"""Semi-supervised modules used either (a) inside the MAML inner loop or
(b) after adaptation (three-stage pipeline).

Everything here is written so that it can be *detached* and used as a target
generator for a differentiable student loss -- which is exactly the interface
that the MAML inner loop accepts.
"""
import numpy as np

from .core import (p_clone, p_sub, softmax, entropy, ce_grad, soft_ce_grad,
                   entropy_grad, combine_grads, knn_affinity, Adam)


# ------------------------------------------------------------ graph / LP core
def label_spreading(W, Y0, alpha=0.9, iters=60, clamp_idx=None, mode="spread"):
    """Graph-based propagation of labels/targets.

    mode="spread" : Zhu & Ghahramani label spreading,
                    Y <- alpha S Y + (1-alpha) Y0   (soft, shrinks to the prior)
    mode="clamp"  : hard-clamped label propagation,
                    Y <- S Y ; Y[clamp] <- Y0[clamp]
                    which converges to the harmonic solution and preserves the
                    labelled values exactly.  This is the right choice for
                    *regression*, where softening to the prior is a fatal bias.
    """
    d = W.sum(1)
    d[d <= 0] = 1.0
    dinv = 1.0 / np.sqrt(d)
    S = W * dinv[:, None] * dinv[None, :]
    Y = Y0.copy()
    if mode == "clamp" and clamp_idx is not None:
        for _ in range(iters):
            Y = S @ Y
            Y[clamp_idx] = Y0[clamp_idx]
        return Y
    for _ in range(iters):
        Y = alpha * (S @ Y) + (1.0 - alpha) * Y0
    if clamp_idx is not None:
        Y[clamp_idx] = Y0[clamp_idx]
    return Y


def label_propagation(F, y_lab, idx_lab, n_class, k=10, alpha=0.9, iters=60,
                      sigma=None):
    """Transductive label propagation on features F.  Returns soft labels (N,C)."""
    N = F.shape[0]
    Y0 = np.zeros((N, n_class))
    Y0[idx_lab, y_lab] = 1.0
    W = knn_affinity(F, k=k, sigma=sigma)
    return label_spreading(W, Y0, alpha=alpha, iters=iters, clamp_idx=idx_lab)


# ------------------------------------------------------------- prototypes
def prototypes(F, y_lab, idx_lab, n_class):
    d = F.shape[1]
    P = np.zeros((n_class, d))
    for c in range(n_class):
        sel = idx_lab[y_lab == c]
        P[c] = F[sel].mean(0) if len(sel) else 0.0
    return P


def prototype_logits(F, y_lab, idx_lab, n_class, tau=0.1, metric="euclid"):
    """Softmax over negative squared distances to class prototypes."""
    P = prototypes(F, y_lab, idx_lab, n_class)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    if metric == "cos":
        s = Fn @ Pn.T
    else:
        d2 = ((F[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        s = -d2
    return softmax(s / tau)


# ------------------------------------------------- SSL terms for the inner loop
def make_entropy_term(net, X_u):
    def g(P):
        return entropy_grad(net, P, X_u)
    return g


def make_proto_distill_term(net, X_s, y_s, X_u, tau=0.1, weight=1.0, mode="soft"):
    """Teacher = prototype classifier built from detached support features."""
    def g(P):
        Fs = net.features(P, X_s)
        Fu = net.features(P, X_u)
        T = prototype_logits(Fs, y_s, np.arange(len(y_s)), int(y_s.max()) + 1, tau)
        # T rows correspond to support; recompute for unlabeled
        Pr = prototypes(Fs, y_s, np.arange(len(y_s)), int(y_s.max()) + 1)
        d2 = ((Fu[:, None, :] - Pr[None, :, :]) ** 2).sum(-1)
        Tu = softmax(-d2 / tau)
        if mode == "hard":
            hard = np.zeros_like(Tu)
            hard[np.arange(len(Tu)), Tu.argmax(1)] = 1.0
            Tu = hard
        loss, dP = soft_ce_grad(net, P, X_u, Tu)
        return weight * loss, [(W * weight, b * weight) for W, b in dP]
    return g


def make_lp_distill_term(net, X_s, y_s, X_u, n_class, k=10, alpha=0.9,
                         weight=1.0, hard=False):
    """Teacher = label propagation over detached features of support+unlabeled."""
    def g(P):
        Fs = net.features(P, X_s)
        Fu = net.features(P, X_u)
        F = np.vstack([Fs, Fu])
        idx = np.arange(len(Fs))
        Y = label_propagation(F, y_s, idx, n_class, k=k, alpha=alpha)
        Tu = Y[len(Fs):]
        if hard:
            h = np.zeros_like(Tu)
            h[np.arange(len(Tu)), Tu.argmax(1)] = 1.0
            Tu = h
        Tu = np.clip(Tu, 1e-6, None)
        Tu = Tu / Tu.sum(1, keepdims=True)
        loss, dP = soft_ce_grad(net, P, X_u, Tu)
        return weight * loss, [(W * weight, b * weight) for W, b in dP]
    return g


# --------------------------------------------------- third stage: self-training
def selftrain(net, P, X_all, y_lab, idx_lab, n_class, X_eval=None,
              rounds=1, steps=100, lr=1e-3, thr=0.0, use_lp=True, k=10,
              alpha=0.9, seed=0, only_last=True, sharp=0.5):
    """Three-stage step 3: pseudo-label the unlabeled set with the *adapted*
    model, then refit.  only_last=True keeps the meta-learned representation
    frozen and adapts only the linear head -- the stable 'anchoring' variant.

    Returns the refined parameters.
    """
    Pc = p_clone(P)
    N = X_all.shape[0]
    unl = np.setdiff1d(np.arange(N), idx_lab)
    for r in range(rounds):
        F = net.features(Pc, X_all)
        if use_lp:
            Y = label_propagation(F, y_lab, idx_lab, n_class, k=k, alpha=alpha)
            T = Y[unl]
        else:
            T = softmax(net.forward(Pc, X_all[unl]))
        conf = T.max(1)
        keep = conf >= thr
        Xu = X_all[unl][keep]
        Tu = T[keep]
        if len(Xu) == 0:
            break
        hard = np.zeros_like(Tu)
        hard[np.arange(len(Tu)), Tu.argmax(1)] = 1.0
        Tu = (1.0 - sharp) * Tu + sharp * hard
        Xa = np.vstack([X_all[idx_lab], Xu])
        Ta = np.vstack([np.eye(n_class)[y_lab], Tu])
        opt = Adam(Pc, lr=lr)
        for _ in range(steps):
            _, g = soft_ce_grad(net, Pc, Xa, Ta)
            if only_last:
                g = [(np.zeros_like(W), np.zeros_like(b)) for W, b in g[:-1]] + [g[-1]]
            Pc = opt.step(Pc, g)
    return Pc
