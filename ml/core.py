"""Minimal NumPy meta-learning core: ReLU MLPs, losses, MAML/FOMAML meta-gradients.

Parameter containers are lists of (W, b) tuples, one per layer.
"""
import numpy as np


# ---------------------------------------------------------------- param utils
def p_clone(P):
    return [(W.copy(), b.copy()) for W, b in P]


def p_add(P, Q, s=1.0):
    return [(W + s * Wq, b + s * bq) for (W, b), (Wq, bq) in zip(P, Q)]


def p_sub(P, Q, s=1.0):
    return [(W - s * Wq, b - s * bq) for (W, b), (Wq, bq) in zip(P, Q)]


def p_scale(P, s):
    return [(W * s, b * s) for W, b in P]


def p_norm(P):
    return float(np.sqrt(sum((W ** 2).sum() + (b ** 2).sum() for W, b in P)))


def p_dot(P, Q):
    return float(sum((W * Wq).sum() + (b * bq).sum() for (W, b), (Wq, bq) in zip(P, Q)))


def p_flat(P):
    return np.concatenate([np.concatenate([W.ravel(), b.ravel()]) for W, b in P])


def p_unflat(v, P):
    """Inverse of p_flat: rebuild parameter list with flat vector v added."""
    out = []
    off = 0
    for W, b in P:
        n = W.size
        out.append((W + v[off:off + n].reshape(W.shape), None))
        off += n
        n = b.size
        out[-1] = (out[-1][0], b + v[off:off + n].reshape(b.shape))
        off += n
    return out


def p_zeros_like(P):
    return [(np.zeros_like(W), np.zeros_like(b)) for W, b in P]


def p_mean(Ps):
    n = len(Ps)
    out = p_scale(Ps[0], 0.0)
    for P in Ps:
        out = p_add(out, P, 1.0 / n)
    return out


# ---------------------------------------------------------------------- model
class MLP:
    """Fully connected ReLU network with a linear output head."""

    def __init__(self, sizes, seed=0, hidden_scale=None, last_scale=0.1):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.P = []
        for i in range(len(sizes) - 1):
            if i == len(sizes) - 2:
                s = last_scale
            else:
                s = np.sqrt(2.0 / sizes[i]) if hidden_scale is None else hidden_scale
            self.P.append((rng.normal(0.0, s, (sizes[i], sizes[i + 1])),
                           np.zeros(sizes[i + 1])))

    def forward(self, P, X, want_feat=False):
        a = X
        for W, b in P[:-1]:
            a = np.maximum(a @ W + b, 0.0)
        feat = a
        W, b = P[-1]
        z = a @ W + b
        if want_feat:
            return z, feat
        return z

    def features(self, P, X):
        a = X
        for W, b in P[:-1]:
            a = np.maximum(a @ W + b, 0.0)
        return a

    def backward(self, P, X, dZ):
        """dL/dP given dL/dZ (shape of the network output)."""
        cache = []
        a = X
        for W, b in P[:-1]:
            cache.append(a)
            a = np.maximum(a @ W + b, 0.0)
        cache.append(a)
        dP = [None] * len(P)
        delta = dZ
        for l in range(len(P) - 1, -1, -1):
            a_in = cache[l]
            W, b = P[l]
            dP[l] = (a_in.T @ delta, delta.sum(0))
            if l > 0:
                da = delta @ W.T
                delta = da * (cache[l] > 0)
        return dP

    def backward_input(self, P, X, dZ):
        """dL/dX given dL/dZ — needed for adversarial perturbations (VAT)."""
        cache = []
        a = X
        for W, b in P[:-1]:
            cache.append(a)
            a = np.maximum(a @ W + b, 0.0)
        cache.append(a)
        delta = dZ
        for l in range(len(P) - 1, -1, -1):
            W, b = P[l]
            if l == 0:
                return delta @ W.T
            da = delta @ W.T
            delta = da * (cache[l] > 0)
        return None


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def entropy(p, eps=1e-12):
    return -(p * np.log(np.clip(p, eps, None))).sum(axis=1)


# --------------------------------------------------------------------- losses
def ce_grad(net, P, X, y, w=None):
    """Cross entropy on integer labels y. Returns (loss, dP)."""
    z = net.forward(P, X)
    p = softmax(z)
    n = len(y)
    ll = -np.log(np.clip(p[np.arange(n), y], 1e-12, None))
    if w is None:
        loss = float(ll.mean())
        dZ = p.copy()
        dZ[np.arange(n), y] -= 1.0
        dZ /= n
    else:
        w = np.asarray(w, dtype=float)
        sw = w.sum()
        loss = float((ll * w).sum() / max(sw, 1e-12))
        dZ = (p - np.eye(p.shape[1])[y]) * (w[:, None] / max(sw, 1e-12))
    return loss, net.backward(P, X, dZ)


def mse_grad(net, P, X, Y):
    z = net.forward(P, X)
    n = z.shape[0]
    diff = z - Y
    loss = float((diff ** 2).sum(axis=1).mean())
    return loss, net.backward(P, X, 2.0 * diff / n)


def soft_ce_grad(net, P, X, T):
    """Cross entropy against soft targets T (N x C), mean-reduced."""
    z = net.forward(P, X)
    p = softmax(z)
    n = T.shape[0]
    loss = float((-(T * np.log(np.clip(p, 1e-12, None))).sum(axis=1)).mean())
    return loss, net.backward(P, X, (p - T) / n)


def entropy_grad(net, P, X):
    """Mean predictive entropy on unlabeled X (fully differentiable)."""
    z = net.forward(P, X)
    p = softmax(z)
    n = len(p)
    loss = float(entropy(p).mean())
    # dH/dz = p * ( log p + H )  (row-wise)
    H = entropy(p)[:, None]
    dZ = p * (np.log(np.clip(p, 1e-12, None)) + H) / n
    return loss, net.backward(P, X, dZ)


# ----------------------------------------------------------- adaptation / meta
def combine_grads(*pairs):
    """Sum several (loss, dP) pairs into (total_loss, dP)."""
    loss = sum(l for l, _ in pairs)
    dP = p_scale(pairs[0][1], 0.0)
    for _, g in pairs:
        dP = p_add(dP, g, 1.0)
    return loss, dP


def inner_adapt(P, inner_grad, alpha, steps):
    """Returns [theta_0, theta_1, ..., theta_steps]."""
    thetas = [p_clone(P)]
    for _ in range(steps):
        _, g = inner_grad(thetas[-1])
        thetas.append(p_sub(thetas[-1], g, alpha))
    return thetas


def hvp_fd(grad_fn, P, v, rel=1e-4):
    """Finite-difference Hessian-vector product H v for grad_fn(P) -> (loss, dP)."""
    nv = p_norm(v)
    if nv < 1e-300:
        return p_zeros_like(P)
    eps = rel * (1.0 + p_norm(P)) / nv
    _, gp = grad_fn(P)
    _, gm = grad_fn(p_sub(P, v, eps))
    return p_scale(p_sub(gp, gm), 1.0 / eps)


def maml_meta_grad(thetas, outer_grad, inner_grad, alpha, second_order=True, rel=1e-4):
    """Backprop through the inner loop.

    thetas: [theta_0, ..., theta_m] produced by inner_adapt.
    outer_grad(P) -> (loss, dP);  inner_grad(P) -> (loss, dP).
    """
    _, gq = outer_grad(thetas[-1])
    if not second_order:
        return gq
    v = gq
    for k in range(len(thetas) - 2, -1, -1):
        h = hvp_fd(inner_grad, thetas[k], v, rel)
        v = p_sub(v, h, alpha)
    return v


# --------------------------------------------------------------- meta-optimiser
class Adam:
    def __init__(self, P, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = p_zeros_like(P)
        self.v = p_zeros_like(P)
        self.t = 0

    def step(self, P, g):
        self.t += 1
        b1, b2, t = self.b1, self.b2, self.t
        self.m = [(b1 * mw + (1 - b1) * gw, b1 * mb + (1 - b1) * gb)
                  for (mw, mb), (gw, gb) in zip(self.m, g)]
        self.v = [(b2 * vw + (1 - b2) * gw * gw, b2 * vb + (1 - b2) * gb * gb)
                  for (vw, vb), (gw, gb) in zip(self.v, g)]
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        upd = [(self.lr * (mw / c1) / (np.sqrt(vw / c2) + self.eps),
                self.lr * (mb / c1) / (np.sqrt(vb / c2) + self.eps))
               for (mw, mb), (vw, vb) in zip(self.m, self.v)]
        return p_sub(P, upd, 1.0)


class SGD:
    def __init__(self, lr=1e-3):
        self.lr = lr

    def step(self, P, g):
        return p_sub(P, g, self.lr)


# ------------------------------------------------------------------ utilities
def accuracy(net, P, X, y):
    return float((net.forward(P, X).argmax(1) == y).mean())


def stratified_sample(rng, y, k, n_class):
    """Pick k indices per class; returns index array."""
    idx = []
    for c in range(n_class):
        cand = np.flatnonzero(y == c)
        if len(cand) <= k:
            idx.append(cand)
        else:
            idx.append(rng.choice(cand, size=k, replace=False))
    return np.concatenate(idx)


def knn_affinity(F, k=10, sigma=None):
    """Symmetric kNN RBF affinity matrix."""
    n = F.shape[0]
    d2 = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    if sigma is None:
        pos = d2[d2 > 0]
        sigma = np.sqrt(np.median(pos)) if pos.size else 1.0
    W = np.exp(-d2 / (2.0 * sigma ** 2 + 1e-12))
    np.fill_diagonal(W, 0.0)
    k = min(k, n - 1)
    thr = np.sort(W, axis=1)[:, -k][:, None]
    W = np.where(W >= thr, W, 0.0)
    return np.maximum(W, W.T)
