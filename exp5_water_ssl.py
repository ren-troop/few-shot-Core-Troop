"""Experiment 5 -- which SSL method fits the REAL water dataset?

Data: water_dataset.mat -- 37 USGS sites x 705 days x 11 covariates,
target = next-day pH, real spatio-temporal structure.

Setting (a deployed new sensor):
  * a backbone is pretrained (fully supervised) on the meta-train sites,
  * at a held-out site we get K labelled days and the FULL unlabelled
    covariate record (remaining train days + the whole test period),
  * every method adapts from the SAME pretrained backbone with the SAME step
    budget, so the only difference is how the unlabelled days are used.

Methods (all in their regression form, since the target is continuous):
  sup        : labelled MSE only
  pl_reg     : Pseudo-Label -- self-training on the model's own predictions,
               no augmentation, ramp-up weight
  pi_reg     : FixMatch's regression analogue -- consistency between a weak and
               a strong augmented view (no threshold: regression has no argmax)
  vat_reg    : VAT -- consistency under the virtual adversarial perturbation,
               D = ||f(x) - f(x+r)||^2, r = eps * g/||g||
  manifold   : graph-Laplacian smoothness on the unlabelled covariates
  sup_lp     : supervise, then transductive label propagation on the covariate graph
"""
import sys, os, json, time, argparse
import numpy as np

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
os.makedirs(os.path.join(WS, "results"), exist_ok=True)   # 输出目录（打包时改名为 results）

from ml.core import (MLP, mse_grad, Adam, p_clone, p_scale, p_add, p_zeros_like,
                     knn_affinity)
from ml.ssl import label_spreading
from exp2_water import WaterData, make_splits, ridge_fit


# ------------------------------------------------------------------ graphs
def build_graph(F, k=10):
    return knn_affinity(F, k=k)


def lp_regression(W, y_lab, idx_lab, iters=200):
    N = W.shape[0]
    Y0 = np.zeros((N, 1)); Y0[idx_lab, 0] = y_lab
    return label_spreading(W, Y0, alpha=0.9, iters=iters, clamp_idx=idx_lab,
                           mode="clamp")[:, 0]


# ------------------------------------------------------------------ terms
def mse_term(net, X, T, weight=1.0):
    T = T.reshape(-1, 1)
    def g(P):
        z = net.forward(P, X)
        n = z.shape[0]
        diff = z - T
        loss = weight * float((diff ** 2).sum(1).mean())
        return loss, net.backward(P, X, 2.0 * weight * diff / n)
    return g


def manifold_term(net, X, L, weight=1.0):
    Z = float(np.abs(L).sum()) + 1e-12
    def g(P):
        z = net.forward(P, X)
        loss = weight * float((z.T @ (L @ z)).item()) / Z
        return loss, net.backward(P, X, 2.0 * weight * (L @ z) / Z)
    return g


def vat_reg_term(net, X, cfg, rng):
    """D = ||f(x) - f(x+r)||^2 ; one power iteration on the squared difference."""
    d = rng.normal(size=X.shape)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    # g = grad_r D |_{r = xi d}:  D = (f(x+xi d) - f(x))^2
    z0 = net.forward(P_holder[0], X)
    zp = net.forward(P_holder[0], X + cfg["xi"] * d)
    dZ = 2.0 * (zp - z0) / len(X)
    g = net.backward_input(P_holder[0], X + cfg["xi"] * d, dZ)
    g /= (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    r = cfg["eps"] * g
    T = net.forward(P_holder[0], X)                    # detached target
    def gfn(P):
        z = net.forward(P, X + r)
        n = z.shape[0]
        diff = z - T
        loss = cfg["lam_u"] * float((diff ** 2).sum(1).mean())
        return loss, net.backward(P, X + r, 2.0 * cfg["lam_u"] * diff / n)
    return gfn


P_holder = [None]      # the current parameters, so VAT can build its direction


def ramp(step, total, warm=0.2):
    return min(1.0, step / max(1.0, warm * total))


# ------------------------------------------------------------------ train
def train(method, net, P0, Xl, yl, Xu, cfg, seed):
    rng = np.random.default_rng(seed)
    P = p_clone(P0)
    P_ema = p_clone(P0)          # lagged teacher for pl_reg (otherwise grad == 0)
    opt = Adam(P, lr=cfg["lr"])
    nl, nu = len(yl), len(Xu)
    graph = None
    if method == "manifold":
        W = build_graph(Xu, k=cfg["graph_k"])
        graph = np.diag(W.sum(1)) - W
    for step in range(cfg["steps"]):
        bl = rng.choice(nl, size=min(cfg["batch_l"], nl), replace=False)
        _, dP = mse_grad(net, P, Xl[bl], yl[bl].reshape(-1, 1))
        if method != "sup" and nu > 0:
            bu = rng.choice(nu, size=min(cfg["batch_u"], nu), replace=False)
            Xb = Xu[bu]
            if method == "pl_reg":
                # Pseudo-Label for regression: the "hard" target is the lagged
                # teacher's prediction on the SAME input (the paper's design);
                # using the current parameters would give exactly zero gradient.
                T = net.forward(P_ema, Xb).ravel()
                w = cfg["lam_u"] * ramp(step, cfg["steps"])
                t = mse_term(net, Xb, T, w)(P)
            elif method == "pi_reg":
                weak = Xb + rng.normal(0, cfg["sigma_w"], Xb.shape)
                strong = Xb + rng.normal(0, cfg["sigma_s"], Xb.shape)
                m = rng.random(Xb.shape) > cfg["drop_s"]
                strong = strong * m
                T = net.forward(P, weak).ravel()
                t = mse_term(net, strong, T, cfg["lam_u"])(P)
            elif method == "vat_reg":
                P_holder[0] = P
                t = vat_reg_term(net, Xb, cfg, rng)(P)
            elif method == "manifold":
                t = manifold_term(net, Xu, graph, cfg["lam_u"])(P)
            else:
                raise ValueError(method)
            dP = p_add(dP, t[1], 1.0)
        P = opt.step(P, dP)
        if method == "pl_reg":
            b = cfg["ema"]
            P_ema = [(b * W1 + (1 - b) * W2, b * b1 + (1 - b) * b2)
                     for (W1, b1), (W2, b2) in zip(P_ema, P)]
    return net, P


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--split", default="random")
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--out", default="results/exp5_water_ssl.json")
    args = ap.parse_args()

    wd = WaterData(rf"{WS}\data\water_dataset.mat")
    tr_sites, te_sites = make_splits(wd, args.split)
    sites = {s: wd.site(int(s)) for s in range(wd.n_site)}

    cfg = dict(hidden=[64, 64], lr=3e-3, steps=300, batch_l=32, batch_u=128,
               lam_u=1.0, sigma_w=0.01, sigma_s=0.10, drop_s=0.20, ema=0.99,
               eps=0.15, xi=1e-6, graph_k=10,
               backbone_steps=4000, backbone_lr=3e-3)
    # consistency-type losses in regression have a trivial constant solution,
    # so lambda must be small; it is selected on held-out SOURCE sites only.
    LAM_GRID = {"sup": [0.0], "pl_reg": [0.0, 0.05, 0.2, 1.0],
                "pi_reg": [0.0, 0.005, 0.02, 0.1],
                "vat_reg": [0.0, 0.005, 0.02, 0.1],
                "manifold": [0.0, 0.005, 0.02, 0.1]}
    if args.quick:
        cfg["steps"] = 60; cfg["backbone_steps"] = 400
        LAM_GRID = {k: v[:2] for k, v in LAM_GRID.items()}
    seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4]
    methods = ["sup", "pl_reg", "pi_reg", "vat_reg", "manifold"]

    # ---- pretrain ONE backbone on the meta-train sites (fully supervised)
    A = np.vstack([sites[int(s)][0][sites[int(s)][2]] for s in tr_sites])
    b = np.concatenate([sites[int(s)][1][sites[int(s)][2]] for s in tr_sites])
    net = MLP([A.shape[1]] + list(cfg["hidden"]) + [1], seed=0, last_scale=0.1)
    P0 = net.P
    opt = Adam(P0, lr=cfg["backbone_lr"])
    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(cfg["backbone_steps"]):
        idx = rng.choice(len(b), 128, replace=False)
        _, g = mse_grad(net, P0, A[idx], b[idx].reshape(-1, 1))
        P0 = opt.step(P0, g)
    print(f"pretrained backbone on {len(tr_sites)} sites / {len(b)} days "
          f"({time.time()-t0:.0f}s)")

    out = {"config": cfg, "split": args.split, "K": args.K,
           "train_sites": [int(s) for s in tr_sites],
           "test_sites": [int(s) for s in te_sites], "methods": methods,
           "results": {}}
    logf = open(rf"{WS}\results\exp5_log.txt", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    log("=" * 78)
    log(f"exp5 REAL water dataset  split={args.split} K={args.K} "
        f"steps={cfg['steps']} seeds={seeds}")

    # ---- 1) select lambda on held-out SOURCE sites (no target labels touched)
    src_val = list(tr_sites[:6]) if len(tr_sites) > 8 else list(tr_sites)
    lam_sel = {}
    log(f"  lambda selection on source sites {[int(s) for s in src_val]}")
    for m in methods:
        best = (np.inf, None)
        for lam in LAM_GRID[m]:
            c2 = dict(cfg); c2["lam_u"] = lam
            tot = []
            for seed in ([0] if args.quick else [0, 1]):
                rng = np.random.default_rng(500 + seed)
                for s in src_val:
                    X, t, idx_tr, idx_te = sites[int(s)]
                    sup_idx = np.sort(rng.choice(idx_tr, size=args.K, replace=False))
                    unl = np.setdiff1d(np.arange(len(t)), sup_idx)
                    _, P = train(m, net, P0, X[sup_idx], t[sup_idx], X[unl],
                                 c2, seed)
                    pred = net.forward(P, X[idx_te]).ravel()
                    tot.append(float(np.mean((pred - t[idx_te]) ** 2)))
            mu = float(np.mean(tot))
            log(f"    {m:10s} lam={lam:<6} source-val MSE={mu:.6f}")
            if mu < best[0]:
                best = (mu, lam)
        lam_sel[m] = best[1]
        log(f"    -> {m}: lambda = {best[1]}")
    out_lam = dict(lam_sel)

    per = {m: [] for m in methods}
    per_lp = {m: [] for m in methods}
    for seed in seeds:
        rng = np.random.default_rng(3000 + seed)
        for s in te_sites:
            X, t, idx_tr, idx_te = sites[int(s)]
            sup_idx = np.sort(rng.choice(idx_tr, size=args.K, replace=False))
            unl = np.setdiff1d(np.arange(len(t)), sup_idx)      # all other days
            Xl, yl, Xu = X[sup_idx], t[sup_idx], X[unl]
            Xte, yte = X[idx_te], t[idx_te]
            rec = {}
            for m in methods:
                c2 = dict(cfg); c2["lam_u"] = lam_sel[m]
                net2, P = train(m, net, P0, Xl, yl, Xu, c2, seed)
                pred = net2.forward(P, Xte).ravel()
                rec[m] = float(np.mean((pred - yte) ** 2))
                # transductive label propagation on the same backbone
                pool = np.arange(len(t))
                F = net2.features(P, X[pool])
                W = build_graph(F, k=cfg["graph_k"])
                pos = np.searchsorted(pool, sup_idx)
                yv = lp_regression(W, yl, pos)
                rec[m + "+lp"] = float(np.mean((yv[idx_te] - yte) ** 2))
            for m in methods:
                per[m].append(rec[m])
                per_lp[m].append(rec[m + "+lp"])

    def st(v):
        v = np.asarray(v, float)
        return [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))]
    # paired gain vs sup (same site & seed)
    base = np.array(per["sup"])
    summ = {}
    for m in methods:
        d = np.array(per[m]) - base
        dlp = np.array(per_lp[m]) - base
        summ[m] = {"mse": st(per[m]), "lp_mse": st(per_lp[m]),
                   "paired_d": st(d), "paired_d_lp": st(dlp),
                   "lambda": lam_sel[m]}
    out["results"] = summ
    out["lambda_source_val"] = out_lam
    log(f"\n  {'method':10s} {'lambda':>7s} {'MSE':>18s} {'paired dMSE':>20s} "
        f"{'MSE+LP':>18s} {'paired dMSE(+LP)':>20s}")
    for m in methods:
        s = summ[m]
        extra = ("                 ref" if m == "sup" else
                 f"{s['paired_d'][0]:+.5f}+-{s['paired_d'][1]:.5f}")
        extra2 = ("                 ref" if m == "sup" else
                  f"{s['paired_d_lp'][0]:+.5f}+-{s['paired_d_lp'][1]:.5f}")
        log(f"  {m:10s} {s['lambda']:>7} {s['mse'][0]:.6f}+-{s['mse'][1]:.6f}   "
            f"{extra:>20s} {s['lp_mse'][0]:.6f}+-{s['lp_mse'][1]:.6f}   {extra2:>20s}")
    log("  (lower MSE is better; 'paired dMSE' = method - sup, negative = SSL better)")
    json.dump(out, open(os.path.join(WS, args.out), "w", encoding="utf-8"), indent=2)
    log("saved -> " + args.out)


if __name__ == "__main__":
    main()
