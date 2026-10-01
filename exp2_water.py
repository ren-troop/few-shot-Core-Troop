"""Experiment 2 -- the water-quality dataset as a few-shot site-adaptation task.

Task = one monitoring site.  Meta-train on a subset of sites, meta-test on
held-out sites.  For a new site we are given:
  * K labelled days (laboratory pH measurements)  -> the "anchor"
  * the FULL unlabelled covariate record (704 days, 11 indices) -> the amplifier
and must predict pH on the held-out (later) period.
"""
import sys, time, json, argparse, os
import numpy as np

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
os.makedirs(os.path.join(WS, "results"), exist_ok=True)   # 输出目录（打包时改名为 results）

from matio import loadmat
from ml.core import (MLP, mse_grad, p_add, p_scale, p_zeros_like, p_clone,
                     inner_adapt, maml_meta_grad, Adam, combine_grads)
from ml.ssl import label_spreading, knn_affinity, selftrain
from ml.core import knn_affinity as _knn


# ------------------------------------------------------------------ data
class WaterData:
    def __init__(self, path):
        d = loadmat(path)
        self.Xtr = np.stack([np.asarray(e, float) for e in d["X_tr"].ravel()], 0)
        self.Xte = np.stack([np.asarray(e, float) for e in d["X_te"].ravel()], 0)
        self.Ytr = d["Y_tr"].astype(float)
        self.Yte = d["Y_te"].astype(float)
        self.n_site = self.Ytr.shape[0]
        self.mu = float(self.Ytr.mean())
        self.sd = float(self.Ytr.std())
        groups = [np.asarray(g).ravel().astype(int) - 1 for g in d["location_group"].ravel()]
        self.groups = groups
        self.features = [str(f) for f in d["features"].ravel()]

    def site(self, s):
        U = np.vstack([self.Xtr[:, s, :], self.Xte[:, s, :]])   # 705 x 11
        y = np.concatenate([self.Ytr[s], self.Yte[s]])          # 705
        X = U[:-1]                                              # day i covariates
        t = (y[1:] - self.mu) / self.sd                         # day i+1 pH (standardised)
        n_tr = self.Xtr.shape[0] - 1                            # 422 usable labelled days
        idx_tr = np.arange(n_tr)
        idx_te = np.arange(self.Xtr.shape[0] - 1, len(t))
        return X, t, idx_tr, idx_te


def make_splits(wd, mode, seed=0):
    rng = np.random.default_rng(seed)
    if mode == "random":
        perm = rng.permutation(wd.n_site)
        n_test = max(6, wd.n_site // 3)
        te = np.sort(perm[:n_test])
        tr = np.sort(perm[n_test:])
    elif mode == "spatial":
        # group 2 (28 coastal sites) -> meta-train, groups 0/1 (Atlanta cluster) -> meta-test
        tr = np.sort(wd.groups[2])
        te = np.sort(np.concatenate([wd.groups[0], wd.groups[1]]))
    else:
        raise ValueError(mode)
    return tr, te


# ------------------------------------------------------------- graph helpers
def build_graph(F, k=10, sigma=None):
    return knn_affinity(F, k=k, sigma=sigma)


def lp_regression(W, y_lab, idx_lab, alpha=0.9, iters=None, mode=None):
    N = W.shape[0]
    Y0 = np.zeros((N, 1))
    Y0[idx_lab, 0] = y_lab
    if mode is None:
        mode = "clamp"
    if iters is None:
        iters = 200 if mode == "clamp" else 60
    Y = label_spreading(W, Y0, alpha=alpha, iters=iters, clamp_idx=idx_lab, mode=mode)
    return Y[:, 0]


# --------------------------------------------------------------- inner terms
def make_manifold_term(net, X_u, L, weight=1.0):
    """Graph-Laplacian smoothness on the unlabelled covariates (differentiable).

    Normalised by the total edge weight so that the term has the same scale as
    the mean-squared error on the labelled points.
    """
    DmW = L
    n = X_u.shape[0]
    Z = float(np.abs(DmW).sum()) + 1e-12

    def g(P):
        z = net.forward(P, X_u)
        quad = float((z.T @ (DmW @ z)).item())
        lbd = weight * quad / Z
        dz = 2.0 * weight * (DmW @ z) / Z
        return lbd, net.backward(P, X_u, dz)
    return g


def make_lp_target_term(net, X_u, target, weight=1.0):
    """MSE against a fixed (detached) label-propagation pseudo-label."""
    T = target.reshape(-1, 1)

    def g(P):
        z = net.forward(P, X_u)
        n = z.shape[0]
        diff = z - T
        loss = weight * float((diff ** 2).sum(axis=1).mean())
        return loss, net.backward(P, X_u, 2.0 * weight * diff / n)
    return g


def scale_term(pair, w):
    l, g = pair
    return w * l, p_scale(g, w)


# ------------------------------------------------------------------ meta
def build_task(wd, s, cfg, rng):
    X, t, idx_tr, idx_te = wd.site(s)
    K = cfg["K"]
    sup = rng.choice(idx_tr, size=min(K, len(idx_tr)), replace=False)
    q = rng.choice(idx_te, size=min(cfg["q_size"], len(idx_te)), replace=False)
    sup = np.sort(sup)
    q = np.sort(q)
    # anchor-based centring: the K labelled days define the site's level/scale.
    # This is what makes the "anchor" meaningful and makes LP's shrinkage toward
    # zero the correct behaviour instead of a bias.
    # NOTE: the covariates predict the *site level*, which is exactly what
    # subtracting the K-label mean would destroy (pooled ridge: 0.63 centred vs
    # 0.34 raw).  The anchors act through the inner loop, not through centring.
    m_s = 0.0
    if cfg.get("center_mode", "none") == "meanstd":
        m_s = float(t[sup].mean())
        s_s = float(t[sup].std()) + 1e-3
    else:
        s_s = 1.0
    return dict(X=X, t=t, sup=sup, q=q, idx_tr=idx_tr, idx_te=idx_te,
                m=m_s, sd=s_s, tc=(t - m_s) / s_s)


def task_graphs(task, cfg):
    X, sup = task["X"], task["sup"]
    pool = cfg["pool"]
    rng = np.random.default_rng(cfg["seed"] + 13)
    unl = rng.choice(np.arange(X.shape[0]), size=min(pool, X.shape[0]), replace=False)
    unl = np.union1d(unl, sup)
    Xu = X[unl]
    W = build_graph(Xu, k=cfg["graph_k"])
    L = np.diag(W.sum(1)) - W
    pos = {int(v): i for i, v in enumerate(unl)}
    return unl, Xu, W, L, pos


def train(cfg, wd, tr_sites, log=None, curve=None):
    net = MLP([cfg["dim"]] + list(cfg["hidden"]) + [1], seed=cfg["seed"],
              last_scale=cfg["last_scale"])
    P = net.P
    opt = Adam(P, lr=cfg["meta_lr"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    kind = cfg["kind"]
    second_order = (kind == "maml2")
    t0 = time.time()
    for it in range(cfg["iters"]):
        gsum = p_zeros_like(P)
        for _ in range(cfg["meta_batch"]):
            s = int(rng.choice(tr_sites))
            task = build_task(wd, s, cfg, rng)
            if kind == "joint":
                _, g = mse_grad(net, P, task["X"][task["sup"]],
                                task["tc"][task["sup"]].reshape(-1, 1))
            else:
                if kind in ("manifold", "lpdistill"):
                    task["_graphs"] = task_graphs(task, cfg)
                inner = make_inner(net, task, cfg)
                outer = (lambda Q, tk=task: mse_grad(
                    net, Q, tk["X"][tk["q"]], tk["tc"][tk["q"]].reshape(-1, 1)))
                thetas = inner_adapt(P, inner, cfg["alpha"], cfg["inner_steps"])
                g = maml_meta_grad(thetas, outer, inner, cfg["alpha"],
                                   second_order=second_order)
            gsum = p_add(gsum, g, 1.0 / cfg["meta_batch"])
        P = opt.step(P, gsum)
        if log is not None and (it + 1) % max(1, cfg["iters"] // 4) == 0:
            log(f"    [{kind}] iter {it+1}/{cfg['iters']}  {time.time()-t0:.0f}s")
        if curve is not None and (it + 1) % cfg["curve_every"] == 0:
            r2 = np.random.default_rng(999)
            ts = [build_task(wd, int(s), cfg, r2) for s in te_sites_for_curve(cfg, wd)]
            a = np.mean([evaluate(net, P, wd, t, "sup", cfg) for t in ts])
            b = np.mean([evaluate(net, P, wd, t, "sup_then_lp", cfg) for t in ts])
            curve.append({"iter": it + 1, "sup": float(a), "sup_then_lp": float(b)})
    return net, P


_CURVE_SITES = {}


def te_sites_for_curve(cfg, wd):
    if "s" not in _CURVE_SITES:
        _CURVE_SITES["s"] = list(cfg["_te_sites"])
    return _CURVE_SITES["s"]


def head_only_filter(g):
    return [(np.zeros_like(W), np.zeros_like(b)) for W, b in g[:-1]] + [g[-1]]


def make_inner(net, task, cfg):
    X, tc, sup = task["X"], task["tc"], task["sup"]
    base = lambda P: mse_grad(net, P, X[sup], tc[sup].reshape(-1, 1))
    kind = cfg["kind"]
    if kind in ("maml2", "fomaml", "sup", "joint"):
        g = base
    elif kind == "manifold":
        unl, Xu, W, L, pos = task["_graphs"]
        term = make_manifold_term(net, Xu, L, weight=cfg["lam"])
        g = (lambda P: combine_grads(base(P), term(P)))
    elif kind == "lpdistill":
        unl, Xu, W, L, pos = task["_graphs"]
        target = lp_regression(W, tc[sup], np.array([pos[int(v)] for v in sup]),
                               alpha=cfg["lp_alpha"])
        term = make_lp_target_term(net, Xu, target, weight=cfg["lam"])
        g = (lambda P: combine_grads(base(P), term(P)))
    else:
        raise ValueError(kind)
    if cfg.get("head_only") and kind != "joint":
        inner = g
        return lambda P: (lambda r: (r[0], head_only_filter(r[1])))(inner(P))
    return g


# ------------------------------------------------------------------- eval
def ridge_fit(A, b, lam=1e-3):
    A1 = np.hstack([A, np.ones((A.shape[0], 1))])
    return np.linalg.solve(A1.T @ A1 + lam * np.eye(A1.shape[1]), A1.T @ b)


def evaluate(net, P0, wd, task, method, cfg):
    X, t, sup, q = task["X"], task["t"], task["sup"], task["q"]
    tc, m_s, s_s = task["tc"], task["m"], task["sd"]
    tq = t[q]

    def mse_centred(pred_c):
        pred = np.asarray(pred_c).ravel() * s_s + m_s
        return float(np.mean((pred - tq) ** 2))

    if method == "support_mean":
        return float(np.mean((m_s - tq) ** 2))
    if method == "ridge":
        w = ridge_fit(X[sup], t[sup])
        p = np.hstack([X[q], np.ones((len(q), 1))]) @ w
        return float(np.mean((p - tq) ** 2))
    if method == "global_ridge":
        w = cfg["_gw"]
        p = np.hstack([X[q], np.ones((len(q), 1))]) @ w
        return float(np.mean((p - tq) ** 2))
    if method == "oracle_ridge":
        w = ridge_fit(X[task["idx_tr"]], t[task["idx_tr"]])
        p = np.hstack([X[q], np.ones((len(q), 1))]) @ w
        return float(np.mean((p - tq) ** 2))

    steps = cfg["eval_steps"]
    if method in ("sup", "manifold", "lpdistill"):
        sub = dict(cfg)
        sub["kind"] = method
        inner = make_inner(net, task, sub)
        th = inner_adapt(P0, inner, cfg["alpha"], steps)
        return mse_centred(net.forward(th[-1], X[q]))

    th = inner_adapt(P0, make_inner(net, dict(task, **{}), dict(cfg, kind="sup")),
                     cfg["alpha"], steps)
    Pf = th[-1]
    pool_full = np.union1d(np.union1d(sup, q), cfg["_pool_idx"])
    pos_sup = np.searchsorted(pool_full, sup)
    pos_q = np.searchsorted(pool_full, q)
    if method == "sup_then_da":
        # distribution alignment: the unlabelled record tells us the site's
        # covariate distribution, hence where its pH level should sit.
        p_pool = net.forward(Pf, X[pool_full]).ravel()
        delta = float(np.mean(t[sup]) - np.mean(p_pool))
        return mse_centred(net.forward(Pf, X[q]).ravel() + delta)
    if method in ("sup_then_lp", "sup_then_lp_feat", "sup_then_st",
                  "sup_then_sthead", "sup_then_lp_seeded"):
        F = (X[pool_full] if method == "sup_then_lp"
             else net.features(Pf, X[pool_full]))
        W = build_graph(F, k=cfg["graph_k"])
        if method == "sup_then_lp_seeded":
            # seed the unlabelled nodes with the adapted model's own predictions
            N = len(pool_full)
            Y0 = np.zeros((N, 1))
            Y0[:, 0] = net.forward(Pf, X[pool_full]).ravel()
            Y0[pos_sup, 0] = tc[sup]
            yv = label_spreading(W, Y0, alpha=cfg["lp_alpha"], iters=200,
                                 clamp_idx=pos_sup, mode="clamp")[:, 0]
        else:
            yv = lp_regression(W, tc[sup], pos_sup, alpha=cfg["lp_alpha"])
        if method in ("sup_then_lp", "sup_then_lp_feat", "sup_then_lp_seeded"):
            return mse_centred(yv[pos_q])
        # self-training: pseudo-label every day, then refit (head / whole net)
        Pc = p_clone(Pf)
        Ta = yv.reshape(-1, 1).copy()
        Ta[pos_sup, 0] = tc[sup]
        Xa = X[pool_full]
        opt = Adam(Pc, lr=cfg["st_lr"])
        for _ in range(cfg["st_steps"]):
            _, g = mse_grad(net, Pc, Xa, Ta)
            if method == "sup_then_sthead":
                g = [(np.zeros_like(Wm), np.zeros_like(b)) for Wm, b in g[:-1]] + [g[-1]]
            Pc = opt.step(Pc, g)
        return mse_centred(net.forward(Pc, X[q]))
    raise ValueError(method)


EVAL_METHODS = ["support_mean", "ridge", "global_ridge", "oracle_ridge",
                "sup", "manifold", "lpdistill", "sup_then_da", "sup_then_lp",
                "sup_then_lp_seeded", "sup_then_lp_feat", "sup_then_st",
                "sup_then_sthead"]


def build_eval_tasks(wd, te_sites, cfg, n_rep=1, seed=None):
    """Deterministic task list shared by every evaluation method."""
    base = cfg["test_seed"] if seed is None else seed
    out = []
    for r in range(n_rep):
        for s in te_sites:
            rng = np.random.default_rng(base + 1000 * r + int(s))
            out.append(build_task(wd, int(s), cfg, rng))
    return out


def run_eval(net, P0, wd, te_sites, cfg):
    tasks = build_eval_tasks(wd, te_sites, cfg, n_rep=cfg.get("eval_reps", 1))
    out = {}
    for m in cfg["eval_methods"]:
        vals = []
        for task in tasks:
            unl, Xu, W, L, pos = task_graphs(task, cfg)
            task["_graphs"] = (unl, Xu, W, L, pos)
            cfg["_pool_idx"] = unl
            vals.append(evaluate(net, P0, wd, task, m, cfg))
        vals = np.array(vals)
        out[m] = [float(vals.mean()), float(vals.std() / np.sqrt(len(vals)))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--kinds", default="rand,joint,maml2,fomaml,manifold,lpdistill")
    ap.add_argument("--split", default="random")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--hidden", default="40,40")
    ap.add_argument("--head-only", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/exp2_water.json")
    args = ap.parse_args()

    wd = WaterData(os.path.join(WS, "data", "water_dataset.mat"))
    tr_sites, te_sites = make_splits(wd, args.split)
    hidden = [int(h) for h in args.hidden.split(",") if h.strip() != ""]
    if not hidden:
        hidden = []
    cfg = dict(
        kind="maml2", split=args.split, K=args.K, q_size=120, pool=192,
        dim=11, hidden=hidden, last_scale=0.1, seed=0, head_only=args.head_only,
        alpha=0.02, inner_steps=3, eval_steps=8,
        meta_lr=args.lr, meta_batch=8, iters=3000, lam=1.0,
        graph_k=5, lp_alpha=0.9, st_lr=1e-3, st_steps=120,
        curve_every=200, eval_methods=EVAL_METHODS, test_seed=2024,
        eval_reps=args.reps,
        _te_sites=[int(s) for s in te_sites],
        _pool_idx=np.arange(0, wd.Xtr.shape[0] + wd.Xte.shape[0] - 1),
    )
    if args.quick:
        cfg["iters"] = 40
        cfg["curve_every"] = 20
    if args.iters:
        cfg["iters"] = args.iters
        cfg["curve_every"] = max(20, args.iters // 6)

    # global ridge over meta-train sites (no adaptation)
    A, b = [], []
    for s in tr_sites:
        X, t, idx_tr, idx_te = wd.site(int(s))
        A.append(X[idx_tr]); b.append(t[idx_tr])
    cfg["_gw"] = ridge_fit(np.vstack(A), np.concatenate(b))

    logf = open(os.path.join(WS, "results", "exp2_log.txt"), "a", encoding="utf-8")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())

    log("=" * 70)
    log(f"exp2 split={args.split} K={args.K} train_sites={len(tr_sites)} "
        f"test_sites={len(te_sites)}")
    log("  meta-train sites: " + str(sorted(int(s) for s in tr_sites)))
    log("  meta-test  sites: " + str(sorted(int(s) for s in te_sites)))

    results, curves = {}, {}
    for kind in args.kinds.split(","):
        t0 = time.time()
        cfg["kind"] = kind
        curve = []
        log(f"  training {kind} ...")
        if kind == "rand":
            net = MLP([cfg["dim"]] + list(cfg["hidden"]) + [1], seed=cfg["seed"] + 777,
                      last_scale=cfg["last_scale"])
            P = net.P
        else:
            net, P = train(cfg, wd, tr_sites, log=log, curve=curve)
        res = run_eval(net, P, wd, te_sites, cfg)
        results[kind] = res
        curves[kind] = curve
        log(f"  == {kind} (train {time.time()-t0:.0f}s)")
        for m, (mu, se) in res.items():
            log(f"       {m:18s} MSE = {mu:.6e} +- {se:.2e}")
    with open(os.path.join(WS, args.out), "w", encoding="utf-8") as f:
        json.dump({"config": {k: v for k, v in cfg.items()
                              if not k.startswith("_")},
                   "train_sites": [int(s) for s in tr_sites],
                   "test_sites": [int(s) for s in te_sites],
                   "results": results, "curves": curves}, f, indent=2)
    log("saved -> " + args.out)


if __name__ == "__main__":
    main()
