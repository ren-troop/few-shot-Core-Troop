"""Experiment 1 -- synthetic clustered N-way few-shot classification.

Question: where should the semi-supervised signal enter?
  (a) inside the MAML inner loop            -- merged variant
  (b) after adaptation (three-stage)        -- proposed pipeline
Training backbones: joint pretraining / MAML (2nd order) / FOMAML /
inner-loop SSL variants.  Plus a random-initialisation control.
"""
import sys, time, json, argparse, os
import numpy as np

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
os.makedirs(os.path.join(WS, "results"), exist_ok=True)   # 输出目录（打包时改名为 results）

from ml.core import (MLP, ce_grad, p_add, p_scale, p_zeros_like, p_clone,
                     inner_adapt, maml_meta_grad, Adam, combine_grads, accuracy)
from ml.ssl import (make_entropy_term, make_proto_distill_term,
                    make_lp_distill_term, label_propagation, prototype_logits,
                    selftrain)
from ml.tasks import sample_gauss_task


def scale_term(pair, w):
    l, g = pair
    return w * l, p_scale(g, w)


def make_net(cfg, seed=None):
    return MLP([cfg["dim"]] + list(cfg["hidden"]) + [cfg["n_way"]],
               seed=cfg["seed"] if seed is None else seed,
               last_scale=cfg["last_scale"])


def make_inner(net, task, kind, cfg):
    Xs, ys = task.Xs, task.ys
    base = lambda P: ce_grad(net, P, Xs, ys)
    if kind in ("maml2", "fomaml", "sup", "joint"):
        return base
    if kind == "ssl_ent":
        U = make_entropy_term(net, task.Xu)
        return lambda P: combine_grads(base(P), scale_term(U(P), cfg["lam"]))
    if kind == "ssl_proto":
        U = make_proto_distill_term(net, Xs, ys, task.Xu, tau=cfg["proto_tau"],
                                    weight=cfg["lam"])
        return lambda P: combine_grads(base(P), U(P))
    if kind == "ssl_lp":
        U = make_lp_distill_term(net, Xs, ys, task.Xu, task.n_way,
                                 k=cfg["lp_k"], alpha=cfg["lp_alpha"],
                                 weight=cfg["lam"])
        return lambda P: combine_grads(base(P), U(P))
    raise ValueError(kind)


def sample(rng, cfg):
    return sample_gauss_task(rng, n_way=cfg["n_way"], dim=cfg["dim"], K=cfg["K"],
                             Q=cfg["Q"], radius=cfg["radius"], sigma=cfg["sigma"],
                             warp=cfg["warp"], noise=cfg["noise"])


# ------------------------------------------------------------- meta-training
def train(kind, cfg, log=None, curve=None):
    net = make_net(cfg)
    P = net.P
    opt = Adam(P, lr=cfg["meta_lr"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    second_order = (kind == "maml2")
    t0 = time.time()
    for it in range(cfg["iters"]):
        gsum = p_zeros_like(P)
        for _ in range(cfg["meta_batch"]):
            task = sample(rng, cfg)
            if kind == "joint":
                _, g = ce_grad(net, P, task.Xs, task.ys)
            else:
                inner = make_inner(net, task, kind, cfg)
                outer = (lambda Q, t=task: ce_grad(net, Q, t.Xq, t.yq))
                thetas = inner_adapt(P, inner, cfg["alpha"], cfg["inner_steps"])
                g = maml_meta_grad(thetas, outer, inner, cfg["alpha"],
                                   second_order=second_order)
            gsum = p_add(gsum, g, 1.0 / cfg["meta_batch"])
        P = opt.step(P, gsum)
        if curve is not None and (it + 1) % cfg["curve_every"] == 0:
            r2 = np.random.default_rng(4242)
            ts = [sample(r2, cfg) for _ in range(cfg["curve_tasks"])]
            a1 = np.mean([evaluate_task(net, P, t, "sup", cfg) for t in ts])
            a2 = np.mean([evaluate_task(net, P, t, "sup_then_lp", cfg) for t in ts])
            curve.append({"iter": it + 1, "sup": float(a1), "sup_then_lp": float(a2)})
            if log:
                log(f"    [{kind}] it {it+1}: sup={a1*100:.2f} +LP={a2*100:.2f}"
                    f"  ({time.time()-t0:.0f}s)")
        elif log is not None and (it + 1) % max(1, cfg["iters"] // 4) == 0:
            log(f"    [{kind}] iter {it+1}/{cfg['iters']}  {time.time()-t0:.0f}s")
    return net, P


# --------------------------------------------------------------------- eval
EVAL_METHODS = ["sup", "sup_ent", "sup_proto", "sup_lp", "sup_then_lp",
                "sup_then_st", "sup_then_sthead", "sup_then_proto", "raw_lp"]


def evaluate_task(net, P0, task, method, cfg):
    Xs, ys, Xq, yq, Xu = task.Xs, task.ys, task.Xq, task.yq, task.Xu
    steps = cfg["eval_steps"]
    if method == "raw_lp":
        F = np.vstack([Xs, Xq])
        ns = len(Xs)
        Y = label_propagation(F, ys, np.arange(ns), task.n_way, k=cfg["lp_k"],
                              alpha=cfg["lp_alpha"])
        return float((Y[ns:].argmax(1) == yq).mean())
    if method in ("sup", "sup_ent", "sup_proto", "sup_lp"):
        kind = {"sup": "sup", "sup_ent": "ssl_ent", "sup_proto": "ssl_proto",
                "sup_lp": "ssl_lp"}[method]
        inner = make_inner(net, task, kind, cfg)
        thetas = inner_adapt(P0, inner, cfg["alpha"], steps)
        return accuracy(net, thetas[-1], Xq, yq)

    thetas = inner_adapt(P0, lambda P: ce_grad(net, P, Xs, ys), cfg["alpha"], steps)
    Pf = thetas[-1]
    F = net.features(Pf, np.vstack([Xs, Xq]))
    ns = len(Xs)
    idx = np.arange(ns)
    if method == "sup_then_lp":
        Y = label_propagation(F, ys, idx, task.n_way, k=cfg["lp_k"],
                              alpha=cfg["lp_alpha"])
        return float((Y[ns:].argmax(1) == yq).mean())
    if method == "sup_then_proto":
        p = prototype_logits(F, ys, idx, task.n_way, tau=cfg["proto_tau"])
        return float((p[ns:].argmax(1) == yq).mean())
    if method in ("sup_then_st", "sup_then_sthead"):
        Pr = selftrain(net, Pf, np.vstack([Xs, Xq]), ys, idx, task.n_way,
                       rounds=cfg["st_rounds"], steps=cfg["st_steps"],
                       lr=cfg["st_lr"], use_lp=True, k=cfg["lp_k"],
                       alpha=cfg["lp_alpha"], only_last=(method == "sup_then_sthead"))
        return accuracy(net, Pr, Xq, yq)
    raise ValueError(method)


def run_eval(net, P0, cfg):
    rng = np.random.default_rng(cfg["test_seed"])
    tasks = [sample(rng, cfg) for _ in range(cfg["n_test_tasks"])]
    out = {}
    for m in cfg["eval_methods"]:
        accs = [evaluate_task(net, P0, t, m, cfg) for t in tasks]
        out[m] = [float(np.mean(accs)), float(np.std(accs) / np.sqrt(len(accs)))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--kinds", default="rand,joint,maml2,fomaml,ssl_ent,ssl_proto,ssl_lp")
    ap.add_argument("--out", default="results/exp1_synth.json")
    args = ap.parse_args()

    cfg = dict(
        n_way=5, dim=8, K=1, Q=15, radius=1.0, sigma=0.35, warp=0.4, noise=1.0,
        hidden=[40, 40], last_scale=0.1, seed=0,
        alpha=0.4, inner_steps=1, eval_steps=3,
        meta_lr=1e-3, meta_batch=16, iters=2500, n_test_tasks=400,
        lam=1.0, proto_tau=0.1, lp_k=12, lp_alpha=0.9,
        st_rounds=2, st_lr=3e-4, st_steps=150,
        curve_every=250, curve_tasks=12,
        eval_methods=EVAL_METHODS, test_seed=12345,
    )
    if args.quick:
        cfg["iters"] = 100
        cfg["n_test_tasks"] = 40
        cfg["curve_every"] = 50
        cfg["curve_tasks"] = 8

    logf = open(os.path.join(WS, "results", "exp1_log.txt"), "a", encoding="utf-8")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())

    log("=" * 70)
    log("exp1 config: " + json.dumps({k: v for k, v in cfg.items()
                                      if k not in ("eval_methods",)}))
    results, curves = {}, {}
    for kind in args.kinds.split(","):
        t0 = time.time()
        curve = []
        log(f"  training {kind} ...")
        if kind == "rand":
            net = make_net(cfg, seed=cfg["seed"] + 777)
            P = net.P
        else:
            net, P = train(kind, cfg, log=log, curve=curve)
        res = run_eval(net, P, cfg)
        results[kind] = res
        curves[kind] = curve
        log(f"  == {kind} (train {time.time()-t0:.0f}s)")
        for m, (mu, se) in res.items():
            log(f"       {m:16s} acc = {mu*100:6.2f} +- {se*100:.2f}")

    with open(os.path.join(WS, args.out), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "results": results, "curves": curves}, f, indent=2)
    log("saved -> " + args.out)


if __name__ == "__main__":
    main()
