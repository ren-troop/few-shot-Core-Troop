"""Experiment 1b -- direct test of mechanism M3.

M3 predicts that the task-specific part of an inner-loop SSL term is O(alpha*m),
so its (harmful) effect should shrink as alpha*m grows, while a *frozen-teacher*
consistency term (which is task-independent by construction) should stay neutral.

One MAML backbone is trained at alpha=0.4 / 1 step; at meta-test we sweep the
adaptation step size and the number of adaptation steps.
"""
import sys, json, os, time
import numpy as np

# 仓库根目录：ssl_maml/experiments/xxx.py 往上三层
WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WS)
os.makedirs(os.path.join(WS, "results", "ssl_maml"), exist_ok=True)  # 结果 JSON（可入库）
os.makedirs(os.path.join(WS, "outputs_ssl_maml"), exist_ok=True)     # 运行日志（outputs*/ 已 gitignore）

from ssl_maml.core import (ce_grad, softmax, accuracy, inner_adapt, combine_grads,
                     p_scale)
from ssl_maml.ssl import make_entropy_term
from ssl_maml.tasks import sample_gauss_task
from ssl_maml.experiments import exp1_synth as E


def make_frozen_teacher_term(net, X_u, P0, weight=1.0):
    """Consistency toward the frozen meta-initialisation (task-independent)."""
    T = softmax(net.forward(P0, X_u))

    def g(P):
        z = net.forward(P, X_u)
        p = softmax(z)
        n = len(p)
        loss = weight * float((-(T * np.log(np.clip(p, 1e-12, None))).sum(1)).mean())
        dZ = weight * (p - T) / n
        return loss, net.backward(P, X_u, dZ)
    return g


def main():
    cfg = dict(
        n_way=5, dim=8, K=1, Q=15, radius=1.0, sigma=0.35, warp=0.4, noise=1.0,
        hidden=[40, 40], last_scale=0.1, seed=0,
        alpha=0.4, inner_steps=1, eval_steps=3,
        meta_lr=1e-3, meta_batch=16, iters=2500, n_test_tasks=400,
        lam=1.0, proto_tau=0.1, lp_k=12, lp_alpha=0.9,
        st_rounds=2, st_lr=3e-4, st_steps=150,
        curve_every=250, curve_tasks=12,
        eval_methods=[],
        test_seed=12345,
    )
    quick = "--quick" in sys.argv
    if quick:
        cfg["iters"] = 100
        cfg["n_test_tasks"] = 60

    print("training MAML backbone (alpha=0.4, 1 step) ...", flush=True)
    t0 = time.time()
    net, P0 = E.train("maml2", cfg)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(cfg["test_seed"])
    tasks = [E.sample(rng, cfg) for _ in range(cfg["n_test_tasks"])]

    rows = []
    for m in (1, 3, 8):
        for alpha in (0.05, 0.1, 0.2, 0.4, 0.8):
            acc = {"sup": [], "ent": [], "frozen": []}
            for t in tasks:
                Xs, ys = t.Xs, t.ys
                base = lambda P: ce_grad(net, P, Xs, ys)
                # supervised inner loop
                th = inner_adapt(P0, base, alpha, m)
                acc["sup"].append(accuracy(net, th[-1], t.Xq, t.yq))
                # + entropy minimisation on the unlabeled pool
                ent = make_entropy_term(net, t.Xu)

                def inner(P, base=base, ent=ent):
                    l1, g1 = base(P)
                    l2, g2 = ent(P)
                    return combine_grads((l1, g1),
                                         (cfg["lam"] * l2, p_scale(g2, cfg["lam"])))
                th = inner_adapt(P0, inner, alpha, m)
                acc["ent"].append(accuracy(net, th[-1], t.Xq, t.yq))
                # + consistency toward the frozen initialisation
                ft = make_frozen_teacher_term(net, t.Xu, P0, cfg["lam"])
                inner = lambda P: combine_grads(base(P), ft(P))
                th = inner_adapt(P0, inner, alpha, m)
                acc["frozen"].append(accuracy(net, th[-1], t.Xq, t.yq))
            rows.append(dict(m=m, alpha=alpha,
                             sup=float(np.mean(acc["sup"])),
                             ent=float(np.mean(acc["ent"])),
                             frozen=float(np.mean(acc["frozen"]))))
            r = rows[-1]
            print(f"  m={m} alpha={alpha:<4} | sup={r['sup']*100:6.2f} "
                  f"| +entropy={r['ent']*100:6.2f} ({(r['ent']-r['sup'])*100:+6.2f}) "
                  f"| +frozen-teacher={r['frozen']*100:6.2f} "
                  f"({(r['frozen']-r['sup'])*100:+6.2f})", flush=True)

    out = os.path.join(WS, "results", "ssl_maml", "exp1b_alpha_sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": {k: v for k, v in cfg.items() if k != "eval_methods"},
                   "rows": rows}, f, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
