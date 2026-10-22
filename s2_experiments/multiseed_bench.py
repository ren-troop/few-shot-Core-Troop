# -*- coding: utf-8 -*-
"""
多 seed + 配对比较：给三段式的结论加上置信区间
================================================================================
单 seed 的问题：
   - 只反映"30 个评测任务之间"的波动；
   - 不反映"换一批训练/测试站点"的波动；
   - 最小的效应（长江 h=1: -0.0131）落在噪声边缘。

本脚本做两件事：
  1) 【配对比较】同一批任务、同一批 support/query 抽样下，先在每个任务上
     d_i = MSE_C(i) - MSE_B(i)，再统计 mean(d_i) 和 SE(d_i)。
     （不配对的算法会把"任务难易"的波动算进误差里，误差棒虚高。）
  2) 【多 seed】seed = 0..4，每个 seed 重新划分站点、重新元训练、重新抽样。
     跨 seed 的离散度，才是"换一批数据还成不成立"的答案。

实验组（4 个，去掉已被证明无信息的两组）：
    A_baseline   100 步纯监督
    B_sameopt    400 步纯监督，一个 Adam     <- 对照：排除"多训练"
    C_sameopt    100 纯监督 + 300 联合，一个 Adam <- 三段式（同优化器）
    D_loose      400 联合，一个 Adam

输出：
    每个配置 x 每个 seed 的 SSLnet 配对均值 + 配对 SE，
    以及跨 seed 的均值 + 跨 seed SE + 为正的 seed 数。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = r"C:\Users\wjx\Desktop\ssl-maml-fewshot"
WS = r"C:\Users\wjx\Desktop\半监督研究"
sys.path.insert(0, REPO)

from ssl_maml.core import (MLP, mse_grad, p_clone, p_add, p_zeros_like,
                           inner_adapt, maml_meta_grad, Adam)

OUT = os.path.join(WS, "multiseed_results.json")

FEATS = ["prec", "temp", "tmax", "et0", "soil", "wind"]
HORIZONS = [1, 7, 30]
DATASETS = [("HuangHe", "yrb_daily.csv"), ("ChangJiang", "yangtze_daily.csv")]
SEEDS = [0, 1]

K_SUP = 20
Q_QRY = 50
ADAPT_STEPS = 100
ADAPT_LR = 3e-3
REFINE_STEPS = 300
REFINE_LR = 3e-3
LAM = 1.0
EMA_BETA = 0.99
RAMP_FRAC = 0.3
BATCH_L = 32
BATCH_U = 128
META_ITERS = 1000
META_BATCH = 8
META_LR = 1e-3
ALPHA = 0.01
INNER_STEPS = 1
HIDDEN = [32, 32]
N_EVAL = 30
TRAIN_FRAC = 0.7
MIN_DAYS = 300
GATE_R2 = 0.2

ARMS = ["A_baseline", "B_sameopt", "C_sameopt", "D_loose"]

# ---------------------------------------------------------------- 数据
_CACHE = {}


def load(csv, h):
    key = (csv, h)
    if key in _CACHE:
        return _CACHE[key]
    df = pd.read_csv(os.path.join(WS, csv), encoding="utf-8-sig")
    df["discharge"] = pd.to_numeric(df["discharge"], errors="coerce")
    df = df.dropna(subset=["discharge"]).copy()
    df["ly"] = np.log1p(df["discharge"].clip(lower=0))
    sites = sorted(df["site"].unique())
    dates = sorted(df["date"].unique())
    nS, nD = len(sites), len(dates)
    F = np.full((nS, nD, len(FEATS) + 1), np.nan)
    for k, f in enumerate(FEATS):
        piv = df.pivot_table(index="site", columns="date", values=f,
                             aggfunc="mean").reindex(index=sites,
                                                     columns=dates)
        v = np.array(piv.to_numpy(), float, copy=True)
        F[:, :, k] = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
    piv = df.pivot_table(index="site", columns="date", values="ly",
                         aggfunc="mean").reindex(index=sites, columns=dates)
    LY = np.array(piv.to_numpy(), float, copy=True)
    LY = (LY - np.nanmean(LY)) / (np.nanstd(LY) + 1e-12)
    F[:, :, -1] = LY
    y = np.full((nS, nD), np.nan)
    y[:, :-h] = LY[:, h:]
    _CACHE[key] = (F, y, sites)
    return F, y, sites


def pools_of(F, y, r):
    ok = (~np.isnan(F[r]).any(1)) & (~np.isnan(y[r]))
    idx = np.where(ok)[0]
    if len(idx) < MIN_DAYS:
        return None, None
    cut = int(len(idx) * TRAIN_FRAC)
    return idx[:cut], idx[cut:]


# ---------------------------------------------------------------- 训练
def make_net(n_feat, seed):
    return MLP([n_feat] + HIDDEN + [1], seed=seed, last_scale=1.0)


def mse_term(net, P, X, T, w=1.0):
    z = net.forward(P, X).ravel()
    n = len(z)
    d = z - T
    return (w * float((d ** 2).mean()),
            net.backward(P, X, (2.0 * w * d / n).reshape(-1, 1)))


def ramp(s, total, warm=RAMP_FRAC):
    return min(1.0, s / max(1.0, warm * total))


def predict(net, P, X):
    return net.forward(P, X).ravel()


def train_phase(net, P, opt, Xl, yl, Xu, rng, n_pure, n_joint):
    Pe = p_clone(P)
    nl = len(yl)
    nu = 0 if Xu is None else len(Xu)
    for s in range(n_pure + n_joint):
        bl = rng.choice(nl, size=min(BATCH_L, nl), replace=False)
        _, dP = mse_grad(net, P, Xl[bl], yl[bl])
        if s >= n_pure and nu > 0:
            bu = rng.choice(nu, size=min(BATCH_U, nu), replace=False)
            T = predict(net, Pe, Xu[bu])
            w = LAM * ramp(s - n_pure, n_joint)
            _, dU = mse_term(net, P, Xu[bu], T, w)
            dP = p_add(dP, dU, 1.0)
        P = opt.step(P, dP)
        Pe = [(EMA_BETA * a + (1 - EMA_BETA) * b,
               EMA_BETA * c + (1 - EMA_BETA) * d)
              for (a, c), (b, d) in zip(Pe, P)]
    return P


def eval_arm(net, P0, F, y, r, tr, te, rng, arm):
    sup = rng.choice(tr, size=min(K_SUP, len(tr)), replace=False)
    qry = rng.choice(te, size=min(Q_QRY, len(te)), replace=False)
    unl = np.setdiff1d(np.arange(len(y[r])), np.concatenate([sup, qry]))
    unl = unl[~np.isnan(F[r][unl]).any(1) & ~np.isnan(y[r][unl])]
    Xl, yl = F[r][sup], y[r][sup].reshape(-1, 1)
    Xu, Xq, yq = F[r][unl], F[r][qry], y[r][qry]

    if arm == "A_baseline":
        P = p_clone(P0)
        P = train_phase(net, P, Adam(P, lr=ADAPT_LR), Xl, yl, None, rng,
                        ADAPT_STEPS, 0)
    elif arm == "B_sameopt":
        P = p_clone(P0)
        P = train_phase(net, P, Adam(P, lr=ADAPT_LR), Xl, yl, None, rng,
                        ADAPT_STEPS + REFINE_STEPS, 0)
    elif arm == "C_sameopt":
        P = p_clone(P0)
        P = train_phase(net, P, Adam(P, lr=ADAPT_LR), Xl, yl, Xu, rng,
                        ADAPT_STEPS, REFINE_STEPS)
    elif arm == "D_loose":
        P = p_clone(P0)
        P = train_phase(net, P, Adam(P, lr=REFINE_LR), Xl, yl, Xu, rng,
                        0, ADAPT_STEPS + REFINE_STEPS)
    else:
        raise ValueError(arm)
    return float(np.mean((predict(net, P, Xq) - yq) ** 2))


def meta_train(F, y, tr, pools, seed):
    net = make_net(F.shape[2], seed)
    P = net.P
    opt = Adam(P, lr=META_LR)
    rng = np.random.default_rng(seed + 1)
    for _ in range(META_ITERS):
        g = p_zeros_like(P)
        for _ in range(META_BATCH):
            r = int(rng.choice(tr))
            a, b = pools[r]
            s = rng.choice(a, size=min(K_SUP, len(a)), replace=False)
            q = rng.choice(b, size=min(Q_QRY, len(b)), replace=False)
            Xs, ys_ = F[r][s], y[r][s].reshape(-1, 1)
            Xq, yq = F[r][q], y[r][q].reshape(-1, 1)
            inn = (lambda Q, X=Xs, Y=ys_: mse_grad(net, Q, X, Y))
            out = (lambda Q, X=Xq, Y=yq: mse_grad(net, Q, X, Y))
            th = inner_adapt(P, inn, ALPHA, INNER_STEPS)
            g = p_add(g, maml_meta_grad(th, out, inn, ALPHA,
                                        second_order=True), 1.0 / META_BATCH)
        P = opt.step(P, g)
    return net, P


def run_one(name, csv, h, seed):
    F, y, sites = load(csv, h)
    pools = {i: pools_of(F, y, i) for i in range(len(sites))}
    ok = [i for i in range(len(sites)) if pools[i][0] is not None]
    if len(ok) < 6:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ok)
    n_te = max(2, int(0.25 * len(perm)))
    te = [int(x) for x in perm[:n_te]]
    tr = [int(x) for x in perm[n_te:]]

    net, P0 = meta_train(F, y, tr, pools, seed)

    per = {a: [] for a in ARMS}
    for _ in range(N_EVAL):
        r = int(rng.choice(te))
        a, b = pools[r]
        for arm in ARMS:
            # 每支都用同一个 rng 起点 -> 同一批 support/query -> 可配对
            per[arm].append(eval_arm(net, P0, F, y, r, a, b,
                                     np.random.default_rng(seed + 99), arm))
    arr = {a: np.array(per[a]) for a in ARMS}
    out = {"seed": seed, "n_sites": len(ok), "n_test": len(te)}
    for a in ARMS:
        out[a] = dict(mse=float(arr[a].mean()),
                      se=float(arr[a].std(ddof=1) / np.sqrt(len(arr[a]))))
    out["A_r2"] = float(1 - out["A_baseline"]["mse"])

    # ---- 配对差 ----
    for tag, x, yarm in [("ssl_net", "C_sameopt", "B_sameopt"),
                         ("three_vs_base", "C_sameopt", "A_baseline"),
                         ("loose_vs_base", "D_loose", "A_baseline")]:
        d = arr[x] - arr[yarm]
        out[tag] = dict(mean=float(d.mean()),
                        se=float(d.std(ddof=1) / np.sqrt(len(d))),
                        n=int(len(d)))
    return out


def main():
    allres = {}
    for name, csv in DATASETS:
        for h in HORIZONS:
            key = "%s_h%d" % (name, h)
            allres[key] = []
            for sd in SEEDS:
                t0 = time.time()
                r = run_one(name, csv, h, sd)
                if r is None:
                    continue
                allres[key].append(r)
                print("  %-16s seed=%d  baseR2=%.3f  ssl_net=%+.4f (pairSE %.4f)  %.0fs"
                      % (key, sd, r["A_r2"], r["ssl_net"]["mean"],
                         r["ssl_net"]["se"], time.time() - t0), flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(allres, f, indent=2, ensure_ascii=False)

    print("")
    print("=" * 90)
    print("多 seed 汇总 (SSLnet = C_sameopt - B_sameopt, 负=半监督有效)")
    print("=" * 90)
    print("  %-16s %8s %12s %10s %10s %10s"
          % ("config", "baseR2", "mean SSLnet", "seedSE", "neg seeds", "verdict"))
    print("  " + "-" * 82)
    for key, rs in allres.items():
        if not rs:
            continue
        vals = np.array([r["ssl_net"]["mean"] for r in rs])
        n_neg = int((vals < 0).sum())
        se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        base = float(np.mean([r["A_r2"] for r in rs]))
        verdict = "有效" if (vals.mean() < 0 and n_neg >= max(3, len(vals) * 0.6)) else (
            "边缘" if n_neg >= 1 else "无效")
        print("  %-16s %8.3f %12.4f %10.4f %7d/%-2d %10s"
              % (key, base, vals.mean(), se, n_neg, len(vals), verdict))
    print("")
    print("saved -> " + OUT)


if __name__ == "__main__":
    main()
