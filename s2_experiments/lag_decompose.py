# -*- coding: utf-8 -*-
"""
三段式效果的因果分解：加"同一优化器"对照组
================================================================================
原有问题：严格三段式(C) 比 联合版(D) 差，但 C 在第二段和第三段之间
          新建了 Adam（动量归零），所以差距里可能混着"优化器重启"的效应。

本脚本用 6 个实验组把三个效应拆开（总步数全部相同 = 100+300）：

    A_baseline      100 步纯监督                     -> 预测
    B_restart       100 纯监督 + 300 纯监督，Adam 重启
    B_sameopt       400 纯监督，一个 Adam 连续跑
    C_restart       100 纯监督 + 300 联合，Adam 重启          <- 原来的严格三段式
    C_sameopt       100 纯监督 + 300 联合，一个 Adam 连续跑    <- 新增对照
    D_loose         400 联合，一个 Adam 连续跑                <- 原来的联合版

三个关键对比：
    C_restart  - C_sameopt   = Adam 重启的效应
    C_sameopt  - B_sameopt   = 半监督的净效应（结构与优化器都固定）
    D_loose    - C_sameopt   = 联合 vs 分离（优化器/总步数/标签预算全一致）
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

OUT = os.path.join(WS, "lag_decompose_results.json")

FEATS = ["prec", "temp", "tmax", "et0", "soil", "wind"]
HORIZONS = [1, 7, 30]
DATASETS = [("HuangHe", "yrb_daily.csv"), ("ChangJiang", "yangtze_daily.csv")]

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
SEED = 0
TRAIN_FRAC = 0.7
MIN_DAYS = 300
GATE_R2 = 0.2

ARMS = ["A_baseline", "B_restart", "B_sameopt",
        "C_restart", "C_sameopt", "D_loose"]


# ------------------------------------------------------------------ 数据
def load(csv, h):
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
    return F, y, sites


def pools_of(F, y, r):
    ok = (~np.isnan(F[r]).any(1)) & (~np.isnan(y[r]))
    idx = np.where(ok)[0]
    if len(idx) < MIN_DAYS:
        return None, None
    cut = int(len(idx) * TRAIN_FRAC)
    return idx[:cut], idx[cut:]


# ------------------------------------------------------------------ 训练
def make_net(n_feat):
    return MLP([n_feat] + HIDDEN + [1], seed=SEED, last_scale=1.0)


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
    """在给定参数和优化器上继续训练：先 n_pure 步纯监督，再 n_joint 步联合。"""
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
    elif arm == "B_restart":
        P = p_clone(P0)
        P = train_phase(net, P, Adam(P, lr=ADAPT_LR), Xl, yl, None, rng,
                        ADAPT_STEPS, 0)
        P = p_clone(P)
        P = train_phase(net, P, Adam(P, lr=REFINE_LR), Xl, yl, None, rng,
                        REFINE_STEPS, 0)
    elif arm == "B_sameopt":
        P = p_clone(P0)
        P = train_phase(net, P, Adam(P, lr=ADAPT_LR), Xl, yl, None, rng,
                        ADAPT_STEPS + REFINE_STEPS, 0)
    elif arm == "C_restart":
        P = p_clone(P0)
        P = train_phase(net, P, Adam(P, lr=ADAPT_LR), Xl, yl, None, rng,
                        ADAPT_STEPS, 0)
        P = p_clone(P)
        P = train_phase(net, P, Adam(P, lr=REFINE_LR), Xl, yl, Xu, rng,
                        0, REFINE_STEPS)
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


def meta_train(F, y, tr, pools):
    net = make_net(F.shape[2])
    P = net.P
    opt = Adam(P, lr=META_LR)
    rng = np.random.default_rng(SEED + 1)
    t0 = time.time()
    for it in range(META_ITERS):
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
        if (it + 1) % 500 == 0:
            print("      iter %d/%d (%.0fs)" % (it + 1, META_ITERS,
                                                time.time() - t0))
    return net, P


def run_one(name, csv, h, log=print):
    log("")
    log("-" * 68)
    log("[%s]  h = %d" % (name, h))
    log("-" * 68)
    F, y, sites = load(csv, h)
    pools = {i: pools_of(F, y, i) for i in range(len(sites))}
    ok = [i for i in range(len(sites)) if pools[i][0] is not None]
    if len(ok) < 6:
        log("  可用点位只有 %d 个, 跳过" % len(ok))
        return None
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(ok)
    n_te = max(2, int(0.25 * len(perm)))
    te = [int(x) for x in perm[:n_te]]
    tr = [int(x) for x in perm[n_te:]]
    log("  可用点位 %d (训练 %d / 测试 %d)" % (len(ok), len(tr), len(te)))

    net, P0 = meta_train(F, y, tr, pools)
    res = {a: [] for a in ARMS}
    for _ in range(N_EVAL):
        r = int(rng.choice(te))
        a, b = pools[r]
        for arm in ARMS:
            res[arm].append(eval_arm(net, P0, F, y, r, a, b,
                                     np.random.default_rng(SEED + 99), arm))
    out = {}
    for arm in ARMS:
        v = np.array(res[arm])
        out[arm] = dict(mse=float(v.mean()),
                        se=float(v.std(ddof=1) / np.sqrt(len(v))),
                        r2=float(1 - v.mean()))
    b0 = out["A_baseline"]["mse"]
    log("  %-14s %14s %10s %8s" % ("arm", "MSE", "vs A", "R2"))
    log("  " + "-" * 50)
    for arm in ARMS:
        o = out[arm]
        log("  %-14s %8.4f+-%.4f %10s %8.3f"
            % (arm, o["mse"], o["se"],
               "-" if arm == "A_baseline" else "%+.4f" % (o["mse"] - b0),
               o["r2"]))

    # ---- 三个关键对比 ----
    cr = out["C_restart"]["mse"]
    cs = out["C_sameopt"]["mse"]
    bs = out["B_sameopt"]["mse"]
    dl = out["D_loose"]["mse"]
    log("")
    log("  分解:")
    log("    C_restart - C_sameopt = %+.4f   (Adam restart effect)"
        % (cr - cs))
    log("    C_sameopt - B_sameopt = %+.4f   (SSL net effect)"
        % (cs - bs))
    log("    D_loose   - C_sameopt = %+.4f   (joint vs separated)"
        % (dl - cs))
    out["decomp"] = dict(adam_restart=float(cr - cs),
                         ssl_net=float(cs - bs),
                         joint_vs_sep=float(dl - cs))
    out["healthy"] = bool(out["A_baseline"]["r2"] > GATE_R2)
    return out


def main():
    res = {}
    for name, csv in DATASETS:
        for h in HORIZONS:
            r = run_one(name, csv, h)
            if r:
                res["%s_h%d" % (name, h)] = r
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("")
    print("=" * 84)
    print("汇总: 三个效应分解 (MSE 变化, 负=更好)")
    print("=" * 84)
    print("  %-18s %8s %12s %10s %12s %14s"
          % ("config", "baseR2", "AdamRestart", "SSLnet", "JointVsSep",
             "C_restart-A"))
    print("  " + "-" * 80)
    for k, v in res.items():
        d = v["decomp"]
        print("  %-18s %8.3f %+12.4f %+10.4f %+12.4f %+14.4f"
              % (k, v["A_baseline"]["r2"], d["adam_restart"], d["ssl_net"],
                 d["joint_vs_sep"],
                 v["C_restart"]["mse"] - v["A_baseline"]["mse"]))
    print("")
    print("  AdamRestart = C_restart - C_sameopt  (优化器重启的影响)")
    print("  SSLnet      = C_sameopt - B_sameopt  (半监督净效应)")
    print("  JointVsSep  = D_loose - C_sameopt    (联合 vs 分离)")
    print("")
    print("saved -> " + OUT)


if __name__ == "__main__":
    main()
