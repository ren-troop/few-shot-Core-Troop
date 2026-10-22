# -*- coding: utf-8 -*-
"""
单文件自包含版本：验证 PL / VAT / FixMatch 放进 MAML inner loop 是否有效。

数据：water_potability.csv（置于 ../data_potability/ 下）
任务：MAML 2-way few-shot 分类（9 维水质特征 -> Potability 0/1）

三种半监督方法（都在 inner loop 的无标签集 ux 上起作用）：
  1) PL（Pseudo-Labeling）：模型给无标签样本打硬伪标签（置信度>tau）再训练
  2) VAT（虚拟对抗训练）：找让输出最"不稳定"的对抗方向 r，让 f(x)≈f(x+ξr)
  3) FixMatch：弱增强出伪标签 -> 强增强(掩码+大噪声)做一致性训练

每个 seed 下 baseline 与三方法共享相同模型初始化 + 相同任务序列，
因此可配对；跨 5 个 seed 做配对 t 检验。

运行：
  python innerloop_ssl_standalone.py           # 完整实验（4方法 x 5种子）
  python innerloop_ssl_standalone.py --quick    # 快速冒烟（单种子/少迭代）
输出：innerloop_ssl_standalone_results.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from scipy.stats import ttest_rel

# 数据路径：相对本脚本所在目录
HERE = os.path.dirname(os.path.abspath(__file__))
POTABILITY = os.path.join(HERE, '..', 'data_potability', 'water_potability.csv')

FEATURES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_potability(csv_path):
    df = pd.read_csv(csv_path)
    for c in FEATURES:
        df[c] = df[c].fillna(df[c].median())          # 缺失值中位数填充
    X = df[FEATURES].values.astype(np.float32)
    y = df['Potability'].values.astype(np.int64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)           # 标准化
    return np.ascontiguousarray(X), np.ascontiguousarray(y)


def split_pool(X, y, seed=0, test_frac=0.3):
    """meta-train / meta-test 两个不重叠样本池。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_te = int(len(X) * test_frac)
    te, tr = idx[:n_te], idx[n_te:]
    return X[tr], y[tr], X[te], y[te]


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# 任务采样：支撑集(K有标签) + 无标签集(U) + 查询集(Q)
# --------------------------------------------------------------------------- #
def make_task_sampler(X, y, K, U, Q, seed):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    qpc = Q // len(classes)

    def sample():
        sx, sy, qx, qy = [], [], [], []
        for c in classes:
            c_idx = np.where(y == c)[0]
            pick = rng.choice(c_idx, K + qpc, replace=False)
            sx.append(X[pick[:K]]); sy.append(y[pick[:K]])
            qx.append(X[pick[K:]]); qy.append(y[pick[K:]])
        sx = np.concatenate(sx); sy = np.concatenate(sy)
        qx = np.concatenate(qx); qy = np.concatenate(qy)
        u_idx = rng.choice(len(X), U, replace=False)
        ux = X[u_idx]
        return (torch.from_numpy(sx), torch.from_numpy(sy).long(),
                torch.from_numpy(ux), torch.from_numpy(qx), torch.from_numpy(qy).long())

    return sample


# --------------------------------------------------------------------------- #
# 损失函数
# --------------------------------------------------------------------------- #
def supervised_loss(out, y):
    return F.cross_entropy(out, y)


def _strong_aug(x):
    """强增强：随机掩码 20% 特征 + 较大高斯噪声(σ=0.1)。"""
    mask = (torch.rand_like(x) > 0.2).float()
    return x * mask + torch.randn_like(x) * 0.1


def pseudo_label_loss(model, p, x, tau):
    out = functional_call(model, p, x)
    prob = torch.softmax(out, dim=1)
    conf, pseudo = prob.max(1)
    mask = conf >= tau
    if mask.sum() == 0:
        return torch.zeros((), device=x.device)
    return F.cross_entropy(out[mask], pseudo[mask])


def vat_loss(model, p, x, xi=0.3, ip=1):
    def divergence(xx, x_ref):
        out = functional_call(model, p, xx)
        ref = torch.softmax(x_ref.detach(), dim=1)
        logp = torch.log_softmax(out, dim=1)
        return (ref * (torch.log(ref + 1e-8) - logp)).sum(1).mean()

    ref_out = functional_call(model, p, x)
    r = xi * F.normalize(torch.randn_like(x).view(len(x), -1), dim=1).view_as(x)
    for _ in range(ip):
        r = r.detach().requires_grad_(True)
        g = torch.autograd.grad(divergence(x + r, ref_out), r)[0]
        g = g.view(len(x), -1)
        g = g / (g.norm(dim=1, keepdim=True) + 1e-8)
        r = xi * g.view_as(x)
    return divergence(x + r.detach(), ref_out)


def fixmatch_loss(model, p, x, tau):
    xw = x + torch.randn_like(x) * 0.01             # 弱增强
    outw = functional_call(model, p, xw)
    prob = torch.softmax(outw, dim=1)
    conf, pseudo = prob.max(1)
    mask = conf >= tau
    if mask.sum() == 0:
        return torch.zeros((), device=x.device)
    outs = functional_call(model, p, _strong_aug(x[mask]))  # 强增强 -> 一致性
    return F.cross_entropy(outs, pseudo[mask])


SEMI_LOSSES = {'PL': pseudo_label_loss, 'VAT': vat_loss,
               'FixMatch': fixmatch_loss}


def semi_loss(method, model, p, x, tau):
    return SEMI_LOSSES[method](model, p, x, tau)


# --------------------------------------------------------------------------- #
# MAML 内层循环：有标签支撑集(监督) + 无标签集(半监督) -> 查询集元损失
# --------------------------------------------------------------------------- #
def inner_loop_meta_loss(model, task, method, cfg):
    sx, sy, ux, qx, qy = task
    p = {n: v.clone() for n, v in model.named_parameters()}
    names = list(p.keys())
    for _ in range(cfg['inner_steps']):
        loss = supervised_loss(functional_call(model, p, sx), sy)
        if method != 'baseline':
            loss = loss + cfg['lam'] * semi_loss(method, model, p, ux, cfg['tau'])
        grads = torch.autograd.grad(loss, list(p.values()), create_graph=True)
        p = {n: p[n] - cfg['inner_lr'] * g for n, g in zip(names, grads)}
    outq = functional_call(model, p, qx)
    return supervised_loss(outq, qy)


def meta_train(model, sampler, method, cfg):
    opt = torch.optim.Adam(model.parameters(), lr=cfg['meta_lr'])
    for it in range(cfg['meta_iters']):
        opt.zero_grad()
        ml = torch.zeros((), dtype=torch.float32)
        for _ in range(cfg['meta_batch']):
            ml = ml + inner_loop_meta_loss(model, sampler(), method, cfg)
        ml = ml / cfg['meta_batch']
        ml.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()


def evaluate(model, sampler, method, cfg, n_tasks):
    model.eval()
    accs = []
    for _ in range(n_tasks):
        sx, sy, ux, qx, qy = sampler()
        p = {n: v.clone() for n, v in model.named_parameters()}
        names = list(p.keys())
        for _ in range(cfg['inner_steps']):
            loss = supervised_loss(functional_call(model, p, sx), sy)
            if method != 'baseline':
                loss = loss + cfg['lam'] * semi_loss(method, model, p, ux, cfg['tau'])
            grads = torch.autograd.grad(loss, list(p.values()))
            p = {n: p[n] - cfg['inner_lr'] * g for n, g in zip(names, grads)}
        with torch.no_grad():
            outq = functional_call(model, p, qx)
            accs.append((outq.argmax(1) == qy).float().mean().item())
    model.train()
    return float(np.mean(accs)), float(np.std(accs))


# --------------------------------------------------------------------------- #
# 主实验
# --------------------------------------------------------------------------- #
def run_one(Xtr, ytr, Xte, yte, method, cfg, seed, eval_tasks):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(cfg['in_dim'], cfg['hidden'])
    sampler = make_task_sampler(Xtr, ytr, cfg['K'], cfg['U'], cfg['Q'], seed)
    meta_train(model, sampler, method, cfg)
    eval_sampler = make_task_sampler(Xte, yte, cfg['K'], cfg['U'], cfg['Q'], seed)
    return evaluate(model, eval_sampler, method, cfg, eval_tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='快速冒烟：单种子少迭代')
    args = ap.parse_args()

    cfg = dict(in_dim=9, hidden=64, K=5, inner_steps=5, inner_lr=0.05,
               meta_lr=1e-3, meta_batch=5, meta_iters=200, lam=1.0,
               U=20, Q=12, xi=0.3, tau=0.5, ip=1)

    methods = ['baseline', 'PL', 'VAT', 'FixMatch']
    seeds = [0, 1, 2, 3, 4]
    eval_tasks = 150
    if args.quick:
        methods = ['baseline', 'PL']
        seeds = [0]
        cfg['meta_iters'] = 20
        eval_tasks = 50
        print('[quick] 快速冒烟模式：baseline+PL / seed=0 / 20 iters')

    if not os.path.exists(POTABILITY):
        raise FileNotFoundError(
            f'找不到数据文件: {POTABILITY}\n请把 water_potability.csv 放到 '
            f'本脚本的上级目录 {os.path.join(HERE, "..", "data_potability")} 下')

    X, y = load_potability(POTABILITY)
    Xtr, ytr, Xte, yte = split_pool(X, y, seed=123)   # 固定切分, 方法间可比

    raw = {m: [] for m in methods}
    for seed in seeds:
        for m in methods:
            t0 = time.time()
            mean, std = run_one(Xtr, ytr, Xte, yte, m, cfg, seed, eval_tasks)
            raw[m].append(mean)
            print(f'[seed{seed}/{m:8s}] acc={mean:.4f}±{std:.4f}  ({time.time()-t0:.1f}s)',
                  flush=True)

    base = np.array(raw['baseline'])
    summary = {}
    for m in ['PL', 'VAT', 'FixMatch']:
        if m not in raw:
            continue
        a = np.array(raw[m])
        diff = a - base
        t, p = ttest_rel(a, base)
        summary[m] = dict(mean=float(a.mean()), std=float(a.std()),
                          diff=float(diff.mean()), diff_std=float(diff.std()),
                          t=float(t), p=float(p), wins=int((a > base).sum()), n=len(a))

    result = dict(cfg=cfg, seeds=seeds, raw=raw, summary=summary)
    out = os.path.join(HERE, 'innerloop_ssl_standalone_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\n===== baseline vs 三方法（配对 t 检验, n=%d）=====' % len(base))
    print(f"baseline   : acc={base.mean():.4f} ± {base.std():.4f}")
    for m in ['PL', 'VAT', 'FixMatch']:
        if m not in summary:
            continue
        s = summary[m]
        print(f"{m:9s}: acc={s['mean']:.4f}±{s['std']:.4f}  "
              f"diff={s['diff']*100:+.2f}pp±{s['diff_std']*100:.2f}  "
              f"p={s['p']:.3f}  胜场 {s['wins']}/{s['n']}")
    print('\n已保存:', out)


if __name__ == '__main__':
    main()