# -*- coding: utf-8 -*-
"""
FixMatch 强增广消融：高斯/掩码 vs Mixup（含 EMA 稳定性修复）。

目的:
  豆包用 Mixup 取代高斯噪声做 FixMatch 的"强增广", 但:
    (1) 回归场景下用"模型自己的连续伪标签"做 Mixup 目标 -> 自参照闭环 -> 不稳定;
    (2) 原代码是回归专用结构(1 维输出 + MSE), 无法直接用于分类 potability;
    (3) lam = max(beta(0.75,0.75)) 每步剧烈抖动, 且逐样本 for 循环 + dtype 混用。

  本脚本在"标签稀缺 + 大量无标签"的原生半监督场景下, 干净地对比:
    baseline      : 纯监督
    fm_gauss      : 强增广 = 随机掩码 + 高斯噪声(分类) / 轻噪声(回归)
    fm_mixup      : 强增广 = Mixup(稳定 lam, 向量化)
    fm_mixup_ema  : 强增广 = Mixup + EMA Teacher(打破回归自参照)

  分类与回归各自用"正确"的伪标签 / 标签混合 / 损失:
    分类: softmax -> argmax 硬伪标签(置信度阈值) -> 混 one-hot 软标签 -> 交叉熵
    回归: 连续伪标签 -> 混连续标签 -> MSE
"""
import os
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from ssl_maml import load_potability, load_ph_regression, split_pool

HERE = os.path.dirname(os.path.abspath(__file__))
POTABILITY = r'c:\Users\wjx\.trae-cn\attachments\6aaaaa65ba62911dcc0c5032\6026c254-36be-44e3-8f71-d96604eb38f0_water_potability.csv'
PH_MAT = os.path.join(HERE, '..', 'data_water_mat', 'water_dataset.mat')


def build_net(in_dim, out_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def weak_aug(x, std=0.01):
    return x + torch.randn_like(x) * std


def strong_aug_gauss_cls(x):
    """分类强增广: 随机掩码 20% 特征 + 较大噪声。"""
    mask = (torch.rand_like(x) > 0.2).float()
    return x * mask + torch.randn_like(x) * 0.1


def mixup_batch(X, y, alpha=4.0):
    """向量化 Mixup。lam 来自 Beta(alpha,alpha), 保证无自配对(循环移位)。"""
    k = X.shape[0]
    if k < 2:
        return X, y
    lam = float(np.random.beta(alpha, alpha))
    offset = int(torch.randint(1, k, (1,)).item())
    perm = (torch.arange(k, device=X.device) + offset) % k
    return lam * X + (1 - lam) * X[perm], lam * y + (1 - lam) * y[perm]


def _onehot(pseudo, n_class):
    return F.one_hot(pseudo, num_classes=n_class).float()


# --------------------------------------------------------------------------- #
# 四种方法的"无标签损失"（task: 'cls' | 'reg'）
# --------------------------------------------------------------------------- #
def loss_fm_gauss(model, p, x, task, tau=0.7):
    if task == 'cls':
        xw = weak_aug(x)
        outw = functional_call(model, p, xw)
        conf, pseudo = torch.softmax(outw, 1).max(1)
        mask = conf >= tau
        if mask.sum() == 0:
            return torch.zeros((), device=x.device)
        outs = functional_call(model, p, strong_aug_gauss_cls(x[mask]))
        return F.cross_entropy(outs, pseudo[mask])
    # 回归: 掩码会自追逐发散, 沿用轻噪声一致性
    xw = weak_aug(x, 0.01)
    target = functional_call(model, p, xw).detach().squeeze(-1)
    outs = functional_call(model, p, x + torch.randn_like(x) * 0.05)
    return F.mse_loss(outs.squeeze(-1), target)


def loss_fm_mixup(model, p, x, task, tau=0.7, alpha=4.0):
    if task == 'cls':
        xw = weak_aug(x)
        outw = functional_call(model, p, xw)
        conf, pseudo = torch.softmax(outw, 1).max(1)
        mask = conf >= tau
        if mask.sum() < 2:
            return torch.zeros((), device=x.device)
        Xc = x[mask]
        yc = _onehot(pseudo[mask], outw.shape[1])
        X_mix, y_mix = mixup_batch(Xc, yc, alpha)
        pred = functional_call(model, p, X_mix)
        return F.cross_entropy(pred, y_mix)
    # 回归: 模型自身连续伪标签做 Mixup 目标(自参照, 豆包原方案)
    xw = weak_aug(x, 0.01)
    with torch.no_grad():
        y = functional_call(model, p, xw)          # (N,1) 伪标签
    X_mix, y_mix = mixup_batch(x, y.detach(), alpha)
    pred = functional_call(model, p, X_mix)
    return F.mse_loss(pred.squeeze(-1), y_mix.squeeze(-1))


def loss_fm_mixup_ema(model, p, teacher_p, x, task, tau=0.7, alpha=4.0):
    """用 EMA Teacher 生成伪标签, 打破"模型自己追自己"的闭环。"""
    if task == 'cls':
        xw = weak_aug(x)
        with torch.no_grad():
            outw = functional_call(model, teacher_p, xw)
        conf, pseudo = torch.softmax(outw, 1).max(1)
        mask = conf >= tau
        if mask.sum() < 2:
            return torch.zeros((), device=x.device)
        Xc = x[mask]
        yc = _onehot(pseudo[mask], outw.shape[1])
        X_mix, y_mix = mixup_batch(Xc, yc, alpha)
        pred = functional_call(model, p, X_mix)
        return F.cross_entropy(pred, y_mix)
    # 回归: teacher 伪标签, 稳定 lam
    xw = weak_aug(x, 0.01)
    with torch.no_grad():
        y = functional_call(model, teacher_p, xw)
    X_mix, y_mix = mixup_batch(x, y.detach(), alpha)
    pred = functional_call(model, p, X_mix)
    return F.mse_loss(pred.squeeze(-1), y_mix.squeeze(-1))


def _semi(method, model, p, teacher_p, x, task, tau):
    if method == 'fm_gauss':
        return loss_fm_gauss(model, p, x, task, tau)
    if method == 'fm_mixup':
        return loss_fm_mixup(model, p, x, task, tau)
    if method == 'fm_mixup_ema':
        return loss_fm_mixup_ema(model, p, teacher_p, x, task, tau)
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# 单次实验
# --------------------------------------------------------------------------- #
def run_once(name, method, n_label, seed, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)

    task = 'cls' if name == 'potability' else 'reg'
    if task == 'cls':
        X, y = load_potability(POTABILITY)
    else:
        X, y = load_ph_regression(PH_MAT)
    Xtr, ytr, Xte, yte = split_pool(X, y, seed=123)

    n = len(Xtr)
    rng = np.random.default_rng(seed)
    if task == 'cls':
        lab_idx = []
        for c in np.unique(ytr):
            ci = np.where(ytr == c)[0]
            lab_idx.append(rng.choice(ci, n_label // 2, replace=False))
        lab_idx = np.concatenate(lab_idx)
    else:
        lab_idx = rng.choice(n, n_label, replace=False)
    unlab_idx = np.setdiff1d(np.arange(n), lab_idx)

    Xlab = torch.from_numpy(Xtr[lab_idx])
    ylab = torch.from_numpy(ytr[lab_idx]).long() if task == 'cls' else torch.from_numpy(ytr[lab_idx]).float()
    Xunlab = torch.from_numpy(Xtr[unlab_idx])
    Xte_t = torch.from_numpy(Xte)
    yte_t = torch.from_numpy(yte).long() if task == 'cls' else torch.from_numpy(yte).float()

    out_dim = 2 if task == 'cls' else 1
    model = build_net(X.shape[1], out_dim, cfg['hidden'])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)

    teacher = None
    if method == 'fm_mixup_ema':
        # teacher 参数 = 模型参数的深拷贝, 每步 EMA 滑动
        teacher = {name: param.clone() for name, param in model.named_parameters()}

    for _ in range(cfg['epochs']):
        model.train()
        out_lab = model(Xlab)
        loss = F.cross_entropy(out_lab, ylab) if task == 'cls' else F.mse_loss(out_lab.squeeze(-1), ylab)
        if method != 'baseline':
            p = {name: param for name, param in model.named_parameters()}
            loss = loss + cfg['lam'] * _semi(method, model, p, teacher, Xunlab, task, cfg['tau'])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if teacher is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    teacher[name].mul_(cfg['ema_decay']).add_(param.data, alpha=1 - cfg['ema_decay'])

    model.eval()
    with torch.no_grad():
        out_te = model(Xte_t)
        if task == 'cls':
            return float((out_te.argmax(1) == yte_t).float().mean().item())
        mse = F.mse_loss(out_te.squeeze(-1), yte_t).item()
        r2 = 1 - mse / float(yte_t.var())
        return r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['potability', 'ph'])
    ap.add_argument('--methods', nargs='+',
                    default=['baseline', 'fm_gauss', 'fm_mixup', 'fm_mixup_ema'])
    ap.add_argument('--n_labels', nargs='+', type=int, default=[10, 20, 40])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--tau', type=float, default=0.7)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--out', default=os.path.join(HERE, 'results_mixup_ablation.json'))
    args = ap.parse_args()

    cfg = dict(epochs=args.epochs, lr=args.lr, hidden=args.hidden,
               lam=args.lam, tau=args.tau, ema_decay=args.ema_decay)
    results = []
    for name in args.datasets:
        for nl in args.n_labels:
            for method in args.methods:
                for seed in args.seeds:
                    m = run_once(name, method, nl, seed, cfg)
                    results.append(dict(dataset=name, method=method,
                                        n_label=nl, seed=seed, metric=m))
                    print(f'[{name}/{method}/nlab={nl}/seed{seed}] = {m:.4f}', flush=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()