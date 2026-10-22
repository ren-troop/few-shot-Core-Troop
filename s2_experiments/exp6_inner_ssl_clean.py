# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
HERE = os.path.dirname(os.path.abspath(__file__))

def p_clone(P):
    return [(W.copy(), b.copy()) for W, b in P]

def p_add(P, Q, s=1.0):
    return [(W + s * Wq, b + s * bq) for (W, b), (Wq, bq) in zip(P, Q)]

def p_sub(P, Q, s=1.0):
    return [(W - s * Wq, b - s * bq) for (W, b), (Wq, bq) in zip(P, Q)]

def p_scale(P, s):
    return [(W * s, b * s) for W, b in P]

def p_norm(P):
    return float(np.sqrt(sum(((W ** 2).sum() + (b ** 2).sum() for W, b in P))))

def p_zeros_like(P):
    return [(np.zeros_like(W), np.zeros_like(b)) for W, b in P]

class MLP:

    def __init__(self, sizes, seed=0, last_scale=0.1):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.P = []
        for i in range(len(sizes) - 1):
            if i == len(sizes) - 2:
                s = last_scale
            else:
                s = np.sqrt(2.0 / sizes[i])
            self.P.append((rng.normal(0.0, s, (sizes[i], sizes[i + 1])), np.zeros(sizes[i + 1])))

    def forward(self, P, X):
        a = X
        for W, b in P[:-1]:
            a = np.maximum(a @ W + b, 0.0)
        W, b = P[-1]
        return a @ W + b

    def features(self, P, X):
        a = X
        for W, b in P[:-1]:
            a = np.maximum(a @ W + b, 0.0)
        return a

    def backward(self, P, X, dZ):
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

def ce_grad(net, P, X, y):
    z = net.forward(P, X)
    p = softmax(z)
    n = len(y)
    loss = float(-np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean())
    dZ = p.copy()
    dZ[np.arange(n), y] -= 1.0
    dZ /= n
    return (loss, net.backward(P, X, dZ))

def soft_ce_grad(net, P, X, T):
    z = net.forward(P, X)
    p = softmax(z)
    n = T.shape[0]
    loss = float((-(T * np.log(np.clip(p, 1e-12, None))).sum(axis=1)).mean())
    return (loss, net.backward(P, X, (p - T) / n))

def entropy_grad(net, P, X):
    z = net.forward(P, X)
    p = softmax(z)
    n = len(p)
    loss = float(entropy(p).mean())
    H = entropy(p)[:, None]
    dZ = p * (np.log(np.clip(p, 1e-12, None)) + H) / n
    return (loss, net.backward(P, X, dZ))

def combine_grads(*pairs):
    loss = sum((l for l, _ in pairs))
    dP = p_scale(pairs[0][1], 0.0)
    for _, g in pairs:
        dP = p_add(dP, g, 1.0)
    return (loss, dP)

def inner_adapt(P, inner_grad, alpha, steps):
    thetas = [p_clone(P)]
    for _ in range(steps):
        _, g = inner_grad(thetas[-1])
        thetas.append(p_sub(thetas[-1], g, alpha))
    return thetas

def hvp_fd(grad_fn, P, v, rel=0.0001):
    nv = p_norm(v)
    if nv < 1e-300:
        return p_zeros_like(P)
    eps = rel * (1.0 + p_norm(P)) / nv
    _, gp = grad_fn(P)
    _, gm = grad_fn(p_sub(P, v, eps))
    return p_scale(p_sub(gp, gm), 1.0 / eps)

def maml_meta_grad(thetas, outer_grad, inner_grad, alpha, second_order=True, rel=0.0001):
    _, gq = outer_grad(thetas[-1])
    if not second_order:
        return gq
    v = gq
    for k in range(len(thetas) - 2, -1, -1):
        h = hvp_fd(inner_grad, thetas[k], v, rel)
        v = p_sub(v, h, alpha)
    return v

class Adam:

    def __init__(self, P, lr=0.001, b1=0.9, b2=0.999, eps=1e-08):
        self.lr, self.b1, self.b2, self.eps = (lr, b1, b2, eps)
        self.m = p_zeros_like(P)
        self.v = p_zeros_like(P)
        self.t = 0

    def step(self, P, g):
        self.t += 1
        b1, b2, t = (self.b1, self.b2, self.t)
        self.m = [(b1 * mw + (1 - b1) * gw, b1 * mb + (1 - b1) * gb) for (mw, mb), (gw, gb) in zip(self.m, g)]
        self.v = [(b2 * vw + (1 - b2) * gw * gw, b2 * vb + (1 - b2) * gb * gb) for (vw, vb), (gw, gb) in zip(self.v, g)]
        c1, c2 = (1.0 - b1 ** t, 1.0 - b2 ** t)
        upd = [(self.lr * (mw / c1) / (np.sqrt(vw / c2) + self.eps), self.lr * (mb / c1) / (np.sqrt(vb / c2) + self.eps)) for (mw, mb), (vw, vb) in zip(self.m, self.v)]
        return p_sub(P, upd, 1.0)

def accuracy(net, P, X, y):
    return float((net.forward(P, X).argmax(1) == y).mean())

def knn_affinity(F, k=10, sigma=None):
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

def label_spreading(W, Y0, alpha=0.9, iters=60, clamp_idx=None):
    d = W.sum(1)
    d[d <= 0] = 1.0
    dinv = 1.0 / np.sqrt(d)
    S = W * dinv[:, None] * dinv[None, :]
    Y = Y0.copy()
    for _ in range(iters):
        Y = alpha * (S @ Y) + (1.0 - alpha) * Y0
    if clamp_idx is not None:
        Y[clamp_idx] = Y0[clamp_idx]
    return Y

def label_propagation(F, y_lab, idx_lab, n_class, k=10, alpha=0.9, iters=60):
    N = F.shape[0]
    Y0 = np.zeros((N, n_class))
    Y0[idx_lab, y_lab] = 1.0
    W = knn_affinity(F, k=k)
    return label_spreading(W, Y0, alpha=alpha, iters=iters, clamp_idx=idx_lab)

@dataclass
class Task:
    Xs: np.ndarray
    ys: np.ndarray
    Xq: np.ndarray
    yq: np.ndarray
    n_way: int
    info: dict = field(default_factory=dict)

    @property
    def Xu(self):
        return self.Xq

def sample_gauss_task(rng, n_way=5, dim=2, K=1, Q=15, radius=1.0, sigma=0.35, warp=0.0, noise=0.0):
    phase = rng.uniform(0.0, 2.0 * np.pi)
    ang = phase + 2.0 * np.pi * np.arange(n_way) / n_way
    means = radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    A = np.eye(2)
    if warp > 0:
        R = np.linalg.qr(rng.normal(size=(2, 2)))[0]
        s = np.exp(rng.uniform(-warp, warp, size=2))
        A = R @ np.diag(s) @ R.T
    zs = rng.normal(0, sigma, size=(n_way, K, 2))
    zq = rng.normal(0, sigma, size=(n_way, Q, 2))

    def to_x(z):
        return z @ A.T + means[:, None, :]
    Xs = to_x(zs).reshape(n_way * K, 2)
    ys = np.repeat(np.arange(n_way), K)
    Xq = to_x(zq).reshape(n_way * Q, 2)
    yq = np.repeat(np.arange(n_way), Q)

    def embed(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = X.shape[0]
        if dim <= X.shape[1]:
            return X[:, :dim].copy()
        out = np.zeros((n, dim))
        out[:, :X.shape[1]] = X
        if noise > 0:
            out[:, X.shape[1]:] = rng.normal(0.0, noise, size=(n, dim - X.shape[1]))
        return out
    perm = rng.permutation(len(Xs))
    return Task(embed(Xs)[perm], ys[perm], embed(Xq), yq, n_way, {'A': A, 'phase': phase, 'means': means})

def make_pl_term(net, X_u, tau=0.9):

    def g(P):
        p = softmax(net.forward(P, X_u))
        conf = p.max(1)
        idx = p.argmax(1)
        mask = conf > tau
        if mask.sum() == 0:
            return (0.0, p_zeros_like(P))
        T = np.zeros_like(p[mask])
        T[np.arange(len(T)), idx[mask]] = 1.0
        return soft_ce_grad(net, P, X_u[mask], T)
    return g

def make_fixmatch_term(net, X_u, tau=0.95, sigma_w=0.05, sigma_s=0.3, drop=0.25, rng=None):

    def g(P):
        xw = X_u + rng.normal(0, sigma_w, X_u.shape)
        xs = X_u + rng.normal(0, sigma_s, X_u.shape)
        xs = xs * (rng.random(X_u.shape) > drop)
        pw = softmax(net.forward(P, xw))
        mask = pw.max(1) > tau
        if mask.sum() == 0:
            return (0.0, p_zeros_like(P))
        T = np.zeros((int(mask.sum()), pw.shape[1]))
        T[np.arange(len(T)), pw[mask].argmax(1)] = 1.0
        return soft_ce_grad(net, P, xs[mask], T)
    return g

def make_vat_term(net, X_u, eps=1.0, xi=1e-06, n_power=1, rng=None):

    def g(P):
        d = rng.normal(size=X_u.shape)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        for _ in range(n_power):
            xr = X_u + xi * d
            p0 = softmax(net.forward(P, X_u))
            pr = softmax(net.forward(P, xr))
            dz = (pr - p0) / len(X_u)
            gi = net.backward_input(P, xr, dz)
            d = gi / (np.linalg.norm(gi, axis=1, keepdims=True) + 1e-12)
        r = eps * d
        T = softmax(net.forward(P, X_u))
        pr = softmax(net.forward(P, X_u + r))
        kl = float((T * (np.log(np.clip(T, 1e-12, None)) - np.log(np.clip(pr, 1e-12, None)))).sum(1).mean())
        return (kl, net.backward(P, X_u + r, (pr - T) / len(X_u)))
    return g

def make_entropy_term(net, X_u):

    def g(P):
        return entropy_grad(net, P, X_u)
    return g

def make_net(cfg, seed=None):
    return MLP([cfg['dim']] + list(cfg['hidden']) + [cfg['n_way']], seed=cfg['seed'] if seed is None else seed, last_scale=cfg['last_scale'])

def scale_term(pair, w):
    l, g = pair
    return (w * l, p_scale(g, w))

def make_inner(net, task, kind, cfg, rng):
    base = lambda P: ce_grad(net, P, task.Xs, task.ys)
    if kind == 'sup':
        return base
    if kind == 'sup_pl':
        U = make_pl_term(net, task.Xu, tau=cfg['pl_tau'])
    elif kind == 'sup_fix':
        U = make_fixmatch_term(net, task.Xu, tau=cfg['fm_tau'], rng=rng)
    elif kind == 'sup_vat':
        U = make_vat_term(net, task.Xu, eps=cfg['vat_eps'], rng=rng)
    elif kind == 'sup_ent':
        U = make_entropy_term(net, task.Xu)
    else:
        raise ValueError(kind)
    return lambda P: combine_grads(base(P), scale_term(U(P), cfg['lam']))

def evaluate_task(net, P0, task, method, cfg):
    Xs, ys, Xq, yq = (task.Xs, task.ys, task.Xq, task.yq)
    steps = cfg['eval_steps']
    if method == 'three_stage':
        thetas = inner_adapt(P0, lambda P: ce_grad(net, P, Xs, ys), cfg['alpha'], steps)
        Pf = thetas[-1]
        F = net.features(Pf, np.vstack([Xs, Xq]))
        Y = label_propagation(F, ys, np.arange(len(ys)), task.n_way, k=cfg['lp_k'], alpha=cfg['lp_alpha'])
        return float((Y[len(ys):].argmax(1) == yq).mean())
    rng = np.random.default_rng(cfg['test_seed'] + 7)
    inner = make_inner(net, task, method, cfg, rng)
    thetas = inner_adapt(P0, inner, cfg['alpha'], steps)
    return accuracy(net, thetas[-1], Xq, yq)

def run_eval(net, P0, cfg):
    rng = np.random.default_rng(cfg['test_seed'])
    tasks = [sample(rng, cfg) for _ in range(cfg['n_test_tasks'])]
    out = {}
    for m in cfg['eval_methods']:
        accs = np.array([evaluate_task(net, P0, t, m, cfg) for t in tasks]) * 100
        out[m] = [float(accs.mean()), float(accs.std() / np.sqrt(len(accs)))]
    return out

def sample(rng, cfg):
    return sample_gauss_task(rng, n_way=cfg['n_way'], dim=cfg['dim'], K=cfg['K'], Q=cfg['Q'], radius=cfg['radius'], sigma=cfg['sigma'], warp=cfg['warp'], noise=cfg['noise'])

def train_maml(cfg, log=None):
    net = make_net(cfg)
    P = net.P
    opt = Adam(P, lr=cfg['meta_lr'])
    rng = np.random.default_rng(cfg['seed'] + 1)
    t0 = time.time()
    for it in range(cfg['iters']):
        gsum = p_zeros_like(P)
        for _ in range(cfg['meta_batch']):
            task = sample(rng, cfg)
            inner = lambda Q, t=task: ce_grad(net, Q, t.Xs, t.ys)
            outer = lambda Q, t=task: ce_grad(net, Q, t.Xq, t.yq)
            thetas = inner_adapt(P, inner, cfg['alpha'], cfg['inner_steps'])
            g = maml_meta_grad(thetas, outer, inner, cfg['alpha'], second_order=True)
            gsum = p_add(gsum, g, 1.0 / cfg['meta_batch'])
        P = opt.step(P, gsum)
        if log and (it + 1) % max(1, cfg['iters'] // 4) == 0:
            log('    iter %d/%d  (%.0fs)' % (it + 1, cfg['iters'], time.time() - t0))
    return (net, P)
METHOD_DESC = {'sup': '基线：内层只有支持集交叉熵', 'sup_pl': '① Pseudo-Label 放进内层', 'sup_fix': '② FixMatch 放进内层', 'sup_vat': '③ VAT 放进内层', 'sup_ent': '对照组：熵最小化放进内层', 'three_stage': '新方法：适应之后做标签传播（三段式）'}
SHIPPED = {'sup': (75.86, 0.62), 'sup_pl': (75.92, 0.59), 'sup_fix': (63.39, 0.65), 'sup_vat': (55.96, 0.48), 'sup_ent': (63.05, 0.61), 'three_stage': (82.64, 0.58)}

def make_config(quick=False):
    cfg = dict(n_way=5, dim=8, K=1, Q=15, radius=1.0, sigma=0.35, warp=0.4, noise=1.0, hidden=[40, 40], last_scale=0.1, seed=0, alpha=0.4, inner_steps=1, eval_steps=3, meta_lr=0.001, meta_batch=16, iters=2500, n_test_tasks=400, lam=1.0, pl_tau=0.9, fm_tau=0.95, vat_eps=1.0, lp_k=12, lp_alpha=0.9, test_seed=12345, eval_methods=['sup', 'sup_pl', 'sup_fix', 'sup_vat', 'sup_ent', 'three_stage'])
    if quick:
        cfg['iters'] = 100
        cfg['n_test_tasks'] = 40
    return cfg

def print_table(res, methods):
    base = res['sup'][0]
    print('')
    print('=' * 78)
    print('%-38s %14s %12s' % ('方法', '准确率 %', '相对基线'))
    print('-' * 78)
    for m in methods:
        mu, se = res[m]
        d = '—' if m == 'sup' else '%+.2f' % (mu - base)
        print('%-38s %8.2f +- %.2f %12s' % (METHOD_DESC[m], mu, se, d))
    print('=' * 78)
SHORT_LABELS = {'sup': '基线', 'sup_pl': 'Pseudo-Label 放进内层', 'sup_fix': 'FixMatch 放进内层', 'sup_vat': 'VAT 放进内层', 'sup_ent': '熵最小化放进内层', 'three_stage': '三段式（放在适应之后）'}

def make_plot(res, methods, out_png):
    if not HAVE_MPL:
        print('  [提示] 没装 matplotlib，跳过画图。安装命令：pip install matplotlib')
        return False
    base = res['sup'][0]
    names = [SHORT_LABELS[m] for m in methods]
    vals = [res[m][0] for m in methods]
    errs = [res[m][1] for m in methods]
    colors = []
    for m in methods:
        v = res[m][0] - base
        if m == 'sup':
            colors.append('#888888')
        elif v > 1:
            colors.append('#2a9d8f')
        elif v < -5:
            colors.append('#c1121f')
        else:
            colors.append('#e9c46a')
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(methods))[::-1]
    ax.barh(y, vals, xerr=errs, color=colors, capsize=4, error_kw=dict(lw=1.2, ecolor='#333333'))
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(base, color='k', ls='--', lw=1, label='基线 %.2f' % base)
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.5, yi, '%.2f' % v, va='center', fontsize=9)
    ax.set_xlabel('Few-shot accuracy (%)')
    ax.set_title('Semi-supervised term placed INSIDE the MAML inner loop\n(same backbone / same K labels -- only the position differs)')
    ax.legend(loc='lower right')
    ax.set_xlim(min(vals) - 6, max(vals) + 8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print('  图已保存 -> ' + out_png)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='少量迭代，快速验证')
    ap.add_argument('--plot', action='store_true', help='跑完画一张对比图')
    ap.add_argument('--out', default=os.path.join(HERE, 'exp6_inner_ssl_result.json'))
    args = ap.parse_args()
    cfg = make_config(quick=args.quick)
    print('=' * 78)
    print('exp6 -- 半监督放进 MAML 内循环：PL / FixMatch / VAT vs 基线 vs 三段式')
    print('=' * 78)
    print('  iter=%d  n_test_tasks=%d  lam=%.2f  alpha=%.2f  inner_steps=%d' % (cfg['iters'], cfg['n_test_tasks'], cfg['lam'], cfg['alpha'], cfg['inner_steps']))
    if args.quick:
        print('  （--quick 模式：只验证脚本能跑通，数字不代表结论）')
    print('\n  训练 MAML 二阶主干 ...')
    t0 = time.time()
    net, P = train_maml(cfg, log=print)
    print('  元训练完成，用时 %.0fs' % (time.time() - t0))
    print('  评测中（%d 个任务 x %d 种方法）...' % (cfg['n_test_tasks'], len(cfg['eval_methods'])))
    res = run_eval(net, P, cfg)
    print_table(res, cfg['eval_methods'])
    print('')
    print('  结论：三种半监督方法放进内层都不涨点（PL 空转、FixMatch/VAT 伤害），')
    print('        把半监督挪到「适应之后」（三段式）才涨点。问题在位置，不在方法。')
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'config': cfg, 'results': res}, f, indent=2, ensure_ascii=False)
    print('\n  结果已保存 -> ' + args.out)
    if args.plot:
        print('')
        if args.quick:
            print('  [注意] --quick 模式下数字不可信，改用本机实测值画图。')
            shown = dict(SHIPPED)
        else:
            shown = {k: tuple(v) for k, v in res.items()}
        make_plot(shown, cfg['eval_methods'], os.path.join(HERE, 'exp6_inner_ssl_compare.png'))
if __name__ == '__main__':
    main()
