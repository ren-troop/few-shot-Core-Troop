import os
import sys
import tempfile


def _pick_writable_tmp():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(here, '_tmp'),
        os.path.join(os.path.expanduser('~'), '_pycharm_tmp'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Temp'),
        os.environ.get('TEMP', ''),
        os.environ.get('TMP', ''),
        os.path.join(here, 'tmp'),
    ]
    for c in cands:
        if not c:
            continue
        try:
            os.makedirs(c, exist_ok=True)
            probe = os.path.join(c, '.write_probe')
            with open(probe, 'w') as f:
                f.write('ok')
            os.remove(probe)
            return c
        except Exception:
            continue
    return None


_TMP_DIR = _pick_writable_tmp()
if _TMP_DIR:
    os.environ['TMP'] = _TMP_DIR
    os.environ['TEMP'] = _TMP_DIR
    os.environ['TMPDIR'] = _TMP_DIR
    tempfile.tempdir = _TMP_DIR
    print(f"[env] 临时目录: {_TMP_DIR}")
else:
    print("[env] 警告: 找不到可写的临时目录，torch 可能无法 import")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 配置区 —— 在 PyCharm 中运行前修改这里
# ============================================================
DATA_PATH = r'C:\Users\wjx\Doubao\chats\2026-09-01\new-chat\data\water_quality\water_dataset.mat'
OUTPUT_DIR = r'C:\Users\wjx\Desktop\半监督研究\_run5\out_fixmatch'

# 超参数
N_SUPPORT = 3          # 每个任务的带标签样本数（3 = 约8%标签，低标签场景）
N_QUERY = 5            # 查询集样本数
META_ITERATIONS = 1000 # 元训练迭代次数
META_LR = 0.001        # 外层学习率
INNER_LR = 0.01        # 内层学习率
INNER_STEPS = 5        # 内层梯度步数
TASK_BATCH_SIZE = 4    # 每轮元训练的任务数
UNSUP_WEIGHT = 0.3     # 半监督损失权重（v2配合Loss尺度修复，从1.0降到0.3）
SEED = 42

# FixMatch 超参数
WEAK_NOISE_STD = 0.01      # 弱增强高斯噪声标准差
MIXUP_ALPHA = 0.75         # Mixup Beta分布参数
CONFIDENCE_FRACTION = 0.3  # 置信度过滤比例（选预测方差最低的30%）
N_WEAK_PREDS = 5           # 弱增强预测次数（用于估计方差）
SPATIAL_MIXUP_PROB = 0.8   # 空间感知Mixup概率（80%同组配对）


# ============================================================
# 1. 数据加载
# ============================================================
def load_water_data(mat_path):
    """加载水质数据集，包含 location_group 空间分组信息。"""
    data = sio.loadmat(mat_path)
    X_tr_cell = data['X_tr']
    Y_tr = data['Y_tr']
    X_te_cell = data['X_te']
    Y_te = data['Y_te']

    # 构建站点→空间组映射
    n_sites = X_tr_cell[0, 0].shape[0]  # 37
    site_groups = np.zeros(n_sites, dtype=np.int64)
    lg = data['location_group']
    for g in range(lg.shape[1]):
        indices = lg[0, g].flatten()
        for idx in indices:
            site_groups[idx - 1] = g
    site_groups_t = torch.LongTensor(site_groups)
    print(f"  空间分组: {np.bincount(site_groups)} 个站点/组")

    train_tasks = []
    for i in range(X_tr_cell.shape[1]):
        X = torch.FloatTensor(X_tr_cell[0, i])
        Y = torch.FloatTensor(Y_tr[:, i]).unsqueeze(1)
        train_tasks.append((X, Y, site_groups_t.clone()))

    test_tasks = []
    for i in range(X_te_cell.shape[1]):
        X = torch.FloatTensor(X_te_cell[0, i])
        Y = torch.FloatTensor(Y_te[:, i]).unsqueeze(1)
        test_tasks.append((X, Y, site_groups_t.clone()))

    return train_tasks, test_tasks


def split_task(X, Y, site_groups, n_support, n_query, seed=None):
    """将一个任务划分为 support / unlabeled / query 三部分。"""
    n = X.shape[0]
    if seed is not None:
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    else:
        idx = torch.randperm(n)

    sup_idx = idx[:n_support]
    qry_idx = idx[n_support:n_support + n_query]
    unl_idx = idx[n_support + n_query:]

    return (X[sup_idx], Y[sup_idx], site_groups[sup_idx],
            X[unl_idx], site_groups[unl_idx],
            X[qry_idx], Y[qry_idx], site_groups[qry_idx])


# ============================================================
# 2. 模型
# ============================================================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=11, hidden_dims=[64, 32], output_dim=1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def functional_forward(model, x, params):
    """MAML inner loop 用的显式参数前向传播。"""
    x = x.clone()
    idx = 0
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            w = params[idx]
            b = params[idx + 1]
            x = F.linear(x, w, b)
            idx += 2
        elif isinstance(layer, nn.ReLU):
            x = F.relu(x)
    return x


# ============================================================
# 3. FixMatch (Mixup版) 半监督损失
# ============================================================
def fixmatch_mixup_loss(model, X_unl, G_unl, params,
                         weak_noise_std=WEAK_NOISE_STD,
                         mixup_alpha=MIXUP_ALPHA,
                         confidence_fraction=CONFIDENCE_FRACTION,
                         n_weak_preds=N_WEAK_PREDS,
                         spatial_mixup_prob=SPATIAL_MIXUP_PROB):
    """
    FixMatch 损失（Mixup强增强版本）：
    步骤1：弱增强(高斯噪声)多次预测 → 均值作伪标签，方差作不确定性
    步骤2：选预测方差最低的 confidence_fraction 样本（Top-K置信度过滤）
    步骤3：空间感知Mixup（优先同空间组配对）生成强增强样本
    步骤4：强增强样本预测与伪标签的MSE一致性损失
    """
    n_unl = X_unl.shape[0]
    if n_unl < 2:
        return torch.tensor(0.0), 0, 0.0

    # 步骤1：弱增强预测 + 方差估计
    model.eval()
    with torch.no_grad():
        preds_list = []
        for _ in range(n_weak_preds):
            X_weak = X_unl + torch.randn_like(X_unl) * weak_noise_std
            preds_list.append(functional_forward(model, X_weak, params))
        pseudo_y = torch.stack(preds_list).mean(dim=0)   # 伪标签
        pred_var = torch.stack(preds_list).var(dim=0).squeeze(1)  # 不确定性

    # 步骤2：Top-K 置信度过滤（选方差最低的30%）
    k = max(2, int(np.ceil(n_unl * confidence_fraction)))
    if k >= n_unl:
        selected_idx = torch.arange(n_unl)
    else:
        _, selected_idx = torch.topk(pred_var, k, largest=False)

    n_confident = k
    mean_pred_var = pred_var[selected_idx].mean().item()
    X_conf = X_unl[selected_idx]
    y_conf = pseudo_y[selected_idx]
    G_conf = G_unl[selected_idx]

    # 步骤3：空间感知 Mixup
    model.train()
    perm = torch.zeros(k, dtype=torch.long)
    for i in range(k):
        if np.random.random() < spatial_mixup_prob:
            same_group = (G_conf == G_conf[i]).nonzero(as_tuple=True)[0]
            same_group = same_group[same_group != i]
            if len(same_group) > 0:
                perm[i] = same_group[torch.randint(len(same_group), (1,)).item()]
            else:
                candidates = torch.arange(k)
                candidates = candidates[candidates != i]
                perm[i] = candidates[torch.randint(len(candidates), (1,)).item()]
        else:
            candidates = torch.arange(k)
            candidates = candidates[candidates != i]
            perm[i] = candidates[torch.randint(len(candidates), (1,)).item()]

    lam = np.random.beta(mixup_alpha, mixup_alpha)
    lam = max(lam, 1 - lam)
    X_mix = lam * X_conf + (1 - lam) * X_conf[perm]
    y_mix = lam * y_conf + (1 - lam) * y_conf[perm]

    # 步骤4：一致性损失
    pred_mix = functional_forward(model, X_mix, params)
    loss = F.mse_loss(pred_mix, y_mix.detach())

    return loss, n_confident, mean_pred_var


# ============================================================
# 4. MAML 训练
# ============================================================
def maml_train(model, train_tasks, device='cpu'):
    """MAML 元训练，inner loop 中加入 FixMatch 半监督损失。"""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=META_LR)
    model.to(device)
    n_tasks = len(train_tasks)

    history = {'meta_loss': [], 'fixmatch_loss': [], 'confident_fraction': [], 'mean_pred_var': []}

    for it in range(META_ITERATIONS):
        meta_optimizer.zero_grad()
        task_indices = np.random.choice(n_tasks, size=TASK_BATCH_SIZE, replace=False)

        batch_qry_loss = 0.0
        batch_fm_loss = 0.0
        batch_conf_frac = 0.0
        batch_mean_var = 0.0

        for ti in task_indices:
            X, Y, G = train_tasks[ti]
            (X_sup, Y_sup, G_sup, X_unl, G_unl, X_qry, Y_qry, G_qry) = \
                split_task(X, Y, G, N_SUPPORT, N_QUERY)

            # Inner loop：5步梯度适应
            params = [p.clone() for p in model.parameters()]
            for step in range(INNER_STEPS):
                pred_sup = functional_forward(model, X_sup, params)
                loss_sup = F.mse_loss(pred_sup, Y_sup)

                loss_fm = torch.tensor(0.0)
                n_conf = 0
                mean_var = 0.0
                if X_unl.shape[0] > 0:
                    loss_fm, n_conf, mean_var = fixmatch_mixup_loss(model, X_unl, G_unl, params)

                total_loss = loss_sup + UNSUP_WEIGHT * loss_fm
                grads = torch.autograd.grad(total_loss, params, create_graph=True, allow_unused=True)
                params = [p - INNER_LR * (g if g is not None else 0)
                          for p, g in zip(params, grads)]

            # Query set 损失（用于外层元更新）
            pred_qry = functional_forward(model, X_qry, params)
            qry_loss = F.mse_loss(pred_qry, Y_qry)

            batch_qry_loss += qry_loss
            batch_fm_loss += loss_fm.item() if isinstance(loss_fm, torch.Tensor) else loss_fm
            batch_conf_frac += n_conf / max(X_unl.shape[0], 1)
            batch_mean_var += mean_var

        # 外层元更新
        meta_loss = batch_qry_loss / TASK_BATCH_SIZE
        meta_loss.backward()
        meta_optimizer.step()

        history['meta_loss'].append(meta_loss.item())
        history['fixmatch_loss'].append(batch_fm_loss / TASK_BATCH_SIZE)
        history['confident_fraction'].append(batch_conf_frac / TASK_BATCH_SIZE)
        history['mean_pred_var'].append(batch_mean_var / TASK_BATCH_SIZE)

        if (it + 1) % 250 == 0:
            print(f"  Iter {it+1}/{META_ITERATIONS} | "
                  f"meta_loss={meta_loss.item():.6f} | "
                  f"fm_loss={batch_fm_loss/TASK_BATCH_SIZE:.6f} | "
                  f"conf_frac={batch_conf_frac/TASK_BATCH_SIZE:.3f}")

    return history


# ============================================================
# 5. 评估
# ============================================================
def evaluate(model, test_tasks, device='cpu'):
    """在全部测试任务上评估。"""
    model.eval()
    all_mse, all_mae, all_r2 = [], [], []

    for ti in range(len(test_tasks)):
        X, Y, G = test_tasks[ti]
        (X_sup, Y_sup, G_sup, X_unl, G_unl, X_qry, Y_qry, G_qry) = \
            split_task(X, Y, G, N_SUPPORT, N_QUERY, seed=int(ti))

        params = [p.clone() for p in model.parameters()]
        for step in range(INNER_STEPS):
            pred_sup = functional_forward(model, X_sup, params)
            loss_sup = F.mse_loss(pred_sup, Y_sup)
            loss_fm = torch.tensor(0.0)
            if X_unl.shape[0] > 0:
                loss_fm, _, _ = fixmatch_mixup_loss(model, X_unl, G_unl, params)
            total_loss = loss_sup + UNSUP_WEIGHT * loss_fm
            grads = torch.autograd.grad(total_loss, params, allow_unused=True)
            params = [p - INNER_LR * (g if g is not None else 0)
                      for p, g in zip(params, grads)]

        with torch.no_grad():
            pred_qry = functional_forward(model, X_qry, params)
            mse = F.mse_loss(pred_qry, Y_qry).item()
            mae = F.l1_loss(pred_qry, Y_qry).item()
            ss_res = ((Y_qry - pred_qry) ** 2).sum().item()
            ss_tot = ((Y_qry - Y_qry.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / (ss_tot + 1e-8)

        all_mse.append(mse)
        all_mae.append(mae)
        all_r2.append(r2)

    return {
        'mse_mean': float(np.mean(all_mse)), 'mse_std': float(np.std(all_mse)),
        'mae_mean': float(np.mean(all_mae)), 'mae_std': float(np.std(all_mae)),
        'r2_mean': float(np.mean(all_r2)), 'r2_std': float(np.std(all_r2)),
    }


# ============================================================
# 6. 主函数
# ============================================================
def main():
    print("=" * 60)
    print("  FixMatch (Mixup版) + MAML —— 水质预测实验")
    print(f"  标签数: {N_SUPPORT}/任务 | 测试集: 全部282任务")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    print("\n  加载数据...")
    train_tasks, test_tasks = load_water_data(DATA_PATH)
    print(f"  训练: {len(train_tasks)} 任务, 测试: {len(test_tasks)} 任务")

    # 训练
    print(f"\n  开始训练 (n_support={N_SUPPORT})...")
    model = MLPRegressor(input_dim=11, hidden_dims=[64, 32], output_dim=1)
    start = time.time()
    history = maml_train(model, train_tasks)
    train_time = time.time() - start
    print(f"  训练耗时: {train_time:.1f}s")

    # 评估
    print("\n  评估 (全部282测试任务)...")
    results = evaluate(model, test_tasks)
    results['train_time'] = train_time

    print(f"\n  {'='*50}")
    print(f"  结果 (n_support={N_SUPPORT})")
    print(f"  {'='*50}")
    print(f"  MSE: {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
    print(f"  MAE: {results['mae_mean']:.6f} ± {results['mae_std']:.6f}")
    print(f"  R2:  {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")

    # 保存
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'fixmatch_mixup_maml_model.pt'))
    with open(os.path.join(OUTPUT_DIR, 'fixmatch_mixup_maml_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    window = 20
    for idx, (key, title, color) in enumerate([
        ('meta_loss', 'Meta Loss', '#FF9800'),
        ('fixmatch_loss', 'FixMatch Loss', '#E91E63'),
        ('confident_fraction', 'Confident Fraction', '#2196F3'),
        ('mean_pred_var', 'Mean Prediction Variance', '#4CAF50'),
    ]):
        ax = axes[idx // 2][idx % 2]
        vals = history[key]
        smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, color=color, linewidth=1.5)
        ax.set_xlabel('Meta-Iteration')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.suptitle('FixMatch-Mixup + MAML Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fixmatch_mixup_maml_training.png'), dpi=150)
    plt.close()

    print(f"\n  结果已保存到 {OUTPUT_DIR}")
    print("  完成！")


if __name__ == '__main__':
    main()
