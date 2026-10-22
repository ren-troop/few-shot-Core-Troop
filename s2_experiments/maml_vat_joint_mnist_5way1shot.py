"""
MAML + VAT 联合实验：方案A —— 5-way 1-shot，2 个区域

方案 A 设置：
  - 2 个合成区域（每区 5 个数字）
  - 区域1: {0, 1, 2, 3, 4}
  - 区域2: {5, 6, 7, 8, 9}
  - 5-way 1-shot 分类
  - 每类 20 张无标签数据
  - 对比：无SSL(标准MAML) vs 有SSL(MAML+VAT)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import copy
import random
import contextlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchvision import datasets, transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# VAT Loss
# ============================================================

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    def __init__(self, xi=0.1, eps=2.0, ip=1):
        super().__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds


# ============================================================
# 模型
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch, track_running_stats=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MAMLNet(nn.Module):
    """4层 stride-2 CNN，输出 n_way 维 logits"""
    def __init__(self, n_way=5, hidden=64):
        super().__init__()
        self.block1 = ConvBlock(1, hidden)
        self.block2 = ConvBlock(hidden, hidden)
        self.block3 = ConvBlock(hidden, hidden)
        self.block4 = ConvBlock(hidden, hidden)
        self.linear = nn.Linear(hidden * 2 * 2, n_way)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# ============================================================
# 数据：2 个区域，每区 5 个数字
# ============================================================

def create_2_regions(data_dir='./mnist_data'):
    """
    方案A：2 个合成区域，每区 5 个数字
      区域1: {0,1,2,3,4}  —— 前5个数字
      区域2: {5,6,7,8,9}  —— 后5个数字
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    region_classes = [
        [0, 1, 2, 3, 4],  # 区域1
        [5, 6, 7, 8, 9],  # 区域2
    ]

    def build_indices(dataset, class_list):
        indices = {}
        targets = dataset.targets.numpy()
        for cls in class_list:
            indices[cls] = np.where(targets == cls)[0].tolist()
        return indices

    regions_train = []
    regions_test = []

    for r_idx, cls_list in enumerate(region_classes):
        tr_idx = build_indices(train_ds, cls_list)
        te_idx = build_indices(test_ds, cls_list)
        regions_train.append({'dataset': train_ds, 'class_indices': tr_idx, 'classes': cls_list})
        regions_test.append({'dataset': test_ds, 'class_indices': te_idx, 'classes': cls_list})
        print(f"  区域{r_idx+1}: 类别{cls_list}, "
              f"训练样本: {sum(len(v) for v in tr_idx.values())}, "
              f"测试样本: {sum(len(v) for v in te_idx.values())}")

    return regions_train, regions_test


def sample_task(region_info, n_way=5, n_support=1, n_unlabeled=20, n_query=10, seed=None):
    """采样一个 few-shot 任务（含无标签数据）"""
    rng = np.random.RandomState(seed) if seed is not None else np.random

    dataset = region_info['dataset']
    classes = region_info['classes']
    class_indices = region_info['class_indices']

    # n_way 必须等于区域类别数
    assert n_way == len(classes), f"区域只有 {len(classes)} 个类，n_way={n_way} 不匹配"

    sup_x, sup_y, unl_x, q_x, q_y = [], [], [], [], []

    for ep_idx, cls in enumerate(classes):
        idx_list = class_indices[cls]
        total = n_support + n_unlabeled + n_query
        selected = rng.choice(idx_list, size=min(total, len(idx_list)), replace=False)

        for i in range(n_support):
            img, _ = dataset[selected[i]]
            sup_x.append(img)
            sup_y.append(ep_idx)

        for i in range(n_support, n_support + n_unlabeled):
            img, _ = dataset[selected[i]]
            unl_x.append(img)

        for i in range(n_support + n_unlabeled, total):
            if i < len(selected):
                img, _ = dataset[selected[i]]
                q_x.append(img)
                q_y.append(ep_idx)

    sup_x = torch.stack(sup_x)
    sup_y = torch.tensor(sup_y)
    unl_x = torch.stack(unl_x) if len(unl_x) > 0 else torch.empty(0, 1, 28, 28)
    q_x = torch.stack(q_x)
    q_y = torch.tensor(q_y)

    return sup_x, sup_y, unl_x, q_x, q_y


# ============================================================
# 标准 MAML（无 SSL）
# ============================================================

def train_baseline(model, region_info, num_iter=3000,
                   alpha=0.4, beta=0.001, meta_batch=8,
                   n_way=5, n_support=1, n_query=10, device='cpu'):
    optimizer = Adam(model.parameters(), lr=beta)
    running_loss = 0.0
    running_acc = 0.0

    for it in range(num_iter):
        meta_grads = {}

        for task_i in range(meta_batch):
            sx, sy, _, qx, qy = sample_task(
                region_info, n_way, n_support, 0, n_query,
                seed=it * 1000 + task_i
            )
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)

            orig = {n: p.data.clone() for n, p in model.named_parameters()}

            # 内循环
            model.zero_grad()
            sup_loss = F.cross_entropy(model(sx), sy)
            sup_loss.backward()

            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.data -= alpha * p.grad

            # 外循环 query loss
            model.zero_grad()
            q_logits = model(qx)
            q_loss = F.cross_entropy(q_logits, qy)
            q_loss.backward()

            with torch.no_grad():
                q_acc = (q_logits.argmax(1) == qy).float().mean().item()

            for n, p in model.named_parameters():
                if p.grad is not None:
                    if n not in meta_grads:
                        meta_grads[n] = p.grad.clone() / meta_batch
                    else:
                        meta_grads[n] += p.grad.clone() / meta_batch

            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.data = orig[n]

            running_loss += q_loss.item()
            running_acc += q_acc

        model.zero_grad()
        for n, p in model.named_parameters():
            if n in meta_grads:
                p.grad = meta_grads[n]
        optimizer.step()
        optimizer.zero_grad()

        if (it + 1) % 200 == 0:
            avg_loss = running_loss / (200 * meta_batch)
            avg_acc = running_acc / (200 * meta_batch)
            print(f"  迭代 {it+1}/{num_iter} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
            running_loss = 0.0
            running_acc = 0.0


# ============================================================
# MAML + VAT（有 SSL）
# ============================================================

def train_vat(model, region_info, num_iter=3000,
              alpha=0.4, beta=0.001, meta_batch=8,
              n_way=5, n_support=1, n_unlabeled=20, n_query=10,
              vat_eps=2.0, vat_alpha=1.0, device='cpu'):
    optimizer = Adam(model.parameters(), lr=beta)
    vat_fn = VATLoss(xi=0.1, eps=vat_eps, ip=1)

    running_loss = 0.0
    running_acc = 0.0
    running_vat = 0.0

    for it in range(num_iter):
        meta_grads = {}

        for task_i in range(meta_batch):
            sx, sy, ux, qx, qy = sample_task(
                region_info, n_way, n_support, n_unlabeled, n_query,
                seed=it * 1000 + task_i
            )
            sx, sy = sx.to(device), sy.to(device)
            ux = ux.to(device)
            qx, qy = qx.to(device), qy.to(device)

            orig = {n: p.data.clone() for n, p in model.named_parameters()}

            # 内循环
            model.zero_grad()
            sup_loss = F.cross_entropy(model(sx), sy)
            sup_loss.backward()

            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.data -= alpha * p.grad

            # 外循环：联合损失
            model.zero_grad()
            q_logits = model(qx)
            q_loss = F.cross_entropy(q_logits, qy)

            vat_loss_val = vat_fn(model, ux)

            meta_loss = q_loss + vat_alpha * vat_loss_val
            meta_loss.backward()

            with torch.no_grad():
                q_acc = (q_logits.argmax(1) == qy).float().mean().item()

            for n, p in model.named_parameters():
                if p.grad is not None:
                    if n not in meta_grads:
                        meta_grads[n] = p.grad.clone() / meta_batch
                    else:
                        meta_grads[n] += p.grad.clone() / meta_batch

            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.data = orig[n]

            running_loss += q_loss.item()
            running_acc += q_acc
            running_vat += vat_loss_val.item()

        model.zero_grad()
        for n, p in model.named_parameters():
            if n in meta_grads:
                p.grad = meta_grads[n]
        optimizer.step()
        optimizer.zero_grad()

        if (it + 1) % 200 == 0:
            avg_loss = running_loss / (200 * meta_batch)
            avg_acc = running_acc / (200 * meta_batch)
            avg_vat = running_vat / (200 * meta_batch)
            print(f"  迭代 {it+1}/{num_iter} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"VAT: {avg_vat:.4f} | "
                  f"Acc: {avg_acc:.4f}")
            running_loss = 0.0
            running_acc = 0.0
            running_vat = 0.0


# ============================================================
# Fine-tune 评估
# ============================================================

def fine_tune_eval(model, sx, sy, qx, qy, lr=0.4, max_steps=5, device='cpu'):
    model = copy.deepcopy(model)
    optimizer = SGD(model.parameters(), lr=lr)
    accs = []

    model.eval()
    with torch.no_grad():
        acc = (model(qx).argmax(1) == qy).float().mean().item()
    accs.append(acc)

    model.train()
    for _ in range(max_steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(sx), sy)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            acc = (model(qx).argmax(1) == qy).float().mean().item()
        accs.append(acc)
        model.train()

    return accs


# ============================================================
# 主实验
# ============================================================

def run_experiment(data_dir='./mnist_data', num_iter=2000, meta_batch=4,
                   n_support=1, n_unlabeled=20, n_query=10,
                   fine_tune_steps=5, num_test_tasks=15,
                   alpha=0.4, beta=0.001,
                   vat_eps=2.0, vat_alpha=1.0, device='cpu'):
    n_way = 5
    n_regions = 2

    print("=" * 70)
    print("MAML + VAT 联合实验 —— 方案A：5-way 1-shot，2个区域")
    print("=" * 70)
    print(f"区域数: {n_regions}（每区5个数字类别）")
    print(f"任务配置: {n_way}-way {n_support}-shot")
    print(f"无标签样本/类: {n_unlabeled}")
    print(f"元训练迭代: {num_iter}")
    print(f"元批大小: {meta_batch}")
    print(f"VAT: eps={vat_eps}, alpha={vat_alpha}")
    print(f"内循环α: {alpha}, 外循环β: {beta}")
    print(f"测试任务/区域: {num_test_tasks}")
    print(f"Fine-tune步数: {fine_tune_steps}")
    print("=" * 70)

    print("\n=== 创建 2 个合成区域 ===")
    regions_train, regions_test = create_2_regions(data_dir)

    all_bl = []
    all_vat = []

    for r_idx in range(n_regions):
        print(f"\n{'='*70}")
        print(f"区域 {r_idx+1}/{n_regions}  (数字 {regions_train[r_idx]['classes']})")
        print(f"{'='*70}")

        # Baseline
        print(f"\n[区域{r_idx+1}] 训练标准 MAML（无SSL）...")
        torch.manual_seed(42 + r_idx)
        np.random.seed(42 + r_idx)
        random.seed(42 + r_idx)

        bl_model = MAMLNet(n_way=n_way).to(device)
        train_baseline(
            bl_model, regions_train[r_idx],
            num_iter=num_iter, alpha=alpha, beta=beta,
            meta_batch=meta_batch, n_way=n_way,
            n_support=n_support, n_query=n_query, device=device
        )

        # MAML + VAT
        print(f"\n[区域{r_idx+1}] 训练 MAML + VAT（有SSL）...")
        torch.manual_seed(42 + r_idx)
        np.random.seed(42 + r_idx)
        random.seed(42 + r_idx)

        vat_model = MAMLNet(n_way=n_way).to(device)
        train_vat(
            vat_model, regions_train[r_idx],
            num_iter=num_iter, alpha=alpha, beta=beta,
            meta_batch=meta_batch, n_way=n_way,
            n_support=n_support, n_unlabeled=n_unlabeled,
            n_query=n_query, vat_eps=vat_eps, vat_alpha=vat_alpha,
            device=device
        )

        # 评估
        print(f"\n[区域{r_idx+1}] 评估 {num_test_tasks} 个测试任务...")
        bl_accs = []
        vat_accs = []

        for t in range(num_test_tasks):
            sx, sy, _, qx, qy = sample_task(
                regions_test[r_idx], n_way, n_support, 0, n_query,
                seed=9000 + r_idx * 100 + t
            )
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)

            bl_a = fine_tune_eval(bl_model, sx, sy, qx, qy,
                                  lr=alpha, max_steps=fine_tune_steps, device=device)
            vat_a = fine_tune_eval(vat_model, sx, sy, qx, qy,
                                   lr=alpha, max_steps=fine_tune_steps, device=device)
            bl_accs.append(bl_a)
            vat_accs.append(vat_a)

        bl_accs = np.array(bl_accs)
        vat_accs = np.array(vat_accs)
        all_bl.append(bl_accs)
        all_vat.append(vat_accs)

        bl_m = bl_accs.mean(axis=0)
        vat_m = vat_accs.mean(axis=0)
        print(f"\n  区域{r_idx+1} 结果 (平均 {num_test_tasks} 任务):")
        print(f"  {'步数':>6} | {'无SSL':>8} | {'有SSL':>8} | {'差值':>8}")
        print(f"  {'-'*42}")
        for s in range(fine_tune_steps + 1):
            print(f"  {s:>6} | {bl_m[s]:>8.2%} | {vat_m[s]:>8.2%} | {vat_m[s]-bl_m[s]:>+8.2%}")

    # 汇总
    print(f"\n{'='*70}")
    print(f"汇总：{n_regions} 个区域的平均结果（最终步）")
    print(f"{'='*70}")
    final_bl = [r[:, -1].mean() for r in all_bl]
    final_vat = [r[:, -1].mean() for r in all_vat]

    print(f"\n{'区域':>8} | {'类别':>14} | {'无SSL':>8} | {'有SSL':>8} | {'提升':>8}")
    print(f"{'-'*55}")
    for i in range(n_regions):
        cls = regions_train[i]['classes']
        print(f"  区域{i+1:>2} | {str(cls):>14} | "
              f"{final_bl[i]:>8.2%} | {final_vat[i]:>8.2%} | "
              f"{final_vat[i]-final_bl[i]:>+8.2%}")

    avg_bl = np.mean(final_bl)
    avg_vat = np.mean(final_vat)
    print(f"{'-'*55}")
    print(f"  {'平均':>6} | {'':>14} | {avg_bl:>8.2%} | {avg_vat:>8.2%} | "
          f"{avg_vat-avg_bl:>+8.2%}")

    # 画图
    plot_results(all_bl, all_vat, regions_train, fine_tune_steps,
                 n_support, n_way, num_test_tasks, alpha, vat_alpha, vat_eps)

    # 保存结果
    save_results(all_bl, all_vat, regions_train, fine_tune_steps,
                 n_support, n_unlabeled, n_way, num_test_tasks,
                 num_iter, meta_batch, alpha, beta, vat_eps, vat_alpha)

    return all_bl, all_vat


# ============================================================
# 画图
# ============================================================

def plot_results(all_bl, all_vat, regions_train, max_steps,
                 n_support, n_way, num_tasks, alpha, vat_alpha, vat_eps):
    steps = np.arange(max_steps + 1)
    n_regions = len(all_bl)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1：各区域最终步对比
    ax1 = axes[0, 0]
    final_bl = [r[:, -1].mean() * 100 for r in all_bl]
    final_vat = [r[:, -1].mean() * 100 for r in all_vat]
    x = np.arange(n_regions)
    w = 0.35
    b1 = ax1.bar(x - w/2, final_bl, w, label='无SSL (MAML)', color='steelblue', alpha=0.85)
    b2 = ax1.bar(x + w/2, final_vat, w, label='有SSL (MAML+VAT)', color='coral', alpha=0.85)
    for b in b1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                 f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for b in b2:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                 f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    labels = [f"区域{i+1}\n{regions_train[i]['classes']}" for i in range(n_regions)]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel(f'{max_steps}步后准确率 (%)')
    ax1.set_title(f'{n_way}-way {n_support}-shot：各区域最终准确率对比')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(max(final_bl), max(final_vat)) * 1.2)

    # 图2：平均学习曲线
    ax2 = axes[0, 1]
    bl_all = np.concatenate(all_bl, axis=0)
    vat_all = np.concatenate(all_vat, axis=0)
    bl_m = bl_all.mean(axis=0) * 100
    bl_s = bl_all.std(axis=0) * 100
    vat_m = vat_all.mean(axis=0) * 100
    vat_s = vat_all.std(axis=0) * 100

    ax2.plot(steps, bl_m, 'b-o', label='无SSL (MAML)', markersize=6, linewidth=2)
    ax2.fill_between(steps, bl_m - bl_s, bl_m + bl_s, alpha=0.15, color='blue')
    ax2.plot(steps, vat_m, 'r-s', label='有SSL (MAML+VAT)', markersize=6, linewidth=2)
    ax2.fill_between(steps, vat_m - vat_s, vat_m + vat_s, alpha=0.15, color='red')
    ax2.set_xlabel('Fine-tune 梯度步数')
    ax2.set_ylabel('Query 准确率 (%)')
    ax2.set_title(f'平均学习曲线 ({n_regions}区域 × {num_tasks}任务)')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.3, max_steps + 0.3)
    ax2.set_ylim(0, 105)
    ax2.set_xticks(steps)

    # 图3：提升幅度
    ax3 = axes[1, 0]
    imp = [(v - b) for b, v in zip(final_bl, final_vat)]
    colors = ['#2ca02c' if i > 0 else '#d62728' for i in imp]
    bars = ax3.bar(range(n_regions), imp, color=colors, alpha=0.85)
    ax3.axhline(y=0, color='black', linewidth=0.8)
    for b, i in zip(bars, imp):
        h = b.get_height()
        va = 'bottom' if h >= 0 else 'top'
        off = 0.2 if h >= 0 else -0.2
        ax3.text(b.get_x() + b.get_width()/2, h + off,
                 f'{h:+.2f}%', ha='center', va=va, fontsize=10, fontweight='bold')
    ax3.set_xticks(range(n_regions))
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel('准确率提升 (%)')
    ax3.set_title(f'有SSL 相对 无SSL 的提升 (VAT α={vat_alpha}, ε={vat_eps})')
    ax3.grid(True, alpha=0.3, axis='y')

    # 图4：各步对比
    ax4 = axes[1, 1]
    step_labels = [f'{s}步' for s in steps]
    x2 = np.arange(len(steps))
    w2 = 0.35
    ax4.bar(x2 - w2/2, bl_m, w2, label='无SSL (MAML)', color='steelblue', alpha=0.85)
    ax4.bar(x2 + w2/2, vat_m, w2, label='有SSL (MAML+VAT)', color='coral', alpha=0.85)
    for i, (b, v) in enumerate(zip(bl_m, vat_m)):
        ax4.text(i - w2/2, b + 0.5, f'{b:.1f}', ha='center', va='bottom', fontsize=8)
        ax4.text(i + w2/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    ax4.set_xlabel('Fine-tune 步数')
    ax4.set_ylabel('准确率 (%)')
    ax4.set_title('各步平均准确率对比')
    ax4.set_xticks(x2)
    ax4.set_xticklabels(step_labels)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, max(max(bl_m), max(vat_m)) * 1.2)

    plt.tight_layout()
    save_path = 'maml_vat_5way1shot_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存: {save_path}")
    plt.close()


# ============================================================
# 保存结果
# ============================================================

def save_results(all_bl, all_vat, regions_train, max_steps,
                 n_support, n_unlabeled, n_way, num_test_tasks,
                 num_iter, meta_batch, alpha, beta, vat_eps, vat_alpha):
    path = 'maml_vat_5way1shot_results.txt'
    n_regions = len(all_bl)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MAML + VAT 联合实验结果 —— 方案A：5-way 1-shot\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"区域数: {n_regions}\n")
        f.write(f"任务: {n_way}-way {n_support}-shot\n")
        f.write(f"无标签/类: {n_unlabeled}\n")
        f.write(f"元迭代: {num_iter}\n")
        f.write(f"元批大小: {meta_batch}\n")
        f.write(f"α={alpha}, β={beta}\n")
        f.write(f"VAT eps={vat_eps}, alpha={vat_alpha}\n")
        f.write(f"测试任务/区域: {num_test_tasks}\n")
        f.write(f"Fine-tune步数: {max_steps}\n\n")

        f.write("各区域最终步准确率:\n")
        f.write(f"{'区域':>8} | {'类别':>14} | {'无SSL':>8} | {'有SSL':>8} | {'提升':>8}\n")
        f.write("-" * 55 + "\n")
        final_bl, final_vat = [], []
        for i in range(n_regions):
            bl = all_bl[i][:, -1].mean()
            va = all_vat[i][:, -1].mean()
            final_bl.append(bl)
            final_vat.append(va)
            cls = regions_train[i]['classes']
            f.write(f"  区域{i+1:>2} | {str(cls):>14} | "
                    f"{bl:>8.2%} | {va:>8.2%} | {va-bl:>+8.2%}\n")
        avg_bl = np.mean(final_bl)
        avg_vat = np.mean(final_vat)
        f.write("-" * 55 + "\n")
        f.write(f"  {'平均':>6} | {'':>14} | {avg_bl:>8.2%} | {avg_vat:>8.2%} | "
                f"{avg_vat-avg_bl:>+8.2%}\n\n")

        f.write("各步平均准确率:\n")
        bl_all = np.concatenate(all_bl, axis=0)
        vat_all = np.concatenate(all_vat, axis=0)
        bl_m = bl_all.mean(axis=0)
        vat_m = vat_all.mean(axis=0)
        f.write(f"{'步数':>6} | {'无SSL':>8} | {'有SSL':>8} | {'差值':>8}\n")
        f.write("-" * 42 + "\n")
        for s in range(max_steps + 1):
            f.write(f"{s:>6} | {bl_m[s]:>8.2%} | {vat_m[s]:>8.2%} | "
                    f"{vat_m[s]-bl_m[s]:>+8.2%}\n")

    print(f"结果已保存: {path}")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='MAML+VAT 方案A：5-way 1-shot')
    parser.add_argument('--data_dir', type=str, default='./mnist_data')
    parser.add_argument('--num_iter', type=int, default=300)
    parser.add_argument('--meta_batch', type=int, default=4)
    parser.add_argument('--n_support', type=int, default=1)
    parser.add_argument('--n_unlabeled', type=int, default=20)
    parser.add_argument('--n_query', type=int, default=10)
    parser.add_argument('--fine_tune_steps', type=int, default=5)
    parser.add_argument('--num_test_tasks', type=int, default=15)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--vat_eps', type=float, default=2.0)
    parser.add_argument('--vat_alpha', type=float, default=1.0)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    run_experiment(
        data_dir=args.data_dir,
        num_iter=args.num_iter,
        meta_batch=args.meta_batch,
        n_support=args.n_support,
        n_unlabeled=args.n_unlabeled,
        n_query=args.n_query,
        fine_tune_steps=args.fine_tune_steps,
        num_test_tasks=args.num_test_tasks,
        alpha=args.alpha,
        beta=args.beta,
        vat_eps=args.vat_eps,
        vat_alpha=args.vat_alpha,
        device=device,
    )


if __name__ == '__main__':
    main()
