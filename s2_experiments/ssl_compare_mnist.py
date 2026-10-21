"""
三种半监督方法对比实验：Pseudo-Label vs VAT vs FixMatch
数据集：MNIST（100 张标注 + 10000 张无标签）

用法：
    python ssl_compare_mnist.py --num_epochs 60 --n_labeled 100

PyCharm直接运行：右键 → Run，约3-5分钟出结果
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssl_modules import PseudoLabelLoss, VATLoss, FixMatchLoss, weak_augment, strong_augment


# ============================================================
# 模型
# ============================================================

class SimpleCNN(nn.Module):
    """简单 CNN（4 层卷积 + 1 层全连接）"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        # 28 -> 14 -> 7 -> 4
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# 数据加载
# ============================================================

def get_mnist_loaders(n_labeled=100, n_unlabeled=10000, batch_size=64, data_dir='./mnist_data'):
    """
    构造半监督 MNIST 数据：
    - labeled_train: n_labeled 张有标签（每类均匀）
    - unlabeled_train: n_unlabeled 张无标签
    - test: 完整测试集
    """
    transform = transforms.Compose([transforms.ToTensor()])

    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # 从训练集中选 n_labeled 张有标签（每类均匀）
    targets = np.array(full_train.targets)
    labeled_indices = []
    per_class = n_labeled // 10
    for c in range(10):
        class_idx = np.where(targets == c)[0]
        chosen = np.random.choice(class_idx, size=per_class, replace=False)
        labeled_indices.extend(chosen)

    # 剩下的选 n_unlabeled 张作为无标签
    all_idx = set(range(len(full_train)))
    labeled_set = set(labeled_indices)
    remaining = list(all_idx - labeled_set)
    unlabeled_indices = np.random.choice(remaining, size=n_unlabeled, replace=False)

    labeled_dataset = Subset(full_train, labeled_indices)
    unlabeled_dataset = Subset(full_train, unlabeled_indices)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"  有标签数据: {len(labeled_indices)} 张")
    print(f"  无标签数据: {len(unlabeled_indices)} 张")
    print(f"  测试数据: {len(test_dataset)} 张")

    return labeled_loader, unlabeled_loader, test_loader


# ============================================================
# 评估
# ============================================================

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ============================================================
# 训练：纯监督 baseline
# ============================================================

def train_baseline(labeled_loader, test_loader, num_epochs, lr, device):
    print("\n=== 训练 Baseline（纯监督） ===")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    test_accs = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for x, y in labeled_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        acc = evaluate(model, test_loader, device)
        test_accs.append(acc)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/n_batches:.4f} | Test Acc: {acc:.4f}")

    return model, test_accs


# ============================================================
# 训练：Pseudo-Label
# ============================================================

def train_pseudo_label(labeled_loader, unlabeled_loader, test_loader,
                       num_epochs, lr, alpha_final, T1, T2, device):
    print(f"\n=== 训练 Pseudo-Label (α_f={alpha_final}, T1={T1}, T2={T2}) ===")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pl_loss_fn = PseudoLabelLoss(alpha_final=alpha_final, T1=T1, T2=T2)
    test_accs = []

    # 计算总步数（用于 α 调度）
    total_steps_per_epoch = min(len(labeled_loader), len(unlabeled_loader))

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_sup_loss = 0
        total_pl_loss = 0
        n_batches = 0

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        for _ in range(total_steps_per_epoch):
            try:
                lx, ly = next(labeled_iter)
                ux, _ = next(unlabeled_iter)
            except StopIteration:
                break

            lx, ly = lx.to(device), ly.to(device)
            ux = ux.to(device)

            optimizer.zero_grad()

            # 监督损失
            sup_loss = F.cross_entropy(model(lx), ly)

            # 伪标签损失
            pl_loss = pl_loss_fn(model, ux)
            alpha = pl_loss_fn.get_alpha(global_step)

            total_loss_val = sup_loss + alpha * pl_loss
            total_loss_val.backward()
            optimizer.step()

            pl_loss_fn.step()
            global_step += 1

            total_loss += total_loss_val.item()
            total_sup_loss += sup_loss.item()
            total_pl_loss += pl_loss.item()
            n_batches += 1

        acc = evaluate(model, test_loader, device)
        test_accs.append(acc)
        if (epoch + 1) % 10 == 0:
            cur_alpha = pl_loss_fn.get_alpha()
            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Sup: {total_sup_loss/n_batches:.4f} | "
                  f"PL: {total_pl_loss/n_batches:.4f} | "
                  f"α={cur_alpha:.3f} | "
                  f"Test Acc: {acc:.4f}")

    return model, test_accs


# ============================================================
# 训练：VAT
# ============================================================

def train_vat(labeled_loader, unlabeled_loader, test_loader,
              num_epochs, lr, vat_eps, vat_alpha, device):
    print(f"\n=== 训练 VAT (eps={vat_eps}, alpha={vat_alpha}) ===")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    vat_fn = VATLoss(xi=0.1, eps=vat_eps, ip=1)
    test_accs = []

    total_steps_per_epoch = min(len(labeled_loader), len(unlabeled_loader))

    for epoch in range(num_epochs):
        model.train()
        total_sup_loss = 0
        total_vat_loss = 0
        n_batches = 0

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        for _ in range(total_steps_per_epoch):
            try:
                lx, ly = next(labeled_iter)
                ux, _ = next(unlabeled_iter)
            except StopIteration:
                break

            lx, ly = lx.to(device), ly.to(device)
            ux = ux.to(device)

            optimizer.zero_grad()

            # 监督损失
            sup_loss = F.cross_entropy(model(lx), ly)

            # VAT 损失（只算无标签数据上的，也可以加有标签的）
            vat_loss_val = vat_fn(model, ux)

            total_loss_val = sup_loss + vat_alpha * vat_loss_val
            total_loss_val.backward()
            optimizer.step()

            total_sup_loss += sup_loss.item()
            total_vat_loss += vat_loss_val.item()
            n_batches += 1

        acc = evaluate(model, test_loader, device)
        test_accs.append(acc)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Sup: {total_sup_loss/n_batches:.4f} | "
                  f"VAT: {total_vat_loss/n_batches:.4f} | "
                  f"Test Acc: {acc:.4f}")

    return model, test_accs


# ============================================================
# 训练：FixMatch
# ============================================================

def train_fixmatch(labeled_loader, unlabeled_loader, test_loader,
                   num_epochs, lr, threshold, lambda_u, device):
    print(f"\n=== 训练 FixMatch (τ={threshold}, λ_u={lambda_u}) ===")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    fm_fn = FixMatchLoss(threshold=threshold)
    test_accs = []

    total_steps_per_epoch = min(len(labeled_loader), len(unlabeled_loader))

    for epoch in range(num_epochs):
        model.train()
        total_sup_loss = 0
        total_fm_loss = 0
        total_mask = 0
        n_batches = 0

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        for _ in range(total_steps_per_epoch):
            try:
                lx, ly = next(labeled_iter)
                ux, _ = next(unlabeled_iter)
            except StopIteration:
                break

            lx, ly = lx.to(device), ly.to(device)
            ux = ux.to(device)

            # 对无标签数据做弱增强和强增强
            with torch.no_grad():
                weak_ux = weak_augment(ux, flip=False, translate=2)  # MNIST 不翻转
            strong_ux = strong_augment(ux, n=2, m=5, cutout_size=8, flip=False, translate=2)
            weak_ux = weak_ux.to(device)
            strong_ux = strong_ux.to(device)

            optimizer.zero_grad()

            # 监督损失
            sup_loss = F.cross_entropy(model(lx), ly)

            # FixMatch 损失
            fm_loss_val, mask_ratio = fm_fn(model, weak_ux, strong_ux)

            total_loss_val = sup_loss + lambda_u * fm_loss_val
            total_loss_val.backward()
            optimizer.step()

            total_sup_loss += sup_loss.item()
            total_fm_loss += fm_loss_val.item()
            total_mask += mask_ratio
            n_batches += 1

        acc = evaluate(model, test_loader, device)
        test_accs.append(acc)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Sup: {total_sup_loss/n_batches:.4f} | "
                  f"FM: {total_fm_loss/n_batches:.4f} | "
                  f"Mask: {total_mask/n_batches:.3f} | "
                  f"Test Acc: {acc:.4f}")

    return model, test_accs


# ============================================================
# 画图
# ============================================================

def plot_comparison(results, num_epochs, n_labeled, n_unlabeled):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：测试准确率曲线
    ax = axes[0]
    colors = {'Baseline': '#666666', 'Pseudo-Label': '#1f77b4', 'VAT': '#ff7f0e', 'FixMatch': '#2ca02c'}
    styles = {'Baseline': '--', 'Pseudo-Label': '-', 'VAT': '-', 'FixMatch': '-'}

    for name, accs in results.items():
        epochs = np.arange(1, len(accs) + 1)
        ax.plot(epochs, np.array(accs) * 100, label=name,
                color=colors.get(name, '#333'),
                linestyle=styles.get(name, '-'),
                linewidth=2, marker='o', markersize=3)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'三种半监督方法对比 (MNIST, {n_labeled}标注/{n_unlabeled}无标签)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右图：最终准确率柱状图
    ax = axes[1]
    names = list(results.keys())
    final_accs = [results[n][-1] * 100 for n in names]
    bar_colors = [colors.get(n, '#333') for n in names]

    bars = ax.bar(names, final_accs, color=bar_colors, alpha=0.85, width=0.6)
    for b, acc in zip(bars, final_accs):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
    ax.set_title(f'最终准确率对比', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    baseline_acc = final_accs[0]
    ax.set_ylim(min(final_accs) - 5, max(final_accs) + 3)

    plt.tight_layout()
    save_path = 'ssl_comparison_mnist.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存: {save_path}")
    plt.close()


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='三种半监督方法对比：Pseudo-Label vs VAT vs FixMatch')
    parser.add_argument('--n_labeled', type=int, default=100, help='有标签数据量')
    parser.add_argument('--n_unlabeled', type=int, default=10000, help='无标签数据量')
    parser.add_argument('--num_epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='batch大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    # Pseudo-Label 参数
    parser.add_argument('--pl_alpha', type=float, default=3.0, help='Pseudo-Label 最终α')
    parser.add_argument('--pl_T1', type=int, default=100, help='Pseudo-Label T1(步数)')
    parser.add_argument('--pl_T2', type=int, default=600, help='Pseudo-Label T2(步数)')
    # VAT 参数
    parser.add_argument('--vat_eps', type=float, default=2.0, help='VAT 扰动大小')
    parser.add_argument('--vat_alpha', type=float, default=1.0, help='VAT 损失权重')
    # FixMatch 参数
    parser.add_argument('--fm_threshold', type=float, default=0.95, help='FixMatch 置信度阈值')
    parser.add_argument('--fm_lambda_u', type=float, default=1.0, help='FixMatch 无监督损失权重')
    args = parser.parse_args()

    # 设种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    print("=" * 70)
    print("半监督学习三方法对比实验")
    print("=" * 70)
    print(f"数据集: MNIST")
    print(f"有标签: {args.n_labeled} 张 | 无标签: {args.n_unlabeled} 张")
    print(f"训练轮数: {args.num_epochs} | 学习率: {args.lr}")
    print("=" * 70)

    # 数据
    print("\n=== 加载数据 ===")
    labeled_loader, unlabeled_loader, test_loader = get_mnist_loaders(
        n_labeled=args.n_labeled, n_unlabeled=args.n_unlabeled,
        batch_size=args.batch_size
    )

    results = {}

    # 1. Baseline
    model_bl, accs_bl = train_baseline(labeled_loader, test_loader, args.num_epochs, args.lr, device)
    results['Baseline'] = accs_bl

    # 2. Pseudo-Label
    model_pl, accs_pl = train_pseudo_label(
        labeled_loader, unlabeled_loader, test_loader,
        args.num_epochs, args.lr, args.pl_alpha, args.pl_T1, args.pl_T2, device
    )
    results['Pseudo-Label'] = accs_pl

    # 3. VAT
    model_vat, accs_vat = train_vat(
        labeled_loader, unlabeled_loader, test_loader,
        args.num_epochs, args.lr, args.vat_eps, args.vat_alpha, device
    )
    results['VAT'] = accs_vat

    # 4. FixMatch
    model_fm, accs_fm = train_fixmatch(
        labeled_loader, unlabeled_loader, test_loader,
        args.num_epochs, args.lr, args.fm_threshold, args.fm_lambda_u, device
    )
    results['FixMatch'] = accs_fm

    # 汇总
    print("\n" + "=" * 70)
    print("最终结果汇总")
    print("=" * 70)
    print(f"{'方法':<18} {'最终准确率':>12} {'相对提升':>12}")
    print("-" * 44)
    baseline = results['Baseline'][-1]
    for name, accs in results.items():
        final = accs[-1]
        improve = final - baseline
        print(f"{name:<18} {final:>11.2%} {improve:>+11.2%}")

    # 画图
    plot_comparison(results, args.num_epochs, args.n_labeled, args.n_unlabeled)

    # 保存结果
    with open('ssl_comparison_results.txt', 'w', encoding='utf-8') as f:
        f.write("半监督学习三方法对比实验结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"数据集: MNIST\n")
        f.write(f"有标签: {args.n_labeled} 张 | 无标签: {args.n_unlabeled} 张\n")
        f.write(f"训练轮数: {args.num_epochs}\n\n")
        f.write(f"{'方法':<18} {'最终准确率':>12} {'相对提升':>12}\n")
        f.write("-" * 44 + "\n")
        for name, accs in results.items():
            final = accs[-1]
            improve = final - baseline
            f.write(f"{name:<18} {final:>11.2%} {improve:>+11.2%}\n")
    print("结果已保存: ssl_comparison_results.txt")


if __name__ == '__main__':
    main()
