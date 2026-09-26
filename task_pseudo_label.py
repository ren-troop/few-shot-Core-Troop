"""
任务：在MNIST上实验Pseudo-Labeling
有标签100张 + 无标签10000张，记录伪标签效果

论文：Pseudo-Label: The Simple and Efficient Semi-Supervised Learning
      Method for Deep Neural Networks (Lee, 2013)

实验设置（对应论文表2）：
  - 有标签数据：100张（每类10张）
  - 无标签数据：10000张
  - 测试集：10000张
  - 网络结构：2个卷积层 + 2个全连接层（论文中的dropNN）

论文公式对应：
  - 公式(14)：伪标签 = argmax(预测概率)
  - 公式(15)：总loss = 有标签loss + α × 无标签loss
  - 公式(16)：α从0线性增长到3

运行方法：
  python task_pseudo_label.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第1部分：定义网络（对应论文表2中的dropNN）
# ============================================================

class SimpleCNN(nn.Module):
    """
    网络结构对应论文表2的dropNN：
    - 2个卷积层（提取特征）
    - 2个全连接层（分类）
    - ReLU激活函数（论文第2.1节）
    - 输出层用sigmoid（论文第2.1节明确说使用sigmoid输出单元）
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   # 论文：卷积层C1
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)   # 论文：卷积层C3
        self.fc1 = nn.Linear(320, 50)                   # 论文：全连接层F5
        self.fc2 = nn.Linear(50, 10)                    # 论文：输出层

    def forward(self, x):
        x = F.relu(self.conv1(x))     # (1,28,28)→(10,24,24)
        x = F.max_pool2d(x, 2)       # →(10,12,12)
        x = F.relu(self.conv2(x))    # →(20,8,8)
        x = F.max_pool2d(x, 2)       # →(20,4,4)
        x = torch.flatten(x, 1)      # →(320,)
        x = F.relu(self.fc1(x))      # →(50,)
        x = self.fc2(x)              # →(10,) 论文用sigmoid，但配合交叉熵等价于softmax
        return x


# ============================================================
# 第2部分：数据加载（对应论文表2的实验设置）
# ============================================================

def load_data(n_labeled=100, n_unlabeled=10000, batch_size=32):
    """
    论文表2设置：
    - n_labeled=100：只用100个有标签数据
    - 剩余数据作为无标签数据
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 加载完整训练集（60000张）
    train_dataset = datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./mnist_data', train=False, download=True, transform=transform)

    # 每类选10张有标签数据，共100张（论文表2的100 labeled设置）
    labels = train_dataset.targets.numpy()
    labeled_indices = []
    for cls in range(10):
        cls_indices = np.where(labels == cls)[0]
        selected = np.random.choice(cls_indices, 10, replace=False)
        labeled_indices.extend(selected)
    labeled_indices = np.array(labeled_indices)

    # 无标签数据：从剩余数据中选10000张
    remaining_indices = np.setdiff1d(np.arange(len(train_dataset)), labeled_indices)
    unlabeled_indices = np.random.choice(remaining_indices, n_unlabeled, replace=False)

    # 创建子集
    labeled_dataset = Subset(train_dataset, labeled_indices)
    unlabeled_dataset = Subset(train_dataset, unlabeled_indices)

    # 有标签数据加载器
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)

    # 无标签数据加载器（只取图片，不取标签）
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=True)

    # 测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return labeled_loader, unlabeled_loader, test_loader


# ============================================================
# 第3部分：训练函数
# ============================================================

def train_baseline(model, device, labeled_loader, optimizer, epoch):
    """
    Baseline：只用有标签数据训练（论文表2的dropNN列）
    """
    model.train()
    total_loss = 0
    for data, target in labeled_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(labeled_loader)


def train_pseudo_label(model, device, labeled_loader, unlabeled_loader,
                       optimizer, epoch, alpha):
    """
    Pseudo-Label训练（对应论文公式15）

    论文公式(15)：
      L = (1/n) * Σ L(y_i, f_i)  +  α(t) * (1/n') * Σ L(y'_i, f'_i)
      |____有标签部分____|         |______无标签部分______|

    论文公式(14)：
      y'_i = argmax(f_i)  → 伪标签 = 预测概率最大的类

    论文公式(16)：
      α(t) = 0,                 当 t < T1
      α(t) = α_f * (t-T1)/(T2-T1), 当 T1 ≤ t < T2
      α(t) = α_f,               当 t ≥ T2
      其中 α_f=3, T1=100, T2=600
    """
    model.train()
    labeled_loss_total = 0
    unlabeled_loss_total = 0
    total_loss_total = 0

    # 把无标签数据做成迭代器，方便循环取
    unlabeled_iter = iter(unlabeled_loader)

    for batch_idx, (labeled_data, labeled_target) in enumerate(labeled_loader):
        labeled_data = labeled_data.to(device)
        labeled_target = labeled_target.to(device)

        # === 有标签部分 ===
        optimizer.zero_grad()
        labeled_output = model(labeled_data)
        labeled_loss = F.cross_entropy(labeled_output, labeled_target)

        # === 无标签部分（公式14+15）===
        try:
            unlabeled_data, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_data, _ = next(unlabeled_iter)
        unlabeled_data = unlabeled_data.to(device)

        with torch.no_grad():
            unlabeled_output = model(unlabeled_data)
            # 论文公式(14)：伪标签 = argmax(预测概率)
            pseudo_labels = unlabeled_output.argmax(dim=1)

        # 用伪标签训练无标签数据
        unlabeled_output = model(unlabeled_data)
        unlabeled_loss = F.cross_entropy(unlabeled_output, pseudo_labels)

        # 论文公式(15)：总loss = 有标签loss + α × 无标签loss
        total_loss = labeled_loss + alpha * unlabeled_loss
        total_loss.backward()
        optimizer.step()

        labeled_loss_total += labeled_loss.item()
        unlabeled_loss_total += unlabeled_loss.item()
        total_loss_total += total_loss.item()

    return (labeled_loss_total / len(labeled_loader),
            unlabeled_loss_total / len(labeled_loader),
            total_loss_total / len(labeled_loader))


def test(model, device, test_loader):
    """测试函数"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


# ============================================================
# 第4部分：主函数
# ============================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 论文表2设置：100个有标签数据
    n_labeled = 100
    n_unlabeled = 10000
    n_epochs = 50

    print(f"\n{'='*60}")
    print(f"Pseudo-Labeling 实验")
    print(f"有标签数据: {n_labeled}张")
    print(f"无标签数据: {n_unlabeled}张")
    print(f"训练轮数: {n_epochs}轮")
    print(f"{'='*60}")

    # 加载数据
    print("\n[1/4] 加载MNIST数据...")
    labeled_loader, unlabeled_loader, test_loader = load_data(
        n_labeled=n_labeled, n_unlabeled=n_unlabeled
    )
    print(f"  有标签batch数: {len(labeled_loader)}")
    print(f"  无标签batch数: {len(unlabeled_loader)}")

    # ============================================================
    # 实验1：Baseline（只用100个有标签数据）
    # ============================================================
    print(f"\n[2/4] 训练Baseline（只用{n_labeled}个有标签数据）...")
    model_baseline = SimpleCNN().to(device)
    optimizer_baseline = optim.SGD(
        model_baseline.parameters(), lr=0.01, momentum=0.9
    )

    baseline_accs = []
    baseline_alphas = []

    for epoch in range(1, n_epochs + 1):
        loss = train_baseline(
            model_baseline, device, labeled_loader, optimizer_baseline, epoch
        )
        test_loss, test_acc = test(model_baseline, device, test_loader)
        baseline_accs.append(test_acc)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    print(f"\n  Baseline最终准确率: {baseline_accs[-1]:.2f}%")

    # ============================================================
    # 实验2：Pseudo-Label（100个有标签 + 10000个无标签）
    # ============================================================
    print(f"\n[3/4] 训练Pseudo-Label（{n_labeled}有标签 + {n_unlabeled}无标签）...")
    model_pl = SimpleCNN().to(device)
    optimizer_pl = optim.SGD(
        model_pl.parameters(), lr=0.01, momentum=0.9
    )

    # 论文公式(16)的参数
    alpha_f = 3.0      # 论文：α_f = 3
    T1 = 10            # 前10轮α=0（论文T1=100步，这里按epoch缩放）
    T2 = 30            # 第10-30轮α线性增长（论文T2=600步）

    pl_accs = []
    pl_alphas = []
    pl_labeled_losses = []
    pl_unlabeled_losses = []

    for epoch in range(1, n_epochs + 1):
        # 论文公式(16)：计算α(t)
        if epoch < T1:
            alpha = 0.0
        elif epoch < T2:
            alpha = alpha_f * (epoch - T1) / (T2 - T1)
        else:
            alpha = alpha_f

        labeled_loss, unlabeled_loss, total_loss = train_pseudo_label(
            model_pl, device, labeled_loader, unlabeled_loader,
            optimizer_pl, epoch, alpha
        )
        test_loss, test_acc = test(model_pl, device, test_loader)

        pl_accs.append(test_acc)
        pl_alphas.append(alpha)
        pl_labeled_losses.append(labeled_loss)
        pl_unlabeled_losses.append(unlabeled_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | α={alpha:.2f} | "
                  f"Labeled Loss: {labeled_loss:.4f} | "
                  f"Unlabeled Loss: {unlabeled_loss:.4f} | "
                  f"Test Acc: {test_acc:.2f}%")

    print(f"\n  Pseudo-Label最终准确率: {pl_accs[-1]:.2f}%")

    # ============================================================
    # 实验3：Oracle（用全部60000个有标签数据，作为上限参考）
    # ============================================================
    print(f"\n[4/4] 训练Oracle（全部60000个有标签数据，上限参考）...")
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
    full_loader = DataLoader(full_train, batch_size=64, shuffle=True)

    model_oracle = SimpleCNN().to(device)
    optimizer_oracle = optim.SGD(
        model_oracle.parameters(), lr=0.01, momentum=0.9
    )

    oracle_accs = []

    for epoch in range(1, n_epochs + 1):
        loss = train_baseline(
            model_oracle, device, full_loader, optimizer_oracle, epoch
        )
        test_loss, test_acc = test(model_oracle, device, test_loader)
        oracle_accs.append(test_acc)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Test Acc: {test_acc:.2f}%")

    print(f"\n  Oracle最终准确率: {oracle_accs[-1]:.2f}%")

    # ============================================================
    # 汇总结果
    # ============================================================
    print(f"\n{'='*60}")
    print(f"实验结果汇总")
    print(f"{'='*60}")
    print(f"{'方法':<20} {'最终准确率':<15} {'最高准确率':<15}")
    print(f"{'-'*50}")
    print(f"{'Baseline (100 labeled)':<20} {baseline_accs[-1]:<15.2f} {max(baseline_accs):<15.2f}")
    print(f"{'Pseudo-Label (100+10000)':<20} {pl_accs[-1]:<15.2f} {max(pl_accs):<15.2f}")
    print(f"{'Oracle (60000 labeled)':<20} {oracle_accs[-1]:<15.2f} {max(oracle_accs):<15.2f}")
    print(f"{'-'*50}")
    improvement = pl_accs[-1] - baseline_accs[-1]
    print(f"Pseudo-Label相对Baseline提升: {improvement:+.2f}%")
    print(f"{'='*60}")

    # ============================================================
    # 保存训练记录到文件
    # ============================================================
    save_path = 'pseudo_label_results.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Pseudo-Labeling 实验记录\n")
        f.write(f"有标签数据: {n_labeled}张\n")
        f.write(f"无标签数据: {n_unlabeled}张\n")
        f.write(f"训练轮数: {n_epochs}轮\n\n")

        f.write("Baseline准确率变化:\n")
        for i, acc in enumerate(baseline_accs):
            f.write(f"  Epoch {i+1:3d}: {acc:.2f}%\n")

        f.write(f"\nPseudo-Label准确率变化:\n")
        for i, (acc, alpha) in enumerate(zip(pl_accs, pl_alphas)):
            f.write(f"  Epoch {i+1:3d}: {acc:.2f}%  (α={alpha:.3f})\n")

        f.write(f"\nOracle准确率变化:\n")
        for i, acc in enumerate(oracle_accs):
            f.write(f"  Epoch {i+1:3d}: {acc:.2f}%\n")

        f.write(f"\n最终结果:\n")
        f.write(f"  Baseline: {baseline_accs[-1]:.2f}%\n")
        f.write(f"  Pseudo-Label: {pl_accs[-1]:.2f}%\n")
        f.write(f"  Oracle: {oracle_accs[-1]:.2f}%\n")
        f.write(f"  提升: {improvement:+.2f}%\n")

    print(f"\n训练记录已保存到: {save_path}")

    # ============================================================
    # 画图
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1：准确率对比
    ax1 = axes[0]
    ax1.plot(range(1, n_epochs+1), baseline_accs, 'b-', label='Baseline (100 labeled)', linewidth=2)
    ax1.plot(range(1, n_epochs+1), pl_accs, 'r-', label='Pseudo-Label (100+10000)', linewidth=2)
    ax1.plot(range(1, n_epochs+1), oracle_accs, 'g--', label='Oracle (60000 labeled)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Pseudo-Labeling效果对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2：α的变化
    ax2 = axes[1]
    ax2.plot(range(1, n_epochs+1), pl_alphas, 'purple', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('α (无标签loss权重)')
    ax2.set_title('论文公式(16): α的退火调度')
    ax2.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label='α_f=3')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3：loss变化
    ax3 = axes[2]
    ax3.plot(range(1, n_epochs+1), pl_labeled_losses, 'b-', label='Labeled Loss', linewidth=2)
    ax3.plot(range(1, n_epochs+1), pl_unlabeled_losses, 'r-', label='Unlabeled Loss', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Pseudo-Label训练Loss变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pseudo_label_results.png', dpi=150, bbox_inches='tight')
    print(f"对比图已保存到: pseudo_label_results.png")

    # 保存模型
    torch.save({
        'baseline_model': model_baseline.state_dict(),
        'pseudo_label_model': model_pl.state_dict(),
        'oracle_model': model_oracle.state_dict(),
        'baseline_accs': baseline_accs,
        'pl_accs': pl_accs,
        'oracle_accs': oracle_accs,
    }, 'pseudo_label_models.pth')
    print(f"模型已保存到: pseudo_label_models.pth")


if __name__ == '__main__':
    main()
