# -*- coding: utf-8 -*-
"""
ssl_modules.py —— 三种半监督损失与数据增强（重建版）
================================================================================
ssl_compare_mnist.py 依赖这个模块，但磁盘上找不到它。按主脚本的调用方式重建：

    PseudoLabelLoss(alpha_final, T1, T2)
        __call__(model, ux)      -> loss tensor
        get_alpha(global_step)   -> 当前 alpha（0 → alpha_final 线性爬升）
        step()                   -> 内部步数 +1

    VATLoss(xi=0.1, eps, ip=1)
        __call__(model, ux)      -> loss tensor

    FixMatchLoss(threshold)
        __call__(model, weak_ux, strong_ux) -> (loss, mask_ratio)

    weak_augment(ux, flip=False, translate=2)                     -> tensor
    strong_augment(ux, n=2, m=5, cutout_size=8, flip=False, translate=2) -> tensor

接口约定与主脚本第 32 / 166 / 199 / 236 / 266 / 296 / 323-324 / 334 行的用法一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ 数据增强
def _shift(x, translate):
    """随机平移（逐样本），边缘用 0 填充。"""
    if translate <= 0:
        return x
    B = x.shape[0]
    out = torch.zeros_like(x)
    padded = F.pad(x, (translate, translate, translate, translate))
    for i in range(B):
        dx = int(torch.randint(-translate, translate + 1, (1,)).item())
        dy = int(torch.randint(-translate, translate + 1, (1,)).item())
        out[i] = padded[i, :, translate + dx:translate + dx + x.shape[-2],
                        translate + dy:translate + dy + x.shape[-1]]
    return out


def _flip(x):
    return torch.flip(x, dims=[-1])


def weak_augment(ux, flip=False, translate=2):
    """弱增强：随机平移（+ 可选水平翻转）。语义保持，标签不变。"""
    out = _shift(ux, translate)
    if flip and torch.rand(1).item() < 0.5:
        out = _flip(out)
    return out


def strong_augment(ux, n=2, m=5, cutout_size=8, flip=False, translate=2):
    """强增强：弱增强 + n 个 cutout 方块遮挡（每个边长 cutout_size）。"""
    out = weak_augment(ux, flip=flip, translate=translate)
    H, W = out.shape[-2], out.shape[-1]
    for _ in range(n):
        for i in range(out.shape[0]):
            cx = int(torch.randint(0, max(1, H - cutout_size), (1,)).item())
            cy = int(torch.randint(0, max(1, W - cutout_size), (1,)).item())
            out[i, :, cx:cx + cutout_size, cy:cy + cutout_size] = 0.0
    return out


# ------------------------------------------------------------------ 伪标签
class PseudoLabelLoss(nn.Module):
    """Lee (2013) Pseudo-Label。

    无标签样本用当前模型 argmax 打硬标签，算交叉熵。
    alpha 按步数从 0 线性爬升到 alpha_final（T1 步前为 0，T2 步后到达终值）。
    """

    def __init__(self, alpha_final=3.0, T1=100, T2=600):
        super().__init__()
        self.alpha_final = float(alpha_final)
        self.T1 = int(T1)
        self.T2 = max(int(T2), int(T1) + 1)
        self._t = 0

    def get_alpha(self, global_step=None):
        t = self._t if global_step is None else int(global_step)
        if t < self.T1:
            return 0.0
        if t >= self.T2:
            return self.alpha_final
        return self.alpha_final * (t - self.T1) / (self.T2 - self.T1)

    def step(self):
        self._t += 1

    def forward(self, model, ux):
        with torch.no_grad():
            logits = model(ux)
            pseudo = logits.argmax(dim=1)
        logp = F.log_softmax(model(ux), dim=1)
        return F.nll_loss(logp, pseudo)


# ------------------------------------------------------------------ VAT
class VATLoss(nn.Module):
    """Miyato et al. (2019) Virtual Adversarial Training。

    用幂迭代找「让输出分布变化最大」的扰动方向 r，然后最小化
    KL( p(y|x) || p(y|x+r) )。目标端 detach。
    """

    def __init__(self, xi=0.1, eps=1.0, ip=1):
        super().__init__()
        self.xi = float(xi)
        self.eps = float(eps)
        self.ip = int(ip)

    def forward(self, model, ux):
        with torch.no_grad():
            logits = model(ux)
            pred = F.softmax(logits, dim=1)

        # 随机初始化单位方向 d
        d = torch.randn_like(ux)
        d = d / (d.norm(p=2, dim=list(range(1, ux.dim())), keepdim=True) + 1e-12)

        # 幂迭代逼近最坏方向
        for _ in range(self.ip):
            d.requires_grad_(True)
            logits_r = model(ux + self.xi * d)
            logp_r = F.log_softmax(logits_r, dim=1)
            kl = F.kl_div(logp_r, pred, reduction="none").sum(dim=1).mean()
            grad = torch.autograd.grad(kl, d)[0]
            d = grad.detach()
            d = d / (d.norm(p=2, dim=list(range(1, ux.dim())),
                            keepdim=True) + 1e-12)

        # 用归一化方向缩放到 eps
        r = self.eps * d
        logits_r = model(ux + r)
        logp_r = F.log_softmax(logits_r, dim=1)
        return F.kl_div(logp_r, pred, reduction="batchmean")


# ------------------------------------------------------------------ FixMatch
class FixMatchLoss(nn.Module):
    """Sohn et al. (2020) FixMatch。

    弱增强预测 max softmax > threshold 的样本才参与；
    对强增强视图算交叉熵，目标端 detach。
    """

    def __init__(self, threshold=0.95):
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, model, weak_ux, strong_ux):
        with torch.no_grad():
            logits_w = model(weak_ux)
            probs_w = F.softmax(logits_w, dim=1)
            conf, pseudo = probs_w.max(dim=1)
            mask = conf.ge(self.threshold)

        mask_ratio = float(mask.float().mean().item())
        if mask.sum() == 0:
            return torch.zeros((), device=weak_ux.device, requires_grad=True), \
                mask_ratio

        logits_s = model(strong_ux)
        loss = F.cross_entropy(logits_s[mask], pseudo[mask])
        return loss, mask_ratio
