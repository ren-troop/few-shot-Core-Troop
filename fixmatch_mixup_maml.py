"""
FixMatch (Mixup Version) + MAML for Water Quality Prediction
===============================================================
Semi-supervised MAML inner-loop with FixMatch adapted for tabular regression.

Key adaptations from original FixMatch (Sohn et al., 2020):
  1. Regression: MSE loss replaces cross-entropy
  2. Tabular data: Mixup replaces RandAugment as strong augmentation
     (interpolated samples stay within convex hull of real data, physically valid)
  3. Confidence: prediction variance across 5 weak-augmented predictions
     (low variance = high confidence), replaces max class probability

Dataset: Water quality regression (MATLAB .mat format)
  - 423 train tasks, 282 test tasks
  - Each task: 37 monitoring sites x 11 physicochemical features
  - Label: normalized water quality score [0,1]

Usage:
  python fixmatch_mixup_maml.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import json
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 1. Data Loading & Task Construction
# ============================================================

def load_water_data(mat_path='data/water_quality/water_dataset.mat'):
    """Load water quality dataset in MAML task format.

    Expected .mat structure:
      X_tr: (1, 423) object array, each element -> (37, 11) feature matrix
      Y_tr: (37, 423) label matrix
      X_te: (1, 282) object array
      Y_te: (37, 282) label matrix
    """
    data = sio.loadmat(mat_path)
    X_tr_cell = data['X_tr']
    Y_tr = data['Y_tr']
    X_te_cell = data['X_te']
    Y_te = data['Y_te']

    train_tasks = []
    for i in range(X_tr_cell.shape[1]):
        X = torch.FloatTensor(X_tr_cell[0, i])   # (37, 11)
        Y = torch.FloatTensor(Y_tr[:, i]).unsqueeze(1)  # (37, 1)
        train_tasks.append((X, Y))

    test_tasks = []
    for i in range(X_te_cell.shape[1]):
        X = torch.FloatTensor(X_te_cell[0, i])
        Y = torch.FloatTensor(Y_te[:, i]).unsqueeze(1)
        test_tasks.append((X, Y))

    return train_tasks, test_tasks


def split_task(X, Y, n_support=3, n_query=5, seed=None):
    """Split a task into support (labeled), unlabeled, and query sets.

    Args:
        n_support: number of labeled support samples
        n_query: number of labeled query samples
        remaining: unlabeled samples for semi-supervised loss

    Returns:
        X_sup, Y_sup, X_unl, X_qry, Y_qry
    """
    n = X.shape[0]
    if seed is not None:
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    else:
        idx = torch.randperm(n)

    sup_idx = idx[:n_support]
    qry_idx = idx[n_support:n_support + n_query]
    unl_idx = idx[n_support + n_query:]

    return (X[sup_idx], Y[sup_idx],
            X[unl_idx],
            X[qry_idx], Y[qry_idx])


# ============================================================
# 2. Model
# ============================================================

class MLPRegressor(nn.Module):
    """Simple MLP for water quality score regression."""
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
    """Forward pass using explicit parameter list (for MAML inner loop).

    This preserves the computation graph for second-order meta-gradients.
    """
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
# 3. FixMatch (Mixup Version) Semi-Supervised Loss
# ============================================================

def fixmatch_mixup_loss(model, X_unl, params,
                         weak_noise_std=0.01,
                         mixup_alpha=0.75,
                         confidence_threshold=0.7,
                         n_weak_preds=5):
    """FixMatch loss with Mixup strong augmentation, adapted for regression.

    Pipeline:
      1. Weak augmentation (small Gaussian noise) x n_weak_preds times
         -> average predictions = pseudo-labels
      2. Confidence estimation: variance across weak predictions
         (low variance = high confidence) -> binary mask
      3. Strong augmentation: Mixup (linear interpolation between pairs
         of real unlabeled samples) -> interpolated pseudo-labels
      4. Loss: MSE between strong-augmented predictions and interpolated
         pseudo-labels, masked by confidence

    Args:
        model: MLPRegressor
        X_unl: (N, 11) unlabeled features
        params: current inner-loop parameters
        weak_noise_std: std of Gaussian noise for weak augmentation
        mixup_alpha: Beta distribution parameter for Mixup
        confidence_threshold: minimum confidence (0-1) to keep a sample
        n_weak_preds: number of weak predictions for confidence estimation

    Returns:
        loss: scalar tensor
        n_confident: number of samples passing confidence filter
    """
    n_unl = X_unl.shape[0]
    if n_unl < 2:
        return torch.tensor(0.0, device=X_unl.device), 0

    # ---- Step 1 & 2: Weak augmentation -> pseudo-labels + confidence ----
    model.eval()
    with torch.no_grad():
        preds_list = []
        for _ in range(n_weak_preds):
            X_weak = X_unl + torch.randn_like(X_unl) * weak_noise_std
            preds_list.append(functional_forward(model, X_weak, params))

        # Average prediction = pseudo-label
        pseudo_y = torch.stack(preds_list).mean(dim=0)  # (N, 1)

        # Variance = uncertainty (low variance = high confidence)
        pred_var = torch.stack(preds_list).var(dim=0).squeeze(1)  # (N,)
        confidence = 1.0 - (pred_var / (pred_var.max() + 1e-8))
        mask = (confidence >= confidence_threshold).float().unsqueeze(1)  # (N, 1)

    n_confident = int(mask.sum().item())
    if n_confident < 2:
        return torch.tensor(0.0, device=X_unl.device), n_confident

    # ---- Step 3: Strong augmentation via Mixup ----
    model.train()
    perm = torch.randperm(n_unl)
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    lam = max(lam, 1 - lam)  # skew toward one sample for stability

    X_mix = lam * X_unl + (1 - lam) * X_unl[perm]
    y_mix = lam * pseudo_y + (1 - lam) * pseudo_y[perm]
    # Confidence mask for mixed samples = min of both
    mask_mix = torch.minimum(mask, mask[perm])

    # ---- Step 4: Consistency loss ----
    pred_mix = functional_forward(model, X_mix, params)
    loss = (F.mse_loss(pred_mix, y_mix.detach(), reduction='none') * mask_mix).mean()

    return loss, n_confident


# ============================================================
# 4. MAML Training with FixMatch Inner Loop
# ============================================================

def maml_train_fixmatch(model, train_tasks,
                         meta_lr=0.001, inner_lr=0.01, inner_steps=5,
                         n_support=3, n_query=5, unsup_weight=1.0,
                         meta_iterations=1000, task_batch_size=4,
                         eval_interval=250, device='cpu', seed=42,
                         **fixmatch_kwargs):
    """MAML meta-training with FixMatch semi-supervised inner loop.

    Inner loop: total_loss = supervised MSE + unsup_weight * FixMatch loss
    Outer loop: meta-gradient from query set supervised loss
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    model.to(device)

    history = {
        'meta_loss': [], 'meta_mae': [],
        'inner_loss_avg': [], 'fixmatch_loss': [], 'n_confident': [],
    }
    n_tasks = len(train_tasks)

    for it in range(meta_iterations):
        meta_optimizer.zero_grad()
        task_indices = np.random.choice(n_tasks, size=task_batch_size, replace=False)

        batch_qry_loss = 0.0
        batch_qry_mae = 0.0
        batch_inner = 0.0
        batch_fm_loss = 0.0
        batch_confident = 0.0

        for ti in task_indices:
            X, Y = train_tasks[ti]
            X_sup, Y_sup, X_unl, X_qry, Y_qry = split_task(X, Y, n_support, n_query)
            X_sup, Y_sup = X_sup.to(device), Y_sup.to(device)
            X_unl = X_unl.to(device)
            X_qry, Y_qry = X_qry.to(device), Y_qry.to(device)

            # Inner loop: functional parameter updates (preserve graph for 2nd order)
            params = [p.clone() for p in model.parameters()]
            task_inner_losses = []

            for step in range(inner_steps):
                pred_sup = functional_forward(model, X_sup, params)
                loss_sup = F.mse_loss(pred_sup, Y_sup)

                loss_fm = torch.tensor(0.0, device=device)
                n_conf = 0
                if X_unl.shape[0] > 0:
                    loss_fm, n_conf = fixmatch_mixup_loss(
                        model, X_unl, params, **fixmatch_kwargs)

                total_loss = loss_sup + unsup_weight * loss_fm
                task_inner_losses.append(total_loss.item())

                grads = torch.autograd.grad(
                    total_loss, params, create_graph=True, allow_unused=True)
                params = [p - inner_lr * (g if g is not None else 0)
                          for p, g in zip(params, grads)]

            # Query set evaluation (outer loop meta-gradient)
            pred_qry = functional_forward(model, X_qry, params)
            qry_loss = F.mse_loss(pred_qry, Y_qry)
            qry_mae = F.l1_loss(pred_qry, Y_qry).item()

            batch_qry_loss += qry_loss
            batch_qry_mae += qry_mae
            batch_inner += np.mean(task_inner_losses)
            batch_fm_loss += loss_fm.item() if isinstance(loss_fm, torch.Tensor) else loss_fm
            batch_confident += n_conf

        # Meta-update
        meta_loss = batch_qry_loss / task_batch_size
        meta_loss.backward()
        meta_optimizer.step()

        history['meta_loss'].append(meta_loss.item())
        history['meta_mae'].append(batch_qry_mae / task_batch_size)
        history['inner_loss_avg'].append(batch_inner / task_batch_size)
        history['fixmatch_loss'].append(batch_fm_loss / task_batch_size)
        history['n_confident'].append(batch_confident / task_batch_size)

        if (it + 1) % eval_interval == 0:
            print(f"  [FixMatch-Mixup] Iter {it+1}/{meta_iterations} | "
                  f"meta_loss={meta_loss.item():.6f} | meta_mae={batch_qry_mae/task_batch_size:.6f} | "
                  f"fm_loss={batch_fm_loss/task_batch_size:.6f} | "
                  f"confident={batch_confident/task_batch_size:.1f}/{X_unl.shape[0]}")

    return history


# ============================================================
# 5. Evaluation
# ============================================================

def evaluate_fixmatch(model, test_tasks,
                      n_support=3, n_query=5, inner_lr=0.01, inner_steps=5,
                      unsup_weight=1.0, n_eval_tasks=50, device='cpu', seed=123,
                      **fixmatch_kwargs):
    """Evaluate FixMatch-MAML on test tasks.

    For each test task: adapt on support+unlabeled via FixMatch inner loop,
    then evaluate on query set.
    """
    torch.manual_seed(seed)
    model.eval()

    task_indices = np.random.choice(len(test_tasks),
                                     size=min(n_eval_tasks, len(test_tasks)),
                                     replace=False)

    all_losses, all_maes, all_r2 = [], [], []

    for ti in task_indices:
        X, Y = test_tasks[ti]
        X_sup, Y_sup, X_unl, X_qry, Y_qry = split_task(
            X, Y, n_support, n_query, seed=int(ti))
        X_sup, Y_sup = X_sup.to(device), Y_sup.to(device)
        X_unl = X_unl.to(device)
        X_qry, Y_qry = X_qry.to(device), Y_qry.to(device)

        params = [p.clone() for p in model.parameters()]
        for step in range(inner_steps):
            pred_sup = functional_forward(model, X_sup, params)
            loss_sup = F.mse_loss(pred_sup, Y_sup)
            loss_fm = torch.tensor(0.0, device=device)
            if X_unl.shape[0] > 0:
                loss_fm, _ = fixmatch_mixup_loss(model, X_unl, params, **fixmatch_kwargs)
            total_loss = loss_sup + unsup_weight * loss_fm
            grads = torch.autograd.grad(total_loss, params, allow_unused=True)
            params = [p - inner_lr * (g if g is not None else 0)
                      for p, g in zip(params, grads)]

        with torch.no_grad():
            pred_qry = functional_forward(model, X_qry, params)
            loss = F.mse_loss(pred_qry, Y_qry).item()
            mae = F.l1_loss(pred_qry, Y_qry).item()
            ss_res = ((Y_qry - pred_qry) ** 2).sum().item()
            ss_tot = ((Y_qry - Y_qry.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / (ss_tot + 1e-8)

        all_losses.append(loss)
        all_maes.append(mae)
        all_r2.append(r2)

    return {
        'mse_mean': np.mean(all_losses), 'mse_std': np.std(all_losses),
        'mae_mean': np.mean(all_maes), 'mae_std': np.std(all_maes),
        'r2_mean': np.mean(all_r2), 'r2_std': np.std(all_r2),
    }


# ============================================================
# 6. Main
# ============================================================

def main():
    print("=" * 60)
    print("  FixMatch (Mixup) + MAML for Water Quality Prediction")
    print("  Low-label setting: n_support=3 (~8% labeled)")
    print("=" * 60)

    device = 'cpu'

    # ---- Load data ----
    print("\n  Loading water quality dataset...")
    train_tasks, test_tasks = load_water_data()
    print(f"  Train: {len(train_tasks)} tasks, Test: {len(test_tasks)} tasks")
    print(f"  Per task: {train_tasks[0][0].shape[0]} samples x "
          f"{train_tasks[0][0].shape[1]} features")

    # ---- Hyperparameters ----
    n_support = 3        # labeled support samples per task
    n_query = 5          # query samples per task
    meta_iterations = 1000
    fixmatch_kwargs = dict(
        weak_noise_std=0.01,      # weak augmentation noise level
        mixup_alpha=0.75,          # Mixup Beta distribution parameter
        confidence_threshold=0.7,   # keep top ~30% most confident samples
        n_weak_preds=5,             # predictions for confidence estimation
    )

    # ---- Train ----
    print(f"\n  Training FixMatch-MAML (n_support={n_support})...")
    model = MLPRegressor(input_dim=11, hidden_dims=[64, 32], output_dim=1)
    start = time.time()
    history = maml_train_fixmatch(
        model, train_tasks,
        meta_lr=0.001, inner_lr=0.01, inner_steps=5,
        n_support=n_support, n_query=n_query, unsup_weight=1.0,
        meta_iterations=meta_iterations, task_batch_size=4,
        eval_interval=250, device=device, seed=42,
        **fixmatch_kwargs)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.1f}s")

    # ---- Evaluate ----
    print("\n  Evaluating on test tasks...")
    results = evaluate_fixmatch(
        model, test_tasks,
        n_support=n_support, n_query=n_query,
        inner_lr=0.01, inner_steps=5, unsup_weight=1.0,
        n_eval_tasks=50, device=device, seed=123,
        **fixmatch_kwargs)
    results['train_time'] = train_time
    results['method'] = 'fixmatch_mixup'

    print(f"\n  {'='*50}")
    print(f"  RESULTS (n_support={n_support}, 50 test tasks)")
    print(f"  {'='*50}")
    print(f"  MSE:  {results['mse_mean']:.6f} +/- {results['mse_std']:.6f}")
    print(f"  MAE:  {results['mae_mean']:.6f} +/- {results['mae_std']:.6f}")
    print(f"  R²:   {results['r2_mean']:.4f} +/- {results['r2_std']:.4f}")
    print(f"  Time: {train_time:.1f}s")

    # ---- Save ----
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/model_fixmatch_mixup.pt')
    with open('results/results_fixmatch_mixup.json', 'w') as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    window = 20
    smoothed = np.convolve(history['meta_loss'], np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='#FF9800', linewidth=1.5)
    ax.set_xlabel('Meta-Iteration')
    ax.set_ylabel('Meta Loss (MSE)')
    ax.set_title('FixMatch-Mixup Training Meta-Loss')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    fm_smoothed = np.convolve(history['fixmatch_loss'], np.ones(window)/window, mode='valid')
    ax.plot(fm_smoothed, color='#E91E63', linewidth=1.5, label='FixMatch loss')
    ax2 = ax.twinx()
    conf_smoothed = np.convolve(history['n_confident'], np.ones(window)/window, mode='valid')
    ax2.plot(conf_smoothed, color='#2196F3', linewidth=1.5, alpha=0.7, label='N confident')
    ax.set_xlabel('Meta-Iteration')
    ax.set_ylabel('FixMatch Loss', color='#E91E63')
    ax2.set_ylabel('N Confident Samples', color='#2196F3')
    ax.set_title('FixMatch Component Loss & Confidence Filter')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/fixmatch_mixup_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Training plot saved to results/fixmatch_mixup_training.png")
    print("\n  Done!")


if __name__ == '__main__':
    main()
