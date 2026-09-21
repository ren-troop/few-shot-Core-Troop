# Meta Batch 提升实验说明

## 目标

本次实现的是 `meta_batch` 提升实验，对比：

```text
meta_batch = 2, 8, 16
```

`meta_batch` 表示一次 outer loop 更新中同时采样多少个任务，并把这些任务的 query loss 求平均后更新元模型参数。它不是单个任务里的样本数量，而是“每次元更新包含的任务数量”。

新增脚本：

```text
meta_batch_ablation.py
```

该脚本会在其他参数保持一致的情况下，依次运行 `meta_batch=2`、`meta_batch=8`、`meta_batch=16`，并输出 CSV 结果，方便比较训练稳定性、验证误差和计算成本。

## 原理

在 MAML / Meta-SGD 中，每次训练不是只看一个任务，而是从任务分布中采样多个任务：

```text
Task 1, Task 2, ..., Task B
```

其中 `B` 就是 `meta_batch`。

每个任务先用自己的 support set 做 inner loop 适应，再用 query set 计算任务损失：

```text
L_query_1, L_query_2, ..., L_query_B
```

然后外循环使用这些任务损失的平均值：

```text
L_meta = (L_query_1 + L_query_2 + ... + L_query_B) / B
```

最后根据 `L_meta` 更新元模型参数。

因此，`meta_batch` 越大，每次更新看到的任务越多，更新方向越能代表整体任务分布，随机波动通常更小。

## 算法提升逻辑

当 `meta_batch=2` 时，每次外循环只平均 2 个任务。如果这 2 个任务刚好振幅、相位或噪声差异很大，梯度方向可能波动明显，表现为：

- `meta_loss` 起伏大。
- `val_query_loss` 不稳定。
- `grad_norm` 可能忽高忽低。

当 `meta_batch` 提升到 8 或 16 时，每次更新会平均更多任务的 query loss，单个异常任务对更新方向的影响被削弱，因此训练曲线通常更平滑，泛化评估更稳定。

但 `meta_batch` 不是越大越好。提升它会带来三类代价：

- 每个 outer step 需要处理更多任务，单步训练时间增加。
- 二阶 MAML / Meta-SGD 需要保存更多计算图，显存占用增加。
- 如果计算资源有限，过大的 `meta_batch` 可能导致训练速度变慢。

因此本实验对比 `2 -> 8 -> 16`，核心是判断：

```text
更大的 meta_batch 是否明显降低验证误差和训练波动；
如果提升有限，是否值得承担额外计算成本。
```

一般建议：

- `meta_batch=2`：适合快速调试，速度快但波动大。
- `meta_batch=8`：适合正式实验的默认设置，稳定性和成本较平衡。
- `meta_batch=16`：适合机器性能允许时验证更大任务平均是否进一步改善稳定性。

## 输入

快速测试命令：

```bash
python meta_batch_ablation.py --steps 2 --inner-steps 1 --val-tasks 2 --log-every 1
```

正式实验命令：

```bash
python meta_batch_ablation.py --algorithm meta-sgd --inner-update semi-supervised --meta-batch-list 2,8,16 --inner-steps 3 --steps 300 --shots 5 --queries 10 --unlabeled-per-task 10 --ssl-weight 0.2 --val-tasks 20 --log-every 50 --output meta_batch_results.csv
```

如果希望三组实验采样的训练任务总数接近一致，可以使用：

```bash
python meta_batch_ablation.py --algorithm meta-sgd --inner-update semi-supervised --meta-batch-list 2,8,16 --inner-steps 3 --total-task-budget 2400 --val-tasks 20 --log-every 50 --output meta_batch_results_fair_tasks.csv
```

此时：

- `meta_batch=2` 大约训练 1200 个 outer step。
- `meta_batch=8` 大约训练 300 个 outer step。
- `meta_batch=16` 大约训练 150 个 outer step。

这样比较的是“相近任务总量下”的表现；如果不用 `--total-task-budget`，则比较的是“相同 outer step 数下”的表现。

主要输入参数：

- `--meta-batch-list`：要比较的 meta batch，默认 `2,8,16`。
- `--inner-steps`：每个任务 inner loop 更新步数，默认 `3`。
- `--steps`：每个 meta batch 设置下的 outer loop 步数。
- `--total-task-budget`：可选。设置后按总任务数自动调整每组 steps。
- `--algorithm`：选择 `maml` 或 `meta-sgd`。
- `--inner-update`：选择 `supervised` 或 `semi-supervised`。
- `--shots`：每个任务 support set 的有标签样本数。
- `--queries`：每个任务 query set 的样本数。
- `--unlabeled-per-task`：每个任务的半监督样本数。
- `--ssl-weight`：半监督损失权重。
- `--first-order`：启用一阶近似，降低显存和运行成本。
- `--output`：CSV 保存路径。

## 输出

脚本输出两类结果。

第一类是终端日志：

```text
[meta_batch=8] algorithm=meta-sgd, device=cpu, inner_update=semi-supervised, inner_steps=3
step=0050 inner_loss=... support_loss=... ssl_loss=... meta_loss=... grad_norm=... val_query_loss=... best_val_query_loss=... elapsed_sec=...
```

第二类是 CSV 文件，默认：

```text
meta_batch_results.csv
```

CSV 中主要字段：

- `meta_batch`
- `configured_steps`
- `sampled_train_tasks`
- `inner_steps`
- `support_loss`
- `ssl_loss`
- `meta_loss`
- `val_query_loss`
- `best_val_query_loss`
- `grad_norm`
- `avg_step_time_sec`
- `meta_lr_mean`
- `meta_lr_min`
- `meta_lr_max`
- `cuda_peak_memory_mb`

分析时优先看：

- `best_val_query_loss`：越低说明验证任务上适应效果越好。
- `val_query_loss`：最终验证误差。
- `grad_norm`：过大或剧烈波动说明训练可能不稳定。
- `avg_step_time_sec`：单步耗时，用于衡量计算成本。

## 当前数据来源

当前实验使用合成 sine wave few-shot 任务：

```text
y = A sin(x + φ)
```

每个任务随机采样不同的振幅 `A` 和相位 `φ`，用于模拟不同区域、站点或生态过程之间的差异。

数据划分方式：

- `support set`：少量有标签样本，用于 inner loop 任务适应。
- `query set`：用于 outer loop 评估适应后的效果。
- `unlabeled set`：用于半监督损失，伪标签带有噪声和置信度。

合成数据的作用是先验证算法逻辑和训练框架是否正确，再迁移到公开数据或真实采集数据。

## 如何接入真实数据

真实数据接入时，主要替换 `maml_meta_sgd_framework.py` 中的 `FewShotTaskSampler`。

建议将真实数据整理为统一表格：

```text
task_id, split, feature_1, feature_2, ..., feature_n, target, label_available
```

字段说明：

- `task_id`：任务编号，可以是流域、站点、区域、时间窗口、生态类型或事件类型。
- `split`：样本划分，取值为 `support`、`query`、`unlabeled`。
- `feature_1 ... feature_n`：输入特征，例如降雨、温度、蒸散发、遥感指数、土壤湿度、历史径流、历史水位等。
- `target`：预测目标，例如径流量、水位、水质指标或生态风险值。
- `label_available`：是否有真实标签。

接入步骤：

1. 按流域、站点、区域或时间窗口划分 `task_id`。
2. 每个任务内部划分 support set、query set 和 unlabeled set。
3. 将 support set 转为 `support_x/support_y`。
4. 将 query set 转为 `query_x/query_y`。
5. 将未标注样本转为 `ssl_x`。
6. 使用已有模型、历史模型或伪标签模块生成 `ssl_y`。
7. 根据伪标签可靠程度生成 `ssl_confidence`。
8. 返回统一 `TaskBatch`，保持训练循环不变。

如果真实数据是多维特征，需要把模型第一层从：

```python
nn.Linear(1, hidden_dim)
```

改为：

```python
nn.Linear(input_dim, hidden_dim)
```

其中 `input_dim` 是真实输入特征数量。

## 建议结果表述

如果 `meta_batch=8` 明显优于 `2`，可以写：

```text
实验结果表明，将 meta batch 从 2 提升至 8 后，每次外循环更新能够平均更多任务的 query loss，减弱单个随机任务对梯度方向的影响，使训练过程更加稳定，并在验证任务上取得更低的 query loss。
```

如果 `meta_batch=16` 相比 `8` 提升有限，可以写：

```text
继续将 meta batch 提升至 16 后，验证误差改善有限，但单步计算时间进一步增加。综合预测性能与计算成本，后续实验优先采用 meta_batch=8 作为默认设置，并保留 2 和 16 作为消融对照。
```
