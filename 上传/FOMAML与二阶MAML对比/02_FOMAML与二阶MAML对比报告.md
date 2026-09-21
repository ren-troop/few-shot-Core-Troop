# FOMAML 对比实验说明

## 目标

本次实验用于对比标准二阶 MAML 和一阶近似 FOMAML 的差异：

```text
标准二阶 MAML：create_graph=True
FOMAML：create_graph=False
```

新增脚本：

```text
fomaml_comparison.py
```

该脚本会在相同数据、相同模型、相同训练参数下分别运行二阶模式和 FOMAML 模式，并记录：

- 显存或进程内存差异。
- 训练速度差异。
- 验证精度差异。

## 原理

MAML 的核心流程是：

```text
先在 support set 上做 inner loop 任务适应
再在 query set 上计算 meta loss
最后用 meta loss 更新初始参数
```

标准二阶 MAML 在 inner loop 更新时保留完整计算图：

```python
create_graph=True
```

这样 outer loop 反向传播时，可以计算“query loss 对初始参数的二阶影响”。它理论上更精确，但显存占用更高、速度更慢。

FOMAML 使用一阶近似：

```python
create_graph=False
```

它不保留 inner loop 的二阶梯度图，只近似使用适应后参数带来的梯度方向。这样可以明显减少计算图规模，通常速度更快、显存更低，但可能带来轻微精度损失。

## 算法提升逻辑

这个实验的目的不是让 FOMAML 一定超过二阶 MAML，而是判断：

```text
是否可以用很小的精度损失换取更快速度和更低显存。
```

如果实验结果显示 FOMAML 的 `val_query_loss` 接近二阶 MAML，但 `avg_step_time_sec` 更低、内存占用更少，那么后续正式实验可以优先采用 FOMAML 作为工程版本。

如果 FOMAML 的验证误差明显变差，说明当前任务仍然依赖二阶梯度信息，应继续保留标准二阶 MAML 作为主实验。

## 输入

快速测试命令：

```bash
python fomaml_comparison.py --steps 2 --meta-batch 2 --inner-steps 1 --val-tasks 2 --log-every 1
```

正式实验命令：

```bash
python fomaml_comparison.py --algorithm maml --inner-update semi-supervised --inner-steps 3 --meta-batch 8 --steps 300 --shots 5 --queries 10 --unlabeled-per-task 10 --ssl-weight 0.2 --val-tasks 20 --log-every 50 --output fomaml_comparison_results.csv
```

如果要在 Meta-SGD 上也比较一阶与二阶模式，可以运行：

```bash
python fomaml_comparison.py --algorithm meta-sgd --inner-update semi-supervised --inner-steps 3 --meta-batch 8 --steps 300 --min-meta-lr 1e-6 --clamp-meta-lr 0.2 --output first_order_meta_sgd_results.csv
```

主要参数：

- `--algorithm`：默认 `maml`，也可以设置为 `meta-sgd`。
- `--modes`：默认 `second-order,fomaml`。
- `--inner-update`：选择 `supervised` 或 `semi-supervised`。
- `--inner-steps`：inner loop 更新步数。
- `--meta-batch`：每次 outer loop 采样的任务数量。
- `--steps`：训练步数。
- `--val-tasks`：验证任务数量。
- `--output`：CSV 输出路径。

## 输出

脚本会输出终端日志，例如：

```text
[mode=second-order] algorithm=maml, create_graph=True
step=0050 meta_loss=... val_query_loss=... val_rmse=... elapsed_sec=... process_memory_mb=...

[mode=fomaml] algorithm=maml, create_graph=False
step=0050 meta_loss=... val_query_loss=... val_rmse=... elapsed_sec=... process_memory_mb=...
```

同时保存 CSV 文件，默认：

```text
fomaml_comparison_results.csv
```

CSV 中主要字段：

- `mode`
- `create_graph`
- `first_order`
- `meta_loss`
- `val_query_loss`
- `val_rmse`
- `best_val_query_loss`
- `grad_norm`
- `elapsed_sec`
- `avg_step_time_sec`
- `start_process_memory_mb`
- `end_process_memory_mb`
- `peak_observed_process_memory_mb`
- `cuda_peak_memory_mb`

其中：

- `val_query_loss` 是验证任务 query set 上的 MSE，越低越好。
- `val_rmse` 是验证误差的平方根，更直观。
- `avg_step_time_sec` 表示平均每个训练 step 的耗时。
- `cuda_peak_memory_mb` 表示 GPU 峰值显存，仅在 CUDA 环境下有效。
- `peak_observed_process_memory_mb` 表示 CPU 进程内存观测值，适合无 GPU 环境下参考。

## 精度、速度、显存如何比较

建议重点比较三类指标：

```text
精度：val_query_loss / val_rmse / best_val_query_loss
速度：elapsed_sec / avg_step_time_sec
显存：cuda_peak_memory_mb
CPU 内存：peak_observed_process_memory_mb
```

判断逻辑：

- 如果 FOMAML 的误差接近二阶 MAML，但速度更快、显存更低，说明一阶近似值得采用。
- 如果 FOMAML 的误差明显更高，说明当前任务需要二阶信息。
- 如果二者误差差异很小，应优先选择计算成本更低的 FOMAML。

## 当前数据来源

当前实验使用合成 sine wave few-shot 任务：

```text
y = A sin(x + φ)
```

每个任务随机采样不同振幅 `A` 和相位 `φ`，用来模拟不同区域、不同站点或不同生态过程之间的任务差异。

当前任务划分为：

- `support set`：少量有标签样本，用于 inner loop。
- `query set`：用于 outer loop 和验证。
- `unlabeled set`：用于半监督损失，包含带噪声的伪标签和置信度。

## 如何接入真实数据

真实数据接入方式与前面实验一致，不需要改 FOMAML 对比脚本，只需要替换主框架中的任务采样器。

建议整理为：

```text
task_id, split, feature_1, feature_2, ..., feature_n, target, label_available
```

字段说明：

- `task_id`：一个任务，可以对应流域、站点、区域、时间窗口或事件类型。
- `split`：样本类型，取值为 `support`、`query`、`unlabeled`。
- `feature_1 ... feature_n`：输入特征，例如降雨、温度、蒸散发、遥感指数、土壤湿度、历史径流、历史水位等。
- `target`：预测目标，例如径流量、水位、水质指标或生态风险值。
- `label_available`：是否有真实标签。

接入后仍然返回统一的 `TaskBatch`：

```text
support_x, support_y
query_x, query_y
ssl_x, ssl_y, ssl_confidence
```

如果真实数据是多维特征，需要把模型第一层从：

```python
nn.Linear(1, hidden_dim)
```

改为：

```python
nn.Linear(input_dim, hidden_dim)
```

## 建议报告写法

可以写为：

```text
为降低二阶 MAML 在多步 inner loop 下的显存占用和运行时间，本项目增加 FOMAML 对比实验，将 inner loop 中的 create_graph 设置为 False，构造一阶近似版本。实验记录标准二阶 MAML 与 FOMAML 在验证误差、训练耗时和显存占用方面的差异，用于判断一阶近似是否能在精度损失较小的情况下提升训练效率。
```
