# Meta-SGD 中 meta_lr 正数约束说明

## 目标

本次修改是给 Meta-SGD 中的可学习内循环学习率 `meta_lr` 增加正数约束，使其在每次外循环优化后被裁剪到：

```text
[min_meta_lr, clamp_meta_lr]
```

默认设置为：

```text
min_meta_lr = 1e-6
clamp_meta_lr = 0.2
```

对应命令行参数：

```bash
--min-meta-lr 1e-6
--clamp-meta-lr 0.2
```

## 原理

Meta-SGD 的任务内更新形式为：

```text
θ' = θ - α ⊙ ∇θ L_inner
```

其中：

- `θ` 是模型参数。
- `θ'` 是经过 inner loop 后的任务适应参数。
- `α` 是 Meta-SGD 学到的逐参数内循环学习率，也就是代码中的 `meta_lr`。
- `L_inner` 是任务内损失，可以是纯监督损失，也可以是半监督损失。

如果 `α` 为正，更新方向是标准梯度下降：

```text
θ' = θ - 正数 * 梯度
```

如果 `α` 变成负数，更新方向会反过来：

```text
θ' = θ - 负数 * 梯度 = θ + 正数 * 梯度
```

这相当于在某些参数上做梯度上升，可能导致任务内适应方向不稳定。虽然原始 Meta-SGD 理论上允许学习更自由的更新方向，但在当前项目里，我们更关注稳定的小样本任务适应，因此把 `meta_lr` 限制为正数更符合实验目标。

## 修改内容

原代码中 `meta_lr` 的约束是：

```text
[-clamp_meta_lr, clamp_meta_lr]
```

也就是允许负数。

现在改为：

```text
[min_meta_lr, clamp_meta_lr]
```

核心逻辑为：

```python
lower = max(min_value, 1e-12)
upper = max(max_value, lower)
meta_lr.clamp_(min=lower, max=upper)
```

这样可以保证：

- `meta_lr` 不会小于一个很小的正数。
- `meta_lr` 不会无限增大。
- 每个参数仍然可以拥有不同的学习率。
- Meta-SGD 保留“学习更新步长”的能力，但避免负学习率带来的方向翻转。

## 输入

主训练脚本运行示例：

```bash
python maml_meta_sgd_framework.py --algorithm meta-sgd --inner-update semi-supervised --inner-steps 3 --meta-batch 8 --steps 300 --min-meta-lr 1e-6 --clamp-meta-lr 0.2 --log-every 50
```

步数消融实验：

```bash
python inner_steps_ablation.py --algorithm meta-sgd --inner-update semi-supervised --inner-steps-list 1,3,5 --meta-batch 8 --steps 300 --min-meta-lr 1e-6 --clamp-meta-lr 0.2 --output inner_steps_results.csv
```

meta batch 消融实验：

```bash
python meta_batch_ablation.py --algorithm meta-sgd --inner-update semi-supervised --meta-batch-list 2,8,16 --inner-steps 3 --steps 300 --min-meta-lr 1e-6 --clamp-meta-lr 0.2 --output meta_batch_results.csv
```

主要相关参数：

- `--min-meta-lr`：`meta_lr` 的最小值，默认 `1e-6`。
- `--clamp-meta-lr`：`meta_lr` 的最大值，默认 `0.2`。
- `--inner-lr`：`meta_lr` 的初始化值，默认 `0.01`。

## 输出

训练日志和 CSV 中会继续输出：

- `meta_lr_mean`
- `meta_lr_min`
- `meta_lr_max`

本次新增：

- `meta_lr_nonpositive_count`

如果正数约束生效，正常情况下：

```text
meta_lr_min > 0
meta_lr_nonpositive_count = 0
```

如果发现 `meta_lr_min` 长期贴近 `min_meta_lr`，说明部分参数的内学习率被压到下界，可能需要降低 `outer_lr` 或调小半监督损失权重。

如果发现 `meta_lr_max` 长期贴近 `clamp_meta_lr`，说明部分参数的更新步长被压到上界，可能需要适当增大上界，或者检查任务损失是否过大。

## 算法提升逻辑

加入正数约束主要提升训练稳定性，而不是直接增加模型容量。

原来的自由 `meta_lr` 可能出现负值，导致部分参数在 inner loop 中朝梯度上升方向更新。对于小样本任务，support set 本来就少，如果内循环方向过于自由，容易造成任务适应不稳定，进而影响 query loss 和 meta-gradient。

正数约束后的逻辑是：

```text
保留 Meta-SGD 的逐参数学习率能力
+ 限制更新方向仍为梯度下降
+ 防止学习率过小或过大
= 提高小样本任务适应过程的稳定性
```

因此，这个修改适合放在项目计划书或实验报告中的“训练稳定性优化”部分。

## 数据来源与真实数据接入

本次修改不改变数据来源。当前代码仍使用合成 sine wave few-shot 任务：

```text
y = A sin(x + φ)
```

该数据用于模拟不同任务之间的差异，验证 MAML / Meta-SGD / 半监督 inner loop 是否能正常运行。

真实数据接入方式也不改变，仍然是替换 `FewShotTaskSampler`，让它返回统一的 `TaskBatch`：

```text
support_x, support_y
query_x, query_y
ssl_x, ssl_y, ssl_confidence
```

真实水文或生态数据建议整理为：

```text
task_id, split, feature_1, feature_2, ..., target, label_available
```

其中 `task_id` 对应流域、站点、区域或时间窗口，`split` 对应 `support`、`query`、`unlabeled`。

## 建议报告写法

可以写为：

```text
为提高 Meta-SGD 训练稳定性，本项目对可学习内循环学习率 meta_lr 增加正数约束，将其裁剪到 [min_meta_lr, clamp_meta_lr] 范围内。该设计保留了 Meta-SGD 按参数学习更新步长的能力，同时避免负学习率导致 inner loop 更新方向反转，从而降低小样本任务适应过程中的不稳定风险。
```
