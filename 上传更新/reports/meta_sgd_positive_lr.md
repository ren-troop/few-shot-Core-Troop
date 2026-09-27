# Meta-SGD 正学习率约束

## 作用

Meta-SGD 不使用单一固定内循环学习率，而是为模型的每个参数学习更新率：

```text
adapted_parameter = parameter - meta_lr * gradient
```

若允许 `meta_lr` 为负，某些参数会沿普通梯度的反方向更新。这在原始 Meta-SGD 解释中可以表示可学习方向，但在本项目的工程设定中会降低学习率含义的可解释性，也可能使少样本和伪标签条件下的适应不稳定。因此实验将每个学习率限制为严格正数并设置上界。

## 实现顺序

每个外循环依次执行：

1. 根据多个任务的 Query 损失反向传播；
2. `optimizer.step()` 同时更新模型参数和 Meta-SGD 学习率；
3. `clamp_meta_lrs(min_value, max_value)` 将学习率限制到合法区间；
4. `extra_metrics()` 读取约束后的统计量并写入日志和 CSV。

下界按 `max(min_value, 1e-12)` 处理，上界按 `max(max_value, lower)` 处理，因此即使用户传入不合理的非正上界，也不会形成空区间。

## 可观测字段

- `meta_lr_mean`：全部逐参数学习率的均值；
- `meta_lr_min`：最小学习率，应不小于配置下界；
- `meta_lr_max`：最大学习率，应不大于配置上界；
- `meta_lr_nonpositive_count`：非正学习率数量，应始终为 0。

约束并不保证模型一定收敛。仍需联合观察验证误差、梯度范数、达到上界或下界的参数比例，以及不同随机种子的稳定性。若大量学习率长期贴住边界，应重新检查上下界、外循环学习率和伪标签权重。
