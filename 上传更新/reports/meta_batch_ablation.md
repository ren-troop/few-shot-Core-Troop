# MetaBatch 2/8/16 消融实验

## 实验目的

MetaBatch 表示一次外循环更新前共同计算 Query 损失的任务数量。较大的 MetaBatch 可以降低单次梯度估计的随机性，但会增加每步计算量。实验同时采用两种控制方式，避免把“训练任务更多”误认为“批大小更优”。

固定设置为 Meta-SGD、半监督内循环、3 个 inner steps、5 个 support、10 个 query、10 个 unlabeled、随机种子 28。

## 相同外循环步数

每种 batch 均运行 300 step：

| MetaBatch | 训练任务数 | 最终验证损失 | 最佳验证损失 | 平均每步时间/s |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 600 | 1.095396 | 1.095396 | 0.0203 |
| 8 | 2400 | 0.302060 | 0.302060 | 0.0694 |
| 16 | 4800 | 0.283480 | 0.283480 | 0.1407 |

该口径下 batch=8 和 16 的验证损失较低，但它们分别处理了 batch=2 的 4 倍和 8 倍任务，因此结果同时包含训练数据量增加的影响。每步耗时随 batch 增大，batch=16 相对 batch=8 的精度收益较小。

结果文件：`results/meta_batch_formal.csv`。

## 相同任务预算

把总训练任务数统一为 2400，外循环步数相应设为 1200、300、150：

| MetaBatch | 外循环步数 | 训练任务数 | 最终验证损失 | 最佳验证损失 | 平均每步时间/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1200 | 2400 | 0.403617 | 0.264293 | 0.0078 |
| 8 | 300 | 2400 | 0.490651 | 0.490651 | 0.0654 |
| 16 | 150 | 2400 | 0.816178 | 0.816178 | 0.1219 |

在当前单随机种子下，小 batch 通过更多外循环更新获得了更好的最佳验证损失。由此不能得出 batch=8/16 在相同训练预算下优于 batch=2 的结论。

结果文件：`results/meta_batch_equal_budget.csv`。

## 学习率约束检查

两组实验的所有配置均满足：

- `meta_lr_min >= 1e-6`；
- `meta_lr_max <= 0.2`；
- `meta_lr_nonpositive_count = 0`。

这些字段在 `optimizer.step()` 后、结果记录前计算，代表实际进入下一次训练的 Meta-SGD 学习率状态。

## 结论

MetaBatch 不能脱离训练任务数和外循环更新次数单独比较。当前数据支持的稳妥表述是：batch=8 可作为计算成本与单步任务多样性之间的工程候选，batch=16 的额外成本较高；最终选型需要对两种预算口径分别进行多随机种子重复实验，并报告均值和标准差。

## 复现命令

```powershell
python -m experiments.meta_batch_ablation --steps 300 --meta-batch-list 2,8,16 --inner-steps 3 --device cpu --output results/meta_batch_formal.csv
python -m experiments.meta_batch_ablation --total-task-budget 2400 --meta-batch-list 2,8,16 --inner-steps 3 --device cpu --output results/meta_batch_equal_budget.csv
```
