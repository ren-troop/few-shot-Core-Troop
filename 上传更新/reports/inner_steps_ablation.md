# 内循环步数 1/3/5 消融实验

## 原理

内循环使用 Support Set 对当前任务进行快速适应。步数太少时可能欠拟合，步数太多时可能对少量支持样本和伪标签过拟合，同时增加二阶梯度的计算成本。本实验在其他参数不变时比较 1、3、5 步。

## 设置

- 算法：Meta-SGD；
- 内循环：半监督；
- MetaBatch：8；
- Support/Query/Unlabeled：5/10/10；
- 外循环：300 step；
- 随机种子：28；
- Meta-SGD 学习率范围：`[1e-6, 0.2]`。

## 结果

| Inner steps | 最终验证损失 | 最佳验证损失 | 最终 meta-loss | 梯度范数 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.576114 | 0.576114 | 1.150378 | 3.570285 |
| 3 | 0.302060 | 0.302060 | 0.632128 | 3.284543 |
| 5 | 0.262543 | 0.262543 | 0.543966 | 3.944254 |

在本次合成任务中，增加内循环步数持续降低验证损失，5 步最好；3 步已取得大部分收益，且计算成本低于 5 步。三个设置的 `meta_lr_nonpositive_count` 均为 0，学习率下界生效。

该结果只覆盖一个随机种子和合成任务。水文区域数据上应重复相同消融，并同时检查验证误差、跨种子标准差和平均每步耗时。若 5 步只带来很小收益，3 步通常更适合作为默认值。

结果文件：`results/inner_steps_formal.csv`。

```powershell
python -m experiments.inner_steps_ablation --steps 300 --meta-batch 8 --inner-steps-list 1,3,5 --device cpu --output results/inner_steps_formal.csv
```
