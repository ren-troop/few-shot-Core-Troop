# 二阶 MAML 与 FO-MAML 对比

## 算法差异

二阶 MAML 在外循环反向传播时保留内循环梯度更新的计算图，meta-gradient 包含二阶导数信息。FO-MAML 在内循环使用 `create_graph=False`，忽略二阶项，以降低时间和内存成本。

两种模式使用相同初始随机种子、任务采样参数、网络、优化器、半监督权重和 3 个 inner steps。唯一的算法开关是 `create_graph`。

## 300-step 结果

| 模式 | create_graph | 验证损失 | 验证 RMSE | 最佳验证损失 | 平均每步时间/s | 观测进程峰值/MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 二阶 MAML | True | 1.671024 | 1.292681 | 1.671024 | 0.0709 | 270.67 |
| FO-MAML | False | 1.986556 | 1.409452 | 1.986556 | 0.0312 | 46.69 |

FO-MAML 的平均每步时间约为二阶 MAML 的 44%，但验证 RMSE 高约 9%。这体现了典型的速度与精度权衡：二阶信息在本次合成任务中带来一定收益，FO-MAML 则明显降低计算成本。

内存数据是同一进程中顺序执行时的 RSS 观测峰值，受 Python/PyTorch 分配器和执行顺序影响。它适合做运行诊断，不能替代独立进程、相同环境下的严格峰值内存基准。GPU 环境应优先比较 `cuda_peak_memory_mb`。

## 结论边界

该结果来自随机种子 28 的合成回归任务。黄河多随机种子实验中，FO-MAML 与二阶 MAML 的平均 RMSE 接近，而 FO-MAML 更快。因此工程上可优先把 FO-MAML 作为大规模消融基线，再用二阶 MAML 复核最终候选配置；是否接受精度损失，应以真实区域数据的多随机种子置信区间决定。

结果文件：`results/fomaml_formal.csv`。

```powershell
python -m experiments.fomaml_comparison --algorithm maml --modes second-order,fomaml --steps 300 --meta-batch 8 --inner-steps 3 --device cpu --output results/fomaml_formal.csv
```
