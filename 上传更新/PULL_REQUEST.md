# 跨区域元学习实验框架

## 内容

- 提供 MAML、FO-MAML、Meta-SGD 的共享 PyTorch 实现。
- 支持监督和基于置信度伪标签的半监督内循环。
- 提供内循环步数、MetaBatch、算法类型和半监督开关消融入口。
- 提供 HydroMLYR 数据哈希校验、窗口特征构造和黄河流域多随机种子实验。
- 记录 loss、RMSE、MAE、梯度范数、学习率统计、耗时和显存/进程内存。
- 数据、模型、图片和运行输出由 `.gitignore` 排除，通过 README 中的数据源和命令复现。

## 复现

```powershell
python -m pip install -r requirements.txt
python -m experiments.meta_batch_ablation --steps 300 --meta-batch-list 2,8,16 --output results/meta_batch_formal.csv
python -m experiments.meta_batch_ablation --total-task-budget 2400 --meta-batch-list 2,8,16 --output results/meta_batch_equal_budget.csv
python -m experiments.yellow_river_multiseed --csv-path data/yellow_river_hydromlyr_model_ready.csv --seeds 28,42,2026 --steps 300
```

## 结果边界

相同外循环步数与相同任务预算属于不同控制变量，报告分别呈现。黄河多随机种子实验未显示半监督伪标签在当前配置下具有稳定优势，FO-MAML 则具有较低的单步耗时；进一步结论需要扩大随机种子、区域划分和超参数范围。
