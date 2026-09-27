# 黄河流域三算法多随机种子实验

## 数据与任务

实验使用 HydroMLYR 连续流量和气象记录。每个流域作为一个任务，33 个流域用于元训练，11 个互斥流域用于验证。输入为 16 个 7 日/14 日气象窗口特征，预测目标为流量 `q`。

每个任务使用 8 个 Support、16 个 Query 和 16 个 Unlabeled 样本；样本在任务内无放回抽取。半监督模式使用 Support 最近邻为 Unlabeled 生成伪标签，并按特征距离生成置信度权重。

## 实验设置

- 算法：二阶 MAML、FO-MAML、Meta-SGD；
- 内循环：监督、半监督；
- Inner steps：3；
- MetaBatch：8；
- 外循环：300 step；
- 随机种子：28、42、2026。

## 汇总结果

| 方法 | 内循环 | RMSE 均值 ± 标准差 | MAE 均值 ± 标准差 | 平均每步时间/s |
| --- | --- | ---: | ---: | ---: |
| FO-MAML | 监督 | 0.6330 ± 0.0776 | 0.4230 ± 0.0459 | 0.0290 |
| FO-MAML | 半监督 | 0.6332 ± 0.0786 | 0.4235 ± 0.0473 | 0.0451 |
| 二阶 MAML | 监督 | 0.6357 ± 0.0792 | 0.4230 ± 0.0501 | 0.0499 |
| 二阶 MAML | 半监督 | 0.6373 ± 0.0804 | 0.4240 ± 0.0506 | 0.0835 |
| Meta-SGD | 监督 | 0.6375 ± 0.0722 | 0.4170 ± 0.0378 | 0.0515 |
| Meta-SGD | 半监督 | 0.6380 ± 0.0723 | 0.4182 ± 0.0379 | 0.0860 |

## 分析

六种配置的平均 RMSE 差异很小，明显小于跨随机种子的标准差，不能据此宣称某个算法在预测精度上稳定占优。Meta-SGD 的平均 MAE 略低，但 RMSE 没有同步改善，说明优势不足以形成稳健结论。

当前半监督配置没有降低平均 RMSE 或 MAE，并增加了约 50% 到 70% 的单步时间。原因可能包括最近邻伪标签信息量有限、特征距离未完全反映水文状态相似性，以及固定 `ssl_weight=0.1` 未按伪标签质量动态调节。它说明当前伪标签策略需要进一步设计，不说明半监督学习本身无效。

FO-MAML 的平均精度与二阶 MAML 接近，但单步耗时最低，因而是当前配置下更适合作为工程基线的方法。正式模型选型仍应增加区域划分重复、更多随机种子和超参数搜索，并报告置信区间。

## 结果文件

- `results/yellow_river_multiseed_all.csv`：18 个单次配置结果；
- `results/yellow_river_multiseed_summary.csv`：按算法和内循环类型汇总的均值与标准差。

```powershell
python -m experiments.yellow_river_multiseed --csv-path data/yellow_river_hydromlyr_model_ready.csv --seeds 28,42,2026 --steps 300 --meta-batch 8 --inner-steps-list 3 --output-dir outputs/yellow_river_multiseed
```
