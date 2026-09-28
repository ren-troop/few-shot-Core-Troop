# 框架、输入输出与数据接入

## 输入

跨区域框架接收一个表格型数据集。每行代表某流域在某个日期的一次有效观测，至少需要：

- 区域列：默认 `region_id`，决定任务归属；
- 目标列：默认 `q`，即要预测的连续值；
- 数值特征列：用于模型输入；
- 可选元数据列：如 `date`、`is_natural`，可通过 `--drop-columns` 排除。

CSV 中的缺失值和非数值列会在数据构建阶段处理。特征和目标只使用训练部分统计量标准化，防止验证区域信息泄漏到训练预处理。

## 输出

训练脚本输出 CSV，每行对应一个完整配置。主要字段包括：

- 配置：算法、是否一阶、内循环类型、步数、MetaBatch、随机种子；
- 数据：训练/验证区域数、特征数、预测目标；
- 损失：support、SSL、inner、meta；
- 评估：归一化 MSE/RMSE、原始尺度 RMSE/MAE；
- 稳定性：梯度范数、Meta-SGD 学习率统计；
- 成本：总耗时、平均每步时间、GPU 峰值显存。

可视化脚本从结果 CSV 生成 PNG/PDF。图片属于可再生输出，不纳入版本控制。

## 算法执行流程

1. 按流域拆分训练区域与验证区域，二者严格互斥。
2. 从一个区域无放回抽取 Support、Query 和 Unlabeled。
3. 在 Support 上计算监督损失；半监督模式再计算置信度加权伪标签损失。
4. MAML 用固定 `inner_lr` 更新，Meta-SGD 用逐参数 `meta_lr` 更新。
5. 重复指定 inner steps，得到任务适应参数。
6. 用适应参数计算 Query loss，并在 meta-batch 内求平均。
7. 外循环更新共同初始化参数；Meta-SGD 还更新并约束逐参数学习率。
8. 在独立验证区域重复适应和评估，转换回原始目标尺度输出 RMSE/MAE。

## HydroMLYR 特征构造

`scripts.prepare_hydromlyr` 从 ZIP 中逐流域读取：

- `meteorological.txt`：气象时间序列；
- `continuous.txt`：连续流量 `q`；
- `natural_basins.txt`：自然流域标识。

脚本按日期对齐流量与气象数据，并构造 7 日和 14 日窗口：

- 累计量：降水 `pre`、蒸发 `evp`、日照 `ssd`；
- 均值：平均温度、平均地温、平均气压、相对湿度、平均风速。

两个窗口各产生 8 个特征，共 16 个输入。输出保留日期、流域、自然属性和流量目标，便于检查来源和扩展新的区域划分。

## 接入自定义数据

```powershell
python -m experiments.full_comparison `
  --dataset csv `
  --csv-path data/my_regions.csv `
  --region-column station_id `
  --target-column discharge `
  --drop-columns date,station_name `
  --min-samples-per-region 40 `
  --steps 300 `
  --output-dir outputs/my_regions
```

当 `shots=8`、`queries=16`、`unlabeled_per_task=16` 时，每个可用区域至少要有 40 行完整样本。这是一次任务无放回采样的结构性下限，不是模型达到可靠预测精度的统计学门限。可靠性门限需通过学习曲线、多区域划分和多随机种子实验确定。
