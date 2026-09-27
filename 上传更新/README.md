# 跨区域 MAML、FO-MAML 与 Meta-SGD 实验

本项目实现面向区域回归任务的二阶 MAML、FO-MAML 和 Meta-SGD，并支持监督与半监督内循环。代码包含合成正弦任务消融、公开数据跨区域对比、黄河流域 HydroMLYR 多随机种子实验、数据校验和结果可视化。

## 目录结构

```text
src/          共享算法框架
experiments/  各项实验入口
scripts/      HydroMLYR 数据校验与特征构造
results/      可直接审阅的文本结果
reports/      实验方法和结果说明
data/         本地数据目录，不纳入版本控制
outputs/      运行时输出目录，不纳入版本控制
```

算法实现只有两套共享模块：

- `src/maml_meta_sgd_framework.py`：合成少样本回归任务及 MAML/Meta-SGD 基础实现。
- `src/region_framework.py`：区域数据读取、任务采样、伪标签、跨区域训练与评估。

实验脚本统一从共享模块导入，不需要复制框架文件。

## 环境安装

建议使用 Python 3.10 或更高版本。在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

只检查命令行参数和模块依赖：

```powershell
python -m experiments.meta_batch_ablation --help
python -m experiments.yellow_river_multiseed --help
```

## HydroMLYR 数据来源与准备

黄河实验使用公开数据集 **CCAM: China Catchment Attributes and Meteorology dataset** 中的 HydroMLYR 部分，来源 DOI：<https://doi.org/10.5281/zenodo.5729444>。

将以下文件从 Zenodo 下载到项目的 `data/` 目录：

- `7_HydroMLYR.zip`
- `readme.txt`
- `8_attribute_descriptions.xlsx`

PowerShell 下载示例：

```powershell
New-Item -ItemType Directory -Force data
Invoke-WebRequest "https://zenodo.org/records/5729444/files/7_HydroMLYR.zip?download=1" -OutFile "data/7_HydroMLYR.zip"
Invoke-WebRequest "https://zenodo.org/records/5729444/files/readme.txt?download=1" -OutFile "data/readme.txt"
Invoke-WebRequest "https://zenodo.org/records/5729444/files/8_attribute_descriptions.xlsx?download=1" -OutFile "data/8_attribute_descriptions.xlsx"
```

校验文件哈希并生成模型输入：

```powershell
python -m scripts.prepare_hydromlyr --data-dir data
```

脚本直接读取 ZIP，不需要手工解压。它会生成：

- `data/hydromlyr_validation_summary.csv`：各流域样本范围和字段检查。
- `data/yellow_river_hydromlyr_model_ready.csv`：模型可读数据。

模型表包含 `region_id`、`date`、`is_natural`、目标 `q`，以及降水、蒸发、日照、温度、地温、气压、湿度和风速的 7 日/14 日统计量，共 16 个数值特征。数据文件和生成图表均由 `.gitignore` 排除，仓库只保存代码、报告和必要的文本结果。

## 算法与数据流

每个流域视为一个任务。训练流域只用于元训练，验证流域只用于跨区域评估，两个集合必须互斥。每个任务从同一流域无放回抽取：

1. Support Set：内循环快速适应。
2. Query Set：计算外循环 meta-loss 或评估误差。
3. Unlabeled Set：半监督模式下生成伪标签并按置信度加权。

区域样本不足 `shots + queries + unlabeled_per_task` 时，该区域不会进入采样池。Support、Query 和 Unlabeled 不会复用同一行，因此评估样本不会参与当前任务的适应过程。

Meta-SGD 为每个模型参数学习内循环更新率。每次外循环优化后，更新率被限制在 `[min_meta_lr, clamp_meta_lr]`；CSV 同时记录 `meta_lr_mean`、`meta_lr_min`、`meta_lr_max` 和 `meta_lr_nonpositive_count`。

## 主要函数

| 函数 | 作用 |
| --- | --- |
| `RegionalTaskSampler.sample` | 按区域构造互不重叠的 Support、Query 和 Unlabeled 数据 |
| `RegionalTaskSampler.make_pseudo_labels` | 使用支持样本最近邻生成伪标签和距离置信度 |
| `compute_inner_loss` | 合成监督损失与半监督加权损失 |
| `MAMLAdapter.adapt` | 使用固定内循环学习率完成任务适应 |
| `MetaSGDAdapter.adapt` | 使用逐参数可学习更新率完成任务适应 |
| `meta_train_step` | 执行一个 meta-batch 的内循环、Query 损失反传和外循环更新 |
| `evaluate` | 在独立验证流域上计算归一化与原始尺度误差 |
| `make_region_dataset` | 读取公开数据或自定义 CSV，完成清洗、标准化和区域划分 |

## 实验命令

### FO-MAML 与二阶 MAML

```powershell
python -m experiments.fomaml_comparison --steps 300 --meta-batch 8 --inner-steps 3 --device auto --output results/fomaml_formal.csv
```

### 内循环步数 1/3/5

```powershell
python -m experiments.inner_steps_ablation --steps 300 --meta-batch 8 --inner-steps-list 1,3,5 --device auto --output results/inner_steps_formal.csv
```

### MetaBatch 2/8/16

相同外循环步数：

```powershell
python -m experiments.meta_batch_ablation --steps 300 --meta-batch-list 2,8,16 --inner-steps 3 --device auto --output results/meta_batch_formal.csv
```

相同任务预算：

```powershell
python -m experiments.meta_batch_ablation --total-task-budget 2400 --meta-batch-list 2,8,16 --inner-steps 3 --device auto --output results/meta_batch_equal_budget.csv
```

### 公开数据快速验证

不准备 CSV 时，可用 scikit-learn California Housing 检查完整跨区域流程：

```powershell
python -m experiments.full_comparison --dataset california --steps 10 --methods "Second-order MAML,FO-MAML,Meta-SGD" --inner-updates supervised,semi-supervised --inner-steps-list 1,3 --output-dir outputs/california_smoke
```

### 黄河流域多随机种子完整实验

```powershell
python -m experiments.yellow_river_multiseed --dataset csv --csv-path data/yellow_river_hydromlyr_model_ready.csv --seeds 28,42,2026 --steps 300 --meta-batch 8 --inner-steps-list 3 --output-dir outputs/yellow_river_multiseed
```

每个入口均应使用 `python -m experiments.<模块名>` 从项目根目录启动，这样共享模块导入路径稳定。

## 结果解释原则

`results/meta_batch_formal.csv` 在每种 batch 下都执行 300 个外循环，因此 batch=8/16 分别处理了 batch=2 的 4/8 倍任务。该设置可以比较固定更新次数下的效果和成本，但不能单独证明大 batch 更优。

`results/meta_batch_equal_budget.csv` 将训练任务数统一为 2400。当前单随机种子结果中，batch=2 的最佳验证损失最低；因此现阶段只能把 batch=8 视为并行度、单步稳定性和训练成本之间的工程候选，不能写成确定的性能结论。正式汇报应增加多个随机种子并报告均值、标准差或置信区间。

黄河多随机种子结果显示六种组合的平均 RMSE 接近，差异小于跨随机种子波动。当前伪标签策略没有带来稳定收益，FO-MAML 的速度优势较明显。结论应限定在现有数据划分、300 step 和当前超参数范围内。

更详细的参数、字段和结论见 `reports/`。
