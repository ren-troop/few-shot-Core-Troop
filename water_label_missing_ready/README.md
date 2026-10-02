# 水质标签缺失基准（water_review_v2）

## 生成与检查

在仓库根目录的 Anaconda Prompt 运行：

```bat
python water_label_missing_ready\prepare_data.py
python water_label_missing_ready\check_data.py --report results\verification\water_checks_local.json
```

脚本默认下载到本机 `water_label_missing_ready/data/raw/`。若网络不通，直接使用你旧压缩包中的两份原始文件（把示例路径换成实际位置）：

```bat
python water_label_missing_ready\prepare_data.py --uci-zip "D:\旧工程\water_label_missing_ready\data\raw\water_quality_uci_733.zip" --potability-csv "D:\旧工程\water_label_missing_ready\data\raw\water_potability.csv"
```

UCI 原始 `.mat` 从 ZIP 内存读取，不需 MATLAB。所有生成 CSV、原始 ZIP 都只保存在本机，Git 忽略它们。仓库中的 manifest 与每个数据集 metadata 用于记录来源、切分和变换参数。

## 来源和可复现性

- UCI 官方下载：https://archive.ics.uci.edu/static/public/733/water%2Bquality%2Bprediction-1.zip
- UCI 页面：https://archive.ics.uci.edu/dataset/733/water%2Bquality%2Bprediction-1
- Water Potability 发布页：https://www.kaggle.com/datasets/adityakadiwal/water-potability
- 本次使用的 CSV 镜像：https://raw.githubusercontent.com/prasadposture/Water-Potability-Project/main/water_potability.csv

镜像不冒充 UCI 官方或 Kaggle 官方下载接口。脚本使用固定 SHA256 验证字节快照，内容变化会报错而不悄悄混入另一版数据。旧 manifest 的 Potability 哈希对应下载时的 CRLF，Git ZIP 中 CSV 变为 LF；已核对两者归一化换行后完全相同。新版记录实际字节哈希和标准 LF 哈希，并只接受这两个已验证快照。

## 预处理

**UCI：** MAT 中实际有 37 个站点，训练序列 423 天、测试序列 282 天。将每个站点-日期展开一行，共 26085 行。保留官方训练序列的前 338 天作训练、后 85 天作验证；官方测试 282 天不变，同一天的所有站点在同一 split，防止按行随机切分混入未来。11 个特征按 MAT 的 features 顺序命名，目标保留 Y_tr/Y_te 原配对，不再额外移位。

UCI 发布方提供的是已处理数据；本脚本不声称恢复了物理单位、精确的原始标签时移或验证了发布方归一化的拟合范围。进一步做严格预测实验前，应核实这些来源细节。`day_index` 是拼接序列内顺序索引，不是精确日历时间。`location_id` 是站点身份，建议后续用来组织任务，不能不加思考地当连续数值特征。跨站点泛化需要另设站点隔离方案，当前切分只评价同站点时间保留情境。

**Water Potability：** 3276 条记录，类别 0/1 为 1998/1278 条。先按类别分层切分 70%/15%/15%（seed=42，实际 2293/491/492 行），再用训练集中位数填补 ph/Sulfate/Trihalomethanes 等缺失特征，按训练均值和总体标准差（ddof=0）做 z-score。验证/测试只应用训练变换。原始缺失的是特征；人为隐藏的是标签，二者不同。此数据无时间戳，不应将它宣称为时间序列。

## 四份 CSV 与字段

每个数据集生成四份：

| 文件 | 标签用途 | 是否允许当半监督训练输入 |
| --- | --- | --- |
| complete.csv | 完整监督基准，带 label_true | 仅完整监督对照；不得给缺失标签模型偷看 |
| missing_20pct.csv | 训练标签约 20% 隐藏 | 可以 |
| missing_40pct.csv | 训练标签约 40% 隐藏 | 可以 |
| missing_60pct.csv | 训练标签约 60% 隐藏 | 可以 |

| 字段 | 定义 |
| --- | --- |
| row_id | 与原行稳定对应，用于比较掩码；不作为特征 |
| split | train / validation / test；不作为特征 |
| day_index | UCI 时间序列索引；需按时间排序构造窗口 |
| location_id | UCI 站点 ID；用于分组或构造任务 |
| UCI 的 11 个指标列 | 输入特征，名称和顺序见 metadata.features |
| z_ph 等 9 个 z_ 列 | Potability 输入，已按训练统计量标准化 |
| label_true | 只出现在 complete.csv，完整真值 |
| label_partial | 缺失版可见标签，被隐藏者为空/NaN |
| label_observed | 1=标签可见，0=标签隐藏；不是模型特征 |

新版缺失 CSV **不包含 label_true**，与旧文件不同。训练时有标签损失仅使用 `observed=True`；无标签样本可用于组内伪标签或一致性方法，不能把 NaN 填成 0 后当真值训练。验证/测试标签全部可见，但只用于评估。

## 缺失机制

每个数据集独立创建 NumPy default_rng(seed=42)，随机排列训练行，隐藏前 round(n_train×缺失率) 行。抽样不看标签或特征，属于训练行上的 MCAR 固定数量随机隐藏。20% 隐藏集合是 40% 的子集，40% 又是 60% 的子集，便于公平比较。整数取整会造成实际比例与名义比例略有差别。

| 数据集 | 训练行 | 20% 隐藏 | 40% 隐藏 | 60% 隐藏 |
| --- | ---: | ---: | ---: | ---: |
| UCI | 12506 | 2501 | 5002 | 7504 |
| Potability | 2293 | 459 | 917 | 1376 |

旧版缺少生成源码，虽然恢复了相同 Potability 分层切分及预处理，但旧掩码的 RNG 过程没有被可靠重建。v2 Potability 使用明确的新掩码，不声称与 v1 缺失行逐条相同；旧元数据保留供追溯。

## 后续给谁使用

交给组内 MAML/Meta-SGD 表格任务、监督基线和伪标签/半监督模块的同学。UCI 可按站点组织任务并按 day_index 构造窗口；窗口应在各 split 内建立，避免跨边界泄漏。Potability 只作为表格分类和标签缺失对照；它不能直接接到当前只接受 28×28 图片的 Omniglot CNN。

提供 `load_benchmark.load_split(...)`，返回 `X/y/observed/row_id/feature_names`，通过显式特征白名单排除标签、掩码和 ID。监督训练例：

```python
from water_label_missing_ready.load_benchmark import load_split
batch = load_split('water_label_missing_ready/data', 'water_potability', 40, 'train')
X_labeled = batch['X'][batch['observed']]
y_labeled = batch['y'][batch['observed']].astype('int64')
X_unlabeled = batch['X'][~batch['observed']]
```

`check_data.py` 检查来源哈希、重新预处理一致性、时间切分、训练标准化、掩码数量/嵌套性、非训练标签完整和缺失版无完整真值列。
