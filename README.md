# few-shot-Core-Troop

<!-- MAML_REVIEW_SUBMISSION_20261002 -->

## Omniglot 元学习探索与水质标签缺失基准

本提交完成学习率/正则化元学习机制探索，以及 Water Quality (UCI 733) / Water Potability 的预处理和人为标签缺失基准，对应项目计划书的公开数据探索与半监督数据准备阶段。

### 当前正式结果

修订后四方法 × 三种子 × 600 次外更新已完成，结果位于 `results/review_v2/`；Meta-SGD 采用已确认的**允许负学习率**设定。

| 方法 | 测试准确率：三种子均值 ± 样本标准差 |
| --- | ---: |
| MAML | 63.43% ± 2.73 pp |
| Meta-SGD | 64.01% ± 2.23 pp |
| Fixed-L2 | 62.39% ± 2.81 pp |
| Meta-L2 | 62.47% ± 1.48 pp |

Meta-L2 每层梯度均有回传，系数相对初值下降约 28%–86%，确认存在实际学习；与 Fixed-L2 的平均差仅 +0.09 pp，配对差值区间包含 0，**性能收益仍缺乏充分证据**。Meta-SGD 比 MAML 平均高 +0.58 pp，同样没有稳定提升的证据。验证曲线末段仍上升，不声称已收敛。

原提交保留在 `results/legacy_v1/`；新旧类别划分不同，不混合统计。用户于2026-10-02补回本机终端检查结果：两套水质数据均通过。输出已记录于 `results/verification/water_checks_local.json`，来源说明见同目录 `WATER_CHECK_EVIDENCE.md`。本轮无需重跑训练或数据检查。

### 从这里开始

1. [电脑操作与提交指南](电脑操作与提交指南.md)：完整复现命令，以及保留原分支、备份并重建干净历史的Git步骤。
2. [评审逐项回复](REVIEW_RESPONSE.md)：5 项阻断与 9 项建议的对应修改和证据。
3. [阶段报告](meta_maml_exploration/docs/阶段进度报告.md) 与 [实验说明](meta_maml_exploration/README.md)。
4. [水质基准说明](water_label_missing_ready/README.md)：来源、字段、缺失机制与下游使用。
5. [PR 标题和描述](PR_DESCRIPTION.md)：复制到远端 PR；本文件交付不代表已修改远端。

### 提交范围

提交源码、Markdown、环境说明、根 `.gitignore`、`results/` 下的小型证据表/JSON、水质 manifest.json 和两个 metadata.json。运行输出目录、权重、图片、原始/生成数据 CSV/ZIP/MAT 和工程压缩包均不进 Git。完整日志留在本机；仓库每版仅保留一份抽样训练日志。

### 数据获取与检查（仓库根目录）

```bat
python water_label_missing_ready\prepare_data.py
python water_label_missing_ready\check_data.py --report results\verification\water_checks_local.json
```

- UCI 官方页面：https://archive.ics.uci.edu/dataset/733/water%2Bquality%2Bprediction-1
- UCI 官方下载：https://archive.ics.uci.edu/static/public/733/water%2Bquality%2Bprediction-1.zip
- Water Potability 发布页：https://www.kaggle.com/datasets/adityakadiwal/water-potability
- 与旧提交一致的 CSV 镜像：https://raw.githubusercontent.com/prasadposture/Water-Potability-Project/main/water_potability.csv
- Omniglot：https://github.com/brendenlake/omniglot

下载脚本核验来源字节校验值；Potability 接受已核验内容相同的 LF/CRLF 两种快照。详细参数见 manifest / metadata。完整复现命令见操作指南；已完成本轮实验者不必重复运行。
