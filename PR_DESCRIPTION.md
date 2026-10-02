# 建议标题

Omniglot 元学习探索：Meta-SGD 与 Meta-L2 对比实验及水质标签缺失基准

# 可复制到 PR 描述的正文

## 背景与改动

完成分配的两项任务：学习率/正则化元学习探索与假设记录；Water Quality (UCI) 和 Water Potability 预处理及训练标签缺失构造。本次修订解决评审指出的大文件入库、Meta-L2 更新证据不足和复现资料不完整问题。

- 仅提交源码、文档和轻量证据；统一根 .gitignore，checkpoint 默认关闭，清理本机绝对路径。旧结果单独保留在 results/legacy_v1。
- Meta-L2 设置独立外学习率0.01，记录每层梯度、raw更新与系数轨迹；补充独立类别验证、超参scope、步长符号分布及跨种子汇总。
- 共用参数定义保证超参透传，校验 initial_l2 > 0；Meta-SGD 按已确认的设定允许负学习率，正数版本仅作为另一个可选消融。
- 提供水质生成、检查、加载脚本与README：训练集MCAR 20/40/60%嵌套标签隐藏，验证/测试完整；缺失版移除label_true，特征预处理使用训练统计量（UCI发布数据已有预处理，范围限制见README）。

## 正式结果

修订后600次外更新、200测试任务、四方法×seed42/43/44已完成。环境：Windows、Python3.11.16、PyTorch2.2.2、torchvision0.17.2、CPU。类别划分为771训练/193验证/659测试；每100轮50个固定验证任务。

| 方法 | 测试准确率均值 ± 种子间样本标准差 |
| --- | ---: |
| MAML | 63.43% ± 2.73 pp |
| Meta-SGD | 64.01% ± 2.23 pp |
| Fixed-L2 | 62.39% ± 2.81 pp |
| Meta-L2 | 62.47% ± 1.48 pp |

Meta-SGD−MAML为+0.58 pp，配对95% t CI为[-3.69,+4.85] pp；尚无稳定提升证据。Meta-L2每层梯度在600轮均非零，系数下降约28%–86%，已核实实际更新；Meta-L2−Fixed-L2仅+0.09 pp，差值区间[-5.30,+5.47] pp，H2的性能收益仍证据不足，不写成成立或不成立。

验证末段仍上升，本轮属于有限预算比较，不宣称收敛。新旧训练类别划分不同，不混用结果。详见 [阶段报告](meta_maml_exploration/docs/阶段进度报告.md)、[正式汇总](results/review_v2/aggregate/aggregate.csv)、[逐层诊断](results/review_v2/diagnostics/l2_layer_diagnostics.csv)。

## 核验与复现

12组任务级统计、配置、日志长度、类别划分、梯度轨迹与精简导出一致性检查通过；此前8项离线回归测试和水质重建检查通过。用户于2026-10-02补充本机水质检查输出，两套数据均为passed（UCI 26085行、Potability 3276行），切分数量与manifest一致；记录见results/verification/water_checks_local.json及WATER_CHECK_EVIDENCE.md。以下命令保留供复现，已完成本轮检查者无需再跑。

仓库根目录、已激活匹配的PyTorch/torchvision环境：

```bat
python -m pip install -r requirements.txt
python water_label_missing_ready\prepare_data.py
python water_label_missing_ready\check_data.py --report results\verification\water_checks_local.json
python meta_maml_exploration\run_all.py --download --seed 42 --device cpu --num-threads 4 --episodes 600 --eval-episodes 200 --meta-batch 4 --val-every 100 --val-episodes 50 --inner-lr 0.4 --outer-lr 0.001 --initial-l2 0.0001 --l2-outer-lr 0.01 --l2-adam-eps 0.00000001 --hidden-channels 32 --plots --output-root outputs_reproduce_seed42
```

将 seed / output-root 同步改为43、44再运行。默认允许负步长，不加 --meta-sgd-positive；新复现使用新输出目录，已完成本轮者不用重跑。完整汇总和Git步骤见操作指南。

```bat
python -m unittest discover -s tests -v
python tools\audit_submission.py --git-index
```

按提交者要求沿用原分支：先保存本地备份，以干净目标分支重建一个提交，只加入本次源码和精简证据，再用显式force-with-lease更新原head分支。仅删除大文件或添加.gitignore不能清理旧提交。该历史重建由提交者按操作指南执行，交付压缩包本身不等于远端已清理。图片作为PR附件，不入Git。
