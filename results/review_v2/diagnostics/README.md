# 本次反馈复核说明

这些表从用户返回的完整反馈ZIP重新计算；不是新训练结果。核对脚本仅针对本轮已约定的四方法、seed42/43/44和600轮配置，不是任意配置的通用评测器。通用跨种子汇总用aggregate_seeds.py。

| 文件 | 含义 |
| --- | --- |
| integrity_report.json | 输入包哈希、配置、环境、核验范围和本机水质报告是否附带 |
| run_integrity.csv | 每组训练轮数、测试任务数、验证次数及设备 |
| l2_layer_diagnostics.csv | 每层初末系数、变化比例、非零梯度/更新次数和梯度量级 |
| learning_rate_sign_summary.csv | 逐种子步长总数、负数数量与百分比、全模型最小/最大值 |
| validation_by_seed.csv | 全部72个验证点的准确率 |
| validation_across_seeds.csv | 每方法每验证轮的三种子均值与样本std |
| validation_final_progress.csv | 每条曲线500至600轮的变化，单位为百分点 |
| test_task_variability.csv | 每次运行200测试任务的准确率std与平均损失，不能与跨训练种子std混用 |

脚本对完整训练记录、测试任务、类划分、配置快照、summary/comparison、梯度与softplus轨迹、精简导出进行数值一致性检查。汇总CSV另用aggregate_seeds.py复算，与反馈包中的四张汇总表一致。

若需要复核原反馈包，在仓库根目录执行（ZIP路径按本机位置调整）：

```bat
python tools\review_feedback.py --feedback-zip outputs_review_feedback.zip --output-dir results\review_v2_recheck
```

原始ZIP、图片、全量日志和任务记录只在本机保留，不提交Git。一致性检查不能证明本机源文件未被修改，也不等于另一机器上的独立复现。此脚本的水质检查范围仍为原反馈包中的元数据。用户后来已补回本机CSV检查的通过输出，见 `../../verification/water_checks_local.json` 与 `../../verification/WATER_CHECK_EVIDENCE.md`；原包的local_water_check_attached=false保留，避免改变历史事实。
