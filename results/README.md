# 结果目录约定

- `review_v2/`：本次修订后真实600次外更新 × 四方法 × 三种子的正式结果，Meta-SGD允许负步长；这是当前报告的主证据。
- `review_v2/aggregate/`：跨种子均值、样本标准差、t CI、区间重叠描述与配对差值。
- `review_v2/seed*/`：各运行摘要、配置、类别划分、验证记录、超参数和符号统计；Meta-L2抽样轨迹。
- `review_v2/diagnostics/`：由返回的完整反馈包重算的一致性检查、逐层L2诊断、符号总量、验证趋势及测试任务离散度；核对脚本为 `tools/review_feedback.py`。
- `legacy_v1/`：原提交的正式结果；旧协议无独立验证，不能与v2混合统计。
- `legacy_v1/aggregate/`：仅对旧数值重新汇总，未重新训练。
- `legacy_v1/water_metadata/`：旧水质来源和预处理记录。
- `verification/`：此前代码/水质重建/短流程检查，不能作为算法效果证据。本机补充水质检查报告也放此处。

每版最多保留一份按20轮抽样的 training_log_sample.csv；L2另保留每层抽样轨迹。完整训练日志、200任务逐条记录、图像保留在本机反馈包，不进Git。diagnostics的小表来源为这些完整记录，便于审查。

历史 accuracy_ci95 沿用旧程序的1.96近似，不伪改原数值；新版单模型任务CI和跨种子CI使用Student t。三种子的样本标准差/CI与单模型测试任务的标准差/CI不是同一个统计量。

用户随后于2026-10-02补回本机水质检查终端输出，两套数据均通过；已记录在 `verification/water_checks_local.json`，来源见 `verification/WATER_CHECK_EVIDENCE.md`。原反馈ZIP未附该报告这一事实不变，故原包integrity_report中的local_water_check_attached仍为false，不代表现在尚未检查。
