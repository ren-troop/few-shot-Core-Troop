# 用户本机水质检查记录

接收日期：2026-10-02（北京时间）。来源：提交者在对话中粘贴的 Anaconda Prompt 完整 JSON 输出。执行命令：

```bat
python water_label_missing_ready\check_data.py --report results\verification\water_checks_local.json
```

| 数据集 | 总行数 | 训练 | 验证 | 测试 | 检查状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| water_quality_uci | 26085 | 12506 | 3145 | 10434 | passed |
| water_potability | 3276 | 2293 | 491 | 492 | passed |

同目录的 water_checks_local.json 按用户贴出的输出原样记录字段与数值；未收到原始 JSON 文件本体，因此不声称与本机文件逐字节一致。行数及切分数量与本工程 manifest 一致。

check_data.py 检查来源文件哈希、从原始数据重建的完整表、训练集特征填充与标准化、UCI时间切分、缺失比例及嵌套性、仅训练集隐藏标签、缺失版移除label_true、可见标签一致性和特征数值有限性。用户返回的结果显示两个数据集都通过这些检查。

本记录来自用户本机运行，不是分析环境中的再次执行，也不消除UCI发布方预处理统计范围未知这一限制。

原始 outputs_review_feedback.zip 确实未附此报告，所以 review_v2/diagnostics/integrity_report.json 的 local_water_check_attached=false 保留原值，表示原反馈包内容。用户随后补回的本机检查结果由本记录说明，不篡改原包核对结论。
