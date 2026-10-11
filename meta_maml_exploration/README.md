# MAML 学习率与正则化元学习探索

## Meta-SGD 学习率约束

Meta-SGD 为模型参数学习内循环步长。约束逻辑统一放在仓库根目录 `src/common/meta_sgd_lr.py`，训练器通过 `apply_meta_sgd_constraint` 调用，并用 `summarize_lr` 记录实际生效的步长。

| 模式 | 生效步长 | 用途 |
| --- | --- | --- |
| `clamp` | 限制在 `[1e-6, 0.2]` | 与崔锦程当前的学习率范围一致；正式重跑脚本采用此模式 |
| `softplus` | `softplus(raw)`，大于 0 但没有 0.2 上限 | 共享工具的推荐默认模式，作为对照变体 |
| `none` | 使用 raw 值，允许负数 | signed 对照变体 |

不要把 `softplus` 当作 `[1e-6, 0.2]` 的限制；只有 `clamp` 才符合这个范围。`run_windows.bat` 显式传入 `--meta-sgd-mode clamp --meta-sgd-clamp-min 0.000001 --meta-sgd-clamp-max 0.2`。

## 方法与当前复跑配置

| 方法 | 内循环步长 | 支持集正则化 |
| --- | --- | --- |
| maml | 固定 0.4 | 无 |
| meta_sgd | 逐参数学习，clamp 到 `[1e-6, 0.2]` | 无 |
| fixed_l2 | 固定 0.4 | 每权重层固定 L2 |
| meta_l2 | 固定 0.4 | 每权重层可学习 L2 |

复跑沿用此压缩包已有的 Omniglot 设置，以便只改变 Meta-SGD 约束：5-way 1-shot、每类 5 query、四个卷积块、600 次外循环更新、每种子 200 个测试任务，seed 42/43/44。每 100 次更新在独立验证类别上评估一次。若要完成组内另行规定的 5-way 5-shot、五种子和三模式消融，需使用单独的实验配置，不能把它与本次 1-shot 复跑混为一表。

## Windows 运行

在 Anaconda Prompt 中进入仓库根目录，确认 PyTorch 与 torchvision 可导入，并安装其余依赖：

```bat
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

短流程检查：

```bat
meta_maml_exploration\quick_check.bat
```

正式运行：

```bat
meta_maml_exploration\run_windows.bat
```

正式脚本会清理旧的 `outputs_review_seed42/43/44`、quick-check 输出、旧反馈压缩包、`results/legacy_v1/`、`results/review_v2/` 和旧训练审计文件，然后训练并重新导出 `results/review_v2/`。旧正式结果不会与新结果混合。若运行中断，重新运行该批处理会清理残留的正式输出后从头开始。

## 输出文件

每个本机 `outputs_review_seed*/<method>/` 包含 `summary.json`、`configuration.json`、`split.json`、`training_log.csv`、`validation_log.csv`、`test_tasks.csv` 和 `learned_hyperparameters.csv`。Meta-SGD 另有 `learning_rate_signs.csv` 与 `clamp_count`；Meta-L2 另有 `l2_trajectory.csv`。`--plots` 生成本机 PNG 图。默认不保存 checkpoint。

批处理末尾运行 `tools/export_results.py`，生成精简的 `results/review_v2/` 和 `outputs_review_feedback_clamp.zip`。汇总含每种子数据、跨种子均值/标准差/置信区间和配对差值；完整日志与图片只放在本机运行目录/反馈压缩包，不提交 Git。

## 水质数据

`water_label_missing_ready/` 的水质数据生成、检查与加载流程与 Meta-SGD 约束无关。Water Potability 是静态分类数据；UCI Water Quality 保留时间和站点结构。缺失标签仅在训练集构造，详细字段和防泄漏约定见 `water_label_missing_ready/README.md`。
