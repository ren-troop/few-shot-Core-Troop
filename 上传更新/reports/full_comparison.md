# 三算法与半监督完整对比设计

## 对比矩阵

完整实验由三个维度组成：

| 维度 | 设置 |
| --- | --- |
| 算法 | 二阶 MAML、FO-MAML、Meta-SGD |
| 内循环数据 | 仅监督、监督加半监督 |
| 内循环步数 | 1、3、5 或用户指定列表 |

所有组合共享相同的数据划分、网络、Support/Query 数量、外循环优化器和评估任务数。这样可分别回答二阶梯度、可学习内循环学习率、伪标签和适应步数是否带来收益。

## 指标解释

- `eval_rmse_raw`：原始目标尺度 RMSE，对较大误差更敏感；
- `eval_mae_raw`：原始尺度 MAE，较易解释典型绝对误差；
- `best_eval_mse_norm`：训练期间最佳归一化验证 MSE；
- `avg_step_time_sec`：每次外循环平均耗时；
- `grad_norm`：外循环梯度范数，用于发现爆炸或异常波动；
- `meta_lr_*`：Meta-SGD 更新率范围和非正数量。

模型选型不能只看单个最终值。优先比较跨随机种子的均值和标准差，再结合耗时、内存和稳定性选择候选方法。

## 黄河结果对应的判断

当前 3 个随机种子、300 step、3 个 inner steps 下：

1. 六种组合平均 RMSE 为 0.6330 到 0.6380，差异远小于标准差 0.0722 到 0.0804。
2. 半监督版本没有显示稳定精度收益，同时增加运行时间。
3. FO-MAML 的平均每步时间最低，精度与二阶 MAML 接近，适合用作后续大规模消融基线。
4. Meta-SGD 的 MAE 略低，但 RMSE 没有同步领先，目前证据不足以认定其整体最优。

## 后续完整验证

建议至少补充 5 个随机种子、多个训练/验证区域划分，并对伪标签权重和置信度阈值做嵌套消融。只有当改进幅度大于跨种子和跨划分波动，且置信区间稳定，才能形成方法优于基线的结论。

```powershell
python -m experiments.full_comparison --dataset csv --csv-path data/yellow_river_hydromlyr_model_ready.csv --region-column region_id --target-column q --drop-columns date,is_natural --methods "Second-order MAML,FO-MAML,Meta-SGD" --inner-updates supervised,semi-supervised --inner-steps-list 1,3,5 --steps 300 --output-dir outputs/full_comparison
```
