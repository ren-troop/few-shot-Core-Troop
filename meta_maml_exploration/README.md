# MAML 学习率与正则化元学习探索

## 四种方法

| 方法 | 内循环步长 | 支持集正则化 | 比较目的 |
| --- | --- | --- | --- |
| maml | 固定 0.4 | 无 | 共同基线 |
| meta_sgd | 逐参数可学习，默认允许负值 | 无 | 检验 H1 |
| fixed_l2 | 固定 0.4 | 每权重层固定 L2 | 排除单纯加入正则化的影响 |
| meta_l2 | 固定 0.4 | 每权重层可学习 L2 | 与 fixed_l2 比较检验 H2 |

这是自定义 Meta-L2 探索变体，不是 MetaReg 论文的完整复现。网络为四个卷积块 + 线性分类头，无 BatchNorm。

## 新版实验协议

5-way 1-shot，每类 5 query；一次外循环更新处理 4 个任务。`--episodes 600` 指 600 次外循环更新，即 2400 个训练任务；不是遍历全数据 600 遍。

官方 background 的 964 个字符类别，固定 `split_seed=2026` 后划为训练 771 类 / 验证 193 类；evaluation 的 659 类只用于最终测试。四方法和三个训练随机种子共用同一类别划分。验证任务由独立 RNG 抽样，每轮验证重用固定的 50 个任务，训练 RNG 不受验证影响。默认每 100 episode 验证一次，结束时再验证；不以测试集挑参数，不按测试分数选择模型。最终测试固定 200 个任务，报告最后一轮模型。

**本轮四方法 × 三种子已按新协议全部重跑。原版没有独立验证，与新版协议不同，不能将旧MAML和新Meta-L2拼成比较表。** 正式结果与诊断见 `../results/review_v2/` 和《阶段进度报告》第6节。

## Meta-L2 的实际公式和诊断

```text
rho_l = inverse_softplus(initial_l2)
lambda_l = softplus(rho_l)
L_support = CrossEntropy + sum_l lambda_l * mean(theta_l ** 2)
theta'_l = theta_l - inner_lr * d(L_support)/d(theta_l)
L_meta = CrossEntropy(query using theta')
```

偏置不加 L2。`initial_l2=1e-4` 对应 `rho≈-9.21029`；正则项对单个权重的梯度是 `2*lambda*theta/N_l`。softplus 的导数在该处约为 `1e-4`。因此层内取平均、参数化导数以及 Adam 的 eps 都可能影响更新；不能把“小变化”直接等同于“梯度不通”或“假设失败”。也不存在系数必须变化多少才算学到的通用阈值。

网络及 Meta-SGD 步长使用 `outer_lr=1e-3`，raw_l2 独立使用 `l2_outer_lr=1e-2`；默认仍为 Adam `eps=1e-8`。`--l2-adam-eps` 可配置，但正式复核先保留默认，避免同时改变多个因素。逐轮记录裁剪前的每层 raw 梯度、总 raw 梯度范数、裁剪前全参数梯度范数、更新后 lambda、raw 参数变化及相对初值变化。系数轨迹含 episode=0 的初值。有限差分测试检查元梯度是否正确。

`--initial-l2` 必须为有限正数；选择不加正则化请用 `maml`。Meta-L2 必须使用二阶梯度，因此默认四方法的一键命令不能加 `--first-order`；要做一阶消融时显式选择 `--methods maml meta_sgd fixed_l2 --first-order`，且单独命名实验。

## Meta-SGD 负步长口径

默认 signed 参数化保留原实验语义：更新尺度和方向一起学习。负步长表示该参数沿 support 梯度方向更新，并不等于整个 support loss 必然增加，也不能据此认定 query 泛化变差。

提供 `--meta-sgd-positive`（softplus）作为**另一个实验变体**，不会自动将旧结果改为正值。本轮已由提交者确认采用允许负学习率的设定，`meta_sgd_positive=false`。组内正数约束模块应作为另一变体明确命名；本轮不截断或取绝对值，也不需要重跑正数版本。

`learned_hyperparameters.csv` 中 `scope=per_parameter` 表示每个参数元素一个步长；`scope=per_layer` 表示每层一个标量，因此 Meta-L2 的 std=0、min=max=mean 是正常定义。`learning_rate_signs.csv` 给出每层负/零/正个数和负值比例。

## 文件入口

- `meta_experiment/config.py`：两个训练入口共享参数定义，防止遗漏透传。
- `train.py`：训练/验证/最终测试，默认不保存 checkpoint，启用须显式 `--save-checkpoint`。
- `run_all.py`：相同配置运行四方法，`--methods` 可选子集，`--plots` 生成本机图。
- `plot_results.py`：本机验证曲线、L2 轨迹、学习率符号图。
- `aggregate_seeds.py`：检查配置、拒绝重复种子/混合协议，输出种子间均值、样本 std、Student t CI、CI 是否重叠和配对差值 CI。
- 根目录 `tools/export_results.py`：精简提交结果，并打包回传诊断结果，不包含权重或数据。

`summary.json` 只记录相对路径，外部数据目录写为 `<external>/目录名`；环境记录含 Python/PyTorch/torchvision/Numpy/Pandas、设备和耗时。输出目录非空时拒绝覆盖，避免混入旧权重或旧结果。重复运行请换输出目录名。

## 结果解释

单模型的 task-level CI 衡量抽样测试任务的不确定性；跨种子的 CI 衡量重复训练结果的不确定性，不能混用。种子间 std 使用 ddof=1；3 种子的 t 临界值约 4.303，因此区间通常较宽。CI 重叠只是描述，不等于显著性检验；同时查看同种子配对差值及其区间。三种子证据有限，不能仅看排序宣称方法优越。

## 参考

- MAML：https://arxiv.org/abs/1703.03400
- Meta-SGD：https://arxiv.org/abs/1707.09835
- PyTorch 参数组：https://docs.pytorch.org/docs/stable/optim.html#per-parameter-options
- Adam：https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
- Omniglot 接口：https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Omniglot.html
