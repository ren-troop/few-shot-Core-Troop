# 半监督信号该放进 MAML 的哪一层？

**研究问题**：把半监督（SSL）损失项放进 MAML 的 inner loop，能提升小样本性能吗？
如果不能，应该放在哪里？

**结论**：框架上放得进去，但**会掉点**；放到**适应之后**才有效，在真实水数据上
把 MSE 降低了 **11.8%–24.1%**。

---

## 1. 三个主要结论

### 结论一：内层 SSL 一致有害

同一个 MAML 主干（合成 5-way 1-shot，准确率 %，400 个测试任务）：

| 无标签用法 | 准确率 | 变化 |
|---|---|---|
| 不用（纯监督适应） | **75.86** | — |
| 内层 + 熵最小化 | 63.05 | **−12.81** |
| 内层 + 原型蒸馏 | 71.64 | **−4.22** |
| 内层 + LP 蒸馏 | 65.96 | **−9.90** |
| **适应之后 + 标签传播** | **82.64** | **+6.78** |

### 结论二：不是调参问题

内循环熵最小化的 `(α, m)` 扫描（Δ准确率，百分点）：

| | m=1 | m=3 | m=8 |
|---|---|---|---|
| α=0.05 | +0.83 | +0.40 | −0.68 |
| α=0.1 | +0.19 | −0.09 | −4.35 |
| α=0.2 | −0.09 | **−12.36** | **−16.69** |
| α=0.4 | −0.17 | **−12.81** | **−23.33** |
| α=0.8 | −0.28 | −7.28 | −17.68 |

**只有「死区」和「断崖」，没有「先升后降」** → 问题在**方向**，不在**力度**。

机理：内层只走 $m$ 步，$\theta' = \theta_0 + O(\alpha m)$，伪标签由 $\theta'$ 产生，
而 $\theta_0$ 是**跨任务共享**的，所以伪标签里**任务特异的部分只有 $O(\alpha m)$** ——
这一项实际退化成「对初始化的正则」，而不是「每个任务各自的半监督适应」。

**对照证据**：把一致性项的教师**冻结在 $\theta_0$**，则 $m=1$ 时它与纯支持集 CE
**逐位完全相同**（69.74/69.74、75.78/75.78、77.31/77.31、77.39/77.39、76.41/76.41），
因为在 $\theta_0$ 处损失和梯度都恰为 0。

同一个熵最小化项，放内层 **−12.81**，放外层（元正则）**+3.58**。

### 结论三：真实水数据上 Pseudo-Label 放在「适应之后」有效

USGS 水数据（37 站 × 705 天 × 11 指标，预测次日 pH）：

| 设定 | 纯监督 MSE | Pseudo-Label MSE | 配对 ΔMSE | 相对 |
|---|---|---|---|---|
| 随机划分 K=10 | 0.378 | **0.299** | −0.0792 ± 0.0165 | **−20.9%** |
| 随机划分 K=30 | 0.387 | **0.294** | −0.0935 ± 0.0459 | **−24.1%** |
| 空间划分 K=30 | 0.283 | **0.250** | −0.0333 ± 0.0066 | **−11.8%** |

对照的另外三种半监督项（都写成回归形式），源域网格搜索选出的权重 λ：
**Π 一致性 0、VAT 0、流形正则 0.1**（微小且不稳定）—— 即"别用"。

> **一个容易踩的坑**：Pseudo-Label 直接用在**回归**上，梯度**恒等于 0**
> （因为目标就等于预测值本身）。必须改成**滞后 EMA 教师**才有效。
> 这一点在分类里被 argmax 的非线性掩盖了。

---

## 2. 文件说明（`ssl_maml/` 下共 10 个文件）

| 文件 | 内容 |
|---|---|
| `README.md` | 本文件 |
| `.gitignore` | 仓库根，已与上游 main 合并 |
| `ssl_maml/core.py` | 核心库：MLP 手写前反向、`inner_adapt`、MAML 二阶元梯度（有限差分 HVP）、Adam/SGD、kNN 邻接 |
| `ssl_maml/ssl.py` | 6 种半监督项：标签传播 / 标签扩散 / 原型 / 熵最小化 / 原型蒸馏 / LP 蒸馏 / 自训练 |
| `ssl_maml/tasks.py` | 任务采样器（含噪声维度与任务随机相位） |
| `ssl_maml/experiments/exp1_synth.py` | **主实验**：合成小样本分类，三种注入位置（内层 / 外层 / 适应之后）对比 |
| `ssl_maml/experiments/exp1b_alpha.py` | **机理实验**：`(α, m)` 扫描 + 冻结教师对照 |
| `ssl_maml/experiments/exp2_water.py` | 真实水数据：元训练 → 极少数标签监督适应 → 自训练 / 标签传播（三段式） |
| `ssl_maml/experiments/exp5_water_ssl.py` | 真实水数据：Pseudo-Label / Π 一致性 / VAT / 流形正则 四种 SSL 项的对比 |
| `ssl_maml/matio.py` | 纯 Python 的 MATLAB `.mat` 读取器（不依赖 scipy） |
| `ssl_maml/prepare_water_data.py` | 数据脚本：从 USGS 官方接口拉原始日值、校验成品文件 |

**依赖**：只需要 `numpy`。元学习、二阶梯度、半监督项、优化器、`.mat` 解析
全部手写实现，不依赖 PyTorch / TensorFlow / scipy / scikit-learn。

---

## 3. 怎么跑

```bash
pip install numpy                    # 唯一的依赖

# 不需要任何数据，直接跑出结论一、结论二（前者约 25 分钟，加 --quick 可几分钟跑通）
python ssl_maml/experiments/exp1_synth.py
python ssl_maml/experiments/exp1b_alpha.py

# 需要数据（见下一节），跑出结论三
python ssl_maml/experiments/exp2_water.py --quick
python ssl_maml/experiments/exp5_water_ssl.py --quick
```

结果 JSON 写在自动创建的 `results/ssl_maml/` 里，运行日志写在 `outputs_ssl_maml/`（该目录已 gitignore）；控制台也会打印汇总表。

`exp1_synth.py` 支持 `--kinds` 指定要跑哪些主干和方法，`--quick` 用少量迭代快速验证；
`exp5_water_ssl.py` 支持 `--split {random,spatial}` 和 `--K`。

---

## 4. 数据

**主实验 `exp1`/`exp1b` 用合成数据，不需要任何数据文件**；只有 `exp2`/`exp5` 需要下面这一份。

### 获取 `water_dataset.mat`

本仓库**不分发**该文件（`.gitignore` 已排除 `*.mat`）。两条路径：

**路径 A：直接取成品文件（最快）**

1. 群文件里搜索 `water_dataset.mat`（约 1.0 MB）。
2. 放到仓库根目录 `data/water_dataset.mat`，即 `few-shot-Core-Troop/data/water_dataset.mat`。
3. 用本节末尾的命令校验。

**路径 B：从 USGS 原始来源自己拉（可独立核查）**

```bash
python ssl_maml/prepare_water_data.py --download
```

把这份数据背后的 **USGS 官方日值**（4 个指标 × 日最大/最小/均值 × 37 个站）
拉到 `data/usgs_raw/`，并生成 `manifest.md` 记录每一条请求的完整 URL。

> **为什么脚本只拉原始数据、不直接生成 `water_dataset.mat`？**
> 成品是 2019 年那篇论文**整理过**的版本：37 个站里有 16 个的连续监测记录
> 2012–2015 年才开始；2015-10 之后**不存在一段 705 天里 37 个站每天都有数据**
> （最长完整区间只有 47 天）。所以论文一定做过缺测填补、或者挑的是非连续的天，
> 而这个规则没有公开；数据还做过按特征 min-max 归一化，用的哪段极值也没公开。
> 重建规则一旦猜错，产出的是一份**同名不同数**的数据，别人拿去跑实验会得到对不上的
> 结果，比不给脚本更糟。所以脚本只做两件能验证的事：拉原始数据、校验成品文件。

### 校验

```bash
python ssl_maml/prepare_water_data.py --verify
```

会逐项检查站点 ID、特征名与顺序、形状；通过时输出：

```
校验 data\water_dataset.mat
  站点 37，特征 11，训练 423 天，测试 282 天
结论: [OK] 文件正确
```

想直接看文件内容（纯 Python 的 `.mat` 读取器，不需要装 scipy）：

```bash
python ssl_maml/matio.py data/water_dataset.mat
```

应当打印出这些变量：

| 变量 | 形状 | 说明 |
|---|---|---|
| `X_tr` | cell 1×423 | 训练期每天一个 37×11 矩阵（站点 × 指标） |
| `X_te` | cell 1×282 | 测试期，同上 |
| `Y_tr` / `Y_te` | 37×423 / 37×282 | pH 目标 |
| `location_ids` | 37×1 | 监测站 ID |
| `features` | cell 1×11 | 11 个水质指标名 |
| `location_group` | cell 1×3 | 站点空间分组 |

合计 **37 站 × 705 天 × 11 指标**，看到这个规模就说明文件没问题。

### 出处与许可

> Liang Zhao, Olga Gkountouna, Dieter Pfoser.
> *Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial
> Topological Constraints.* **ACM TSAS** 5(3), Article 19, 2019.
> DOI: [10.1145/3339823](https://doi.org/10.1145/3339823)

数据源自 USGS 地表水监测记录，经该论文整理为 `water_dataset.mat`。
**数据集许可独立于本仓库**，请遵守其自身条款。

---

## 5. 另外两个理论结果

**VAT 的秩定理**：对 softmax 模型，$D_{KL}(p(y|x)\,\|\,p(y|x{+}r))$ 在 $r=0$ 处的 Hessian 满足

```
rank(H) = min(I, C − 1)
```

所以**二分类（C=2）时秩恒为 1，VAT 精确退化为无监督 FGSM**（沿决策边界法向），
标量回归同理（rank 1）。数值验证：秩恰为 1，特征向量与梯度 $|\cos| = 1.000000$，
相对误差 1e−12 量级。**VAT 只在输出维度 ≥ 2 时才真正有用。**

**元梯度正确性**：二阶元梯度与有限差分校验的一致性（相对误差 ~1.8e−5）。

（这两项的验证脚本不在本包的 10 个文件里，需要的话可以单独提供。）

---

## 6. 统计协议

- **配对差**：所有增益都是同一任务、同一随机种子下 `method − baseline`，报告均值 ± 标准误。
- **源域选参**：所有半监督权重都在**源域任务**上网格搜索，**完全不碰目标域标签**。

---

## 7. 引用与许可

```bibtex
@misc{fewshot-ssl-maml,
  title  = {Where should the semi-supervised signal go? Inner-loop vs. post-adaptation in MAML},
  author = {wenjunxiong-17},
  year   = {2025}
}
```

引用的论文（本仓库只引用、不分发 PDF 或全文）：
Finn et al. *MAML* (ICML 2017)；Sohn et al. *FixMatch* (NeurIPS 2020)；
Lee. *Pseudo-Label* (ICML 2013 Workshop)；Miyato et al. *VAT* (IEEE TPAMI 2019)；
Ren et al. *Meta-Learning for Semi-Supervised Few-Shot Classification* (ICLR 2018)；
Zhao et al. *ACM TSAS* 5(3), 2019。

**代码许可**：MIT。数据集与论文版权归各自作者。

---

## 8. 与仓库里其他脚本的关系

本模块是**重构版**。它把每一个半监督项统一成同一个可插拔接口

```python
def term(P) -> (loss: float, dP: list[(W, b)])
```

所以**同一个损失项可以不加修改地放到三个不同位置**（MAML 内循环 / 元训练外层 /
适应之后）——这正是本项目回答"半监督信号该放哪"这个科学问题所需要的能力。

仓库根目录下另有三个早期脚本 —— `fixmatch_mixup_ablation.py`、
`fixmatch_mixup_maml.py`、`task_pseudo_label.py`（PR #2 合入）—— 它们是
FixMatch + MixUp 方向的消融实现，接口与实验设置都与本模块不同。

**去重计划**：本模块通过评审后，将以**单独一个 PR** 把上述三个脚本标注为
legacy 并归档（移入归档目录 / 在文件头加 legacy 说明）。
**本 PR 不删除任何既有文件。**
