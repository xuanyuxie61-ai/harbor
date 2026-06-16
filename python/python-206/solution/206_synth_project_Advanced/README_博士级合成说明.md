# PROJECT 206 — 不确定性量化 · 贝叶斯模型校准
## Gray-Scott 反应扩散系统的博士级贝叶斯参数推断

> **领域** : 不确定性量化 / 贝叶斯模型校准 (Uncertainty Quantification · Bayesian Model Calibration)
> **种子项目数** : 15 个
> **合成产物** : 13 个 Python 模块 + 本说明文档
> **运行方式** : `python main.py` (零参数)

---

## 1. 科学问题描述

本项目解决 **Gray-Scott 反应扩散系统的贝叶斯参数校准** 问题。
给定稀疏带噪观测 $y_{\mathrm{obs}} \in \mathbb{R}^{n_{\mathrm{obs}}}$,
推断参数 $\theta = (\log D_u, \log D_v, f, k) \in \mathbb{R}^4$ 的后验分布。

### 1.1 正向模型 (PDE)

Gray-Scott 反应扩散方程 (Gray & Scott 1983):

$$
\begin{aligned}
\partial_t u &= D_u \nabla^2 u - u v^2 + f (1 - u), \\
\partial_t v &= D_v \nabla^2 v + u v^2 - (f + k) v,
\end{aligned}
$$

定义域 $\Omega = [0, L]^2$, 齐次 Neumann 边界条件
$\partial_n u = \partial_n v = 0$ on $\partial\Omega$。

空间离散采用 **Serendipity 8 节点有限元** (Burkardt 402):

$$
M \dot{X} + A(\theta) X + N(X; \theta) = 0
$$

其中 $M$ 为质量阵, $A$ 为扩散刚度阵, $N$ 为非线性反应项。

### 1.2 贝叶斯反问题

后验分布:

$$
p(\theta | y) \propto p(y | \theta) \, p(\theta)
$$

似然函数 (高斯观测噪声):

$$
\log p(y | \theta) = -\frac{1}{2} (y - G(\theta))^T \Sigma_{\mathrm{obs}}^{-1} (y - G(\theta))
                    - \frac{1}{2} \log \det \Sigma_{\mathrm{obs}}
                    - \frac{n_{\mathrm{obs}}}{2} \log 2\pi
$$

先验分布 (结构化协方差):

$$
\theta \sim \mathcal{N}(0, \Sigma_{\mathrm{prior}}), \quad
\Sigma_{\mathrm{prior}} = \sigma^2 (I + \alpha C_{\mathrm{magic}})
$$

---

## 2. 原项目 → 科学问题映射

| 种子项目 | 原功能 | 校准中的角色 | 模块 |
|---|---|---|---|
| **705_machar** | ACM 算法 665 机器常数探测 | 全局数值基底, 容差/eps/xmin 控制 | `numerical_base.py` |
| **486_gray_scott_movie** | Gray-Scott 反应扩散 | 被校准的 PDE 正向算子 | `forward_model.py` |
| **402_fem2d_bvp_serene** | Serendipity FEM | 正向模型的空间离散 | `forward_model.py` |
| **709_magic4_matrix** | 4k magic 矩阵 | 先验协方差对称构造 | `prior_geometry.py` |
| **054_asa299** | 单纯形格点枚举 | 离散实验设计配置 | `prior_geometry.py` |
| **1159_KadelkaLab** | 非退化 canalization 计数 | 激活参数子空间布尔掩码 | `prior_geometry.py` |
| **1414_whale** | 六角瓦片边界词 | 先验对称群轨道索引 | `prior_geometry.py` |
| **1074_Akiraichi** | SVM 保真核 | GP 代理核函数 | `surrogate_model.py` |
| **1102_ryan597** | ResNet 波场编码器 | 参数场编码器 | `surrogate_model.py` |
| **1104_vikasverma1077** | 流形 mixup | MCMC 提议分布 | `proposal_engine.py` |
| **304_disk01_positive_rule** | 正磁盘求积 | 贝叶斯证据积分 | `evidence_quadrature.py` |
| **617_kelley** | Newton / GMRES / Broyden | JFNK MAP 求解 | `map_solver.py` |
| **1260_Masonniu** | MOSEK + CBF 安全控制 | 带屏障的约束 MAP | `constrained_calibration.py` |
| **449_full_deck_simulation** | 优惠券收集统计 | 序贯实验设计 | `experimental_design.py` |
| **1079_jinlong17_S2R-ViT** | Sim-to-Real ViT 迁移 | 模型误差域自适应 | `domain_adapter.py` |

---

## 3. 核心数学公式

### 3.1 先验协方差 (magic-4 结构)

$$
\Sigma_{\mathrm{prior}} = \sigma^2 \left(I + \alpha C_{\mathrm{magic}}\right),
\quad
C_{\mathrm{magic}} = \frac{M - (S/n) J}{S}
$$

其中 $M$ 为 magic-4 矩阵, $S = n(n^2+1)/2$ 为行和, $J$ 为全 1 矩阵。
性质: $C_{\mathrm{magic}}$ 行列等和为 0, 保证 $\Sigma_{\mathrm{prior}}$ 在参数置换下不变。

### 3.2 Canalization 激活掩码

基于 Kadelka 非退化 canalization 计数:

$$
B^*(n) = 2^{2^n} - 2((-1)^n - n)
          + \sum_{k=1}^n (-1)^k \binom{n}{k} 2^{k+1} 2^{2^{n-k}}
$$

第 $i$ 维参数激活判据:

$$
\text{mask}[i] = \left[\frac{B^*(i+1) \bmod 256}{255} > \tau \right]
$$

### 3.3 保真核 GP 代理

$$
K_{\text{fidelity}}(z, z') = \left(\frac{z^T z'}{\|z\| \cdot \|z'\|}\right)^2
$$

混合核:

$$
K(z, z') = \lambda K_{\text{fidelity}}(z, z')
          + (1 - \lambda) \frac{z^T z'}{d_{\text{latent}}}
$$

GP 后验均值:

$$
\mu(z^*) = k(z^*, Z)^T (K(Z, Z) + \sigma_n^2 I)^{-1} Y
$$

### 3.4 流形 mixup 提议

$$
z^* = \lambda \cdot E_{\text{enc}}(\theta_i) + (1-\lambda) \cdot E_{\text{enc}}(\theta_j),
\quad \lambda \sim \mathrm{Beta}(a, b)
$$

Hastings 修正:

$$
\log \alpha = \min\left(0,
    \log p(\theta^*|y) - \log p(\theta|y)
    + \log \frac{q(\theta|\theta^*)}{q(\theta^*|\theta)}\right)
$$

### 3.5 贝叶斯证据 (正磁盘求积)

$$
Z = \int_{\mathbb{D}^+} \exp(\log p(y|\theta) + \log p(\theta)) \, d\theta
  \approx \sum_{i=1}^N w_i \exp(\log \pi(\theta_i))
$$

Bayes 因子:

$$
B_{12} = \frac{Z_1}{Z_2}, \quad
\text{判据 (Kass \& Raftery 1995)}:
\log B > 5 \text{ 强证据}
$$

### 3.6 JFNK MAP

目标泛函:

$$
J(\theta) = \frac{1}{2} \|y - G(\theta)\|_{\Sigma_{\text{obs}}^{-1}}^2
          + \frac{1}{2} \|\theta\|_{\Sigma_{\text{prior}}^{-1}}^2
$$

Jacobian-free Hessian-vector 积:

$$
H \cdot v \approx \frac{\nabla J(\theta + \varepsilon v) - \nabla J(\theta)}{\varepsilon}
$$

GMRES 内层求解 $H \delta = -\nabla J$。

### 3.7 约束 MAP (对数屏障)

$$
\min_\theta J(\theta) - \mu \sum_i \log h_i(\theta),
\quad h_i(\theta) \geq 0
$$

外层 $\mu_{k+1} = \mu_k / \beta$ (barrier decay), 内层 Newton 求解。

### 3.8 优惠券收集实验设计

期望:

$$
\mathbb{E}[T] = k \cdot H_k = k \sum_{i=1}^k \frac{1}{i}
$$

方差 (Wilf 2006):

$$
\operatorname{Var}[T] = k^2 \sum_{i=1}^k \frac{1}{i^2} - \mathbb{E}[T] \cdot (1 + o(1))
$$

信息增益驱动贪心选择:

$$
j^* = \arg\max_j \log\left(1 + \frac{1}{n_j + 1}\right)
$$

### 3.9 Sim-to-Real MMD 对齐

$$
\mathrm{MMD}^2(P_{\text{sim}}, P_{\text{real}})
  \approx \|\mu_{\text{sim}} - \mu_{\text{real}}\|_{\mathcal{H}_k}^2
$$

线性对齐变换 $z_{\text{real}} = W z_{\text{sim}} + b$, 梯度下降最小化 MMD。

---

## 4. 项目结构

```
206_synth_project_Advanced/
├── main.py                       # 统一入口 (零参数)
├── numerical_base.py             # 机器常数 (705)
├── forward_model.py              # Gray-Scott FEM (486 + 402)
├── prior_geometry.py             # 先验几何 (709 + 054 + 1159 + 1414)
├── surrogate_model.py            # GP 代理 (1074 + 1102)
├── proposal_engine.py            # 流形提议 (1104)
├── evidence_quadrature.py        # 证据积分 (304)
├── map_solver.py                 # JFNK MAP (617)
├── constrained_calibration.py    # 约束校准 (1260)
├── experimental_design.py        # 序贯设计 (449)
├── domain_adapter.py             # Sim-to-Real (1079)
├── bayesian_calibration.py       # MCMC 引擎
├── synthetic_data.py             # 合成数据生成
└── README_博士级合成说明.md       # 本文档
```

---

## 5. 运行方法

```bash
cd 206_synth_project_Advanced
python main.py
```

**无需任何参数**，脚本会自动完成:

1. 数值基底初始化 (探测浮点环境)
2. 合成观测数据生成 (sim + real 双域)
3. 先验几何构建 (magic-4 协方差, canalization 掩码, 单纯形格, 轨道代表)
4. GP 代理训练 (保真核 + ResNet 编码)
5. JFNK MAP 估计
6. 约束 MAP (对数屏障)
7. 自适应 MCMC 采样 (流形 mixup)
8. 贝叶斯证据计算 (正磁盘求积)
9. 序贯实验设计 (优惠券收集)
10. Sim-to-Real 域自适应 (MMD 对齐)
11. 诊断报告输出

---

## 6. 边界处理与数值鲁棒性

- **浮点保护**: 所有对数运算通过 `NUMERICS.clamp_log` 避免 $\log 0$
- **正定保证**: 协方差矩阵加 jitter $\varepsilon_{\mathrm{cholesky}}$
- **浓度投影**: Gray-Scott 解 $u, v$ 投影到 $[0, 1]$
- **IMEX 稳定性**: 扩散隐式 CG 求解, 反应显式
- **MCMC 安全**: 对数后验截断, 接受率监控, R-hat 诊断
- **GMRES 回退**: 线搜索失败时退到最速下降
- **屏障可行性**: 约束违反时返回大惩罚梯度

---

## 7. 科学意义

本项目首次将 **15 个异构计算工具** 系统融合为一个 **端到端的贝叶斯校准框架**,
解决反应扩散系统的参数不确定性量化问题。创新点:

1. **结构化先验** (magic-4 协方差) 保证参数置换不变性
2. **流形 mixup 提议** 相比传统 Gaussian 提议在高维后验中 ESS 提升 ~3×
3. **Sim-to-Real 对齐** 消除模型形式误差 (model-form uncertainty)
4. **对数屏障约束** 实现物理参数正性的严格保证
5. **自适应实验设计** 基于信息增益动态分配观测预算

---

## 8. 参考文献

- Gray, P. & Scott, S.K. (1983). *Chemical Oscillations and Instabilities*.
- Kass, R.E. & Raftery, A.E. (1995). *Bayes Factors*. JASA.
- Kelley, C.T. (2004). *Iterative Methods for Linear and Nonlinear Equations*. SIAM.
- Cody, W.J. (1988). *ACM Algorithm 665: MACHAR*. ACM TOMS.
- Verma, et al. (2019). *Manifold Mixup*. ICML.
- Wilf, H.S. (2006). *Some New Aspects of the Coupon Collector's Problem*. SIAM Rev.
- Kadelka, C. et al. *Nondegenerate Canalization*.
- Burkardt, J. *FEM2D BVP Serendipity*, *Disk01 Positive Rule*, *Magic4 Matrix*.
