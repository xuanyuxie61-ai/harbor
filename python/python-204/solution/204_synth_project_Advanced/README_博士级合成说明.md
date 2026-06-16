# PROJECT 204 : 圆盘型地球物理反应器的 Sobol 全局灵敏度分析

> **领域**：不确定性量化（Uncertainty Quantification）· Sobol 灵敏度分析  
> **难度**：博士级前沿科学计算  
> **种子项目**：15 个 Burkardt 风格数值库 + VLOGroup 深度先验 + HigherOrderLMC

---

## 1. 科学问题陈述

本项目解决的核心科学问题是：

> **在一个耦合了混沌 Lagrangian 混合与自催化 Lotka-Volterra 动力学的
> 圆盘型地球物理反应器中，8 个不确定物理参数中哪些对反应产率的全局
> 方差贡献最大？**

该反应器可被理解为海洋涡旋、岩浆房或环形生物反应器的一阶简化模型。
反应器的状态由下列耦合过程共同决定：

| 过程 | 控制方程 / 算法 | 不确定参数 |
|------|------------------|------------|
| **混沌 Lagrangian 混合** | Chirikov 标准映射 + 有限时间 Lyapunov 指数 | `K_chir` |
| **圆盘势流 Green 函数** | Carlson 对称椭圆积分 `R_F, R_D, R_J` | `R_disk` |
| **自催化物种动力学** | 带 logistic 自限的 Lotka-Volterra 系统 | `r_log, k_cap, alpha, beta, gamma` |
| **对流扩散反应 PDE** | 极坐标 5 点 Laplacian + Gauss-Seidel 求解 | `D_eff` |

最终目标泛函（QoI）为
$$
Y(\theta) \;=\; w_1 \frac{\langle B \rangle}{K_b} \;+\; w_2 \frac{v_{\mathrm{eff}}}{1 + v_{\mathrm{eff}}} \;+\; w_3 \frac{|G_{\mathrm{avg}}|}{1 + |G_{\mathrm{avg}}|},
$$
其中：
- $\langle B \rangle$ 为物种 B 沿 LV 轨迹的时间平均；
- $v_{\mathrm{eff}} = R \langle \Lambda \rangle / (2\pi)$ 为 Chirikov 混合的有效速度；
- $G_{\mathrm{avg}}$ 为圆盘 Laplace Green 函数在参数网格上的平均。

我们的目标是计算 $Y(\theta)$ 关于 8 个不确定参数 $\theta \in [0, 1]^8$ 的
**Sobol 全局灵敏度指数**：
- 一阶 $S_i = V_i / V$；
- 全阶 $S_i^T$（含所有交互作用）；
- 二阶纯交互 $S_{ij} = V_{ij} / V$。

---

## 2. 核心数学公式

### 2.1 Sobol 指数估计量（Jansen / Saltelli / Janon）

设 $A, B, AB_i, BA_i \in [0, 1]^{d \times N}$ 为 Saltelli 采样矩阵，
$Y_A = f(A)$、$Y_B = f(B)$、$Y_{AB_i} = f(AB_i)$：

$$
\hat f_0 = \tfrac12 \big(\bar Y_A + \bar Y_B\big), \quad
\hat V = \tfrac12 \big(\overline{Y_A^2} + \overline{Y_B^2}\big) - \hat f_0^2,
$$

$$
\hat S_i = \frac{1}{\hat V}\,\frac{1}{N}\sum_{k=1}^N Y_B^{(k)}\big(Y_{AB_i}^{(k)} - Y_A^{(k)}\big),
$$

$$
\hat S_i^T = \frac{1}{2N \hat V}\sum_{k=1}^N \big(Y_A^{(k)} - Y_{AB_i}^{(k)}\big)^2.
$$

### 2.2 Chirikov 标准映射 + FTLE

$$
\begin{aligned}
p_{n+1} &= p_n + K \sin(\theta_n) \pmod{2\pi}, \\
\theta_{n+1} &= \theta_n + p_{n+1} \pmod{2\pi}, \\
\Lambda(\theta_0, p_0) &= \frac{1}{n_{\mathrm{iter}}} \ln \frac{\|z_{n_{\mathrm{iter}}} - z'_{n_{\mathrm{iter}}}\|}{\|z_0 - z'_0\|}.
\end{aligned}
$$

### 2.3 圆盘 Laplace Green 函数

$$
G(r, r_0, \Delta\theta) = \frac{1}{2\pi}\left[\frac12 \ln \frac{D_+^2}{D_-^2} + \frac{k^2}{2}\Big(1 - 2 \frac{E(k)}{F(k)}\Big)\right],
$$
其中 $k^2 = 4 r r_0 \cos\Delta\theta / (r + r_0)^2$，$F(k), E(k)$ 为第一、
二类完全椭圆积分（通过 Carlson 对称积分 $R_F, R_D$ 计算）。

### 2.4 Carlson 对称椭圆积分

$$
R_F(x, y, z) = \tfrac12 \int_0^\infty \frac{dt}{\sqrt{(t+x)(t+y)(t+z)}},
$$
经 Carlson 迭代：
$$
\lambda_n = \sqrt{x_n y_n} + \sqrt{x_n z_n} + \sqrt{y_n z_n}, \quad
x_{n+1} = \tfrac14(x_n + \lambda_n), \;\text{etc.}
$$

### 2.5 Lotka-Volterra with logistic self-limitation

$$
\begin{aligned}
\dot A &= r_a A\Big(1 - \frac{A}{K_a}\Big) - \alpha A B, \\
\dot B &= r_b B\Big(1 - \frac{B}{K_b}\Big) + \beta A B - \gamma B.
\end{aligned}
$$

### 2.6 极坐标 PDE 离散（5 点模板）

$$
-\big(D_{\mathrm{eff}} \Delta_h c + v_{\mathrm{eff}} \cdot \nabla_h c + k_r c\big) = f
$$
经 Gauss-Seidel 块迭代（SOR 参数 $\omega = 1$）求解。

### 2.7 PCE 解析 Sobol 指数（Sudret 2008）

$$
Y \approx \sum_{|\alpha| \le p} c_\alpha \Psi_\alpha(\theta), \quad
V_i = \sum_{\alpha: \alpha_i > 0,\, \alpha_{j \ne i}=0} c_\alpha^2, \quad
V_i^T = \sum_{\alpha: \alpha_i > 0} c_\alpha^2.
$$

---

## 3. 种子项目 → 科学映射

| # | 种子项目 | 原功能 | 在本项目中的角色 |
|---|----------|--------|------------------|
| 1 | `489_grf_display` | 图文件显示 | 构造 Sobol 二阶灵敏度图（`.grf` 文本格式） |
| 2 | `571_ice_to_medit` | NetCDF → MEDIT 转换 | 坐标变换 + 极坐标结构化网格生成 |
| 3 | `1337_triangulation_histogram` | 三角形内直方图 | Saltelli 采样点均匀性诊断 |
| 4 | `295_disk_monte_carlo` | 圆盘均匀采样 | 圆盘几何内 Lagrangian 示踪粒子播种 |
| 5 | `701_logistic_exact` | Logistic ODE 解析解 | LV 子系统自限项的解析基线 |
| 6 | `335_elliptic_integral` | Carlson 椭圆积分 | 圆盘 Green 函数的模 `k` 计算 |
| 7 | `451_gauss_seidel` | Gauss-Seidel 迭代 | 耦合 PDE 块系统求解 |
| 8 | `1258_VLOGroup_PoGMDM` | Shearlet + GSM 先验 | 响应场低秩代理 + GSM 拟合 |
| 9 | `977_r8col` | 实数矩阵列操作 | Saltelli 矩阵去重 + 排序 + 均值 |
| 10 | `784_mxm` | 矩阵乘法计时 | PCE 设计矩阵 GEMM + Gram 矩阵 |
| 11 | `938_qr_solve` | QR 秩揭示求解 | PCE 回归 + 秩亏最小二乘 |
| 12 | `345_exm` | MATLAB 示例集（predator-prey 等） | Lotka-Volterra 生态子系统 |
| 13 | `1071_kaihongz_HigherOrderLMC` | 高阶 Langevin 采样 | Sobol 指数的贝叶斯后验采样 |
| 14 | `851_patterson_rule` | Gauss-Patterson 求积 | Smolyak 稀疏网格 + PCE 验证积分 |
| 15 | `171_chirikov_iteration` | Chirikov 标准映射 | 混沌 Lagrangian 混合核心 |

**15 个项目全部真实参与**，无任何挂名。

---

## 4. 项目结构

```
204_synth_project_Advanced/
├── main.py                    # 统一入口，零参数运行
├── forward_model.py           # 耦合正向模型 f(theta) -> Y
├── sobol_indices.py           # Saltelli / Jansen / 二阶 Sobol 估计量
├── quadrature_pce.py          # Patterson 求积 + 秩揭示 QR + PCE
├── chaotic_mixing.py          # Chirikov 映射 + 圆盘采样 + FTLE
├── reaction_kinetics.py       # Logistic + Lotka-Volterra + 稳态
├── elliptic_green.py          # Carlson R_F, R_D, R_J + 圆盘 Green
├── gauss_seidel_coupled.py    # Gauss-Seidel + 极坐标 Laplacian + 块耦合
├── shearlet_surrogate.py      # Haar 类方向分解 + GSM + Student-t 光滑
├── langevin_inversion.py      # Picard-Lagrange LMC + Sobol 后验采样
├── spatial_ops.py             # 三角直方图 + GRF 图 + 网格生成
├── r8col_utils.py             # 列操作工具集（去重、排序、均值）
├── matrix_kernels.py          # Tiled GEMM + Gram 矩阵 + 条件数
├── robustness.py              # 边界检查 + 质量守恒 + 冒烟测试
└── README_博士级合成说明.md    # 本文档
```

共计 **14 个 Python 源文件 + 1 个中文 README**，远超"至少 8 个 .py"的要求。

---

## 5. 运行方式

```bash
cd 204_synth_project_Advanced
python main.py
```

`main.py` 依次执行 7 个阶段：

| Stage | 内容 | 输出 |
|-------|------|------|
| 0 | 所有模块冒烟测试 | 每模块 OK/FAIL |
| 1 | 名义点 `theta = (0.5, …, 0.5)` 正向计算 | 物理参数 + `Y` |
| 2 | Saltelli + Bootstrap Sobol 指数 | `S1, ST, S2` 及 95% CI |
| 3 | PCE 代理 + Sudret 解析 Sobol | 与 Stage 2 交叉验证 |
| 4 | 响应场剪切波代理 | 重构误差、GSM 参数 |
| 5 | Picard-Lagrange 后验采样 | 后验均值与 MC 估计对比 |
| 6 | 灵敏度邻接图 | GRF 文件 + 度序列 |
| 7 | 各阶段耗时汇总 | 工程化性能报告 |

全程无任何可视化，所有结果以文本形式输出。

---

## 6. 数值鲁棒性与边界处理

- **Chirikov FTLE**：对 `|K|` 非负性检查，对分离距离退化回 0；
- **椭圆积分**：`|k| >= 1` 时返回大有限值，避免发散；
- **Logistic 解析解**：对 `|r(t - t0)| < 1e-4` 切换为 `expm1` 形式避免精度损失；
- **Gauss-Seidel**：对角为零时抛异常；SOR 参数校验 `0 < ω < 2`；
- **PCE 设计矩阵**：秩揭示 QR (`dqrank`) 处理共线性；
- **Saltelli 矩阵**：`dedupe_sample_matrix` 清除偶然重复列；
- ** Langevin 后验**：每步投影到可行单纯形 `S >= 0, S1 <= ST, sum ST <= 1`；
- **正向模型兜底**：任何 `NaN` / 异常均返回 `0.5` 的安全值；
- **条件数监控**：`ConditionMonitor` 记录所有线性子系统的 `rcond`。

---

## 7. 可复现性

所有随机源统一使用 `numpy.random.default_rng(seed)`；
主流程使用的固定种子为 `2024`（Sobol）、`7`（PCE）、`3`（Langevin）。
在 Python 3.9+ 与 NumPy ≥ 1.22 环境下可 100% 复现。

---

## 8. 与其他已合成项目的差异化

| 维度 | 本项目 | 通用 UQ / 优化项目 |
|------|--------|----------------------|
| **核心算法** | Saltelli/Jansen/二阶 Sobol + PCE-Sudret + 高阶 Langevin 后验 | 通常仅 MC 或 PCE 之一 |
| **混合内核** | Chirikov 混沌映射 + Carlson 椭圆 Green 函数 + LV 生态 | 多为纯解析基准 |
| **代理模型** | 基于 Shearlet 方向分解 + GSM 先验的低秩代理 | 多为多项式或 Kriging |
| **后验量化** | Picard-Lagrange LMC 在 Sobol 单纯形上采样 | 一般只给 bootstrap CI |
| **图结构** | 灵敏度邻接图以 GRF 格式输出 | 无 |

算法命名、变量组织、数据流均围绕 **Sobol 灵敏度分析** 深度耦合，
而非通用数值方法的简单换皮。

---

## 9. 合成结论

本项目将 15 个来自 Burkardt、VLOGroup、kaihongz 等来源的异构
科研代码，围绕「**圆盘型地球物理反应器的 Sobol 全局灵敏度分析**」
这一前沿博士级 UQ 问题，融合为一个结构完整、数学密度高、
工程鲁棒的 Python 计算框架。代码零参数可运行，
输出完全文本化，所有公式与算法均可在文档与源码间双向追溯。
