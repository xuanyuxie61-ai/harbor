# 格点 QCD 强子谱学 — 博士级合成说明

## 一、项目概述

本项目将 **15 个科研代码种子项目** 的核心算法融合重构，面向 **粒子物理：格点 QCD 强子谱学** 这一前沿博士级科学问题，构建了一套可运行的 Python 科研计算流水线。

项目模拟了从规范场生成、Wilson 梯度流平滑、Wilson-Dirac 传播子求解、强子关联函数构造，到变分法能谱提取、手征动力学分析与重整化群流计算的完整流程。

---

## 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 合成后角色 |
|------|--------|----------|-----------|
| 1 | 274_dg1d_maxwell | 1D 间断 Galerkin (DG) 谱方法 + LSERK 时间推进 | `gauge_dg_update.py`：规范场的 fictitious-time DG 谱演化，用于冷却预处理 |
| 2 | 209_conte_deboor | Muller 复根法、Hermite 三次样条 (calccf) | `variational_spectrum.py`：关联函数样条平滑、特征多项式求根 |
| 3 | 460_gegenbauer_cc | Gegenbauer-Clenshaw-Curtis 积分 | `quadrature_physics.py`：动量空间角向积分、衰变常数计算 |
| 4 | 826_ode_euler_backward | 后向 Euler 隐式 ODE 求解 | `wilson_flow.py`：梯度流方程的隐式时间推进 |
| 5 | 615_kdv_exact | KdV 孤子 sech² 精确解 | `correlator_builder.py`：孤子波包试探函数用于重子关联函数增强 |
| 6 | 1127_sphere_stereograph | 球极投影 | `lattice_gauge.py`：SU(2) 群元的球极投影参数化 |
| 7 | 355_fd1d_advection_lax_wendroff | Lax-Wendroff 平流格式 | `fermion_solver.py`：Wilson-Dirac 算符 hopping 项的二阶离散 |
| 8 | 765_midpoint_adaptive | 自适应隐式中点法 | `wilson_flow.py`：梯度流的自适应步长控制 |
| 9 | 1266_toms178 | Hooke-Jeeves 直接搜索优化 | `variational_spectrum.py`：smearing 参数优化 |
| 10 | 1386_vanderpol_ode | Van der Pol 非线性振子 | `chiral_dynamics.py`：手征场的非线性阻尼振荡模型 |
| 11 | 091_biochemical_nonlinear_ode | 反应网络与化学计量矩阵 | `chiral_dynamics.py` 与 `reaction_rg.py`：夸克-介子耦合动力学与 RG 流网络 |
| 12 | 004_alpert_rule | Alpert 奇异/振荡积分规则 | `quadrature_physics.py`：传播子奇异核的积分处理 |
| 13 | 632_lagrange | Lagrange 插值基函数 | `correlator_builder.py`：关联函数在非格点位置的插值重构 |
| 14 | 670_levy_dragon_chaos | 迭代函数系统 (IFS) | `lattice_gauge.py`：规范场构型的混沌热化噪声注入 |
| 15 | 991_r8pp | 对称正定打包矩阵 Cholesky 分解 | `matrix_algebra.py`：关联矩阵预处理与高斯采样 |

---

## 三、新增数学物理模型与核心公式

### 3.1 Wilson 纯规范作用量

格点上的 SU(N_c) 规范场由链路变量 $U_\mu(x) \in SU(N_c)$ 描述。Wilson plaquette action 为：

$$
S_G = \frac{\beta}{N_c} \sum_{x,\mu<\nu} \text{Re}\,\text{Tr}\left[ \mathbf{1} - P_{\mu\nu}(x) \right]
$$

其中 plaquette 定义为：

$$
P_{\mu\nu}(x) = U_\mu(x) \, U_\nu(x+\hat\mu) \, U_\mu^\dagger(x+\hat\nu) \, U_\nu^\dagger(x)
$$

### 3.2 Wilson-Dirac 算符

Wilson-Dirac 算符在格点上离散为：

$$
D_w(x,y) = \frac{1}{\kappa} \delta_{xy} - \frac{1}{2} \sum_\mu \left[ (1-\gamma_\mu) U_\mu(x) \delta_{x+\hat\mu,y} + (1+\gamma_\mu) U_\mu^\dagger(x-\hat\mu) \delta_{x-\hat\mu,y} \right]
$$

其中跳跃参数 $\kappa$ 与裸夸克质量 $m_0$ 的关系为：

$$
\kappa = \frac{1}{2m_0 a + 8}
$$

### 3.3 Wilson 梯度流

梯度流方程定义为：

$$
\dot{V}_\mu(x,t) = -g_0^2 \, \partial_{V_\mu} S_G[V(t)] = F_\mu(x,t) \, V_\mu(x,t)
$$

其中 $F_\mu(x,t)$ 为来自 staples 的 su(2) 代数力。本项目中使用后向 Euler 与自适应隐式中点法进行时间推进：

$$
Y_m = Y_n + \theta \Delta t \, F(Y_m) Y_m, \quad Y_{n+1} = \frac{1}{\theta} Y_m + \left(1 - \frac{1}{\theta}\right) Y_n
$$

### 3.4 强子关联函数与有效质量

两点关联函数定义为：

$$
C_H(t) = \sum_{\mathbf{x}} \langle O_H(\mathbf{x},t) \, O_H^\dagger(\mathbf{0},0) \rangle
$$

对于大 $t$，关联函数呈指数衰减：

$$
C_H(t) \approx Z_H \, e^{-m_H t}
$$

有效质量由对数比给出：

$$
m_H^{\text{eff}}(t) = \frac{1}{\Delta t} \log\frac{C_H(t)}{C_H(t+\Delta t)}
$$

### 3.5 变分广义本征值方法 (GEVP)

为了提取激发态，构造 $N_{op} \times N_{op}$ 关联矩阵 $C_{ij}(t)$，求解广义本征值问题：

$$
C(t) \, v_n = \lambda_n(t,t_0) \, C(t_0) \, v_n
$$

其中本征值满足：

$$
\lambda_n(t,t_0) = A_n \, e^{-m_n (t-t_0)} + O\left(e^{-m_{n+1}(t-t_0)}\right)
$$

### 3.6 KdV 孤子波包试探函数

借鉴 KdV 方程的孤子解，构造重子关联函数的试探波包：

$$
\psi_{\text{solitonic}}(x,t;v) = -\frac{v}{2} \, \text{sech}^2\!\left( \frac{\sqrt{v}}{2}(x - vt - a) \right)
$$

### 3.7 Gegenbauer-Clenshaw-Curtis 积分

对于具有 Gegenbauer 权 $(1-x^2)^{\lambda-1/2}$ 的积分：

$$
I_\lambda[f] = \int_{-1}^{1} (1-x^2)^{\lambda-1/2} f(x)\,dx \approx \frac{\Gamma(\lambda+1/2)\sqrt{\pi}}{\Gamma(\lambda+1)} \, u_0
$$

其中 $u_r$ 由 Chebyshev 偶系数递推确定。

### 3.8 Gell-Mann–Oakes–Renner (GMOR) 关系

手征对称性自发破缺导致 Goldstone 玻色子（π 介子），其质量满足：

$$
m_\pi^2 f_\pi^2 = (m_u + m_d) \, B_0 \, f_0^2
$$

### 3.9 重整化群流方程

SU(3) 的 β 函数（两圈）：

$$
\mu \frac{dg}{d\mu} = \beta(g) = -\frac{\beta_0 \, g^3}{16\pi^2} - \frac{\beta_1 \, g^5}{256\pi^4}
$$

其中：

$$
\beta_0 = 11 - \frac{2}{3} N_f, \quad \beta_1 = 102 - \frac{38}{3} N_f
$$

跑动耦合的单圈解析解：

$$
\alpha_s(\mu) = \frac{4\pi}{\beta_0 \, \log(\mu^2/\Lambda_{QCD}^2)}
$$

### 3.10 SU(2) 球极投影

SU(2) 群元参数化为 $U = a_0 \mathbf{I} + i \mathbf{a} \cdot \boldsymbol{\sigma}$，球极投影映射为：

$$
q_i = \frac{a_i}{1+a_0}, \quad a_0 = \frac{1-|q|^2}{1+|q|^2}, \quad a_i = \frac{2q_i}{1+|q|^2}
$$

---

## 四、文件结构

```
034_synth_project/
├── main.py                      # 统一入口，零参数运行
├── matrix_algebra.py            # SPD 打包矩阵 Cholesky 分解 (991_r8pp)
├── lattice_gauge.py             # 格点几何、SU(2) 规范场、IFS 热化、球极投影
├── gauge_dg_update.py           # DG 谱方法规范场演化 (274_dg1d_maxwell)
├── wilson_flow.py               # Wilson 梯度流：后向 Euler + 自适应中点法
├── fermion_solver.py            # Wilson-Dirac 算符 + CG 传播子求解
├── correlator_builder.py        # 强子关联函数、Lagrange 插值、KdV 孤子增强
├── variational_spectrum.py      # GEVP、Muller 求根、样条、Hooke-Jeeves 优化
├── quadrature_physics.py        # Gegenbauer 与 Alpert 高阶积分
├── chiral_dynamics.py           # 手征振子、夸克-介子反应网络
├── reaction_rg.py               # RG 流方程、跑动耦合、多能标反应网络
└── README_博士级合成说明.md      # 本文档
```

---

## 五、运行方式

在项目目录下执行：

```bash
python main.py
```

无需任何命令行参数。程序将自动完成：
1. 规范场构型生成与 IFS 混沌热化
2. DG 谱方法 fictitious-time 演化
3. Wilson 梯度流平滑
4. Wilson-Dirac 传播子 CG 求解
5. π 介子与核子关联函数构造
6. 变分 GEVP 能谱提取与 smearing 参数优化
7. 关联函数样条插值平滑
8. Gegenbauer / Alpert 高阶积分（衰变常数、自能）
9. 手征动力学与 Goldstone 玻色子分析
10. 重整化群流方程积分

---

## 六、科学问题说明

### 6.1 本项目解决的科学问题

格点 QCD（Lattice Quantum Chromodynamics）是从第一性原理出发研究强相互作用非微扰性质的标准方法。本项目聚焦于 **强子谱学**（Hadron Spectroscopy），即：

- 从夸克和胶子的基本自由度出发，数值计算强子（如 π 介子、核子）的质量；
- 通过梯度流技术抑制 UV 涨落、改善算符与基态的重叠；
- 利用变分方法（GEVP）同时提取基态与激发态；
- 通过手征有效场论（ChPT）验证 GMOR 关系；
- 追踪耦合常数随能标的 RG 演化。

### 6.2 数值方法的博士级难度

- **高阶谱方法**：DG 间断 Galerkin 方法具有谱精度收敛率；
- **隐式刚性 ODE 求解**：Wilson 梯度流为刚性方程，采用后向 Euler 与自适应中点法；
- **大型稀疏线性系统**：Wilson-Dirac 算符在 $4^3 \times 8$ 格点上为 $512 \times 2 = 1024$ 维复线性系统，使用 CG 迭代求解；
- **广义本征值问题**：变分法涉及非对称矩阵的广义本征值分解；
- **奇异积分处理**：Alpert 规则处理传播子在零动量处的红外奇异性；
- **多尺度耦合分析**：RG 流方程与多能标反应网络描述耦合常数的跨尺度演化。

---

## 七、边界处理与数值鲁棒性

1. **周期性/反周期性边界条件**：空间方向周期，时间方向反周期（费米子）；
2. **步长限制**：DG 演化中限制投影坐标变化量 $|\Delta q| < 0.5$ 以保持 SU(2) 稳定性；
3. **正则化**：GEVP 中关联矩阵添加 $10^{-10} \mathbf{I}$ 避免奇异；
4. **阻尼迭代**：Wilson flow 的后向 Euler 使用阻尼因子 0.5 保证定点收敛；
5. **CG 安全退出**：当 $p^\dagger A p \approx 0$ 或达到最大迭代次数时自动终止；
6. **除零保护**：所有分母均添加微小正数（如 `1e-10`、`1e-15`）。

---

## 八、作者与致谢

本项目基于 15 个科研代码种子项目的核心算法，经深度重构与科学问题映射而成。
合成方向：**粒子物理 — 格点 QCD 强子谱学**。
