# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：多因子 HJM 利率期限结构模拟平台

本项目基于 Heath‑Jarrow‑Morton (HJM) 框架对利率期限结构进行仿真与不确定性量化。入口文件 `main.py` 负责编排整体流程；其余 Python 文件各自承担独立的计算模块。在后续 benchmark 中，仅保留 `main.py`，要求根据本描述重新实现所缺失的模块文件。

---

## 文件清单与职责概要

| 文件名 | 主要职责 |
|--------|----------|
| `main.py` | 零参数入口，调用所有模块完成从市场校准、网格生成、动力学模拟到债券定价和不确定性量化的完整流程，并输出性能指标。 |
| `fem_maturity_grid.py` | 二维非结构化三角网格生成（矩形域）、二次三角形元（T6）基函数与导数、刚度/质量矩阵组装、Dirichlet 边界处理、泊松方程与热方程求解。 |
| `hjm_model.py` | 多因子 HJM 模型类，负责波动率结构计算、无套利漂移项、随机动力学状态推进以及一条完整的前向利率曲线路径模拟。 |
| `polynomial_chaos_uq.py` | 概率论 Hermite 多项式及其乘积多项式、多维指标集合生成、多项式混沌展开求值与 Sobol 敏感性分析。 |
| `sparse_linear_algebra.py` | 三元组格式（ST）稀疏矩阵的读/写、带宽估计、稀疏矩阵‑向量乘以及稀疏线性系统求解器封装。 |
| `special_functions.py` | 对数正态分布 PDF / CDF / 逆 CDF 和采样、高精度标准正态逆 CDF（Wichura 算法）、Lambert W 函数近似（上下分支）。 |
| `stochastic_dynamics.py` | Lorenz‑96 混沌系统、Duffing 振子、Oregonator 反应系统的参数与右端项计算，以及将三个系统状态投影为 HJM 多因子波动率驱动项的耦合机制。 |
| `term_structure_pde.py` | HJM 漂移项的数值积分、前向利率 PDE 的空间半离散化（有限差分）、债券价格与零息收益率的计算、完整期限结构 PDE 的隐式求解。 |
| `time_stepping.py` | 显式 RK2、RK3 单步、嵌入式 RK23 积分器（带误差估计）、后向 Euler 单步以及自适应步长 RK23 积分器。 |
| `yield_curve_calibration.py` | Shepard 二维逆距离加权插值、Horner 多项式求值、多项式最小二乘拟合及收益率曲线特征（峰、谷、拐点）提取。 |

---

## 模块边界与核心接口

### 1. 有限元网格与空间离散 (`fem_maturity_grid.py`)

- 生成矩形域上的节点坐标和六节点二次三角形单元（T6），节点编号约定为角点、边中点、内部中心点按规则排列。
- 提供三角形面积计算、参考三角形到物理三角形的映射。
- 二次基函数及其梯度基于面积坐标实现，局部编号为 1..6。
- 提供三角形高斯积分规则（1 点或 3 点）。
- `assemble_fem_matrices` 组装全局刚度矩阵 A 和质量矩阵 M（稀疏 COO→CSR），支持可选的零阶反应项系数函数。
- `apply_dirichlet_bc` 识别矩形边界节点，将矩阵对应行置为恒等式，右端项置为边界值。
- `solve_poisson_fem` 利用组装矩阵、右端项函数和边界条件求解泊松方程 `-Δu = f`。
- `solve_heat_fem` 使用后向 Euler 格式 `(M + dt*A) u_{n+1} = M u_n + dt*f` 求解热方程，返回最终状态和历史轨迹。

### 2. HJM 多因子模型 (`hjm_model.py`)

- 类 `HJMMultiFactorModel` 封装模型参数：因子数、基准波动率、指数衰减参数、Lorenz‑96、Duffing、Oregonator 的系统参数、多项式混沌展开阶数与维度。
- 构造函数计算各动力学系统的默认初值，并生成多维混沌指标集合。
- `volatility_structure` 根据当前时间和剩余期限以及三个动力学状态，计算多因子波动率向量。结构包含：指数衰减因子、斜率因子、混沌耦合因子，并加入非负截断。
- `drift_term` 通过 `term_structure_pde.musiela_drift` 计算 HJM 无套利漂移。
- `evolve_stochastic_dynamics` 使用 RK3 步进分别推进 Lorenz‑96、Duffing 和 Oregonator 系统一个时间步，对 Oregonator 采用子步以保证稳定性。
- `simulate_path` 在给定的期限网格上，逐步更新随机动力学和用有限差分隐式格式求解前向利率 PDE（调用 `term_structure_pde.forward_rate_pde_rhs`），返回时间序列、利率历史及动力学状态历史。

### 3. 多项式混沌与不确定性量化 (`polynomial_chaos_uq.py`)

- 实现概率论 Hermite 多项式 `He_n(x)` 的系数递推 (`hep_coefficients`)、单点求值 (`hep_value`)、批量求值 (`hep_values`)。
- 多维乘积多项式求值 `hermite_product_polynomial_value`。
- `generate_multi_indices(d, p)` 生成总次不超过 p 的 d 维多维指标集。
- `polynomial_chaos_expand` 根据系数、多维指标和标准正态样本计算混沌展开的随机输出。
- `sobol_sensitivity` 基于混沌系数计算总方差与各维主效应 Sobol 指标，利用阶乘权重。

### 4. 稀疏线性代数 (`sparse_linear_algebra.py`)

- 读写 ST 格式文本文件（三元组：行、列、数值），自动处理 1‑based 与 0‑based 索引及头部注释。
- 内部实现 COO ↔ 三元组转换。
- `estimate_bandwidth` 计算稀疏矩阵的半带宽。
- `sparse_matvec` 稀疏矩阵‑向量乘法。
- `solve_sparse_system` 封装 `spsolve` 或 `splu` 求解稀疏线性系统，并返回求解信息（含残差）。

### 5. 特殊函数与分布 (`special_functions.py`)

- 对数正态分布：PDF、CDF（基于误差函数近似）、逆 CDF（调用标准正态逆）以及采样。
- 标准正态逆 CDF 采用 Wichura 算法（AS 241），基于有理多项式逼近，分三个区域计算。
- Lambert W 函数近似（WAPR 算法）：支持上、下分支，并可接受偏移量模式，末尾使用一次 Halley 迭代精化。

### 6. 随机动力学系统 (`stochastic_dynamics.py`)

- `lorenz96_parameters` / `lorenz96_deriv`：维度、强迫项、初值扰动及右端项（循环边界）。
- `duffing_parameters` / `duffing_deriv`：硬/软弹簧参数、初值及包含立方刚度和外部周期驱动的二阶方程。
- `oregonator_parameters` / `oregonator_deriv`：由化学反应机理导出的无量纲参数，三维状态变量 u, v, w 的刚性动力学。
- `multi_factor_coupling`：将三个系统的归一化特征量（混沌强度、位移、流动性水平）通过可配置的耦合矩阵投影到多因子波动率空间，并取绝对值保证非负。

### 7. 期限结构 PDE 求解 (`term_structure_pde.py`)

- `musiela_drift`：用 10 点 Gauss‑Legendre 积分计算 HJM 漂移项 `Σ σ_i(t,T)·∫_t^T σ_i(t,u) du`。
- `forward_rate_pde_rhs`：根据当前前向利率曲线，采用中心差分（支持非均匀网格）构造对流‑扩散方程的空间离散化矩阵 A_fd 和强迫向量
