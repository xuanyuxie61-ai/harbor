# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 微反应器混合与反应强化多尺度计算框架（PROJECT_138）

本项目是一个面向微反应器设计优化的博士级计算框架，无需外部输入参数即可运行完整的数值分析。代码包含 12 个独立模块，由 `main.py` 统一调度，依次执行 PDE 求解、催化剂分布优化、动力学参数估计、稳定性分析、质量‑能量平衡、降阶模型构建、混合质量统计、操作条件优化、稀疏矩阵运算、网络拓扑分析、热应力分析和离散负载规划等步骤。

保留 `main.py`，删除其余所有 `.py` 文件后，请根据本描述补全缺失的模块文件。

## 模块清单

### 1. `reactor_pde_solver.py` — 微通道对流‑扩散‑反应 PDE 稳态求解器

**核心类：** `MicroreactorPDESolver`

**功能：** 求解一维微通道内的稳态浓度场和温度场，控制方程为对流‑扩散‑反应耦合 PDE，反应速率采用 Arrhenius 型表达式。空间离散使用有限体积法，时间推进采用伪时间迭代至收敛。

**主要接口：**
- `solve_steady_state(max_iter, tol)` → 返回浓度 \(C(x)\) 和温度 \(T(x)\) 数组。
- `compute_conversion_and_yield(C)` → 计算出口转化率。
- `compute_peclet_damkohler(C, T)` → 计算整体 Peclet 数和 Damköhler 数。

**关键细节：** 构造函数接受通道长度、网格数、扩散系数、流速、Arrhenius 参数、反应级数、物性参数、壁面传热系数等。内部空间离散利用迎风格式处理对流项，中心差分处理扩散项，反应源项使用点隐式 Newton‑Raphson 局部迭代。边界条件为入口 Dirichlet、出口零梯度、壁面 Robin 传热。

### 2. `catalyst_placement_cvt.py` — 基于 CVT 的催化剂最优分布

**核心类：** `CatalystCVTPlacer`

**功能：** 使用 Centroidal Voronoi Tessellation (CVT) 算法优化二维或三维区域内催化剂颗粒的空间分布，使得覆盖率均匀。通过 Lloyd 迭代（Monte Carlo 采样逼近 Voronoi 单元质心）最小化能量泛函。

**主要接口：**
- `iterate()` → 执行 Lloyd 迭代，返回生成元坐标、能量和最大位移。
- `compute_uniformity_index()` → 计算均匀度指数 \(\eta\)。
- `get_catalyst_loading_map(grid_res)` → 生成规则网格上的催化剂负载密度图（仅 2D）。

**关键细节：** 构造函数接受维数、生成元数量、区域边界、密度权重函数、采样点数、最大迭代次数、容差。内部实现重要性采样生成符合密度函数的随机点，计算点到生成元的最近邻归属，通过质心公式更新生成元并裁剪到边界。提供缺省的均匀密度函数。

### 3. `kinetics_parameter_estimation.py` — 反应动力学参数最小二乘估计

**核心类：** `KineticsParameterEstimator`

**功能：** 基于实验数据（浓度、温度、反应速率）估计 Arrhenius 参数（指前因子 \(A\)、活化能 \(E_a\)、反应级数 \(n\)）。采用非线性最小二乘，结合线搜索的 Gauss‑Newton 迭代，并利用 QR 分解（Householder 方法）求解线性化子问题。同时提供基于 QR 的置信区间计算。

**主要接口：**
- `estimate_arrhenius_parameters(concentrations, temperatures, rates)` → 返回 \((A, E_a, n, \text{residual})\)。
- `qr_factorize(A)` → 返回 \((Q, R)\)。
- `solve_least_squares(A, b)` → 使用 QR 分解求解最小二乘系统。
- `compute_confidence_intervals(A, b, x_est)` → 返回参数标准差估计。

**关键细节：** 构造函数接受气体常数。参数估计先进行一维格点搜索获得初始猜测，再通过 Gauss‑Newton 迭代（嵌入 Levenberg‑Marquardt 阻尼）和回溯线搜索优化原始残差。QR 分解采用 Householder 反射，回代求解上三角系统。

### 4. `stability_eigenanalysis.py` — 稳态线性稳定性特征值分析

**核心类：** `ReactorStabilityAnalyzer`

**功能：** 对稳态浓度‑温度场进行线性稳定性分析，构建 Jacobian 矩阵并计算特征值，判断稳态是否稳定，定位临界 Damköhler 数区间，计算热爆炸风险指数。

**主要接口：**
- `analyze_stability(C_steady, T_steady, A_arr, Ea, n_order)` → 返回全部特征值、最大实部、稳定性标志。
- `compute_critical_damkohler_bracket(...)` → 扫描 Damköhler 数范围，返回失稳前后的 Da 和最大实部。
- `compute_thermal_explosion_index(max_real)` → 计算热爆炸风险指数。

**关键细节：** 构造函数接受网格数、物理参数、壁面传热系数等。Jacobian 矩阵由对流‑扩散算子与反应速率导数组成，通过迎风+中心差分构建算子，并组合成 2N×2N 矩阵。使用特征值分解判断稳定性，Da 扫描通过缩放指前因子实现。

### 5. `mass_energy_balance.py` — 质量‑能量耦合平衡定点迭代求解器

**核心类：** `MassEnergyBalanceSolver`

**功能：** 对全混流微反应器（CSTR 模型）求解质量与能量平衡耦合方程，采用定点迭代 \(T^{(n+1)} = G(T^{(n)})\)，其中 \(G\) 由能量平衡显式推导。支持多起点求解以捕获多重稳态，可计算分岔指示器。

**主要接口：**
- `solve_fixed_point(T_guess, max_iter, tol)` → 返回稳态温度、浓度、迭代次数、收敛标志。
- `multi_start_solve(n_st
