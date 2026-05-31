# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：湍流燃烧火焰面模型数值模拟系统（python-156）

## 项目简介
本项目实现了一套基于稳态层流火焰面模型（Steady Laminar Flamelet Model, SLFM）的湍流燃烧数值模拟工具。核心思想是在混合分数空间内求解温度与化学组分的分布，通过标量耗散率耦合湍流效应，并集成了点火/熄火极限分析、火焰前锋几何、湍流随机场、网格生成、守恒校验等多个子系统。整个系统采用有限元、有限差分等数值方法，辅以多种数学模型（椭圆积分、多项式求根、蛋形曲线、Voronoi 分区等）来模拟和分析燃烧现象。

## 文件结构与职责
项目包含多个 Python 模块，每个模块负责一个相对独立的科学计算子任务。**后续将仅保留 `main.py`，其余 `.py` 文件将被删除**，需要根据本描述重新实现它们。各文件职责如下：

| 文件名 | 职责概述 |
|--------|----------|
| `flamelet_core.py` | 定义火焰面模型的核心物理常量和基础函数：标量耗散率分布、混合气体密度与分子量、一步 Arrhenius 反应速率、边界条件等。 |
| `fem_thermal_solver.py` | 基于线性有限元法求解稳态火焰面温度方程的非线性迭代器。 |
| `fem_quadratic_solver.py` | 基于二次有限元法求解燃料/氧化剂质量分数的对流-扩散-反应方程。 |
| `fd_scalar_solver.py` | 基于有限差分法求解标量耗散率的修正方程（含湍流衰减项）。 |
| `mesh3d_generator.py` | 基于距离函数的 3D 非结构网格生成器（圆柱形燃烧室）。 |
| `turbulent_random_field.py` | 使用线性同余生成器（LCG）生成湍流速度脉动场，并据此计算标量耗散率脉动。 |
| `ignition_polynomial.py` | 基于 WDK 多项式求根算法分析点火与熄火临界 Damköhler 数。 |
| `flame_front_shape.py` | 使用蛋形曲线族参数化火焰前锋皱褶形状，并计算其表面积及皱褶因子。 |
| `arrhenius_kinetics.py` | 提供 Arrhenius 速率常数计算、进展变量 ODE 积分和绝热火焰温度估算。 |
| `flame_instability.py` | 基于受迫振动方程模拟 Darrieus-Landau 热扩散不稳定性及热声振荡，计算不稳定性增长率。 |
| `elliptic_special.py` | 实现 Carlson 对称形式的不完全椭圆积分 RF，用于火焰曲率计算和 Markstein 长度/曲率修正火焰速度。 |
| `spatial_partition.py` | 基于 Voronoi 图进行二维区域面积计算，并提供一维混合分数空间域分解和分区质量分析。 |
| `matrix_bandwidth.py` | 分析 FEM/FDM 刚度矩阵的几何带宽，估算稀疏存储需求。 |
| `stoichiometric_polynomial.py` | 利用多项式乘法（卷积）计算化学反应网络中满足化学计量变化的路径数，评估机理复杂度。 |
| `conservation_validator.py` | 对模拟结果进行质量守恒和能量守恒的加权校验和验证。 |
| `exact_benchmark.py` | 提供解析解（Manufactured Solution 和高斯渐近解），计算数值解与精确解之间的 L²、L∞、H¹ 误差。 |
| `main.py` | 主控程序，串联上述所有模块，执行完整的数值模拟流程并输出结果。 |

## 模块详细功能与接口说明

### 1. `flamelet_core.py` – 核心物理模型
- **主要内容**：全局物理常数（通用气体常数、压力、分子量、反应动力学参数、化学计量比等）；函数包括：
  - `scalar_dissipation_rate(Z, chi_st, Z_st)`：返回高斯型标量耗散率分布 χ(Z)，峰值位于化学计量混合分数处。
  - `mixture_molecular_weight(Z)`：混合气体的平均分子量。
  - `density_mixture(Z, T)`：理想气体状态方程计算密度。
  - `reaction_rate_one_step(T, Y_F, Y_O, Z)`：一步 Arrhenius 反应速率，包含点火门槛和温度自限因子的修正，可选混合分数空间限制。
  - `temperature_equation_rhs(Z, T, Y_F, Y_O, chi_st)`：返回扩散系数 κ 和温度源项。
  - `flamelet_boundary_conditions()`：返回左右边界的温度及质量分数。
  - `thermal_diffusivity_ref()`：参考热扩散系数 α。
- **接口特征**：大量使用 NumPy 向量化操作，支持标量和数组输入。其他模块依赖此文件中的函数和常量。

### 2. `fem_thermal_solver.py` – 温度场 FEM 求解器
- **功能**：求解弱形式 `∫ κ(Z) T' V' dZ = ∫ S_T V dZ`，其中 κ = ρχ/2，源项 S_T 近似为高斯热源并含有温度自限因子。
- **主要函数**：
  - `kappa_func(Z, T, chi_st)`: 计算等效扩散系数。
  - `source_func(Z, T, Y_F, Y_O, chi_st)`: 估计反应热源项，利用 `flamelet_core` 中的一步反应速率和典型参数。
  - `solve_fem_thermal(n, Z_nodes, T_init, Y_F_init, Y_O_init, chi_st, tol, max_iter)`: 外部调用入口。使用线性有限元（2 节点单元），2 点 Gauss-Legendre 积分，组装三对角刚度矩阵和右端项，应用 Dirichlet 边界条件，通过迭代（含欠松弛）求解非线性问题。
- **依赖**：`flamelet_core`。

### 3. `fem_quadratic_solver.py` – 组分质量分数 FEM 求解器
- **功能**：求解 `- d/dZ [D(Z) dY/dZ] = ω̇/ρ`，用于更新燃料或氧化剂的质量分数。
- **主要函数**：
  - `solve_fem_quadratic_species(n, Z_nodes, species_type, T_field, chi_st, tol, max_iter)`: 基于二次 Lagrange 单元（每个单元 3 节点），3 点 Gauss-Legendre 积分，线性化反应源项并组装刚度矩阵和载荷向量，Dirichlet 边界条件（左、右边界值来自 `flamelet_boundary_conditions`），迭代求解线性系统。
- **依赖**：`flamelet_core`。

### 4. `fd_scalar_solver.py` – 标量耗散率修正方程求解器
- **功能**：求解一维扩散-衰减方程： `-d/dZ [ D_eff dχ/dZ ] + C_χ ω_turb/k_turb · χ = S(Z)`，其中扩散系数非线性依赖 χ。
- **主要函数**：
  - `solve_fd_scalar_dissipation(n, Z_nodes, chi_st, C_chi, omega_turb, k_turb, tol, max_iter)`: 使用非均匀网格中心差分，组装三对角线性系统，迭代更新扩散系数直至收敛。
- **依赖**：`flamelet_core`（用于获取参考扩散系数和初始标量耗散率分布）。

### 5. `mesh3d_generator.py` – 3D 网格生成
- **功能**：基于 DistMesh 思想，在圆柱形燃烧室内生成四面体网格。
- **主要函数**：
  - `distance_cylinder(p, R, H)`: 圆柱的有向距离函数。
  - `target_edge_length(p, h0, R)`: 边界加密的目标边长。
  - `generate_mesh_3d(h0, R, H, iteration_max, pfix)`: 主函数，通过初始撒点、Delaunay 三角剖分、基于边长力的节点位置迭代优化、边界投影等步骤生成网格。
