# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# PROJECT_78: 血动脉脉动流与壁面剪切应力分析

## 项目概述
本项目是一个计算流体力学（CFD）仿真代码库，模拟动脉中的脉动血流与壁面剪切应力（WSS）分布。项目整合了多个独立算法模块，覆盖几何建模、数值积分、线性代数求解、随机扩散、网格生成、血管力学、脉冲波传播、血流网络分配、最优控制以及时间推进的CFD求解器。主入口为 `main.py`，它调用各模块完成一个完整的计算流程，并输出分析报告。

所有模块（除 `main.py`）的源代码将被移除，要求根据本描述重新实现它们。每个模块的功能边界、核心数据结构和关键算法关系如下所述。

---

## 模块说明

### 1. `geometry_utils.py` – 血管几何建模与基础物理工具
**职责**：提供高精度圆周率计算、素数判定、有限元三角形网格、血管截面几何参数、Womersley数、雷诺数、Murray定律以及其他数值鲁棒性工具。

**核心内容**：
- **π 计算**（Spigot/BBP 算法）：函数 `pi_spigot(n_digits)` 使用 BBP 公式计算 π 到指定小数位数的字符串；`pi_high_precision()` 返回双精度浮点 π。
- **素数工具**：`is_prime(n)` 试除法判定素数；`prime_sieve(limit)` 埃拉托斯特尼筛法生成素数数组；`bifurcation_prime_level(level)` 判断血管分叉级别是否为素数级（用于标记关键节点）。
- **有限元网格**：类 `FEMMesh` 存储节点（`nodes`）和三角形单元（`elements`），提供 `element_area(idx)`、`total_area()`、`scale_to_area(target_area)` 等方法。
- **血管几何**：`circular_cross_section_area(radius)`、`circular_cross_section_perimeter(radius)` 基于 π 计算面积和周长。
- **血流动力学参数**：`womersley_number(radius, kinematic_viscosity, angular_frequency)` 计算 Womersley 数；`reynolds_number(mean_velocity, diameter, kinematic_viscosity)` 计算雷诺数；`murray_law_radius(r_parent, n_children, bifurcation_angle_deg)` 基于 Murray 定律计算子管半径。
- **安全数值函数**：`safe_sqrt(x)`、`safe_divide(a, b, default)`。

**被 `main.py` 调用**：用于初始几何参数计算、分叉级别标记、FEM 网格面积验证等。

---

### 2. `linear_algebra_core.py` – 稀疏三对角线性系统求解器
**职责**：实现 R83T 格式的三对角矩阵存储、矩阵-向量运算、迭代求解器（Jacobi、Gauss-Seidel、共轭梯度）以及 Thomas 直接解法，并包含构造 Womersley 方程三对角系统的函数。

**核心内容**：
- **类 `R83TMatrix`**：以 `(n,3)` 数组存储三对角矩阵（次对角线、主对角线、超对角线）。提供类方法 `from_diagonals(sub, main, super)` 和 `dif2(n)`（构造 DIF2 测试矩阵），以及 `to_dense()`、`eigenvalue_dif2(i)` 等辅助方法。
- **矩阵运算**：`r83t_mv(A, x)`、`r83t_mtv(A, x)`、`r83t_res(A, x, b)` 分别实现矩阵-向量乘、转置乘和残差计算。
- **迭代求解器**：`r83t_jacobi_solve(A, b, ...)`、`r83t_gauss_seidel_solve(A, b, ...)`、`r83t_cg_solve(A, b, ...)` 均返回解向量、迭代次数和最终残差。CG 要求矩阵对称正定。
- **直接求解器**：`thomas_algorithm(A, b)` 使用 Thomas 算法（三对角追赶法）求解。
- **Womersley 系统构造**：`build_womersley_tridiagonal(n_r, alpha_w, dt, dr, kinematic_viscosity)` 返回 `R83TMatrix`，用于隐式时间步进离散的系数矩阵。

**被 `main.py` 调用**：用于 DIF2 矩阵求解测试，以及 `pulsatile_cfd_engine.py` 中求解每个时间步的三对角系统。

---

### 3. `quadrature_rules.py` – 数值积分与插值分析
**职责**：提供正六边形区域上的 Lyness 对称求积、Gauss–Legendre 求积节点与权重计算、Runge 函数分析及 WSS 分布插值误差评估。

**核心内容**：
- **六边形求积**：`hexagon01_area()` 返回单位正六边形面积；`hexagon_lyness_rule03()` 和 `hexagon_lyness_rule07()` 分别返回代数精度 5 和 9 的节点与权重；`integrate_on_hexagon(f, rule_id)` 在六边形区域上积分。
- **Gauss–Legendre 求积**：`legendre_gauss_nodes_weights(n)`
