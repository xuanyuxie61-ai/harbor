# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 生物质热解反应器多物理场耦合模拟系统（python-140）

本项目是一个生物质热解反应器多物理场耦合模拟的数值平台，集成反应动力学、传热、颗粒运动学、网格生成、有限元组装和高精度数值积分等多种计算模块。主入口为 `main.py`，它按顺序调用各个模块，完成从几何建模到结果输出的完整工作流。所有参数均在 `main.py` 内部设定，零命令行参数运行。

以下列出除 `main.py` 外需要实现的所有 Python 模块，描述其职责、核心数据结构和对外接口。实现时应保证接口与 `main.py` 中的调用方式兼容，算法思想与描述一致，但具体数值参数和实现细节需要自主设计。

## 1. utils.py — 通用数值工具

提供项目共用的基础数值功能，包括：

-   `safe_exp(x)`: 数值稳定的指数函数，适用于阿伦尼乌斯公式中可能出现的大指数，通过裁剪防止溢出。
-   `safe_divide(a, b)`: 安全除法，避免除零，返回指定填充值。
-   `check_bounds(values, lower, upper)`: 检查数组是否在指定区间内，越界时发出警告并裁剪到边界。
-   `timestamp()`: 打印当前日期时间。
-   `finite_diff_jacobian(func, x)`: 使用中心差分计算任意多变量向量函数的雅可比矩阵。
-   `cond_number_estimate(A)`: 基于矩阵范数近似估算方阵的条件数，用于判断数值稳定性。

## 2. geometry_utils.py — 几何定义与变换

提供反应器几何体的有向距离函数（SDF）和二维几何变换。

-   关键 SDF 函数：
    -   `cylinder_distance_function(p, radius, center)`: 二维圆柱截面的有向距离。
    -   `rectangle_distance_function(p, xmin, xmax, ymin, ymax)`: 矩形区域的有向距离。
    -   `torus_distance_function(p, R_major, R_minor)`: 三维环面的有向距离（用于描述环形反应器的中心线几何）。
-   组合运算：
    -   `union_distance(d1, d2)`, `intersect_distance(d1, d2)`, `diff_distance(d1, d2)`: 对有向距离进行并、交、差集操作。
-   二维变换：
    -   `rotation_matrix_2d(theta)`: 生成旋转矩阵。
    -   `reflect_vector(v, axis)`: 沿 x 或 y 轴反射向量。
-   `compute_reactor_boundary_word(n_segments, reactor_type, **kwargs)`: 生成不同反应器截面（圆柱、矩形、环面中心线）的多边形边界点，用于可视化或网格生成。
-   `circumcenter(p1, p2, p3)`: 计算三角形的外心，用于网格质量评估。

## 3. reactor_mesh.py — 非结构化网格生成与细化

负责二维反应器截面网格生成和局部细化。

-   `distmesh_2d(fd, fh, h0, bbox, pfix, max_iter)`: 基于有向距离函数 `fd` 和网格尺寸函数 `fh` 生成二维非结构化三角形网格。算法通过力平衡迭代移动内部节点，并利用 Delaunay 三角化。返回节点坐标 `(N,2)` 和单元索引 `(NT,3)`（0‑based）。
-   `ball_grid(n, r, c)`: 在给定半径 `r` 和球心 `c` 的三维球体内生成均匀分布的网格点，用于离散催化剂颗粒内部。
-   `triangulation_refine_local(node_xy, element_node, element_to_refine)`: 对指定三角形单元进行 1:4 中点细分，返回细化后的节点数组和单元集合。
-   `compute_mesh_quality(p, t)`: 计算网格质量指标（内切圆半径与外接圆半径之比）的数组，质量接近 1 为等边三角形。

## 4. pyrolysis_kinetics.py — 热解反应动力学

实现生物质多组分热解反应网络的动力学建模与求解。

-   类 `BiomassPyrolysisKinetics`:
    -   描述一个包含纤维素、半纤维素、木质素三组分的并行热解网络，涉及五种阿伦尼乌斯反应，状态向量包含 7 个组分质量分数。
    -   构造时设定各反应的指前因子和活化能（典型文献值），并给定初始质量分数。
    -   方法 `reaction_rates(T)`: 根据温度返回 5 个反应速率常数。
    -   方法 `deriv(t, y, T_func)`: 计算给定时刻和状态的 ODE 右端项，其中温度由外部函数 `T_func(t)` 提供。在导数计算中包含产物分配系数，保证质量保守，并对状态进行必要的数值稳定处理。
    -   方法 `solve_rk4(tspan, y0, n_steps, T_func)`: 使用经典四阶 Runge‑Kutta 方法积分动力学 ODE，返回时间序列和状态矩阵。
    -   方法 `solve_midpoint(tspan, y0, n_steps, T_func, theta, max_iter)`: 使用隐式中点法求解，用不动点迭代处理隐式方程，最后将结果投影回可行域。
-   辅助函数：
    -   `doughnut_pyrolysis_flow(t, y, m_param, n_param)`: 环面反应器内的非线性流动方程组（三变量），右端项由参数 `m`, `n` 控制。
    -   `solve_doughnut_flow_rk4(...)`: 用 RK4 求解该流动方程，返回轨迹。

## 5. thermal_model.py — 反应器一维传热求解

求解生物质反应器内沿轴向的一维非稳态热传导‑对流‑反应耦合问题。

-   类 `BandedMatrixSolver`:
    -   实现紧凑带状矩阵的无选主元 LU 分解（`factor`）和回代求解（`solve`），对应于三对角或一般带状存储。
    -   构造时需指定矩阵规模 `n`、下带宽 `ml` 和上带宽 `mu`。
-   类 `ThermalReactorModel`:
    -   封装一维传热方程，包含密度、比热容、有效导热系数、气速等物理参数，以及轴向离散化为 `nx` 个节点。
    -   `build_system_matrix(dt)`: 为隐式欧拉时间离散构造三对角带状矩阵（含对流迎风格式），返回 `(3, nx)` 的带状存储数组。
    -   `solve_timestep(T_old, dt, Q_source, T_inlet)`: 完成一个时间步的温度更新，使用带状求解器。
    -   `simulate(T_init, dt, n_steps, Q_func, T_inlet)`: 执行完整时间推进，其中 `Q_func` 接受 `(t, x)` 返回各空间节点的反应热源。
-   独立函数 `compute_reaction_heat_source(x, T, kinetics, y_mass, reaction_enthalpy)`: 利用动力学对象计算某空间位置的近似反应热源密度。

## 6. quadrature_integrator.py — 高精度数值积分

提供与反应器模拟相关的多种求积方法。

-   `generalized_hermite_integral(expon, alpha)`: 返回权函数为 |x|^α exp(-x^2) 的单项式积分精确值（用于误差验证）。
-   `gauss_hermite_nodes_weights(n, alpha)`: 生成标准或广义 Gauss‑Hermite 求积节点和权重。
-   `integrate_daem_activation_energy(E, sigma, T, n_quad)`: 利用 Gauss‑Hermite 求积计算分布式活化能模型（DAEM）中活化能高斯分布下的有效反应速率因子。
-   `square01_monte_carlo_integrate(f, n_samples)`: 在单位正方形上使用蒙特卡洛方法计算积分，返回均值和标准误差估计。
