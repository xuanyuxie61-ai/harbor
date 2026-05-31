# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

## 项目描述：精馏塔传质与能效优化科学计算平台

本项目是一个面向精馏塔设计和分析的 Python 科学计算工具集，涵盖汽液平衡热力学、传质动力学、塔板几何建模、能效优化、填料随机堆积、压力波动模拟以及不确定性量化等多个模块。入口程序 `main.py` 通过一系列演示函数调用各个模块，验证整体系统的正确性。你的任务是仅根据这份描述和保留的 `main.py`，实现所有缺失的辅助模块（`.py` 文件），使 `python main.py` 能够正常运行并输出合理的结果。

项目要求实现的核心模块及其职责如下：

### 1. `utils.py` – 通用工具函数
提供数值安全和边界检查的基础功能，被其他所有模块依赖。

**主要函数：**
- `safe_divide(a, b, fill_value)`：安全除法，避免除零。
- `clip_with_warning(x, xmin, xmax, name)`：将变量裁剪到指定区间，越界时打印警告。
- `ensure_positive(x, eps, name)`：保证变量为正，将小于等于零的元素提升到 `eps`。
- `relative_change(new, old)`：计算新旧值之间的最大相对变化，用于收敛判断。
- `thermo_factor_check(T, Tmin, Tmax)`：检查温度（并可扩展至压力）是否在合理热力学区间内，返回裁剪后的安全值。

### 2. `vle_thermodynamics.py` – 汽液平衡 (VLE) 热力学计算
实现多组分非理想体系的相平衡计算，基于多项式谱展开和数值求积。

**主要功能与函数：**
- `jacobi_polynomial(m, n, alpha, beta, x)`：利用递推关系计算 Jacobi 多项式在给定点上的值，参数 α, β > -1。返回形状为 (m, n+1) 的矩阵。
- `laguerre_compute(norder, alpha)`：计算 Gauss-Laguerre 求积的节点和权重，用于形如 ∫₀^∞ e⁻ˣ x^α f(x) dx 的积分。内部实现 Newton 迭代求解 Laguerre 多项式根。
- `laguerre_quadrature_integrate(f, norder, alpha, transform)`：使用 Laguerre-Gauss 求积计算加权积分，支持可选的变量变换。
- `antoine_vapor_pressure(T, A, B, C)`：Anoine 方程计算饱和蒸气压（输入温度 `T` 为 °C，公式返回 Pa）。常数从 `main.py` 传入。
- `wilson_parameters(V, Lambda_ij, T)`：根据二元交互能参数和温度计算 Wilson 参数矩阵 Λ；`wilson_activity_coefficient(x, V, Lambda_ij, T)` 利用 Wilson 方程求活度系数。
- `vle_flash_calculation(x, P_total, T, A_ant, B_ant, C_ant, V, Lambda_ij)`：等温闪蒸，返回归一化汽相组成 `y`，相平衡常数 `K` 和活度系数 `gamma`。低压简化模型 y_i = K_i * x_i，K_i = γ_i * P_i^sat / P_total。
- `vle_relative_volatility(K)`：以最大 K 为参考计算相对挥发度。
- `activity_coefficient_spectral_expansion(x_range, nc, alpha_jac, beta_jac, n_modes)`：利用 Jacobi 多项式在归一化坐标 ξ=2x-1 上构造活度系数谱表示（返回基函数矩阵，非直接活度系数）。

### 3. `property_interpolation.py` – 物性插值与数值积分
提供实验数据插值和沿塔高的传质通量积分。

**主要函数：**
- `shepard_interp_1d(nd, xd, yd, p, ni, xi)`：一维 Shepard 反距离加权插值。当插值点与数据点重合时分配全权重，否则使用距离的 -p 次方。支持 p=0 等权平均。
- `quad_trapezoid(f_func, a, b, n)`：复合梯形数值积分。
- `interpolate_vle_data(z_data, T_data, x_data, y_data, z_query, p)`：沿塔高位置 `z` 对温度、液相和汽相组成进行 Shepard 插值（每次按组分分别插值），并归一化组成。
- `integrate_mass_transfer_flux(z_nodes, N_A_func)`：用梯形积分计算沿塔高的总传质通量。

### 4. `tray_geometry_mesh.py` – 塔板几何建模与网格划分
生成矩形塔板的四边形网格，并基于 Murphree 效率的概念计算局部和平均效率。

**主要函数：**
- `drectangle(p, x1, x2, y1, y2)`：计算点到矩形的有符号距离（内部为负，外部为正）。
- `reference_to_physical_q4(q4, n, rs)`：将 Q4 参考单元（R,S∈[0,1]）映射到物理坐标，使用双线性形函数。
- `q4_jacobian(q4, rs)`：计算映射的 Jacobian 矩阵及其行列式（面积缩放因子）。
- `generate_tray_mesh(tray_width, tray_height, nx, ny)`：生成均匀矩形网格，返回节点坐标、单元拓扑（逆时针四点）和各单元面积。
- `compute_local_efficiency_on_mesh(nodes, elements, x_liq, y_vap, K_eq)`：在每个网格单元上利用给定液相组成、汽相组成和平衡常数估算局部 Murphree 效率，效率定义基于 (y* - y_in) 的差商，对各组分取均值并裁剪到 [0,1]。
- `mesh_average_efficiency(nodes, elements, areas, E_local)`：面积加权计算平均效率。

### 5. `mass_transfer_dynamics.py` – 传质动力学与常微分方程 (ODE) 系统
集成多种 ODE 模型，包括扩散、混合、对流和精馏塔全塔动态物料平衡。

**主要函数：**
- `rk45_integrate(yprime, tspan, y0, n_steps, projection=None)`：显式 Runge-Kutta 4/5 阶积分器，返回时间序列、解序列和每步误差估计。支持每步后的投影函数，用于裁剪或归一化状态。
- `maxwell_stefan_diffusion(y, D_matrix, c_total)`：三组分 Maxwell-Stefan 扩散方程右端项，状态向量包含组成和通量。通量变化由组分差的阻尼交互力驱动，组成变化受通量影响。
- `simulate_three_component_diffusion(y0, D_matrix, c_total, tspan, n_steps)`：调用 RK45 模拟扩散过程。
- `langford_deriv(t, xyz, a, b, c, d, e, f)`：Langford 非线性系统右端项，用于描述局部湍流混合。
- `simulate_langford_mixing(xyz0, tspan, n_steps, ...)`：模拟 Langford 混合。
- `lorenz96_deriv(t, y, n, force)`：Lorenz96 混沌系统右端项，状态变量环状连接。
- `simulate_lorenz96_convection(y0, tspan, n_steps, force)`：模拟 Lorenz96 对流。
- `distillation_column_deriv(t, state, n_trays, nc, F, z_feed, q_feed, L, V, holdup, alpha_rel, tray_eff)`：精馏塔全塔动态物料平衡 ODE 右端项，考虑各板汽液相流量、进料、持液量、相对挥发度和 Murphree 效率。状态向量按塔板顺序排列各组分液相摩尔分数。组成边界硬限制防止越界。
- `simulate_distillation_dynamics(...)`：使用 RK45 积分和组成归一化投影模拟精馏塔动态，返回时间、状态、误差和按时间
