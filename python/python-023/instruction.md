# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：空间等离子体波粒相互作用自适应相空间准线性输运模拟

## 整体目标
本项目构建了一个博士级的计算框架，用于模拟磁层空间中 whistler 模电磁波与电子的回旋共振相互作用，以及由此导致的非热电子在分形湍流磁场中的相空间输运。核心物理包括回旋共振条件、准线性扩散张量、Fokker‑Planck 方程、Vlasov‑Maxwell 色散关系等。主流程在 `main.py` 中组织，其余文件提供各个计算模块。

## 模块职责与关键接口

### main.py
- 总控脚本，运行完整的波粒相互作用模拟流程。
- 依次调用所有子模块，顺序如下：
  1. 设置物理参数并打印。
  2. 生成分形磁通管结构，并映射为磁场扰动。
  3. 求解 whistler 模色散关系，得到复频率。
  4. 初始化非热电子分布（Kappa + 非中心Beta）。
  5. 积分粒子 Lorentz 力的轨道。
  6. 检测满足回旋共振条件的粒子。
  7. 组装准线性扩散稀疏算子。
  8. 矩阵指数时间演化求解扩散方程。
  9. 多项式混沌展开量化磁场不确定性的影响。
  10. 多维 Lagrange 插值重构相空间分布，并计算速度空间矩、处理波场时间序列。
- 最终输出模拟统计信息和各阶段结果字典。

### dispersion_relation.py
- 实现 whistler 模色散关系的 Newton‑Raphson 求解器。
- 核心物理模型：平行传播右旋偏振波的动力学色散函数，涉及等离子体色散函数 \(Z(\zeta)\) 及其导数。
- 关键函数：
  - `plasma_dispersion_function(zeta)`：计算复 \(Z(\zeta)\)，对小参数用 Faddeeva 函数，对大参数用渐进展开以保证数值稳定性。
  - `whistler_dispersion_residual(omega, k, params)`：返回色散函数值 \(D(k,\omega)\) 及其对 ω 的解析导数。
  - `solve_whistler_dispersion(k, params, ...)`：反向通信 Newton 迭代求解复频率 ω，内置冷等离子体近似作为初始猜测，含步长限制与收敛检查。
- 输入：波数 k，物理参数字典；输出：复频率 ω 或 None（失败时）。

### distribution_models.py
- 非热电子速度分布函数模型。
- 提供 3D Kappa 分布、非中心不完全Beta分布尾巴、复合分布及逃逸/存活概率。
- 关键函数：
  - `kappa_nonthermal_distribution(n_particles, v_max, v_te, kappa, params)`：使用接受‑拒绝方法采样生成粒子速度矢量，并计算对应的复合分布函数值，返回分布值数组和速度网格。
  - `escape_probability(t, tau_esc)` 和 `survival_probability`：描述粒子从共振区逃逸的概率。

### fractal_magnetic_field.py
- 使用 Menger 海绵的迭代函数系统（IFS）生成分形磁通管点云。
- 功能包括：分形点生成、盒计数分形维数计算、将分形点映射为空间磁场扰动函数。
- 关键函数：
  - `generate_fractal_flux_tubes(n_points)`：返回形状 (n_points, 3) 的点云。
  - `map_fractal_to_magnetic_field(points, B0, fractal_scale)`：返回一个可调用对象，根据位置计算带有分形调制的磁场向量。

### file_sequence_processor.py
- 波场时间序列数据的处理器，包含序列文件名生成、模拟场振幅与相位演化、统计量计算、网格重排和时间自相关。
- 关键函数：
  - `process_field_timeseries(omega_solutions, params, n_frames)`：基于色散解模拟波包的时间演化，返回统计字典（振幅范围、主导频率等）。
  - `extract_grid_from_sequence(data_sequence, nx, ny)`：将一维序列重排为二维网格。
  - `compute_temporal_correlation(field_series, max_lag)`：计算归一化时间自相关函数。

### pce_expansion.py
- 基于 Legendre 多项式的多项式混沌展开（PCE），用于量化磁场随机涨落对分布函数不确定性的影响。
- 支持多指标枚举、归一化多项式计算、Galerkin 投影。
- 关键函数：
  - `polychaos_magnetic_uncertainty(v_parallel, v_perp, params, n_stochastic, p_degree)`：执行 PCE 分析，返回分布函数的均值场和方差场（二维网格）。

### phase_space_lagrange.py
- 多维 Lagrange 插值重构相空间分布函数，使用重心 Lagrange 公式和 Chebyshev 节点抑制 Runge 现象。
- 关键函数：
  - `lagrange_phase_space_reconstruction(v_parallel, v_perp, f_grid, params, n_cheb)`：在速度网格上执行二维张量积插值，返回重构后的分布函数网格。

### quasilinear_diffusion.py
- 组装准线性扩散系数和离散扩散算子矩阵。
- 基于 Kennel‑Engelmann 理论，计算扩散张量的离散分量 \(D_{\parallel\parallel}, D_{\perp\perp}, D_{\parallel\perp}\)，并用中心差分构建稀疏/稠密矩阵。
- 关键函数：
  - `assemble_ql_diffusion_matrix(v_parallel, v_perp, omega_solutions, params, n_stochastic, p_degree)`：返回扩散算子矩阵 A 和归一化初始条件 rhs 向量。

### matrix_exponential_solver.py
- 用于 Fokker‑Planck 方程时间推进的矩阵指数求解器。
- 提供 Pade 逼近计算稠密矩阵指数和 Krylov 子空间近似计算 exp(A)v。
- 关键函数：
  - `evolve_diffusion_operator(A, f0, dt, n_steps, use_krylov)`：执行多步矩阵指数演化，自动选择 Pade 直接矩阵乘或 Krylov 方法，并带有数值稳定性检查。

### moment_integrator.py
- 速度空间各阶矩的数值积分器。
- 利用轴对称性将 3D 积分化为 2D（\(v_{\parallel}, v_{\perp}\)）复合 Simpson 规则，计算密度扰动、平行/垂直温度、温度各向异性、热流和熵。
- 关键函数：
  - `compute_velocity_space_moments(v_parallel, v_perp, f_grid, params)`：返回包含所有矩的字典。

### particle_orbit.py
- 带电粒子在电磁场中的轨道积分器，基于高敏感性 ODE 思想。
- 核心方法：Boris 推进器（相空间体积保持），可选的 Dormand‑Prince RK45 步，以及李雅普诺夫指数计算。
- 关键函数：
  - `integrate_lorentz_orbits(x0, v0, B0_vec, params, t_span, n_steps)`：返回所有粒子的轨道状态 （位置+速度），并估计最大李雅普诺夫指数。

### resonance_voronoi.py
- 基于回旋共振条件和距离范数的共振粒子检测，以及 Voronoi 邻域搜索。
- 关键函数：
  - `detect_resonant_particles(v_grid, omega_solutions, params, resonance_width, p_norm)`：返回满足共振条件的粒子索引列表。
  - `voronoi_nearest_neighbor`：执行最近邻搜索。
  - `compute_resonance_region_volume`：通过 Monte Carlo 估计共振区域体积。

### sparse_assembler.py
- 稀疏矩阵工具箱，提供 COO 转 CSR 格式、CSR 矩阵‑向量乘法、条件数估计、Harwell‑Boeing 文件输出。
- 关键函数：
  - `sparse_matrix_operations(A_dense, tol)`：对稠密矩阵进行稀疏化并转换为 CSR，返回统计信息字典。
- 主要服务于扩散矩阵的稀疏处理。

## 模块间关系
- `main.py` 作为顶层
