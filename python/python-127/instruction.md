# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 项目描述：人工耳蜗电刺激电场分布数值模拟系统

## 概述
本项目是一个用于人工耳蜗电刺激电场分布模拟的数值计算系统。系统以蜗牛壳状螺旋几何为基础，对患者耳蜗进行个性化建模，结合电极阵列配置、有限元法和解析方法求解组织中的电势分布，并进一步模拟螺旋神经节神经元的膜电位响应、神经激活斑图的时空演化、患者参数变异性统计以及基于 SVD 的降维分析。

## 文件结构

### `main.py`
统一入口脚本，执行完整的仿真流程。零参数即可运行。该文件调用其余所有模块，串联以下步骤：
1. 耳蜗几何建模
2. 电极阵列配置
3. 有限元网格生成与电场求解
4. 高阶 Laplacian 算子与解析电势验证
5. 神经元膜电位响应模拟
6. 神经激活反应-扩散斑图演化
7. 高精度三角形数值积分
8. SVD 患者参数降维分析与电极配置优化
9. 患者变异性统计建模
10. 一维电流连续性方程求解
每个步骤完成后打印关键中间结果，并执行若干物理一致性断言。

### `cochlea_geometry.py`
耳蜗三维几何建模模块。

**核心类**：`CochleaGeometry`
- 使用对数螺旋参数方程（由初始半径 `r0`、螺旋紧缩系数 `b`、最大螺旋角 `theta_max` 控制）描述蜗轴中心线。
- 内部构建中心线离散点、切向量和法向量。
- 提供的方法包括：按角度插值中心线坐标；计算点到蜗轴的有符号距离（基于最近线段投影）；基于一组已知测量点，用三次样条插值重建患者个性化耳蜗轮廓；构建螺旋神经节神经元图拓扑（节点位置沿骨螺旋板分布，边为邻接连接，权重为距离倒数）；以及获取特定角度位置鼓阶横截面四边形。

### `electrode_array.py`
电极阵列配置模块。

**核心类**：`ElectrodeArray`
- 描述含多个铂铱合金电极的线性阵列，包含电极半径、间距、插入深度和离蜗轴偏移等物理参数。
- 主要方法：`place_along_modiolar_axis(geometry)` 沿蜗轴放置电极，得到电极二维坐标；`set_currents` 设置各电极刺激电流（μA 输入，内部转换为 A）；`monopolar_stimulus` 和 `tripolar_stimulus` 分别实现单极和三极刺激配置；`get_source_terms(mesh_nodes)` 将电极电流转换为有限元网格上的高斯点源分布。

### `fem_solver.py`
二维有限元求解器模块。

**核心类**：`FEM2DSolver`
- 基于 Galerkin 方法在三角网格上求解 Poisson 方程 ∇·(σ∇V) = -I_e。
- 构造函数接收节点坐标、三角形元素数组和电导率（标量或逐元素）。
- 内部按单元组装全局刚度矩阵（稀疏 CSR 格式）。
- `solve(source, dirichlet_nodes, dirichlet_values)` 实现电势求解，支持指定 Dirichlet 边界条件。
- 提供 `compute_gradient(V)` 计算单元内的电势梯度，以及 `compute_activation_function(V)` 计算沿蜗轴方向的电势二阶导数（激活函数），用于神经刺激分析。

**辅助函数**：`generate_cochlea_mesh(geometry, n_radial, n_angular)` 基于 `CochleaGeometry` 生成耳蜗鼓阶截面的结构化三角形网格。

### `laplacian_operator.py`
高阶 Laplacian 离散算子模块。包含：
- `laplacian_5point(V, dx, dy)` — 经典五点差分格式。
- `laplacian_9point(V, dx, dy)` — 九点高阶格式（O(h⁴) 截断误差）。
- `build_differ_matrix(n, stencil)` — 构建 Vandermonde 型差分模板矩阵，用于求解任意阶导数的差分系数。
- `high_order_derivative_coefficients(order, stencil)` — 利用上述矩阵求差分系数。
- `laplacian_matrix_1d(n, dx, stencil_type)` — 以稀疏矩阵形式构建一维 Laplacian（支持二阶中心或四阶 compact 格式）。
- `anisotropic_laplacian_5point(V, dx, dy, sigma_xx, sigma_yy)` — 各向异性守恒型五点 Laplacian。

### `potential_field.py`
电势场的解析计算模块，基于特殊函数展开。

核心函数：
- `gegenbauer_polynomial_value(m, alpha, x)` — 递推计算 Gegenbauer 多项式值，附带归一化常数计算。
- `sincn(x)` — 归一化 sinc 函数。
- `sinc_interpolation_1d(x_samples, f_samples, x_query)` — Whittaker-Shannon sinc 插值。
- `analytical_potential_spherical(r, theta, I_source, sigma, R_cochlea, n_terms, alpha)` — 球谐（Gegenbauer/Legendre）展开的解析电势。
- `cylindrical_potential_line_source(rho, z, z_e, I_e, sigma)` — 无限长圆柱中线源基本解。
- `multi_electrode_superposition(electrode_positions, electrode_currents, query_points, sigma)` — 基于点源基本解叠加的多电极电势。

### `neural_membrane.py`
螺旋神经节神经元膜电位动力学模块。

**核心类**：
- `SimplifiedSGNModel` — FitzHugh-Nagumo 型简化模型（二维 ODE 系统）。`simulate` 方法使用 `scipy.solve_ivp` 进行积分，并通过事件检测捕获动作电位发放时刻。
- `DetailedSGNModel` — Hodgkin-Huxley 风格详细模型（四维 ODE 系统，包含 Na⁺ 和 K⁺ 门控变量动力学，以及对温度的 Q10 校正）。同样通过 `simulate` 方法积分并返回发放时刻。

**辅助函数**：`biphasic_pulse(t, amplitude, phase_width_ms, interphase_gap_ms)` — 生成临床常用的双相脉冲刺激电流波形。

### `reaction_diffusion.py`
神经激活模式反应-扩散模块。

**核心类**：`NeuralActivationPattern`
- 求解 Gray-Scott 型耦合反应-扩散方程，描述激活密度和抑制因子的时空演化。
- 构造函数定义网格、扩散系数及反应参数。
- `initialize(seed_pattern)` 支持高斯、随机或均匀初始条件。
- `evolve(n_steps, stimulus_history, dt)` 进行显式欧拉时间步进，返回历史记录。
- `compute_spread_metrics()` 计算激活区域面积、质心和空间展宽。

**辅助函数**：`neural_activation_rd(...)` — 单步更新函数，接受自定义 Laplacian 函数以支持不同差分格式。

### `quadrature_integration.py`
高精度三角形数值积分模块。

基于 Witherden-Vincent 对称求积规则（精度 1-5 阶），实现以下功能：
- `triangle_quadrature_rule(precision)` — 返回参考三角形上的求积点和权重。
- `integrate_triangle(f, vertices, precision)` — 在任意三角形上积分标量函数。
- `integrate_over_mesh_elements(f, nodes, elements, precision)` — 在三角网格所有单元上进行积分。
- `test_quadrature_precision(max_precision)` — 通过单项式积分验证各精度阶的准确性。

### `svd_analysis.py`
基于 SVD 的患者参数降维与电场分析模块。

**核心类**：`PatientSVDAnalyzer`
- 输入患者特征矩阵（每列为一个患者的特征向量），进行中心化后执行经济型 SVD。
- 提供主成分提取、奇异值获取、方差解释比例、累积方差比例、投影、重构、低秩近似和压缩比计算。

**辅助函数**：
- `generate_synthetic_patient_data(n_patients, n_features, n_modes, noise_level)` — 生成合成数据用于
