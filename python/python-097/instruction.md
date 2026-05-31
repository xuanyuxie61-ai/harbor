# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目 python-097：微波器件 FDTD 仿真系统

## 项目概述
本项目实现了一套三维时域有限差分（FDTD）电磁仿真工具，用于分析圆柱形微波谐振腔的电磁模式特性。系统融合了多个数值计算与物理建模模块，最终通过 `main.py` 统一调用完成仿真流程：构建几何模型、生成 Yee 网格、配置材料、初始化 FDTD 引擎与 PML 吸收边界、施加激励源、时域推进与能量监测、本征模式提取以及数值验证。

后续 benchmark 将保留 `main.py`，其余 `.py` 文件会被删除，需由 agent 根据本描述重新实现缺失的模块。

## 模块清单与职责

### 1. `physics_constants.py` —— 物理常数与电磁学基础算子
- 定义真空物理常数：磁导率 `MU_0`、介电常数 `EPSILON_0`、光速 `C_0`、波阻抗 `ETA_0`。
- 提供电场旋度 `curl_electric_to_magnetic(E, dx, dy, dz)` 和磁场旋度 `curl_magnetic_to_electric(H, dx, dy, dz)` 的中心差分实现，分别用于法拉第定律和安培定律的更新。
- 提供能量密度函数 `electromagnetic_energy_density(E, H, epsilon, mu)`、坡印廷矢量 `poynting_vector(E, H)`。
- 提供辅助函数：品质因数 `quality_factor(omega, W_stored, P_loss)`、CFL 稳定性条件 `cfl_condition_3d(dx, dy, dz, c_max)`、色散关系 `wavenumber_frequency_relation(omega, epsilon, mu)`。

### 2. `yee_grid.py` —— Yee 交错网格生成与管理
- 定义 `YeeGrid3D` 类：存储三维矩形均匀网格的节点数、尺寸、步长、主网格坐标（电场节点）和交错网格坐标（磁场节点），并提供 `cell_volume()` 和网格形状查询。
- 定义 `CylindricalYeeGrid` 类：用于圆柱坐标系 (r,z) 的二维网格，处理轴心 r=0 处的奇异性，提供 `radial_weighting()` 和轴边界条件。
- 提供工厂函数：`generate_rectangular_grid(...)` 生成平移后的矩形网格，`generate_polar_grid_2d(...)` 生成极坐标网格。

### 3. `geometry_shapes.py` —— 几何形状定义与材料分布
- 实现多种腔体/介质几何体：
  - `CylindricalCavity`：圆柱形腔体，提供体积、表面积、TE/TM 模式截止频率的理论计算，以及点判属函数 `is_inside`。
  - `CoaxialCavity`：同轴腔体，提供特性阻抗、单位长度电容和点判属。
  - `CircleSegmentDielectric`：圆扇形介质块（用于模式调谐），提供面积、形心计算及点判属。
  - `ParametricProfile`：基于离散轮廓点的参数化形状（可描述波纹壁等），通过线性插值给出半径，并可判断点是否在内。
- 提供 `create_corrugated_wall_profile` 函数生成余弦波纹轮廓。
- 提供 `assign_material_properties(grid, shapes)` 函数，根据几何形状列表为网格的每一个节点分配介电常数、磁导率和电导率（基于网格坐标和形状的 `is_inside` 方法）。

### 4. `fdtd_engine.py` —— FDTD 核心引擎
- 定义 `FDTD3DEngine` 类：
  - 构造函数接收 `YeeGrid3D`、材料参数场 `epsilon, mu, sigma`、可选的激励源函数、时间步长（或基于 CFL 自动计算）。
  - 预计算磁场更新系数 `ch = dt/mu` 和电场隐式处理系数 `ce1, ce2`（源自 backward Euler 思想处理导电损耗）。
  - 方法：`update_magnetic()`（法拉第定律）、`update_electric()`（安培定律）、`apply_pec_boundary()`（理想导体边界，切向电场置零）、`apply_source()`、`compute_energy()` 和 `compute_power_loss()`。
  - `step()` 方法按顺序执行磁场更新、电场更新、边界条件、时间推进和源激励；`run(n_steps, ...)` 方法循环调用并收集能量历史。
- 定义 `HarmonicSource` 类：正弦调制高斯脉冲激励源，可在指定网格点注入特定场分量，支持 `call(t, engine)` 接口。
- 提供函数 `stability_analysis_2d_scalar(kx, ky, dx, dy, dt, c)` 用于二维标量波动的数值色散分析。

### 5. `pml_boundary.py` —— 完美匹配层吸收边界
- 定义 `PMLBoundary3D` 类：
  - 接收 Yee 网格、PML 厚度、多项式阶数等参数，自动计算最大电导率 `sigma_max` 并在边界区域构建多项式电导率分布 (`sigma_x, sigma_y, sigma_z`)。
  - 内部存储分裂场变量 (`Ex_y`, `Ex_z`, `Hx_y` 等)，用于基于分裂场形式的 PML 更新。
  - `update_electric_pml(...)` 和 `update_magnetic_pml(...)` 方法根据拉伸坐标 Maxwell 方程更新场量，只在 PML 区域应用，非 PML 区域直接保持原场。
  - `compute_reflection_estimate()` 给出理论反射系数估计。

### 6. `eigenmode_solver.py` —— 电磁模式求解与 PageRank 分析
- 实现迭代特征值求解：
  - `power_method_eigenmode(A_func, x0, ...)`：幂方法求最大模特征值。
  - `inverse_power_method(A_func, x0, sigma_shift, ...)`：带位移的逆幂方法，用 Richardson 迭代近似求解线性系统，提取离给定位移最近的特征值。
- 提供 `build_fd_helmholtz_operator_2d(...)` 构造二维标量亥姆霍兹算子的有限差分矩阵‑向量乘法函数（五点差分格式，Dirichlet 边界）。
- `compute_cavity_modes_2d(...)` 基于逆幂方法和正交化计算二维谐振腔的前若干个模式，返回模式频率、波数、场分布等。
- `power_flow_pagerank_analysis(E, H, dx, dy, dz, damping, n_iter)` 将电磁场能量密度类比 PageRank 中的转移矩阵，通过阻尼迭代得到每个空间网格点的“能量重要性”分布。

### 7. `special_matrices.py` —— 特殊结构矩阵运算
- 提供 Hankel、Toeplitz、循环矩阵的构造：`hankel_matrix(n, x)`、`toeplitz_matrix(n, t)`、`circulant_matrix(n, c)`。
- `hankel_inverse_fiedler(n, x)` 利用 Fiedler 公式通过求解两个线性系统并组合四个辅助矩阵来获得 Hankel 矩阵的逆，同时返回原矩阵。
- `solve_toeplitz_system(n, t, b)` 使用 Levinson‑Durbin 递归求解 Toeplitz 系统。
- `antenna_array_impedance_matrix(n_elements, spacing_wavelength, ka)` 返回具有 Toeplitz 结构的均匀线阵互阻抗矩阵（简化模型）。

### 8. `quadrature_rules.py` —— 高阶数值积分规则
- 内置 Wandzura 三角形对称积分点与权重（7 阶和 13 阶）。
- `integrate_triangle_wandzura(f, vertices, rule_degree)` 将给定三角形映射至参考三角形并应用对应规则计算积分。
- `gauss_legendre_1d(n)` 返回一维 Gauss‑Legendre 积分节点与权重（n≤10 时查表，其余用 numpy 计算）。
- `integrate_3d_pyramid_gauss
