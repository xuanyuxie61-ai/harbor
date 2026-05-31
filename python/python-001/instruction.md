# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：不规则小行星多尺度引力场建模与近距轨道长期稳定性分析系统

本系统针对碎石堆结构小行星，构建“多面体‑球谐耦合”高保真引力场模型，并在此基础上分析近表面轨道的长期稳定性、碰撞风险与最优悬停策略。项目采用模块化设计，各模块覆盖几何建模、引力场计算、有限元分析、轨道积分、参数优化、碰撞风险评估以及数据读写等功能。

以下描述各源文件的核心职责、主要接口和模块边界，后续实现者需要根据本描述和提供的 `main.py` 恢复剩余代码。

## 依赖关系概览
* `special_functions.py` – 基础数学库，无内部依赖。
* `asteroid_geometry.py` – 依赖 NumPy，提供形状生成与几何量计算。
* `gravity_harmonics.py` – 依赖 `special_functions.py` 中的缔合 Legendre 函数和阶乘比。
* `gravity_polyhedron.py` – 依赖 NumPy，对顶点/面片操作；可能调用 `gravity_harmonics.py` 中的类。
* `fem_gravity.py` – 依赖 NumPy，内部实现带状矩阵存储和前代求解器，用于泊松方程。
* `orbit_integrator.py` – 依赖 NumPy，提供确定性/随机轨道积分器。
* `orbit_optimization.py` – 依赖 NumPy，实现实验设计与回溯搜索。
* `collision_risk.py` – 依赖 NumPy，利用多面体表面信息进行分析。
* `data_io.py` – 依赖 NumPy，负责文件读写。

所有模块均由 `main.py` 统一调用，`main.py` 中展示了各模块的典型使用方式。

## special_functions.py
负责提供引力场建模所需的特殊函数与正交多项式工具。
* `lngamma_lanczos(z)` – 使用 Lanczos 近似计算 Gamma 函数的自然对数，返回值和错误标志。
* `gamma_lanczos(z)` – 基于对数 Gamma 计算 Gamma 函数值，处理边界情况。
* `factorial_ratio(n, m)` – 数值稳定地计算 n!/m!，用于归一化系数。
* `legendre_to_monomial_matrix(n_max)` / `monomial_to_legendre_matrix(n_max)` – 构建 Legendre 基与单项式基之间的转换矩阵。
* `gegenbauer_to_monomial_matrix(n_max, alpha)` – 生成 Gegenbauer 多项式到单项式的转换矩阵（Legendre 是其特例）。
* `associated_legendre_normalized(n, m, x)` – 计算全归一化缔合 Legendre 函数值，使用列递推。
* `associated_legendre_schmidt(n, m, x)` – 计算 Schmidt 半归一化缔合 Legendre 函数。

自定义异常：`SpecialFunctionError`。

## asteroid_geometry.py
实现小行星的三维形状建模，包括分形表面扰动和三角剖分。
* `sierpinski_ifs_transforms(scale)` – 返回 Sierpinski Carpet 的 8 个仿射变换，用于 IFS 迭代。
* `generate_fractal_profile(n_iterations, scale, seed)` – 生成分形表面轮廓点集（二维）。
* `polygon_area_2d(vertices)` – 用鞋带公式计算二维多边形面积。
* `is_collinear(v1, v2, v3, eps)` – 判断三点是否近似共线。
* `is_convex_vertex(poly, i)` – 判断多边形顶点是否为凸顶点。
* `point_in_triangle_2d(p, a, b, c)` – 检查点是否在三角形内部（重心坐标法）。
* `ear_clip_triangulation(poly)` – 实现耳切法，将简单多边形剖分为三角形，返回顶点索引数组。
* `generate_asteroid_cross_section(...)` – 生成椭圆基础加分形扰动的二维截面轮廓。
* `revolve_to_3d(poly2d, z_scale)` – 将二维轮廓绕 z 轴旋转生成三维顶点和三角面片。
* `polyhedron_volume_and_com(vertices, faces)` – 通过四面体累加计算多面体体积和质心。
* `triangle_area_3d(v1, v2, v3)` / `surface_area(vertices, faces)` – 计算三维三角形面积及多面体总表面积。

自定义异常：`AsteroidGeometryError`。

## gravity_harmonics.py
基于球谐展开计算小行星外部引力场。
* `compute_stokes_coefficients_from_shape(vertices, faces, density, n_max, r_ref)` – 由多面体形状和均匀密度，通过四面体数值积分估算球谐系数 Cnm 和 Snm，并返回参考半径。
* `SphericalHarmonicGravity` 类：
  * 构造时接收 GM、参考半径、C/S 系数矩阵和最大阶数。
  * `potential(pos)` – 计算给定位置处的引力势。
  * `acceleration(pos)` – 计算引力加速度向量（通过球坐标偏导数转换到笛卡尔坐标）。
  * `gradient_fd(pos, h)` – 用有限差分验证加速度。

依赖 `special_functions` 中的 `associated_legendre_normalized` 和 `factorial_ratio`。

## gravity_polyhedron.py
实现 Werner‑Scheeres 多面体引力模型，适用于近场精确计算。
* `edge_factor(r1, r2)` – 计算边贡献中的对数项。
* `face_solid_angle(r1, r2, r3)` – 计算三角形对面点的立体角。
* `polyhedron_gravity_potential(pos, vertices, faces, density, g_const)` – 由多面体模型计算外部引力势（求和面贡献和边贡献）。
* `polyhedron_gravity_acceleration(pos, vertices, faces, density, g_const, fd_step)` – 通过中心差分近似加速度。
* `combined_gravity_model(pos, ...)` – 组合多面体法与球谐法，利用平滑过渡权重在近/远场之间切换。

## fem_gravity.py
使用有限元方法求解小行星内部引力势分布的二维泊松方程。
* `wathen_element_matrix()` – 返回一个 8 节点单元的参考矩阵（实际未在后续装配中使用，仅作为参考）。
* `assemble_fem_system_2d(nx, ny, density_grid, g_const, dx, dy)` – 在规则矩形网格上组装二维泊松方程的刚度矩阵（采用 4 节点矩形单元）和右端项，施加 Dirichlet 边界条件，并将稀疏矩阵转换为带状下三角存储格式。
* `dense_to_r8blt(K, ml)` – 将稠密矩阵的下三角部分打包为 R8BLT 带状存储。
* `r8blt_sl(n, ml, a, b)` – 带状下三角前代求解器，直接求解线性方程组。
* `solve_internal_potential_2d(...)` – 高层接口：构建默认非均匀密度剖面，调用装配与求解，返回势场网格和坐标。
* `internal_gravity_from_potential(phi, x_coords, y_coords)` – 由势场数值梯度计算引力加速度分量。

自定义异常：`FEMGravityError`。

## orbit_integrator.py
提供小行星附近轨道的确定性及随机积分器。
* `line_ncc_rule(n, a, b)` – 生成 Newton‑Cotes 闭型求积公式的节点和权重。
* `newton_cotes_integrate(f, a, b, n, n_sub)` – 复合 Newton‑Cotes 数值积分。
* `srk4_ti_step(x, t, h, q, fi, gi)` – 时间不变系统的四阶随机 Runge‑Kutta 单步。
* `rk4_step(x, t, h, f)` – 经典四阶 Runge‑Kutta 单步。
* `OrbitalDynamics` 类：
  * 初始化时绑定引力加速度函数、太阳引力常数、SRP 系数等。
  * `deterministic_rhs(state, t)` – 组装右端项（引力 +
