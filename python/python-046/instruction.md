# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# InSAR 形变监测与断层滑动反演计算框架

本项目是一个面向地球物理 InSAR 断层滑动反演的 Python 科学计算库，由多个模块组成，协同完成从断层网格生成、有限元建模、动力学模拟，到正演计算、正则化反演和不确定性估计的完整科研流程。主入口为 `main.py`，它按固定流程调用各模块，验证结果并输出汇总报告。

你需要补全除 `main.py` 之外的所有源码文件，确保每个模块的功能、接口与后文描述一致。代码需要能够被 `main.py` 成功导入并运行通过。

## 模块概览

| 文件名 | 核心职责 |
|--------|----------|
| `utils.py` | 通用数值工具（有限性检查、归一化、三角形面积、正定性保障等） |
| `fault_geometry.py` | 断层面 2D/3D 网格生成与地表观测网格定义 |
| `fem_elasticity.py` | 二维弹性力学 T3 有限元求解器与一维 FEM 基函数插值 |
| `numerical_quadrature.py` | 一维/二维数值积分（复合梯形、高斯-勒让德、三角形高斯求积） |
| `sparse_matrix.py` | CCS 格式稀疏矩阵存储、矩阵向量乘法、矩阵链最优顺序 |
| `spectral_basis.py` | Legendre 和 Hermite 多项式基函数及其二维混合展开 |
| `rate_state_dynamics.py` | 速率-状态摩擦动力学（ODE 系统，弹簧-滑块模型） |
| `insar_forward.py` | Okada 弹性半空间位错模型与 InSAR LOS 正演投影 |
| `regularization.py` | 拉普拉斯平滑算子构造、Tikhonov/L‑curve/GCV 正则化参数选取 |
| `inversion_core.py` | Nelder‑Mead 优化器与断层滑动线性/非线性反演框架 |

## 各模块详细说明

### utils.py
- 提供一组建模过程中频繁使用的辅助函数。
- 关键函数：
  - `check_finite(arr, name)` — 检查数组是否全部有限，否则抛出异常。
  - `clip_to_range(x, xmin, xmax)` — 数值裁剪。
  - `normalize_vector(v)` — L2 归一化。
  - `compute_triangle_area(p1, p2, p3)` — 用行列式计算 2D 三角形面积。
  - `ensure_positive_definite(A, min_eig)` — 通过对特征值截断强制矩阵正定。
- 这些函数被几乎所有其他模块调用。

### fault_geometry.py
- 定义两个核心类：
  - `FaultMesh` — 表示一个矩形断层面的参数化网格。
    - 构造函数参数：沿走向长度、沿倾向宽度、走向角、倾角、走向/倾向方向结点数、是否自适应。
    - 内部生成 2D 局部坐标系（x 沿走向，y 沿倾向）下的结点 `nodes` 和三角形单元 `elements`（由四边形拆分或 Delaunay 三角化得到）。
    - 如果 `adaptivity=True`，采用 CVT（重心 Voronoi 图）思路，使用密度函数（在断层中部加权）和简化 Lloyd 松弛调整采样点，最后通过 `scipy.spatial.Delaunay` 生成三角形网格。
    - 主要方法：`_build_regular_quadrilateral_mesh`（均匀四边形拆分），`_build_cvt_adaptive_mesh`（自适应），`_mark_boundary_nodes`（标记边界结点），`get_element_centroids`（计算三角形形心），`map_to_3d`（将 2D 坐标转换到三维空间），`element_areas`（计算每个三角形面积）。
  - `SurfaceGrid` — 定义矩形区域上的等间距地表观测网格。
    - 构造函数参数：x / y 范围、nx / ny（像素数）。
    - 生成 `points` 属性，`(N,2)` 数组，按网格展开。

### fem_elasticity.py
- 提供平面应变线性弹性有限元计算和一维 FEM 基函数工具。
- 类：
  - `FEMElasticity2D`
    - 接受结点 `nodes`、三角形单元 `elements`、杨氏模量 `E`、泊松比 `nu` 进行初始化。
    - 内部存储 Lamé 常数和平面应变弹性矩阵 `D`（3×3）。
    - 核心方法：`assemble_stiffness_matrix`（组装刚度矩阵，使用 T3 常数 B 矩阵，积分用单元面积乘局部刚度矩阵），`assemble_mass_matrix`（组装集中质量矩阵），`apply_dirichlet_bc`（修改 K 和 F 施加 Dirichlet 条件），`solve_static`（求解 K u = F 得到位移场），`compute_stress_at_elements`（根据位移计算单元应力）。
    - 私用方法：`_t3_basis_derivatives`（计算 T3 三角形基函数的导数），`_build_B_matrix`（构造应变-位移 B 矩阵）。
  - `FEM1DBasis`
    - 类方法：`local_basis_1d(order, node_x, eval_x)` — 用拉格朗日插值构造一维基函数值。
    - `interpolate_1d(node_x, node_v, eval_x)` — 使用基函数进行有限元插值。

### numerical_quadrature.py
- 提供多种数值积分规则，用于有限元刚度矩阵、断层积分等。
- 函数：
  - `composite_trapezoidal(f, a, b, n)` — 复合梯形积分。
  - `gauss_legendre_integral(f, a, b, n)` — n 点 Gauss‑Legendre 求积（包含内部辅助 `gauss_legendre_nodes_weights`）。
  - `triangle_gauss_rule(order)` — 返回参考三角形上的求积节点和权重，支持 1、3、7 点规则。
  - `integrate_over_triangle(f, p1, p2, p3, order)` — 将物理三角形映射到参考三角形进行积分。
  - `integrate_2d_grid(f, xlim, ylim, nx, ny)` — 二维矩形上的复合梯形积分。

### sparse_matrix.py
- CCS（压缩列存储）稀疏矩阵格式，用于处理大型有限元系统或格林函数矩阵。
- 类 `CCSMatrix`：
  - 属性：`m, n, colptr, rowind, data`。
  - 构造时检查数组一致性。
  - 方法：`multiply_vector(x)` 计算 A·x，`transpose_multiply_vector(x)` 计算 A^T·x，`to_dense()` 转回稠密矩阵，静态方法 `from_dense(A, tol)` 从稠密矩阵构造 CCS。
- 函数：
  - `matrix_chain_optimal_order(dims)` — 动态规划求矩阵链乘法最优顺序。
  - `build_parenthesization(split, i, j)` — 重构括号化表达式字符串。
  - `sparse_matrix_multiply_chain(matrices)` — 顺序相乘列表中的矩阵（支持稠密和 CCS 混合）。
- 类 `SparseLinearOperator` — 包装 CCS 矩阵，提供迭代求解器接口。

### spectral_basis.py
- 实现 Legendre 与概率论 Hermite 多项式的计算，并混合两者构造二维谱基函数。
- 函数：
  - `legendre_polynomial_values(m, n, x)` — 计算 P0…Pn 在样本点 x 上的值（递推公式）。
  - `legendre_polynomial_derivative(m, n, x)` — 同理计算导数。
  - `hermite_probabilist_coefficients(n)` — 返回 He_n 的系数（递推或系数表）。
  - `hermite_probabilist_values_array(max_n, x)` — 计算 He0…He_max_n 在点 x 上的值。
  - `mixed_legendre_hermite_basis_2d(x
