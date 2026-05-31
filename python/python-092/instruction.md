# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：室内声场射线追踪与模态分析系统

本项目是一个三维室内声学仿真工具集，结合低频有限元（FEM）和高频几何声学（射线追踪）方法，并包含贝叶斯参数反演。代码分为多个模块，`main.py` 作为入口脚本编排整个分析流程：从房间几何定义、四面体网格生成、Helmholtz 方程求解、模态分析、射线追踪、边缘衍射计算，到基于 DREAM MCMC 的吸声系数反演和不确定性传播。

以下各模块描述对应将被删除的 `.py` 文件。每个模块需要提供特定的类和/或函数，其接口已在 `main.py` 的导入语句中体现。你需要根据职责描述和 `main.py` 中的调用方式重新实现这些文件，实现细节可自由发挥，但必须保持整体功能一致。

---

## room_geometry.py
**职责**：房间几何建模与表面预处理。
- 提供有向距离函数（SDF），用于定义带有立柱的吸收性长方体房间。核心函数是 `dshoebox_with_pillars(p)` 和一个均匀网格尺寸函数 `huniform(p)`。
- 提供 `extract_room_surfaces()` 提取房间六个表面的三角形顶点数据。
- 提供 `compute_surface_normals(surfaces)` 计算各表面外法向量。
- 提供 `room_surface_areas(surfaces)` 和 `room_total_volume()` 分别计算各表面面积和房间总体积（需考虑立柱体积）。
- 提供 `compute_sabine_reverberation_time(absorption_coeffs, surfaces)` 基于 Sabine 公式计算混响时间 T60。
- 内部使用几何工具函数，如长方体/球体 SDF 的组合（集合差、交集）。

## mesh_generator.py
**职责**：三维非结构化四面体网格生成与局部加密。
- 实现 `distmesh_3d(fd, fh, h0, box, ...)`，基于 DistMesh 算法，通过力平衡迭代和 Delaunay 再三角化生成四面体网格。输入有向距离函数和网格尺寸函数，输出节点数组和四面体索引数组。
- 提供 `refine_mesh_near_boundary(p, t, fd, h0)` 在距离函数接近零的区域进行局部细化。
- 提供 `mesh_statistics(p, t)` 返回节点数、单元数、体积范围、质量统计等。
- 提供 `surftri(p, t)` 从四面体网格提取边界三角形面。
- 提供 `simp_qual_3d(p, t)` 计算四面体单元质量（内切球与外接球半径比）。
- 依赖 `scipy.spatial.Delaunay` 进行四面体剖分。

## fem_acoustics.py
**职责**：三维 Helmholtz 方程有限元求解。
- 定义物理常数 `C_AIR`（声速）。
- 实现形函数、梯度、单元雅可比计算等基础函数。
- 核心函数 `assemble_helmholtz_system(p, t, freq, ...)` 组装全局刚度矩阵 K、质量矩阵 M 和载荷向量 F，形成系统 `(K - k² M) p = F`，其中 k 为波数。返回 COO 格式的稀疏矩阵（来自 `sparse_linalg` 的 `SparseCOO`）以及波数 k。支持单点频域源。
- 提供 `solve_helmholtz_cg(p, t, freq, ...)` 利用共轭梯度法求解近似 SPD 系统（取实部），用于验证流程。
- 提供 `compute_sound_pressure_level(pressure)` 由复数声压计算声压级 (SPL)。
- 提供 `compute_intensity(pressure, p, t, freq)` 基于 FEM 节点解计算单元中心的声强向量。
- 依赖 `sparse_linalg` 模块的 `SparseCOO`、`assemble_sparse_from_triplets` 和 `conjugate_gradient`，以及内部求积规则。

## sparse_linalg.py
**职责**：稀疏矩阵代数运算库。
- 定义 `SparseCOO` 类，表示坐标格式（COO）的稀疏矩阵，包括属性 `rows`、`cols`、`vals`、`shape`。提供方法：矩阵-向量乘法 `mv()`、转置乘法 `mtv()`、残差计算、转为稠密矩阵等。
- 提供函数 `assemble_sparse_from_triplets(rows, cols, vals, n)`，通过累加相同索引的三元组构建 `SparseCOO` 矩阵。
- 提供独立函数 `conjugate_gradient(A_sparse, b, ...)` 求解 SPD 线性系统，遵循标准 CG 迭代。
- 还提供 `jacobi_iteration`、Cholesky 分解 `r8po_fa`、基于 Cholesky 因子的求解 `r8po_sl`、行列式和逆矩阵计算以及 Jacobi 迭代求解器。

## modal_analysis.py
**职责**：声学模态分析与特征频率求解。
- 提供 `rectangular_room_modes(Lx, Ly, Lz, max_order)` 解析计算刚性壁长方体房间的模态频率 (l, m, n)。
- 提供 `schroeder_frequency(room_volume, total_surface_area, absorption_coeff_avg)` 估算 Schroeder 频率，区分模态控制区与扩散区。
- 提供 `inverse_iteration(K_sparse, M_sparse)` 逆迭代法求解广义特征值问题的最小特征对，返回频率、模态向量和 Rayleigh 商。内部使用 `sparse_linalg` 的 CG 求解器。
- 提供 `zero_rc_brent(func, a, b)` 实现 Brent 求根法，用于特征频率精细搜索（可选实现）。
- 提供 `compute_modal_density(room_volume, freq)` 和 `modal_overlap_factor(modes, damping_ratio)` 计算模态密度和模态重叠因子。
- 提供 `rayleigh_quotient_iteration` 加速模态改进（可选）。

## quadrature_rules.py
**职责**：高阶数值积分规则。
- 实现单位三角形上的对称求积规则 `triangle_symq_rule(precision)`，返回一组重心坐标和权重，支持多个精度等级。
- 提供 `integrate_over_triangle(func, v0, v1, v2, precision)` 利用上述规则在任意三角形上积分标量函数。
- 实现单位球内的精确单项式积分 `ball01_monomial_integral(e)`，基于 Gamma 函数公式。
- 提供 `ball01_sample(n)` 在单位球内均匀采样，用于蒙特卡洛积分和射线方向采样。
- 提供 `ball01_volume()` 返回单位球体积。
- 还包含线段采样工具：`line01_sample_ergodic`（黄金比例低差异序列）等。

## ray_tracer.py
**职责**：室内高频声场的蒙特卡洛射线追踪。
- 实现 `sobol_generate(m, n)` 生成低差异序列（可近似）；`sample_directions_sobol(n_rays)` 在球面上均匀采样射线方向。
- 提供 `ray_plane_intersection()`、`reflect_direction()`、`scatter_direction()` 等几何计算函数。
- 核心函数 `trace_ray(...)` 追踪单条射线的多次反射，记录路径长度、能量衰减和命中的表面序列，考虑散射效应。
- 提供 `monte_carlo_ray_tracing(surfaces, normals, absorption, source_pos, ...)` 发射多条射线，累积能量衰减曲线 (EDC)，并通过 Schroeder 反向积分法计算 T60 和早期衰减时间 EDT。
- 提供 `build_reflection_graph(surfaces, normals, absorption, n_rays)` 通过射线统计构建表面间能量转移概率矩阵，并返回表面名称列表。
- 提供 `compute_room_response_stats(...)` 计算平均自由程、平均反射次数等统计量。
- 依赖 `quadrature_rules.line01_sample_ergodic` 可能用于参数化（非硬性）。

## edge_diffraction.py
**职责**：基于几何衍射
