# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 异构HPC热‑电耦合模拟与调度系统

本项目是一个集成式的科学计算系统，模拟异构高性能计算平台上的任务调度，同时耦合温度场求解和不确定性量化。整个系统由多个模块组成，主入口 `main.py` 调用各模块完成完整的计算流程：构造异构平台、生成随机任务、训练性能代理模型、进行蒙特卡洛不确定性分析和数值积分、求解有限元热场并进行网格自适应，最后执行贪心调度与局部搜索优化。

## 项目文件说明

### `main.py`
项目入口脚本。零参数运行，依次调用其它模块完成以下流程：
1. 2D 有限元热求解与误差分析、网格自适应细化。
2. 蒙特卡洛不确定性量化（椭圆采样、超球面积分、对偶变量方差缩减等）。
3. 性能代理模型训练与预测（Chebyshev 插值和最小二乘拟合）。
4. 随机任务集生成与可靠性评估。
5. 数值积分演示（Vandermonde 求积、金字塔积分、复合2D积分及误差估计）。
6. 异构平台构建、拓扑变换、贪心调度与局部搜索改进，输出调度指标。

运行 `main.py` 时会导入以下所有模块，因此删掉其余 `.py` 文件后，复现者需要根据本描述实现这些模块，以保证 `main.py` 正常运行。

---

### `utils.py`
提供通用数学与数值工具，供多个模块调用。

**主要函数：**
- `rref_matrix(a, tol=None)`：计算矩阵的简化行阶梯形 (RREF) 以及一个伪行列式值。返回变换后的矩阵和行列式。
- `hypersphere_surface_area(dim)`：返回 dim 维单位超球面的表面积。
- `hypersphere_volume(dim)`：返回 dim 维单位超球体的体积。
- `check_positive_definite_symmetric(mat, tol)`：判断方阵是否为对称正定矩阵（基于特征值或 Cholesky 尝试）。
- `cholesky_factor(a)`：对对称正定矩阵进行 Cholesky 分解，返回上三角矩阵 U，使得 A = UᵀU。
- `safe_log(x)`、`safe_sqrt(x)`：对极小值或负数做安全处理的 log 和 sqrt。
- `binomial_coeff(n, k)`：计算二项式系数 C(n, k)，采用迭代乘积以避免大数溢出。

---

### `mesh_transform.py`
网格几何变换与自适应细化工具。提供 2D 仿射变换函数和三角形网格质量评估与细化逻辑。

**主要函数：**
- `rotation_matrix_2d(angle_rad)`：返回 2D 旋转矩阵（绕原点）。
- `dilation_matrix_2d(sx, sy)`：返回 2D 缩放矩阵。
- `translation_vector_2d(tx, ty, n_points)`：生成 n_points 个重复的平移向量。
- `affine_transform_2d(points, A, b)`：对形状 (2, N) 的点集执行仿射变换 y = A x + b。A 默认为单位阵，b 默认零。
- `transform_mesh(nodes, elements, A, b)`：对有限元网格的节点施加仿射变换，保持单元拓扑不变。
- `polygon_surface_quality(nodes, elements)`：计算每个三角形单元的质量指标（与等边三角形的接近程度）。返回质量数组、最小质量和平均质量。节点索引视为 1‑based。
- `adaptive_refinement_markers(nodes, elements, gradient_field, threshold_ratio)`：根据节点上的梯度场（形状 (2, N)）计算每个单元的平均梯度模，标记梯度最大的阈值比例单元为需细化。
- `refine_marked_elements(nodes, elements, marker)`：对被标记的三角形单元执行 1‑to‑4 细化（连接边中点生成四个子三角形）。维护边中点映射，避免重复添加节点。返回新节点数组和新单元数组（索引 1‑based）。

---

### `heterogeneous_platform.py`
异构计算平台建模，包括处理器性能与功耗模型、平台拓扑与通信延迟。

**类 `Processor`：**
属性：处理器 ID、类型 ('CPU','GPU','FPGA','TPU')、峰值浮点性能 (GFLOPS)、内存带宽 (GB/s)、空闲/峰值功耗 (W)、热阻 (K/W)、2D 位置坐标、当前温度 (K)、利用率 (0‑1)。
- `effective_performance(ambient_temp, alpha_thermal)`：根据功耗计算当前温度，并返回考虑温度降频后的有效性能（峰值 × 降频因子）。
- `execution_time(workload_flops, compute_intensity)`：基于 Roofline 模型估计执行时间，同时考虑计算瓶颈和访存瓶颈。

**类 `HeterogeneousPlatform`：**
包含多个 Processor 实例、环境温度、通信延迟矩阵（基于处理器间欧氏距离的简化模型）、拓扑变换状态。
- `add_processor(proc)`：添加处理器。
- `build_default_platform()`：构建一个包含 2 个 CPU、1 个 GPU、1 个 FPGA 的默认平台，并建立通信矩阵。
- `apply_topology_transform(A, b)`：对全体处理器位置施加仿射变换，更新通信矩阵，并记录累积变换。
- `rotate_topology(angle_deg)`、`scale_topology(sx, sy)`：调用 `mesh_transform` 中的辅助函数生成变换矩阵，然后应用拓扑变化。
- `get_total_power()`：计算平台当前总功耗。
- `reset_utilization()`：将所有处理器利用率置零。
- `snapshot_state()`：返回包含各处理器温度、有效性能、总功耗的字典。

---

### `fem_thermal_solver.py`
二维稳态热传导有限元求解器，使用线性三角形单元。

**主要函数：**
- `build_rectangular_mesh(nx, ny, xl, xr, yb, yt)`：在矩形区域上生成由三角形组成的均匀网格。节点按列优先排列，索引从 1 开始。返回节点坐标数组 (2, node_num) 和单元节点数组 (3, element_num)。
- `fem2d_poisson_solve(nx, ny, source_func, exact_func, ..., conductivity)`：求解方程 `-k ∇² u = f`，边界条件由 `exact_func(x,y) -> (u, dudx, dudy)` 确定 Dirichlet 值。组装时使用三角形线性形函数，积分采用边中点三点求积规则。若提供精确解则据此施加边界并计算 L2 和 H1 误差。返回：数值解 u、节点坐标、单元拓扑、L2 误差、H1 误差。
- `extract_gradient_at_nodes(u, node_xy, element_node)`：从解 u 中恢复节点梯度，通过对每个单元的常量梯度按面积加权平均。

---

### `monte_carlo_uq.py`
蒙特卡洛不确定性量化模块，用于在高维参数空间中采样并估计统计量。

**主要函数：**
- `uniform_in_sphere01_map(dim_num, n, rng)`：在 dim_num 维单位球内均匀采样（利用标准正态方向向量和径向逆变换）。返回形状 (dim_num, n) 的点集。
- `ellipse_sample(n, A, r, rng)`：在二维椭圆 {x | xᵀ A x ≤ r²} 内均匀采样，依靠 Cholesky 分解将单位球点映射到椭圆。
- `ellipse_area(A, r)`：返回上述椭圆的面积。
- `hypersphere01_monomial_integral(dim, expon)`：计算单位超球面上单项式 ∫ ∏ x_i^{e_i} dS 的精确值（利用 Gamma 函数，奇数指数时积分为零）。
- `hypersphere_monte_carlo_integral(dim, n_samples, func, rng)`：用 Monte Carlo 方法估计超球面上函数的积分，返回积分值和标准误差。
- `hypercube_distance_stats(dim, n_samples, rng)`：估算 dim 维超立方体内两个随机均匀点欧氏距离的均值和方差。
- `antithetic_variates_integral
