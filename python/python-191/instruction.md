# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目：python-191 — MPI 并行科学计算矩阵乘法综合验证平台

本项目构建一个博士级科学计算代码库，核心主题为“高性能计算：MPI 并行矩阵乘法”。项目融合了 15 个种子项目的关键算法，通过一个统一入口 `main.py` 进行自动实验验证。除了保留的 `main.py`，其余所有支撑模块需要被复现。这些模块涵盖了并行矩阵乘法、矩阵指数与天体力学、高级插值与根查找、数值积分、有限元拓扑、概率采样、素数分布负载均衡、以及多项式哈希校验等多个领域。

## 需要复现的模块文件

以下每个 Python 文件各自完成一个独立领域的功能，它们被 `main.py` 调用以运行预设的实验序列。请为每个文件编写代码实现所述功能，保持与原始设计一致的接口行为。

### 1. `mpi_cannon_multiply.py` — MPI 风格并行稠密矩阵乘法
- **职责**：实现 Cannon 算法和 SUMMA 算法的单机多进程模拟，用于并行稠密矩阵乘法。
- **主要功能**：
  - `cannon_multiply(A, B, num_processes)`：按照 Cannon 算法执行 C = A @ B。要求：将矩阵在 sqrt(p)×sqrt(p) 的进程网格上进行分块，进行初始对齐（A 块左移，B 块上移），然后通过 q 步的本地乘法与循环移位完成计算。当矩阵尺寸过小时应回退到串行 `numpy.dot`；若尺寸不能整除进程网格边长，需要填充或截断。
  - `mpi_summa_multiply(A, B, num_processes)`：SUMMA 算法实现，通过在块多列上广播 A 的列块和 B 的行块执行外积累加。
  - `frobenius_error(C_parallel, C_reference)`：计算两个矩阵的相对 Frobenius 范数误差。
  - 辅助函数：用于选取最接近完全平方数的进程数，以及块乘法的工作函数。
- **关键要求**：使用 Python 的 `multiprocessing` 进行并行化；正确处理进程通信模拟（通过数据拷贝实现移位）；确保浮点结果误差在合理范围。

### 2. `matrix_exponential_ode.py` — 矩阵指数与开普勒变分方程
- **职责**：实现矩阵指数的缩放平方 Padé 近似，以及 Kepler 二体问题的状态转移矩阵（STM）积分与扰动传播。
- **主要功能**：
  - `matrix_exponential_pade(A, order)`：采用 [order/order] Padé 逼近计算 exp(A)，配合 2 的幂次缩放保证范数条件。
  - 开普勒相关函数：
    - `kepler_derivatives(state)`：返回 Kepler Hamilton 系统的导数（位置 q 导数等于动量 p，p 的引力加速度）。
    - `kepler_hessian(state)`：计算 Hamilton 量对位置变量的 Hessian 矩阵。
    - `kepler_variational_matrix(state)`：组装 4×4 变分方程矩阵 M，使得状态转移矩阵满足 dPhi/dt = M @ Phi。
    - `integrate_kepler_stm(y0, t_span, n_steps)`：使用经典 4 阶 Runge-Kutta 方法同时积分 4 维状态和 16 维 STM，返回终态和 STM 矩阵。
    - `propagate_perturbation(y0, delta_y0, t, n_steps)`：利用 STM 传播初始扰动。
    - `exponential_growth_rate(A)`：计算矩阵的最大实部特征值，表征指数增长率。
- **关键要求**：正确实现 RK4 积分器，正确处理 STM 的初始化为单位矩阵；Kepler 导数在碰撞附近应有数值保护（如设置最小距离阈值）。

### 3. `chebyshev_hermite_approx.py` — Chebyshev 代理根查找与 Hermite 插值
- **职责**：基于 Chebyshev 多项式对光滑函数进行代理逼近，通过伴随矩阵求根（CPR 方法）；利用 Hermite 插值（含导数值）进行核矩阵元素近似。
- **主要功能**：
  - `chebyshev_nodes(a, b, N)`：生成区间 [a,b] 上的 N+1 个第二类 Chebyshev 节点。
  - `chebyshev_coefficients(f_vals)`：从节点函数值计算 Chebyshev 系数（离散余弦变换风格）。
  - `chebyshev_companion_matrix(c)`：根据截断后的 Chebyshev 系数构造伴随矩阵，供特征值法求根。
  - `cpr_roots(f, a, b, N, tau, sigma)`：对函数 f 在 [a,b] 上执行 CPR 根查找，返回实根和插值残差估计。需支持向量化函数。
  - `hermite_divided_differences(x, y, yp)`：为 Hermite 插值构造扩展节点和差商表。
  - `hermite_evaluate(z, d, x_eval)`：用牛顿形式计算 Hermite 插值多项式的值。
  - `approximate_matrix_element(kernel, xi, xj, order, method)`：通过 Chebyshev 或 Hermite 方法近似核函数在两个变量方向上的元素值。
- **关键要求**：伴随矩阵的构造要遵循标准 Chebyshev 同伴矩阵形式；插值过程注意区间映射；Hermite 差商表需处理重复节点导数条件。

### 4. `legendre_quadrature_kernel.py` — Gauss-Legendre 快速求积与圆盘积分
- **职责**：实现 Gauss-Legendre 求积节点的快速算法（GLR），并提供单位圆盘的高阶积分公式，用于核矩阵组装。
- **主要功能**：
  - `legendre_compute_glr(n)`：用 GLR 算法计算 n 个 Gauss-Legendre 节点和权重。需实现 Legendre 多项式的递推求值及导数，并利用渐近初始猜测和 Newton 迭代精确求解节点。
  - `rescale_quadrature(x, w, a, b)`：将 Legendre 规则从 [-1,1] 缩放到任意区间。
  - `disk01_rule(nr, nt)`：构造单位圆盘积分规则，径向使用 Legendre 节点经变量变换求得，角度均匀分布。返回径向权重、径向节点、角度节点。
  - `integrate_disk_kernel(kernel, nr, nt)`：利用圆盘规则积分一个二元函数。
  - `construct_kernel_matrix_1d(nodes, kernel_func, quadrature_order)`：用 GL 求积在 1D 上构造 L2 核矩阵。
  - `construct_kernel_matrix_2d_disk(nodes, kernel_func, nr, nt)`：在单位圆盘上构造 2D 核矩阵（如 RBF 核）。
- **关键要求**：GLR 算法中 Legendre 导数在端点需妥善处理；圆盘积分权重需归一化；核矩阵构造支持给定节点和基函数。

### 5. `sparse_fem_topology.py` — 有限元网格拓扑与稀疏刚度矩阵
- **职责**：管理三角形面网格，计算几何属性和节点连通性，并组装二维 Laplacian 刚度矩阵（COO 格式）。同时提供受“trinity 拼图”启发的稀疏覆盖模式生成。
- **主要类与函数**：
  - `TriangularMesh` 类：存储节点坐标和三角形元素索引；提供方法：验证拓扑有效性、计算节点度数、计算三角形面积、构建 CSR 风格的邻接模式、以及基于 XY 投影的二维 Laplacian 刚度矩阵组装。
  - `trinity_tile_cover_pattern(region_triangles, tile_types)`：生成两个二进制稀疏矩阵 A1（覆盖约束）和 A2（瓦片使用约束），模拟覆盖问题中的线性系统模式。
  - `sparse_matrix_vector_product(data, row_ind, col_ind, x, n_rows)`：COO 格式的稀疏矩阵向量乘法。
  - `sparsity_ratio(data, n_rows, n
