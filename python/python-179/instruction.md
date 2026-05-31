# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目概述

本项目是一个综合性的计算数学实验框架，名为“参数化反应‑扩散‑形态演化系统的高维响应面低秩张量近似”。其核心目标是通过参数化一维反应‑扩散方程求解，构造一个包含空间、时间及多个物理/几何参数的高维解张量，并利用张量列车（Tensor Train, TT）分解进行低秩压缩与分析。

整个项目由多个功能模块构成，主入口为 `main.py`。后续将保留 `main.py`，删除其余 `.py` 文件，需要其他 agent 根据本描述重新实现所有被依赖的模块。

# 模块描述

## system_utils.py —— 系统初始化与数值工具

提供全局常量（如机器精度 `EPS`、秩阈值 `TOL_RANK`）、一个带时间戳的 `Logger` 类，以及若干数值安全函数（`robust_sqrt`、`clip_to_range`、`check_finite` 等）。`initialize_project()` 返回一个 `Logger` 实例，用于整个运行的日志记录。

## domain_generator.py —— 参数化计算域生成

负责生成具有生物形态特征的一维或二维计算域，融合了鸟类蛋形参数化轮廓与手部合成轮廓。

- `chebyshev_nodes_1d(a, b, n)`：在区间 `[a,b]` 上生成第一类 Chebyshev 节点，用于抑制 Runge 现象。
- `chicken_egg_half_profile(B, L, w, x)`：鸡形蛋的半高轮廓，基于最大宽度 `B`、长度 `L` 和偏移 `w` 的解析公式。
- `pyriform_egg_half_profile(B, L, w, x)`：梨形蛋的半高轮廓。
- `universal_egg_half_profile(B, L, w, D, x)`：通用蛋形轮廓，通过混合参数 `D` 调节鸡形与梨形的比例。
- `generate_parametric_radial_domain(n, B, L, w, D)`：生成基于通用蛋形的 Chebyshev 节点径向坐标，用作一维有限元网格。
- `compute_radial_cross_section(x, B, L, w, D)`：计算给定轴向坐标处的径向半径。
- `hand_outline_polygon(n_points)`：利用 Fourier 描述子生成一个合成手部轮廓的多边形顶点。
- `hand_ellipse_fourier_approx(n_points)`：返回手部轮廓及其 SVD 的前若干主成分方向。

## fem_discretization.py —— 一维有限元离散化

实现一维分段线性（hat 函数）有限元，用于求解扩散方程的空间离散。

- `hat_basis(x, xi)`：计算 hat 函数基在任意点 `x` 处的值。
- `assemble_fem_matrices_1d(nodes, diffusion_coeff)`：组装质量和刚度矩阵（对称三对角），采用局部单元矩阵公式。
- `extract_tridiagonal(A)`：将对称三对角矩阵转换为紧凑 3×n 格式。
- `integrate_on_2d_section(r_nodes, z_nodes, f_values)`：在二维轴对称截面的三角形网格上使用重心 Gauss 求积计算积分。
- `project_function_to_fem(nodes, func)`：将连续函数插值投影到 FEM 节点空间。
- `fem_l2_norm(nodes, u, M)`：通过质量矩阵计算离散 L² 范数。

## reaction_kinetics.py —— 反应动力学源项

提供参数化的反应项，用于反应‑扩散方程的源项（R(u)）。

- `twoway_reaction_term(u, k1, k2)`：双向一级线性反应速率项。
- `twoway_exact_solution(t, u0, k1, k2)`：双向反应的精确解（用于验证）。
- `vanderpol_reaction_term(u, mu)`：范德波尔型非线性反应项。
- `parametric_reaction_source(u, k1, k2, mu, mix_ratio)`：混合线性与非线性反应的参数化源项，由混合权重 `mix_ratio` 控制。
- `reaction_jacobian_diagonal(u, k1, k2, mu, mix_ratio)`：反应项关于 `u` 的对角 Jacobian。

## tridiagonal_solver.py —— 三对角矩阵求解器（R83 格式）

以紧凑 3×n 数组存储三对角矩阵，支持多种迭代求解器。

- `r83_mv(A, x)` / `r83_mtv(A, x)`：矩阵‑向量乘法及其转置。
- `r83_dif2(n)`：构造一维离散 Laplacian（-1, 2, -1 模板）。
- `r83_res(A, x, b)`：残差计算。
- `r83_cg(A, b, ..., tol, max_iter)`：共轭梯度法（CG）求解 SPD 三对角系统。
- `r83_cr_fa(A)` / `r83_cr_sl(fac, b)`：循环约化法的因子分解与回代求解。
- `r83_jac_sl(A, b, ...)`：Jacobi 迭代求解。
- `r83_gs_sl(A, b, ...)`：Gauss‑Seidel 迭代求解。

## randomized_sketching.py —— 随机采样与低秩近似

用于随机化 SVD、测试矩阵生成和环形域随机采样。

- `annulus_sample(n, pc, r1, r2)`：在二维环形域内均匀生成随机点。
- `randomized_range_finder(A, k, p)`：通过高斯随机投影和 power iteration 找到矩阵列空间的近似基。
- `randomized_svd(A, k, p)`：基于随机范围寻找的截断 SVD。
- `hilbert_matrix(m, n)`：生成病态 Hilbert 矩阵，用于验证低秩近似算法。
- `low_rank_test_matrix(m, n, rank)`：生成已知精确秩的测试矩阵。
- `adaptive_rank_threshold(s, tol)`：基于奇异值衰减确定有效数值秩。

## quadrature_integrator.py —— 高维数值积分与误差估计

- `van_der_corput_sequence(n, base)`：生成一维 van der Corput 序列。
- `hammersley_sequence(n, d)`：生成 d 维 Hammersley 低差异序列。
- `grid_integrate_1d(f, a, b, n)`：一维复合梯形积分。
- `monte_carlo_integrate(f, dim, n, bounds, seed)`：标准 Monte Carlo 积分，返回估计值和标准误差。
- `qmc_integrate(f, dim, n, bounds)`：基于 Hammersley 序列的拟 Monte Carlo 积分。
- `estimate_tensor_frobenius_norm_mc(tensor, n_samples)`：通过随机采样估计稠密张量的 Frobenius 范数。

## rank_analysis.py —— 代数秩分析与 Hankel 张量

- `rref_compute(A, tol)`：计算简化行阶梯形（RREF）及主元列。
- `rref_rank(A, tol)` / `rref_columns(A, tol)`：通过 RREF 计算数值秩或提取列空间基。
- `unfold_tensor(tensor, mode)`：张量的 mode‑k 展开（matricization）。
- `tensor_multilinear_ranks(tensor, tol)`：计算张量的多线性秩（各 mode 展开的秩）。
- `estimate_tensor_train_ranks(tensor, tol)`：估计张量列车格式所需的 TT‑秩。
- `collatz_polynomial_next(p)` / `collatz_polynomial_sequence(p0, max_steps)`：在 GF(2) 上生成 Collatz 多项式序列。
- `build_hankel_tensor_from_sequence(seq, dimensions)`：从一维序列构造高阶 Hankel 张量。
- `hankel_matrix_from_sequence(s, m, n)`：构造 Hankel 矩阵。

## nmf_initializer.py —— 非负矩阵/张量分解初始化

- `random_contingency_table(nrow, n
