# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目：python-175 – 随机椭圆 PDE 的广义多项式混沌展开与不确定性量化

## 总体目标

本项目是一个综合性的科学计算基准，模拟处理不确定性传播与分析的工作流。核心问题是一维随机椭圆偏微分方程（PDE），扩散系数 \(a(x,\xi)\) 随机依赖一列独立随机变量 \(\xi = (\xi_1,\dots,\xi_d)\)（均匀或高斯分布）。项目通过对随机空间采用广义多项式混沌（gPC）展开，在空间上进行有限元离散，组合形成伪谱投影或侵入式 Galerkin 方法求解随机 PDE。进一步工作包括：计算输出统计矩和全局敏感性指数（Sobol 指数），执行贝叶斯参数推断（使用 DREAM MCMC 采样器），以及进行混沌敏感性、谱稳定性和收敛性诊断。

代码被组织为多个独立的子模块，每个子模块实现一组特定的数学算法或工具。随后这些模块被 `main.py` 中的演示流程调用，该流程依次运行自测并展示主要工作流步骤。**在后续的基准测试中，将只保留 `main.py`，而其他所有 `.py` 文件将被删除。参与者需要根据本描述重新实现这些缺失的模块，使得 `main.py` 能够无错误运行并输出正确的结果。**

## 文件结构与模块功能

缺失的模块及各自的职责如下：

### 1. `orthogonal_polynomials.py` – 正交多项式与 Gauss 求积

实现一维经典正交多项式的求值和系数生成，以及通过 Jacobi 矩阵方法计算 Gauss 节点与权重。

- **关键函数/接口**：
  - `legendre_eval(n, x)`、`chebyshev_eval(n, x)`、`hermite_eval(n, x)`：分别计算 n 阶 Legendre、第一类 Chebyshev、物理学家 Hermite 多项式在标量或数组 x 上的值（使用三项递推，避免数值不稳定）。
  - `legendre_coeffs(n)`、`chebyshev_coeffs(n)`、`hermite_coeffs(n)`：返回对应多项式的单项式系数（从低次到高次）。
  - `poly_horner(coeffs, x)`：利用 Horner 算法计算由系数给出的多项式在点 x 的值。
  - `normalized_legendre_eval(n, x)`：计算归一化的 Legendre 多项式值，使其在 \([-1,1]\) 上与权重 1 构成标准正交基。
  - `gauss_legendre_nodes_weights(n)`：通过构造对称三对角 Jacobi 矩阵并求特征值，计算 n 点 Gauss-Legendre 求积节点和权重。
  - `gauss_chebyshev_nodes_weights(n)`：类似于 Gauss-Chebyshev。
  - `polynomial_roots_via_companion(coeffs)`：将多项式的系数转换为伴随矩阵，通过特征值分解求所有根。
  - `jacobi_matrix_legendre(n)`、`jacobi_matrix_chebyshev(n)`：构建相应的 Jacobi 矩阵（内部辅助函数）。
  - `test_orthogonal_polynomials`：自测函数，检查正交性等性质。

### 2. `multidim_polynomial.py` – 多维多指标集代数

处理 gPC 中所需的多指标（multi-index）的枚举、排序和求值。

- **关键函数/接口**：
  - `multi_index_total_degree(alpha)`：返回多指标的总次数。
  - `multi_index_grlex_compare(alpha, beta)`：按 graded lexicographic 次序比较两个多指标。
  - `multi_index_rank_grlex(alpha, d)` 和 `multi_index_unrank_grlex(rank, d)`：基于组合公式实现多指标在 grlex 序下的排位与反演。
  - `enumerate_multi_indices_grlex(d, max_degree)`：生成所有总次数 ≤ max_degree 的 d 维多指标，并按 grlex 排序。`enumerate_multi_indices_total_degree` 是其别名。
  - `sparse_grid_index_set(d, level, rule)`: 根据指定规则（`"tensor"`、`"total"`、`"hyperbolic"`）生成稀疏指标集。总次数截断规则使用 grlex 枚举，双曲截断使用乘积条件，全张量乘积生成完全网格。
  - `multivariate_orthogonal_basis(alpha, xi, poly_eval_1d)`：计算多维正交多项式基函数的值，即一维基函数按维度的乘积。`poly_eval_1d` 是一个接受次数和采样点的可调用对象（如从 `orthogonal_polynomials.py` 导入的归一化多项式求值函数）。
  - `test_multidim_polynomial`：自测函数，验证 rank/排列一致性。

### 3. `sparse_linear_solver.py` – 稀疏线性代数与 CG 求解

提供稀疏矩阵存储格式和迭代求解器，用于大规模耦合系统。

- **关键类别** `SparseMatrixCOO`：
  - 构造函数接收行数和列数。
  - `add_entry(i, j, v)`：添加非零元。
  - `to_dense()`：转换为稠密 NumPy 数组（用于小型系统）。
  - `mv(x)`：稀疏矩阵-向量乘法 \(y = A x\)。
  - `mtv(x)`：转置的矩阵-向量乘法 \(y = A^T x\)。
  - `residual_norm(x, b)`：计算残差范数。
- **关键函数**：
  - `sparse_from_dense(A)`：从稠密数组构建稀疏矩阵。
  - `conjugate_gradient_sparse(A_sparse, b, x0, max_iter, tol, atol)`：实现共轭梯度法，返回解和包含迭代次数、残差、收敛标志的字典。要求矩阵为 SPD。
  - `jacobi_preconditioned_cg(...)`：实现带 Jacobi 预条件的 CG。
  - `test_sparse_linear_solver`：自测函数。

### 4. `quadrature_rules.py` – 高精确定性/随机求积规则

集合了多类求积公式，用于数值积分和随机投影。

- **关键函数**：
  - `gauss_legendre_tensor(d, n_per_dim)`：d 维张量积 Gauss-Legendre 规则，返回节点和权重。
  - `padua_points_and_weights(degree)`：生成二维 Padua 点及对应插值/求积权重，用于双变量多项式插值。
  - `twb_triangle_rule(strength)`：给出单位三角形上的 TWB 求积节点和权重（根据强度选择预计算规则）。
  - `triangle_unit_monomial_integral(ex, ey)`：单位三角形上 \(x^{ex} y^{ey}\) 的精确积分公式。
  - `triangle_to_standard(nodes)`：将单位三角形节点映射到标准三角形 \([-1,1]^2\)。
  - `smolyak_sparse_grid(d, level, poly_family)`：Smolyak 稀疏网格求积，使用组合系数叠加张量积 Gauss-Legendre 规则（1D 规则点数随指标 i 增加）。返回组合后的节点和权重。
  - `test_quadrature_rules`：自测函数，检验多项式的精确积分。

### 5. `fem1d_solver.py` – 一维有限元求解器

实现一维椭圆问题的 Galerkin 有限元离散，使用分段线性 hat 函数。

- **关键函数**：
  - `uniform_mesh_1d(xL, xR, n_elem)`：生成均匀节点。
  - `build_fem1d_system(mesh, a_func, c_func, f_func, bc_left, bc_right)`：组装全局刚度矩阵（包含扩散和反应项）和载荷向量。处理 Dirichlet 边界条件（修改方程）。返回 `SparseMatrixCOO` 格式的矩阵和右端向量。
    - 边界条件元组：`('D', value)` 或 `('N', flux
