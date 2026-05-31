# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：二维可压缩湍流边界层直接数值模拟平台

本项目是一个面向二维可压缩 Navier-Stokes 方程的直接数值模拟（DNS）框架，融合谱元法、有限体积法、本征正交分解（POD）、马尔可夫链蒙特卡洛（MCMC）不确定性量化与空化概率分析等多种数值方法。入口文件 `main.py` 串联所有模块，按阶段执行网格生成、数值积分验证、谱元离散与泊松求解、可压缩 NS 求解、线性代数诊断、POD 分析、MCMC 采样、空化风险评估及收敛诊断。后续将保留 `main.py`，删除其余 `.py` 文件，由其他 agent 根据本描述补全缺失模块。

---

## 文件结构与职责概览

| 文件名 | 职责 |
|--------|------|
| `main.py` | 主入口，调度所有模块完成10个阶段计算，无命令行参数 |
| `utils_numerical.py` | 底层数值工具：正交矩阵行列式验证、二分法求根、安全除法与平方根、CFL计算、minmod 限制器 |
| `linear_algebra_engine.py` | 线性代数支持：Hager 条件数估计、部分选主元 LU 分解及求解、初等行变换、Jacobi 预处理与迭代 |
| `mesh_generator.py` | 自适应网格生成：Voronoi 非结构网格、IFS 分形细化、边界层采样、谱元 GLL 网格 |
| `quadrature_library.py` | 高阶积分规则：六边形 Lyness-Monegato 规则、三角形 Wandzura 规则、参考域到物理域映射及积分函数 |
| `spectral_element_discretization.py` | 谱元离散：一维/二维 DCT、DCT 泊松求解器、Hermite 插值、谱导数、有限元质量/刚度矩阵组装 |
| `compressible_ns_core.py` | 可压缩 NS 求解器核心：守恒变量、原始变量、Roe 数值通量、MUSCL 重构、粘性通量、RK3 时间推进、边界条件 |
| `turbulence_pod_analysis.py` | 湍流 POD 模态分析：快照 POD、模态重构、湍动能与 Reynolds 应力、Galerkin 系数、模态动力学 |
| `mcmc_sampler.py` | MCMC 统计采样：马尔可夫转移矩阵、Metropolis-Hastings 采样器、湍流参数后验采样、稳态分布 |
| `cavitation_probability.py` | 空化概率模型：局部空化概率、多位置联合概率、空化初生准则、全场风险评估、成核率计算 |
| `diagnostics_convergence.py` | 诊断与收敛监测：收敛阶估计、GCI 网格收敛指标、能量/质量守恒、CFL 稳定性监测、格式化输出 |

---

## 模块详细说明

### 1. `utils_numerical.py` — 底层数值工具
提供与 CFD 和线性代数相关的稳健低层函数。

- **`safe_divide` / `safe_sqrt`**：带保护的除法和平方根，防止除零或负数开根。
- **`check_cfl`**：根据对流速度、声速和粘性系数计算满足 CFL 条件的最大时间步长。
- **`limiter_minmod`**：minmod 限制器，用于 MUSCL 重构中抑制激波附近振荡，支持可调参数 θ。
- **`bisection_root_find`**：二分法求非线性方程的根，用于状态方程反解（如从总能反求温度），具备区间扩展能力。
- **`detq_orthogonal`**：验证正交矩阵的行列式，用于几何守恒律检查。

### 2. `linear_algebra_engine.py` — 线性代数引擎
支持稀疏/稠密线性系统的求解与诊断。

- **`condition_hager`**：通过迭代估计矩阵的 L1 条件数，避免显式求逆，用于监测离散算子病态程度。
- **`lu_decomposition_with_pivot`**：部分选主元 Doolittle LU 分解，返回 L、U、置换矩阵 P。
- **`solve_lu`**：利用 LU 分解结果求解线性方程组。
- **`elementary_row_scale` / `elementary_row_swap` / `elementary_row_axpy`**：初等行变换辅助函数。
- **`jacobi_preconditioner`**：构造 Jacobi 预处理器（对角逆矩阵）。
- **`apply_jacobi_precond`**：使用 Jacobi 迭代求解 Ax=b。

### 3. `mesh_generator.py` — 自适应网格生成
生成 CFD 计算所需的结构化与非结构化网格。

- **`generate_voronoi_mesh`**：基于随机生成点构造 Voronoi 图背景网格，计算每个单元的归属和面积。
- **`ifs_adaptive_refinement`**：利用迭代函数系统（IFS）生成分形点集，可在指定区域以高概率接受点，实现自适应加密。
- **`sample_boundary_points`**：边界层法向采样，采用双曲正切拉伸确保近壁面分辨率，支持平板和翼型。
- **`generate_spectral_element_mesh`**：生成基于 Chebyshev 点的谱元网格，y 方向可选指数拉伸。

### 4. `quadrature_library.py` — 高阶数值积分库
提供六边形和三角形区域的高精度积分规则。

- **`hexagon_lyness_rule`**：返回指定规则的积分点、权重和代数精度，用于正六边形上的 Lyness 对称高斯积分。
- **`wandzura_triangle_rule`**：返回指定规则的三角形积分点、权重和精度，基于 Wandzura-Xiao 规则。
- **`reference_to_physical_t3`**：将参考三角形上的积分点映射到物理三角形，并给出 Jacobian 与行列式。
- **`integrate_scalar_on_triangle`** / **`integrate_scalar_on_hexagon`**：对用户提供的标量场在三角形或六边形上执行数值积分。

### 5. `spectral_element_discretization.py` — 谱元离散模块
实现快速变换与谱精度算子。

- **`discrete_cosine_transform_1d` / `inverse_discrete_cosine_transform_1d`**：一维 DCT-II 与 DCT-III 变换。
- **`dct_poisson_solver_2d`**：利用 2D DCT 在频域求解二维泊松方程，满足齐次 Neumann 边界条件，适合压力投影步。
- **`hermite_interpolant_coeffs` / `hermite_interpolant_eval`**：构造并求值 Hermite 插值多项式，可同时满足函数值与导数值，用于通量重构保证 C1 连续性。
- **`hermite_interpolant_derivative`**：计算 Hermite 插值多项式的导数差商表。
- **`spectral_derivative_1d`**：基于 Chebyshev 节点的谱方法一阶导数矩阵。
- **`assemble_fem_mass_matrix_2d` / `assemble_fem_stiffness_matrix_2d`**：在线性三角形单元上组装一致质量矩阵和 Laplace 刚度矩阵。
- **`apply_boundary_conditions_matrix`**：对线性系统施加 Dirichlet 边界条件。

### 6. `compressible_ns_core.py` — 可压缩 NS 求解器核心
构造求解器类 `CompressibleNSSolver`，管理网格、场变量、时间推进与边界条件。

- **初始化参数**：网格数、区域尺寸、比热比、雷诺数、普朗特数、马赫数、壁面温度。自动计算参考状态、粘性系数与网格（含 y 方向指数拉伸）。
- **守恒变量 Q**：形状 `(ny, nx, 4)` 分别存储 ρ, ρu, ρv, ρE。
- **`initialize_field`**：用误差函数近似构造 Blasius 边界层初始场。
- **`primitive
