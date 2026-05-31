# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Python 项目描述：离子通道选择性通透的多尺度计算框架

## 目标
这个 Python 项目是一个研究钾离子通道（KcsA）选择性通透机制的多尺度计算框架。入口文件 `main.py` 调用了多个科学计算模块，依次完成通道几何建模、静电势求解、离子输运模拟、自由能计算等任务。

我们将保留 `main.py`，并删除除 `main.py` 之外的所有 `.py` 文件。你需要**重新实现所有缺失的模块**，使得 `main.py` 能够正确运行并输出有意义的结果。下面的描述给出了每个缺失文件的职责、核心数据结构、关键函数接口以及模块之间的依赖关系，但不提供任何已有的完整源代码或精确的实现细节。

## 缺失模块清单（按文件名）

### 1. `channel_geometry.py`
- 负责构建钾离子通道的二维轴对称三角网格。
- 核心类：`Triangulation`，维护节点数组、三角形单元数组和邻居信息数组。
- 提供 `build_neighbors`（根据单元连通性自动构建邻居关系）、`refine_local`（对指定三角形进行局部四等分细化，并处理共享边的邻居单元）、`element_centers`（计算三角形重心）、`in_selectivity_filter`（判断点是否位于简化的选择性滤器区域）以及 `adaptive_refine_filter`（对滤器区域进行多级自适应细化）。
- 独立函数 `build_channel_mesh_2d(nz, nr)` 根据简化的 KcsA 几何（长度 4.5 nm、滤器收窄、腔体等）生成一个二维轴对称截面的 `Triangulation` 对象。

### 2. `potential_field.py`
- 负责通道内静电势场的自洽求解。
- `DielectricProfile` 类：根据通道几何构建空间变化的介电常数分布（蛋白区域 4.0，水区域 78.5），并通过平滑阶跃函数模拟界面过渡。包含 `gradient_eps` 方法计算介电常数的数值梯度。
- `PotentialSolver` 类：利用 Poisson‑Boltzmann 方程自洽迭代求解电势。关键方法包括 `_fixed_charge_density`（生成滤器区域的固定负电荷密度）、`_mobile_charge_density`（基于线性化 Debye‑Hückel 近似计算移动离子电荷密度）、`solve`（执行自洽迭代，利用有限差分 Laplacian，处理介电梯度修正项并应用 Neumann 边界条件）。迭代使用松弛因子控制收敛。

### 3. `finite_difference.py`
- 提供高阶有限差分算子，用于泊松方程的离散化和 Nernst‑Planck 方程的求解。
- 包含 `vandermonde_like`（构造 Vandermonde 矩阵）、`differ_stencil`（通过解线性方程组计算任意阶导数所需的差分系数）。
- `build_laplacian_1d`：构建一维四阶精度的 Laplacian 矩阵（内部采用五点中心差分，边界降为二阶，支持 Dirichlet 边界）。
- `build_laplacian_3d`：利用 Kronecker 积构造三维各向异性 Laplacian 矩阵（Neumann 边界）。
- `apply_laplacian_3d`：无矩阵存储地应用三维各向同性 Laplacian 算子（七点模板），并在边界实施 Neumann 零法向导数条件。

### 4. `special_functions.py`
- 提供统计力学和电化学中需要的特殊数学函数。
- 包含高精度误差函数 `erf_cody`（基于 Cody 有理逼近）、物理学家 Hermite 多项式 `hermite_phys`、Laguerre 多项式 `laguerre_poly`、Legendre 多项式 `legendre_poly`（均通过递推关系计算）。
- `log_gamma_lanczos`：用 Lanczos 近似计算 ln Γ(z)。
- `spherical_harmonic_norm` 与 `associated_legendre`：计算球谐函数相关量。
- 统计力学工具：`partition_function_harmonic`（谐振子配分函数）、`debeye_huckel_kappa`（Debye 长度倒数）、`boltzmann_factor`。

### 5. `ion_transport.py`
- 求解三维 Nernst‑Planck 对流‑扩散方程，描述 K⁺ 和 Na⁺ 的浓度演化。
- 核心类 `NernstPlanckSolver`：需要指定网格、扩散系数和离子电价，利用算子分裂法（Strang 分裂）将扩散步和迁移步分开推进。内部方法包括 `_diffusion_step`（纯扩散更新）、`_migration_step`（电场引起的迁移更新）和 `solve_step`（组合两者完成一个时间步）。此外提供 `steady_state_flux` 计算离子通量密度，`permeability_coefficient` 估算通透系数和选择性比。
- 独立函数 `pnp_steady_state_iterator` 实现了 Poisson‑Nernst‑Planck 方程组的自洽迭代循环，交替求解电势和更新浓度。

### 6. `monte_carlo_integrator.py`
- 高维数值积分工具集，用于配分函数和自由能计算。
- 球面相关：`sphere01_sample`（用高斯投影法在单位球面上均匀采样）、`sphere01_monomial_integral`（单项式球面积分精确公式）、`spherical_mean_integrand`（蒙特卡洛球面均值）。
- Fibonacci 格点规则：`fibonacci`（生成 Fibonacci 数）、`fibonacci_lattice_2d`（二维标准正方形区域的格点积分）、`lattice_rule_nd`（多维标准格点积分）。
- Clenshaw‑Curtis 稀疏网格：`cc_abscissa`（一维节点坐标）、`cc_weights`（对应权值）、`sparse_grid_cc_1d`（一维稀疏网格节点与权值）、`tensor_product_grid`（张量积网格）、`sparse_grid_cc_smolyak`（Smolyak 构造多维稀疏网格，利用增量公式组合不同层数的一维规则，并对重复节点合并）、`integrate_sparse_grid`（使用稀疏网格计算超立方体积分）。

### 7. `brownian_dynamics.py`
- 进行离子在静电势场中的 overdamped Langevin 布朗动力学模拟。
- `IonParticle` 类：储存单个离子的位置、电荷、半径、质量及轨迹历史。
- `BrownianDynamicsEngine` 类：基于 Euler‑Maruyama 离散化实现动力学步进。`_random_displacement` 生成满足涨落耗散定理的高斯随机位移；`_force_field` 通过中心差分从电势场插值电场力，并叠加通道壁的简谐约束力；`step` 组合漂移和随机扩散项更新离子位置；`run` 对多粒子执行多步模拟。
- 独立函数 `compute_mean_square_displacement` 从轨迹计算均方位移，并通过线性拟合估算有效扩散系数。

### 8. `transition_network.py`
- 将离子在通道中的离散跃迁建模为 Markov 状态网络。
- `TransitionNetwork` 类：存储状态数和转移速率矩阵 K，提供 `add_transition` 添加单向跃迁速率；`compute_degrees` 计算各节点的流入/流出速率；`is_eulerian_path` 判定有向图是否存在欧拉路径（用于衡量连续传导能力）；`steady_state_probability` 通过幂迭代求主方程的稳态概率；`mean_first_passage_time` 求解平均首次通过时间；`conductivity` 估算单通道电导（pS 量级）。
- 辅助函数 `build_kcsa_k_channel_network` 和 `build_na_leaky_network` 分别预设 K⁺ 和 Na⁺ 在网络中的典型跃迁速率，构建 6 状态一维链（入口‑滤器四个位点‑出口）。

### 9. `lattice_occupation.py`
- 用一维或二维离散晶格模型处理离子在通道结合位点上的多体占据统计。
- `LatticeChannel` 类：基于晶格形状和结合能构建系统，`valid
