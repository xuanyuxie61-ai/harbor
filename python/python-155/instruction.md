# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 量子行走搜索算法项目描述 (python-155)

该项目实现量子行走搜索算法，并融合多种图结构上的离散时间与连续时间量子行走、数值方法、参数优化和诊断工具。代码分布于多个 `.py` 文件中，`main.py` 作为统一入口依次运行 14 个实验。后续你需要根据本描述，在仅保留 `main.py` 的前提下，复现所有缺失的模块文件。

## 文件职责总览

| 文件名 | 职责 |
|--------|------|
| `utils.py` | 基础数值工具与常用矩阵/向量操作 |
| `matrix_solvers.py` | 专用线性方程组求解器（Toeplitz、Vandermonde、三对角、CG 等） |
| `geometry_mesh.py` | 几何离散化：二维三角剖分、球面立方网格、六边形格子 |
| `lattice_states.py` | 高维格点状态枚举与参数网格生成（Diophantine 方程、Clenshaw-Curtis 稀疏网格） |
| `quantum_operators.py` | 量子算子构造：硬币、平移、预言机、哈密顿量、酉演化 |
| `quantum_walk_core.py` | 量子行走核心动力学类（DTQW、CTQW、多维行走、搜索增强） |
| `search_algorithm.py` | 搜索算法实现与分析（二维网格、六边形格子、剖分域、超立方体等） |
| `numerical_quadrature.py` | 数值积分与正交规则（六边形 Stroud 规则、梯形/辛普森、病态函数测试） |
| `optimization_diagnostics.py` | 参数优化与收敛诊断（牛顿法、随机采样、优惠券收集、收敛率） |

## 模块详细描述

### 1. `utils.py` – 基础工具

提供项目各处复用的安全数学与数据结构工具：
- 安全除法 `safe_divide`（防止除零）。
- 向量归一化 `normalize_vector`。
- 幂次判断及获取 `is_power_of_two`、`next_power_of_two`。
- 整数向量最大公约数 `gcd_vec`。
- 矩阵强制厄米性 `ensure_hermitian` 和幺正性 `ensure_unitary`。
- 数值钳位 `clamp`。
- 生成切比雪夫节点 `chebyshev_nodes`。
- 构造 Hadamard 矩阵 `hadamard_matrix`（要求阶数为 2 的幂，否则自动填充并截断）。
- 构造 Grover 扩散硬币矩阵 `grover_coin`。
- 构造一维离散拉普拉斯矩阵 `discrete_laplacian_1d`（支持周期边界）。
- 二维网格四邻居索引获取 `grid_neighbors_2d`。
- 多维索引与一维扁平索引相互转换：`tensor_index_to_flat`、`flat_to_tensor_index`。

### 2. `matrix_solvers.py` – 线性方程组求解器

提供用于量子行走运算符和离散化问题的高效专用求解器：
- **Toeplitz 矩阵**：`r8to_mv`（矩阵‑向量乘），`r8to_sl`（求解，优先调用 SciPy 的 Toeplitz 求解器，否则退化为稠密求解）。
- **Vandermonde 矩阵**：`r8vm_mv`（向量乘），`r8vm_sl`（求解方阵，检测节点重复以避免奇异性）。
- **三对角矩阵（R83 格式）**：`r83_mv`（向量乘）；`r83_cg`（共轭梯度法）；`r83_cr_fa`（循环约化因子化）、`r83_cr_sl`（循环约化求解）；`r83_jac_sl`（Jacobi 迭代）；`r83_gs_sl`（Gauss‑Seidel 迭代）。
- **反向通信共轭梯度**：`cg_rc`（状态机式的 CG 执行器，通过 job 标志控制外部计算矩阵‑向量乘与预调节）；`cg_rc_solve`（高层封装，自动调用外部矩阵乘法函数，并具备 NaN/发散时的稠密回退机制）。
- **Wathen 测试矩阵**：`wathen_order`（返回矩阵阶数），`wathen`（生成有限元一致质量矩阵，基于 8‑节点 serendipity 单元）。

### 3. `geometry_mesh.py` – 几何离散化

提供多种几何模型的离散化与邻接结构：
- **二维三角剖分**：`generate_2d_mesh` 根据多边形边界生成初始三角网格（从质心进行扇形剖分），并使用 `refine_mesh` 通过边二分法细化直至满足最大边长约束；`mesh_adjacency` 从三角形单元构建边邻接表。
- **立方球面网格**：`sphere_cubed_grid_point_count`、`sphere_cubed_grid_line_count` 返回细分后的理论点数/边数；`cubed_grid_ijk_to_xyz` 将立方体索引映射到单位球面坐标；`generate_cubed_sphere_grid` 生成球面网格点和线段端点；`cubed_sphere_adjacency` 根据线段构建邻接表（通过最近点匹配）。
- **六边形格子**：`hexagon_unit_vertices`（单位正六边形顶点）、`hexagon_area`（面积）；`generate_hexagonal_lattice` 生成具有若干环的六边形点阵；`hexagonal_adjacency` 根据点间距离建立相邻关系（仅连接距离等于间距的点）。
- **六边形 Stroud 积分规则**：`hexagon_stroud_rule1` 至 `hexagon_stroud_rule4` 返回不同精度的积分点与权重。
- `hexagon_monomial_integral` 计算 `x^p y^q` 在单位六边形上的精确积分（使用 Steger 多边形矩公式），并辅以组合数函数 `comb`。

### 4. `lattice_states.py` – 格点状态与参数网格

提供高维状态枚举与参数空间采样方法：
- **Diophantine 方程非负整数解**：`diophantine_nd_check`（解存在性检验）；`diophantine_nd_nonnegative`（回溯求全部非负解）；`diophantine_nd_nonnegative_bounded`（带边界限制的解）。
- **Clenshaw‑Curtis 稀疏网格**：`cc_level_to_order`（级别到积分点数的映射）；`cc_abscissa`（一维横坐标）；`generate_cc_sparse_grid` 和 `constrained_parameter_grid` 分别生成各向同性和各向异性（加权）稀疏网格，通过级别组合和笛卡尔积构造点集并去重。
- **随机水平采样**：`random_level_sample` 从数组中随机抽取若干水平值作为等高线水平；`analyze_probability_landscape` 对概率网格进行统计（最小值、最大值、均值、标准差、水平间距）。
- `build_hypercube_states` 返回超立方体的所有顶点坐标。
- `diophantine_constrained_states` 将 Diophantine 解转换为多维整数数组。

### 5. `quantum_operators.py` – 量子算子

构造量子行走所需的各类算子：
- **分段常数函数**：`piecewise_constant_1d` 和 `piecewise_constant_2d`，可根据断点数组和值数组构造一维/二维分段常数可调用函数，内部进行边界钳位和网格搜索。
- **硬币算子**：`hadamard_coin`、`grover_coin`、`fourier_coin` 分别返回 Hadamard、Grover 扩散、离散傅里叶变换硬币矩阵；`custom_phase_coin` 返回对角相位硬币。
- **平移算子**：`shift_operator_1d` 为一位量子行走构建平移算子（支持周期或反射边界）；`shift_operator_graph` 为任意图构建 Szegedy 风格平移算子（根据邻接表计算边
