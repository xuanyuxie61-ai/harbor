# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目概述：三维阻挫自旋玻璃基态与动力学全栈计算框架

本 Python 项目实现了一套用于凝聚态物理中阻挫磁性与自旋玻璃的数值研究工具。入口脚本 `main.py` 串联多个模块，完成晶格构造、交换矩阵建立、四元数自旋初始化、能量景观优化（模拟退火 + 贪心松弛）、自旋动力学积分、谱分析、磁畴统计以及一维 Ginzburg‑Landau 序参量自适应有限元求解。**保留 `main.py`，其他 `.py` 文件将被删除，需要根据本描述复现所有缺失模块。**

## 涉及的文件及其职责

| 文件名 | 职责概述 |
|--------|----------|
| `utils.py` | 数值鲁棒性工具：安全除/开方、自旋归一化裁剪、RMS 范数、Skyline 稀疏矩阵‑向量乘、从三对角构建 Skyline、一维直方图统计、单位三角形区域直方图、素数判定与列表生成。提供全局常量 `EPS_MACHINE`、`EPS_SQRT`。 |
| `spin_lattice.py` | 阻挫晶格构造：`PyrochloreLattice` 类（三维烧绿石，周期性边界，最近邻与次近邻键，可加键无序）、`PrimeFrustratedLattice` 类（二维素数调制方格，随机符号模拟阻挫，可生成全耦合矩阵）。提供二维自旋图的连通分量标记函数 `connected_components_2d_spin_map`。 |
| `spin_quaternion.py` | 四元数代数与自旋旋转：归一化、共轭、乘法、旋转三维矢量、四元数与旋转矩阵互转、轴角到四元数、球面线性插值（slerp）、生成均匀随机四元数、自旋矢量与四元数的互相转换。 |
| `exchange_matrix.py` | 交换矩阵构造：一维/二维离散拉普拉斯型交换矩阵（支持 Dirichlet/Neumann/周期边界）；将一维阵转为 Skyline 稀疏格式；计算有效交换场 `apply_exchange_operator` 和交换能量 `exchange_energy`；向耦合矩阵添加高斯键无序 `add_disorder`。 |
| `energy_landscape.py` | 能量景观优化：Brent 方法一维极小搜索 `local_min_brent`；单自旋转轴旋转线搜索 `line_search_spin_rotation`；全系统贪心松弛 `greedy_relaxation`；模拟退火 `simulated_annealing_spin_glass`。 |
| `eigen_analysis.py` | 特征值分析：幂法求主导特征对 `power_method`；逆迭代（含位移）求近指定值特征对 `inverse_iteration`；直接对角化求谱隙与软模 `spectral_gap_and_soft_modes`；一维自旋波色散 `spin_wave_dispersion_1d`；由谱隙估算关联长度 `correlation_length_from_gap`。 |
| `spin_dynamics.py` | 自旋动力学：计算有效场 `effective_field`；Landau‑Lifshitz‑Gilbert 方程右端项 `llg_rhs`；显式欧拉积分 `euler_integrate_llg`；隐式梯形法积分（含简化 Newton 迭代）`trapezoidal_integrate_llg`；Brusselator 型自旋泵浦 ODE 及其积分 `integrate_brusselator_pump`；Fisher‑KPP 行波精确解构造磁畴壁磁化分布 `domain_wall_magnetization`；从轨迹提取总磁化强度 `compute_magnetization_trajectory`。 |
| `domain_analysis.py` | 磁畴分析：从二维自旋投影提取连通域尺寸统计与熵 `extract_domain_statistics`；自旋极角直方图 `spin_orientation_histogram`；二维径向关联函数 `radial_distribution_function_2d`；磁化强度时间序列的熵产生率 `entropy_rate_from_trajectory`；单位三角形内自旋分布均匀性分析 `analyze_triangle_spin_distribution`。 |
| `adaptive_fem_solver.py` | 一维序参量自适应有限元：P1 线性基函数；组装三对角系统 `assemble_tridiagonal_system`（含 Dirichlet/Neumann 边界）；Thomas 算法求解三对角 `solve_tridiagonal`；基于曲率的局部网格加密 `refine_mesh_locally`；主流程 `adaptive_fem_order_parameter`（带误差估计与迭代终止控制）。 |

## 模块依赖与整体流程

`main.py` 中的 `main()` 函数按顺序执行以下步骤，每个步骤都依赖上述模块提供的 API：

1. **晶格构造**  
   生成 `PyrochloreLattice` 和 `PrimeFrustratedLattice` 实例，获得格点数、键列表与耦合矩阵 `J_3d`、`J_2d`。
2. **交换矩阵与稀疏格式**  
   调用 `exchange_laplacian_1d` 构造一维拉普拉斯矩阵；利用 `build_skyline_from_tridiagonal` 与 `skyline_mv` 验证 Skyline 格式正确性；通过 `add_disorder` 为矩阵加无序；用 `exchange_laplacian_2d`（尽管此处未直接调用，但模块提供该接口以待扩展）演示二维构造思想。
3. **自旋初始化**  
   使用 `random_spin_quaternion` 生成随机四元数，经 `q_to_spin_vector` 转换为 3D 单位矢量，填充系统所有自旋。
4. **能量景观优化**  
   依次运行 `simulated_annealing_spin_glass`（输出最优构型与能量历史）和 `greedy_relaxation`（局部线搜索精化）；示例调用 `line_search_spin_rotation` 展示单自旋 Brent 线搜索。
5. **谱分析与自旋波**  
   对 2D 矩阵调用 `spectral_gap_and_soft_modes` 得全谱及谱隙；使用 `power_method` 和 `inverse_iteration` 求主导特征值与逆迭代软模；计算一维自旋波色散 `spin_wave_dispersion_1d` 和关联长度 `correlation_length_from_gap`。
6. **自旋动力学**  
   分别用 `euler_integrate_llg`（显式欧拉）和 `trapezoidal_integrate_llg`（隐式梯形）积分 LLG 方程，得到自旋轨迹；用 `compute_magnetization_trajectory` 提取总磁化强度序列；积分 Brusselator 型自旋泵 `integrate_brusselator_pump`；通过 `domain_wall_magnetization` 构造 Fisher‑KPP 行波形磁畴壁分布。
7. **磁畴分析与统计**  
   将 3D 自旋投影为 2D 图，调用 `extract_domain_statistics` 进行连通域分析；计算极角直方图 `spin_orientation_histogram`；调用 `analyze_triangle_spin_distribution` 分析三角形内分布；利用 `radial_distribution_function_2d` 获得径向关联；由磁化强度轨迹计算熵产生率 `entropy_rate_from_trajectory`。
8. **自适应有限元**  
   定义系数函数 A(x), B(x), F(x)，调用 `adaptive_fem_order_parameter` 求解一维 Ginzburg‑Landau 型边值问题，获得自适应节点、解、能量密度及迭代历史。

各模块之间通过函数参数传递 Numpy 数组、标量及函数回调，不涉及全局状态。

## 核心算法与数据结构说明（要求复现的要点）

### 1. 四元数与自旋表示 (`spin_quaternion.py`)
- 实现四元数乘法和归一化。
- 轴角 → 四元数：给定旋转轴（单位向量）和弧度角，计算对应
