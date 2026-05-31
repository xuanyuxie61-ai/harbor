# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 高超声速边界层转捩预测项目描述

## 概述

本项目实现一套用于**高超声速边界层转捩预测**的计算工具，覆盖网格生成、基流求解、线性稳定性分析（LST）、e^N 放大因子积分、转捩位置预测及不确定性量化等关键步骤。物理背景为可压缩边界层流动，重点捕捉 Mack 第二模态不稳定性及转捩前沿的光滑性优化。

程序入口为 `main.py`，它串联整个计算流程并输出结果。执行流程为：
1. 设置全局参数（马赫数、雷诺数、壁温比等）
2. 生成平板边界层结构化/非结构化计算网格
3. 求解基流（自相似温度场与速度剖面）
4. 验证有限元基函数并构建 FEM 矩阵
5. 进行 Chebyshev 谱方法的线性稳定性分析
6. 谱积分精度验证及 e^N 积分
7. 转捩位置预测与前沿优化
8. 蒙特卡洛不确定性量化
9. 输出数据文件与报告

`main.py` 文件将**保留**，其余模块文件将被删除。你需要根据本描述**补全所有缺失的 Python 文件**，使其能够正确配合 `main.py` 运行。

---

## 文件清单与功能概述

| 文件名 | 职责简述 |
|--------|----------|
| `main.py` | 流程主控，调用各模块并输出结果（**保留，不需实现**） |
| `mesh_generator.py` | 边界层计算网格生成（结构化网格、三角形剖分、球面波矢网格） |
| `utils.py` | 数学工具与辅助函数（质因数分解、Chebyshev 节点/微分矩阵、Blasius 解、粘性公式等） |
| `thermal_solver.py` | 可压缩边界层自相似能量方程求解（温度场、速度、摩擦系数、热流） |
| `fem_basis.py` | 四面体线性有限元基函数、坐标变换、质量/刚度矩阵装配 |
| `stability_analysis.py` | 可压缩线性稳定性求解器（算子构建、特征值求解、Jordan 分析、模态追踪） |
| `spectral_integrator.py` | 谱积分与求积法则（Chebyshev 精确度验证、球面三角形积分、e^N 积分） |
| `transition_predictor.py` | 转捩预测与优化（e^N 方法、多站位预测、转捩前沿优化、感受性系数） |
| `monte_carlo_sampler.py` | 参数空间采样与不确定性量化（LHS 采样、序贯最优停止、MC 传播） |
| `data_io.py` | 数据格式转换与报告输出（XY 文件、VTK 结构化网格、TECPLOT、特征值谱、转捩报告） |

---

## 模块详细说明

### mesh_generator.py
提供二维/三维高超声速边界层网格生成。

**类**：`BoundaryLayerMesh`
- 构造参数：平板长度 `L`、域高 `H`、流向/法向节点数 `Nx`, `Ny`、雷诺数 `Re`、马赫数 `Ma`。
- 方法：
  - `wall_normal_stretching(n, h_max, beta)`：计算壁面法向几何拉伸坐标，保证第一层网格高度满足 `y+ < 1`（利用 Blasius 摩阻估算摩擦速度）。
  - `generate_flat_plate_mesh()`：生成二维平板结构化网格，返回节点数组 `(nx*ny, 2)` 以及 `nx, ny`。
  - `generate_triangles_from_structured(nx, ny)`：将四边形网格剖分为三角形单元，返回三角形节点索引数组。
  - `triangle_neighbors(triangle_num, triangle_node)`：基于边的邻居单元搜索，返回每个三角形的三个邻居索引（-1 为边界）。
  - `boundary_nodes(nx, ny)`：标记壁面、入口、出口、远场节点。

**函数**：
- `sphere_wavevector_grid(lat_num, long_num)`：生成单位球面上的波矢方向离散点，用于三维稳定性分析中的展向波数扫描，返回 `(N,3)` 数组。
- `save_xy_data(filename, x, y)` / `read_xy_data(filename)`：简单的二维坐标数据读写。

### utils.py
提供数值计算基础工具，供多个模块调用。

**主要函数**：
- `prime_factors(n)`：质因数分解，返回升序因子列表。
- `optimal_chebyshev_order(target_n, max_prime)`：寻找不小于 `target_n` 且 `N+1` 的质因数均不超过 `max_prime` 的最小整数 `N`，用于选择适合快速变换的 Chebyshev 阶数。
- `normalize_array(arr, method)`：对数组进行 min‑max 或 z‑score 归一化。
- `safe_divide(a, b, fill_value)`：逐元素安全除法，分母过小以 `fill_value` 填充。
- `blasius_function(eta)`：计算不可压缩 Blasius 边界层相似解 `f(η)`, `f'`, `f''`，采用级数展开结合渐近修正。
- `compressible_blasius_velocity(eta, Ma, gamma, Pr, Tw_Te)`：基于 Crocco‑Busemann 能量积分给出可压缩速度、温度、粘性系数剖面。
- `sutherland_viscosity(T, T_ref, mu_ref, S)`：Sutherland 粘性公式。
- `chebyshev_nodes(n, a, b)`：生成区间 `[a,b]` 上的 Chebyshev‑Gauss‑Lobatto 节点。
- `chebyshev_diff_matrix(n, a, b)`：构造对应的 Chebyshev 谱微分矩阵，含区间缩放。

### thermal_solver.py
求解高超声速边界层自相似能量方程，获取基流温度、速度及物性剖面。

**类**：`HypersonicThermalSolver`
- 构造参数：`Ma`, `Re`, `Pr`, `gamma`, `Tw_over_Te`, `L`, `N_eta`, `eta_max`。
- 内部建立均匀 `η` 网格。
- `solve_self_similar_energy(epsilon, max_iter)`：用 Jacobi 松弛迭代求解能量方程的离散形式，耦合速度剖面（来自 Blasius 解）和温度相关的输运系数。返回字典包含 `eta`, `T`, `u`, `mu`, `rho`, `iterations`, `diff`。
- `compute_wall_heat_flux(solution)`：基于壁面温度梯度计算无量纲壁面热流（近似 Stanton 数）。
- `compute_skin_friction(solution)`：基于壁面速度梯度和粘性计算摩擦系数。

### fem_basis.py
三维四面体有限元基函数与矩阵装配。

**函数**：
- `tet4_basis(t, p)`：计算物理四面体 `t`（3×4）在点 `p`（3×N）处的线性基函数值，基于体积坐标和四面体体积公式，返回 `(4,N)`。
- `tetrahedron_volume(t)`：计算四面体体积（使用边向量行列式）。
- `reference_to_physical_tet4(t, xi)`：将参考四面体坐标 `ξ`（3×N）映射到物理坐标，基于等参变换。
- `physical_to_reference_tet4(t, x)`：通过求解线性方程组将物理坐标逆映射到参考坐标。
- `reference_tet4_sample(n)`：在参考四面体内生成均匀随机采样点。
- `build_fem_mass_matrix(nodes, tetrahedra, rho)`：构建一致质量矩阵，采用单元循环并利用线性四面体的解析积分公式。
- `build_fem_stiffness_matrix(nodes, tetrahedra, mu_field)`：构建扩散刚度矩阵，计算常数梯度并进行加权积分。

所有矩阵使用稀疏或稠密组装均可，但需支持密度场或扩散系数场。

### stability_analysis.py
可压缩边界层线性稳定性分析的谱方法求解器
