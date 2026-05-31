# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：自适应谱‑有限元粒子方法负载均衡框架

本项目实现一个高性能科学计算框架，结合粒子方法、有限元场求解、多重网格加速、谱分析以及动态负载均衡技术，适用于等离子体物理中粒子‑网格（PIC）模拟等场景。框架以 `main.py` 作为统一入口，通过串联各功能模块完成从粒子初始化、轨迹积分、场求解、负载评估到动态重划分的完整流程。后续将保留 `main.py` 并删除其余 `.py` 文件，要求根据本描述复现所有缺失的模块代码。

## 模块总览

| 文件名 | 职责 |
|--------|------|
| `main.py` | 主程序，调用其余所有模块完成端到端仿真流程 |
| `utils.py` | 提供数值稳定性工具、网格基础运算、常数定义及多重网格辅助操作 |
| `fast_summation.py` | 快速求和：Toeplitz 矩阵‑向量乘法、球面/球体采样、粒子前缀和、多极子展开 |
| `fem_solver.py` | 二维有限元求解器：P1 三角形基函数、刚度/质量矩阵组装、泊松方程求解及插值 |
| `load_balancer.py` | 动态负载均衡：处理器域分解、负载评估、正交递归二分法（ORB）及扩散均衡 |
| `mesh_generator.py` | 自适应网格生成：Q4 四边形网格、基于粒子负载的细化、三角剖分 |
| `multigrid_poisson.py` | 一维和二维多重网格泊松求解器（V‑cycle） |
| `particle_dynamics.py` | 粒子混沌动力学：Rucklidge/Arneodo 系统、RKF45 自适应积分、粒子负载沉积 |
| `quadrature_rules.py` | 三角形求积规则及网格上的数值积分与矩量计算 |
| `spectral_analysis.py` | 谱方法：拉盖尔/广义拉盖尔多项式、切比雪夫节点插值、径向分布谱展开及谱导数 |

## 模块详细说明

### `utils.py`
- **物理常数与容限**：定义机器精度 `EPSILON_MACHINE`、最大迭代次数、泊尔兹曼常数等。
- **数值稳定性函数**：
  - `safe_divide`：避免除零的安全除法。
  - `check_bounds`：将数组截断至指定区间并警告越界。
- **几何辅助**：
  - `compute_triangle_area`：通过向量叉积计算二维三角形有向面积，含退化检测。
  - `reference_to_physical_q4`：使用双线性形函数将参考四边形 `[0,1]²` 中的点映射到物理四边形。
- **网格索引转换**：`mesh_base_one` 将元素节点索引统一转换为 1‑based。
- **多重网格辅助**：
  - `gauss_seidel_sweep`：一维 Gauss‑Seidel 迭代扫掠。
  - `restrict_coarse_to_fine`：粗到细网格的线性延拓。
  - `restrict_fine_to_coarse`：细到粗网格的完全加权限制。
  - `is_power_of_two`：检查整数是否为 2 的幂。

### `fast_summation.py`
- **Toeplitz 矩阵‑向量乘**：
  - `toeplitz_mv`：根据 `2N‑1` 个独立元素实现朴素 O(N²) 乘法，矩阵由第一行和第一列定义。
  - `toeplitz_embedded_fft_mv`：将 Toeplitz 矩阵嵌入循环矩阵，借助 FFT 实现 O(N log N) 加速。
- **球面/球体采样**：
  - `sample_unit_ball_positive`：在三维单位球内均匀采样，并取绝对值得到正八分球点（用于粒子初始化）。
  - `sample_unit_sphere_surface`：用正态分布归一化法在任意维度单位球面上均匀采样。
- **前缀和与粒子计数**：
  - `compute_prefix_sum_2d`：将二维粒子分布栅格化后构建前缀和数组。
  - `query_region_count`：利用前缀和在 O(1) 时间内查询矩形区域内粒子数。
- **多极子展开**：`multipole_expansion` 对一组带权重的粒子计算至指定阶数（单极子、偶极子、四极子）的展开系数。
- **相互作用核构建**：`build_interaction_matrix_toeplitz` 根据给定核函数 `K(r)` 和网格间距生成一维 Toeplitz 表示。

### `fem_solver.py`
- **类 `FEMSystem`**：
  - 初始化给定节点坐标（形状 `(n,2)`）和三角形单元连接（1‑based）。
  - 自动检测位于矩形外边界上的边界节点。
  - `basis_t3`：对指定三角形计算 P1 线性基函数值及其在任意评估点处的梯度。
  - `assemble_stiffness_matrix`：组装泊松方程对应的稀疏刚度矩阵（使用常数梯度近似）。
  - `assemble_mass_matrix`：组装一致质量矩阵（P1 显式公式）。
  - `project_function_l2`：将节点函数值进行 L2 投影（求解 `M u = M f`）。
  - `solve_poisson`：接收右端项并求解带 Dirichlet 边界条件的泊松方程，使用直接消去法。
  - `interpolate_to_points`：将有限元解用重心坐标插值到任意空间点，若点在三角形外则回退到最近节点。

### `load_balancer.py`
- **类 `LoadBalancer`**：
  - 管理处理器网格（将处理器总数分解为 `px × py` 均匀域划分）。
  - `compute_loads`：统计各子域内的粒子数（可叠加场求解开销），得到每个处理器的负载。
  - `imbalance_factor`：计算负载不均衡因子 `max/mean`。
  - `find_optimal_split`：沿给定轴排序粒子并寻找使两侧负载差最小的切分位置。
  - `recursive_bisection`：实现正交递归二分法（ORB），根据粒子分布标准差选择切分轴，按负载比例递归分配处理器。
  - `rebalance`：评估当前负载，若超过阈值则执行 ORB 重划分，估算迁移粒子数并更新域分解。
  - `evaluate_efficiency`：输出平均负载、标准差、不均衡因子、变异系数和理论并行效率。
- **独立函数** `diffusion_based_load_balance`：基于扩散的负载均衡算法，在多步迭代中通过邻接处理器之间的负载流动实现平衡。

### `mesh_generator.py`
- **`MeshElement` 类**：代表单个网格单元，存储节点索引、类型（Q4/T3）、细化层级、面积和负载。
- **`QuadMesh` 类**：
  - 初始化建立均匀四边形网格（`nx × ny` 个单元），节点按行优先排列。
  - `evaluate_load`：快速定位粒子所属单元并统计粒子数。
  - `refine_by_load`：若单元负载超过平均值的 `(1+θ)` 倍且未达到最大层级，则将其四等分为四个子四边形（需插入边中点和中心点）。
  - `triangulate_elements`：将所有 Q4 单元剖分为三角形（T3），返回节点数组和 1‑based 三角形连接矩阵。
  - `get_element_centers`：求每个单元的几何中心。
- **独立函数**：
  - `build_delaunay_triangulation`：尝试使用 `scipy.spatial.Delaunay` 生成 Delaunay 剖分，若不可用则回退到简单矩形剖分。
  - `compute_mesh_bandwidth`：根据单元连接矩阵计算网格半带宽。

### `multigrid_poisson.py`
- **`MultigridPoisson1D`**：求解一维泊松方程 `-u''=f`，要求网格数为 2 的幂。
