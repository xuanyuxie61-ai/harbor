# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：聚合物玻璃化转变分子动力学模拟

## 概览

本项目实现一个粗粒化分子动力学（MD）模拟框架，用于研究聚合物熔体在温度淬火过程中的玻璃化转变行为。系统由多条柔性链组成，每条链包含若干粗粒化单体（bead）。通过追踪比容、自由体积、结构有序度及扩散系数随温度的变化，利用多种数值方法分析玻璃化转变温度 Tg、VFT 方程参数和脆性指数。

模拟流程（见 `main.py`）分为七步：
1. 系统初始化（构建聚合物链、力场、积分器、温度协议等）
2. 高温平衡化
3. 温度淬火扫描与物理量采集
4. 玻璃化转变分析
5. 热扩散分析
6. 稀疏矩阵求解演示
7. 结果汇总输出

本项目将保留 `main.py`，其余模块文件将被移除。你需要根据以下描述重新实现它们。

## 文件与模块功能

### 1. `polymer_chain.py` – 聚合物链构建
- 提供 **PolymerChain** 类，用于生成多链聚合物的初始构象和基本物理量。
- 使用 3D 自回避随机游走（SARW）初始化每条链的位置，避免单体过近的重叠。
- 速度按 Maxwell-Boltzmann 分布初始化，并减去质心速度以避免整体漂移。
- 提供函数 **generate_ellipse_cross_section**，用于在椭圆截面内生成网格点，表示链横截面单体分布。
- 实现方法：`radius_of_gyration`（回转半径）、`end_to_end_distance`、`kinetic_energy`、`instantaneous_temperature`，以及边界条件应用 `apply_pbc`。

**核心数据成员**：`n_chains`、`beads_per_chain`、`n_total`、`positions`、`velocities`、`forces`、`masses`、`box`、`chain_starts`。

### 2. `force_field.py` – 力场定义
- 包含类 **ForceField**，定义粗粒化聚合物体系的势能模型。
- 三种势能项：
  - **Lennard-Jones 非键势**：使用截断距离和能量平移，保证截断处连续。
  - **FENE 键合势**：有限伸长非线性弹性势能，限制键的最大伸缩。
  - **弯曲角势**：保持链刚性的角度约束。
- 提供计算单个键或角度的能量/力标量的函数，以及向量化的总力计算函数 `compute_total_forces`（求和三种贡献）和总势能计算 `total_potential_energy`。所有力计算均考虑最小像约定处理周期边界。

### 3. `integrator.py` – 分子动力学积分器
- 提供 **VelocityVerletIntegrator** 类，实现 Velocity Verlet 时间积分，并包含 CFL 条件检查。
- 提供函数 **rk23_step** 和 **rk23_integrate**，实现 2/3 阶嵌入 Runge-Kutta 方法（用于 Nose-Hoover 热浴耦合）。
- 提供 **NoseHooverIntegrator** 类，利用扩展变量实现恒温控制。其 `step` 方法用 RK23 积分更新热浴变量 ξ，并对速度施加阻尼修正。

### 4. `thermostat.py` – 热浴与温度控制
- 定义 **TemperatureProtocol** 类，生成温度-时间协议（线性、阶梯、对数降温），并提供冷却速率计算。
- 实现两个常用恒温器：
  - **AndersenThermostat**：随机碰撞，从 Maxwell 分布重采样速度。
  - **BerendsenThermostat**：弱耦合速度缩放。
- 恒温器的 `apply` 方法接受速度、质量、目标温度和时间步长，返回调整后的速度。

### 5. `cvt_sampler.py` – CVT 自由体积分析器
- 包含类 **CVTSampler**，用于通过 Centroidal Voronoi Tessellation 分析聚合物体系中的自由体积。
- 核心算法：
  1. 在模拟盒子内随机采样点，根据聚合物单体位置计算非均匀密度 ρ(x)（自由体积权重函数）。
  2. 为每个采样点找到最近的生成器（Voronoi 胞）。
  3. 将生成器更新为对应 Voronoi 区域的密度加权质心。
  4. 应用反射边界和盒子投影确保生成器在盒子内。
  5. 迭代直至收敛，记录能量历史。
- `iterate` 方法执行上述迭代并返回生成器位置和通过蒙特卡洛估计的 Voronoi 体积。
- 提供 `free_volume_fraction` 计算硬球模型近似的自由体积分数，以及 `structural_order_parameter` 基于能量变化评估有序度。

### 6. `heat_diffusion.py` – 热传导分析
- 提供两个热传导求解器：
  - **HeatDiffusion1D**：一维显式有限差分求解器，求解 ∂T/∂t = α ∂²T/∂x² + Q。支持单步时间推进和稳态求解，内置 CFL 稳定性条件。
  - **HeatDiffusion2DFEM**：二维简化有限元/有限差法求解器，使用五点 Laplacian 离散和 Jacobi 隐式迭代（向后 Euler）。可求解稳态热传导（通过 Jacobi 松弛），并估算有效热导率。
- 边界条件通过调用 `boundary_T` 函数设定。

### 7. `glass_transition.py` – 玻璃化转变分析
- 包含函数 **vft_equation** 和 **vft_viscosity** 计算 VFT 方程。
- 提供 **regula_falsi** 函数实现试位法求根，用于精确确定 Tg。
- 实现 **GlassTransitionAnalyzer** 类：
  - `add_data_point` 累积温度、比容、能量数据。
  - `linear_fit` 进行最小二乘线性拟合。
  - `find_tg_tangent_intersection` 通过高低温区的切线交点估计 Tg 及热膨胀系数。
  - `find_tg_regula_falsi` 使用试位法改进 Tg 估计。
  - `vft_fit` 通过搜索 Vogel 温度 T0 拟合 VFT 参数。
  - `fragility_index` 计算脆性指数 m。
  - `configurational_entropy` 计算 Adam-Gibbs 型构型熵。

### 8. `sparse_solver.py` – 稀疏矩阵求解
- 提供压缩列存储（CCS）格式的稀疏矩阵类 **SparseCCS**。
  - 可从稠密矩阵构造（忽略小于阈值的条目）。
  - 支持与稠密矩阵相互转换。
  - 实现矩阵-向量乘法 `matvec` 和转置乘法 `transpose_matvec`。
  - 提供稀疏度 `sparsity_ratio` 和单元素读写。
- 函数 **conjugate_gradient** 使用共轭梯度法求解对称正定线性系统 Ax=b。
- 函数 **build_neighbor_sparse_matrix** 根据原子位置和截断半径构建邻居稀疏矩阵（0-1 邻接矩阵）。

### 9. `numeric_utils.py` – 数值工具
- 提供素数相关函数：`prime_count` 计算素数个数，`generate_primes` 生成前 n 个素数，`seeded_random` 基于素数索引生成确定性随机数。
- 提供 **2D Legendre-Gauss 积分**：`legendre_gauss_nodes` 计算节点和权重，`integrate_2d_gauss` 对二元函数做积分（用于径向分布函数积分）。
- 通用工具：`safe_divide`（避免除零）、`soft_cutoff`（平滑截断权重）、`distance_matrix_pbc`（考虑 PBC 的距离矩阵）、`mean_squared_displacement`（MSD 计算）。

## 模块间主要依赖关系

- `main.py` 直接引用所有其他模块，充当装配和驱动层。
- `
