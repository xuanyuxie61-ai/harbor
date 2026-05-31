# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：高性能计算检查点容错与重启系统

本项目构建了一套面向大规模并行数值模拟的容错框架，核心思想是结合**多级检查点树**、**统计故障预测**、**状态压缩**及**最优恢复决策**，在长时间 PDE 时间推进过程中自动应对硬件故障，保证计算结果的有效性并最小化额外开销。项目以三维对流‑扩散‑反应方程的有限元求解为主线，通过若干相互独立的科学计算模块组成端到端的演示系统。

下文列出各源文件及其职责、关键数据结构和算法概念，用以指导后续复现工作。保留文件 **main.py** 作为集成入口，其余 `.py` 文件需要根据本描述从零实现。

---

## 模块划分与职责

### 1. `mesh_geometry.py` – 三维四面体网格
- 提供类 `TetrahedralMesh`，存储节点坐标 `nodes`（N×3）和单元索引 `elements`（M×4）。
- 包含静态方法 `generate_uniform_box`，将长方体区域划分为小立方体，每个立方体再剖分为 6 个四面体，返回网格实例。
- 提供方法：计算单元体积（标量三重积的 1/6）、单元重心、全局边界框、最大单元直径。
- 网格数据用于后续 PDE 求解和检查点状态的空间索引。

### 2. `pde_solver.py` – 三维对流‑扩散‑反应有限元求解器
- 类 `AdvectionDiffusionSolver` 基于 P1 四面体元进行时间推进。
- 构造函数接收网格、扩散系数、速度场、反应率，负责组装 lumped 质量矩阵和近似扩散算子（此处简化为沿节点排序方向的一维三对角结构）。
- 提供方法：
  - `step_explicit`：显式 Euler 时间步，包含扩散、对流、非线性反应项，并强制 Dirichlet 边界。
  - `initial_condition`：生成高斯或随机初始场。
  - `compute_energy`：计算离散能量泛函（涉及单元内常数梯度）。
- 依赖 `TetrahedralMesh`、`quadrature_engine`、`sparse_linear_algebra`。

### 3. `checkpoint_tree.py` – 多级存储层次树
- 定义类 `CheckpointNode` 表示单个存储节点（内存/本地磁盘/远程 PFS），包含层级、读写带宽、容量、单位存储成本。
- 定义类 `CheckpointTree`，根节点固定，提供：
  - 节点收集、按层次筛选。
  - `expected_recovery_time`：根据各层次恢复概率和数据量计算期望恢复时间（沿树向上累加父级传输开销）。
  - `total_storage_cost`：统计在指定层级同时保存副本的存储成本。
  - `tree_distance`：计算两节点在树中的边数（用于冗余放置）。
  - `hierarchical_cluster_placement`：将 N 个检查点轮询分配到叶子节点以实现负载分散。
- `build_default_tree` 函数构造典型的三级树：DRAM (0) → Local_SSD (1) → Remote_PFS (2)。

### 4. `fault_model.py` – 故障到达统计模型与预测
- 类 `GammaFaultModel` 用 Gamma 分布建模故障间隔时间，提供 PDF、CDF、生存函数、风险率、均值、方差、熵、分位数、随机采样。
  - 内部使用 `special_functions` 中的`gammad`（不完全 Gamma）计算 CDF，`digamma` 计算熵。
  - 分位数通过 Chi2 逆 CDF 实现。
- 类 `FaultPredictor` 维护历史故障间隔列表，实现：
  - `observe`：记录间隔。
  - `test_increase`：基于最近样本的方差膨胀检验（利用 Chi2 分布），判断故障率是否显著上升。
  - `recommended_checkpoint_interval`：通过矩估计拟合 Gamma 分布并取中位数分位数作为推荐间隔。
- 依赖 `special_functions` 中有关 Gamma、Chi2、Digamma 的实现。

### 5. `state_compression.py` – 状态压缩与重建
- SVD 低秩压缩：
  - `svd_compress` 对二维矩阵执行截断 SVD，返回低秩因子和近似矩阵；可自动处理一维向量。
  - `svd_reconstruct` 从因子重建完整矩阵。
  - `optimal_rank` 根据奇异值能量占比阈值选择最优秩。
- 三角插值压缩（一维谱插值）：
  - `trigcardinal` 实现三角基函数（奇偶分支）。
  - `trig_interpolant` 在等距节点上对给定数据构造三角插值并可求值任意点。
  - `compress_state_trig` 将一维状态向下采样到粗网格并记录坐标与值。
  - `reconstruct_state_trig` 从粗网格数据重建细网格状态。
- 所有功能均使用 NumPy，输出为纯数组或元组。

### 6. `recovery_mdp.py` – 恢复策略的马尔可夫决策过程
- 类 `CheckpointMDP` 将计算流程建模为 5 状态（Compute、Checkpoint、Verify、Recover、Done）× 3 动作（Memory、Local、Remote）的 MDP。
- 构造函数接收故障概率、恢复成功概率、每状态‑动作对的即时成本，内部构建转移概率矩阵。
- 提供方法：
  - `value_iteration`：值迭代求解最优策略，返回各状态的值与最优动作。
  - `stationary_distribution`：对给定动作计算非吸收态子链的稳态分布。
  - `expected_time_to_done`：模拟从 Compute 到 Done 的期望步数/成本。
- 不依赖外部模块，仅使用 NumPy。

### 7. `sampling_optimizer.py` – 拉丁超立方采样与鲁棒优化
- 函数 `latin_random` 生成 `dim_num × point_num` 的拉丁超立方样本（分布区间 [0,1]）。
- 类 `CheckpointStrategyOptimizer`：
  - 通过 `sample_parameters` 将 LHS 样本变换到故障率、带宽比、状态规模、压缩比的物理范围。
  - 静态方法 `objective` 定义单位时间期望损失（检查点开销密度 + 故障重启损失密度）。
  - `optimize_interval` 在一组候选间隔中找出使损失最小的间隔。
  - `robust_optimize` 对全部样本执行优化，返回平均间隔、标准差、损失分布等统计信息。

### 8. `sparse_linear_algebra.py` – 稀疏线性代数与迭代求解器
- 类 `R83SMatrix` 表示固定三对角系数 `[sub, diag, sup]` 的方形矩阵，提供：
  - 矩阵‑向量乘、转置乘、残差、转换为稠密矩阵。
  - 静态方法 `dif2` 构造经典离散 Laplacian 的三对角形式。
- 三个线性方程组求解器（均针对 `R83SMatrix`）：
  - `r83s_cg`：共轭梯度法。
  - `r83s_jacobi`：Jacobi 迭代。
  - `r83s_gauss_seidel`：Gauss‑Seidel 迭代。
- `cholesky_decompose`：对对称矩阵进行 Cholesky 分解 `L L^T`，返回下三角矩阵、秩亏数和故障标识，用于协方差矩阵分解等。

### 9. `special_functions.py` – 核心特殊函数
- 实现以下统计/数值函数，部分基于经典 FORTRAN 算法的移植：
  - `alnorm(x, upper)`：标准正态尾部概率。
  - `gammad(x, p)`：不完全 Gamma 函数 P(x,p)，处理大参数与小参数分支。
  - `ppchi2(p, v, g)`：Chi2 分布的分位数（逆 CDF）。
  - `ppnd(p)`：标准正态的逆 CDF。
  - `digamma(x)`：Dig
