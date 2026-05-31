# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：二维界面演化的自适应高阶水平集方法

本项目实现一个基于水平集方法（Level Set Method）的二维界面演化求解器，支持曲率驱动流、外部力场、体积守恒、拓扑检测、自适应网格、降阶建模及参数不确定性量化等功能。所有代码围绕笛卡尔网格上的 Hamilton–Jacobi 方程求解展开，并整合了 WENO5 离散、TVD‑RK3 时间推进、符号距离函数重初始化等关键数值技术。

后续 benchmark 将保留 `main.py` 作为统一演示入口，删除其余 `.py` 文件。你需要根据本描述重新实现缺失的模块，使其能够与 `main.py` 配合运行并产生有意义的结果。

---

## 文件职责概览

### `levelset_function.py`
定义 `LevelSetFunction` 类，封装二维水平集函数的网格、数值及其基本几何操作。  
- **职责**：提供水平集的初始化（圆、椭圆、星形、双圆、矩形等），计算梯度、法向量、曲率，提取零等值线，估计体积与界面长度，并支持基于 PDE 的符号距离函数重初始化。  
- **核心数据结构**：二维笛卡尔网格，存储水平集值 `phi`（形状 `(nx, ny)`）。  
- **关键算法思想**：曲率通过中心差分计算二阶偏导；法向量由梯度归一化得到；界面长度使用平滑 Dirac 函数近似积分；符号距离重初始化采用迎风格式的显式迭代。

### `hj_solver.py`
实现 Hamilton–Jacobi 方程的数值求解器 `HJSolver`，并包含剪切流场生成器 `ShearFlow`。  
- **职责**：组合曲率项、平流项和外部力场，构造右端项函数，使用 TVD‑RK3 时间推进求解界面演化。同时提供 CFL 条件计算和一维行波精确解验证。  
- **关键算法**：利用 `LevelSetFunction` 实例计算曲率和梯度，结合中心差分形成半离散算子；`ShearFlow` 提供简单剪切、涡对、振荡剪切等速度场。

### `reinitialization.py`
提供 `Reinitializer` 类，将水平集函数恢复为符号距离函数。  
- **职责**：实现 Godunov 迎风格式的重初始化 PDE 迭代，以及 Jacobi 风格的变体。支持暴力快速行进法近似（用于小规模验证）。可检查 SDF 性质（梯度偏离 1 的程度）。  
- **核心思想**：求解 ∂φ/∂τ + sign(φ₀)(|∇φ|-1)=0，用光滑化的 sign 函数和 Godunov 梯度模计算。

### `curvature_flow.py`
包含 `CurvatureFlow` 类，用于计算曲率驱动的流速场和几何能量，以及 Lebedev 球面积分工具。  
- **职责**：计算平均曲率流速度、Willmore 流右端项，使用 Lebedev 节点在球面上积分；估计 Willmore 能量、界面面积、高斯映射方差。  
- **关键算法**：曲面 Laplace‑Beltrami 算子通过笛卡尔网格近似；Willmore 能量使用 Dirac 函数加权积分。

### `adaptive_mesh.py`
实现自适应网格生成与节点优化 `AdaptiveMesh`。  
- **职责**：根据水平集函数值定义尺寸函数（界面附近加密），执行全局均匀细化，并通过 Centroidal Voronoi Tessellation (CVT) 优化节点分布。还提供了三角形面积和质量度量工具。  
- **核心算法**：尺寸函数采用 tanh 形式；CVT 使用 Lloyd 迭代在密度加权（由水平集导出）下近似 Voronoi 单元的质心。

### `topology_tracker.py`
提供 `TopologyTracker` 类，用于检测界面拓扑变化（分裂/合并）。  
- **职责**：在界面带状区域内构建图，通过 BFS 查找连通分量，近似计算 Euler 示性数，记录拓扑历史并报告事件。  
- **关键算法**：基于邻接图的连通分量分析，BFS 求最短距离。

### `volume_corrector.py`
实现体积守恒修正器 `VolumeCorrector` 和外部力场生成器 `ExternalForcing`。  
- **职责**：通过平移水平集值使体积恢复目标值（Brent 法或二分法）；提供 ripple 式、振荡式、重力式等外部法向力场。  
- **核心思想**：将体积守恒转化为求 λ 使 V(φ+λ)=V₀，利用 Brent 法求解非线性方程。外部力场将非线性 ODE 思想推广至时空场。

### `convergence_analysis.py`
包含 `ConvergenceAnalysis`（计算 L²、L∞、H¹ 误差和收敛阶）和 `ReducedOrderModel`（基于 POD‑SVD 的降阶模型）。  
- **职责**：评估数值解的误差与收敛阶；对快照矩阵进行去均值后 SVD，按能量阈值截断，提供投影、重构、降阶算子等功能。  
- **关键算法**：SVD 分解，能量截断准则，降阶 Galerkin 投影。

### `sampling_engine.py`
提供 Latin 超立方中心采样器 `LatinCenterSampler` 和不确定性量化工具 `UncertaintyQuantification`。  
- **职责**：在多维参数空间中生成均匀覆盖的样本点；基于这些样本估计函数均值、方差和 Sobol 一阶敏感性指数。  
- **核心算法**：Latin Center 采样保证每维投影均匀；近似 Sobol 指数通过构造混合矩阵并计算乘积均值实现。

### `optimizer.py`
包含矩阵链最优括号化 `MatrixChainOptimizer`、约束满足暴力搜索 `ConstraintSatisfier` 和算子顺序优化器。  
- **职责**：动态规划求解矩阵链乘法最少运算次数；暴力搜索布尔变量赋值以判断公式可满足性；应用矩阵链思想优化线性算子链的浮点运算量。  
- **典型联系**：约束满足可处理 Young 方程（多相接触角）问题。

### `numerical_utils.py`
底层数值工具模块。  
- **职责**：提供 WENO5 正/负通量重构、WENO5 空间导数、TVD‑RK3 时间步、二/四阶中心差分、二维 Laplacian，以及复数矩阵的 Cholesky 分解、LU 分解、QR 分解和三角方程组求解。  
- **关键算法**：WENO5 采用三个候选模板和光滑指示子计算非线性权重；复数线性代数模仿经典 LINPACK 的流程。

### `main.py`（保留文件）
演示入口，依次调用各模块展示功能：初始化水平集、重初始化、曲率流、时间演化（含体积修正与拓扑追踪）、自适应网格与 CVT、收敛性分析与 POD、拉丁采样与 UQ、优化器、复数线性代数。

## 模块间关系和数据流

1. **基础层**：`numerical_utils` 提供差分、积分、分解等原子操作，被 `levelset_function`、`hj_solver`、`reinitialization` 等直接引用。
2. **核心描述**：`LevelSetFunction` 承载网格和水平集值，几乎所有其它模块都需要一个 `LevelSetFunction` 实例作为输入或成员。
3. **演化管线**：`HJSolver` 依赖 `LevelSetFunction` 计算右端项，`Reinitializer` 用于周期性地恢复 SDF 性质，`VolumeCorrector` 保持体积守恒，`TopologyTracker` 记录拓扑事件。这些在 `main.py` 的时间步循环中协同工作。
4. **几何与网格优化**：`CurvatureFlow` 和 `AdaptiveMesh` 使用 `LevelSetFunction` 计算几何量或尺寸场，可独立运行，也可嵌入演化流程中评估界面质量。
5. **分析与降阶**：`ConvergenceAnalysis` 通过构造精确解或连续快照评估算法精度；`ReducedOrderModel` 将多个时间步的水平集向量作为快照，构建 POD 基，以支持后续快速模拟。
6. **辅助工具**：`
