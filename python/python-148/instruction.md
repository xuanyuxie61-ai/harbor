# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 多尺度神经场脑机接口信号解码系统

本项目构建了一套端到端的脑机接口（BCI）神经信号解码仿真系统融合神经群体动力学、二维神经场 PDE 求解、谱分析、连接组拓扑渗流、稳定性分析和最优电极空间采样等模型模块。

保留的唯一入口文件是 `main.py`，它调用核心流水线类 `BCIPipeline` 并输出多个分析结果。其他所有 `.py` 文件将被删除。你的任务是依据下面的描述重新实现这些缺失的文件使 `main.py` 能够正常运行并生成相似的科学指标输出。

---

## 模块文件与职责概述

### `utils.py` — 数学与数值工具库
提供整个系统依赖的底层数学函数和数值算法主要包括：
- 利用 **Clenshaw 递推** 高效求 Chebyshev 级数值的函数。
- 从连续函数通过 Chebyshev‑Gauss‑Lobatto 节点计算 Chebyshev 系数的函数。
- 生成周期锯齿波的函数（用于驱动神经振荡器）。
- 神经激活函数（如 sigmoid 与 softplus）及其稳定实现。
- 经典 **四阶 Runge‑Kutta 单步积分器**。
- 计算标准 **Gauss‑Legendre 节点与权重** 的函数（调用 numpy.polynomial）。
- 重心 **Lagrange 插值** 用于 Chebyshev 节点上的稳定计算。
- 从邻接矩阵构造图拉普拉斯矩阵的函数 `sparse_adjacency_to_laplacian`。
- 矩阵指数幂级数截断近似与数值稳定的 softmax 等辅助功能。

### `neural_mass_ode.py` — 神经群体振荡器与状态室模型
该文件实现多种神经元集群的电生理模型：
- **`EIOscillator`**：描述兴奋‑抑制（E‑I）Wilson‑Cowan 型神经质量模型其动力学由一对 ODE 组成包含 sigmoid 激活函数和锯齿波外部驱动提供模拟方法和由 E、I 状态计算局部场电位（LFP）的函数。
- **`AIRPopulationDynamics`**：借鉴 SIR 流行病模型构建的 Active‑Inactive‑Refractory 神经元群体动力学其中连接强度受外部兴奋输入调制并保持总神经元数守恒。
- **`MultiPopulationArray`**：管理多个 E‑I 振荡器通道引入通道间扩散耦合可同时模拟多通道 LFP 信号提供多通道模拟和 LFP 提取方法。

### `neural_field_solver.py` — 二维神经场 PDE 求解器
实现 Amari 神经场方程的数值求解：
- 提供生成规则矩形网格的函数 `generate_square_grid`（支持单元中心、顶点等偏移选项）以及多边形内部三角化网格点生成函数 `generate_polygon_grid_points`。
- 定义 **Mexican‑hat 连接核** （高斯差分）。
- **`NeuralFieldSolver`**：在规则网格上预计算连接核矩阵使用半隐式 Euler 或 RK4 进行时间推进模拟场活动还支持计算空间 2D FFT 功率谱。
- **`NeuralFieldWithGaussQuadrature`**：利用 Gauss‑Legendre 积分在网格单元内进行子像素精度积分预计算基于高斯求积的核矩阵以提高空间积分精度。

### `electrode_sampling.py` — 最优电极空间采样与皮层几何
提供电极布局优化和皮层表面建模：
- 实现二维点/三角形包含测试（叉积法）和点/多边形包含测试（射线法）。
- 生成六边形同心格点的函数 `hexagonal_grid_points`。
- **Lloyd CVT 迭代** 用于优化电极位置（蒙特卡罗近似 Voronoi 质心）。
- **`CorticalSurfaceGeometry`**：用球冠近似局部皮层表面可计算表面高度、法向量并生成六边形或 CVT 优化的电极 3D 位置。
- **`ElectrodeArray`**：管理电极阵列调用几何对象生成布局通过 Delaunay 三角剖分评估空间覆盖度提供采样神经场的接口。

### `connectome_topology.py` — 脑连接组拓扑与渗流分析
构建脑区连接图并进行渗流相变分析：
- **`BrainConnectomeGraph`**：生成随机加权对称邻接矩阵（支持对数正态分布权重）并保证图连通计算图拉普拉斯、Fiedler 值（第二小特征值）有效电阻、可通讯性矩阵（矩阵指数）并在图上模拟热扩散。
- **`NeuralPercolationAnalyzer`**：在二维格点上模拟位置渗流通过洪水填充标记连通分量检测跨越簇（左‑右或上‑下）估计渗流临界阈值并计算特定簇的分形维数（盒计数法）。
- **`ConnectomePercolationBridge`**：将连接组中的激活脑区映射到渗流格点模拟信息扩散分析传播后的最大簇大小与是否形成跨越簇。

### `spectral_signal_analysis.py` — 神经信号谱分析与最优空间采样
提供时频分析与二维最优采样工具：
- 实现 **Clausen 函数** 的分段 Chebyshev 展开用于周期性相位分析。
- 生成 **Padua 点集** 以及对应的近似积分权重这些点是正方形区域上代数最优的插值节点。
- **`ChebyshevSpectrumAnalyzer`**：对信号进行离散 Chebyshev 变换（基于 DCT）提取 Chebyshev 系数、谱能量、主导模式与直流分量并可重建信号。
- **`GaussLegendreSignalIntegrator`**：利用 Gauss‑Legendre 积分高精度计算信号的能量泛函及前四阶统计矩（均值、方差、偏度、峰度）。
- **`SpatialPaduaSampler`**：在 Padua 点处对二维标量场进行采样和积分并通过径向基函数插值到规则网格。

### `stability_and_roots.py` — 神经动力系统稳定性与根界分析
提供线性化系统的稳定性判据：
- 实现计算矩阵 **l₁、l₂、l∞ 对数范数** 的函数用于估计矩阵指数的上界。
- 实现 **Cauchy 多项式根界** 算法（区间加倍+二分法）获得特征值模的上界。
- 从方阵提取特征多项式系数的函数（Faddeev‑LeVerrier 算法）。
- 计算 E‑I 神经质量模型在给定状态处的 Jacobian 矩阵的函数。
- **`BCIStabilityAnalyzer`**：使用 Newton‑Raphson 寻找平衡点分析开环系统的 Jacobian 特征值、对数范数和根界判断稳定性；还支持计算闭环系统稳定性（加入线性反馈）并用 QR 分解近似计算有限时间 Lyapunov 指数。

### `bci_decoder.py` — BCI 解码核心流水线
这是顶层集成模块实现完整的信号生成→特征提取→解码→分析流程：
- **`SyntheticBCISignalGenerator`**：组装神经群体阵列、皮层几何、电极阵列、神经场求解器和连接组来生成多通道 LFP 信号、神经场时空活动以及在电极位置采样的电生理数据集。
- **`BCIFeatureExtractor`**：调用 Chebyshev 谱分析器和 Gauss‑Legendre 积分器提取每个电极信号的时域特征（统计矩、能量、主导模式等）并计算电极间的空间自相关和功率方差。
- **`BCIDecoder`**：采用带 L2 正则化的线性回归（岭回归）从特征向量映射到运动意图向量（例如 2D 速度）提供训练和解码方法。
- **`BCIPipeline`**：将上述组件与稳定性分析和连接组渗流桥接组合在一起实现单试次运行、训练‑测试解码评估以及完整的综合分析其 `run_full_analysis` 方法返回一个字典包含信号摘要、稳定性指标、连接组参数、解码误差和电极几何信息（这些正是 `main.py` 期望输出的字段）。

---

## 模块间依赖关系

- `utils.py` 被几乎其他所有模块直接调用提供基础数学函数。
- `neural_mass_ode.py` 依赖 `utils.py`（sigmoid、sawtooth、rk4）。
- `neural_field_solver.py` 依赖
