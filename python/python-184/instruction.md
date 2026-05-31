# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：python-184 — 时间序列预测与异常检测的多物理场耦合计算框架

## 整体概述
该项目是一个面向复杂时间序列的博士级计算框架，融合自回归建模、图谱分析、径向基函数重建、偏微分方程平滑、非线性动力学、贝叶斯求积、几何嵌入、离散模式检测以及制造解验证等多种技术。主要目标是对含有非线性、多尺度和未知噪声的时间序列进行预测、平滑、重构和异常检测，并通过多方法融合给出综合异常评分与评估指标。

框架入口为 `main.py`，它生成带有注入结构性异常的合成时间序列，然后依次调用各模块完成计算，最终输出各项指标（预测误差、异常检测的 Precision / Recall / F1 / AUC 等）。其余模块提供独立的算法实现，需要在代码复现时根据描述补全。

## 文件结构与职责

| 文件名 | 职责 |
|--------|------|
| `main.py` | 主程序入口，生成合成序列、协调各模块、融合异常得分、计算最终性能指标。需要保留，其他文件需复现。 |
| `toeplitz_solver.py` | 提供 Toeplitz 线性系统的 Levinson‑Durbin 求解器，用于 Yule‑Walker 方程和一般对称 Toeplitz 系统。 |
| `ar_predictor.py` | 基于 AR(p) 模型的时间序列预测器，包含参数拟合、多步预测、预测区间以及特征根稳定性分析（Newton‑Maehly 求根）。 |
| `graph_anomaly_detector.py` | 通过构造相似度图（k‑NN 或 ε‑球）进行异常检测，包含连通分量分析、PageRank 异常分数以及基于图拉普拉斯 Fiedler 向量的谱异常分数。 |
| `rbf_reconstructor.py` | 径向基函数（RBF）插值与重建，支持多种核函数，采用留一法重构误差作为异常得分，可用于缺失值插补。 |
| `pde_spatiotemporal_model.py` | 基于一维有限元的偏微分方程求解器：显式热方程平滑噪声，以及反应‑扩散方程模拟非线性演化模式。 |
| `nonlinear_ode_dynamics.py` | 非线性常微分方程模型：一个四物种生化反应网络（经典 RK4 积分，守恒量投影），以及扩展 Brusselator 振荡器及其 Lyapunov 指数估计。 |
| `quadrature_bayesian.py` | 广义 Gauss‑Hermite 求积规则的构造（通过三对角 Jacobi 矩阵的特征分解），用于贝叶斯预测分布的矩计算。 |
| `embedding_geometry.py` | 时延嵌入构造以及基于嵌入点云的几何质量分析，包括局部三角化质量评分和利用假近邻法（FNN）估计最小嵌入维度。 |
| `numerical_robustness.py` | 数值鲁棒性工具集：机器精度浮点遍历（nextafter / prevfloat）、Regula Falsi 求根、高维球 Monte‑Carlo 积分、Mahalanobis 球概率以及矩阵条件数敏感性分析。 |
| `manufactured_verification.py` | 采用制造解方法验证 PDE 求解器，生成解析解与强迫项，进行数值求解并比较误差，同时支持空间收敛阶研究。 |
| `discrete_pattern_kernel.py` | 在 GF(2) 上实现一维离散卷积，用于检测二进制时间序列中的突发连续异常或交替模式，并估计 Markov 转移矩阵与熵率。 |
| `triangular_feature_integrator.py` | 在参考三角形上实现对称数值求积，将三个时刻的特征映射到二维单纯形上，提取三角形上的积分特征。 |

## 模块边界与核心算法说明

### 1. Toeplitz 系统与 AR 建模
- **`ToeplitzSolver`** 利用 Levinson‑Durbin 递推高效求解对称正定 Toeplitz 系统。主要提供 `solve_yule_walker`（由自相关序列计算 AR 系数和反射系数）和 `solve_toeplitz`（求解一般 Toeplitz 方程）。递推过程中会检查预测误差功率，并对反射系数进行边界处理以防止奇异性。另外提供 Schur‑Cohn 稳定性判据（所有反射系数模小于 1）。
- **`ARPredictor`** 依赖 `ToeplitzSolver`，通过有偏自相关估计和 Levinson‑Durbin 得到 AR 参数，并估计噪声标准差。实现多步递推预测（使用自身历史与已预测值），以及基于正态假设的预测区间（脉冲响应累积方差）。内部还用 Newton‑Maehly 算法同时求解特征多项式的所有复根：利用 Horner 法则求值及导数，再通过 deflation 项防止收敛到同一根，最终给出特征根的模、时间常数、频率和阻尼比，用于稳定性/模态分析。

### 2. 图结构异常检测
- **`GraphAnomalyDetector`** 将时间序列窗口嵌入为图节点：构建 k‑NN 或 ε‑球邻接矩阵（无向图）。计算连通分量（BFS），进行 PageRank 异常评分（迭代直至收敛，得分低为异常），以及谱异常评分（计算归一化拉普拉斯矩阵的 Fiedler 向量，其绝对值作为异常度）。所有方法均封装在 `detect` 入口中。

### 3. RBF 重建与异常评分
- **`RBFReconstructor`** 实现多核（高斯、Multiquadric、逆 MQ、薄板样条）的 RBF 插值。通过求解带 Tikhonov 正则化的线性系统获得插值权值。提供重建和留一法异常评分：对每个点，用其余点训练模型并计算在该点的重构误差。

### 4. PDE 时空平滑
- **`PDE1DHeatExplicit`** 使用一维线性有限元（集中质量矩阵）和显式欧拉时间推进求解热方程。它会根据网格尺寸和 κ 自动调整时间步长以满足 CFL 稳定性条件。支持可选的源项函数。
- **`ReactionDiffusion1D`** 在一维非均匀网格上实现反应‑扩散方程，反应项为 logistic 增殖与 Michaelis‑Menten 消耗的组合。提供显式欧拉和 Heun（二阶）两种时间积分格式，扩散项采用变网格中心差分。

### 5. 非线性 ODE 动力学
- **`BiochemicalODE`** 描述四物种（酶、底物、复合物、产物）的生化反应网络，基于 stoichiometric 矩阵和反应速率（带 Michaelis‑Menten 饱和项）。使用经典 RK4 积分，并在每一步通过投影法近似维持守恒量，同时强制非负性。
- **`ExtendedBrusselator`** 模拟含 Hopf 分岔的非线性振荡器，提供雅可比矩阵，以及基于有限时间轨道扰动法的最大 Lyapunov 指数估计。

### 6. 贝叶斯求积
- **`GenHermiteQuadrature`** 通过构造对称三对角 Jacobi 矩阵（直接使用标准 Hermite 情形或平移缩放）并计算其特征值与特征向量，得到节点和权重。可对任意被积函数进行数值积分，并用于计算预测函数的矩（将后验假设为高斯分布，通过变量变换搭配标准 Hermite 权重）。

### 7. 时延嵌入与几何分析
- **`EmbeddingGeometry`** 根据 Takens 嵌入定理构造延迟嵌入矩阵。对每个嵌入点，利用 k 近邻构成局部三角形，评估其几何质量（最小内角归一化值、内切/外接圆半径比），综合为异常得分（质量低表示异常）。还实现了假近邻法（FNN），通过比较相邻嵌入维度的距离比估计最小嵌入维度。

### 8. 数值鲁棒性工具
- **`NumericalRobustness`** 提供一组与浮点精度和安全求根相关的工具：`next_float` / `prev_float` 用于机器精度下的浮点遍历；`regula_falsi` 实现 Illinios 变
