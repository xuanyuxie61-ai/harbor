# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# PROJECT_189 项目描述

## 项目概述
本项目实现了一套基于策略梯度（REINFORCE、自然梯度、GAE）的强化学习框架，用于控制一个包含锯齿波强迫、放牧型非线性耦合的四阶非线性振荡网络。整个科学计算栈覆盖特殊函数、线性代数、谱方法、随机过程、约束优化与状态空间几何分析。入口脚本 `main.py` 会依次执行数值验证、策略梯度训练以及状态空间三角剖分分析。

后续复现任务中，`main.py` 将被保留，其余 `.py` 文件需要根据本描述与 `main.py` 中可见的接口重新实现。

## 文件清单与职责

### `special_functions.py`
提供三类特殊函数：
- **正弦积分 Si(x)**：由分段级数展开、贝塞尔辅助展开以及渐近展开组合计算，用于奖励函数中的软饱和项。
- **贝塞尔函数零点**：采用 Halley 迭代法计算 J_n(x) 或 Y_n(x) 的第 k 个正零点，并提供批量计算前 k_max 个零点的接口。这些零点用于物理系统的特征模态分析和谱滤波器设计。
- **不完全 Beta 函数 I_x(p,q)** 与 **Beta CDF**：基于连分式或级数的高精度实现，为信任区域策略优化提供置信概率计算。

### `linear_algebra.py`
提供博士级数值线性代数工具：
- **RREF 求解**：将矩阵化为行简化阶梯形（RREF），并基于该分解求解可能亏秩的线性系统 A·x = b，同时提供秩的计算。
- **周期三对角矩阵（R83P）求解**：对带循环边界的周期三对角矩阵进行 LU 分解，内部使用非周期三对角分解与 Schur 补技巧；提供前后代入法求解原始系统及转置系统。
- **Toeplitz Cholesky 分解**：对半正定 Toeplitz 矩阵递推计算下三角 Cholesky 因子 L，并基于此因子实现从 Toeplitz 协方差中生成高斯样本。

### `spectral_basis.py`
提供谱方法与特征表示：
- **PCA（Turk‑Pentland 技巧）**：当样本数远小于特征维度时，通过小矩阵特征分解计算主成分向量、特征值和均值，支持单点和批量投影与重建。
- **Legendre 多项式及乘积基**：实现一元 Legendre 多项式递推求值，以及多元乘积多项式；支持生成所有总次数不超过指定上限的基函数矩阵。
- **贝塞尔谱滤波器**：利用贝塞尔零点生成围绕系统共振频率的带通滤波器。

### `stochastic_processes.py`
提供随机探索与统计工具：
- **方向均匀采样**（d 维单位球面）与**布朗运动轨迹生成**：基于独立高斯增量和随机方向。
- **Ornstein‑Uhlenbeck 过程**：模拟均值回归的连续时间随机噪声，用于动作探索。
- **多元正态距离统计**：通过 Monte Carlo 估计 d 维标准正态空间中两点欧氏距离的期望与方差，并提供理论 χ 均值公式。
- **高斯核矩阵**：计算状态点之间的 RBF 核矩阵，sigma 选用中位数启发式。

### `dynamical_system.py`
实现受控非线性振荡环境：
- **锯齿波驱动函数**：生成归一化锯齿波信号 S(t)。
- **放牧耦合非线性项**：描述振荡子系统（x1, x2）与生态子系统（x3, x4）之间的状态耦合。
- **环境类 `ControlledNonlinearOscillator`**：维护四维状态，提供 `reset`、`step` 方法；支持 Euler 和 RK4 两种积分器；奖励函数结合状态范数、动作范数和正弦积分项；参考轨迹使用贝塞尔函数 J0 生成准周期轨道。

### `policy_network.py`
基于谱基的参数化随机策略：
- **`SpectralPolicyNetwork` 类**：
  - 使用多元 Legendre 乘积多项式基作为特征，均值由参数与基向量的线性组合给出。
  - 对数标准差参数化以保持正性，并夹在预设范围内。
  - 动作采样可选用 Toeplitz 相关结构的高斯噪声（借助 `linear_algebra.py` 的 Toeplitz 采样）。
  - 提供 `log_prob`、`grad_log_prob`（返回均值参数和对数标准差参数的解析梯度）以及参数展平/恢复接口。

### `value_approximator.py`
值函数逼近与优势估计：
- **`PCAStateRepresentation`**：可对高维观测拟合 PCA 子空间并进行投影。
- **`SpectralValueFunction`**：利用 Legendre 乘积基在低维状态上做线性最小二乘回归；支持正规方程和基于 RREF 的求解；提供单点和批量预测。
- **辅助函数**：计算折扣回报序列 `compute_discounted_returns` 和广义优势估计（GAE）`generalized_advantage_estimate`。

### `natural_gradient.py`
自然策略梯度计算：
- 从状态‑动作样本中**估计 Fisher 信息矩阵**（样本外积平均加规则化）。
- **共轭梯度法**：以矩阵‑向量乘积形式实现 CG，阻尼项提升数值稳定性；配套 `fisher_vector_product` 函数直接计算 F·v 而不显式构造矩阵。
- **自然梯度更新方向** `natural_gradient_update`：支持 CG、直接求逆或回退到 CG 三种模式。
- **`NaturalPolicyGradientOptimizer` 类**：封装学习率与 KL 约束线搜索，执行一步自然梯度参数更新。

### `constrained_optimizer.py`
约束优化工具：
- **LP 投影** `lp_action_projection`：将违反线性约束 C·a ≤ d 及盒式界限的动作投影到可行域，通过求解引入辅助变量后的线性规划实现。
- **信任区域概率** `trust_region_probability`：利用不完全 Beta 函数估计策略更新满足 KL 约束的置信概率。
- **信任区域检查** `check_trust_region`：结合直接阈值与概率判断是否满足约束。
- **余弦退火学习率调度器** `CosineAnnealingScheduler`：周期性调整学习率。

### `mesh_geometry.py`
状态空间三角剖分与插值：
- **`StateSpaceTriangulation` 类**：对点云进行 Delaunay 三角剖分，实现单纯形定位、重心坐标计算、顶点值插值以及单纯形体积计算。
- **自适应网格加密** `adaptive_mesh_refinement`：根据局部值函数梯度估计在变化剧烈区域插入新采样点，直至达到最大点数。

### `policy_gradient_core.py`
核心训练器：
- **`TrajectoryBuffer`**：存储一条轨迹的状态、动作、奖励等数据。
- **`PolicyGradientTrainer` 类**：
  - 集成环境、策略网络、值函数、自然梯度优化器、余弦退火学习率以及 OU 探索噪声。
  - 提供 `collect_episode` 采集轨迹（可选确定性策略），`update` 执行一次策略与值函数更新。
  - 策略更新计算标准梯度（含熵奖励），加入 GAE 优势，进行梯度截断，再通过自然梯度优化器更新参数。
  - 值函数更新使用谱回归拟合回报。
  - `train` 方法运行多轮迭代并打印指标。

### `main.py`（保留）
项目入口，依次执行：
1. 数值验证：调用特殊函数、线性代数、随机过程、信任区域等模块进行正确性检查并打印结果。
2. 策略梯度训练：实例化 `PolicyGradientTrainer` 并训练，随后测试并输出平均奖励、参考轨迹跟踪误差。
3. 状态空间网格分析：采样轨迹点，进行 PCA 降维至二维，构建 `StateSpaceTriangulation`，输出剖分统计与插值测试。
最后汇总运行时间与关键指标。该文件在复现时不需更改，其余模块需根据它暴露的接口实现。

## 核心数据流与交互
- 环境（`dynamical_system.py`）产生观测 → 策略（`policy_network
