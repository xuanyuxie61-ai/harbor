# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：神经随机最优控制与深度强化学习 — 统一数值框架

本项目是一个面向神经计算科研的 Python 数值实验平台。它围绕受随机微分方程（SDE）驱动的神经群体动力学，构建最优控制与深度强化学习的完整计算管线。主要功能包括：神经动力学建模、SDE 数值积分、Hamilton-Jacobi-Bellman（HJB）方程的有限元求解、策略优化、状态空间离散化、高维数值积分、收敛性与稳定性分析，以及基于 Actor-Critic 的强化学习训练。

后续 benchmark 将**保留 `main.py`，删除其他所有 `.py` 文件**。要求其他 agent 根据本描述以及 `main.py` 中的 import 和调用方式，自行实现各模块缺失的代码。

## 文件结构与职责

### `main.py`（保留，不修改）
项目的统一入口，以顺序执行的脚本形式组织所有实验环节。它导入所有子模块，依次进行环境检测、神经动力学验证、SDE 积分对比、HJB 方程有限元求解、Actor-Critic 训练、Nelder‑Mead 策略优化、高维积分、CVT 状态空间离散化、收敛性与稳定性分析，最后输出汇总结果。该文件定义了各环节的参数、辅助函数，并显式调用了各子模块提供的类和函数，agent 可以通过这些调用推断子模块必须实现的接口。

### `neural_mass_dynamics.py` —— 神经质量模型动力学
提供 Wilson‑Cowan 型兴奋‑抑制神经群体模型。负责：
- 以持久化字典管理模型参数（时间常数、耦合强度、激活函数参数、噪声强度等）。
- 实现 S 型激活函数（带防溢出保护）及其向量化版本。
- 计算模型右端导数（包括可选控制输入）。
- 计算系统 Jacobian 矩阵（用于局部稳定性分析）。
- 基于线性化特征值或简化公式估计振荡周期。
- 提供 LQR 型运行代价和终端代价函数，用于最优控制问题。

`main.py` 调用的关键函数：`get_neural_parameters`、`neural_mass_deriv`、`neural_mass_jacobian`、`neural_oscillation_period`、`compute_running_cost`、`compute_terminal_cost`、`sigmoid_activation`。

### `sde_integrator.py` —— 随机微分方程数值积分器
实现三种 SDE 数值格式以及相关辅助工具：
- **Euler‑Maruyama 方法**（强收敛阶 0.5）。
- **Milstein 方法**（强收敛阶 1.0），需要额外提供扩散系数对状态的导数。
- **随机显式中点法（Stochastic Explicit Midpoint）**，对 Stratonovich 型 SDE 有更好的稳定性。
- 布朗运动采样函数。
- 均方稳定性条件的理论检验函数。
- 强误差计算函数：通过粗细网格对比估计收敛行为。

`main.py` 调用了 `euler_maruyama`、`milstein_method`、`stochastic_explicit_midpoint`、`mean_square_stability_check`、`compute_strong_error`。

### `hjb_fem_solver.py` —— Hamilton‑Jacobi‑Bellman 方程有限元求解器
利用三维四面体网格上的线性有限元求解受控系统的后向 HJB 方程。
- **几何工具**：计算四面体体积、网格质量度量、生成规则三维区域上的 Kuhn 四面体剖分。
- **有限元组装**：对给定的漂移场和扩散场，在 P1 单元上组装质量矩阵 M 和聚合刚度/对流矩阵 A。其中扩散效应通过有效扩散系数（扩散矩阵乘积的迹）体现。
- **线性求解器**：实现针对一般非对称矩阵的双共轭梯度法（BiCG），并在失败时回退到直接解法。
- **HJB 时间步进**：采用隐式欧拉法从终端条件逆向求解，每个时间层通过离散化控制空间取极小值，并用 BiCG 求解大型稀疏系统。

`main.py` 调用了 `regular_tetrahedral_mesh`、`assemble_fem_matrices`、`solve_hjb_backward`、`tetrahedron_volume`、`compute_tet_quality`、`bicg_solver`。

### `policy_optimizer.py` —— 策略优化器
基于无梯度优化和抽样评估的策略求解工具。
- **Nelder‑Mead 单纯形优化**：针对非光滑目标函数（如受噪声扰动的轨迹代价）的鲁棒无梯度算法，包含反射、扩展、收缩等标准操作以及防坍塌机制。
- **策略代价评估**：通过蒙特卡洛运行并截取分位数来提高鲁棒性。
- **参数化策略**：提供线性反馈控制律和二次反馈控制律的函数，参数直接编码反馈增益。

`main.py` 调用了 `nelder_mead_optimize`、`linear_feedback_policy`、`evaluate_policy_cost`。

### `state_space_tools.py` —— 状态空间离散化与编码
将连续状态空间映射为离散表示以简化控制与学习。
- **变度量张量**：支持欧氏、Fisher 信息型、各向异性等度量。
- **CVT（质心 Voronoi 剖分）Lloyd 迭代**：利用变度量距离在区域内生成最优生成元，用于状态量化。
- **最近邻映射**：将连续状态归到最近的生成元索引。
- **状态编码器**：实现基于置换的索引编码与解码，用于信息混淆或压缩存储。
- **轨迹序列化**：以字典形式导出/导入状态-时间-控制序列。

`main.py` 使用了 `cvt_lloyd_iterate`、`state_to_index`、`StateEncoder`、`serialize_state_trajectory`、`metric_tensor`。

### `quadrature_engine.py` —— 高维数值积分引擎
提供在神经控制和随机分析中常用的积分工具。
- **金字塔区域积分**：实现 Jaskowiec‑Sukumar 对称积分规则（精度阶 p=0 ~ 6），并封装为在整个参考金字塔上积分任意标量函数的方法。
- **Gauss‑Hermite 一维求积**：将物理学家节点变换为概率积分（期望），支持任意标准差。
- **蒙特卡洛期望**：通过用户提供的采样器估计期望和标准误差，具备异常值过滤。

`main.py` 调用了 `pyramid_jaskowiec_rule`、`integrate_over_pyramid`、`gauss_hermite_quad_1d`、`monte_carlo_expectation`。

### `rl_agent.py` —— Actor‑Critic 强化学习智能体
针对连续状态、连续动作的在线策略学习组件。
- **RBF 值函数逼近器（Critic）**：使用径向基函数网络拟合值函数，支持梯度更新。
- **高斯策略（Actor）**：以 RBF 特征为输入的参数化高斯策略，输出均值通过 tanh 缩放到动作界内，支持采样和策略梯度更新（利用优势函数）。
- **Actor‑Critic 智能体**：组合 Critic 和 Actor，提供动作选择、单步 TD 训练、完整回合运行。训练使用 TD(0) 误差作为优势估计。

`main.py` 中直接实例化并使用 `ActorCriticAgent`。

### `numeric_utils.py` —— 数值工具集
提供跨越多个实验环节的基础数值功能。
- **环境检测**：报告 NumPy 版本、浮点精度等。
- **数值稳定性断言**：检查数组是否存在 NaN/Inf 等。
- **二分法求根**：实现经典二分法，支持误差容限和最大迭代次数，并封装为 Bang‑Bang 控制切换时间的搜索。
- **基准测试函数**：Rosenbrock 函数及其梯度、Himmelblau 函数，以及通用的优化器基准测试接口。
- **安全数值运算**：安全除法、软裁剪（可微近似硬裁剪）。

`main.py` 调用了 `check_environment`、`assert_numeric_stability`、`bisection_find_root`、`
