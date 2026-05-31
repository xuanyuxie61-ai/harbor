# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：惯性约束聚变（ICF）内爆多物理耦合模拟

本项目是一个一维球对称拉格朗日坐标系下的惯性约束聚变（ICF）内爆模拟框架。它耦合了激光能量沉积、电子热传导、流体动力学、DT 聚变反应、离子-电子能量弛豫、不稳定性分析以及中子输运等多个物理过程。各个物理模块之间通过共享状态数组（密度、温度、速度、物质组成等）进行数据交换，并由主时间循环统一推进。项目的核心算法源自多个经典数值计算方法，包括有限元、共轭梯度、高斯求积、自适应 Runge-Kutta 等。

## 文件清单与职责

- **`icf_parameters.py`**：定义所有物理常数、靶丸几何与材料参数、激光参数、数值控制参数、聚变反应参数。提供统一的参数类，其他模块通过导入这些类获取常数。
- **`mesh_generator.py`**：生成一维球形径向网格（`RadialMesh` 类），支持非均匀网格（在烧蚀层附近加密）并确保关键界面处有节点。同时提供 RCM 矩阵重排序与一维有限元邻接表构建。
- **`geometry_laser.py`**：创建 NIF 式的多束激光几何排布（`LaserBeam` 类），计算激光束与球面的交点、入射角，并生成激光能量在径向网格上的几何沉积权重。
- **`state_equation.py`**：等离子体状态方程模块，计算电子/离子压强、内能、声速，包含理想气体、电子简并压、库仑修正、辐射压，以及 Saha 电离度计算。使用 `quadrature_rules` 中的 Fermi-Dirac 积分。
- **`hydrodynamics.py`**：一维拉格朗日球对称流体力学求解器（`LagrangeHydro` 类），求解动量方程和能量方程，包含人工粘性、显式 predictor-corrector 时间推进，以及 CFL 条件的时间步估计。
- **`heat_conduction.py`**：电子热传导求解模块，基于 Spitzer-Harm 热导率（含热流限制），用隐式离散构建三对角线性系统，通过共轭梯度法求解节点温度。
- **`laser_propagation.py`**：激光在等离子体中的传播与能量沉积模型，基于逆轫致辐射吸收，计算激光随径向深度的衰减，并得到体沉积率。
- **`fusion_reactions.py`**：DT 聚变反应率计算（Bosch-Hale 参数化）、alpha 粒子沉积、中子蒙特卡洛输运（使用 Niederreiter 低差异序列），以及中子能谱直方图统计。
- **`instability_analysis.py`**：Rayleigh-Taylor 和 Richtmyer-Meshkov 不稳定性分析，包括 IFS 表面粗糙度生成、模式增长率谱计算、能量流网络构建与 PageRank 评估。
- **`time_integrator.py`**：自适应时间积分器，包含 RKF45 嵌入式 Runge-Kutta 方法与 Cliff 随机数生成器，用于稳健的时间推进，也提供备用显式 Euler 步。
- **`diagnostics.py`**：通过多角度视线诊断重建三维内爆形状，基于 theodolite 项目算法，包括模拟诊断阵列、弦长测量、对称性残差最小化和形状参数反演，现未在主循环中使用。
- **`eos_opacity.py`**：包含理想气体+辐射压状态方程、Saha 电离、轫致辐射不透明度与 Rosseland 平均不透明度计算，以及电子热导率和能量流有向图构建，现未在主循环中使用。
- **`fem1d_radiation.py`**：一维球坐标辐射扩散方程的有限元求解器，包含 Lagrange 形函数、刚度/质量矩阵组装、theta-方法时间推进，依赖于 `matrix_utils` 的三对角 CG 求解器。
- **`laser_coupling.py`**：激光与等离子体耦合的更详细模型，包括逆轫致吸收系数、临界密度、激光包络的傍轴波动方程、以及多径积分沉积轮廓，现未在主循环中使用。
- **`matrix_utils.py`**：稀疏矩阵工具，包括 RCM 重排序和三对角矩阵的共轭梯度法求解器（`r83v_cg`），供辐射扩散和热传导等模块使用。
- **`monte_carlo_neutron.py`**：另一个中子蒙特卡洛输运实现，使用 Niederreiter 准随机序列和 Cliff RNG，支持中子源采样、输运跟踪和能谱分箱，与 `fusion_reactions` 中的中子输运功能类似。
- **`quadrature_rules.py`**：基于 Golub-Welsch 特征值方法的高斯求积法则生成器，支持 Gauss-Jacobi、Legendre、Chebyshev 求积，并提供 Fermi-Dirac 积分和 Planck/Rosseland 平均积分的计算。
- **`quadrature_utils.py`**：封装 SciPy 的 Gauss-Jacobi 和 Gauss-Legendre 求积规则，提供区间缩放接口，用于其他模块的数值积分。
- **`rayleigh_taylor.py`**：Rayleigh-Taylor 不稳定性建模，包含经典和烧蚀致稳 RT 增长率、非线性模式耦合的 IFS 混沌模型、混合宽度估计等，独立于 `inst
