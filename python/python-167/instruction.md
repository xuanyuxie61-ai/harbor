# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：六足机器人多足步态优化系统

## 项目目标
本项目构建一个多模块计算的六足机器人步态优化系统。系统整合了地形建模、运动学/动力学仿真、中枢模式发生器（CPG）、轨迹规划、稳定性分析以及混沌全局优化等环节，最终自动寻找一组步态参数以改善机器人在非平坦地形上的运动表现。项目保留 `main.py` 作为整体流程入口，需实现其余所有支持模块。

## 涉及文件及职责概要

### `utils.py`
通用基础工具模块，提供：
- 高精度墙钟计时器 `Timer`
- 矩阵数值奇异性检测 `check_numerical_singularity`
- 安全除法 `safe_divide` 和鲁棒平方根 `robust_sqrt`
- 数值裁剪函数 `clip_to_bounds`
- 中心差分 Jacobian 近似、Householder 反射矩阵、Gershgorin 圆盘计算等

该模块被本项目几乎所有其他模块间接使用。

### `config_parser.py`
机器人配置解析模块，基于 XML 递归解析器思想：
- `parse_robot_config` 将 URDF/SDF 样式的 XML 字符串解析为嵌套字典结构
- `extract_link_params` 从字典中提取各连杆的质量、质心位置和惯性张量

### `terrain_model.py`
地形建模与足部接触分析：
- `TriangulatedTerrain` 类：表示三角网格地形。支持从 Triangle 格式的 `.node`/`.ele` 文件或 ASCII STL 文件载入；提供对给定平面坐标 `(x,y)` 的投影点高度和面法向量查询；实现基于重心坐标的三角形插值。
- `QuadrilateralTerrainPatch` 类：四边形表面双线性插值，可计算参数点位置、Jacobian 和法向量。
- `generate_sample_terrain` 函数：生成一个由三角函数构造的示例非平坦地形网格。

### `robot_kinematics.py`
多足机器人单腿运动学与接触约束：
- `SerialLegKinematics` 类：基于 DH 参数的三自由度串联腿正向运动学和几何 Jacobian。提供阻尼最小二乘法数值逆运动学求解，内部利用 `utils.clip_to_bounds` 保证关节角在限位范围内。
- `JointLimitConstraint` 类：将关节旋转空间测地距离映射为软约束惩罚，计算惩罚梯度。
- `FootContactGeometry` 类：定义 Coulomb 摩擦锥约束，提供摩擦锥违反量计算和接触力矩计算。

### `gait_dynamics.py`
步态动力学与中枢模式发生器：
- `CPGNetwork` 类：包含六个耦合 Hopf 振荡器的网络，可描述三足交替步态所需的相位关系。提供状态方程右函数及其解析出的相位、振幅。
- `TrapezoidalIntegrator` 类：基于固定点迭代的梯形法 ODE 积分器。
- `StanceSwingAutomaton` 类：基于 CPG 相位和相邻腿状态的离散支撑/摆动自动机，通过局部规则决定各腿的支撑或摆动相。
- `LegDynamics` 类：单腿的简化质量‑弹簧‑阻尼动力学，可计算给定关节力、接触力和当前 Jacobian 下的加速度。

### `numerical_solver.py`
大规模数值线性代数求解器：
- `CholeskySolver`：对称正定矩阵的 Cholesky 分解与方程组求解，内部使用 `utils.robust_sqrt` 处理对角元。
- `BlockTridiagonalSolver`：块三对角线性系统的正向消去和回代求解。
- `Radix2FFT`：基‑2 快速傅里叶变换及逆变换，并提供功率谱密度计算。
- `VandermondeSolver`：Vandermonde 线性系统的 O(n²) 求解，以及对应的多项式求值。
- `MatrixMultiplyBenchmark`：矩阵乘法运算。

### `trajectory_planner.py`
足端轨迹规划与优化：
- `TSPBruteForce` 类：旅行商问题暴力求解器，用于对候选足端落点进行最优排序。
- `PolynomialSwingTrajectory` 类：利用 5 次多项式拟合满足位置、速度、加速度边界条件的摆动轨迹。
- `FootfallPlanner` 类：综合落点规划器，结合 TSP 排序与多项式轨迹生成器，生成单腿 3D 摆动轨迹（水平直线加竖直拱形轨迹）。

### `stability_optimizer.py`
多足机器人稳定性分析与约束优化：
- `SupportPolygon` 类：用 Graham 扫描法计算所有支撑足位置的凸包，提供点包含性测试和到多边形边界的带符号距离。
- `StabilityMargin` 类：计算静态稳定性裕度（COM 到边界的距离）以及动态零力矩点（ZMP）位置。
- `SupportGraphCentrality` 类：将支撑状态建模为图，利用 PageRank 幂迭代评估各腿在整体稳定性中的重要性。
- `LinearStabilityConstraint` 类：将支撑多边形转化为线性不等式约束，为 COM 生成带安全裕度的可行域。

### `chaotic_search.py`
混沌全局优化模块：
- `LogisticMap` 类：生成 Logistic 混沌序列。
- `BarnsleyFernIFS` 类：基于 Barnsley 蕨类迭代函数系统在二维参数空间中采样。
- `ChaoticSimulatedAnnealing` 类：混沌模拟退火优化器，利用混沌扰动和 Metropolis 接受准则搜索最优解。
- `GaitParameterOptimizer` 类：步态参数综合优化器。优化变量包括步态周期、步幅、抬腿高度、耦合强度和阻尼系数；适应度函数综合考虑稳定性裕度和能量消耗。先使用 IFS 生成候选初始解，再通过混沌模拟退火进行精细搜索。

## 模块交互与流程概要
保留的 `main.py` 将依次调用上述模块完成：
1. 解析默认的六足机器人 XML 配置，提取各连杆物理参数。
2. 演示运动学正逆解、关节限位约束以及接触几何。
3. 构建地形网格，演示高度查询、四边形插值和文件读写。
4. 使用 CPG 网络与梯形积分器进行步态动力学仿真，结合支撑‑摆动自动机生成状态序列，并演示单腿动力学。
5. 运行 Cholesky 分解、块三对角求解、FFT 频谱分析、Vandermonde 插值等数值计算。
6. 对足端落点进行 TSP 排序并生成多项式摆动轨迹。
7. 计算支撑多边形与稳定性裕度，进行支撑图中心性分析和线性稳定性约束生成。
8. 最终调用混沌全局优化器 `GaitParameterOptimizer` 优化步态参数，并输出结果。

每个模块的公开接口（类名与主要方法签名）需与 `main.py` 中的调用保持兼容，但具体的数值策略、容差、迭代次数等可由实现者自行设定，只需保证功能正确且数值鲁棒。
