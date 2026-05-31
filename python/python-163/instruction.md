# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 项目描述：地热储层热-流-固耦合模拟

## 概述
本项目实现一个面向增强型地热系统（EGS）的热-水-力（THM）全耦合数值模拟。程序从零参数运行，生成合成数据，完成以下工作：
- 随机渗透率场与热物性场生成
- 基于达西定律与 Forchheimer 修正的流体流动
- 考虑对流-扩散的热传输（ETD-RK4 时间步进）
- 多孔弹性力学变形
- 裂缝网络马尔可夫演化
- 蒙特卡洛不确定性量化
- 数值收敛与误差分析

## 文件职责与模块边界

### `main.py`
程序入口。串联所有模块，依次执行参数初始化、材料属性构建、随机场生成、正交多项式拟合、网格划分与质量分析、积分规则测试、热传输求解、多孔弹性求解、裂缝/井筒/UQ 计算，最后输出结果文件。需保留该文件，后续 agent 将基于缺失的模块文件复现功能。

### `thm_model.py`
定义 THM 耦合系统的核心物理参数与状态容器。
- `THMParameters` 类：存储储层几何、网格、岩石/流体属性、初始/边界条件、数值控制参数，并预计算有效热物性。
- `THMState` 类：保存压力、温度、位移三个主变量场及当前时间。
- 提供若干辅助函数，如达西速度、有效热容、有效热导率、比奥模量、热扩散系数、应变张量、多孔弹性应力、温压依赖的流体密度和粘度等。

### `spline_properties.py`
基于三次样条（not-a-knot 边界条件）构建温度/应力依赖的材料属性插值器。
- `CubicSplineInterpolator` 类：从离散数据点计算分段三次多项式系数，支持求值和一/二阶导数。
- 提供默认样条函数：花岗岩热导率–温度、水粘度–温度、孔隙度–有效应力。

### `stochastic_fields.py`
生成储层非均质随机场。
- `StochasticDiffusivity2D` 类：基于截断的 Karhunen-Loève 展开构造二维随机扩散系数场。
- `LogNormalPermeabilityField` 类：用对数正态模型与近似特征函数生成一维和二维渗透率场。
- `RandomSampler` 类：提供随机采样工具（最小/最大值估计、拉丁超立方采样、蒙特卡洛期望）。
- `generate_stochastic_permeability_realization` 函数：为给定参数生成三维渗透率实现。

### `orthogonal_fit.py`
通过离散加权正交多项式拟合岩石属性的一维空间分布。
- `OrthogonalPolynomialFit` 类：利用三项递推构建正交基，计算展开系数，支持求值和单基函数评估。
- 提供便捷函数：`fit_permeability_field_1d`、`fit_thermal_conductivity_field_1d`。

### `mesh_geometry.py`
网格生成与几何分析。
- `Triangle` 类：计算三角形边长、面积、角度、外心/内心、质量指标等。
- `Tetrahedron` 类：计算四面体体积、边长、质量、形心。
- `triangle_mesh_quality` 函数：统计三角形网格质量。
- `reservoir_boundary_polygon`：返回简化的储层二维多边形。
- `triangulate_polygon_simple`：简单耳切法三角剖分。
- `generate_structured_hex_mesh`：生成结构化六面体网格。
- `convert_quadratic_tet_to_linear`：十节点二次四面体转四节点线性四面体。
- `reservoir_tetrahedral_mesh`：将六面体网格剖分为四面体网格。

### `quadrature_rules.py`
三维体积积分的高精度数值积分规则。
- 提供一维 Gauss-Legendre 规则（1–5 阶）。
- `cube_rule`：基于张量积构建长方体上的乘积高斯规则。
- `hexahedron_witherden_rule`：返回单位六面体上的对称积分规则，精度可达 11 阶。
- `integrate_scalar_field_hexahedron` / `integrate_vector_field_hexahedron`：对任意六面体区域做标量/矢量场积分。

### `etdrk4_solver.py`
热传输方程的时间推进求解器。
- `ETDRK4Solver1D`：一维周期性对流-扩散方程的谱方法 ETD-RK4 求解器。在傅里叶空间中预计算线性算子与系数，通过轮廓积分得到 ETD 系数，提供 `step` 和 `solve` 方法。
- `ETDRK4ThermalSolver`：二维热传输求解器，采用显式有限差分处理扩散和迎风对流，自动子步满足稳定性条件，并施加边界条件与物理限幅。

### `lambert_flow.py`
利用 Lambert W 函数求解井筒流动与非线性压力方程。
- `lambert_w_approx`：实数域 Lambert W 函数的高精度近似（分区域初值 + Halley 迭代）。
- `wellbore_pressure_drop_lambert`：基于达西-魏斯巴赫与 Forchheimer 修正的井筒压降计算。
- `solve_transcendental_pressure`：解形如 p·exp(βp)=C 的超越方程。
- `injection_well_pressure`：稳定注入井压力（Peaceman 等效半径）。

### `risk_fracture.py`
裂缝网络演化的马尔可夫模型。
- `FractureMarkovModel` 类：构建邻接矩阵与状态转移矩阵，通过幂迭代求稳态分布，提供分布演化与平均首次到达时间。
- `fracture_aperture_markov_evolution`：模拟热应力循环下裂缝开度的离散状态马尔可夫演化。
- `effective_permeability_from_fracture_network`：基于立方律的平均裂缝渗透率估计。

### `convergence_analysis.py`
数值误差
