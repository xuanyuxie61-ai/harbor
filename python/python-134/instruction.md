# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# PEMFC 质子交换膜燃料电池水管理数值模拟系统

本项目的目标是构建一个博士级多物理场耦合数值模拟平台，用于研究质子交换膜燃料电池（PEMFC）内部的水管理过程。系统采用模块化设计，整合了化学计量学、电化学反应动力学、静电学、暂态扩散与对流传输、多孔介质输运、蒙特卡洛材料特性估计、最优传感器布局、结构化线性代数以及实验数据生成等多个子领域。主入口 `main.py` 以零参数方式驱动全部流程，顺序调用十二个科学计算模块，完成从物理参数初始化到残差分析的全闭环仿真。

## 项目文件结构及各模块职责

### `main.py`
主控程序，负责定义物理参数，并按逻辑顺序调用所有模块。它本身不包含深层算法，而是作为顶层编排器，将各模块的输出串联起来，并在控制台打印关键中间结果与最终汇总。

### `stoichiometry_balancer.py`
化学计量学整数平衡模块。利用 Diophantine 方程（整数高斯消元）对 PEMFC 阴极氧还原反应（ORR）进行原子/电荷守恒的整数平衡，返回最小整数系数组合并验证残差。提供函数：`balance_orr_stoichiometry` 和 `verify_stoichiometry_solution`。

### `electrochemistry_kinetics.py`
电化学反应动力学模块。基于 Butler‑Volmer 方程描述电极局部电流密度与过电位的关系，计算温度修正的交换电流密度，并提供由目标电流反解活化过电位的功能。同时包含多组分反应源项计算与守恒量提取，支撑后续极化曲线生成。

### `proton_potential_solver.py`
质子电势场求解模块。在二维矩形域上求解变系数稳态 Poisson 方程，得到膜内质子电势分布。模块内嵌膜电导率与水含量的经验关系，并支持对结果进行 Hermite 三次样条插值。同时提供闭合形式的解析解用于验证（`proton_potential_exact`）。核心函数：`solve_proton_potential` 和 `interpolate_proton_potential_hermite`。

### `membrane_water_transport.py`
膜内水传输瞬态求解器。求解一维对流‑扩散‑反应方程，描述膜内水含量 λ(z,t) 的演化。采用隐式向后 Euler 时间积分处理刚性，并通过 Picard 迭代处理非线性系数；空间离散使用中心差分，线性系统通过 Thomas 算法直接求解。同时提供用于验证的波动方程精确解和刚性 ODE 精确解。

### `porous_gdl_transport.py`
气体扩散层（GDL）多孔介质传质模块。模拟 GDL 中液态水饱和度 s(z,t) 的非线性扩散过程。毛细扩散系数基于 Leverett J‑function 与相对渗透率模型计算；时间推进采用显式格式，并依据稳定性条件自适应调整时间步长。模块提供 Barenblatt 自相似解析解用于验证（`porous_medium_exact`）。

### `mesh_generator.py`
三维四面体网格生成与细化模块。生成简化 PEMFC 计算域（包含流道、脊、膜、催化层、GDL 的分层结构）的初始四面体网格，并实现 Liu & Joe 的 8‑子四面体细化算法：对每条边插入中点，将每个四面体剖分为 8 个子单元，并去除退化单元。提供网格质量评估（体积统计）。

### `monte_carlo_clustering.py`
催化层蒙特卡洛估计模块。在四面体网格上通过蒙特卡洛采样估计催化层有效扩散系数（基于 Bruggeman 修正与孔隙连通性），以及水团簇尺寸分布。采样时利用参考四面体到物理四面体的仿射变换；采用重心坐标法在单元内均匀撒点。

### `optimal_sampling.py`
最优传感器布置模块。基于 Centroidal Voronoi Tessellation (CVT) 和 Lloyd 迭代，在膜平面上寻找密度加权最优测点位置。密度函数反映水含量监测优先级，由连个高斯峰组合构成；初始生成点由离散 PDF 采样得到，迭代过程在离散网格上计算 Voronoi 质心。

### `banded_linear_algebra.py`
结构化线性代数模块。集中实现三类高效代数运算：对称正定带状矩阵（R8PBL 紧凑存储）的生成、格式转换与矩阵‑向量乘法；密集矩阵的 LU 分解（部分主元）及求解（LINPACK 风格）；Hankel 矩阵的 Cholesky 分解以及基于自相关函数的协方差因子构造。最终提供一个综合求解带状线性系统的性能测试案例。

### `synthetic_experiments.py`
合成实验数据生成模块。根据参数化物理模型产生燃料电池极化曲线（I‑V 特性，包含活化、欧姆、浓差过电位）、电化学阻抗谱（基于简化 Randles 电路）以及多温度‑湿度条件下的性能扫描数据，用于验证模型的实验一致性。

### `convergence_analysis.py`
收敛性与残差分析模块。对多物理场求解结果计算各控制方程的无量纲化离散残差（质子 Poisson 方程、膜水传输方程、GDL 多孔介质方程），进行网格收敛性研究（利用 Richardson 外推估算收敛阶），并给出膜内整体水的质量平衡误差。

## 核心数据流与模块间关系

`main.py` 中 `setup_physical_parameters` 函数定义包含温度、压力、交换电流密度、膜厚度、GDL 厚度等近百个物理参数的字典 `params`，该字典几乎贯穿所有模块，作为主要的科学与数值配置载体。

- 首先由 `stoichiometry_balancer` 完成化学计量分析，产生原子守恒的整数解。
- 随后 `mesh_generator` 生成并细化三维四面体网格，为蒙特卡洛估计提供几何信息。
- 电化学动力学是由 `electrochemistry_kinetics` 独立计算交换电流密度与 Butler‑Volmer 曲线，并为后续极化曲线生成提供基础。
- 二维电势场 `solve_proton_potential` 接收参数和可选的水含量场，返回电势矩阵与网格，供残差分析和插值使用。
- 膜内水传输求解器 `solve_membrane_water_transport` 利用参数与电流密度分布（如未提供则采用默认线性分布），输出稳态水含量剖面与时间网格，后续用于质量平衡和 Hankel 协方差估计。
- GDL 饱和度由 `solve_gdl_saturation` 求解，返回饱和度与空间坐标，并参与残差分析。
- 蒙特卡洛模块 `estimate_effective_diffusivity_monte_carlo` 使用细化网格与参数估计有效扩散系数；`estimate_water_cluster_distribution` 则输出团簇阈值与占比。
- 最优采样模块 `optimize_sensor_placement` 根据参数中指定的传感器数量进行 Lloyd 迭代，输出物理平面内的最优传感器坐标。
- 线代数模块 `solve_banded_linear_system` 演示集成求解性能，内部调用 R8PBL 和 LINPACK 子程序。
- 合成实验模块 `generate_polarization_curve` 和 `generate_impedance_spectrum` 分别返回极化曲线和阻抗谱数据，用于性能评估。
- 收敛性分析 `compute_residuals` 综合电势场、水含量剖面、GDL 饱和度及参数，返回各方程的无量纲残差；`compute_mass_balance_error` 计算膜内总水平衡。

整个工作流体现了从底层化学反应平衡、微观传输特性估计，到宏观场求解、传感器布局优化，再到数据合成与误差分析的完整数值模拟闭环。每个模块均独立设计并暴露最小化的公共接口，便于复现和维护。

## 各模块关键算法概览（不含实现细节）

**化学计量平衡**  
通过整数高斯消元求解齐次 Diophantine 方程组，获得原子守恒通解的整数基与特解，并选取最小正整数组合作为 ORR 的平衡系数。

**电化学动力学**  
Butler‑Volmer 方程采用对称/非对称传递系数形式，通过剪切指数参数防止数值溢出；活化过电位反算则使用 Tafel 近似或完整反演。

**质子电势场**  
二维 Poisson 方程的变系数五点
