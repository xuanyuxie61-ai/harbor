# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：表面催化反应分子动力学模拟集成系统

本项目构建了一个面向 Pt(111) 表面 CO 氧化催化反应（2CO + O₂ → 2CO₂）的多尺度模拟框架，结合了表面结构建模、势能面构建、电子结构计算、随机动力学积分、反应‑扩散方程、蒙特卡洛采样、马尔可夫链主方程和反应坐标分析等模块。代码由多个 Python 文件组成，主入口为 `main.py`，其余文件提供各子模块的实现。

## 文件清单与职责

- **`utils.py`**  
  提供物理常数、单位转换、常用势函数和网格生成工具。  
  包含：SI 制物理常数（玻尔兹曼常数、基元电荷、原子质量单位等）；Pt(111) 晶格常数；数量转换函数（如 `kb_t_ev` 将温度转换为 eV 能量标度）；解析势函数（Morse 势、Lennard‑Jones 势）；Arrhenius 速率公式；一维和多维均匀网格生成函数；安全除法等辅助函数。  
  几乎所有其他模块都依赖此文件。

- **`catalyst_surface.py`**  
  定义 Pt(111) 表面晶格结构与吸附位点管理。  
  核心类：`Pt111Surface`。  
  功能：构建 FCC(111) 面心立方表面原子坐标（ABC 堆垛），生成高对称吸附位点（top、bridge、fcc‑hollow、hcp‑hollow），计算位点能量；采用一维元胞自动机（规则 30）演化位点占据态以模拟吸附/脱附过程；使用三维 Centroidal Voronoi Tessellation (CVT) 优化吸附位点分布（Lloyd 算法）；计算表面覆盖率、横向相互作用能等。  
  依赖 `utils.py` 中的常数和网格工具。

- **`potential_surface.py`**  
  构建和分析催化剂表面势能面（PES）。  
  核心类：`PotentialEnergySurface`。  
  功能：基于多维多项式插值拟合从头算数据点，使用 Vandermonde 矩阵和最小二乘求解多项式系数；提供势能值、梯度和 Hessian 矩阵的计算；利用 Newton‑Raphson 方法搜索鞍点（过渡态）；通过线性插值路径估计反应活化能（NEB 近似）。  
  包含一个演示函数 `build_co_oxidation_pes_demo`，用于生成 CO 氧化反应的解析模型势能并拟合。  
  依赖 `utils.py` 中的势函数和物理常数。

- **`tight_binding.py`**  
  紧束缚电子结构计算模块。  
  核心类：`TightBindingSolver`。  
  功能：基于 Slater‑Koster 参数化形式构建表面体系的 Hamiltonian 矩阵（含截断半径和指数衰减的 hopping 积分）；求解本征值问题并计算电子态密度（DOS，高斯展宽）；通过 Fermi‑Dirac 分布计算能带能量；实现压缩边界带状矩阵分解（R8CBB 风格的 LU 分解）；支持 SLAP Triad 格式的稀疏矩阵输入/输出；附带一维 Poisson 方程有限差分求解器。  
  依赖 `utils.py` 中的常数和温度转换。

- **`langevin_integrator.py`**  
  随机分子动力学积分器，用于模拟表面吸附物种在热涨落下的运动。  
  核心类：  
  * `LangevinIntegrator` — 实现 BAOAB 分裂方案（二阶精度）的 Langevin 动力学积分器，包含初始化、时间步进、动能/温度计算、均方位移 (MSD) 和扩散系数估计。  
  * `StochasticReactionDynamics` — 将 Langevin 动力学与 Gillespie 随机模拟算法结合，用于模拟表面吸附、脱附、扩散和反应的事件。  
  依赖 `utils.py` 中的单位转换常数，以及 `catalyst_surface.py` 和 `potential_surface.py` 的类。

- **`reaction_diffusion.py`**  
  一维反应‑扩散方程求解器，用于宏观尺度表面覆盖度演化。  
  核心类：  
  * `ReactionDiffusion1D` — 使用有限差分法求解稳态和时间依赖的反应‑扩散方程，支持 Dirichlet 和 Neumann 边界条件，采用 Newton 迭代处理非线性反应项。  
  * `LangmuirHinshelwoodKinetics` — 基于 Langmuir‑Hinshelwood 机理的 CO 氧化动力学模型，包含覆盖度演化的常微分方程及其数值积分（RK4），并计算稳态覆盖率。  
  依赖 `utils.py` 中的温度转换。

- **`monte_carlo.py`**  
  蒙特卡洛采样与数值积分模块。  
  核心类：  
  * `MonteCarloSampler` — 在超矩形内均匀采样，估计高维积分并分析收敛性；计算热活化吸附概率。  
  * `QuadratureIntegrator` — 提供复合梯形、复合 Simpson 和三点 Gauss‑Legendre 求积公式。  
  * `PiecewiseLinearProductIntegral` — 精确计算两个分段线性函数乘积的积分，并用于热平均反应速率常数估计。  
  依赖 `utils.py` 中的常数。

- **`markov_kinetics.py`**  
  马尔可夫链主方程求解器，用于描述表面反应网络的状态跃迁。  
  核心类：`SurfaceReactionNetwork`。  
  功能：枚举代表性表面构型状态；基于吸附、脱附和反应事件构建转移速率矩阵；求解主方程时间演化；计算稳态分布、转换频率（TOF）、平均首次通过时间（MFPT）和稳态熵产生率。  
  依赖 `numpy` 和 `utils.py` 中的常数。

- **`svd_reaction_coords.py`**  
  反应坐标降维与主成分分析模块。  
  核心类：`ReactionCoordinateAnalyzer`。  
  功能：对分子动力学轨迹进行奇异值分解（SVD），提取主成分（反应坐标）；计算方差贡献率、自由能剖面、集体性指数；实现可交换性分析（Committor）。  
  包含一个测试轨迹生成函数 `generate_test_trajectory`。  
  依赖 `utils.py` 中的常数。

- **`main.py`**  
  主入口脚本，按模块顺序调用上述所有子模块，展示一个完整的 Pt(111) 表面 CO 氧化催化反应模拟流程。  
  依次执行：表面结构构建与 CVT 优化、势能面构建与过渡态搜索、紧束缚电子结构计算、Langevin 动力学与 Gillespie 事件模拟、反应‑扩散方程与 LH 动力学、蒙特卡洛积分与采样、马尔可夫链主方程分析、SVD 反应坐标分析。  
  每个模块的输出被打印到控制台，用于验证代码正确性。

## 模块间关系

- `utils.py` 是基础设施层，被所有模块导入。
- `catalyst_surface.py` 和 `potential_surface.py` 为 `langevin_integrator.py`、`tight_binding.py` 等提供表面几何和能量信息。
