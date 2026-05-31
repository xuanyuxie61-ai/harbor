# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 项目描述：火箭发动机燃烧不稳定性的多物理场耦合分析系统

## 项目目标
本项目围绕"液体火箭发动机燃烧室热声不稳定性"这一工程问题，构建一个从几何建模到热声耦合预测的完整仿真分析链。系统入口为 `main.py`（零参数运行），其他 `.py` 文件将在后续被删除，需要 agent 根据本描述重新实现。

## 涉及的文件与模块职责

### 1. `main.py` — 主控与流程编排
项目的唯一入口文件，负责：
- 导入所有模块并按顺序执行九个分析步骤。
- 步骤依次为：燃烧室几何建模、喷注器面板布局优化、喷雾液滴分布优化、气液两相流动分析、一维燃烧波求解、声场模态分析、火焰传递函数计算、热声耦合振荡器预测、综合稳定性评估。
- 每个步骤调用对应模块的核心类/函数，打印摘要信息。

### 2. `utils.py` — 数值工具与物理常数
提供底层数学函数和火箭发动机相关的物理常数（单位：SI）：
- **CORDIC算法**：基于迭代旋转的高精度三角函数（正弦、余弦、反正切），用于几何角度计算。
- **Gamma函数与圆积分**：计算半整数Gamma函数、单位圆上单项式积分，用于截面统计特性分析。
- **数值稳定性工具**：安全除法、鲁棒平方根、数组有限性检查。
- **推进剂物性计算**：基于简化模型估计燃烧温度与理想比冲。

### 3. `geometry_model.py` — 燃烧室几何建模
包含类 `CombustionChamberGeometry`，负责：
- 定义圆柱燃烧室-收敛段-喉部-扩张段的关键几何参数。
- 基于分段函数计算任意轴向位置的半径与截面积。
- 计算截面矩、声学等效长度、纵向模态频率。
- 生成轴对称四边形网格（顶点、单元、边界标签），支持轴向/径向节点加密。
- 利用Joukowsky保角变换生成喷管型线。

### 4. `injector_layout.py` — 喷注器面板优化布局
包含类 `InjectorLayoutOptimizer`，负责：
- 在圆形面板上生成候选喷注单元位置（极坐标环分布或三角形密铺）。
- 将布局问题建模为受间距和总流量约束的价值最大化问题。
- 提供贪心启发式算法和暴力枚举求解器，输出选中单元集合、总流量、布局均匀性指标。
- 计算氧燃比空间分布（考虑壁面燃料膜冷却效应）。

### 5. `spray_dynamics.py` — 喷雾液滴动力学与分布优化
包含类 `SprayDistributionCVT`，负责：
- 基于Centroidal Voronoi Tessellation (CVT) 优化液滴在燃烧室内的空间分布。
- Lloyd迭代：在柱坐标系下采样、分配Voronoi区域、更新生成器位置、计算能量泛函。
- 生成初始分布（含液滴尺寸、速度），采用Rosin-Rammler分布。
- 基于$d^2$-law模拟液滴蒸发寿命（含阻力耦合与边界处理）。
- 输出喷雾统计特性（Sauter平均直径、浓度分布等）。

### 6. `two_phase_flow.py` — 气液两相流动分析
包含类 `StokesDropletFlow` 和 `TwoPhaseFlowSolver`，负责：
- 基于Stokes方程计算低雷诺数下液滴周围的速度场、压力场、表面剪切应力。
- 计算考虑内部循环的阻力系数（Hadamard-Rybczynski公式）及Basset历史力。
- 估计Nusselt/Sherwood数用于传热传质。
- 提供简化的1D稳态两相流求解器，沿轴向推进求解气相与液滴场量（含蒸发、动量交换、质量守恒）。

### 7. `combustion_wave.py` — 一维燃烧波传播
包含类 `NewtonInterpolation`、`CombustionRateModel` 和 `ReactionDiffusionSolver`，负责：
- 实现Newton差商插值，用于燃烧速率-压力关系曲线的插值与外推。
- 基于Saint-Robert定律（$r_b = a P^n$）的燃烧速率模型。
- 使用Jacobi迭代求解稳态一维反应-扩散方程（Arrhenius反应源项），获得温度场、火焰位置与厚度、数值火焰速度。
- 基于Zeldovich-Frank-Kamenetskii大活化能渐近理论估计层流火焰速度。

### 8. `acoustic_modes.py` — 声场模态分析
包含类 `FEMBasis2DTriangle`、`AcousticModeAnalyzer` 和 `FEMHelmholtzSolver`，负责：
- 二维三角形Lagrange基函数的构造与求值/梯度计算。
- 计算纵向、径向、切向声学模态频率与模态形状（基于解析公式和Bessel函数零点）。
- 模态正交性积分验证。
- Rayleigh准则计算（$R = \int p' q' dV$）及模态阻尼率估计。
- 提供简化的一维FEM Helmholtz本征值求解器（线性基函数，刚性壁/开放端边界条件）。

### 9. `flame_response.py` — 火焰传递函数与插值稳定性
包含类 `ChebyshevNDInterpolation`、`LebesgueStabilityAnalyzer` 和 `FlameTransferFunction`，负责：
- 多维Chebyshev级数插值（Clenshaw递推求值）。
- 计算Lagrange插值的Lebesgue函数和常数，比较等距节点与Chebyshev节点的稳定性。
- 火焰传递函数解析模型（包含时间延迟和一阶低通滤波）。
- 利用Newton插值从离散频率点重构FTF。
- Nyquist图数据生成与稳定性裕度（增益裕度、相位裕度）分析。

### 10. `thermoacoustic_oscillator.py` — 热声耦合振荡器
包含类 `ThermoacousticOscillator` 和 `MultiModeThermoacousticSystem`，负责：
- 建立带非线性饱和（Van der Pol型）的单模态热声振荡器ODE。
- 使用四阶Runge-Kutta积分求解压力脉动时程。
- 计算振幅包络、增长率、极限环振幅、振荡指标。
- 多模态耦合系统模型：包含模态间耦合矩阵的ODE系统及其RK4积分。

## 核心算法与数据结构
- **离散化/网格**：轴对称矩形/四边形网格（顶点列表、单元索引、边界标签），1D有限元网格。
- **数值方法**：CORDIC迭代、Newton插值、Chebyshev级数、CVT Lloyd迭代、Jacobi迭代、显式Euler与RK4积分。
- **物理模型**：Joukowsky变换、Stokes阻力/d²蒸发定律/Arrhenius反应/Zeldovich理论、Helmholtz方程/Rayleigh准则、火传递函数（n-τ模型）、Van der Pol振荡器。
- **数据结构**：多数函数返回字典（包含field arrays、收敛指标、物理参数），关键类保持明确的构造参数和内部状态（如几何参数、液滴位置/直径/速度）。

## 模块间依赖关系
- `main.py` 依赖所有其它模块。
- `two_phase_flow.py` 依赖 `geometry_model.py`。
- `utils.py` 被多数模块使用（常数、函数）。
- `flame_response.py` 中的 `FlameTransferFunction.interpolate_ftf` 可能调用 `combustion_wave.py` 中的 `NewtonInterpolation`。
- 其余模块之间基本独立，通过 `main.py` 串联数据流。

## 要求实现的要点
- 保持给定的API签名（类名、方法名、参数含义）与文档字符串中的功能描述一致。
- 实现应使 `main.py` 能够零参数运行并输出有意义的物理结果，不需追求高精度数值。
- 鼓励利用 `numpy` 进行向量化计算。
- 无需实现任何可视化或文件I/O。
```
