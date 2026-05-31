# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# Python 地幔对流与板块运动数值模拟系统

## 项目概述
本项目是一个模块化的地幔对流数值模拟系统，集成了地球物理模型、数值方法、谱方法、网格生成与诊断分析等功能。所有计算基于 Python 和 NumPy/SciPy，主入口为 `main.py`，通过调用其他各模块完成演示性计算。您的任务是：在仅保留 `main.py` 的前提下，根据本描述重新实现所有缺失的 `.py` 文件，使得程序能正确运行并输出与原始项目一致的计算演示。

## 文件职责与核心要求

### 1. `spherical_geometry.py`
- **PiSpigot 类**：实现 spigot 算法计算高精度 π 值（可适当降精度或回退到 `math.pi`，但算法骨架应存在）。
- **SphericalGeometry 类**：管理地球表面半径、核幔边界半径，提供：
  - 球壳表面积、球壳体积计算。
  - 球坐标 `(r, θ, φ)` 与直角坐标 `(x, y, z)` 之间的批量转换。
  - 生成球壳内结构化网格点。
  - 必要的输入范围裁剪与边界处理。
- 数值均为双精度，半径、角度等采用国际单位（km）或弧度。

### 2. `mesh_generator.py`
- **PolygonTriangulator 类**：基于耳切法（ear-clipping）对简单多边形进行三角剖分。要求支持逆时针顶点顺序、共线和相交判断，返回 1‑based 的三角形索引。
- **VoronoiTessellator 类**：在二维矩形区域上用离散化方法近似计算 Voronoi 图，给定生成元，返回网格标签和各单元近似面积。
- **UnicycleIndexer 类**：提供循环置换索引的生成与序列转换，用于边界节点循环编号。
- **MantleMesh2D 类**：生成环形扇区的结构化三角形网格，并能复用三角剖分和 Voronoi 进行复合形状网格化。

### 3. `spectral_basis.py`
- **GramSchmidt 类**：提供经典和修正 Gram‑Schmidt 正交化算法，将矩阵列向量正交/单位化。
- **TrigonometricBasis 类**：实现基于 Dirichlet 核的三角函数基数，用于周期域上的插值或谱展开。需处理奇偶阶数的不同分母形式，并保证在节点处返回 1。
- **LagrangeInterpolation 类**：构建 Lagrange 插值基矩阵，并实现一维插值。
- **SpectralExpansion 类**：组合径向正交基（通过 Gram‑Schmidt 对幂函数基正交化得到）和角向三角函数基，用于谱展开。

### 4. `quadrature_engine.py`
- **GaussLegendre 类**：利用 Davis‑Rabinowitz 方法计算 Gauss‑Legendre 节点与权重，并提供 `[-1, 1]` 上的一维积分功能（支持仿射映射到任意区间）。
- **GaussLaguerre 类**：计算广义 Gauss‑Laguerre 节点与权重（通过 Jacobi 矩阵特征分解），并实现区间 `[a, ∞)` 上含权重 `(x-a)^α exp(-b(x-a))` 的积分。
- **Quadrature2D 类**：提供矩形上二维乘积 Gauss‑Legendre 积分，以及三角形上的简化积分（可使用形心权重法）。
- **HypercubeSampler 类**：在 `[0,1]^m` 中均匀采样，并提供蒙特卡洛积分估计（均值和标准误差）。

### 5. `mantle_physics.py`
- **MantleConstants**：定义地球地幔的物理常数（半径、密度、热膨胀系数、热扩散率、参考粘度、活化能等）。
- **ViscosityModel**：实现 Arrhenius 型和 Frank‑Kamenetskii 近似温度‑深度依赖粘度公式，需对输入温度做物理范围限定，并对输出粘度做截断。
- **DensityModel**：基于 Boussinesq 近似计算密度和浮力（相对于参考温度）。
- **DimensionlessNumbers**：提供瑞利数、努塞尔数、普朗特数、佩克莱特数的静态计算方法，含输入检验。
- **StokesPhysics**：实现从流函数 `ψ` 到极坐标速度分量 `(u_r, u_θ)` 的中心差分映射。
- **ThermalPhysics**：实现二维极坐标下的拉普拉斯算子、平流项以及单位体积生热率的计算。

### 6. `stokes_solver.py`
- **StokesSolver 类**：基于谱‑Galerkin 思想的 Stokes 流求解器（实际以有限差分‑迭代简化实现）。
  - 初始化时构建径向基（对 Legendre‑like 多项式进行 Gram‑Schmidt 正交化）和角向基数。
  - `compute_velocity_from_streamfunction`：根据温度场求解流函数泊松方程（用 Jacobi 迭代），并获得速度场。有效瑞利数可内置为常数。
  - `spectral_project_temperature`：将温度场投影到谱基上，用于模态分析。

### 7. `thermal_solver.py`
- **GrazingChemicalExchange 类**：描述上/下地幔化学储库交换的非线性耦合 ODE 系统，实现右端项计算和四阶 Runge‑Kutta 时间积分，并限制状态变量范围。
- **ThermalSolver 类**：
  - 生成初始温度场（可包含线性传导剖面或叠加正弦扰动以触发对流）。
  - 时间步进：采用全显式方法（平流+扩散+热源），自动检测 CFL 和扩散稳定性条件并限制时间步长；边界条件为内边界等温（CMB）、外边界等温（表面）、角向周期性；对温度做物理约束。
  - 计算表面无量纲热流，可供后续努塞尔数计算使用。

### 8. `diagnostics.py`
- **TemperatureHistogram 类**：基于排序样本构建离散累积分布并导出分段线性 PDF，提供规范化处理，并计算微分熵。
- **ChandrupatlaRootFinder 类**：实现 Chandrupatla 混合二次/二分法求根算法，封装临界瑞利数搜索（需实现 `find_critical_rayleigh`，通过给定的 Nusselt 函数寻找使 Nu≈1.05 的 Ra）。
- **LissajousForcing 类**：利用 Lissajous 曲线参数生成周期性强迫信号，并计算由此导致的地表热流调制。
- **ParameterSampler 类**：在参数边界内进行蒙特卡罗采样，计算模型输出均值和标准差。
- **MantleDiagnostics 类**：包装以上工具，提供温度场综合统计（均值、标准差、熵、直方图等）。

## 模块交互关系
- `main.py` 依次调用各模块的演示函数，验证子域正确性。
- 谱基、积分器、物理模块被 `stokes_solver.py` 和 `thermal_solver.py` 使用。
- `diagnostics.py` 有时会调用 `mantle_physics.py` 中的无量纲数计算（如 Nusselt 数），因此需保持模块间的接口一致。
- 所有文件中均使用 NumPy 的数组接口，并注意输入验证、形状兼容性以及浮点安全处理（如除以零保护）。

## 实现提示
- 不需要完全复现原始代码的每一行，但必须保证公开类和方法签名与 `main.py` 中的调用兼容。
- 数学公式的实现可采用标准数值方法，但要注意边界条件、稳定性限制、异常输入保护。
- 对于涉及随机性的模块（蒙特卡洛采样），保留可复现的种子参数设置。
- 不允许使用外部项目特定的第三方库，只能使用 `numpy`、`scipy` 等标准科学计算库（已在环境中提供）。
- 最终系统应从 `main.py` 运行，无错误输出并打印所有演示结果。
```
