# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 酶催化反应过渡态搜索框架——项目描述

本项目的目标是为酶催化反应过渡态搜索提供一个博士级科学计算框架。代码由多个模块构成，每个模块负责一个独立的子任务，如分子拓扑分析、静电势计算、势能面插值、反应路径优化、过渡态验证、热力学积分等。整个框架通过 `main.py` 统一调用，`main.py` 内嵌演示数据，零参数运行即可展示各个模块的功能。

## 文件列表与主要职责

- **main.py**  
  程序入口。依次调用各模块的导出函数，执行分子拓扑解析、活性位点静电势求解、势能面 RBF 插值、ODE 动力学积分、热力学积分、构型空间铺砌、弦方法/NEB 路径优化、过渡态验证以及特殊函数计算，最后输出综合报告。该文件本身不包含复杂逻辑，仅用于编排演示流程并输出结果。

- **molecular_topology.py**  
  实现分子拓扑与图结构分析。提供 `XYZParser` 类用于从文本解析 XYZ 格式的分子坐标，`MolecularGraph` 类基于原子坐标和共价半径构建化学键图（距离截断），并能计算连通分量、查找催化三联体（如 Ser-His-Asp 模式）以及输出 METIS 格式的图表示。模块级函数 `analyze_molecular_topology` 封装完整的分析管道并返回图对象和统计结果字典。

- **electrostatics.py**  
  负责酶活性位点的二维 Poisson-Boltzmann 方程求解。`Poisson2DSolver` 类在规则网格上利用五点差分模板构建线性系统，通过 Gauss-Seidel 迭代计算电势分布，支持 Dirichlet/Neumann 边界和蛋白板障碍物。`pic_charge_density` 函数实现粒子云网格法（双线性权重）将粒子电荷分配到网格生成密度。此外还包含精确解验证函数 `poisson_2d_exact_solution` 以及静电稳定化能计算函数 `electrostatic_stabilization_energy`。

- **pes_surface.py**  
  从稀疏的从头算能量数据构建连续的势能面。`PESInterpolator` 类封装了多维径向基函数（RBF）插值，支持多二次、逆多二次、薄板样条和高斯核函数。用户需先调用 `compute_weights` 求解线性系统得到权重，然后通过 `interpolate` 评估任意点的势能，`gradient` 和 `hessian` 方法提供数值梯度与 Hessian。辅助函数 `estimate_r0` 可基于数据点分布自动估计核函数的形状参数。

- **dynamics_ode.py**  
  提供常微分方程积分器及示教模型。`RKF45Integrator` 实现了 Runge-Kutta-Fehlberg 自适应步长方法，带有误差控制和步长调整。`GlycolysisModel` 是 Selkov 糖酵解振荡模型，输出平衡解和 Jacobian 矩阵。`ReactionCoordinateDynamics` 可根据自由能函数的数值梯度模拟反应坐标的确定性演化。顶层函数 `integrate_glycolysis` 调用 RKF45 积分器记录振荡轨迹。

- **thermodynamic_quadrature.py**  
  热力学积分与几何矩计算工具。`GaussLaguerreQuadrature` 类通过构造 Jacobi 矩阵并利用 `IMTQLX` 对角化算法生成广义 Gauss-Laguerre 正交规则，用于形如 ∫(x-a)^α e^{-b(x-a)} f(x) dx 的积分。`PolygonMoments` 提供任意多边形区域上的非归一化矩、归一化矩和中心矩的 Steger 算法；`HexagonMoments` 专用于单位正六边形的矩计算。`ThermodynamicIntegration` 类根据能量剖面计算自由能势垒、活化自由能及熵贡献。

- **configuration_tiling.py**  
  构型空间的离散化与采样。`PentominoShapes` 定义了 12 种标准 pentomino 形状（如 F、I、L、T 等），用于铺砌模型。`ConfigurationTiling` 在给定的二维反应坐标范围内建立网格，通过能量截断标记可及区域，并利用 T 形拼板进行贪心铺砌覆盖，返回采样点坐标。`ConfigurationSpaceSampler` 实现 Metropolis Monte Carlo 采样以及基于线性插值的反应路径采样。

- **string_optimizer.py**  
  弦方法路径优化模块。`PathParameterization` 提供弧长参数化和等弧长重插值（三次样条）。`StringMethod` 类实现标准的弦方法演化：每次迭代计算各图像的势能与梯度，构造正交于路径的力分量，更新图像后重新参数化，直至力收敛。亦支持攀爬图像 NEB 的变体 `climbing_image_neb`。`SequenceManager` 用于生成序列文件名，`SymmetryOperations` 提供三维坐标的旋转、反射以及 C2v 对称操作，用于生成对称等价构型。

- **transition_state.py**  
  过渡态搜索与验证的核心模块。`NEBOptimizer` 实现标准的 Nudged Elastic Band 优化和攀爬图像 NEB，内部自动计算路径切向力与弹性力，并保持端点固定。`TransitionStateVerifier` 根据梯度范数和 Hessian 特征值判断鞍点，提取虚频并计算 Wigner 隧道校正因子；`rate_constant_tst` 基于过渡态理论计算反应速率常数。`ReactionPathAnalysis` 提供活化能计算、反应坐标值提取和路径曲率分析等辅助功能。

- **sparse_operations.py**  
  稀疏矩阵运算支持。`CRSMatrix` 类以压缩行存储格式表示稀疏矩阵，提供矩阵‑向量乘法和残差计算，并可从稠密矩阵构造。`build_molecular_hessian_crs` 根据原子坐标、力常数和截断距离构建分子体系 Hessian 矩阵的 CRS 表示。`lanczos_eigenvalue_solver` 实现 Lanczos 迭代算法，从稀疏矩阵中提取极端特征值，用于验证 Hessian 的谱特性。

- **special_math.py**  
  高级特殊数学函数。`clausen_function` 通过分段 Chebyshev 级数展开精确计算 Clausen 积分 Cl₂(x)。`periodic_torsion_potential` 利用傅里叶级数与 Clausen 修正项构建周期性二面角扭转势能。`angular_partition_function` 在给定温度下数值积分计算角向配分函数。

## 核心算法与数据流

整个框架按以下流程组织（与 `main.py` 中的演示一致）：

1. **分子拓扑** —— 从内嵌的 XYZ 数据出发，通过 `XYZParser.parse_string` 获得原子列表与坐标，再利用 `MolecularGraph` 构建键连关系，分析连通分量和催化三联体，输出 METIS 格式图。  
2. **静电势** —— 在二维网格上，用 `pic_charge_density` 分配电荷密度，然后由 `Poisson2DSolver` 的 Gauss-Seidel 迭代求解电势和电场，并与精确解对比验证。整个过程在无量纲化参数下进行，以避免数值溢出。  
3. **势能面插值** —— 在双阱势能面上随机采样，调用 `estimate_r0` 估计形状参数，通过 `PESInterpolator` 计算 RBF 权重，评估插值精度和梯度。  
4. **动力学（ODE）** —— 积分糖酵解模型得到振荡轨迹，展示 RKF45 积分器的自适应步长特性；同时演示一维反应坐标的确定性演化。  
5. **热力学积分** —— 生成 Gauss-Laguerre 正交规则并积分测试函数；计算正六边形和任意多边形的几何矩；从能量剖面计算活化自由能。  
6. **构型空间铺砌** —— 使用 Pentomino 形状在二维构型空间进行铺砌采样，并通过 Metropolis MC 采样器演示粗粒化模拟。  
7. **弦方法/NEB 路径优化** —— 基于 RBF 插值得到的势能面，先用 `
