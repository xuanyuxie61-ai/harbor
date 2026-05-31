# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 系外行星大气光谱反演系统

本项目是一个用于系外行星透射光谱模拟与参数反演的科学计算工具。它通过正向模型生成模拟观测数据，并利用非线性最小二乘优化和贝叶斯马尔可夫链蒙特卡洛（MCMC）方法，从噪声光谱中恢复大气温度、化学丰度、云层性质等参数。整个系统围绕一系列物理和数值模块构建，每个模块封装了特定的算法或物理过程，最终由 `main.py` 统一调度执行完整的反演流水线。

## 模块总览

项目包含以下 Python 源文件，其中 `main.py` 为入口，其余文件将由你根据本描述重新实现：

- `main.py`：主控程序，生成模拟数据、执行反演与诊断，并输出结果。
- `atmospheric_model.py`：行星大气物理模型（温度剖面、化学平衡、云层）。
- `data_io.py`：科学数据读写与格式转换。
- `inversion_solver.py`：非线性反演求解器（二分法、Broyden 方法、Levenberg-Marquardt、Tikhonov 正则化）。
- `mesh_generator.py`：一维/二维大气网格生成与自适应加密。
- `monte_carlo_sampler.py`：贝叶斯采样器（MCMC、嵌套采样、单纯形采样、粉红噪声生成）。
- `radiative_transfer.py`：辐射传输方程求解与透射光谱计算。
- `sparse_linear_algebra.py`：稀疏矩阵 CRS 格式存储与 GMRES 求解器。
- `spectral_synthesis.py`：分子吸收截面与 Voigt 线型计算。
- `sphere_quadrature.py`：球面角度积分与 Delta-Eddington 近似。

下面逐一说明各模块的职责、核心接口和所采用的算法，以便你能够独立实现并保证与 `main.py` 的调用兼容。

## 1. atmospheric_model.py

**职责**：建立行星大气的物理参数化模型，包括温度-压强（T-P）剖面、化学平衡丰度、行星重力场以及云层光学厚度。该模块封装了行星基本物理常数和大气状态方程，是整个正向模型的基础。

**核心类**：
- `AtmosphericProfile`：描述行星大气的竖直剖面。存储行星质量、半径、恒星质量、轨道距离等参数，计算平衡温度，并提供重力加速度、T-P 剖面（Guillot 模型与等温模型）、压强分层网格、大气标高、以及从压强剖面反算海拔高度的方法。
- `ChemicalEquilibrium`：基于简化化学平衡计算各物种的体积混合比（VMR）。参数化考虑温度、压强、金属丰度和碳氧比，并支持计算平均分子量以及对丰度进行对数正态不确定性采样。
- `CloudModel`：简化的云层模型，使用高斯型光学厚度分布描述云层在特定压强范围内的消光贡献。

**关键算法**：
- Guillot 温度剖面：基于内部热流和辐照的解析解，表达为光学厚度的函数。
- 对数等间距压强网格构造。
- 简化化学平衡参数化：VMR ∝ (P/P0)^α · exp(-E/kT)，结合边界裁剪。
- 云层光学厚度：τ(P) = τ0 · exp(-[(logP - logPc)/σ]^2)。

**与 main.py 的关系**：`main.py` 使用 `AtmosphericProfile` 生成行星实例，用 `ChemicalEquilibrium` 计算真实化学丰度，用 `CloudModel` 构建云层，并通过这些对象计算正向模型所需的温度、气体成分和云光学厚度。

## 2. data_io.py

**职责**：提供科学数据读写的结构化容器和格式转换工具。它融合了类似 TECPLOT 格式的节点/单元数据管理，以及光谱 ASCII 表格的读写，并支持元数据的 JSON 序列化。

**核心类**：
- `TecDataset`：模拟科学数据集容器，包含变量名、节点数据（二维数组）和单元连接关系，支持按变量名存取数据，并提供 ASCII 格式的读写方法。
- 独立函数：`write_spectrum_ascii` 和 `read_spectrum_ascii` 用于读写三列（波长、流量、误差）光谱文件；`save_json_metadata` 和 `load_json_metadata` 用于保存和加载 JSON 元数据。

**关键算法**：并无复杂算法，主要是格式解析和数组拼接。`TecDataset.read_ascii` 需要解析文件头部的 TITLE、VARIABLES 和 ZONE 行，然后分节点数据行和单元连接行。`write_spectrum_ascii` 和 `read_spectrum_ascii` 处理简单的空格分隔数值文件。

**与 main.py 的关系**：`main.py` 调用 `write_spectrum_ascii` 输出模拟观测光谱和反演结果光谱，使用 `save_json_metadata` 保存反演元数据。`TecDataset` 虽然是模块的一部分，但在 `main.py` 中仅被导入而未直接使用，保留它以备后续扩展。

## 3. inversion_solver.py

**职责**：提供非线性方程求解和参数反演所需的各种数值优化算法。这些算法用于从观测光谱中估计大气参数。

**核心类**：
- `BisectionRootFinder`：反向通信式二分法求根，要求函数在区间两端异号，通过逐步调用 `step()` 方法在外部控制迭代，适合嵌入复杂流程中。
- `BroydenSolver`：Broyden 拟牛顿法求解向量非线性方程组 F(x)=0。通过有限差分初始化 Jacobian 逆，并利用低秩更新避免每次重新计算全 Jacobian。
- `LevenbergMarquardt`：非线性最小二乘优化器，求解 min Σ r_i(x)^2。采用阻尼
