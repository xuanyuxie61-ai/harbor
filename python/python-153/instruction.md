# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：量子机器学习核方法研究工具包

本项目围绕量子机器学习中的核方法，提供了一系列计算与数值分析模块。项目入口为 `main.py`，它从多个实现模块中导入并演示各类功能。在后续 benchmark 中，将保留 `main.py`，删除所有其他 `.py` 文件；你需要根据本描述，重新实现这些被删除的模块，使得 `main.py` 可以正常运行。

目标是复现以下文件中的核心功能，不必逐行还原源码，但需保持模块接口和整体行为一致。

## 总体结构

项目包含 10 个功能模块和 1 个入口文件。模块之间的关系大致如下：

- `utils.py` – 基础数学工具与量子门定义，被多个模块使用。
- `randomness_engine.py` – 量子随机数引擎，使用 Park‑Miller 线性同余生成器，支持多种分布采样。
- `reaction_diffusion_kernel.py` – 基于 Gray‑Scott 反应扩散方程的特征映射，将非线性动力学模式编码为量子电路参数。
- `stroud_integrator.py` – 基于 Stroud 规则的多维数值积分，用于高维量子期望值计算。
- `quantum_monte_carlo.py` – 量子蒙特卡洛方法，包含 Feynman‑Kac 路径积分、量子行走核估计和马尔可夫链命中时间统计。
- `kernel_matrix_analysis.py` – 量子核矩阵构造与分析，包括 Vandermonde 行列式、Chebyshev 谱微分矩阵、条件数估计和核岭回归求解。
- `variational_optimizer.py` – 变分优化器，基于 Broyden 拟牛顿法，可进行 VQE 能量最小化。
- `parallel_circuit_simulator.py` – 模拟 CUDA 并行调度的量子电路模拟器，含量子门张量积构造和 PageRank 谱分析。
- `geometric_feature_map.py` – 极小曲面几何与射线法多边形判定，用于几何量子核和量子态空间分析。
- `stability_analysis.py` – 数值稳定性分析工具，涵盖 von Neumann 放大因子、CFL 条件、扩散稳定性、核矩阵条件数与 Trotter 误差估计。

`main.py` 会导入这些模块的公开接口并依次运行演示，因此你的实现必须提供与 `main.py` 调用相匹配的函数和类。

---

## 各文件详细职责

### 1. `utils.py` – 基础工具与量子门库

- **职责**：提供通用数学函数（扩展欧几里得、快速幂、安全除法、概率裁剪、向量归一化、格式化输出等）以及标准量子门定义。
- **关键组件**：
  - `extended_gcd`, `mod_inverse`, `power_mod` 等模运算工具。
  - `normalize_vector`, `clip_probability`, `print_section`, `print_subsection` 等辅助函数。
  - `QuantumGateLibrary` 类：包含静态方法返回 2×2 复矩阵，如 Pauli 门、Hadamard 门、旋转门 `Rx`, `Ry`, `Rz`（参数为角度）、`CNOT`（4×4 矩阵）等。
- **注意**：这些工具会被多个其他模块使用，保持接口一致即可。

### 2. `randomness_engine.py` – 量子随机数引擎

- **职责**：实现基于 Park‑Miller 线性同余生成器（参数 `a=16807, m=2^31-1`）的伪随机数序列，支持跳跃（skip‑ahead）和多种概率分布采样。
- **核心类**：`QuantumRandomnessEngine`
  - 构造函数接受一个正整数种子。
  - 方法包括：`uniform_01`（返回 [0,1] 浮点）、`uniform_ab`、`uniform_int`、`uniform_disk`（单位圆盘内的复数采样）、`uniform_sphere_nd`（n 维单位球面向量）、`jump_ahead`（一次性跳过 N 步）、`generate_sequence`（生成长度为 n 的序列）。
  - 内部使用 Schrage 分解防止 32 位整数溢出。
- **附加函数**：
  - `box_muller_transform`：将两个 [0,1] 均匀随机数转换为标准正态分布。
  - `quantum_random_hermitian`：生成 n×n 随机厄米矩阵。
  - `quantum_random_unitary`：生成 n×n 随机酉矩阵。

### 3. `reaction_diffusion_kernel.py` – 反应扩散驱动的量子特征映射

- **职责**：利用 Gray‑Scott 反应扩散方程组（U, V 两个浓度场）生成空间模式，并将这些模式映射为量子电路参数。
- **关键组件**：
  - `laplacian9_torus`：使用 9‑点高阶离散拉普拉斯算子（假设周期边界），输入 2D 场和网格间距，返回近似拉普拉斯场。
  - `gray_scott_step`：执行一步显式 Euler 时间推进的 Gray‑Scott 方程，包含反应项与扩散项，自动检查稳定性并裁剪浓度。
  - `gray_scott_simulation`：运行完整模拟，接受网格尺寸、步数、扩散系数和动力学参数，返回最终的 U, V 场。
  - `advection_ftcs_step`：一维对流方程 FTCS 格式的单步推进（周期边界），已知该格式无条件不稳定，主要用于教学和稳定性对比。
  - `pattern_to_quantum_parameters`：将 2D 模式场通过插值映射为形状 `(n_layers, n_qubits)` 的参数数组，范围 `[-π/2, π/2]`。
  - `ReactionDiffusionFeatureMap` 类：
    - 初始化时指定量子比特数、层数以及 Gray‑Scott 参数。
    - 方法 `generate_pattern` 生成并缓存一个反应扩散斑图。
    - 方法 `get_parameters` 接收一个数据点（与量子比特数同维），结合缓存的斑图生成并返回最终的量子旋转门参数（每层每比特一个参数，范围 `[-π, π]`）。

### 4. `stroud_integrator.py` – Stroud 多维求积

- **职责**：提供低精度多维积分规则，用于计算高斯权重 `exp(-||x||^2)` 或超立方体均匀权重下的积分，支持对量子期望值的数值积分。
- **核心函数**：
  - `en_r2_monomial_integral`：计算单项式在全空间高斯权重下的精确积分（用 Gamma 函数）。
  - `cn_leg_monomial_integral`：计算单项式在超立方体 `[-1,1]^N` 上的精确积分。
  - `stroud_en_r2_03_1`、`stroud_en_r2_05_1`、`stroud_cn_leg_03_1`：生成对应不同维度、不同阶数（3 阶或 5 阶）的 Stroud 积分节点和权重。
- **核心类**：`StroudIntegrator`
  - 构造函数接受维度 `n_dim` 和规则类型（如 `"en_r2_03"`），根据规则生成 `nodes` 和 `weights`。
  - `integrate` 方法：接受一个函数 `f: R^N -> float`，计算加权和 `∑ w_i f(x_i)`。
  - 提供 `integrate_vectorized` 版本，批量计算。
- **额外函数**：`gaussian_quadrature_kernel_expectation`：利用 `StroudIntegrator` 计算核函数在高斯分布下的期望值。

### 5. `quantum_monte_carlo.py` – 量子蒙特卡洛与路径积分

- **职责**：实现几种基于蒙特卡洛采样的量子计算相关估计方法。
- **主要函数**：
  - `feynman_kac_2d_estimator`：使用离散随机行走模拟椭圆域内 Feynman‑Kac 公式的解，返回估计值和精确解（椭圆势函数 PDE）。参数包括椭圆半轴、初始
