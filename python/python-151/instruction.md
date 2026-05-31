# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 变分量子本征求解器 (VQE) 项目描述

## 项目概述
本项目是一个模块化的变分量子本征求解器（Variational Quantum Eigensolver, VQE），用于估计分子体系的基态能量。它将量子化学的哈密顿量构建、参数化量子电路（ansatz）的表示与操作、经典优化策略以及测量统计后处理整合为一条完整的计算流水线。项目借鉴并融合了多个数值算法（如带状矩阵求解、高斯求积规则、有限元插值、拒绝采样、陀螺仪动力学等），以支持从简单的状态向量模拟到带噪声的有限采样仿真的全流程。

## 文件与模块说明

### `main.py`
程序入口与集成演示。负责顺序调用各个模块的核心功能，运行示例计算并输出验证信息。该文件本身不含具体算法实现，只通过导入和组合其它模块展示 VQE 流水线。后续将予以保留，其他模块文件会被删除。

### `vqe_core.py`
VQE 求解器的中枢。包含两个主要类：
- `VQESolver`：整合哈密顿量、ansatz、测量采样器和优化器，封装了精确能量计算、带噪声能量估计、梯度计算（有限差分与参数位移规则）以及完整的 `run_vqe` 优化流程。
- `VQEConvergenceAnalysis`：收敛性诊断工具，用于分析能量景观、估计谱隙和 VQE 误差上界。

该模块依赖其余所有模块，定义了 VQE 的顶层接口。

### `ansatz_tree.py`
参数化量子电路的完全二叉树表示及圆弧参数初始化。主要内容：
- `TreeNode` 数据类：表示树中的一个量子门节点，包含门类型、作用量子比特、参数、父子指针等。
- `AnsatzTree` 类：基于完全二叉树的 ansatz，支持构建硬件高效电路（HEA），以及自适应添加算子层（ADAPT‑VQE 风格）。提供参数收集/设置、中序遍历、状态向量演化（支持 RX/RY/RZ/CNOT/ISWAP 门的矩阵作用）。
- 辅助函数：`circle_arc_grid_params` 在圆弧上均匀采样参数点；`initialize_parameters_on_bloch_circle` 将参数初始化为 Bloch 赤道上的值；`is_tree_adjacency` 判断邻接矩阵是否为树结构。

### `optimizer_geodesic.py`
测地线梯度流优化器与样条步长控制。主要组成部分：
- `GyroscopeDynamics`：模拟阻尼陀螺仪的欧拉角动力学（ODE），用于描述参数空间的旋转流。
- `HermiteCubicSpline`：基于节点与节点导数的分段三次 Hermite 插值，提供评估、导数和积分功能。
- `GeodesicVQEOptimizer`：VQE 优化器，结合动量下降、Hermite 样条局部曲面近似以及陀螺仪大步探索来最小化能量函数。

### `hamiltonian_builder.py`
分子哈密顿量构建与数值积分工具。主要组成部分：
- 一系列用于高斯‑拉盖尔求积规则的函数（`class_matrix_laguerre` 构造 Jacobi 矩阵，`imtqlx` 执行隐式 QL 对角化，`gauss_laguerre_rule` 生成节点和权重）。
- 随机矩阵生成：`r8mat_orth_uniform` 生成随机正交矩阵，`r8symm_gen` 生成指定特征值分布的对称矩阵。
- `MolecularHamiltonian` 类：构建第二量子化的分子电子哈密顿量，提供单电子和双电子积分的简化模型，以及通过 Jordan‑Wigner 变换生成 Pauli 字符串字典。还包含计算精确基态能量（全组态相互作用）的方法。
- `compute_radial_integral_gauss_laguerre`：使用 Laguerre 求积计算径向积分。

### `pauli_operator.py`
Pauli 算符代数与稀疏线性代数。核心内容：
- `PAULI_MATRICES` 字典与 `pauli_commutator` 函数：定义单量子比特 Pauli 矩阵的基本运算。
- `PauliString` 类：多量子比特 Pauli 字符串，支持矩阵表示、乘积、对易子、权重和支持计算。
- `rref_compute` 和 `rref_solve`：实现行最简形（RREF）计算与利用 RREF 求解线性系统。
- `extract_independent_paulis`：利用 RREF 从 Pauli 字符串集合中提取线性无关子集。
- `ShermanMorrisonSolver` 类：基于 Sherman‑Morrison 公式的低秩矩阵逆更新与求解，支持矩阵‑向量乘法和直接求解。
- `build_pauli_hamiltonian`：从系数字典构造哈密顿量矩阵。

### `banded_solver.py`
带状对称正定矩阵运算与 PDE 离散化。主要内容：
- `PDEParameters` 类：管理 PDE 物理参数的默认值。
- 一系列 PDE 辅助函数：`pde_coefficients` 定义耦合抛物型 PDE 的系数，`pde_boundary_conditions` 提供混合边界条件，`pde_initial_condition` 给出初值。
- `BandedMatrix` 类：实现对称正定带状矩阵（R8PBL 格式），提供矩阵‑向量乘法、稠密转换、带状 Cholesky 分解、求解线性系统，以及 DIF2 矩阵的解析特征值与特征向量。
- `finite_difference_discretize`：将一维 PDE 离散化为带状线性系统 `A u = f`。
- `solve_steady_pde`：求解稳态 PDE 作为 VQE 的经典基准验证。

### `fem_sampler.py`
有限元插值与期望值采样。主要内容：
- `bracket4`：二分查找区间定位。
- `fem1d_interpolate`：一维分段线性插值（FEM 评估）。
- 3D 四面体工具：`tetrahedron_volume` 计算体积，`basis_mn_tet4` 计算线性基础函数值。
- `project_sample_to_fem3d`：将采样数据投影到三维 FEM 网格。
- `FEMExpectationSampler` 类：利用一维 FEM 插值估计能量，从测量比特串构建概率密度，并通过梯形积分计算期望值。

### `measurement_sampler.py`
量子测量统计与拒绝采样。主要内容：
- `chebyshev2_sample`：通过拒绝采样从 Chebyshev 第二型分布中抽取样本。
- `cvt_density_sample`：从特定密度中采样（与 CVT 相关的分布）。
- `QuantumMeasurementSampler` 类：模拟有限次投影测量，包含 Pauli 期望值的有限样本估计、基旋转（Hadamard、HS† 门）以及比特串采样功能。

### `molecular_grid.py`
分子积分网格生成与 CVT 优化。主要内容：
- `hypercube_grid`：生成多维超立方体笛卡尔积网格。
- `cvtm_1d_optimize`：一维镜像周期 CVT（Lloyd 迭代）优化节点位置。
- `MolecularIntegralGrid` 类：管理分子积分网格，包含构建 3D 网格、计算 Slater 型原子轨道、一体与双电子积分的数值近似，以及利用 CVT 优化径向网格。

## 核心数据流与交互
1. **哈密顿量构建**：`MolecularHamiltonian` 生成二次量子化哈密顿量，并转换为 Pauli 字符串集合。
2. **ansatz 表示**：`AnsatzTree` 用二叉树描述参数化电路，支持参数向量的读写和状态向量演化。
3. **能量估计**：`VQESolver` 依据 ansatz 参数演化状态向量，然后通过 `QuantumMeasurementSampler`（含噪声）或直接矩阵运算（精确）计算哈密顿量期望值。
4. **优化**：`GeodesicVQEOptimizer` 利用梯度信息（由 `energy_gradient_parameter_shift` 或 `energy_gradient_finite_diff` 提供）更新参数，其内部可能调用 `HermiteCubicSpline` 和
