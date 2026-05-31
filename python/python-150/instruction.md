# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目概述

本项目是一个面向分子性质预测的图神经网络（GNN）系统，融合了谱图卷积、消息传递、物理约束损失、不确定性量化、几何松弛、静电求解、数值积分等组件。整个项目由多个 Python 模块组成，`main.py` 作为统一入口串联所有功能，该文件将保留，其余所有 `.py` 文件需根据本描述重新实现。

# 核心文件与职责

## 1. `molecular_graph.py` – 分子图构建与谱分析

定义类 `MolecularGraph`，负责将分子表示为图。  
- 初始化接收原子坐标、键列表（包含键级）以及原子特征矩阵，内部构建稀疏邻接矩阵（COO 格式）、度矩阵、拉普拉斯矩阵、归一化拉普拉斯矩阵。  
- 使用幂迭代估计拉普拉斯矩阵最大特征值 λ_max，用于缩放。  
- 归一化拉普拉斯操作 `apply_normalized_laplacian(x)` 实现缩放后的矩阵乘法，供 Chebyshev 卷积使用。  
- 基于节点-边关联矩阵和转移矩阵，通过幂迭代计算原子重要性向量（类似 PageRank）。  
- 提供 `build_demo_molecules()` 函数，返回若干预定义分子（如 H₂O、CH₄、苯环）的 `MolecularGraph` 对象。

**依赖**：仅依赖 NumPy。

## 2. `chebyshev_conv.py` – Chebyshev 谱图卷积与插值

提供 `ChebyshevGraphConv` 类，实现基于 Chebyshev 多项式展开的谱图卷积层。  
- 初始化参数：输入 / 输出特征维度、Chebyshev 阶数 K。  
- 前向传播接收节点特征矩阵 `x` 和一个可调用对象 `laplacian_mul`（用于施加归一化拉普拉斯），按递推关系计算 K 个多项式项，最后通过可学习的参数矩阵线性组合出输出。层内包含权重 `theta` 和偏置 `bias`。  

同时提供函数 `chebyshev_coefficients_1d` 和 `chebyshev_value_1d`，用于对一维数据进行 Chebyshev 插值：通过构造 Vandermonde 系统求解系数，然后使用 Clenshaw 递推在任意点求值。

**依赖**：NumPy。

## 3. `dataset.py` – 合成分子数据集

定义 `SyntheticMoleculeDataset` 类，生成用于训练和评估的小分子数据集。  
- 依据预定义的分子模板（坐标、键、原子特征）加入随机扰动，为每个分子创建 `MolecularGraph` 实例。  
- 计算三个目标性质：**原子化能**（结合原子自身能量、Morse 键势、谐波角势和静电作用）、**HOMO‑LUMO 能隙**（简化模型）以及**偶极矩**（基于电荷与位置）。  
- 提供 `__getitem__` 返回分子图、目标字典及原子序数向量。  
- 提供 `train_test_split` 方法进行数据集划分。

**依赖**：`molecular_graph` 中的 `MolecularGraph`；NumPy。

## 4. `dynamics_integrator.py` – 隐式结构松弛器

实现隐式后向 Euler 单步积分器 `backward_euler_step`，用于求解形如 `dy/dt = f(y)` 的 ODE。  
- 采用 Newton‑Raphson 迭代求解残差方程，需要提供右端项 `f` 和雅可比函数 `df`。  
- 在此基础上提供 `damped_gradient_flow` 函数，将分子坐标展平，以负梯度为右端项，重复调用后向 Euler 步进行能量极小化。  
- 提供 Lennard‑Jones 势能、精确梯度以及有限差分 Hessian 函数，用于演示几何松弛。

**依赖**：NumPy。

## 5. `electrostatic_solver.py` – 分子静电势求解器

定义 `ElectrostaticSolver` 类，在三维网格上求解 Poisson 方程以获得静电势。  
- 初始化参数：模拟盒子尺寸、网格分辨率、介电常数等。  
- `deposit_charge`：将原子点电荷通过三线性权重（Cloud‑in‑Cell）沉积到网格上，得到电荷密度。  
- `solve_poisson`：使用有限差分（7 点模板）和 Gauss‑Seidel 迭代（带 SOR 加速）求解离散 Poisson 方程。  
- `compute_electric_field`：从静电势通过中心差分计算电场分量，边界采用单侧差分。  
- `interpolate_field_to_atoms`：将网格电场三线性插值回原子位置。  
- `compute_electrostatic_energy`：基于网格势和密度估算静电能。  

另外提供函数 `maxwell_boltzmann_velocity` 根据温度和原子质量采样速度。

**依赖**：NumPy。

## 6. `feature_engineering.py` – 分子特征工程与编码

集合多种分子描述符：  
- `threshold_binarize`、`double_threshold_encode`、`molecular_fingerprint`：对连续特征进行阈值二值化或多级编码，形成类指纹表示。  
- `coulomb_matrix`：根据原子坐标和原子序数构建 Coulomb 矩阵（对角与非对角元素按静电规则填充），然后按行范数排序并截断/填充至固定尺寸。  
- `radial_distribution_histogram`：计算分子内原子对的径向分布函数直方图，用球壳体积进行权重归一化。  
- `compute_atom_features`：由原子序数构造基础理化特征（如原子序数、电负性、范德华半径、电离能等近似值）。  
- `encode_molecular_features`：组合 Coulomb 矩阵、RDF 直方图、多层阈值指纹以及多项式描述符，最终输出归一化的综合特征向量。

**依赖**：`polynomial_basis` 中的 `compute_polynomial_descriptors`；NumPy。

## 7. `gnn_model.py` – 图神经网络模型

定义消息传递神经网络组件：  
- `MLP`：两层全连接网络，用于消息函数和更新函数。  
- `MPNNLayer`：单层消息传递，对每条边通过 MLP 计算消息并聚合到节点，再经过更新 MLP 得到新节点特征。  
- `MolecularMPNN`：整体模型，包含多个 MPNN 层，前置 Chebyshev 谱卷积作为初始滤波。输入为图、原子序数；输出原子能量、总能量、原子电荷以及不确定性参数（γ, ν, α, β）。  
  - 边特征由键级、距离倒数、电荷乘积项等组成。  
  - 全局读出通过节点特征求和池化后经过 MLP 获得修正项。  
  - 不确定性估计由 `EvidentialRegressor` 头部产生。

**依赖**：`chebyshev_conv`、`feature_engineering`（原子特征构造）、`uncertainty_model`；NumPy。

## 8. `graph_utils.py` – 图分析与高维几何工具

提供一系列与图结构和距离统计相关的函数：  
- `greedy_graph_partition`：根据节点权重和邻接矩阵进行贪心二部图划分。  
- `fermat_factor`、`graph_hash_fingerprint`：利用 Fermat 因数分解生成图的拓扑指纹。  
- `diophantine_nonnegative_solutions`、`parity_violation_check`：枚举线性 Diophantine 方程的非负解并检查奇偶约束。  
- `hypercube_distance_stats`、`descriptor_space_uniformity`：在描述符空间中随机采样点对，计算距离均值/方差以及均匀性度量。  
- `spherical_basis_angles`、`angular_descriptor`：生成二维圆上的均匀角度基，并计算分子中指定原子周围的各向异性角度投影描述符。

**依赖**：NumPy。

## 9. `numerical_quadrature.py` – 数值积分模块

提供二维三角形和一维线段上的数值求积规则：  
- 三角形对称规则 `triangle_unit
