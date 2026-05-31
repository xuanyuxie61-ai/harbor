# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：聚合反应动力学与分子量分布多尺度耦合模拟

本项目是一个化学工程模拟框架，用于研究自由基聚合反应器的动力学行为、产物分子量分布（MWD）、反应器空间离散化以及参数不确定性传播。系统包含多个独立模块，每个模块对应一个 `.py` 文件，最后通过 `main.py` 统一调度，执行从反应动力学到数值验证的完整流程。

## 文件清单与职责

- `polymerization_kinetics.py`：自由基聚合反应动力学模型，使用矩方法描述活性链和死链的浓度演化，并通过自适应步长 RK4(5) 求解 ODE 系统。
- `reaction_diffusion_dg.py`：一维不连续伽辽金（DG）反应-扩散-对流求解器，用于模拟反应器轴向的温度/浓度分布。
- `molecular_weight_distribution.py`：分子量分布建模，包含 Flory-Schulz 分布、截断对数正态分布、多边形域矩计算以及由矩重构分布的方法。
- `monte_carlo_chain_sampler.py`：聚合物链构象空间的蒙特卡洛采样，包括椭球内均匀采样、圆盘内随机三角形面积估计（混合效率）、粗粒化链采样和临界孔径估计。
- `uncertainty_quantification.py`：基于稀疏网格 Gauss-Hermite 随机配点法的不确定性量化，可计算模型输出的统计矩和 Sobol 敏感度指数。
- `vandermonde_reconstruction.py`：使用 Vandermonde 插值重构连续分子量分布曲线，并计算分布的矩和导数。
- `nonlinear_solver.py`：二维非线性反应-扩散方程（凝胶效应）的 Newton-Krylov 求解器，包含稀疏 Jacobian 组装和正交性检验。
- `cvt_reactor_discretization.py`：基于 Centroidal Voronoi Tessellation（CVT）的反应器空间离散化，通过 Lloyd 算法在密度函数下生成最优节点。
- `quadrature_validation.py`：多维求积规则的精确性验证，支持 Gauss-Hermite 单项式积分测试和收敛阶估计。
- `main.py`：主程序入口，依次调用以上模块，执行九大模拟步骤并输出统计结果。

## 模块功能概要

### 1. 聚合反应动力学 (`polymerization_kinetics.py`)
- **核心模型**：自由基链式聚合机理（引发、增长、终止、链转移），通过矩方法建立 ODE 系统，状态向量包含单体、引发剂、链转移剂浓度以及活性链和死链的 0‑2 阶矩。
- **关键类/函数**：`PolymerizationParameters` 管理所有动力学参数并提供 Arrhenius 温度修正；`integrate_polymerization` 使用自适应 Cash-Karp RK4(5) 积分；`compute_conversion_and_pdi` 从状态矩阵计算转化率、DP_n、DP_w、PDI 和分子量；`exact_solution_batch` 给出简化模型的解析解用于验证。
- **边界**：提供动力学参数管理、ODE 右端项、初始状态、积分和结果后处理，不涉及反应器空间分布。

### 2. 一维 DG 反应-扩散求解器 (`reaction_diffusion_dg.py`)
- **核心模型**：求解一维对流-扩散-反应方程，采用结点型不连续 Galerkin 方法，使用 Jacobi 多项式、Gauss-Lobatto 节点、Vandermonde 矩阵和 LDG 通量。
- **关键类/函数**：`DG1DReactionDiffusion` 类封装网格构建、算子组装、通量计算和右端项，`solve` 方法通过有限差分/RK4 推进至最终时间。
- **边界**：仅处理一维标量方程，边界条件为 Dirichlet 零，通量采用中心/迎风混合。

### 3. 分子量分布建模 (`molecular_weight_distribution.py`)
- **核心功能**：提供 Flory-Schulz 分布概率质量函数和前四阶解析矩；截断正态分布和截断对数正态分布的 PDF 及采样；多边形域上的矩积分（Steger 公式）；由矩重构分布（Gamma、对数正态、最大熵）；以及流场展宽对矩的修正。
- **关键函数**：`flory_schulz_distribution`、`flory_schulz_moments`、`lognormal_mwd_pdf`、`polygon_moment`、`mwd_from_moments`、`local_mwd_broadening`。
- **边界**：专注于分布形状和矩计算，不涉及时间演化或反应动力学。

### 4. 链构象蒙特卡洛 (`monte_carlo_chain_sampler.py`)
- **核心功能**：椭球内均匀采样（Cholesky 分解 + 球内采样）；圆盘内随机三角形面积估计（混合效率）；粗粒化自由连接链的末端距采样；由回转半径和孔隙率估计临界孔径。
- **关键函数**：`ellipse_sample`、`disk_triangle_picking`、`mixing_efficiency_estimate`、`coarse_grained_chain_mc`、`critical_pore_size`。
- **边界**：蒙特卡洛采样，不依赖反应动力学。

### 5. 不确定性量化 (`uncertainty_quantification.py`)
- **核心算法**：Smolyak 稀疏网格构造（Gauss-Hermite 一维规则），生成多维节点与权重；通过加权求和估计模型输出的均值、方差、偏度、峰度；利用条件期望近似计算一阶 Sobol 敏感度指数。
- **关键函数**：`sparse_grid_hermite`、`propagate_uncertainty`、`sensitivity_index_sobol`。
- **边界**：输入为任意 `model_func`，输出统计量，与具体物理模型解耦。

### 6. Vandermonde 分布重构 (`vandermonde_reconstruction.py`)
- **核心算法**：构造 scaled Vandermonde 矩阵，求解插值多项式系数；支持切比雪夫节点重配置以抑制 Runge 现象；使用 Horner 法则求值；在 log 坐标下重构连续分子量分布，并计算分布导数和矩。
- **关键函数**：`vandermonde_interp_coef`、`polyval_horner`、`reconstruct_m
