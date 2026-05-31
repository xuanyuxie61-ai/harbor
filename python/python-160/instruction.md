# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目概述

本项目模拟一个下吸式生物质气化反应器，采用顺序模块化方法集成多个物理模型，包括：反应器几何定义、生物质颗粒粒径分布分析、化学计量与元素平衡、Arrhenius 动力学与 Markov 状态转移、热力学平衡求解、辐射与导热传热、一维流动求解（含非均匀网格与周期三对角系统）、炭颗粒燃尽寿命分析、序列化反应器状态模拟以及自适应网格生成。

项目入口文件为 `main.py`，该文件调用所有其他模块的接口，执行一次完整的气化流程仿真并输出指标。你需要根据以下模块描述，补全除 `main.py` 以外的全部缺失 Python 文件，使仿真可以正常运行。每个模块必须提供描述中列出的类、函数与公有方法，但无需与原始实现完全一致，只需保证行为正确、接口兼容。

---

# 模块职责与接口要求

## `reactor_geometry.py` — 反应器几何与 3D 网格

**职责**：定义下吸式气化炉的圆柱形反应器几何，并提供坐标变换和简化的三维三角网格数据结构。

**关键类与接口**：

- **`CylindricalReactor`**
  - 构造参数：总高 `H`、半径 `R`、各段高度 `H_bed`（干燥/热解段）、`H_combustion`（燃烧段）、`H_reduction`（还原段）。
  - 方法：`volume()`（总体积）、`cross_section_area()`（截面积）、`zone_volume(zone)`（给定区域体积）、`zone_for_z(z)`（根据轴向坐标返回区域名）、`cylindrical_to_cartesian(r, theta, z)`、`cartesian_to_cylindrical(x, y, z)`。
  - 方法 `map_circle_transform(A, norm_type, num_points)`：用一个 2×2 矩阵变换单位圆上的点，返回变换后的点集和矩阵条件数。

- **`Mesh3D`**
  - 方法：`add_vertex(x, y, z)`、`add_face(i, j, k)`、`face_area(face_idx)`、`total_surface_area()`、`bounding_box()`、`sample_on_surface(num_samples)`。
  - 用于构建简单的三维三角面片网格，支持面积计算和按面积权重采样表面点。

**提示**：坐标变换参考圆的 L1、L2、L∞ 范数采样；`map_circle_transform` 中通过 SVD 计算条件数。

---

## `biomass_psd.py` — 粒径分布分析

**职责**：表征生物质原料颗粒的粒径分布，支持参数化分布（Rosin‑Rammler、对数正态）与经验直方图，并计算与传热传质相关的特征量。

**关键类**：

- **`BiomassPSD`**
  - 初始化接受粒径下限 `d_min` 和上限 `d_max`。
  - 核心方法：
    - `rosin_rammler_cdf(d, d_50, n)` 和 `rosin_rammler_pdf(d, d_50, n)`：计算 Rosin‑Rammler 的累积分布和概率密度，`d` 可以是标量或数组，需处理边界和异常值。
    - `lognormal_pdf(d, mu, sigma)`：对数正态概率密度函数。
    - `sauter_mean_diameter(d_50, n, distribution, num_points)`：用数值积分（梯形法）计算 Sauter 平均直径 \(d_{32} = \sum d_i^3 / \sum d_i^2\)。支持 `'rosin-rammler'` 和 `'lognormal'` 两种分布，使用相应 PDF 进行积分。
    - `specific_surface_area(d_32, particle_density)`：根据球形假设计算比表面积 \(6/(\rho_p \, d_{32})\)。
    - `build_histogram(samples)`：利用 `stats_utils.setup_discrete_histogram` 根据样品构建经验直方图。
    - `mean_diameter_from_histogram()`：由经验直方图通过积分计算平均粒径。
    - `biot_number(d_32, h_conv, k_char)`：毕渥数。
    - `thiele_modulus(d_32, rate_const, D_eff)`：梯勒模数。
    - `effectiveness_factor(phi)`：球形催化剂的效率因子，需处理 φ 很小时的极限。

---

## `stoichiometry.py` — 化学计量矩阵与元素平衡

**职责**：管理气化反应的元素组成矩阵，提供高斯‑若尔当消元、秩、零空间基，以及整数系数的约简与分解。

**关键类与静态方法**：

- **`StoichiometricMatrix`**
  - 构造函数接受生物质元素式 `biomass_formula`（C, H, O, N, S 原子比，默认为 `(1.0, 1.4, 0.6, 0.01, 0.005)`）。内部构建一个 5×11 的元素‑物种矩阵 `self.A`，物种顺序固定为：`biomass, H2O, O2, CO, CO2, H2, CH4, N2, H2S, tar, char`。
  - 方法：`row_swap(i, j)`、`row_scale(s, i)`、`row_axpy(s1, i1, s2, i2)`（行操作使用 1‑基索引）。
  - `gauss_jordan_elimination(rhs=None)`：返回行简化阶梯形矩阵，若提供 `rhs` 则同时求解线性系统。
  - `rank()`：通过 SVD 计算矩阵秩。
  - `nullspace_basis()`：返回方程 \(A \nu = 0\) 的零空间基。
  - `validate_balance(stoich_vector)`：验证给定化学计量向量是否满足元素守恒。

- **`StoichiometricReducer`**（提供静态方法）
  - `gcd_two(a, b)`、`gcd_list(values)`：计算整数列表的最大公约数。
  - `reduce_coefficients(coeffs)`：将系数除以最大公约数以获得最小整数比。
  - `fermat_reduce(n)`：用费马分解法找到整数 n 的两个因子 \(a, b\) 使 \(n = a^2 - b^2\)。

---

## `kinetics_model.py` — 化学动力学与 Markov 状态模型

**职责**：实现气化的全局反应动力学（Arrhenius 速率），Markov 链描述粒子在反应区间的状态转移，以及二项分布辅助计算。

**关键类**：

- **`ArrheniusRate`**
  - 构造参数：指前因子 `A`，活化能 `Ea`。
  - 方法：`rate(T)` 返回速率常数 \(k = A \exp(-E_a/(R T))\)；`derivative_dk_dT(T)` 返回温度导数。

- **`GasificationKinetics`**
  - 初始化时定义七个反应（R1 到 R7）的 `ArrheniusRate` 对象、反应焓 `dH`，以及一个 7 行（物种）×7 列（反应）的化学计量系数矩阵 `nu`。物种顺序：`['C','O2','CO2','CO','H2O','H2','CH4']`。
  - 方法：`reaction_rates(T, conc)` 根据幂律动力学计算每个反应的速率，其中水煤气变换反应 R6 需使用可逆速率，调用 `wgs_equilibrium(T)`。
  - `wgs_equilibrium(T)`：通过范特霍夫方程计算 WGS 平衡常数（\(\Delta H = -41.2\,\text{kJ/mol}\), \(\Delta S = -42.3\,\text{J/(mol·K)}\)）。
  - `species_production_rates(T, conc)`：利用 `nu` 矩阵计算各物种净生成率。
  - `heat_of_reaction(T, conc
