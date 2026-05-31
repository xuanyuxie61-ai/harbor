# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：GRB 余辉辐射机制综合模拟

## 概述
该项目模拟伽马射线暴（GRB）余辉的多物理过程，构建了一个零参数集成管线的离散步解题。它包含喷流流体力学、冲击波扩散、磁重联、粒子加速、不透明度插值、辐射扩散有限元求解、谱能量分布蒙特卡洛积分、光子转移矩阵、谱矩分析、逆康普顿级联、各向异性扩散张量构造及矩阵输入输出等模块。主入口脚本 `main.py` 按顺序调用各模块，输出模拟统计量，展示物理机制间的耦合。

## 模块文件及职责

### `grb_jet_hydro.py`
提供轴对称相对论喷流的流体力学计算。包含：
- `compute_jet_profiles`：在柱坐标(r,z)网格上生成速度场（基于流函数螺旋模型）、质量密度及洛伦兹因子，并返回连续性方程的残差。
- `lorentz_factor`：根据速度分量计算洛伦兹因子。
- 其他辅助函数如网格生成、流函数及其导数、连续性残差等，用于构造零散度的螺旋流场。

### `blast_wave_diffusion.py`
基于多孔介质方程（PME）的冲击波扩散模型。包含：
- `blast_wave_energy_density_profile`：将 PME 自相似解映射到物理参数（Eiso，nism，γad），给出给定时间处的能量密度径向分布。
- `porous_medium_exact`：潘佩尔自相似解及其一、二阶导数的计算。
- `porous_medium_parameters`：管理模型默认参数（如幂指数 m，常数 c 与 δ）。

### `magnetic_spiral.py`
螺旋磁场几何建模。包含：
- `spiral_array`：生成给定厚度的向外螺旋整数阵列。
- `prime_spiral_mask`：标记螺旋中的质数位置。
- `magnetic_pitch_angle_grid`：根据角速度和轴向速度计算螺距角 ψ(r) 及 Bφ/Bz 比。
- `magnetization_parameter`：根据密度、磁场强度和洛伦兹因子计算磁化参数 σ。

### `reconnection_automaton.py`
用二维元胞自动机（仿照生命游戏）模拟磁重联动态。包含：
- `life_update`：执行一步网格更新（根据邻居数决定激活/熄灭）。
- `initialize_reconnection_sites`：生成初始激活位点（随机或中心分布）。
- `evolve_reconnection`：演化指定步数并记录每步的活跃细胞数和耗散功率。

### `particle_acceleration.py`
随机龙格-库塔方法求解电子加速的福克-普朗克方程。包含：
- `accelerate_electrons`：对粒子群的洛伦兹因子进行时间推进，采用冲击加速漂移和玻姆扩散系数。
- 内部函数如 `fi_dsa`（系统加速率）、`gi_dsa`（随机噪声幅度）以及 `rk4_ti_step`（Kasdin 随机RK4算法）。

### `opacity_interpolator.py`
二维分段线性（三角剖分）插值，用于构建和查询不透明度表 κ(ρ,T)。包含：
- `build_opacity_table`：在 (log ρ, log T) 网格上生成由汤姆逊散射和韧致辐射组成的总不透明度。
- `interpolate_opacity`：对给定的 (ρ,T) 查询点进行插值。
- 底层 `pwl_interp_2d` 支持向量化查询。

### `radiation_diffusion_fem.py`
一维辐射扩散方程的有限元求解（Neumann 边界条件）。包含：
- `solve_radiation_diffusion`：组装质量矩阵和刚度矩阵，投影初值，将半离散系统用刚性 ODE 求解器积分，返回时-空解。
- `assemble_mass_matrix`、`assemble_stiffness_matrix`：生成三角形线元的 M、K 矩阵。
- `nonlinear_source`：施加非线性源项（常数、线性、平方、立方项）。
- `basic_hat`：定义帽函数基函数。

### `fem_matrix_assembly.py`
基于 serendipity 四边形单元的 Wathen 有限元质量矩阵组装，用于 2D 辐射传输。包含：
- `solve_wathen_system`：构建稠密的系统矩阵，生成右端项，用共轭梯度法求解。
- `wathen_st`：生成稀疏（行、列、值）三元组表示的矩阵。
- `cg_sparse`：共轭梯度法解线性方程组。

### `photon_transfer_matrix.py`
构建描述光子能群间转移（康普顿散射）的 Markov 转移矩阵，并求稳态分布。包含：
- `build_compton_transfer_matrix`：生成含逃逸态的增加散射、下散射、自散射概率的矩阵。
- `compute_photon_stats`：计算稳态向量、平均散射次数和 Compton‑y 参数。
- `incidence_to_transition`、`power_rank`：列随机化及幂迭代求最大特征向量。

### `spectral_moments.py`
谱矩分析，基于 Hankel 正定矩阵。包含：
- `synthetic_grb_moments`：产生折断幂律谱的解析矩 (μ0 … μN)。
- `build_hankel_from_moments`：利用矩构建 Hankel 矩阵。
- `compute_spectral_moments_from_hankel`：从 Hankel 矩阵恢复辐射通量、平均频率和谱宽。
- `hankel_spd_cholesky_lower`：构造给定对角线元素和次对角线元素的 Hankel 矩阵的下三角 Cholesky 因子。

### `sed_triangle_integrator.py`
在 (γ,θ) 三角域上蒙特卡洛积分计算同步辐射通量。包含：
- `monte_carlo_sed`：均匀采样三角形，计算同步辐射被积函数（包含近似同步函数 F(x)），返回通量和标准误差。
- `triangle_area`、`triangle_sample`：计算三角形面积并在其中均匀采样点。
- `integrand_sed`：计算给定 (γ,θ) 的辐射贡献（幂律电子分布、同步函数近似）。

### `spectrum_interpolator.py`
多方法光谱插值。包含：
- `interpolate_spectrum`：根据用户选择（线性、拉格朗日或最近邻）对 (ν, νFν) 数据插值。
- `interp_linear`：分段线性插值。
- `interp_lagrange`、`lagrange_value`：拉格朗日基函数评估与多项式插值。
- `interp_nearest`：最近邻插值。

### `discrete_cascade.py`
基于子集和动态规划的离散逆康普顿级联。包含：
- `discrete_ic_cascade`：根据离散电子洛伦兹因子，将种子光子能量乘以 γ² 的组合得出可达光子能量列表。
- `subset_sum_table`、`subset_sum_find`：DP 表构造和子集回溯。
- `cascade_compactness`：计算辐射致密参数 ℓ。

### `anisotropic_tensor.py`
各向异性扩散张量构造。包含：
- `magic_anisotropic_field`：生成随半径变化的扩散张量场，张量形式 Dij = D⊥ δij + (D∥‑D⊥) bi bj，其中 b 由磁场方向决定，调制标度参考幻方模式。
- `anisotropic_diffusion_tensor`：按给定 D⊥、D∥ 和 pitch angle 构造 3×3 扩散张量。
- `magic3`、`magicsquare`：生成 3×3 或 n×n 幻方。

### `matrix_io.py`
矩阵导出为 Matrix Market 格式。包含：
- `export_grb_matrix`：将稠密矩阵写出为数组格式和坐标格式的 .mtx 文件。
- `dense_to_mm_array`、`sparse_to_mm_coordinate`：转换内部表示为 MM 文本。
- `read_mm_coordinate`：读取 MM 坐标文件。

## 主流程 (`main.py`)
`
