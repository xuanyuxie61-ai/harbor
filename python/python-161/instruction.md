# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：钙钛矿太阳能电池多物理场耦合模拟系统

## 目标
根据提供的 `main.py` 文件（仿真入口），恢复所有缺失的辅助模块。`main.py` 依赖一组模块实现从光谱采样、光吸收电荷产生、材料插值、漂移-扩散输运、缺陷统计、复合模型、机械应力到不确定性量化以及离子迁移致迟滞的完整多物理场链路。你需要实现这些模块，保证整体流程能够运行并输出效率评估。

## 文件清单

| 文件名 | 核心功能 |
|--------|----------|
| `spectrum_sampler.py` | AM1.5G 光谱的离散 CDF 构建与反变换采样，生成光子波长和入射角样本 |
| `absorption_integrator.py` | 三维楔形体上的高斯求积规则，计算光吸收与载流子产生率的体积分 |
| `material_interpolator.py` | 基于二维分段线性插值的钙钛矿材料参数（带隙、迁移率、吸收系数）随温度和卤素配比的查询 |
| `sparse_matrix_io.py` | Matrix Market 格式的稀疏矩阵读写，以及漂移-扩散方程离散后的稀疏 Jacobian 构建 |
| `drift_diffusion_solver.py` | 时间相关的漂移-扩散-泊松方程瞬态求解（固定点中点法），并用 humps ODE 验证求解器精度 |
| `mesh_triangulation.py` | 多晶薄膜的二维三角网格生成、PLY 文件 I/O，以及面积/重心计算 |
| `defect_monte_carlo.py` | 缺陷的随机位置采样（三角形内均匀分布）、对数正态缺陷密度、载流子寿命计算，以及基于寿命统计的 PDF/CDF 模型 |
| `recombination_models.py` | 辐射复合、Auger 复合、带尾态复合（使用 Gauss‑Laguerre 积分）的计算 |
| `mechanical_stress.py` | 薄膜热应力、屈曲临界应力、后屈曲挠度，以及换热应力引起的效率损失估计 |
| `uncertainty_pce.py` | 基于 Hermite 多项式的混沌展开（PCE）对光电效率进行不确定性量化与敏感性分析 |
| `coupled_ion_migration.py` | 离子‑载流子耦合 ODE 系统（包含 Euler 求解器），模拟 I‑V 扫描迟滞以及类似捕食者‑猎物模型的振荡行为 |
| `model_reduction.py` | SVD 分解、低秩近似、POD 基提取及漂移-扩散系统的 Galerkin 投影降阶 |

## 模块详细说明

### 1. `spectrum_sampler.py`
通过构建 AM1.5G 太阳光谱的二维离散直方图（波长 λ 和入射角 θ），生成归一化的二维概率密度函数 (PDF) 和累积分布函数 (CDF)。利用 CDF 反变换将均匀随机数映射为 (λ, θ) 样本点。同时提供波长到光子能量的转换。

- **关键函数/接口**
  - `build_am15_spectrum(...)` → 返回 (λ 格点, θ 格点, 归一化 PDF, 二维 CDF)
  - `discrete_cdf_to_xy(...)` → 将 `[0,1]` 随机数映射为归一化坐标样本
  - `sample_photons(n_photons)` → 从光谱采样指定数量的光子波长和角度
  - `photon_energy_ev(lambda_nm)` → 波长(nm)→光子能量(eV)
- **依赖**：numpy，物理常数（Planck 常数、光速、元电荷）

### 2. `absorption_integrator.py`
在三维楔形体区域（0≤x,0≤y,x+y≤L，-d/2≤z≤d/2）上构造张量积高斯求积规则：三角形平面使用 Dunavant 规则，z 方向使用 Gauss‑Legendre。利用 Beer‑Lambert 定律结合光谱参数计算载流子产生率，并通过楔形体上的求积计算总产生率。还包含单项式精确积分（用于验证求积规则的精确性）。

- **关键函数/接口**
  - `wedge01_volume(...)` → 楔形体体积
  - `wedge01_integral(exponents, ...)` → 单项式精确积分
  - `generate_wedge_gauss_rule(order_xy, order_z, ...)` → 返回求积节点和权重
  - `compute_carrier_generation_rate(absorption_coeff, irradiance_fn, photon_energy_ev_fn, ...)` → 总产生率、节点、产生密度、权重
  - `test_exactness(...)` → 通过单项式测试求积精确性
- **依赖**：numpy, typing，以及三角形 Dunavant 规则的内部表

### 3. `material_interpolator.py`
在 (温度 T, 卤素配比 x) 二维矩形网格上存储带隙 `Eg`、电子/空穴迁移率 `μ_n, μ_p` 和吸收系数 `α`。通过构建 Varshni 模型、幂律温度关系及 Tauc 定律生成表格数据。提供二维分段线性插值（三角剖分）进行任意点的参数查询。

- **关键类/函数**
  - `pwl_interp_2d_scalar(...)` / `pwl_interp_2d_vector(...)` → 二维分段线性插值
  - `class PerovskiteMaterial`：`get_params(T, x)` → 返回包含 bandgap、迁移率、吸收系数的字典
  - 内部表格构建函数：`_build_eg_table`, `_build_mu_n_table`, `_build_mu_p_table`, `_build_alpha_table`
- **依赖**：numpy, 物理常数

### 4. `sparse_matrix_io.py`
提供 COO 格式的稀疏矩阵容器 `SparseMatrix`，能够写入/读取 Matrix Market (.mtx) 文件。同时包含 `build_drift_diffusion_jacobian` 函数，根据 Scharfetter‑Gummel 离散格式构建一维漂移‑扩散‑泊松系统的三耦合块 Jacobian（电子连续性、空穴连续性、Poisson 方程）。

- **关键类/接口**
  - `class SparseMatrix`：`add(i,j,val)`, `to_dense()`, `nnz()`
  - `build_drift_diffusion_jacobian(N, dx, mu_n, mu_p, D_n, D_p, E, n, p, kT_q)` → 返回稀疏 Jacobian (3N×3N)
  - `write_matrix_market(A, filename)` / `read_matrix_market(filename)`
- **辅助**：Bernoulli 函数稳定计算
- **依赖**：numpy

### 5. `drift_diffusion_solver.py`
实现固定点中点时间推进算法 `midpoint_fixed_time_stepper` 求解瞬态漂移‑扩散‑泊松方程。一维器件中同时演化电子浓度、空穴浓度和电势，空间离散采用 Scharfetter‑Gummel 格式，边界条件为 Dirichlet。通过 humps ODE 的精确解进行数值精度验证。

- **关键函数/接口**
  - `midpoint_fixed_time_stepper(f, y0, dt, theta, ...)` → 单步中点时间推进
  - `solve_transient_drift_diffusion_1d(N, L, T, mu_n, mu_p, eps_r, ND, NA, G, R_fn, tspan, n_steps, ...)` → 时间序列的解 (n, p, φ 历史)
  - `humps_deriv(t, y)` / `humps_exact(x)` → humps 测试函数及精确解
  - `verify_solver()` → 返回 L2 误差
- **依赖**：numpy, 本模块的中点法

### 6. `mesh_triangulation.py`
生成代表钙钛矿多晶薄膜截面的随机扰动矩形网格，对每个晶粒进行对角线三角剖分。提供 `TriMesh` 类管理顶点
