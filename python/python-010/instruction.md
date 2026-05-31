# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：宇宙大尺度结构 N 体模拟（python‑010）

该项目实现了一个简单的宇宙学 N 体模拟流水线，用于研究 ΛCDM 平坦宇宙中的暗物质结构形成。系统由多个独立 Python 模块组成，每个模块负责一组明确定义的物理计算或数值算法。主程序 `main.py` 保留，你的任务是**根据本描述以及 `main.py` 中的调用关系，实现所有缺失的 `.py` 文件**（即除 `main.py` 外的全部源码文件）。

缺失模块列表：
`cosmology.py`、`density_field.py`、`halo_finder.py`、`initial_conditions.py`、`linalg_utils.py`、`nbody_integrator.py`、`pm_solver.py`、`power_spectrum.py`、`statistics.py`、`utils.py`。

描述将说明每个文件的职责、核心接口、主要算法及其与其它模块的协作方式。你需要保证实现的函数签名和行为与 `main.py` 中的调用完全兼容。

---

## 1. 模块总览与边界

整个模拟流程由 `main.py` 驱动，大致包含以下步骤：
1. 设置 ΛCDM 宇宙学参数（`cosmology.Cosmology`）。
2. 生成初始条件（Zeldovich 近似），使用物质功率谱与 transfer function（`initial_conditions`）。
3. 使用粒子网格（PM）方法求解引力并积分粒子运动（`pm_solver.PMSolver`、`nbody_integrator.NBodyIntegrator`）。
4. 演化过程中进行密度场对流测试（`density_field.AdvectionSolver`）。
5. 估计功率谱、相关函数、统计量（`power_spectrum`），识别暗物质晕（FOF、球形过密度）（`halo_finder`）。
6. 执行统计检验与数值工具验证（`statistics`、`linalg_utils`）。
7. 输出日志与辅助信息（`utils` 提供文件/字符串工具）。

所有模块均使用 NumPy 进行数组运算，物理常数与宇宙学参数集中在 `Cosmology` 中传递。

---

## 2. 各文件详细职责与关键接口

### `cosmology.py` — 宇宙学背景演化

**职责**：基于平坦 ΛCDM 模型，提供**背景膨胀历史**、**线性增长因子**以及相关宇宙学量。

**核心类 `Cosmology`**
- 构造器接受 Planck 2018 典型参数：`h`, `Omega_m`, `Omega_b`, `Omega_Lambda`, `Omega_r`, `T_cmb`, `sigma8`, `ns`。检查总 Ω_tot 是否接近 1，以及部分参数的合理性。
- 属性：`H0`（km/s/Mpc）、`G`（引力常数）、`c`、`rho_crit_0`。
- 关键方法（只列出功能，不给出公式细节）：
  - `H(a)`：尺度因子 a 处的 Hubble 参数 H(a)。要求 a > 0。
  - `dH_da(a)`, `dlnH_dlna(a)`：用于增长因子方程。
  - `Omega_m_a(a)`：返回随尺度因子演化的物质密度参数 Ω_m(a)。
  - `scale_factor_evolution_rhs(t, y)`：单变量 ODE 右侧函数（dy/dt = a H(a)），用于数值积分尺度因子历史。
  - `linear_growth_rhs(a, y)`：将线性增长因子二阶 ODE 化为一阶系统的右侧函数（y = [D, dD/da]）。
  - `rk12_integrate(rhs, t_span, y0, n_steps)`：显式 Runge‑Kutta (1,2) 积分器（Euler + Heun），输出时间序列、解序列、误差估计。
  - `compute_scale_factor_history(...)`：积分宇宙时间并返回 a(t) 历史。
  - `compute_linear_growth_factor(a_min, a_max, n_steps)`：积分 D(a)，用早期近似（D≈a）作为初值，最后归一化到 D(a=1)=1。
  - `bisect_root_finder(func, a, b, tol, max_iter)`：二分法求根，用于求解特定红移时刻。
  - `age_of_universe(a_target)`：用复合 Simpson 法则计算达到 a_target 的宇宙年龄（积分变量换为 ln a）。
  - `delta_c(z)`：球对称坍缩模型下的线性临界过密度（近似含 Ω_m 对数修正）。
  - `comoving_distance(z, n_int)`：共动距离的 Simpson 积分。

---

### `density_field.py` — 密度场对流演化（Lax‑Wendroff）

**职责**：实现一维和三维对流方程的 Lax‑Wendroff 有限差分格式，用于检验密度场的守恒性。

**核心类 `AdvectionSolver`**
- 构造器：`nx`（网格数）、`dx`（格距）、`c`（对流速度）。
- 关键方法：
  - `lax_wendroff_step_1d(u, dt)`：单步一维 Lax‑Wendroff 更新，周期性边界，内部检查 CFL 条件。
  - `evolve_1d(u0, t_final, n_steps)`：返回时间序列和场历史。
  - `lax_wendroff_step_3d_x(u, dt)`、`_y`、`_z`：分别在 x/y/z 方向执行一维 Lax‑Wendroff。
  - `strang_split_3d(u, dt)`：Strang 分裂的三维对流步。
  - `evolve_3d_density_field(rho0, t_final, n_steps)`：三维密度场演化，返回时间序列与历史场。
- 独立函数 `test_mass_conservation()`：在周期域中用高斯包测试一维质量守恒，返回相对误差。

---

### `halo_finder.py` — 暗物质晕识别与统计

**职责**：FOF 分组、球形过密度质量计算、密度场水平集分析、球面方向采样及角距统计、质量函数估计。

**核心类 `HaloFinder`**
- 构造器：`L`（盒子边长），可选 `linking_length`（默认使用经验公式 `0.2 * (L³/N_part)^{1/3}`）。
- 关键方法：
  - `fof_groups(pos, mass)`：采用网格分箱加速的 FOF 算法，返回每个晕的粒子索引列表和晕质量数组。
  - `spherical_overdensity_mass(pos, mass, center, rho_crit, Delta=200)`：累计排序质量，找出满足平均密度 ≥ Δ·ρ_crit 的最大半径和对应质量。
- **独立函数**：
  - `level_set_volume_analysis(delta_grid, L, n_levels)`：对密度场取等值面，计算不同阈值以上的体积分数。
  - `sample_sphere_positive_distance(n_samples, rng)`：在单位球面上生成随机方向（取各分量绝对值）。
  - `angular_distance_histogram(directions, n_bins)`：计算随机方向对之间的角距离分布直方图，并归一化为概率密度。
  - `halo_mass_function_from_groups(halo_mass, volume, n_bins)`：从晕质量列表估计质量函数 dn/dlnM 及 Poisson 误差。

---

### `initial_conditions.py` — 初始条件生成

**职责**：利用 Eisenstein & Hu transfer function、功率谱归一化以及 Zeldovich 近似生成暗物质粒子的初始位移、速度和密度场。同时提供拉丁超立方采样和 Gauss‑Hermite 求积。

**核心类**
- `TransferFunction`：封装零重子物质 transfer function T(k)，构造器接收 Cosmology 对象。
- `PowerSpectrum`：构造器接收 Cosmology 和可选的 TransferFunction；内部通过 `σ8` 约束自动归一化振幅；`__call__(k)` 返回 P(k)。

**主要函数**
- `
