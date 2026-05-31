# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 托卡马克磁约束聚变综合模拟系统 – 项目描述

## 1. 项目概述
本项目实现一个托卡马克等离子体的多物理场数值模拟工具，涵盖 Grad‑Shafranov 平衡求解、引导中心漂移、D‑T 聚变动力学、能量输运延迟微分模型、碰撞统计与输运系数、磁面几何分析、有限元刚度矩阵与矩阵市场格式转换、湍流谱分析以及 MHD 稳定性分析等核心模块。所有模块以 Python 编写，物理参数统一由 `parameters.py` 提供，主程序 `main.py` 作为入口依次调用各模块并输出结果。

## 2. 文件清单及功能总览
| 文件名 | 主要职责 |
|--------|----------|
| `parameters.py` | 定义全局物理常量（磁导率、元电荷、质量等）与 ITER‑like 几何/等离子体参数，并提供若干参数字典生成函数。 |
| `collision_transport.py` | 库仑碰撞频率、平均自由程、超球面速度采样统计、矩形随机点距离统计及经典/新经典输运系数计算。 |
| `equilibrium_solver.py` | Grad‑Shafranov 平衡方程 Picard 迭代求解，磁通函数重构及磁场分量计算。 |
| `fusion_kinetics.py` | D‑T 聚变反应速率参数化、燃烧动力学模拟、轫致辐射、α 加热及 Lawson 判据估算。 |
| `geometry_utils.py` | 磁面内外判定（射线法）、极向面积/环向体积计算、Fekete 插值点选取、曲率分析及三角形网格生成。 |
| `matrix_algebra.py` | 上三角 Toeplitz 矩阵求解、Matrix Market/Harwell‑Boeing 格式读写、全局刚度矩阵组装与共轭梯度求解。 |
| `mhd_stability.py` | MHD 状态 Markov 转移模型、理想 δW 能量原理简化计算、Mercier 判据及临界比压估算。 |
| `particle_drift.py` | 引导中心速度演化（类比刚体 Euler 方程），RK2 积分及磁矩绝热不变性检验。 |
| `quadrature_engine.py` | Gauss‑Legendre 求积、三角形对称求积、局部刚度矩阵组装、Fekete 点插值及环形体积分工具。 |
| `spectral_analysis.py` | 等离子体湍流信号的 FFT 功率谱分析、波数谱计算、增长率拟合及阿尔芬色散关系。 |
| `transport_dde.py` | 基于 Mackey‑Glass 型延迟微分方程的能量输运模型，含 RK4 积分与历史插值。 |
| `main.py` | 统一入口，顺序执行上述所有模块的演示用例，打印结果。该文件在 benchmark 中保留。 |

## 3. 各模块详细说明

### 3.1 `parameters.py`
- **作用**：集中管理所有物理常数、几何尺寸、等离子体参数及数值计算控制参数。
- **主要内容**：定义 `MU0`, `EPS0`, `KB`, `QE`, `ME`, `MD`, `MT`, `R0`, `a_minor`, `B0`, `KAPPA`, `DELTA`, `q0`, `q_edge`, `N_E_AXIS`, `T_E_AXIS`, `Z_EFF` 等一系列常量和配置项；提供 `get_equilibrium_params()`, `get_fusion_params()`, `get_transport_params()`, `get_drift_params()` 等函数，返回特定模拟所需的参数字典。
- **被依赖**：几乎被所有其他模块导入使用。

### 3.2 `collision_transport.py`
- **作用**：计算等离子体库仑碰撞特征量及输运系数。
- **核心函数**：
  - `coulomb_logarithm(n_e, T_e_eV)`：估算 Coulomb 对数，处理低温截断和裁剪。
  - `electron_ion_collision_frequency(n_e, T_e_eV, Z_eff)`：电子‑离子碰撞频率。
  - `thermal_velocity_electron(T_e_eV)`：电子热速度。
  - `mean_free_path(n_e, T_e_eV, Z_eff)`：电子平均自由程。
  - `hypersphere_velocity_sampling(m_dim, n_samples, T_e_eV)`：在多维球面上随机采样速度方向，统计夹角分布。
  - `rectangle_collision_distance_stats(a, b, n_samples)`：矩形区域内随机点对的欧氏距离统计。
  - `compute_transport_coefficients(n_e, T_e_eV, B, Z_eff, q, R0, a)`：综合计算经典扩散系数、新经典扩散系数、电子和离子热导率等。
- **依赖**：从 `parameters` 导入物理常量。

### 3.3 `equilibrium_solver.py`
- **作用**：求解轴对称 Grad‑Shafranov 方程，重构极向磁通和磁场。
- **核心函数**：
  - `miller_boundary(theta, R0, a, kappa, delta)`：由极向角生成等离子体边界坐标。
  - `pressure_profile(psi_norm, …)`：归一化磁通下的压强剖面。
  - `f_profile(psi_norm, …)`：极向电流函数 F = R B_φ 的剖面。
  - `gs_operator(psi, R_grid, Z_grid)`：计算 Grad‑Shafranov 椭圆算子的有限差分近似。
  - `solve_grad_shafranov(max_iter, tol, relaxation, nr, nz)`：Picard 迭代求解，输出磁通矩阵、坐标网格及迭代信息。
  - `compute_magnetic_field(psi, R_grid, Z_grid)`：由磁通数值计算磁场三个分量。
- **算法要点**：使用二阶中心差分离散，结合松弛迭代和 Dirichlet 边界；安全因子剖面近似计算。
- **依赖**：`parameters.py` 中的几何与网格参数。

### 3.4 `fusion_kinetics.py`
- **作用**：模拟氘‑氚聚变反应动力学及功率计算。
- **核心函数**：
  - `dt_reactivity_bosch_hale(Ti_keV)`：Bosch‑Hale 参数化的反应率 <σv>。
  - `simplified_reactivity(Ti_keV)`：简化的反应率模型，用于快速计算。
  - `fusion_derivative(t, y, k_eff, tau_p, tau_He, S_D, S_T)`：四维（n_D, n_T, n_He, n_n）速率方程右端。
  - `simulate_fusion_burn(fusion_params, Ti_keV, n_steps)`：前向欧拉法求解燃烧过程，返回密度、聚变功率和增益因子 Q。
  - `compute_bremsstrahlung(n_e, T_e_eV, Z_eff)`：轫致辐射功率损失密度。
  - `compute_alpha_heating(n_e, Ti_keV)`：α 粒子加热功率密度。
  - `lawson_criterion(Ti_keV, eta)`：Lawson 判据的近似值。
- **依赖**：从 `parameters` 获取聚变能、质量等常量和初始条件字典。

### 3.5 `geometry_utils.py`
- **作用**：磁面几何判定、面积/体积计算、谱元插值节点（Fekete 点）选取及曲率分析。
- **核心函数**：
  - `point_in_flux_surface(R_test, Z_test, theta_poly, n_theta)`：射线法判断点是否在 LCFS 内部。
  - `compute_poloidal_area(theta_poly, n_theta)`：Green 定理计算极向截面面积。
  - `compute_toroidal_volume(theta_poly, n_theta, n_radial)`：数值积分计算环向体积并给出解析近似。
  - `fekete_points_on_flux_surface(m, n_sample)`：在 LCFS
