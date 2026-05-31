# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 超表面电磁调控与全息 — Python 项目描述

## 项目概述

本项目是一个用于**超表面电磁调控与全息（Metasurface Electromagnetic Control and Holography）**的博士级综合计算框架。它融合了多个数值算法，完成非周期超表面布局优化、相位剖面设计、远场球谐展开、散射算符压缩、波传播模拟、非线性共振建模、网格处理、点云 I/O、制造公差分析以及全息图优化等任务。

主干脚本 `main.py` 会调用其余模块中的核心函数，按固定顺序执行一套完整的物理计算流水线。后续评估中，`main.py` 将被保留，而其他所有的拓展模块（.py 文件）将被删除，需要根据本描述重新实现。

---

## 文件清单与职责

| 文件名 | 主要职责 |
|---|---|
| `cvt_placer.py` | 基于 Centroidal Voronoi Tessellation (CVT) 的生成元布局优化，支持二维平面（可加权）和单位球面 |
| `phase_spectrum.py` | 多项式求值与 Legendre/Chebyshev 谱展开，用于二维全息相位剖面的设计与重构 |
| `spherical_expansion.py` | 连带 Legendre 函数、归一化球谐函数计算，远场球谐展开与重构，Mie 散射系数估算 |
| `scattering_operator.py` | 构建近轴散射传输矩阵，通过 SVD 实现低秩近似与压缩误差评估 |
| `wave_propagation.py` | 常微分方程数值求解（RK4、RK23），梯度折射率层中的波传播，以及角谱传播法 |
| `nonlinear_resonator.py` | 非线性振子模型（分段橡胶带、Duffing、大角度摆），用于描述 meta‑atom 的非线性相位响应 |
| `mesh_handler.py` | 简易网格文件的读写、维度检测、网格质量（四面体/三角形）计算与测试网格生成 |
| `geometry_io.py` | 三维点云（XYZ 格式）的读写，随机 meta‑atom 点云生成及统计量计算 |
| `tolerance_analysis.py` | 蒙特卡洛相位误差模拟，良率估计，误差 CDF 计算，灵敏度分析 |
| `moment_integrals.py` | 方形区域上的单项式解析积分、Gauss‑Legendre 数值积分、场矩计算与验证 |
| `hologram_io.py` | 二进制/多相位全息图配置文件的读写，Gerchberg‑Saxton 相位恢复算法，配置物理约束检查 |

`main.py` 负责顺序调用以上模块，设置物理参数、完成全局计算并输出结果文件。

---

## 模块功能概述与边界

### cvt_placer.py – Voronoi 镶嵌布局优化

**核心思想**：利用 Lloyd 松弛算法，通过迭代将一组生成元移动到其 Voronoi 区域的质心，获得均匀或按强度加权分布的点集。用于设计非周期超表面布局以抑制光栅瓣。

**主要接口**：
- `lloyd_relaxation_square(n_generators, n_steps, density_func, …)` 在矩形区域内返回优化后的二维生成元坐标。若提供密度函数，则按该强度分布采样加权；否则均匀 CVT。
- `lloyd_relaxation_sphere(n_generators, n_steps, …)` 在单位球面上生成均匀分布的点集，每次迭代将质心投影回球面。
- `compute_voronoi_areas_square / sphere(…)` 通过蒙特卡洛采样估算各 Voronoi 区域的面积。

**算法要点**：
- 密度加权采样使用拒绝采样，以包络分布生成样本点。
- Voronoi 质心计算根据最近生成元分配样本，加权质心公式内部处理特殊情况（无样本时保留原位置）。
- 球面版本使用正态分布采样并归一化到单位球面，距离用欧氏距离近似。

**模块边界**：仅依赖 NumPy，提供纯算法函数。

---

### phase_spectrum.py – 多项式谱展开相位设计

**核心思想**：利用 Legendre 或 Chebyshev 多项式的张量积对二维相位函数进行谱展开，从而实现低阶模态驱动的相位调控。

**主要接口**：
- `horner_eval(coeffs, x)` Horner 法则求多项式值。
- `legendre_polynomials(n, x)` / `legendre_polynomials_array(n, x_arr)` 计算 Legendre 多项式及其导数值。
- `design_hologram_phase_2d(x_grid, y_grid, coeffs, Lx, Ly)` 根据 Legendre 谱系数生成二维相位剖面。
- `reconstruct_phase_from_spectrum(phase_samples, x_nodes, y_nodes, max_degree)` 从离散相位样本通过最小二乘拟合恢复谱系数。
- `chebyshev_nodes(n)` 返回 Chebyshev 配置点。
- `phase_gradient_2d(…)` 计算相位梯度。

**算法要点**：
- 多项式通过递推式生成，保证数值稳定。
- 谱展开设计时坐标先映射到 `[-1,1]`。
- 拟合采用正规方程加 Tikhonov 正则化。

**模块边界**：纯数学工具，不涉及电磁传播。

---

### spherical_expansion.py – 球谐展开与 Mie 散射

**核心思想**：将远场散射场投影到球谐函数基上，以实现紧凑表示，并可计算小粒子 Mie 散射系数。

**主要接口**：
- `spherical_harmonic_y(l, m, theta, phi)` 计算归一化球谐函数值（含 Condon‑Shortley 相位）。
- `expand_far_field_spherical(field_samples, theta_grid, phi_grid, l_max)` 数值积分获得球谐系数。
- `reconstruct_far_field(coeffs, theta_grid, phi_grid)` 由系数重构远场。
- `scattering_coefficients_mie(l_max, k, a, eps_r, mu_r)` 小粒子近似下的电多极子散射系数。

**算法要点**：
- 连带 Legendre 函数通过递推计算，包含归一化因子。
- 展开积分权重使用 `sinθ` 几何因子和网格差分。
- Mie 系数仅给出 Rayleigh 散射主导项及高阶衰减。

**模块边界**：与电磁传播无关，仅提供基函数和投影。

---

### scattering_operator.py – 散射算符构建与 SVD 压缩

**核心思想**：超表面可视为线性传输矩阵，通过 SVD 低秩近似可以大幅度降低计算复杂度。

**主要接口**：
- `build_scattering_operator(n_pixels, aperture_size, wavelength, phase_profile, amplitude_profile)` 构建复传输矩阵（基于 sinc 核的角谱耦合模型）。
- `svd_compress_scattering(S, rank)` 进行 SVD 分解并返回秩‑R 近似、压缩比等指标。
- `evaluate_compression_error(S, rank_list)` 评估不同截断秩的相对误差。
- `apply_scattering_operator(S, field_in)` 将算符作用于入射场。

**算法要点**：
- 传输核使用 `sinc` 函数与截止波数，体现衍射受限耦合。
- SVD 采用 `numpy.linalg.svd`，误差和压缩比通过奇异值分布计算。

**模块边界**：仅处理矩阵代数，函数本身不包含物理传播。

---

### wave_propagation.py – 波传播与 ODE 求解

**核心思想**：基于慢变包络近似，将传播方程转化为一阶复 ODE，用 Runge‑Kutta 法求解；同时提供角谱法传播。

**主要接口**：
- `rk4_integrate(f, t_span, y0, n_steps)` 经典四阶 Runge‑Kutta。
- `rk23_integrate(f, t_span, y0, n_steps)` 带局部误差估计的 RK23。
- `propagate_plane_wave_scalar(k0, z_span, n_eff_func, E0, n_steps)` 标量
