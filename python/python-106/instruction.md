# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 项目描述：纳米光子学等离子体模拟（python-106）

本项目是一个多尺度的纳米光子学与等离子体模拟工具集，用于分析金属纳米颗粒阵列的集体光学响应、热载流子输运及波导模态。项目包含一个主驱动模块 `main.py` 和若干物理计算模块，每个模块封装一类计算任务。下文列出所有涉及的文件及其职责，并以中等粒度描述模块边界、核心数据结构和主要算法。

---

## 文件清单与职责概览

### `main.py`
- **角色**：主入口，零参数运行即可执行全流程仿真。
- **职责**：
  - 定义材料参数（Drude 模型、纳米球几何参数）。
  - 按顺序调用其他所有模块的功能：
    1. 纳米颗粒布局（CVT）；
    2. 单球米散射谱；
    3. 耦合偶极集体响应；
    4. FFT 谱分析；
    5. 小波多分辨率热点分析；
    6. 连通分量热点检测；
    7. 3D Gauss-Legendre 能量积分；
    8. 集体共振频率搜索；
    9. 热电子随机游走输运；
    10. 1D 有效波导模式求解；
    11. 域划分负载均衡；
    12. 张量网格生成与 CFL 检查。
  - 输出各阶段的关键指标（共振波长、热点数目、能量、收集效率等）。

### `mie_theory.py`
- **角色**：单个金属纳米球的 Mie 散射理论计算。
- **核心函数**：
  - `drude_permittivity(omega, omega_p, gamma, eps_inf)`：根据 Drude 模型计算复介电常数。
  - `mie_cross_sections(omega, a, eps_medium, ...)`：计算消光截面和散射截面。内部利用 Mie 系数（`a_l`, `b_l`）的级数展开，调用 `mie_coefficients`。
  - `generate_sphere_surface_grid(n_theta, n_phi, radius)`：生成球面网格点，用于离散化。
- **算法/数据结构**：基于 Riccati-Bessel 函数计算 Mie 系数；频率可以是标量或数组；大小参数 `x = k·a` 决定所需的最大多极阶数 `lmax`。
- **依赖**：`scipy.special.spherical_jn/yn` 用于贝塞尔函数。

### `dipole_coupling.py`
- **角色**：耦合偶极模型（CDM），计算纳米颗粒阵列的集体极化响应。
- **核心函数**：
  - `dyadic_green_tensor(r_vec, k)`：计算自由空间并矢格林张量 G(r)，用于描述两点间偶极耦合。
  - `build_coupling_matrix(positions, polarizabilities, omega, eps_medium)`：构建整个系统的相互作用矩阵 `A = diag(1/α) - G`（形状 `3N×3N`，复数）。
  - `incident_plane_wave(positions, E0, kvec, pol)`：构造平面波入射场向量（`3N` 维）。
  - `solve_dipole_moments(A, b)`：求解 `A p = b`，对小系统用直接求解，大系统用 `scipy.sparse.linalg.bicgstab`（带稀疏矩阵回退）。
  - `build_coupling_graph(positions, omega, eps_medium, threshold)`：根据近场耦合强度构建有向图，边表示强耦合连接。
  - `polarizability_clausius_mossotti(eps_particle, eps_medium, volume)`：Clausius-Mossotti 极化率公式。
- **关键结构**：每个颗粒的偶极矩为 3 分量，位置为 N×3，极化率为 N 元复数数组。

### `spectral_analysis.py`
- **角色**：时域信号的频谱分析，提取等离子体共振的频域特征。
- **核心函数**：
  - `power_spectral_density(time_series, dt)`：计算实值时间序列的功率谱密度。
  - `spectral_response_dipole(p_t, dt)`：从偶极矩时序（N×3 数组）计算辐射谱强度（利用 ω⁴ 加权）。
- **算法**：内部使用基于 Cooley-Tukey 的串行复 FFT（为健壮性以 `numpy.fft` 为后端）。假设输入长度为 2 的幂。

### `wavelet_field.py`
- **角色**：对二维近场分布进行 Haar 小波多分辨率分析。
- **核心函数**：
  - `haar_1d_transform` / `haar_1d_inverse`：一维正向/逆向 Haar 变换。
  - `haar_2d_transform` / `haar_2d_inverse`：二维 Haar 变换（先列后行/先行后列）。
  - `extract_multiresolution_hotspots(field, threshold_factor)`：对变换后的细节系数进行统计，识别热点出现的空间尺度（像素为单位），返回尺度列表和各级系数统计量。
- **数据结构**：输入/输出均为二维实数数组（大小最好是 2 的幂）。

### `hotspot_detector.py`
- **角色**：通过连通分量分析检测“热点”区域，并提取多边形边界，同时计算热载流子产生率。
- **核心函数**：
  - `find_connected_components_2d(field, threshold)`：使用 4-连通区域生长法标记高于阈值的区域（二维）。返回标签数组和分量数。
  - `find_connected_components_3d(field, threshold)`：三维版本，使用 6-连通。
  - `extract_hotspot_polygons_2d(field, dx, dy, threshold_factor)`：基于连通分量标记，用极角排序提取凸包边界，返回物理坐标下的顶点列表及各热点平均强度。
  - `polygon_contains_point(polygon, q)`：基于绕数算法的点包含测试。
  - `hot_carrier_generation_rate(field, omega, eps_metal, dx, dy, dz)`：计算单位体积热载流子产生率 `G_hc ∝ |E|² Im[ε]/ω`。
- **关键**：阈值相对于均值强度的倍数自动确定；输出多边形供后续可视化或密度估算。

### `volume_integrator.py`
- **角色**：三维 Gauss-Legendre 数值积分，用于计算电磁能量、吸收功率等体积分。
- **核心函数**：
  - `gauss_legendre_3d_set(a, b, nx, ny, nz)`：生成矩形盒内张量积节点和权重。
  - `integrate_over_box(f, a, b, nx, ny, nz)`：通用积分接口。
  - `electromagnetic_energy_density_integral(epsilon, mu, E_field_func, H_field_func, a, b, ...)`：计算 `U = ½∫(ε|E|²+μ|H|²) dV`。
  - `absorbed_power_integral(omega, epsilon, E_field_func, a, b, ...)`：计算 `P_abs = ½ ω ε₀ Im[ε] ∫|E|² dV`。
  - `test_exactness_monomial(a, b, max_total_degree, ...)`：用单项式测试积分精度。
- **算法**：高斯-勒让德节点通过 `numpy.polynomial.legendre.leggauss` 获取，映射到任意区间；权重相应缩放。

### `resonance_finder.py`
- **角色**：寻找单球和耦合阵列的等离子体共振频率。
- **核心函数**：
  - `bisection_method(f, a, b, tol, max_iter)`：经典二分法求根。
  - `expand_bracket(f, a0, b0, ...)`：自动扩大区间直到找到变号区间。
  - `find_single_sphere_resonance(eps_medium, omega_p, gamma, eps_inf)`：解 `Re[ε
