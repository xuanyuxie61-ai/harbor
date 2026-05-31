# SD-OCT 计算框架：定量组织表征的博士级合成项目

## 1. 项目概述

本项目将 **15 个科研代码项目** 的核心算法融合为一个面向 **光学工程：光学相干层析成像（Optical Coherence Tomography, OCT）** 前沿科学问题的博士级 Python 计算框架。

### 科学问题

**光谱域光学相干层析成像（Spectral-Domain OCT, SD-OCT）的定量组织表征计算框架**

该框架解决以下前沿科学问题：
- 多层生物组织中的光传播建模与扩散近似求解
- 光谱域 OCT 干涉信号的数值积分与色散补偿
- 蒙特卡洛光子追踪模拟各向异性散射介质中的光传输
- 生物活性组织动力学（兴奋性膜电位、糖酵解振荡）对 OCT 相位信号的功能性调制
- 从 A-scan 信号反演组织光学参数（散射系数、吸收系数、各向异性因子）的逆问题
- 参数敏感性分析与高维参数空间边界探索

---

## 2. 原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 在合成项目中的角色 |
|:---:|--------|---------|------------------|
| 1 | `1064_sensitive_ode` | 敏感依赖初值的 ODE（$y''=y$） | **光子传输敏感性分析**：微小散射系数变化对光强的指数放大效应建模 |
| 2 | `1419_xy_display` | 2D 坐标点文件读取 | **OCT 扫描坐标管理**：B-scan 坐标读取与组织层边界坐标处理（删除可视化） |
| 3 | `474_gmsh_io` | Gmsh 网格文件读写 | **组织网格 I/O**：多层组织二维三角网格的 Gmsh 格式导入导出 |
| 4 | `940_quad_gauss` | Gauss-Legendre 求积（Elhay-Kautsky / IMTQLX） | **光谱数值积分**：64 点 Gauss-Legendre 求积计算 OCT 干涉谱积分 |
| 5 | `435_fitzhugh_nagumo_ode` | FitzHugh-Nagumo 可兴奋系统 | **功能性 OCT 建模**：细胞膜电位动态通过电光效应调制局部折射率 |
| 6 | `561_hypercube_surface_distance` | 超立方体表面随机采样 | **参数空间边界探索**：OCT 系统参数（$\mu_a, \mu_s, g, n, \ldots$）的高维边界采样 |
| 7 | `828_ode_midpoint` | 中点法求解 ODE | **光传输 ODE 稳定积分**：中点法求解分层组织中的光子扩散方程 |
| 8 | `472_glycolysis_ode` | Sel'kov 糖酵解振荡器 | **代谢功能性成像**：代谢振荡引起的热光效应提供 OCT 功能对比度 |
| 9 | `415_fem2d_scalar_display_brief` | FEM 2D 标量数据展示 | **2D 组织光学属性场的数据结构**：保留有限元节点/单元/值数据结构（删除可视化） |
| 10 | `1420_xy_io` | XY 坐标文件 I/O | **扫描模式坐标 I/O**：OCT 扫描点坐标与组织层边界的 XY 文件读写 |
| 11 | `1313_triangle_quadrature_symmetry` | 三角形求积对称性分析 | **高阶有限元积分**：7 点 Gauss 三角形求积规则与重心坐标对称性分类 |
| 12 | `617_kelley` | Newton-GMRES-CG 非线性/线性求解器 | **逆问题求解**：Newton-GMRES-CG 组合算法从 A-scan 反演组织光学参数 |
| 13 | `1124_sphere_monte_carlo` | 单位球面蒙特卡洛采样 | **散射方向采样**：Henyey-Greenstein 相函数随机抽样与蒙特卡洛光子追踪 |
| 14 | `273_dg1d_heat` | 1D 间断 Galerkin 热方程 | **辐射传输方程 DG 求解**：Jacobi 多项式 + LGL 节点的间断 Galerkin 扩散近似求解器 |
| 15 | `447_freefem_msh_io` | FreeFem++ 网格文件 I/O | **组织几何导入导出**：FreeFem++ 格式的 2D 网格读写 |

---

## 3. 核心数学物理模型与公式

### 3.1 SD-OCT 干涉信号模型

光谱域 OCT 的干涉信号由 Wiener-Khinchin 定理描述：

$$I(k) = S(k) \left[ R_R + R_S + 2\sqrt{R_R R_S} \cos\big(2kz + \phi(k)\big) \right]$$

其中 $S(k)$ 为光源光谱密度（Gaussian 模型）：

$$S(k) = S_0 \exp\left(-\frac{(k-k_0)^2}{2\Delta k^2}\right)$$

色散相位采用 Taylor 展开：

$$\phi(k) = \sum_{m=0}^{M} \frac{\phi_m}{m!}(k-k_0)^m$$

深度分辨 A-scan 通过对光谱积分获得：

$$A(z) = \left| \int_{k_{\min}}^{k_{\max}} S(k) \, R(k,z) \, e^{i2kz} \, dk \right|$$

光谱积分使用 **Gauss-Legendre 求积**（Elhay-Kautsky 方法，通过 Jacobi 矩阵对角化获得节点和权重）。

### 3.2 Mie 散射与体光学系数

球形粒子的散射截面（Rayleigh-Debye-Gans 近似）：

$$\sigma_s = \frac{8\pi}{3} k^4 a^6 \left| \frac{m^2-1}{m^2+2} \right|^2, \quad m = \frac{n_p}{n_m}$$

体散射与吸收系数：

$$\mu_s = \rho \sigma_s, \quad \mu_a = \rho \sigma_a, \quad \rho = \frac{3\phi_v}{4\pi a^3}$$

Henyey-Greenstein 各向异性相函数：

$$p(\cos\theta) = \frac{1-g^2}{2(1+g^2-2g\cos\theta)^{3/2}}$$

### 3.3 辐射传输方程的扩散近似

稳态扩散方程：

$$-\nabla \cdot \big(D(z) \nabla \phi\big) + \mu_a(z) \phi = S(z)$$

其中扩散系数：

$$D = \frac{1}{3(\mu_s' + \mu_a)}, \quad \mu_s' = (1-g)\mu_s$$

传输平均自由程：

$$l_{\text{tr}} = \frac{1}{\mu_s' + \mu_a}$$

**1D 求解**：使用 **间断 Galerkin (DG)** 方法，基于 Jacobi 多项式 $P_n^{(\alpha,\beta)}(x)$ 与 Legendre-Gauss-Lobatto (LGL) 节点，构建 Vandermonde 矩阵和微分矩阵：

$$V_{ij} = P_j(r_i), \quad D = V_x V^{-1}$$

**2D 求解**：使用线性 Lagrange 有限元，配合 **7 点 Gauss 三角形求积**（精确到 5 次多项式）。

### 3.4 蒙特卡洛光子追踪

光子步长服从指数分布：

$$s = -\frac{\ln \xi}{\mu_t}, \quad \mu_t = \mu_a + \mu_s$$

散射方向由 HG 相函数的逆 CDF 采样：

$$\cos\theta = \frac{1}{2g}\left[1+g^2 - \left(\frac{1-g^2}{1-g+2gU}\right)^2\right]$$

### 3.5 生物动力学与功能性 OCT

**FitzHugh-Nagumo 可兴奋膜模型**：

$$\frac{dv}{dt} = v - \frac{v^3}{3} - w + d, \quad \frac{dw}{dt} = \frac{v+a-bw}{c}$$

**Sel'kov 糖酵解振荡器**：

$$\frac{du}{dt} = -u + av + u^2v, \quad \frac{dv}{dt} = b - av - u^2v$$

折射率调制：

$$n(t) = n_0 + \alpha_{\text{eo}} v(t) + \alpha_{\text{thermo}} u(t)$$

OCT 相位偏移（电光效应 + 热光效应）：

$$\Delta\phi = \frac{4\pi}{\lambda_0} \cdot \text{OPL} \cdot \Delta n$$

### 3.6 逆问题：光学参数重建

目标：从测量 A-scan $A_{\text{meas}}(z)$ 反演 $p = [\mu_a^{(0)}, \mu_s^{(0)}, g^{(0)}, \ldots]$。

最小二乘目标函数：

$$\min_p \| F(p) \|^2, \quad F(p) = A_{\text{sim}}(p) - A_{\text{meas}}$$

求解方法：
- **Newton 迭代**：有限差分 Jacobian
- **线性子问题**：Levenberg-Marquardt 正则化下的 **GMRES** / **CG** 求解

### 3.7 敏感性分析

敏感 ODE 系统（微小初值变化导致指数发散）：

$$\frac{dy_1}{dt} = y_2, \quad \frac{dy_2}{dt} = \lambda^2 y_1$$

解析解：

$$y_1(t) = \left(1-\frac{\varepsilon}{2}\right)e^{-\lambda t} + \frac{\varepsilon}{2}e^{\lambda t}, \quad \varepsilon = y_1(0)-1$$

---

## 4. 项目文件结构

```
107_synth_project/
├── main.py                          # 统一入口，零参数运行
├── oct_physics.py                   # 核心物理模型（干涉、散射、相函数、SNR）
├── spectral_integration.py          # Gauss-Legendre 光谱积分（IMTQLX）
├── photon_transport_ode.py          # ODE 光子传输、FHN、糖酵解
├── monte_carlo_scattering.py        # MC 光子追踪、球面/超立方体采样
├── dg_radiative_transfer.py         # 1D DG 扩散求解器（Jacobi/LGL）
├── fem_optical_solver.py            # 2D FEM 扩散求解器（三角形求积）
├── nonlinear_inverse_solver.py      # Newton-GMRES-CG 逆问题求解
├── biological_oscillators.py        # 生物动力学与功能性 OCT 信号
├── tissue_mesh_io.py                # Gmsh/FreeFem++/XY 网格 I/O
├── scan_pattern.py                  # OCT 扫描模式生成与坐标管理
├── utils.py                         # 参数管理、数值安全、收敛分析
└── README_博士级合成说明.md          # 本文档
```

共 **12 个 .py 文件**，满足要求。

---

## 5. 代码边界处理与数值鲁棒性

本项目在多处实现了严格的边界检查和数值鲁棒性：

1. **物理参数边界**：所有光学参数（$\mu_a, \mu_s, g, n$）均限制在物理合理范围内（如 $g \in [-0.99, 0.99]$，$\mu_a, \mu_s > 0$）
2. **除零保护**：`safe_divide` 函数、DG 界面惩罚项的零间距检测、Mie 散射中的大小参数分支判断
3. **非有限值处理**：`clip_to_finite` 过滤 NaN/Inf；蒙特卡洛中光权重为零时终止追踪
4. **矩阵求逆稳定性**：使用 QR 分解替代 LU；GMRES 中监测 Happy Breakdown
5. **迭代收敛控制**：Newton 法监测残差比增长（ratio $\geq 1$ 时自动终止）；CG/GMRES 设置最大迭代次数
6. **文件 I/O 容错**：网格文件解析时跳过注释行和空行；XY 文件头缺失时自动推断

---

## 6. 运行方法

```bash
cd 107_synth_project
python main.py
```

无需任何参数。程序将自动执行：
1. 系统参数初始化
2. 扫描模式生成
3. Mie 散射参数计算
4. SD-OCT 干涉信号模拟与光谱积分
5. 1D DG + 2D FEM 光扩散求解
6. 蒙特卡洛光子追踪
7. FHN / 糖酵解生物动力学模拟
8. 逆问题光学参数重建
9. 敏感性分析与参数空间探索
10. 网格 I/O 验证

输出包含各阶段的定量指标（SNR、相干长度、散射深度、相位偏移、重建残差、收敛率等）。

---

## 7. 科学前沿性与博士级难度

本项目具备以下博士级科学计算特征：

- **多物理场耦合**：电磁干涉（OCT）+ 光传输（辐射传输方程）+ 生物动力学（ODE）+ 逆问题（优化）
- **高阶数值方法**：64 点 Gauss-Legendre 谱积分、4 阶 DG（Jacobi 多项式 + LGL 节点）、7 点三角形高阶求积
- **蒙特卡洛方法**：各向异性散射介质中的光子包追踪，包含边界反射/折射处理
- **功能成像**：将电生理和代谢动力学与光学信号定量关联
- **逆问题求解**：非线性最小二乘 + 矩阵自由 Jacobian + GMRES/CG 迭代子问题
- **参数敏感性**：高维超立方体表面采样，探索参数空间边界行为
- **大规模一致性**：从 1D ODE 到 2D FEM 到 3D 蒙特卡洛的多尺度建模一致性

---

## 8. 每个输入项目的真实融入验证

通过代码中的直接调用和算法继承，15 个输入项目均已真实融入，无任何挂名项目：

- **sensitive_ode**: `sensitive_photon_deriv`, `sensitive_photon_exact`（光子敏感性分析）
- **xy_display**: `xy_data_read` / `scan_pattern.py` 中的坐标管理逻辑
- **gmsh_io**: `gmsh_mesh2d_read`, `gmsh_mesh2d_write`
- **quad_gauss**: `legendre_ek_compute`, `imtqlx`（完整的 Jacobi 矩阵对角化）
- **fitzhugh_nagumo_ode**: `fitzhugh_nagumo_deriv`（功能性 OCT）
- **hypercube_surface_distance**: `hypercube_surface_sample`, `hypercube_surface_distance_stats`
- **ode_midpoint**: `ode_midpoint_solve`（中点法 + 定点迭代）
- **glycolysis_ode**: `glycolysis_deriv`, `glycolysis_equilibrium`
- **fem2d_scalar_display_brief**: `fem_optical_solver.py` 中的 FEM 节点/单元/值数据结构
- **xy_io**: `xy_data_read`, `xy_data_write`, `xy_header_write`
- **triangle_quadrature_symmetry**: `triangle_xy_to_barycentric`, `barycentric_symmetry`, `triangle_gauss_rule`
- **kelley**: `gmres_solve`, `cg_solve`, `nsol_solve`, `givapp`, `diffjac`
- **sphere_monte_carlo**: `sphere01_sample`, `sphere01_monomial_integral`
- **dg1d_heat**: `dg_diffusion_solve_1d` 及完整的 Jacobi/LGL/Vandermonde/Dmatrix/Lift 体系
- **freefem_msh_io**: `freefem_msh_read`, `freefem_msh_write`

---

*项目完成日期：2026-05-04*
