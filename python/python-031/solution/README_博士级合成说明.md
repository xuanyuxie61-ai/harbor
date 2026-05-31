# 中子星crust层核pasta相计算平台

## 项目概述

本项目围绕**核物理：中子星crust层核pasta相**这一前沿博士级科学问题，融合15个种子项目的核心算法，构建了一套完整的核pasta相计算平台。该平台能够计算核物质状态方程、pasta相几何结构、库仑势、反应-扩散动力学、温度演化、振动模式以及相图稳定性分析。

## 科学背景

在中子星crust层（密度约 $10^{11}-10^{14}$ g/cm$^3$，即 $0.001-0.1$ fm$^{-3}$），核物质在强库仑相互作用与表面张力竞争下，自组织成各种非均匀结构，即所谓的**核pasta相**（Nuclear Pasta Phases）：

- **球状相（gnocchi）**：核物质球嵌入核子气体
- **柱状相（spaghetti）**：核物质柱
- **片状相（lasagna）**：核物质片
- **管状相（anti-spaghetti）**：气体柱嵌入核物质
- **泡状相（anti-gnocchi）**：气体泡嵌入核物质

这些结构对中子星的热演化、粘滞性、引力波辐射等有着重要影响。

## 核心物理公式

### 1. Skyrme能量密度泛函

$$\mathcal{H} = \frac{\hbar^2}{2m}\tau + \frac{t_0}{2}\left[\left(1+\frac{x_0}{2}\right)\rho^2 - \left(x_0+\frac{1}{2}\right)(\rho_n^2+\rho_p^2)\right] + \frac{t_3}{24}\left[\left(1+\frac{x_3}{2}\right)\rho^{\alpha+2} - \left(x_3+\frac{1}{2}\right)\rho^\alpha(\rho_n^2+\rho_p^2)\right]$$

### 2. 总能量/核子

$$\frac{E}{A} = \frac{E_{\text{bulk}}}{A} + \frac{E_{\text{surf}}}{A} + \frac{E_{\text{Coulomb}}}{A} + \frac{E_{\text{lattice}}}{A}$$

其中：
- $E_{\text{surf}}/A = \sigma \cdot S/(\rho V)$
- $E_{\text{Coulomb}}/A = \frac{3}{10}\frac{e^2}{R_{\text{WS}}}\left(\frac{\rho_p}{\rho}\right)^2 f_C(u)$
- $E_{\text{lattice}} = -0.9 \frac{(Ze)^2}{2R_{\text{WS}}}$

### 3. 泊松方程（有限元离散）

$$\nabla^2 \Phi = -4\pi e \rho_p(\mathbf{r})$$

弱形式：
$$\int (\nabla \Phi \cdot \nabla v)\, dV = 4\pi e \int \rho_p v\, dV$$

### 4. 反应-扩散方程

$$\frac{\partial \rho_n}{\partial t} = D_n \nabla^2 \rho_n + \lambda_{\beta^+} \rho_p - \lambda_{\beta^-} \rho_n$$
$$\frac{\partial \rho_p}{\partial t} = D_p \nabla^2 \rho_p - \lambda_{\beta^+} \rho_p + \lambda_{\beta^-} \rho_n$$

### 5. 中子星冷却方程

$$C_V \frac{dT}{dt} = -\varepsilon_\nu + \varepsilon_{\text{crust}}$$

其中 $C_V = \frac{\pi^2}{2} N(0) k_B^2 T$，$\varepsilon_\nu \propto T^8$（modified Urca）。

### 6. 贝塞尔函数展开（柱坐标库仑势）

$$\Phi(r) = \frac{2e\rho_p}{R} \sum_n \frac{J_0(\alpha_{0n} r/R)}{\alpha_{0n}^3 J_1(\alpha_{0n})}$$

### 7. CVT能量泛函

$$G(\mathbf{X}) = \sum_i \int_{\Omega_i} \rho(\mathbf{x}) |\mathbf{x} - \mathbf{x}_i|^2 \, d\mathbf{x}$$

### 8. Ginzburg-Landau相变动力学

$$\frac{d\psi}{dt} = -\gamma \frac{\delta F}{\delta \psi}, \quad F = \int \left[a(T)\psi^2 + b\psi^4 + c(\nabla\psi)^2\right] dV$$

## 文件结构与种子项目映射

| 文件 | 功能 | 融入的种子项目 |
|------|------|--------------|
| `nuclear_eos.py` | Skyrme核状态方程、非中心t分布参数不确定性 | 051_asa243 (非中心t分布) |
| `geometry_pasta.py` | 五种pasta相几何建模、多面体积分 | 1246_tetrahedron_felippa_rule, 1409_wedge_integrals, 1326_triangle01_integrals, 530_hexagon_stroud_rule, 956_quadrilateral_surface_display |
| `coulomb_solver.py` | 有限元泊松方程求解、库仑能计算 | 411_fem2d_project, 410_fem2d_predator_prey_fast |
| `reaction_diffusion.py` | 核子反应-扩散动力学、熵产生率 | 350_fd_predator_prey, 410_fem2d_predator_prey_fast |
| `ode_integrator.py` | 温度演化ODE、不稳定/刚性ODE积分 | 1374_unstable_ode, 1283_tough_ode |
| `bessel_modes.py` | 柱坐标本征模式、振动频率、形变能 | 080_besselj_zero |
| `cvt_sampler.py` | CVT结构优化、N维蒙特卡洛积分 | 247_cvt_2d_lumping, 1113_sphere_cvt, 1209_test_int_nd |
| `phase_diagram.py` | 相图构建、稳定性分析、转变密度 | （综合各模块） |
| `main.py` | 统一入口，零参数运行 | （综合调度） |

## 运行方法

```bash
cd Synthesis-project-python/031_synth_project
python main.py
```

程序将自动执行以下计算流程：
1. 核物质状态方程计算（不同密度下的能量、压强、对称能、不可压缩系数）
2. 五种pasta相的几何参数与能量景观
3. 解析与有限元库仑势求解
4. β衰变率、扩散系数与1D反应-扩散模拟
5. 中子星冷却模拟、不稳定ODE与刚性ODE测试
6. 贝塞尔零点与柱相振动频率
7. N维蒙特卡洛积分与2D CVT结构优化
8. 相图计算、稳定性分析与相转变密度

## 关键数值结果示例

- **饱和密度**: $\rho_0 \approx 0.16$ fm$^{-3}$，$E/A \approx -33.5$ MeV
- **对称能**: $E_{\text{sym}} \approx 13.3$ MeV
- **相变序列**（$x_p = 0.3$，$T = 0$）:
  - $\rho \approx 0.02-0.06$ fm$^{-3}$: spaghetti
  - $\rho \approx 0.06-0.10$ fm$^{-3}$: lasagna
  - $\rho \approx 0.10-0.14$ fm$^{-3}$: anti-spaghetti
- **贝塞尔零点**: $J_0$ 第一个零点 $\alpha_{1,0} = 2.404826$

## 边界处理与数值鲁棒性

- 所有密度、质子分数输入均进行合法性检查
- FEM求解失败时自动回退到解析近似
- 反应扩散中加入非负密度约束和CFL条件自适应
- ODE积分中使用绝对/相对容差控制
- 贝塞尔零点计算中使用最大迭代次数限制
- 数值积分中使用截断避免除零

## 删除的可视化内容

原始种子项目中的以下可视化代码已被删除：
- `quadrilateral_surface_display.m` 中的 `fill3`、`plot` 等图形绘制
- `fd_predator_prey.m` 中的 `figure`、`plot`、`print` 等
- `cvt_2d_lumping.m` 中的 `surf`、`plot` 等
- `fe2d_predator_prey_fast.m` 中的 `trisurf`、`colorbar` 等

## 作者与参考文献

- 科学模型参考: Watanabe et al., PRL 103, 121101 (2009); Ravenhall et al., PRL 50, 2066 (1983)
- Skyrme参数参考: Dutra et al., PRC 85, 035201 (2012)
- 冷却模型参考: Friman & Maxwell, ApJ 232, 541 (1979)
