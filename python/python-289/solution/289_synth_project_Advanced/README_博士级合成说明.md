# PROJECT 289 — 1D 平板位形回旋动力学湍流模拟器

> **领域**：计算等离子体物理 / 湍流输运 / gyrokinetic 模拟  
> **方法**：高阶有限差分 + 隐式时间推进 + 线性稳定性分析（小规模可复现实验）

---

## 1. 项目概述

本项目将一个完整的 **1D 平板位形 (slab-geometry) 回旋动力学 (gyrokinetic) 湍流模拟器** 实现为可直接运行的 Python 数值实验室。它面向**博士级前沿科学计算问题**：

**核心科学问题**：在磁化等离子体中，离子温度梯度 (ITG) 不稳定性驱动的湍流输运是托卡马克约束损失的主要机制。本项目实现并研究：

1. **线性稳定性**：求解 slab ITG 本征值问题，获得复数本征频率 ω = ω_r + iγ
2. **阈值搜索**：二分法定位临界 (R/L_Ti)_c
3. **带状流残余**：Rosenbluth-Hinton 理论预言的残余水平 φ_res / φ_0
4. **湍流输运统计**：热通量的 PDF、间歇性指数、重尾特征

整个模拟从物理常数、平衡位形、速度空间离散、FLR 算子、碰撞算子、时间推进器、到诊断后处理，**全部从零实现**，仅依赖 numpy 和 scipy。

---

## 2. 核心科学公式

### 2.1 回旋动力学方程（线性化）

$$\frac{\partial \delta h}{\partial t} + v_\parallel \hat{b}_0 \cdot \nabla \delta h - i \omega_{T*}\left[1 + \eta_i \left(\frac{v^2}{v_{th}^2} - \frac{3}{2}\right)\right] J_0(k_\perp \rho_s) \frac{e\phi}{T_i} F_0 + i\omega_d(v_\parallel)\delta h = 0$$

其中 $\delta h = \delta f - \frac{e\langle\phi\rangle_R}{T_i}\left[1 + \eta_i\left(\frac{v^2}{v_{th}^2} - \frac{3}{2}\right)\right]F_0$ 是非绝热部分。

### 2.2 FLR 算子

$$\Gamma_0(b) = I_0(b) e^{-b}, \quad b = \frac{k_\perp^2 \rho_s^2}{2}$$

### 2.3 回旋动力学 Poisson 方程（准中性条件）

$$(1 + \tau_e)(1 - \Gamma_0)\phi = Z_i \int \langle \delta f \rangle_R \, d^3v$$

### 2.4 Lorentz 碰撞算子

$$C[\delta f] = \frac{\nu_D(v)}{2} \frac{\partial}{\partial \xi}\left[(1-\xi^2)\frac{\partial \delta f}{\partial \xi}\right], \quad \xi = \frac{v_\parallel}{v}$$

在 Legendre 基底下本征值为 $\lambda_\ell = -\frac{\nu_D}{2}\ell(\ell+1)$。

### 2.5 Rosenbluth-Hinton 带状流残余

$$\frac{\phi_{res}}{\phi_0} = \frac{1}{1 + 1.6 q^2 / \sqrt{\varepsilon}}$$

### 2.6 高阶有限差分

4 阶紧致 (Padé) 格式：

$$\frac{1}{6}f'_{j-1} + \frac{2}{3}f'_j + \frac{1}{6}f'_{j+1} = \frac{f_{j+1} - f_{j-1}}{2h}$$

4 阶中心差分二阶导数：

$$f''_j = \frac{-\frac{1}{12}f_{j-2} + \frac{4}{3}f_{j-1} - \frac{5}{2}f_j + \frac{4}{3}f_{j+1} - \frac{1}{12}f_{j+2}}{h^2}$$

### 2.7 时间积分器

**隐式梯形法 (Crank-Nicolson)**：

$$y_{n+1} = y_n + \frac{h}{2}\left[f(t_n, y_n) + f(t_{n+1}, y_{n+1})\right]$$

稳定函数 $R(z) = \frac{1+z/2}{1-z/2}$，A-稳定。

**IMEX 梯形法**：对刚性线性项 L 隐式，对非线性项 N 显式：

$$(I - \frac{\Delta t}{2}L)y^{n+1} = (I + \frac{\Delta t}{2}L)y^n + \frac{\Delta t}{2}(N^n + N^{n+1,guess})$$

### 2.8 湍流热通量 PDF 拟合

重尾分布：$P(|Q|) \sim \exp(-b |Q|^\alpha)$，其中间歇性指数 $\alpha \in (0.4, 0.8)$ 标志雪崩输运事件。

---

## 3. 项目结构

```
289_synth_project_Advanced/
├── main.py                    # 统一入口（零参数运行）
├── __init__.py                # 包初始化
├── physics_constants.py       # 物理常数、平衡位形、等离子体参数
├── velocity_space.py          # 速度空间：Gauss-Patterson、CVT、Bessel
├── radial_mesh.py             # 1D 有限元网格、三角剖分、相空间 I/O
├── finite_difference.py       # 高阶有限差分算子、带状矩阵 R8GB
├── gyroaverage.py             # 有限拉莫尔半径 (FLR) 算子
├── collision_operator.py      # Lorentz 碰撞算子、Toeplitz 求解
├── time_integrator.py         # 隐式 / IMEX 时间推进器
├── linear_eigenvalue.py       # 线性 ITG 本征值问题、阈值搜索
├── zonal_flow.py              # Rosenbluth-Hinton 带状流残余
├── turbulence_diagnostics.py  # 诊断：SG 滤波、PDF、合成噪声
├── results_summary.json       # 运行结束后生成的 JSON 摘要
└── README_博士级合成说明.md    # 本文档
```

共 **11 个 .py 文件**（含 `main.py` 与 `__init__.py`）。

---

## 4. 种子项目映射表

本项目严格融合了 15 个输入种子项目的核心算法，每个都承担真实角色：

| # | 种子项目 | 核心算法 | 在本项目中的角色 |
|---|---|---|---|
| 1 | 1232_dhrichards_BridgingTheGapFigures | Savitzky-Golay 2D 滤波、球谐展开、specfab 算子演化、特征线回溯、温度参数化 | `turbulence_diagnostics.py` 中的 SG 滤波；`time_integrator.py` 的特征线追踪；`physics_constants.py` 的参数化平衡 |
| 2 | 1425_xyzf_display | XYZ/XYZF 3D 点面 I/O | `radial_mesh.py::PhaseSpaceIO` 5D 相空间二进制 I/O |
| 3 | 392_fem1d_heat_steady | 1D FEM 稳态热传导 | `radial_mesh.py::RadialMesh1D.solve_dirichlet` + `zonal_flow.py` 带状流残余椭圆求解 |
| 4 | 117_brc_data | 带噪声的随机采样 | `turbulence_diagnostics.py::SyntheticDiagnostic` 合成诊断噪声 |
| 5 | 851_patterson_rule | Gauss-Patterson 求积 | `velocity_space.py::patterson_rule_ab` 速度空间矩积分 |
| 6 | 831_ode_trapezoidal | 隐式梯形 ODE 求解 | `time_integrator.py::ode_trapezoidal` |
| 7 | 979_r8gb | 带状矩阵 R8GB PLU 分解 | `finite_difference.py::BandedMatrixR8GB` |
| 8 | 389_fem1d_function_10_display | FEM 分段线性函数 I/O | `radial_mesh.py::RadialMesh1D.from_file` / `.save` |
| 9 | 080_besselj_zero | Bessel 函数零点与求值 | `velocity_space.py::jn_eval`、`jn_zeros`；`gyroaverage.py` 中的 J₀ 应用 |
| 10 | 377_fem_neumann | basic_hat + Neumann 反应扩散 | `radial_mesh.py::basic_hat`；碰撞算子的 Neumann 边界 |
| 11 | 245_cvt_1d_nonuniform | 非均匀密度 CVT | `velocity_space.py::cvt_1d_nonuniform` 自适应速度网格 |
| 12 | 999_r8sto | 对称 Toeplitz 求解 (Levinson) | `collision_operator.py::CollisionMatrix` 周期卷积 |
| 13 | 1354_triangulation_triangle_neighbors | 三角形邻接拓扑 | `radial_mesh.py::TriangularPoloidalMesh` 极向非结构网格 |
| 14 | 1001_Eddien826_SA | 经纬度区域平均、气候数据 | `turbulence_diagnostics.py::regional_mean_2d` 通量面平均 |
| 15 | 540_histogram_display | 堆叠直方图 | `turbulence_diagnostics.py::FluxPDF` 热通量 PDF 统计 |

---

## 5. 运行方法

```bash
cd "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/289_synth_project/289_synth_project_Advanced"
python main.py
```

**零参数运行**，所有物理参数在代码中配置。

### 预期输出

```
PROJECT 289 -- 1D slab gyrokinetic turbulence simulator
Domain:  计算等离子体 / 湍流输运 / gyrokinetic 模拟
          高阶有限差分 + 稳定性分析 (小规模可复现实验)
========================================================================

Step 0 -- Equilibrium construction
  ion sound speed  c_s      = 2.1885e+05 m/s
  ion cyclotron    omega_ci = 1.1974e+08 rad/s
  ...

Step 5 -- Slab ITG eigenvalue problem
  top 5 eigenvalues (Re, Im):
    omega = +0.00000 + +3.03580 j
  max growth rate = +3.03580

Step 8 -- Synthetic turbulence time series and diagnostics
  heat-flux kurtosis = 24.2296     (heavy tail, non-Gaussian)
  stretched-exp (alpha, b) = (0.78, 2.13)
  ...

summary written to  results_summary.json
Pipeline complete
  wall-clock time: ~17 s
```

运行结束后会在当前目录生成 `results_summary.json`，包含全部关键物理量。

---

## 6. 关键技术点

### 6.1 数值方法

- **高阶紧致差分**：4 阶 Padé 格式求解一阶导数，4 阶中心差分求解二阶导数，边界用单侧 4 阶格式
- **带状矩阵 LU**：完全移植 LINPACK `r8gb` 风格的带状 PLU 分解
- **隐式梯形 + IMEX**：Crank-Nicolson 处理刚性项，SSP-RK3 处理对流项
- **Gauss-Patterson 嵌套求积**：速度空间高阶积分
- **CVT Lloyd 迭代**：Maxwellian 加权的自适应速度网格

### 6.2 边界处理与鲁棒性

- 所有物理量都有 positivity floor（密度、温度、扩散系数）
- Bessel 函数使用 scipy 的 AMOS 稳定实现，零点用二分法精确求取
- 带状矩阵求解有 dense-LU fallback
- 所有时间步都检测 NaN / Inf 并提前中止
- Poisson 方程分母有 ±ε 钳位避免除零
- Toeplitz 求解先尝试 Cholesky，失败则 fallback 到 dense LU

### 6.3 工程复杂度

- 11 个独立模块，每个都承担明确的物理角色
- 严格的类型注解与数据类 (`dataclass`)
- 完整的 self-check（每个模块都有 `if __name__ == "__main__"` 自检）
- 主入口统一调度 9 个物理步骤，每步都有 try/except 保护
- JSON 摘要自动序列化所有 numpy 类型

---

## 7. 可扩展方向

本项目是 **博士级科研计算** 的最小可复现实验。可扩展方向：

1. **δf 粒子模拟**：用 CVT 生成的最优标记粒子做 Monte-Carlo 速度空间积分
2. **非线性项**：实现 E×B 非线性对流 `v_E · ∇δf`，用特征线方法
3. **电磁效应**：加入 A_∥ 自由度，升级为电磁 gyrokinetic
4. **真实几何**：用 s-alpha 或 Miller 平衡替换平板近似
5. **谱方法**：在极向用球谐展开替换 ballooning harmonics
6. **GPU 加速**：将带状矩阵求解与速度空间积分迁移到 CUDA

---

## 8. 参考文献

1. Frieman & Chen, *Phys. Fluids* **25**, 502 (1982) — 非线性 gyrokinetic 方程
2. Hasegawa & Mima, *Phys. Fluids* **21**, 87 (1978) — HM 方程
3. Rosenbluth & Hinton, *Phys. Rev. Lett.* **80**, 724 (1998) — 带状流残余
4. Romanelli, *Phys. Fluids B* **1**, 1018 (1989) — slab ITG
5. Lele, *J. Comp. Phys.* **103**, 16 (1992) — 紧致差分格式
6. Abel et al., *Plasma Phys. Control. Fusion* **54**, 124010 (2012) — 碰撞算子
7. Helander & Sigmar, *Collisional Transport in Magnetized Plasmas* (CUP 2002)
8. Terry, *Rev. Mod. Phys.* **88**, 021003 (2016) — 湍流综述
9. Zhang & Jin, *Computation of Special Functions* (Wiley 1996) — Bessel 函数
10. Burkardt, SLATEC-style `r8gb`, `r8sto`, `fem1d_*` 系列 — 数值算法移植来源

---

**作者**：自动合成工作流 DA  
**日期**：2026-06-08  
**语言**：Python 3（依赖：numpy, scipy）
