# README_博士级合成说明.md

## 激光诱导空化气泡崩溃的多物理场合成计算框架

---

## 一、项目概述

本项目围绕**计算流体力学：气泡动力学与空化**这一前沿科学领域，基于 15 个种子项目的核心算法，融合构建了一个面向激光诱导空化气泡近壁面非球形崩溃动力学的博士级多物理场计算框架。

### 科学问题

激光诱导空化（Laser-Induced Cavitation, LIC）是强激光脉冲在液体中聚焦产生等离子体，随后等离子体绝热膨胀形成空化气泡的复杂物理过程。气泡在生长至最大半径后，因周围液体惯性作用发生剧烈崩溃，产生高达 GPa 级的瞬态压力脉冲和高速微射流，对 nearby 固体壁面造成空蚀损伤。本项目的核心科学问题为：

> **如何在统一的计算框架中，耦合求解气泡壁运动、非球形变形、压力波传播、微团破碎及随机成核等多物理过程，并定量预测崩溃能量分配与壁面载荷？**

---

## 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 科学映射 |
|:---:|:---|:---|:---|
| 1 | `475_gmsh_to_fem` | GMSH 网格读取与 FEM 格式转换 | **FEM 网格生成模块**：为压力波传播求解器提供三角形网格数据结构 |
| 2 | `1221_test_nonlin` | 23 个非线性测试问题与 Jacobian | **Newton-Krylov 耦合求解器**：非线性残差函数与数值 Jacobian |
| 3 | `180_circle_map` | 2×2 矩阵对单位圆的映射 | **气泡椭圆变形分析**：条件数 κ(A) 量化非球形变形程度 |
| 4 | `060_axon_ode` | Hodgkin-Huxley 神经元 ODE | **气泡壁运动 ODE**：四变量 [R, dR/dt, T_g, n_g] 扩展系统 |
| 5 | `095_bisection_integer` | 整数二分根搜索 | **Blake 临界半径搜索**：二分法求解 p_g - p_∞ + 2σ/R = 0 |
| 6 | `231_cube_exactness` | 3D Legendre 求积精确度测试 | **高阶体积分**：Gauss-Legendre 3D 求积用于气泡体积计算 |
| 7 | `1014_rbf_interp_2d` | 2D RBF 插值（MQ/TPS/Gaussian/IMQ） | **3D 压力场重建**：径向基函数插值重构气泡周围三维压力分布 |
| 8 | `167_chebyshev2_rule` | Gauss-Chebyshev Type 2 求积 | **曲面边界积分**：权重 sqrt((x-a)(b-x)) 适合圆形/球形边界问题 |
| 9 | `301_disk01_monte_carlo` | 单位圆盘均匀采样 | **表面位点分布**：Monte Carlo 生成空化核初始位置 |
| 10 | `334_ellipsoid_monte_carlo` | 椭球均匀采样（Cholesky 变换） | **非球形核分布**：模拟表面粗糙度导致的非均匀成核位点 |
| 11 | `449_full_deck_simulation` | 多次独立实验统计 | **成核统计框架**：多次独立成核实验的均值、方差、极值统计 |
| 12 | `1183_supreme_vacancy` | 概率累积模型 | **位点激活概率**：1 - Π(1 - p_i) 计算至少一个位点被激活的概率 |
| 13 | `1367_tsp_random` | 随机搜索旅行商问题 | **能量路径优化**：将能量分配视为组合优化问题，随机搜索最优路径 |
| 14 | `655_leaf_chaos` | 迭代函数系统（IFS） | **微团混沌破碎**：四组仿射变换模拟不同尺度涡旋对微团的输运 |
| 15 | `410_fem2d_predator_prey_fast` | 2D FEM 反应扩散方程 | **压力波 FEM 求解器**：Newmark-β 时间离散 + 三角形单元弱形式 |

---

## 三、核心数学物理模型与公式

### 3.1 扩展 Rayleigh-Plesset 方程

不可压缩液体中球形气泡的径向运动：

```
R * d²R/dt² + (3/2) * (dR/dt)² = (p_g - p_∞)/ρ - 4μ*(dR/dt)/R - 2σ/(ρR)
```

其中：
- `R(t)`：气泡瞬时半径
- `p_g`：气泡内气体压力
- `p_∞`：远场液体压力
- `ρ`：液体密度
- `μ`：动力粘度
- `σ`：表面张力系数

### 3.2 Keller-Miksis 可压缩修正

考虑液体可压缩性的声学修正：

```
(1 - dR/dt/c) * R * d²R/dt² + (3/2 - dR/dt/(2c)) * (dR/dt)²
= (1 + dR/dt/c) * (p_g - p_∞)/ρ + (R/(ρc)) * d(p_g)/dt
```

其中 `c` 为液体中的声速。

### 3.3 Van der Waals 状态方程

气泡内真实气体状态方程：

```
(p_g + a_vdw * n_g²/V²) * (V - n_g*b_vdw) = n_g * R_g * T_g
```

### 3.4 非球形变形（Legendre 展开）

气泡界面用球谐函数展开：

```
r(θ, t) = R(t) * [1 + Σ_{n=2}^{N} a_n(t) * P_n(cosθ)]
```

第 n 阶模式的动力学方程：

```
d²a_n/dt² + 3(dR/dt/R) * da_n/dt - (n-1)(d²R/dt²/R) * a_n
= -(n-1)σ/(ρR³) * (n+2)(n-1) * a_n - 2μ(n-1)(n+2)/(ρR²) * da_n/dt
```

### 3.5 Blake 临界半径

气泡失稳的临界条件：

```
R_crit = √(2σ / (3(p_∞ - p_v)))
```

### 3.6 经典成核理论（CNT）

成核自由能势垒：

```
ΔG* = 16πσ³ / (3(p_v - p_∞)²)
```

成核速率：

```
J = J_0 * exp(-ΔG* / (k_B T))
```

### 3.7 FEM 弱形式（压力波方程）

声学波动方程的弱形式：

```
∫_Ω (∂²p/∂t²) v dΩ + c² ∫_Ω ∇p · ∇v dΩ = c² ∫_{∂Ω} (∂p/∂n) v dS
```

时间离散采用 Newmark-β 方法（β=1/4, γ=1/2）：

```
M * a^{n+1} + c² K * p^{n+1} = f^{n+1}
v^{n+1} = v^n + Δt/2 * (a^n + a^{n+1})
p^{n+1} = p^n + Δt * v^n + (Δt²/4) * (a^n + a^{n+1})
```

### 3.8 RBF 插值

三维径向基函数插值：

```
p(x) = Σ_{j=1}^{N} w_j * φ(||x - x_j||)
```

常用核函数：
- 多元二次（MQ）：`φ(r) = √(r² + r0²)`
- 高斯：`φ(r) = exp(-r²/r0²)`

### 3.9 混沌破碎（IFS）

微团运动的迭代函数系统：

```
x_{k+1} = A_j * x_k + b_j,   j ∈ {0,1,2,3}
```

Lyapunov 指数：`λ = lim_{n→∞} (1/n) Σ ln(||δx_{k+1}|| / ||δx_k||)`

计盒维数：`D_box = lim_{ε→0} log(N(ε)) / log(1/ε)`

### 3.10 高阶数值积分

Gauss-Chebyshev Type 2 求积：

```
∫_a^b f(x) * √((x-a)(b-x)) dx ≈ Σ_i w_i f(x_i)
x_i = (a+b)/2 + (b-a)/2 * cos(iπ/(n+1))
w_i = π/(n+1) * sin²(iπ/(n+1))
```

Gauss-Legendre 3D 求积：

```
∫∫∫ f(x,y,z) dxdydz ≈ Σ_i Σ_j Σ_k w_i w_j w_k f(x_i, y_j, z_k)
```

---

## 四、文件结构

```
080_synth_project/
├── main.py                    # 统一入口，零参数运行
├── utils.py                   # 物理常数、Cholesky 分解、Monte Carlo 采样工具
├── rayleigh_plesset_solver.py # 扩展 R-P / Keller-Miksis 求解 + Blake 二分 + Newton 稳态
├── bubble_shape_deformation.py # 椭圆变形映射 + Legendre 模式 + IFS 混沌破碎
├── surface_integrals.py       # Gauss-Chebyshev / Gauss-Legendre 高阶数值积分
├── fem_pressure_wave.py       # 2D FEM 压力波传播求解器
├── rbf_pressure_field.py      # 3D RBF 压力场重建
├── nucleation_statistics.py   # 随机成核统计模型
├── nonlinear_coupling.py      # 多物理场 Newton-Krylov 耦合求解器
├── energy_dissipation.py      # 能量分析与路径优化
└── README_博士级合成说明.md    # 本文档
```

---

## 五、运行方式

```bash
cd /mnt/data/zpy/sci-swe/source\ code/Synthesis-project-python/080_synth_project
python main.py
```

程序无需任何输入参数，所有物理参数在 `main.py` 中预设。运行后将在控制台输出 14 个计算模块的结果，涵盖从临界半径分析到能量优化的完整计算流程。

---

## 六、工程复杂性与数值鲁棒性

1. **边界处理**：所有涉及除法的运算均使用 `safe_divide`，避免除以零；半径、温度、摩尔数等物理量均施加正数约束。
2. **Jacobian 正则化**：Newton 迭代中 Jacobian 奇异时自动回退到最小二乘求解。
3. **线搜索阻尼**：Newton 步长采用回溯线搜索，确保残差单调下降。
4. **Picard 回退**：Newton 法不收敛时自动回退到 Picard 迭代。
5. **FEM 稳定性**：使用集中质量矩阵替代一致质量矩阵，提高显式时间推进的稳定性。
6. **RBF 正则化**：插值矩阵添加微量单位矩阵扰动，避免病态。
7. **ODE 事件处理**：`solve_ivp` 设置最大步长限制，确保时间分辨率。

---

## 七、科学意义

本合成项目将 15 个看似独立的数值算法项目，通过**激光诱导空化气泡动力学**这一前沿科学问题有机融合。项目涵盖了：
- **微尺度 ODE 动力学**（气泡壁运动）
- **偏微分方程数值解**（FEM 压力波）
- **无网格方法**（RBF 插值）
- **随机过程与统计物理**（成核模型）
- **非线性科学与混沌**（微团破碎）
- **优化理论**（能量路径搜索）

这些模块相互耦合，形成了一个完整的多物理场计算平台，可用于预测空化泡崩溃产生的壁面损伤、优化水声器件设计、以及理解细胞内激光手术中的物理机制。
