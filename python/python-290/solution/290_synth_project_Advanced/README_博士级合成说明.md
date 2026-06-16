# PROJECT_290：计算等离子体 - 阿尔芬波与高能粒子相互作用

## 博士级合成项目说明文档

**科学领域**: 计算等离子体物理  
**核心问题**: 阿尔芬波与高能粒子的非线性相互作用  
**数值方法**: 高阶有限差分 + 本征值稳定性分析  
**规模**: 小规模可复现实验

---

## 一、项目概述

本项目实现了一个面向磁约束聚变等离子体中**阿尔芬波-高能粒子相互作用**问题的完整数值模拟系统。在 ITER 等托卡马克装置中，来自中性束注入（NBI）和聚变反应的 α 粒子等高能粒子可以通过共振将能量传递给阿尔芬波，引发**环向阿尔芬本征模（TAE）**等不稳定性，进而散射高能粒子、降低聚变约束性能。

本项目围绕以下控制方程展开：

### 1.1 简化 MHD 方程（阿尔芬波）

$$
\frac{\partial \psi}{\partial t} = -\nabla_\parallel \phi + \eta \nabla^2 \psi
$$

$$
\frac{\partial U}{\partial t} = v_A^2 \nabla_\parallel J - [\phi, U] + \nu \nabla^2 U
$$

其中 $\psi$ 为扰动磁通量函数，$\phi$ 为电势，$U = \nabla^2\phi$ 为涡度，$J = \nabla^2\psi$ 为扰动电流，$\nabla_\parallel = (1/qR_0)\partial/\partial\theta$ 为沿磁力线方向导数，$[\cdot,\cdot]$ 为泊松括号。

### 1.2 高能粒子漂移动力学方程

$$
\frac{\partial f}{\partial t} + v_\parallel \mathbf{b}\cdot\nabla f + \mathbf{v}_d\cdot\nabla f + \frac{\mu}{m}(\mathbf{b}\cdot\nabla B)\frac{\partial f}{\partial v_\parallel} = C[f] + S
$$

其中 $f = f(\mathbf{r}, v_\parallel, \mu, t)$ 为高能粒子分布函数，$\mu = mv_\perp^2/(2B)$ 为绝热不变量磁矩，$\mathbf{v}_d$ 为曲率与 $\nabla B$ 漂移速度。

### 1.3 波-粒子共振条件

通行粒子共振：
$$
\omega - n\Omega_\phi - k_\parallel v_\parallel = 0
$$

捕获粒子弹跳共振：
$$
\omega - n\Omega_\phi - p\Omega_b = 0
$$

---

## 二、15 个种子项目到科学问题的映射

| # | 种子项目 | 原领域 | 映射到本项目 | 关键算法 |
|---|---------|--------|-------------|---------|
| 1 | 1412_weekday_zeller | 日期计算 (Zeller 同余) | 周期性边界处理 | 模运算循环缠绕，用于环向/极向周期边界 |
| 2 | 329_ellipse_distance | Monte Carlo 距离统计 | EP 轨道统计 | Monte Carlo 采样估计粒子轨道位移分布 |
| 3 | 042_asa144 | 固定边际约束随机化 | 约束分布重构 | 在守恒量约束下随机化分布函数 |
| 4 | 258_cvt_metric | 度量张量 CVT | 自适应网格 | 基于度量张量的自适应网格加密 |
| 5 | 187_clausen | Clausen 函数 Chebyshev 展开 | 等离子体色散函数 | Clenshaw 递推、Chebyshev 级数 |
| 6 | 203_companion_matrix | 正交基伴随矩阵 | 色散关系求根 | Chebyshev 伴随矩阵特征值法 |
| 7 | 1434_zombie_ode | 多室 ODE (SZR) | 多种高能粒子群体 | 多室耦合 ODE + 守恒量追踪 |
| 8 | 573_image_boundary | 图像边界提取 | 等离子体边界几何 | 边界轮廓提取与 POLY 格式输出 |
| 9 | 987_r8pbl | SPD 带状矩阵 | 隐式时间推进 | 带状矩阵压缩存储与矩阵向量乘 |
| 10 | 1095_blip_bio-model | 多尺度 LBM 模拟 | 速度空间碰撞 | MRT 碰撞算子、速度空间离散 |
| 11 | 1336_triangulation_display | 三角网格显示 | 磁通量面网格 | 三角化邻接关系计算 |
| 12 | 1162_rPMDD | 时间序列统计分析 | 增长率诊断 | 时序分析、DiD、Fisher 检验 |
| 13 | 182_circle_positive_distance | 圆上距离统计 | EP 轨道位移 | 曲线几何上的 Monte Carlo 统计 |
| 14 | 474_gmsh_io | Gmsh 网格格式 | 网格数据 I/O | Gmsh MSH 格式读写 |
| 15 | 224_cplex_solution_read | CPLEX 解文件解析 | 特征值结果解析 | 结构化数据解析与重构 |

---

## 三、项目结构与文件说明

```
290_synth_project_Advanced/
├── main.py                        统一入口 (零参数运行)
├── __init__.py                    包初始化
├── plasma_config.py               等离子体物理参数定义
├── magnetic_geometry.py           磁平衡几何与自适应网格
├── boundary_handler.py            周期性边界与螺旋映射
├── high_order_operators.py        高阶有限差分算子与带状矩阵
├── alfven_wave_solver.py          阿尔芬波 RK4/CN 时间推进求解器
├── energetic_particle_kinetics.py 高能粒子动力学与 MRT 碰撞
├── dispersion_analysis.py         色散关系与特殊函数
├── stability_eigenvalue.py        本征值稳定性分析与 δW 方法
├── conservation_monitor.py        守恒量监测与约束分布重构
├── statistical_diagnostics.py     增长率估计、频谱分析、统计检验
├── solution_io.py                 模拟数据 I/O 与结果解析
├── output/                        输出目录 (运行时生成)
└── README_博士级合成说明.md        本文档
```

共 **12 个 .py 文件**，**1 个 .md 文档**，**1 个 main.py 入口**。

---

## 四、核心算法与关键公式

### 4.1 高阶有限差分（2N 阶中心差分）

一阶导数：
$$
f'(x_i) \approx \frac{1}{h}\sum_{k=1}^{N} c_k \left[f(x_{i+k}) - f(x_{i-k})\right]
$$

4阶精度系数：$c_1 = 4/3, \quad c_2 = -1/12$

6阶精度系数：$c_1 = 3/4, \quad c_2 = -3/20, \quad c_3 = 1/60$

二阶导数：
$$
f''(x_i) \approx \frac{1}{h^2}\left[d_0 f(x_i) + \sum_{k=1}^{N} d_k \left(f(x_{i+k}) + f(x_{i-k})\right)\right]
$$

### 4.2 四阶 Runge-Kutta 时间推进

$$
k_1 = f(t_n, y_n)
$$
$$
k_2 = f\left(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2}k_1\right)
$$
$$
k_3 = f\left(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2}k_2\right)
$$
$$
k_4 = f(t_n + \Delta t, y_n + \Delta t\, k_3)
$$
$$
y_{n+1} = y_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

CFL 稳定性条件：
$$
\Delta t < \text{CFL} \cdot \frac{\Delta r}{v_A}, \quad \text{CFL} \leq 1
$$

### 4.3 带状矩阵压缩存储（R8PBL 格式）

对于 N×N 对称正定矩阵，半带宽 ML：
- 存储规模：$(ML+1) \times N$ 实数
- 对角元：`A_data[0, j] = A(j, j)`
- 第 k 条次对角：`A_data[k, j] = A(j+k, j)`

利用对称性的矩阵向量乘：
$$
b_i = A_{ii}x_i + \sum_{k=1}^{ML}\left[A_{i+k,i}x_i + A_{i+k,i}x_{i+k}\right]
$$

### 4.4 等离子体色散函数（Fried-Conte 函数）

$$
Z(\zeta) = \frac{1}{\sqrt{\pi}} \int_{-\infty}^{\infty} \frac{e^{-t^2}}{t - \zeta}\,dt, \quad \text{Im}(\zeta) > 0
$$

小参数展开：$Z(\zeta) \approx -2\zeta\left(1 - \frac{2\zeta^2}{3} + \cdots\right)$

大参数渐近：$Z(\zeta) \approx i\sqrt{\pi} e^{-\zeta^2} - \frac{1}{\zeta}\sum_{n=0}^{\infty} \frac{(2n-1)!!}{(2\zeta^2)^n}$

### 4.5 Chebyshev 伴随矩阵求根

多项式 $p(x) = \sum_{k=0}^n c_k T_k(x)$ 的根可通过构造 Chebyshev 伴随矩阵：

$$
A = \begin{pmatrix}
0 & 1/2 & & & \\
1/2 & 0 & 1/2 & & \\
& \ddots & \ddots & \ddots & \\
& & 1/2 & 0 & 1/2 \\
-\frac{c_0}{2c_n} & \cdots & & -\frac{c_{n-1}}{2c_n} & 0
\end{pmatrix}
$$

特征值即为多项式的根。

### 4.6 阿尔芬连续谱与 TAE 间隙

连续谱分支：
$$
\omega_m(r) = \left|\frac{n}{R_0}\left(1 - \frac{m}{nq(r)}\right)\right| v_A
$$

TAE 间隙频率：
$$
\omega_{\text{TAE}} \approx \frac{(2m+1)n v_A}{2qR_0}\left(1 - \frac{\varepsilon^2}{4} + \cdots\right)
$$

### 4.7 能量原理（δW 方法）

$$
\delta W = \delta W_{\text{mag}} + \delta W_{\text{kin}} + \delta W_\parallel + \delta W_{\text{EP}}
$$

稳定判据：$\delta W > 0 \iff$ 稳定

### 4.8 MRT 碰撞算子

$$
\hat{C} = -M^{-1} S M (f - f^{eq})
$$

BGK 简化：
$$
f^{n+1} = f^n - \frac{\nu \Delta t}{1 + \nu \Delta t}(f^n - f^{eq})
$$

---

## 五、模拟流程

```
main.py 执行流程
│
├─ Step 1:  等离子体参数配置 (plasma_config.py)
│   计算 v_A, Ω_ci, d_i, ρ_s, β, β_fast 等
│
├─ Step 2:  磁平衡几何与网格 (magnetic_geometry.py)
│   生成椭圆截面通量面、POLY 文件、Gmsh 网格、CVT 自适应网格
│
├─ Step 3:  高能粒子动力学 (energetic_particle_kinetics.py)
│   初始化麦克斯韦分布、MRT 碰撞演化、守恒量追踪
│
├─ Step 4:  边界处理与高阶算子验证
│   周期边界、磁力线追踪、有理面检查、收敛阶验证
│
├─ Step 5:  阿尔芬波时间推进 (alfven_wave_solver.py)
│   初始化 TAE 模、RK4 推进、能量监测
│
├─ Step 6:  守恒量监测 (conservation_monitor.py)
│   追踪能量/磁通/正则动量、约束分布重构
│
├─ Step 7:  本征值稳定性分析 (stability_eigenvalue.py)
│   构建线性算子矩阵、本征值分解、δW 能量原理
│
├─ Step 8:  色散关系与连续谱 (dispersion_analysis.py)
│   等离子体色散函数、TAE 频率、Chebyshev 求根、MC 轨道统计
│
├─ Step 9:  统计诊断 (statistical_diagnostics.py)
│   增长率估计、FFT 频谱、模分解、Fisher 检验、DiD 分析
│
└─ Step 10: 保存结果 (solution_io.py)
    eigenvalue_result.json, simulation_history.json,
    plasma_boundary.poly, flux_surfaces.msh 等
```

---

## 六、运行方式

### 6.1 零参数运行

```bash
cd /mnt/data/zpy/sci-swe/source code/Synthesis-project-python/290_synth_project/290_synth_project_Advanced
python main.py
```

无需任何命令行参数，程序自动完成从参数配置到结果保存的完整流程。

### 6.2 输出文件

运行后在 `output/` 目录生成：
- `eigenvalue_result.json`：本征值分析结果
- `simulation_history.json`：时间序列历史
- `stability_result.json`：稳定性分析结果
- `plasma_params.json`：等离子体参数
- `plasma_boundary.poly`：等离子体边界 POLY 文件
- `flux_surfaces.msh`：磁通量面 Gmsh 网格

### 6.3 预期运行时间

在小规模配置（32×16 网格、80 时间步）下，约 **11 秒**完成全部模拟。

---

## 七、关键物理结果解读

### 7.1 典型输出

```
阿尔芬速度 v_A = 1.34e+07 m/s
离子回旋频率 Ω_ci = 2.39e+08 rad/s
离子惯性长度 d_i = 5.58e-02 m
等离子体比压 β = 3.82e-07
高能粒子 β_fast = 1.61e-04
TAE 频率 f_TAE ≈ 2098 kHz
最大增长率 γ_max ≈ 2.78e+07 s⁻¹
```

### 7.2 稳定性诊断

- **本征值分析**：通过线性算子矩阵的特征值实部判断稳定性，γ > 0 表示不稳定性
- **能量原理（δW）**：δW > 0 表示系统稳定，δW < 0 表示不稳定
- **Nyquist 判据**：通过色散函数 D(ω) 在原点的包围数判断不稳定模数量

### 7.3 物理意义

TAE 不稳定性由高能粒子的逆向波-粒子共振驱动。当 β_fast 超过临界阈值时：
$$
\beta_{\text{fast}} > \beta_{\text{crit}} \sim \frac{1}{n} \cdot \frac{\omega_d}{\omega_{\text{TAE}}}
$$
TAE 模增长率 γ 变为正值，高能粒子被反常输运损失。

---

## 八、数值方法特性

### 8.1 高阶精度

支持 2、4、6、8 阶中心差分，空间截断误差为 $O(h^{2N})$。通过收敛阶验证确保达到理论精度。

### 8.2 周期性边界

环向/极向坐标通过模运算自动缠绕，避免边界处场值不连续。

### 8.3 数值稳定性

- CFL 条件控制时间步长
- 边界吸收层抑制反射
- 守恒量监测追踪数值误差
- 约束分布重构保持物理一致性

### 8.4 边界处理

- $r=0$ 处使用 L'Hôpital 规则处理柱坐标奇点
- 有理面 $q = m/n$ 处的连续谱奇异性通过正则化处理
- 边界层指数衰减吸收

---

## 九、扩展性与可复现性

本项目为**小规模可复现实验**设计，所有参数可通过 `PlasmaParameters` 类调整：
- 网格分辨率 (n_r, n_theta, n_phi)
- 有限差分阶数 (fd_order)
- 时间步数和 CFL 数
- 等离子体参数 (B₀, n_e, T_e, ...)
- 高能粒子参数 (E_fast, n_fast, ...)

所有输出文件使用 JSON 格式，便于后续分析和第三方复查。

---

## 十、科学问题总结

本项目解决的**前沿博士级科学问题**：

> **在托卡马克磁约束聚变等离子体中，如何准确预测高能粒子驱动的阿尔芬本征模（TAE/EAE）不稳定性的阈值、增长率和频率？**

该问题的核心挑战包括：
1. 多尺度耦合（宏观 MHD 波 + 微观粒子轨道）
2. 非线性波-粒子共振
3. 复杂几何中的连续谱结构
4. 高维相空间中的碰撞动力学

本项目通过高阶有限差分 + 本征值稳定性分析 + 多物理场耦合，提供了一个完整的小规模可复现研究平台。

---

*项目作者: DA 博士级合成项目 PROJECT_290*  
*合成日期: 2026*
