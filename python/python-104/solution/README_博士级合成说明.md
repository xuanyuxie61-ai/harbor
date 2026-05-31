# README — 博士级合成说明

## 项目概述

**项目名称**: 自适应光学波前校正系统的高保真数值模拟  
**科学领域**: 光学工程 — 自适应光学波前校正  
**合成语言**: Python  
**合成目录**: `/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/104_synth_project`

本项目将 **15 个种子科研代码项目** 的核心算法融合重构为一个面向**前沿博士级**自然科学计算问题的完整 Python 项目，用于模拟和研究地基大口径望远镜在强大气湍流条件下的自适应光学（AO）波前校正性能。

---

## 一、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 在合成项目中的角色 |
|------|--------|----------|-------------------|
| 1 | `718_matlab_commandline` | 文件操作、数据日志 | `data_io.py` 的波前数据/参数日志记录功能；`adaptive_sampling.py` 的采样点日志输出 |
| 2 | `858_pendulum_double_ode` | 双摆混沌 ODE 动力学 | `deformable_mirror.py` 中快速倾斜镜（FSM）双轴机械振动的耦合动力学模型 |
| 3 | `841_ozone_ode` | 臭氧光化学 ODE | `atmosphere_turbulence.py` 中大气光化学-热耦合 ODE，描述紫外光解引起的局部折射率扰动 |
| 4 | `967_r83v` | R83V 三对角矩阵线性代数（CG/Jacobi/GS/CR） | `wavefront_reconstruction.py` 中基于三对角系统的 zonal 波前重构快速求解器 |
| 5 | `901_porous_medium_exact` | 多孔介质方程 Barenblatt 精确解 | `atmosphere_turbulence.py` 中折射率非线性扩散修正（PME 耗散模型） |
| 6 | `709_magic4_matrix` | 4 阶幻方矩阵生成 | `deformable_mirror.py` 中驱动器非周期布局的幻方优化，避免共振简并 |
| 7 | `054_asa299` | 单形格点枚举 | `zernike_modes.py` 中 Zernike 系数在 L1 约束单形内的离散搜索优化 |
| 8 | `832_ode_sweep_parfor` | ODE 参数扫描 | `closed_loop_control.py` 中 PI 控制增益与带宽的参数空间扫描优化 |
| 9 | `932_pyramid_grid` | 金字塔结构化网格 | `shack_hartmann_sensor.py` 中金字塔型子孔径网格划分；`wavefront_propagation.py` 中金字塔光线追迹 |
| 10 | `140_caustic` | 圆内焦散线几何 | `wavefront_propagation.py` 中光波焦散奇点检测与焦散网络生成 |
| 11 | `060_axon_ode` | Hodgkin-Huxley 神经轴突模型 | `closed_loop_control.py` 中高速控制电路的 HH 门控响应模型 |
| 12 | `1248_tetrahedron_integrals` | 单位四面体单项式积分 | `optical_transfer.py` 中四面体域上的光学传递函数积分与相位矩计算 |
| 13 | `1420_xy_io` | 2D 几何数据文件 I/O | `data_io.py` 中 XY/XYF/XYL 格式的波前数据、Zernike 系数、子孔径斜率读写 |
| 14 | `199_collatz_recursive` | Collatz 递归序列 | `iterative_utils.py` 中 Collatz-like 自适应步长调度与收敛控制 |
| 15 | `262_cvt_triangle_uniform` | 三角形域 CVT 最优剖分 | `adaptive_sampling.py` 中基于 CVT 的自适应采样点分布与网格加密 |

---

## 二、新增数学物理模型与核心公式

### 2.1 大气湍流相位屏生成

#### Kolmogorov 功率谱
在单位圆盘上，相位扰动的功率谱密度为：

$$
\Phi(f) = 0.023 \, r_0^{-5/3} \, (f_x^2 + f_y^2)^{-11/6}
$$

其中 $r_0$ 为 Fried 参数，与视宁度（seeing）的关系为：

$$
r_0 = \frac{\lambda}{\theta_{\text{seeing}} / 206265}
$$

#### von Kármán 谱（含外尺度 $L_0$）
$$
\Phi_{\text{vK}}(f) = \Phi(f) \cdot (f^2 + f_0^2)^{-11/6}, \quad f_0 = \frac{1}{L_0}
$$

#### Hufnagel-Valley $C_n^2$ 模型
$$
C_n^2(h) = 0.00594 \left(\frac{v}{27}\right)^2 (10^{-5} h)^{10} e^{-h/1000}
+ 2.7\times10^{-16} e^{-h/1500} + A e^{-h/100}
$$

### 2.2 多孔介质非线性扩散修正（源自 901）

折射率扰动 $\delta n$ 满足多孔介质方程（PME）：

$$
\frac{\partial (\delta n)}{\partial t} = D_{\text{eff}} \nabla^2 \big( (\delta n)^m \big), \quad m > 1
$$

Barenblatt 自相似解：

$$
\alpha = \frac{1}{m-1}, \quad \beta = \frac{1}{m+1}, \quad \gamma = \frac{m-1}{2m(m+1)}
$$

$$
u(x,t) = \max\!\left(0, \; (t+\delta)^{-\beta} \left[ c - \gamma \left( \frac{x}{(t+\delta)^{\beta}} \right)^2 \right]^{\alpha} \right)
$$

### 2.3 光化学-热耦合 ODE（源自 841）

状态向量 $\mathbf{y} = [\delta n, \delta T, [\text{O}_3], I_{\text{UV}}]^T$，光解速率：

$$
k_1(t) = 0.01 \cdot \max\!\left(0, \; \sin\frac{2\pi t}{t_d}\right)
$$

ODE 系统：

$$
\begin{aligned}
\frac{d(\delta n)}{dt} &= q_{\text{heat}} k_1 \, [\text{O}_3] - k_2 \, \delta n - k_3 \, \delta n \, \delta T \\
\frac{d(\delta T)}{dt} &= q_{\text{heat}} k_1 \, [\text{O}_3] - k_{\text{thermal}} \, \delta T \\
\frac{d[\text{O}_3]}{dt} &= -k_1 \, [\text{O}_3] + k_3 \, \delta n \, \delta T \\
\frac{dI_{\text{UV}}}{dt} &= -k_{\text{absorb}} I_{\text{UV}} [\text{O}_3] + S(t)
\end{aligned}
$$

积分采用经典四阶 Runge-Kutta（RK4）：

$$
\mathbf{k}_1 = h \mathbf{f}(t_n, \mathbf{y}_n), \quad
\mathbf{k}_2 = h \mathbf{f}\!\left(t_n+\frac{h}{2}, \mathbf{y}_n+\frac{\mathbf{k}_1}{2}\right)
$$

$$
\mathbf{k}_3 = h \mathbf{f}\!\left(t_n+\frac{h}{2}, \mathbf{y}_n+\frac{\mathbf{k}_2}{2}\right), \quad
\mathbf{k}_4 = h \mathbf{f}(t_n+h, \mathbf{y}_n+\mathbf{k}_3)
$$

$$
\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{1}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
$$

### 2.4 Zernike 多项式

径向多项式：

$$
R_n^m(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! \, ((n+m)/2 - k)! \, ((n-m)/2 - k)!} \rho^{n-2k}
$$

角向函数：

$$
\Phi_m(\theta) = \begin{cases}
\cos(m\theta), & m \ge 0 \\
\sin(|m|\theta), & m < 0
\end{cases}
$$

### 2.5 波前重构（Southwell 离散化 + R83V 求解）

斜率-相位关系：

$$
s_x(i+0.5, j) = \frac{\phi_{i+1,j} - \phi_{i,j}}{\Delta x}, \quad
s_y(i, j+0.5) = \frac{\phi_{i,j+1} - \phi_{i,j}}{\Delta y}
$$

最小二乘目标导出正规方程 $\mathbf{A} \boldsymbol{\phi} = \mathbf{b}$，其中 $\mathbf{A}$ 为三对角矩阵，采用 R83V 格式存储并求解。

**共轭梯度法（CG）**：

$$
\alpha_k = \frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{p}_k^T \mathbf{A} \mathbf{p}_k}, \quad
\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k
$$

$$
\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k \mathbf{A} \mathbf{p}_k, \quad
\beta_k = \frac{\mathbf{r}_{k+1}^T \mathbf{r}_{k+1}}{\mathbf{r}_k^T \mathbf{r}_k}
$$

$$
\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k
$$

**循环约化法（Cyclic Reduction）**：基于 Hockney (1965) 的奇偶重排与块 LU 分解。

### 2.6 Shack-Hartmann 传感器

子孔径斜率：

$$
s_x^{(k)} = \frac{1}{A_k} \iint_{\text{subap}_k} \frac{\partial W}{\partial x} \, dA, \quad
s_y^{(k)} = \frac{1}{A_k} \iint_{\text{subap}_k} \frac{\partial W}{\partial y} \, dA
$$

质心算法：

$$
x_c = \frac{\sum x \, I(x,y)}{\sum I(x,y)}, \quad
y_c = \frac{\sum y \, I(x,y)}{\sum I(x,y)}
$$

### 2.7 变形镜与快速倾斜镜

**高斯影响函数**：

$$
I_k(x,y) = \exp\!\left( -\frac{(x-x_k)^2 + (y-y_k)^2}{2\sigma^2} \right)
$$

**镜面变形**：

$$
\phi_{\text{DM}}(x,y) = \sum_{k=1}^{N_{\text{act}}} v_k \, I_k(x,y)
$$

**FSM 双轴动力学（耦合双摆，源自 858）**：

$$
\ddot{\theta}_1 = \frac{-g(2m_1+m_2)\sin\theta_1 - m_2 g \sin(\theta_1-2\theta_2) - 2m_2(\dot{\theta}_2^2 l_2 + \dot{\theta}_1^2 l_1 \cos\Delta)\sin\Delta + T_1}{2l_1(m_1+m_2-m_2\cos^2\Delta)}
$$

其中 $\Delta = \theta_1 - \theta_2$。

### 2.8 焦散奇点检测

波前 Hessian：

$$
\mathbf{H} = \begin{bmatrix} W_{xx} & W_{xy} \\ W_{xy} & W_{yy} \end{bmatrix}
$$

焦散条件：

$$
\det(\mathbf{H}) = W_{xx} W_{yy} - W_{xy}^2 < 0
$$

### 2.9 光学传递函数

瞳孔函数：

$$
P(x,y) = A(x,y) \exp\big(i \phi(x,y)\big)
$$

OTF：

$$
\text{OTF}(f_x, f_y) = \iint P(x,y) P^*(x-\lambda z f_x, y-\lambda z f_y) \, dx dy
$$

Strehl 比：

$$
S = \frac{\displaystyle\Big|\iint_{\text{pupil}} e^{i\phi} \, dA\Big|^2}{A^2}
$$

### 2.10 四面体积分（源自 1248）

单位四面体 $T$ 上的单项式积分：

$$
\iiint_T x^{e_1} y^{e_2} z^{e_3} \, dV = \frac{e_1! \, e_2! \, e_3!}{(e_1+e_2+e_3+3)!}
$$

### 2.11 CVT 自适应采样（源自 262）

Lloyd 迭代：

$$
\mathbf{z}_i^{(k+1)} = \frac{1}{|V_i|} \iint_{V_i} \mathbf{x} \, dA, \quad
V_i = \{ \mathbf{x} \in \Omega \mid \|\mathbf{x} - \mathbf{z}_i\| \le \|\mathbf{x} - \mathbf{z}_j\|, \forall j \}
$$

### 2.12 Hodgkin-Huxley 控制电路（源自 060）

门控速率函数：

$$
\alpha_n(V) = 0.01 \frac{10-V}{e^{(10-V)/10}-1}, \quad
\beta_n(V) = 0.125 e^{-V/80}
$$

$$
\alpha_m(V) = 0.1 \frac{25-V}{e^{(25-V)/10}-1}, \quad
\beta_m(V) = 4.0 e^{-V/18}
$$

$$
\alpha_h(V) = 0.07 e^{-V/20}, \quad
\beta_h(V) = \frac{1}{e^{(30-V)/10}+1}
$$

膜电位方程：

$$
C \frac{dV}{dt} = I_{\text{ext}} - n^4 G_K (V-E_K) - m^3 G_{\text{Na}} h (V-E_{\text{Na}})
$$

### 2.13 Collatz 自适应步长（源自 199）

步长调度：

$$
s = \frac{s_{\text{base}}}{1 + 2^{\lfloor \log_2(r / s_{\text{base}}) \rfloor}}
$$

其中 $r$ 为残差范数。

### 2.14 闭环 PI 控制

控制律：

$$
u(t) = K_p \, e(t) + K_i \int_0^t e(\tau) \, d\tau
$$

带宽受限执行器（一阶低通）：

$$
\frac{du_{\text{filt}}}{dt} = \frac{u - u_{\text{filt}}}{\tau_c}, \quad \tau_c = \frac{1}{2\pi f_c}
$$

---

## 三、文件结构与修改说明

```
104_synth_project/
├── main.py                          # 统一入口，零参数运行
├── data_io.py                       # 数据读写（1420_xy_io + 718）
├── iterative_utils.py               # 迭代工具（199_collatz_recursive）
├── zernike_modes.py                 # Zernike 模式 + 单形枚举（054_asa299）
├── atmosphere_turbulence.py         # 湍流 + PME + 光化学 ODE（841 + 901）
├── wavefront_propagation.py         # 波前传播 + 焦散（140_caustic + 932）
├── shack_hartmann_sensor.py         # SH 传感器（932_pyramid_grid）
├── wavefront_reconstruction.py      # 波前重构 + R83V 求解（967_r83v）
├── optical_transfer.py              # OTF/PSF + 四面体积分（1248）
├── deformable_mirror.py             # DM + FSM 动力学（709 + 858）
├── adaptive_sampling.py             # CVT 采样（262 + 718）
├── closed_loop_control.py           # 闭环控制 + HH + 参数扫描（832 + 060）
└── README_博士级合成说明.md         # 本文档
```

---

## 四、科学问题与解决能力

本项目解决的核心科学问题是：

> **在强大气湍流和复杂热-化学耦合条件下，地基大口径望远镜如何通过 Shack-Hartmann 波前传感器、变形镜和快速倾斜镜的协同工作，实现衍射极限成像？**

具体能力包括：
1. **高保真湍流模拟**：结合 Kolmogorov 谱、PME 非线性扩散和光化学-热 ODE，生成具有物理真实性的大气相位屏。
2. **多方法波前重构**：支持 modal（Zernike 基）和 zonal（Southwell 离散化 + R83V 快速求解）两种重构策略。
3. **完整闭环控制**：包含 PI 控制器、带宽受限执行器、HH 型高速电路响应，以及控制参数空间扫描优化。
4. **光学性能评估**：计算 OTF、MTF、PSF、Strehl 比、包围能量和焦散奇点分布。
5. **自适应采样优化**：基于 CVT 在高梯度区域自动加密采样点。

---

## 五、运行方式

### 环境要求
- Python >= 3.8
- NumPy
- SciPy

### 运行命令
```bash
cd "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/104_synth_project"
python main.py
```

### 运行流程
1. 初始化望远镜和 AO 系统参数
2. 生成综合大气湍流相位屏（Kolmogorov + PME 修正 + 光化学-热扰动）
3. Shack-Hartmann 传感器采样并提取子孔径斜率
4. Zernike 模态分解与 zonal 波前重构
5. 变形镜面形计算与 FSM 双轴动力学响应
6. 闭环 PI 控制迭代校正
7. 光学传递函数、Strehl 比、焦散奇点分析
8. 控制参数扫描优化
9. CVT 自适应采样与日志输出

### 输出文件
运行后在 `output/` 目录下生成：
- `system_parameters.log` — 系统参数记录
- `subaperture_slopes.txt` — 子孔径斜率数据
- `zernike_coefficients_turbulence.txt` — Zernike 系数
- `adaptive_sampling_points.txt` — 自适应采样点坐标

---

## 六、质量检查清单

- [x] 原目录未被修改
- [x] 合成后项目为 Python 语言
- [x] 新目录完整包含合成后的项目
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] **15 个输入项目均已真实融入**，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性（NaN 检测、clip、限幅、非零除保护）
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成
- [x] 无可视化代码
