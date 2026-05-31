# 博士级合成说明：随机 Fisher-KPP 反应-扩散-对流方程的数值积分与不确定性量化

## 一、项目概述

本项目围绕**计算数学：随机微分方程数值积分**这一前沿领域，将 15 个种子科研代码项目的核心算法融合为一个博士级科学计算系统。

### 1.1 核心科学问题

求解一维随机 Fisher-Kolmogorov-Petrovsky-Piskunov (Fisher-KPP) 反应-扩散-对流方程：

$$
dU(t,x) = \Big[ \varepsilon \frac{\partial^2 U}{\partial x^2} - v \frac{\partial U}{\partial x} + r U\Big(1 - \frac{U}{K}\Big) \Big] dt + \sigma_{\text{noise}} \, U\Big(1 - \frac{U}{K}\Big) \, dW(t,x),
$$

其中：
- $U(t,x)$ 为未知随机场；
- $\varepsilon > 0$ 为扩散系数；
- $v$ 为对流速度；
- $r > 0$ 为内禀增长率；
- $K > 0$ 为环境容纳量；
- $\sigma_{\text{noise}}$ 为乘性噪声强度；
- $dW(t,x)$ 为柱维纳过程（cylindrical Wiener process），经截断后成为有限维 $Q$-维纳过程。

该方程是**随机偏微分方程（SPDE）**理论中的经典模型，在种群遗传学（基因频率波前传播）、燃烧理论（火焰锋面稳定性）和神经科学（随机神经网络活动波）中具有广泛的物理背景。

### 1.2 项目文件清单

| 序号 | 文件名 | 核心功能 | 融入的种子项目 |
|------|--------|----------|----------------|
| 1 | `wiener_process.py` | $Q$-维纳过程构造、L'Ecuyer 高质量 RNG、Box-Muller 变换 | 1040_rnglib |
| 2 | `mesh_generation.py` | CVT Lloyd 自适应网格、四面体网格、Fibonacci 螺旋采样 | 676_line_cvt_lloyd, 1247_tetrahedron_grid, 427_fibonacci_spiral |
| 3 | `spatial_operators.py` | 混合空间离散：FTCS、DG 通量、FEM 质量/刚度矩阵、Lax-Wendroff | 434_fisher_pde_ftcs, 272_dg1d_burgers, 404_fem2d_heat_rectangle, 065_ball_and_stick_display |
| 4 | `stochastic_rk.py` | 随机 Runge-Kutta（Platen SRK）、Milstein、半隐式 Euler、自适应 RK12 | 1029_rk12, 1164_stiff_ode, 907_praxis |
| 5 | `spde_core.py` | SPDE 主求解器、能量/质量守恒监测、边界条件处理 | 404_fem2d_heat_rectangle, 434_fisher_pde_ftcs |
| 6 | `monte_carlo_uq.py` | 蒙特卡洛系综、反变量法方差缩减、分层采样、Gelman-Rubin 诊断 | 533_high_card_parfor, 154_chain_letter_tree |
| 7 | `parameter_calibration.py` | PRAXIS 无梯度优化、Rosenbrock 基准、SPDE 扩散系数反演 | 907_praxis, 898_polynomials |
| 8 | `strategy_selector.py` | 基于局部 Peclet/CFL 指标的自适应格式选择（Reversi 博弈决策） | 1022_reversi_game |
| 9 | `numerical_utils.py` | 带状矩阵求解、Dirichlet/Neumann BC 施加、SVD 排序 | 404_fem2d_heat_rectangle, 907_praxis |
| 10 | `main.py` | 统一入口：执行全部 5 组实验并输出文本报告 | — |

---

## 二、原项目到科学问题的映射

### 2.1 434_fisher_pde_ftcs → 空间算子与反应项

原项目使用 FTCS 格式求解确定性 Fisher PDE：

$$
\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} + u(1-u).
$$

本项目将其扩展为**随机**情形，保留了 FTCS 的中心差分模板：

$$
D^2[U]_i = \frac{2}{h_{i-1}+h_i} \Big( \frac{U_{i+1}-U_i}{h_i} - \frac{U_i-U_{i-1}}{h_{i-1}} \Big),
$$

并引入 KPP 反应项 $f(U) = r U(1-U/K)$。

### 2.2 272_dg1d_burgers → 间断 Galerkin 数值通量

原项目求解 1D Burgers 方程的 DG 方法。本项目提取其**局部 Lax-Friedrichs 数值通量**：

$$
\hat{f}(U_L, U_R) = \frac{1}{2}\big(f_L + f_R\big) - \frac{\alpha}{2}(U_R - U_L), \quad \alpha = \max |f'(U)|,
$$

用于处理对流主导的梯度层区域，抑制 Gibbs 振荡。

### 2.3 1029_rk12 → 自适应随机 Runge-Kutta

原项目提供显式 RK1/RK2 误差估计。本项目将其思想推广到**随机微分方程**，构造嵌入对：

$$
\begin{aligned}
Y^{(1)}_{n+1} &= Y_n + f(Y_n)h + g(Y_n)\Delta W_n, \\
Y^{(2)}_{n+1} &= Y_n + \tfrac{1}{2}\big(f(Y_n)+f(Y^{(1)}_{n+1})\big)h + \tfrac{1}{2}\big(g(Y_n)+g(Y^{(1)}_{n+1})\big)\Delta W_n.
\end{aligned}
$$

局部误差 $e_n = |Y^{(2)}_{n+1} - Y^{(1)}_{n+1}|$ 用于自适应步长控制：

$$
h_{\text{new}} = h \cdot \min\Big\{5, \max\Big\{0.2, 0.9\sqrt{\frac{\text{tol}}{e_n}}\Big\}\Big\}.
$$

### 2.4 676_line_cvt_lloyd → 自适应空间网格优化

原项目实现一维 Lloyd 算法求解 Centroidal Voronoi Tessellation。本项目将其作为**自适应网格生成器**，最小化密度加权能量泛函：

$$
E(z_1,\dots,z_n) = \sum_{i=1}^n \int_{V_i} \rho(x) \|x - z_i\|^2 \, dx,
$$

其中密度函数 $\rho(x)$ 根据 Fisher-KPP 波前梯度集中特性设计：

$$
\rho(x) = \exp(-\lambda |x - x_c|) + 0.1.
$$

### 2.5 1022_reversi_game → 自适应数值格式决策

原项目为棋盘游戏的贪心最优策略搜索。本项目将其改造为**数值格式自适应选择器**：将每个空间节点视为棋盘格子，局部状态为物理指标向量 $s_i = [\text{Pe}_i, \text{CFL}_i, |\nabla U|_i, |\Delta U|_i]$，通过贪心评分选择最优离散格式（centered / upwind / Lax-Wendroff / shock-capturing）。

### 2.6 1040_rnglib → 高质量随机数与维纳过程

原项目实现 L'Ecuyer 的 CMRG 伪随机数生成器。本项目保留其核心递推思想：

$$
\begin{aligned}
x_{1,n} &= (40014 \cdot x_{1,n-1}) \mod 2147483647, \\
x_{2,n} &= (40692 \cdot x_{2,n-1}) \mod 2145483479,
\end{aligned}
$$

并进一步构造**截断 $Q$-维纳过程**：

$$
W_N(t,x) = \sum_{k=1}^{N} \sqrt{q_k} \, e_k(x) \, \beta_k(t), \quad q_k = \sigma^2 k^{-2\alpha}, \; \alpha > 0.5.
$$

### 2.7 533_high_card_parfor → 蒙特卡洛最优停止与分层

原项目估计最优停止策略（秘书问题变体）。本项目提取其**蒙特卡洛统计估计框架**，引入反变量法（Antithetic Variates）和分层采样（Stratified Sampling）进行方差缩减。分层估计量方差为：

$$
\text{Var}_{\text{strat}} = \sum_{l=1}^{L} \frac{N_l}{N^2} \sigma_l^2.
$$

### 2.8 065_ball_and_stick_display → Lax-Wendroff 格式

原项目演示 Lax-Wendroff 格式的时空模板。本项目提取其数值格式用于**对流主导区域**：

$$
U_i^{n+1} = U_i^n - \frac{c}{2}(U_{i+1}^n - U_{i-1}^n) + \frac{c^2}{2}(U_{i+1}^n - 2U_i^n + U_{i-1}^n), \quad c = \frac{v\Delta t}{\Delta x}.
$$

### 2.9 154_chain_letter_tree → 层次聚类距离矩阵

原项目构建距离矩阵并进行层次聚类。本项目将其用于**蒙特卡洛样本的层次距离度量**，构造伪距离矩阵以评估不同随机种子系综之间的统计距离，辅助分层方差缩减。

### 2.10 404_fem2d_heat_rectangle → 有限元矩阵组装与带状求解

原项目实现 2D 热方程的 FEM 求解。本项目将其**质量/刚度矩阵组装思想**降维到 1D，采用 lumped-mass 近似：

$$
M_i = \frac{1}{2}(h_{i-1} + h_i), \quad K_{ii} = \varepsilon\Big(\frac{1}{h_{i-1}} + \frac{1}{h_i}\Big) + \frac{|v|}{2}.
$$

并保留 LINPACK DGB 风格的带状矩阵存储与边界条件处理逻辑。

### 2.11 1247_tetrahedron_grid → 四面体网格生成

原项目生成标准四面体上的重心坐标网格。本项目保留其**重心坐标插值公式**：

$$
x = \frac{i v_1 + j v_2 + k v_3 + l v_4}{n}, \quad i+j+k+l = n,
$$

作为高维空间离散化的理论基础（虽然演示代码以 1D 为主，但框架可直接推广到 3D）。

### 2.12 1164_stiff_ode → 刚性 SDE 隐式处理

原项目求解刚性 ODE $y' = \lambda(\cos t - y)$。本项目将其**隐式处理思想**推广到刚性随机微分方程，采用半隐式 Euler：

$$
(I - h A) Y_{n+1} = Y_n + h f_{\text{nonlin}}(Y_n) + g(Y_n)\Delta W_n,
$$

其中 $A$ 为线性漂移的离散矩阵，使时间步长摆脱刚性约束 $h < 2/\lambda$。

### 2.13 898_polynomials → 优化基准测试函数

原项目提供 Rosenbrock、Camel 等经典多项式测试函数。本项目将其作为**无梯度优化器 PRAXIS 的验证基准**：

$$
\text{Rosenbrock}(x) = \sum_{i=1}^{n-1} \Big[ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \Big].
$$

### 2.14 907_praxis → 主方向无梯度优化

原项目实现 Brent 的 PRAXIS 算法。本项目提取其核心思想——**沿近似 Hessian 主轴进行无梯度搜索**，并通过 SVD 周期性更新搜索方向：

$$
V_{\text{new}} = U \Sigma^{-1}, \quad \text{由 } V^T D V \text{ 的 SVD 分解得到}.
$$

用于 SPDE 扩散系数的反演标定。

### 2.15 427_fibonacci_spiral → 拟均匀采样

原项目生成 Fibonacci 螺旋点。本项目将其推广为**圆盘上的拟均匀采样**：

$$
r_k = R \sqrt{\frac{k}{N - 0.5}}, \quad \theta_k = \frac{2\pi k}{\phi^2}, \quad \phi = \frac{1+\sqrt{5}}{2},
$$

该分布最小化 Riesz $s$-能量，适用于随机采样的空间初始化。

---

## 三、核心数学物理模型与公式

### 3.1 SPDE 的弱形式与 Galerkin 投影

对 Fisher-KPP SPDE 乘以测试函数 $\phi \in H_0^1(0,L)$ 并积分，得到弱形式：

$$
\begin{aligned}
d\langle U, \phi \rangle &= \Big[ -\varepsilon \Big\langle \frac{\partial U}{\partial x}, \frac{\partial \phi}{\partial x} \Big\rangle - v \Big\langle \frac{\partial U}{\partial x}, \phi \Big\rangle + r \big\langle U(1-U/K), \phi \big\rangle \Big] dt \\
&\quad + \sigma_{\text{noise}} \big\langle U(1-U/K), \phi \big\rangle \, dW(t).
\end{aligned}
$$

采用分段线性有限元基函数 $\{\phi_i\}$，得到 SDE 系统：

$$
M \, d\mathbf{U} = \big[ -K \mathbf{U} + \mathbf{R}(\mathbf{U}) \big] dt + \mathbf{G}(\mathbf{U}) \, d\mathbf{W}(t),
$$

其中 $M_{ij} = \int \phi_i \phi_j dx$ 为质量矩阵，$K_{ij} = \int \varepsilon \phi_i' \phi_j' + v \phi_i \phi_j' dx$ 为刚度矩阵。

### 3.2 Platen 显式强阶 1.0 SRK

对于对角噪声 SDE $dX = f(X)dt + g(X)dW$，Platen 格式为：

$$
\begin{aligned}
H_1 &= Y_n + f(Y_n)h + g(Y_n)\sqrt{h}, \\
H_2 &= Y_n + f(Y_n)h - g(Y_n)\sqrt{h}, \\
Y_{n+1} &= Y_n + f(Y_n)h + \frac{1}{2}\big(g(H_1)+g(H_2)\big)\Delta W_n \\
&\quad + \frac{1}{2\sqrt{h}}\big(g(H_1)-g(H_2)\big)\big(\Delta W_n^2 - h\big).
\end{aligned}
$$

该格式的**强收敛阶为 1.0**，优于 Euler-Maruyama 的 0.5。

### 3.3 Milstein 修正（乘性噪声）

对于乘性噪声 $g(U) = \sigma U(1-U/K)$，其 Jacobian 为 $g'(U) = \sigma(1 - 2U/K)$，Milstein 格式为：

$$
Y_{n+1} = Y_n + f(Y_n)h + g(Y_n)\Delta W_n + \frac{1}{2} g(Y_n) g'(Y_n) \big(\Delta W_n^2 - h\big).
$$

### 3.4 能量估计与稳定性

对 Fisher-KPP SPDE 施加 Itô 公式，$L^2$ 能量满足：

$$
\mathbb{E}\big[ \|U(t)\|^2 \big] \leq C \exp\big( (2r + \sigma_{\text{noise}}^2) t \big).
$$

数值格式需保持该指数增长上界，否则视为不稳定。代码中通过 `compute_energy` 方法进行实时监测。

### 3.5 自适应 Peclet-CFL 判据

局部 Peclet 数与 CFL 数：

$$
\text{Pe}_i = \frac{|v| h_i}{\varepsilon}, \qquad \text{CFL}_i = \frac{|v| \Delta t}{h_i}.
$$

决策规则：
- $\text{Pe} < 2$ 且 $\text{CFL} < 0.5$：中心差分；
- $\text{Pe} \geq 2$ 且 $\text{CFL} < 1$：迎风差分；
- $\text{CFL} \geq 1$：Lax-Wendroff 或时间步长缩减。

### 3.6 参数反演的失配泛函

扩散系数 $\varepsilon$ 的反演目标泛函：

$$
\mathcal{J}(\varepsilon) = \frac{1}{2M} \sum_{j=1}^{M} \big(U(x_j, t_j; \varepsilon) - Y_j^{\text{obs}}\big)^2 + \frac{\alpha_{\text{reg}}}{2} \varepsilon^2.
$$

由于 $\partial U / \partial \varepsilon$ 难以解析求得，采用无梯度 PRAXIS 优化器求解 $\varepsilon^* = \arg\min \mathcal{J}(\varepsilon)$。

---

## 四、运行方式

### 4.1 环境要求
- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7（可选，本项目主要依赖 NumPy）

### 4.2 执行命令

```bash
cd Synthesis-project-python/180_synth_project
python main.py
```

**无需任何命令行参数。** `main.py` 将自动执行以下 5 组实验：

1. **SPDE 单路径求解**：对比 Euler-Maruyama、Platen SRK、Milstein 三种时间积分方法；
2. **蒙特卡洛不确定性量化**：80 样本反变量法 + 分层采样，估计波前位置的统计期望与方差；
3. **无梯度参数标定**：Rosenbrock 基准验证 + SPDE 扩散系数 $\varepsilon$ 的反演；
4. **自适应策略选择**：基于局部 Peclet/CFL 指标的自适应格式决策；
5. **Fibonacci 螺旋与 CVT 网格**：空间采样与网格优化演示。

实验结果将打印到终端，并保存为 `summary_report.txt`。

---

## 五、工程鲁棒性与边界处理

1. **非负物理约束**：Fisher-KPP 解满足 $U \geq 0$，代码中通过 `np.clip(u, 0, K*1.5)` 强制截断；
2. **病态矩阵处理**：带状/隐式矩阵求解时，若条件数 $> 10^{14}$，自动回退到 `np.linalg.lstsq`；
3. **步长正性保护**：自适应步长调整后通过 `h = min(dt, tf - t)` 防止越界；
4. **密度函数下界**：CVT 密度函数 $\rho(x) \geq 0.1$，避免除以零；
5. **Wiener 增量截断**：极小特征值 $q_k < 10^{-15}$ 时数值截断到机器精度级别。

---

## 六、项目创新点与博士级难度体现

1. **多尺度耦合**：同时处理扩散（抛物型）、对流（双曲型）与随机 forcing（Itô 积分）三种截然不同的数学结构；
2. **高阶随机积分**：实现了强阶 1.0 的 Platen SRK 与 Milstein 修正，超越工程上常用的 Euler-Maruyama；
3. **自适应策略**：将棋盘博弈思想转化为数值格式的局部自适应选择，体现了跨学科算法融合；
4. **无梯度反演**：在缺乏伴随方程（adjoint）的条件下，通过 PRAXIS 主方向法实现 SPDE 参数标定；
5. **系统的不确定性量化**：集成蒙特卡洛、方差缩减、Gelman-Rubin 收敛诊断的完整 UQ 流程；
6. **高质量随机数**：基于 L'Ecuyer CMRG 而非默认 Mersenne Twister，保证了大规模并行蒙特卡洛的统计独立性。
