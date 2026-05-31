# 精馏塔传质与能效优化 — 博士级 Python 科研代码合成项目

## 1. 项目概述

本项目围绕 **化学工程：精馏塔传质与能效优化** 这一前沿博士级科学问题，将 15 个种子科研代码项目的核心算法融合重构为一个完整的、可零参数运行的 Python 科学计算平台。

精馏是化学工程中能耗最大的分离操作之一，占全球工业能耗的约 3%。在多组分非理想体系中，汽液平衡、传质动力学、塔板效率、填料堆积、压力波动及操作不确定性之间存在强耦合。本项目通过高阶数值方法（Jacobi 谱方法、Laguerre-Gauss 求积、Runge-Kutta-Fehlberg 积分、有限差分法、蒙特卡洛采样、Sobol 敏感性分析等），系统性地建立了一套从热力学基础到能效优化的全链条计算框架。

---

## 2. 原项目到科学问题的映射

| 原项目 | 核心算法 | 在合成项目中的角色 |
|:---|:---|:---|
| 420_fermat_factor | 费马因数分解（从 √n 开始搜索接近因数） | **能效优化模块**：将总成本函数类比为"待分解整数"，从 √C 附近开始搜索最优塔板数 N 与回流比 R 的组合 |
| 1260_three_body_ode | 三体问题引力动力学 ODE | **传质动力学模块**：将三体相互作用力映射为三组分 Maxwell-Stefan 扩散方程中的交互阻力项 |
| 945_quad_trapezoid | 复合梯形数值积分 | **物性插值模块**：沿塔高离散节点积分传质通量 |
| 953_quadrilateral_mesh | 四边形 Q4 等参映射 | **塔板几何模块**：将参考单元 [0,1]² 映射到物理塔板，计算局部效率与 Jacobian 面积元 |
| 1071_shepard_interp_1d | 反距离加权 Shepard 插值 | **物性插值模块**：从离散实验数据点插值汽液平衡曲线 |
| 640_laguerre_integrands | Laguerre-Gauss 高斯求积 | **VLE 热力学模块**：计算无穷温度区间上的相平衡积分 |
| 305_dist_plot | 矩形有符号距离函数 | **塔板几何模块**：定义塔板边界，判断节点是否在塔板内部 |
| 1037_rk45 | Runge-Kutta 4/5 阶显式积分 | **传质动力学模块**：求解精馏塔动态物料平衡 ODE、Maxwell-Stefan 扩散、Langford 混合、Lorenz96 对流 |
| 645_langford_ode | Langford 非线性湍流 ODE | **传质动力学模块**：刻画塔板上局部湍流混合的非线性动力学 |
| 682_line_lines_packing | 线段随机填充（Renyi 停车问题） | **填料模拟模块**：模拟填料塔内随机堆积的几何分布，估算空隙率 |
| 607_jacobi_polynomial | Jacobi 多项式递推计算 | **VLE 热力学模块**：在组成区间 [-1,1] 上对活度系数进行谱展开 |
| 703_lorenz96_ode | Lorenz96 混沌对流 ODE | **传质动力学模块**：模拟塔内大尺度对流的混沌特性 |
| 042_asa144 | 随机列联表生成（给定行/列和） | **不确定性量化模块**：生成具有给定总摩尔流量的随机组分分布 |
| 366_fd1d_wave | 一维波动方程有限差分 | **压力波动模块**：模拟塔内压力扰动的轴向传播 |
| 529_hexagon_monte_carlo | 六边形区域蒙特卡洛积分 | **不确定性量化模块**：在六边形操作参数空间内采样，量化参数不确定性对能效的影响 |

---

## 3. 新增数学物理模型与核心公式

### 3.1 多组分汽液平衡（VLE）

低压简化平衡方程：

$$
y_i P = x_i \gamma_i P_i^{\text{sat}}
$$

其中活度系数 $\gamma_i$ 由 Wilson 方程计算：

$$
\ln \gamma_i = 1 - \ln\left(\sum_j x_j \Lambda_{ij}\right) - \sum_k \frac{x_k \Lambda_{ki}}{\sum_j x_j \Lambda_{kj}}
$$

Wilson 参数：

$$
\Lambda_{ij} = \frac{V_j}{V_i} \exp\left(-\frac{\lambda_{ij} - \lambda_{ii}}{RT}\right)
$$

饱和蒸气压由 Antoine 方程给出：

$$
\log_{10} P_i^{\text{sat}} = A_i - \frac{B_i}{T + C_i}
$$

相对挥发度：

$$
\alpha_i = \frac{K_i}{K_{\text{ref}}}, \quad K_i = \frac{\gamma_i P_i^{\text{sat}}}{P}
$$

### 3.2 Jacobi 多项式谱展开

在组成变量 $\xi = 2x - 1 \in [-1,1]$ 上，活度系数可展开为：

$$
\gamma(x) \approx \sum_{n=0}^{N} a_n P_n^{(\alpha,\beta)}(\xi)
$$

Jacobi 多项式满足递推关系：

$$
\begin{aligned}
P_0^{(\alpha,\beta)}(x) &= 1 \\
P_1^{(\alpha,\beta)}(x) &= \frac{(2+\alpha+\beta)x + (\alpha-\beta)}{2} \\
c_1 P_n &= (c_3 + c_2 x) P_{n-1} + c_4 P_{n-2}
\end{aligned}
$$

### 3.3 Laguerre-Gauss 求积

用于无穷温度区间的积分：

$$
\int_0^{+\infty} e^{-x} x^{\alpha} f(x) \, dx \approx \sum_{i=1}^{n} w_i f(x_i)
$$

节点 $x_i$ 为广义 Laguerre 多项式 $L_n^{(\alpha)}(x)$ 的根，权重：

$$
w_i = \frac{\Gamma(\alpha+1) \prod_{j=2}^{n} c_j}{L_n^{(\alpha)\prime}(x_i) \, L_{n-1}^{(\alpha)}(x_i)}
$$

### 3.4 Maxwell-Stefan 扩散方程（三组分）

将三体引力相互作用映射为组分间扩散阻力：

$$
\frac{dx_i}{dt} = \frac{N_i}{c_t}, \qquad
\frac{dN_i}{dt} = c_t \sum_{j \neq i} \frac{x_j - x_i}{(|x_j - x_i| + \varepsilon) D_{ij}} \times 10^{-4}
$$

其中 $\varepsilon = 10^{-3}$ 为饱和项，防止奇点。

### 3.5 精馏塔动态物料平衡

第 $j$ 块塔板、第 $i$ 个组分的物料平衡：

$$
\frac{d(M_j x_{i,j})}{dt} = L_{j-1} x_{i,j-1} + V_{j+1} y_{i,j+1} - L_j x_{i,j} - V_j y_{i,j} + F_j z_{i,j}
$$

实际汽相组成（Murphree 效率）：

$$
y_{i,j} = E_j \, y_{i,j}^{*} + (1 - E_j) \, y_{i,j+1}, \qquad y_{i,j}^{*} = K_{i,j} x_{i,j}
$$

边界处理（投影步）：每步积分后对每块板的组成进行裁剪与归一化：

$$
x_{i,j} \leftarrow \frac{\max(x_{i,j}, 10^{-12})}{\sum_k \max(x_{k,j}, 10^{-12})}
$$

### 3.6 四边形 Q4 等参映射

参考单元 $(R,S) \in [0,1]^2$ 到物理单元的映射：

$$
\begin{aligned}
X(R,S) &= \sum_{i=1}^{4} N_i(R,S) X_i \\
Y(R,S) &= \sum_{i=1}^{4} N_i(R,S) Y_i
\end{aligned}
$$

形函数：

$$
\begin{aligned}
N_1 &= (1-R)(1-S) \\
N_2 &= R(1-S) \\
N_3 &= RS \\
N_4 &= (1-R)S
\end{aligned}
$$

Jacobian 行列式给出局部面积微元：

$$
|J| = \left|\frac{\partial(X,Y)}{\partial(R,S)}\right|
$$

### 3.7 矩形有符号距离函数

$$
d(p) = -\min\left(-x_1 + p_x,\; x_2 - p_x,\; -y_1 + p_y,\; y_2 - p_y\right)
$$

内部 $d < 0$，边界 $d = 0$，外部 $d > 0$。

### 3.8 塔板局部 Murphree 效率

$$
E_{OG} = \frac{y_{\text{out}} - y_{\text{in}}}{y^{*} - y_{\text{in}}}
$$

面积加权平均效率：

$$
\bar{E} = \frac{\sum_e A_e E_e}{\sum_e A_e}
$$

### 3.9 Runge-Kutta-Fehlberg（RK45）积分

每步计算 4 阶和 5 阶近似，误差估计：

$$
e_{n+1} = |y_{n+1}^{(5)} - y_{n+1}^{(4)}|
$$

Butcher 表系数（经典 RKF）：

$$
\begin{array}{c|cccccc}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1/4 & 1/4 & 0 & 0 & 0 & 0 & 0 \\
3/8 & 3/32 & 9/32 & 0 & 0 & 0 & 0 \\
12/13 & 1932/2197 & -7200/2197 & 7296/2197 & 0 & 0 & 0 \\
1 & 439/216 & -8 & 3680/513 & -845/4104 & 0 & 0 \\
1/2 & -8/27 & 2 & -3544/2565 & 1859/4104 & -11/40 & 0 \\
\hline
& 16/135 & 0 & 6656/12825 & 28561/56430 & -9/50 & 2/55 \\
& 25/216 & 0 & 1408/2565 & 2197/4104 & -1/5 & 0 \\
\end{array}
$$

### 3.10 Langford 局部湍流混合模型

$$
\begin{aligned}
\frac{dx}{dt} &= (z - b)x - dy \\
\frac{dy}{dt} &= dx + (z - b)y \\
\frac{dz}{dt} &= c + az - \frac{z^3}{3} - (x^2 + y^2)(1 + ez) + fzx^3
\end{aligned}
$$

参数取 $a=3.0, b=1.5, c=1.0, d=0.5, e=0.2, f=0.1$。

### 3.11 Lorenz96 大尺度对流混沌模型

$$
\frac{dy_i}{dt} = (y_{i+1} - y_{i-2}) y_{i-1} - y_i + F
$$

其中 $i$ 为环状索引（对应塔板高度离散），$F=8.0$ 为外力参数。

### 3.12 类费马分解优化算法

总成本函数 $C(N,R)$ 的最小化类比于费马因数分解：

$$
C(N,R) \approx N \cdot R \quad \Rightarrow \quad \text{从 } \sqrt{C_{\max}} \text{ 附近搜索最优 } (N,R)
$$

Gilliland 关联验证优化结果的一致性：

$$
X = \frac{R - R_{\min}}{R + 1}, \quad
Y = \frac{N - N_{\min}}{N + 1}
$$

$$
Y = 1 - \exp\left[\frac{(1 + 54.4X)(X - 1)}{11 + 117.2X} \sqrt{X}\right]
$$

### 3.13 填料塔 Ergun 压降方程

$$
\frac{\Delta P}{L} = 150 \frac{(1-\varepsilon)^2 \mu u}{\varepsilon^3 d_p^2} + 1.75 \frac{(1-\varepsilon) \rho u^2}{\varepsilon^3 d_p}
$$

空隙率由随机堆积密度估算：

$$
\varepsilon = 1 - \frac{n_{\text{packing}} V_{\text{single}}}{V_{\text{column}}}
$$

### 3.14 一维波动方程（压力传播）

$$
c^2 \frac{\partial^2 P}{\partial z^2} = \frac{\partial^2 P}{\partial t^2}
$$

有限差分格式（CFL 条件 $|\alpha| = |c \Delta t / \Delta z| \le 1$）：

$$
\begin{aligned}
P_j^{n+1} &= 2(1 - \alpha^2) P_j^n + \alpha^2 (P_{j+1}^n + P_{j-1}^n) - P_j^{n-1} \quad (n \ge 2) \\
P_j^{1} &= \frac{\alpha^2}{2} P_{j+1}^0 + (1 - \alpha^2) P_j^0 + \frac{\alpha^2}{2} P_{j-1}^0 + \Delta t \, P_t^0
\end{aligned}
$$

### 3.15 Sobol 一阶敏感性指标

$$
S_i = \frac{\mathbb{V}_{X_i}\bigl(\mathbb{E}_{X_{\sim i}}(Y \mid X_i)\bigr)}{\mathbb{V}(Y)}
$$

使用 Saltelli 蒙特卡洛估计：

$$
S_i \approx \frac{1}{N} \sum_{j=1}^{N} f(B_j) \bigl(f(A_B^{(i)})_j - f(A_j)\bigr) \Big/ \hat{\mathbb{V}}(Y)
$$

### 3.16 Shepard 反距离加权插值

$$
f(x) = \sum_{i=1}^{N} w_i(x) f_i, \qquad w_i(x) = \frac{|x - x_i|^{-p}}{\sum_j |x - x_j|^{-p}}
$$

当 $x = x_i$ 时，$w_i = 1$，其余为 0，保证精确插值。

### 3.17 梯形数值积分

$$
\int_a^b f(z) \, dz \approx \frac{h}{2} \left[ f(z_0) + 2\sum_{i=1}^{n-1} f(z_i) + f(z_n) \right], \quad h = \frac{b-a}{n}
$$

---

## 4. 项目文件结构

```
132_synth_project/
├── main.py                         # 统一入口，零参数运行
├── utils.py                        # 通用辅助函数（安全除法、边界检查等）
├── vle_thermodynamics.py           # VLE 热力学：Jacobi 多项式 + Laguerre 求积 + Wilson 方程
├── property_interpolation.py       # 物性插值：Shepard 插值 + 梯形积分
├── tray_geometry_mesh.py           # 塔板几何：四边形 Q4 映射 + 有符号距离函数 + 局部效率
├── mass_transfer_dynamics.py       # 传质动力学：RK45 + Maxwell-Stefan + Langford + Lorenz96 + 精馏塔动态
├── efficiency_optimizer.py         # 能效优化：类费马分解搜索 + Gilliland 关联 + 成本模型
├── packing_simulation.py           # 填料模拟：线段随机填充 + Ergun 压降 + 空隙率
├── uncertainty_quantification.py   # 不确定性量化：六边形蒙特卡洛 + 随机列联表 + Sobol 分析
├── pressure_wave_dynamics.py       # 压力波动：一维波动方程有限差分
└── README_博士级合成说明.md        # 本文档
```

---

## 5. 合成后的项目能够解决什么科学问题

1. **多组分非理想体系汽液平衡预测**：基于 Wilson 方程与 Jacobi 谱展开，计算活度系数与相平衡常数，适用于乙醇-水-甲醇等强非理想体系。
2. **塔板局部传质效率评估**：通过四边形网格离散塔板表面，计算各单元的局部 Murphree 点效率，并给出面积加权平均效率。
3. **精馏塔动态行为模拟**：求解刚性物料平衡 ODE 系统，预测开工、负荷变动、进料扰动等瞬态过程中的组成分布演化。
4. **Maxwell-Stefan 扩散过程分析**：模拟三组分体系中组分间交互扩散的动力学过程，揭示非理想扩散对传质速率的非线性影响。
5. **塔内混沌混合与对流不稳定性**：利用 Langford 和 Lorenz96 模型分别刻画局部湍流混合与大尺度对流的混沌特性。
6. **理论塔板数与回流比耦合优化**：基于类费马分解的全局搜索与 Gilliland 关联验证，寻找年度总成本最小的操作-设计组合。
7. **填料塔堆积特性与压降预测**：通过随机线段填充模拟填料几何分布，结合 Ergun 方程预测压降与传质效率。
8. **操作参数不确定性量化**：使用六边形区域蒙特卡洛积分、随机列联表生成和 Sobol 敏感性分析，评估温度、压力、回流比等参数波动对能效的影响。
9. **压力波动传播与稳定性分析**：基于一维波动方程有限差分，模拟塔内压力扰动的传播过程，评估操作稳定性。

---

## 6. 运行方式

### 环境要求
- Python 3.8+
- NumPy

### 运行命令
```bash
cd 132_synth_project
python main.py
```

程序将自动执行全部 8 个科学计算模块，输出各模块的关键结果到控制台，并在最后给出计算结果汇总。整个运行过程约需 5–10 秒（取决于 CPU 性能），无需任何用户输入参数。

---

## 7. 质量检查清单

- [x] 原目录（`all_code/` 下各项目）未被修改
- [x] 合成后的项目为 Python 语言
- [x] 新目录完整包含合成后的项目（10 个文件）
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] 15 个输入项目均已真实融入合成项目，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性（组成裁剪、CFL 条件检查、除零保护等）
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 无可视化相关内容
