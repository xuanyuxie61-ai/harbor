# 海洋内波破碎与混合参数化综合模拟系统

## PROJECT_57 博士级合成说明

---

## 一、项目概述

本项目围绕**海洋科学：海洋内波破碎与混合参数化**这一前沿科学问题，将15个种子项目的核心算法融合重构为一个多尺度、多物理过程耦合的博士级海洋内波数值模拟系统。

### 科学问题

海洋密度分层中的内波（Internal Waves）是海洋中能量传播的重要载体。当内波在传播过程中因非线性增强、剪切不稳定或地形相互作用而发生破碎时，会触发湍流混合，对海洋热量、盐度和动量的垂向输运产生关键影响。本系统旨在：

1. **模拟内波的非线性生成与传播动力学**
2. **预测内波破碎事件的发生概率与空间分布**
3. **参数化湍流混合过程，量化能量耗散与扩散系数**
4. **追踪内波能量在三维海洋中的最优传播路径**

---

## 二、原项目到科学问题的映射

| 编号 | 原项目 | 核心算法 | 科学映射 | 融入文件 |
|:---:|:---|:---|:---|:---|
| 1 | `776_monomial_symmetrize` | 排列对称化/组合数学 | 波数空间能量谱的对称化处理，保证物理守恒 | `turbulence_parameterization.py` |
| 2 | `322_duffing_ode` | 非线性Duffing振子 | 海洋内波的非线性振荡方程（含阻尼、强迫、浮力修正） | `internal_wave_dynamics.py` |
| 3 | `536_hilbert_curve_3d` | 3D Hilbert空间填充曲线 | 海洋三维分层区域的空间索引与高效数据遍历 | `spatial_indexing.py` |
| 4 | `702_logistic_ode` | Logistic增长ODE | 内波能量的非线性饱和与衰减过程 | `internal_wave_dynamics.py` |
| 5 | `655_leaf_chaos` | IFS迭代函数系统 | 混合斑块的分形空间分布建模 | `monte_carlo_breaking.py` |
| 6 | `271_dg1d_advection` | 1D间断Galerkin对流求解器 | 内波传播方程的高阶谱元数值求解 | `spectral_discretization.py` |
| 7 | `438_flies_simulation` | 圆盘随机距离蒙特卡洛 | 随机相位内波模态叠加的统计模拟 | `monte_carlo_breaking.py` |
| 8 | `696_locker_simulation` | 置换循环搜索策略 | 内波模态间能量交换的置换循环结构分析 | `optimal_path.py` |
| 9 | `244_cvt_1d_lumping` | 1D CVT/Lloyd算法 | 海洋垂向最优采样点分布（密度加权） | `mesh_generation.py` |
| 10 | `287_dijkstra` | Dijkstra最短路径 | 内波能量传播的最优路径搜索 | `optimal_path.py` |
| 11 | `496_haar_transform` | Haar小波变换 | 内波信号的时频分析与破碎事件检测 | `wavelet_analysis.py` |
| 12 | `1416_wishart_matrix` | Wishart矩阵采样 | 雷诺应力张量的随机采样（Bartlett分解） | `turbulence_parameterization.py` |
| 13 | `194_cobweb_plot` | 不动点迭代/蛛网图 | 混合效率不动点方程的迭代求解 | `turbulence_parameterization.py` |
| 14 | `137_casino_simulation` | 乘法随机过程 | 内波能量级联的随机耗散模拟 | `monte_carlo_breaking.py` |
| 15 | `1341_triangulation_order1_display` | 三角网格剖分 | 海洋水平区域的Delaunay三角离散化 | `mesh_generation.py` |

---

## 三、新增数学物理模型与核心公式

### 3.1 海洋物理基础

**密度剖面：**
$$\rho(z) = \rho_0 + \frac{d\rho}{dz} \cdot z$$

**Brunt-Väisälä 浮力频率：**
$$N^2 = -\frac{g}{\rho_0} \frac{d\rho}{dz}, \quad N = \sqrt{N^2}$$

**梯度Richardson数（剪切不稳定判据）：**
$$Ri = \frac{N^2}{(\partial u/\partial z)^2 + (\partial v/\partial z)^2}$$

当 $Ri < 0.25$（Miles-Howard临界值）时，流动发生Kelvin-Helmholtz不稳定，触发内波破碎。

### 3.2 线性内波色散关系

$$\omega^2 = \frac{N^2 k_h^2 + f^2 m^2}{k_h^2 + m^2}$$

其中 $k_h = \sqrt{k_x^2 + k_y^2}$ 为水平波数，$m$ 为垂向波数，$f$ 为科里奥利参数。

**群速度：**
$$c_{gx} = \frac{k_x}{\omega} \frac{(N^2 - f^2)m^2}{(k_h^2 + m^2)^2}, \quad c_{gz} = -\frac{m}{\omega} \frac{(N^2 - f^2)k_h^2}{(k_h^2 + m^2)^2}$$

### 3.3 非线性内波动力学（Duffing型）

将Duffing振子推广至海洋内波场景：

$$\frac{d^2\xi}{dt^2} + \delta \frac{d\xi}{dt} + \alpha_{eff} \xi + \beta \xi^3 = \gamma \cos(\omega t) + F_{coriolis}$$

其中：
- $\alpha_{eff} = \alpha + N^2$（浮力修正的恢复力系数）
- $F_{coriolis} = -f \cdot \dot{\xi}$（科里奥利效应）

**能量方程（Logistic-like衰减）：**
$$\frac{dE}{dt} = rE\left(1 - \frac{E}{E_{max}}\right) - \varepsilon_{diss}$$

其中 $\varepsilon_{diss} = \delta \dot{\xi}^2$ 为机械能耗散率。

### 3.4 Korteweg-de Vries (KdV) 内波方程

$$\frac{\partial \eta}{\partial t} + c \frac{\partial \eta}{\partial x} + \alpha \eta \frac{\partial \eta}{\partial x} + \beta \frac{\partial^3 \eta}{\partial x^3} = 0$$

采用**伪谱法**（FFT-based）求解，结合分裂步法处理线性与非线性项。

### 3.5 间断Galerkin (DG) 谱元方法

内波传播方程的弱形式：
$$\int_{\Omega_k} \frac{\partial u}{\partial t} \phi \, dx - \int_{\Omega_k} a u \frac{\partial \phi}{\partial x} \, dx + \oint_{\partial \Omega_k} \hat{F} \phi \, ds = \int_{\Omega_k} S \phi \, dx$$

**数值通量（Upwind）：**
$$\hat{F} = \frac{a}{2}(u^- + u^+) - \frac{|a|}{2}(u^+ - u^-)$$

**低存储5级RK-45时间积分：**
$$u^{(i)} = \alpha_i u^{(i-1)} + \Delta t \cdot \text{RHS}(u^{(i-1)}, t^{(i-1)})$$

### 3.6 Haar小波时频分析

一级Haar变换：
$$a_i = \frac{v_{2i-1} + v_{2i}}{\sqrt{2}}, \quad d_i = \frac{v_{2i-1} - v_{2i}}{\sqrt{2}}$$

**多尺度能量谱：**
$$E_j = \sum_k |d_{j,k}|^2$$

破碎事件检测基于高频细节系数的阈值判断。

### 3.7 湍流耗散率参数化（Osborn, 1980）

$$\varepsilon = \nu \left|\frac{\partial u}{\partial z}\right|^2 \cdot f(Ri), \quad f(Ri) = \max\left(0, 1 - \frac{Ri}{Ri_c}\right)$$

**垂向涡扩散系数：**
$$K_z = \frac{\Gamma \varepsilon}{N^2}$$

### 3.8 混合效率不动点方程

$$\Gamma = f(\Gamma) = \frac{\Gamma_{max}}{1 + \alpha \cdot Ri \cdot \Gamma}$$

采用蛛网图不动点迭代求解，迭代格式：
$$\Gamma_{n+1} = \frac{\Gamma_{max}}{1 + \alpha \cdot Ri \cdot \Gamma_n}$$

### 3.9 Wishart分布与雷诺应力张量

**Bartlett分解：**
$$W = R^T \cdot AU \cdot R, \quad AU = C^T \cdot C$$

其中 $C$ 为上三角随机矩阵：
- 对角线：$C_{ii} = \sqrt{\chi^2(df - i + 1)}$
- 上三角：$C_{ij} \sim \mathcal{N}(0,1)$ ($i < j$)

雷诺应力张量：
$$\tau_{ij} = -\rho_0 \langle u'_i u'_j \rangle$$

### 3.10 蒙特卡洛内波破碎概率

随机相位叠加模型：
$$u(z,t) = \sum_{m=1}^{M} A_m \sin\left(\frac{m\pi z}{H}\right) \cos(\omega_m t + \theta_m)$$

破碎概率：
$$P_{break} = \frac{1}{N_{realizations}} \sum_{k=1}^{N_{realizations}} \mathbb{1}_{[Ri < 0.25]}^{(k)}$$

### 3.11 能量传播最优路径

图模型 $G = (V, E, w)$：
- 节点 $V$：海洋离散化分层点 $(z_i, x_j)$
- 边权重 $w_{ij} = \Delta x / |c_g|$（传播时间）

Dijkstra算法求解：
$$d[v] = \min_{u \to v} \{d[u] + w(u,v)\}$$

### 3.12 CVT最优垂向离散化

能量泛函：
$$E(\{z_i\}) = \sum_j \int_{V_j} \rho(z) \cdot |z - z_j|^2 \, dz$$

Lloyd迭代：
$$z_j^{(n+1)} = \frac{\int_{V_j^{(n)}} z \cdot \rho(z) \, dz}{\int_{V_j^{(n)}} \rho(z) \, dz}$$

---

## 四、文件结构与功能说明

| 文件名 | 功能 | 核心公式/算法 |
|:---|:---|:---|
| `main.py` | 统一入口，零参数运行 |  orchestrates all modules |
| `ocean_physics.py` | 海洋物理基础参数与公式 | $N(z)$, $Ri$, $\omega(k,m)$, $\varepsilon$ |
| `internal_wave_dynamics.py` | 非线性内波动力学 | Duffing ODE, KdV方程, 波作用量 |
| `spectral_discretization.py` | DG谱元数值求解 | Jacobi多项式, Vandermonde矩阵, RK-45 |
| `wavelet_analysis.py` | Haar小波时频分析 | 多尺度分解, 破碎检测, 能量谱 |
| `spatial_indexing.py` | 3D Hilbert空间索引 | h↔(x,y,z)双向转换, 局部性保持指数 |
| `mesh_generation.py` | 最优空间离散化 | CVT/Lloyd算法, Delaunay三角剖分 |
| `monte_carlo_breaking.py` | 蒙特卡洛破碎模拟 | 随机相位叠加, 乘法随机过程, IFS分形 |
| `optimal_path.py` | 能量传播路径追踪 | Dijkstra算法, 置换循环, 射线追踪 |
| `turbulence_parameterization.py` | 湍流混合参数化 | Wishart采样, 不动点迭代, 波数对称化 |

---

## 五、运行方式

### 环境要求
- Python 3.8+
- NumPy
- SciPy

### 运行命令
```bash
cd 057_synth_project
python main.py
```

### 运行输出
程序自动执行以下9个模块：
1. 海洋物理参数计算
2. 非线性内波动力学
3. DG谱元数值求解
4. Haar小波时频分析
5. 3D Hilbert空间索引
6. CVT与三角网格生成
7. 蒙特卡洛破碎模拟
8. 最优能量传播路径
9. 湍流混合参数化

最终输出综合结果摘要，包含关键物理量的统计信息。

### 运行时间
在普通CPU上约 **10-30秒**。

---

## 六、边界处理与数值鲁棒性

1. **密度/浮力频率**：使用 `np.clip` 限制在物理合理范围
2. **Richardson数**：避免除零（剪切过小时设为大值 $10^6$）
3. **DG求解器**：解限制在 $[-10, 10]$，防止数值爆炸
4. **Hilbert转换**：坐标截断在 $[0, N-1]$ 范围
5. **不动点迭代**：结果限制在 $[0, \Gamma_{max}]$
6. **Cholesky分解**：对非正定矩阵添加正则化扰动
7. **KdV伪谱法**：使用FFT确保周期性边界条件

---

## 七、科学难度说明

本项目涉及以下博士级科学计算内容：

1. **多物理场耦合**：非线性ODE + PDE + 随机过程 + 湍流参数化
2. **高阶数值方法**：间断Galerkin谱元法、伪谱法、低存储RK-45
3. **高级统计方法**：Wishart分布采样、蒙特卡洛模拟、不动点分析
4. **空间优化理论**：CVT能量泛函极小化、Delaunay三角剖分
5. **组合数学应用**：置换循环分解、波数空间对称群
6. **分形几何**：迭代函数系统(IFS)建模混合斑块
7. **图论算法**：Dijkstra最短路径在能量传播中的应用
8. **时频分析**：Haar小波多尺度能量谱分解

---

## 八、代码规范

- 所有代码符合PEP 8规范
- 每个模块包含详细的docstring说明
- 边界条件均有显式处理
- 数值稳定性通过clip和正则化保证
- 随机种子固定（seed=57）保证可复现性
