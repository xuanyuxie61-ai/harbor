# 中尺度对流系统数值预报综合模拟系统 — 博士级合成说明

## 项目概述

本项目基于 **15 个科研代码种子项目** 的核心算法, 围绕 **大气科学: 中尺度对流系统 (Mesoscale Convective System, MCS) 数值预报** 这一前沿领域, 融合构建了一个完整的博士级自然科学计算系统。

中尺度对流系统是引发暴雨、冰雹、大风和龙卷等灾害性天气的重要天气系统, 其数值预报涉及热力学、流体力学、微物理学、不确定性量化与观测网络优化等多学科交叉问题。本系统集成了从探空资料前处理、对流动力学积分、滞弹性压力求解、微物理参数化、不确定性量化到观测网络优化的完整预报链条。

---

## 一、原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 在 MCS 预报系统中的角色 |
|:---:|:---|:---|:---|
| 1 | **1137_spquad** | Clenshaw-Curtis 稀疏网格求积 (Smolyak 构造) | 高维参数空间不确定性量化 (UQ), 用于集合预报参数扰动的高效采样 |
| 2 | **1270_toms443** | Lambert W 函数 | 求解 Clausius-Clapeyron 方程的隐式反演, 用于饱和水汽压计算 |
| 3 | **213_contour_gradient_3d** | 3D 梯度计算 | 水汽通量辐合 (MFC) 诊断、散度计算、Laplacian 模板 |
| 4 | **1134_spiral_pde** | Barkley 型反应-扩散 PDE (9 点 Laplacian) | 对流单体触发与传播的激发介质动力学模拟 |
| 5 | **665_legendre_rule** | Gauss-Legendre 求积规则 (Golub-Welsch) | 垂直方向高精度积分 (可降水量、CAPE/CIN 精确计算) |
| 6 | **998_r8st** | 稀疏矩阵 COO 格式 + CG/Jacobi 迭代 | 滞弹性压力方程的大型稀疏线性系统求解 |
| 7 | **1257_tetrahedron01_monte_carlo** | 单位四面体蒙特卡洛积分 | 三维降水体积的蒙特卡洛估算 |
| 8 | **1269_toms291** | log-Gamma 函数 (Stirling 渐近) | 微物理 Gamma 分布雨滴谱的参数计算 |
| 9 | **1315_triangle_svg** | 三角形几何 (面积、重心、重心坐标) | 地表三角形网格上的感热/潜热通量面积分 |
| 10 | **558_hypercube_grid** | 多维张量积网格生成 | 集合预报参数空间的结构化采样 |
| 11 | **1427_zero_brent** | Brent 法求根 (二分+割线+反二次插值) | 精确确定 LCL/LFC/EL 等关键高度 |
| 12 | **792_nearest_interp_1d** | 一维最近邻插值 | 探空廓线的快速垂直插值 |
| 13 | **642_laguerre_product** | Laguerre 多项式 + 广义 Polynomial Chaos | 微物理参数不确定性的随机 Galerkin 展开 |
| 14 | **807_nonlin_fixed_point** | 不动点迭代 + Newton-Raphson | 饱和调整的非线性热力学平衡求解 |
| 15 | **146_ccvt_reflect** | 约束重心 Voronoi 镶嵌 (Lloyd 迭代) | 最优雷达/气象站观测网络的空间布局优化 |

---

## 二、新增数学物理模型与核心公式

### 2.1 大气热力学核心

#### Clausius-Clapeyron 饱和水汽压方程

$$ e_s(T) = e_0 \exp\left[ \frac{L_v}{R_v} \left( \frac{1}{T_0} - \frac{1}{T} \right) \right] $$

其中 $e_0 = 611.2\,\text{Pa}$, $L_v = 2.501 \times 10^6\,\text{J/kg}$, $R_v = 461.51\,\text{J/(kg·K)}$, $T_0 = 273.15\,\text{K}$。

#### 潜在温度

$$ \theta = T \left( \frac{p_0}{p} \right)^{R_d/c_p} $$

#### 虚温

$$ T_v = T(1 + 0.608 q_v) $$

#### CAPE (对流有效位能) 与 CIN (对流抑制能量)

$$ \text{CAPE} = \int_{z_{\text{LFC}}}^{z_{\text{EL}}} g \frac{T_{v,\text{parcel}} - T_{v,\text{env}}}{T_{v,\text{env}}} \, dz $$

$$ \text{CIN} = \int_{0}^{z_{\text{LFC}}} g \frac{T_{v,\text{parcel}} - T_{v,\text{env}}}{T_{v,\text{env}}} \, dz $$

#### 湿绝热过程 (伪绝热近似)

$$ \frac{dT}{dp} = \frac{1}{p} \frac{R_d T + L_v q_s}{c_p + \frac{L_v^2 q_s \varepsilon}{R_v T^2}} $$

#### 饱和调整 (不动点迭代)

热力学约束方程组:
$$ \begin{cases} q_{\text{total}} = q_v + q_l \\ q_v = q_{\text{sat}}(T) = \frac{\varepsilon e_s(T)}{p - (1-\varepsilon)e_s(T)} \\ T_{\text{new}} = T_{\text{old}} + \frac{L_v}{c_p}(q_{\text{sat}}(T) - q_{\text{total}}) \end{cases} $$

通过不动点映射 $T = g(T)$ 迭代求解, 并引入 Aitken $\Delta^2$ 加速:
$$ x_{k+1}^{\text{accel}} = x_{k+1} - \frac{(x_{k+1} - x_k)^2}{x_{k+1} - 2x_k + x_{k-1}} $$

### 2.2 Lambert W 函数 (隐式热力学反演)

求解 $W(z) e^{W(z)} = z$ 的主分支 $W_0(z)$, 使用分段初始猜测 + Halley 迭代:

- 近分支点 ($z \approx -1/e$): 级数展开 $W \approx -1 + p - p^2/3 + 11p^3/72$
- 小 $z$: Padé 近似 $W \approx z(1 + z(1 + 3z/2))/(1 + 5z/2)$
- 大 $z$: 对数渐近 $W \approx L_1 - L_2 + L_2/L_1 + \cdots$

其中 $L_1 = \ln z$, $L_2 = \ln L_1$。

### 2.3 垂直方向 Gauss-Legendre 高精度积分

通过坐标变换将气压坐标 $p \in [p_{\text{top}}, p_{\text{sfc}}]$ 映射到标准区间 $[-1, 1]$:

$$ p(\xi) = \frac{p_{\text{sfc}} - p_{\text{top}}}{2}\xi + \frac{p_{\text{sfc}} + p_{\text{top}}}{2}, \quad \xi \in [-1, 1] $$

可降水量:
$$ \text{PW} = \frac{1}{g} \int_{p_{\text{top}}}^{p_{\text{sfc}}} q_v(p) \, dp \approx \frac{1}{g} \cdot \frac{p_{\text{sfc}} - p_{\text{top}}}{2} \sum_{i=1}^{n} w_i \, q_v(p(\xi_i)) $$

Gauss-Legendre 节点与权重通过 **Golub-Welsch 算法** 生成: 对称三对角 Jacobi 矩阵的特征值给出节点, 特征向量第一分量的平方给出权重。

### 2.4 对流动力学: Barkley 型反应-扩散系统

将 MCS 中的对流单体抽象为激发介质中的螺旋波/行波, 控制方程为:

$$ \frac{\partial U}{\partial t} = D_u \nabla^2 U + \frac{1}{\varepsilon} U(1-U)\left(U - \frac{V+\beta}{\alpha}\right) $$

$$ \frac{\partial V}{\partial t} = D_v \nabla^2 V + \delta \nabla^2 V + U - V $$

- $U \in [0,1]$: 归一化对流强度 (快变量/activator)
- $V \in [0,1]$: 归一化环境湿度/不稳定度积累 (慢变量/inhibitor)
- $\varepsilon = 0.002$: 时间尺度分离参数
- 9 点高阶 Laplacian 模板 (周期性边界):

$$ \nabla^2 A \approx \frac{1}{6\Delta x^2} \begin{bmatrix} 1 & 4 & 1 \\ 4 & -20 & 4 \\ 1 & 4 & 1 \end{bmatrix} \star A $$

时间积分采用 **四阶 Runge-Kutta**:
$$ \mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4) $$

### 2.5 滞弹性压力方程与稀疏 CG 求解

滞弹性近似下的压力方程为椭圆型:
$$ \nabla \cdot (\rho_0 \nabla \phi) = \nabla \cdot (\rho_0 B \hat{k}) + 2J(u,v) $$

离散化后得到大型稀疏线性系统 $A\mathbf{p} = \mathbf{b}$, 其中 $A$ 为 SPD 矩阵。使用 **共轭梯度法 (CG)** 求解:

$$ \begin{aligned} &\mathbf{r}_0 = \mathbf{b} - A\mathbf{x}_0, \quad \mathbf{p}_0 = \mathbf{r}_0 \\ &\alpha_k = \frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{p}_k^T A \mathbf{p}_k}, \quad \mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k \\ &\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k A \mathbf{p}_k, \quad \beta_k = \frac{\mathbf{r}_{k+1}^T \mathbf{r}_{k+1}}{\mathbf{r}_k^T \mathbf{r}_k} \\ &\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k \end{aligned} $$

若 CG 不收敛, 自动回退到 **Jacobi 迭代** 作为鲁棒后备方案。

### 2.6 水汽通量辐合 (Moisture Flux Convergence)

三维水汽通量向量: $\mathbf{F} = \rho q_v \mathbf{V}$

水汽通量辐合:
$$ -\nabla \cdot \mathbf{F} = -\left[ \frac{\partial(\rho q_v u)}{\partial x} + \frac{\partial(\rho q_v v)}{\partial y} + \frac{\partial(\rho q_v w)}{\partial z} \right] $$

使用二阶中心差分计算梯度:
$$ \frac{\partial f}{\partial x} \approx \frac{f_{i+1} - f_{i-1}}{2\Delta x} $$

边界处自动降级为一阶前/后向差分, 并进行边界值截断保护。

### 2.7 随机微物理参数化: Laguerre Polynomial Chaos

微物理参数 (如凝结时间尺度 $\tau$) 存在不确定性, 使用广义 Polynomial Chaos 展开:

$$ q(\xi) = \sum_{i=0}^{P} q_i L_i(\xi) $$

其中 $\xi$ 为服从指数分布的随机变量, $L_i(\xi)$ 为 **Laguerre 多项式**, 满足三项递推:

$$ (n+1)L_{n+1}(x) = (2n+1-x)L_n(x) - nL_{n-1}(x) $$

正交性:
$$ \int_0^{\infty} L_i(x) L_j(x) e^{-x} \, dx = \delta_{ij} $$

Gauss-Laguerre 求积节点通过 Newton 迭代精化 Laguerre 多项式的根获得, 权重为:
$$ w_i = \frac{1}{n \cdot [L_{n-1}(x_i)]^2} $$

Gamma 分布雨滴谱 (Khrgian-Mazin 谱):
$$ N(D) = N_0 D^{\mu} \exp(-\Lambda D) $$

第 $k$ 阶矩:
$$ M_k = \int_0^{\infty} D^k N(D) \, dD = \frac{N_0 \, \Gamma(k+\mu+1)}{\Lambda^{k+\mu+1}} $$

### 2.8 稀疏网格不确定性量化 (Smolyak 构造)

对 $d$ 维参数空间的全张量积求积需要 $N^d$ 个节点, 而稀疏网格仅需 $O(N(\log N)^{d-1})$:

$$ A(q,d) = \sum_{|\mathbf{i}|_1 \leq q} (\Delta^{i_1} \otimes \cdots \otimes \Delta^{i_d}) $$

其中 $\Delta^i = Q^i - Q^{i-1}$ 为差分求积算子。1D 基规则采用 **Clenshaw-Curtis**:
- 节点: $x_k = \cos(k\pi/n)$, $k = 0, \ldots, n$
- 权重通过 Waldvogel 快速算法 (DCT 相关) 计算

预报响应的统计量:
$$ \mathbb{E}[f] = \sum_{i} w_i \, f(\mathbf{x}_i), \quad \text{Var}[f] = \sum_i w_i f(\mathbf{x}_i)^2 - (\mathbb{E}[f])^2 $$

### 2.9 四面体蒙特卡洛降水体积估算

标准单位四面体 (顶点 $(0,0,0),(1,0,0),(0,1,0),(0,0,1)$) 上的精确单项式积分:

$$ \int_{T} x^{e_1} y^{e_2} z^{e_3} \, dV = \frac{e_1! \, e_2! \, e_3!}{(e_1+e_2+e_3+3)!} $$

随机点生成采用指数分布归一化法: 若 $E_1, E_2, E_3 \sim \text{Exp}(1)$, 令 $S = E_1+E_2+E_3$, 则:

$$ (x,y,z) = \left( \frac{E_1}{S}, \frac{E_2}{S}, \frac{E_3}{S} \right) $$

在物理四面体上的映射:
$$ \mathbf{x}_{\text{phys}} = \mathbf{v}_0 + J \, \boldsymbol{\xi}, \quad J = [\mathbf{v}_1-\mathbf{v}_0, \, \mathbf{v}_2-\mathbf{v}_0, \, \mathbf{v}_3-\mathbf{v}_0] $$

体积: $V = |\det J| / 6$。

### 2.10 地表通量三角形面积分

三角形面积 (叉积公式):
$$ A = \frac{1}{2} \| (\mathbf{v}_1 - \mathbf{v}_0) \times (\mathbf{v}_2 - \mathbf{v}_0) \| $$

重心坐标:
$$ \lambda_0 = \frac{A(\mathbf{p},\mathbf{v}_1,\mathbf{v}_2)}{A(\mathbf{v}_0,\mathbf{v}_1,\mathbf{v}_2)}, \quad \lambda_1 = \frac{A(\mathbf{v}_0,\mathbf{p},\mathbf{v}_2)}{A(\mathbf{v}_0,\mathbf{v}_1,\mathbf{v}_2)}, \quad \lambda_2 = \frac{A(\mathbf{v}_0,\mathbf{v}_1,\mathbf{p})}{A(\mathbf{v}_0,\mathbf{v}_1,\mathbf{v}_2)} $$

感热通量:
$$ H = \rho c_p C_d |\mathbf{V}| (T_{\text{sfc}} - T_{\text{air}}) $$

潜热通量:
$$ LE = \rho L_v C_d |\mathbf{V}| (q_{\text{sfc}} - q_{\text{air}}) $$

### 2.11 最优观测网络: 约束重心 Voronoi 镶嵌 (CCVT)

CVT 能量:
$$ E = \sum_{i=1}^{n} \int_{V_i} \rho(\mathbf{x}) \|\mathbf{x} - \mathbf{g}_i\|^2 \, d\mathbf{x} $$

Lloyd 迭代 (Monte Carlo 近似):
1. 在区域内采样大量点 $\{\mathbf{s}_k\}$
2. 为每个样本找到最近生成子: $i^*(k) = \arg\min_j \|\mathbf{g}_j - \mathbf{s}_k\|^2$
3. 更新生成子为 Voronoi 单元的质心: $\mathbf{g}_i^{\text{new}} = \frac{\sum_{k \in V_i} \mathbf{s}_k}{|V_i|}$
4. 对越界生成子进行反射处理: $g > g_{\max} \Rightarrow g' = 2g_{\max} - g$
5. 重复直至能量收敛

---

## 三、项目文件结构与修改说明

```
058_synth_project/
├── main.py                          # 统一入口, 零参数运行
├── thermodynamics.py                # 热力学核心 (种子 1270, 1269, 1427, 807)
├── vertical_quadrature.py           # Gauss-Legendre 垂直积分 (种子 665)
├── moisture_flux.py                 # 水汽通量辐合与梯度 (种子 213, 1134)
├── convection_dynamics.py           # 反应-扩散对流动力学 (种子 1134)
├── anelastic_solver.py              # 滞弹性压力稀疏 CG 求解 (种子 998)
├── microphysics.py                  # Laguerre 混沌微物理 (种子 642, 1269)
├── uncertainty_quantification.py    # 稀疏网格 UQ (种子 1137)
├── ensemble_generator.py            # 集合采样与插值 (种子 558, 792)
├── precipitation_estimator.py       # 四面体 MC 降水估算 (种子 1257)
├── surface_geometry.py              # 三角形通量积分 (种子 1315)
├── observation_optimizer.py         # CCVT 观测网络优化 (种子 146)
└── README_博士级合成说明.md          # 本文档
```

### 各文件改造要点

1. **thermodynamics.py**: 融合 Lambert W 函数 (toms443)、log-Gamma (toms291)、Brent 求根 (zero_brent) 和不动点迭代 (nonlin_fixed_point), 实现完整的大气热力学诊断链。
2. **vertical_quadrature.py**: 将 IQPACK/Golub-Welsch 算法 (legendre_rule) 应用于气压坐标的垂直积分, 替代粗糙的梯形法。
3. **moisture_flux.py**: 去除 3D 可视化的 contour/surf/quiver, 保留纯数值梯度计算 (gradient_2d_centered, gradient_3d_centered), 并加入 9 点 Laplacian 模板 (来自 spiral_pde)。
4. **convection_dynamics.py**: 将 torus 上的 spiral PDE 改造为 MCS 对流动力学模型, 使用 RK4 时间积分替代原 MATLAB ode45 接口。
5. **anelastic_solver.py**: 将 MATLAB 稀疏矩阵工具箱 (r8st) 翻译为 Python COO 格式, 实现 CG 和 Jacobi 双求解器。
6. **microphysics.py**: 将 Laguerre 多项式内积 (laguerre_product) 扩展为完整的 Polynomial Chaos 微物理参数化框架, 结合 log-Gamma 计算雨滴谱矩。
7. **uncertainty_quantification.py**: 将 1D/2D 稀疏网格 (spquad) 扩展为任意维度的 Smolyak 构造, 用于集合预报参数不确定性传播。
8. **ensemble_generator.py**: 结合 hypercube_grid 的张量积采样与 nearest_interp_1d 的快速插值, 构建集合预报参数生成与探空廓线插值系统。
9. **precipitation_estimator.py**: 将 tetrahedron01_monte_carlo 的积分方法应用于三维降水体积估算, 支持物理四面体的随机采样。
10. **surface_geometry.py**: 去除 triangle_svg 的可视化部分, 保留三角形面积、重心坐标和面积分数学, 用于地表能量平衡计算。
11. **observation_optimizer.py**: 将 ccvt_reflect 的 Lloyd 迭代移植到 Python, 优化雷达和自动气象站的空间布局。

---

## 四、合成后的项目能够解决什么科学问题

本项目构建了一个端到端的 **中尺度对流系统数值预报原型系统**, 能够解决以下科学问题:

1. **对流触发条件诊断**: 基于探空资料自动计算 CAPE、CIN、LCL、LFC、EL 等关键对流指标, 使用 Brent 求根法精确定位自由对流高度和平衡高度。

2. **对流动力学预报**: 通过 Barkley 型反应-扩散系统模拟对流单体的触发、传播和组织化过程, 适用于 MCS 冷池传播和线状对流形成的研究。

3. **滞弹性压力场重建**: 求解简化的滞弹性压力方程, 为三维速度场提供无辐散约束, 是中尺度模式动力核心的基础组件。

4. **水汽输送诊断**: 计算水汽通量辐合场, 识别 MCS 发展过程中的水汽汇聚区域, 这是暴雨预报的关键诊断量。

5. **微物理过程参数化**: 使用 Laguerre Polynomial Chaos 量化微物理参数不确定性, 计算凝结率和降水率, 支持 Gamma 分布雨滴谱分析。

6. **集合预报不确定性量化**: 通过稀疏网格求积高效传播多参数不确定性, 给出预报量 (如 CAPE) 的均值和标准差, 为概率预报提供支撑。

7. **降水体积估算**: 使用四面体蒙特卡洛积分估算三维空间内的总降水量, 适用于雷达-模式同化中的降水检验。

8. **地表能量平衡**: 在三角形网格上积分感热和潜热通量, 计算下边界对 MCS 发展的热力学强迫。

9. **观测网络优化**: 使用 CCVT 算法优化雷达和气象站的空间布局, 为观测系统模拟试验 (OSSE) 提供理论指导。

---

## 五、如何运行

### 环境要求
- Python 3.8+
- NumPy

### 运行方式
```bash
cd Synthesis-project-python/058_synth_project
python main.py
```

程序将自动执行以下 11 个阶段, 无需任何输入参数:
1. 探空数据前处理与热力学诊断
2. 集合预报参数采样
3. 对流动力学积分
4. 滞弹性压力方程求解
5. 水汽通量辐合诊断
6. 随机微物理参数化
7. 稀疏网格不确定性量化
8. 降水体积蒙特卡洛估算
9. 地表通量面积分
10. 观测网络优化
11. 综合诊断与数值验证

### 预期输出
程序将输出各阶段的物理诊断量 (CAPE、CIN、PW、降水率、通量等) 以及数值验证结果 (Gauss-Legendre 积分误差、Lambert W 精度、log-Gamma 精度), 最后报告总运行时间。

---

## 六、数值鲁棒性与边界处理

本项目在多处实现了严格的边界处理和数值鲁棒性保护:

- **热力学模块**: 温度截断在 [150 K, 350 K], 比湿截断在 [0, 0.05], 水汽压非负保护; Lambert W 的分段初始猜测避免 `log(log(x))` 奇点; 饱和调整的不动点迭代带 Aitken 加速和发散保护。
- **梯度/散度模块**: 边界处自动降级为一阶差分; 所有梯度输出通过 `np.where(np.isfinite(...), ...)` 清除 NaN/Inf。
- **稀疏求解器**: CG 迭代带零除保护; 若 CG 不收敛自动回退到 Jacobi 迭代; 压力场零均值规范化处理 Neumann 边界的奇异性。
- **微物理模块**: 降水率上限截断 (100 mm/hr); Gamma 分布矩计算中 `lam <= 0` 的保护返回; log-Gamma 对非正参数返回 `-inf`。
- **四面体采样**: 指数采样和为零时的回退机制; 物理四面体体积小于阈值时直接返回 0。
- **CCVT 优化**: 空 Voronoi 单元的回退处理; 反射边界后的二次截断确保生成子始终在域内。

---

## 七、科学复杂度说明

本项目的科学计算复杂度达到博士级, 体现在:

1. **多物理场耦合**: 同时求解热力学、流体力学 (滞弹性)、微物理和随机偏微分方程。
2. **高维不确定性量化**: 3 维参数空间上的 Smolyak 稀疏网格, 避免了全张量积的维度灾难。
3. **特殊函数与隐式方程**: Lambert W、log-Gamma、Brent 求根、不动点迭代等非初等函数和方程的数值求解。
4. **稀疏线性代数**: 大规模稀疏矩阵的 CG 迭代求解, 是气象模式动力核心的关键算法。
5. **正交多项式理论**: Laguerre 多项式的三项递推、根求取、Gauss 求积和广义 Fourier 展开。
6. **变分最优设计**: CVT 能量最小化的 Lloyd 迭代, 属于计算几何与最优控制的交叉。
7. **误差控制**: 多重数值验证机制 (Gauss-Legendre 积分精度、特殊函数参考值比对) 确保结果可信。
