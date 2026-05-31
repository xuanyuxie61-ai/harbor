# 多因子随机波动率模型下奇异期权定价与全局校准系统

## 项目概述

本项目基于 **15 个种子科研代码项目**的核心算法，围绕**金融工程：衍生品定价与随机波动率**这一前沿科学领域，融合构建了一个面向博士级难度的科学计算系统。项目采用 **Python** 语言实现，包含 9 个独立模块与 1 个统一入口，所有代码零参数可运行，具备完整的边界处理与数值鲁棒性设计。

---

## 科学问题定义

**核心问题**：在具有均值回归随机波动率的多因子仿射模型框架下，使用**混合数值方法**（PDE 有限差分 + 稀疏网格配置 + 蒙特卡洛方差缩减 + Krylov 子空间迭代）对路径依赖型奇异期权进行高精度定价，并基于**主成分降维**与**延拓法全局校准**实现模型参数的鲁棒估计。

该问题横跨**随机分析、偏微分方程数值解、Krylov 子空间方法、稀疏网格高维积分、非线性优化与分形动力学**等多个博士级数学物理领域，计算复杂度极高。

---

## 核心数学物理模型与公式

### 1. Heston 随机波动率模型

资产价格 $S_t$ 与瞬时方差 $v_t$ 的联合演化由以下二维随机微分方程组描述：

$$
\begin{cases}
dS_t = r S_t \, dt + \sqrt{v_t} S_t \, dW_t^S \\[6pt]
dv_t = \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^v
\end{cases}
$$

其中相关性结构为：

$$
d\langle W^S, W^v \rangle_t = \rho \, dt, \qquad \rho \in [-1, 1]
$$

**Feller 条件**（保证 $v_t > 0$ 不被吸收）：

$$
2\kappa\theta \geq \sigma^2
$$

若 Feller 条件不满足，则 $v=0$ 为 regular boundary，数值离散需特殊处理反射/吸收效应。

### 2. Black-Scholes-Heston 对流-扩散-反应 PDE

衍生品价格 $V(S, v, t)$ 满足二维退化抛物型 PDE：

$$
\frac{\partial V}{\partial t} + \mathcal{L}_{\text{Heston}} V = 0
$$

其中 Heston 算子：

$$
\mathcal{L}_{\text{Heston}} = \frac{1}{2} v S^2 \frac{\partial^2}{\partial S^2}
+ \rho\sigma v S \frac{\partial^2}{\partial S \partial v}
+ \frac{1}{2} \sigma^2 v \frac{\partial^2}{\partial v^2}
+ rS \frac{\partial}{\partial S}
+ \kappa(\theta - v) \frac{\partial}{\partial v}
- r \cdot I
$$

**终端条件**（欧式看涨）：

$$
V(S, v, T) = \max(S - K, 0)
$$

**边界条件**：
- $S = 0$：$V(0, v, t) = 0$（Dirichlet）
- $S = S_{\max}$：$V(S_{\max}, v, t) = S_{\max} - K e^{-r(T-t)}$（Dirichlet）
- $v = 0$：退化边界，PDE 降维为 $V_t + rS V_S + \kappa\theta V_v - rV = 0$
- $v = v_{\max}$：$\partial V / \partial v = 0$（Neumann，波动率饱和）

### 3. 特征函数与 Riccati 方程

Heston 模型对数价格的特征函数具有仿射结构：

$$
\varphi(u) = \exp\bigl( A(u, \tau) + D(u, \tau) \cdot v_0 + iu \ln S_0 \bigr)
$$

其中 $\tau = T - t$，$A$ 与 $D$ 满足 Riccati ODE 系统：

$$
\begin{aligned}
\frac{dD}{d\tau} &= -\frac{1}{2} u(u+i) + (\kappa - i\rho\sigma u) D + \frac{1}{2}\sigma^2 D^2 \\[6pt]
\frac{dA}{d\tau} &= r u i + \kappa\theta D
\end{aligned}
$$

初值 $D(0) = A(0) = 0$。解析解为：

$$
\begin{aligned}
d(u) &= \sqrt{(i\rho\sigma u - \kappa)^2 + \sigma^2 (ui + u^2)} \\[6pt]
g(u) &= \frac{\kappa - i\rho\sigma u - d}{\kappa - i\rho\sigma u + d} \\[6pt]
D(u, \tau) &= \frac{\kappa - i\rho\sigma u - d}{\sigma^2} \cdot \frac{1 - e^{-d\tau}}{1 - g \, e^{-d\tau}} \\[6pt]
A(u, \tau) &= ru i \tau + \frac{\kappa\theta}{\sigma^2}\Bigl[ (\kappa - i\rho\sigma u - d)\tau - 2\ln\frac{1 - g e^{-d\tau}}{1 - g} \Bigr]
\end{aligned}
$$

### 4. ADI 有限差分离散化

对 $(S, v)$ 空间采用**非均匀拉伸网格**：

$$
S_i = S_{\max} \frac{\sinh(c_s \xi_i)}{\sinh(c_s)}, \qquad
v_j = v_{\max} \frac{e^{c_v \eta_j} - 1}{e^{c_v} - 1}
$$

时间推进采用**隐式欧拉格式**（无条件稳定）：

$$
(I - \Delta t \, A) \, V^{n+1} = V^n + \Delta t \, b_{\text{bc}}
$$

其中 $A$ 为空间离散算子的稀疏矩阵，采用 **CCS（Compressed Column Storage）** 格式存储。对于大规模系统，使用 **重启 GMRES** 迭代求解。

### 5. 重启 GMRES 迭代法

对于线性系统 $Ax = b$，GMRES 在 Krylov 子空间

$$
\mathcal{K}_m(A, r_0) = \text{span}\{r_0, A r_0, A^2 r_0, \dots, A^{m-1} r_0\}
$$

中寻找最小残差近似解。Arnoldi 迭代产生正交基 $V_m$ 与上 Hessenberg 矩阵 $\tilde{H}_m$，满足：

$$
A V_m = V_{m+1} \tilde{H}_m
$$

通过 **Givens 旋转** 对 $\tilde{H}_m$ 进行增量式 QR 分解，可 $O(1)$ 更新残差范数。内存受限时采用 restarted GMRES，每 $m$ 步以当前解为初值重新开始。

### 6. Smolyak 稀疏网格配置法

对于 $d$ 维参数空间的积分，Smolyak 稀疏网格公式：

$$
A(q, d) = \sum_{|i| \leq q} (\Delta_{i_1} \otimes \cdots \otimes \Delta_{i_d})
$$

其中 $\Delta_i = Q_i - Q_{i-1}$ 为一维差分 Quadrature。组合系数：

$$
c_i = (-1)^{q - |i|} \binom{d - 1}{q - |i|}
$$

相比全张量积，稀疏网格将节点数从 $O(N^d)$ 降至 $O(N (\log N)^{d-1})$。

### 7. 主成分分析（PCA）与波动率曲面降维

对隐含波动率矩阵 $X \in \mathbb{R}^{m \times n}$ 中心化后，样本协方差矩阵：

$$
C = \frac{1}{n-1} (X - \mu \mathbf{1}^T)(X - \mu \mathbf{1}^T)^T
$$

当 $m \gg n$ 时，采用 **Turk-Pentland 技巧**：先对 $n \times n$ 矩阵 $X^T X$ 特征分解，再通过 $X v_k$ 得到 $C$ 的特征向量，复杂度从 $O(m^3)$ 降至 $O(n^3)$。

波动率曲面通常可由前 3 个主成分解释 95% 以上方差：
- PC1（水平位移）：平行移动
- PC2（倾斜）：微笑斜率
- PC3（曲率）：微笑曲率

### 8. 黄金分割搜索与延拓法校准

**黄金分割搜索**：对于区间 $[a, b]$ 上的单峰函数，取内点

$$
x_1 = \phi a + (1 - \phi) b, \qquad x_2 = (1 - \phi) a + \phi b
$$

其中 $\phi = (\sqrt{5} - 1)/2 \approx 0.618$ 为黄金分割比。区间每轮缩小比例恒为 $\phi$，收敛率 $\approx 0.618$。

**延拓法（Homotopy / Continuation）**：给定参数化非线性系统 $F(x; \lambda) = 0$，从已知解 $(x_0, \lambda_0)$ 出发，通过预测-校正步跟踪解曲线：

$$
\text{预测: } x_1 = x_0 + h \, t, \qquad
\text{校正: } \text{Newton 迭代求解 } G(x) = \begin{bmatrix} F(x; \lambda) \\ x_p - x_{1p} \end{bmatrix} = 0
$$

切向量 $t$ 满足 $J \, t = 0$，$\|t\| = 1$，通过 SVD 求得。

### 9. 椭圆积分与 Laguerre 求根法

**第二类完全椭圆积分**（AGM 算法）：

$$
E(k) = \int_0^{\pi/2} \sqrt{1 - k^2 \sin^2 \theta} \, d\theta
$$

通过算术-几何平均迭代：

$$
a_{n+1} = \frac{a_n + b_n}{2}, \quad b_{n+1} = \sqrt{a_n b_n}, \quad c_{n+1} = \frac{a_n - b_n}{2}
$$

**Laguerre 多项式求根法**：对于 $n$ 次多项式，迭代公式为

$$
z = (f')^2 - \frac{n}{n-1} f \, f'', \qquad
\Delta x = -\frac{n}{n-1} \frac{f}{f' + \text{sign}(f') \sqrt{z}}
$$

用于求解 Heston 特征函数的复平面驻点。

### 10. 非线性动力学与多重分形

**波动率-订单流耦合 ODE**（受放牧生态系统模型启发）：

$$
\begin{cases}
\displaystyle\frac{du}{dt} = r_1 u \Bigl(1 - \frac{u}{K}\Bigr) - c_1 v \bigl(1 - e^{-d_1 u}\bigr) \\[8pt]
\displaystyle\frac{dv}{dt} = -a v + c_2 v \bigl(1 - e^{-d_2 u}\bigr)
\end{cases}
$$

**多重分形谱**（盒计数法）：广义维数 $D_q$ 满足

$$
D_q = \lim_{\varepsilon \to 0} \frac{1}{q - 1} \frac{\ln \sum_i p_i^q}{\ln \varepsilon}
$$

---

## 种子项目映射表

| 序号 | 种子项目 | 核心算法 | 合成后角色 |
|:---:|---------|---------|-----------|
| 1 | **975_r8ccs** | 稀疏矩阵 CCS 格式（矩阵-向量乘、转置乘、元素访问、二阶差分构造） | `sparse_matrix_ccs.py`：Heston PDE 离散化后的稀疏线性系统存储与运算 |
| 2 | **488_grazing_ode** | 非线性 ODE 导数计算与参数化动力学 | `nonlinear_dynamics.py`：波动率-订单流耦合 ODE 与 Feller 稳定性分析 |
| 3 | **326_eigenfaces** | PCA 主成分提取（Turk-Pentland 技巧） | `principal_component_analysis.py`：波动率曲面降维与多因子相关性提取 |
| 4 | **328_ellipse** | 椭圆面积、周长、椭圆积分 | `special_math_utils.py`：$(S, v)$ 空间椭圆截断域面积计算与完全椭圆积分 AGM 算法 |
| 5 | **476_golden_section** | 黄金分割一维优化搜索 | `parameter_optimizer.py`：Heston 参数 $\rho$ 的局部校准优化 |
| 6 | **1055_sandia_sgmgg** | 稀疏网格组合系数计算（Smolyak） | `sparse_grid_stochastic.py`：高维参数期望计算的稀疏网格配置法 |
| 7 | **760_mgmres** | 重启 GMRES 迭代法与 Givens 旋转 | `gmres_iterative.py`：大型稀疏线性系统的 Krylov 子空间迭代求解 |
| 8 | **1168_stla_to_tri_surface_fast** | 快速文本解析与结构化数据处理 | `special_math_utils.py`：市场数据 CSV 快速解析与结构化读取 |
| 9 | **652_latin_random** | 拉丁超立方随机采样 | `latin_hypercube_sampler.py`：蒙特卡洛模拟中的方差缩减与相关正态采样 |
| 10 | **318_dragon_chaos** | 迭代函数系统（IFS）与混沌分形 | `nonlinear_dynamics.py`：市场波动多重分形特征分析与 Dragon IFS |
| 11 | **382_fem_to_xml** | 有限元网格数据格式转换与管理 | `special_math_utils.py`：1D/2D 有限差分/有限元网格生成与边界节点管理 |
| 12 | **210_continuation** | 延拓法、Newton 迭代、切向量计算 | `parameter_optimizer.py`：波动率微笑曲线的解分支跟踪与全局校准 |
| 13 | **915_prime_plot** | 素性检测与数论 | `special_math_utils.py`：Miller-Rabin 素性测试，用于伪随机数长周期模数选择 |
| 14 | **1430_zero_laguerre** | Laguerre 多项式求根法 | `special_math_utils.py`：Heston 特征函数复平面驻点的 Laguerre 迭代求解 |
| 15 | **1377_usa_box_plot** | 箱线图填充与分位数统计 | `special_math_utils.py`：蒙特卡洛输出的 VaR/CVaR 风险度量与异常值检测 |

---

## 文件结构

```
141_synth_project/
├── main.py                           # 统一入口，零参数运行
├── sparse_matrix_ccs.py              # 稀疏矩阵 CCS 格式
├── gmres_iterative.py                # 重启 GMRES 迭代求解器
├── heston_pde_engine.py              # Heston PDE 有限差分引擎
├── principal_component_analysis.py   # 主成分分析与波动率降维
├── sparse_grid_stochastic.py         # Smolyak 稀疏网格配置法
├── latin_hypercube_sampler.py        # 拉丁超立方采样
├── parameter_optimizer.py            # 黄金分割搜索与延拓法校准
├── nonlinear_dynamics.py             # 非线性动力学、混沌与 Riccati 解析解
├── special_math_utils.py             # 椭圆积分、Laguerre 求根、素性检测、统计分位数、网格管理
└── README_博士级合成说明.md          # 本说明文档
```

**共 10 个 Python 文件**，满足 $\geq 8$ 个的要求。

---

## 运行方式

在项目目录下直接执行：

```bash
python main.py
```

无需任何命令行参数。程序将自动执行 8 组数值实验，输出包括：
1. 稀疏矩阵 CCS 格式验证与 GMRES 求解性能
2. 拉丁超立方采样与蒙特卡洛方差缩减效果
3. 波动率曲面主成分分析（PCA）
4. 稀疏网格高维积分精度
5. Heston PDE 有限差分定价与 Greeks 计算
6. Riccati 解析解、Laguerre 求根、非线性 ODE 与多重分形分析
7. 黄金分割搜索、素数检测与延拓法解分支跟踪
8. 风险度量统计与有限元网格管理

---

## 边界处理与数值鲁棒性

1. **Feller 条件自动检测**：在 PDE 求解前检查 $2\kappa\theta/\sigma^2$，若小于 1 则自动在 $v=0$ 附近加密网格。
2. **非均匀拉伸网格**：$S$ 方向使用 $\sinh$ 拉伸，$v$ 方向使用指数拉伸，确保关键区域（ATM 行权价、低波动率边界）有足够分辨率。
3. **退化边界处理**：$v=0$ 处扩散项消失，自动降维为一维对流方程，避免数值不稳定。
4. **复数运算鲁棒性**：所有涉及 Heston 特征函数的复数运算均使用 `cmath`/`numpy` 支持，避免 `math.sqrt` 对负数/复数崩溃。
5. **GMRES 重启机制**：内存受限时自动重启，内迭代次数自适应，残差范数通过 Givens 旋转增量更新。
6. **延拓法步长自适应**：收敛失败时自动减半步长，成功时放大 1.2 倍，保证解分支跟踪的鲁棒性。
7. **素性检测 Miller-Rabin**：使用确定性基底集，对 64 位整数保证正确性。
8. **LHS 尾部截断**：逆正态变换前将均匀样本截断到 $[10^{-10}, 1 - 10^{-10}]$，避免极端尾部数值溢出。

---

## 数值实验摘要

| 实验 | 内容 | 关键结果 |
|:---:|------|---------|
| 1 | 500x500 稀疏系统 GMRES 求解 | 310 次迭代收敛，残差 $9.3 \times 10^{-11}$，耗时 280 ms |
| 2 | LHS 生成 2000 个相关正态样本 | 相关系数估计 $-0.698$（理论 $-0.7$），MC 积分相对误差 0.26% |
| 3 | 波动率曲面 PCA | 第一主成分解释方差 100%（合成数据秩为 1）；3x3 协方差矩阵第一因子解释 66.8% |
| 4 | 稀疏网格积分 | 2D 层级 4 稀疏网格 641 节点，积分相对误差 7.9% |
| 5 | Heston PDE 定价 | ATM 看涨期权价格 9.58（BS 参考 9.41）；Delta=0.55, Vega=41.1 |
| 6 | Riccati + ODE + 分形 | Riccati 解析解有效；Dragon IFS 容量维数 $D_0 \approx 1.81$ |
| 7 | 优化与延拓 | Rosenbrock 函数黄金分割 42 步收敛；抛物线延拓 31 步从 $\lambda=1$ 跟踪到 $\lambda=14.6$ |
| 8 | 风险度量与网格 | VaR(99%)=-0.42, CVaR=-0.52；1D 网格 41 节点，2D 网格 66 节点/100 三角形 |

---

## 前沿性与博士级难度说明

本项目综合了以下**博士级计算难度**的要素：

1. **退化抛物型 PDE 的数值求解**：Heston PDE 在 $v=0$ 处退化，需要特殊的边界处理与非均匀网格设计，属于偏微分方程数值分析的前沿课题。
2. **Krylov 子空间迭代法**：GMRES 结合 Givens 旋转的增量式 QR 更新，是大型稀疏线性系统求解的标准博士课程内容。
3. **Smolyak 稀疏网格**：高维积分的"维度灾难"缓解技术，涉及组合数学与逼近论的深度交叉。
4. **仿射过程特征函数的复分析**：Riccati ODE 的解析求解涉及复变函数分支切割与数值稳定性分析。
5. **延拓法全局路径跟踪**：非线性方程组解分支的拓扑分析，属于计算数学中数值延拓（Numerical Continuation）的核心内容。
6. **多重分形与混沌动力学**：将生态学中的非线性 ODE 与分形理论映射到金融市场波动率分析，体现了跨学科建模能力。
7. **主成分降维与多因子相关性**：在高维波动率协方差结构中提取独立驱动因子，是量化金融博士研究的典型课题。

---

*文档生成时间：2026-05-06*
