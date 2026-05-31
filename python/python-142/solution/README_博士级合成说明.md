# 信用风险违约相关性建模 —— 博士级科研代码合成项目

## 项目概述

本项目围绕**金融工程：信用风险违约相关性建模**这一前沿科学领域，将 15 个原始科研代码项目的核心算法融合重构为一个面向博士级研究的多模块 Python 计算框架。

项目全名：**空间-时序多主体信用风险违约相关性 PDE-Copula-网络混合框架**（Spatial-Temporal Multi-Name Credit Risk Default Correlation Modeling via PDE-Copula-Network Hybrid Framework）。

---

## 一、原项目到科学问题的映射

| 原项目编号 | 原项目核心内容 | 合成后角色 |
|-----------|--------------|-----------|
| 1123_sphere_llt_grid | 球面经纬度三角网格生成 | **球面信用风险区域划分**：将全球经济地理映射到单位球面，生成离散区域中心用于空间违约相关性建模 |
| 054_asa299 | N维单形格点枚举 | **组合权重与相关性矩阵离散化**：枚举投资组合权重配置及相关性矩阵特征值网格，用于压力测试 |
| 1259_theta_method | Theta 方法求解 ODE | **违约传染动态数值积分**：求解 Oregonator 启发的信用网络传染 ODE 系统 |
| 590_interp | 多维插值与节点生成 | **信用利差与违约概率期限结构插值**：Clenshaw-Curtis/Fejér 节点用于快速积分变换 |
| 628_knapsack_values | 0/1 背包测试数据 | **信用组合资本优化**：在 Basel III 风险资本约束下选择最优信用资产组合 |
| 359_fd1d_display | 1D 有限差分可视化 | **结构模型 PDE 有限差分离散**：求解违约概率密度的病态边值问题 |
| 725_matlab_map | 地理映射与 Voronoi | **球面 Voronoi 区域剖分**：构建区域间违约传染的地理拓扑网络 |
| 1131_sphere_voronoi | 球面 Voronoi 图 | **球面 Voronoi 面积计算**：Girard 公式计算区域经济权重 |
| 139_cauchy_principal_value | Cauchy 主值积分 | **Lévy 跳扩散模型特征函数逆变换**：奇异积分主值计算 |
| 1397_voronoi_neighbors | 平面 Voronoi 邻接 | **区域违约传染邻接矩阵**：基于 Delaunay 三角剖分构建传染网络 |
| 480_gram_schmidt | Gram-Schmidt 正交化 | **宏观经济因子正交化**：消除 CreditMetrics 多因子模型中的多重共线性 |
| 1311_triangle_lyness_rule | 三角形高精度求积 | **Copula 联合违约概率数值积分**：Lyness-Jespersen 规则用于二元高斯 Copula |
| 838_oregonator_ode | Oregonator 化学 ODE | **违约传染动力学模型**：将 BZ 化学反应振荡类比为信用网络违约波传播 |
| 151_cg_ne | CG-NE 法方程求解 | **隐含相关性校准**：从 CDO 市场价格反推相关性参数的最小二乘问题 |
| 572_ill_bvp | 病态边值问题 | **小扩散系数结构模型**：epsilon-摄动下的违约障碍稳态密度 BVP |

---

## 二、核心科学模型与数学公式

### 2.1 多因子正交化模型 (模块 1)

在 CreditMetrics / Vasicek 多因子框架中，资产 $i$ 的违约指示变量为：

$$X_i = \sum_{k=1}^{K} B_{i,k} F_k + \sqrt{1 - \sum_{k=1}^{K} B_{i,k}^2} \cdot Z_i$$

其中 $F_k \sim N(0,1)$ 为系统性因子，$Z_i \sim N(0,1)$ 为特质性冲击。通过**修正 Gram-Schmidt (MGS)** 正交化：

$$v_j = a_j - \sum_{i<j} \langle a_j, q_i \rangle q_i, \quad q_j = \frac{v_j}{\|v_j\|}$$

消除因子载荷矩阵 $B$ 的列间共线性，得到正交因子载荷 $B_{\perp}$。

### 2.2 单形格点枚举 (模块 2)

投资组合权重 $w$ 的可行域为 $N$ 维标准单形：

$$\Delta_N = \left\{ w \in \mathbb{R}^N_{\geq 0} : \sum_{i=1}^{N} w_i = 1 \right\}$$

通过格点枚举算法 (AS 299) 生成离散代表点集：

$$\mathcal{L}(N, T) = \left\{ x \in \mathbb{Z}^N_{\geq 0} : \sum_{i=1}^{N} x_i = T \right\}, \quad w = \frac{x}{T}$$

用于系统性探索不同行业配置下的违约相关性敏感性。

### 2.3 CG-NE 隐含相关性校准 (模块 3)

从 CDO 分券市场价格反推隐含相关性，构成最小二乘问题：

$$\min_{\delta\rho} \| J \cdot \delta\rho - (P_{\text{market}} - P_{\text{model}}) \|^2$$

等价于法方程：

$$(J^T J) \delta\rho = J^T \cdot r$$

采用带 Tikhonov 正则化的 CG-NE：

$$\min \|A x - b\|^2 + \lambda \|x\|^2 \;\Leftrightarrow\; (A^T A + \lambda I) x = A^T b$$

迭代格式：

$$\alpha_k = \frac{z_k^T z_k}{(A d_k)^T (A d_k) + \lambda d_k^T d_k}, \quad x_{k+1} = x_k + \alpha_k d_k$$

### 2.4 信用曲线插值 (模块 4)

违约概率期限结构 $PD(T)$ 的分段线性插值：

$$PD(T) = PD_i \frac{T_{i+1} - T}{T_{i+1} - T_i} + PD_{i+1} \frac{T - T_i}{T_{i+1} - T_i}, \quad T \in [T_i, T_{i+1}]$$

Clenshaw-Curtis 节点：

$$x_i = \cos\left(\frac{n-i}{n-1}\pi\right), \quad i = 0, \dots, n-1$$

### 2.5 球面 Voronoi 空间相关性 (模块 5)

将全球经济区域映射到单位球面 $S^2$，通过 Delaunay 三角剖分构建 Voronoi 对偶。区域 $i$ 与 $j$ 的空间违约相关性采用球面高斯核：

$$\rho_{ij}^{\text{spatial}} = \rho_0 \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)$$

其中 $d_{ij} = \arccos(|\langle \mathbf{x}_i, \mathbf{x}_j \rangle|)$ 为球面中心角。Voronoi 单元面积由 Girard 公式计算：

$$\text{Area}(\triangle) = (A + B + C - \pi) \cdot R^2$$

### 2.6 违约传染 ODE 系统 (模块 6)

受 Oregonator (BZ 反应) 启发的信用网络传染动力学：

$$\begin{aligned}
\frac{du}{dt} &= \frac{qv - uv + u(1-u)}{\eta_1} \\
\frac{dv}{dt} &= \frac{-qv - uv + fw}{\eta_2} \\
\frac{dw}{dt} &= u - w
\end{aligned}$$

其中 $u(t)$ 为违约强度，$v(t)$ 为传染压力，$w(t)$ 为系统性缓冲。数值求解采用 **Theta 方法**：

$$Y_{n+1} = Y_n + h \left[ \theta F(t_n, Y_n) + (1-\theta) F(t_{n+1}, Y_{n+1}) \right]$$

$\theta = 0.5$ 时为 Crank-Nicolson 格式（二阶精度，A-稳定）。

网络级联增强：

$$\lambda_i^{\text{network}} = \lambda_i^{\text{local}} + \gamma \sum_j A_{ij} \lambda_j^{\text{local}}$$

### 2.7 结构模型病态 BVP (模块 7)

资产价值稳态密度满足 Fokker-Planck 方程，经变量替换映射为奇异摄动 BVP：

$$\varepsilon y''(x) - x y'(x) + y(x) = 0, \quad x \in [-1, 1]$$

$$y(-1) = 2, \quad y(1) = 1$$

有限差分离散（自适应迎风格式，局部 Peclet 数判定）：

$$\text{Pe}_i = \frac{|x_i| h}{2\varepsilon} > 2 \;\Rightarrow\; \text{采用 upwind 差分}$$

Merton 模型解析违约概率：

$$PD(T) = \Phi\left( -\frac{\ln(V_0 / D) + (\mu - \sigma^2/2)T}{\sigma\sqrt{T}} \right)$$

### 2.8 Copula 积分与 Cauchy 主值 (模块 8)

二元高斯 Copula 联合违约概率：

$$C(u, v; \rho) = \Phi_2\left(\Phi^{-1}(u), \Phi^{-1}(v); \rho\right)$$

通过 Lyness-Jespersen 三角形求积规则进行高精度数值积分：

$$\int_T f(x,y) \, dA = |\det J| \sum_i w_i f(x_i, y_i)$$

Cauchy 主值积分（Gauss-Legendre，偶数点对称抵消）：

$$\text{CPV}\int_a^b \frac{f(t)}{t-x} \, dt = \sum_i w_i \frac{f(x_i') - f(x)}{x_i' - x}$$

### 2.9 信用组合优化 (模块 9)

双约束 0/1 背包问题：

$$\begin{aligned}
\max_{s \in \{0,1\}^N} \; & \sum_i v_i s_i \\
\text{s.t.} \; & \sum_i C_i s_i \leq C_{\text{total}} \\
& \sum_i PD_i \cdot LGD_i \cdot EAD_i \cdot s_i \leq \text{VaR}_{\text{limit}}
\end{aligned}$$

动态规划递推：

$$dp[i][w] = \max\left(dp[i-1][w], \; dp[i-1][w-w_i] + v_i\right)$$

---

## 三、项目文件结构

```
142_synth_project/
├── main.py                           # 统一入口，零参数运行
├── factor_orthogonalization.py       # 因子正交化 (480_gram_schmidt)
├── lattice_enumeration.py            # 单形格点枚举 (054_asa299)
├── linear_solver_cgne.py             # CG-NE 法方程求解 (151_cg_ne)
├── interpolation_surfaces.py         # 信用曲线插值 (590_interp)
├── spherical_correlation_grid.py     # 球面 Voronoi 相关性 (1123, 1131, 1397, 725)
├── default_contagion_ode.py          # 违约传染 ODE (838, 1259)
├── structural_default_bvp.py         # 结构模型病态 BVP (572, 359)
├── copula_quadrature.py              # Copula 积分与 CPV (1311, 139)
├── portfolio_knapsack.py             # 信用组合优化 (628)
├── utils.py                          # 通用数值工具
└── README_博士级合成说明.md           # 本文档
```

---

## 四、运行方式

```bash
cd /mnt/data/zpy/sci-swe/source code/Synthesis-project-python/142_synth_project
python main.py
```

程序将顺序执行 9 个独立模块和 1 个综合整合模块，输出各模块的数值结果摘要。无需任何命令行参数。

---

## 五、科学问题总结

本项目解决的核心科学问题为：

> **如何在一个统一计算框架下，耦合宏观因子结构、地理空间相关性、网络传染动态、结构模型稳态密度及 Copula 依赖结构，以实现对多主体信用组合违约相关性的高精度量化分析与资本优化配置？**

该问题涉及：
- **线性代数**：因子载荷正交化、相关性矩阵最近投影、Cholesky 分解
- **常微分方程数值解**：Theta 方法、刚性系统、边界层分析
- **偏微分方程 / 边值问题**：奇异摄动 BVP、自适应迎风有限差分
- **数值积分**：高斯求积、Cauchy 主值、三角形对称求积
- **组合优化**：动态规划、多约束背包问题
- **网络科学**：Voronoi 剖分、邻接矩阵、级联动力学

具备完整的边界处理、数值鲁棒性设计与多尺度建模能力。
