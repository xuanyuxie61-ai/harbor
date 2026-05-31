# README_博士级合成说明.md

## 项目主题

**金融工程：高维非线性风险平价投资组合优化**
——基于流形学习、谱方法与随机偏微分方程的风险度量

---

## 一、项目概述

本项目将 15 个原始科研代码项目的核心算法融合重构，面向金融工程中的前沿博士级问题：**在高维、强相关、非正态的金融市场中，如何构建真正风险分散化的投资组合**。项目集成了谱方法、球面几何嵌入、网络分析、随机微分方程、组合优化与单纯形格点搜索等多元数学工具，形成一个统一的 Python 计算框架。

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 在金融工程中的角色 |
|---|---|---|
| 161_chebyshev_matrix | Chebyshev 谱微分矩阵与网格 | **高精度风险度量**：利用谱方法对收益率 CDF 进行指数收敛精度插值，计算 VaR/CVaR |
| 181_circle_monte_carlo | 单位圆随机采样与单变量积分 | **高维球面采样**：生成相关性保持的随机冲击，用于蒙特卡洛积分与路径模拟 |
| 1205_test_digraph_arc | 有向图弧列表与 PageRank 网络 | **系统性风险识别**：构建资产依赖网络，用 PageRank 度量系统重要性 |
| 891_polygonal_surface_display | 多边形顶点/面数据解析 | **凸包分析**：提取收益-风险空间的凸包顶点，近似有效前沿 |
| 1236_tet_mesh_quality | 四面体质量评估（体积、行列式） | **分散度几何指标**：将协方差矩阵子块的体积与条件数作为分散度度量 |
| 307_distance_to_position_sphere | 球面距离到坐标的非线性最小二乘 | **相关性球面嵌入**：把相关性矩阵映射到高维球面，进行几何风险分析 |
| 609_jai_alai_simulation | 蒙特卡洛锦标赛模拟 | **Bootstrap 与尾部模拟**：模拟资产"竞争"过程，估计相对表现概率 |
| 132_caesar | Caesar 移位密码 | **数据敏感性扰动**：对收益率序列进行循环移位加噪声，测试策略鲁棒性 |
| 1138_spring_double_ode | 双弹簧耦合 ODE | **市场传染动力学**：推广为多维耦合 SDE，描述资产间的动量-回归耦合 |
| 1286_trapezoidal | 梯形隐式 ODE 求解器 | **SDE 数值积分**：梯形隐式格式求解耦合随机微分方程，保证数值稳定性 |
| 1170_stochastic_heat2d | 二维随机热方程（稀疏矩阵有限差分） | **风险传播模拟**：将风险冲击视为热源，用随机热方程模拟其在网络中的扩散 |
| 1352_triangulation_svg | 三角剖分数据结构 | **相似性网络构建**：Delaunay 剖分构建资产相似性网络，识别潜在传染路径 |
| 578_image_double | 矩阵双线性插值/上采样 | **协方差矩阵精细化**：对子样本协方差矩阵进行上采样，提升估计精度 |
| 054_asa299 | 单纯形格点枚举 | **组合权重格点搜索**：在标准单纯形上枚举有理点，探索带整数约束的组合 |
| 306_distance_to_position | 欧氏距离到坐标的 MDS 嵌入 | **低维相似性嵌入**：经典多维尺度分析将资产映射到二维平面，用于网络可视化替代分析 |

---

## 三、新增数学物理模型与核心公式

### 3.1 Chebyshev 谱方法风险度量

对收益率样本 $\{r_i\}_{i=1}^N$，经验累积分布函数（ECDF）为

$$
\hat{F}_N(r) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(r_i \le r).
$$

将 $r$ 通过仿射变换映射到 $[-1,1]$，在 Chebyshev 节点

$$
x_j = \cos\left(\frac{\pi j}{n}\right), \quad j=0,1,\dots,n
$$

上构造插值。利用重心 Lagrange 插值公式

$$
p(x) = \frac{\sum_{j=0}^n \frac{w_j}{x-x_j} \hat{F}_N(r_j)}{\sum_{j=0}^n \frac{w_j}{x-x_j}},
\quad w_0=w_n=\frac{1}{2},\; w_j=(-1)^j,
$$

高精度求解分位点。VaR 与 CVaR 定义为

$$
\text{VaR}_\alpha = \inf\{r \in \mathbb{R} \mid \hat{F}_N(r) \ge \alpha\},
$$

$$
\text{CVaR}_\alpha = \frac{1}{\alpha} \int_{-\infty}^{\text{VaR}_\alpha} r \, d\hat{F}_N(r).
$$

谱微分矩阵 $D$ 满足 $(Dv)_i \approx p'(x_i)$，其元素为

$$
D_{ij} = \frac{c_i}{c_j} \frac{(-1)^{i+j}}{x_i-x_j}\;(i\ne j),\quad
D_{ii} = -\sum_{j\ne i} D_{ij},
$$

其中 $c_0=c_n=2$, $c_j=1$ ($1\le j\le n-1$)。

### 3.2 球面嵌入与几何分散度

资产相关性矩阵 $C$ 诱导球面距离

$$
d_{ij}^{\text{sphere}} = \arccos(C_{ij}) \in [0,\pi].
$$

通过非线性最小二乘求解经纬度坐标 $(\phi_i,\lambda_i)$，使得

$$
\sum_{i<j} \bigl(d_{ij}^{\text{sphere}} - d_{\text{haversine}}(\phi_i,\lambda_i,\phi_j,\lambda_j)\bigr)^2 \to \min.
$$

球面几何分散度指数定义为

$$
\mathcal{D} = \frac{1}{n^2} \sum_{i,j} \|\mathbf{p}_i - \mathbf{p}_j\|^2
= 2 - 2\Bigl\|\frac{1}{n}\sum_i \mathbf{p}_i\Bigr\|^2,
$$

其中 $\mathbf{p}_i \in \mathbb{S}^2$。当资产完全重合时 $\mathcal{D}=0$；均匀分布时 $\mathcal{D}\to 2$。

### 3.3 网络风险与 PageRank

设资产依赖网络的邻接矩阵为 $A$，行随机矩阵为

$$
P_{ij} = \frac{A_{ij}}{\sum_k A_{ik}}.
$$

Google 矩阵

$$
G = \alpha P + \frac{1-\alpha}{n} \mathbf{1}\mathbf{1}^T,
$$

PageRank 向量 $\mathbf{x}$ 满足 $\mathbf{x} = G^T \mathbf{x}$，$\sum_i x_i = 1$。

在金融语境下，$x_i$ 越大表示资产 $i$ 的系统重要性越高。

### 3.4 随机热方程风险扩散

将金融网络映射到二维区域 $\Omega=[0,1]^2$，风险场 $u(x,y)$ 满足稳态随机热方程

$$
-\nabla \cdot \bigl(a(x,y;\boldsymbol{\omega}) \nabla u\bigr) = f(x,y),
\quad u|_{\partial\Omega} = g,
$$

其中扩散系数 $a(x,y;\boldsymbol{\omega}) = \omega_1 + \omega_2 \sin(\pi x)\sin(\pi y)$ 体现市场不确定性。

采用五点有限差分离散，构造稀疏线性系统 $A\mathbf{u}=\mathbf{F}$，通过稀疏直接求解器获得稳态风险分布。

### 3.5 风险平价优化

设协方差矩阵为 $\Sigma$，组合权重为 $\mathbf{w}$，组合风险为

$$
\sigma(\mathbf{w}) = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}.
$$

资产 $i$ 的边际风险贡献（MRC）与风险贡献（RC）分别为

$$
\text{MRC}_i = \frac{(\Sigma \mathbf{w})_i}{\sigma(\mathbf{w})},\qquad
\text{RC}_i = w_i \cdot \text{MRC}_i.
$$

风险平价要求

$$
\text{RC}_i = b_i \, \sigma(\mathbf{w}),\quad \sum_i b_i = 1,
$$

即 $w_i (\Sigma \mathbf{w})_i = b_i (\mathbf{w}^T \Sigma \mathbf{w})$。

本项目采用循环坐标下降（CCD）算法（Spinu, 2013）：

$$
x_i^{k+1} = \sqrt{\frac{b_i}{\sum_{j\ne i} \Sigma_{ij} x_j^k}},\qquad
w_i = \frac{x_i}{\sum_j x_j}.
$$

### 3.6 分散化比率与有效赌注数

分散化比率（Choueifaty & Coignard, 2008）

$$
\text{DR}(\mathbf{w}) = \frac{\mathbf{w}^T \boldsymbol{\sigma}}{\sqrt{\mathbf{w}^T \Sigma \mathbf{w}}}.
$$

有效赌注数（Meucci, 2009）

$$
\text{ENB} = \exp\!\Bigl(-\sum_i p_i \ln p_i\Bigr),
\quad p_i = \frac{\text{RC}_i}{\sum_j \text{RC}_j}.
$$

### 3.7 耦合市场动力学

推广双弹簧 ODE 模型至高维资产系统：

$$
\begin{cases}
\displaystyle \frac{du_i}{dt} = v_i, \\[6pt]
\displaystyle \frac{dv_i}{dt} = \frac{1}{m_i}\Bigl[-k_1 u_i + \sum_j K_{2,ij}(u_j - u_i) - \gamma v_i\Bigr] + \sigma_i \dot{W}_i(t),
\end{cases}
$$

其中 $u_i$ 为价格偏离均衡的幅度，$v_i$ 为动量，$K_2$ 为资产间耦合矩阵，$W_i(t)$ 为独立维纳过程。

对确定性部分采用梯形隐式格式（A-稳定，截断误差 $O(\Delta t^3)$）：

$$
\mathbf{Y}_{n+1} = \mathbf{Y}_n + \frac{\Delta t}{2}\bigl[\mathbf{f}(t_n,\mathbf{Y}_n) + \mathbf{f}(t_{n+1},\mathbf{Y}_{n+1})\bigr] + \mathbf{g}(t_n,\mathbf{Y}_n) \sqrt{\Delta t}\,\mathbf{Z}_n.
$$

### 3.8 单纯形格点枚举与组合搜索

对 $n$ 维标准单纯形上的整数格点，满足

$$
x_i \ge 0,\quad \sum_{i=1}^n x_i = t,
$$

按逆字典序生成。组合权重为 $\mathbf{w} = \mathbf{x}/t$。格点总数为

$$
\binom{n+t-1}{t}.
$$

通过遍历所有格点寻找最小方差或最大夏普比率组合。

### 3.9 多维尺度分析（MDS）

对距离矩阵 $D$，定义双中心化矩阵

$$
B = -\frac{1}{2} J D^{(2)} J,\quad J = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T,
$$

其中 $D^{(2)}_{ij} = D_{ij}^2$。对 $B$ 谱分解 $B = V\Lambda V^T$，取前 $d$ 个正特征值，嵌入坐标为

$$
X = V_d \sqrt{\Lambda_d}.
$$

---

## 四、文件结构说明

```
144_synth_project/
├── main.py                      # 统一入口，零参数运行
├── chebyshev_pricing.py         # Chebyshev 谱方法 + 单位圆积分
├── spherical_embedding.py       # 球面 MDS 嵌入 + 高维采样
├── network_risk.py              # 网络构建 + PageRank + 随机热方程 + Delaunay
├── monte_carlo_simulator.py     # GBM 模拟 + Bootstrap + 锦标赛模拟
├── portfolio_optimizer.py       # 马科维茨 + 风险平价 (CCD/牛顿) + 分散化指标
├── dynamics_model.py            # 耦合 ODE/SDE + 梯形隐式求解器
├── simplex_search.py            # 单纯形格点枚举 + 体积/质量指标 + 格点搜索
├── utils.py                     # Caesar 扰动 + 矩阵上采样 + 凸包 + MDS + 条件数
└── README_博士级合成说明.md      # 本文档
```

---

## 五、合成后的项目能解决什么科学问题

1. **高精度尾部风险估计**：利用 Chebyshev 谱方法克服传统线性插值的低精度缺陷，为极端风险（VaR/CVaR）提供指数收敛的数值估计。

2. **高维相关结构的几何刻画**：通过球面嵌入将抽象的相关性矩阵转化为可计算的几何对象，利用角度距离和分散度指数量化组合的真实分散化程度。

3. **系统性风险的识别与预警**：结合 PageRank 与网络风险贡献度，识别"大而不能倒"的核心资产；通过随机热方程模拟风险冲击在网络中的时空演化。

4. **真正风险分散化的组合构建**：实现风险平价优化算法，使各资产对组合总风险的贡献相等，避免马科维茨优化中的权重集中问题。

5. **市场传染的动态模拟**：通过耦合 SDE 模型模拟单一资产受冲击后，风险如何通过耦合网络传播至整个系统，为压力测试提供定量工具。

6. **可行域的系统探索**：单纯形格点枚举提供了在不依赖梯度信息的条件下，系统搜索带约束组合优化问题的全局最优解的能力。

---

## 六、如何运行

确保环境中已安装 `numpy` 和 `scipy`：

```bash
pip install numpy scipy
```

进入项目目录并直接运行：

```bash
cd Synthesis-project-python/144_synth_project
python main.py
```

程序无需任何命令行参数，将自动完成以下流程：
1. 生成 8 资产、252 日合成收益率数据
2. Chebyshev 谱方法计算 VaR/CVaR
3. 球面嵌入与几何分散度分析
4. 资产网络构建、PageRank 与风险扩散模拟
5. 蒙特卡洛路径模拟与 Bootstrap 统计推断
6. 最小方差组合、风险平价组合与约束风险平价组合优化
7. 耦合动力学风险传染模拟
8. 单纯形格点搜索与协方差矩阵质量评估
9. 工具函数验证与结果汇总

---

## 七、质量检查清单

- [x] 原目录未被修改
- [x] 合成后的项目为 Python 语言
- [x] 新目录完整包含合成后的项目（9 个 .py 文件 + 1 个 README）
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] 15 个输入项目均已真实融入合成项目，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性（正则化、截断、fallback、条件检查）
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 已删除所有可视化相关内容
