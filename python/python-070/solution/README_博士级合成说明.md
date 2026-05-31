# 生态建模：渔业资源管理与最优捕捞策略 —— 博士级合成说明文档

## 一、项目概述

本项目基于 **15 个科研代码种子项目**，在 **生态建模：渔业资源管理与最优捕捞策略** 的科学领域内，融合构造了一个面向前沿科学问题的博士级 Python 计算项目。

### 核心科学问题

> **多尺度渔业资源动态建模与最优捕捞控制**：
> 在年龄结构种群动力学、空间扩散-对流、非线性密度依赖补充、环境不确定性及生态系统状态转换的多重约束下，求解使长期贴现经济收益最大化的最优捕捞策略，并量化其生态风险。

该问题涉及：
- **偏微分方程**（年龄/空间结构种群分布）
- **最优控制理论**（Pontryagin 最大值原理、动态规划）
- **随机分析与不确定性量化**（拟蒙特卡洛、风险评估）
- **相场理论**（生态系统双稳态与状态转换）
- **图优化与聚类分析**（海洋保护区网络、栖息地分区）
- **高维数值积分**（三维栖息地生物量估算）

---

## 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 在合成项目中的角色 |
|:---:|--------|---------|------------------|
| 1 | `941_quad_monte_carlo` | Monte Carlo 一维积分 | **不确定性量化**：随机种群参数的风险评估 |
| 2 | `387_fem1d_bvp_quadratic` | 二次元有限元法求解 1D BVP | **年龄结构种群动态**：McKendrick-von Foerster 方程稳态求解 |
| 3 | `695_local_min_rc` | Brent 反向通信局部优化 | **最优捕捞努力量**：Schaefer-Gordon 模型的利润最大化 |
| 4 | `076_bellman_ford` | 带负权图最短路径 | **MPA 网络优化**：海洋保护区生态连通性路径规划 |
| 5 | `498_hammersley` | Hammersley 低差异序列 | **不确定性量化**：多维参数空间的 QMC 采样 |
| 6 | `1076_sigmoid` | Sigmoid 高阶导数展开 | **补充量模型**：Allee 效应阈值函数的数学基础 |
| 7 | `211_continuity_exact` | 无散度二维速度场构造 | **空间生态学**：海洋流场对鱼卵/仔稚鱼被动输运的驱动 |
| 8 | `1381_vandermonde` | Bjorck-Pereyra $O(n^2)$ 算法 | **年龄-体长关系**：Vandermonde 快速多项式插值 |
| 9 | `229_cube_arbq_rule` | 3D 立方体高精度求积 | **栖息地积分**：三维海域生物量体积估算 |
| 10 | `1128_sphere_stereograph_display` | 球面立体投影 | **空间生态学**：地球曲面到平面的坐标映射 |
| 11 | `394_fem1d_nonlinear` | 非线性 FEM（Picard + Newton） | **密度依赖种群**：非线性扩散-反应方程求解 |
| 12 | `621_kmeans_fast` | Elkan 快速 K-Means | **栖息地聚类**：基于环境特征的渔区生态分区 |
| 13 | `003_allen_cahn_pde` | Allen-Cahn 相场方程 | **状态转换**：渔业生态系统双稳态与崩溃预警 |
| 14 | `931_pyramid_felippa_rule` | 金字塔域求积规则 | **栖息地积分**：水柱锥形域的生物量积分 |
| 15 | `679_line_felippa_rule` | 1D Gauss-Legendre 求积 | **栖息地积分**：深度剖面生物量线积分 |

**保证**：全部 15 个输入项目均已真实融入合成项目，无遗漏、无挂名。

---

## 三、新增数学物理模型与核心公式

### 3.1 种群动态模型

#### 3.1.1 Schaefer-Gordon 生物经济模型
稳态生物量与捕捞努力量的关系：
$$B^*(E) = K \left(1 - \frac{qE}{r}\right), \quad \text{当 } qE \leq r$$

其中：
- $r$：内禀增长率（1/年）
- $K$：环境承载力（吨）
- $q$：可捕系数
- $E$：捕捞努力量

#### 3.1.2 年龄结构稳态方程（McKendrick-von Foerster）
$$-D_a \frac{d^2 N}{da^2} + \frac{dN}{da} + M(a)N = R\delta(a-a_0)$$

采用二次元有限元离散，3 点 Gauss-Legendre 求积组装刚度矩阵：
$$A_{ij} = \sum_{e} \sum_{q} w_q \left[ p(x_q) \phi_i'(x_q) \phi_j'(x_q) + q(x_q) \phi_i(x_q) \phi_j(x_q) \right]$$

#### 3.1.3 非线性密度依赖模型
$$-\frac{d}{dx}\left(p(x)\frac{du}{dx}\right) + q(x)u + u\frac{du}{dx} = f(x)$$

采用 **Picard 迭代**（前 5 步）+ **Newton 迭代**（后续步）耦合求解：
- Picard：$u^{(k)} \nabla u^{(k+1)} \approx u^{(k)} \nabla u^{(k)}$
- Newton：Jacobian 项 $J_{ij} = \frac{\partial F_i}{\partial u_j}$ 包含 $u_{old} \cdot \phi_j' + \phi_j \cdot u_{old}'$

### 3.2 补充量模型（Stock-Recruitment）

#### 3.2.1 Beverton-Holt 模型
$$R(S) = \frac{\alpha S}{1 + \beta S}$$

渐近补充量：$R_\infty = \alpha / \beta$

#### 3.2.2 Ricker 模型
$$R(S) = \alpha S \exp(-\beta S)$$

最优亲体量：$S_{opt} = 1/\beta$，最大补充量 $R_{max} = \alpha / (e\beta)$

#### 3.2.3 Sigmoid-Allee 修正模型（本项目创新）
$$R_{SA}(S) = \frac{\alpha S}{1 + \beta S} \cdot \sigma\bigl(\gamma(S - S_{crit})\bigr)$$

其中 $\sigma(z) = 1/(1+e^{-z})$ 为平滑阈值函数，$S_{crit}$ 为 Allee 效应临界亲体量。

**导数公式**（用于稳定性分析）：
$$\frac{dR_{SA}}{dS} = R_{BH}'(S) \cdot \sigma(z) + R_{BH}(S) \cdot \sigma'(z) \cdot \gamma$$

其中 $\sigma'(z) = \sigma(z)(1-\sigma(z))$。

### 3.3 Sigmoid 高阶导数展开

Sigmoid 函数：
$$s(x) = \frac{1}{1+e^{-x}}$$

$n$ 阶导数的幂级数展开：
$$s^{(n)}(x) = \sum_{j=1}^{n+1} c_j \cdot s(x)^j$$

系数由组合公式计算：
$$c_k = \sum_{j=0}^{k} (-1)^{k-j} \binom{k}{j} (j+1)^n$$

### 3.4 最优控制与经济模型

#### 3.4.1 贴现利润目标泛函
$$J(E) = \int_0^T e^{-\delta t} \bigl[ p q E(t) B(t) - c E(t) \bigr] dt$$

稳态近似：
$$\Pi(E) = \bigl[p q E B^*(E) - c E\bigr] \cdot \frac{1-e^{-\delta T}}{\delta}$$

#### 3.4.2 最优捕捞努力量
通过 **Brent 方法**（黄金分割 + 抛物线插值）求解：
$$E^* = \arg\max_{E \in [0, E_{max}]} \Pi(E)$$

Brent 方法收敛率：
- 最坏情况（仅黄金分割）：线性收敛，比率 $\varphi - 1 = 1/\varphi \approx 0.618$
- 最好情况（连续抛物线插值）：超线性收敛，阶数 $\approx 1.324$

#### 3.4.3 最大可持续产量（MSY）
$$E_{MSY} = \frac{r}{2q}, \quad Y_{MSY} = \frac{rK}{4}$$

### 3.5 空间生态学

#### 3.5.1 无散度速度场
由流函数 $\Phi(Z) = (1-\cos(C\pi Z))(1-Z)^2$ 构造：
$$U(X,Y) = 10 \frac{\partial}{\partial Y}\bigl[\Phi(X)\Phi(Y)\bigr] = 10\Phi(X)\Phi'(Y)$$
$$V(X,Y) = -10 \frac{\partial}{\partial X}\bigl[\Phi(X)\Phi(Y)\bigr] = -10\Phi(Y)\Phi'(X)$$

解析验证：
$$\nabla \cdot \mathbf{v} = \frac{\partial U}{\partial X} + \frac{\partial V}{\partial Y} = 10\Phi'(X)\Phi'(Y) - 10\Phi'(Y)\Phi'(X) = 0$$

#### 3.5.2 对流-扩散方程（鱼卵输运）
$$\frac{\partial C}{\partial t} + U\frac{\partial C}{\partial x} + V\frac{\partial C}{\partial y} = D\nabla^2 C - \lambda C$$

采用 **一阶迎风格式**（Upwind）保证 CFL 稳定性：
$$U > 0: \quad \frac{\partial C}{\partial x} \approx \frac{C_{i,j} - C_{i-1,j}}{\Delta x}$$

#### 3.5.3 球面立体投影
从南极 $S=(0,0,-1)$ 到切平面 $Z=1$：
$$q_1 = \frac{2p_1}{1+p_3}, \quad q_2 = \frac{2p_2}{1+p_3}, \quad q_3 = 1$$

逆变换：
$$\mathbf{p} = \frac{(4e_1,\; 4e_2,\; 4-\|\mathbf{e}\|^2)}{4+\|\mathbf{e}\|^2}$$

### 3.6 不确定性量化

#### 3.6.1 Hammersley 低差异序列
M 维 Hammersley 点：
$$r_1(i) = \frac{i \bmod N}{N}, \quad r_{j+1}(i) = \sum_{k} \frac{d_k}{p_j^k}$$

其中 $p_j$ 为第 $j$ 个素数，$d_k$ 为 $i$ 的 $p_j$ 进制展开位数。

#### 3.6.2 QMC 积分收敛率
经典 MC：$O(N^{-1/2})$，QMC：$O\bigl((\log N)^M / N\bigr)$

#### 3.6.3 对数正态参数采样
若 $X \sim \text{LogNormal}(\mu_{ln}, \sigma_{ln})$：
$$\sigma_{ln}^2 = \ln\left(1 + \frac{\sigma^2}{\mu^2}\right), \quad \mu_{ln} = \ln\mu - \frac{\sigma_{ln}^2}{2}$$

### 3.7 栖息地积分

#### 3.7.1 多维张量积求积
$$\int_{\Omega} f(\mathbf{x})\, dV \approx \sum_{i} w_i f(\mathbf{x}_i)$$

#### 3.7.2 Gauss-Legendre 节点与权重（5 点）
在 $[-1,1]$ 上：
$$x_k \in \left\{0, \pm\frac{1}{3}\sqrt{5-2\sqrt{10/7}}, \pm\frac{1}{3}\sqrt{5+2\sqrt{10/7}}\right\}$$

#### 3.7.3 立方体域映射
参考域 $[-1,1]^3 \to [a,b]\times[c,d]\times[e,f]$：
$$x = \frac{b-a}{2}\xi + \frac{a+b}{2}, \quad J = \frac{(b-a)(d-c)(f-e)}{8}$$

### 3.8 快速 K-Means 聚类

#### 3.8.1 Elkan 三角不等式加速
$$\|\mathbf{x}_i - \boldsymbol{\mu}_j\| \geq \max\left(0,\; \|\mathbf{x}_i - \boldsymbol{\mu}_{c(i)}\| - \|\boldsymbol{\mu}_{c(i)} - \boldsymbol{\mu}_j\|\right)$$

若下界已超过当前最优距离，则跳过对中心 $j$ 的距离计算。

### 3.9 Allen-Cahn 相场模型

#### 3.9.1 控制方程
$$\frac{\partial u}{\partial t} = \nu \nabla^2 u - \frac{1}{2\xi^2} u(u^2-1) + \varepsilon E(t)(1-u^2)\text{sgn}(u)$$

#### 3.9.2 双稳态势函数
$$F(u) = \frac{1}{4}(u^2-1)^2$$

- $u = +1$：高生物量态（健康渔业）
- $u = -1$：低生物量态（衰退渔业）
- $u = 0$：不稳定临界点

#### 3.9.3 自由能泛函
$$\mathcal{E}[u] = \int \left[\frac{\nu}{2}|\nabla u|^2 + \frac{1}{4\xi^2}(u^2-1)^2\right] dx$$

#### 3.9.4 时间推进（四阶 Runge-Kutta）
$$\begin{aligned}
k_1 &= \Delta t \cdot f(t_n, u_n) \\
k_2 &= \Delta t \cdot f(t_n + \tfrac{\Delta t}{2}, u_n + \tfrac{k_1}{2}) \\
k_3 &= \Delta t \cdot f(t_n + \tfrac{\Delta t}{2}, u_n + \tfrac{k_2}{2}) \\
k_4 &= \Delta t \cdot f(t_n + \Delta t, u_n + k_3) \\
u_{n+1} &= u_n + \frac{k_1 + 2k_2 + 2k_3 + k_4}{6}
\end{aligned}$$

### 3.10 Vandermonde 快速求解

#### 3.10.1 Bjorck-Pereyra $O(n^2)$ 算法
求解 $V\mathbf{x} = \mathbf{b}$，其中 $V_{ij} = \alpha_j^{i-1}$：

前向消去（差商计算）：
$$x_j \leftarrow x_j - \alpha_k \cdot x_{j-1}, \quad j=n,\ldots,k+1$$

后向替换：
$$x_j \leftarrow \frac{x_j}{\alpha_j - \alpha_{j-k-1}}, \quad x_j \leftarrow x_j - x_{j+1}$$

---

## 四、合成项目文件结构与修改说明

```
070_synth_project/
├── main.py                     # 统一入口，零参数运行
├── utils.py                    # 数值配置、Legendre 多项式、Thomas 算法
├── recruitment_models.py       # 补充量模型 + Sigmoid 高阶导数
├── vandermonde_interp.py       # Bjorck-Pereyra 算法 + 多维 Vandermonde 插值
├── population_dynamics.py      # 二次元 FEM + 非线性 FEM（Picard/Newton）
├── spatial_ecology.py          # 无散度流场 + 立体投影 + 对流扩散
├── optimal_harvest.py          # Brent 优化 + Bellman-Ford + Schaefer-Gordon
├── uncertainty_quantification.py # Hammersley QMC + MC 积分 + 风险评估
├── habitat_integration.py      # 立方体/金字塔/线段求积规则
├── stock_clustering.py         # Elkan 快速 K-Means 聚类
└── regime_shift.py             # Allen-Cahn 相场 + RK4 时间推进
```

### 各文件修改要点

1. **`utils.py`**（新建）：
   - 整合 `prime.m` 的素数表思想，提供全局数值配置
   - 从 `fem1d_bvp_quadratic` 提取 3 点 Gauss-Legendre 规则
   - 从 `fem1d_nonlinear` 提取 Thomas 三对角求解算法
   - 新增 Legendre 多项式递推、安全除法等鲁棒性工具

2. **`recruitment_models.py`**（基于 `1076_sigmoid`）：
   - 将 Sigmoid 系数计算从 MATLAB 翻译为 Python
   - 新增 Beverton-Holt、Ricker 模型
   - **创新**：Sigmoid-Allee 修正模型及其导数公式

3. **`vandermonde_interp.py`**（基于 `1381_vandermonde`）：
   - 翻译 `pvand.m`、`dvand.m`、`bidim.m` 的核心算法
   - 新增一维插值和二维嵌套求解接口

4. **`population_dynamics.py`**（基于 `387_fem1d_bvp_quadratic` + `394_fem1d_nonlinear`）：
   - 完整翻译二次元 FEM 刚度矩阵组装流程
   - 翻译非线性 FEM 的 Picard + Newton 迭代耦合
   - 新增年龄结构 McKendrick-von Foerster 求解器
   - 新增 L² 误差估计

5. **`spatial_ecology.py`**（基于 `211_continuity_exact` + `1128_sphere_stereograph_display`）：
   - 翻译 `uv_spiral.m` 的无散度速度场公式
   - 翻译 `sphere_stereograph.m` 及其逆变换
   - 翻译 `icos_shape.m` 的二十面体顶点公式
   - 新增对流-扩散方程求解（迎风格式）
   - 新增鱼卵被动扩散模拟

6. **`optimal_harvest.py`**（基于 `695_local_min_rc` + `076_bellman_ford`）：
   - 将 Brent 反向通信算法改写为 Python 类封装
   - 翻译 Bellman-Ford 图算法
   - 新增 Schaefer-Gordon 生物经济模型
   - 新增 MPA 网络连通性优化

7. **`uncertainty_quantification.py`**（基于 `498_hammersley` + `941_quad_monte_carlo`）：
   - 翻译 `hammersley_sequence.m` 和 `hammersley_value.m`
   - 翻译 van der Corput 序列生成
   - 翻译 `quad_monte_carlo.m` 的积分框架
   - 新增 Beasley-Springer-Moro 逆正态 CDF
   - 新增渔业参数风险评估（对数正态采样）

8. **`habitat_integration.py`**（基于 `229_cube_arbq_rule` + `931_pyramid_felippa_rule` + `679_line_felippa_rule`）：
   - 翻译 `line_unit_o01`~`o05` 的 Gauss-Legendre 规则
   - 翻译 `cube_arbq` 和 `cube_arbq_size`
   - 翻译 `pyramid_unit_o01`、`pyramid_unit_o05`、`pyramid_unit_monomial`
   - 新增多维域积分接口和生物量估算函数

9. **`stock_clustering.py`**（基于 `621_kmeans_fast`）：
   - 翻译 `kmeans_fast.m` 的 Elkan 三角不等式加速逻辑
   - 翻译 `anchors.m` 的最远优先初始化
   - 翻译 `calcdist.m`、`alldist.m`
   - 新增渔业栖息地分区接口

10. **`regime_shift.py`**（基于 `003_allen_cahn_pde`）：
    - 翻译 `allen_cahn_deriv.m` 的右端项公式
    - 翻译 `laplacian_interval.m` 的 Neumann 边界处理
    - 新增四阶 Runge-Kutta 时间推进
    - 新增捕捞强迫项和自由能泛函计算

11. **`main.py`**（新建）：
    - 统一调用全部 10 个模块
    - 9 大演示章节，覆盖从种群动态到状态转换的完整流程

---

## 五、运行方式

```bash
cd Synthesis-project-python/070_synth_project
python main.py
```

程序无需任何命令行参数，运行后依次输出：
1. 鱼类种群补充量模型（Sigmoid-Allee 修正）
2. Vandermonde 快速算法验证
3. 年龄结构种群稳态 FEM 求解
4. 海洋流场与仔稚鱼被动扩散
5. 最优捕捞策略与 MPA 网络优化
6. 不确定性量化与风险评估（QMC）
7. 三维栖息地生物量体积积分
8. 渔业栖息地快速 K-Means 聚类
9. Allen-Cahn 相场状态转换模拟

---

## 六、科学意义与创新性

### 6.1 科学意义
本项目构建了一个**多尺度、多物理过程耦合**的渔业资源管理计算框架：
- **微观尺度**：年龄结构动力学（FEM 求解 PDE）
- **中观尺度**：空间扩散与对流（速度场 + 平流扩散方程）
- **宏观尺度**：最优经济控制（Brent 优化 + Bellman-Ford 网络）
- **不确定性**：参数随机性量化（QMC + 对数正态分布）
- **临界现象**：生态系统状态转换（Allen-Cahn 相场）

### 6.2 工程鲁棒性
- 所有数值运算包含**边界条件检查**（非负生物量、参数范围验证）
- **安全除法**处理（避免除以零）
- **CFL 稳定性条件**自动满足（时间步长自适应调整）
- **数值截断**防止浮点溢出（Sigmoid、指数函数）
- 迭代算法设置**最大迭代次数上限**和**收敛阈值**

### 6.3 博士级难度体现
1. **跨学科融合**：将有限元、图论、最优控制、随机分析、相场理论整合于单一科学问题
2. **前沿模型**：Sigmoid-Allee 修正补充模型、捕捞强迫 Allen-Cahn 方程
3. **高级算法**：Bjorck-Pereyra $O(n^2)$ Vandermonde 求解、Elkan 三角不等式加速、Brent 超线性优化
4. **完整闭环**：从种群动态 → 空间分布 → 最优策略 → 风险评估 → 状态预警

---

## 七、质量检查清单

- [x] 原目录未被修改
- [x] 合成后项目为 Python 语言
- [x] 新目录完整包含合成后的项目（11 个 .py 文件）
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] **全部 15 个输入项目已真实融入合成项目**
- [x] `main.py` 已实际运行通过，零参数且无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 无可视化代码残留
