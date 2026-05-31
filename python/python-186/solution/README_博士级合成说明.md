# 社交网络传播动力学：多模态耦合传播模拟系统

## 一、项目概述

本项目是基于 **15 个科研代码种子项目** 融合合成的博士级 Python 科研计算项目，聚焦于**数据科学：社交网络传播动力学**这一前沿领域。

核心科学问题为：**基于复杂网络拓扑、空间异质性与延迟反馈的多模态耦合传播动力学模型**。项目构建了一个从社交网络拓扑分析、空间异质性建模、随机接触模拟、流行病-信息耦合动力学演化、传播前沿高分辨率数值模拟，到参数校准与统计推断的完整科研计算 pipeline。

---

## 二、原项目到科学问题的映射

| 序号 | 原始种子项目 | 核心算法/思想 | 合成后角色 |
|:---:|:---|:---|:---|
| 1 | `413_fem2d_sample` (2D有限元采样) | 三角形T3基函数、重心坐标、点定位 | `spatial_mesh.py`：空间风险场的有限元插值与采样 |
| 2 | `182_circle_positive_distance` (几何概率) | 单位圆正象限蒙特卡洛距离估计 | `stochastic_contact.py`：随机接触模型的几何概率基础 |
| 3 | `1424_xyz_io` (3D点云I/O) | XYZ格式读写、螺旋点云生成 | `data_io.py`：多维特征数据的I/O与点云生成 |
| 4 | `1023_rigid_body_ode` (刚体欧拉方程) | ODE右端项构造、守恒量验证 | `epidemic_dynamics.py`：SEAIHR流行病ODE系统构造与守恒检查 |
| 5 | `709_magic4_matrix` (4阶幻方) | 互补对方法构造幻方 | `stochastic_contact.py`：幻方矩阵用于网络鲁棒性测试 |
| 6 | `796_neighbors_to_metis_graph` (邻居图转换) | 网格邻接关系提取、图格式转换 | `network_topology.py`：社交网络图结构构建与连通性保证 |
| 7 | `1238_tet_mesh_refine` (四面体网格细化) | 中点细分、边唯一性识别 | `spatial_mesh.py`：2D三角网格中点细化（4子三角形） |
| 8 | `867_persistence` (流式统计) | Welford在线算法、二分法状态机 | `parameter_calibration.py`：流式统计与二分法参数校准 |
| 9 | `975_r8ccs` (稀疏矩阵CCS) | 压缩列存储、稀疏矩阵-向量乘 | `sparse_algebra.py`：图拉普拉斯稀疏矩阵与幂迭代 |
| 10 | `441_floyd` (Floyd-Warshall) | 全源最短路径动态规划 | `network_topology.py`：网络最短路径与全局效率计算 |
| 11 | `707_mackey_glass_dde` (Mackey-Glass DDE) | 延迟微分方程、历史函数 | `epidemic_dynamics.py`：信息传播反馈的Mackey-Glass型延迟动力学 |
| 12 | `582_image_normalize` (图像归一化) | min-max/z-score标准化 | `data_io.py`：多维特征标准化与预处理 |
| 13 | `097_bisection_rc` (二分法求根) | 区间二分、反向通信 | `parameter_calibration.py`：接触率beta的标定 |
| 14 | `272_dg1d_burgers` (1D DG Burgers) | 间断Galerkin、LGL节点、数值通量 | `propagation_front.py`：信息传播前沿的DG高分辨率模拟 |
| 15 | `902_power_method` (幂法特征值) | 幂迭代、Rayleigh商、收敛判定 | `network_topology.py` 与 `sparse_algebra.py`：PageRank中心性与拉普拉斯谱分析 |

---

## 三、新增数学物理模型与核心公式

### 3.1 社交网络拓扑模型

**随机块模型 (Stochastic Block Model, SBM)**：

$$
A_{ij} \sim \begin{cases} 
\text{Bernoulli}(p_{in}) & \text{if } c_i = c_j \\
\text{Bernoulli}(p_{out}) & \text{if } c_i \neq c_j
\end{cases}
$$

**Floyd-Warshall 全源最短路径**：

$$
D_{ij}^{(k)} = \min\left(D_{ij}^{(k-1)},\; D_{ik}^{(k-1)} + D_{kj}^{(k-1)}\right)
$$

**网络全局效率 (Latora-Marchiori)**：

$$
E(G) = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{1}{d_{ij}}
$$

**介数中心性 (Brandes算法)**：

$$
C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}
$$

**PageRank-like 中心性 (幂法)**：

$$
\mathbf{y}_{k+1} = \frac{\alpha M \mathbf{y}_k + (1-\alpha)\frac{1}{n}\mathbf{1}}{\|\alpha M \mathbf{y}_k + (1-\alpha)\frac{1}{n}\mathbf{1}\|}
$$

$$\lambda_{k+1} = \mathbf{y}_{k+1}^T P \mathbf{y}_{k+1} \quad \text{(Rayleigh商)}$$

### 3.2 空间异质性有限元模型

**三角形T3基函数**：

$$
N_1(\xi,\eta) = 1 - \xi - \eta, \quad N_2(\xi,\eta) = \xi, \quad N_3(\xi,\eta) = \eta
$$

**重心坐标插值**：

$$
f(\mathbf{p}) = \sum_{i=1}^{3} \lambda_i f(\mathbf{v}_i), \quad \lambda_i = \frac{A(\mathbf{p}, \mathbf{v}_{i+1}, \mathbf{v}_{i+2})}{A(\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3)}
$$

**有限元刚度矩阵 (Poisson算子)**：

$$
K_{ij} = \sum_e \int_{T_e} \nabla\phi_i \cdot \nabla\phi_j \, dA = \sum_e A_e \, \mathbf{G}^T \mathbf{G}
$$

其中梯度矩阵 $\mathbf{G}$ 的元素为 $G_{ki} = \frac{\partial N_k}{\partial x_i}$。

### 3.3 稀疏矩阵与图拉普拉斯

**图拉普拉斯矩阵**：

$$
L = D - A, \quad D_{ii} = \sum_j A_{ij}
$$

**CCS稀疏矩阵-向量乘法**：

$$
y_i = \sum_{j: A_{ij} \neq 0} A_{ij} x_j
$$

### 3.4 SEAIHR 流行病-信息耦合动力学

**SEAIHR模型方程组**：

$$
\begin{aligned}
\frac{dS}{dt} &= -\beta \frac{S(I + \eta_A A)}{N} + \omega R \\
\frac{dE}{dt} &= \beta \frac{S(I + \eta_A A)}{N} - \sigma E \\
\frac{dA}{dt} &= (1-p_{sym})\sigma E - \gamma_A A \\
\frac{dI}{dt} &= p_{sym}\sigma E - (\gamma_I + \alpha_H) I \\
\frac{dH}{dt} &= \alpha_H I - (\gamma_H + \mu) H \\
\frac{dR}{dt} &= \gamma_A A + \gamma_I I + \gamma_H H - \omega R \\
\frac{dD}{dt} &= \mu H
\end{aligned}
$$

**信息反馈延迟动力学 (Mackey-Glass型)**：

$$
\beta(t) = \beta_0 \exp\left(-k_\beta \cdot I_{info}(t-\tau)\right)
$$

$$
\frac{dI_{info}}{dt} = \frac{\beta_{info} \cdot I_{past}}{1 + I_{past}^{n_{info}}} - \gamma_{info} I_{info}
$$

其中 $I_{past} = I(t-\tau) + A(t-\tau)$ 为延迟感染人数。

**基本再生数 $R_0$**：

$$
R_0 = \underbrace{\frac{\beta \, p_{sym}}{\gamma_I + \alpha_H}}_{R_0^{sym}} + \underbrace{\frac{\beta \, \eta_A (1-p_{sym})}{\gamma_A}}_{R_0^{asym}}
$$

**4阶Runge-Kutta积分**：

$$
\begin{aligned}
\mathbf{k}_1 &= f(t_n, \mathbf{y}_n) \\
\mathbf{k}_2 &= f(t_n + \tfrac{\Delta t}{2}, \mathbf{y}_n + \tfrac{\Delta t}{2}\mathbf{k}_1) \\
\mathbf{k}_3 &= f(t_n + \tfrac{\Delta t}{2}, \mathbf{y}_n + \tfrac{\Delta t}{2}\mathbf{k}_2) \\
\mathbf{k}_4 &= f(t_n + \Delta t, \mathbf{y}_n + \Delta t \, \mathbf{k}_3) \\
\mathbf{y}_{n+1} &= \mathbf{y}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
\end{aligned}
$$

局部截断误差 $O(\Delta t^5)$，全局误差 $O(\Delta t^4)$。

### 3.5 传播前沿 DG 模拟 (Burgers方程)

**1D粘性Burgers方程**（模拟信息传播前沿的激波结构）：

$$
\frac{\partial u}{\partial t} + \frac{\partial}{\partial x}\left(\frac{u^2}{2}\right) = \varepsilon \frac{\partial^2 u}{\partial x^2}
$$

**Nodal Discontinuous Galerkin 空间离散**：

参考单元 $[-1,1]$ 上的LGL节点 $\{r_i\}_{i=0}^{N}$，物理单元映射：

$$
x^k(r) = \frac{1-r}{2} x_L^k + \frac{1+r}{2} x_R^k
$$

**Legendre Vandermonde矩阵**：

$$
V_{ij} = P_j(r_i), \quad j = 0, \ldots, N
$$

**微分矩阵**：

$$
D_r = V_r V^{-1}
$$

**Lax-Friedrichs数值通量**：

$$
\hat{f} = \frac{1}{2}(u_M^2 - u_P^2) - \frac{\alpha}{2}(u_M - u_P), \quad \alpha = \max|u|
$$

**5-stage low-storage RK4时间积分**：

$$
\begin{aligned}
\mathbf{res}^{(k)} &= a_k \, \mathbf{res}^{(k-1)} + \Delta t \, \mathbf{RHS}(\mathbf{u}^{(k-1)}) \\
\mathbf{u}^{(k)} &= \mathbf{u}^{(k-1)} + b_k \, \mathbf{res}^{(k)}
\end{aligned}
$$

### 3.6 随机接触与几何概率

**几何接触率模型**：

个体在二维圆盘上均匀分布，接触率随距离指数衰减：

$$
\lambda_{ij} = \sqrt{a_i a_j} \exp\left(-\frac{d_{ij}}{d_0}\right)
$$

**单位圆上两点平均距离 (Monte Carlo)**：

$$
\mathbb{E}[d] \approx \frac{1}{N_{pairs}} \sum_{(p,q)} \sqrt{(x_p-x_q)^2 + (y_p-y_q)^2}
$$

**渗流阈值估计**：

对于 Erdős-Rényi 随机图：

$$
p_c \approx \frac{1}{n}
$$

### 3.7 参数校准与统计推断

**Welford在线方差算法**：

$$
\begin{aligned}
\bar{x}_n &= \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n} \\
M_{2,n} &= M_{2,n-1} + (x_n - \bar{x}_{n-1})(x_n - \bar{x}_n) \\
s_n^2 &= \frac{M_{2,n}}{n-1}
\end{aligned}
$$

**二分法求根 (标定接触率$\beta$)**：

给定目标 $R_0^{target}$，求解 $g(\beta) = R_0(\beta) - R_0^{target} = 0$：

$$
\beta^{(k+1)} = \begin{cases} 
\beta_{low}^{(k)} & \text{if } g(\beta_{low}) \cdot g(\beta_{mid}) < 0 \\
\beta_{mid}^{(k)} & \text{otherwise}
\end{cases}
$$

收敛速度：线性，$|\epsilon_{k+1}| \approx \frac{1}{2}|\epsilon_k|$。

**AIC信息准则**：

$$
\text{AIC} = 2k - 2\ln(L), \quad \text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}
$$

### 3.8 数据预处理与PCA

**主成分分析**：

$$
\mathbf{C} = \frac{1}{n-1} \mathbf{X}_c^T \mathbf{X}_c = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^T
$$

投影到前 $k$ 个主成分：

$$
\mathbf{Z} = \mathbf{X}_c \mathbf{V}_{:,1:k}
$$

---

## 四、项目文件结构

```
186_synth_project/
├── main.py                          # 统一入口，零参数可运行
├── network_topology.py              # 社交网络拓扑分析 (Floyd-Warshall, PageRank, 中心性)
├── spatial_mesh.py                  # 空间网格生成、有限元采样、扩散算子
├── sparse_algebra.py                # CCS稀疏矩阵、图拉普拉斯、稀疏幂迭代
├── epidemic_dynamics.py             # SEAIHR+信息耦合ODE/DDE、RK4积分器
├── propagation_front.py             # 1D DG方法求解Burgers传播前沿
├── parameter_calibration.py         # Welford流式统计、二分法校准、MLE、AIC
├── stochastic_contact.py            # 几何概率接触模型、幻方测试、渗流阈值
├── data_io.py                       # XYZ点云I/O、特征归一化、PCA
├── utils.py                         # 数值稳定性检查、熵、KL散度
└── README_博士级合成说明.md         # 本文档
```

---

## 五、合成后的项目能够解决什么科学问题

1. **复杂社交网络的结构分析**：量化网络效率、中心性、聚类系数、度分布熵，识别关键传播节点。
2. **空间异质性对传播的影响**：通过有限元方法建模地理空间上的风险场异质性，评估不同区域的风险差异。
3. **多模态耦合传播动力学**：同时模拟生物病原体传播与信息传播之间的双向反馈机制（信息改变行为→改变接触率→改变疫情）。
4. **传播前沿的高分辨率追踪**：使用间断Galerkin方法捕捉信息/疫情传播前沿的激波结构，避免数值扩散导致的虚假平滑。
5. **参数反演与校准**：基于观测数据，通过最大似然估计和二分法反演关键流行病学参数（如接触率$\beta$）。
6. **网络鲁棒性与渗流分析**：评估社交网络在边移除下的连通性变化，估计传播阈值。
7. **大规模稀疏计算**：利用CCS稀疏矩阵格式高效处理大规模网络邻接矩阵与拉普拉斯矩阵运算。

---

## 六、如何运行

### 环境要求
- Python 3.8+
- NumPy

### 运行方式
```bash
cd 186_synth_project
python main.py
```

程序无需任何输入参数，运行后将自动执行以下8个阶段：
1. 社交网络拓扑构建与分析
2. 空间异质性建模 (有限元方法)
3. 随机接触模型与几何概率
4. 流行病-信息耦合传播动力学
5. 传播前沿高分辨率模拟 (间断Galerkin)
6. 参数校准与统计推断
7. 多维数据I/O与特征分析
8. 综合指标与数值稳定性检查

### 运行示例输出
```
======================================================================
  社交网络传播动力学: 多模态耦合传播模拟系统
======================================================================

[阶段1] 社交网络拓扑构建与分析
--------------------------------------------------
  网络全局效率: 1.5641
  介数中心性范围: [0.0020, 0.0405]
  主特征值 (PageRank): 1.0000
  ...

[阶段8] 综合指标与数值稳定性检查
--------------------------------------------------
  所有数值检查通过 ✓
```

---

## 七、边界处理与数值鲁棒性设计

- **非负约束**：所有流行病学状态变量在RK4积分后强制 $y \geq 0$，防止人口数出现负值。
- **守恒修正**：SEAIHR模型中检查总人口守恒性 $dN/dt \approx 0$，偏差过大时进行归一化修正。
- **零除保护**：所有除法运算均包含分母保护（如 `max(denominator, 1e-10)`）。
- **NaN/Inf检测**：`utils.check_numerical_stability` 全局监控数值异常。
- **网格外点处理**：FEM采样中，点在网格外时使用最近节点值作为 fallback。
- **稀疏矩阵验证**：CCS格式构造时自动验证行索引排序与指针一致性。
- **幻方边界修正**：当输入 $n$ 不是4的倍数时，自动调整为有效的阶数。
- **DDE历史插值**：延迟项通过线性插值从已存储的历史数据中恢复，保证 $t < t_0$ 时使用初始历史函数。

---

## 八、科学难度说明

本项目具备博士级科学计算难度，体现在：

1. **多尺度耦合**：同时处理网络拓扑尺度（图论）、空间连续尺度（有限元）、时间演化尺度（ODE/DDE）三个不同尺度的耦合问题。
2. **高阶数值方法**：采用4阶Runge-Kutta时间积分、间断Galerkin谱元空间离散、Legendre-Gauss-Lobatto节点，属于计算数学前沿方法。
3. **延迟微分方程**：引入Mackey-Glass型延迟反馈，使系统从有限维ODE变为无限维DDE，显著增加分析难度。
4. **复杂网络分析**：融合PageRank中心性、Brandes介数中心性、Watts-Strogatz聚类系数、Latora-Marchiori效率等多指标网络分析。
5. **参数反演问题**：从观测数据反演流行病学参数是一个经典的反问题（ill-posed），需要结合最大似然估计、信息准则等统计方法。
6. **稀疏线性代数**：大规模稀疏矩阵的存储与运算本身就是高性能科学计算的核心课题。
