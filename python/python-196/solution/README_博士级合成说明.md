# 异构HPC集群热-电耦合模拟任务调度系统 — 博士级合成说明

## 一、项目概述

本项目围绕**高性能计算：异构计算任务调度**这一前沿科学领域，将15个原始科研代码项目的核心算法融合重构为一个面向异构高性能计算（HPC）集群的博士级科学计算系统。

**核心科学问题**：在现代数据中心中，热-电耦合物理模拟（如芯片热管理、电池热失控预测）需要在由CPU、GPU、FPGA、TPU等异构处理器组成的集群上进行高效调度。由于任务执行时间具有随机性、处理器性能受温度动态降频影响、且任务间存在复杂的数据依赖，如何构建一个集物理模拟、不确定性量化、性能预测与智能调度于一体的计算框架，是当前HPC领域的前沿挑战。

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 在合成项目中的角色 | 融入文件 |
|--------|----------|-------------------|----------|
| 500_hand | 2D仿射变换（旋转、平移、缩放、线性变换） | **网格自适应细化**中的几何变换算子；**处理器拓扑布局**的仿射映射 | `mesh_transform.py`, `heterogeneous_platform.py` |
| 331_ellipse_monte_carlo | 椭圆内Cholesky采样 | **材料参数不确定性量化**：在椭圆置信域内采样导热系数等物理参数 | `monte_carlo_uq.py` |
| 1022_reversi_game | 棋盘博弈贪心最优策略 | **任务抢占与迁移策略**：将调度状态映射到博弈棋盘，使用贪心策略选择最优迁移 | `scheduler_engine.py` |
| 850_partition_greedy | 贪心分区负载均衡 | **异构任务初步分配**：将任务按计算量降序分配到负载最轻的处理器 | `scheduler_engine.py` |
| 891_polygonal_surface_display | 多边形表面节点/面拓扑读取 | **FEM网格质量评估**：三角形单元质量指标计算（面积-边长比） | `mesh_transform.py`, `fem_thermal_solver.py` |
| 950_quadrature_weights_vandermonde | Vandermonde矩阵求积权重 | **热源项数值积分**：构造高精度Newton-Cotes求积公式 | `quadrature_integrator.py` |
| 847_pariomino | 整数规划与RREF求解 | **任务-处理器映射**：使用约束矩阵RREF分析解空间结构，贪婪搜索可行映射 | `scheduler_engine.py`, `utils.py` |
| 556_hypercube_distance | 超立方体随机点距离统计 | **高维任务特征相似度分析**：在5维特征空间中统计任务间欧氏距离分布 | `monte_carlo_uq.py` |
| 935_pyramid_monte_carlo | 金字塔区域单积分 | **芯片散热结构积分**：金字塔形散热片的热流密度精确积分 | `quadrature_integrator.py` |
| 566_hypersphere_monte_carlo | 超球体表面积与采样 | **高维参数空间探索**：超球面蒙特卡洛积分用于不确定性传播 | `monte_carlo_uq.py`, `utils.py` |
| 033_asa076 | 正态分布CDF（Hill算法AS 66） | **系统可靠性评估**：计算任务在给定时间内完成的概率 | `task_workload_model.py` |
| 014_approx_chebyshev | Chebyshev插值逼近 | **性能代理模型**：基于Chebyshev节点构建快速执行时间预测器 | `performance_surrogate.py` |
| 657_least_squares_approximant | 最小二乘多项式拟合（SVD） | **性能代理模型**：SVD正则化最小二乘拟合处理器性能曲面 | `performance_surrogate.py` |
| 698_log_normal | 对数正态分布采样与PDF | **任务执行时间随机建模**：用LogNormal描述异构处理器上的执行时间波动 | `task_workload_model.py` |
| 409_fem2d_poisson_rectangle_linear | 2D FEM求解Poisson方程 | **核心物理求解器**：求解稳态热传导方程获取温度场 | `fem_thermal_solver.py` |

---

## 三、新增数学物理模型与核心公式

### 3.1 稳态热传导方程（Poisson方程）

在芯片封装域 $\Omega = [0,1]\times[0,1]$ 上，温度场 $T(x,y)$ 满足：

$$
- k \nabla^2 T(x,y) = Q(x,y), \quad (x,y) \in \Omega
$$

边界条件（Dirichlet）：

$$
T(x,y) = T_b(x,y), \quad (x,y) \in \partial\Omega
$$

其中 $k$ 为导热系数 $[\text{W}/(\text{m}\cdot\text{K})]$，$Q$ 为体积热源密度 $[\text{W}/\text{m}^3]$。

**FEM弱形式**：寻找 $T_h \in V_h$ 使得

$$
k \int_\Omega \nabla T_h \cdot \nabla v_h \, d\Omega = \int_\Omega Q v_h \, d\Omega, \quad \forall v_h \in V_h^0
$$

离散后得到线性系统：$\mathbf{A}\mathbf{T} = \mathbf{F}$，其中单元刚度矩阵元为

$$
A_{ij}^{(e)} = k \cdot |\Omega_e| \cdot w_q \sum_{q} \left( \frac{\partial \phi_i}{\partial x}\frac{\partial \phi_j}{\partial x} + \frac{\partial \phi_i}{\partial y}\frac{\partial \phi_j}{\partial y} \right)_{(x_q,y_q)}
$$

### 3.2 对数正态执行时间模型

任务 $i$ 在处理器 $j$ 上的执行时间 $X_{ij}$ 服从对数正态分布：

$$
X_{ij} \sim \text{LogNormal}(\mu_{ij}, \sigma^2)
$$

其中

$$
\mu_{ij} = \mu_i^{(0)} - \ln\left(\frac{P_j}{P_{\text{ref}}}\right)
$$

概率密度函数：

$$
f(x; \mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\!\left(-\frac{1}{2}\left(\frac{\ln x - \mu}{\sigma}\right)^2\right), \quad x > 0
$$

**系统可靠性**（所有任务按时完成的联合概率）：

$$
R_{\text{sys}} = \exp\!\left( \sum_i \ln \Phi\!\left( \frac{\ln t_i^{(\text{alloc})} - \mu_{ij}}{\sigma} \right) \right)
$$

其中 $\Phi(\cdot)$ 为标准正态CDF，采用 **David Hill 的 Algorithm AS 66** 高精度近似：

$$
\Phi(z) = \begin{cases}
0.5 - z\left(p - qy/(y+a_1+b_1/(y+a_2+b_2/(y+a_3)))\right), & z \le 1.28 \\
r e^{-y} / \left(z + c_1 + d_1/(z+c_2+d_2/(\cdots))\right), & z > 1.28
\end{cases}
$$

其中 $y = z^2/2$，$p=0.39894228044$，$r=0.398942280385$ 等系数详见 `task_workload_model.py`。

### 3.3 Roofline性能模型

处理器 $j$ 的理论执行时间由Roofline模型决定：

$$
t_{ij}^{(\text{exec})} = \max\!\left( \frac{W_i}{P_j^{(\text{eff})}}, \; \frac{W_i}{B_j \cdot I_i} \right)
$$

其中：
- $W_i$：任务 $i$ 的浮点运算量 [FLOPs]
- $P_j^{(\text{eff})}$：处理器 $j$ 的有效峰值算力 [FLOPs/s]
- $B_j$：内存带宽 [bytes/s]
- $I_i$：计算强度 [FLOPs/byte]

**温度降频效应**：

$$
P_j^{(\text{eff})} = P_j^{(\text{peak})} \cdot \max\!\left(0, \; 1 - \alpha_{\text{th}} (T_j - T_{\text{amb}})\right)
$$

其中芯片结温 $T_j = T_{\text{amb}} + R_{\text{th}}^{(j)} \cdot P_{\text{diss}}^{(j)}$。

### 3.4 Chebyshev性能代理模型

为加速调度决策，在Chebyshev节点上构建插值代理：

**Chebyshev节点**：

$$
x_k = \frac{a+b}{2} + \frac{b-a}{2} \cos\!\left(\frac{2k+1}{2n}\pi\right), \quad k=0,\dots,n-1
$$

**Newton差商形式**：

$$
p(x) = d_0 + (x-x_0)\bigl(d_1 + (x-x_1)(d_2 + \cdots)\bigr)
$$

其中 $d_j$ 为 $j$ 阶差商，通过递归计算：

$$
d_j^{(m)} = \frac{d_j^{(m-1)} - d_{j-1}^{(m-1)}}{x_j - x_{j-m}}
$$

### 3.5 超球面与椭圆采样（不确定性量化）

**超球面表面积**：

$$
S_m = \begin{cases}
\displaystyle\frac{2\pi^{m/2}}{(m/2 - 1)!}, & m \text{ 为偶数} \\[8pt]
\displaystyle\frac{2^m \pi^{(m-1)/2}}{(m-1)!!}, & m \text{ 为奇数}
\end{cases}
$$

**椭圆内均匀采样**：对正定矩阵 $\mathbf{A}$ 描述的区域 $\mathbf{x}^T \mathbf{A} \mathbf{x} \le r^2$：

$$
\mathbf{A} = \mathbf{U}^T \mathbf{U} \quad\text{(Cholesky分解)} \\
\mathbf{Y} \sim \text{Uniform}(\mathcal{B}_2(0,r)), \quad \mathbf{X} = \mathbf{U}^{-1}\mathbf{Y}
$$

### 3.6 数值积分

**Vandermonde求积权重**：给定节点 $\{x_i\}_{i=1}^n$，求解

$$
\sum_{i=1}^n w_i x_i^{k-1} = \frac{b^k - a^k}{k}, \quad k=1,\dots,n
$$

即 $\mathbf{V}\mathbf{w} = \mathbf{r}$，其中 $V_{k,i} = x_i^{k-1}$。

**单位金字塔精确积分**：对单积分 $\int_{\mathcal{P}} x^{e_1} y^{e_2} z^{e_3} \, dV$，若 $e_1,e_2$ 均为偶数：

$$
I = \frac{2}{e_1+1} \cdot \frac{2}{e_2+1} \sum_{i=0}^{2+e_1+e_2} (-1)^i \binom{2+e_1+e_2}{i} \frac{1}{i+e_3+1}
$$

否则 $I = 0$。

### 3.7 多目标调度优化

调度问题形式化为混合整数规划：

$$
\min \; \alpha_1 C_{\max} + \alpha_2 E_{\text{total}} + \alpha_3 (1 - R_{\text{sys}})
$$

约束：
- $\sum_j x_{ij} = 1$（每个任务恰好分配到一个处理器）
- $C_{\max} \ge s_i + t_{ij} x_{ij}$（makespan下界）
- $T_j \le T_{\max}^{(\text{safe})}$（温度安全约束）

采用**贪心分区启发式**（源自 `partition_greedy`）进行初步分配，再通过**局部搜索**改进。

---

## 四、项目文件结构

```
196_synth_project/
├── main.py                      # 统一入口，零参数运行
├── utils.py                     # 通用工具：RREF、超球体公式、Cholesky、安全数学运算
├── mesh_transform.py            # 网格仿射变换与自适应细化（hand + polygonal_surface）
├── task_workload_model.py       # 对数正态建模 + 正态CDF（log_normal + asa076）
├── quadrature_integrator.py     # Vandermonde求积 + 金字塔积分（quadrature + pyramid）
├── monte_carlo_uq.py            # 椭圆/超球体/超立方体蒙特卡洛（ellipse + hypersphere + hypercube）
├── performance_surrogate.py     # Chebyshev插值 + 最小二乘拟合（approx_chebyshev + least_squares）
├── fem_thermal_solver.py        # 2D FEM热求解器（fem2d_poisson + polygonal_surface拓扑）
├── heterogeneous_platform.py    # 异构平台建模：CPU/GPU/FPGA/TPU（hand几何变换 + 热阻模型）
├── scheduler_engine.py          # 调度引擎：贪心分区 + ILP映射 + 博弈抢占（partition + pariomino + reversi）
└── README_博士级合成说明.md      # 本说明文档
```

---

## 五、合成后的项目能够解决什么科学问题

1. **芯片热管理模拟**：求解稳态热传导方程，获得芯片封装内的温度场分布，识别热点区域。
2. **材料参数不确定性量化**：通过蒙特卡洛方法在椭球/超球置信域内采样材料参数，量化温度预测的不确定性。
3. **异构任务调度优化**：在CPU+GPU+FPGA异构集群上，考虑执行时间随机性、温度约束与能耗，生成近似最优调度方案。
4. **性能快速预测**：利用Chebyshev插值和最小二乘代理模型，避免昂贵的全基准测试，快速评估不同任务-处理器配对的执行时间。
5. **网格自适应细化**：根据温度梯度场自动标记并细化高梯度区域单元，提高热点区域计算精度。

---

## 六、如何运行

### 环境要求
- Python >= 3.8
- NumPy >= 1.20

### 运行命令
```bash
cd 196_synth_project
python main.py
```

### 预期输出
程序将依次执行以下模块并输出结果：
1. **FEM Thermal Solver**：网格信息、L2/H1误差、自适应细化统计
2. **Monte Carlo UQ**：椭圆面积、超球面体积、距离统计、蒙特卡洛积分值
3. **Performance Surrogate**：Chebyshev与最小二乘代理模型的误差与预测对比
4. **Task Workload Modeling**：对数正态采样统计、正态CDF表、任务可靠性
5. **Numerical Quadrature**：Vandermonde求积结果、金字塔积分、2D复合积分误差
6. **Heterogeneous Platform & Scheduling**：处理器配置、拓扑变换、调度指标（makespan、能耗、系统可靠性）

整个流程约 0.1–0.3 秒，无需任何外部输入或参数。

---

## 七、关键数值结果示例

- **FEM L2 误差**：$9.92 \times 10^{-3}$（17×17 网格，512 单元）
- **Chebyshev 代理最大误差**：$6.36 \times 10^{-11}$
- **Vandermonde 求积误差**（$e^x$ 在 $[0,1]$）：$8.59 \times 10^{-7}$
- **系统可靠性**：$\approx 0.997$
- **调度 makespan**：$\approx 141$ 秒（20 个任务，4 处理器）

---

## 八、工程鲁棒性与边界处理

- **数值稳定性**：所有除法操作均设置最小正数保护（`max(..., 1e-12)` 或 `1e-15`）
- **矩阵正定性检查**：Cholesky 分解前强制验证对称正定性
- **安全数学运算**：`safe_log`、`safe_sqrt` 防止 `log(0)` 和 `sqrt(负数)`
- **温度约束**：性能降频因子 `max(0, ...)` 保证不会出现负性能
- **边界条件处理**：FEM 中边界节点精确施加 Dirichlet 条件
- **网格质量监控**：自适应细化前检查单元质量，退化单元（面积接近零）自动跳过
