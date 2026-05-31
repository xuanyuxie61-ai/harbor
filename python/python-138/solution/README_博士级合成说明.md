# PROJECT_138：微反应器混合与反应强化多尺度计算框架

## 1. 项目概述

本项目围绕**化学工程：微反应器混合与反应强化**这一前沿领域，基于 15 个种子科研代码项目的核心算法，融合构建了一个面向博士级科学计算的多尺度数值框架。

微反应器（Microreactor）因其极高的比表面积、优异的传热传质性能，在现代过程强化（Process Intensification）中占据核心地位。本项目涵盖从分子尺度的反应动力学参数估计，到介观尺度的催化剂分布优化与混合质量评估，再到宏观尺度的反应器网络拓扑设计、稳态 PDE 求解、稳定性分析及薄壁结构热应力评估，形成了一个完整的计算链条。

---

## 2. 科学问题与核心公式

### 2.1 对流-扩散-反应耦合控制方程

微通道内的浓度场 $C(x,t)$ 与温度场 $T(x,t)$ 满足：

$$
\frac{\partial C}{\partial t} + u \frac{\partial C}{\partial x} = D_m \frac{\partial^2 C}{\partial x^2} - r(C,T)
$$

$$
\rho c_p \frac{\partial T}{\partial t} + \rho c_p u \frac{\partial T}{\partial x} = \lambda \frac{\partial^2 T}{\partial x^2} + (-\Delta H) r(C,T) - \frac{4 h_w}{d_h}(T - T_w)
$$

其中 Arrhenius 反应速率为：

$$
r(C,T) = A \exp\!\left(-\frac{E_a}{RT}\right) C^n
$$

稳态求解采用内部 Newton-Raphson 隐式迭代，对流项采用一阶迎风格式保证单调性，扩散项为中心差分。

**无量纲特征数：**

$$
\text{Pe} = \frac{u L}{D_m}, \quad \text{Da} = \frac{A \exp(-E_a/(RT_{\text{avg}})) L}{u}
$$

### 2.2 催化剂最优分布——Centroidal Voronoi Tessellation (CVT)

将催化剂颗粒的空间排布建模为 CVT 能量最小化问题：

$$
E = \sum_{i=1}^{N} \int_{V_i} \rho(\mathbf{x}) \|\mathbf{x} - \mathbf{z}_i\|^2 \, d\mathbf{x}
$$

最优条件为 Lloyd 映射不动点：

$$
\mathbf{z}_i = \frac{\int_{V_i} \rho(\mathbf{x}) \mathbf{x} \, d\mathbf{x}}{\int_{V_i} \rho(\mathbf{x}) \, d\mathbf{x}}
$$

算法采用 Monte Carlo 采样近似积分，通过 Lloyd 迭代逼近 CVT。

### 2.3 反应动力学参数估计——QR 非线性最小二乘

直接在原尺度上优化目标泛函：

$$
\min_{A, E_a, n} \sum_{i=1}^{m} \left[ r_i^{\text{exp}} - A \exp\!\left(-\frac{E_a}{R T_i}\right) C_i^n \right]^2
$$

采用 Gauss-Newton / Levenberg-Marquardt 迭代：

$$
(J^T J + \lambda I) \delta = -J^T \mathbf{f}
$$

其中 Jacobian $J$ 的列向量为：

$$
\frac{\partial r}{\partial \ln A} = r, \quad \frac{\partial r}{\partial E_a} = -\frac{r}{RT}, \quad \frac{\partial r}{\partial n} = r \ln C
$$

初值通过格点搜索（$n \in [0, 2.5]$）结合线性回归获得，避免局部极小陷阱。

### 2.4 稳态线性稳定性分析

在稳态解 $(C_s, T_s)$ 附近引入小扰动 $(c', \theta')$，线性化后得到：

$$
\frac{d}{dt} \begin{bmatrix} c' \\ \theta' \end{bmatrix} = J \begin{bmatrix} c' \\ \theta' \end{bmatrix}
$$

Jacobian $J$ 为 $2N_x \times 2N_x$ 分块矩阵：

$$
J = \begin{bmatrix}
A_c - \text{diag}(\partial r / \partial C) & -\text{diag}(\partial r / \partial T) \\
\dfrac{-\Delta H}{\rho c_p} \text{diag}(\partial r / \partial C) & A_T + \dfrac{-\Delta H}{\rho c_p} \text{diag}(\partial r / \partial T) - \beta_w I
\end{bmatrix}
$$

稳定性判据：
- 若 $\max \text{Re}(\lambda_k) < 0$：稳态渐近稳定
- 若 $\exists k: \text{Re}(\lambda_k) > 0$：存在热失控风险

热爆炸风险指数定义为：

$$
I_{TE} = \tanh\bigl( \max(0, \text{Re}(\lambda_{\max})) \cdot \tau_{\text{res}} \bigr)
$$

### 2.5 质量-能量耦合平衡定点迭代

对 CSTR 型微反应器，联立稳态方程并构造定点映射：

$$
G(T) = \frac{\rho c_p Q T_0 + (-\Delta H) Q C_0 + UA \, T_c - (-\Delta H) Q \, C(T)}{\rho c_p Q + UA}
$$

其中 $C(T)$ 由质量平衡隐式确定。迭代格式为 $T^{(n+1)} = G(T^{(n)})$，并引入阻尼因子避免振荡。

分岔指示器：

$$
\left| \frac{dG}{dT} \right|_{ss} < 1 \; \Rightarrow \; \text{局部收敛}
$$

### 2.6 POD 降阶模型与 Modified Gram-Schmidt

对快照矩阵 $U = [\mathbf{u}_1, \ldots, \mathbf{u}_m]$ 执行 Modified Gram-Schmidt：

$$
\mathbf{v}_k = \mathbf{u}_k - \sum_{j=1}^{k-1} \langle \mathbf{v}_k, \boldsymbol{\phi}_j \rangle \boldsymbol{\phi}_j, \quad \boldsymbol{\phi}_k = \frac{\mathbf{v}_k}{\|\mathbf{v}_k\|}
$$

POD 模态通过 SVD 截断获得，能量占比为：

$$
\eta_{\text{POD}} = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{m} \sigma_i^2}
$$

### 2.7 混合质量多变量统计度量

定义混合缺陷指标：

$$
M_d = \frac{\mathbb{E}[\|\mathbf{c} - \mu \mathbf{1}\|_2]}{\sigma \sqrt{d}}
$$

KL 散度衡量实际分布与理想多元正态分布的差异：

$$
D_{KL} = \frac{1}{2}\Bigl[ \text{tr}(\Sigma_0^{-1}\Sigma) + (\boldsymbol{\mu} - \boldsymbol{\mu}_0)^T \Sigma_0^{-1} (\boldsymbol{\mu} - \boldsymbol{\mu}_0) - d + \ln\frac{\det \Sigma_0}{\det \Sigma} \Bigr]
$$

两区域间的 Mahalanobis 距离：

$$
D_M = \sqrt{(\boldsymbol{\mu}_a - \boldsymbol{\mu}_b)^T \Sigma^{-1} (\boldsymbol{\mu}_a - \boldsymbol{\mu}_b)}, \quad \Sigma = \frac{\Sigma_a + \Sigma_b}{2}
$$

### 2.8 操作条件优化

**黄金分割搜索：** 对单峰目标函数在区间 $[a,b]$ 上寻找最优值，收缩比为黄金比例：

$$
\tau = \frac{\sqrt{5} - 1}{2} \approx 0.618
$$

**Newton 法：** 使用精确 Hessian 的阻尼 Newton 迭代，搜索方向由 $H \mathbf{p} = -\nabla f$ 给出，线搜索采用 Armijo 条件。

### 2.9 稀疏 Skyline 矩阵运算

对称稀疏 skyline (R8SS) 格式存储下三角列条带。矩阵-向量乘法按列遍历：

$$
y_j += a_{kj} x_j, \quad y_i += a_{kj} x_j, \quad y_j += a_{kj} x_i
$$

其中 $i$ 遍历列 $j$ 的非对角元行号。

### 2.10 反应器网络拓扑

微反应器网络建模为有向图 $G=(V,E)$。Eulerian 路径存在定理：

- 闭合回路：$\forall v, \text{indegree}(v) = \text{outdegree}(v)$
- 开放路径：恰有一个节点出度比入度大 1，一个节点入度比出度大 1

生成树计数采用 Kirchhoff 矩阵树定理：

$$
\tau(G) = \det(L_{n-1})
$$

其中 $L_{n-1}$ 为删除最后一行一列的 Laplacian。

### 2.11 薄板热应力分析（双调和方程）

薄板挠度 $w(x,y)$ 满足双调和方程：

$$
\nabla^4 w = \frac{\partial^4 w}{\partial x^4} + 2 \frac{\partial^4 w}{\partial x^2 \partial y^2} + \frac{\partial^4 w}{\partial y^4} = 0
$$

构造解析试函数：

$$
w(x,y) = \bigl[ a \cosh(gx) + b \sinh(gx) + cx \cosh(gx) + dx \sinh(gx) \bigr] \bigl[ e \cos(gy) + f \sin(gy) \bigr]
$$

该函数精确满足 $\nabla^4 w = 0$。热应力通过曲率计算：

$$
\sigma_x = -\frac{Ez}{1-\nu^2}(\kappa_x + \nu \kappa_y) - \frac{E \alpha_T \Delta T}{1-\nu}
$$

von Mises 等效应力：

$$
\sigma_{vm} = \sqrt{\sigma_x^2 + \sigma_y^2 - \sigma_x \sigma_y + 3 \tau_{xy}^2}
$$

### 2.12 离散催化剂负载整数规划

催化剂在各区域的整数分配满足丢番图方程：

$$
\sum_{i=1}^{n} a_i x_i = B, \quad x_i \in \mathbb{Z}_{\geq 0}
$$

采用递归回溯枚举所有非负整数解，结合分支限界进行剪枝。贪心启发式按效益比 $w_i / a_i$ 降序填充。

---

## 3. 原项目映射与角色分配

| 原种子项目 | 核心算法 | 合成项目角色 |
|---|---|---|
| `1060_schroedinger_linear_pde` | 线性 PDE 求解、守恒量监测、参数管理 | `reactor_pde_solver.py`：对流-扩散-反应 PDE 稳态求解器 |
| `238_cvt` | Centroidal Voronoi Tessellation (Lloyd 迭代) | `catalyst_placement_cvt.py`：催化剂空间分布优化 |
| `938_qr_solve` | QR 分解、最小二乘求解 | `kinetics_parameter_estimation.py`：反应动力学参数 QR/非线性最小二乘估计 |
| `172_chladni_figures` | 双调和特征值问题、Laplacian 离散 | `stability_eigenanalysis.py`：稳态 Jacobian 特征值稳定性分析 |
| `194_cobweb_plot` | 定点迭代、收敛性分析 | `mass_energy_balance.py`：质量-能量耦合平衡定点迭代求解 |
| `480_gram_schmidt` | Gram-Schmidt 正交化 | `reduced_order_basis.py`：MGS 正交基构造与 POD 降阶 |
| `819_normal01_multivariate_distance` | 多元正态距离统计 | `mixing_quality_statistics.py`：混合质量多变量统计度量 |
| `439_florida_cvt_geo` | 地理约束 CVT、几何质心计算 | `catalyst_placement_cvt.py`：边界框约束与密度加权质心计算 |
| `1222_test_opt` | 优化测试问题、梯度/Hessian | `reactor_optimization.py`：操作条件 Newton 优化与 Rosenbrock 测试 |
| `997_r8ss` | Skyline 对称稀疏矩阵-向量乘法 | `sparse_matrix_ops.py`：R8SS 格式稀疏矩阵运算 |
| `476_golden_section` | 黄金分割搜索 | `reactor_optimization.py`：单参数停留时间优化 |
| `286_digraph_arc` | 有向图 Eulerian 路径判定 | `network_topology.py`：微反应器网络流路 Eulerian 分析 |
| `482_graph_arc` | 无向图 Pruefer 编码/解码、生成树 | `network_topology.py`：网络树拓扑设计与生成树计数 |
| `087_biharmonic_exact` | 双调和方程精确解、残差验证 | `thermal_stress_analysis.py`：薄板双调和热应力解析评估 |
| `289_diophantine_nd` | N 维丢番图非负整数解枚举 | `discrete_catalyst_loading.py`：离散催化剂负载整数规划 |

---

## 4. 文件结构与运行方式

### 4.1 文件清单

```
138_synth_project/
├── main.py                           # 统一入口，零参数运行
├── reactor_pde_solver.py             # PDE 稳态求解器
├── catalyst_placement_cvt.py         # CVT 催化剂分布优化
├── kinetics_parameter_estimation.py  # 动力学参数 QR/非线性估计
├── stability_eigenanalysis.py        # 线性稳定性特征值分析
├── mass_energy_balance.py            # 质量-能量定点迭代
├── reduced_order_basis.py            # MGS/POD 降阶基
├── mixing_quality_statistics.py      # 混合质量统计
├── reactor_optimization.py           # 黄金分割 + Newton 优化
├── sparse_matrix_ops.py              # Skyline 稀疏矩阵运算
├── network_topology.py               # 网络拓扑与 Eulerian 分析
├── thermal_stress_analysis.py        # 双调和热应力分析
├── discrete_catalyst_loading.py      # 离散催化剂整数规划
└── README_博士级合成说明.md           # 本文档
```

### 4.2 运行方法

```bash
cd 138_synth_project
python main.py
```

程序无需任何输入参数，运行后将依次执行 12 个计算模块，输出各模块的关键结果与汇总报告。

---

## 5. 工程鲁棒性设计

1. **参数边界校验**：所有模块在构造函数中对物理参数（长度、浓度、温度、流量等）进行合法性检查，非法输入抛出 ValueError。
2. **数值稳定性**：
   - PDE 求解器对非正浓度/温度做安全截断
   - 反应速率设置上界限制避免指数爆炸
   - 动力学估计采用格点搜索 + Gauss-Newton 双重策略避免局部极小
   - 稳定性分析边界条件采用单位化固定而非极端大数
3. **迭代收敛保护**：
   - 定点迭代引入阻尼因子和物理上界（温度上限 2000 K）
   - Newton 法配备回溯线搜索与 Hessian 不正定回退机制
   - CVT Lloyd 迭代对空单元进行随机重置
4. **矩阵运算安全**：
   - Thomas 算法除零保护
   - Cholesky 失败自动回退 LU
   - Skyline 矩阵维度严格校验
