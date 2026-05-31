# 深海半潜式平台非线性水动力响应与疲劳可靠性综合分析系统

## README — 博士级合成说明文档

---

## 一、项目概述

本项目基于 **15 个科研代码种子项目**，围绕 **计算流体力学：海洋平台水动力响应** 这一前沿领域，融合重构为一个面向深海半潜式平台非线性水动力响应与疲劳可靠性分析的博士级综合计算系统。

项目涵盖以下核心科学问题：

1. **势流理论面板法**：计算波浪绕射/辐射水动力系数（附加质量、辐射阻尼）；
2. **非线性时域响应**：BDF2 隐式积分求解平台六自由度运动方程；
3. **压力泊松方程与速度势**：SOR 迭代求解流场压力修正与 Laplace 速度势；
4. **自适应网格生成**：CVT Lloyd 算法优化平台周围节点分布，Delaunay 三角剖分质量控制；
5. **方向波谱与随机波浪**：JONSWAP 频谱 + Mitsuyasu 方向扩散，驱动蒙特卡洛波面合成；
6. **疲劳累积损伤与可靠性**：雨流计数 + Miner 法则 + 马尔可夫链海况转移模型；
7. **波数离散化约束**：丢番图方程约束 Floquet-Bloch 模式与 Bragg 共振条件；
8. **稀疏矩阵优化**：R8NCF 坐标格式 + RCM 重排序，支撑大规模 FEM/CFD 线性系统。

---

## 二、原项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 合成后角色 |
|:---:|---------|---------|-----------|
| 1 | `566_hypersphere_monte_carlo` | 超球面均匀采样、Gamma 函数矩积分 | **wave_spectrum.py**：波向积分蒙特卡洛、方向分布球面采样 |
| 2 | `676_line_cvt_lloyd` | 一维 Lloyd 算法、CVT 能量泛函 | **mesh_geometry.py**：平台周围垂向节点自适应优化分布 |
| 3 | `074_bdf2` | BDF2 隐式多步法、Newton 非线性求解 | **platform_dynamics.py**：平台六自由度时域动力响应积分 |
| 4 | `849_partition_brute` | 组合分区优化、子集枚举 | **platform_dynamics.py**：6-DOF 子系统负载均衡分区 |
| 5 | `1039_rng_cliff` | Cliff 混沌伪随机数生成器 | **wave_spectrum.py**：随机波浪相位确定性生成 |
| 6 | `1349_triangulation_rcm` | Reverse Cuthill-McKee 带宽缩减 | **sparse_matrix.py**：稀疏矩阵 RCM 重排序预处理 |
| 7 | `877_poisson_2d` | 五点差分 + Jacobi 迭代 | **poisson_solver.py**：二维泊松方程离散与求解框架 |
| 8 | `986_r8ncf` | 坐标格式稀疏矩阵、SpMV | **sparse_matrix.py**：R8NCF 稀疏矩阵存储与运算 |
| 9 | `155_change_diophantine` | N 维丢番图方程回溯求解 | **spectral_analysis.py**：波数空间整数约束离散化 |
| 10 | `1335_triangulation_delaunay_discrepancy` | Delaunay 不一致度、局部角度优化 | **mesh_geometry.py**：水线面三角剖分质量评估 |
| 11 | `1080_simplex_integrals` | 单纯形精确矩积分、Dirichlet 采样 | **mesh_geometry.py**：三角形 FEM 质量矩阵精确积分 |
| 12 | `455_gaussian_2d` | 各向异性二维高斯核 | **wave_spectrum.py**：方向扩散函数各向异性建模 |
| 13 | `1095_snakes_probability` | 马尔可夫链转移矩阵、矩阵幂方法 | **fatigue_reliability.py**：海况转移概率与稳态分布 |
| 14 | `882_polygon` | 多边形面积、形心、惯性矩、固角、三角剖分 | **mesh_geometry.py** + **hydrodynamics.py**：水线面水静力学、面板法几何 |
| 15 | `1099_sor` | 逐次超松弛 SOR 迭代 | **poisson_solver.py**：压力泊松方程 SOR 加速求解 |

**每一个输入项目均已真实融入合成项目**，无遗漏、无挂名。

---

## 三、新增数学物理模型与核心公式

### 3.1 JONSWAP 方向波谱模型

频率谱（JONSWAP）：

$$
S(f) = \alpha g^2 (2\pi)^{-4} f^{-5} \exp\!\left[-\frac{5}{4}\left(\frac{f}{f_p}\right)^{-4}\right] \gamma^{\exp\left[-\frac{(f-f_p)^2}{2\sigma^2 f_p^2}\right]}
$$

其中 $\alpha = 0.0081$，$\gamma = 3.3$，$\sigma = 0.07\;(f \le f_p)$ 或 $0.09\;(f > f_p)$。

方向扩散函数（Mitsuyasu 型，各向异性高斯）：

$$
D(\theta) = G(n) \left|\cos\frac{\theta - \theta_m}{2}\right|^{2n}, \quad \int_{-\pi}^{\pi} D(\theta)\,d\theta = 1
$$

方向波谱：

$$
S(f,\theta) = S(f) \cdot D(\theta)
$$

### 3.2 线性 Airy 波运动学

色散关系（有限水深）：

$$
\omega^2 = gk \tanh(kh)
$$

入射波速度势：

$$
\phi_I = \frac{Ag}{\omega} \frac{\cosh[k(z+h)]}{\cosh(kh)} \sin(kx\cos\beta + ky\sin\beta - \omega t)
$$

水平/垂向速度：

$$
u = \frac{\partial\phi_I}{\partial x}, \quad w = \frac{\partial\phi_I}{\partial z}
$$

### 3.3 势流理论面板法

格林函数（三维自由表面，镜像法近似）：

$$
G(\mathbf{x}, \boldsymbol{\xi}) = \frac{1}{|\mathbf{x}-\boldsymbol{\xi}|} + \frac{1}{|\mathbf{x}-\boldsymbol{\xi}^*|}
$$

附加质量与辐射阻尼：

$$
A_{ij}(\omega) = -\rho \,\mathrm{Re}\!\left[\iint_S \phi_j \, n_i \, dS\right], \quad
B_{ij}(\omega) = \rho\omega \,\mathrm{Im}\!\left[\iint_S \phi_j \, n_i \, dS\right]
$$

### 3.4 六自由度平台运动方程

Cummins 时域方程：

$$
(\mathbf{M} + \mathbf{A}_\infty)\ddot{\boldsymbol{\xi}}(t) + \int_0^\infty \mathbf{K}(\tau)\dot{\boldsymbol{\xi}}(t-\tau)\,d\tau + \mathbf{C}\boldsymbol{\xi}(t) = \mathbf{F}_{\text{wave}}(t) + \mathbf{F}_{\text{moor}}(t)
$$

其中：
- $\mathbf{M}$：6×6 刚体质量/惯性矩阵
- $\mathbf{A}_\infty$：无限频率附加质量
- $\mathbf{K}(t)$：延迟函数（Retardation function）
- $\mathbf{C}$：静水恢复力矩阵
- $\boldsymbol{\xi} = [\xi_1, \xi_2, \xi_3, \xi_4, \xi_5, \xi_6]^T = [\text{surge}, \text{sway}, \text{heave}, \text{roll}, \text{pitch}, \text{yaw}]^T$

静水恢复力矩阵关键元素：

$$
C_{33} = \rho g A_{wp}, \quad C_{44} = \rho g I_{xx} + \rho g V_{disp}(z_{cob} - z_{cog}), \quad C_{55} = \rho g I_{yy} + \rho g V_{disp}(z_{cob} - z_{cog})
$$

### 3.5 BDF2 隐式时间积分

BDF2 步进公式：

$$
\frac{3\mathbf{y}^{n+1} - 4\mathbf{y}^n + \mathbf{y}^{n-1}}{2\Delta t} = \mathbf{f}(t^{n+1}, \mathbf{y}^{n+1})
$$

起步步采用 Backward Euler + 外推校正：

$$
\mathbf{y}_h = \text{BE}(\Delta t/2), \quad \mathbf{y}^1 = 2\mathbf{y}_h - \mathbf{y}^0
$$

非线性方程使用阻尼不动点迭代求解。

### 3.6 压力泊松方程

不可压缩 Navier-Stokes 的压力泊松方程：

$$
\nabla^2 p = \rho \left(\frac{\partial u_i}{\partial x_j}\right)\!\left(\frac{\partial u_j}{\partial x_i}\right)
$$

离散化（五点差分）：

$$
\frac{u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_{i,j}}{dx \cdot dy} = f_{i,j}
$$

SOR 迭代：

$$
u_i^{\text{new}} = (1-\omega)\nu_i + \frac{\omega}{a_{ii}}\left[b_i - \sum_{j<i} a_{ij}\nu_j^{\text{new}} - \sum_{j>i} a_{ij}\nu_j^{\text{old}}\right]
$$

### 3.7 系泊力（悬链线模型）

悬链线方程：

$$
L = \frac{H}{w}\sinh\!\left(\frac{wx_f}{H}\right) + \frac{EA}{w}\left[\text{arsinh}\!\left(\frac{wx_f}{H}\right) - \frac{wx_f}{H}\right]
$$

其中 $H$ 为水平张力，$w$ 为单位长度线重，$L$ 为索长，$x_f$ 为水平跨距。

### 3.8 疲劳累积损伤（Miner 法则）

S-N 曲线（Basquin 方程）：

$$
N = a \cdot S^{-m}
$$

Miner 线性累积损伤：

$$
D = \sum_i \frac{n_i}{N_i} = \sum_i \frac{n_i}{a \cdot S_i^{-m}}
$$

当 $D \ge 1$ 时判定为疲劳失效。

### 3.9 马尔可夫链海况转移模型

离散状态转移：

$$
\boldsymbol{\pi}^{(t+1)} = \boldsymbol{\pi}^{(t)} \mathbf{P}
$$

稳态分布满足：

$$
\boldsymbol{\pi} = \boldsymbol{\pi}\mathbf{P}, \quad \sum_s \pi_s = 1
$$

长期期望年疲劳损伤率：

$$
\mathbb{E}[D] = \sum_s \pi_s D_s
$$

### 3.10 丢番图波数约束

Floquet-Bloch 波数离散化：

$$
\mathbf{k} = \left(\frac{h_1 \cdot 2\pi}{L_x}, \frac{h_2 \cdot 2\pi}{L_y}\right), \quad h_1, h_2 \in \mathbb{Z}
$$

Bragg 共振条件（立柱阵列）：

$$
2k\cos\theta = n \cdot \frac{2\pi}{L}, \quad n \in \mathbb{Z}^+
$$

丢番图方程：

$$
a_1 x_1 + a_2 x_2 + \cdots + a_m x_m = b, \quad x_i \in \mathbb{Z}_{\ge 0}
$$

### 3.11 水线面水静力学

多边形面积（Shoelace 公式）：

$$
A = \frac{1}{2}\sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i)
$$

二阶矩（用于稳心高度）：

$$
I_{xx} = \int_\Omega y^2\,dA, \quad I_{yy} = \int_\Omega x^2\,dA
$$

稳心半径：

$$
BM = \frac{I}{V_{disp}}
$$

### 3.12 单纯形精确积分

m 维单位单纯形上的单矩积分：

$$
\int_\Delta \prod_{i=1}^m x_i^{e_i}\,dV = \frac{\prod_{i=1}^m \Gamma(e_i+1)}{\Gamma\bigl(m + \sum e_i + 1\bigr)}
$$

### 3.13 RCM 带宽缩减

Reverse Cuthill-McKee 重排序通过 BFS 层次遍历最小化稀疏矩阵带宽：

$$
\text{bandwidth}(A) = \max_{a_{ij} \neq 0} |i - j| + 1
$$

---

## 四、项目文件结构与说明

```
079_synth_project/
├── main.py                          # 统一入口，零参数运行
├── utils.py                         # 通用工具：Gamma、安全反余弦、丢番图校验、Cliff RNG
├── wave_spectrum.py                 # 波谱模型：JONSWAP、方向扩散、Airy 波、群速度
├── mesh_geometry.py                 # 网格与几何：CVT Lloyd、多边形几何、Delaunay 检查、单纯形积分
├── sparse_matrix.py                 # 稀疏矩阵：R8NCF 格式、SpMV、Laplacian 构建、RCM 重排序
├── poisson_solver.py                # 泊松求解：SOR、Jacobi、压力泊松、Laplace 速度势
├── platform_dynamics.py             # 平台动力学：BDF2 积分、系泊力、DOF 分区优化
├── hydrodynamics.py                 # 水动力：面板法、Morison 方程、Airy 运动学
├── fatigue_reliability.py           # 疲劳可靠性：雨流计数、Miner 损伤、马尔可夫链、可靠度
└── spectral_analysis.py             # 谱分析：丢番图约束、响应谱 RA0、谱带宽参数
```

---

## 五、如何运行

在终端中执行：

```bash
cd "Synthesis-project-python/079_synth_project"
python main.py
```

程序将自动完成以下 10 个分析阶段并输出结果：

1. **平台几何建模与水静力学特性**（水线面面积、形心、惯性矩、稳心半径）
2. **自适应网格生成与质量评估**（CVT 节点优化、Delaunay 质量检查、FEM 精确积分）
3. **稀疏矩阵构建与 RCM 带宽优化**（Laplacian 矩阵、带宽缩减比）
4. **势流理论水动力系数计算**（附加质量、辐射阻尼、波浪力传递函数）
5. **方向波谱合成与波浪运动学**（JONSWAP 谱、方向扩散、Airy 波速度/压力）
6. **平台六自由度动力响应时域分析**（BDF2 隐式积分、位移/转角统计）
7. **压力泊松方程与速度势求解**（SOR 迭代、压力场、Laplace 速度势）
8. **疲劳累积损伤与可靠性分析**（雨流计数、Miner 损伤、马尔可夫链、可靠度指标 β）
9. **丢番图波数约束与谱分析**（整数解枚举、Bragg 共振、Floquet-Bloch 模式、响应谱）
10. **综合分析结果汇总**

---

## 六、边界处理与数值鲁棒性

1. **Gamma 函数**：使用 Lanczos 近似对数域计算，防止大参数溢出；
2. **反余弦函数**：输入严格限制在 $[-1, 1]$，防止浮点误差导致 `ValueError`；
3. **Cliff RNG**：非法种子自动回退到 0.3，NaN 检测与重播种；
4. **BDF2 积分**：阻尼不动点迭代 + 发散检测 + 物理边界截断（位移 ±50 m，转角 ±0.5 rad）；
5. **系泊力**：牛顿迭代中 $H$ 下限保护、大参数 `sinh` 截断、平台位移限制 ±200 m；
6. **SOR 求解**：对角元非零检测、松弛因子范围校验 (0, 2)；
7. **稀疏矩阵**：索引越界检查、排列维度一致性校验；
8. **丢番图方程**：GCD 可除性预检查、非负约束验证；
9. **波数计算**：`tanh` 大参数截断、深水/有限水深自动切换。

---

## 七、科学问题与工程价值

本项目合成的计算系统可直接用于以下前沿科研与工程问题：

- **深海半潜式平台概念设计阶段的水动力性能快速评估**
- **极端海况下平台非线性耦合响应的时域预报**
- **疲劳关键节点的长期损伤累积与寿命预测**
- **基于马尔可夫链的可靠度分析与风险决策**
- **波浪-结构-系泊耦合系统的数值仿真与优化**
- **Floquet-Bloch 理论在海洋结构物周期性阵列中的 Bragg 共振分析**

---

*文档生成时间：2026-05-05*
*合成依据：sci-project-synthesis-python skill 规范*
