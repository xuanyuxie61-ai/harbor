# 智能电网潮流优化与暂态稳定性分析综合平台

## 项目概述

本项目是一个面向**能源系统：智能电网潮流优化**的博士级科研计算平台。项目基于 15 个 MATLAB 种子项目的核心算法，融合重构为一个统一入口（`main.py`）、零参数可运行的 Python 科研计算系统。平台涵盖电网拓扑建模、牛顿-拉夫逊潮流计算、暂态稳定性时域仿真、最优经济调度与机组组合动态规划、负荷马尔可夫预测、加权最小二乘状态估计、可靠性分析与电压稳定裕度评估等核心模块。

---

## 一、种子项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 合成后的角色 |
|------|----------|----------|-------------|
| 1 | `178_circle_distance` | 圆上随机点距离 PDF | 线路可靠性模型中的几何概率与弧长故障率映射 |
| 2 | `1330_triangulation` | Delaunay 三角剖分 (r8tris2) | 电网拓扑几何建模，节点连接的平面剖分优化 |
| 3 | `176_circle_arc_grid` | 圆弧网格生成 | 环形配电网母线节点的参数化坐标生成 |
| 4 | `158_change_polynomial` | 多项式卷积（生成函数） | 多机组容量聚合的动态规划生成函数法 |
| 5 | `1298_triangle_analyze` | 三角形几何分析 | 三相不平衡度的相量三角形几何判据 |
| 6 | `998_r8st` | 稀疏矩阵 CG 迭代求解 | 大规模潮流修正方程的稀疏迭代求解器 |
| 7 | `568_i4lib` | 整数矩阵 RREF | 电网可观测性分析的整数秩判定 |
| 8 | `216_control_bio_homework` | 最优控制框架 | 最优潮流控制的数学规划思想迁移 |
| 9 | `549_humps` | 非线性 ODE 测试函数 | 暂态仿真中非线性摇摆方程的数值积分 |
| 10 | `281_diff2_center` | 中心差分二阶导数 | 潮流雅可比矩阵 Hessian 项的数值验证 |
| 11 | `716_markov_letters` | 马尔可夫双字母频率 | 电力负荷离散时间马尔可夫链预测模型 |
| 12 | `1421_xyf_display` | 拓扑数据结构管理 | 电网节点-边邻接表与连接性管理 |
| 13 | `156_change_dynamic` | 动态规划 | 机组组合问题的多时段状态转移 DP |
| 14 | `1139_spring_ode` | 弹簧阻尼 ODE | 多机电力系统机电振荡摇摆方程 |
| 15 | `759_mesh2d_write` | 网格数据 I/O | 电网拓扑连接性与元素数据结构 |

---

## 二、新增数学物理模型与核心公式

### 2.1 电网拓扑与 Delaunay 三角剖分

节点坐标参数化（圆弧网格）：
\[
x(\theta) = c_x + r \cos\theta, \quad y(\theta) = c_y + r \sin\theta
\]

Delaunay 空外接圆准则（二维）：对三角形顶点 \(i,j,k\)（逆时针排列）和任意其他节点 \(p\)：
\[
\det \begin{bmatrix}
x_p & y_p & x_p^2+y_p^2 & 1 \\
x_i & y_i & x_i^2+y_i^2 & 1 \\
x_j & y_j & x_j^2+y_j^2 & 1 \\
x_k & y_k & x_k^2+y_k^2 & 1
\end{bmatrix} > 0
\]

### 2.2 节点导纳矩阵

对每条线路阻抗 \(z = r + jx\)，导纳 \(y = 1/z = g - jb\)：
\[
Y_{ii} = \sum_{k \in N(i)} y_{ik} + j\frac{b_{shunt}}{2}, \quad
Y_{ij} = -y_{ij} \;(i \neq j)
\]

### 2.3 牛顿-拉夫逊潮流方程（极坐标）

\[
P_i = |V_i| \sum_{j} |V_j| \left( G_{ij} \cos\theta_{ij} + B_{ij} \sin\theta_{ij} \right)
\]
\[
Q_i = |V_i| \sum_{j} |V_j| \left( G_{ij} \sin\theta_{ij} - B_{ij} \cos\theta_{ij} \right)
\]

雅可比子块（标准形式）：
\[
\begin{aligned}
H_{ii} &= \frac{\partial P_i}{\partial \theta_i} = -Q_i - B_{ii}|V_i|^2 \\
N_{ii} &= \frac{\partial P_i}{\partial |V_i|} |V_i| = P_i + G_{ii}|V_i|^2 \\
J_{ii} &= \frac{\partial Q_i}{\partial \theta_i} = P_i - G_{ii}|V_i|^2 \\
L_{ii} &= \frac{\partial Q_i}{\partial |V_i|} |V_i| = Q_i - B_{ii}|V_i|^2
\end{aligned}
\]

牛顿迭代格式（带阻尼线搜索）：
\[
x_{k+1} = x_k + \alpha_k \Delta x, \quad \alpha_k \in (0,1]
\]
其中 \(\alpha_k\) 通过监控失配范数单调性自适应选取。

### 2.4 暂态稳定性摇摆方程

单机无穷大系统：
\[
M \frac{d^2\delta}{dt^2} + D \frac{d\delta}{dt} = P_m - P_e, \quad
M = \frac{2H}{\omega_s}
\]
\[
P_e(\delta) = \frac{E' V_\infty}{X} \sin\delta
\]

状态空间形式：
\[
\frac{d}{dt} \begin{bmatrix} \delta \\ \omega \end{bmatrix}
= \begin{bmatrix} \omega - \omega_s \\ (P_m - P_e - D(\omega-\omega_s))/M \end{bmatrix}
\]

四阶龙格-库塔单步积分：
\[
\begin{aligned}
k_1 &= h f(t_n, y_n) \\
k_2 &= h f(t_n + h/2, y_n + k_1/2) \\
k_3 &= h f(t_n + h/2, y_n + k_2/2) \\
k_4 &= h f(t_n + h, y_n + k_3) \\
y_{n+1} &= y_n + \frac{k_1 + 2k_2 + 2k_3 + k_4}{6}
\end{aligned}
\]

**等面积法则（Equal-Area Criterion）**：临界切除角 \(\delta_{cr}\) 满足
\[
\int_{\delta_0}^{\delta_{cr}} P_m \,d\delta
= \int_{\delta_{cr}}^{\delta_{max}} \left( P_{max}^{post} \sin\delta - P_m \right) d\delta
\]

### 2.5 经济调度（等微增率准则）

二次成本函数：
\[
C_i(P_{G,i}) = a_i P_{G,i}^2 + b_i P_{G,i} + c_i
\]

最优性条件（未达边界的机组）：
\[
\frac{dC_i}{dP_{G,i}} = 2a_i P_{G,i} + b_i = \lambda
\]

解析解：
\[
P_{G,i}(\lambda) = \frac{\lambda - b_i}{2a_i}
\]

\(\lambda\) 通过二分搜索确定，使 \(\sum P_{G,i} = P_D\)。

### 2.6 机组组合动态规划

状态 \(s_t = (u_t, \tau_t)\)，其中 \(u_t \in \{0,1\}\) 为启停状态，\(\tau_t\) 为持续时段数。

Bellman 方程：
\[
V_t(u, \tau) = C_t(u) + \min_{u'} \left\{ V_{t-1}(u', \tau') + C_{switch}(u' \to u) \right\}
\]

### 2.7 负荷马尔可夫链

离散状态转移矩阵：
\[
P_{ij} = \Pr(X_{t+1} = j \mid X_t = i), \quad \sum_j P_{ij} = 1
\]

熵率：
\[
H(P) = -\sum_i \pi_i \sum_j P_{ij} \log_2 P_{ij}
\]

### 2.8 WLS 状态估计

量测方程：
\[
z = h(x) + \varepsilon, \quad \varepsilon \sim \mathcal N(0, R)
\]

目标函数：
\[
J(x) = [z - h(x)]^T R^{-1} [z - h(x)]
\]

高斯-牛顿迭代：
\[
\Delta x = (H^T R^{-1} H)^{-1} H^T R^{-1} [z - h(x)]
\]

### 2.9 电压稳定性（PV 曲线鼻点）

\[
P = \frac{EV}{X} \sin\delta, \quad
Q = \frac{EV}{X} \cos\delta - \frac{V^2}{X}
\]

消去 \(\delta\) 后得到关于 \(V^2\) 的二次方程。鼻点判据（判别式为零）：
\[
\Delta = (2QX - E^2)^2 - 4X^2(P^2 + Q^2) = 0
\]

电压裕度：
\[
\text{Margin} = \frac{P_{max} - P_0}{P_{max}} \times 100\%
\]

### 2.10 三相不平衡度（对称分量法）

\[
\begin{bmatrix} V_0 \\ V_+ \\ V_- \end{bmatrix}
= \frac{1}{3} \begin{bmatrix}
1 & 1 & 1 \\
1 & \alpha & \alpha^2 \\
1 & \alpha^2 & \alpha
\end{bmatrix}
\begin{bmatrix} V_a \\ V_b \\ V_c \end{bmatrix}, \quad
\alpha = e^{j2\pi/3}
\]

不平衡度：
\[
\varepsilon = \frac{|V_-|}{|V_+|} \times 100\%
\]

---

## 三、项目文件结构

```
165_synth_project/
├── main.py                  # 统一入口，零参数运行
├── utils.py                 # 通用数值工具（圆弧网格、多项式卷积、中心差分、RREF、RK4）
├── grid_topology.py         # Delaunay 三角剖分与电网拓扑建模
├── sparse_matrix.py         # 稀疏矩阵 COO 存储、CG 与 Jacobi 迭代求解
├── power_flow.py            # 牛顿-拉夫逊潮流计算（带阻尼线搜索）
├── transient_stability.py   # 单机/多机摇摆方程与暂态稳定性仿真
├── optimal_dispatch.py      # 经济调度（等微增率）与机组组合 DP
├── load_markov.py           # 负荷马尔可夫预测与熵率分析
├── state_estimation.py      # WLS 状态估计与可观测性分析
├── reliability.py           # 线路可靠性、电压稳定裕度、三相不平衡度
└── README_博士级合成说明.md  # 本文档
```

---

## 四、运行方式

```bash
cd 165_synth_project
python3 main.py
```

程序无需任何命令行参数，自动执行全部模块并输出计算结果。

---

## 五、质量检查

- [x] 原目录未被修改
- [x] 合成项目为 Python 语言
- [x] 包含 10 个 `.py` 文件（≥8 个）
- [x] `main.py` 零参数可运行，执行无报错
- [x] 已删除所有可视化相关内容
- [x] 代码具备边界处理与数值鲁棒性
- [x] 包含大量数学物理公式
- [x] 中文说明文档已生成
- [x] 每个输入种子项目均已真实融入
