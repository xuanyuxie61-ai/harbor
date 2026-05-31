# 六足机器人多足步态优化系统 —— 博士级合成说明

## 一、项目概述

本项目围绕 **"机器人学：多足机器人步态优化"** 领域，基于 15 个种子科研代码项目的核心算法，融合构建了一个面向前沿科学问题的博士级 Python 计算系统。

### 核心科学问题

多足机器人在非结构化地形上的自适应步态优化是一个高度复杂的耦合动力学-优化问题，涉及：
- **非线性振荡器网络**（CPG）的同步与相位锁定
- **大规模稀疏/块结构线性系统**的高效数值求解
- **混合连续-离散动力学**（支撑/摆动自动机）的稳定分析
- **非凸多模态参数空间**的全局混沌优化
- **接触力学**（摩擦锥、碰撞检测）的实时约束处理

该系统集成了运动学、动力学、地形建模、轨迹规划、稳定性分析与全局优化六大子系统，构成了一个完整的多足机器人步态优化科研平台。

---

## 二、原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法/思想 | 合成后角色 |
|:---:|-----------|-------------|-----------|
| 1 | `1310_triangle_io` | 三角形网格文件（.node/.ele）读写 | **地形三角网格建模与 I/O**：`terrain_model.py` 中 `TriangulatedTerrain.from_node_element()` 实现地形网格的持久化读写，用于足部-地形碰撞检测 |
| 2 | `1288_trapezoidal_fixed` | 固定点迭代梯形法 ODE 求解 | **步态动力学积分器**：`gait_dynamics.py` 中 `TrapezoidalIntegrator` 对 CPG 耦合振荡器网络进行时间积分，局部截断误差 $O(h^3)$ |
| 3 | `671_life` | Conway 生命游戏细胞自动机 | **支撑-摆动离散状态机**：`gait_dynamics.py` 中 `StanceSwingAutomaton` 将生命游戏的局部邻居交互思想映射为多足机器人支撑/摆动相位的离散切换规则 |
| 4 | `1417_wtime` | 墙钟计时器 | **性能计时与算法评估**：`utils.py` 中 `Timer` 类提供高精度计时，用于各数值模块的性能基准测试 |
| 5 | `726_matlab_mistake` | MATLAB 常见数值/逻辑错误示例 | **数值稳定性检测框架**：`utils.py` 中 `check_numerical_singularity`、`safe_divide`、`robust_sqrt` 等函数系统处理除零、奇异矩阵、负值开方等边界条件 |
| 6 | `1418_xml2struct` | XML 递归解析器 | **机器人 URDF/SDF 配置解析**：`config_parser.py` 中 `xml2dict()` 递归将 XML 节点树映射为嵌套字典，提取 link 质量、惯性张量、关节限位等物理参数 |
| 7 | `1168_stla_to_tri_surface_fast` | ASCII STL 文件快速解析 | **足部三维模型解析**：`terrain_model.py` 中 `TriangulatedTerrain.from_stl_ascii()` 快速读取机器人足部 STL 模型为三角网格，用于精确接触面计算 |
| 8 | `1115_sphere_distance` | 球面距离概率密度 | **关节空间测地距离约束**：`robot_kinematics.py` 中 `JointLimitConstraint` 将球面测地距离概念映射为关节旋转空间的极限约束，定义软约束势函数梯度 |
| 9 | `845_pagerank2` | 稀疏邻接矩阵构建 | **支撑图中心性分析**：`stability_optimizer.py` 中 `SupportGraphCentrality` 将支撑状态建模为有向图，用 PageRank 特征向量中心性评估各腿对整体稳定性的贡献 |
| 10 | `786_nas` | NASA 基准（Cholesky、FFT、块三对角、矩阵乘法） | **大规模数值线性代数引擎**：`numerical_solver.py` 中分别实现 `CholeskySolver`（质量矩阵分解）、`BlockTridiagonalSolver`（多腿耦合系统）、`Radix2FFT`（步态频域分析）、`MatrixMultiplyBenchmark`（基础运算） |
| 11 | `1358_trinity` | 三角网格区域线性规划约束 | **支撑多边形 LP 约束**：`stability_optimizer.py` 中 `LinearStabilityConstraint` 将凸包区域转化为半空间表示 $A \cdot p \leq b$，作为 COM/ZMP 可行域的硬约束 |
| 12 | `1363_tsp_brute` | 旅行商问题暴力求解 | **足端落点排序优化**：`trajectory_planner.py` 中 `TSPBruteForce` 将候选落点集的最优访问顺序建模为 TSP，最小化重心偏移与能量消耗 |
| 13 | `1004_r8vm` | Vandermonde 矩阵求解 | **足端摆动多项式轨迹插值**：`trajectory_planner.py` 中 `PolynomialSwingTrajectory` 用 Vandermonde 型约束系统求解 5 次多项式系数，保证轨迹 $C^2$ 连续 |
| 14 | `956_quadrilateral_surface_display` | 四边形表面双线性插值 | **四边形地形单元插值**：`terrain_model.py` 中 `QuadrilateralTerrainPatch` 实现双线性映射 $p(\xi,\eta)$ 与 Jacobian/法向量计算，去除可视化保留数学核心 |
| 15 | `655_leaf_chaos` | Barnsley 蕨类迭代函数系统（IFS） | **混沌全局步态参数优化**：`chaotic_search.py` 中 `BarnsleyFernIFS` 与 `ChaoticSimulatedAnnealing` 利用混沌遍历性在参数空间进行全局探索，避免梯度法陷入局部最优 |

---

## 三、新增数学物理模型与核心公式

### 3.1 改进 DH 参数正向运动学

对 3-DOF 串联腿，第 $i$ 个连杆的齐次变换矩阵（Craig 约定）：

$$
^{i-1}T_i = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

足端位姿 $T_{ee} = \prod_{i=1}^{3} {}^{i-1}T_i$。

### 3.2 几何 Jacobian

线速度部分的 Jacobian：

$$
J_v = \begin{bmatrix} z_0 \times (p_{ee}-p_0) & z_1 \times (p_{ee}-p_1) & z_2 \times (p_{ee}-p_2) \end{bmatrix}
$$

其中 $z_i$ 为关节轴方向，$p_i$ 为关节原点。

### 3.3 阻尼最小二乘逆运动学

迭代公式：

$$
\Delta q = J^T \left( J J^T + \lambda^2 I \right)^{-1} \Delta x
$$

$\lambda$ 为阻尼系数，保证奇异构型附近的数值稳定性。

### 3.4 耦合 Hopf 振荡器 CPG

单个 Hopf 极限环：

$$
\begin{aligned}
\dot{x} &= \alpha(\mu - r^2)x - \omega y \\
\dot{y} &= \alpha(\mu - r^2)y + \omega x
\end{aligned}
\quad \text{其中 } r^2 = x^2 + y^2
$$

$N$ 个振荡器耦合：

$$
\begin{aligned}
\dot{x}_i &= \alpha(\mu - r_i^2)x_i - \omega_i y_i + \sum_j c_{ij}(x_j - x_i) \\
\dot{y}_i &= \alpha(\mu - r_i^2)y_i + \omega_i x_i + \sum_j c_{ij}(y_j - y_i)
\end{aligned}
$$

### 3.5 梯形法固定点迭代

对 $\dot{y} = f(t,y)$：

$$
y_{n+1}^{(k+1)} = y_n + \frac{h}{2}\left[ f(t_n, y_n) + f(t_{n+1}, y_{n+1}^{(k)}) \right]
$$

局部截断误差 $O(h^3)$，全局误差 $O(h^2)$。

### 3.6 单腿动力学（简化）

$$
M\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau + J^T f_c
$$

其中 $M$ 为惯性矩阵，$C$ 为科氏力与离心力矩阵，$G$ 为重力项，$f_c$ 为足端接触力。

### 3.7 Cholesky 分解

对 SPD 矩阵 $A = LL^T$：

$$
\begin{aligned}
L_{ii} &= \sqrt{A_{ii} - \sum_{k=0}^{i-1} L_{ik}^2} \\
L_{ji} &= \frac{1}{L_{ii}}\left( A_{ji} - \sum_{k=0}^{i-1} L_{jk}L_{ik} \right), \quad j > i
\end{aligned}
$$

### 3.8 块三对角系统求解（Thomas 块版本）

对块三对角系统 $A_i x_{i-1} + B_i x_i + C_i x_{i+1} = d_i$，前向消元：

$$
\begin{aligned}
U_{i-1} &= L_{i-1}^{-1} C_{i-1} \\
L_i &= B_i - A_i U_{i-1}
\end{aligned}
$$

### 3.9 基-2 Cooley-Tukey FFT

DFT 分解：

$$
\begin{aligned}
X_k &= E_k + e^{-i2\pi k/N} O_k \\
X_{k+N/2} &= E_k - e^{-i2\pi k/N} O_k
\end{aligned}
$$

复杂度从 $O(N^2)$ 降至 $O(N \log N)$。

### 3.10 Vandermonde 多项式插值

Vandermonde 矩阵 $V_{ij} = x_i^{j-1}$，插值条件 $Vc = y$。

递推求解算法复杂度 $O(n^2)$，远优于通用高斯消元 $O(n^3)$。

### 3.11 5 次多项式摆动轨迹

边界条件（$C^2$ 连续）：

$$
\begin{aligned}
p(0) &= p_s, & p(T) &= p_e \\
\dot{p}(0) &= v_s, & \dot{p}(T) &= v_e \\
\ddot{p}(0) &= a_s, & \ddot{p}(T) &= a_e
\end{aligned}
$$

约束矩阵为 $6 \times 6$ 的广义 Vandermonde 型矩阵。

### 3.12 零力矩点（ZMP）

简化平面模型：

$$
\begin{aligned}
x_{ZMP} &= x_{COM} - \frac{z_{COM} \cdot a_x}{g} \\
y_{ZMP} &= y_{COM} - \frac{z_{COM} \cdot a_y}{g}
\end{aligned}
$$

### 3.13 PageRank 中心性

幂迭代：$r^{(k+1)} = \alpha M r^{(k)} + (1-\alpha)v$。

### 3.14 混沌模拟退火

Metropolis 接受准则：

$$
P(\text{accept}) = \begin{cases}
1 & \Delta E \leq 0 \\
\exp(-\Delta E / T) & \Delta E > 0
\end{cases}
$$

扰动由 Logistic 混沌映射驱动：$x_{n+1} = r x_n (1-x_n)$，$r \in [3.57, 4]$。

### 3.15 Gershgorin 圆盘定理

特征值 $\lambda$ 必位于某个圆盘：

$$
D_i = \{ z \in \mathbb{C} : |z - a_{ii}| \leq R_i \}, \quad R_i = \sum_{j \neq i} |a_{ij}|
$$

### 3.16 摩擦锥约束（Coulomb）

$$
\|f_t\| \leq \mu f_n
$$

其中 $f_t = f_c - (f_c \cdot n)n$ 为切向分量，$f_n = f_c \cdot n$ 为法向分量。

---

## 四、项目文件结构

```
167_synth_project/
├── main.py                     # 统一入口，零参数运行
├── utils.py                    # 数值稳定性、计时、Gershgorin、Householder
├── config_parser.py            # XML 配置解析（xml2struct 映射）
├── terrain_model.py            # 三角网格地形、四边形插值、STL 解析
├── robot_kinematics.py         # DH 运动学、Jacobian、逆解、关节限位、接触几何
├── gait_dynamics.py            # CPG 网络、梯形法积分、支撑自动机、腿动力学
├── numerical_solver.py         # Cholesky、块三对角、FFT、Vandermonde、矩阵乘法
├── trajectory_planner.py       # TSP 落点排序、5 次多项式轨迹、综合规划器
├── stability_optimizer.py      # 支撑多边形、ZMP、PageRank、LP 约束
├── chaotic_search.py           # Logistic 混沌、IFS、混沌模拟退火、参数优化
└── README_博士级合成说明.md     # 本文档
```

共 **10 个 .py 文件**，超过最低 8 个的要求。

---

## 五、合成后项目能解决的科学问题

1. **多足机器人在非平坦地形上的自适应步态生成**：通过 CPG 网络与地形反馈耦合，实现六足机器人的三足交替步态（tripod gait）自动生成与相位锁定。

2. **足部-地面接触力学实时分析**：基于三角网格地形与四边形补丁的双线性插值，实现足端高度查询、法向量计算与摩擦锥约束检验。

3. **全身动力学高效数值求解**：利用 Cholesky 分解与块三对角求解器，处理多腿耦合的大规模线性系统，支持实时仿真需求。

4. **步态稳定性定量评估**：通过支撑多边形凸包、ZMP 计算、支撑图 PageRank 中心性，提供多尺度的稳定性度量。

5. **全局步态参数优化**：利用混沌遍历性（Logistic 映射 + IFS）在高度非凸的参数空间中寻找接近最优的步态周期、步幅、抬腿高度与耦合强度组合。

6. **足端轨迹光滑规划**：基于 TSP 优化的落点排序与 5 次多项式插值，保证摆动相的 $C^2$ 连续性，避免冲击与滑移。

---

## 六、运行方式

```bash
cd Synthesis-project-python/167_synth_project
python main.py
```

程序将自动执行以下完整流程：
1. 解析机器人 URDF XML 配置
2. 正向/逆向运动学计算与接触几何分析
3. 生成并查询示例非平坦地形
4. CPG 网络积分与支撑-摆动状态机演化
5. Cholesky / 块三对角 / FFT / Vandermonde 数值求解演示
6. TSP 落点排序与多项式摆动轨迹生成
7. 支撑多边形、ZMP、PageRank 稳定性分析
8. 混沌全局步态参数优化
9. 全局数值鲁棒性检查

**无需输入任何参数，无可视化输出。**

---

## 七、边界处理与数值鲁棒性

本系统在以下关键位置实施了严格的边界处理：

- **除零保护**：`safe_divide`、`robust_sqrt` 对奇异点进行截断与回退
- **矩阵奇异性检测**：`check_numerical_singularity` 通过条件数与零对角双重检测
- **关节限位裁剪**：所有关节角通过 `clip_to_bounds` 强制进入物理可行区间
- **凸包退化处理**：`SupportPolygon` 对 0/1/2 个点的退化情况进行单独分支
- **三角形面积为零**：地形查询中跳过退化三角形
- **Cholesky 非正定性**：加入 `eps` 扰动保证平方根内为正
- **FFT 长度检查**：强制要求输入长度为 2 的幂，否则报错
- **Vandermonde 重复节点**：显式检测并返回错误码
- **CPG 积分稳定性**：梯形法固定点迭代保证隐式格式的数值稳定性
- **混沌映射参数边界**：Logistic 参数 $r$ 被裁剪至混沌区间 $[3.57, 4.0]$

---

## 八、复审结论

- [x] 原目录未被修改
- [x] 合成后项目为 Python 语言
- [x] 新目录完整包含合成后项目（10 个 .py 文件 + 1 个文档）
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] **15 个输入项目全部真实融入**，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成
- [x] 已删除所有可视化相关内容
