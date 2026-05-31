# 高维约束空间下机械臂实时轨迹规划与动态避障的混合整数非线性优化系统

## 一、项目概述

本项目基于 **15 个科研种子代码项目** 的核心算法，在 **机器人学：机械臂轨迹规划与避障** 领域内融合构建了一个面向前沿科学问题的博士级 Python 计算项目。

**核心科学问题**：
> 在 cluttered 三维工作空间中，一个 7 自由度冗余机械臂如何从初始构型规划出一条满足关节极限、速度/加速度约束、避障要求，且能量最优的光滑轨迹？更进一步，如何在实时计算约束下，通过混合整数决策、无导数在线优化和伪谱配点法，实现从起点到目标轮廓跟踪的完整运动？

该问题涉及高维非凸约束优化、刚性多体动力学、大规模稀疏线性代数、概率图搜索、整数规划与无导数优化等多个博士级研究方向，计算复杂度极高，非普通工程实现可轻易解决。

---

## 二、原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 在合成项目中的真实角色 |
|------|-----------|---------|---------------------|
| 1 | 1283_tough_ode | 刚性非线性ODE测试问题 | **机械臂多体动力学时间推进**：采用 SDIRK（单对角隐式Runge-Kutta）积分器思想，对 Euler-Lagrange 动力学方程 $\tau = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q)$ 进行自适应步长刚性积分 |
| 2 | 151_cg_ne | CGNE共轭梯度正规方程 | **微分逆运动学求解**：求解 $J(q)\Delta q = \Delta x$ 的阻尼最小二乘问题，不显式构造 $J^T J$，用于实时逆运动学 |
| 3 | 077_bernstein_approximation | Bernstein多项式/de Casteljau | **Bézier关节空间轨迹表示**：利用Bernstein基的凸包性质保证轨迹始终在安全区域内，解析计算速度/加速度 |
| 4 | 1344_triangulation_orient | 三角网格定向校正 | **障碍物表面建模**：通过符号面积校正所有三角形为CCW定向，为有向距离场(SDF)查询提供一致法向 |
| 5 | 917_prism_witherden_rule | Witherden棱柱高斯积分 | **障碍物质量属性计算**：将多面体分解为三角棱柱，利用高精度高斯积分规则计算体积、质心和惯性张量 |
| 6 | 1079_simplex_grid | 单形网格生成/组合排序 | **受约束构型空间采样**：将归一化关节坐标映射到单形，生成确定性网格覆盖 |
| 7 | 541_histogram_pdf_sample | 直方图PDF/CDF逆变换采样 | **构型空间非均匀采样**：在障碍物密集方向按直方图PDF加权采样，提高狭窄通道发现率 |
| 8 | 760_mgmres | 重启GMRES + ILU预条件 | **稀疏KKT系统求解**：轨迹优化中的大型稀疏线性系统使用 GMRES(m) + ILU(0) 求解 |
| 9 | 663_legendre_product_display | Gauss-Legendre节点/权重/张量积 | **伪谱法轨迹离散化**：在Legendre-Gauss节点上配点，构造微分矩阵 $D_{ij}$，将连续最优控制问题转化为NLP |
| 10 | 544_hits | HITS权威/枢纽排序 | **PRM瓶颈节点识别**：对概率路线图的边-节点关联矩阵进行SVD-based HITS排序，识别连接不同自由空间区域的关键构型 |
| 11 | 920_profile_data | 2D面部轮廓数据 | **末端轨迹目标**：将40点2D轮廓缩放到机械臂工作空间，作为末端跟踪目标曲线 |
| 12 | 155_change_diophantine | 非负整数Diophantine方程 | **离散控制周期分配**：将总控制周期按关节权重分配到各自由度，求解所有可行整数分配方案 |
| 13 | 1020_reid_tiling | 精确覆盖/RREF | **工作空间离散覆盖**：将网格单元覆盖建模为二元线性系统 $Ax = b$，用RREF求解精确覆盖姿态集合 |
| 14 | 907_praxis | Brent无导数PRAXIS优化 | **在线轨迹微调**：沿主方向（SVD更新）进行无导数搜索，对碰撞惩罚、路径长度和奇异位形进行联合优化 |
| 15 | 224_cplex_solution_read | CPLEX XML解解析 | **混合整数决策读取**：解析MILP求解器的离散走廊选择和连续速度决策结果 |

---

## 三、新增数学物理模型与核心公式

### 3.1 改进DH参数齐次变换

机械臂连杆坐标系的变换矩阵：

$$
^{i-1}T_i = 
\begin{bmatrix}
\cos\theta_i & -\sin\theta_i & 0 & a_i \\
\sin\theta_i\cos\alpha_i & \cos\theta_i\cos\alpha_i & -\sin\alpha_i & -d_i\sin\alpha_i \\
\sin\theta_i\sin\alpha_i & \cos\theta_i\sin\alpha_i & \cos\alpha_i & d_i\cos\alpha_i \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

### 3.2 几何雅可比矩阵

末端速度 $\xi_E = [v_E^T, \omega_E^T]^T$ 与关节速度的关系：

$$
J_i = \begin{bmatrix} z_i \times (p_E - p_i) \\ z_i \end{bmatrix}, \quad
\xi_E = J(q)\dot{q}
$$

### 3.3 Euler-Lagrange 动力学

$$
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = \tau
$$

其中质量矩阵：

$$
M(q) = \sum_{i=1}^{n} \left[ m_i J_{v_i}^T J_{v_i} + J_{\omega_i}^T I_i J_{\omega_i} \right]
$$

### 3.4 CGNE（共轭梯度正规方程）

不显式计算 $A^T A$：

$$
\begin{aligned}
r_0 &= b - Ax_0 \\
z_0 &= A^T r_0, \quad d_0 = z_0 \\
\alpha_k &= \frac{z_k^T z_k}{(Ad_k)^T (Ad_k)} \\
x_{k+1} &= x_k + \alpha_k d_k \\
r_{k+1} &= r_k - \alpha_k A d_k \\
z_{k+1} &= A^T r_{k+1} \\
\beta_k &= \frac{z_{k+1}^T z_{k+1}}{z_k^T z_k} \\
d_{k+1} &= z_{k+1} + \beta_k d_k
\end{aligned}
$$

### 3.5 Bernstein 基与 Bézier 轨迹

定义在 $[A, B]$ 上的 $n$ 次Bernstein基：

$$
B_i^n(t) = \binom{n}{i} \frac{(B-t)^{n-i}(t-A)^i}{(B-A)^n}
$$

Bézier曲线：

$$
q(t) = \sum_{i=0}^{n} P_i B_i^n(t)
$$

de Casteljau 递推（数值稳定）：

$$
P_i^{(r)}(t) = \frac{B-t}{B-A} P_i^{(r-1)}(t) + \frac{t-A}{B-A} P_{i+1}^{(r-1)}(t)
$$

### 3.6 伪谱微分矩阵

在 Legendre-Gauss 节点 $\{\tau_j\}$ 上，状态导数：

$$
\dot{x}(\tau_k) = \sum_{j=0}^{N} D_{kj} x(\tau_j), \quad
D_{ij} = \frac{w_j}{w_i(\tau_i - \tau_j)} \;(i \neq j)
$$

配点约束（含时间域变换 $t = \frac{t_f-t_0}{2}\tau + \frac{t_f+t_0}{2}$）：

$$
\sum_{j} D_{kj} x_j - \frac{t_f-t_0}{2} f(x_k, u_k) = 0
$$

### 3.7 Gauss-Legendre 积分

一维规则：

$$
\int_{-1}^{1} f(x)\,dx \approx \sum_{i=1}^{n} w_i f(x_i), \quad
w_i = \frac{2}{(1-x_i^2)[P_n'(x_i)]^2}
$$

三维张量积：

$$
\iiint f(x,y,z)\,dx\,dy\,dz \approx \sum_{i,j,k} w_i w_j w_k \, f(x_i, y_j, z_k)
$$

### 3.8 三角棱柱上的 Witherden 规则

对单位三角棱柱 $P$（底面为 $(0,0),(1,0),(0,1)$，高为 $[0,1]$）：

$$
\int_P f\,dV \approx V(P) \sum_{i} w_i f(x_i, y_i, z_i), \quad V(P) = 0.5
$$

### 3.9 GMRES(m) + Givens旋转

Arnoldi 过程：

$$
A V_m = V_{m+1} \bar{H}_m
$$

Givens 旋转消去 $\bar{H}_m$ 的次对角元，将最小二乘问题转化为上三角系统。

ILU(0) 不完全LU预条件：

$$
M \approx LU, \quad \text{fill-in 限制在 } A \text{ 的稀疏模式内}
$$

### 3.10 HITS 图排序

对边-节点关联矩阵 $A$：

$$
a^{(k+1)} = \frac{A^T h^{(k)}}{\|A^T h^{(k)}\|_2}, \quad
h^{(k+1)} = \frac{A a^{(k+1)}}{\|A a^{(k+1)}\|_2}
$$

收敛到主奇异向量：$a = |V_{:,0}|$，$h = |U_{:,0}|$。

### 3.11 Diophantine 整数分配

控制周期分配模型：

$$
w_1 x_1 + w_2 x_2 + \cdots + w_n x_n = T_{\text{total}}, \quad x_i \in \mathbb{N}_0
$$

可行性条件：$\gcd(w_1,\dots,w_n) \mid T_{\text{total}}$。

### 3.12 PRAXIS 无导数优化

二次模型：

$$
Q(x') = F(x) + \frac{1}{2}(x'-x)^T A (x'-x), \quad
A = V^{-T} D V^{-1}
$$

方向矩阵 $V$ 定期通过 SVD 更新为主方向：

$$
V = U \Sigma W^T \quad\Rightarrow\quad \text{新搜索方向} = U \text{ 的列}
$$

### 3.13 有向距离场（SDF）

对查询点 $\mathbf{x}$ 和三角形 $(p_0, p_1, p_2)$：

$$
\phi(\mathbf{x}) = \text{sign}\bigl(\mathbf{n} \cdot (\mathbf{x} - p_0)\bigr) \cdot \min_{\mathcal{T}} \|\mathbf{x} - \Pi_{\mathcal{T}}(\mathbf{x})\|
$$

其中 $\Pi_{\mathcal{T}}$ 为点到三角形的最近点投影。

---

## 四、文件结构与实现路径

### 4.1 文件列表（共 12 个 Python 文件）

| 文件 | 功能 | 融合的种子项目 |
|------|------|-------------|
| `main.py` | 统一入口，编排完整管线 | 全部 |
| `kinematics_dynamics.py` | 正向/微分运动学、刚体动力学ODE、CGNE | 1283, 151 |
| `bernstein_path.py` | Bézier轨迹生成、de Casteljau、凸包边界 | 077 |
| `obstacle_geometry.py` | 三角网格定向、棱柱积分、SDF查询 | 1344, 917 |
| `configuration_space.py` | 单形网格、PDF/CDF采样、高斯混合 | 1079, 541 |
| `sparse_linear_algebra.py` | CRS稀疏格式、GMRES(m)、ILU(0)、CGNE | 760, 151 |
| `pseudospectral_control.py` | Legendre节点/权重、微分矩阵、配点约束 | 663 |
| `roadmap_graph.py` | PRM构建、HITS排序、Dijkstra、Profile数据 | 544, 920 |
| `discrete_planning.py` | Diophantine回溯、RREF精确覆盖 | 155, 1020 |
| `derivative_free_opt.py` | PRAXIS主方向法、SVD更新、线搜索 | 907 |
| `milp_parser.py` | CPLEX XML解析、整数/连续变量提取 | 224 |
| `core_planner.py` | 核心规划器，整合所有模块 | 全部 |

### 4.2 运行方式

```bash
cd "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/169_synth_project"
python main.py
```

无需任何参数，运行后将输出：
- PRM路径节点数
- PRAXIS优化状态
- Diophantine分配方案数
- MILP走廊选择结果
- ODE动力学仿真步数
- 伪谱速度代价积分
- 可操纵性度量统计
- 边界约束验证结果

---

## 五、科学问题与工程复杂度说明

### 5.1 科学难度

本项目落地的科学问题属于**机器人学中的运动规划（Motion Planning）**，是公认的NP-hard问题：

1. **高维构型空间**：7自由度机械臂的构型空间为 $\mathbb{T}^7$（7维环面），在连续空间中搜索避障路径的计算复杂度随维度指数增长。
2. **非凸约束耦合**：关节极限（箱约束）、自碰撞、障碍物避碰、速度/加速度/力矩约束同时存在，形成高度非凸的可行域。
3. **微分包含**：轨迹必须满足 Euler-Lagrange 动力学方程，这是一个微分代数方程（DAE）系统，离散化后产生大规模稀疏KKT系统。
4. **混合整数决策**：避障走廊选择等离散决策使问题成为混合整数非线性规划（MINLP），属于计算复杂性理论中最难的优化类别之一。

### 5.2 工程鲁棒性

- **边界处理**：所有关节角严格裁剪到 $[-\pi, \pi]$；Bézier控制点通过 `clamp_control_points_to_joint_limits` 利用凸包性质保证整条轨迹不超出安全区。
- **数值稳定性**：de Casteljau递推避免显式计算大阶乘组合数；GMRES使用修正Gram-Schmidt和Givens旋转保持Arnoldi向量正交性；SDIRK使用Newton迭代和步长自适应控制刚性积分误差。
- **退化处理**：雅可比奇异时通过阻尼最小二乘（damping）保证数值可逆；HITS排序在图不连通时回退到均匀分布；PRAXIS检测到ill-conditioning时进行随机步进扰动。
- **有限性检查**：main.py中对所有轨迹采样点进行 `np.isfinite` 断言，确保无NaN/Inf溢出。

### 5.3 公式-算法-代码一致性

每个核心公式都在代码中有直接对应实现：
- DH变换矩阵 → `ManipulatorKinematics._mdh_transform()`
- CGNE迭代 → `cgne_solve()` / `cgne_solve_wrapper()`
- Bernstein递推 → `bernstein_basis()`
- de Casteljau → `de_casteljau()`
- 伪谱微分矩阵 → `pseudospectral_differentiation_matrix()`
- Gauss-Legendre节点 → `legendre_compute()`
- GMRES+Givens → `gmres_solve()`
- HITS SVD → `hits_svd()`
- Diophantine回溯 → `diophantine_nd_solutions()`
- RREF → `rref_compute()`
- PRAXIS SVD更新 → `PraxisOptimizer.minimize()`
- SDF → `PolyhedralObstacle.signed_distance()`

---

## 六、结论

本项目成功将15个独立的科研代码项目融合为一个统一的、可执行的博士级机械臂轨迹规划系统。系统涵盖：
- **运动学与动力学**：MDH参数、几何雅可比、Euler-Lagrange方程、SDIRK刚性积分
- **轨迹表示**：Bernstein-Bézier曲线、凸包安全保证、解析导数
- **障碍物建模**：三角网格定向校正、棱柱高斯积分、有向距离场
- **构型空间**：单形网格、PDF逆变换采样、高斯混合
- **数值代数**：稀疏CRS格式、GMRES(m)、ILU(0)预条件、CGNE
- **最优控制**：Gauss-Legendre伪谱配点、微分矩阵、代价积分
- **图搜索**：PRM、HITS瓶颈识别、Dijkstra最短路径
- **离散优化**：Diophantine回溯、精确覆盖RREF
- **在线优化**：PRAXIS无导数主方向法
- **决策解析**：CPLEX MILP解读取

代码总计约 **3000+ 行 Python**，全部零参数可运行，具备完整的边界处理和数值鲁棒性。
