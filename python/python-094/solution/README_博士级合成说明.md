# README_博士级合成说明.md

## 项目概述

**项目名称**：非线性声学冲击波传播高阶数值模拟与自适应降阶建模系统

**科学领域**：声学工程 — 非线性声学 shock wave 传播

**合成目标**：基于 15 个科研代码种子项目的核心算法，融合构建一个面向前沿声学工程问题的博士级 Python 科研计算项目。项目围绕水中高频超声（1 MHz）非线性传播中的冲击波形成、能量级联、熵产生等物理现象，集成了谱方法、有限体积激波捕捉、Strang 分裂、SVD 降阶模型、CVT 网格优化、高维数值积分、传感器阵列优化、自适应网格细分、矩阵链动态规划优化、混沌随机采样以及数值完整性校验等先进算法。

---

## 一、原项目到科学问题的映射

本项目将 15 个输入种子项目的核心算法全部真实融入合成系统，无一遗漏、无一挂名。具体映射关系如下：

| 序号 | 种子项目 | 核心算法 | 科学融合角色 |
|------|----------|----------|--------------|
| 1 | `1004_r8vm` | Vandermonde 矩阵紧凑存储、Björck-Pereyra 求解、行列式、矩阵-向量乘法 | `spectral_solver.py`：构造 Legendre/Chebyshev 谱微分矩阵，用于 Burgers 方程的高阶空间离散 |
| 2 | `250_cvt_3d_sampling` | 3D CVT Lloyd 算法 | `mesh_generator.py`：在声学计算域内生成 CVT 优化网格，提升冲击波前沿区域的空间分辨率 |
| 3 | `1365_tsp_greedy` | 贪心算法求解旅行商问题 | `sensor_optimizer.py`：优化声学传感器阵列的测量路径，最小化总行程 |
| 4 | `1146_square_hex_grid` | 六边形网格生成 | `mesh_generator.py`：作为初始网格生成器，为 CVT 优化提供种子点 |
| 5 | `740_matrix_chain_dynamic` | 动态规划求解矩阵链最优括号化 | `matrix_chain_optimizer.py`：优化声学算子链（微分-质量-刚度矩阵）的乘法顺序，降低 FLOPs |
| 6 | `1294_tri_surface_to_obj` | 三角面元到 OBJ 格式转换、法向量计算 | `geometry_utils.py`：计算声学散射体（NACA 翼型）表面的三角面元法向量，支持射线反射模拟 |
| 7 | `1039_rng_cliff` | Cliff 混沌随机数生成器 | `random_generator.py`：用于 Monte Carlo 积分中的高质量伪随机采样，以及 LHS 分层采样 |
| 8 | `704_luhn` | Luhn 校验和算法 | `luhn_checksum_adapter.py`：改造为浮点数组与网格拓扑的数值完整性校验工具，保障大规模并行计算的数据一致性 |
| 9 | `1298_triangle_analyze` | 三角形几何分析（面积、角度、外接圆、内切圆、质量因子） | `geometry_utils.py`：评估声学网格质量；`triangle_refiner.py`：作为自适应细分的几何判据 |
| 10 | `785_naca` | NACA 4位对称/弯度翼型生成 | `geometry_utils.py`：生成水下声学实验中的标准翼型边界，模拟非定常流动诱发的声散射 |
| 11 | `1314_triangle_refine` | 三角形递归细分与数值积分 | `triangle_refiner.py`：在冲击波陡峭梯度区域进行自适应细分，提高局部积分精度 |
| 12 | `1346_triangulation_q2l` | 二次三角剖分到线性转换 | `triangle_refiner.py`：将 6 节点二次声学单元转换为 4 个 3 节点线性单元，兼容显式时间推进格式 |
| 13 | `239_cvt_1_movie` | CVT 迭代（find_closest, cvt_iterate） | `mesh_generator.py`、`sensor_optimizer.py`：提供高效的最近邻搜索与 Lloyd 迭代核心 |
| 14 | `1187_svd_fingerprint` | SVD 分解与低秩近似 | `svd_rom.py`：对冲击波时空场进行 POD 降阶，提取主导模态，构建 Galerkin 投影 ROM |
| 15 | `805_nintlib` | 多维 Romberg 积分、Monte Carlo 积分、tuple 生成 | `romberg_integrator.py`：计算声束总能量、频谱能量密度等高维积分；`main.py` 中演示验证 |

---

## 二、新增数学物理模型与核心公式

### 2.1 非线性 Burgers 方程（一维冲击波模型）

一维粘性 Burgers 方程是描述有限振幅声波在非线性介质中传播形成冲击波的最简模型：

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu_{\mathrm{eff}} \frac{\partial^2 u}{\partial x^2}
$$

其中 $u$ 为质点速度，$\nu_{\mathrm{eff}}$ 为包含热粘滞耗散的有效粘性系数。代码中使用 Chebyshev-Gauss-Lobatto (CGL) 节点构造谱微分矩阵 $D$，通过 RK4 时间推进求解。

### 2.2 KZK 方程（轴对称非线性声束）

Khokhlov-Zabolotskaya-Kuznetsov (KZK) 方程描述弱非线性、弱衍射的轴对称声束传播：

$$
\frac{\partial p}{\partial z} = \frac{1}{2 k_0} \nabla_{\perp}^2 p + \frac{\delta}{2 c_0^3} \frac{\partial^2 p}{\partial \tau^2} + \frac{\beta}{\rho_0 c_0^3} p \frac{\partial p}{\partial \tau}
$$

三项分别对应：**衍射**、**热粘滞吸收**、**非线性对流**。代码中使用 **Strang 分裂**：
1. 衍射半步 $\Delta z/2$（显式 Crank-Nicolson 近似）
2. 非线性+吸收整步 $\Delta z$（谱方法 RK4）
3. 衍射半步 $\Delta z/2$

### 2.3 冲击波形成距离（Fubini-Ghiron 解）

对于平面波，Fubini 解析解给出冲击波形成距离：

$$
x_s = \frac{1}{\beta \, k_0 \, M_0}, \quad M_0 = \frac{p_0}{\rho_0 c_0^2}
$$

其中 $\beta = 1 + B/(2A)$ 为介质非线性系数，$M_0$ 为声源 Mach 数。程序自动计算此距离作为物理特征尺度。

### 2.4 Gol'dberg 数

衡量非线性与吸收竞争的无量纲数：

$$
N_G = \frac{1}{\alpha_{\mathrm{cl}} \, x_s}
$$

当 $N_G \gg 1$ 时非线性主导，冲击波陡峭化显著；当 $N_G \ll 1$ 时吸收主导，波形保持光滑。

### 2.5 Tait 状态方程（高压水介质）

描述水在高压下的非线性压缩性：

$$
p = B \left[ \left( \frac{\rho}{\rho_0} \right)^{\gamma} - 1 \right]
$$

参数：$\gamma = 7.0$，$B = 3.046 \times 10^8$ Pa。由状态方程导出局部非线性声速：

$$
c(p) = c_0 + \frac{\beta \, p}{\rho_0 c_0}
$$

### 2.6 熵产生率（Rankine-Hugoniot 框架）

冲击波面的熵产生率通过速度梯度估计：

$$
\dot{s} = \frac{\rho_0 \nu}{T_0} \int_{\Sigma} \left( \frac{\partial u}{\partial x} \right)^2 dx
$$

### 2.7 谱微分矩阵

对于 CGL 节点 $x_j = \cos(\pi j / N)$，一阶谱微分矩阵元素为：

$$
D_{ij} = \frac{c_i}{c_j} \frac{(-1)^{i+j}}{x_i - x_j}, \quad i \neq j; \qquad c_0 = c_N = 2, \; c_j = 1 \; (j \neq 0, N)
$$

$$
D_{ii} = -\frac{x_i}{2(1 - x_i^2)}, \quad 0 < i < N
$$

### 2.8 SVD-POD 降阶模型

快照矩阵 $S \in \mathbb{R}^{N_{\mathrm{space}} \times N_{\mathrm{time}}}$ 的 SVD：

$$
S = U \Sigma V^T = \sum_{i=1}^{R} \sigma_i u_i v_i^T
$$

$r$ 阶 POD 近似及相对误差：

$$
S_r = U_r \Sigma_r V_r^T, \qquad \varepsilon_r = \frac{\|S - S_r\|_F}{\|S\|_F} = \sqrt{\frac{\sum_{i=r+1}^{R} \sigma_i^2}{\sum_{i=1}^{R} \sigma_i^2}}
$$

### 2.9 Godunov 数值通量（Burgers 方程）

对于 $f(u) = u^2/2$，Godunov 通量为：

$$
F(u_L, u_R) = \begin{cases}
\min_{u \in [u_L, u_R]} f(u), & u_L \le u_R \\
\max_{u \in [u_R, u_L]} f(u), & u_L > u_R
\end{cases}
$$

### 2.10 高维声能量积分

声束总能量（轴对称+延迟时间）：

$$
E = \int_0^{r_{\max}} \int_0^{z_{\max}} \int_{-\tau_{\max}}^{\tau_{\max}}
\frac{p^2(r, z, \tau)}{2 \rho_0 c_0^2} \, 2\pi r \, d\tau \, dz \, dr
$$

分别使用 Monte Carlo 与 Romberg 多维积分求解。

### 2.11 矩阵链最优括号化（动态规划）

对于矩阵链 $A_1 \times A_2 \times \dots \times A_n$，最优子结构：

$$
m[i, j] = \min_{i \le k < j} \left\{ m[i, k] + m[k+1, j] + d_{i-1} d_k d_j \right\}
$$

### 2.12 CVT 能量函数与 Voronoi 分割

CVT 最小化能量泛函：

$$
\mathcal{E}(\{z_i\}) = \int_{\Omega} \min_i \|x - z_i\|^2 \rho(x) \, dx
$$

Lloyd 迭代将生成器逐步移向 Voronoi 单元的质心。

---

## 三、合成项目文件结构与实现路径

### 3.1 文件清单（共 13 个 Python 文件）

| 文件名 | 功能 | 融合的种子项目 |
|--------|------|--------------|
| `main.py` | 统一入口，零参数运行，协调 12 个阶段完整流程 | 全部 |
| `shock_physics.py` | 非线性声学物理模型（Burgers/KZK/Tait/熵产生） | 物理基础 |
| `mesh_generator.py` | 六边形网格 + CVT Lloyd 优化 | 1146, 250, 239 |
| `spectral_solver.py` | Vandermonde 谱方法 + 谱微分矩阵 | 1004 |
| `nonlinear_pde_solver.py` | Strang 分裂 KZK 求解器 + Godunov FV 激波捕捉 | 集成平台 |
| `romberg_integrator.py` | 多维 Romberg / Monte Carlo 积分 | 805 |
| `svd_rom.py` | SVD-POD 降阶模型 + DMD | 1187 |
| `sensor_optimizer.py` | CVT 传感器布局 + TSP 贪心路径优化 | 1365, 239, 250 |
| `geometry_utils.py` | NACA 翼型 + 三角形分析 + 三角面元处理 | 785, 1298, 1294 |
| `triangle_refiner.py` | 三角形递归细分 + Q2L 转换 + AMR | 1314, 1346 |
| `matrix_chain_optimizer.py` | 矩阵链动态规划优化 | 740 |
| `random_generator.py` | Cliff RNG + LHS 采样 | 1039 |
| `luhn_checksum_adapter.py` | 浮点数组校验和 + 拓扑完整性校验 | 704 |

### 3.2 运行方式

```bash
cd Synthesis-project-python/094_synth_project
python main.py
```

无需任何命令行参数。程序自动执行以下 12 个阶段：

1. **物理参数初始化**：水中 1 MHz 超声，500 kPa 峰值压力，计算冲击波形成距离、Gol'dberg 数等特征量。
2. **网格生成**：六边形网格 + CVT Lloyd 优化，评估网格质量。
3. **几何边界**：NACA 0012 翼型生成，三角形几何质量分析。
4. **Burgers 谱求解**：Ricker 子波初值，Chebyshev 谱微分 + RK4，Vandermonde 验证。
5. **KZK 方程求解**：高斯声束初值，Strang 分裂，5000 步轴向传播。
6. **激波捕捉校验**：Godunov 有限体积格式处理方波强间断。
7. **声能量积分**：Monte Carlo 与 Romberg 三维积分对比。
8. **SVD-POD 降阶**：对 Burgers 解提取主导模态，评估压缩比与误差。
9. **传感器优化**：12 传感器 CVT 布局 + TSP 贪心路径规划。
10. **三角形 AMR**：自适应细分与二次-线性单元转换。
11. **矩阵链优化**：声学算子链的动态规划最优顺序。
12. **随机采样与校验**：Cliff RNG、LHS、Luhn 风格数值完整性校验。

---

## 四、合成后项目解决的科学问题

### 4.1 核心科学问题

本项目面向**非线性声学冲击波传播的全流程数值模拟与降阶分析**，解决以下博士级科学问题：

1. **冲击波形成动力学**：通过 Burgers 方程与 KZK 方程的耦合求解，定量描述高频超声在水中从弱非线性波形到强冲击波的过渡过程，包括波形陡峭化、谐波级联、熵产生等关键物理机制。

2. **多尺度数值方法融合**：在一套系统中同时集成谱方法（高阶光滑区精度）、有限体积激波捕捉（强间断稳定性）、Strang 算子分裂（多物理过程解耦），实现从波长尺度到冲击波厚度尺度的跨尺度模拟。

3. **高维能量与不确定性量化**：使用 Romberg 外推与 Monte Carlo 采样计算三维声能量积分，为声学剂量评估与生物效应预测提供定量工具。

4. **数据驱动的降阶建模**：通过 SVD-POD 从全阶数值解中提取主导时空模态，构建 Galerkin 投影降阶模型，为实时参数扫描、在线监测与数字孪生提供计算加速路径。

5. **最优实验设计**：将 CVT 空间覆盖理论与 TSP 路径规划结合，解决声学传感器阵列的最优布局与测量路径设计问题，提升实验效率与场重构精度。

6. **计算性能优化**：通过矩阵链动态规划最小化算子应用 FLOPs，通过自适应网格细化在冲击波前沿集中计算资源，实现高效、鲁棒的工程级模拟。

### 4.2 边界处理与数值鲁棒性

- **物理边界**：Dirichlet 零边界（Burgers）、轴对称 Neumann 边界（KZK 径向）、衰减边界（tau 方向）。
- **CFL 条件**：自动计算并限制时间步长，设置 $N_z$ 上限防止内存爆炸。
- **数值截断**：压力上限截断（1 GPa）、速度上限检查（10 倍声速）、非有限值检测与异常恢复。
- **状态校验**：Tait 方程奇点规避、Mach 数范围检查、网格节点唯一性检查、Vandermonde 奇异性检查。
- **数据完整性**：Luhn 风格浮点校验和与统计一致性校验，保障大规模计算结果可信。

---

## 五、关键公式索引

| 公式名称 | 所在文件 | 用途 |
|----------|----------|------|
| Burgers 方程 | `shock_physics.py` | 一维冲击波演化 |
| KZK 方程 | `shock_physics.py`, `nonlinear_pde_solver.py` | 轴对称声束传播 |
| Fubini 冲击波形成距离 | `shock_physics.py` | 物理特征尺度 |
| Gol'dberg 数 | `shock_physics.py` | 非线性/吸收竞争 |
| Tait 状态方程 | `shock_physics.py` | 高压介质压缩性 |
| 谱微分矩阵 (CGL) | `spectral_solver.py` | 高阶空间离散 |
| Vandermonde 行列式 | `spectral_solver.py` | 矩阵条件数评估 |
| Godunov 通量 | `nonlinear_pde_solver.py` | 激波捕捉 |
| Strang 分裂 | `nonlinear_pde_solver.py` | 多算子解耦 |
| Romberg 外推 | `romberg_integrator.py` | 高维积分加速 |
| SVD-POD 低秩近似 | `svd_rom.py` | 降阶建模 |
| DMD 特征动力学 | `svd_rom.py` | 模态稳定性分析 |
| CVT 能量泛函 | `mesh_generator.py`, `sensor_optimizer.py` | 最优空间覆盖 |
| 矩阵链 DP | `matrix_chain_optimizer.py` | 运算顺序优化 |
| NACA 翼型厚度 | `geometry_utils.py` | 声学边界几何 |
| 三角形质量因子 | `geometry_utils.py` | 网格质量评估 |
| 熵产生率估计 | `shock_physics.py` | 热力学不可逆性 |

---

## 六、质量检查清单

- [x] 原目录未被修改
- [x] 合成后项目为 Python 语言
- [x] 新目录完整包含合成后项目（13 个 .py 文件）
- [x] 只有一个博士级数学/物理科学计算问题已落地为可执行代码
- [x] **15 个输入项目均已真实融入**，无遗漏、无挂名
- [x] **`main.py` 已实际运行通过**，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 无可视化相关代码
