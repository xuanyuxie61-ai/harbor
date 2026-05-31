# README_博士级合成说明.md

## 永磁同步电机电磁场有限元分析 — 博士级科研代码合成项目

---

## 一、项目概述

本项目基于 **15 个科研代码种子项目** 的核心算法，在 **电磁学：电机电磁场有限元分析** 这一前沿科学领域内，融合构建了一个面向博士级难度的多物理场耦合计算框架。

**核心科学问题**：
> 考虑非线性磁材料（硅钢片 B-H 曲线）、材料参数不确定性（对数正态分布）、以及转子偏心故障的永磁同步电机（PMSM）电磁-结构耦合多场分析。

该问题涵盖：
- **二维轴对称静磁场的有限元离散与求解**
- **非线性磁材料的 Newton-Raphson 迭代**
- **三维场投影与端部效应分析**
- **Maxwell 应力张量的电磁转矩计算**
- **涡流损耗的时谐场估计**
- **转子动力学、偏心振动与陀螺效应耦合**
- **数值稳定性分析与刚度矩阵条件数估计**

---

## 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 在合成项目中的角色 |
|------|--------|----------|-------------------|
| 1 | `698_log_normal` | 对数正态分布 PDF/CDF/采样/矩估计 | `material_physics.py`：硅钢片磁导率、永磁体剩磁等材料参数的不确定性量化 |
| 2 | `1312_triangle_monte_carlo` | 三角形单元蒙特卡洛积分 | `quadrature_engine.py`：验证有限元刚度矩阵组装中的三角形单元积分精度 |
| 3 | `447_freefem_msh_io` | FreeFEM .msh 网格文件读写 | `mesh_engine.py`：电机 2D 截面网格的输入输出与数据结构管理 |
| 4 | `491_grid_display` | 网格节点/单元/边界数据结构 | `mesh_engine.py`：2D 有限元网格的拓扑组织与边界边提取 |
| 5 | `1337_triangulation_histogram` | 三角剖分质量统计 | `mesh_engine.py`：三角形纵横比、内角、面积等网格质量指标计算 |
| 6 | `414_fem2d_scalar_display` | 2D 标量 FEM 数据后处理 | `fem2d_axi.py`：节点-单元数据关系处理，为 B 场重构提供基础 |
| 7 | `261_cvt_square_uniform` | CVT（重心 Voronoi 镶嵌）网格生成 | `mesh_engine.py`：电机定子/转子/气隙区域的高质量三角形网格生成 |
| 8 | `495_gyroscope_ode` | 陀螺仪刚体转动 ODE | `rotor_multiphysics.py`：高速电机转子的陀螺效应与欧拉角动力学 |
| 9 | `383_fem1d` | 1D 有限元方法 | `fem1d_radial.py`：电机气隙方向的 1D 径向磁势简化模型 |
| 10 | `418_fem3d_project` | 3D FEM 投影（TET4 基函数） | `fem3d_projection.py`：将 2D 截面磁势场投影到 3D 轴向长度分布 |
| 11 | `1089_sling_ode` | 吊索非线性振荡 ODE | `rotor_multiphysics.py`：转子偏心故障的非线性径向振动模型 |
| 12 | `207_condition` | 矩阵 L1 条件数估计（Hager/LINPACK） | `numerical_analysis.py`：有限元刚度矩阵的病态程度评估 |
| 13 | `441_floyd` | Floyd-Warshall 最短路径 | `mesh_engine.py`：网格节点连通性分析，估算磁路最短长度 |
| 14 | `947_quadmom` | 正交多项式与高斯求积（矩方法） | `quadrature_engine.py`：高斯-勒让德求积与自定义权函数的矩方法求积 |
| 15 | `1387_vanderpol_ode_period` | Van der Pol 周期估计（Urabe 公式） | `rotor_multiphysics.py`：偏心故障非线性振动的周期渐近估计 |

**每一个输入项目都在合成项目中承担了真实计算角色，无遗漏、无挂名。**

---

## 三、新增数学物理模型与核心公式

### 3.1 非线性磁材料 B-H 曲线模型

基于反正切函数的经验模型：

$$
B(H) = \mu_0 H + \frac{2 B_s}{\pi} \arctan\!\left( \frac{\pi (\mu_r - 1) \mu_0 H}{2 B_s} \right)
$$

其中：
- $\mu_0 = 4\pi \times 10^{-7}$ H/m 为真空磁导率
- $\mu_r$ 为初始相对磁导率
- $B_s$ 为饱和磁通密度

微分磁导率（增量磁导率）：

$$
\mu_{\text{diff}}(H) = \frac{dB}{dH} = \mu_0 + \frac{(\mu_r - 1)\mu_0}{1 + \left( \frac{\pi (\mu_r - 1) \mu_0 H}{2 B_s} \right)^2}
$$

磁阻率（reluctivity）用于有限元刚度矩阵组装：

$$
\nu(B) = \frac{H(B)}{B} = \frac{1}{\mu_{\text{apparent}}(B)}
$$

### 3.2 轴对称静磁场控制方程

在二维轴对称假设下，磁矢势仅含 $z$ 分量 $A_z(x,y)$，控制方程为：

$$
\frac{\partial}{\partial x}\!\left( \nu \frac{\partial A_z}{\partial x} \right) + \frac{\partial}{\partial y}\!\left( \nu \frac{\partial A_z}{\partial y} \right) = -J_z
$$

其中 $J_z$ 为轴向电流密度。磁通密度与磁矢势的关系：

$$
B_x = \frac{\partial A_z}{\partial y}, \quad B_y = -\frac{\partial A_z}{\partial x}, \quad |B| = \sqrt{B_x^2 + B_y^2}
$$

### 3.3 Galerkin 弱形式与有限元离散

采用三角形 P1 Lagrange 元，$A_z \approx \sum_j A_j N_j(x,y)$，弱形式为：

$$
\sum_j A_j \int_{\Omega_e} \nu \, \nabla N_i \cdot \nabla N_j \, d\Omega = \int_{\Omega_e} J_z N_i \, d\Omega
$$

对于 P1 元，$\nabla N_i$ 在单元内为常数：

$$
\nabla N_i = \frac{1}{2|\Delta|} \begin{bmatrix} y_j - y_k \\ x_k - x_j \end{bmatrix}
$$

其中 $(i,j,k)$ 为三角形顶点的轮换下标，$|\Delta|$ 为单元面积。

单元刚度矩阵：

$$
K_{ij}^{(e)} = \nu \, |\Delta| \, (\nabla N_i \cdot \nabla N_j)
$$

### 3.4 Maxwell 应力张量与电磁转矩

Maxwell 应力张量：

$$
\mathbf{T} = \frac{1}{\mu} \left( \mathbf{B} \otimes \mathbf{B} - \frac{1}{2} |\mathbf{B}|^2 \mathbf{I} \right)
$$

即分量形式：

$$
T_{xx} = \frac{1}{\mu}\!\left(B_x^2 - \frac{1}{2}|\mathbf{B}|^2\right), \quad T_{xy} = \frac{B_x B_y}{\mu}, \quad T_{yy} = \frac{1}{\mu}\!\left(B_y^2 - \frac{1}{2}|\mathbf{B}|^2\right)
$$

电磁转矩（二维简化，气隙中径向/切向分量）：

$$
\tau = L \oint r^2 T_{r\theta} \, d\theta = \frac{L}{\mu_0} \oint r^2 B_r B_\theta \, dl
$$

其中 $B_r = B_x \cos\theta + B_y \sin\theta$，$B_\theta = -B_x \sin\theta + B_y \cos\theta$。

### 3.5 涡流损耗（时谐场近似）

对于时谐场 $A_z(\mathbf{r},t) = \hat{A}_z(\mathbf{r}) e^{j\omega t}$，涡流电流密度：

$$
\mathbf{J}_e = -j\omega\sigma \hat{A}_z \, \mathbf{e}_z
$$

涡流损耗功率：

$$
P_{\text{eddy}} = \frac{1}{2\sigma} \int_{\Omega} |\mathbf{J}_e|^2 \, dV = \frac{\omega^2 \sigma}{2} \int_{\Omega} |\hat{A}_z|^2 \, dV
$$

### 3.6 永磁体退磁曲线

采用线性退磁曲线近似：

$$
B = \mu_0 \mu_{\text{rec}} H + B_r
$$

其中 $\mu_{\text{rec}}$ 为回复磁导率（NdFeB 约 1.05），$B_r$ 为剩磁。矫顽力：

$$
H_c = \frac{B_r}{\mu_0 \mu_{\text{rec}}}
$$

### 3.7 材料参数不确定性（对数正态分布）

若随机变量 $X$ 服从对数正态分布，则 $\ln X \sim \mathcal{N}(\mu_{\ln}, \sigma_{\ln}^2)$，PDF 为：

$$
f(x; \mu_{\ln}, \sigma_{\ln}) = \frac{1}{x \sigma_{\ln} \sqrt{2\pi}} \exp\!\left( -\frac{(\ln x - \mu_{\ln})^2}{2\sigma_{\ln}^2} \right), \quad x > 0
$$

由均值 $\mathbb{E}[X]$ 和方差 $\text{Var}[X]$ 反解参数：

$$
\sigma_{\ln}^2 = \ln\!\left(1 + \frac{\text{Var}[X]}{\mathbb{E}[X]^2}\right), \quad \mu_{\ln} = \ln\mathbb{E}[X] - \frac{1}{2}\sigma_{\ln}^2
$$

### 3.8 高斯-勒让德求积与矩方法

$n$ 点 Gauss-Legendre 求积精确积分次数 $\leq 2n-1$ 的多项式：

$$
\int_{-1}^{1} f(x) \, dx \approx \sum_{i=1}^{n} w_i f(x_i)
$$

节点 $x_i$ 和权重 $w_i$ 通过 Legendre 多项式的 Jacobi 矩阵特征值问题求解（Golub-Welsch 算法）。Jacobi 矩阵的递推系数：

$$
\alpha_k = 0, \quad \beta_k = \frac{k}{\sqrt{4k^2 - 1}}
$$

矩方法求积：给定权函数 $w(x)$ 的前 $2n+1$ 个矩 $m_k = \int x^k w(x) \, dx$，构造 Hankel 矩阵 $H_{ij} = m_{i+j}$，通过 Cholesky 分解提取 Jacobi 矩阵系数，进而求解特征值问题得到求积节点和权重。

### 3.9 转子动力学方程

永磁同步电机转子运动方程：

$$
J \frac{d\omega}{dt} = \tau_{\text{em}} - \tau_{\text{load}} - B_d \omega - \tau_{\text{cog}}(\theta)
$$

齿槽转矩的傅里叶近似：

$$
\tau_{\text{cog}}(\theta) = \sum_k T_k \sin(N_s k \theta)
$$

### 3.10 偏心故障非线性振动

偏心位移 $(u,v)$ 满足：

$$
m_r \frac{d^2 u}{dt^2} + c_r \frac{du}{dt} + k_r u = F_u + m_r e \omega^2 \cos\theta
$$

在极坐标下融合 sling_ode 的非线性结构，引入软弹簧恢复力 $s(r-r_0)^3$。

### 3.11 陀螺效应（欧拉方程）

高速转子的体坐标系角动量方程：

$$
\begin{aligned}
A_1 \frac{d\omega_1}{dt} &= (A_2 - A_3) \omega_2 \omega_3 + M_1 \\
A_2 \frac{d\omega_2}{dt} &= (A_3 - A_1) \omega_3 \omega_1 + M_2 \\
A_3 \frac{d\omega_3}{dt} &= (A_1 - A_2) \omega_1 \omega_2 + M_3
\end{aligned}
$$

### 3.12 非线性振动周期估计（Urabe 公式）

对于强非线性 Van der Pol 型振荡器，周期渐近展开：

$$
T(\mu) \approx (3 - 2\ln 2)\mu + \frac{3\alpha}{\mu^{1/3}} - \frac{\ln\mu}{3\mu} + \frac{C}{\mu}
$$

其中 $\alpha = 2.338107$ 为 Airy 函数首个零点，$C = 3\ln 2 - \ln 3 - 1.5 + b_0 - 2d$。

### 3.13 刚度矩阵条件数与数值稳定性

对称正定矩阵 $K$ 的谱条件数：

$$
\kappa_2(K) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

Hager 的 $L_1$ 条件数估计算法通过迭代求解辅助线性系统估计 $||K^{-1}||_1$：

$$
\text{cond}_1(K) = ||K||_1 \cdot ||K^{-1}||_1
$$

Peclet 数（对流-扩散稳定性判据）：

$$
\text{Pe} = \frac{|v| h}{2D}
$$

当 $\text{Pe} > 1$ 时，标准 Galerkin 方法可能出现数值振荡。

### 3.14 1D 径向有限元方程

柱坐标下的轴对称磁势方程：

$$
-\frac{d}{dr}\!\left( \nu(r) \, r \frac{dA_\theta}{dr} \right) + \frac{\nu(r)}{r} A_\theta = J_s(r)
$$

线性基函数：

$$
\phi_i(r) = \frac{r_{i+1} - r}{h_i}, \quad \phi_{i+1}(r) = \frac{r - r_i}{h_i}
$$

### 3.15 三角形单元 Dunavant 高斯求积

7 点 Dunavant 规则（精确到 5 次多项式）在面积坐标下的节点与权重：

- 重心点：$(1/3, 1/3, 1/3)$，$w = 9/40$
- 类型 A（3 个轮换点）：$a = (6+\sqrt{15})/21$，$w_A = (155-\sqrt{15})/1200$
- 类型 B（3 个轮换点）：$b = (6-\sqrt{15})/21$，$w_B = (155+\sqrt{15})/1200$

### 3.16 3D TET4 有限元基函数

四面体体积坐标（重心坐标）：

$$
\phi_i(x,y,z) = \frac{V_i(x,y,z)}{V}, \quad \sum_{i=1}^{4} \phi_i = 1
$$

其中 $V$ 为四面体有向体积：

$$
V = \frac{1}{6} \det\!\begin{bmatrix} x_2-x_1 & x_3-x_1 & x_4-x_1 \\ y_2-y_1 & y_3-y_1 & y_4-y_1 \\ z_2-z_1 & z_3-z_1 & z_4-z_1 \end{bmatrix}
$$

基函数梯度：

$$
\nabla \phi_i = \frac{(\mathbf{p}_j - \mathbf{p}_k) \times (\mathbf{p}_l - \mathbf{p}_k)}{6V}
$$

---

## 四、项目文件结构与实现路径

```
100_synth_project/
├── main.py                    # 统一入口，零参数运行
├── mesh_engine.py             # CVT网格生成 + FreeFEM I/O + 质量统计 + Floyd连通性
├── quadrature_engine.py       # Gauss-Legendre求积 + 三角形Dunavant求积 + 蒙特卡洛 + 矩方法
├── material_physics.py        # 非线性B-H曲线 + 永磁体模型 + 对数正态不确定性
├── fem1d_radial.py            # 1D径向有限元（Thomas算法求解三对角系统）
├── fem2d_axi.py               # 2D轴对称电磁场FEM核心（稀疏矩阵、Dirichlet边界、B场重构）
├── fem3d_projection.py        # 3D场投影（TET4基函数、拉伸网格、端部效应力）
├── rotor_multiphysics.py      # 转子动力学 + 偏心振动 + 陀螺效应 + 非线性周期估计
├── numerical_analysis.py      # Hager条件数估计 + LINPACK估计 + FEM误差估计 + Peclet数
└── README_博士级合成说明.md    # 本文档
```

### 各文件对原项目的具体融合方式

**`mesh_engine.py`**
- `CVTMeshGenerator` 类：直接移植 `261_cvt_square_uniform` 的 Lloyd 迭代算法，扩展至环形扇区几何。
- `Mesh2D` 类：融合 `491_grid_display` 的节点-单元数据结构和 `1337_triangulation_histogram` 的网格质量统计指标（纵横比、内角、面积比）。
- `floyd_warshall_magnetic_path`：直接实现 `441_floyd` 的三重循环最短路径算法，用于网格节点连通性分析。
- `write_msh` / `read_msh`：移植 `447_freefem_msh_io` 的 FreeFEM .msh 格式解析逻辑。

**`quadrature_engine.py`**
- `GaussLegendreQuadrature`：基于 `947_quadmom` 的 Golub-Welsch 特征值思想，通过 `scipy.linalg.eigh_tridiagonal` 构造 Legendre 高斯求积。
- `TriangleGaussianQuadrature`：实现 Dunavant 7 点规则，用于有限元刚度矩阵的高精度数值积分。
- `TriangleMonteCarlo`：移植 `1312_triangle_monte_carlo` 的参考单元采样与物理单元映射算法，增加方差估计与置信区间。
- `MomentMethodQuadrature`：直接复现 `947_quadmom` 的 `moment_method` 核心算法（Hankel 矩阵 → Cholesky → Jacobi 矩阵 → 特征值问题）。

**`material_physics.py`**
- `NonlinearMagneticMaterial`：基于反正切函数构建 B-H 曲线模型，提供磁阻率、微分磁导率等物理量计算。
- `PermanentMagnet`：线性退磁曲线模型，计算剩磁、矫顽力、等效磁化电流。
- `LogNormalUncertainty`：完整移植 `698_log_normal` 的 PDF、CDF、反函数、采样、均值、方差等统计函数。

**`fem1d_radial.py`**
- 完整重构 `383_fem1d` 的算法流程：`init` → `geometry` → `assemble` → `solve` → `output`。
- 保留线性基函数 `phi` 及其导数的解析表达式。
- 保留 Thomas 算法求解三对角系统的核心逻辑。
- 扩展至柱坐标下的径向磁势方程。

**`fem2d_axi.py`**
- 基于 `414_fem2d_scalar_display` 的节点-单元关系处理，构建 2D 轴对称 FEM 求解器。
- 实现 P1 元的形状函数梯度、单元刚度矩阵与载荷向量组装。
- 支持稀疏矩阵（`scipy.sparse.csr_matrix`）与 Dirichlet 罚函数边界条件。
- 实现 Maxwell 应力张量计算、电磁转矩积分、磁场储能与涡流损耗估计。

**`fem3d_projection.py`**
- 移植 `418_fem3d_project` 的 `basis_mn_tet4` 与 `fem3d_transfer` 思想。
- 将 2D 三角形网格拉伸为 3D 棱柱并细分为 TET4 四面体。
- 组装 3D 一致质量矩阵，求解 L² 投影方程。
- 计算 3D 磁场储能与端部轴向漏磁力。

**`rotor_multiphysics.py`**
- `RotorDynamics`：构建电机转子运动 ODE，耦合电磁转矩、负载、阻尼与齿槽转矩。
- `EccentricVibration`：融合 `1089_sling_ode` 的非线性振动结构，加入偏心激振力与软弹簧恢复力。
- `GyroscopicEffects`：直接移植 `495_gyroscope_ode` 的欧拉角速率方程与角动量方程。
- `NonlinearPeriodEstimator`：完整复现 `1387_vanderpol_ode_period` 的 Cook、Cartwright、Grimshaw、Dorodnicyn、Urabe 五种周期估计公式。

**`numerical_analysis.py`**
- `ConditionEstimator`：移植 `207_condition` 的 `condition_hager` 与 `condition_linpack` 核心算法，支持稠密矩阵与稀疏矩阵。
- `FEMErrorEstimator`：实现 P1 元的先验 H¹ 误差估计、网格 Peclet 数计算、刚度矩阵正定性检查。
- `NumericalRobustness`：提供安全除法、梯度裁剪、收敛监控、正则化等工程鲁棒性工具。

---

## 五、合成后的项目能够解决什么科学问题

1. **电机电磁场分布计算**：在考虑非线性硅钢片 B-H 曲线的前提下，求解永磁同步电机 2D 截面的磁矢势、磁通密度分布。

2. **材料不确定性量化**：通过对数正态分布模型，量化硅钢片磁导率、永磁体剩磁等材料参数的不确定性对电磁场计算结果的影响。

3. **电磁力与转矩计算**：基于 Maxwell 应力张量，计算气隙中的电磁转矩，评估电机的机电能量转换效率。

4. **涡流损耗估计**：在时谐场近似下，估算定子/转子铁芯中的涡流损耗，为电机热设计提供输入。

5. **三维端部效应分析**：通过 2D→3D 场投影，估算电机端部漏磁与轴向电磁力，辅助绕组端部结构设计。

6. **转子动力学与故障诊断**：耦合电磁转矩与转子运动方程，模拟转子偏心故障引起的非线性振动，估算故障特征频率。

7. **高速陀螺效应分析**：对于高速永磁电机，分析转子陀螺效应对支承系统的影响。

8. **数值稳定性评估**：估计有限元刚度矩阵的条件数，评估离散格式的数值稳定性与计算可靠性。

---

## 六、项目运行方式

### 环境要求
- Python 3.8+
- NumPy
- SciPy

### 运行命令

```bash
cd 100_synth_project
python main.py
```

**无需输入任何参数**，程序将自动执行以下完整流程：
1. 生成电机 2D 截面网格
2. 定义材料属性与不确定性模型
3. 1D 径向有限元分析
4. 2D 轴对称有限元分析
5. 3D 场投影与端部效应分析
6. 高斯求积与蒙特卡洛积分验证
7. 转子动力学耦合分析
8. 数值稳定性评估

### 输出说明

程序在控制台输出各步骤的计算结果，包括：
- 网格节点/单元数量与质量指标
- 磁矢势、磁通密度的数值范围
- 磁场储能、电磁转矩、涡流损耗
- 稳态转速、偏心位移、振动周期
- 条件数、Peclet 数、误差估计

---

## 七、设计决策与边界处理

### 7.1 数值鲁棒性措施

- **退化单元检测**：在网格生成和有限元组装阶段检查三角形面积，面积小于 $10^{-14}$ 时抛出异常。
- **边界保护**：所有除法操作使用安全除法（`safe_divide`），分母接近零时替换为极小正数。
- **梯度裁剪**：非线性迭代中的梯度范数超过阈值时自动裁剪，防止数值爆炸。
- **残差监控**：Newton-Raphson 迭代中检查残差是否为有限值，非有限时提前终止并抛出异常。
- **刚度矩阵正则化**：对角元接近零时自动添加 $10^{-12}$ 正则项，保证可逆性。
- **Dirichlet 罚函数法**：使用 $10^{16}$ 量级的罚系数固定边界值，避免直接消元带来的稀疏结构破坏。

### 7.2 工程复杂度

- **多场耦合**：电磁场（FEM）→ 结构动力学（ODE）→ 数值稳定性分析，三个物理域在一个框架内协同计算。
- **多尺度空间**：1D 径向简化 → 2D 截面全模型 → 3D 轴向投影，三个空间维度层层递进。
- **多精度积分**：高斯求积（确定性）+ 蒙特卡洛（随机性）+ 矩方法（自定义权函数），三种积分策略互为验证。
- **多物理量输出**：磁场、转矩、损耗、力、位移、转速、周期、条件数等 10+ 个物理量同时计算。

### 7.3 删除的非科学内容

- 所有 `matplotlib` / `trisurf` / `plot` 等可视化代码已移除。
- 所有图形界面、交互式输入、PNG 保存等后处理代码已移除。
- 仅保留数值计算、矩阵运算、ODE 求解、统计分析等核心科学算法。

---

## 八、参考文献与理论来源

1. Zienkiewicz O.C., Taylor R.L. *The Finite Element Method*, 6th Ed., Butterworth-Heinemann, 2005.
2. Hager W.W. "Condition Estimates", *SIAM J. Sci. Stat. Comput.*, 5(2):311-316, 1984.
3. Golub G.H., Welsch J.H. "Calculation of Gaussian Quadrature Rules", *Math. Comp.*, 23(106):221-230, 1969.
4. Du Q., Faber V., Gunzburger M. "Centroidal Voronoi Tessellations", *SIAM Review*, 41(4):637-676, 1999.
5. Dunavant D.A. "High Degree Efficient Symmetrical Gaussian Quadrature Rules for the Triangle", *Int. J. Numer. Meth. Eng.*, 21(6):1129-1148, 1985.
6. Cartwright M.L. "Van der Pol's equation for relaxation oscillations", *Annals of Math. Studies*, 29:3-18, 1952.
7. Urabe M. "Periodic Solutions of Van der Pol's Equation", *J. Sci. Hiroshima Univ.*, 24(2):197-199, 1960.
8. Moulton F.R. *Differential Equations*, Dover, 1958. (Gyroscope)
9. Elhay S., Kautsky J. "Algorithm 655: IQPACK", *ACM TOMS*, 13(4):399-415, 1987.
10. Burkardt J. *Numerical Analysis Codes*, https://people.math.sc.edu/Burkardt/ (所有种子项目来源)
