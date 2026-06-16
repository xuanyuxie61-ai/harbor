# 计算等离子体：Dusty Plasma 晶体结构模拟
## 高阶有限差分与稳定性分析（博士级合成项目）

---

## 一、项目概述

本项目是一个**前沿博士级计算等离子体物理模拟程序**，围绕 **dusty plasma（粉尘等离子体）晶体结构模拟** 展开。项目融合了 15 个种子项目的核心算法，在计算等离子体物理领域内进行了深度耦合改造，实现了从物理建模、数值求解到稳定性分析的全流程计算。

### 科学背景

在低温射频放电等离子体中，微米量级的固态粉尘颗粒通过收集电子和离子而带负电（电荷量可达 10³-10⁵ 个电子电荷）。粉尘颗粒间的屏蔽库仑（Yukawa）相互作用为：

$$
\phi_Y(r) = \frac{Q_d^2}{4\pi\varepsilon_0 r} \exp\left(-\frac{r}{\lambda_D}\right)
$$

其中 $Q_d$ 为粉尘电荷，$\lambda_D$ 为等离子体德拜长度。

系统的无量纲耦合参数为：

$$
\Gamma = \frac{Q_d^2}{4\pi\varepsilon_0 a_{ws}} \frac{\exp(-\kappa)}{k_B T_d}
$$

其中 $a_{ws} = 1/\sqrt{\pi n_d}$ 为 Wigner-Seitz 半径，$\kappa = a_{ws}/\lambda_D$ 为屏蔽参数。

当 $\Gamma > \Gamma_c \approx 137$ 时，粉尘系统发生结晶，形成二维六角晶格结构。

---

## 二、种子项目映射关系

| 序号 | 种子项目 | 核心算法 | 在合成项目中的角色 |
|------|---------|---------|-------------------|
| 1 | 281_diff2_center | 中心差分二阶导数 | 高阶有限差分算子基础（Fornberg 算法）|
| 2 | 507_hb_io | Harwell-Boeing 稀疏矩阵 I/O | 动力学矩阵的稀疏存储与读写 |
| 3 | 1381_vandermonde | Vandermonde 矩阵与 Bjorck-Pereyra 算法 | 粉尘电荷分布多项式插值 |
| 4 | 1318_triangle_symq_rule_original | 三角形对称求积规则 | Yukawa 屏蔽势的 Wigner-Seitz 元积分 |
| 5 | 522_hermite_polynomial | Hermite 多项式与 Gauss-Hermite 求积 | 粉尘振荡模式的谱分解 |
| 6 | 185_circles | 参数化圆（截面几何）| 粉尘颗粒截面与填充率分析 |
| 7 | 971_r8bto | 分块 Toeplitz 矩阵与 Schur 补递推 | 周期晶格的动力学矩阵求解 |
| 8 | 634_lagrange_basis_display | Lagrange 基函数求值 | 鞘层电势的 Lagrange 重构 |
| 9 | 862_pendulum_ode | 摆的 ODE 精确解 | 粉尘振荡 ODE 的类比（鞘层约束势）|
| 10 | 380_fem_to_tec | FEM 网格格式转换 | 晶体网格数据导出接口 |
| 11 | 782_msm_to_mm | Matrix Market 格式写入 | 矩阵格式互操作 |
| 12 | 1102_ryan597_RepresentationLearningWaves | 波的表示学习与谱方法 | 粉尘晶格波（DLW）的谱表示 |
| 13 | 016_arclength | 弧长积分（梯形法则）| 粉尘轨迹弧长的计算 |
| 14 | 615_kdv_exact | KdV 方程的精确孤子解 | 非线性粉尘声波基准测试 |
| 15 | 941_quad_monte_carlo | Monte Carlo 积分 | 构型热力学采样 |

---

## 三、核心模块说明

### 3.1 物理常数模块 (`dusty_plasma_physics_constants.py`)

封装 CODATA 2018 基本物理常数，定义 `DustyPlasmaRegime` 类，计算：
- **OML 漂浮电势**：Newton-Raphson 求解 $e\phi_f/k_BT_e$
- **德拜长度**：$\lambda_D^{-2} = \lambda_{De}^{-2} + \lambda_{Di}^{-2}$
- **粉尘电荷**：$Z_d = 4\pi\varepsilon_0 r_d |\phi_f| / e$
- **耦合参数**：$\Gamma$ 判断结晶/液态/气态区域

### 3.2 高阶有限差分 (`fd_high_order_stencil.py`)

基于 Fornberg 算法与预计算模板，实现：
- **2/4/6/8 阶精度**的中心差分模板
- 1D/2D  Laplacian 与梯度算子
- 周期/镜像/截断边界条件处理
- **收敛阶验证**：通过网格细化测试确认精度

预计算的二阶导数模板：
- 2 阶：$[1, -2, 1]/h^2$
- 4 阶：$[-1, 16, -30, 16, -1]/(12h^2)$
- 6 阶：$[2, -27, 270, -490, 270, -27, 2]/(180h^2)$

### 3.3 Vandermonde 电荷插值 (`vandermonde_charge_interpolation.py`)

实现：
- Vandermonde 矩阵构造
- **Bjorck-Pereyra 算法**：$O(N^2)$ 稳定求解
- 双变量 Vandermonde 用于 2D 电荷分布重构
- 条件数分析评估插值适定性

### 3.4 对称求积与 Madelung 常数 (`quadrature_screening_integral.py`)

实现 Xiao-Gimbutas 风格的三角形对称求积：
- 1/3/4/5/7 次代数精度规则
- 六边形 Wigner-Seitz 元分解为 6 个三角形
- **Madelung 常数**：$M(\kappa) = \sum_{j\neq 0} \exp(-\kappa r_j/a_{ws}) / (r_j/a_{ws})$

### 3.5 Hermite 谱模式 (`hermite_spectral_modes.py`)

基于物理学家 Hermite 多项式：
- 三项递推：$H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)$
- Hermite 函数：$\psi_n(x) = (2^n n! \sigma\sqrt{\pi})^{-1/2} H_n(x/\sigma) e^{-x^2/2\sigma^2}$
- **Gauss-Hermite 求积**：通过 Jacobi 矩阵特征值计算（Golub-Welsch 算法）
- 约束势中粉尘振荡模式的能谱与热分布

### 3.6 六角晶格几何 (`hexagonal_lattice_geometry.py`)

实现 `HexagonalLattice` 类：
- 格点坐标生成：$\mathbf{R}_{n_1,n_2} = n_1\mathbf{a}_1 + n_2\mathbf{a}_2$
- Wigner-Seitz 元（六边形 Voronoi 元胞）
- 配位壳层分析
- 填充率 $\eta = n_d \pi r_d^2$
- 倒格子矢量：$\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi\delta_{ij}$
- Debye-Waller 因子：$f_{DW}(G) = \exp(-\langle u^2 \rangle |G|^2 / 2)$
- 轨迹弧长的谱方法计算

### 3.7 分块 Toeplitz 动力学矩阵 (`block_toeplitz_dynamical.py`)

Yukawa 耦合下 1D 粉尘链的动力学矩阵：
$$D_{n,j} = \omega_{pd}^2 \times \begin{cases} 2\sum_k K_L(ka) & n=j \\ -K_L(|n-j|a) & n\neq j \end{cases}$$

力常数：
- 纵向：$K_L(r) = \frac{e^{-\kappa r}}{r^3}(2 + 2\kappa r + \kappa^2 r^2)$
- 横向：$K_T(r) = \frac{e^{-\kappa r}}{r^3}(1 + \kappa r)$

实现：
- 分块 Toeplitz 矩阵向量积（从 971_r8bto 的 `r8bto_mv`）
- **Schur 补递推求解**（`r8bto_sl`）
- 稠密 LU 分解与回代（`r8ge_fa`, `r8ge_sl`）
- 色散关系 $\omega(q)$ 通过块对角化计算

### 3.8 Lagrange 电势重构 (`lagrange_potential_reconstruction.py`)

鞘层电势 $\phi(z)$ 从粉尘位置测量值重构：
- Lagrange 基函数：$L_i(x) = \prod_{j\neq i} \frac{x-x_j}{x_i-x_j}$
- 电场 $E = -d\phi/dz$ 通过 Lagrange 导数计算
- Chebyshev 节点优化（抑制 Runge 现象）
- **Lebesgue 常数**评估插值稳定性

### 3.9 粉尘晶格波 (`dust_lattice_wave.py`)

`DustLatticeWaveField` 类实现 DLW 谱表示：
- 位移场：$u_n(t) = \frac{1}{N}\sum_q A(q,t) e^{iqna}$
- 色散关系（纵向/横向分支）
- 高斯波包初始化
- Velocity Verlet 时间演化
- 群速度 $v_g = d\omega/dq$
- 声子态密度 $g(\omega)$

### 3.10 KdV 孤子基准 (`kdv_soliton_benchmark.py`)

非线性粉尘声波的 KdV 方程：
$$\frac{\partial \phi}{\partial t} + A\phi\frac{\partial \phi}{\partial \xi} + B\frac{\partial^3 \phi}{\partial \xi^3} = 0$$

实现：
- **sech² 孤子精确解**：$\phi = \phi_m \text{sech}^2((\xi - ut)/\Delta)$
- 有理函数解（奇异性测试）
- 双孤子相互作用
- 守恒量 $I_1, I_2, I_3$
- 残差 $R = \phi_t + A\phi\phi_x + B\phi_{xxx}$ 验证

### 3.11 Monte Carlo 热力学 (`monte_carlo_thermodynamics.py`)

`DustyPlasmaMonteCarlo` 类实现 Metropolis 采样：
- Yukawa 总能量计算
- 最小镜像约定（周期边界）
- 平均能量 $\langle U/N \rangle$
- 比热 $C_v = \Gamma^2 \text{Var}(E)/N$
- 对关联函数 $g(r)$
- **Lindemann 熔化判据**：$\delta_L > \delta_c \approx 0.15$

### 3.12 稳定性特征值分析 (`stability_eigenvalue_analysis.py`)

`CrystalStabilityAnalyzer` 类：
- 对称特征值分解（`np.linalg.eigh`）
- **Born 稳定性判据**：$C_{11} > 0, C_{66} > 0, C_{11} > |C_{12}|$
- 幂迭代求最大特征值
- 反迭代求指定特征值
- 模态分解：$u = \sum_k c_k v_k$
- $\varepsilon$-伪谱分析

### 3.13 稀疏矩阵 I/O (`sparse_matrix_io.py`)

实现两种标准格式：
- **Harwell-Boeing (HB)**：CSC 格式，稀疏矩阵高效存储
- **Matrix Market (MM)**：NIST 标准交换格式
- 支持实数/复数/对称/斜对称等类型

---

## 四、数学公式汇总

### 4.1 等离子体物理公式

| 公式 | 表达式 |
|------|--------|
| 电子德拜长度 | $\lambda_{De} = \sqrt{\varepsilon_0 T_e / (n_e e^2)}$ |
| 离子德拜长度 | $\lambda_{Di} = \sqrt{\varepsilon_0 T_i / (n_i e^2)}$ |
| 总德拜长度 | $\lambda_D^{-2} = \lambda_{De}^{-2} + \lambda_{Di}^{-2}$ |
| OML 漂浮电势 | $\exp(-\zeta) = \sqrt{m_e/(m_i \tau)} (1 + \zeta/\tau)$ |
| 粉尘电荷 | $Z_d = 4\pi\varepsilon_0 r_d \|\phi_f\| / e$ |
| 耦合参数 | $\Gamma = \frac{Q_d^2}{4\pi\varepsilon_0 a_{ws}} e^{-\kappa} / (k_B T_d)$ |
| 粉尘等离子体频率 | $\omega_{pd} = \sqrt{n_d Q_d^2 / (\varepsilon_0 m_d)}$ |
| 粉尘声速 | $C_{DA} = \sqrt{Z_d k_B T_e / m_d}$ |

### 4.2 数值方法公式

| 方法 | 公式 |
|------|------|
| 2 阶差分 | $f''(x) \approx (f(x+h) - 2f(x) + f(x-h))/h^2$ |
| 4 阶差分 | $f''(x) \approx (-f_{2h} + 16f_h - 30f + 16f_{-h} - f_{-2h})/(12h^2)$ |
| Vandermonde | $V_{ij} = x_i^j$ |
| BP 算法 | $b_i^{(n)} = (b_i^{(n-1)} - b_{i-1}^{(n-1)})/(x_i - x_{i-n})$ |
| Hermite 递推 | $H_{n+1} = 2x H_n - 2n H_{n-1}$ |
| Yukawa 势 | $V(r) = e^{-\kappa r}/r$ |
| 力常数（纵）| $K_L = e^{-\kappa r}(2 + 2\kappa r + \kappa^2 r^2)/r^3$ |
| KdV 孤子 | $\phi = \phi_m \text{sech}^2((x-ut)/\Delta)$ |

---

## 五、运行方法

### 5.1 环境要求

- Python 3.8+
- NumPy（标准科学计算库）

### 5.2 运行命令

```bash
cd 297_synth_project_Advanced
python main.py
```

**零参数**：直接运行即可，无需任何命令行参数。

### 5.3 输出

程序将打印 14 个计算步骤的完整结果：
1. 等离子体参数与耦合区域
2. 晶格几何与填充率
3. 高阶差分模板验证
4. 屏蔽积分与 Madelung 常数
5. Vandermonde 电荷插值
6. Lagrange 鞘层电势重构
7. Hermite 谱模式分析
8. 分块 Toeplitz 动力学矩阵
9. 特征值稳定性分析
10. 粉尘晶格波传播
11. KdV 孤子基准测试
12. Monte Carlo 热力学
13. 稀疏矩阵 I/O 测试
14. 轨迹弧长计算

最终输出 **RESULTS SUMMARY** 包含所有关键物理量。

---

## 六、项目文件结构

```
297_synth_project_Advanced/
├── main.py                              # 统一入口
├── dusty_plasma_simulator.py            # 顶层模拟调度器
├── dusty_plasma_physics_constants.py    # 物理常数与等离子体区域
├── fd_high_order_stencil.py             # 高阶有限差分模板
├── sparse_matrix_io.py                  # HB/MM 稀疏矩阵 I/O
├── vandermonde_charge_interpolation.py  # Vandermonde 电荷插值
├── quadrature_screening_integral.py     # 对称求积与 Madelung 常数
├── hermite_spectral_modes.py            # Hermite 谱模式
├── hexagonal_lattice_geometry.py        # 六角晶格几何
├── block_toeplitz_dynamical.py          # 分块 Toeplitz 动力学矩阵
├── lagrange_potential_reconstruction.py # Lagrange 电势重构
├── dust_lattice_wave.py                 # 粉尘晶格波
├── kdv_soliton_benchmark.py             # KdV 孤子基准
├── monte_carlo_thermodynamics.py        # Monte Carlo 热力学
├── stability_eigenvalue_analysis.py     # 特征值稳定性分析
└── README_博士级合成说明.md             # 本文档
```

共 **15 个 .py 文件**（含入口 main.py），超过要求的最少 8 个。

---

## 七、科学问题与前沿性

### 7.1 解决的核心问题

本项目针对 **实验室粉尘等离子体晶体** 的计算模拟，具体解决：

1. **晶体稳定性判据**：通过特征值分析判定晶格是否稳定
2. **波传播特性**：计算 DLW 的色散关系与群速度
3. **非线性波动力学**：KdV 孤子作为非线性基准
4. **相变热力学**：Monte Carlo 模拟熔化转变
5. **数值方法验证**：高阶差分精度、谱方法正交性

### 7.2 前沿性体现

- **强耦合等离子体物理**：$\Gamma > 100$ 区域的结晶行为是凝聚态物理与等离子体物理的交叉前沿
- **高阶数值方法**：4-8 阶差分模板、谱方法、自适应求积的组合应用
- **多尺度耦合**：从单颗粒 OML 充电到集体 DLW 模式
- **可复现性**：零参数运行，固定随机种子，所有结果可精确重现

### 7.3 工程复杂度

- **边界处理**：周期/镜像/截断三种边界条件
- **数值鲁棒性**：NaN/Inf 防护、奇异矩阵最小二乘回退、零间距保护
- **格式兼容性**：HB 与 MM 两种国际标准的完整实现
- **精度控制**：收敛阶测试、条件数监控、残差验证

---

## 八、典型运行输出示例

```
========================================================================
  DUSTY PLASMA CRYSTAL SIMULATION
  High-Order Finite Difference & Stability Analysis
========================================================================

[Step 1] Initializing dusty plasma regime...
  Plasma regime: VALID
    n_e [m^-3]                = 1.000e+15
    T_e [eV]                  = 2.500
    kappa                     = 1.3939
    Gamma                     = 13.05
    Crystal regime            = INTERMEDIATE

[Step 9] Eigenvalue stability analysis...
  Crystal stable: True
  Negative modes: 0
  Min eigenvalue: 9.887e-01
  Condition number: 7.60e+01
  Born criteria satisfied: True

------------------------------------------------------------------------
  RESULTS SUMMARY
------------------------------------------------------------------------
  Plasma regime valid           : True
  Crystal stable                : True
  Gamma                         : 13.05
  Madelung constant             : 2.1817
  Condition number              : 75.98
  MC acceptance                 : 0.993
------------------------------------------------------------------------
  All simulation steps completed successfully.
------------------------------------------------------------------------
```

---

## 九、参考文献

1. Shukla, P.K. & Mamun, A.A., *Introduction to Dusty Plasma Physics*, IoP (2002)
2. Morfill, G.E. et al., "Crystallization of dusty plasmas", *Phys. Rev. Lett.* **83**, 1598 (1999)
3. Hamaguchi, S. & Farouki, R.T., "Thermodynamics of strongly coupled dusty plasmas", *Phys. Rev. E* **56**, 4671 (1997)
4. Fornberg, B., "Generation of finite difference formulas", *Math. Comp.* **51**, 699 (1988)
5. Bjorck, Å. & Pereyra, V., "Solution of Vandermonde systems", *Math. Comp.* **24**, 893 (1970)
6. Xiao, H. & Gimbutas, Z., "Quadrature rules on the triangle", *Comput. Math. Appl.* **59**, 663 (2010)
7. Melandso, F., "Lattice waves in dust plasma crystals", *Phys. Plasmas* **6**, 1738 (1999)
8. Rao, N.N., Shukla, P.K. & Yu, M.Y., "Dust acoustic solitons", *Planet. Space Sci.* **38**, 543 (1990)
9. Born, M. & Huang, K., *Dynamical Theory of Crystal Lattices*, Oxford (1954)
10. Trefethen, L.N. & Embree, M., *Spectra and Pseudospectra*, Princeton (2005)

---

## 十、许可证

本项目为学术合成项目，遵循原始种子项目的许可协议。仅用于科研与教学目的。

---

**合成完成日期**：2026年6月
**项目规模**：15 个 Python 源文件，~3500 行代码
**科学领域**：计算等离子体物理 / 强耦合 dusty plasma / 高阶数值方法
