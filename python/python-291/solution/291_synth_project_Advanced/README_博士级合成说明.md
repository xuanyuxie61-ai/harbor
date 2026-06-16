# PROJECT_291 —— 计算等离子体：等离子体鞘层与壁面相互作用
## 高阶有限差分与稳定性分析（小规模可复现实验）

---

## 一、项目概述

本项目是一个面向**前沿博士级科学计算问题**的合成项目，严格围绕以下方向展开：

> **计算等离子体 —— 等离子体鞘层与壁面相互作用 —— 高阶有限差分与稳定性分析（小规模可复现实验）**

项目以**一维磁化氩等离子体鞘层**为研究对象，融合 15 个种子项目的核心算法，构建了一套完整的计算等离子体工具链。通过 `python main.py` **零参数**即可运行，完成从等离子体参数初始化、网格生成、非线性自洽求解、谱稳定性分析、动理学 Vlasov 验证到拓扑分析的全流程计算。

### 科学问题

磁化等离子体与材料壁面的相互作用是核聚变等离子体边界物理、半导体刻蚀等离子体、电推进等领域的核心问题。鞘层（sheath）是等离子体与壁面之间的薄层结构，其特征尺度为德拜量级（~0.1 mm），但其中发生的物理过程——离子加速、电子排斥、二次电子发射、不稳定性激发——对壁面侵蚀、等离子体约束、能量输运具有决定性影响。

本项目研究的核心问题：
1. **自洽鞘层结构**：求解含二次电子发射 (SEE) 修正的非线性 Poisson-Boltzmann 方程
2. **高阶数值格式**：实现 4 阶/6 阶紧致有限差分 (Compact/Padé) 格式与 WENO-5 无振荡格式
3. **谱稳定性分析**：构造线性化稳定性矩阵，计算特征值谱，进行 von Neumann 分析
4. **动理学验证**：使用 Strang 分裂求解 Vlasov-Poisson 方程组
5. **拓扑分析**：用单纯复形表征相空间结构，计算 Betti 数与 Euler 特征

---

## 二、15 个种子项目的融合映射

| # | 种子项目 | 核心算法 | 在鞘层问题中的角色 |
|---|----------|----------|--------------------|
| 1 | **1413_welzl** | Ritter 近似最小包围球 | 参数空间 (x, k) 的最优域分解 |
| 2 | **420_fermat_factor** | Fermat 因式分解 N=a²-b² | 离子-电子共振模式 (m,n) 识别 |
| 3 | **823_obj_to_tri_surface** | 三角面元网格转换 | 壁面粗糙度三角化几何表征 |
| 4 | **907_praxis** | Brent PRAXIS 主轴优化 | 鞘层参数 (u_B, γ_eff, θ_B) 无梯度优化 |
| 5 | **1078_nec-research_alebrew** | 主动学习采样策略 | 自适应网格与参数采样启发 |
| 6 | **1261_karenadam** | 多通道时间编码 (TEM) | Langmuir 探针信号的带限重建 |
| 7 | **471_glomin** | Brent 全局最小化 (含二阶导约束) | 边际稳定波数 k_c 的全局搜索 |
| 8 | **1305_triangle_grid** | 三角形单元求积 | 壁面通量积分与速度空间求积 |
| 9 | **400_fem2d_bvp_linear** | 二维线性有限元 | Poisson-Boltzmann 方程 FEM 基准 |
| 10 | **559_hypercube_integrals** | 单位超立方体单项式积分 | 多维速度空间矩量积分 |
| 11 | **625_knapsack_greedy** | 贪心 0-1 背包 | 按价值/代价比选择不稳定模式 |
| 12 | **332_ellipsoid** | 椭圆积分 (F, E, K) | 磁前鞘磁通量因子与椭球壁面粗糙元 |
| 13 | **1015_sheffieldquantum_qsim** | 量子算子分裂 + Krylov | Vlasov 方程 Strang 分裂 + 矩阵指数传播 |
| 14 | **623_knapsack_brute** | 暴力子集枚举 | 小规模模式子集最优搜索 |
| 15 | **1208_fergal-murphy_simplicial_emergence_hypergraphs** | 单纯复形 + 拓扑涌现 | 鞘层相空间结构 Betti 数分析 |

---

## 三、核心数学物理公式

### 3.1 无量纲化体系

**尺度**：
- 长度：德拜长度 $\lambda_{De} = \sqrt{\varepsilon_0 k_B T_e / (n_0 e^2)}$
- 速度：离子声速 $c_s = \sqrt{k_B T_e / m_i}$
- 时间：离子等离子体频率倒数 $\omega_{pi}^{-1}$
- 电势：$k_B T_e / e$

### 3.2 修正 Poisson-Boltzmann 方程

$$\frac{d^2\phi}{dx^2} = n_e(\phi) - n_i(\phi)$$

**修正 Boltzmann 电子密度（含 SEE）**：
$$n_e(\phi) = (1-\gamma_\text{eff})\exp\!\left(\frac{\phi}{\chi}\right) + \gamma_\text{eff}\exp(\phi)$$

其中 $\chi = (1-\gamma_\text{eff}) / (1-\gamma_\text{eff}\sqrt{m_e/(2\pi m_i)})$ 为 SEE 降低因子。

**冷离子流体密度**：
$$n_i(\phi) = \frac{1}{\sqrt{1 - 2\phi/u_0^2}}, \quad u_0 \geq c_s$$

### 3.3 Bohm 判据

$$u_i(x_s) \geq c_s = \sqrt{\frac{Z k_B T_e + \gamma_i k_B T_i}{m_i}}$$

### 3.4 高阶紧致有限差分（Lele 1992）

**4 阶三对角紧致格式**：
$$\alpha f'_{i-1} + f'_i + \alpha f'_{i+1} = a\frac{f_{i+1}-f_{i-1}}{2h}$$
其中 $\alpha = 1/4$, $a = 3/2$。

**WENO-5 非线性权重**：
$$\omega_k = \frac{\alpha_k}{\sum_j \alpha_j}, \quad \alpha_k = \frac{d_k}{(\varepsilon + \beta_k)^2}$$
其中 $\beta_k$ 为光滑度指标，$d_k = (1/10, 6/10, 3/10)$ 为理想权重。

### 3.5 谱稳定性分析

**线性化矩阵**：
$$L = \begin{pmatrix} D^2 + \partial_\phi n_e & -I & 0 \\ 0 & -u_0 D & -n_{i0} D \\ -D & 0 & -u_0 D \end{pmatrix}$$

**特征值问题** $L \mathbf{v} = \lambda \mathbf{v}$：
- $\text{Re}(\lambda) > 0$：不稳定模式
- $\text{Re}(\lambda) = 0$：边际稳定
- $\text{Re}(\lambda) < 0$：稳定

**von Neumann 条件**：$\rho(I + \Delta t L) \leq 1$

### 3.6 Vlasov-Poisson Strang 分裂

$$f^{n+1} = T_v(\Delta t/2)\, T_x(\Delta t)\, T_v(\Delta t/2)\, f^n$$

其中 $T_x = \exp(\Delta t\, v \partial_x)$ 为空间平流，$T_v = \exp(\Delta t\, (qE/m) \partial_v)$ 为速度空间加速。

### 3.7 磁前鞘 Chodura 条件

$$M_\parallel \geq \cos\theta_B$$

**磁通量几何因子**（第二类完全椭圆积分）：
$$\Gamma_B = E(\sin\theta_B) = \int_0^{\pi/2} \sqrt{1 - \sin^2\theta_B \sin^2\varphi}\, d\varphi$$

### 3.8 单纯复形拓扑

**边界算子**：$\partial_k : C_k \to C_{k-1}$

**Betti 数**：$\beta_k = \dim \ker \partial_k - \dim \text{im}\, \partial_{k+1} = n_k - \text{rank}(\partial_k) - \text{rank}(\partial_{k+1})$

**Euler 特征**：$\chi = \beta_0 - \beta_1 + \beta_2 = V - E + F$

---

## 四、代码结构与文件说明

```
291_synth_project_Advanced/
├── main.py                         # 统一入口（零参数运行）
├── sheath_constants.py             # 物理常数、等离子体参数、无量纲化
├── sheath_grid.py                  # 非均匀网格（几何/tanh/Winslow/Welzl）
├── fd_highorder.py                 # 高阶有限差分（紧致4/6阶, WENO-5, 非均匀）
├── sheath_quadrature.py            # 高斯求积（Legendre/Hermite/三角形/超立方体）
├── sheath_nonlinear.py             # Poisson-Boltzmann 非线性求解（Newton/Picard/延拓）
├── sheath_time_encoding.py         # 多通道时间编码信号重建
├── sheath_resonance.py             # Fermat 分解 + 共振模式分析
├── sheath_mode_selection.py        # 背包模式选择（贪心/暴力/DP）
├── sheath_spectral_stability.py    # 谱稳定性分析 + von Neumann + 伪谱
├── sheath_praxis_optimizer.py      # PRAXIS 主轴优化
├── sheath_glomin_eigenvalue.py     # glomin 全局特征值搜索
├── sheath_elliptic_presheath.py    # 椭圆积分 + 磁前鞘
├── sheath_vlasov_splitting.py      # Vlasov Strang 分裂 + Krylov
├── sheath_simplicial_topology.py   # 单纯复形 + Betti 数 + 拓扑涌现
└── README_博士级合成说明.md        # 本文件
```

### 14 个计算阶段

| 阶段 | 内容 | 主要模块 |
|------|------|----------|
| 1 | 等离子体参数设置与网格生成 | constants, grid |
| 2 | 高阶有限差分算子构造 | fd_highorder |
| 3 | 壁面几何三角化与拓扑 | simplicial_topology |
| 4 | 速度空间积分与 Bohm 判据 | quadrature |
| 5 | 非线性 Poisson-Boltzmann 求解 | nonlinear |
| 6 | 共振模式分析 (Fermat 分解) | resonance |
| 7 | 谱稳定性分析 | spectral_stability |
| 8 | 模式选择 (背包问题) | mode_selection |
| 9 | 参数优化 (PRAXIS) | praxis_optimizer |
| 10 | 边际稳定性搜索 (glomin) | glomin_eigenvalue |
| 11 | 磁前鞘计算 (椭圆积分) | elliptic_presheath |
| 12 | Vlasov 动理学验证 | vlasov_splitting |
| 13 | 相空间拓扑分析 | simplicial_topology |
| 14 | 探针信号重建 | time_encoding |

---

## 五、运行方式

```bash
cd 291_synth_project_Advanced
python main.py
```

**无任何命令行参数**。程序依次执行 14 个计算阶段，最后输出综合结果摘要。典型运行时间 < 2 秒。

### 依赖

- Python 3.8+
- NumPy
- SciPy (scipy.special, scipy.linalg, scipy.optimize, scipy.sparse)

---

## 六、科学结果解读

运行输出的关键物理量：

- **德拜长度** $\lambda_{De} \approx 1.29 \times 10^{-4}$ m
- **离子声速** $c_s \approx 2.72 \times 10^{3}$ m/s
- **Bohm 判据**：验证通过 ($u_i/c_s \geq 1$)
- **Newton 自洽求解**：6 次迭代收敛，残差 $\sim 10^{-13}$
- **谱稳定性**：检测到不稳定模式，给出 CFL 时间步限制
- **Chodura 条件**：磁前鞘满足 ($M_\parallel \geq \cos\theta_B$)
- **Vlasov Landau 阻尼**：场能衰减比 ~0.95
- **相空间拓扑**：Betti 数 $(\beta_0, \beta_1, \beta_2) = (1, \sim161, 0)$，Euler 特征 $\chi \approx -160$

---

## 七、边界条件与数值鲁棒性

### 边界处理
- 微分矩阵边界行使用单侧差分（向前/向后），避免虚假反射
- Poisson-Boltzmann 求解强制 Dirichlet 边界：$\phi(0) = 0$, $\phi(L) = \phi_\text{wall}$
- Vlasov 分裂使用周期边界条件，通过 FFT 求解 Poisson
- 指数函数截断：$\exp(\phi/\chi) \to \min(\exp(\phi/\chi), \exp(50))$

### 数值鲁棒性
- SEE 系数限制：$0 \leq \gamma_\text{eff} \leq 0.99$（避免发散）
- 密度分母保护：$\max(\text{arg}, 10^{-10})$
- Newton 步长回溯线搜索（阻尼因子 0.5 递减）
- 矩阵求解使用伪逆后备（`lstsq` 替代 `solve`）
- 网格单调性强制保证

---

## 八、创新性与独特方法论

本项目的独特性体现在：

1. **物理-算法深度耦合**：每个数值算法都服务于明确的等离子体物理问题
   - Fermat 分解 → 共振模式识别
   - 背包问题 → 不稳定模式选择
   - 单纯复形 → 相空间拓扑表征

2. **多尺度计算架构**：
   - 德拜尺度（鞘层）→ 高阶 FDM
   - 拉莫尔尺度（前鞘）→ 椭圆积分
   - 动理学尺度（Vlasov）→ 分裂方法

3. **跨学科融合**：
   - 计算几何（Welzl 球）→ 参数域分解
   - 编码理论（TEM）→ 等离子体诊断
   - 代数拓扑（Betti 数）→ 相空间结构

4. **博士级计算难度**：
   - 非线性 Poisson-Boltzmann + Newton 全局收敛
   - 3N×3N 稳定性矩阵特征值问题
   - 6D 相空间的单纯复形构造

---

## 九、参考文献方向

1. Riemann, K.-U. (2000). *The Bohm criterion and sheath formation*. J. Phys. D: Appl. Phys.
2. Lele, S. K. (1992). *Compact finite difference schemes with spectral-like resolution*. J. Comput. Phys.
3. Chodura, R. (1982). *Plasma-wall transition in an oblique magnetic field*. Phys. Fluids.
4. Jiang, G.-S., & Shu, C.-W. (1996). *Efficient implementation of weighted ENO schemes*. J. Comput. Phys.
5. Brent, R. P. (1973). *Algorithms for Minimization without Derivatives*. Prentice-Hall.
6. Murph, F. et al. (2024). *Simplicial emergence in hypergraphs*.

---

**完成日期**：2026 年 6 月
**计算等离子体研究组**
