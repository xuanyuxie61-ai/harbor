# 博士级科研代码合成说明

## 项目名称
**中微子振荡概率与参数反演：高阶有限差分与稳定性分析（小规模可复现实验）**

## 科学问题描述

本项目聚焦于计算高能物理中的一个前沿问题：**三代中微子在真空与变密度物质中传播时的味振荡现象，以及从观测数据中反演 PMNS 混合参数**。该问题位于粒子物理、天体物理与数值分析交叉地带，核心科学困难在于：

1. **量子演化方程**：中微子味态 $|\nu_\alpha\rangle$ ($\alpha = e,\mu,\tau$) 沿传播距离 $x$ 的演化遵循 Schrödinger 型方程
   $$
   i\,\frac{d}{dx}\,\Psi(x) \;=\; H(x)\,\Psi(x),\qquad \Psi \in \mathbb{C}^3,
   $$
   其中哈密顿量 $H(x) = H_{\text{vac}} + V(x)$ 包含真空项与 MSW 物质势。

2. **真空哈密顿量**：
   $$
   H_{\text{vac}} \;=\; \frac{1}{2E}\,U\,\text{diag}(0,\,\Delta m_{21}^2,\,\Delta m_{31}^2)\,U^\dagger,
   $$
   $U$ 为 $3\times 3$ PMNS 混合矩阵，依赖三个混合角 $(\theta_{12},\theta_{23},\theta_{13})$ 与一个 CP 破坏相 $\delta_{CP}$。

3. **MSW 物质效应**：Wolfenstein 势 $V_{CC} = \sqrt{2}\,G_F\,N_e(x)$ 仅在 $\nu_e$ 分量上添加，使有效混合角发生密度依赖的畸变，在共振条件
   $$
   2 E\,V_{CC}(x) \;=\; \Delta m^2\cos 2\theta
   $$
   处出现 MSW 共振。

4. **振荡概率**：
   $$
   P(\nu_\alpha \to \nu_\beta; L) \;=\; \bigl|\,[\exp(-iHL)]_{\beta\alpha}\,\bigr|^2.
   $$

5. **参数反演目标**：从若干基线 $L_k$ 与能量 $E_j$ 下的概率观测 $P_{jk}^{\text{obs}}$，最小化
   $$
   \chi^2(\boldsymbol{\theta}) \;=\; \sum_{j,k}\,\frac{\bigl(P_{jk}^{\text{pred}}(\boldsymbol{\theta}) - P_{jk}^{\text{obs}}\bigr)^2}{\sigma_{jk}^2},
   $$
   反演参数向量 $\boldsymbol{\theta} = (\theta_{12},\theta_{23},\theta_{13},\delta_{CP},\Delta m_{21}^2,\Delta m_{31}^2)$。

6. **数值挑战**：$H(x)$ 沿传播路径变化时，需要用高阶有限差分/有限元离散化 Schrödinger 方程，并对时间（即传播距离）积分格式进行 von Neumann 稳定性分析，保证概率守恒与数值稳定性。

## 15 个种子项目到科学问题的映射

| 种子项目 | 核心算法 | 在本项目中的角色 | 承载文件 |
|---|---|---|---|
| `1275_cognizant-ai-labs_red-paper` | 贝叶斯神经网络 / 随机变分推断 | 参数反演的不确定性量化（后验采样） | `uncertainty_quantification.py` |
| `353_fd1d_advection_ftcs` | FTCS 有限差分格式 | 高阶 FD 离散化 Schrödinger 方程的基底 | `high_order_finite_difference.py` |
| `725_matlab_map` (florida_voronoi) | Voronoi 图 / 边界增广点 | 参数空间 Voronoi 分割用于局部代理 | `mesh_generation.py` |
| `603_jacobi` | Jacobi 迭代与 Jacobi 本征值旋转 | 线性反演子问题迭代与厄米矩阵对角化 | `band_matrix_operations.py` |
| `922_puzzles` (casino_puzzle) | 乘性随机过程 / 几何平均 | CP 相位随机游走的几何平均检验 | `monte_carlo_sampling.py` |
| `580_image_mesh2d` | 2D 三角形网格生成 | 参数空间 $(\theta_{12},\theta_{23})$ 三角形剖分 | `mesh_generation.py` |
| `404_fem2d_heat_rectangle` | 二维有限元热传导 | 1D/2D FEM 求解太阳密度分布方程 | `matter_density_fem.py` |
| `987_r8pbl` | 对称正定带状矩阵压缩存储 | 中微子哈密顿量带状存储与 matvec | `band_matrix_operations.py` |
| `574_image_contrast` | 局部对比度增强 | 振荡信号锐化与噪声抑制算子 | `numerical_utils.py` |
| `418_fem3d_project` | 3D 有限元投影 | 从采样点到 FEM 网格的概率投影 | `matter_density_fem.py` |
| `476_golden_section` | 黄金分割搜索 | 坐标下降中一维线搜索 | `parameter_inversion.py` |
| `567_hypersphere_positive_distance` | 正超球面均匀采样 | PMNS 参数空间超球面采样 | `monte_carlo_sampling.py` |
| `1223_gev26_clpbounds` | CLP 条件线性规划 / 乘子 bootstrap | 振荡参数的置信界估计 | `uncertainty_quantification.py` |
| `1066_vincentdufourdecieux_...-KMC` | KMC 动力学蒙特卡罗 / Arrhenius 外推 | 能谱外推与随机味跃迁模拟 | `monte_carlo_sampling.py` |
| `741_matrix_exponential` | Padé 缩放平方 / Taylor / 本征分解 | 精确演化算子 $\exp(-iHL)$ | `matrix_evolution.py` |

## 项目文件结构

```
226_synth_project_Advanced/
├── main.py                          # 统一入口（零参数）
├── neutrino_physics.py              # PMNS 矩阵、哈密顿量、MSW 有效角
├── high_order_finite_difference.py  # 高阶 FD 算子与 von Neumann 分析
├── matter_density_fem.py            # FEM 求解太阳密度分布 + 3D 投影
├── band_matrix_operations.py        # R8PBL 带状矩阵 + Jacobi 迭代/本征值
├── matrix_evolution.py              # 三种矩阵指数算法 + 演化算子
├── mesh_generation.py               # 2D 三角网格 + Voronoi 分割
├── parameter_inversion.py           # 黄金分割 + GP 代理 + 坐标下降反演
├── monte_carlo_sampling.py          # 超球面采样 + KMC + Arrhenius + 赌场悖论
├── uncertainty_quantification.py    # CLP 界 + Cross-fitting + Bootstrap
├── numerical_utils.py               # 对比度增强、log-sum-exp、特殊函数
└── README_博士级合成说明.md          # 本文档
```

## 核心科学公式汇总

### 1. PMNS 矩阵（PDG 参数化）

$$
U = R_{23}(\theta_{23})\;\Gamma_\delta\;R_{13}(\theta_{13})\;R_{12}(\theta_{12}),
$$
$$
R_{12} = \begin{pmatrix} c_{12} & s_{12} & 0 \\ -s_{12} & c_{12} & 0 \\ 0 & 0 & 1 \end{pmatrix},\quad
R_{13} = \begin{pmatrix} c_{13} & 0 & s_{13}e^{-i\delta} \\ 0 & 1 & 0 \\ -s_{13}e^{i\delta} & 0 & c_{13} \end{pmatrix},\quad
R_{23} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & c_{23} & s_{23} \\ 0 & -s_{23} & c_{23} \end{pmatrix}.
$$

### 2. 物质中的有效混合角（双味近似）

$$
\sin^2 2\theta_m = \frac{\sin^2 2\theta}{(\cos 2\theta - A/\Delta m^2)^2 + \sin^2 2\theta},\qquad A = 2 E V_{CC}.
$$

### 3. 高阶有限差分模板（$2k$ 阶中心差分）

$$
\left.\frac{d^2 u}{dx^2}\right|_i \;\approx\; \frac{1}{dx^2}\sum_{m=-k}^{k} c_m\,u_{i+m},
$$

其中 2 阶：$c = (-1, 2, -1)$；4 阶：$c = \tfrac{1}{12}(-1, 16, -30, 16, -1)$ 等。

### 4. von Neumann 放大因子

- **FTCS**：$g(\omega) = 1 - i\nu\sin(\omega\,dx)$，恒不稳定；
- **RK4**：$g = 1 + z + z^2/2 + z^3/6 + z^4/24,\quad z = -i\nu\sin(\omega\,dx)$；
- **Crank-Nicolson / IMR / Magnus2**：$g = (1+z/2)/(1-z/2)$，满足 $|g|\equiv 1$（A-稳定）。

### 5. Padé(6,6) 缩放平方法

$$
\exp(A) \;\approx\; [D_{66}(A)]^{-1} N_{66}(A),\qquad
\exp(A) = \bigl[\exp(A/2^s)\bigr]^{2^s}.
$$

### 6. $\chi^2$ 反演目标泛函

$$
\chi^2(\boldsymbol\theta) = \sum_{j,k} \frac{\bigl(P_{jk}^{\text{pred}}(\boldsymbol\theta) - P_{jk}^{\text{obs}}\bigr)^2}{\sigma_{jk}^2} \;+\; \lambda\,\|\boldsymbol\theta - \boldsymbol\theta_0\|^2.
$$

### 7. CLP 置信界

$$
\min/\max\; q^\top b \quad \text{s.t.}\quad A b \leq 0,\; \mathbf{1}^\top b = 1,\; b \geq 0,
$$

配合乘子 bootstrap 得到一致置信区间。

## 运行方式

```bash
cd 226_synth_project_Advanced
python main.py
```

零参数运行，依次执行 10 个模块的完整计算流程，控制台输出所有结果与验证信息。

## 计算输出内容

1. **PMNS 矩阵构造与幺正性验证**
2. **真空与物质哈密顿量计算**
3. **物质中有效混合角（MSW 共振）**
4. **2/4/6/8 阶 FD 算子的收敛阶测试**
5. **FTCS / Lax-Wendroff / Leapfrog 格式的 von Neumann 稳定性**
6. **太阳密度剖面的解析与 FEM 求解**
7. **R8PBL 带状矩阵存储与 Jacobi 线性/本征值求解**
8. **三种矩阵指数算法精度对比**
9. **中微子真空振荡概率 $P(\nu_e \to \nu_\mu)$ 沿基线的扫描**
10. **2D 三角形网格 + Voronoi 参数空间分割**
11. **黄金分割搜索 + GP 代理的坐标下降反演**
12. **正超球面采样 + Arrhenius 外推 + KMC + 赌场悖论**
13. **CLP 置信界 + Cross-fitting + 乘子 Bootstrap**
14. **局部对比度增强、log-sum-exp、特殊函数、KL/JS 散度**

## 工程亮点

- **边界处理**：混合角裁剪至 $[0,\pi/2]$；$\delta$ 模 $2\pi$；密度/能量非负校验；概率裁剪至 $[0,1]$；PMNS 幺正性 SVD 投影。
- **数值鲁棒性**：log-sum-exp 避免溢出；带状矩阵利用 Hermitian 结构仅存半带宽；Jacobi 旋转对角化实对称矩阵；条件数监测与 Tikhonov 正则化。
- **单位制一致性**：能量统一为 eV，距离为 m（传播子内部转换为 eV$^{-1}$），密度为 g/cm$^3$ 或 mol/cm$^3$。
- **可复现性**：所有随机种子固定，结果完全确定。

## 难度说明

本项目覆盖了以下博士级计算难点：

1. 3×3 复非对角哈密顿量的变密度传播（MSW + 高阶 FD）
2. Schrödinger 演化子的酉性保持（Magnus / CN / Padé）
3. 非线性 $\chi^2$ 反演 + GP 代理 + 坐标下降
4. 参数空间 Voronoi 分割 + 局部多项式代理
5. CLP 置信界 + 乘子 bootstrap 的渐近一致性
6. 高维超球面采样 + Arrhenius 能量外推
7. Jacobi 本征值旋转与带状压缩存储
8. 高阶 FD 格式的 von Neumann 稳定性完整扫描

## 作者
博士级科研代码合成项目 · 2026
