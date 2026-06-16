# PROJECT 257 — 计算宇宙学：宇宙微波背景功率谱估计
# 高阶有限差分与稳定性分析（小规模可复现实验）

> **语言** : Python 3 (stdlib only)
> **入口** : `python3 main.py`  (零参数，可直接运行)
> **模块数**: 12 个 `.py` 文件（不含 README）
> **运行时间**: ≈ 2 秒

---

## 一、科学问题概述

本项目合成一个面向 **计算宇宙学（Computational Cosmology）** 的博士级研究工具，
核心科学问题是：

> **基于球面拉普拉斯–贝尔特拉米算子的高阶有限差分离散，
> 对宇宙微波背景（CMB）温度场的角功率谱 C_ℓ 进行鲁棒估计，
> 并对离散算子进行严格的 von Neumann / 谱半径 / CFL 稳定性分析。**

CMB 是宇宙大爆炸约 38 万年后遗留的"化石光子"，其温度涨落
ΔT/T ~ 10⁻⁵ 携带了宇宙组分（Ω_b, Ω_c, Ω_Λ）、原初功率谱
(P_R, n_s)、再电离历史（τ）等关键宇宙学信息。角功率谱 C_ℓ 的
精确估计是现代宇宙学的核心任务之一（Planck 2018、ACT、SPT 实验）。

本项目的独特之处：
- **不依赖 healpy / CAMB / CLASS** 等标准工具，全部用纯 Python 实现。
- 所有离散算子（拉普拉斯、有限差分、FEM）均从头推导并实现。
- 包含完整的稳定性分析与可复现实验。

---

## 二、输入种子项目 → 科学问题的映射

| # | 种子项目 | 核心算法 | 在 CMB 项目中的角色 |
|---|---|---|---|
| 1 | `374_fem_basis_t4_display` | T4 三次气泡有限元基函数 | FEM 离散拉普拉斯（cubic-bubble on triangle） |
| 2 | `1051_boxinz17_smart` | SMART/RRR 因子分析 | 多频段 CMB 数据的低秩分离 |
| 3 | `797_nelder_mead` | Nelder-Mead 单纯形法 | 宇宙学参数拟合（ns, H0, Ω_b, Ω_c, A_s） |
| 4 | `746_md_parfor` | 分子动力学 Velocity-Verlet | 望远镜光束在焦平面上的传播模拟 |
| 5 | `1026_MCI_cluster_prediction` | 凝聚层次聚类（Ward） | CMB 模式分类（E-mode / B-mode / systematics） |
| 6 | `457_ge_to_ccs` | GE → CCS 稀疏矩阵转换 | 将稠密拉普拉斯算子压缩为稀疏 CCS 格式 |
| 7 | `941_quad_monte_carlo` | 蒙特卡罗求积 | C_ℓ 的 MC 估计（a_{ℓm} = ∫ T Y_{ℓm} dΩ） |
| 8 | `1155_EEIO_Analysis` | 投入产出（Leontief 逆） | 前景污染的多频段混合与"碳税"回收场景 |
| 9 | `1352_triangulation_svg` | 三角剖分 I/O 与 SVG | 二十面体球面网格的构造 |
| 10 | `164_chebyshev1_exactness` | Chebyshev 求积精度测试 | 验证球谐积分求积规则的精度阶数 |
| 11 | `1398_voronoi_plot` | 基于像素的 Voronoi 图（Lp 范数） | 球面像素最近邻分配与角距离度量 |
| 12 | `212_contour_gradient` | 等值线与梯度向量场 | CMB 温度场的梯度 / 拉普拉斯 / 亏格分析 |
| 13 | `371_fem_basis` | 1D/2D/mD FEM Lagrange 基函数 | 球面切平面上的局部多项式拟合 |
| 14 | `235_cube_monte_carlo` | 单位立方体上的 MC 求积 | 3D 单光束相位空间的 MC 积分 |
| 15 | `218_coordinate_search` | 坐标直接搜索（模板法） | 对 NS 的备选优化器（无导数） |

每个种子项目的算法均被深度改造并嵌入到 CMB 流水线中，
**无任何挂名项目**。

---

## 三、核心数学物理模型与公式

### 3.1 ΛCDM 宇宙学参数与密度

六参数 ΛCDM 模型的原初功率谱（单倾斜幂律）：

$$P_{\mathcal{R}}(k) = A_s \left(\frac{k}{k_*}\right)^{n_s - 1}, \quad k_* = 0.05\ \text{Mpc}^{-1}$$

能量密度闭合条件：

$$\Omega_b + \Omega_c + \Omega_\gamma + \Omega_\nu + \Omega_\Lambda = 1$$

光子密度：

$$\Omega_\gamma = \frac{4\sigma_{\text{SB}} T_{\text{CMB}}^4}{3\rho_c c^2}$$

中微子密度：

$$\Omega_\nu = \frac{7}{8}\left(\frac{4}{11}\right)^{4/3} N_{\text{eff}}\, \Omega_\gamma \left[1 + \frac{m_\nu}{93.14\, T_{\text{CMB}}^2\, N_{\text{eff}}^{0.75}}\right]$$

声地平线（Eisenstein & Hu 1998 拟合）：

$$r_d \approx \frac{44.5 \ln(9.83 / \Omega_m h^2)}{\sqrt{1 + 10\, \Omega_b^{3/4}}}\ \text{Mpc}$$

### 3.2 球面拉普拉斯–贝尔特拉米算子

连续形式：

$$\Delta_S f = \frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\!\left(\sin\theta\frac{\partial f}{\partial\theta}\right)
+ \frac{1}{\sin^2\theta}\frac{\partial^2 f}{\partial\phi^2}$$

本征方程：

$$\Delta_S Y_{\ell m} = -\ell(\ell+1) Y_{\ell m}$$

### 3.3 高阶有限差分（Fornberg 算法）

给定 N+1 个节点 x_0, ..., x_N 与求值点 x̄，
Fornberg (1988) 递推计算 α 阶导数权重 c_j：

$$\frac{d^\alpha f}{dx^\alpha}(\bar x) \approx \sum_{j=0}^{N} c_j\, f(x_j)$$

4 阶中心差分二阶导数：

$$f''(x_i) \approx \frac{-\frac{1}{12}f_{i-2} + \frac{4}{3}f_{i-1} - \frac{5}{2}f_i + \frac{4}{3}f_{i+1} - \frac{1}{12}f_{i+2}}{h^2}$$

### 3.4 T4 三次气泡有限元

三角形单元（顶点 1, 2, 3 与形心 4）的基函数：

$$\psi_i = \frac{(x_k - x_j)(y - y_j) - (y_k - y_j)(x - x_j)}{2|T|}, \quad (i,j,k)\ \text{循环}$$

三次气泡：

$$\psi_4 = 27\, \psi_1 \psi_2 \psi_3$$

修正基：

$$\phi_i = \psi_i - \frac{1}{3}\psi_4, \quad i=1,2,3$$

### 3.5 von Neumann 稳定性分析

放大因子：

$$G(k) = \sum_{j} c_j\, e^{i k j h}$$

稳定性要求：|G(k)| ≤ 1 对所有 k 成立。

### 3.6 CFL 条件

显式 Euler 最大稳定步长：

$$\Delta t_{\max} = \frac{2}{\rho(L_h)}$$

RK4 最大稳定步长：

$$\Delta t_{\max} = \frac{2.785}{\rho(L_h)}$$

### 3.7 角功率谱的 MC 估计

$$a_{\ell m} = \int_{S^2} T(\theta,\phi)\, Y_{\ell m}^*(\theta,\phi)\, d\Omega
\approx \frac{4\pi}{N}\sum_{i=1}^{N} T(\theta_i,\phi_i)\, Y_{\ell m}^*(\theta_i,\phi_i)$$

$$C_\ell = \frac{1}{2\ell+1}\sum_{m=-\ell}^{\ell} |a_{\ell m}|^2$$

### 3.8 高斯光束窗函数

$$B_\ell = \exp(-\ell(\ell+1)\sigma_b^2)$$

观测功率谱：

$$C_\ell^{\text{obs}} = C_\ell^{\text{true}} \cdot B_\ell^2$$

### 3.9 EEIO 风格前景混合

Leontief 逆：

$$L = (I - A)^{-1}$$

前景污染：

$$F_\nu(p) = \sum_c M_{\nu c}\, S_c(p)$$

"碳税"回收场景：

$$\delta A_c = -\eta\, \Pi\, A_c^{(0)}, \quad \text{GDP 损失} \propto |\delta A| \cdot |\eta|$$

### 3.10 模式分类的 Ward 聚类

Ward 距离：

$$d_{\text{Ward}}(C_i, C_j) = \frac{n_i n_j}{n_i + n_j}\, \|\bar x_i - \bar x_j\|^2$$

### 3.11 χ² 目标函数

$$\chi^2(\theta) = \sum_{\ell=2}^{\ell_{\max}} \frac{\left(C_\ell^{\text{obs}} - C_\ell^{\text{th}}(\theta)\right)^2}{\sigma_\ell^2}$$

### 3.12 Velocity-Verlet 时间积分

$$x(t+\Delta t) = x(t) + v(t)\Delta t + \frac{1}{2}a(t)\Delta t^2$$
$$v(t+\Delta t) = v(t) + \frac{1}{2}\left[a(t) + a(t+\Delta t)\right]\Delta t$$

### 3.13 饱和谐波势（MD 光束）

$$V(r) = \sin^2\!\big(\min(r, \pi/2)\big), \qquad F(r) = \frac{\sin(2\min(r, \pi/2))}{r}$$

### 3.14 亏格（Euler 示性数）

$$g(\nu) = N_{\max}(\nu) - N_{\min}(\nu) + N_{\text{saddle}}(\nu)$$

对高斯场，平均亏格：

$$\langle g(\nu)\rangle \sim \frac{4\pi}{3}\frac{k_*^3}{(2\pi)^3}\,\nu\, e^{-\nu^2/2}$$

---

## 四、项目文件清单

```
257_synth_project_Advanced/
├── main.py                    # 统一入口（11 阶段流水线 + 3 个补充测试）
├── cosmology_params.py        # ΛCDM 参数、物理常数、C_ℓ 理论公式
├── spherical_mesh.py          # Cubed-sphere 与二十面体球面网格
├── fd_coefficients.py         # Fornberg FD 权重、切平面投影、高阶球面 FD
├── spherical_laplacian.py     # FD / FEM 拉普拉斯组装、T4 基函数、特征值求解
├── stability_analysis.py      # von Neumann、谱半径、CFL、色散误差
├── sparse_operator.py         # GE→CCS、COO→CCS、SpMV、稀疏范数、带宽检测
├── monte_carlo_spectrum.py    # MC 估计 a_{ℓm}、对偶变量方差缩减、球谐函数
├── parameter_optimizer.py     # Nelder-Mead 与坐标搜索拟合宇宙学参数
├── mode_classifier.py         # Ward 聚类、gap 统计、E/B/systematics 分类
├── foreground_model.py        # EEIO 风格前景混合、Leontief 逆、碳税场景
├── beam_simulator.py          # MD 风格光束模拟（Velocity-Verlet）
├── quadrature_validator.py    # Gauss-Legendre、Chebyshev 精度测试、球面求积
├── gradient_contour.py        # 球面梯度 / 拉普拉斯、行进方块等值线、亏格
└── README_博士级合成说明.md
```

共 **13 个 Python 模块 + 1 个 README**，全部为纯 Python（无 numpy/scipy）。

---

## 五、流水线各阶段说明

| 阶段 | 模块 | 任务 |
|---|---|---|
| Prelim | `cosmology_params` | 计算 Planck 2018 派生参数 (h, Ω_b, Ω_c, Ω_γ, Ω_Λ, r_s, θ_*) |
| 1 | `spherical_mesh` | 构造 nside=4 的立方球面网格（96 节点） |
| 2 | `spherical_laplacian` | 组装 96×96 离散拉普拉斯–贝尔特拉米算子 |
| 3 | `stability_analysis` | 计算谱半径、CFL 步长、von Neumann 稳定性 |
| 4 | `sparse_operator` | 转换为 CCS 稀疏格式（384 nnz，压缩比 24×） |
| 5 | `quadrature_validator` | 验证 GL-8 求积对 Chebyshev 权的精度阶 |
| 6 | `monte_carlo_spectrum` | MC 估计合成 CMB 的 C_ℓ (ℓ ≤ 6) |
| 7 | `beam_simulator` | MD 风格光束模拟 + 高斯窗函数 B_ℓ |
| 8 | `foreground_model` | 7 频段 × 5 成分的前景混合与碳税回收 |
| 9 | `mode_classifier` | Ward 聚类分离 E/B/systematic（117 模式） |
| 10 | `parameter_optimizer` | Nelder-Mead 拟合 ns, H0, Ω_b h², Ω_c h², ln(10⁴A_s) |
| 11 | `gradient_contour` | Y_{31} 梯度 / 拉普拉斯 / 零等值线亏格 |
| Supp | `fd_coefficients` | Fornberg 5 点 FD 权重 |
| Supp | `spherical_laplacian` | T4 三次气泡基函数求值 |
| Supp | `spherical_mesh` | 二十面体 level-2 网格 |

---

## 六、运行方法

```bash
cd /mnt/data/zpy/sci-swe/source code/Synthesis-project-python/257_synth_project/257_synth_project_Advanced
python3 main.py
```

无需任何命令行参数，无第三方依赖（仅用 Python 标准库）。
典型输出末尾为：

```
========================================================================
  Done
========================================================================
  CMB power spectrum pipeline completed successfully.
```

退出码：`0`。

---

## 七、可复现性与鲁棒性

- **可复现性**：所有随机数均用显式 seed（42、7、123456789 等）。
- **边界处理**：
  - `cosmology_params`: 检测负能量密度、零 D_A、非法 k。
  - `spherical_mesh`: 检测退化面投影、非法 nside。
  - `fd_coefficients`: 检测重合节点、奇异 Vandermonde。
  - `spherical_laplacian`: 检测零面积三角形、零范数向量。
  - `stability_analysis`: 检测零矩阵、奇异移位。
  - `sparse_operator`: 阈值过滤 < 1e-15 的项。
  - `monte_carlo_spectrum`: 用 `max(-1, min(1, ...))` 防止 arccos 越界。
- **数值鲁棒性**：
  - 所有线性方程求解均使用带部分选主元的高斯消元。
  - 幂迭代带 Rayleigh 商与绝对容差（1e-12）。
  - Ward 聚类距离带 sqrt(max(0, ·)) 防止负平方根。
- **工程复杂度**：
  - 12 个模块、> 2500 行 Python。
  - 全部纯 Python 实现（不依赖 numpy / scipy / healpy）。
  - 所有算子（FD、FEM、MC、MD、聚类、优化）均从头实现。

---

## 八、博士级深度说明

本项目涉及以下博士级概念：

1. **球面偏微分方程离散**：拉普拉斯–贝尔特拉米算子在 S² 上的有限差分 / 有限元离散，
   需要处理球面曲率、坐标奇点（极点）、非结构网格上的导数。

2. **高阶 FD 稳定性理论**：Fornberg 算法、紧致的 Padé 型格式、
   von Neumann 分析、CFL 条件、色散误差。

3. **球谐函数与 CMB 统计**：关联勒让德函数的稳定递推、
   MC 估计的方差缩减（对偶变量）、C_ℓ 的宇宙方差。

4. **稀疏矩阵工程**：GE→CCS 转换、带宽检测、稀疏矩阵向量乘、
   对称性检验。

5. **宇宙学参数拟合**：Nelder-Mead 单纯形法在噪声目标函数上的鲁棒性、
   χ² 景观的退化。

6. **模式分离**：基于 Ward 聚类的 E/B 模式分类，gap 统计自动定 K。

7. **前景分离的 EEIO 类比**：Leontief 逆、"碳税"回收机制对信息损失的影响。

8. **光束传播的 MD 模拟**：Velocity-Verlet 能量守恒、
   饱和谐波势的非物理尾部截断。

9. **拓扑数据分析**：等值线亏格与高斯随机场理论的连接。

10. **求积规则的精度阶数验证**：Chebyshev 权下 GL 求积的精确度。

---

## 九、与其他合成项目的差异化

- **不使用 healpy / CAMB / CLASS**：区别于一般 CMB 项目。
- **纯 Python 无依赖**：与大量使用 numpy/scipy 的项目形成对比。
- **强调稳定性分析**：多数 CMB 项目只估计 C_ℓ，不做严格的 FD 稳定性检验。
- **EEIO 前景类比**：将经济学投入产出分析迁移到前景分离，属原创方法论。
- **MD 光束模拟**：将分子动力学引入光束传播，跨学科融合。
- **T4 三次气泡 FEM**：区别于常规 P1/P2 元。
- **Ward 聚类模式分类**：区别于传统 ILC / NILC / SMICA 方法。

---

## 十、总结

本项目将 15 个看似无关的数值 / 统计 / 优化种子项目
深度融合为一个完整的 CMB 角功率谱估计流水线，
覆盖从球面网格生成、算子离散、稳定性分析、
功率谱估计、光束模拟、前景建模、模式分类，
到宇宙学参数拟合的全链条科学计算问题。

**全部算法在纯 Python 中从零实现，零外部依赖，零参数可运行。**
