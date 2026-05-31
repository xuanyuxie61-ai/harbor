# 极端天气事件归因分析系统（CEEAS）——博士级合成说明

## 一、项目概述

本项目基于 **15 个科研代码项目**的核心算法，融合构建了一个面向**气候科学：极端天气事件归因分析**的博士级计算框架。项目采用 Python 语言实现，包含 14 个 `.py` 文件，统一入口为 `main.py`，零参数可运行。

### 核心科学问题

**如何将极端天气事件（如极端降水、热浪）归因于人为气候变化？**

这是一个前沿的气候科学问题，涉及：
1. **极端事件的空间聚类与尺度识别**（渗流理论）
2. **不规则气候区域上的物理量积分**（三角网格/球面/楔形求积）
3. **大气能量级串动力学**（非线性 stiff ODE）
4. **归因概率的不确定性量化**（蒙特卡洛集合 + 子集距离）
5. **空间相关结构的 Pfaffian 建模**（斜对称矩阵行列式）
6. **多尺度谱分析与 FFT 优化**（质因数分解）

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 在合成项目中的角色 |
|:---|:---|:---|
| **992_r8ri** | R8RI 稀疏矩阵存储与运算 | `sparse_climate_matrix.py`：构建大规模气候网格 Laplacian 稀疏矩阵，用于空间平滑和扩散算子 |
| **1347_triangulation_quad** | 三角网格上的积分估计 | `triangulation_quadrature.py`：在极端事件区域的 Delaunay 三角网格上积分能量/水汽通量 |
| **1404_wdk** | Weierstrass-Durand-Kerner 多项式求根 | `climate_interpolation.py`：求解特征多项式根，识别极端事件主导空间尺度 |
| **865_percolation_simulation** | 二维渗流模拟与连通分量分析 | `climate_percolation.py`：识别极端降水/温度异常的空间连通簇，计算序参量与关联长度 |
| **792_nearest_interp_1d** | 一维最近邻插值 | `climate_interpolation.py`：气候数据重网格化（regridding），将不同分辨率数据对齐 |
| **939_quad_fast_rule** | Fejér/Gauss-Legendre 快速积分 | `fast_spectral_quadrature.py`：大气垂直柱上的快速谱积分（水汽、能量） |
| **1280_toms923** | Parlett-Reid 算法计算 Pfaffian | `covariance_pfaffian.py`：极端事件空间分布的斜对称协方差建模，配分函数计算 |
| **437_flame_ode** | 火焰 stiff ODE + Lambert W 精确解 | `energy_cascade_ode.py`：大气能量级串模型，描述极端事件从扰动到饱和的非线性动力学 |
| **1126_sphere_quad** | 球面二十面体求积 / 蒙特卡洛 | `spherical_climate_quad.py`：全球平均辐射强迫、能量收支的球面积分 |
| **683_line_monte_carlo** | 线段蒙特卡洛采样 | `monte_carlo_ensemble.py`：一维能量级串方程的蒙特卡洛积分验证 |
| **1330_triangulation** | Delaunay 三角剖分（增量/朴素） | `delaunay_mesh.py`：极端事件区域自适应三角网格生成 |
| **911_prime_factors** | 质因数分解 | `spectral_analysis.py`：FFT 长度优化（混合基 FFT），气候周期分析 |
| **118_brc_naive** | 十亿记录聚合（城市温度统计） | `regional_aggregation.py`：区域气候统计聚合（均值、极值、频率、REI 指数） |
| **1407_wedge_felippa_rule** | 楔形区域（三角柱）求积 | `wedge_atmosphere_quad.py`：三维大气柱-三角棱柱区域的物理量体积分 |
| **1177_subset_distance** | 子集 Hamming 距离统计 | `monte_carlo_ensemble.py`：集合成员间极端事件空间模式的距离分析 |

---

## 三、新增数学物理模型与核心公式

### 3.1 渗流理论模型（Percolation Theory）

极端事件识别采用 **site percolation** 模型：

$$U_{ij} = \begin{cases} 1 & \text{if } \phi_{ij} > \theta \\ 0 & \text{otherwise} \end{cases}$$

其中 $\phi_{ij}$ 为标准化气候异常场，$\theta$ 为阈值（通常取 2.0，即 2 个标准差）。

**序参量**（Order Parameter）：
$$P_\infty = \frac{S_{\max}}{N_{\text{total}}}$$

**关联长度**（Correlation Length）：
$$\xi^2 = \frac{2 \sum_s s^2 n_s}{\sum_s s n_s}$$

其中 $n_s$ 为尺寸为 $s$ 的簇数量，$\xi$ 反映极端事件的空间相关尺度。

二维方格渗流阈值：$p_c \approx 0.592746$，临界指数 $\nu = 4/3$。

### 3.2 Delaunay 三角剖分与自适应网格

对极端事件连通分量进行 Delaunay 三角剖分，满足**空外接圆条件**：

对于三角形 $T = (p_i, p_j, p_k)$，其外接圆内部不包含任何其他点 $p_m$。

三角形面积：
$$A_T = \frac{1}{2} \left| x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j) \right|$$

### 3.3 高维数值积分公式

#### (a) 三角网格求积（重心法则 + 高阶 Gauss 规则）

**重心法则**（精度 1）：
$$\int_{\Omega} f \, dA \approx \sum_{T \in \mathcal{T}} |T| \cdot \frac{f(v_1) + f(v_2) + f(v_3)}{3}$$

**7 点 Gauss 规则**（精度 5）：
$$\int_T f \, dA \approx |T| \sum_{k=1}^{7} w_k f(P_k)$$

其中权重与节点见 `triangulation_quadrature.py`。

#### (b) 球面求积（二十面体细分）

单位球面三角形面积（L'Huilier 定理）：
$$\tan\frac{E}{4} = \sqrt{ \tan\frac{s}{2} \tan\frac{s-a}{2} \tan\frac{s-b}{2} \tan\frac{s-c}{2} }$$

其中 $a,b,c$ 为球面三角形的边长，$s = (a+b+c)/2$，$E$ 为球面过剩，面积 $= E \cdot R^2$（$R=1$）。

球面蒙特卡洛：
$$\int_{S^2} f(\Omega) \, d\Omega \approx 4\pi \cdot \frac{1}{N} \sum_{k=1}^{N} f(x_k)$$

#### (c) 快速谱求积（Gauss-Legendre）

Legendre 点通过对称三对角矩阵特征值分解获得：
$$T = \text{tridiag}(\beta, 0, \beta), \quad \beta_k = \frac{1}{2\sqrt{1 - (2k)^{-2}}}$$

$$x_i = \text{eigvec}_0(i), \quad w_i = 2 \cdot \text{eigval}(i)^2$$

大气总水汽含量：
$$\text{TCWV} = \int_0^{z_{\text{top}}} \rho(z) q(z) \, dz$$

#### (d) 楔形区域求积（3D 大气柱）

楔形区域定义：$0 \leq x, 0 \leq y, x+y \leq 1, -1 \leq z \leq 1$

通过张量积组合三角形规则与线段规则：
$$(x,y,z) = (\xi_i, \eta_i, \zeta_j), \quad w_{ij} = w_i^{\text{tri}} \cdot w_j^{\text{line}}$$

### 3.4 Pfaffian 协方差模型

对于斜对称矩阵 $K$（$K^T = -K$），定义反对称指数核：
$$K_{ij} = \text{sgn}(x_i - x_j) \cdot \exp\left(-\frac{|x_i - x_j|}{\xi}\right)$$

**Pfaffian** 通过 Parlett-Reid LTL 分解计算：
$$A = P^T L T L^T P, \quad \text{pf}(A) = (-1)^{\text{swaps}} \prod_{k=1}^{N/2} T_{2k-1, 2k}$$

关键性质：$\text{pf}(K)^2 = \det(K)$

高斯随机场配分函数：
$$Z = \sqrt{|\text{pf}(K)|}, \quad \ln Z = \frac{1}{2} \ln |\text{pf}(K)|$$

### 3.5 能量级串动力学（Stiff ODE）

类比火焰模型的能量级串方程：
$$\frac{dy}{d\tau} = y^2 - y^3 = y^2(1-y)$$

其中 $y = E/E_{\text{eq}}$ 为归一化能量强度。

**精确解**（Lambert W 函数）：
$$y(\tau) = \frac{1}{W\left(A \cdot e^{A-\tau}\right) + 1}, \quad A = \frac{1}{y_0} - 1$$

Lambert W 定义：$W(z) \cdot e^{W(z)} = z$

**饱和时间**：
$$\tau_{\text{sat}} = A - W\left(A \cdot e^A \cdot \left(\frac{1}{\epsilon} - 1\right)\right)$$

### 3.6 蒙特卡洛集合不确定性

集合成员生成：
$$\phi^{(m)} = \phi_0 + \sigma \cdot \mathcal{N}(0, 1), \quad m = 1, \ldots, M$$

Hamming 距离（空间模式差异）：
$$d_H(A, B) = \sum_i \left|\mathbf{1}_A(i) - \mathbf{1}_B(i)\right|$$

归因共识：
$$\text{Consensus}_i = \mathbb{1}\left[ \frac{1}{M} \sum_{m=1}^M \mathbf{1}(\phi_i^{(m)} > \theta) \geq \rho \right]$$

### 3.7 区域极端事件指数（REI）

$$\text{REI}_R = w_1 \frac{\mu_R}{\mu_{\text{global}}} + w_2 \frac{x_{\max,R}}{x_{\max,\text{global}}} + w_3 \cdot f_R$$

其中 $f_R$ 为区域极端事件频率。

### 3.8 谱分析与 FFT 优化

最优 FFT 长度要求质因数仅含 $\{2, 3, 5\}$：
$$N_{\text{opt}} = \min\{ n \geq N_0 : \text{prime\_factors}(n) \subseteq \{2,3,5\} \}$$

混合基 FFT 复杂度：
$$\mathcal{O}\left(N \sum_i e_i p_i\right), \quad N = \prod_i p_i^{e_i}$$

---

## 四、项目文件结构

```
065_synth_project/
├── main.py                          # 统一入口，零参数运行
├── sparse_climate_matrix.py         # 稀疏矩阵（992_r8ri）
├── climate_percolation.py           # 渗流分析（865_percolation_simulation）
├── delaunay_mesh.py                 # Delaunay 三角剖分（1330_triangulation）
├── triangulation_quadrature.py      # 三角网格求积（1347_triangulation_quad）
├── spherical_climate_quad.py        # 球面求积（1126_sphere_quad）
├── fast_spectral_quadrature.py      # 快速谱求积（939_quad_fast_rule）
├── wedge_atmosphere_quad.py         # 楔形区域求积（1407_wedge_felippa_rule）
├── monte_carlo_ensemble.py          # 蒙特卡洛集合（683 + 1177）
├── covariance_pfaffian.py           # Pfaffian 协方差（1280_toms923）
├── climate_interpolation.py         # 插值与求根（792 + 1404）
├── energy_cascade_ode.py            # 能量级串 ODE（437_flame_ode）
├── regional_aggregation.py          # 区域聚合（118_brc_naive）
├── spectral_analysis.py             # 谱分析（911_prime_factors）
└── README_博士级合成说明.md
```

---

## 五、运行方式

```bash
cd Synthesis-project-python/065_synth_project
python main.py
```

无需任何参数，程序将自动生成合成气候数据并执行完整的归因分析流程，输出：
- 极端事件渗流统计（占据概率、序参量、关联长度）
- Delaunay 三角网格信息
- 区域能量/水汽积分结果
- 全球辐射强迫估计
- 三维大气柱能量积分
- 集合归因不确定性（Hamming 距离、共识区域）
- Pfaffian 空间相关分析
- 能量级串动力学模拟
- 区域极端事件指数（REI）
- 主导周期识别与 FFT 优化验证

---

## 六、工程特性

- **边界处理**：所有数值模块包含零值、负值、除零等边界检查
- **数值鲁棒性**：Pfaffian 计算包含斜对称性验证，球面求积包含 clip 防越界
- **模块化设计**：每个 `.py` 文件可独立运行自测试
- **零可视化**：已删除所有图形输出，仅保留数值结果
- **无外部数据依赖**：所有数据通过合成生成，无需额外输入文件
