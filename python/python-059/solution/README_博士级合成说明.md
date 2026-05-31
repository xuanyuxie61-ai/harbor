# 气溶胶辐射效应与气候反馈综合分析系统 — 博士级合成说明

**项目编号**: PROJECT_59  
**科学领域**: 大气科学 — 气溶胶辐射效应与气候反馈  
**合成语言**: Python 3  
**合成日期**: 2026-05-03

---

## 一、项目概述

本项目将 **15 个独立的科研算法项目** 融合重构为一个面向**大气科学前沿**的博士级综合计算系统。核心科学问题为：

> **在全球尺度上，定量分析气溶胶微物理特性（粒径分布、混合态、折射率）如何通过 Mie 散射与辐射传输过程影响大气辐射收支，并进一步驱动云凝结核（CCN）活化与气候反馈。**

系统涵盖从 **纳米级粒子微物理 → 米级辐射传输 → 百公里级全球统计反演** 的多尺度耦合计算，难度达到自然科学研究生的博士论文级别。

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 在本项目中的科学角色 |
|---|---|---|
| 1099_sor | SOR 迭代求解线性系统 | **辐射传输方程离散化后的高效求解器**。将一维平面平行大气的离散坐标辐射传输方程转化为大型稀疏线性系统，使用逐次超松弛法（SOR）迭代求解辐射强度场。 |
| 1273_toms515 | 组合数字典序生成 | **气溶胶粒径分档优化**。从连续粒径分布中按字典序最优选取代表性粒径档（bins），最小化数值离散化误差。 |
| 173_chrominoes | 颜色计数分类 | **气溶胶混合态分类**。将粒子表面离散为 m×n 网格，C 种化学组分按颜色计数公式分类，计算内混/外混指数 χ。 |
| 1335_triangulation_delaunay_discrepancy | Delaunay 三角化质量度量 | **全球大气网格质量评估**。对球面经纬度网格进行三角化，计算 Delaunay 离散度与最小角指标，确保辐射通量计算网格无病态畸变。 |
| 306_distance_to_position | 距离矩阵反演低维坐标 | **气溶胶排放源反演定位**。由地面观测站的浓度差异构造伪距离矩阵，通过多维标度法（MDS）结合非线性最小二乘反演排放源的三维空间位置。 |
| 1246_tetrahedron_felippa_rule | 四面体高斯求积 | **非球形气溶胶粒子体积积分**。对立方体、六面体等复杂粒子形状进行三维高精度数值积分，计算等效光学截面积。 |
| 186_cities | 经纬度球面距离计算 | **全球观测网络几何**。计算站点间大圆距离，用于空间协方差建模与网格距离矩阵构建。 |
| 029_asa053 | Wishart 随机矩阵生成 | **AOD 观测协方差统计建模**。生成符合 Wishart 分布的样本协方差矩阵，用于评估遥感反演的不确定性。 |
| 661_legendre_polynomial | 勒让德多项式递推计算 | **散射相函数的勒让德展开**。将 Mie 散射相函数展开为勒让德级数，供辐射传输方程的散射积分使用。 |
| 1148_square_minimal_rule | 正方形最小点高斯求积 | **辐射通量角积分**。对立体角进行二维高斯积分，计算辐射通量与加热率。 |
| 1422_xyl_display | 点线几何结构处理 | **大气分层边界定义**。将大气层顶/层底视为几何线段，定义射线追踪的边界条件。 |
| 094_bisection | 二分法根搜索 | **临界相对湿度与饱和蒸气压求根**。在 Kohler 理论中用于求解临界过饱和度的隐式方程。 |
| 1011_random_walk_3d_simulation | 三维随机游走 | **蒙特卡洛光子传输模拟**。模拟光子在气溶胶-云大气中的多次散射随机行走，估算有效光学厚度与逃逸概率。 |
| 702_logistic_ode | Logistic 增长方程 | **CCN 活化动力学**。描述气溶胶粒子在过饱和环境下的活化分数随时间的 Logistic 增长。 |
| 165_chebyshev1_rule | 第一类切比雪夫求积 | **角度散射积分的奇异性处理**。处理辐射传输中 1/sqrt(1-μ²) 权重的奇异积分。 |

---

## 三、新增数学物理模型与核心公式

### 3.1 气溶胶微物理：多模态对数正态分布

气溶胶粒径分布采用经典的多模态对数正态模型：

$$
n(r) = \sum_{k=1}^{K} \frac{N_k}{\sqrt{2\pi} \, r \ln\sigma_{g,k}} \exp\!\left( -\frac{(\ln r - \ln r_{m,k})^2}{2\ln^2\sigma_{g,k}} \right)
$$

其中 $N_k$ 为第 $k$ 模态的总数浓度，$r_{m,k}$ 为几何中值粒径，$\sigma_{g,k}$ 为几何标准差。

### 3.2 有效介质近似：Bruggeman 方程

对于内混气溶胶粒子，等效复折射率 $m_{\text{eff}}$ 通过 Bruggeman 自洽方程求解：

$$
\sum_{i} f_i \frac{m_i^2 - m_{\text{eff}}^2}{m_i^2 + 2m_{\text{eff}}^2} = 0
$$

其中 $f_i$ 为第 $i$ 种化学组分的体积分数，$m_i$ 为其复折射率。本项目使用 Newton 迭代法求解该非线性复方程。

### 3.3 Mie 散射：消光截面与相函数

小参数 ($x = 2\pi r/\lambda \ll 1$) 下的消光效率由 Rayleigh-Gans 近似给出：

$$
Q_{\text{ext}} = \frac{8}{3}x^4 \left| \frac{m^2-1}{m^2+2} \right|^2 + 4x \, \Im\!\left( \frac{m^2-1}{m^2+2} \right)
$$

对于中等尺度 ($0.1 < x < 50$)，采用 van de Hulst 近似：

$$
Q_{\text{ext}} = 2 - 4e^{-\rho\tan\beta}\frac{\cos\beta}{\rho}\sin(\rho-\beta) - 4e^{-\rho\tan\beta}\left(\frac{\cos\beta}{\rho}\right)^2\cos(\rho-2\beta)
$$

其中 $\rho = 2x(n_r - 1)$, $\tan\beta = n_i/(n_r - 1)$。

散射相函数采用 Henyey-Greenstein 近似：

$$
P_{\text{HG}}(\cos\Theta) = \frac{1-g^2}{(1+g^2-2g\cos\Theta)^{3/2}}
$$

其勒让德展开系数为 $a_\ell = g^\ell$：

$$
P_{\text{HG}}(\cos\Theta) = \sum_{\ell=0}^{\infty} \frac{2\ell+1}{4\pi} a_\ell P_\ell(\cos\Theta)
$$

### 3.4 辐射传输方程与 SOR 求解

一维平面平行大气的辐射传输方程（RTE）：

$$
\mu \frac{dI(\tau,\mu)}{d\tau} = I(\tau,\mu) - J(\tau,\mu)
$$

源函数包含多次散射与热发射：

$$
J(\tau,\mu) = \frac{\omega}{2}\int_{-1}^{1} P(\mu,\mu') I(\tau,\mu') \, d\mu' + (1-\omega)B(\tau)
$$

采用离散坐标法（Discrete Ordinates）将角度积分替换为高斯-勒让德求和，深度方向采用迎风差分，最终得到线性代数系统 $A\mathbf{I} = \mathbf{b}$。

使用 **逐次超松弛法 (SOR)** 迭代求解：

$$
I_i^{(k+1)} = (1-\omega_{\text{SOR}}) I_i^{(k)} + \frac{\omega_{\text{SOR}}}{A_{ii}} \left( b_i - \sum_{j<i} A_{ij} I_j^{(k+1)} - \sum_{j>i} A_{ij} I_j^{(k)} \right)
$$

收敛条件：$\|\mathbf{I}^{(k+1)} - \mathbf{I}^{(k)}\|_\infty < 10^{-10}$。

### 3.5 蒙特卡洛光子传输

光子在气溶胶大气中的自由程服从指数分布：

$$
l = -\frac{\ln\xi}{\beta_e}, \quad \xi \sim U(0,1)
$$

散射方向由 HG 相函数的解析反演抽样：

$$
\cos\Theta = \frac{1+g^2 - \left( \frac{1-g^2}{1-g+2g\xi} \right)^2}{2g}
$$

步进更新采用球坐标旋转：

$$
\begin{aligned}
x_{n+1} &= x_n + l \sin\theta \cos\phi \\
y_{n+1} &= y_n + l \sin\theta \sin\phi \\
z_{n+1} &= z_n + l \cos\theta
\end{aligned}
$$

### 3.6 源区反演：多维标度法 (MDS)

由观测浓度 $C_i$ 构造伪距离：

$$
d_i = L \cdot W\!\left( \frac{Q}{4\pi D L C_i} \right)
$$

其中 $W$ 为 Lambert W 函数。距离矩阵 $D_{ij} = |d_i - d_j|$ 经双中心化后特征值分解得到相对位置：

$$
B = -\frac{1}{2} J D^{(2)} J = V\Lambda V^T, \quad X = V_{\text{dim}} \sqrt{\Lambda_{\text{dim}}}
$$

### 3.7 CCN 活化：Köhler 理论与 Logistic 动力学

Köhler 临界过饱和度：

$$
S_{\text{crit}} = \sqrt{\frac{4A^3}{27B}}, \quad A = \frac{2\sigma_w M_w}{RT\rho_w}, \quad B = \frac{\nu M_w V_s}{M_s}
$$

活化分数的 Logistic 演化：

$$
\frac{df_{\text{act}}}{dt} = r f_{\text{act}} \left( 1 - \frac{f_{\text{act}}}{K(S)} \right)
$$

其中饱和活化分数 $K(S)$ 采用 Abdul-Razzak & Ghan 参数化：

$$
K(S) = \frac{1}{2}\left[ 1 - \text{erf}\!\left( \frac{\ln(S_{\text{crit}}/S)}{\sqrt{2}\ln\sigma_g} \right) \right]
$$

### 3.8 统计协方差：Wishart 分布与 EOF

AOD 理论协方差矩阵采用指数衰减空间相关模型：

$$
\Sigma_{ij} = \sigma_{\text{AOD}}^2 \exp\!\left( -\frac{d_{ij}}{L} \right)
$$

其样本协方差服从 Wishart 分布 $W_p(n, \Sigma)$：

$$
f(S) = \frac{|S|^{(n-p-1)/2} \exp(-\text{tr}(\Sigma^{-1}S)/2)}{2^{np/2} |\Sigma|^{n/2} \Gamma_p(n/2)}
$$

经验正交函数（EOF）分解：

$$
\Sigma = V\Lambda V^T, \quad \text{PC}_k(t) = V_k^T \mathbf{x}(t)
$$

### 3.9 高维数值积分引擎

- **四面体积分** (Felippa O24): 在单位四面体 $\{x,y,z \ge 0, x+y+z \le 1\}$ 上使用 24 点高斯规则，代数精度达到 6 阶。
- **正方形积分** (Minimal Rule): 在 $[-1,1]^2$ 上使用最小点高斯求积，避免多余计算点。
- **切比雪夫积分**: 处理带 $1/\sqrt{(x-a)(b-x)}$ 权重的奇异积分，节点为 $x_i = \frac{a+b}{2} + \frac{b-a}{2}\cos\frac{(2i-1)\pi}{2n}$，权重 $w_i = \frac{\pi(b-a)}{2n}$。

---

## 四、模块结构与文件说明

| 文件名 | 功能 | 对应原项目 |
|---|---|---|
| `main.py` | 统一入口，零参数运行，编排全部计算流程 | — |
| `aerosol_microphysics.py` | 粒径分布、混合态、Bruggeman 折射率、分档优化 | 173, 1273 |
| `mie_scattering.py` | 勒让德多项式、HG 相函数、Mie 截面 | 661 |
| `radiative_transfer_solver.py` | RTE 离散化、SOR 迭代、辐射通量 | 1099 |
| `monte_carlo_photon.py` | 3D 蒙特卡洛光子随机游走 | 1011 |
| `atmospheric_mesh.py` | 球面网格、Delaunay 质量、大气分层 | 1335, 186, 1422 |
| `inverse_source.py` | 浓度反演、MDS 定位 | 306 |
| `aerosol_activation.py` | Köhler 理论、Logistic CCN 活化 | 702 |
| `statistical_covariance.py` | Wishart 采样、EOF 分析 | 029 |
| `quadrature_engine.py` | 四面体/正方形/切比雪夫求积 | 1246, 1148, 165 |
| `numerical_utils.py` | 二分法、组合数、正态抽样、WH 变换 | 094, 1273, 029 |

---

## 五、合成后的项目能够解决什么科学问题

1. **气溶胶直接辐射效应估算**：通过 Mie 理论计算消光截面，结合辐射传输方程求解大气加热率，定量评估气溶胶对地表-大气辐射收支的直接扰动。

2. **气溶胶-云相互作用（ACI）参数化**：基于 Köhler 理论和 Logistic 动力学计算 CCN 活化谱，为气候模式中的云微物理参数化提供约束。

3. **排放源归因与反演**：利用地面观测网络的浓度梯度，通过非线性最小二乘反演排放源位置与强度，服务于空气质量管理。

4. **遥感观测不确定性量化**：通过 Wishart 协方差建模与 EOF 分析，评估全球 AOD 遥感产品的空间采样误差与主模态结构。

5. **蒙特卡洛辐射基准**：为复杂气溶胶-云场景提供独立于解析方法的蒙特卡洛基准解，验证辐射传输模式精度。

---

## 六、运行方式

### 环境要求
- Python 3.10+
- NumPy (已安装)
- 无额外依赖（纯 NumPy 实现，确保可移植性）

### 运行命令
```bash
cd Synthesis-project-python/059_synth_project
python main.py
```

程序将自动执行以下流程并输出结果：
1. 气溶胶微物理参数化与粒径分档
2. Mie 散射相函数勒让德展开
3. 辐射传输方程 SOR 求解与通量诊断
4. 三维蒙特卡洛光子传输统计
5. 全球网格质量评估
6. 气溶胶源区反演定位
7. CCN 活化动力学
8. AOD 统计协方差与 EOF 分析
9. 高维数值积分验证
10. 数值工具验证

**运行时间**: 普通 CPU 上约 5~15 秒。

---

## 七、边界处理与数值鲁棒性

本项目在以下方面实施了严格的边界处理与数值鲁棒性设计：

- **参数边界检查**：所有物理参数（粒径、过饱和度、不对称因子等）均在函数入口处进行合法性校验，非法输入抛出带明确信息的异常。
- **浮点安全**：`safe_acos` 函数将输入截断到 $[-1, 1]$，防止 `arccos` 定义域错误；`wilson_hilferty_chi_square` 使用 `abs()` 避免负数开方。
- **发散检测**：SOR 迭代中监控残差增长，若出现 `NaN/Inf` 或范数超过 $10^{30}$，立即终止并返回当前状态。
- **矩阵正则化**：协方差矩阵构造后检查最小特征值，若存在零或负特征值，自动添加微小正定偏移量确保 Cholesky 分解可行。
- **溢出保护**：`logistic_exact` 中指数项超过 700 时直接返回承载量 $K$，避免 `exp` 溢出。

---

## 八、质量检查清单

- [x] 原目录未被修改
- [x] 合成后的项目为 Python 语言
- [x] 新目录完整包含合成后的项目
- [x] 只有一个博士级数学/物理科学计算问题已落地为可执行代码
- [x] **15 个输入项目均已真实融入**，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 已删除所有可视化相关内容
