# LHC 超越标准模型（BSM）新物理信号提取与共振峰重建平台

## 项目概述

本项目将 **15 个科研代码种子项目** 的核心算法融合重构为一个面向**粒子物理：超越标准模型新物理信号**领域的博士级数值分析平台。项目模拟了从探测器响应、径迹重建、信号处理到统计推断的完整 BSM 搜索流程，以 LHC 中暗区重媒介子 $Z' \to \ell^+ \ell^-$ 共振信号提取为核心科学问题。

---

## 一、原项目到科学问题的映射

| 编号 | 原项目名称 | 核心算法 | 在合成项目中的角色 |
|:---:|-----------|---------|-------------------|
| 1 | 353_fd1d_advection_ftcs | 1D 平流方程 FTCS 差分 | **detector_response.py**: 带电粒子在探测器材料中能量沉积的一维平流-扩散模拟（改进为稳定 Lax-Wendroff 格式） |
| 2 | 579_image_edge | NEWS 边缘检测算子 | **detector_response.py**: 在二维探测器击中矩阵上识别粒子径迹边界，提取信号区域 |
| 3 | 385_fem1d_approximate | 1D 有限元近似拟合 | **track_reconstruction.py**: 拟合径迹能量损失 dE/dx 分布，用于粒子鉴别（PID） |
| 4 | 1366_tsp_moler | TSP 旅行商启发式求解 | **track_reconstruction.py**: 优化探测器各层间击中点关联路径，重建完整径迹 |
| 5 | 1187_svd_fingerprint | SVD 低秩近似 | **signal_processing.py**: 对径迹图像进行 PCA 降维，构建信号/背景判别器 |
| 6 | 928_pwl_interp_2d_scattered | 散乱数据 2D PWL 插值 | **interpolation_utils.py**: 在探测器非均匀响应区域重建能量击中位置映射 |
| 7 | 993_r8row | 矩阵行快速排序 | **matrix_solver.py**: 对事例数据矩阵按能量/动量排序，用于高效事件选择 |
| 8 | 012_aperiodic_tile | 非周期平铺算法 | **detector_response.py**: 构造探测器像素单元的准晶排列几何，研究系统误差 |
| 9 | 1110_sparse_interp_nd | 多维稀疏网格 Lagrange 插值 | **interpolation_utils.py**: 在 BSM 参数空间进行 Smolyak 稀疏网格扫描 |
| 10 | 437_flame_ode | 火焰增长 ODE | **shower_model.py**: 类比电磁簇射的指数增长与饱和过程，描述能量沉积演化 |
| 11 | 131_c8lib | 复数矩阵运算 | **matrix_solver.py**: 求解复数散射振幅矩阵，处理 Z' Breit-Wigner 传播子 |
| 12 | 123_burgers_pde_etdrk4 | Burgers 方程 ETD-RK4 谱方法 | **shower_model.py**: 模拟强子化过程中的非线性能量密度激波传播 |
| 13 | 628_knapsack_values | 背包问题数据 | **parameter_scan.py**: 在积分光度限制下优化分析通道选择组合 |
| 14 | 518_hermite_cubic | Hermite 三次样条 | **track_reconstruction.py**: 平滑重建连续径迹，提取曲率与动量信息 |
| 15 | 1003_r8utt | 上三角 Toeplitz 求解 | **matrix_solver.py**: 快速求解探测器响应矩阵的反卷积问题 |

---

## 二、新增数学物理模型与核心公式

### 2.1 Z' 玻色子 Breit-Wigner 共振模型

在 $U(1)'$ 规范扩展模型中，$Z'$ 是额外的有质量规范玻色子。其传播子为：

$$\hat{\Delta}_{\mu\nu}(q) = \frac{-g_{\mu\nu} + q_\mu q_\nu / M_{Z'}^2}{q^2 - M_{Z'}^2 + i M_{Z'} \Gamma_{Z'}}$$

对于无质量外腿（$e^+ e^- \to \ell^+ \ell^-$），流守恒抑制 $q_\mu q_\nu$ 项，标量传播子简化为：

$$D(s) = \frac{1}{s - M_{Z'}^2 + i M_{Z'} \Gamma_{Z'}}$$

微分截面（$Z'$ 贡献，极化求和后）：

$$\frac{d\sigma_{Z'}}{d\Omega} = \frac{\alpha_{em}^2}{4s} \left[ A_{Z'}(1+\cos^2\theta) + B_{Z'}\cos\theta \right] |D(s)|^2 s^2$$

其中耦合系数：
$$A_{Z'} = (g_\ell^{V2} + g_\ell^{A2})(g_q^{V2} + g_q^{A2}), \quad B_{Z'} = 4 g_\ell^V g_\ell^A g_q^V g_q^A$$

### 2.2 有效场论（EFT）接触相互作用

在质量远大于 $\sqrt{s}$ 的重媒介子积分掉后，产生有效四费米子算符：

$$\mathcal{L}_{\text{CI}} = \frac{2\pi}{\Lambda^2} \sum_{i,j=L,R} \eta_{ij} (\bar{\ell}_i \gamma^\mu \ell_i)(\bar{q}_j \gamma_\mu q_j)$$

对截面的修正：
$$\delta\sigma_{\text{CI}} = \frac{\pi \alpha_{em}^2}{2 \Lambda^4} \left[ (\eta_{LL}^2 + \eta_{RR}^2)(1 + \cos^2\theta) + 2\eta_{LR}^2(1 - \cos^2\theta) \right] s$$

### 2.3 探测器能量沉积的平流-扩散方程

带电粒子穿过探测器敏感体积时，能量沉积满足：

$$\frac{\partial E}{\partial t} = -v \frac{\partial E}{\partial x} + D \frac{\partial^2 E}{\partial x^2}$$

数值离散采用 Lax-Wendroff 格式：

$$u_j^{n+1} = u_j^n - \frac{c\Delta t}{2\Delta x}(u_{j+1}^n - u_{j-1}^n) + \frac{c^2 \Delta t^2}{2\Delta x^2}(u_{j+1}^n - 2u_j^n + u_{j-1}^n) + \frac{D\Delta t}{\Delta x^2}(u_{j+1}^n - 2u_j^n + u_{j-1}^n)$$

### 2.4 电磁簇射的火焰 ODE 类比

火焰 ODE 描述半径增长：
$$\frac{dy}{dt} = y^2 - y^3 = y^2(1-y)$$

在簇射物理中的类比：
- $y$：归一化簇射能量
- $y^2$ 项：簇射指数增长（对产生 + bremsstrahlung）
- $y^3$ 项：饱和效应（电离损失主导）

精确解含 Lambert W 函数：
$$y(t) = \frac{1}{W(A e^{A-t}) + 1}, \quad A = \frac{1}{\delta} - 1$$

### 2.5 强子化的 Burgers 方程

Burgers 方程模拟非线性能量沉积：
$$\frac{\partial u}{\partial t} = -\frac{1}{2}\frac{\partial(u^2)}{\partial x} + \nu \frac{\partial^2 u}{\partial x^2}$$

数值方法采用 Kassam-Trefethen 的 ETD-RK4 谱方法，在 Fourier 空间处理线性粘性项，物理空间处理非线性对流项：

$$v_{n+1} = E v_n + f_1 N(v_n) + 2f_2 [N(a) + N(b)] + f_3 N(c)$$

其中 $N(v) = g \cdot \mathcal{F}[(\mathcal{F}^{-1}[v])^2]$，$g = -0.5ik$。

### 2.6 有限元 dE/dx 拟合

最小化加权泛函：
$$J = w_a \sum_i (\text{FEM}(x_i) - E_i)^2 + w_d \sum_j (\text{FEM}''(x_j))^2 + w_b [\text{FEM}(0)^2 + \text{FEM}(L)^2]$$

转化为超定线性方程组 $A \mathbf{c} = \mathbf{b}$，通过最小二乘法求解有限元系数 $\mathbf{c}$。

### 2.7 Hermite 三次样条

区间 $[x_1, x_2]$ 上的 Hermite 插值：
$$p(x) = f_1 H_{00}(t) + h d_1 H_{10}(t) + f_2 H_{01}(t) + h d_2 H_{11}(t)$$

其中 $t = (x - x_1)/h$，基函数：
$$H_{00}(t) = 2t^3 - 3t^2 + 1, \quad H_{10}(t) = t^3 - 2t^2 + t$$
$$H_{01}(t) = -2t^3 + 3t^2, \quad H_{11}(t) = t^3 - t^2$$

### 2.8 上三角 Toeplitz 矩阵快速求解

探测器点扩散函数（PSF）离散化形成 Toeplitz 矩阵 $K$，反卷积需求解 $K \mathbf{x} = \mathbf{b}$。

利用 Toeplitz 结构的回代算法（复杂度 $O(N^2)$）：
$$x(j) = \frac{b(j)}{a_0}, \quad x(i) \leftarrow x(i) - a_{j-i} x(j) \quad (i = 0, \ldots, j-1)$$

### 2.9 Smolyak 稀疏网格插值

在 $d$ 维参数空间 $(M_{Z'}, g_q, \Gamma_{Z'}/M_{Z'})$ 中，Smolyak 公式：
$$A(q,d) = \sum_{|\ell|_1 \leq q} c(\ell) \bigotimes_{i=1}^d U^{\ell_i}$$

组合系数：
$$c(\ell) = (-1)^{|\ell|_1 - d} \binom{d-1}{|\ell|_1 - d}$$

相比全张量积网格 $O(N^d)$，稀疏网格将节点数降至 $O(N(\log N)^{d-1})$。

### 2.10 统计推断：CL_s 方法

95% CL 排除限基于 CL_s 检验：
$$\text{CL}_s = \frac{\text{CL}_{s+b}}{\text{CL}_b}$$

简化的 Asimov 显著性：
$$Z = \sqrt{2\left[(s+b)\ln\left(1+\frac{s}{b}\right) - s\right]}$$

---

## 三、项目文件结构

```
040_synth_project/
├── main.py                  # 统一入口，零参数运行
├── bsm_physics.py           # BSM 物理模型（Z' 传播子、EFT、散射振幅）
├── matrix_solver.py         # 特殊矩阵求解（复数 LU、行排序、Toeplitz）
├── detector_response.py     # 探测器模拟（平流-扩散、边缘检测、非周期几何）
├── track_reconstruction.py  # 径迹重建（TSP、Hermite 样条、FEM 拟合）
├── shower_model.py          # 簇射模型（火焰 ODE、Burgers PDE、强子化）
├── signal_processing.py     # 信号处理（SVD 降维、PCA、峰搜索）
├── interpolation_utils.py   # 多维插值（2D PWL、Smolyak 稀疏网格）
├── parameter_scan.py        # 参数扫描（背包优化、发现潜力）
├── event_selection.py       # 事例选择与统计推断
└── README_博士级合成说明.md   # 本文档
```

---

## 四、各模块功能详解

### 4.1 bsm_physics.py
- `ZPrimeModel` 数据类：封装 $Z'$ 模型参数，含幺正性边界检查
- `breit_wigner_propagator()`: 计算复数传播子，处理极点规避
- `dilepton_cross_section()`: 计算 $e^+e^- \to \ell^+\ell^-$ 微分截面
- `eft_contact_interaction()`: EFT 接触相互作用修正
- `scattering_amplitude_matrix()`: 构建复数振幅矩阵（融入 c8lib 思想）

### 4.2 matrix_solver.py
- `c8mat_fss()`: 复数矩阵多右端项 LU 分解与求解（来自 131_c8lib）
- `r8row_sort_quick_a()`: 矩阵行字典序快速排序（来自 993_r8row）
- `r8utt_sl()`: 上三角 Toeplitz 矩阵快速回代（来自 1003_r8utt）
- `detector_deconvolution_toeplitz()`: Tikhonov 正则化反卷积

### 4.3 detector_response.py
- `advection_diffusion_energy_deposit()`: 稳定格式的 1D 能量沉积模拟（来自 353_fd1d_advection_ftcs，FTCS 改进为 Lax-Wendroff）
- `news_edge_detector()`: NEWS 边缘检测（来自 579_image_edge）
- `detector_hit_map()`: 模拟 LHC 硅径迹探测器的二维击中图
- `aperiodic_detector_geometry()`: 非周期像素排列（来自 012_aperiodic_tile）
- `detector_energy_resolution()`: 参数化能量分辨率 $\sigma_E/E = a/\sqrt{E} \oplus b \oplus c/E$

### 4.4 track_reconstruction.py
- `tsp_track_association()`: TSP 启发式优化层间击中关联（来自 1366_tsp_moler）
- `hermite_cubic_spline()`: 分段 Hermite 三次样条平滑（来自 518_hermite_cubic）
- `estimate_momentum_from_curvature()`: 圆拟合提取横向动量 $p_T = 0.3 B R$
- `fem1d_track_fit()`: 有限元拟合 dE/dx（来自 385_fem1d_approximate）
- `particle_id_from_dedx()`: 基于 Bethe-Bloch 曲线的粒子鉴别

### 4.5 shower_model.py
- `lambert_w_approx()`: Lambert W 函数 Halley 迭代求解
- `flame_ode_solve()`: 火焰 ODE 数值求解（来自 437_flame_ode）
- `electromagnetic_shower_profile()`: 簇射纵向剖面
- `burgers_hadronization_pde()`: Burgers 方程 ETD-RK4 求解（来自 123_burgers_pde_etdrk4）
- `hadronization_energy_spectrum()`: Lund 碎裂模型近似

### 4.6 signal_processing.py
- `svd_low_rank_approximation()`: SVD 截断低秩近似（来自 1187_svd_fingerprint）
- `singular_value_entropy()`: 奇异值谱 Shannon 熵 $S = -\sum p_i \ln p_i / \ln r$
- `signal_background_discriminator()`: PCA 投影构建判别分数
- `pca_denoise()`: 基于方差阈值的 PCA 噪声抑制
- `resonance_peak_finder()`: 滑动窗口共振峰搜索

### 4.7 interpolation_utils.py
- `pwl_interp_2d_scattered()`: 散乱数据二维 PWL 插值（来自 928_pwl_interp_2d_scattered）
- `sparse_interp_nd_value()`: Smolyak 稀疏网格多维 Lagrange 插值（来自 1110_sparse_interp_nd）
- `bsm_cross_section_interp_2d()`: BSM 参数空间双线性插值

### 4.8 parameter_scan.py
- `knapsack_channel_selection()`: 0/1 背包动态规划优化通道选择（来自 628_knapsack_values）
- `smolyak_parameter_scan()`: 三维参数空间稀疏网格采样
- `discovery_potential()`: Asimov 近似发现潜力评估

### 4.9 event_selection.py
- `reconstruct_invariant_mass()`: 四动量重建不变质量 $M = \sqrt{(E_1+E_2)^2 - (\vec{p}_1+\vec{p}_2)^2}$
- `generate_signal_events()`: Breit-Wigner 分布接受-拒绝采样
- `generate_drell_yan_background()`: Drell-Yan 背景逆变换采样
- `cl_s_limit()`: CL_s 假设检验
- `run_full_analysis()`: 完整分析流程（直方图构建、峰搜索、限值计算）

---

## 五、如何运行

在项目目录下执行：

```bash
python main.py
```

程序将零参数运行，依次执行以下模块：
1. Z' 玻色子 Breit-Wigner 共振模型计算
2. 特殊结构矩阵求解演示
3. 探测器能量沉积与边缘检测
4. 径迹重建与 Hermite 样条平滑
5. 电磁簇射与强子化模型
6. SVD 降维与信号/背景判别
7. 散乱数据与稀疏网格插值
8. 背包优化与发现潜力评估
9. 完整信号分析与统计推断

---

## 六、科学问题说明

本项目解决的核心科学问题是：

> **在 LHC 高能对撞数据中提取超越标准模型的 $Z'$ 重媒介子共振信号，并通过多维参数扫描与统计推断排除或发现新物理。**

具体包括：
- **探测器层面**：模拟带电粒子在硅微条/像素探测器中的能量沉积、噪声和分辨率效应
- **重建层面**：将各探测层的孤立击中点关联为完整径迹，提取动量和粒子种类
- **分析层面**：在双轻子不变质量谱中寻找偏离标准模型背景的共振峰
- **推断层面**：计算 95% CL 排除限和发现潜力，优化分析通道选择

代码中注入了大量前沿物理公式，包括：
- 规范玻色子的 Breit-Wigner 传播子与幺正性约束
- 有效场论四费米子接触相互作用
- Bethe-Bloch 能量损失与粒子鉴别
- 电磁簇射的纵向发展与强子化碎裂函数
- CL_s 统计检验与 Asimov 数据集近似

---

## 七、边界处理与数值鲁棒性

- **传播子极点规避**：当 $s \approx M_{Z'}^2$ 时分母接近零，代码自动添加 $10^{-12}$ 阻尼
- **ODE 刚性处理**：火焰 ODE 采用隐式梯形法，Burgers PDE 采用 ETD-RK4 谱方法
- **排序稳定性**：行排序采用三向分区（Dutch National Flag），正确处理相等元素
- **Toeplitz 奇异检查**：对角元接近零时抛出异常而非崩溃
- **能量非负约束**：所有能量相关计算强制下限为 0
- **质量范围裁剪**：插值查询超出网格时自动截断到边界
- **背包容量非负**：动态规划前对所有参数取正并取整
