# PROJECT_91：超声成像波束形成与反演 —— 博士级合成说明

## 1. 项目概述

本项目围绕**声学工程：超声成像波束形成与反演**这一前沿科学领域，融合15个种子项目的核心算法，构建了一个面向多物理场耦合超声层析成像的博士级科学计算平台。

### 核心科学问题

**多物理场耦合下的超声层析成像与介质参数反演**：在非均匀流动介质（Navier-Stokes流场）中，利用多阵元超声阵列进行波束形成成像，考虑压电换能器的瞬态动力学、声波的非线性传播效应、造影剂微泡的扩散增强，以及从多视角传播时间数据中反演介质声速分布的精确层析重建问题。

---

## 2. 原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 在合成项目中的角色 |
|:---:|---|---|---|
| 1 | 1349_triangulation_rcm | Reverse Cuthill-McKee稀疏矩阵重排序 | **acoustic_fem_mesh.py**：声学有限元网格的RCM带宽优化，降低Helmholtz方程离散后的大规模稀疏系统求解复杂度 |
| 2 | 668_levenshtein_distance | 动态规划编辑距离 | **inverse_tomography.py**：A-scan回波序列的符号化比对与组织类型分类（正常组织/肿瘤/囊肿） |
| 3 | 1164_stiff_ode | 刚性ODE模型与隐式求解 | **transducer_dynamics.py**：压电换能器振子的高Q值阻尼振动ODE，以及Lindberg精确解的数值验证 |
| 4 | 362_fd1d_heat_steady | 一维稳态方程有限差分 | **helmholtz_solver.py**：1D/2D Helmholtz方程（声学频域波动方程）的中心差分离散与边界条件处理 |
| 5 | 496_haar_transform | Haar小波正交变换 | **wavelet_denoising.py**：超声A-scan回波信号的多分辨率分解、Donoho-Johnstone通用阈值去噪与多尺度特征提取 |
| 6 | 326_eigenfaces | PCA主成分分析（Turk-Pentland技巧） | **pca_feature_extraction.py**：超声B-scan图像的高维数据降维、特征脸提取与重建误差分析 |
| 7 | 958_quality | 网格质量度量（Q/Alpha/Gamma/D） | **mesh_quality.py**：声学FEM三角网格的Q度量、Alpha角度量、Gamma均匀性度量与带宽分析 |
| 8 | 674_lindberg_exact | Lindberg刚性ODE精确解 | **transducer_dynamics.py**：与stiff_ode协同，提供精确基准验证隐式梯形法的二阶收敛性 |
| 9 | 425_ffmatlib | FEM数据P1插值到规则网格 | **flow_acoustic_coupling.py**：三角形网格上流场数据到矩形规则网格的重心坐标插值 |
| 10 | 787_navier_stokes_2d_exact | 2D Navier-Stokes精确解集 | **flow_acoustic_coupling.py**：Taylor-Green涡精确解作为背景流场，耦合到声学方程 |
| 11 | 119_brownian_motion_simulation | Wiener过程随机游走 | **microbubble_diffusion.py**：超声造影剂微泡在血管中的布朗运动扩散（Stokes-Einstein关系） |
| 12 | 416_fem2d_scalar_display_gpl | FEM标量场三角网格处理 | **flow_acoustic_coupling.py**与**acoustic_fem_mesh.py**：三角网格数据结构和标量场插值处理 |
| 13 | 926_pwl_interp_1d | 分段线性插值 | **helmholtz_solver.py**：声压场从计算网格到传感器位置的插值重建，以及帽子函数基函数构造 |
| 14 | 1096_sncndn | Jacobi椭圆函数（Bulirsch AGM算法） | **nonlinear_acoustics.py**：Burgers方程周期行波解的cn²椭圆函数表示，非线性参数B/A估计 |
| 15 | 569_i4mat_rref2 | 整数矩阵行最简形（纯整数运算） | **inverse_tomography.py**：投影矩阵的精确秩分析、欠定性判定与零空间结构诊断 |

---

## 3. 新增数学物理模型与核心公式

### 3.1 Helmholtz方程（声学频域波动方程）

时谐声波在介质中传播满足:

$$
\nabla^2 p + k^2 p = -f(\mathbf{x}), \quad k = \frac{\omega}{c} = \frac{2\pi f}{c}
$$

其中 $p$ 为复声压，$k$ 为波数，$c$ 为声速。

**一维中心差分离散**:

$$
\frac{p_{i-1} - 2p_i + p_{i+1}}{h^2} + k^2 p_i = -f_i
$$

截断误差 $O(h^2)$，数值色散:

$$
k_{\text{num}}^2 = \frac{4}{h^2}\sin^2\left(\frac{kh}{2}\right) \approx k^2 - \frac{k^4 h^2}{12}
$$

**Sommerfeld吸收边界条件(ABC)**:

$$
\frac{\partial p}{\partial n} = ik \cdot p
$$

### 3.2 压电换能器刚性ODE模型

阻尼受迫振动方程:

$$
m\ddot{u} + c\dot{u} + ku = F(t)
$$

状态空间形式:

$$
\frac{d}{dt}\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -k/m & -c/m \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} + \begin{bmatrix} 0 \\ F(t)/m \end{bmatrix}
$$

**隐式梯形法（Crank-Nicolson）**:

$$
\mathbf{y}_{n+1} = \left(\mathbf{I} - \frac{h}{2}\mathbf{A}\right)^{-1} \left[\mathbf{y}_n + \frac{h}{2}\left(\mathbf{A}\mathbf{y}_n + \mathbf{g}_n + \mathbf{g}_{n+1}\right)\right]
$$

无条件稳定，适用于刚性比 $S = |\text{Re}(\lambda_{\max})| / |\text{Re}(\lambda_{\min})| \gg 1$ 的系统。

### 3.3 Lindberg刚性ODE精确解

模型方程:

$$
y' = \lambda(\cos\omega t - y) + \omega\sin\omega t
$$

精确解:

$$
y(t) = e^{\lambda t}\cos(\omega t) + \sin(\omega t)
$$

### 3.4 Burgers方程与Jacobi椭圆函数

Burgers方程:

$$
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2 u}{\partial x^2}
$$

**周期行波解**（用Jacobi椭圆函数表示）:

$$
u(x,t) = A \cdot \text{cn}^2\left(k(x - ct) \,\big|\, m\right)
$$

其中 $\text{cn}(u|m)$ 为Jacobi椭圆余弦函数，模数 $m \in [0,1]$。

**Bulirsch AGM算法**计算椭圆函数:

$$
\begin{aligned}
a_{n+1} &= \frac{a_n + b_n}{2} \\
b_{n+1} &= \sqrt{a_n b_n} \\
c_{n+1} &= \frac{a_n - b_n}{2}
\end{aligned}
$$

收敛至 $a_\infty = b_\infty = \text{AGM}(a_0, b_0)$。

**非线性参数B/A**:

$$
\frac{B}{A} = 2\rho_0 c_0 \left.\frac{\partial c}{\partial p}\right|_{p=0}
$$

二次谐波生成效率:

$$
\frac{P_2}{P_1} \approx \left(\frac{B}{A} + 2\right) \frac{\pi f z P_1}{2\rho_0 c_0^3}
$$

### 3.5 流-声耦合方程

对流Helmholtz方程（Ma $\ll$ 1近似）:

$$
\nabla^2 P + k^2 P = -\frac{2ik}{c_0}(\mathbf{v} \cdot \nabla P)
$$

**Taylor-Green涡精确解**:

$$
\begin{aligned}
u(x,y,t) &= \sin(x)\cos(y) \cdot e^{-2\nu t} \\
v(x,y,t) &= -\cos(x)\sin(y) \cdot e^{-2\nu t} \\
p(x,y,t) &= \frac{1}{4}(\cos 2x + \cos 2y) \cdot e^{-4\nu t}
\end{aligned}
$$

### 3.6 微泡扩散与Stokes-Einstein关系

扩散系数:

$$
D = \frac{k_B T}{6\pi\eta r}
$$

**Bjerknes声辐射力**:

$$
F = \frac{4\pi r^3 k P_0^2}{3\rho_0 c_0^2}
$$

### 3.7 波束形成与阵列信号处理

**延迟叠加波束形成（DAS）**:

$$
b(t,\theta) = \sum_{n=0}^{N-1} w_n \cdot s_n\left(t - \Delta\tau_n(\theta)\right)
$$

其中 $w_n$ 为窗函数加权，$\Delta\tau_n$ 为聚焦时延。

**-6dB波束宽度**:

$$
\theta_{3\text{dB}} \approx 0.886 \frac{\lambda}{Nd}
$$

### 3.8 PCA降维（Turk-Pentland技巧）

协方差矩阵特征分解:

$$
\mathbf{S} = \frac{1}{N}\sum_{i=1}^{N}(\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T
$$

当 $d \gg N$ 时，计算 $\mathbf{A}^T\mathbf{A}$（$N \times N$）而非 $\mathbf{A}\mathbf{A}^T$（$d \times d$）。

### 3.9 整数精确RREF

整数行最简形通过纯整数运算（加减、GCD约分）实现，完全避免浮点舍入误差:

$$
\text{IRREF}(\mathbf{A}) = \mathbf{R}, \quad \text{其中 } R_{ij} \in \mathbb{Z}
$$

### 3.10 超声层析反演

射线理论传播时间:

$$
T_i = \int_{L_i} \frac{1}{c(x,y)}\,ds = \sum_j A_{ij} m_j
$$

Tikhonov正则化解:

$$
\mathbf{m} = \mathbf{V}\,\text{diag}\left(\frac{\sigma_i}{\sigma_i^2 + \lambda^2}\right)\,\mathbf{U}^T \Delta\mathbf{t}
$$

---

## 4. 项目文件结构

```
091_synth_project/
├── main.py                          # 统一入口，零参数运行
├── acoustic_fem_mesh.py             # 声学网格生成与RCM重排序
├── mesh_quality.py                  # 网格质量评估（Q/Alpha/Gamma度量）
├── transducer_dynamics.py           # 压电换能器刚性ODE动力学
├── helmholtz_solver.py              # Helmholtz方程FDM求解与分段线性插值
├── nonlinear_acoustics.py           # 非线性声学椭圆函数解
├── flow_acoustic_coupling.py        # 流-声耦合与FEM插值
├── microbubble_diffusion.py         # 微泡布朗运动扩散模拟
├── wavelet_denoising.py             # Haar小波去噪与多尺度特征
├── pca_feature_extraction.py        # PCA降维与B-scan特征提取
├── ultrasound_beamforming.py        # 超声阵列波束形成与动态聚焦
├── inverse_tomography.py            # 断层反演与整数RREF精确分析
└── README_博士级合成说明.md         # 中文说明文档
```

---

## 5. 运行方式

```bash
cd /mnt/data/zpy/sci-swe/source\ code/Synthesis-project-python/091_synth_project
python main.py
```

**无需任何参数**，程序将自动执行全部10个模块的完整计算流程，
输出各模块的科学计算结果与精度验证指标。

---

## 6. 科学问题深度说明

### 6.1 为什么这是博士级问题？

1. **多物理场耦合**：同时考虑声学波动、流体力学（Navier-Stokes）、
   非线性介质响应、随机扩散（布朗运动）四种物理机制的相互作用。

2. **高阶数值方法**：隐式梯形法（无条件稳定）、SVD正则化反演、
   整数精确线性代数、Jacobi椭圆函数解析解等多种高级数值技术集成。

3. **工程鲁棒性**：全面的边界条件处理（Dirichlet/Neumann/ABC）、
   数值色散分析、矩阵条件数检查、GCD约分防止整数溢出、伪逆fallback机制。

4. **前沿科学计算**：超声层析成像是医学成像和工业无损检测的前沿方向，
   涉及反问题求解、稀疏信号处理、多尺度分析等现代计算数学核心课题。

### 6.2 计算难度量化

- 稀疏线性系统规模: 可达 $10^4 \times 10^4$（RCM优化后带宽降低至 $O(\sqrt{N})$）
- 波束形成计算: 64阵元 × 2048采样点 × 100聚焦深度 = $1.3 \times 10^7$ 次操作
- 微泡扩散模拟: 500粒子 × 500步 × 2D = $5 \times 10^5$ 次随机位移计算
- 整数RREF: 80×64矩阵的纯整数高斯消元

---

## 7. 边界处理与数值鲁棒性

1. **网格质量检查**：自动剔除Q度量 < 0.1或Alpha度量 < 0.1的病态三角形。
2. **矩阵条件数监控**：当条件数 > $10^{14}$ 时自动切换为正则化求解。
3. **插值边界处理**：外推时clamp到边界区间，防止数值发散。
4. **椭圆函数数值修正**：sn值超出[-1,1]时自动截断并修正cn=0。
5. **反射边界条件**：微泡模拟中采用镜面反射而非周期边界，更符合物理实际。
6. **GCD约分保护**：整数RREF中每一步都进行GCD约分，防止整数溢出。

---

## 8. 作者与参考文献

本项目由15个经典科学计算代码项目融合合成，原始项目均来自
John Burkardt等学者的开源科学计算库（MIT/GPL协议）。

核心参考文献:
1. Cuthill, E. & McKee, J. (1969). "Reducing the bandwidth of sparse symmetric matrices"
2. Turk, M. & Pentland, A. (1991). "Eigenfaces for recognition", J. Cognitive Neuroscience
3. Bulirsch, R. (1965). "Numerical calculation of elliptic integrals and elliptic functions"
4. Lindberg, B. (1974). "On a dangerous property of methods for stiff differential equations"
5. Kino, G.S. (1987). "Acoustic Waves: Devices, Imaging, and Analog Signal Processing"
6. Jensen, J.A. (1996). "Estimation of Blood Velocities Using Ultrasound"
7. Donoho, D.L. & Johnstone, I.M. (1994). "Ideal spatial adaptation by wavelet shrinkage"
