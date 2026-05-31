# GRB 余辉辐射机制合成项目 — 博士级合成说明

## 一、项目概述

本项目将 **15 个独立科研代码项目** 融合为一个面向 **天体物理：伽马射线暴（GRB）辐射机制** 的博士级科研计算项目。项目以 Python 语言实现，围绕 GRB 喷流动力学、磁重联、粒子加速、辐射转移与能谱计算等前沿问题展开。

### 核心科学问题

> **构建一个自洽的 GRB 余辉多波段辐射机制数值模拟框架**，涵盖：
> 1. 相对论喷流的流体动力学（零散度螺旋速度场）；
> 2. 冲击波非线性扩散（Barenblatt 自相似解）；
> 3. 随机粒子加速（Fokker-Planck 方程 + 随机 Runge-Kutta）；
> 4. 辐射转移有限元求解（Neumann 边界条件）；
> 5. 同步辐射 / 逆康普顿能谱的三角形蒙特卡洛积分；
> 6. Hankel 谱矩分析与谱线形状识别；
> 7. 磁重联元胞自动机与螺旋磁场结构；
> 8. 离散逆康普顿级联（子集和问题）。

---

## 二、原项目到科学问题的映射

| 序号 | 原种子项目 | 合成后角色 | 科学功能 |
|:---:|---|---|---|
| 1 | `211_continuity_exact` | `grb_jet_hydro.py` | 相对论连续性方程的零散度螺旋速度场，用于喷流横向-轴向速度剖面计算 |
| 2 | `901_porous_medium_exact` | `blast_wave_diffusion.py` | 多孔介质方程（PME）的 Barenblatt 自相似解，映射为 GRB 冲击波能量密度演化 |
| 3 | `1171_stochastic_rk` | `particle_acceleration.py` | Kasdin 四阶随机 Runge-Kutta，求解 Fokker-Planck SDE，模拟电子在冲击波中的扩散加速 |
| 4 | `377_fem_neumann` | `radiation_diffusion_fem.py` | 一维反应-扩散方程的有限元离散（质量矩阵 + 刚度矩阵 + Neumann 边界），求解辐射能密度演化 |
| 5 | `1401_wathen_matrix` | `fem_matrix_assembly.py` | Wathen 有限元稀疏矩阵组装 + 共轭梯度（CG）求解器，用于二维辐射转移线性系统 |
| 6 | `1308_triangle_integrands` | `sed_triangle_integrator.py` | 三角形区域上的蒙特卡洛积分，计算 `(γ, θ)` 参数空间内的同步辐射能谱 |
| 7 | `590_interp` | `spectrum_interpolator.py` | Lagrange / 线性 / 最近邻插值，重建离散能谱 `νF_ν` 的连续形式 |
| 8 | `927_pwl_interp_2d` | `opacity_interpolator.py` | 二维分段线性插值（三角形剖分），用于 `(ρ, T)` 空间内的 Rosseland 平均不透明度查表 |
| 9 | `1405_web_matrix` | `photon_transfer_matrix.py` | 网络图转移矩阵 + 幂迭代，求解光子在能量 bin 间的稳态分布（Markov 链蒙特卡洛） |
| 10 | `506_hankel_spd` | `spectral_moments.py` | Hankel 正定矩矩阵的 Cholesky 分解，用于谱矩问题与 Christoffel-Darboux 核 |
| 11 | `1371_ulam_spiral` | `magnetic_spiral.py` | Ulam 螺旋数组，离散化喷流中螺旋磁场的缠绕数 `B_φ / B_z = tan ψ(r)` |
| 12 | `671_life` | `reconnection_automaton.py` | Conway 生命游戏元胞自动机，模拟磁重联活性位点的时空演化 |
| 13 | `1178_subset_sum` | `discrete_cascade.py` | 动态规划子集和表，追踪离散逆康普顿级联中可实现的光子能量组合 |
| 14 | `719_matlab_compiler` | `anisotropic_tensor.py` | 幻方（Magic Square）结构，构建各向异性扩散张量 `D_ij = D_⊥ δ_ij + (D_∥ − D_⊥) b_i b_j` |
| 15 | `782_msm_to_mm` | `matrix_io.py` | Matrix Market 格式转换，导出辐射转移稀疏矩阵以供外部求解器使用 |

---

## 三、新增数学物理模型与核心公式

### 3.1 相对论喷流连续性方程

在柱坐标 `(r, φ, z)` 下，相对论质量守恒方程为：

$$
\frac{\partial (\Gamma \rho)}{\partial t} + \nabla \cdot (\Gamma \rho \mathbf{v}) = 0
$$

其中 Lorentz 因子：

$$
\Gamma = \frac{1}{\sqrt{1 - v^2/c^2}}
$$

引入流函数 `Ψ(r,z)`，使得：

$$
\Gamma \rho v_r = -\frac{1}{r}\frac{\partial \Psi}{\partial z}, \quad
\Gamma \rho v_z = \frac{1}{r}\frac{\partial \Psi}{\partial r}
$$

速度场由标量流函数 `Φ(z) = (1 - \cos(C\pi z))(1-z)^2` 构造：

$$
U(X,Y) = 10 \frac{\partial \Phi}{\partial Y}\Phi(X), \quad
V(X,Y) = -10 \frac{\partial \Phi}{\partial X}\Phi(Y)
$$

自动满足 `∂U/∂X + ∂V/∂Y = 0`。

### 3.2 冲击波非线性扩散（Barenblatt 解）

将多孔介质方程（PME）映射为冲击波能量密度演化：

$$
\frac{\partial u}{\partial t} = \Delta(u^m)
$$

Barenblatt 自相似解：

$$
\alpha = \frac{1}{m-1}, \quad \beta = \frac{1}{m+1}, \quad
\gamma_{\text{PME}} = \frac{m-1}{2m(m+1)}
$$

$$
u(r,t) = (t+\delta)^{-\beta}\left[C - \gamma_{\text{PME}}\left(\frac{r}{(t+\delta)^{\beta}}\right)^2\right]_{+}^{\alpha}
$$

冲击波总能量：

$$
E(t) = \int_0^{r_f(t)} 4\pi r^2 u(r,t)\,dr \propto (t+\delta)^{-\beta(3\alpha-1)}
$$

### 3.3 随机粒子加速（Fokker-Planck）

电子能量演化的 SDE：

$$
\frac{d\gamma}{dt} = A(\gamma) + \sqrt{2D(\gamma)}\,\eta(t)
$$

Bohm 扩散系数：

$$
D(\gamma) = \frac{1}{3} r_L c = \frac{1}{3}\frac{\gamma m_e c^2}{eB}c
$$

系统加速率（DSA）：

$$
A(\gamma) = \frac{4}{3}\frac{u_1 - u_2}{c}\gamma
$$

采用 Kasdin (1995) 四阶随机 Runge-Kutta 格式离散，系数为：

$$
\begin{aligned}
a_{21} &= 2.71644396264860, & a_{31} &= -6.95653259006152, \\
a_{32} &= 0.78313689457981, & a_{51} &= 0.47012396888046, \\
q_1 &= 2.12709852335625, & q_2 &= 2.73245878238737
\end{aligned}
$$

### 3.4 辐射扩散有限元（Neumann 边界）

一维辐射扩散方程：

$$
\frac{\partial E_{\text{rad}}}{\partial t} = \frac{\partial}{\partial x}\left[D(x)\frac{\partial E_{\text{rad}}}{\partial x}\right] + \Gamma(E_{\text{rad}})
$$

齐次 Neumann 边界条件：

$$
\left.\frac{\partial E_{\text{rad}}}{\partial x}\right|_{x=0} = 0, \quad
\left.\frac{\partial E_{\text{rad}}}{\partial x}\right|_{x=1} = 0
$$

Galerkin 弱形式导出半离散系统：

$$
M \dot{\mathbf{u}} = -K\mathbf{u} + \mathbf{f}(\mathbf{u})
$$

质量矩阵（一致质量）：

$$
M = \frac{h}{6}\begin{pmatrix}
2 & 1 & & \\
1 & 4 & 1 & \\
& \ddots & \ddots & \ddots \\
& & 1 & 2
\end{pmatrix}
$$

刚度矩阵：

$$
K = \frac{1}{h}\begin{pmatrix}
1 & -1 & & \\
-1 & 2 & -1 & \\
& \ddots & \ddots & \ddots \\
& & -1 & 1
\end{pmatrix}
$$

### 3.5 同步辐射三角形蒙特卡洛积分

同步辐射单电子发射率：

$$
j_\nu(\gamma,\theta) = \frac{\sqrt{3}\,e^3 B\sin\theta}{4\pi m_e c^2} F\left(\frac{\nu}{\nu_c}\right)
$$

临界频率：

$$
\nu_c(\gamma) = \frac{3eB\sin\theta}{4\pi m_e c}\gamma^2
$$

同步辐射函数近似（Crusius & Schlickeiser 1986）：

$$
F(x) \approx \begin{cases}
1.808\,x^{1/3}e^{-x}, & x < 10^{-3} \\
\sqrt{\frac{\pi x}{2}}e^{-x}, & x > 10 \\
\displaystyle\frac{1.808 x^{1/3}e^{-x}(1+0.16x^{2/3})}{1+0.53x^{2/3}}, & \text{中间区}
\end{cases}
$$

三角形蒙特卡洛积分：

$$
F_\nu \approx A_T \cdot \frac{1}{N}\sum_{i=1}^{N} j_\nu(\gamma_i,\theta_i) N(\gamma_i)
$$

其中 `A_T` 为三角形面积，采样采用重心坐标：

$$
\lambda_1 = 1-\sqrt{r_1}, \quad \lambda_2 = \sqrt{r_1}(1-r_2), \quad \lambda_3 = \sqrt{r_1}r_2
$$

### 3.6 光子转移 Markov 链

光子能量 bin 间的转移矩阵 `T` 满足列随机性：

$$
\sum_j T_{ji} = 1
$$

稳态分布 `n*` 满足：

$$
T n^* = n^*
$$

由 Perron-Frobenius 定理，若 `T` 不可约且非周期，则幂迭代：

$$
n_{k+1} = T n_k
$$

收敛到唯一稳态。Compton-y 参数：

$$
y = \sum_i n^*_i \frac{4kT_e}{m_e c^2} \tau_{\text{es},i}
$$

### 3.7 Hankel 谱矩与 Cholesky 分解

谱矩定义：

$$
\mu_k = \int_0^\infty \nu^k F_\nu\,d\nu
$$

Hankel 矩矩阵：

$$
H_{ij} = \mu_{i+j}, \quad i,j = 0,\dots,N-1
$$

由 Hamburger 矩问题，`H` 为正定 Hankel 矩阵当且仅当谱测度 `dΦ(ν)=F_ν dν` 为正测度。Cholesky 分解 `H = LL^T` 的递推公式（Al-Homidan & Alshahrani 2009）：

$$
L_{ij} = \frac{\alpha - \beta}{L_{jj}}, \quad
\alpha = \sum_{s=1}^{q} L_{qs}L_{rs}, \quad
\beta = \sum_{t=1}^{j-1} L_{it}L_{jt}
$$

其中 `q,r` 由 `i+j` 的奇偶性决定。

### 3.8 各向异性扩散张量

磁场对齐坐标系中的扩散张量：

$$
\tilde{D} = \text{diag}(D_\perp, D_\perp, D_\parallel)
$$

旋转到实验室坐标系（`b = B/|B|`）：

$$
D_{ij} = D_\perp \delta_{ij} + (D_\parallel - D_\perp) b_i b_j
$$

对于螺旋磁场 `b = (0, \sin\psi, \cos\psi)`，幻方（Magic Square）结构用于调制 `D_\parallel/D_\perp` 的径向变化。

### 3.9 离散逆康普顿级联（子集和）

多级散射后光子能量：

$$
\varepsilon_n \approx \varepsilon_0 \prod_{k=1}^n \gamma_k^2
$$

取对数后转化为子集和问题：

$$
\Delta E = \sum_i w_i, \quad w_i \propto \ln(\gamma_i^2)
$$

动态规划子集和表 `table[j] = w` 表示能量偏移 `j` 可由权重 `w` 结尾的子集实现。

紧凑度参数：

$$
\ell = \frac{L \sigma_T}{R m_e c^3}
$$

### 3.10 Wathen 有限元矩阵

对于 `NX × NY` 网格的 8 节点 serendipity 单元，矩阵阶数：

$$
N = 3NX \cdot NY + 2NX + 2NY + 1
$$

局部一致质量矩阵（密度 `ρ` 缩放）：

$$
\mathbf{M}_e = \frac{\rho}{180}\begin{pmatrix}
6 & -6 & 2 & -8 & 3 & -8 & 2 & -6 \\
-6 & 32 & -6 & 20 & -8 & 16 & -8 & 20 \\
\vdots & & & & & & & \vdots
\end{pmatrix}
$$

共轭梯度法求解 `A x = b`：

$$
\begin{aligned}
&\mathbf{r}_0 = \mathbf{b} - A\mathbf{x}_0, \quad \mathbf{p}_0 = \mathbf{r}_0 \\
&\alpha_k = \frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{p}_k^T A \mathbf{p}_k}, \quad
\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k \\
&\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k A\mathbf{p}_k, \quad
\beta_k = \frac{\mathbf{r}_{k+1}^T \mathbf{r}_{k+1}}{\mathbf{r}_k^T \mathbf{r}_k}, \quad
\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k
\end{aligned}
$$

---

## 四、项目文件结构

```
008_synth_project/
├── main.py                          # 统一入口，零参数运行
├── grb_jet_hydro.py                 # 喷流流体动力学 (seed 211)
├── blast_wave_diffusion.py          # 冲击波非线性扩散 (seed 901)
├── particle_acceleration.py         # 随机粒子加速 (seed 1171)
├── radiation_diffusion_fem.py       # 辐射扩散 FEM (seed 377)
├── fem_matrix_assembly.py           # Wathen 矩阵 + CG (seed 1401)
├── sed_triangle_integrator.py       # SED 三角形蒙特卡洛 (seed 1308)
├── spectrum_interpolator.py         # 能谱插值 (seed 590)
├── opacity_interpolator.py          # 不透明度 2D 插值 (seed 927)
├── photon_transfer_matrix.py        # 光子转移 Markov 链 (seed 1405)
├── spectral_moments.py              # Hankel 谱矩 (seed 506)
├── magnetic_spiral.py               # 螺旋磁场 (seed 1371)
├── reconnection_automaton.py        # 磁重联元胞自动机 (seed 671)
├── discrete_cascade.py              # 离散级联子集和 (seed 1178)
├── anisotropic_tensor.py            # 各向异性扩散张量 (seed 719)
├── matrix_io.py                     # Matrix Market I/O (seed 782)
└── README_博士级合成说明.md          # 本文档
```

---

## 五、运行方式

在项目根目录下执行：

```bash
python main.py
```

无需任何命令行参数。程序将依次执行：
1. 喷流速度场与连续性残差计算
2. Barenblatt 冲击波能量密度剖面
3. 螺旋磁场几何与磁化参数
4. 磁重联元胞自动机演化
5. 随机 Runge-Kutta 电子加速
6. 不透明度表插值
7. 辐射扩散 FEM 求解
8. Wathen 矩阵组装与 CG 求解
9. 光子转移稳态分布
10. Hankel 谱矩与 Cholesky 分解
11. 同步辐射 SED 三角形蒙特卡洛积分
12. 多方法能谱插值
13. 离散逆康普顿级联
14. 各向异性扩散张量
15. Matrix Market 矩阵导出

---

## 六、科学意义

本项目将原本散落于数值分析、组合优化、图论、矩阵计算等领域的 15 个独立算法，通过严格的数学物理映射，整合为一套完整的 **GRB 余辉辐射机制计算框架**。每个原项目的核心算法均承担了不可替代的真实科学角色：

- **连续性方程** 保证喷流质量守恒；
- **PME 自相似解** 描述冲击波能量传播；
- **随机 RK** 捕捉电子加速的随机扩散本质；
- **FEM + CG** 求解辐射转移的大规模稀疏线性系统；
- **三角形蒙特卡洛** 处理高维相空间 `(γ, θ)` 的能谱积分；
- **子集和 DP** 揭示离散级联的量子化能量特征；
- **Hankel 矩矩阵** 提供谱分析的正定代数结构；
- **元胞自动机** 模拟磁重联的非线性时空演化。

该框架具备数值鲁棒性（边界截断、除零保护、NaN 检测）与工程可扩展性，可直接作为 GRB 多波段余辉拟合的后端计算引擎。
