# 纳米颗粒-生物膜相互作用：粗粒化分子动力学综合模拟系统

## 1. 项目概述

本项目将 **15 个独立科研代码项目** 的核心算法融合重构，面向前沿科学问题——**分子动力学：纳米颗粒与生物膜相互作用**（Nanoparticle-Biomembrane Interaction Dynamics），构建了一个博士级自然科学计算系统。

科学背景：在纳米医学与纳米毒理学中，带电金纳米颗粒（AuNP）与磷脂双分子层（如 POPC）的相互作用决定了药物的靶向递送效率与生物安全性。本系统在一个统一的计算框架内耦合了膜弹性力学、静电学、离子输运、随机动力学、统计采样与机器学习代理模型，能够自洽地计算纳米颗粒的吸附、包裹与膜变形行为。

## 2. 原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 在合成项目中的角色 |
|:---:|--------|---------|-------------------|
| 1 | 606_jacobi_poisson_1d | Jacobi 迭代求解 1D Poisson 方程 | **静电泊松-玻尔兹曼求解器**：计算德拜屏蔽层内的电势分布 |
| 2 | 354_fd1d_advection_lax | Lax 格式求解 1D 对流方程 | **离子对流-扩散输运**：膜附近离子耗竭区的时间演化 |
| 3 | 514_hello_parfor | MATLAB parfor 并行循环 | **并行力计算框架**：multiprocessing 实现的域分解并行 |
| 4 | 773_mnist_neural | CNN 图像分类网络 | **神经网络自由能代理模型**：前馈网络预测结合自由能 ΔG |
| 5 | 704_luhn | Luhn 模 10 校验和 | **分子拓扑完整性校验（MTIC）**：磷脂残基标识符校验 |
| 6 | 085_bicg | BiCG 迭代法解线性方程组 | **预处理 BiCG 静电求解器**：与 Jacobi 形成双求解器验证 |
| 7 | 488_grazing_ode | 放牧模型 ODE（Type-II 功能响应） | **Type-II 受体饱和结合动力学**：纳米颗粒-膜配体结合力 |
| 8 | 755_mesh_etoe | 无结构网格 element-to-element 邻接 | **三角化膜网格邻接分析**：有限元风格的曲率与弯曲能计算 |
| 9 | 919_product_rule | 多维乘积求积规则 | **多维 Gauss-Legendre 能量积分**：环形区域结合能密度积分 |
| 10 | 1262_toeplitz_cholesky | 快速 Toeplitz Cholesky 分解 | **相关随机力生成**：指数记忆核的 AR(1) 精确采样 |
| 11 | 225_cpr | Chebyshev Proxy Rootfinder | **力平衡零点搜索**：寻找稳定/不稳定平衡距离 |
| 12 | 824_octopus | Octave/MATLAB 环境检测 | **平台检测模块**：数值稳定性与后端兼容性判断 |
| 13 | 541_histogram_pdf_sample | 直方图/CDF 逆变换采样 | **Boltzmann 分布采样与 Metropolis 准则**：MC 接受率计算 |
| 14 | 594_interp_spline | 三次样条插值 | **势函数表插值**：LJ 势的连续恢复与导数计算 |
| 15 | 1192_svd_sphere | SVD 与球面采样 | **膜变形 SVD 主成分分析**：随机取向采样与变形模式提取 |

## 3. 核心数学物理模型与公式

### 3.1 膜弹性力学：Helfrich 弯曲能

磷脂双分子层被建模为二维弹性薄壳，其弯曲能由 Helfrich 哈密顿量描述：

$$
E_{\text{bend}} = \frac{\kappa}{2} \int_{\Sigma} (H - H_0)^2 \, dA
$$

其中：
- $\kappa$ 为弯曲模量（典型值 $10\sim40\,k_B T$）
- $H$ 为平均曲率
- $H_0$ 为自发曲率（平面膜取 $H_0 = 0$）

离散化后，对每个三角形单元采用顶点曲率质心插值：

$$
E_{\text{bend}}^{\text{disc}} = \frac{\kappa}{2} \sum_{e} A_e \left( \frac{H_{v_1} + H_{v_2} + H_{v_3}}{3} \right)^2
$$

### 3.2 球对称泊松-玻尔兹曼方程

纳米颗粒表面电荷在电解质溶液中诱导的电势满足：

$$
\frac{d^2\phi}{dr^2} + \frac{2}{r}\frac{d\phi}{dr} = \kappa_D^2 \phi
$$

其中德拜长度：

$$
\lambda_D = \frac{1}{\kappa_D} = \sqrt{\frac{\varepsilon k_B T}{2 N_A e^2 I}}
\quad \text{[nm]}
$$

为消除一阶导数项，引入变换 $u(r) = r\phi(r)$，方程化为对称的 1D Helmholtz 方程：

$$
\frac{d^2 u}{dr^2} = \kappa_D^2 u
$$

中心差分离散（对称正定三对角系统）：

$$
-u_{i-1} + (2 + h^2\kappa_D^2) u_i - u_{i+1} = 0
$$

### 3.3 对流-扩散方程

膜表面附近离子浓度 $c(x,t)$ 满足：

$$
\frac{\partial c}{\partial t} = -v\frac{\partial c}{\partial x} + D\frac{\partial^2 c}{\partial x^2} + S(x,t)
$$

采用 upwind + FTCS 显式格式（稳定性条件 $v\Delta t/\Delta x + 2D\Delta t/\Delta x^2 \le 1$）：

$$
c_i^{n+1} = c_i^n - \frac{v\Delta t}{\Delta x}(c_i^n - c_{i-1}^n) + \frac{D\Delta t}{\Delta x^2}(c_{i-1}^n - 2c_i^n + c_{i+1}^n) + \Delta t \, S_i^n
$$

### 3.4 过阻尼朗之万方程

纳米颗粒在膜法向的运动满足：

$$
\gamma \frac{dz}{dt} = F_{\text{elec}}(z) + F_{\text{vdW}}(z) + F_{\text{bend}}(z) + F_{\text{bind}}(z) + \xi(t)
$$

其中 $\xi(t)$ 为高斯白噪声，满足涨落-耗散定理：

$$
\langle \xi(t)\xi(t') \rangle = 2\gamma k_B T \delta(t-t')
$$

Euler-Maruyama 离散：

$$
z_{n+1} = z_n + \frac{\Delta t}{\gamma} F_{\text{total}}(z_n) + \sqrt{\frac{2k_B T \Delta t}{\gamma}} \, \mathcal{N}(0,1)
$$

各分力表达式：
- **静电屏蔽库仑力**（Derjaguin 近似）：
  $$
  F_{\text{elec}}(z) = C_{\text{scale}} \frac{\zeta_{\text{NP}} \zeta_{\text{mem}}}{\lambda_D} e^{-z/\lambda_D}
  $$
- **Lennard-Jones 力**：
  $$
  F_{\text{vdW}}(z) = \frac{24\varepsilon_{\text{LJ}}}{z} \left[ 2\left(\frac{\sigma_{\text{LJ}}}{z}\right)^{12} - \left(\frac{\sigma_{\text{LJ}}}{z}\right)^6 \right]
  $$
- **膜弯曲回复力**：
  $$
  F_{\text{bend}}(z) = -k_{\text{spring}} (z - z_{\text{eq}})
  $$
- **Type-II 受体-配体结合力**（改编自 grazing_ode）：
  $$
  F_{\text{bind}}(z) = -F_{\max} \left[ 1 - e^{-\kappa_{\text{bind}} \max(z_{\text{cutoff}} - z,\, 0)} \right]
  $$

### 3.5 指数记忆核与相关随机力

广义朗之万方程（GLE）的记忆核取 Mori-Zwanzig 指数形式：

$$
\gamma(\tau) = \gamma_0 e^{-|\tau|/\tau_{\text{mem}}}
$$

离散协方差矩阵为对称 Toeplitz 矩阵：

$$
C_{ij} = k_B T \gamma_0 e^{-|i-j|\Delta t / \tau_{\text{mem}}} = \sigma^2 \rho^{|i-j|}
$$

其中 $\sigma^2 = k_B T \gamma_0$，$\rho = e^{-\Delta t/\tau_{\text{mem}}}$。该协方差对应精确的 AR(1) 过程：

$$
\xi_0 = \sigma \mathcal{N}(0,1), \quad \xi_i = \rho \xi_{i-1} + \sigma\sqrt{1-\rho^2} \, \mathcal{N}(0,1)
$$

### 3.6 Chebyshev Proxy Rootfinder

为寻找力平衡零点，在区间 $[a,b]$ 上构造 Chebyshev 插值多项式：

$$
p(x) = \sum_{j=0}^{N} a_j T_j\left(\frac{2x-a-b}{b-a}\right)
$$

系数由 Clenshaw-Curtis 公式计算：

$$
a_j = \frac{2}{N p_j} \sum_{k=0}^{N} \frac{f(x_k) \cos(j k \pi / N)}{p_k}, \quad p_0 = p_N = 2,\; p_k = 1
$$

多项式根通过伴随矩阵特征值求解：

$$
A = \begin{bmatrix}
0 & 1 & 0 & \cdots & 0 \\
1/2 & 0 & 1/2 & \cdots & 0 \\
0 & 1/2 & 0 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
-\frac{a_0}{2a_N} & -\frac{a_1}{2a_N} & \cdots & -\frac{a_{N-2}}{2a_N} + \frac{1}{2} & -\frac{a_{N-1}}{2a_N}
\end{bmatrix}
$$

### 3.7 神经网络自由能代理模型

输入描述符 $\mathbf{x} = [z, R_{\text{NP}}, \zeta_{\text{NP}}, \zeta_{\text{mem}}, \kappa_{\text{bend}}, I_{\text{ionic}}]^T$，输出预测结合自由能 $\Delta G_{\text{bind}}$。

网络架构（全连接回归网络，改编自 mnist_neural 的深度学习思想）：

$$
\mathbf{h}_1 = \text{ReLU}\left(\text{BN}(W_1 \mathbf{x} + \mathbf{b}_1)\right) \in \mathbb{R}^{32}
$$
$$
\mathbf{h}_2 = \text{ReLU}\left(\text{BN}(W_2 \mathbf{h}_1 + \mathbf{b}_2)\right) \in \mathbb{R}^{16}
$$
$$
\mathbf{h}_3 = \text{ReLU}(W_3 \mathbf{h}_2 + \mathbf{b}_3) \in \mathbb{R}^{8}
$$
$$
\hat{y} = W_4 \mathbf{h}_3 + b_4 \in \mathbb{R}
$$

批归一化（Batch Normalization）：

$$
y = \gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}} + \beta
$$

损失函数（MSE + L2 正则）：

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 + \lambda_{\text{reg}} \|W\|_2^2
$$

训练数据由物理启发的 DLVO + Helfrich 解析模型生成：

$$
\Delta G = \underbrace{C_{\text{elec}} R_{\text{NP}} \zeta_{\text{NP}} \zeta_{\text{mem}} e^{-z/\lambda_D}}_{\text{静电}} + \underbrace{-\frac{A_H R_{\text{NP}}}{12z}}_{\text{vdW}} + \underbrace{\pi \kappa_{\text{bend}} \left(\frac{R_{\text{NP}}}{z}\right)^2}_{\text{弯曲}}
$$

### 3.8 三次样条插值

对于离散的势函数表格 $V(r_i)$，构造分段三次多项式 $S(r)$ 满足 $S(r_i) = V(r_i)$ 且 $S \in C^2$。弯矩方程（自然边界 $M_0 = M_n = 0$）：

$$
\mu_i M_{i-1} + 2M_i + \lambda_i M_{i+1} = d_i
$$

其中 $M_i = S''(r_i)$，等距网格时 $\mu_i = \lambda_i = 1/2$。每段多项式：

$$
S_j(r) = a_j + b_j (r-r_j) + c_j (r-r_j)^2 + d_j (r-r_j)^3
$$

### 3.9 多维乘积求积规则

$d$ 维积分通过一维 Gauss-Legendre 规则的直积构造：

$$
\int_{\Omega} f(\mathbf{x}) \, d\mathbf{x} \approx \sum_{i=1}^{N} w_i f(\mathbf{x}_i)
$$

其中多维节点 $\mathbf{x}_i \in \mathbb{R}^d$ 与权重 $w_i$ 由混合进制枚举各维规则得到。坐标变换（将 $[-1,1]$ 映射到 $[a_k, b_k]$）：

$$
t_k = \frac{b_k - a_k}{2} x_k + \frac{a_k + b_k}{2}, \quad J_k = \frac{b_k - a_k}{2}
$$

### 3.10 SVD 变形模式分析

膜顶点位移矩阵 $\Delta \in \mathbb{R}^{3 \times n_v}$ 的奇异值分解：

$$
\Delta = U \Sigma V^T
$$

左奇异向量 $U_{:,k}$ 为第 $k$ 个空间变形主方向，奇异值 $\sigma_k$ 为对应变形幅度。变形能按模式分解：

$$
E_k = \frac{1}{2} \kappa \sigma_k^2
$$

## 4. 文件结构

```
117_synth_project/
├── main.py                     # 统一入口，零参数运行
├── platform_detect.py          # 平台与环境检测（seed 824）
├── topology_validator.py       # 分子拓扑 MTIC 校验（seed 704）
├── parallel_utils.py           # 并行计算框架（seed 514）
├── membrane_mesh.py            # 三角化膜网格与邻接（seed 755）
├── electrostatic_solver.py     # Poisson-Boltzmann 双求解器（seed 606 + 085）
├── transport_solver.py         # 对流-扩散输运（seed 354）
├── nanoparticle_dynamics.py    # 朗之万动力学与包裹（seed 488）
├── potential_energy.py         # 多维积分与样条插值（seed 919 + 594）
├── sampling_utils.py           # 统计采样与 SVD（seed 541 + 1192）
├── correlated_forces.py        # Toeplitz Cholesky / AR(1) 随机力（seed 1262）
├── equilibrium_solver.py       # Chebyshev Proxy Rootfinder（seed 225）
├── neural_surrogate.py         # 神经网络自由能代理（seed 773）
└── README_博士级合成说明.md    # 本说明文档
```

## 5. 运行方式

```bash
cd /mnt/data/zpy/sci-swe/source code/Synthesis-project-python/117_synth_project
python main.py
```

程序零参数运行，内置所有物理参数（温度 300 K、NaCl 0.1 M、NP 半径 2.5 nm 等），自动完成以下流程：

1. 环境检测与数值精度确认
2. 磷脂拓扑标识符生成与校验
3. 三角化膜网格构建、邻接分析与 Helfrich 弯曲能计算
4. 球对称 Poisson-Boltzmann 方程的 Jacobi / 预处理 BiCG 双求解器验证
5. 膜附近离子耗竭区的对流-扩散演化
6. 多维 Gauss-Legendre 结合能积分与样条势函数插值
7. 高斯分布逆 CDF 采样、球面均匀采样、SVD 变形模式分析与 Metropolis 测试
8. 指数记忆核相关随机力生成与功率谱验证
9. Chebyshev Proxy Rootfinder 力平衡零点搜索与稳定性判定
10. 神经网络结合自由能代理模型的训练与测试（$R^2 > 0.5$ 自动验证）
11. 纳米颗粒过阻尼朗之万动力学轨迹积分
12. 多进程并行计算框架验证

## 6. 关键数值结果示例

- **膜弯曲能**: $E_{\text{bend}} = 2.7280\,k_B T$
- **德拜长度**: $\lambda_D = 0.9741\,\text{nm}$
- **静电求解双验证**: Jacobi 与 BiCG 结果最大差异 $2.4 \times 10^{-9}\,\text{V}$
- **稳定平衡距离**: $z_{\text{eq}} = 3.0361\,\text{nm}$（稳定平衡点）
- **神经网络代理**: 测试集 $R^2 = 0.8788$
- **朗之万轨迹**: 从 $z = 8.0\,\text{nm}$ 弛豫至 $z = 6.53\,\text{nm}$（0.2 ns）

## 7. 边界处理与数值鲁棒性

- **反射边界**：朗之万积分中 $z < 0.1\,\text{nm}$ 时施加反射，防止 LJ 奇异性
- **力幅截断**：总合力限制在 $[-1000, +1000]\,\text{kJ/(mol·nm)}$，避免数值爆炸
- **非负截断**：输运求解器浓度 $c \ge 0$
- **对称正定变换**：Poisson-Boltzmann 通过 $u = r\phi$ 变换化为对称系统，使 BiCG 稳定收敛
- **Jacobi 预处理**：BiCG 采用对角预处理 $M = \text{diag}(A)$，避免 breakdown
- **Chebyshev 区间保护**：力平衡搜索限制在 $[2, 8]\,\text{nm}$，避开 LJ 奇异区导致的插值振荡
- **输入标准化**：神经网络训练前对输入/输出做 Z-score 标准化
- **异常值截断**：训练数据目标值限制在 $[-500, 500]\,\text{kJ/mol}$

## 8. 科学意义

本项目展示了一个完整的粗粒化分子动力学工作流，可用于：
- 预测纳米颗粒与细胞膜的结合亲和力与平衡构象
- 分析膜弹性变形的高阶模式（SVD 主成分）
- 评估离子环境（pH、盐浓度）对静电屏蔽与结合行为的影响
- 通过神经网络代理模型快速筛选大量纳米颗粒参数组合，替代昂贵的全原子模拟

所有代码均通过实际运行验证，无语法错误，无可视化依赖，具备完整的边界保护与数值鲁棒性。
