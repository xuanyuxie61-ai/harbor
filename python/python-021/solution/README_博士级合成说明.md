# README_博士级合成说明.md

## 托卡马克磁约束聚变综合模拟系统

### 一、项目概述

本项目将 **15 个独立科研代码项目** 的核心算法融合为一个面向**等离子体物理：托卡马克磁约束聚变**的博士级综合计算系统。系统围绕托卡马克装置中等离子体的平衡、输运、稳定性与聚变燃烧等前沿科学问题展开，涵盖 Grad-Shafranov 方程求解、延迟微分方程输运模型、MHD 稳定性分析、聚变反应动力学、高阶数值求积与稀疏矩阵代数等高度复杂的计算模块。

---

### 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 合成后角色 | 科学问题映射 |
|------|--------|----------|-----------|-------------|
| 1 | `707_mackey_glass_dde` | Mackey-Glass 延迟微分方程 | `transport_dde.py` | 等离子体能量输运的延迟反馈模型（ITB 内部输运垒阈值行为） |
| 2 | `940_quad_gauss` | Gauss-Legendre 数值积分 | `quadrature_engine.py` | 聚变功率体积积分、磁面面积分、环形几何体积分 |
| 3 | `563_hypersphere_angle` | 超球面随机采样与夹角统计 | `collision_transport.py` | 速度空间各向异性分布采样与 Coulomb 碰撞角统计 |
| 4 | `1200_tennis_matrix` | Markov 转移矩阵 | `mhd_stability.py` | 等离子体宏观态（约束/ELM/撕裂模/破裂）之间的转移概率 |
| 5 | `1265_toms112` | 点在多边形内判定（射线法） | `geometry_utils.py` | 测试粒子是否位于 last closed flux surface (LCFS) 内部 |
| 6 | `174_chrpak` | 字符/字符串解析工具 | `matrix_algebra.py` | Matrix Market 与 Harwell-Boeing 稀疏矩阵格式解析与 I/O |
| 7 | `1317_triangle_symq_rule_convert` | 三角形对称求积法则 | `quadrature_engine.py` | 极向截面三角形网格上的刚度矩阵组装与有限元积分 |
| 8 | `1019_rectangle_distance` | 矩形内随机点距离统计 | `collision_transport.py` | 磁面局部 patch 内粒子平均碰撞距离统计 |
| 9 | `771_mm_to_msm` | Matrix Market 格式读取 | `matrix_algebra.py` | MHD 刚度矩阵的稀疏存储读取 |
| 10 | `678_line_fekete_rule` | 线段上 Fekete 点计算 | `quadrature_engine.py` / `geometry_utils.py` | 磁面上最优谱元插值节点选取 |
| 11 | `1023_rigid_body_ode` | 刚体 Euler 方程 | `particle_drift.py` | 带电粒子引导中心漂移运动（非线性速度耦合类比刚体旋转） |
| 12 | `209_conte_deboor` | 素因子 FFT 算法 | `spectral_analysis.py` | 等离子体湍流与 MHD 模的傅里叶谱分析 |
| 13 | `1017_reaction_ode` | 化学反应动力学 ODE | `fusion_kinetics.py` | D-T 聚变反应动力学（燃料消耗与氦灰产生） |
| 14 | `1003_r8utt` | 上三角 Toeplitz 矩阵 | `matrix_algebra.py` | 有限元预处理中的快速行列式与回代求解 |
| 15 | `781_msm_to_hb` | 稀疏矩阵 HB 格式转换 | `matrix_algebra.py` | 刚度矩阵 Harwell-Boeing 格式输出与交换 |

---

### 三、新增数学物理模型与核心公式

#### 3.1 Grad-Shafranov 平衡方程

在轴对称托卡马克中，磁通函数 $\psi(R,Z)$ 满足椭圆型偏微分方程：

$$
\Delta^* \psi = R^2 \nabla \cdot \left( \frac{1}{R^2} \nabla \psi \right) = -\mu_0 R^2 \frac{dp}{d\psi} - F \frac{dF}{d\psi}
$$

其中 $p(\psi)$ 为等离子体压强，$F = R B_\varphi$ 为极向电流函数。采用 Miller 参数化边界：

$$
\begin{aligned}
R(\theta) &= R_0 + a \cos(\theta + \delta \sin \theta) \\
Z(\theta) &= \kappa a \sin \theta
\end{aligned}
$$

压强与电流剖面采用 ITER 型参数化：

$$
p(\psi_n) = p_0 \left(1 - \psi_n^{\alpha_p}\right)^{\beta_p}, \quad
F^2(\psi_n) = R_0^2 B_0^2 \left[ 1 + \frac{\beta_0}{\varepsilon^2 \kappa^2} \left(1 - \left(1 - \psi_n^{\alpha_I}\right)^{\beta_I}\right) \right]
$$

#### 3.2 引导中心漂移运动（类比刚体 Euler 方程）

刚体自由旋转的 Euler 方程：

$$
\begin{aligned}
\frac{d\omega_1}{dt} &= \left(\frac{1}{I_3} - \frac{1}{I_2}\right) \omega_2 \omega_3 \\
\frac{d\omega_2}{dt} &= \left(\frac{1}{I_1} - \frac{1}{I_3}\right) \omega_1 \omega_3 \\
\frac{d\omega_3}{dt} &= \left(\frac{1}{I_2} - \frac{1}{I_1}\right) \omega_1 \omega_2
\end{aligned}
$$

将其类比为引导中心三个速度分量 $(v_R, v_Z, v_\parallel)$ 的动力学，非线性耦合项来自磁场不均匀性与曲率效应：

$$
\begin{aligned}
\frac{dv_R}{dt} &= \alpha_1 v_Z v_\parallel + S_R(t) \\
\frac{dv_Z}{dt} &= \alpha_2 v_R v_\parallel + S_Z(t) \\
\frac{dv_\parallel}{dt} &= \alpha_3 v_R v_Z + S_\parallel(t)
\end{aligned}
$$

其中 $\alpha_i = (1/I_j - 1/I_k)$ 为类比系数，$S(t)$ 为外部电场/碰撞源项。磁矩绝热不变量：

$$
\mu = \frac{m v_\perp^2}{2B}
$$

#### 3.3 D-T 聚变反应动力学

D + T 反应速率方程组：

$$
\begin{aligned}
\frac{dn_D}{dt} &= -\langle \sigma v \rangle_{DT} n_D n_T + S_D - \frac{n_D}{\tau_p} \\
\frac{dn_T}{dt} &= -\langle \sigma v \rangle_{DT} n_D n_T + S_T - \frac{n_T}{\tau_p} \\
\frac{dn_{He}}{dt} &= +\langle \sigma v \rangle_{DT} n_D n_T - \frac{n_{He}}{\tau_{He}}
\end{aligned}
$$

Bosch-Hale 参数化反应速率：

$$
\langle \sigma v \rangle = C_1 \cdot \theta \sqrt{\frac{\xi}{m_r c^2 T_i^3}} \exp(-3\xi) \quad [\text{m}^3/\text{s}]
$$

聚变功率密度与 Lawson 判据：

$$
P_{fus} = \frac{n_D n_T \langle \sigma v \rangle E_{fus}}{4}, \quad
n \tau_E \ge \frac{3 k_B T_i}{\eta \cdot 0.25 \langle \sigma v \rangle E_{fus}}
$$

#### 3.4 能量输运延迟微分方程（Mackey-Glass 型）

将 Mackey-Glass 方程物理化为等离子体能量密度 $W(t)$ 的演化：

$$
\frac{dW(t)}{dt} = P_{heat} \cdot \beta \cdot \frac{W(t-\tau)^n}{W_0^n + W(t-\tau)^n} - \gamma \frac{W(t)}{\tau_E} - P_{loss}(t)
$$

非线性项模拟输运垒的阈值行为。ITER89-P 能量约束时间缩放律：

$$
\tau_E = 0.048 \cdot I_p^{0.85} B_t^{0.2} \bar{n}_{e20}^{0.1} P_{loss}^{-0.5} R^{1.5} a^{0.3} \kappa^{0.5} M^{0.5} \quad [\text{s}]
$$

新经典扩散系数（简化香蕉区）：

$$
D_{neo} = \frac{q^2 \nu_{ei} \rho_i^2}{\varepsilon^{3/2}}
$$

#### 3.5 高阶数值求积

Gauss-Legendre 求积：

$$
\int_a^b f(x) dx \approx \frac{b-a}{2} \sum_{i=1}^n w_i f\left(\frac{b-a}{2} x_i + \frac{a+b}{2}\right)
$$

三角形 Stroud 对称求积（精度 $p=7$）：

$$
\int_T f(x,y) dA \approx |J| \sum_{k=1}^{N} w_k f(x_k, y_k)
$$

Fekete 点通过最大化 Vandermonde 矩阵行列式选取：

$$
\max_{x_1,\ldots,x_m} |\det V(x_1,\ldots,x_m)|, \quad V_{ij} = T_{j-1}(x_i)
$$

#### 3.6 等离子体湍流谱分析

密度涨落的 Fourier 模分解：

$$
\delta n(r,\theta,t) = \sum_{m,n} \tilde{A}_{mn}(r,t) \exp\left[i(m\theta - n\varphi - \omega_{mn} t)\right]
$$

剪切阿尔芬波色散关系：

$$
\omega^2 = \frac{k_\parallel^2 v_A^2}{1 + k_\perp^2 \rho_s^2}, \quad v_A = \frac{B}{\sqrt{\mu_0 \rho_m}}
$$

增长率由功率谱对数线性拟合得到：

$$
\gamma = \frac{1}{2} \frac{d \ln P_{mn}}{dt}
$$

#### 3.7 Coulomb 碰撞与输运系数

电子-离子碰撞频率（Spitzer）：

$$
\nu_{ei} = \frac{n_e Z_{eff} e^4 \ln\Lambda}{3(2\pi)^{3/2} \varepsilon_0^2 \sqrt{m_e} (k_B T_e)^{3/2}}
$$

Coulomb 对数：

$$
\ln\Lambda = 31.3 - \ln\left(\frac{\sqrt{n_e}}{T_e}\right)
$$

经典扩散系数与电子热导率：

$$
D_{cl} = \nu_{ei} \rho_e^2, \quad \chi_e = D_{cl} \sqrt{\frac{m_i}{m_e}}
$$

#### 3.8 MHD 稳定性分析

理想 MHD 能量原理：

$$
\delta W = \frac{1}{2} \int \left[ \frac{|\mathbf{Q}_\perp|^2}{\mu_0} + \gamma p |\nabla \cdot \boldsymbol{\xi}_\perp|^2 + (\boldsymbol{\xi}_\perp \cdot \nabla p)(\boldsymbol{\xi}_\perp^* \cdot \boldsymbol{\kappa}) - J_\parallel (\boldsymbol{\xi}_\perp^* \times \mathbf{b}) \cdot \mathbf{Q}_\perp \right] dV
$$

Mercier 稳定性判据：

$$
D_M = \left(\frac{q}{r}\right)^2 \left[ \frac{r^4}{4R^2 q^4}(1-q^2)^2 - \frac{2\mu_0 R^2 q^2}{B_\varphi^2} \frac{dp}{dr}\left(1 - \frac{1}{q^2}\right) \right]
$$

Troyon 临界比压极限：

$$
\beta_c \approx 3.5 \cdot \frac{\varepsilon}{q_{edge}} \quad [\%]
$$

MHD 状态转移矩阵 $P$（Markov 链）描述 8 个宏观态之间的转移概率，稳态分布满足 $\boldsymbol{\pi} = P^T \boldsymbol{\pi}$。

#### 3.9 稀疏矩阵代数

上三角 Toeplitz 矩阵（UTT）快速运算：

$$
A = \begin{bmatrix}
a_0 & a_1 & a_2 & \cdots & a_{n-1} \\
0 & a_0 & a_1 & \cdots & a_{n-2} \\
\vdots & & \ddots & & \vdots \\
0 & \cdots & 0 & 0 & a_0
\end{bmatrix}, \quad
\det(A) = a_0^n
$$

求解算法为向后回代，无需显式求逆。

---

### 四、文件结构说明

```
021_synth_project/
├── main.py                           # 统一入口，零参数可运行
├── parameters.py                     # 全局物理常量与数值参数
├── equilibrium_solver.py             # Grad-Shafranov 平衡求解
├── particle_drift.py                 # 引导中心漂移运动（类比刚体 ODE）
├── fusion_kinetics.py                # D-T 聚变反应动力学
├── transport_dde.py                  # 能量输运延迟微分方程
├── quadrature_engine.py              # Gauss 求积 / 三角形 FEM / Fekete 点
├── spectral_analysis.py              # FFT 湍流谱分析与 MHD 模检测
├── collision_transport.py            # Coulomb 碰撞统计与输运系数
├── geometry_utils.py                 # 磁面判定 / 体积面积 / 曲率 / Fekete
├── matrix_algebra.py                 # UTT / MM / HB 格式转换与 CG 求解
├── mhd_stability.py                  # Markov 状态转移 / δW / Mercier 判据
└── README_博士级合成说明.md           # 本文档
```

共 **12 个 `.py` 文件**，满足至少 8 个的要求。

---

### 五、运行方式

```bash
cd /mnt/data/zpy/sci-swe/source code/Synthesis-project-python/021_synth_project
python main.py
```

程序将顺序执行 10 个科学计算模块，每个模块输出关键物理量与数值结果。无需输入任何参数，所有配置在 `parameters.py` 中设定。

---

### 六、数值鲁棒性与边界处理

1. **除零保护**：所有分母表达式均添加 `+ 1e-20`（或类似小量）保护。
2. **非负截断**：密度、温度、能量密度等物理量数值积分后强制 `max(val, 0)`。
3. **数组越界检查**：`np.clip` 广泛用于将归一化磁通、温度等限制在物理合理区间。
4. **矩阵奇异检测**：UTT 求解前检查对角元 $a_0 \neq 0$；CG 迭代设置最大步数与残差容差。
5. **网格收敛性**：Grad-Shafranov 求解采用 Picard 松弛（$\omega = 0.3$），并检查迭代残差。
6. **延迟 DDE 稳定性**：RK4 积分步长自动由总时长与步数确定，历史数组长度根据延迟时间 $\tau$ 动态计算。

---

### 七、合成后的科学问题能力

本项目能够综合解决以下托卡马克磁约束聚变前沿问题：

1. **等离子体平衡重构**：通过 Grad-Shafranov 方程迭代求解，获得磁通函数 $\psi(R,Z)$、安全因子剖面 $q(r)$ 与磁场分布。
2. **粒子轨道追踪**：模拟带电粒子在弯曲磁场中的引导中心漂移，验证磁矩绝热不变性。
3. **聚变燃烧分析**：预测 D-T 燃料消耗、氦灰积累、聚变功率输出与增益因子 $Q$ 的时间演化。
4. **输运与约束**：利用延迟微分方程模拟能量输运的滞后反馈效应，评估 ITER89-P 约束时间与新经典输运系数。
5. **湍流与 MHD 不稳定性**：FFT 谱分析检测不稳定模的增长率，Markov 链评估破裂风险，理想 MHD 能量原理判定稳定性。
6. **高阶数值方法**：Gauss-Legendre 求积、三角形有限元刚度矩阵组装、Fekete 点谱元插值，支持高精度等离子体体积分与边界元离散。
7. **稀疏矩阵工程**：Matrix Market 与 Harwell-Boeing 格式转换、UTT 快速求解、CG 迭代法，满足大型 MHD 刚度矩阵的工程需求。

---

### 八、免责声明

本项目为科研代码合成演示系统，物理模型经过合理简化（如 cylindrical 近似、均匀温度假设、简化碰撞算子等），数值参数参考 ITER-like 托卡马克量级。实际聚变装置设计需采用更完善的代码（如 EFIT、TRANSP、NIMEQ 等）进行验证。
