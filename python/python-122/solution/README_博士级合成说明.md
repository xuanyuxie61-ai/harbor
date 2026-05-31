# 脑血流动力学多尺度耦合模拟与血管网络质量评估系统

## 项目概述

本项目基于 **15 个种子科研项目** 的核心算法，在 **生物医学：脑血流动力学与血管网络** 领域融合构建了一个面向前沿科学问题的博士级计算系统。项目模拟从宏观脑动脉血压波传播、中观血管网络几何生成与质量评估、微观组织氧扩散与代谢，到细胞级血管重构动力学的完整多尺度脑血流动力学过程。

---

## 一、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 在合成项目中的角色 |
|------|--------|----------|-------------------|
| 1 | `646_laplace_radial_exact` | 径向 Laplace 方程精确解 | **脑血管稳态压力场计算**：微血管层面压力分布满足 ∇²P = 0，径向解 P(r) = a·log(r) + b 描述圆形血管截面内的压力梯度 |
| 2 | `1086_sir_ode` | SIR 传染病 ODE | **血氧三室运输模型**：将含氧血(S)、正在释氧血(I)、已耗氧血(R)建模为 SIR 型室模型 |
| 3 | `908_predator_prey_ode` | Lotka-Volterra ODE | **内皮细胞增殖-凋亡动力学**：增殖细胞(E)与凋亡信号(A)的捕食者-猎物竞争关系 |
| 4 | `958_quality` | 网格质量度量 | **血管网络拓扑质量评估**：Q-度量、Alpha-度量、球填充度量、半带宽等评估血管网格质量 |
| 5 | `434_fisher_pde_ftcs` | KPP-Fisher 反应扩散 PDE | **脑组织氧扩散与代谢**：FTCS 格式求解氧在组织中的扩散-反应方程 |
| 6 | `609_jai_alai_simulation` | 竞争排队模拟 | **血细胞分叉竞争模型**：红细胞在血管分叉点按竞争强度排队选择流向 |
| 7 | `308_distmesh` | 2D 距离函数网格生成 | **Willis 环二维截面网格**：基于 SDF 与力平衡迭代生成脑血管二维网格 |
| 8 | `038_asa111` | 正态分布逆 CDF (AS 111) | **血流统计变异性采样**：用逆变换法生成服从正态分布的脉动血流样本 |
| 9 | `309_distmesh_3d` | 3D 距离函数网格生成 | **三维脑血管网格生成**：生成脑实质血管树的三维四面体网格 |
| 10 | `199_collatz_recursive` | 递归序列 | **血管分支递归生成**：类 Collatz 规则递归生成多级血管分叉树 |
| 11 | `1248_tetrahedron_integrals` | 四面体积分 | **血管区域血容量计算**：基于四面体剖分的体积积分 |
| 12 | `1310_triangle_io` | 三角网格 I/O | **血管网格数据读写**：TRIANGLE 格式的节点/单元文件读写 |
| 13 | `266_cycle_brent` | Brent 循环检测 | **血流周期性检测**：检测脑血流自动调节系统中的周期振荡 |
| 14 | `562_hypersphere` | 超球均匀采样 | **血管截面蒙特卡洛采样**：在血管截面上均匀采样红细胞位置 |
| 15 | `126_burgers_time_inviscid` | 无粘 Burgers 方程 | **脑动脉压力波传播**：Godunov 格式求解非线性压力脉冲在弹性动脉中的传播 |

---

## 二、新增数学物理模型与核心公式

### 2.1 血压波传播：无粘 Burgers 方程

脑动脉中的压力脉冲传播可用一维非线性波动方程描述：

$$
\frac{\partial u}{\partial t} + \frac{\partial}{\partial x}\left(\frac{u^2}{2}\right) = 0
$$

采用 **Godunov 格式** 进行数值求解。数值通量通过求解局部 Riemann 问题得到：

$$
\hat{f}(u_L, u_R) = \begin{cases}
f(u_L), & u_L \geq u_R, \; u^* > 0 \\
f(u_R), & u_L \geq u_R, \; u^* \leq 0 \\
f(u_L), & u_L < u_R, \; u_L > 0 \\
f(u_R), & u_L < u_R, \; u_R < 0 \\
0, & \text{其他}
\end{cases}
$$

其中 $u^*$ 为 Riemann 问题的中间状态，$f(u) = u^2/2$。

### 2.2 稳态压力场：Laplace 方程与 Poiseuille 定律

微血管层面，稳态血流满足 Laplace 方程：

$$
\nabla^2 P = 0
$$

其二维径向精确解为：

$$
P(r) = a \ln r + b, \quad r = \sqrt{x^2 + y^2}
$$

圆管中的体积流量由 **Poiseuille 定律** 给出：

$$
Q = \frac{\pi r^4 \Delta P}{8 \mu L}
$$

其中 $\mu$ 为血液动力粘度（约 $3.5 \times 10^{-3}$ Pa·s）。

### 2.3 组织氧扩散：Krogh 模型与反应扩散方程

脑组织氧合遵循反应-扩散方程：

$$
\frac{\partial C}{\partial t} = D \nabla^2 C + \lambda C\left(1 - \frac{C}{C_{\max}}\right) - k_{\text{met}} C
$$

**Krogh 圆柱模型** 的稳态解析解：

$$
P(r) = P_c - \frac{M_0}{4 D_t}(r^2 - R_c^2) + \frac{M_0 R_t^2}{2 D_t} \ln\frac{r}{R_c}
$$

氧消耗采用 **Michaelis-Menten 动力学**：

$$
V(C) = \frac{V_{\max} C}{K_m + C}
$$

### 2.4 血管重构动力学

**血氧三室模型**（SIR 型）：

$$
\begin{aligned}
\frac{dS}{dt} &= -\alpha \frac{SI}{N} + \gamma R \\
\frac{dI}{dt} &= \alpha \frac{SI}{N} - \beta I \\
\frac{dR}{dt} &= \beta I - \gamma R
\end{aligned}
$$

**内皮细胞竞争模型**（Lotka-Volterra 型）：

$$
\begin{aligned}
\frac{dE}{dt} &= \alpha_E E - \beta_E E A \\
\frac{dA}{dt} &= -\gamma_A A + \delta_A E A
\end{aligned}
$$

**剪切应力调控的半径演化**：

$$
\frac{dr}{dt} = k_\tau \left|\frac{\tau - \tau_{\text{ref}}}{\tau_{\text{ref}}}\right|^{1/2} \text{sgn}(\tau - \tau_{\text{ref}}) - k_R (r - r_0)
$$

壁面剪切应力：

$$
\tau_w = \frac{4 \mu Q}{\pi r^3}
$$

### 2.5 血管分支：Murray 定律与递归生成

**Murray 定律**（能量最小化）：

$$
r_0^3 = r_1^3 + r_2^3
$$

对称分支时：

$$
r_1 = r_2 = \frac{r_0}{2^{1/3}}
$$

**分支角度关系**（由能量最小化导出）：

$$
\cos\theta_1 = \frac{r_0^4 + r_1^4 - r_2^4}{2 r_0^2 r_1^2}, \quad
\cos\theta_2 = \frac{r_0^4 + r_2^4 - r_1^4}{2 r_0^2 r_2^2}
$$

### 2.6 血液流变学

**Fahraeus-Lindqvist 效应**（Pries et al., 1992）：

$$
\mu_{\text{app}} = \mu_{\text{plasma}} \left[1 + (\mu_{0.45} - 1) \frac{(1 - Hct)^C - 1}{(1 - 0.45)^C - 1}\right]
$$

其中：

$$
\mu_{0.45} = 220 e^{-1.3d} + 3.2 - 2.44 e^{-0.06 d^{0.645}}
$$

$$
C = (0.8 + e^{-0.075d})\left(-1 + \frac{1}{1 + 10^{-11}d^{12}}\right) + \frac{1}{1 + 10^{-11}d^{12}}
$$

**分叉处红细胞压积分配**（Pries 两相模型）：

$$
\frac{Hct_{d1}}{Hct_{d2}} = \left(\frac{Q_{d1}}{Q_{d2}}\right)^n \left(\frac{D_{d1}}{D_{d2}}\right)^m, \quad n \approx 1.0, \; m \approx 0.5
$$

### 2.7 网格质量度量

**Q-度量**（三角形单元质量）：

$$
Q(T) = \frac{(b+c-a)(c+a-b)(a+b-c)}{abc} = \frac{2r_{\text{in}}}{r_{\text{out}}}
$$

其中 $r_{\text{in}}$ 为内切圆半径，$r_{\text{out}}$ 为外接圆半径。对于正三角形，$Q = 1$。

**n 维球体积**：

$$
V_n(r) = \frac{\pi^{n/2} r^n}{\Gamma(n/2 + 1)}
$$

### 2.8 血流周期性检测

**Brent 循环检测算法**：对于迭代映射 $x_{n+1} = f(x_n)$，在 $O(\mu + \lambda)$ 时间内检测进入周期前的步数 $\mu$ 与周期长度 $\lambda$。

**脑血管自动调节映射**：

$$
CBF_{n+1} = CBF_n + k_1 \frac{MAP - MAP_{ss}}{MAP_{ss}} - k_2 \frac{CBF_n - CBF_{ss}}{CBF_{ss}} - k_3 \sin(2\pi f_{\text{heart}} n \Delta t)
$$

### 2.9 Windkessel 模型

动脉出口压力的离散化 Windkessel 模型：

$$
P(t) + RC \frac{dP}{dt} = R Q_{\text{in}}(t)
$$

离散格式：

$$
P_{n+1} = \frac{P_n + R Q_n \Delta t / C}{1 + \Delta t / (RC)}
$$

---

## 三、合成项目文件结构

```
122_synth_project/
├── main.py                          # 统一入口，零参数运行
├── cerebral_vascular_mesh.py        # 2D/3D 脑血管网格生成 (distmesh 2D/3D)
├── blood_pressure_wave.py           # 血压波传播 + 压力场求解 (Burgers + Laplace)
├── tissue_oxygen_diffusion.py       # 组织氧扩散与代谢 (Fisher PDE)
├── vascular_remodeling.py           # 血管重构动力学 (SIR + Predator-Prey)
├── blood_cell_dynamics.py           # 血细胞动力学与统计变异性 (Jai-Alai + ASA111)
├── network_topology_quality.py      # 网络拓扑质量评估 (Quality Measures)
├── hemodynamic_integrals.py         # 血流积分与随机采样 (Tetrahedron + Hypersphere + Triangle I/O)
├── vascular_branching.py            # 血管分支递归生成 (Collatz-like Recursion)
├── flow_cycle_analysis.py           # 血流周期性与循环检测 (Brent Cycle Detection)
└── README_博士级合成说明.md          # 中文说明文档
```

---

## 四、各文件的具体改造与实现

### `cerebral_vascular_mesh.py`
- **来源**：`308_distmesh` (2D) + `309_distmesh_3d` (3D)
- **改造**：将 MATLAB 的 distmesh 算法完整移植为 Python/NumPy/SciPy 实现，增加了 `generate_willis_ring_mesh()` 与 `generate_cerebral_vessel_3d()` 两个面向脑血管几何的封装函数。
- **科学应用**：生成 Willis 环截面与脑实质血管树的三维计算网格。

### `blood_pressure_wave.py`
- **来源**：`126_burgers_time_inviscid` + `646_laplace_radial_exact`
- **改造**：移植 Godunov 数值通量求解无粘 Burgers 方程；实现二维/三维径向 Laplace 精确解；新增 Windkessel 出口模型、Poiseuille 流量计算、 vascular network 压力场线性求解器。
- **科学应用**：模拟压力脉冲在脑动脉的传播与稳态压力分布。

### `tissue_oxygen_diffusion.py`
- **来源**：`434_fisher_pde_ftcs`
- **改造**：将一维 FTCS 格式扩展为 1D/2D 径向形式；新增 Michaelis-Menten 氧消耗与 Krogh 圆柱解析解。
- **科学应用**：模拟氧气从毛细血管向脑实质的扩散与代谢消耗。

### `vascular_remodeling.py`
- **来源**：`1086_sir_ode` + `908_predator_prey_ode`
- **改造**：将两个独立的 ODE 系统耦合为六维状态向量 `[S, I, R, E, A, r]`，新增剪切应力调控半径的项；实现 Murray 分支定律与壁面剪切应力计算。
- **科学应用**：模拟血氧运输与血管内皮细胞动态重构的耦合过程。

### `blood_cell_dynamics.py`
- **来源**：`609_jai_alai_simulation` + `038_asa111`
- **改造**：移植 jai-alai 竞争排队为血细胞分叉竞争模型；完整实现 ASA111 正态逆 CDF 算法（ppnd）；新增 Fahraeus-Lindqvist 粘度公式、红细胞压积分配、脉动血流随机模型。
- **科学应用**：模拟红细胞在血管分叉的竞争行为与血流统计涨落。

### `network_topology_quality.py`
- **来源**：`958_quality`
- **改造**：移植 Q-度量、Alpha-度量、球填充度量、半带宽度量；新增四面体质量度量；封装为 `vascular_network_quality_report()` 综合评估函数。
- **科学应用**：评估脑血管网格的计算适用性与几何质量。

### `hemodynamic_integrals.py`
- **来源**：`1248_tetrahedron_integrals` + `562_hypersphere` + `1310_triangle_io`
- **改造**：移植单位四面体单项式积分、体积计算与采样；移植超球面/体内均匀采样并扩展为血管截面采样与蒙特卡洛流量积分；移植 TRIANGLE 格式的节点/单元读写。
- **科学应用**：计算血管血容量、截面红细胞分布、蒙特卡洛估计流量。

### `vascular_branching.py`
- **来源**：`199_collatz_recursive`
- **改造**：将 Collatz 递归思想转化为血管分支规则（半径大于毛细血管阈值则二分，否则终止）；实现 Murray 定律角度计算；构建完整的 `VascularBranch` 树结构。
- **科学应用**：递归生成符合生理规律的多级脑血管树。

### `flow_cycle_analysis.py`
- **来源**：`266_cycle_brent`
- **改造**：移植 Brent 循环检测算法；新增脑血管自动调节离散映射、FFT 频谱分析、血流状态自动分类。
- **科学应用**：检测脑血流的周期性异常（如血管痉挛导致的异常振荡）。

### `main.py`
- **改造**：唯一入口文件，串联全部 9 个模块，执行 9 个阶段的完整模拟流程并输出结果。

---

## 五、合成项目解决的科学问题

本项目解决的核心科学问题是：

> **如何在多尺度框架下，从宏观血压波传播到微观组织氧合，完整模拟脑血管系统的血流动力学行为，并评估血管网络几何对计算精度与生理功能的影响？**

具体包括：

1. **宏观尺度**：脑动脉压力脉冲的非线性传播（Burgers 方程）与 Windkessel 出口效应
2. **网络尺度**：脑血管网格的自动生成（DistMesh 2D/3D）与拓扑质量评估
3. **传输尺度**：血氧三室模型（SIR）与红细胞分叉竞争（Jai-Alai）
4. **组织尺度**：氧气在脑实质中的扩散-反应过程（Fisher-KPP PDE）与 Krogh 模型
5. **细胞尺度**：内皮细胞增殖-凋亡竞争（Predator-Prey）与剪切应力诱导的血管重构
6. **结构尺度**：符合 Murray 定律的递归血管树生成与分支角度优化
7. **诊断尺度**：血流周期性异常检测（Brent 循环检测 + FFT 频谱分析）

---

## 六、运行方式

### 环境要求
- Python 3.8+
- NumPy
- SciPy

### 运行命令
```bash
cd Synthesis-project-python/122_synth_project
python main.py
```

程序将自动执行以下流程：
1. 生成 Willis 环 2D 网格与脑实质 3D 网格
2. 评估网格拓扑质量
3. 模拟 Burgers 压力波传播与 Laplace 压力场
4. 求解组织氧扩散 PDE
5. 求解耦合血管重构 ODE
6. 模拟血细胞竞争与统计变异性
7. 递归生成血管树
8. 检测血流周期性
9. 计算血容量与蒙特卡洛流量积分

运行结束后，结果汇总保存至 `simulation_report.txt`。

---

## 七、数值鲁棒性与边界处理

- **Laplace 径向解**：在 $r=0$ 处加入截断保护 $r_{\min} = 10^{-14}$，避免对数与除零奇异
- **Burgers Godunov 通量**：完整处理冲击波与稀疏波的全部情形分支，确保数值稳定性
- **FTCS 扩散**：自动检测 CFL 条件，超出稳定性限制时自适应调整时间步长
- **ODE 求解**：采用 `scipy.integrate.solve_ivp` 的 RK45 方法，带自动步长控制
- **网格生成**：力平衡迭代带收敛判据，外部节点通过梯度步投影回边界
- **正态逆 CDF**：对 $p \leq 0$ 或 $p \geq 1$ 返回错误标志，防止数值溢出
- **血管树递归**：半径下限截断（毛细血管阈值），防止无限递归
- **线性求解**：conductance 矩阵添加正则项 $10^{-12}I$，对奇异系统使用最小二乘回退

---

## 八、参考文献与理论依据

1. Persson, P.O. & Strang, G. (2004). A Simple Mesh Generator in MATLAB. *SIAM Review*, 46(2), 329-345.
2. Beasley, J.D. & Springer, S.G. (1977). Algorithm AS 111: The Percentage Points of the Normal Distribution. *Applied Statistics*, 26(1), 118-121.
3. Pries, A.R., et al. (1992). Blood Flow in Microvascular Networks. *Microcirculation*, 1(2).
4. Krogh, A. (1919). The Number and Distribution of Capillaries in Muscles with Calculations of the Oxygen Pressure Head Necessary for Supplying the Tissue. *Journal of Physiology*, 52(6), 409-415.
5. Murray, C.D. (1926). The Physiological Principle of Minimum Work. *Proceedings of the National Academy of Sciences*, 12(3), 207-214.
6. Brent, R.P. (1980). An Improved Monte Carlo Factorization Algorithm. *BIT*, 20(2), 176-184.
7. Kermack, W.O. & McKendrick, A.G. (1927). A Contribution to the Mathematical Theory of Epidemics. *Proceedings of the Royal Society A*, 115(772), 700-721.
8. Fisher, R.A. (1937). The Wave of Advance of Advantageous Genes. *Annals of Eugenics*, 7(4), 355-369.
