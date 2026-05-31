# PROJECT_54：海洋酸化与碳循环综合模拟系统

## 一、项目概述

本项目围绕**海洋科学：海洋酸化与碳循环**这一前沿领域，将 15 个种子科研代码项目的核心算法融合重构为一个面向博士级科学计算的 Python 综合模型系统。该系统涵盖海水碳酸盐化学平衡求解、三维碳库存数值积分、碳输送反应动力学、海洋锋面检测、监测网络优化、碳封存深度优化以及参数不确定性量化等关键模块。

## 二、科学问题背景

### 2.1 海洋酸化与碳循环

工业革命以来，大气 CO₂ 浓度已从约 280 ppm 上升至 420 ppm 以上，其中约 30% 被海洋吸收。CO₂ 进入海水后形成碳酸，导致海洋 pH 下降（ ocean acidification ），同时改变碳酸盐系统的平衡状态：

```
CO₂ + H₂O ⇌ H₂CO₃ ⇌ H⁺ + HCO₃⁻ ⇌ 2H⁺ + CO₃²⁻
```

关键科学指标包括：
- **pH**：表层海水 pH 已从工业化前的 ~8.17 下降至 ~8.05
- **碳酸钙饱和度 (Ω)**：Ω = [Ca²⁺][CO₃²⁻] / Ksp，当 Ω < 1 时碳酸盐溶解
- **海-气 CO₂ 通量**：F = k_w · K₀ · (pCO₂_海洋 − pCO₂_大气)
- **人为碳库存**：C_anth = ∭ ρ · (DIC_现代 − DIC_工业化前) dV

### 2.2 核心数学模型

**碳酸盐系统质子平衡方程**（非线性，需数值求根）：

$$f([H^+]) = TA - DIC \cdot (\alpha_1 + 2\alpha_2) - \frac{K_w}{[H^+]} + [H^+] - \frac{B_T \cdot K_B}{K_B + [H^+]} = 0$$

其中各物种分数由平衡常数决定：

$$\alpha_0 = \frac{[H^+]^2}{[H^+]^2 + K_1[H^+] + K_1K_2}, \quad \alpha_1 = \frac{K_1[H^+]}{D}, \quad \alpha_2 = \frac{K_1K_2}{D}$$

**Brunt-Väisälä 浮力频率**（层化稳定性）：

$$N^2 = -\frac{g}{\rho_0} \frac{d\rho}{dz}$$

**一维垂向碳输送方程**：

$$\frac{\partial DIC}{\partial t} = -w\frac{\partial DIC}{\partial z} + \frac{\partial}{\partial z}\left(K_z \frac{\partial DIC}{\partial z}\right) + J_{bio} - J_{air\text{-}sea} + J_{remin}$$

**三箱碳循环模型**（大气-表层海洋-深层海洋）：

$$\begin{aligned}
\frac{dN_1}{dt} &= -k_{12}N_1 + k_{21}N_2 + F_{anthro} \\
\frac{dN_2}{dt} &= k_{12}N_1 - k_{21}N_2 - k_{23}N_2 + k_{32}N_3 - \frac{N_2 - N_2^0}{R_{Revelle}} \\
\frac{dN_3}{dt} &= k_{23}N_2 - k_{32}N_3
\end{aligned}$$

## 三、种子项目映射与改造方法

| 序号 | 种子项目 | 核心算法 | 在合成项目中的角色 | 改造文件 |
|:---:|:---|:---|:---|:---|
| 1 | 1073_shepard_interp_nd | N 维 Shepard 反距离插值 | 稀疏海洋观测数据（DIC/TA/T/S）的空间插值 | sparse_interpolation.py |
| 2 | 1431_zero_muller | Muller 二次插值根查找（复数） | 求解碳酸盐系统质子平衡方程 f([H⁺])=0 | carbonate_chemistry.py |
| 3 | 490_grf_io | GRF 图文件 I/O | 海洋功能区连通性网络的构建与序列化 | region_connectivity.py |
| 4 | 1366_tsp_moler | Moler 随机启发式 TSP | 监测站点最优巡航路径规划 | sensor_network.py |
| 5 | 518_hermite_cubic | Hermite 三次样条 | 海水垂直剖面（T/S/DIC）的高阶插值与层化分析 | vertical_profiles.py |
| 6 | 752_mesh_bandwidth | FEM 网格带宽分析 | Q4 海洋网格稀疏矩阵带宽计算 | ocean_mesh.py |
| 7 | 834_opt_golden | 黄金分割搜索 | 最优海洋碳封存深度的单峰优化 | sequestration_optimizer.py |
| 8 | 953_quadrilateral_mesh | Q4 四边形网格工具包 | 二维海洋 basin 网格生成、面积计算、边界识别 | ocean_mesh.py |
| 9 | 343_euler | 显式前向欧拉法 | 碳输送 ODE 和三箱碳循环模型的时间积分 | carbon_transport.py |
| 10 | 255_cvt_corn | CVT/Lloyd 算法（圆盘） | 监测传感器的最优空间分布 | sensor_network.py |
| 11 | 059_autocatalytic_ode | 自催化反应动力学 | 碳酸盐快速动力学缓冲过程的简化模型 | carbon_transport.py |
| 12 | 579_image_edge | NEWS 差分边缘检测 | 海洋锋面（温度/盐度/DIC 梯度）的自动识别 | front_analysis.py |
| 13 | 232_cube_felippa_rule | 3D 立方体高斯求积 | 长方体海域碳库存的三维数值积分 | quadrature_engine.py |
| 14 | 1250_tetrahedron_keast_rule | Keast 四面体求积 | 非结构化四面体单元上的高精度积分 | quadrature_engine.py |
| 15 | 565_hypersphere_integrals | 超球面积分与采样 | 高维参数空间的不确定性量化与敏感性分析 | uncertainty_analysis.py |

## 四、项目文件结构

```
054_synth_project/
├── main.py                      # 统一入口，零参数运行
├── carbonate_chemistry.py       # 碳酸盐化学 + Muller 根查找
├── ocean_mesh.py                # Q4 网格生成 + 带宽分析
├── vertical_profiles.py         # Hermite 样条 + 层化分析
├── carbon_transport.py          # 碳输送 ODE + 箱式模型
├── quadrature_engine.py         # 3D 立方体/四面体求积 + 超球面采样
├── sensor_network.py            # CVT 布点 + TSP 路径规划
├── sparse_interpolation.py      # Shepard N 维插值
├── front_analysis.py            # NEWS/Sobel 锋面检测
├── sequestration_optimizer.py   # 黄金分割封存优化
├── uncertainty_analysis.py      # 超球面不确定性量化
├── region_connectivity.py       # 区域连通性图网络
└── README_博士级合成说明.md     # 本文档
```

## 五、核心公式与算法

### 5.1 平衡常数（温度-盐度依赖）

**碳酸第一解离常数 K₁**（Lueker et al., 2000, total scale）：

$$pK_1 = \frac{3633.86}{T_K} - 61.2172 + 9.67770 \ln T_K - 0.011555 \, S + 0.0001152 \, S^2$$

**碳酸第二解离常数 K₂**（Lueker et al., 2000, total scale）：

$$pK_2 = \frac{471.78}{T_K} + 25.929 - 3.16967 \ln T_K - 0.01781 \, S + 0.0001122 \, S^2$$

**CO₂ 溶解度 K₀**（Weiss, 1974）：

$$\ln K_0 = -60.2409 + \frac{93.4517}{T_{100}} + 23.3585 \ln T_{100} + S \left(0.023517 - 0.023656 \, T_{100} + 0.0047036 \, T_{100}^2\right)$$

### 5.2 Muller 根查找法

对于方程 f(x)=0，已知三点 (x_old, f_old), (x_mid, f_mid), (x_new, f_new)，拟合抛物线：

$$P(x) = a(x-x_{new})^2 + b(x-x_{new}) + c$$

其中：

$$a = \frac{(x_{mid}-x_{new})(f_{old}-f_{new}) - (x_{old}-x_{new})(f_{mid}-f_{new})}{(x_{old}-x_{new})(x_{mid}-x_{new})(x_{old}-x_{mid})}$$

$$b = \frac{(x_{old}-x_{new})^2(f_{mid}-f_{new}) - (x_{mid}-x_{new})^2(f_{old}-f_{new})}{(x_{old}-x_{new})(x_{mid}-x_{new})(x_{old}-x_{mid})}$$

$$x_{\pm} = x_{new} + \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

选择使 |f(x)| 更小的根，收敛阶 ≈ 1.84。

### 5.3 Hermite 三次样条

在区间 [z₁, z₂] 上，设 h = z₂ − z₁，df = (f₂ − f₁)/h：

$$P(z) = f_1 + (z-z_1)\left[d_1 + (z-z_1)\left(c_2 + (z-z_1)c_3\right)\right]$$

$$c_2 = -\frac{2d_1 - 3df + d_2}{h}, \quad c_3 = \frac{d_1 - 2df + d_2}{h^2}$$

### 5.4 三维高斯求积

对于长方体 [a,b]×[c,d]×[e,f]，张量积规则：

$$\iiint_V f(x,y,z)\,dx\,dy\,dz \approx \sum_{i=1}^{n_x}\sum_{j=1}^{n_y}\sum_{k=1}^{n_z} w_i w_j w_k \, f(x_i, y_j, z_k)$$

其中 (x_i, w_i) 为 1D Gauss-Legendre 节点和权重。

### 5.5 Keast 四面体求积

在参考四面体上的积分：

$$\int_T f\,dV = |\det J| \sum_k w_k \, f(\xi_k, \eta_k, \zeta_k)$$

其中 (ξ_k, η_k, ζ_k) 为重心坐标映射后的求积点。

### 5.6 CVT 最优布点

Lloyd 算法迭代：

$$g_i^{(t+1)} = \frac{\int_{V_i(g^{(t)})} x \, \rho(x) \, dx}{\int_{V_i(g^{(t)})} \rho(x) \, dx}$$

其中 V_i 为第 i 个生成元的 Voronoi 单元，ρ(x) 为采样密度函数。

### 5.7 TSP Moler 启发式

对于距离矩阵 D，随机初始回路 π，迭代进行：
1. **2-opt 反转**：随机选 i,j，反转 π[i:j]，若路径缩短则接受
2. **单点插入**：随机移除一点并插入新位置，若路径缩短则接受

### 5.8 黄金分割搜索

对于单峰函数 f(x) 在 [a,b] 上，内点：

$$x_1 = \varphi a + (1-\varphi)b, \quad x_2 = (1-\varphi)a + \varphi b, \quad \varphi = \frac{\sqrt{5}-1}{2} \approx 0.618$$

比较 f(x₁) 与 f(x₂)，保留含极小值的子区间，收敛比 = φ。

### 5.9 超球面参数敏感性

在 m 维标准化参数空间，单位球面 S^{m−1} 上均匀采样：

$$\omega = \frac{X}{\|X\|}, \quad X \sim \mathcal{N}(0, I_m)$$

Sobol 一阶敏感性指数：

$$S_i = \frac{\text{Var}(\mathbb{E}[Y|\theta_i])}{\text{Var}(Y)} \approx 1 - \frac{\text{Var}_{\theta_{\sim i}}(Y)}{\text{Var}(Y)}$$

## 六、运行方式

```bash
cd Synthesis-project-python/054_synth_project
python main.py
```

程序无需任何输入参数，运行后依次执行 11 个科学计算模块的演示，输出包括：
- 碳酸盐化学系统求解结果（pH、pCO₂、Ω）
- 海洋网格拓扑分析
- 垂直剖面 Hermite 样条插值与 Brunt-Väisälä 频率
- 碳输送反应动力学模拟
- 三维碳库存数值积分
- CVT 传感器布点与 TSP 最优路径
- 稀疏观测 Shepard 插值
- 海洋锋面检测
- 碳封存深度优化
- 参数不确定性量化与 Sobol 敏感性分析
- 海洋区域连通性图网络分析

## 七、数值验证

本项目使用 **pyCO2SYS**（海洋碳酸盐系统计算的国际标准 Python 包）对核心碳酸盐化学模块进行交叉验证：

| 参数 | 本系统输出 | pyCO2SYS 验证值 | 相对误差 |
|:---|:---|:---|:---|
| pH (T=15°C, S=35, DIC=2000, TA=2300) | 8.193 | 8.199 | < 0.1% |
| pCO₂ (μatm) | 265.9 | 262.8 | ~1.2% |
| Ω_aragonite | 3.162 | 3.203 | ~1.3% |
| pK₁ | 5.940 | 5.940 | < 0.01% |
| pK₂ | 9.129 | 9.129 | < 0.01% |

## 八、边界处理与数值鲁棒性

1. **Muller 根查找**：处理退化情形（分母接近零时退化为割线法），收敛检验包括 |f(x)|、绝对增量和相对增量三重容差
2. **碳酸盐系统**：DIC/TA 非负检验、温度/盐度物理范围检验、复数输入自动取实部
3. **Hermite 样条**：区间外推使用最近区间、单调性检验、导数估计的中心差分边界处理
4. **Euler 积分**：内部状态非负约束、边界 Neumann 条件
5. **数值积分**：权重归一化、退化单元体积检验
6. **TSP**：距离矩阵对称化、单节点退化处理
7. **黄金分割**：搜索区间端点惩罚、多情景鲁棒性检验

## 九、科学计算难度说明

本项目具备博士级科学计算复杂度：
- **非线性方程求解**：碳酸盐系统涉及 6 个温度-盐度依赖的平衡常数与质子平衡的非线性耦合
- **多尺度动力学**：从快速碳酸盐化学平衡（秒级）到全球碳循环（百年尺度）的多时间尺度整合
- **高维不确定性**：6 维参数空间上的 Monte Carlo 敏感性分析
- **多物理场耦合**：物理输送（平流-扩散）、化学平衡、生物泵、地质封存的耦合模拟
- **最优控制问题**：碳封存深度的多目标（稳定性-滞留时间-酸化影响）帕累托优化
