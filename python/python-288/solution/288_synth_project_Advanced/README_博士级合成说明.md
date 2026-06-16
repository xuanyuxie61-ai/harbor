# 博士级科研代码合成说明文档

## PROJECT 288: 边界等离子体输运与偏滤器热负荷 — 高阶有限差分与稳定性分析

### 项目概述

本项目是面向**计算等离子体物理**方向的博士级科研项目合成，聚焦**边界等离子体输运与偏滤器热负荷**问题。通过融合15个不同来源的数值计算与科学工程种子项目，构建了一个完整的、可复现的小规模科学计算实验平台。

**核心科学问题：** 在托卡马克装置中，边界等离子体（SOL区域）的输运过程直接决定偏滤器靶板的热负荷分布，这关系到聚变反应堆第一壁材料的安全和寿命。本项目使用高阶有限差分与不连续Galerkin方法，对平行于磁力线和垂直于磁力线方向的等离子体输运进行数值模拟，并通过von Neumann稳定性分析确保数值格式的可靠性。

---

### 科学问题背景

#### 托卡马克中的边界等离子体

在托卡马克磁约束聚变装置中，等离子体被强磁场约束在环形真空室内。然而在等离子体边界（分离面之外）的**刮削层（Scrape-Off Layer, SOL）**区域，开放的磁力线将等离子体引导至偏滤器靶板。这一区域的关键物理过程包括：

1. **平行输运**：沿磁力线方向的快速输运（电子热传导为主）
2. **垂直输运**：跨磁力线的反常扩散（湍流驱动）
3. **鞘层边界条件**：等离子体与材料表面的相互作用
4. **偏滤器靶板热负荷**：决定材料寿命的关键参数

#### 控制方程

**平行方向输运方程：**

$$\frac{\partial u}{\partial t} + a \frac{\partial u}{\partial s} = D_\parallel \frac{\partial^2 u}{\partial s^2} + S(s)$$

其中 $u$ 为等离子体密度或温度，$a$ 为平行流速（可达声速 $c_s = \sqrt{(T_e + T_i)/m_i}$），$D_\parallel$ 为平行扩散系数，$S(s)$ 为源项。

**垂直方向扩散方程：**

$$-\nabla \cdot (\mathbf{D}_\perp \cdot \nabla T) = Q$$

其中 $\mathbf{D}_\perp$ 为各向异性扩散张量（包含经典扩散与反常扩散贡献）。

**Spitzer-Harm平行热传导系数：**

$$\kappa_\parallel = 3.2 \frac{n_e k_B^2 T_e \tau_e}{m_e e} \approx 2.56 \times 10^3 \frac{T_e^{5/2}}{\ln\Lambda} \quad [\text{W/(m·eV)}]$$

**偏滤器靶板热负荷：**

$$q_{\text{target}} = \gamma_{\text{sheath}} \cdot n_{\text{target}} \cdot T_{\text{target}} \cdot c_s$$

其中 $\gamma_{\text{sheath}} \approx 7$ 为鞘层热透射系数，$c_s = \sqrt{(T_e + T_i)/m_i}$ 为离子声速。

**Bohm判据**：在鞘层边缘，平行流速必须达到或超过声速：

$$v_\parallel \geq c_s = \sqrt{\frac{T_e + T_i}{m_i}}$$

---

### 15个种子项目的融合映射

| 序号 | 原始项目 | 核心算法 | 在合成项目中的角色 | 对应模块 |
|------|---------|---------|-------------------|---------|
| 1 | 1109_marekgluza_Fidelity_witnesses | 量子保真度度量 | 等离子体状态保真度追踪 | plasma_physics.py |
| 2 | 736_matman | 矩阵行操作、LU分解 | 线性系统求解核心 | iterative_solver.py |
| 3 | 109_boundary_word_right | 边界字表示、多边形网格 | 偏滤器靶板几何描述 | divertor_geometry.py |
| 4 | 152_cg_rc | 反向通信共轭梯度法 | 稀疏线性系统迭代求解 | iterative_solver.py |
| 5 | 664_legendre_product_polynomial | Legendre多项式递推与求值 | 高阶有限差分格式构造 | legendre_fd.py |
| 6 | 271_dg1d_advection | 不连续Galerkin方法 | 平行方向DG输运求解 | dg_parallel_solver.py |
| 7 | 408_fem2d_poisson_rectangle | 二维Poisson FEM | 垂直方向扩散FEM求解 | perp_diffusion.py |
| 8 | 1403_wavelet | Daubechies小波变换 | 热负荷信号诊断分析 | wavelet_diagnostics.py |
| 9 | 351_fd_to_tec | 数据格式化输出 | 模拟结果数据输出 | output_diagnostics.py |
| 10 | 926_pwl_interp_1d | 分段线性插值 | 等离子体剖面插值 | profile_tools.py |
| 11 | 561_hypercube_surface_distance | 高维采样统计 | Monte Carlo不确定性分析 | parameter_scan.py |
| 12 | 756_mesh_vtoe | 顶点到单元逆映射 | SOL网格拓扑构建 | divertor_geometry.py |
| 13 | 832_ode_sweep_parfor | 参数扫描框架 | 多参数依赖性扫描 | parameter_scan.py |
| 14 | 535_hilbert_curve | Hilbert空间填充曲线 | 网格遍历缓存优化 | divertor_geometry.py |
| 15 | 077_bernstein_approximation | Bernstein多项式逼近 | 等离子体剖面光滑重建 | legendre_fd.py |

---

### 项目文件结构

```
288_synth_project_Advanced/
├── __init__.py                  # 包初始化
├── main.py                      # 统一入口（零参数可运行）
├── plasma_physics.py            # 核心物理常数与方程
├── divertor_geometry.py         # 偏滤器几何与SOL网格
├── legendre_fd.py               # Legendre高阶有限差分
├── dg_parallel_solver.py        # DG平行输运求解器
├── perp_diffusion.py            # FEM垂直扩散求解器
├── iterative_solver.py          # 线性系统迭代求解
├── stability_analysis.py        # von Neumann稳定性分析
├── wavelet_diagnostics.py       # 小波热负荷诊断
├── profile_tools.py             # 剖面插值与重建
├── parameter_scan.py            # 参数扫描与不确定性
├── output_diagnostics.py        # 数据输出与诊断
├── README_博士级合成说明.md     # 本文档
├── simulation_summary.txt       # 运行时生成的摘要
└── divertor_heat_flux.txt       # 运行时生成的热负荷数据
```

---

### 核心算法与公式

#### 1. Legendre多项式与GLL节点

**三项递推关系：**
$$P_0(x) = 1, \quad P_1(x) = x$$
$$(n+1)P_{n+1}(x) = (2n+1)x P_n(x) - n P_{n-1}(x)$$

**GLL节点：** $(1-x^2)P'_N(x) = 0$ 的根，包括端点 $x = \pm 1$

**GLL积分权重：** $w_j = \frac{2}{N(N+1)[P_N(x_j)]^2}$

#### 2. 紧致有限差分格式（Pade格式）

**四阶紧致一阶导数格式：**
$$\frac{1}{4}f'_{i-1} + f'_i + \frac{1}{4}f'_{i+1} = \frac{3}{2}\frac{f_{i+1} - f_{i-1}}{2h}$$

**六阶紧致格式：**
$$\frac{1}{3}f'_{i-1} + f'_i + \frac{1}{3}f'_{i+1} = \frac{14}{9}\frac{f_{i+1} - f_{i-1}}{2h} + \frac{1}{9}\frac{f_{i+2} - f_{i-2}}{4h}$$

#### 3. Bernstein多项式逼近

$$B_{i,n}(x) = \binom{n}{i} \left(\frac{x-a}{b-a}\right)^i \left(\frac{b-x}{b-a}\right)^{n-i}$$

**Bernstein逼近算子：** $B_n(f)(x) = \sum_{i=0}^n f(x_i) B_{i,n}(x)$

**Weierstrass逼近定理保证一致收敛性**

#### 4. DG弱形式

在每个单元 $K$ 上，DG弱形式为：

$$\int_K \frac{\partial u}{\partial t} v \, dx - \int_K F(u) \frac{\partial v}{\partial x} dx + [F^* v]_{\partial K} = \int_K S v \, dx$$

**Lax-Friedrichs数值通量：**
$$F^* = \frac{1}{2}(F(u_L) + F(u_R)) - \frac{1}{2}\lambda_{\max}(u_R - u_L)$$

#### 5. von Neumann稳定性分析

**放大因子定义：** $u_j^n = G^n e^{i k j h}$

**四阶紧致格式放大因子：**
$$G = \frac{1 - i \cdot \text{CFL} \cdot (2a\sin(kh) + 2b\sin(2kh))}{1 + 2\alpha\cos(kh)}$$

**稳定性条件：** $|G| \leq 1$

#### 6. Coulomb对数

$$\ln\Lambda = 24 - \ln\left(\frac{\sqrt{n_e[\text{cm}^{-3}]}}{T_e[\text{eV}]}\right) \quad (T_e > 10 \text{ eV})$$

$$\ln\Lambda = 23 - \ln\left(\sqrt{n_e[\text{cm}^{-3}]} \cdot T_e[\text{eV}]^{-3/2}\right) \quad (T_e < 10 \text{ eV})$$

#### 7. 小波变换

**Daubechies滤波器系数** $h_k$ 满足：
- 正交性：$\sum_k h_k h_{k-2n} = \delta_{n,0}$
- 消失矩：$\sum_k (-1)^k k^m h_k = 0, \quad m = 0, \ldots, p-1$
- 归一化：$\sum_k h_k = \sqrt{2}$

**离散小波变换：** $a[n] = \sum_k h_{k-2n} x[k]$, $d[n] = \sum_k g_{k-2n} x[k]$

---

### 数值方法概述

#### 空间离散
- **平行方向：** 不连续Galerkin方法（1-5阶多项式）
- **垂直方向：** 有限元方法（线性三角形元）
- **有限差分：** 4/6阶紧致（Pade）格式

#### 时间积分
- **显式：** 五级低存储Runge-Kutta
- **隐式：** Crank-Nicolson
- **CFL条件：** $\Delta t \leq \text{CFL} \cdot h / (|a|(2N+1))$

#### 边界条件
- **偏滤器靶板：** Bohm鞘层条件
- **上游：** Dirichlet/Neumann
- **对称面：** 周期性/反射

---

### 运行方法

```bash
cd 288_synth_project_Advanced
python main.py
```

程序将自动执行8个阶段的模拟：
1. 物理参数初始化与基本物理量计算
2. 偏滤器几何与SOL网格生成
3. 高阶Legendre有限差分格式构造
4. DG方法平行输运求解
5. 垂直扩散FEM求解
6. 数值格式稳定性分析
7. 等离子体剖面与靶板热负荷诊断
8. 参数扫描、小波分析与不确定性量化

无需任何输入参数，所有物理参数已在代码中硬编码。

---

### 输出文件

程序运行后生成以下文件：

- `simulation_summary.txt`：完整模拟摘要报告
- `divertor_heat_flux.txt`：靶板热负荷分布数据

控制台输出包含所有阶段的详细计算结果和诊断信息。

---

### 关键物理结果解读

#### 典型ITER级参数下的物理量

| 物理量 | 典型值 | 说明 |
|-------|-------|------|
| Coulomb对数 | 13.1 | 弱耦合等离子体 |
| 电子-离子碰撞频率 | 1.5e6 s^-1 | 决定输运时间尺度 |
| Spitzer热传导 | 1.3 W/(m·eV) | 平行方向快速传热 |
| Bohm扩散 | 1.2 m²/s | 垂直反常输运上限 |
| 离子声速 | 9.8e4 m/s | Bohm判据临界速度 |
| 等离子体β | 1e-4 | 低β（磁压主导） |
| Debye长度 | 1.4e-5 m | 鞘层厚度标度 |

#### 靶板热负荷参数依赖性

- 靶板热负荷随上游密度和温度单调增加
- 再循环系数R增大时，靶板温度降低但密度增加
- Monte Carlo不确定性量化给出90%置信区间

---

### 边界处理与数值鲁棒性

本项目在所有关键模块中均实现了边界条件处理和数值鲁棒性保障：

1. **物理量边界约束**
   - Coulomb对数强制 $\geq 2$（弱耦合下限）
   - 碰撞频率设置极小值防止除零
   - 密度/温度强制非负

2. **数值稳定性保障**
   - DG方法自动满足CFL条件
   - 三对角系统使用Thomas算法保证数值稳定
   - 紧致格式隐式部分通过追赶法求解

3. **网格质量检查**
   - 三角形退化检测
   - 边长比监控
   - Jacobian正定性检验

4. **异常处理**
   - NaN/Inf检测与恢复
   - 线性系统条件数监控
   - 参数扫描容错

---

### 创新性总结

本项目的创新点体现在：

1. **多方法融合**：将DG、FEM、紧致差分、小波分析、Bernstein逼近等多种数值方法有机融合在一个统一的等离子体物理框架下
2. **物理-算法深度耦合**：每个数值模块都直接对应特定的等离子体物理过程
3. **多尺度分析**：从Debye长度到系统尺度的多尺度物理建模
4. **不确定性量化**：通过Monte Carlo方法系统评估参数不确定性
5. **可复现性**：所有参数明确指定，小规模实验可在普通计算机上快速复现

---

### 适用场景

- 博士研究生计算等离子体物理课程项目
- 偏滤器物理入门研究
- 数值方法（DG、FEM、紧致差分）在聚变等离子体中的应用演示
- 科学计算代码工程实践范例

---

### 依赖项

- Python 3.7+
- NumPy
- SciPy (仅 `scipy.special.comb`)

无其他外部依赖，可在任何标准Python环境中运行。

---

### 参考文献

1. Hesthaven, J.S. & Warburton, T. "Nodal Discontinuous Galerkin Methods", Springer, 2008
2. Braginskii, S.I. "Transport Processes in a Plasma", Reviews of Plasma Physics, 1965
3. Stangeby, P.C. "The Plasma Boundary of Magnetic Fusion Devices", IOP, 2000
4. LeVeque, R.J. "Finite Difference Methods for Ordinary and Partial Differential Equations", SIAM, 2007
5.ITER Physics Basis, Nuclear Fusion 39 (1999)
