# 煤粉燃烧 NOx 生成多尺度耦合模拟系统

## 项目概述

本项目将 **15 个种子科研代码项目** 的核心算法融合重构，面向**燃烧科学：煤粉燃烧 NOx 生成**这一前沿博士级科学问题，构建了一个多尺度耦合的自然科学计算框架。

项目包含 **12 个 Python 模块文件** 与 **1 个统一入口 main.py**，零参数可直接运行，涵盖从煤粉颗粒几何、内部扩散反应、详细 NOx 化学动力学、一维燃烧器有限元求解、颗粒群空间分布、参数空间优化到系统刚性诊断与统计后处理的完整科学计算流程。

---

## 一、原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 合成后科学角色 |
|:---:|---|---|---|
| 1 | `1282_tortoise` | 边界词编码几何、瓦片索引映射 | **particle_geometry.py**：煤粉颗粒轮廓的离散方向编码与形状分类 |
| 2 | `093_bird_egg` | 通用蛋形参数化公式 | **particle_geometry.py**：煤粉颗粒的非球形参数化建模（蛋形/梨形） |
| 3 | `083_bezier_surface` | 双三次贝塞尔曲面求值 | **particle_geometry.py**：颗粒表面贝塞尔控制网参数化 |
| 4 | `1167_stla_to_tri_surface` | 表面三角化与面积计算 | **particle_geometry.py**：颗粒表面三角网格生成与比表面积计算 |
| 5 | `080_besselj_zero` | 贝塞尔函数零点 Newton 迭代 | **particle_diffusion.py**：球形颗粒内部扩散问题的特征值分析 |
| 6 | `818_normal_ode` | 正态分布 ODE 精确解验证 | **reaction_kinetics.py**：数值积分器的精度验证基准 |
| 7 | `826_ode_euler_backward` | 后向 Euler 隐式格式 | **reaction_kinetics.py**：刚性化学反应系统的隐式时间积分 |
| 8 | `389_fem1d_function_10_display` | 一维 P1 有限元 | **reactor_fem1d.py**：一维燃烧器温度/浓度场 FEM 求解 |
| 9 | `592_interp_equal` | 牛顿差商插值、Horner 求值 | **thermophysical_props.py**：温度依赖热力学物性的高阶插值 |
| 10 | `249_cvt_3d_lumping` | Lloyd CVT 质心 Voronoi 剖分 | **particle_population.py**：颗粒群空间分布与聚并模型 |
| 11 | `537_hilbert_curve_display` | Hilbert 空间填充曲线 | **parameter_search.py**：多维参数空间高效遍历采样 |
| 12 | `631_l4lib` | Park-Miller LCG 伪随机数 | **parameter_search.py**：随机参数采样与燃烧工况搜索 |
| 13 | `610_jordan_matrix` | 随机 Jordan 标准型生成 | **stiffness_analysis.py**：燃烧 Jacobian 的 Jordan 结构刚度诊断 |
| 14 | `118_brc_naive` | 分组聚合统计 | **data_statistics.py**：多工况燃烧模拟结果的分组后处理 |
| 15 | `502_hand_data` | 离散轮廓数据拟合 | **data_statistics.py**：响应面多项式拟合与不确定性量化 |

---

## 二、新增数学物理模型与核心公式

### 2.1 颗粒几何模型（蛋形 + Bezier + 三角化）

**通用蛋形轮廓（Narushin 公式）**：

$$
y(x) = \frac{B}{2}\sqrt{\frac{L^2 - 4x^2}{L^2 + 8wx + 4w^2}}
$$

其中 $L$ 为颗粒长度，$B$ 为最大宽度，$w$ 为最大宽度偏移。当 $w=0$ 时退化为椭圆。

**双三次贝塞尔曲面求值**：

$$
\mathbf{X}(u,v) = \sum_{i=0}^{3}\sum_{j=0}^{3} B_{i,3}(u) \, B_{j,3}(v) \, \mathbf{P}_{ij}
$$

Bernstein 基函数：

$$
B_{k,3}(t) = \binom{3}{k} t^k (1-t)^{3-k}
$$

**表面面积 via 三角化**：

$$
A = \sum_{\triangle} \frac{1}{2} \| (\mathbf{v}_2 - \mathbf{v}_1) \times (\mathbf{v}_3 - \mathbf{v}_1) \|
$$

**球形度**：

$$
\Psi = \frac{A_{\text{sphere}}(V)}{A_{\text{particle}}} \le 1
$$

### 2.2 颗粒内部扩散–反应耦合（Bessel 特征值方法）

球形坐标下的准稳态扩散–反应方程：

$$
\frac{1}{r^2}\frac{d}{dr}\left(r^2 \frac{dC}{dr}\right) = \frac{k}{D_{\text{eff}}} C = \frac{\phi^2}{R_p^2} C
$$

**Thiele 模数**：

$$
\phi = R_p \sqrt{\frac{k}{D_{\text{eff}}}}
$$

**一级反应球形颗粒有效因子**：

$$
\eta = \frac{3}{\phi}\left(\frac{1}{\tanh\phi} - \frac{1}{\phi}\right)
$$

**球形 Bessel 函数** $j_0(x) = \sin x / x$ 的零点 $\lambda_{0,k}$ 为扩散算子的特征值，瞬态解可展开为：

$$
C(r,t) = \sum_{k=1}^{\infty} a_k \exp\!\left(-\frac{D_{\text{eff}}\lambda_{0,k}^2 t}{R_p^2}\right) j_0\!\left(\frac{\lambda_{0,k} r}{R_p}\right)
$$

**有效扩散系数**（含 Knudsen 扩散）：

$$
D_{\text{eff}} = \frac{\varepsilon}{\tau} \left(\frac{1}{D_{\text{bulk}}} + \frac{1}{D_K}\right)^{-1}, \quad D_K = \frac{2}{3}r_{\text{pore}}\sqrt{\frac{8RT}{\pi M}}
$$

### 2.3 详细 NOx 反应机理（Arrhenius 动力学）

包含 15 条基元反应，覆盖三大 NOx 生成路径：

1. **Thermal NOx**（扩展 Zeldovich）：
   $$\text{N}_2 + \text{O} \rightleftharpoons \text{NO} + \text{N}$$
   $$\text{N} + \text{O}_2 \rightleftharpoons \text{NO} + \text{O}$$

2. **Fuel NOx**（燃料氮氧化）：
   $$\text{HCN} + \text{O} \rightarrow \text{NO} + \text{CH}$$
   $$\text{NH}_3 + \text{O} \rightarrow \text{NO} + \text{H}_2$$

3. **Prompt NOx**（Fenimore）：
   $$\text{CH} + \text{N}_2 \rightarrow \text{HCN} + \text{N}$$

4. **NO Reburn / 热分解**（限制非物理积累）：
   $$\text{NO} + \text{CH} \rightarrow \text{HCN} + \text{O}$$
   $$2\text{NO} \rightarrow \text{N}_2 + \text{O}_2$$

**Arrhenius 速率定律**：

$$
k_r(T) = A_r T^{b_r} \exp\!\left(-\frac{E_r}{RT}\right)
$$

**物种质量生成速率**：

$$
\omega_i^{\text{mass}} = M_i \sum_{r} (\nu_{i,r}^{\text{prod}} - \nu_{i,r}^{\text{reac}}) \, k_r(T) \prod_{j}[X_j]^{\alpha_{j,r}}
$$

### 2.4 刚性 ODE 时间积分（后向 Euler + Newton 迭代）

Batch reactor 控制方程：

$$
\frac{dY_i}{dt} = \frac{1}{\rho}\omega_i(Y,T), \quad i=1,\dots,N_{\text{spec}}
$$

**后向 Euler 隐式格式**（A-稳定）：

$$
\mathbf{Y}^{n+1} = \mathbf{Y}^n + \Delta t \, \mathbf{f}(\mathbf{Y}^{n+1})
$$

**阻尼 Newton 迭代**：

$$
(\mathbf{I} - \Delta t \mathbf{J})\Delta\mathbf{Y} = -(\mathbf{Y} - \mathbf{Y}^n - \Delta t \mathbf{f}(\mathbf{Y}))
$$

其中 Jacobian $\mathbf{J} = \partial\mathbf{f}/\partial\mathbf{Y}$ 通过有限差分计算。时间步长采用步长加倍法自适应控制。

### 2.5 一维燃烧器有限元求解（SUPG 稳定化）

**能量方程**：

$$
\rho u c_p \frac{dT}{dx} = \frac{d}{dx}\!\left(k\frac{dT}{dx}\right) + Q_{\text{comb}}(x) - Q_{\text{rad}}(x)
$$

**NO 质量分数方程**：

$$
\rho u \frac{dY_{\text{NO}}}{dx} = \frac{d}{dx}\!\left(\rho D_{\text{NO}}\frac{dY_{\text{NO}}}{dx}\right) + S_{\text{NO}}(x)
$$

**Peclet 数与 SUPG 稳定化参数**：

$$
\text{Pe} = \frac{\rho u h}{2k}, \quad \tau = \frac{h}{2|\rho u|}\left(\coth\text{Pe} - \frac{1}{\text{Pe}}\right)
$$

**P1 线性基函数**：

$$
\phi_j(x) = \frac{x - x_{j-1}}{h_{j-1}} \;\text{on}\; [x_{j-1},x_j], \quad \phi_j(x) = \frac{x_{j+1} - x}{h_j} \;\text{on}\; [x_j,x_{j+1}]
$$

### 2.6 颗粒群 CVT 空间分布（Lloyd 算法 + 质量集中）

**CVT 能量泛函**：

$$
\mathcal{E}(\{g_i\}) = \sum_{i=1}^{N}\int_{V_i} \rho(\mathbf{x}) \|\mathbf{x} - g_i\|^2 \, d\mathbf{x}
$$

**密度加权质心更新（Lumping）**：

$$
g_i^{\text{new}} = \frac{\sum_{s \in V_i} \rho(s) \, s}{\sum_{s \in V_i} \rho(s)}
$$

**燃烧器密度分布**（模拟中心射流 + 外环回流）：

$$
\rho(r) = \rho_{\text{jet}} \exp\!\left(-\frac{r^2}{0.09}\right) + \rho_{\text{recirc}} \exp\!\left(-\frac{(r-0.6)^2}{0.04}\right)
$$

### 2.7 Smoluchowski 聚并方程

离散形式：

$$
\frac{dN_k}{dt} = \frac{1}{2}\sum_{i+j=k}\beta_{ij}N_i N_j - N_k\sum_j\beta_{kj}N_j
$$

**Saffman-Turner 湍流聚并核**：

$$
\beta_{ij} = C_{\text{agg}}(d_i + d_j)^3 \sqrt{T}
$$

### 2.8 多维参数空间采样

**Hilbert 空间填充曲线**（2D 精确映射）：

$$
(rx, ry) \mapsto \text{quadrant rotation/flip} \Rightarrow (x, y) \in [0, 2^m-1]^2
$$

**Park-Miller LCG**：

$$
\text{seed}_{n+1} = 16807 \cdot \text{seed}_n \pmod{2^{31}-1}
$$

### 2.9 Jordan 刚度诊断

**刚度比**：

$$
S = \frac{|\text{Re}(\lambda_{\text{fast}})|}{|\text{Re}(\lambda_{\text{slow}})|}
$$

对于燃烧系统，$S \sim 10^{20}$，必须使用隐式方法。

**Jordan 块结构**：

$$
J_k(\lambda) = \lambda I + N, \quad N_{i,i+1}=1
$$

### 2.10 响应面拟合与不确定性量化

**2D 多项式响应面**（Vandermonde + 岭回归）：

$$
R(x_1,x_2) = \sum_{i=0}^{p}\sum_{j=0}^{p-i} c_{ij}\, x_1^i x_2^j
$$

**系数求解**：

$$
\mathbf{c} = (\boldsymbol{\Phi}^T\boldsymbol{\Phi} + \lambda\mathbf{I})^{-1}\boldsymbol{\Phi}^T\mathbf{y}
$$

---

## 三、合成项目文件结构

```
158_synth_project/
├── main.py                     # 统一入口，零参数运行
├── reaction_mechanism.py       # 15 反应详细 NOx 机理数据库
├── reaction_kinetics.py        # 刚性 ODE 求解器（后向 Euler）
├── reactor_fem1d.py            # 1D 燃烧器 SUPG-FEM 求解
├── particle_geometry.py        # 煤粉颗粒几何建模（蛋形/Bezier/三角化）
├── particle_diffusion.py       # 颗粒内部扩散-反应（Bessel/Thiele）
├── thermophysical_props.py     # 热力学物性牛顿插值
├── particle_population.py      # 颗粒群 CVT 分布与 Smoluchowski 聚并
├── parameter_search.py         # Hilbert 曲线 + LCG 参数优化
├── stiffness_analysis.py       # Jordan 矩阵刚度诊断
├── data_statistics.py          # 分组统计、响应面、UQ
└── utils.py                    # 通用数值工具（Newton、Gauss-Legendre、Thomas 算法）
```

---

## 四、合成后的项目能够解决什么科学问题

1. **颗粒尺度**：计算非球形煤粉颗粒（蛋形/梨形）的表面积、体积、球形度，评估形状对燃烧速率的影响。
2. **孔尺度**：通过 Thiele 模数与有效因子，定量判断 char 氧化和 fuel-N 释放受动力学控制还是扩散控制。
3. **反应动力学尺度**：求解包含 16 种物种、15 条反应的详细 NOx 化学机理，分离 thermal/fuel/prompt/reburn 四条路径的贡献。
4. **反应器尺度**：在一维燃烧器内耦合求解温度场和 NO 浓度场，评估出口 NOx 排放水平。
5. **群体尺度**：通过 CVT 和 Smoluchowski 方程模拟颗粒群的空间分布与聚并演化。
6. **优化尺度**：在过量空气比、粒径、峰值温度、停留时间的四维参数空间中搜索最低 NOx 排放工况。
7. **数值诊断**：通过 Jacobian 特征值分析和 Jordan 结构，定量诊断燃烧系统的刚性程度并推荐积分方法。
8. **统计后处理**：对大量模拟工况进行分组统计、响应面拟合和 Monte Carlo 不确定性量化。

---

## 五、如何运行

### 环境要求
- Python 3.8+
- NumPy

### 运行方式
```bash
cd Synthesis-project-python/158_synth_project
python main.py
```

无需任何命令行参数。程序将依次执行以下 8 个科学计算步骤并输出结果：

1. 颗粒几何建模（两种形状）
2. 颗粒内部扩散–反应分析
3. Batch Reactor NOx 动力学模拟
4. 1D 燃烧器 FEM 温度/NO 场求解
5. 颗粒群 CVT 分布与聚并
6. 燃烧参数空间优化
7. 系统刚度分析
8. 统计后处理与不确定性量化

典型运行时间：**< 3 秒**（普通桌面 CPU）。

---

## 六、关键数值结果示例

```
Batch Reactor:
  Final NO  = 179.5 ppm
  Final NO2 = 101.2 ppm
  Pathway: thermal = +6.30e-04, fuel = +6.30e-04 kg/(m^3·s)

1D Burner (L=5 m):
  Peak temperature = 1899 K
  Max NO = 562 ppm
  Outlet NO = 281 ppm

Stiffness at 1800 K:
  Stiffness ratio = 3.0e20
  Fastest time scale = 2.6e-12 s
  Recommended method: implicit (backward Euler)
```

---

## 七、设计原则与鲁棒性保障

- **边界处理**：所有温度、浓度、质量分数均经过物理范围裁剪（如 $T \in [200, 5000]$ K，$Y_i \in [0, 1]$）。
- **数值稳定性**：FEM 采用 SUPG 稳定化防止高 Peclet 数下的数值振荡；ODE 积分采用 A-稳定的后向 Euler 与自适应步长；矩阵求解失败时自动回退到最小二乘解。
- **溢出防护**：指数函数 `safe_exp` 限制在 `[-700, 700]`；Bessel 函数 `j_0` 在零点邻域使用泰勒展开避免除零。
- **无可视化**：已删除所有绘图、图像生成代码，纯数值计算输出。
