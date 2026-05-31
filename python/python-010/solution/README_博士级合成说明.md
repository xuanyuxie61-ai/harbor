# README_博士级合成说明.md

## 宇宙大尺度结构 N 体模拟 —— ΛCDM 粒子网格演化系统

---

## 一、项目概述

本项目将 **15 个科研代码种子项目** 的核心算法融合重构为一个面向**天体物理：宇宙大尺度结构 N 体模拟**的博士级 Python 科研计算项目。

### 1.1 科学问题

在 **ΛCDM（Lambda Cold Dark Matter）平坦宇宙学**框架下，通过**粒子网格（Particle-Mesh, PM）方法**追踪暗物质粒子从早期宇宙到今天的非线性引力演化，系统研究：

1. **密度扰动的线性增长与非线性坍缩**：利用 Zeldovich 近似生成符合原初功率谱的初始条件，通过 PM 方法求解泊松方程驱动粒子运动。
2. **功率谱 $P(k)$ 的演化**：从模拟密度场估计物质功率谱，验证结构增长理论。
3. **暗物质晕的质量函数 $dn/dM$**：采用 Friends-of-Friends（FOF）与球形过密度（SO）判据识别暗物质晕，并与 Press-Schechter 理论预言比较。
4. **结构形成的统计各向同性**：通过球面随机采样与角距离分布检验宇宙大尺度结构的统计均匀性。

### 1.2 核心物理模型与公式

#### Friedmann 方程（平坦 ΛCDM）

$$
H^2(a) = H_0^2 \left[ \Omega_m a^{-3} + \Omega_r a^{-4} + \Omega_\Lambda \right]
$$

其中 $H_0 = 100h$ km/s/Mpc 为当前 Hubble 常数，$\Omega_m, \Omega_r, \Omega_\Lambda$ 分别为物质、辐射和暗能量的密度参数。

#### 线性增长因子方程

密度扰动 $\delta$ 的线性演化由增长因子 $D(a)$ 描述：

$$
\frac{d^2 D}{da^2} + \frac{1}{2a}\left(3 + \frac{d\ln H}{d\ln a}\right) \frac{dD}{da} - \frac{3\Omega_m(a)}{2a^2} D = 0
$$

#### 泊松方程（共动坐标）

引力势 $\Phi$ 满足：

$$
\nabla^2 \Phi(\mathbf{x}) = 4\pi G a^2 \bar{\rho} \, \delta(\mathbf{x})
$$

在傅里叶空间直接求解：

$$
\Phi_k = -\frac{4\pi G a^2 \bar{\rho} \, \delta_k}{k^2}
$$

#### Zeldovich 近似

粒子从拉格朗日坐标 $\mathbf{q}$ 到欧拉坐标 $\mathbf{x}$ 的位移：

$$
\mathbf{x}(\mathbf{q}, t) = \mathbf{q} + D(t) \, \mathbf{S}(\mathbf{q}), \quad \mathbf{S} = -\nabla \Psi, \quad \nabla^2 \Psi = -\delta
$$

#### Press-Schechter 质量函数

暗物质晕的数量密度：

$$
\frac{dn}{d\ln M} = \sqrt{\frac{2}{\pi}} \frac{\bar{\rho}}{M^2} \frac{\delta_c}{\sigma(M)} \left| \frac{d\ln \sigma}{d\ln M} \right| \exp\left(-\frac{\delta_c^2}{2\sigma^2(M)}\right)
$$

其中 $\delta_c \approx 1.686$ 为线性临界过密度，$\sigma(M)$ 为质量尺度 $M$ 上的密度涨落均方根。

#### Lax-Wendroff 格式

一维对流方程 $\partial \rho / \partial t + c \, \partial \rho / \partial x = 0$ 的二阶离散格式：

$$
\rho_j^{n+1} = \rho_j^n - \frac{c\Delta t}{2\Delta x}(\rho_{j+1}^n - \rho_{j-1}^n) + \frac{c^2 \Delta t^2}{2\Delta x^2}(\rho_{j+1}^n - 2\rho_j^n + \rho_{j-1}^n)
$$

稳定性条件（CFL）：$|c|\Delta t / \Delta x \leq 1$。

---

## 二、种子项目映射与融合方式

| 编号 | 原始种子项目 | 核心算法 | 合成后融入位置 | 科学角色 |
|:---:|---|---|---|---|
| 1 | 431_filum | 文件字符统计、行计数、文件名解析、字符串处理 | `utils.py` | 模拟日志文件的元数据管理与文本解析 |
| 2 | 806_nonlin_bisect | 非线性方程二分法求根 | `cosmology.py` | 求解特定红移对应的宇宙年龄、过密度阈值等隐式方程 |
| 3 | 1029_rk12 | Runge-Kutta 1-2 阶显式积分（含误差估计） | `cosmology.py`, `nbody_integrator.py` | 尺度因子背景演化与粒子轨道自适应时间积分 |
| 4 | 355_fd1d_advection_lax_wendroff | 一维对流方程 Lax-Wendroff 有限差分 | `density_field.py` | 密度场数值平流演化与质量守恒性检验 |
| 5 | 348_fair_dice_simulation | 离散概率分布逆变换采样（CDF 法） | `statistics.py` | 离散模式的随机采样与概率质量函数检验 |
| 6 | 805_nintlib | 多维 Monte Carlo 数值积分 | `power_spectrum.py` | 高阶统计量（相关函数、速度分布矩）的多维积分计算 |
| 7 | 525_hermite_rule | Gauss-Hermite 求积规则生成 | `initial_conditions.py` | 高斯速度分布函数的精确矩计算与节点生成 |
| 8 | 651_latin_edge | Latin 超立方 edge 采样 | `initial_conditions.py` | 宇宙学参数空间的均匀探索与初始条件正则化 |
| 9 | 137_casino_simulation | 乘法随机过程演化 | `statistics.py` | 密度扰动非线性 regime 的粗粒化随机演化模型 |
| 10 | 702_logistic_ode | Logistic 常微分方程参数管理 | `cosmology.py` | 宇宙学参数持久化与背景演化模型参数封装 |
| 11 | 024_asa005 | 标准正态分布累积密度函数（AS 66） | `statistics.py` | 高斯随机场阈值穿越概率与密度峰值统计 |
| 12 | 978_r8crs | 稀疏矩阵 CRS 格式与矩阵-向量乘法 | `linalg_utils.py` | Poisson 方程稀疏离散算子的存储与运算 |
| 13 | 1125_sphere_positive_distance | 单位球面正象限随机采样 | `halo_finder.py` | 天球坐标系中随机视线方向生成与角向分布检验 |
| 14 | 981_r8ge | 稠密矩阵 PLU 分解（LINPACK 风格） | `linalg_utils.py` | 小尺度团块内部动力学矩阵的直接求解 |
| 15 | 667_levels | 等值线/水平集随机采样 | `halo_finder.py` | 密度场水平集体积分析（结节、纤维、空腔分类） |

---

## 三、项目文件结构

```
010_synth_project/
├── main.py                          # 统一入口，零参数可运行
├── cosmology.py                     # ΛCDM 宇宙学背景演化
├── initial_conditions.py            # Zeldovich 初始条件生成
├── pm_solver.py                     # 粒子网格引力求解器（CIC + FFT）
├── nbody_integrator.py              # N 体 Leapfrog / RK12 积分器
├── density_field.py                 # Lax-Wendroff 密度场对流求解
├── power_spectrum.py                # 功率谱、相关函数、质量函数
├── halo_finder.py                   # FOF / SO 暗物质晕识别
├── statistics.py                    # 统计工具（正态 CDF、离散采样、随机行走）
├── linalg_utils.py                  # 稠密 LU、稀疏 CRS、Thomas 算法
├── utils.py                         # 文件 I/O 与文本工具
└── README_博士级合成说明.md         # 本文档
```

---

## 四、各模块核心改造说明

### 4.1 `cosmology.py`

- **融合 rk12**：将 Runge-Kutta 1-2 阶自适应积分器用于尺度因子 $a(t)$ 和线性增长因子 $D(a)$ 的演化，实现局部截断误差估计。
- **融合 nonlin_bisect**：将二分法求根用于求解特定红移对应的宇宙年龄，以及暗物质晕 virial 半径的隐式方程。
- **融合 logistic_ode**：借鉴参数持久化思想，封装 Planck 2018 基准宇宙学参数，提供参数更新与一致性校验。
- **新增公式**：Friedmann 方程、共动距离积分、临界过密度 $\delta_c(z)$、物质密度参数演化 $\Omega_m(a)$。

### 4.2 `initial_conditions.py`

- **融合 latin_edge**：Latin 超立方采样用于生成均匀分布的宇宙学参数探索点，验证初始条件的空间均匀性。
- **融合 hermite_rule**：通过 Jacobi 矩阵特征值分解生成 Gauss-Hermite 节点与权重，用于验证速度分布的一维高斯矩。
- **融合 fair_dice_simulation / alnorm**：离散模式采样与正态分布概率计算用于高斯随机场构造。
- **新增公式**：Eisenstein-Hu transfer function、原初功率谱 $P(k) = A_s k^{n_s} T^2(k)$、Zeldovich 位移场 $S = -\nabla \Psi$、top-hat 窗函数 $W(kR)$。

### 4.3 `pm_solver.py`

- **融合 r8crs / r8ge**：稀疏矩阵 CRS 格式与稠密 PLU 分解被封装用于一维泊松方程的直接法求解（作为 FFT 解法的数值验证）。
- **新增公式**：三维离散 Poisson 七点 stencil、CIC（Cloud-in-Cell）质量分配权重公式 $W = \prod_i (1 - |x_{p,i} - x_{g,i}|/dx)$。

### 4.4 `nbody_integrator.py`

- **融合 rk12**：RK12 自适应步长用于粒子运动方程的误差控制，与 Leapfrog 辛积分器形成对比。
- **新增公式**：共动坐标运动方程 $d^2\mathbf{x}/dt^2 + 2H(a)d\mathbf{x}/dt = -a^{-2}\nabla \Phi$、时间步长限制 $\Delta t < \eta \sqrt{\varepsilon / |g|}$、Leapfrog 能量守恒误差 $|\Delta E/E| \sim O(\Delta t^2)$。

### 4.5 `density_field.py`

- **融合 fd1d_advection_lax_wendroff**：完整复现一维 Lax-Wendroff 算法，并推广至三维 Strang 分裂格式。
- **新增公式**：多维分量分裂 $L_x(\Delta t/2) L_y(\Delta t/2) L_z(\Delta t) L_y(\Delta t/2) L_x(\Delta t/2)$、质量守恒检验 $M = \sum \rho_j \Delta x$。

### 4.6 `power_spectrum.py`

- **融合 nintlib / monte_carlo_nd**：Monte Carlo 多维积分用于计算相关函数与速度分布的高阶矩，提供标准误差估计。
- **新增公式**：功率谱定义 $P(k) = V \langle |\delta_k|^2 \rangle$、Wiener-Khinchin 定理 $\xi(r) = (1/2\pi^2) \int k^2 P(k) \sin(kr)/(kr) dk$、Press-Schechter 质量函数、质量方差 $\sigma^2(R) = (1/2\pi^2) \int k^2 P(k) W^2(kR) dk$。

### 4.7 `halo_finder.py`

- **融合 levels**：密度场的水平集分析，计算不同阈值下的等值面所包围体积分数，用于宇宙 Web 结构（结节、纤维、面、空腔）分类。
- **融合 sphere_positive_distance**：单位球面正象限随机采样用于生成随机视线方向，计算角距离分布以检验各向同性。
- **新增公式**：FOF 连接长度 $b L / N^{1/3}$、球形过密度判据 $\bar{\rho}(<R_\Delta) = \Delta \rho_{crit}$、晕质量 $M_\Delta = (4\pi/3) R_\Delta^3 \Delta \rho_{crit}$、质量函数 $dn/d\ln M$。

### 4.8 `statistics.py`

- **融合 asa005 (alnorm)**：Hill (1973) AS 66 算法的有理函数近似用于标准正态累积分布函数，支持尾部指数衰减近似。
- **融合 fair_dice_simulation**：逆变换法离散采样，通过累积分布函数 $F_i$ 与均匀随机数 $u$ 的匹配实现。
- **融合 casino_simulation**：乘法随机行走模型 $x_{n+1} = f_{win/loss} \cdot x_n$，用于检验长时间统计行为的稳定性。
- **新增公式**：top-hat 窗函数 $W(kR) = 3[\sin(kR) - kR\cos(kR)]/(kR)^3$、高斯随机场方差-功率谱关系。

### 4.9 `linalg_utils.py`

- **融合 r8ge**：完整复现 LINPACK 风格的 PLU 分解（选主元、行交换、乘子计算、行消去），前向代入与回代求解。
- **融合 r8crs**：压缩行存储（CRS）稀疏矩阵格式，实现稀疏矩阵-向量乘法 $y_i = \sum_{k=row[i]}^{row[i+1]-1} val[k] \cdot x_{col[k]}$。
- **新增公式**：一维/三维离散 Laplace 算子构造、Thomas 算法（三对角方程组 $O(n)$ 直接求解）。

### 4.10 `utils.py`

- **融合 filum**：文件行计数、字符统计、扩展名提取与替换、字符大小写转换等基础 I/O 工具，用于模拟日志与快照文件管理。

---

## 五、运行方式

### 环境要求

- Python 3.8+
- NumPy

### 运行命令

```bash
cd Synthesis-project-python/010_synth_project
python main.py
```

无需任何命令行参数。程序将自动执行：

1. ΛCDM 宇宙学参数初始化与背景演化计算
2. Zeldovich 初始条件生成
3. PM N 体演化（Leapfrog 积分）
4. Lax-Wendroff 密度场对流与质量守恒检验
5. 功率谱与相关函数估计
6. 暗物质晕识别（FOF + 球形过密度）
7. 质量函数统计与 Press-Schechter 理论比较
8. 水平集分析与各向同性检验
9. Monte Carlo 积分与统计验证
10. 线性代数数值精度检验

### 预期输出

运行时间约 **1-2 秒**，终端输出包含各物理量的数值结果、统计检验通过信息以及总耗时。

---

## 六、数值鲁棒性与边界处理

本项目在多处实施了严格的数值鲁棒性设计：

1. **宇宙学参数校验**：`Cosmology.__init__` 中检查平坦性、$h$ 与 $\Omega_m$ 的物理合理范围。
2. **除零保护**：泊松方程求解中 $k^2$ 的零模被显式排除；Thomas 算法中主元小于 $10^{-15}$ 时抛出异常。
3. **CFL 条件检查**：`density_field.py` 中 Lax-Wendroff 步前强制检验 $|c|\Delta t / \Delta x \leq 1$。
4. **周期性边界处理**：所有粒子位置、密度场卷积、CIC 分配均采用模运算保证 $x \in [0, L)$。
5. **能量守恒监测**：Leapfrog 为二阶辛积分器，长时间演化中能量误差有界。
6. **稀疏矩阵合法性**：CRS 格式构造时验证 `row_ptr` 的单调性与首尾一致性。

---

## 七、科学难度评估

本项目涉及的前沿科学计算难点包括：

- **非线性引力 N 体演化**：PM 方法结合 CIC 分配与 FFT 泊松求解，涉及 $O(N \log N)$ 复杂度的三维卷积。
- **原初功率谱与结构增长**：Eisenstein-Hu transfer function、线性增长因子二阶 ODE、Zeldovich 近似。
- **晕识别与质量函数**：FOF 连通分量搜索、球形过密度隐式求解、Press-Schechter 理论比较。
- **高阶数值方法**：RK12 自适应步长、Lax-Wendroff + Strang 分裂、Gauss-Hermite 正交求积、Monte Carlo 多维积分误差分析。
- **大规模稀疏/稠密线性代数**：PLU 分解、CRS 格式、Thomas 算法。

整体问题深度达到**天体物理博士计算课程**水平，涵盖宇宙学、计算天体物理、数值分析、统计物理等多个交叉领域。
