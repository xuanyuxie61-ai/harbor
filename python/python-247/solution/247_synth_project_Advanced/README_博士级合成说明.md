# PROJECT_247 · 暗物质晕形成与并合树构建
# —— 高阶有限差分与稳定性分析（小规模可复现实验）

## 一、项目概述

本项目是面向**计算天体物理**前沿问题的博士级科研代码合成项目。研究的核心科学问题是：

> 在一个小尺度、完全可复现的数值实验中，如何：
>   1. 构建暗物质晕的谱表示引力势；
>   2. 从并合事件的组合枚举与最优路径规划中重建 halo 并合树；
>   3. 用高阶有限差分格式求解晕内对流、相空间输运与密度振荡；
>   4. 对这些格式进行严格的 von Neumann 稳定性分析。

所有子模块均围绕"暗物质 halo"这一物理对象展开，变量命名、公式推导、代码组织均与该领域深度耦合，**不是通用数值方法的简单换皮**。

---

## 二、输入种子项目 → 科学问题映射

共使用 **15 个种子项目**的核心算法，每一个都承担真实的物理角色：

| 序号 | 种子项目 | 核心算法 | 在本项目中的物理角色 |
| ---: | :--- | :--- | :--- |
| 1 | 1034_tsengshao (对流涡旋) | 涡量-流函数 + 5 阶迎风格式 | 热晕气体中对流驱动的次级环流，导致角动量向内输运（§9 `convection_vortex_halo.py`） |
| 2 | 202_combo (Balanced sequences) | Catalan 数、Dyck 词 rank/unrank | 用平衡括号序列枚举并合树拓扑（§4 `merger_tree_enumerator.py`） |
| 3 | 462_gegenbauer_polynomial | Elhay-Kautsky Gauss-Gegenbauer 求积 | NFW 密度剖面的 Gegenbauer 谱展开（§2 `halo_gegenbauer_potential.py`） |
| 4 | 1356_trig_interp | 三角 Lagrange 插值 (Austin–Trefethen 2017) | halo 内准周期轨道的高精度重采样（§3 `trigonometric_orbit_interpolator.py`） |
| 5 | 1018_twiecki_bg_inhib (Pool + retry) | 并行分析池 + 指数退避重试 | 对多棵并合树进行鲁棒的并行分析（§11 `analysis_pool.py`） |
| 6 | 1367_tsp_random | TSP 随机采样 + 2-opt 局部优化 | 在 (log M, z) 空间中寻找最可能的并合顺序（§5 `minimal_merger_path.py`） |
| 7 | 1072_yogurt-ybn (拓扑重构) | 策略 1：化简；策略 2：韧性 | 并合树化简（去掉亚阈值分支）+ 主枝韧性指数（§4） |
| 8 | 456_gaussian_prime_spiral | Gauss 素数螺旋轨迹 | 确定性、非周期、可复现的子 halo 初始位置播种（§1 `gauss_prime_seeding.py`） |
| 9 | 1128_cadet_RDM (SMB 四区) | 平衡-色散模型 + 端口切换 | 冷流 / 维里气体 / 循环风 / 空洞 四区吸积环（§8 `multizone_accretion.py`） |
| 10 | 495_gyroscope_ode | 陀螺仪 6D 欧拉方程 | 三轴 halo 在外潮汐 torque 下的进动（§7 `halo_ode_systems.py`） |
| 11 | 838_oregonator_ode | Oregonator 化学振荡器 | 重释为晕核密度-气体-恒星形成极限环振荡（§7） |
| 12 | 831_ode_trapezoidal | 隐式梯形 ODE 积分器 | 用于上述两个 ODE 系统的 A-稳定积分（§7） |
| 13 | 1012_pkmtum_energy_coarse_graining | 正规化流仿射 + softmax 双射 | 6D 相空间分布的粗粒化，并度量 Liouville 体积偏离（§6 `phase_space_flow_coarse_grain.py`） |
| 14 | 200_collocation (Horner) | Horner 多项式求值 | 引力势的数值稳定 Horner 求值（§2） |
| 15 | 949_quadrature_least_squares | 最小二乘求积权重 | 从含噪样本估计 halo 质量积分（§10 `least_squares_quadrature.py`） |

额外模块 `stability_von_neumann.py` 对 5 阶迎风格式、平衡-色散格式、梯形积分器做严格的 **von Neumann 稳定性分析**（§12），给出放大因子 |G(θ)| 与最大稳定 CFL。

---

## 三、核心数学物理公式

### 3.1 NFW 密度剖面与 Gegenbauer 展开

NFW 密度对比：

$$
\Delta(r) = \frac{\rho(r)}{\rho_s} = \frac{1}{(r/r_s)(1 + r/r_s)^2}
$$

在 $x \in [-1, 1]$（对应 $r \in [r_s, R_{\rm vir}]$）上以 Gegenbauer 多项式展开：

$$
u(x) = (1 - x^2)^{\alpha + 1/2} \, \Delta(R_{\rm vir} x)
      = \sum_{k=0}^{N} a_k \, C_k^{(\alpha)}(x)
$$

展开系数通过 Elhay–Kautsky 方法（对称三对角 Jacobi 矩阵的 Golub–Welsch 对角化）获得 Gauss–Gegenbauer 节点与权重：

$$
\mu_0 = \frac{2^{2\alpha+1}\,\Gamma(\alpha+1)^2}{\Gamma(2\alpha+2)}, \qquad
b_i^2 = \frac{4 i (\alpha+i)^2 (2\alpha+i)}{(2\alpha+2i-1)(2\alpha+2i+1)(2\alpha+2i)^2}
$$

### 3.2 三角 Lagrange 插值（Austin–Trefethen 2017）

给定非均匀节点 $x_j$ 与函数值 $y_j$，周期插值基函数：

$$
L_j(x) = \prod_{k \neq j} \frac{\sin\frac{x - x_k}{2}}{\sin\frac{x_j - x_k}{2}}
$$

其导数由对数导数公式

$$
\frac{d}{dx} L_j(x) = L_j(x) \sum_{m \neq j} \tfrac{1}{2} \cot\!\left(\frac{x - x_m}{2}\right)
$$

给出，可解析地得到轨道速度 $dr/dt$、$d\varphi/dt$。

### 3.3 并合树 = 平衡括号序列

节点数为 $N$ 的并合树与长度为 $2N$ 的 Dyck 词一一对应，计数为 Catalan 数：

$$
C_N = \frac{1}{N+1}\binom{2N}{N}
$$

通过 ballot 数递推实现 rank / unrank：

$$
B(a, b) = \binom{a+b}{a} - \binom{a+b}{a+1}
$$

### 3.4 并合路径 TSP 度量

 progenitor  $i \leftrightarrow j$ 距离：

$$
d(i, j) = \sqrt{\left(\frac{\Delta \log_{10} M}{\sigma_M}\right)^2
              + \left(\frac{\Delta z}{\sigma_z}\right)^2}
$$

以 Burkardt 的 `tsp_random` 随机采样启发式 + 2-opt 局部优化求近似最优环游。

### 3.5 相空间粗粒化流

正规化流双射 $T: (x,v) \mapsto (y,w)$ 由仿射双射与 softmax 行随机双射复合而成：

$$
A_{ij} = \frac{\exp(A'_{ij})}{\sum_k \exp(A'_{ik})}, \qquad
y = A x + b
$$

变化变量公式：

$$
p_Y(y) = p_X(T^{-1}(y)) \, \bigl|\det \tfrac{dT^{-1}}{dy}\bigr|
$$

精确 Vlasov 动力学下 Liouville 定理要求 $|\det J| = 1$；偏离量 $\bigl||\det J| - 1\bigr|$ 即为粗粒化信息损失的度量。

### 3.6 Oregonator 型晕核密度振荡

$$
\eta_1 \dot u = q v - u v + u(1 - u), \qquad
\eta_2 \dot v = -q v - u v + f w, \qquad
\dot w = u - w
$$

$(u, v, w)$ 分别对应晕核密度对比、气体分数、恒星形成库。

### 3.7 三轴 halo 进动（陀螺仪）

$$
\dot\psi  = \frac{\omega_1 \sin\phi + \omega_2 \cos\phi}{\sin\theta}, \qquad
\dot\theta = \omega_1 \cos\phi - \omega_2 \sin\phi, \qquad
\dot\varphi = \omega_3 - \cos\theta\,\dot\psi
$$

$$
A_1 \dot\omega_1 = (A_2 - A_3)\omega_2 \omega_3 + M_1, \qquad
M_1 = -m_{\rm tidal} A_1 \sin\theta \cos\phi, \;\ldots
$$

### 3.8 隐式梯形积分

$$
y_{n+1} = y_n + \frac{h}{2}\bigl[f(t_n, y_n) + f(t_{n+1}, y_{n+1})\bigr]
$$

每步通过带松弛的不动点迭代求解，对刚性 halo 振荡问题是 A-稳定的。

### 3.9 四区吸积环（平衡-色散模型）

$$
\frac{\partial c}{\partial t} + u \frac{\partial c}{\partial z}
  = D \frac{\partial^2 c}{\partial z^2}
    - \frac{1-\varepsilon}{\varepsilon} \frac{\partial q}{\partial t},
\qquad
\frac{\partial q}{\partial t} = k (c - q)
$$

显式迎风格式 + 中心差分扩散，带 CFL 自适应限幅。

### 3.10 5 阶迎风格式

对流项 $\partial_x f$ 的向后偏置 5 阶 stencil（$u > 0$ 时）：

$$
f'_i = \frac{2 f_{i-3} - 15 f_{i-2} + 60 f_{i-1} - 20 f_i - 30 f_{i+1} + 3 f_{i+2}}{60 \Delta x}
$$

### 3.11 最小二乘求积

给定节点 $x_i$，求积权重由法方程

$$
V^\top V \, R = V^\top, \qquad W = Q^\top R
$$

给出；其中 $V_{ik} = x_i^k$，$Q_k = \int_a^b x^k dx = (b^{k+1} - a^{k+1})/(k+1)$。

### 3.12 von Neumann 放大因子

- 5 阶迎风格式 + 显式 Euler：$G(\theta) = 1 - \nu \sum_s c_s e^{i s \theta}$
- 平衡-色散格式：$G(\theta) = 1 - \mathrm{CFL}_u (1 - e^{-i\theta}) - 2 \mathrm{CFL}_D (1 - \cos\theta)$
- 隐式梯形：$G = (1 + \tfrac{1}{2}h\lambda)/(1 - \tfrac{1}{2}h\lambda)$，对所有 $\mathrm{Re}\,\lambda < 0$ 均有 $|G| \le 1$

---

## 四、修改文件一览

| 文件 | 职责 | 关键算法来源 |
| :--- | :--- | :--- |
| `main.py` | 统一入口，顺序调用 12 个 stage | — |
| `gauss_prime_seeding.py` | 子 halo 初始位置播种 | 种子 #8 |
| `halo_gegenbauer_potential.py` | 谱表示引力势 | 种子 #3, #14 |
| `trigonometric_orbit_interpolator.py` | 轨道重采样与速度 | 种子 #4 |
| `merger_tree_enumerator.py` | 并合树枚举、化简、韧性 | 种子 #2, #7 |
| `minimal_merger_path.py` | 并合顺序 TSP 优化 | 种子 #6 |
| `phase_space_flow_coarse_grain.py` | 相空间粗粒化 | 种子 #13 |
| `halo_ode_systems.py` | 晕核振荡 + 进动 + 梯形积分 | 种子 #10, #11, #12 |
| `multizone_accretion.py` | 四区吸积环 | 种子 #9 |
| `convection_vortex_halo.py` | 对流涡旋 + 5 阶迎风 | 种子 #1 |
| `least_squares_quadrature.py` | 最小二乘求积 | 种子 #15 |
| `analysis_pool.py` | 并行分析池 | 种子 #5 |
| `stability_von_neumann.py` | von Neumann 稳定性分析 | 原创整合 |

---

## 五、运行方法

本环境只需 **Python 3 + NumPy**，无需任何其他第三方依赖。

```bash
cd /mnt/data/zpy/sci-swe/source\ code/Synthesis-project-python/247_synth_project/247_synth_project_Advanced
python main.py
```

`main.py` 零参数即可运行，依次执行 12 个 stage：

1. Gauss 素数螺旋播种子 halo 位置
2. 构造 Gegenbauer 谱 halo 势
3. 三角插值重构 halo 轨道
4. 枚举并化简并合树，计算主枝韧性
5. 在 (log M, z) 空间求最优并合路径
6. 正规化流粗粒化相空间分布
7. 积分晕核密度振荡 + 三轴进动 ODE
8. 推进四区吸积环
9. 推进对流涡旋
10. 最小二乘求积估计 halo 质量
11. 并行池分析多棵并合树
12. 输出 von Neumann 稳定性报告

运行完毕会打印每个 stage 的关键数值结果，并在末尾显示总耗时。

---

## 六、科学意义与可扩展方向

- **博士级难点**：将 5 类截然不同的数值方法（谱展开、三角插值、组合枚举、正规化流、高阶有限差分）统一到同一个 halo 物理对象上，并在每个接口处维持物理自洽（如 Liouville 定理检验、CFL 自适应、Horner 稳定性）。
- **可复现性**：所有随机数均用固定种子（`random.Random(2024)`、`np.random.default_rng(0)` 等），任何机器可得到完全一致的结果。
- **可扩展方向**：
  - 将 5 阶迎风格式替换为 WENO-Z，研究非线性稳定性；
  - 用真实的 Consistent-Trees 数据驱动并合树枚举；
  - 把正规化流换为基于 Monge-Ampère 的最优输运粗粒化；
  - 引入自适应网格细化 (AMR) 以解析 sub-kpc 子结构。

---

## 七、工程鲁棒性

- **边界处理**：所有 ODE 右端函数都做了 `np.clip(..., -1e6, 1e6)` 防爆破；涡量在轴 $R=0$ 上被强制归零；最小二乘求积使用 Tikhonov 正则化 $\lambda = 10^{-12}$ 防病态；隐式梯形带下松弛避免迭代发散。
- **CFL 自适应**：对流-扩散步与 5 阶迎风步都按当地速度场动态调整 $\Delta t$。
- **重试机制**：分析池的 retry 装饰器提供 3 次指数退避重试，使单棵树数值失败不影响整体统计。
- **无可视化**：所有 stage 只输出数值/字符串，不含任何 matplotlib / GUI 代码。

---

> **作者**: DA 博士级合成工作流
> **领域**: 计算天体物理 · 暗物质晕形成与并合树构建 · 高阶有限差分与稳定性分析
> **语言**: Python 3 + NumPy
> **入口**: `python main.py` （零参数）
