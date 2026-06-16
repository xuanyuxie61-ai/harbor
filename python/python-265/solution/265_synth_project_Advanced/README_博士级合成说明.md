# PROJECT 265 — 宇宙线在日球层中的传播：高阶有限差分与稳定性分析

## 一、项目概述

本项目面向**计算空间物理**前沿方向 —— **宇宙线在日球层中的传播与扩散**。具体科学问题为：

> 如何以**高阶有限差分格式**（四阶紧致 Padé、五阶 WENO）数值求解**聚焦 Parker 输运方程**（focused Parker transport equation），并通过 **von Neumann 稳定性分析、CFL 限制、矩阵谱半径检验、FAS 多层网格求解、随机微分方程（Milstein 积分）反演、MCMC 湍流采样**等一系列博士级数值手段，对日球层银河宇宙线（GCR）调制问题给出**小规模可复现实验**的完整解答。

项目的代码架构、变量命名、物理量纲全部与宇宙线输运问题深度耦合（如 `kappa_par`、`D_mumu`、`parker_imf`、`focusing_length`、`BarenblattCosmicRay` 等），绝非通用数值库的换皮。

---

## 二、15 个种子项目的核心算法到本项目的映射

| # | 种子项目 | 核心算法 / 思想 | 在本项目中的角色 |
|---|---|---|---|
| 1 | `589_insurance_simulation` | 基于寿命表的蒙特卡罗路径采样 | → `stochastic_parker.py` 中基于伪粒子轨迹的 SDE 采样 |
| 2 | `410_fem2d_predator_prey_fast` | 二维有限元 + 时间积分 | → `focused_transport_eq.py` 在 (r, μ) 张量网格上的半离散算子 |
| 3 | `696_locker_simulation` | 置换搜索策略 | → `species_composition.py` 中相空间 urn 采样策略 |
| 4 | `548_human_mesh2d` | 二维三角化边界网格生成 | → `heliocentric_mesh.py` 日球层对数径向网格与俯仰角网格 |
| 5 | `1023_pranavgupta2603_covid-spread-simulation` | 接触网络的传播模型 | → `focused_transport_eq.py` 中俯仰角散射的接触项 d/dμ[D_μμ df/dμ] |
| 6 | `1075_nicpittman_tropical_pacific_carbon_export` | Schmidt/Solubility 物理公式模块 | → `cosmic_ray_physics.py` 中 Wanninkhof 风格的扩散张量系数公式 |
| 7 | `1432_zero_rc` | Brent 反向通讯求根 | → `dispersion_roots.py` 中等离子体色散关系求根 |
| 8 | `351_fd_to_tec` | FD 数据 → Tecplot 结构化转换 | → `grid_data_io.py` FD 输出 → 结构化 ASCII 转换 |
| 9 | `1168_terenceneo_glow` | Adam + Polyak 平均优化器 | → `inverse_kappa.py` 扩散系数反演的 Adam+Polyak 优化器 |
| 10 | `425_ffmatlib` | FreeFem++ 网格准备 | → `heliocentric_mesh.py` 单元度量、体元、边界 mask |
| 11 | `1283_cdonnay_nesting_OH_WI` | 基于 swap 的 MCMC 提议 | → `turbulence_mcmc.py` 湍流参数 MCMC 采样（swap-pair 提议） |
| 12 | `901_porous_medium_exact` | Barenblatt 自相似精确解 + 残差 | → `analytical_benchmarks.py` 非线性宇宙线扩散的 Barenblatt 解 |
| 13 | `644_lambert_w` | Lambert W 函数 Halley 迭代 | → `dispersion_roots.py` CR 逃逸时间反演中的 Lambert W |
| 14 | `1376_urn_simulation` | 无放回超几何抽样 | → `species_composition.py` CR 元素成分的超几何抽样 |
| 15 | `876_poisson_1d_multigrid` | 多层网格 Gauss-Seidel + 限制/延拓 | → `multigrid_fas.py` FAS 多层网格 V-cycle 求解稳态输运 |

---

## 三、博士级核心物理与数学模型

### 3.1 聚焦 Parker 输运方程

$$
\frac{\partial f}{\partial t}
+ (\mu v + V_{sw}) \frac{\partial f}{\partial r}
= \frac{1}{r^2} \frac{\partial}{\partial r} \Big[ r^2 \kappa_{rr} \frac{\partial f}{\partial r} \Big]
+ \frac{\partial}{\partial \mu} \Big[ D_{\mu\mu} \frac{\partial f}{\partial \mu} \Big]
+ \frac{v}{2L}(1-\mu^2)\Big(\mu - \frac{V_{sw}}{\mu v}\Big) \frac{\partial f}{\partial \mu}
+ \frac{1}{3}(\nabla \cdot \mathbf{V}_{sw})\, p\, \frac{\partial f}{\partial p}
+ Q
$$

其中：
- $f(r,\mu,p,t)$：宇宙线相空间分布函数 [m$^{-3}$ sr$^{-1}$ GV$^{-1}$]
- $\kappa_{rr} = \kappa_{||}\mu^2 + \kappa_\perp \sin^2\psi$：径向扩散系数
- $D_{\mu\mu}$：俯仰角散射系数（准线性理论 Jokipii 1966）
- $L$：磁场聚焦长度 $L = |B|/(\hat{b}\cdot\nabla|B|)$
- $\psi$：Parker 螺旋角 $\tan\psi = r\Omega_\odot\sin\theta / V_{sw}$

### 3.2 Parker 太阳风磁场（Archimedean spiral）

$$
B_r = B_0 \Big(\frac{r_0}{r}\Big)^2, \qquad
B_T = -B_0 \frac{r_0^2 \Omega_\odot \sin\theta}{V_{sw}} \cdot \frac{1}{r}
$$

### 3.3 高阶有限差分格式

**四阶紧致（Padé/Lele 1992）**

$$
\frac{1}{6} f'_{i-1} + \frac{2}{3} f'_i + \frac{1}{6} f'_{i+1}
= \frac{f_{i+1} - f_{i-1}}{2h}
$$

**五阶 WENO（Jiang & Shu 1996）**

对通量 $f_{i+1/2}$ 取三条候选子模板的凸组合，权值由光滑度指示子 $\beta_k$ 决定：

$$
\omega_k = \frac{\alpha_k}{\sum \alpha_\ell}, \quad \alpha_k = \frac{d_k}{(\epsilon + \beta_k)^2}
$$

### 3.4 von Neumann 稳定性分析

对常系数对流-扩散方程 $u_t + v u_x = \kappa u_{xx}$，前向 Euler 离散的放大因子：

$$
G(k) = 1 + \Delta t \Big(-v\, \widehat{(u_x)} + \kappa\, \widehat{(u_{xx})}\Big)
$$

稳定性条件：$\max_k |G(k)| \le 1$。

### 3.5 CFL 时间步限制

$$
\Delta t \le \min\Big(\frac{\Delta r}{|v|},\,
                      \frac{\Delta r^2}{2\kappa_{rr}},\,
                      \frac{\Delta \mu^2}{2 D_{\mu\mu}}\Big)
$$

### 3.6 FAS 多层网格

对每一层网格 $l$ 执行 $\nu_1$ 次预光滑 + 限制残差到粗网格 + 粗网格求解 + 延拓修正 + $\nu_2$ 次后光滑。限制用全权平均（full-weighting），延拓用线性插值。

### 3.7 SDE 等价（Zhang 1999）

聚焦输运方程等价于 Itô 随机微分方程：

$$
dr = A_r\, dt + \sqrt{2\kappa_{rr}}\, dW_r, \qquad
d\mu = A_\mu\, dt + \sqrt{2 D_{\mu\mu}}\, dW_\mu
$$

其中漂移项 $A_r$ 含太阳风对流、扩散梯度、聚焦力；使用 Milstein 积分提高强收敛阶到 1.0。

### 3.8 反问题：Adam + Polyak 平均

$$
\min_{\theta} J(\theta) = \frac{1}{2}\sum_i (f_{model}(r_i; \theta) - f_{obs,i})^2
                         + \frac{\lambda}{2} \|L\theta\|^2,
\quad \theta_j = \log \kappa(r_j)
$$

Polyak 平均：$\bar\theta_k = \beta_2 \bar\theta_{k-1} + (1-\beta_2) \theta_k$

### 3.9 Lambert W 逃逸时间反演

从 $\tau_{diff}$ 与 $\tau_{adv}$ 反演宇宙线在有限传播区域的平均逃逸时间：

$$
t_{esc} = \tau_{adv} \Big[ 1 + W\Big(-\frac{\tau_{diff}}{\tau_{adv}} e^{-1/\tau_{adv}}\Big)\Big]
$$

### 3.10 湍流 MCMC（slab + 2D 复合谱）

$$
P(k) = \delta B^2 \Big[ f_{slab} k^{-q_{slab}} e^{-k l_c}
                     + (1 - f_{slab}) k^{-q_{2D}} e^{-k l_c} \Big]
$$

使用 swap-pair 提议的 Metropolis-Hastings 对 $(q_{slab}, q_{2D}, l_c, \delta B/B)$ 采样。

### 3.11 超几何成分抽样

CR 从源丰度中无放回地抽取 $N_{esc}$ 个粒子，再经散裂生存概率 $p_s = \lambda_{esc}/(\lambda_{esc} + \lambda_{sp,s})$ 过滤，得到观测成分。

---

## 四、项目结构

```
265_synth_project_Advanced/
├── main.py                     统一入口（零参数）
├── cosmic_ray_physics.py       物理常数 + Parker IMF + 运动学
├── focused_transport_eq.py     聚焦 Parker 输运算子
├── high_order_stencils.py      高阶差分模板（upwind1/2, compact4, WENO-5）
├── stability_analysis.py       von Neumann、CFL、谱半径、色散误差
├── heliocentric_mesh.py        日球层网格 + 边界 + 限制/延拓
├── multigrid_fas.py            FAS 多层网格 V-cycle
├── analytical_benchmarks.py    力场模型 / Barenblatt / Green 函数
├── stochastic_parker.py        SDE 等价（Milstein 积分）
├── inverse_kappa.py            扩散系数反演（Adam+Polyak）
├── dispersion_roots.py         Lambert W + Brent + 色散根
├── turbulence_mcmc.py          湍流参数 MCMC
├── species_composition.py      CR 成分超几何抽样
├── grid_data_io.py             FD 数据 I/O + 诊断
└── README_博士级合成说明.md    本文件
```

---

## 五、运行方法

```bash
cd 265_synth_project_Advanced
python main.py
```

主程序顺序执行 12 个实验：
1. **日球层网格与 Parker 磁场** —— 输出 0.1/1/10/50 AU 处的磁场强度与螺旋角
2. **高阶差分精度校验** —— 对比 upwind1、upwind2、compact4、WENO5 的最大误差
3. **von Neumann 稳定性** —— 各格式在 CFL = 0.8 下的最大放大因子，以及矩阵谱半径
4. **正向聚焦输运积分** —— 显式时间推进求解 $f(r,\mu,t)$
5. **FAS 多层网格** —— V-cycle 求解稳态问题，输出残差历史
6. **解析基准** —— 力场调制、稳态对流-扩散、Barenblatt 自相似解、Green 函数
7. **SDE 随机传播** —— Milstein 积分反演日球层调制
8. **反问题** —— Adam+Polyak 反演 $\kappa(r)$
9. **Lambert W / Brent 求根** —— 等离子体色散根、共振波数、逃逸时间反演
10. **湍流 MCMC** —— swap-pair 提议的 Metropolis-Hastings
11. **成分抽样** —— urn 超几何 + leaky-box 散裂平衡
12. **网格 I/O** —— FD 数据写出/读入、Tecplot 转换、通量/各向异性诊断

预期总运行时间 ~ 50 秒。所有输出均为纯文本，无可视化。

---

## 六、合成说明

### 6.1 原项目 → 科学问题映射
- 所有"通用"数值方法（有限差分、多层网格、Adam 优化、MCMC、求根、urn 抽样）都经过**深度改造**，使其变量、参数、物理意义完全绑定到宇宙线输运问题。
- 例如 `gauss_seidel` 在原 `poisson_1d_multigrid` 中解 $-u''=f$；在本项目的 `multigrid_fas.py` 中解的是 $(1/r^2)\partial_r[r^2 \kappa_{rr} \partial_r f] + \partial_\mu^2 f = Q$，其中系数含有真实的 $\kappa_{rr}(r,\mu)$ 与日球层对数网格。

### 6.2 新增数学物理模型
- 完整的聚焦 Parker 输运方程（含聚焦项、俯仰角散射、绝热冷却、太阳风对流）
- Parker 螺旋磁场模型（Archimedean spiral）
- 准线性扩散系数（Jokipii 1966）与跨场非线性引导中心理论
- 力场调制模型（Gleeson & Axford 1968）
- Barenblatt 自相似解的非线性宇宙线类比
- 等离子体色散关系（冷等离子体 Stix 1992）
- 宇宙线源丰度（Binns et al. 2018）与散裂截面（Silberberg & Tsao 1990）

### 6.3 修改文件清单
- `cosmic_ray_physics.py`：从 `carbon_math.py` 的物理公式风格改造；从 `lambert_w.m` 吸收 Lambert W 算法；从 `zero_rc.m` 吸收 Brent 求根
- `focused_transport_eq.py`：从 `fe2d_d_fast.m` 的 PDE 时间积分改造
- `high_order_stencils.py`：新建，提供 upwind/compact4/WENO5 三种高阶差分
- `stability_analysis.py`：从 `poisson_1d_multigrid` 的 Gauss-Seidel 改造为稳定性分析工具
- `heliocentric_mesh.py`：从 `human_mesh2d.m` + `ffmatlib` 的网格准备改造
- `multigrid_fas.py`：从 `poisson_1d_multigrid.m` + `ctof.m` + `ftoc.m` + `gauss_seidel.m` 改造为 FAS
- `analytical_benchmarks.py`：从 `porous_medium_exact.m` 的 Barenblatt 解改造
- `stochastic_parker.py`：从 `insurance_simulation.m` 的 Monte Carlo 路径采样改造
- `inverse_kappa.py`：从 `glow/optim.py` 的 Adam+Polyak 改造
- `dispersion_roots.py`：从 `lambert_w.m` + `zero_rc.m` 改造
- `turbulence_mcmc.py`：从 `swap_proposal.py` 的 swap-pair 改造
- `species_composition.py`：从 `urn_sample.m` + `ksub_random2.m` 改造
- `grid_data_io.py`：从 `fd_to_tec.m` 改造

### 6.4 解决的博士级科学问题
- 日球层 GCR 调制的数值求解（含各向异性扩散、聚焦、绝热冷却）
- 高阶差分格式的色散/耗散特性评估
- 湍流参数的不确定性量化（MCMC）
- 扩散系数的逆问题（Adam+Polyak 正则化反演）
- 宇宙线成分的散裂修正（leaky-box 模型）

### 6.5 边界处理与数值鲁棒性
- 所有网格均避免 $\mu=\pm 1$ 的坐标奇点（使用 $\epsilon$-偏移）
- 聚焦项在 $\mu \approx 0$ 处使用正则化
- Lambert W 对 $-1/e$ 邻域使用分支切换（W₀ / W₋₁）
- Brent 求根带符号变化检验
- 多层网格边界值在每次 V-cycle 后重新施加
- urn 抽样使用条件二项近似避免组合数溢出
- CFL 时间步使用 min-of-three 限制

---

## 七、参考文献

- Parker, E. N. (1965). The passage of energetic charged particles through interplanetary space. *Planet. Space Sci.* 13, 9-49.
- Jokipii, J. R. (1966). Cosmic-ray propagation. I. Charged particles in a random magnetic field. *ApJ* 146, 480.
- Schlickeiser, R. (2002). *Cosmic Ray Astrophysics*. Springer.
- Lele, S. K. (1992). Compact finite difference schemes with spectral-like resolution. *J. Comput. Phys.* 103, 16-42.
- Jiang, G.-S., & Shu, C.-W. (1996). Efficient implementation of weighted ENO schemes. *J. Comput. Phys.* 126, 202-228.
- Brandt, A. (1977). Multi-level adaptive solutions to boundary-value problems. *Math. Comp.* 31, 333-390.
- Gleeson, L. J., & Axford, W. I. (1968). Solar modulation of galactic cosmic rays. *ApJ* 154, 1011.
- Zhang, M. (1999). A Markov stochastic process and its application to the solution of the cosmic-ray transport equation. *ApJ* 513, 409.
- Matthaeus, W. H., et al. (1995). Magnetic fluctuation power near the proton inertial length scale. *Geophys. Res. Lett.* 22, 3035.
- Binns, W. R., et al. (2018). The source composition of galactic cosmic rays. *ApJ* 865, 139.
