# 宇宙大尺度结构 N 体模拟: 高阶有限差分与稳定性分析
## Computational Astrophysics — LSS N-body with High-Order FD & Stability Analysis

**项目编号**: 246_synth_project_Advanced
**科学领域**: 计算天体物理 — 宇宙大尺度结构形成
**语言**: Python 3 (零参数可运行)
**规模**: 小规模可复现实验 (N_grid=16, N_particles=4096, N_steps=8)

---

## 1. 科学问题陈述

本项目解决**冷暗物质 (CDM) 宇宙大尺度结构形成**的数值模拟问题。
从早期宇宙 (z ≈ 49) 的高斯随机密度扰动出发,通过非线性引力不稳定性,
模拟宇宙网 (cosmic web) 的形成过程,包括:
- **暗物质 halo** 的坍缩 (对应 Sheth-Tormen 质量函数)
- **Filaments** (纤维结构) 与 **Voids** (空洞) 的演化
- **两点相关函数** ξ(r) 与 **功率谱** P(k) 的非线性增长

### 控制方程 (共动坐标)

$$
\frac{\partial \delta}{\partial t} + \frac{1}{a} \nabla \cdot [(1+\delta) \mathbf{v}] = 0
\quad \text{(连续性方程)}
$$

$$
\frac{\partial \mathbf{v}}{\partial t} + H(a) \mathbf{v}
+ \frac{1}{a} (\mathbf{v} \cdot \nabla) \mathbf{v}
= -\frac{1}{a} \nabla \Phi
\quad \text{(Euler 方程)}
$$

$$
\nabla^2 \Phi = 4\pi G \bar{\rho} a^2 \delta
\quad \text{(Poisson 方程)}
$$

其中 $a(t)$ 为尺度因子,$H(a) = \dot{a}/a$ 为 Hubble 参数,
$\delta = (\rho - \bar{\rho})/\bar{\rho}$ 为密度对比。

### 本项目的核心方法论

1. **高阶紧致有限差分** (2/4/6/8 阶) 用于空间离散化
2. **FFT + 带状矩阵 LU** 双路径 Poisson 求解器
3. **Von Neumann 稳定性分析** (ITP 求根 + 幂法求谱半径)
4. **CMA-ES 自动参数标定** (软化长度 ε 与时间步长 Δt)
5. **Sheth-Tormen halo 质量函数** 作为物理 benchmark

---

## 2. 输入种子项目 → 科学映射

| # | 种子项目 | 核心算法 | 本项目中的角色 | 所在模块 |
|---|---------|---------|--------------|---------|
| 1 | **1429_zero_itp** | ITP (Interpolate- Truncate-Project) 求根 | 求解色散关系 $\rho(G) = 1$ 的根 → 临界 CFL 数 $\Delta t_\mathrm{crit}$ | `stability_analysis.py::zero_itp` |
| 2 | **1068_jaweriaamjad_internalstate** | JSON 配置加载 + 路径相对仓库根解析 | 宇宙学参数 (`CosmoParams`) 管理 | `cosmo_config.py` |
| 3 | **201_colored_noise** | f_alpha 有色噪声 (功率谱 $P(k) \propto k^{-\alpha}$) | 生成原初密度扰动 $\delta(\mathbf{k}) = \sqrt{P(k)} \cdot \mathcal{N}$ | `initial_conditions.py` |
| 4 | **1059_Glyphosate_crystallization** | OpenFF 拓扑创建 | 网格节点/边/面邻接结构 (`MeshTopology`) | `mesh_topology.py::MeshTopology` |
| 5 | **615_kdv_exact** | KdV 精确解 + 残差 | 密度波 benchmark (sech² soliton / 有理精确解) | `time_integrator.py::kdv_*` |
| 6 | **1290_Dohoon1_Halitosis** | PCR 微生物分组 + 伪随机模采样 | 原初宇宙随机模 QMC 采样 | `initial_conditions.py::prime_qmc_phases` |
| 7 | **1201_de-ranit_GPP** | CMA-ES 参数标定 + L-BFGS 精调 | 自动调优 (ε, Δt) | `parameter_calibration.py` |
| 8 | **666_legendre_shifted_polynomial** | 移位 Legendre 三项递推 | 高阶质量赋值核 / Gauss-Legendre 求积 | `legendre_kernel.py` |
| 9 | **236_cube_surface_distance** | 立方体表面两点距离统计 | 周期盒子内距离分布 (相关函数 Monte Carlo) | `mesh_topology.py::cube_surface_distance_stats` |
| 10 | **367_fd2d_heat_steady** | 2D 稳态热传导 FD | 扩展到 3D 周期 Poisson,高阶 FD 模板 | `poisson_solver.py`, `finite_difference.py` |
| 11 | **786_nas** | N-body Acceleration Search | 哈希网格邻居搜索 + CIC/TSC 质量赋值 | `nas_search.py` |
| 12 | **884_polygon_distance** | 多边形三角剖分 + 三角形面积 | Delaunay 三角剖分 (宇宙网 filament 识别) | `mesh_topology.py::polygon_triangulate` |
| 13 | **915_prime_plot** | 素数筛 + 素数散点 | Halton QMC 序列 (粒子初始位置) | `initial_conditions.py::prime_sieve`, `sample_particle_positions` |
| 14 | **979_r8gb** | 带状矩阵 PLU 分解 + 回代 | 1D 隐式紧致差分求解器 | `finite_difference.py::r8gb_fa_python`, `r8gb_sl_python` |
| 15 | **902_power_method** | 幂法 + 复数特征值幂法 | 放大矩阵谱半径 $\rho(G)$ → 稳定性判定 | `stability_analysis.py::power_method`, `power_method2` |

---

## 3. 新增数学物理模型与核心公式

### 3.1 Eisenstein-Hu 无重子转移函数

$$
T(k) = \frac{L(q)}{L(q) + C q^2}, \quad
L(q) = \ln(2e + 1.8 q), \quad
C = 14.7 + \frac{133}{169 + 0.3 q^{-1/4}}, \quad
q = \frac{k}{\Omega_m h^2 \text{Mpc}^{-1}}
$$

### 3.2 原初功率谱

$$
P(k) = A_s \left(\frac{k}{k_*}\right)^{n_s - 1} T^2(k)
$$

### 3.3 Zel'dovich 近似位移场

$$
\psi_i(\mathbf{k}) = \frac{i k_i \delta(\mathbf{k})}{|\mathbf{k}|^2 D_1(a)}
$$

### 3.4 高阶紧致差分 (Lele 1992)

$$
\beta f''_{i-1} + f''_i + \beta f''_{i+1}
= a \frac{f_{i+1}-2f_i+f_{i-1}}{h^2}
+ b \frac{f_{i+2}-2f_i+f_{i-2}}{4h^2}
+ c \frac{f_{i+3}-2f_i+f_{i-3}}{9h^2}
$$

### 3.5 Sheth-Tormen halo 质量函数

$$
\frac{dn}{dM} = \frac{\bar{\rho}}{M} \frac{d\ln\sigma^{-1}}{dM} f_\mathrm{ST}(\nu)
$$

$$
f_\mathrm{ST}(\nu) = A \sqrt{\frac{2a}{\pi}}
\left[1 + (a\nu^2)^{-p}\right] \nu \exp\left(-\frac{a\nu^2}{2}\right)
$$

其中 $\nu = \delta_c / \sigma(M,z)$, $A = 0.3222$, $a = 0.707$, $p = 0.3$。

### 3.6 Von Neumann 稳定性

放大矩阵 $G(\mathbf{k}, \Delta t)$ 的谱半径:

$$
\rho(G) \leq 1 + \mathcal{O}(\Delta t)
$$

临界 CFL 数由 ITP 求根: $\rho(G(\mathbf{k}, \Delta t_\mathrm{crit})) = 1$。

### 3.7 CMA-ES 代价函数

$$
\mathcal{C}(\varepsilon, \Delta t)
= \sum_k \frac{(P_\mathrm{num}(k) - P_\mathrm{ref}(k))^2}{\sigma_P^2}
+ \lambda \left(\frac{\varepsilon}{0.1} + \frac{\Delta t}{0.01}\right)
$$

---

## 4. 文件结构

```
246_synth_project_Advanced/
├── __init__.py                  # 包初始化
├── __main__.py                  # python -m 入口
├── main.py                      # 统一入口 (12 个 stage)
├── cosmo_config.py              # 参数配置加载 (seed 1068)
├── cosmo_config.json            # JSON 配置文件
├── legendre_kernel.py           # Legendre 多项式核 (seed 666)
├── finite_difference.py         # 高阶 FD + 带状矩阵 (seeds 367+979+666)
├── poisson_solver.py            # Poisson 求解器 (seeds 367+979)
├── initial_conditions.py        # 初始条件生成 (seeds 201+1290+915)
├── mesh_topology.py             # 网格拓扑 (seeds 1059+884+236)
├── time_integrator.py           # 时间积分 + KdV (seed 615)
├── stability_analysis.py        # 稳定性分析 (seeds 1429+902)
├── parameter_calibration.py     # CMA-ES 标定 (seed 1201)
├── nas_search.py                # 邻居搜索 + 质量赋值 (seed 786)
├── diagnostics.py               # 诊断工具 (P(k), ξ(r), 能量)
├── results/                     # 输出目录
└── README_博士级合成说明.md     # 本文档
```

---

## 5. 运行方式

### 5.1 直接运行 (零参数)

```bash
cd 246_synth_project_Advanced
python main.py
```

输出 12 个 stage 的诊断结果,保存 `results/params_used.json` 与 `results/summary.txt`。

### 5.2 作为模块运行

```bash
cd ..
python -m 246_synth_project_Advanced
```

### 5.3 自定义参数

编辑 `cosmo_config.json`:
```json
{
  "box_length": 64.0,
  "n_grid": 16,
  "n_particles": 4096,
  "omega_m": 0.308,
  "omega_l": 0.692,
  "hubble": 0.6781,
  "z_init": 49.0,
  "z_final": 0.0,
  "n_steps": 8,
  "fd_order": 4,
  "softening_eps": 0.05,
  "sigma_8": 0.8159,
  "spectral_index": 0.9667,
  "seed": 42
}
```

---

## 6. 预期输出 (示例)

```
========================================================================
  Stage 1: 加载宇宙学参数 (seed 1068 internalstate/config_utils)
========================================================================
  盒子边长 L = 64.0 Mpc/h
  网格 N = 16 (总 4096 个网格单元)
  ...

========================================================================
  Stage 2: Legendre 多项式与求积验证 (seed 666)
========================================================================
  ∫ P_0 P_0 = +2.000000e+00  (期望 +2.000e+00)  [OK]
  ∫ P_1 P_1 = +6.666667e-01  (期望 +6.667e-01)  [OK]
  ...

========================================================================
  Stage 3: 高阶 FD 精度验证 (seeds 367 + 979 + 666)
========================================================================
  FD order 2:  max|err_explicit| = 1.283e-02,  max|err_compact(c4)| = 2.481e-05
  FD order 4:  max|err_explicit| = 6.583e-05,  max|err_compact(c4)| = 2.481e-05
  3D Laplacian (N=16, 4阶) max|err| = ~1e-15

========================================================================
  Stage 7: KdV 精确解残差基准 (seed 615)
========================================================================
  sech² soliton 残差: max|r| = 2.776e-17
  有理精确解残差: max|r| = 1.526e-05

========================================================================
  Stage 8: von Neumann 稳定性分析 (seeds 1429 + 902)
========================================================================
  幂法: 矩阵 [[2,1],[1,3]] 主特征值 λ = 3.618034  (精确 3.618034)
  ITP 求根 x³=2:  x = 1.2599210499  (精确 1.2599210499)

#  模拟完成 — 总耗时: ~1.5s
```

---

## 7. 博士级难度要点

1. **高阶紧致差分**: Lele (1992) 紧致格式,通过三对角系统隐式求解,
   需要 Sherman-Morrison 周期处理 + Thomas 算法。
2. **多网格 V-cycle**: 粗糙网格修正 + 全加权限制 + 三线性延拓。
3. **Von Neumann 稳定性**: 复数放大矩阵,使用 `power_method2` 估计复数特征值,
   ITP 算法求解非线性稳定性边界。
4. **CMA-ES + L-BFGS 混合优化**: 全局探索 + 局部精调,代价函数基于
   Sheth-Tormen 物理 benchmark。
5. **周期边界条件**: FFT 对角化 + Ewald 求和思想 + 周期哈希网格。
6. **引力软化与时间步长权衡**: 通过 CMA-ES 自动标定,代价函数包含
   功率谱偏差与计算成本。

---

## 8. 可复现性

- 所有随机数通过 `np.random.default_rng(seed)` 控制,默认 seed = 42
- 参数保存到 `results/params_used.json`
- 诊断保存到 `results/summary.txt`
- 在小规模配置 (N=16, N_p=4096, N_steps=8) 下,1.5 秒内完成

---

## 9. 依赖

仅需 Python 标准库 + NumPy + SciPy (用于 L-BFGS 精调):

```
numpy >= 1.20
scipy >= 1.7
```

无其他第三方依赖。
