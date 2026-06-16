# 超新星爆发辐射流体模拟：高阶有限差分与稳定性分析
## —— 博士级科学计算合成项目（PROJECT 250）

**领域**：计算天体物理 · 超新星爆发 · 辐射流体 · 高阶有限差分 · 稳定性分析

**规模**：小规模可复现实验，单节点纯 Python 实现，零参数可运行

---

## 一、项目定位与科学目标

本项目构建一个面向**核心坍缩超新星 (CCSN) 爆发机制**的 1D 球对称辐射流体求解器，融合 **15 个异质科研项目**的核心算法，在博士级计算难度下统一处理以下科学问题：

1. **激波传播与 Rankine-Hugoniot 跳跃**：WENO5 + HLL 通量在球坐标几何守恒形式下捕捉主激波
2. **辐射-物质耦合**：隐式 FLD (flux-limited diffusion) + 蒙特卡罗中微子包输运
3. **SASI 极限环振荡**：Van der Pol 振子模拟 + SSD 空间-谱分解提取主导模式
4. **线性稳定性分析**：Jordan 块分解 + von Neumann 放大矩阵谱半径判定
5. **熵产与热力学第二律**：贝叶斯后验推断 + 因果性 bootstrap 检验

每个模块都耦合了真实的超新星物理公式，非通用数值方法的简单换皮。

---

## 二、15 个种子项目 → 超新星物理的映射

| # | 种子项目 | 原算法思想 | 在超新星中的物理映射 | 对应模块 |
|---|----------|-----------|-------------------|---------|
| 1 | `304_disk01_positive_rule` | 单位圆盘正权求积 | **离散纵标 S_N 角向求积** (辐射输运方向积分) | `radiation_quadrature.py` |
| 2 | `1013_jaewansh_KMOU-RDP-GARCH-model` | GARCH(1,1) 波动率聚集 | **随机中微子光度** L_ν(t) 起伏 | `neutrino.py` |
| 3 | `591_interp_chebyshev` | Chebyshev 多项式插值 | **κ(ρ,T) 不透明度双切比雪夫表** | `opacity.py` |
| 4 | `744_md` | 分子动力学 velocity-Verlet | **Lagrangian 抛射物示踪粒子** + 显式流体求解器 | `tracer.py`, `hydro.py` |
| 5 | `1222_EyringML_...Bagged_TimeSeries_Causality` | Bootstrap + PCMCI 因果 | **后激波→中微子 Granger 因果检验** | `causality.py` |
| 6 | `320_duel_simulation` | 交替命中 Monte Carlo | **中微子包在壳层间的吸收/散射/穿透** | `neutrino.py` |
| 7 | `1205_abartolo-tb_Bayesian-Second-Law` | 贝叶斯热力学第二律 | **熵产率 Σ 的 Gamma 后验** + 第二律一致性检验 | `bayesian_entropy.py` |
| 8 | `1387_vanderpol_ode_period` | Van der Pol 极限环周期 | **SASI 振荡** (30 ms 周期, Urabe 公式) | `sasi_dynamics.py` |
| 9 | `1103_Zangle-Lab_QPV_and_Particle_Tracking` | MSD + looping path length | **示踪粒子扩散诊断** (α-指数 + 缠绕度) | `tracer.py` |
| 10 | `760_mgmres` | Restarted GMRES(m) | **隐式辐射扩散** (I - Δt L) E_r = E_r^n | `implicit_solver.py` |
| 11 | `965_r83s` | 稀疏三对角存储 r83 | **带状 Jacobian** 紧凑存储 | `implicit_solver.py` |
| 12 | `1127_nschawor_eeg-mu-alpha-development` | Spatial Spectral Decomposition | **SASI 多模式分解** (l=1, l=2) | `sasi_dynamics.py` |
| 13 | `1337_triangulation_histogram` | 三角剖分内点直方图 | **激波曲面各向异性** χ² 检验 | `shock_geometry.py` |
| 14 | `610_jordan_matrix` | Jordan 标准型 | **Jacobian 特征值 + 非正规度** 稳定性 | `stability.py` |
| 15 | `006_anishchenko_ode` | 混沌振子参数化 | **SASI 湍流驱动项** 非线性能量注入 | `sasi_dynamics.py` |

---

## 三、核心数学物理模型

### 3.1 球坐标守恒律

$$
\frac{\partial U}{\partial t} + \frac{1}{r^2}\frac{\partial (r^2 F)}{\partial r} = S(U), \quad U = \begin{pmatrix}\rho \\ \rho v \\ E\end{pmatrix}, \quad F = \begin{pmatrix}\rho v \\ \rho v^2 + P \\ v(E+P)\end{pmatrix}
$$

源项 $S = S_\text{geom} + S_\text{grav} + S_\text{rad}$：
- 几何曲率 $S_\text{geom} = (0, 2P/r, 0)^T$
- 点质量引力 $S_\text{grav} = (0, -\rho GM/r^2, -\rho GM v/r^2)^T$
- 辐射耦合 $S_\text{rad} = (-\chi_F F_r/c, \ldots, -\chi_a c E_r + \chi_a c a T^4)$

### 3.2 物态 (EOS)

$$
P = P_\text{ion} + P_e + P_\text{rad} = \frac{\rho k_B T}{\mu m_p} + \frac{8\pi\sqrt{2}\, m_e^{3/2}}{3 h^3}(k_B T)^{5/2} F_{3/2}(\eta_e) + \frac{1}{3}aT^4
$$

其中 $F_{3/2}$ 为 Fermi-Dirac 积分，$\eta_e$ 为电子简并参数。声速：

$$
c_s^2 = \Gamma_1 \frac{P}{\rho}, \quad \Gamma_1 = \beta + \frac{(4-3\beta)^2 (\gamma_\text{ion}-1)}{\beta + 36(\gamma_\text{ion}-1)(4-3\beta)/10}
$$

### 3.3 不透明度 (Chebyshev 双插值)

$$
\log_{10}\kappa(\log_{10}T, \log_{10}\rho) \approx \sum_{k=0}^{N_T-1}\sum_{j=0}^{N_\rho-1} c_{kj}\, T_k(\xi_T)\, T_j(\xi_\rho)
$$

Kramers 自由-自由 + Thomson 电子散射 + $e^\pm$ 对高温修正。

### 3.4 隐式辐射扩散 (GMRES + ILU(0))

$$
(I - \Delta t\, L) E_r^{n+1} = E_r^n, \quad L(E_r) = \nabla\cdot(D\nabla E_r) - \chi_a c E_r + \chi_a c a T^4
$$

采用 r83 带状存储 + Restarted GMRES(m=20) + ILU(0) 预条件，收敛判据：

$$
\|r_k\| / \|r_0\| < \epsilon_\text{rel} \wedge \|r_k\| < \epsilon_\text{abs}
$$

### 3.5 随机中微子源 (GARCH(1,1))

$$
L_\nu(t) = L_0 e^{-t/\tau_\nu} + \varepsilon_t, \quad \varepsilon_t = \sigma_t z_t,\; z_t \sim N(0,1)
$$
$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2, \quad \alpha + \beta < 1
$$

### 3.6 SASI 极限环 (修正 Van der Pol)

$$
\ddot x - \mu(1 - x^2/A^2)\dot x + \omega_0^2 x = \varepsilon \sin(\omega_\text{drive} t)
$$

Urabe 周期公式：
- 小 $\mu$: $T \approx \frac{2\pi}{\omega_0}\left(1 + \frac{\mu^2}{16} + \frac{11\mu^4}{3072}\right)$
- 大 $\mu$: $T \approx \frac{\mu}{\omega_0}(3 - 2\ln 2)$

### 3.7 线性稳定性 (Jordan 形式)

$$
\delta \dot U = J\, \delta U \;\Rightarrow\; \delta U(t) = e^{Jt}\delta U(0)
$$

$J$ 含 Jordan 块 $J_k = \lambda_k I + N$ 时，$e^{J_k t} = e^{\lambda_k t}\sum_{j=0}^{m-1} t^j N^j / j!$，谱半径 $\rho(G(k)) \le 1$ 为 von Neumann 稳定条件。

### 3.8 贝叶斯熵产推断

$$
\Sigma \sim \text{Gamma}(\alpha_0, \beta_0), \quad \Phi_\nu = K\Sigma + \varepsilon,\; \varepsilon \sim N(0, \sigma^2)
$$

共轭更新：$\alpha_\text{post} = \alpha_0 + N/2$, $\beta_\text{post} = \beta_0 + \frac{1}{2\sigma^2}\sum (\Phi_i/K)^2$.

第二律检验：$\langle \Sigma \rangle > 2\sigma(\Sigma) \Rightarrow$ 一致.

### 3.9 Granger 因果 (Bootstrap)

$$
F = \frac{(\text{RSS}_r - \text{RSS}_u)/p}{\text{RSS}_u/(n-2p-1)} \sim F(p, n-2p-1)
$$

Moving block bootstrap (MBB) 估计 $F$ 的 95% 可信区间.

### 3.10 拉格朗日示踪 (velocity-Verlet)

$$
r(t+\Delta t) = r(t) + v(t)\Delta t + \tfrac{1}{2}a(t)\Delta t^2
$$
$$
v(t+\Delta t) = v(t) + \tfrac{1}{2}(a(t) + a(t+\Delta t))\Delta t
$$
$$
a = -\frac{GM}{r^2} + \frac{1}{\rho}\frac{dP}{dr} - \frac{v - v_\text{fluid}}{t_\text{stop}}
$$

MSD: $\text{MSD}(\Delta t) = \langle |r(t+\Delta t) - r(t)|^2 \rangle \propto \Delta t^\alpha$.

---

## 四、文件清单与职责

| 文件 | 角色 | 关键类 / 函数 |
|------|------|--------------|
| `main.py` | 统一入口 (零参数) | `main()` 编排全流程 |
| `constants.py` | CGS 物理常数 | `C_LIGHT`, `G_GRAV`, `M_SUN`, `A_RADIATION`... |
| `mesh.py` | 球对称 Lagrangian 网格 | `SphericalMesh` (几何递进, Courant dt, 激波定位) |
| `eos.py` | 物态 (离子 + e± + 辐射) | `EquationOfState` (Fermi-Dirac 积分近似) |
| `opacity.py` | 不透明度表 | `OpacityTable` (双切比雪夫), `ChebyshevInterpolator1D` |
| `radiation_quadrature.py` | S_N 角求积 | `DiscreteOrdinateQuadrature`, `level_symmetric_sn`, flux limiter |
| `reconstruction.py` | WENO5 重构 | `WENO5Reconstructor`, HLL / Lax-Friedrichs 通量 |
| `hydro.py` | 显式辐射流体 | `RadiationHydroSolver` (WENO5+HLL+RK3+正性限幅) |
| `implicit_solver.py` | GMRES 隐式求解 | `BandMatrix`, `RestartedGMRES`, `ilu0_preconditioner` |
| `shock_geometry.py` | 激波曲面 | `ShockSurface` (Y_lm), `IcosphereMesh`, `rankine_hugoniot` |
| `neutrino.py` | 中微子源与输运 | `GARCHLuminosity`, `MonteCarloNeutrinoTransport`, `NeutrinoSpectrum` |
| `stability.py` | 线性稳定性 | `jordan_block`, `von_neumann_amplification`, `stability_criterion` |
| `sasi_dynamics.py` | SASI 动力学 | `VanDerPolOscillator`, `SpatialSpectralDecomposition` |
| `causality.py` | 时间序列因果 | `granger_causality`, `bagged_granger_causality`, MBB |
| `tracer.py` | 拉格朗日示踪 | `TracerParticles` (MSD, looping, 逃逸率) |
| `bayesian_entropy.py` | 贝叶斯熵诊断 | `BayesianEntropyProduction`, `total_entropy` |
| `diagnostics.py` | 全局诊断 | 引力结合能, 动能, 内能, 致密度 ξ, α_crit |
| `initial.py` | 前身星初值 | `progenitor_profile`, `insert_shock` |

---

## 五、运行方法

```bash
cd /mnt/data/zpy/sci-swe/source\ code/Synthesis-project-python/250_synth_project/250_synth_project_Advanced
python main.py
```

零参数，输出 13 个诊断段落到标准输出，无可视化。典型输出：

```
======================================================================
 1. 球对称网格构造
======================================================================
  单元数: 64,  r ∈ [1.00e+06, 5.00e+08] cm
...
======================================================================
 13. 全局诊断
======================================================================
  引力结合能 E_grav = -7.356e+53 erg = -735.630 foe
  总质量 M = 24.551 M_sun
  激波半径 = 1988.13 km

 完成
  总运行时间: 0.255 s
```

---

## 六、工程鲁棒性与边界处理

- **正性保护 (Zhang-Shu 2010)**：每步强制 $\rho > \varepsilon_\rho$, $P > \varepsilon_P$, 并在 EOS 反解温度前限幅比内能
- **Courant 自适应**：`dt = C_CFL · min(dr / (|v| + c_s))`, 下限 1e-14 s
- **加速度/速度限幅**：示踪粒子 $|a| < 10^{18}$ cm/s², $|v| < 0.1c$
- **GARCH 平稳性检查**：$\alpha + \beta < 1$, $\omega > 0$ 在构造时强制
- **GMRES 幸运中断**：当 $h_{j+1,j} \approx 0$ 时提前收敛
- **Chebyshev 节点选择**：在 $\log T$, $\log \rho$ 域避免端点 Runge 现象
- **激波定位熵判据**：$d\ln s / d\ln r > 2.5$ 识别主激波，避免噪声误触发
- **ILU(0) 主元保护**：当主元 $|d_i| < 10^{-30}$ 时正则化
- **Granger 检验正则化**：$(Z^T Z + 10^{-8} I)$ 防止共线

---

## 七、可复现性

- 全部随机种子固定：`rng = np.random.default_rng(20260607)`
- 网格参数、EOS 参数、GARCH 参数、SASI 参数均在 `main.py` 头部显式声明
- 无任何外部数据依赖，无任何可视化依赖
- 单文件入口 `main.py`，单节点即可复现全部输出

---

## 八、可扩展方向 (博士论文级)

1. **2D/3D 扩张**：将 1D 球对称升级到 $(r, \theta)$ 轴对称，引入角向 WENO + 跨极轴处理
2. **多群中微子输运**：从 MC 单能升级到 $E_\nu$ 多群离散纵标 (S_N × 能量)
3. **真实 EOS 表**：替换解析 EOS 为 Shen/Lattimer-Swesty 表格，双三次插值
4. **GPU 加速**：将 WENO5 + GMRES 用 CuPy/JAX 实现，支持 $10^4$ 单元
5. **数据驱动 SASI 预报**：用本项目生成的时间序列训练 LSTM/Transformer 预测激波 revival
6. **自适应网格加密 (AMR)**：在激波附近动态加密，保持总单元数

---

## 九、核心物理常数参考

| 量 | 符号 | 值 (CGS) |
|----|------|----------|
| 光速 | $c$ | $2.998\times 10^{10}$ cm/s |
| 引力常数 | $G$ | $6.674\times 10^{-8}$ cm³/g/s² |
| 太阳质量 | $M_\odot$ | $1.989\times 10^{33}$ g |
| Boltzmann 常数 | $k_B$ | $1.381\times 10^{-16}$ erg/K |
| 质子质量 | $m_p$ | $1.673\times 10^{-24}$ g |
| 辐射常数 | $a$ | $7.566\times 10^{-15}$ erg/cm³/K⁴ |
| Stefan-Boltzmann | $\sigma$ | $5.670\times 10^{-5}$ erg/cm²/s/K⁴ |
| 1 foe | - | $10^{51}$ erg |

---

**合成完成时间**：2026-06-07
**语言**：Python 3.10+, 仅依赖 `numpy`, `scipy` (标准科学计算栈)
**行数**：约 2400 行 Python，无冗余可视化代码
