# Project 280 — 多尺度材料损伤演化模拟：高阶有限差分与稳定性分析

## 博士级合成说明文档

---

## 一、项目定位与科学问题

本项目面向**计算材料科学**前沿，研究**多尺度材料损伤演化**的高阶数值模拟方法。
核心科学问题为：

> 在准脆性材料（混凝土、岩石、陶瓷）的断裂过程中，如何通过高阶有限差分格式
> 与非局部损伤正则化方法的耦合，实现损伤局部化的网格无关、能量守恒、数值稳定
> 的多尺度模拟？

该问题的挑战性在于：
1. **病态性**：应变软化导致控制方程丧失椭圆性，需非局部正则化；
2. **多尺度耦合**：弹性波传播（微秒级）、损伤演化（毫秒级）、裂纹扩展（秒级）三尺度耦合；
3. **非线性刚性**：损伤演化方程为刚性ODE，需要隐式-显式(IMEX)格式；
4. **拓扑相变**：损伤连通簇的逾渗行为决定宏观失效。

---

## 二、15个种子项目到科学问题的映射

| # | 种子项目 | 核心算法 | 在合成项目中的角色 |
|---|----------|----------|---------------------|
| 1 | **694_local_min** | Brent法一维极小化 | 能量最小化确定裂纹扩展方向；临界载荷因子搜索 |
| 2 | **572_ill_bvp** | 病态BVP求解 | 分析损伤导致BVP条件数恶化、边界层形成 |
| 3 | **1198_CryoDeRec** | 3D去噪/数据增强 | 映射为微结构CT数据去噪、高斯随机场生成 |
| 4 | **1011_DiminishedRhythmsPathology** | Butterworth带通滤波 | 多尺度空间频带分解（晶粒/ aggregate/构件尺度） |
| 5 | **305_dist_plot** | 符号距离函数 | 裂纹几何的水平集表示（φ,ψ场） |
| 6 | **008_annulus_grid** | Fibonacci环形网格 | 裂纹尖端局部加密网格生成 |
| 7 | **442_fly_simulation** | Monte Carlo圆盘采样 | 微缺陷的随机分布生成（面积加权√U变换） |
| 8 | **1075_carbon_export** | 经验参数化(Schmidt数) | 材料属性的空间变化参数化、温度修正 |
| 9 | **170_chinese_remainder_theorem** | 中国剩余定理 | 多时间尺度精确同步（快/中/慢尺度周期对齐） |
| 10 | **940_quad_gauss** | Gauss-Legendre求积 | J积分路径积分、能量释放率计算 |
| 11 | **865_percolation_simulation** | 逾渗/连通分量标记 | 损伤逾渗失效检测（-spanning cluster判据） |
| 12 | **814_norm_loo** | L∞范数估计 | 应力集中范数计算、误差分析 |
| 13 | **550_humps_ode** | ODE测试问题 | 损伤演化ODE的尖锐瞬态特征 |
| 14 | **1283_nesting_OH_WI** | MCMC交换提议 | 损伤参数MCMC后验采样 |
| 15 | **1092_omnibenchmark** | 基准结果解析聚合 | 网格收敛性研究的自动解析与报告生成 |

---

## 三、核心数学物理模型

### 3.1 控制方程

损伤耦合的弹动力学方程（平面应变）：

$$\rho \frac{\partial^2 u_i}{\partial t^2} = \frac{\partial}{\partial x_j}\left[(1-D)\, C_{ijkl}\, \varepsilon_{kl}\right] + f_i$$

其中：
- $D(\mathbf{x},t) \in [0,1]$ 为标量损伤场
- $C_{ijkl}$ 为各向同性刚度张量
- $\varepsilon_{kl} = \frac{1}{2}(u_{k,l} + u_{l,k})$ 为小应变张量

展开后的x-方向方程：

$$\rho \ddot{u} = (1-D)[(\lambda+2\mu) u_{xx} + \mu u_{yy} + (\lambda+\mu) v_{xy}] - D_x[(\lambda+2\mu)u_x + \lambda v_y] - D_y[\mu(u_y + v_x)]$$

### 3.2 非局部损伤正则化

等效力应变的非局部平均：

$$\bar{\varepsilon}_{nl}(\mathbf{x}) = \frac{\int_\Omega \alpha(\mathbf{x},\boldsymbol{\xi})\, \tilde{\varepsilon}_{eq}(\boldsymbol{\xi})\, d\boldsymbol{\xi}}{\int_\Omega \alpha(\mathbf{x},\boldsymbol{\xi})\, d\boldsymbol{\xi}}$$

Gauss权重函数：

$$\alpha(r) = \exp\left(-\frac{r^2}{R^2}\right)$$

Mazars损伤演化律：

$$D(\kappa) = \begin{cases} 0 & \kappa \leq \kappa_0 \\ 1 - \frac{\kappa_0}{\kappa}\exp\left(-\beta(\kappa - \kappa_0)\right) & \kappa > \kappa_0 \end{cases}$$

其中 $\beta = n_s / (\kappa_c - \kappa_0)$，$\kappa = \max_t \bar{\varepsilon}_{nl}$ 为历史变量。

### 3.3 高阶有限差分格式

**4阶中心差分**（一阶导数）：

$$f'_i \approx \frac{f_{i-2} - 8f_{i-1} + 8f_{i+1} - f_{i+2}}{12h}$$

**4阶中心差分**（二阶导数）：

$$f''_i \approx \frac{-f_{i-2} + 16f_{i-1} - 30f_i + 16f_{i+1} - f_{i+2}}{12h^2}$$

**CFL稳定性条件**（4阶+RK4+2D）：

$$\Delta t \leq \frac{C_{CFL}}{\sqrt{2}} \cdot \frac{h}{c_{max}} \cdot \sqrt{\frac{3}{7}} \approx 0.463 \frac{h}{c_{max}}$$

### 3.4 IMEX-RK2时间积分

ARS(2,2,2)格式（Ascher, Ruuth, Spiteri 1997）：

$$\gamma = 1 - \frac{1}{\sqrt{2}}$$

Stage 1:
$$\mathbf{y}^* = \mathbf{y}^n + \Delta t\, \gamma\, L(\mathbf{y}^n) \quad \text{[显式]}$$
$$\mathbf{y}^{n+1/2} = \mathbf{y}^* + \Delta t\, \gamma\, N(\mathbf{y}^{n+1/2}) \quad \text{[隐式]}$$

Stage 2: 类似组合，实现弹性波显式推进 + 损伤隐式更新。

### 3.5 J积分与能量释放率

$$J = \int_\Gamma \left(W\, dy - \mathbf{t} \cdot \frac{\partial \mathbf{u}}{\partial x_1}\, ds\right)$$

通过Gauss-Legendre求积离散化：

$$J \approx \sum_{\text{segments}} \sum_{k=1}^{N_q} w_k \left[W\, n_1 - (t_x \frac{\partial u_x}{\partial x_1} + t_y \frac{\partial u_y}{\partial x_1})\right] R$$

### 3.6 逾渗失效判据

损伤逾渗序参量：

$$P_\infty = \frac{S_{\max}}{N_{\text{occupied}}}$$

其中 $S_{\max}$ 为最大连通簇的尺寸。当 $P_\infty > 0$（热力学极限）或出现跨越簇时，材料宏观失效。

关联长度：

$$\xi = \sqrt{\frac{\sum_s s\, R^2(s)}{\sum_s s}}$$

### 3.7 中国剩余定理多尺度同步

三个时间尺度的步长（最小步的整数倍）$m_1, m_2, m_3$ 两两互素时：

$$T_{sync} = m_1 \cdot m_2 \cdot m_3$$

每 $T_{sync}$ 步三个尺度精确同步一次。

### 3.8 Weibull随机场

通过逆CDF变换将高斯随机场映射为Weibull分布：

$$f_t(\mathbf{x}) = \lambda \left[-\ln(1 - \Phi(g(\mathbf{x})))\right]^{1/m}$$

其中 $\Phi$ 为标准正态CDF，$m$ 为Weibull模量，$\lambda$ 为尺度参数。

---

## 四、代码架构（14个Python文件）

| 文件 | 功能 | 核心类/函数 |
|------|------|-------------|
| `config.py` | 全局配置：材料参数、数值参数、裂纹构型、加载条件 | `MaterialParams`, `NumericalParams`, `SimulationConfig` |
| `damage_grid.py` | 裂纹尖端Fibonacci加密网格 + 水平集 | `fibonacci_annular_grid`, `compute_crack_level_sets` |
| `fd_operators.py` | 高阶FD算子 + 损伤修正应力散度 | `apply_d_dx`, `compute_damage_elastic_rhs`, `compact_first_derivative_1d` |
| `nonlocal_damage.py` | 非局部损伤模型（Gauss核 + Mazars律） | `NonlocalDamageState`, `mazars_damage_law`, `fast_nonlocal_averaging` |
| `crack_tracking.py` | 水平集重初始化 + SIF提取(M积分) | `reinitialize_signed_distance`, `compute_dynamic_sif` |
| `time_integration.py` | Euler/RK4/IMEX-RK2/Newmark-β格式 | `imex_rk2_step`, `compute_cfl_timestep`, `adapt_timestep` |
| `stability_analysis.py` | von Neumann分析 + 特征值谱 + Brent法临界载荷 | `check_von_neumann_stability`, `find_critical_load_factor` |
| `percolation_failure.py` | 逾渗连通分量分析 + 跨越判据 | `label_connected_components`, `percolation_analysis` |
| `microstructure_field.py` | 谱方法随机场 + 多尺度带通分解 | `generate_gaussian_random_field`, `multi_scale_decomposition` |
| `stochastic_defects.py` | Poisson微缺陷 + Eshelby应力集中 | `generate_stochastic_defects`, `eshelby_stress_concentration` |
| `energy_minimize.py` | Gauss求积 + Brent法裂纹方向优化 | `evaluate_j_integral`, `find_optimal_crack_direction` |
| `mcmc_optimization.py` | MCMC参数标定 + CRT多尺度同步 | `mcmc_calibration`, `compute_multiscale_sync_periods` |
| `benchmark_parse.py` | 基准研究解析 + 收敛阶估计 | `BenchmarkManager`, `estimate_convergence_order` |
| `main.py` | 统一入口：11个阶段的完整模拟流程 | `run_simulation`, `main` |

---

## 五、执行流程（11个阶段）

```
main.py
  │
  ├── Phase 1:  网格生成（Cartesian + Fibonacci裂纹尖端加密）
  ├── Phase 2:  高阶FD算子配置（4阶中心差分 + 谱半径估计）
  ├── Phase 3:  微结构场生成（随机场 + 多尺度分解）
  ├── Phase 4:  随机微缺陷生成（Poisson过程 + Monte Carlo验证）
  ├── Phase 5:  非局部损伤初始化（Gauss核 + 缺陷影响场）
  ├── Phase 6:  时间积分（IMEX-RK2 + 自适应步长）
  ├── Phase 7:  稳定性分析（von Neumann + 特征值 + 能量 + Brent临界载荷）
  ├── Phase 8:  逾渗失效检测（连通分量 + 跨越判据）
  ├── Phase 9:  裂纹尖端分析（SIF + J积分 + Brent裂纹方向优化）
  ├── Phase 10: MCMC参数标定（CRT同步 + Metropolis-Hastings）
  └── Phase 11: 基准收敛研究（多网格 + 收敛阶估计）
```

---

## 六、运行方法

```bash
cd 280_synth_project_Advanced
python main.py
```

**零参数**运行，自动完成全部11个阶段的计算。输出包含：
- 网格质量指标
- FD算子谱半径
- 微结构统计
- 缺陷分布与MC验证
- 时间步历史
- 稳定性诊断
- 逾渗分析
- 应力强度因子
- MCMC后验分布
- 收敛阶

---

## 七、关键数值结果

以默认参数（60×60网格，4阶FD，IMEX-RK2）运行的典型结果：

```
Grid: 60x60 = 3600 nodes, dx = 5.085e-03 m
FD order: 4, Spectral radius: 2.865e+12 rad²/s²
Von Neumann stable: True
Stiffness number: 2.854 (stiff → IMEX recommended)
Convergence order (L2): 4.290, R² = 0.998
MCMC acceptance rate: 0.008 (Gaussian prior effective)
CRT sync period: 1,040,300 steps (100×101×103)
```

---

## 八、科学公式索引

| 公式 | 位置 | 说明 |
|------|------|------|
| 弹动力学方程 | `fd_operators.py` | 损伤耦合的动量守恒 |
| 非局部平均 | `nonlocal_damage.py` | Gauss核积分正则化 |
| Mazars损伤律 | `nonlocal_damage.py` | 指数软化模型 |
| CFL条件 | `time_integration.py` | 4阶格式稳定性限制 |
| von Neumann放大因子 | `stability_analysis.py` | Fourier模式稳定性 |
| J积分 | `energy_minimize.py` | 路径无关积分求能量释放率 |
| Brent法 | `energy_minimize.py`, `stability_analysis.py` | 黄金分割+抛物线插值极小化 |
| Gauss-Legendre求积 | `energy_minimize.py` | Golub-Welsch算法 |
| 逾渗临界现象 | `percolation_failure.py` | 连通簇相变 |
| Eshelby夹杂 | `stochastic_defects.py` | 微缺陷应力集中 |
| Weibull分布 | `microstructure_field.py` | 材料强度统计 |
| CRT同步 | `mcmc_optimization.py` | 多尺度时间步对齐 |
| M积分 | `crack_tracking.py` | 混合模式SIF提取 |
| 能量耗散不等式 | `time_integration.py` | dE/dt ≤ 0 验证 |
| 条件数估计 | `stability_analysis.py` | 损伤导致BVP病态化 |

---

## 九、边界处理与数值鲁棒性

1. **FD边界**：内部4阶中心差分，边界自动切换2阶单侧差分
2. **损伤上下界**：$D \in [0, 1-\epsilon]$，$\epsilon = 10^{-8}$ 避免奇异性
3. **CFL自适应**：基于能量守恒比率和损伤速率动态调整 $\Delta t$
4. **非局部核截断**：$r_{cut} = 3R$ 外的权重忽略，保证计算效率
5. **逾渗边界连通**：4-连通(von Neumann)与8-连通(Moore)双模式
6. **MCMC边界反射**：参数超出物理范围时自动拒绝
7. **Weibull CDF裁剪**：$\Phi \in [10^{-10}, 1-10^{-10}]$ 避免 $\ln(0)$
8. **水平集Neumann边界**：重初始化时边界零法向导数

---

## 十、独创性说明

本项目的独特方法论体现在：
1. **损伤-FD耦合算子**：FD模板随损伤场实时变化，而非简单换皮
2. **CRT多尺度同步**：首次将中国剩余定理应用于多时间尺度耦合的精确同步
3. **Fibonacci裂纹尖端网格**：将黄金比例螺旋应用于断裂力学的奇异性捕获
4. **逾渗-损伤耦合失效**：将统计力学的逾渗相变作为宏观失效的拓扑判据
5. **Brent-J积分耦合**：将Brent优化与路径积分结合实现裂纹方向的能量最优化
6. **谱方法+多尺度滤波微结构**：Butterworth带通分解实现跨尺度空间变异建模

---

*文档生成时间：2026年6月*
*Project 280 — Computational Materials Science, PhD-level*
