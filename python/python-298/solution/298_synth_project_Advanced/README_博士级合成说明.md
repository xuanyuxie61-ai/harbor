# PROJECT 298: 计算等离子体 PIC 方法高阶有限差分稳定性分析

## 项目概述

本项目为**博士级前沿科学计算项目**, 围绕计算等离子体物理中的**粒子模拟 PIC 方法**展开,
重点研究**高阶有限差分格式**与**稳定性分析**在小规模可复现实验中的应用。

### 科学问题

1D 静电粒子模拟 (Particle-in-Cell, PIC) 中:
- 高阶有限差分 (2/4/6/8 阶) 如何影响数值色散关系?
- 不同精度阶数的截断误差如何修正 Landau 阻尼率?
- 如何通过 Von Neumann 分析确定稳定参数区域?
- 粒子数、网格分辨率与时间步长的耦合效应如何?

## 核心物理模型

### 1. 等离子体基本参数

```
等离子体频率:  ω_pe = √(n_e * e² / (ε₀ * m_e))
德拜长度:      λ_D = √(ε₀ * k_B * T_e / (n_e * e²))
热速度:        v_th = √(2 * k_B * T_e / m_e)
库仑对数:      ln Λ = 23 - ln(√(n_e * 1e-6) * T_e[eV]^(-3/2))
```

### 2. Bohm-Gross 色散关系 (Langmuir 波)

```
ω² = ω_pe² + 3 * k² * v_th²
```

### 3. PIC 数值色散关系

```
sin²(ω * dt / 2) / (dt/2)² = ω_pe² * D_fd(k*dx) * S_shape(k*dx)
```

其中:
- `D_fd(k*dx) = (k_num / k)²` 为有限差分校正
- `S_shape = [sin(k*dx/2) / (k*dx/2)]^(2*shape_order)` 为形状因子

### 4. Landau 阻尼率 (解析)

```
ω_r ≈ ω_pe * √(1 + 3 * k² * λ_D²)
γ   ≈ -√(π/8) * ω_pe / (k*λ_D)³ * exp(-1/(2*k²*λ_D²) - 3/2)
```

### 5. Boris 粒子推进

```
Step 1: v⁻ = vⁿ + (q*dt)/(2m) * Eⁿ
Step 2: t = (q*dt)/(2m) * B,  s = 2t/(1+t²)
        v' = v⁻ + v⁻ × t
        v⁺ = v⁻ + v' × s
Step 3: vⁿ⁺¹ = v⁺ + (q*dt)/(2m) * Eⁿ
```

### 6. PIC 稳定性判据

```
CFL 条件:         v_max * dt / dx < 0.9
等离子体频率约束:  ω_pe * dt < 2 (实际 ω_pe * dt < 0.2 避免数值加热)
德拜分辨率:       dx ≤ λ_D / 10
粒子数约束:       N_p > N_grid * λ_D / L
```

## 文件结构

```
298_synth_project_Advanced/
├── main.py                  # 统一入口 (12 个 PART, 零参数运行)
├── plasma_constants.py      # 物理常数与等离子体公式
├── high_order_fd.py         # 高阶有限差分模板 (2/4/6/8 阶)
├── poisson_solver.py        # 带状矩阵泊松求解器 (直接法 + FFT + 多重网格)
├── boris_pusher.py          # Boris 粒子推进 (非相对论/相对论)
├── root_finder.py           # 求根算法 (Brent + Newton-Maehly + Sylvester)
├── velocity_quadrature.py   # 速度空间积分与采样 (高维球/立方体/CVT)
├── grid_generator.py        # 网格生成 (椭球/映射/Chebyshev)
├── particle_loader.py       # 粒子加载 (均匀/quiet-start/双流/bump-on-tail)
├── dispersion_analysis.py   # 数值色散分析
├── parameter_sweep.py       # 参数扫描 (含性能计时)
├── stability_analyzer.py    # 稳定性分析 (素数/组合/Von Neumann)
├── diagnostics.py           # 诊断与鲁棒性评估
└── README_博士级合成说明.md   # 本文档
```

## 种子项目融合映射

| 种子项目 | 核心算法 | 在本项目中的角色 |
|---------|---------|----------------|
| 333_ellipsoid_grid | 椭球网格生成 | 速度空间椭球网格 + 粒子分布 |
| 719_matlab_compiler | 矩阵输出 | 格式化矩阵诊断输出 |
| 910_prime | 素数筛法/因子分解 | FFT 友好网格点选择 |
| 801_newton_maehly | 多项式顺序求根 | 数值色散关系求解 |
| 832_ode_sweep_parfor | 参数扫描 | (k,ω,dx,dt) 空间扫描 |
| 145_ccn_rule | 组合数/Fornberg 算法 | 高阶差分模板系数 |
| 178_circle_distance | 球面距离统计 | CVT 采样均匀性评估 |
| 986_r8ncf | 带状矩阵存储 | 泊松方程高效求解 |
| 1113_seizyml | 模型鲁棒性分析 | PIC 鲁棒性指标 |
| 553_hyperball_integrals | 高维球面积分 | 速度空间矩计算 |
| 1029_TimingofOneShot | 矩阵生成 | 参数扫描矩阵构造 |
| 255_cvt_corn | CVT 采样 | 速度空间最优采样 |
| 1427_zero_brent | Brent 求根 | 色散关系变号区间求根 |
| 896_polynomial_resultant | Sylvester 结式 | 公根分析与共振判断 |
| 232_cube_felippa_rule | 立方体高斯求积 | 多维速度空间积分 |

## 运行方法

```bash
cd 298_synth_project_Advanced
python main.py
```

无需任何参数, 程序将自动完成:
1. 物理参数计算
2. 网格设置与质量评估
3. 高阶差分模板分析
4. 速度空间积分验证
5. 求根算法测试
6. 数值色散分析
7. 稳定性分析
8. PIC Landau 阻尼模拟
9. 后处理诊断
10. 收敛性研究
11. 椭球网格应用
12. 参数扫描性能分析

## 关键数值结果

### 典型等离子体条件
- 电子数密度 n_e = 1.0 × 10²⁰ m⁻³
- 电子温度 T_e = 100 eV
- 等离子体频率 ω_pe = 5.641 × 10¹¹ rad/s
- 德拜长度 λ_D = 7.434 × 10⁻⁶ m
- 热速度 v_th = 5.931 × 10⁶ m/s

### 高阶差分截断误差 (k*dx = 0.5π)
- 2阶: 18.9%
- 4阶: 5.4%
- 6阶: 1.8%

### 稳定性判据
- CFL 数: < 0.9
- ω_pe * dt = 0.1 (满足稳定性约束)
- dx / λ_D = 0.078 (满足德拜分辨率)

## 算法特色

1. **领域深度耦合**: 所有算法严格围绕 PIC 方法设计, 变量命名体现等离子体物理
2. **多尺度分析**: 从单粒子 Boris 推进到宏观模式增长率
3. **高阶精度**: 实现 2/4/6/8 阶有限差分并系统对比
4. **完整稳定性框架**: CFL + Von Neumann + 参数扫描
5. **博士级复杂度**: 涉及等离子体色散、Landau 阻尼、数值加热等前沿课题

## 参考文献

1. Birdsall, C.K. & Langdon, A.B. *Plasma Physics via Computer Simulation*, IOP, 2004.
2. Hockney, R.W. & Eastwood, J.W. *Computer Simulation Using Particles*, Adam Hilger, 1988.
3. Brent, R.P. *Algorithms for Minimization Without Derivatives*, Dover, 2002.
4. Dawson, J.M. "Particle Simulation of Plasmas", *Reviews of Modern Physics*, 1983.
5. Langdon, A.B. "On Eliminating Self-Heating in PIC Simulations", *JCP*, 1979.
6. Fornberg, B. "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids", *Math. Comp.*, 1988.
