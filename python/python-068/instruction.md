# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Spatial Eco-Epidemiological Dynamics Project

本项目模拟两个竞争物种在空间异质栖息地中的耦合种群动力学与跨物种传染病传播。核心模型是一组反应-扩散-对流偏微分方程(PDE)，使用傅里叶谱方法结合指数时间差分(ETDRK4)求解；同时包含一个空间均匀的均场ODE近似，采用自适应的隐式中点法求解。项目还包括流行病学分布、栖息地贝塞尔曲面建模、六边形数值积分、制造解验证、网格细化、多维特征空间积分、稀疏矩阵工具以及感染斑块的空间几何分析等多种辅助模块。

入口文件 `main.py` 将保留，其余 `.py` 文件会被移除。你需要根据下面的描述重新实现所有缺失的模块，使得 `main.py` 能够正常运行。

## 文件清单与职责

| 文件 | 职责 |
|------|------|
| `adaptive_midpoint.py` | 自适应隐式中点ODE求解器（含Milne误差估计）及生态-传染病均场ODE右端项 |
| `eco_epi_pde.py` | 核心PDE类，封装反应-扩散-对流系统、栖息地资料、初始化和统计量 |
| `epidemic_distributions.py` | 不完全Gamma函数（AS 147），Gamma以及传染病相关的累积分布函数 |
| `etdrk4_solver.py` | 指数时间差分四阶Runge-Kutta（ETDRK4）谱求解器，处理刚性扩散-对流部分 |
| `exact_solutions.py` | 用于数值验证的制造解析解（人口场、Burgers型解）以及Gauss-Hermite求积 |
| `habitat_surface.py` | 双三次贝塞尔曲面，用于生成空间变化的承载力`K(x,y)`和增长率`r(x,y)` |
| `hexagon_quadrature.py` | 正六边形上的Stroud求积规则，用于斑块级别的种群密度积分 |
| `mesh_refinement.py` | 2D三角形网格细化、一致网格生成和基于梯度的自适应细化指示器 |
| `multi_dim_quadrature.py` | 构造多维张量积求积规则并在多维特征空间（例如易感性、毒力）上积分 |
| `numerical_robustness.py` | 面向机器精度的浮点邻域计算、安全密度截断、阈值判定和2D网格抽稀 |
| `reaction_kinetics.py` | 改进的Selkov型非线性反应项（感染增强、Allee效应）及均场均衡分析 |
| `sparse_matrix_utils.py` | 稀疏矩阵与三元组格式转换、耦合Jacobian稀疏模式构建 |
| `spatial_geometry.py` | 最小包围圆（Welzl算法）和感染斑块连通域分析（面积、边界圆） |

---

## 模块详细描述

### `adaptive_midpoint.py`

**职责**：实现θ-方法形式的隐式中点法单步，利用`scipy.optimize.fsolve`求解隐式方程；结合Milne设备估计局部截断误差，提供自适应步长控制的完整求解器。还包括一个空间均匀的生态-传染病ODE右端函数。

**关键函数与接口**：
- `implicit_midpoint_step(y, t, dt, f, theta=0.5)`：返回一步后的状态 `y_{n+1}`，参数 `theta=0.5` 对应隐式中点法。
- `milne_lte_estimate(y_mid, y_pred, dt, dt_prev, dt_prev2)`：基于当前解与预测解的差估计局部误差。
- `adaptive_midpoint_solve(f, y0, t_span, dt_init, abstol, reltol, theta)`：执行自适应积分，返回包含时间序列 `t`、状态序列 `y`、步数和拒绝次数的字典。内部使用简单的对分/放大步长策略，以及安全因子 `kappa`、最小/最大步长限制。
- `mean_field_eco_epi_ode(t, y, params)`：计算空间均匀模型的右端项。状态向量 `[S1,I1,R1,S2,I2,R2]`，参数包括承载能力、增长率、种内/种间传播系数、恢复率、死亡率、竞争系数等。ODE形式为 logistic 型增长加交叉传播力。

**依赖**：`numpy`, `scipy.optimize.fsolve`。

---

### `eco_epi_pde.py`

**职责**：定义核心类 `EcoEpidemicPDE`，封装周期边界条件下的 2D 反应-扩散-对流系统。管理6个场变量（S1, I1, R1, S2, I2, R2），分别拥有各自的扩散系数 `D` 和平流速度 `(vx, vy)`。提供由 `habitat_surface` 生成的承载力 `K` 和增长率 `r` 空间分布图。非线性项调用 `reaction_kinetics.compute_reaction_terms`。

**关键类与接口**：
- `EcoEpidemicPDE(nx, ny, Lx, Ly, params)`：
  - 属性：网格坐标 `x`, `y`；栖息地地图 `K` 和 `r`（由 `habitat_surface.create_habitat_carrying_capacity` 和 `create_growth_rate_map` 创建）；扩散系数列表 `D`；平流速度 `vx`, `vy`。
  - `default_params()`：静态方法，返回包含所有生物学和物理参数的字典，例如扩散系数、传播系数、恢复率、死亡率、阿利系数以及栖息地曲面所需的参数。
  - `nonlinear_terms(u)`：输入形状 `(6, nx, ny)` 的物理空间场，返回反应项 `rhs`，内部调用 `reaction_kinetics.compute_reaction_terms`。
  - `compute_total_populations(u)`：对每个场在空间上求和（面积加权的数值积分），返回一个包含各分量及总种群 `N1`, `N2` 的字典。
  - `compute_reproduction_numbers(u)`：计算空间变化的感染基本再生数 `R0_1` 和 `R0_2` 并返回均值和最大值。
  - `initialize_state(seed)`：使用高斯脉冲和随机噪声初始化两个物种的空间分布，包括局部感染的起始点。

**依赖**：`numpy`, `scipy.fft.fft2/ifft2`（虽然此类本身不使用FFT，但主流程中使用）、`habitat_surface`、`reaction_kinetics`。

---

### `epidemic_distributions.py`

**职责**：实现 AS 147 算法计算正则化下不完全 Gamma 函数，并在此基础上提供 Gamma 分布的概率密度和累积分布，以及流行病学中常用的世代间隔分布和传染期分布。

**关键函数与接口**：
- `incomplete_gamma(x, p)`：返回 `(value, ifault)`，`value` 为下正则化不完全 Gamma 值，`ifault` 为错误标志（0:正常,1:输入无效,2:下溢）。内部使用对数 Gamma 函数提升数值稳定性，并通过连分式/级数展开计算。
- `gamma_pdf(t, shape, scale)`：Gamma 密度。
- `gamma_cdf(t, shape, scale)`：优先使用 `scipy.special.gammainc`，若不可用则调用 `incomplete_gamma` 并归一化。
- `generation_interval_distribution(t, mean, std)`：根据给定的均值和标准差（默认为 5.2 和 1.7），返回以天为单位的 Gamma 密度向量。
- `cumulative_generation_interval(t, mean, std)`：对应的累积分布。
- `infectious_period_cdf(t, gamma_rate)`：基于指数分布的累积分布 `1 - exp(-γ t)`，默认恢复率为 `1/5`。

**依赖**：`numpy`, `math.lgamma`, 可选 `scipy.special.gammainc`。

---

### `etdrk4_solver.py`

**职责**：实现 ETDRK4 算法的谱求解器，有效处理刚性线性扩散‑对流部分。预计算指数算子
