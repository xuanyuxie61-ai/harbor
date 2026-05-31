# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# CO2 Capture Amine Absorption Dynamics – 项目描述

本项目是一个用于模拟胺法 CO₂ 捕集过程的集成化学工程计算框架，涉及反应动力学、传质、热力学、过程优化、降解路径分析等多个子模型。项目入口为 `main.py`，它示例性地调用所有已实现的模块并展示结果。

后续将**只保留 `main.py`**，删除其他所有 `.py` 文件。你的任务是**根据本描述重新实现被删除的模块**，使 `main.py` 能够成功运行并输出合理结果。

---

## 文件与模块职责

### 1. `utils.py`
**职责**：提供物理常量、输入校验、数值安全函数和基础数学工具，被几乎所有其他模块依赖。

- 定义物理常量：`R_GAS`（理想气体常数），`STANDARD_TEMP`（298.15 K），`STANDARD_PRESSURE`。
- 提供校验函数 `validate_positive`（检查数值是否为正/非负），`clip_concentration`（裁剪浓度的最小值至很小的正数）。
- 提供安全数学函数：`safe_log`，`safe_divide`。
- 提供动力学与热力学常用函数：
  - `arrhenius_rate(A, Ea, T)`：根据阿伦尼乌斯公式计算速率常数。
  - `van_t_hoff(K0, dH, T, T0)`：温度相关的平衡常数。
  - `wilke_chang_diffusion(T, mu, Vb, alpha_assoc)`：液相扩散系数关联式。
  - `hatta_number(k2, D_A, c_B, k_L)`：第二级反应的气液吸收 Hatta 数。
  - `enhancement_factor_hatta(Ha, E_infinite)`：增强因子估算。
- 提供光谱方法用的节点与矩阵工具：
  - `chebyshev_nodes(n, a, b)`：生成第二类 Chebyshev 节点。
  - `chebyshev_differentiation_matrix(n, a, b)`：构造 Chebyshev 微分矩阵。
- 提供一个格式化打印函数 `print_section(title)`。

**关键点**：所有模块应从此处导入所需常量与工具函数，不要重复实现。

---

### 2. `reaction_kinetics.py`
**职责**：实现 CO₂‑胺反应的动力学模型，基于两性离子机理，并提供负载率计算与批量吸收模拟。

- 类 `AmineKinetics`：
  - 初始化参数根据胺类型（`"MEA"`, `"MDEA"`, `"PZ"`）加载不同动力学数据（指前因子、活化能、pKa 等）。
  - 方法 `k2(T)`：返回二级反应速率常数（阿伦尼乌斯形式）。
  - 方法 `pKa_T(T)`：温度相关 pKa。
  - 方法 `base_catalysis_contribution(T, amine_conc, OH_conc, H2O_conc)`：计算两性离子去质子化的碱催化贡献之和。
  - 方法 `reaction_rate(T, c_CO2, c_amine, c_OH)`：总体吸收速率，考虑去质子化增强。
  - 方法 `carbamate_hydrolysis_rate(T, c_carbamate, c_H2O)`：氨基甲酸盐水解速率。
- 类 `CO2LoadingCalculator`：
  - 基于给定的 `AmineKinetics` 实例工作。
  - 方法 `equilibrium_loading(T, P_CO2, c_amine_total)`：使用改进的 Kent‑Eisenberg 关系式计算平衡 CO₂ 负载率（alpha）。
  - 方法 `kinetic_loading_estimate(T, P_CO2, c_amine_total, contact_time)`：基于接触时间估计实际负载率。
- 函数 `simulate_batch_absorption(T, P_CO2, c_amine0, t_span, amine_type, n_steps)`：
  - 建立一个四状态（CO₂、RNH₂、RNHCOO⁻、RNH₃⁺）的 ODE 系统，包含气液传质项和反应项。
  - 使用 BDF3 积分器求解并返回时间和状态矩阵。

**依赖**：`utils`，`ode_integrators`（用于批量吸收模拟）。

---

### 3. `ode_integrators.py`
**职责**：提供刚性/非刚性常微分方程积分器，以及基于周期检测的分析。

- 函数 `explicit_trapezoidal(f, tspan, y0, n_steps)`：Heun 方法（显式梯形法）求解非刚性 ODE。
- 函数 `bdf3_solver(f, tspan, y0, n_steps)`：三阶向后差分公式（BDF3）求解刚性 ODE，前两步用显式 RK3 启动，后续步骤使用 `fsolve` 求解隐式方程，包含失败回退到 BDF1 的逻辑。
- 函数 `gear_bdf2(f, tspan, y0, n_steps)`：Gear’s BDF2 方法（备选）。
- 函数 `predator_prey_like_cycles(f, tspan, y0, n_steps, threshold)`：求解 ODE 后，检测第一个状态变量的局部极大值，计算周期和振幅，返回周期字典、时间、状态。
- 函数 `solve_stiff_amine_ode(f, tspan, y0, n_steps, stiffness_threshold)`：通过数值估计雅可比矩阵特征值来计算刚度比，自动选择 BDF3 或显式梯形法。

**依赖**：`utils`，`numpy`，`scipy.optimize.fsolve`。

---

### 4. `spectral_methods.py`
**职责**：使用 Chebyshev 谱方法求解扩散‑反应边值问题，以及 2D 多项式拟合。

- 类 `ChebyshevQuadrature`：
  - 基于 Jacobi 矩阵特征值方法生成 Chebyshev‑2 型求积节点和权重。
  - 构造函数接收点数 `n_points, a, b`，计算规则。
  - 方法 `integrate(f)`：计算带权重 `sqrt((x-a)*(b-x))` 的积分。
- 函数 `integrate_reaction_rate_profile(rate_func, z_a, z_b, n)`：利用上述求积类计算膜厚度上的反应速率积分。
- 类 `SpectralDiffusionSolver`：
  - 内部利用 Chebyshev 微分矩阵求解二阶方程。
  - 方法 `solve_film_diffusion_reaction(D_diff, k_rxn, delta, c_interface, c_bulk)`：求解 `D d²c/dz² – k c = 0`，边界条件 `c(0)=c_i`，`c(delta)=c_b`，返回浓度剖面和界面通量。
  - 方法 `solve_channel_flow(mu, delta_p, L, R)`：求解泊肃叶流动。
- 函数 `polynomial_fit_2d_vandermonde(x, y, z, degree)`：构建 2D Vandermonde 矩阵，最小二乘拟合多项式 `sum c_{ij} x^i y^j`（i+j ≤ degree），返回系数和条件数。
- 函数 `evaluate_2d_polynomial(coeffs, degree, x, y)`：评估该多项式。

**依赖**：`utils`（含 `chebyshev_nodes`, `chebyshev_differentiation_matrix`），`numpy`。

---

### 5. `mass_transfer.py`
**职责**：双膜理论传质模型与填料塔轴向分布模拟。

- 类 `TwoFilmModel`：
  - 初始化时根据温度和压力计算亨利常数、液相和气相传质系数、膜厚度等。
  - 方法 `solve_interface(P_CO2_bulk, c_CO2_bulk, c_amine_bulk, k2_rate)`：迭代求解界面条件，计算 Hatta 数、瞬间增强因子，最终得到总传质系数、通量和界面浓度。
  - 方法 `film_profile(...)`：调用光谱求解器获得液膜内浓度剖面和通量。
- 函数 `generate_film_grid(n_points, delta, centering
