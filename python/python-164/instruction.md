# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# PEM 燃料电池阴极催化剂层衰减多物理场耦合模拟系统

本项目通过多物理场耦合模拟，研究质子交换膜燃料电池（PEMFC）阴极催化剂层（CCL）在长期运行中的衰减行为。涉及电化学反应动力学、氧气扩散、Pt纳米颗粒奥斯特瓦尔德熟化、碳载体腐蚀、电化学活性表面积（ECSA）损失评估、催化剂负载优化以及形貌退化分析等多个子模型。  
入口文件 `main.py` 集成各模块，按顺序执行完整模拟流程并输出综合结果。  
需要从 `main.py` 的导入关系及调用逻辑出发，补全以下模块（文件）的实现。

## 模块文件及其职责

### 1. `ccl_grid.py` – 操作条件参数网格生成
- 负责生成阴极催化剂层的多维物理参数空间采样网格。
- **核心函数**：
  - `hypercube_grid(m, n, ns, a, b, c)`：在 *m* 维超立方体内生成规则网格，通过逐维直积得到坐标矩阵，支持不同的中心模式。
  - `generate_ccl_parameter_grid()`：预设 5 个物理参数（温度、相对湿度、电池电位、Pt负载、碳比表面积）及其范围，调用上述函数生成网格并返回包含网格、参数名、数目等的字典。
  - `sample_operating_condition(grid_data, index)`：从网格中提取单个操作条件点。

### 2. `butler_volmer.py` – 电化学动力学求解
- 基于 Butler‑Volmer 方程和 Tafel 修正，求解氧还原反应（ORR）的过电位及电流密度。
- **核心函数**：
  - `butler_volmer_current(eta, j0, alpha_a, alpha_c, n, T)`：直接计算给定过电位下的电流密度，包含指数溢出保护和近似处理。
  - `exchange_current_density(T, C_O2, ...)`：根据温度、氧气浓度和材料参数计算修正的交换电流密度（使用 Tafel 公式及相关系数）。
  - `solve_overpotential_muller(...)`：利用阻尼牛顿法求解非线性 Butler‑Volmer 方程，寻找过电位 *η*，包含步长限制和边界保护。
  - `solve_overpotential_wdk(...)`：尝试基于泰勒展开和多项式求根的 Weierstrass‑Durand‑Kerner 法求解（实际退化为 Muller 法以提高鲁棒性）。
  - `orr_kinetic_parameters(T)`：返回阴极 ORR 的典型动力学参数集（传递系数、电子数、参考电流等）。

### 3. `diffusion_solver.py` – 传质扩散方程求解
- 求解 CCL 中的一维稳态扩散‑反应方程（二阶常微分方程），使用有限差分/中心差分离散化为三对角线性系统，并利用直接法求解。
- **核心函数**：
  - `solve_diffusion_tridiagonal(D_eff, k_rxn, L_ccl, C_0, N)`：构建三对角矩阵，设置 Dirichlet/Neumann 边界条件，用 **Thomas 算法**（追赶法）求解浓度分布。
  - `solve_diffusion_banded(...)`：同样方程，使用带状 LU 分解（`r8gb_fa`/`r8gb_sl`）求解，用于验证一致性。
  - `thomas_algorithm(lower, diag, upper, rhs)`：标准追赶法实现。
  - `r83_cr_fa`、`r83_cr_sl`：循环约化法分解与求解（用于三对角系统，可作为 Thomas 的替代，但 main 中未直接使用）。
  - `r8gb_fa`、`r8gb_sl`：一般带状矩阵的 LU 分解和前代/回代求解。
  - `effective_diffusivity(D_bulk, epsilon, tau)`：基于 Bruggeman 修正计算有效扩散系数（依赖孔隙率）。

### 4. `ripening_model.py` – Pt 纳米颗粒奥斯特瓦尔德熟化
- 模拟催化剂层中 Pt 颗粒在 Ostwald 熟化与溶解‑再沉积过程中的粒径分布演化，基于 Gibbs‑Thomson/Kelvin 方程和 LSW 理论。
- **核心函数**：
  - `kelvin_solubility(r, gamma, V_m, T, C_sat_inf)`：计算曲率修正的溶解度。
  - `critical_radius(gamma, V_m, T, C_bulk, C_sat_inf)`：确定熟化临界半径（大于该半径的颗粒长大，反之溶解）。
  - `ripening_rate(r, D, V_m, C_sat_inf, C_bulk, gamma, T)`：计算单个颗粒半径的变化速率，带有物理限幅。
  - `evolve_size_distribution(radii, ...)`：使用显式欧拉法步进演化颗粒尺寸分布，返回整个演化历史。
  - `lsw_analytical_r3(t, r0, ...)`：LSW 理论预测的平均半径随时间的立方根增长。
  - `disk_distance_stats_monte_carlo(radii1, radii2)`：蒙特卡洛方法统计颗粒间距离，用于评估聚集效应。
  - `moment_size_distribution(radii, k)`：计算尺寸分布的第 *k* 阶矩。
  - `pt_dissolution_parameters()`：返回 Pt 溶解/熟化的典型物理参数（界面能、摩尔体积、扩散系数等）。

### 5. `carbon_corrosion.py` – 碳载体腐蚀动力学与传播
- 描述碳腐蚀电化学反应及其引起的碳比表面积空间演化，采用一维守恒律形式，使用多种有限体积/有限差分数值格式求解。
- **核心函数**：
  - `corrosion_current_density(E, ...)`：根据电极电位计算简化的 Tafel 腐蚀电流。
  - `carbon_mass_loss_rate(j_corr, A_carbon)`：基于电流和法拉第定律计算质量损失速率。
  - `corrosion_front_velocity(E, T)`：经验模型估计腐蚀前沿的移动速度。
  - `solve_corrosion_propagation(u0, nx, nt, dx, dt, v_front, k_corr, theta_pore, method)`：离散方程 `du/dt + v·du/dx = -k·u·θ`，提供 **Godunov**、**Lax‑Wendroff**、**MacCormack** 三种格式，包含 CFL 条件检查和自动步长调整。
  - `numerical_flux_godunov(u_left, u_right, v)`：Godunov 迎风通量。
  - `structural_integrity_loss(S_c_current, S_c_initial)`：计算结构完整性损失比例。

### 6. `ecsa_calculator.py` – ECSA 损失与稳定性分析
- 计算基于颗粒尺寸分布的电化学活性表面积，评估多种衰减机制下的 ECSA 演化，并利用雅可比矩阵主导特征值进行系统稳定性分析。
- **核心函数**：
  - `ecsa_from_size_distribution(radii, rho_pt)`：由粒径列表计算单位质量活性表面积（m²/g_Pt）。
  - `total_ecsa_loss_model(t_hours, ECSA0, params)`：综合溶解‑熟化（立方根律）、碳腐蚀（指数衰减）和毒化（线性衰减）三种效应的 ECSA 损失模型。
  - `voltage_loss_from_ecsa(ECSA_ratio, b_tafel)`：由 ECSA 下降估算 Tafel 电压损失。
  - `ecsa_loss_kinetics(ECSA, t, k1, k2)`：简单的两次方动力学解析解。
  - `build_stability_jacobian(n_species, rate_constants, interaction_matrix)`：组装描述衰减系统局部线性化的雅可比矩阵。
  - `power_method_eigenvalue(A, y0, ...)`：幂法迭代计算矩阵主导特征值及特征向量。
  - `stability_analysis_max_eigenvalue(J)`：根据主导特征值判断系统稳定性（稳定/不稳定/临界）。

### 7. `catalyst_optimizer.py` – 催化剂负载优化
- 使用单变量优化
