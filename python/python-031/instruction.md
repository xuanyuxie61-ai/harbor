# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：中子星 crust 核 pasta 相计算平台

## 项目目标
本项目模拟中子星外壳层（crust）中核物质在强库仑相互作用与表面张力竞争下形成的非均匀“核 pasta”结构。通过整合多项数值算法，计算五种 pasta 相的几何性质、库仑能、振动模式、相图以及冷却/反应扩散动力学，从而为研究中子星 crust 物理提供统一计算平台。

## 项目文件与职责
项目由以下模块组成。`main.py` 作为主入口，依次调用各模块完成全部计算流程并输出结果；其余文件提供核心物理模型和数值方法。

### `main.py`
项目入口，零参数可运行。组织 8 个计算环节：
- 状态方程（`nuclear_eos`）
- 几何建模（`geometry_pasta`）
- 库仑势求解（`coulomb_solver`）
- 反应扩散模拟（`reaction_diffusion`）
- ODE 温度演化（`ode_integrator`）
- 柱坐标本征模式（`bessel_modes`）
- CVT 采样优化（`cvt_sampler`）
- 相图与稳定性分析（`phase_diagram`）

### `nuclear_eos.py`
核状态方程模块。基于 Skyrme 能量密度泛函计算均匀核物质的性质。
- **数据结构/常量**：`SKYRME_PARAMS` 字典（Skyrme 参数集），物理常数 `HBAR2_2M`、`M_NUCLEON`。
- **核心函数**：
  - `skyrme_energy_density(rho_n, rho_p, params)`：返回能量密度与压强。
  - `nuclear_matter_properties(rho, x_p, params)`：返回字典，包含每核子能量、压强、对称能、不可压缩系数。
  - `parameter_uncertainty_t_stat(estimates, true_value, std_errors, confidence)`：利用非中心 t 分布进行参数不确定性分析（融入 asa243 算法）。
- **依赖**：`numpy`, `scipy.special.gammaln`, 自定义的正态分布函数 `alnorm`。

### `geometry_pasta.py`
几何结构与积分规则模块。定义五种 pasta 相的类并计算几何特征，集成多种多面体积分规则。
- **类**：`PastaPhase`（基类）及其子类 `GnocchiPhase`、`SpaghettiPhase`、`LasagnaPhase`、`AntiSpaghettiPhase`、`AntiGnocchiPhase`。每个子类依据填充率 `u` 计算特征尺度（半径/厚度）、表面积、体积，并实现 `coulomb_factor` 方法。
- **工厂函数**：`create_pasta_phase(phase_id, density, proton_fraction, u)`。
- **积分工具**：单位三角形、单位楔形、单位四面体上的单项式积分，以及六边形 Stroud 积分规则（1-, 3-, 5-, 7-阶）。还包括四边形双线性插值。
- **其他函数**：`pasta_energy_landscape` 扫描密度范围计算各相的能量景观。

### `coulomb_solver.py`
库仑势有限元求解模块。使用二维三角形单元（T3）求解泊松方程，获得电势分布和库仑能。
- **有限元函数**：
  - `basis_mn_t3`：计算三角形单元上基函数及其导数。
  - `assemble_stiffness`：遍历单位组装稀疏刚度矩阵与载荷向量，使用单点积分，载荷项包含质子密度函数。
  - `apply_dirichlet_bc`：施加 Dirichlet 边界条件，修改矩阵和右端项。
  - `solve_poisson_fem`：组装、施加边界、调用 `spsolve` 求解，并积分计算库仑能。
- **辅助函数**：
  - `analytical_coulomb`：基于几何因子和修正系数的解析库仑能近似，作为有限元失败时的回退。
  - `wigner_seitz_coulomb`：构造简单极坐标网格，利用 `geometry_pasta` 创建 pasta 相密度分布，调用 FEM 求解器。
- **依赖**：`geometry_pasta`、`scipy.sparse`。

### `bessel_modes.py`
柱坐标本征模式与贝塞尔函数零点计算模块。
- **零点计算**：`besselj_zero(n, nt)` 使用 Newton 迭代求解 J_n(x) 的前 nt 个零点，初始猜测根据 n 大小采用不同经验公式。
- **柱对称电势**：`cylinder_coulomb_potential(r, R_cyl, rho_p, n_modes)` 对 J_0 零点展开求和，限制在柱内区域。
- **振动频率**：`cylinder_vibration_frequencies(R_cyl, sigma, rho, n_modes)` 计算 Rayleigh 不稳定性下的柱相振动频率。
- **其他电势**：`spherical_coulomb_potential`（球对称精确解）、`sheet_coulomb_potential`（片状相）。
- **形变能**：`pasta_deformation_energy` 根据相类型和形变幅度计算表面形变能。

### `cvt_sampler.py`
质心 Voronoi 镶嵌采样（CVT）与多维积分模块。
- **2D CVT**：`cvt_2d_lumping` 实现 Lloyd 迭代，基于用户提供的密度函数在 [-1,1]^2 上生成 CVT 点集，返回能量和位移历史。
- **球面 CVT**：`sphere_cvt_step` 和 `voronoi_areas_direct` 提供简化版球面 CVT 步骤和面积计算，依赖凸包或采样近似。
- **多维积分**：`nd_integrand_gaussian`、`nd_integrand_coulomb` 等被积函数，以及 `monte_carlo_nd_integral` 通用蒙特卡洛积分器。
- **pasta 优化**：`optimize_pasta_cvt` 结合 `geometry_pasta` 创建相并利用 CVT 优化核团空间分布，返回生成点和面积分布。

### `reaction_diffusion.py`
核子反应-扩散动力学模块。
- **β 衰变率**：`beta_decay_rates(temperature, rho_n, rho_p, electron_chemical_potential)` 使用费米积分近似计算正 / 逆 β 衰变率。
- **扩散系数**：`diffusion_coefficient` 采用 Stokes-Einstein 公式。
- **一维求解器**：`fd_reaction_diffusion_1d` 采用显式 Euler、中心差分处理扩散，支持 Neumann/Dirichlet/periodic 边界，自动调整时间步长以满足 CFL 条件并记录历史。
- **二维有限元求解器**：`fe_reaction_diffusion_2d` 使用质量集中和显式步进，通过简化刚度矩阵组装（基于面积与梯度近似）推进密度场。
- **熵产生**：`entropy_production_rate` 计算不可逆过程的熵产生。

### `ode_integrator.py`
温度演化与结构动力学 ODE 积分模块。
- **中子星冷却**：`crust_cooling_ode` 计算温度 T 和累积加热 Q 的导数，使用中微子发光度（修正 Urca 过程）和简并费米气体比热。`solve_crust_cooling` 封装求解流程。
- **不稳定 ODE**：`unstable_exact`、`unstable_deriv`、`solve_unstable_system` 提供精确解与数值解对比，用于测试。
- **刚性 ODE**：`tough_deriv` 和 `solve_tough_system` 定义并求解一个四维刚性系统。
- **相变动力学**：`phase_transition_kinetics` 实现 Ginzburg-Landau 型的序参量演化方程。

### `phase_diagram.py`
相图与稳定性分析模块。综合各物理组分计算 pasta 相的总能量，并生成相图、评估稳定性。
- **表面张力**
