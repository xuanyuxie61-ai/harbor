# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：三维随机 Fisher-KPP 方程的数值积分、不确定性量化与参数反演

本项目是一个用于求解一维随机 Fisher‑KPP（反应‑扩散‑对流）方程的计算数学工具集。主要功能包括：空间自适应网格生成、随机 Wiener 过程的截断近似、多种确定性/随机时间积分格式、蒙特卡洛不确定性量化、无梯度参数标定以及基于局部物理特征的策略选择器。所有模块均用 Python 编写，主入口点为 `main.py`，其余文件实现各子模块，在 benchmark 中需被重新实现。

---

## 文件清单与职责

### 1. `main.py` — 统一实验入口
- 包含一组自包含实验，直接调用其他模块以演示和验证所有功能。
- 实验 1：单条随机路径求解，对比 Euler‑Maruyama、Platen 强 1.0 阶显式 SRK 和 Milstein 方法。
- 实验 2：蒙特卡洛不确定性量化，使用分层采样与反变量方差缩减，计算期望、方差、RMSE 及 Gelman‑Rubin R‑hat 统计量，并输出层次距离矩阵。
- 实验 3：无梯度优化验证——先对 Rosenbrock 函数测试 PRAXIS 优化器，再用作 SPDE 扩散系数 ε 的反演。
- 实验 4：自适应数值格式选择策略，基于解场的局部 Peclet 与 CFL 数推荐全局格式。
- 实验 5：Fibonacci 螺旋圆盘采样与 CVT Lloyd 网格优化演示。
- 最后生成文本摘要报告。
- 依赖所有其他 `.py` 文件（除 `strategy_selector` 等可直接被调用外，多数通过显式导入使用）。

### 2. `mesh_generation.py` — 网格生成与节点优化
- 提供一维与二维空间网格生成方法，用于 SPDE 的空间离散。
- **核心函数：**
  - `cvt_lloyd_1d(n, a, b, density, it_num, tol)`  
    在一维区间上执行重心 Voronoi 网格（CVT）Lloyd 迭代。根据密度函数调整节点分布，直至收敛。使用 Simpson 法则或梯形法则数值计算 Voronoi 区域的质心。
  - `fibonacci_spiral_disk(n, R)`  
    在半径为 R 的圆盘上通过 Fibonacci 螺旋生成拟均匀分布点，基于黄金比例角度增量。
  - `tetrahedron_grid(n, vertices)` 与 `tetrahedron_grid_count(n)`  
    在给定四面体上生成规则重心坐标网格，支持三维情形（本项目仅用于演示）。
  - `adaptive_density_function(x, steepness, center)`  
    指数衰减密度函数，用于在指定中心附近加密网格。
  - `generate_composite_mesh_1d(n_base, a, b, steepness, center)`  
    组合以上工具，生成一维自适应复合网格：先运行 CVT 获得优化节点，然后可根据需要进一步加密（此处直接返回 CVT 节点）。
- 可用性：`main.py` 中实验 1、2、3、4 使用 `generate_composite_mesh_1d` 生成空间点集；实验 5 直接调用 `fibonacci_spiral_disk` 和 `cvt_lloyd_1d`。

### 3. `wiener_process.py` — 时空 Wiener 过程与随机数生成
- 构造截断 Q‑Wiener 过程，用于驱动 SPDE 中的随机扰动。
- **主要类：**
  - `LEcuyerRNG(seed1, seed2)`  
    组合多重递归伪随机数生成器，提供 `uniform` 和 `gaussian`（Box‑Muller）采样，支持反变量缓存。
  - `QWienerProcess(spatial_grid, n_modes, alpha, sigma, rng, use_antithetic)`  
    在空间网格上构造有限维 Karhunen–Loève 展开：$$W_N(t,x)=\sum_{k=1}^{N} \sqrt{q_k}\, e_k(x)\,\beta_k(t)$$  
    其中特征函数为正弦基（对应 Dirichlet 边界），特征值按幂律 $$q_k = \sigma^2 k^{-2\alpha}$$ 衰减。  
    - `increment(dt)` 生成时间步内的 Wiener 增量，返回长度为网格点数的向量。  
    - 支持反变量法（`use_antithetic=True`）以减小方差。  
    - 提供强误差估计和谱截断误差的近似公式（用于分析而非必需计算）。
- `spde_core.py` 和 `monte_carlo_uq.py` 中的采样回路均依赖此类生成随机扰动。

### 4. `spatial_operators.py` — 空间微分算子离散化
- 实现非均匀网格上的一维对流‑扩散‑反应算子，并提供有限元质量/刚度矩阵装配以及 DG 风格数值通量。
- **主要类：**
  - `SpatialDiscretization1D(x, epsilon, velocity, reaction_rate, carrying_capacity)`  
    存储空间网格和物理参数，并计算局部 Peclet 数（$$Pe_i = |v| h_i/\varepsilon$$）。
    - `diffusion_operator(u)`：非均匀网格上的二阶中心差分扩散项。
    - `advection_operator(u, scheme)`：依据局部 Peclet 数在中心差分、迎风、Lax‑Wendroff（附人工粘性）之间自动选择格式（`"auto"`）。
    - `reaction_operator(u)`：Fisher‑KPP 反应项 $$r u (1-u/K)$$，并对负值做截断保护。
    - `full_rhs_deterministic(u, scheme)`：返回上述三项的代数和（扩散－对流＋反应）。
    - `assemble_fem_matrices()`：返回 lumped‑mass 质量矩阵和迎风稳定 Galerkin 刚度矩阵（均为 dense 形式），用于半隐式时间推进。
    - `dg_numerical_flux(u_left, u_right)`：间断 Galerkin 风格的局部 Lax‑Friedrichs 通量。
- `spde_core.py` 在构造漂移函数和预计算隐式线性部分时依赖此类；`parameter_calibration.py` 的 solver_factory 也会实例化该类。

### 5. `stochastic_rk.py` — 随机 Runge‑Kutta 方法与自适应步长
- 实现多种 Itô SDE 的时间步进格式，并提供自适应步长控制。
- **独立函数：**
  - `sde_euler_maruyama_step(y, f, g, h, dW)`：标准 Euler‑Maruyama 步。
  - `sde_srk_platen_step(y, f, g, h, dW)`：显式强 1.0 阶 Platen 格式，使用两个试探点。
  - `sde_milstein_step(y, f, g, dg, h, dW)`：需要扩散函数的导数，强 1.0 阶。
  - `stiff_sde_semiimplicit_step(y, f_lin, f_nonlin, g, h, dW)`：将线性漂移部分隐式处理，其余显式，求解线性系统。
  - `adaptive_rk12_sde_step(y, f, g, h, dW, tol)`：基于 Euler‑Maruyama 与 Heun 嵌入对计算局部误差，自动调整步长，返回新步长、接受标志和新解。
- **类 `StochasticIntegrator`**  
  封装上述方法选择与步长参数，`spde_solver.py` 通过该类的实例调用单步操作。

### 6. `spde_core.py` — SPDE 核心求解器
- 将空间算子、Wiener 过程和时间积分器组合，求解一维随机 Fisher‑KPP 方程。
- **类 `SPDESolver1D(spatial, wiener, integrator, sigma_noise, dirichlet_bc, neumann_bc)`**
  - 预计算 FEM 矩阵，构造隐式线性漂移矩阵供半
