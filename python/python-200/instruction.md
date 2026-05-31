# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：基于自动微分的分子动力学高阶导数计算与多尺度分析

本项目是一个高性能计算框架，用于从原子间势能出发，通过自动微分、谱方法、有限元、弹性力学和热力学分析，构建多尺度材料响应预测系统。项目包含多个相互依赖的 Python 模块，统一入口为 `main.py`（保留），其余模块需要根据本描述复现。

## 项目结构与模块概览

| 文件名 | 主要职责 |
|--------|----------|
| `autodiff_core.py` | 前向模式自动微分引擎，支持一阶导数（Dual Number）和二阶导数（Hyper‑Dual Number） |
| `potential_models.py` | 分子间势能模型（Lennard‑Jones、高斯势），提供与自动微分兼容的势能函数 |
| `md_engine.py` | 分子动力学模拟引擎，基于 Velocity Verlet 积分，包含温度控制、边界条件、统计量收集 |
| `chebyshev_spectral.py` | Chebyshev 谱插值与微分，支持谱精度求解微分方程边值问题 |
| `cvt_optimizer.py` | Centroidal Voronoi Tessellation (CVT) 采样优化，用于非均匀密度节点分布 |
| `biharmonic_elasticity.py` | 一维双调和方程有限差分解法，用于纳米梁/薄膜的弹性变形分析 |
| `lattice_geometry.py` | 晶格几何描述、六方网格边界追踪、Diophantine 方程求解、晶格构造 |
| `sampling_methods.py` | 多维采样方法（Latin Hypercube、Latin Center、三角形网格、分层采样、准随机序列） |
| `tetrahedral_fem.py` | 四面体网格生成与有限元体积积分，用于从原子坐标到连续密度场的映射 |
| `thermodynamics.py` | 热力学量计算：比热、弹性常数、径向分布函数、热膨胀系数、熵 |
| `utils.py` | 通用工具：计时器、直方图统计、数值鲁棒性辅助、矩阵工具 |

## 模块详细说明

### 1. `autodiff_core.py` – 自动微分引擎

**核心思想**：基于 Dual Number 的前向模式自动微分，通过重载算术运算和初等函数，在前向传播中同时获得函数值和精确导数。引入 Hyper‑Dual Number 实现二阶导数。

**主要数据结构**：
- `DualScalar`：包含 `val`（函数值）和 `der`（一阶导数）两个字段，支持 `+,-,*,/,**,neg` 以及初等函数（`sin, cos, exp, log, sqrt, abs, min`）。
- `HyperDualScalar`：用于二阶导数，字段为 `f0, f1, f2, f12`，分别对应原值、对第一方向的导数、对第二方向的导数、混合偏导数，同样支持算术运算和初等函数。

**关键函数**：
- `dual_sin`, `dual_cos`, `dual_exp`, `dual_log`, `dual_sqrt`, `dual_abs`, `dual_min` 等对 `DualScalar` 的数学函数。
- 对应的 `hdual_*` 系列函数用于 `HyperDualScalar`。
- 向量级接口：`grad_scalar_func_ad`（按分量构造 dual 计算梯度）、`directional_derivative_ad`（一次性计算方向导数）、`mixed_partial_hyperdual`（混合偏导数）、`hessian_scalar_func_fd`（有限差分 Hessian，备用）、`jacobian_vector_func`（有限差分 Jacobian）。

**实现要求**：
- 实现 `DualScalar` 和 `HyperDualScalar` 类，正确重载算术运算，严格按照代数规则例如 `(a+εa')·(b+εb') = ab + ε(ab'+a'b)`。
- 提供数学函数，应用链式法则。
- 实现梯度、方向导数、混合偏导数的计算函数，利用 dual 数传播。

### 2. `potential_models.py` – 势能模型

**核心功能**：提供 Lennard‑Jones 12‑6 势、高斯修正势，以及组合势能，同时支持标量、Dual 和 Hyper‑Dual 版本，以便与自动微分引擎无缝集成。

**关键函数**：
- `lennard_jones_potential(r, epsilon, sigma)`：标量版本。
- `lennard_jones_force(r, epsilon, sigma)`：解析导数。
- `lennard_jones_dual(r, epsilon, sigma)`：返回 `DualScalar`，在 r 接近零时进行截断处理。
- `lennard_jones_hyperdual(r, epsilon, sigma)`：返回 `HyperDualScalar`。
- `gaussian_potential_2d` 和 `gaussian_potential_dual`：二维各向异性高斯势。
- `total_potential_lj(positions, epsilon, sigma, rcut, box_size)`：计算 N 粒子体系的总 LJ 势能，含平滑截断函数。
- `total_forces_lj`：解析力计算。
- `total_potential_with_gaussian`：组合势能。
- `virial_stress_lj`：维里应力张量。

**实现要求**：
- 正确实现 LJ 公式及其导数，包括截断处理。
- 支持周期性边界条件（通过最小镜像约定）。
- Dual 版本需利用 `autodiff_core` 中的类型和函数构造表达式。

### 3. `md_engine.py` – 分子动力学引擎

**核心功能**：Velocity Verlet 积分器、Berendsen 弱耦合热浴、周期性边界条件、能量/温度/压强追踪。

**主要类**：`MDEngine`
- 初始化参数：粒子数、维度、质量、时间步长、盒子尺寸、LJ 参数、截断半径、目标温度、热浴耦合常数。
- 状态：`pos, vel, acc, force`。
- 方法：
  - `initialize_positions_lattice(lattice_type)`：正方或六方晶格初始化。
  - `initialize_velocities_maxwell_boltzmann()`：按 Maxwell‑Boltzmann 分布初始化速度，并去除质心速度。
  - `apply_periodic_boundary()`：周期性边界条件。
  - `compute_forces_and_energies()`：调用 `potential_models` 计算力和势能，动能。
  - `compute_temperature(kinetic)`：从动能计算瞬时温度（k_B=1）。
  - `compute_pressure(potential, kinetic)`：结合维里项计算压强。
  - `berendsen_thermostat(current_temp)`：速度缩放因子。
  - `velocity_verlet_step(apply_thermostat)`：执行一个时间步，包含位置更新
