# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 超构表面相位调控全波仿真与稳健性优化系统

本项目围绕光学超构表面的电磁仿真、优化设计与不确定性分析，构建了一个多模块计算平台。主入口文件为 `main.py`，它串联了以下模块的功能并执行完整流程。实现时需保持 `main.py` 不变，并补全其余所有 Python 模块，使程序能够正常运行且输出合理结果。

## 模块清单与职责概要

| 文件名 | 职责描述 |
|--------|----------|
| `maxwell_fem.py` | 二维横磁（TM）模式亥姆霍兹方程的有限元求解，用于计算单个纳米柱单元的散射场。 |
| `metasurface_grid.py` | 基于重心 Voronoi 镶嵌（CVT）的超构表面纳米柱阵列空间排布优化与参数分配。 |
| `phase_quadrature.py` | 高精度数值积分（Witherden 四边形规则）计算纳米柱截面的有效相位延迟、极化率及波导模式色散。 |
| `multipole_moments.py` | 从散射场或远场数据中提取电磁多极矩（电偶极矩、磁偶极矩等）并计算辐射功率。 |
| `uncertainty_quantify.py` | 使用 Smolyak 稀疏网格和 Gauss‑Hermite 积分进行制造误差传播与灵敏度分析。 |
| `process_sampling.py` | 模拟纳米加工工艺中的随机误差，包括三角形/多边形内均匀采样、高度随机场生成、多变量高斯采样及重采样插值。 |
| `topology_optimize.py` | 基于动态规划的离散相位量化以及压缩感知/贪心方法的逆向设计。 |
| `wavefront_trace.py` | 将超构表面相位分布转化为加权图，用 Dijkstra 算法追踪光线路径与等光程面。 |
| `phase_surface.py` | 基于极小曲面（平均曲率流）理论的相位平滑，支持悬链面、螺旋面相位轮廓生成及曲率能量评估。 |
| `convergence_utils.py` | 数值收敛性分析工具：L²、L∞、H¹ 范数计算，长方体内随机点距离统计，收敛阶估计，Richardson 外推，GCI 指标以及 Monte‑Carlo 收敛检验。 |

## 各模块详细描述

### 1. maxwell_fem.py — 有限元电磁散射求解

**类：** `MaxwellFEM2D`

- **初始化参数：** 波长、硅与空气折射率、PML 宽度与最大电导率。
- **功能：** 在矩形区域内求解 TM 极化亥姆霍兹方程，使用六节点二次三角形单元，稀疏矩阵存储，包含 PML 吸收边界。
- **核心方法：**
  - `build_rectangular_mesh(nx, ny, xlim, ylim)` 返回节点坐标数组 `(node_num,2)` 和单元索引数组 `(elem_num,6)`。
  - `epsilon_profile(x, y, pillar_center, pillar_size)` 返回点 `(x,y)` 处的复相对介电常数。
  - `pml_stretch(x, y, xlim, ylim)` 返回复坐标拉伸因子。
  - 基函数与积分：提供 T6 单元的形状函数及梯度（在参考坐标下），以及三角形上的七点高斯积分规则。
  - `assemble_system(…)` 组装全局刚度矩阵（CSC 格式）和右端项，支持平面波源。
  - `apply_dirichlet_boundary(…)` 在外边界施加简化的一阶吸收条件（强制 E=0）并调整矩阵。
  - `solve_scattering(…)` 完整流程：网格生成 → 组装 → 边界处理 → 稀疏求解 → 返回复电场分布 `E_z`、节点和单元。
  - `compute_transmission_phase(…)` 提取指定 y 坐标线上的透射场相位。

### 2. metasurface_grid.py — CVT 超构表面网格

**类：** `MetasurfaceCVT`

- **初始化参数：** 矩形区域边界（物理尺寸）。
- **功能：** 利用与相位梯度相关的密度函数生成自适应 Voronoi 镶嵌，实现纳米柱排布优化。
- **核心方法：**
  - `density_function(x, y, target_phase_func)` 根据目标相位梯度（数值微分或默认球面波相位）计算采样密度，包含边缘增强。
  - `sample_density(n, …)` 使用拒绝采样生成符合密度的随机点。
  - `voronoi_centroid(generator, samples)` 将样本点分配到最近生成器并计算新重心（Monte‑Carlo 近似）。
  - `compute_cvt(…)` Lloyd 迭代求解 CVT，返回最终生成器坐标和能量历史。
  - `assign_pillar_parameters(…)` 根据目标相位函数为每个生成器分配纳米柱高度和宽度（基于简化的波导模型）。
  - `compute_voronoi_areas(…)` 用 Monte‑Carlo 估计各 Voronoi 单元的面积。

### 3. phase_quadrature.py — 高精度数值积分

**类：** `PhaseQuadrature`

- **初始化参数：** 波长、硅与空气折射率。
- **思路：** 使用 Witherden 四边形积分规则在单位正方形上积分，映射到纳米柱截面矩形。
- **外部函数：** `quadrilateral_witherden_rule(p)` 返回精度阶数 p 对应的积分点数、坐标和权重（内嵌多个预存规则表）。
- **核心方法：**
  - `map_to_pillar(xu, yu, cx, cy, width, height)` 坐标映射。
  - `local_effective_index(…)` 判断点是否在柱内，返回折射率。
  - `integrate_phase_delay(…)` 利用有效折射率计算截面平均相位延迟和传输幅度（包含反射损耗模型）。
  - `integrate_polarizability(…)` 计算等效极化率（Clausius‑Mossotti 型近似）。
  - `integrate_energy_density(…)` 对任意场函数在截面上积分。
  - `compute_dispersion_relation(…)` 将矩形柱等效为平面波导，用数值求根给出前几阶模式的有效折射率列表。

### 4. multipole_moments.py — 多极矩提取

**类：** `MultipoleExtractor`

- **初始化参数：** 波长、背景折射率，并预设物理常数。
- **核心方法：**
  - `extract_dipole_moments(r_obs, E_scat, H_scat)` 在多个观测点上通过最小二乘反演电偶极矩（p）和磁偶极矩（m），使用远场近似表达式。
  - `_levi_civita(i, j, k)` 辅助的 Levi‑Civita 符号。
  - `multipole_moment_method(angular_samples, field_samples, max_order)` 利用角向场采样构建 Hankel 型矩矩阵，通过 Cholesky 分解提取球谐展开系数。
  - `_spherical_harmonic(l, m, theta, phi)` 调用 `scipy.special.sph_harm`。
  - `radiation_powers(p, m, Q_e=None, Q_m=None)` 计算电/磁偶极、四极辐射功率及总功率。

### 5. uncertainty_quantify.py — 制造误差不确定性量化

**类：** `UncertaintyQuantify`

- **初始化参数：** 随机维度数和 Smolyak 最大层数。
- **核心方法：**
  - `hermite_gauss_rule(order)` 返回一维 Gauss‑Hermite 节点和权重（对应 exp(-x²) 权重）。
  - `level_to_order_open(level)` 层到节点数的映射。
  - `sparse_grid_hermite()` 构造多维稀疏网格点与组合权重，处理重复点合并。
  - `propagate_moments(model_func, sigma_params)` 通过稀疏网格积分计算输出值的均值、方差、偏度、
