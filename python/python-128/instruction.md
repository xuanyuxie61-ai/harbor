# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：三维肿瘤细胞趋化迁移多尺度计算框架（python‑128）

## 1. 项目概述

本项目构建了一个用于模拟 **肿瘤细胞在三维趋化因子浓度场中的迁移行为** 的多尺度计算框架。框架整合了 网格生成、反应‑扩散‑对流偏微分方程求解、个体细胞动力学（椭球几何、周期细胞相位、趋化敏感性）、蒙特卡洛采样、高阶求积、降阶建模（POD/SVD）以及自适应空间采样与周期信号插值 等多个模块。最终由 `main.py` 统一编排执行，生成数值结果摘要。

后续将保留 `main.py`，删除其他所有 `.py` 文件。本描述用于指导重新实现这些被删除的模块，使 `main.py` 可以正确运行。

## 2. 文件清单与职责概览

| 文件名                 | 主要职责                                                         |
|------------------------|------------------------------------------------------------------|
| `mesh_engine.py`       | 三维四面体网格数据结构、均匀长方体网格生成、网格细化、边界提取、Gmsh 格式导出 |
| `chemotaxis_solver.py` | 三维趋化因子反应‑扩散‑对流方程的数值求解器（方向分裂、隐式扩散、迎风对流） |
| `cell_dynamics.py`     | 椭球形细胞个体（含形状、周期相位）及其迁移 ODE，以及群体统计与敏感性分析 |
| `monte_carlo_sampler.py` | 在椭球体内进行均匀蒙特卡洛采样，估计细胞体积与受体‑配体结合概率   |
| `quadrature_rules.py`  | 单位棱柱上的高阶对称求积规则，以及细胞‑ECM 接触力学积分和细胞内平均浓度计算 |
| `rom_analysis.py`      | 基于 SVD 的 POD 降阶模型，用于浓度场快照分解、重构与 Galerkin 投影 |
| `adaptive_grid.py`     | 基于非均匀密度 CVT 的自适应采样器，以及周期性信号的三角基插值       |
| `cell_cycle.py`        | 细胞周期相位模型（G1/S/G2/M），包括循环置换、连续 Markov 链推进和群体敏感性权重 |
| `special_math.py`      | Jacobi 椭圆函数（AGM 法）、CORDIC 三角函数、三对角线性系统求解器 |

其中 `main.py` 是唯一的入口脚本，它会导入上述各模块的公共接口并依次执行各个步骤。

## 3. 模块详细说明

### 3.1 `mesh_engine.py` – 四面体网格

**功能**  
提供三维四面体网格的创建、查询、细化、边界提取以及与 Gmsh 格式的相互转换。

**关键数据结构**  
- `TetrahedralMesh` 类：
  - 属性 `nodes` (N×3 数组) 和 `elements` (M×4 整数数组，0‑based 索引)。
  - 方法 `element_volume(idx)` 计算单个四面体有向体积的绝对值。
  - 方法 `total_volume()` 累加所有单元体积。
  - 方法 `refine_uniform()` 执行中点剖分一致细化（每个四面体拆为 8 个子四面体）。
  - 方法 `compute_centroids()` 返回每个单元质心。
  - 方法 `compute_boundary_faces()` 使用面计数法提取位于网格表面的三角形面片。

**工具函数**  
- `generate_uniform_box_mesh(xlim, ylim, zlim, nx, ny, nz)` 生成一个长方体区域内均匀的六面体剖分，并将每个六面体分解为 6 个四面体，返回 `TetrahedralMesh` 实例。
- `gmsh_format_string(mesh)` 将网格对象序列化为 Gmsh ASCII 格式的字符串（版本 2.2，三角形类型号为 4）。
- `parse_stl_like_surface(nodes, faces)` 模拟 STL 表面三角形输入的去重与索引标准化。

**依赖**  
仅使用 `numpy`，无项目内模块依赖。

### 3.2 `chemotaxis_solver.py` – 趋化因子 PDE 求解器

**功能**  
在三维长方体区域上求解趋化因子浓度 \(c(\mathbf{x},t)\) 满足的反应‑扩散‑对流方程：

\[
\frac{\partial c}{\partial t} = D \nabla^2 c - \mathbf{v}\cdot\nabla c + R(c) - \lambda c,
\]

其中 \(R(c) = V_{\max}\frac{c}{K_m + c}\) 为 Michaelis–Menten 产生项，\(\lambda\) 为降解速率。

**核心类**  
- `ChemotaxisSolver`：
  - 构造函数接收网格点数、区域范围、扩散系数 \(D\)、降解常数 \(\lambda\)、\(V_{\max}\)、\(K_m\) 等。
  - 内部存储三维浓度数组 `c`，以及坐标向量。
  - 方法 `set_initial_condition(c0_func)` 根据可调用函数初始化浓度场。
  - 方法 `step(vx, vy, vz, dt=None)` 执行一个时间步。采用方向分裂（Strang 型）：依次在 x、y、z 方向先隐式向后 Euler 求解扩散，再一阶迎风处理对流，最后半隐式处理反应源项与降解。当 `dt` 为 `None` 时自动根据 CFL 和扩散约束计算安全时间步长。
  - 方法 `gradient()` 用中心差分（边界为单侧差分）计算浓度梯度 \(\nabla c\)，返回三个分量数组。
  - 方法 `total_mass()` 近似积分得总质量。
  - 私有方法：`_reaction_source`、`_safe_dt`、`_solve_1d_diffusion_implicit`（利用三对角求解器）、`_advection_step_leapfrog_1d`（实为迎风格式）。

**依赖**  
- `numpy`
- `special_math.tridiag_solve` — 用于隐式扩散时的三对角系统求解。

### 3.3 `cell_dynamics.py` – 细胞迁移动力学与几何

**功能**  
定义单个椭球形细胞代理 `CellAgent` 和细胞群体 `CellPopulation`，并实现椭球几何量的计算（表面积、体积）。

**关键类与函数**  

- `CellAgent`：
  - 属性：位置 `position` (3‑向量)，形状 `shape` (三元组 `(a,b,c)` 半轴，已降序排列)，细胞周期相位 `phase` (整数 0–3)，速度 `velocity`，趋化敏感系数 `sensitivity`。
  - `chemotaxis_velocity(grad_c, mu, gamma)` 计算饱和趋化速度 \(\mu \frac{\nabla c}{1+\gamma |\nabla c|}\)。
  - `ecm_drag(ecm_density_func, beta)` 根据局部 ECM 密度返回阻力系数。
  - `stochastic_component(sigma=0.05)` 生成高斯白噪声。
  - `step(grad_c, dt, ecm_density_func, ...)` 整合速度项推进位置。

- `CellPopulation`：
  - 管理 `n_cells` 个 `CellAgent`，提供平均位置、扩散度（位置标准差）、总表面积、总体积等统计。
  - `step_all(grad_c_func, dt, ...)` 对全体细胞执行一步迁移。
  - `sensitivity_analysis(...)` 对代表性细胞施加微小初始扰动，返回 L2 偏差随时间的变化，用于评估初值敏感性。

- `ellipsoid_surface_area_rudolf(a,b,c,p=1.6075)` 用 Knud‑Thomsen 近似公式估算椭球表面积。
- `ellipsoid_surface_area_elliptic(a,b,c)` 通过数值积分计算第一、二类不完全椭圆积分，得到更精确的表面积（需要实现辛普森积分，但不依赖外部椭圆积分函数可直接计算）。
- `ellipsoid_volume(a,b,c)` 直接计算椭球体积 \(\frac{4}{3}\pi a b c\)。

**
