# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：水声传播宽角抛物方程（WAPE）建模系统

本项目是一个面向深海复杂环境的水声传播数值模拟系统，基于宽角抛物方程（Wide‑Angle Parabolic Equation, WAPE）方法。系统由 `main.py` 统一调度，其余模块分别负责环境建模、网格生成、声源初始化、边界条件、PE 求解器、模态分析、散射与混响及传播损失后处理。运行 `main.py` 即可完成从参数设置到结果输出的全流程。

## 文件结构

### main.py
主入口脚本。依次调用各模块完成以下流程：
1. 创建 `OceanEnvironment` 实例，设置海水声速剖面、吸收、海底地形等参数。
2. 生成深度和水平计算网格，构造 `PEMesh` 对象。
3. 生成初始声源场并归一化。
4. 初始化边界条件处理器 `BoundaryConditionHandler`。
5. 使用 `ParabolicSolver` 进行场步进求解。
6. 进行简正波分析（`NormalModeAnalyzer`）和模态约束验证。
7. 计算体积散射与混响（`VolumeScatteringModel`, `ReverberationModel`）。
8. 计算传播损失（`PropagationLoss`）、接收器阵列响应、收敛区与声影区，并分析多径统计。
9. 进行空间相关性分析（`SpatialCorrelation`）。
10. 验证若干数值工具（多项式转换、三角积分、特殊函数等）。

### environment.py
定义海洋环境类 `OceanEnvironment`，用于存储和查询声学环境参数。
- 核心属性：声速剖面参数（Munk 模型）、频率、吸收系数、海底参数、密度等。
- 主要方法：
  - `sound_speed(z)`：返回 Munk 声速剖面。
  - `absorption_db_per_km(f_khz)` / `absorption_np_per_m(f_khz)`：Thorp 吸收公式。
  - `wavenumber(z)`：复数波数 k(z) = ω/c(z) + iα。
  - `refractive_index(z)` / `refractive_index_squared_deviation(z)`：折射率及其偏差。
  - `density(z)` / `impedance(z)`：海水密度和声阻抗。
  - `bathymetry(r)`：根据参数化模型（tanh 斜坡 + 高斯山丘）计算海底深度。
  - `seabed_reflection_coefficient(theta)`：Rayleigh 反射系数。
- 工具函数 `safe_divide`：防除零的除法。

### mesh_builder.py
生成 PE 求解所需的计算网格，并管理水域掩码。
- 函数：
  - `generate_depth_grid(z_max, nz, stretch_power, z_axis)`：生成非均匀深度网格，使用拉伸映射，可选声道轴附近加密。
  - `generate_range_grid(r_max, dr)`：均匀水平网格。
  - `point_in_polygon(x_poly, y_poly, x0, y0)`：射线交叉算法判断点是否在多边形内（用于水域掩码）。
- 类 `PEMesh`：
  - 初始化时根据 `r_grid`、`z_grid` 和 `OceanEnvironment` 构建二维网格，计算每个水平步的海底深度，生成节点掩码（`node_mask`，标记有效水域节点），并构建三角形单元列表（用于后处理）。
  - 方法：`get_1d_slice(m)`、`global_index(m,n)`、`local_index(idx)`、`adaptive_range_step(m)`、`mesh_quality_stats()`。

### source_field.py
生成抛物方程的初始声场 `u(0,z)`。
- 主要函数：
  - `gaussian_starter(z, z_s, w0, k0, R_c)`：高斯束源。
  - `green_starter(z, z_s, k0)`：基于 Hankel 函数远场近似的格林函数源。
  - `directional_factor(theta, ka)`：圆形活塞方向性因子。
  - `sinc_interpolate(z_query, z_grid, u_grid)`：基于归一化 sinc 的带限插值。
  - `build_initial_field(z_grid, z_s, source_type, **kwargs)`：根据类型构建初始场。
  - `source_power_normalization
