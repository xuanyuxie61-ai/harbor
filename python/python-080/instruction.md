# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 项目描述：激光诱导空化气泡多物理场模拟框架

本项目的目标是通过 Python 实现一个博士级多物理场计算框架，用于模拟激光诱导空化气泡在近壁面处的非球形崩溃动力学。程序主入口为 `main.py`，其余模块文件需要根据本描述自行实现。所有模块文件分别为：

- `utils.py`
- `rayleigh_plesset_solver.py`
- `bubble_shape_deformation.py`
- `surface_integrals.py`
- `fem_pressure_wave.py`
- `rbf_pressure_field.py`
- `nucleation_statistics.py`
- `nonlinear_coupling.py`
- `energy_dissipation.py`

`main.py` 调用上述模块，完成参数设置、问题求解与结果输出。**你的任务**：仅保留 `main.py`，删除其他所有 `.py` 文件，然后根据本描述为每个模块重新实现缺失的代码，使得 `main.py` 可以无错误运行。

---

## 模块职责与边界

### 1. `utils.py` – 物理常数与通用数值工具

该模块提供：

- 一组 SI 单位制下的物理常数，如水的密度、粘度、表面张力、声速、蒸汽压、环境压力、通用气体常数、玻尔兹曼常数等。
- 数值稳定性函数 `safe_divide(a, b, default=0.0)`，用于避免除零错误。
- 矩阵分解与求解工具，包括 Cholesky 分解 `r8po_fa` 和对应的回代求解 `r8po_sl`，用于后续椭球采样。
- 均匀采样函数：
  - `uniform_in_sphere01_map(m, n)` – 在 m 维单位超球体内均匀采样 n 个点。
  - `disk01_sample(n)` – 在单位圆盘内均匀采样 n 个二维点。
  - `ellipsoid_sample(m, n, a_mat, v, r)` – 从 m 维椭球内部均匀采样，椭球由矩阵 `a_mat`、中心 `v` 和半径 `r` 定义。
- 单项式求值函数 `monomial_value(m, n, e, x)` – 对于一组点计算单项式 ∏ x_i^{e_i} 的值。
- 简单的矩阵打印函数 `print_matrix` 和时间戳输出函数 `timestamp`。

该模块仅提供基础工具，不涉及物理模型。

---

### 2. `rayleigh_plesset_solver.py` – 气泡壁运动方程与临界半径

该模块实现气泡动力学核心 ODE 系统：

- 两种气泡壁运动方程的右端函数：
  - **Rayleigh‑Plesset 方程**（不可压缩假设），包含惯性项、压力差、表面张力、粘性耗散。
  - **Keller‑Miksis 方程**（可压缩液体修正），增加声辐射和声速相关项。
- 气泡内部气体的 **Van der Waals 状态方程**以及绝热近似；同时包含 **Plesset‑Zwick 热传导修正**，用于计算气泡内气体温度的变化。
- ODE 求解器 `solve_rayleigh_plesset`，它封装 `scipy.integrate.solve_ivp`，根据参数选择使用 RP 或 KM 方程，返回时间历程解对象。
- 临界空化核半径的二分法搜索 `critical_nucleation_radius_bisection`，依据 Blake 临界条件（压力平衡与表面张力）在给定区间内寻根。
- 稳态非线性系统的定义与求解：
  - `nonlinear_bubble_residue` 构造稳态残差（半径、速度、温度、气体摩尔数）。
  - `nonlinear_bubble_jacobian` 用数值差分计算该残差的 Jacobian 矩阵。
  - `solve_steady_state_newton` 使用 Newton 法求解气泡的稳态解，并包含边界保护（半径、温度、摩尔数非负）。

本模块与 `utils.py` 紧耦合，使用其中的物理常数和安全除法。其返回的稳态解与时间历程被 `main.py` 用于后续分析。

---

### 3. `bubble_shape_deformation.py` – 非球形变形与混沌破碎

该模块处理气泡形状偏离球形的效应：

- **椭圆变形映射**：将一个正定对称矩阵 `A` 作用于单位圆，得到变形的气泡截面。函数 `bubble_ellipse_shape` 生成变形边界点；`ellipse_condition_number` 计算矩阵条件数，衡量变形程度。
- **Legendre 模式扰动**：利用 Legendre 多项式展开描述微小非球形扰动。
  - `deformation_velocity_potential` 计算表面速度势（仅表面处近似）。
  - `mode_amplitude_odes` 构建高阶模式振幅的 ODE 系统，包含表面张力恢复力、粘性阻尼、与径向运动的耦合项。ODE 状态向量包含半径、径向速度以及多个模式振幅及其时间导数。
- **混沌微团破碎**：使用迭代函数系统（IFS）模拟气泡崩溃后期微团的随机运动。
  - `chaotic_microfragmentation` 通过四个预定义的二维仿射变换（代表不同尺度的拉伸、压缩、旋转剪切）对大量粒子进行迭代，返回最终位置分布并估算平均 Lyapunov 指数。
  - `fragmentation_dimension` 用计盒法估计最终点集的分形维数。
- 辅助工具：`compute_deformation_tensor` 从离散边界点通过最小二乘拟合恢复形状矩阵；`safe_divide` 本地安全除法。

此模块与 `surface_integrals` 无直接依赖，但其变形形状可被积分模块用于表面积或体积计算（后者接口仅需半径函数）。

---

### 4. `surface_integrals.py` – 气泡表面积与体积的高阶数值计算

该模块提供气泡几何量与能量的精确积分工具：

- Gauss‑Chebyshev 第二类求积规则：`chebyshev2_nodes_weights` 生成节点和权重，`chebyshev_surface_integral` 在一维边界曲线上执行带权积分的求积。
- 三维 Gauss‑Legendre 求积规则：`gauss_legendre_3d` 生成立方体区域的节点与权重。
- 表面积计算 `bubble_surface_area_quadrature`：接收一个描述气泡表面形状的函数 `r(theta, phi)`，在球坐标下使用数值微分计算度量，再对角度积分。采用 Gauss‑Legendre 与等距混合求积。
- 体积计算 `bubble_volume_quadrature`：同样基于所述形状函数，利用体积公式对角度直接积分。
- 表面能 `surface_tension_energy` 由表面积乘以表面张力得到。
- 液体动能 `kinetic_energy_integral`：在球形近似下给出解析估计（含非球形一阶修正）。
- 压力做功 `pressure_work_integral` 由压差乘以体积得到。
- 精确度测试 `legendre_3d_exactness_test`：对三维单项式积分，比较数值与精确解，用于验证积分规则的代数精度。

所有积分函数均依赖 `numpy` 和 `scipy` 的 `leggauss`、`legendre` 等标准函数。

---

### 5. `fem_pressure_wave.py` – 二维有限元压力波传播

该模块用二维线性三角形有限元求解声学波动方程：

- `generate_square_mesh(a, b, h)` 利用 Delaunay 三角剖分在正方形区域生成计算网格，返回节点坐标和单元连接列表，并剔除退化单元。
- `fem_matrices_2d(nodes, elements, c_sound, rho)` 组装一致质量矩阵 `M` 和刚度矩阵 `K`，基于线性形函数的梯度计算。
- `apply_boundary_conditions` 在气泡壁附近节点施加 Dirichlet 条件（压力等于壁面压力），对外边界可进行吸收边界近似处理，通过修改矩阵实现。
- `solve_pressure_wave_fem` 是主时间推进求解器，使用 Newmark‑β 方法（隐式）对声学波动方程进行时间离散
