# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目：燃烧科学—爆轰波结构与传播（Python 科学计算系统）

本项目构建面向爆轰波结构与传播问题的博士级科学计算系统，融合了多个数值方法模块。入口文件 `main.py` 将保留，它依次调用各个模块完成完整的计算流程。你的任务是依据本描述，重建除 `main.py` 外所有缺失的 Python 源文件，使得 `main.py` 能够正常运行并输出正确结果。

项目围绕以下几个核心物理和数值主题：
- 爆轰中的 Chapman–Jouguet (CJ) 与 von Neumann 状态计算
- 一维 ZND 爆轰结构的常微分方程求解
- 二维可压缩反应 Euler 方程的有限体积法求解
- 高维化学反应速率的稀疏网格插值
- 热力学积分的高精度对称求积
- 爆轰波前的自适应三角形网格生成（含线性三角形基函数）
- 蒙特卡洛点火概率与临界热点分析
- 反应网络图的构建与分析
- 基于 ZND 剖面的线性稳定性分析

所有模块共享一套物理常数和工具函数（文件 `combustion_utils.py`）。详细说明如下。

---

## 模块一：通用物理常数与工具函数
**文件：** `combustion_utils.py`

**职责：** 提供全局物理常数、参数边界检查以及爆轰流体力学中的基本关系式。其他所有模块都依赖此文件。

**主要内容：**
- 物理常数：通用气体常数、标准大气压、默认物质参数（比热比、单位质量释热、活化能、指前因子、点火温度、未燃气体密度/压强/温度、摩尔质量）。
- 输入检查函数：`check_positive`, `check_nonnegative`, `check_interval`（确保参数合法）。
- Arrhenius 反应速率 `arrhenius_rate(T, A, Ea, R)`：计算 k = A·exp(−Ea/(R·T))，含指数溢出保护。
- 理想气体热力学关系：
  - `specific_heat_ratio_cv_cp(gamma)`：由比热比求定容、定压比热。
  - `sound_speed(T, gamma, W_mol)` 和 `sound_speed_from_prho(p, rho, gamma)`：计算声速。
- ZND 反应进度演化：`znd_progress_variable_derivative(lambda_var, T, A, Ea, n_order, R)` 返回 dλ/dt。
- Rankine–Hugoniot 激波跃变关系：
  - `rankine_hugoniot_pressure_ratio(M, gamma)` 和 `rankine_hugoniot_density_ratio(M, gamma)` 给出激波前后压比和密度比。
- CJ 爆轰速度的简化解析式 `cj_detonation_velocity(gamma, Q, p0, rho0)`。
- von Neumann 尖峰状态：`von_neumann_spike_conditions(D, gamma, p0, rho0)` 通过激波马赫数给出波后压强、密度、温度和马赫数。
- 由比内能和反应进度计算温度：`temperature_from_energy(e, lambda_var, Q, cv)`。
- 2×2 矩阵工具：`cholesky_factor(a)` 对对称正定矩阵做 Cholesky 分解，`solve_lower_triangular(L, b)` 解下三角方程组（用于椭圆采样等）。

---

## 模块二：反应动力学与热力学状态
**文件：** `reaction_kinetics.py`

**职责：** 封装可反应理想气体的状态描述和欧拉通量计算，提供守恒变量与原始变量之间的转换以及化学反应源项。

**核心类：** `ReactiveState`
- 存储原始变量：密度 ρ、速度分量 u, v、比内能 e 和反应进度 λ ∈ [0,1]。
- 方法 `to_conservative()`：输出守恒向量 [ρ, ρu, ρv, E, ρλ]，其中总能量 E = ρe + 0.5ρ(u²+v²)。
- 类方法 `from_conservative(U, gamma, Q)`：由守恒向量反推原始变量，对密度和进度变量进行截断保护。
- `pressure(gamma, Q, W_mol)`：根据理想气体状态方程计算压强，扣除未燃化学能贡献。
- `temperature(gamma, Q, W_mol)`：通过状态方程和压强计算温度。

**主要函数：**
- `chemical_source_term(state, ...)`：返回 dU/dt 的化学源项，即能量源项（释热）和进度变量源项。
- `euler_flux_x(state, ...)` 和 `euler_flux_y(state, ...)`：分别计算 x 和 y 方向的欧拉通量向量（质量、动量、能量、进度输运）。
- `reactive_euler_rhs(U, dx, dy, ...)`：在二维均匀网格上计算反应欧拉方程的右端项，使用 Lax‑Friedrichs 型数值通量进行空间离散，并加上化学源项。

---

## 模块三：一维 ZND 爆轰结构求解器
**文件：** `znd_structure.py`

**职责：** 求解一维定常 ZND 模型，描述从 von Neumann 尖峰到完全反应平衡的爆轰波内部剖面。

**核心类：** `ZNDSolver`
- 构造时接收反应动力学参数和未燃气体状态。
- `cj_velocity()`：计算解析 CJ 速度。
- `von_neumann_state(D)`：根据指定爆轰速度 D 计算激波后的 von Neumann 状态（ρ, p, T, u）。
- `_rhs(y, D)`：实现波系坐标下的 ODE 右端项，状态向量 y = [ρ, u, p, λ]；利用质量、动量和能量守恒的微分关系推导密度、速度、压强的变化率，结合 Arrhenius 反应速率给出 dλ/dξ。
- `solve(D, ximax, npts)`：使用改进的 Euler 法（Heun 法）积分获得解数组 (xi, sol)，其中 sol 的每一列对应 ρ, u, p, λ。
- `induction_length(xi, sol, threshold)` 和 `half_reaction_length(xi, sol)`：由反应进度剖面提取特征长度。

---

## 模块四：二维反应 Euler 方程求解器
**文件：** `euler_reactive_solver.py`

**职责：** 提供二维可压缩反应 Euler 方程的显式时间推进求解器，以及用于隐式方法（如 Jacobian 存储）的稀疏矩阵格式。

**辅助类：** `SparseCRS`（压缩稀疏行格式）
- 存储矩阵的行指针、列索引和值数组。
- 实现 `multiply(x)` 计算 y = A·x，和 `multiply_transpose(x)` 计算 y = Aᵀ·x。

**核心类：** `ReactiveEulerSolver`
- 构造参数：网格分辨率、空间步长、比热比、释热、反应动力学常数。
- `initialize_cj_planar_wave(D, rho0, p0, ...)`：用 tanh 函数构造沿 x 方向传播的平面爆轰波初始场，从波前状态平滑过渡到 CJ 平衡态，填充守恒量数组 U。
- `_rhs(U)`：组合空间导数的通量贡献和化学源项贡献。
- `_spatial_rhs(U)`：分别沿 x 和 y 方向进行 Lax‑Friedrichs 数值通量分裂，计算通量散度项。
- `_source_rhs(U)`：遍历所有网格点调用 `chemical_source_term` 获得化学源项。
- `step_rk3(dt)`：执行三阶 TVD Runge–Kutta 时间步进。
- `compute_cfl_dt(cfl)`：基于局部最大波速计算满足 CFL 条件的时间步长。
- `advance(t_final, cfl, n_print)`：自动时间步循环推进到指定时刻。

---

## 模块五：高维化学流形稀疏
