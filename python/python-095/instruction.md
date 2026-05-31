# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 主动噪声控制综合仿真项目（python-095）

本项目围绕三维封闭空间中的多通道宽带主动噪声控制（ANC）实现了多个相互协作的仿真模块。  
所有算法均以 Python 文件分模块实现，并由一个零参数入口脚本 `main.py` 串联成完整的验证流程。

## 文件总览与依赖关系

| 文件名 | 主要职责 |
|--------|----------|
| `special_functions.py` | 数学基础：不完全 Beta 函数、digamma/trigamma、cos 幂积分、对数 Beta 函数 |
| `tridiagonal_acoustics.py` | 一维管道 Helmholtz 方程求解、Thomas 三对角算法、刚性 ODE 验证 |
| `spherical_array_geometry.py` | 球面 Fibonacci 网格生成、球谐变换与阵列指向性 |
| `sparse_acoustics.py` | 稀疏声传播矩阵（ST↔CCS 格式转换与向量乘法） |
| `source_phase_optimizer.py` | 单频声源相位优化（基于 Chandrupatla 求根） |
| `adaptive_filter.py` | 多通道自适应滤波（QR 最小二乘、FxLMS 类、批量 ANC 设计） |
| `optimal_source_selection.py` | 次级声源子集选择（子集和交换、贪心选择） |
| `statistical_noise_model.py` | 多通道噪声 Dirichlet 建模与自适应步长调整 |
| `integrals_radiation.py` | 圆形活塞辐射器 Rayleigh 积分与指向性 |
| `nonlinear_ode_dynamics.py` | 非线性自适应动力学 ODE 与稳定性边界分析 |
| `acoustic_room_model.py` | 三维房间声学有限元模型与 RCM 网格重排序 |
| `main.py` | 主驱动脚本，依次调用以上所有模块进行综合演示 |

**依赖关系**：  
- `integrals_radiation.py` 依赖 `special_functions.py`（`cos_power_int`）。  
- `statistical_noise_model.py` 依赖 `special_functions.py`（`digamma`, `trigamma`）。  
- `main.py` 依赖所有上述模块，并在最后会调用 `special_functions` 的几个函数进行独立验证。  
- 其余各模块彼此独立，无相互调用。

---

## 模块功能说明

### 1. `special_functions.py` – 特殊数学函数库
提供 ANC 理论及统计建模中必需的特殊函数：
- `betain(x, p, q, beta_log)` – 计算正则化不完全 Beta 函数 \(I_x(p,q)\)，基于连分式/级数展开。
- `digamma(x)` – 计算 digamma 函数 \(\psi(x) = \Gamma'(x)/\Gamma(x)\)，采用渐进展开与递推。
- `trigamma(x)` – 计算 trigamma 函数 \(\psi'(x)\)，同样使用级数展开。
- `cos_power_int(a, b, n)` – 计算 \(\int_a^b \cos^n(t)\,dt\)，利用降幂递推。
- `log_beta(p, q)` – 对数 Beta 函数。
- `incomplete_beta_cdf` – 包装函数，处理边界情况。

### 2. `tridiagonal_acoustics.py` – 一维管道声学与刚性 ODE
- **Thomas 算法**：`tridiagonal_solver(a, b, c, d)` 求解三对角线性系统，支持单右端项或多列右端项。
- **管道 Helmholtz 求解器**：`pipe_helmholtz_solver(L, N, k, source_profile)`  
  对两端封闭的一维管道离散 Helmholtz 方程，构造三对角矩阵并分别求解实部与虚部，得到复数声压场。
- **Lindberg 刚性 ODE 测试**：  
  - `lindberg_exact_solution(t)` 返回简化声学模态耦合系统的精确解及其导数。  
  - `lindberg_residual(t, y, dydt)` 计算 ODE 残差，用于验证数值积分解的正确性。

### 3. `spherical_array_geometry.py` – 球面阵列几何布置
- **Fibonacci 球面网格**：`sphere_fibonacci_grid_points(ng, radius)`  
  利用黄金比例生成近似均匀分布的球面采样点，用于构成误差麦克风或次级声源阵列。
- **球谐变换矩阵**：`spherical_harmonic_transform_matrix(points, L_max)`  
  根据球坐标计算实数形式的球谐函数矩阵，用于从采样的声压估计球谐系数。
- **阵列波束形成指向性**：`spherical_array_directivity(weights, points, theta, phi, k)`  
  计算给定复权重的波束形成输出模值。

### 4. `sparse_acoustics.py` – 稀疏声学传递矩阵
- **`SparseAcousticMatrix` 类**：支持两种稀疏格式：
  - **ST 格式**（行、列、值三元组存储）。
  - **CCS 格式**（按列压缩存储：列起始指针、行索引、数值）。
  - 方法：`add_entry`, `st_to_ccs`（压缩过程中可合并重复位置），`ccs_mv`, `st_mv`（矩阵‑向量乘），`to_dense`（仅供调试）。
- **房间耦合图**：`generate_room_coupling_graph(n_nodes, connection_prob, seed)`  
  生成对称随机稀疏邻接矩阵，模拟房间内节点间声学耦合。
- **传递矩阵生成**：`acoustic_transfer_matrix_sparse(sensor_positions, source_positions, k, ...)`  
  基于距离截断构造稀疏的声学传递矩阵（值取 1/r），模拟自由场或考虑简单镜像的传播。

### 5. `source_phase_optimizer.py` – 次级声源相位优化
- **Chandrupatla 求根算法**：`zero_chandrupatla(f, x1, x2, epsilon, delta, max_iter)`  
  混合逆二次插值与二分法的稳健求根方法，要求区间端点函数值异号。
- **单源相位优化**：`optimize_source_phase(H_col, d, amplitude, phi_bounds)`  
  以总声场能量最小化为目标，定义相位导数函数并利用 Chandrupatla 在多个子区间搜索驻点，同时检查区间端点，得到最优相位及对应的最小能量。

### 6. `adaptive_filter.py` – 多通道自适应滤波与 QR 最小二乘
- **正则化最小二乘**：`qr_least_squares(A, b, lam, w0)`  
  通过正规方程法 \(\left(A^TA + \lambda I\right) w = A^T b + \lambda w_0\) 求解，使用 Cholesky 或直接解线性系统，失败时回退到伪逆。
- **秩揭示 QR 最小二乘**：`qr_rank_revealing_ls(A, b, tol_factor)`  
  使用 Householder QR 分解 \(AP=Q[R_{11}\ R_{12}; 0\ 0]\)，根据对角元判定秩，截断求解最小范数解，增强数值稳定性。
- **多通道 FxLMS 类**：`MultichannelFxLMS`  
  维护参考信号缓冲区和自适应权值矩阵，提供：
  - `filter_reference(x_new)` – 利用给定的次级通路估计对参考信号进行滤波。
  - `update(x_new, error)` – 基于滤波‑x LMS 规则更新权值（含泄漏因子）。
  - `predict_output(x_new)` – 计算当前滤波器输出。
- **批量 ANC 设计**：`batch_multichannel_anc_design(H, X, d, reg_lambda)`  
  构造包含全卷积的有效回归矩阵，并使用秩揭示 QR 求解最优滤波器系数。

### 7. `optimal_source_selection.py` – 最优次级声源子集选择
- **功率预算子集选择**：`subset_sum_swap_anc(candidate_powers, desired_power_budget)`
