# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：视网膜神经编码与视觉通路的多尺度计算模型

本项目构建了一个从光感受器到神经节细胞层的多尺度视网膜计算模型，模拟视觉刺激在视网膜中的处理过程。整个项目由多个功能模块协同构成，`main.py` 作为统一入口逐阶段执行完整流程，输出各环节的量化结果。

---

## 文件清单与职责

项目除 `main.py` 外共包含 9 个功能模块文件：

| 文件名 | 主要职责 |
|--------|---------|
| `retina_geometry.py` | 生成六边形密排感光细胞阵列；Delaunay 三角化构建计算网格；网格质量评估；凸包与多边形矩计算。 |
| `photoreceptor.py` | 光感受器光适应的稳态扩散方程求解（Jacobi 迭代）；光电流的 Clenshaw‑Curtis 数值积分；光转导级联 ODE 的 Runge‑Kutta 4 阶求解。 |
| `bipolar_cell.py` | 差异高斯（DoG）感受野模型；Legendre 多项式计算与 Gauss‑Legendre 求积；感受野的 Legendre 正交基展开与重构。 |
| `synaptic_diffusion.py` | Gray‑Scott 反应‑扩散模型的数值模拟（模拟突触间隙神经递质时空演化）；突触传递效能评估。 |
| `neural_encoding.py` | 三角插值基函数与脉冲模式重建；Horner 多项式求值；脉冲序列统计分析与编码效率（互信息下界）计算。 |
| `spike_generation.py` | 非齐次泊松脉冲序列的 thinning 生成；Faure 准随机序列生成；高维神经参数空间采样。 |
| `network_solver.py` | 带状矩阵的 LU 分解与求解；稠密矩阵 LU 分解；压缩边界带状（CBB）系统的 Schur 补求解；视网膜网络连接矩阵构建。 |
| `stimulus_generator.py` | 标准视觉刺激生成（正弦光栅、高斯斑点、漂移光栅、白噪声）；字典序组合生成与突触子集探索；数学字符串安全求值。 |
| `math_utils.py` | 通用数学工具：梯形/Simpson 积分、连带 Legendre 函数、球谐函数、矩阵条件数估计、对数 Gamma 函数、误差函数近似等。 |

---

## 各模块核心功能与算法概要

### 1. `retina_geometry.py` — 视网膜几何离散化与网格质量评估

**关键函数：**

- `generate_hexagonal_photoreceptor_array(radius, n_rings)`  
  返回一个 (N,2) 数组，表示以中心为零点、按六边形密排规则生成的细胞中心坐标。第 n 层环上共有 6n 个点，通过在六边形相邻顶点间插值得到。

- `delaunay_triangulation_2d(points)`  
  输入二维点集，返回 (M,3) 的三角形顶点索引数组。采用朴素 O(N⁴) 算法：遍历三点组合，用有向面积保证逆时针顺序，并检查外接圆内不包含任何其他点（利用四点行列式判定外接圆条件）。

- `evaluate_mesh_quality(points, triangles)`  
  返回网格质量指标字典，包含：ALPHA 度量（最小内角与 60° 的比值）、Q 度量（内切圆半径与外接圆半径之比的 2 倍）、三角形面积统计、边界边列表。边界边通过有向边对消法提取（内部边被两三角形以相反方向共享，未被反向的边为边界边）。

- `convex_hull_2d(points)`  
  用 Jarvis March（卷包裹）算法计算二维点集的凸包，返回逆时针排列的边界顶点索引。

- `hexagon_moment_integral(p, q, vertices)`  
  利用 Green 公式与二项式展开计算任意多边形区域上单项式 x^p y^q 的积分。

---

### 2. `photoreceptor.py` — 光感受器光转导与光适应

**关键函数：**

- `solve_light_adaptation_steady_state(nx, ny, ...)`  
  使用 Jacobi 迭代求解稳态扩散‑反应方程 D·∇²C + S = 0，采用五点差分格式离散 Laplacian，边界为 Dirichlet 条件。返回稳态浓度场、迭代次数和最终误差。

- `integrate_photocurrent_clenshaw_curtis(intensity_profile, a, b, n)`  
  用 Clenshaw‑Curtis 求积计算光电流积分。节点为 Chebyshev 极端点 cos(jπ/(n‑1))，权重通过离散余弦变换计算。积分区间经线性变换映射到 [-1,1]。

- `solve_phototransduction_rk4(I_light_func, y0, t_span, dt, params)`  
  采用经典四阶 Runge‑Kutta 方法求解三个状态变量（PDE*、cGMP、Ca²⁺）的 ODE 系统，该 ODE 由函数 `phototransduction_ode` 定义，涵盖了激活、合成与水解过程，并使用 Hill 型动力学描述鸟苷酸环化酶活性。

---

### 3. `bipolar_cell.py` — 双极细胞感受野与 Legendre 基展开

**关键函数：**

- `legendre_polynomial_value(n, x)`  
  利用三项递推关系计算 Legendre 多项式 P₀(x) … Pₙ(x) 在给定点集上的值。

- `legendre_polynomial_zeros(n)`  
  通过构造对称 Jacobi 矩阵并求解特征值得到 n 阶 Legendre 多项式的零点，用于 Gauss‑Legendre 求积节点。

- `gauss_legendre_quadrature(n)`  
  返回 n 点 Gauss‑Legendre 求积的节点（即零点）和权重，权重由零点处的导数推导。

- `dog_receptive_field(x, y, A_c, sigma_c, A_s, sigma_s)`  
  计算差异高斯（DoG）感受野的空间响应：中心正高斯减去周围负高斯，参数包括振幅与半径。

- `compute_bipolar_response_convolution(stimulus, rf_params, grid_spacing)`  
  利用二维 Gauss‑Legendre 求积近似计算刺激与感受野的卷积响应，节点映射到实际空间坐标并通过最近邻插值访问像素。

- `decompose_rf_with_legendre_basis(spatial_profile, max_degree, n_quad)`  
  将一维空间感受野轮廓用 Legendre 多项式的正交投影展开，系数通过 Gauss‑Legendre 求积计算。

- `reconstruct_rf_from_legendre_coeffs(coeffs, x)`  
  由 Legendre 系数与多项式求值重构轮廓。

---

### 4. `synaptic_diffusion.py` — 突触间隙反应‑扩散模拟

**关键函数：**

- `simulate_synaptic_transmission(nx, ny, n_steps, Du, Dv, F, K, dt, dx, dy, initial_condition, boundary)`  
  使用 Gray‑Scott 模型模拟两种物质 U（前体）和 V（活性神经递质）的时空演化。时间推进采用显式前向 Euler 法；Laplacian 在周期性边界下用 9‑点模板（精度 O(h⁴)），在 Dirichlet 边界下用 5‑点模板。初始条件支持局部释放、波前和随机分布。返回浓度场历史快照和终态。

- `compute_synaptic_efficacy(V_field, threshold, receptor_density)`  
  根据神经递质浓度场 V 计算峰值浓度、激活面积占比以及加权总效能。

---

### 5. `neural_encoding.py` — 脉冲发放模式分析与编码效率

**关键函数：**

- `trig_interp_basis(x, k)`  
  计算三角插值基函数，k 为奇数时形为 sin(kπx/2)/(k·sin(πx/2))，偶数时使用 tan 分母，并对奇异点做极限处理。

- `tr
