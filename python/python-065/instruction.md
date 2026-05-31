# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 极端天气事件归因分析系统 (CEEAS) — 项目描述

本项目是一个面向气候科学的极端天气事件归因分析框架，集成了多项数值算法。整体流程包括：生成合成气候异常场、利用渗流理论识别极端事件连通簇、对事件区域进行 Delaunay 三角剖分与网格积分、计算全球辐射强迫与大气柱垂直积分、通过蒙特卡洛集合模拟评估归因不确定性、使用 Pfaffian 协方差模型以及能量级联动力学模拟极端事件发展等。所有代码以 **main.py** 为入口，其他模块提供具体算法支持，下面分别说明各模块的职责和关键接口。

---

## 文件：climate_interpolation.py
**职责**：提供一维最近邻插值与 Weierstrass-Durand-Kerner (WDK) 多项式求根算法，用于气候数据的重网格化及特征多项式根求解。

**主要公共函数**：
- `nearest_interp_1d(xd, yd, xi)`：对已排序的一维数据点进行最近邻插值，返回插值点处的函数值。
- `wdk_roots(c, tol, max_iter)`：输入多项式系数（按幂次排列），使用 WDK 迭代求所有复数根，内部基于 Cauchy 界给出初值猜测，并在每次迭代中更新各根估计值直至收敛。
- `regrid_field_nearest(lon_src, lat_src, field_src, lon_tgt, lat_tgt)`：利用最近邻插值将二维气候场（如经纬度网格）插值到目标网格，逐个纬度、经度选择距离最近的源格点。
- `dominant_scale_analysis(field_1d, max_degree)`：对一维气候场进行 FFT 求功率谱，用多项式拟合并求其导数的根，提取主导空间尺度对应的频率，返回实正根列表及其它诊断量。

**核心算法思想**：
- 最近邻插值通过搜索排序后的坐标，比较左右邻居距离确定最近点。
- WDK 求根并行地修正每个根的估计，直至前后变化小于容差。

---

## 文件：climate_percolation.py
**职责**：基于二维渗流理论分析极端天气事件的空间连通性。将气候异常场二值化，标记连通分量，计算跨越性簇、序参量和关联长度。

**主要公共函数**：
- `detect_extreme_grid(anomaly_field, threshold)`：根据阈值将异常场二值化为极端事件占据格点（0/1）。
- `components_2d(a)`：对二值数组采用深度优先搜索标记 4-邻域连通分量，返回每个格点的分量标号。
- `spanning_analysis(cls, m, n)`：分析哪些分量在水平或垂直方向上贯通整个域，返回两类跨越分量个数及各分量尺寸。
- `percolation_order_parameter(component_sizes, total_sites)`：计算渗流序参量 P∞ = 最大簇尺寸 / 总格点数。
- `correlation_length_estimate(component_sizes, threshold_bins)`：由簇尺寸分布估计关联长度 ξ。
- `run_percolation_attribution(anomaly_field, threshold)`：封装上述步骤，返回一个字典，包含占据概率、跨越分析结果、序参量及关联长度等。

**核心算法**：渗流模型中的占据概率 p、连通分量标记、跨越检测以及基于有限尺寸标度的临界指数概念用于解释极端事件的空间组织性。

---

## 文件：covariance_pfaffian.py
**职责**：实现斜对称矩阵的 Pfaffian 计算（Parlett‑Reid LTL 分解），并应用于极端事件空间相关的高斯随机场建模。

**主要公共函数**：
- `pfaffian_LTL(A_in)`：计算 N×N 斜对称矩阵的 Pfaffian，要求 N 为偶数（奇数时返回 0）。算法通过逐次消除及主元交换保符号，乘积形式给出 Pfaffian。
- `build_skew_covariance_from_kernel(nodes, kernel_func)`：由节点坐标和反对称核函数构造斜对称协方差矩阵。
- `gaussian_random_field_log_partition(K)`：利用 Pfaffian 计算高斯随机场配分函数的对数。
- `extreme_event_pfaffian_correlation(nodes, correlation_length)`：基于反对称指数核，对给定节点集构造协方差矩阵并计算 Pfaffian，代表极端事件构型的概率权重。

**核心算法**：斜对称矩阵的 LTL 分解逐行约化，Pfaffian 等于分解过程中若干对角元乘积再乘以行列交换引入的符号。

---

## 文件：delaunay_mesh.py
**职责**：实现平面点集的 Delaunay 三角剖分（朴素算法），用于为极端事件连通簇构建自适应网格。

**主要公共函数**：
- `points_delaunay_naive_2d(node_xy)`：对二维点集，使用空外接圆准则遍历所有三元组，筛选出符合 Delaunay 条件的三角形，返回三角形数量与顶点索引数组。
- `build_event_mesh(component_mask, grid_x, grid_y)`：对连通分量标记数组中的每个分量，提取格点坐标并去重，调用朴素剖分得到三角网格，同时计算各三角形面积，返回各分量网格信息的字典。
- 辅助函数 `triangle_area_2d`、`circumcenter_2d` 等用于几何计算。

**核心算法**：Delaunay 条件要求所有三角形的外接圆内不包含其他点，朴素实现通过三重循环检查来实现 O(N⁴) 复杂度，仅适用于小规模点集。

---

## 文件：energy_cascade_ode.py
**职责**：基于大气能量级联模型（归一化 ODE dy/dt = y²(1-y)）模拟极端事件能量的发展与饱和，并提供精确解（用 Lambert W 函数）和数值解（RK4）。

**主要公共函数**：
- `energy_cascade_exact(t, y0)`：给定初始值 y0∈(0,1) 和时间序列，返回精确解，利用 Lambert W 函数计算。
- `solve_energy_cascade_rk4(t_span, y0, n_steps)`：用经典四阶 Runge–Kutta 方法数值求解 ODE，并处理 [0,1] 范围约束，返回时间和能量轨迹。
- `energy_saturation_time(y0, epsilon)`：计算能量达到指定饱和度所需的时间。
- `atmospheric_energy_model(intensity, tau, delta)`：结合精确解提供单点查询。

**核心模型**：方程中 y² 项代表大尺度能量输入，y³ 项代表湍流耗散，描绘了极端事件从稳态增长至饱和的动力学。

---

## 文件：fast_spectral_quadrature.py
**职责**：提供基于 FFT/三对角矩阵特征分解的快速谱求积规则，用于大气垂直积分等。

**主要公共函数**：
- `fejer1_integrate_fast(f, n)`：在 [-1,1] 上采用 Fejér 第一类规则（利用 IFFT 快速计算权重）积分给定函数。
- `gauss_legendre_integrate_fast(f, n)`：通过对称三对角矩阵的特征值/特征向量得到 Gauss‑Legendre 节点和权重并积分。
- `integrate_vertical_column(z_levels, values, method)`：将垂直坐标映射到标准区间，调用相应求积规则进行积分。
- `compute_total_column_water_vapor(z, q, rho)`：计算柱水汽含量 TWCW = ∫ ρ·q dz，调用垂直柱积分。

**核心思想**：Fejér 规则基于 DCT/FFT 加速权重生成；Gauss‑Legendre 规则利用 Gauss‐Lobatto 型三对角矩阵的谱分解获得节点和权重。

---

## 文件：monte_carlo_ensemble.py
**职责**：蒙特卡洛采样与集合距离分析，用于不确定性量化及归因一致性评估。

**主要公共函数**：
- `line01_sample_random(n, seed)`：在 [0,1] 上均匀随机采样。
- `monte_carlo_line_integ
