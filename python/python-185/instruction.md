# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# Benchmark 项目描述：基于自适应三角剖分与谱稀疏表示的压缩感知图像重建

## 项目背景

本项目模拟一个博士层次的数据科学研究：在严重欠采样条件下，利用压缩感知（Compressed Sensing）技术从少量测量中重建医学图像（如 MRI、CT）。系统融合了谱稀疏表示、自适应网格细化、空间先验建模、动态扩散、高阶误差估计以及非相干采样模式设计等多个计算模块。你将得到一个可运行的 `main.py`，它会依次调用这些模块中的函数进行演示。你的任务是**根据本描述和 `main.py` 中暴露的函数签名与调用方式，重新实现被删除的 Python 文件**。

## 项目结构

项目根目录包含以下文件（`main.py` 将保留，其他 `.py` 文件需要你重新实现）：

- `main.py`（保留）
- `cs_detector.py`
- `dynamic_reconstruction.py`
- `error_estimator.py`
- `fast_solver.py`
- `mesh_adaptive.py`
- `mesh_refinement.py`
- `sampling_pattern.py`
- `spatial_prior.py`
- `spectral_basis.py`
- `support_optimizer.py`

下面依次说明每个缺失文件的作用及其需要对外暴露的主要函数。

---

### 1. `cs_detector.py` —— 压缩感知稀疏检测与 L1 最小化重建

本模块提供基于迭代软阈值算法（ISTA/FISTA）和正交匹配追踪（OMP）的稀疏重建求解器，以及高斯随机测量矩阵的构造。

需要实现的函数（名称、大致输入输出）：

- `soft_thresholding(x, lambda_)`  
  输入：向量 `x`，阈值 `lambda_`  
  输出：软阈值处理后的向量（L1 近端算子）

- `ista_reconstruction(A, y, lambda_, max_iter, tol, x0)`  
  使用 ISTA 求解 Basis Pursuit Denoising。  
  输入：感知矩阵 `A`，测量向量 `y`，正则化参数 `lambda_`，可选迭代参数和初始解。  
  输出：稀疏系数向量。

- `fista_reconstruction(A, y, lambda_, max_iter, tol, x0)`  
  使用 FISTA（快速 ISTA）求解，收敛速度为 O(1/k²)。参数和输出同 ISTA。

- `orthogonal_matching_pursuit(A, y, sparsity, max_iter)`  
  贪婪的 OMP 算法，在已知稀疏度 `sparsity` 下重建信号。  
  输入：感知矩阵 `A`，测量向量 `y`，目标稀疏度。  
  输出：重建向量和支持集索引。

- `build_sensing_matrix_gaussian(m, N, normalize)`  
  生成一个 `m×N` 的高斯随机测量矩阵，默认进行列归一化。

**注意**：`cs_detector` 中的 ISTA/FISTA 需要自行计算感知矩阵的 Lipschitz 常数（`A^T A` 的谱范数）以确定步长；OMP 需通过最小二乘更新支持集。

---

### 2. `dynamic_reconstruction.py` —— 动态图像序列的隐式中点法重建

实现了二维扩散‑衰减偏微分方程的数值求解，以及结合时间平滑约束的压缩感知重建。

需要实现的函数：

- `discrete_laplacian_2d(I, h)`  
  计算二维离散拉普拉斯算子（5 点模板），边界采用 Neumann 条件近似。

- `diffusion_rhs(t, I, D, alpha)`  
  返回扩散‑衰减 PDE 的右端项 `D·∇²I - α·I`。

- `midpoint_fixed_step(f, t0, I0, dt, ...)`  
  固定点迭代中点法单步推进。

- `midpoint_implicit_step(f, t0, I0, dt, ...)`  
  隐式中点法单步推进（用 Picard 迭代求解中点状态）。

- `solve_dynamic_diffusion(I0, tspan, n_steps, D, alpha, method)`  
  求解二维扩散衰减方程的时间演化，返回时间序列和图像序列。`method` 可选 `'implicit'` 或 `'fixed'`。

- `dynamic_cs_reconstruction(measurements, Phi, Psi, lambda_reg, temporal_smoothness)`  
  利用 FISTA 进行一帧重建的简单封装（内部调用 `cs_detector.fista_reconstruction`）。

---

### 3. `error_estimator.py` —— 基于高阶数值积分的重建误差估计

提供三角形和金字塔上的数值积分规则，以及图像重建质量评估（L² 误差、PSNR、SSIM 等）。

需要实现的函数：

- `twb_rule_n(strength)` – 返回三角形 TWB 规则的节点数。
- `twb_rule_data(strength)` – 返回 TWB 规则的节点坐标和权重字典。
- `integrate_triangle_unit_monomial(ex, ey)` – 计算单位三角形上单项式 `x^ex y^ey` 的精确积分（利用阶乘公式）。
- `integrate_over_triangle(f_values, rule_strength)` – 使用 TWB 规则求积分。
- `pyramid_unit_volume()` – 返回单位金字塔的理论体积。
- `pyramid_witherden_rule_data(degree)` – 返回金字塔上 Witherden‑Vincent 型规则的三维节点和权重（采用张量积型近似）。
- `compute_l2_error_image(true_image, recon_image, use_triangle_quad)`  
  计算两幅图像之间的 L² 误差、PSNR、SSIM 等指标，可选地利用三角形积分规则改善误差估计。
- `compute_reconstruction_quality(true_image, recon_image)`  
  便捷封装函数，内部调用 `compute_l2_error_image`。

---

### 4. `fast_solver.py` —— 三对角共轭梯度快速求解器

利用三对角矩阵的 R83 存储格式加速求解正规方程，适用于带状近似后的线性系统。

需要实现的函数：

- `r83_mv(m, n, a, x)`  
  计算 R83 格式三对角矩阵 `a` 与向量 `x` 的乘积。
- `r83_cg(n, a, b, x, max_iter, tol)`  
  共轭梯度法求解三对角线性系统 `A x = b`，其中 `A` 以 R83 格式存储。
- `construct_tridiagonal_from_dense(A)`  
  从稠密方阵中提取三对角元素并转换为 R83 格式。
- `solve_normal_equations_cg(A, y, lambda_reg, max_iter, tol)`  
  构造 `A^T A + λI` 的三对角近似，并用 R83‑CG 求解 `(A^T A + λI)x = A^T y`。

---

### 5. `mesh_adaptive.py` —— 自适应三角剖分与质量度量

在图像域生成均匀三角剖分，并基于图像梯度和三角形质量指标（ALPHA、Q 度量）进行自适应细化。

需要实现的函数：

- `triangle_area(p1, p2, p3)` – 三角形有向面积。
- `arc_cosine_safe(c)` – 安全的反余弦（截断到 [-1,1]）。
- `alpha_measure_single(p1, p2, p3)` – 计算单个三角形的 ALPHA 质量度量（最小角与 60° 的比值）。
- `q_measure_single(p1, p2, p3)` – 计算 Q 度量（内切圆半径与外接圆半径之比）。
- `evaluate_triangulation_quality(nodes, triangles)`  
  输入节点数组和三角形索引数组，返回包含最小/平均 ALPHA、Q 值、面积统计等的字典。
- `generate_uniform_triangulation(width, height, nx, ny)`  
  在矩形域上生成均匀的三角剖分（每个矩形分成两个三角形）。
- `adaptive_refinement_by_gradient(image, nodes, triangles, quality_threshold)`  
  根据图像梯度和三角形质量，对低质量高梯度区域进行细化，添加形心节点并重新划分
