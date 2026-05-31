# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 软体机器人运动学建模 — 项目描述

## 项目目标
本项目是一个用于软体机器人连续体运动学与动力学建模的计算框架。它融合了 Cosserat 杆理论、超弹性本构、横截面有限元分析、谱/间断 Galerkin 离散、降阶模型、逆运动学、路径规划以及双调和方程验证等多个模块，最终在 `main.py` 中串连演示。后续将保留 `main.py`，删除其他 `.py` 文件，你需要根据本描述重新实现这些缺失的模块。

## 文件结构
- `main.py` — 统一入口，调用所有子模块进行集成测试（保留，不要求实现）。
- `mesh_utils.py` — 网格生成与采样质量评估。
- `section_fem.py` — 横截面有限元分析（T6单元、高斯求积、截面属性计算）。
- `spectral_dg.py` — 谱方法与间断 Galerkin 离散化。
- `hyperelastic_law.py` — 超弹性本构模型与化学‑力学耦合。
- `cosserat_core.py` — Cosserat 杆运动学核心（hat/vee 映射、曲率、带状矩阵、前向运动学）。
- `dynamics_solver.py` — 动力学时间积分（Cauchy theta 法、低存储 RK4、Cosserat 动力学）。
- `inverse_kinematics.py` — 逆运动学与散乱数据插值。
- `rom_pod.py` — POD/SVD 降阶模型。
- `path_planner.py` — 动态规划路径规划。
- `validation_biharmonic.py` — 双调和方程精确解与验证。

## 模块详细说明

### 1. `mesh_utils.py`
**职责**：提供一维和二维网格生成工具，以及点集均匀性度量。

**核心功能**：
- 一维等距网格生成，支持五种不同的端点包含方式（函数 `line_grid`）。
- Chebyshev 节点生成（函数 `chebyshev_grid`）。
- 简单多边形（逆时针顶点）的扇形三角剖分（函数 `triangulate_polygon`）。
- 点集均匀性评估指标 — diaphony（函数 `diaphony_compute`）。
- 椭圆内均匀采样（函数 `sample_ellipse`）及多边形面积计算（`compute_polygon_area`）。
- 横截面网格细化，通过添加中心点将三角形细分（函数 `refine_cross_section_mesh`）。

**关键接口**：`line_grid(n, a, b, c)` 返回 `(n,)` 数组；`triangulate_polygon(vertices)` 返回 `(nodes, triangles)`；`refine_cross_section_mesh` 基于 `triangulate_polygon` 进行细化。

### 2. `section_fem.py`
**职责**：利用二次三角形单元（T6）和高斯求积计算软体机器人横截面的几何属性与刚度。

**核心功能**：
- T6 形函数及其在参考三角形上的梯度（`shape_t6`, `grad_shape_t6`）。
- 三角形上的高斯‑勒让德求积规则（`gauss_legendre_triangle`），支持不同精度阶数。
- 在物理三角形上数值积分（`integrate_triangle`），通过雅可比行列式进行坐标变换。
- 计算截面面积、形心、惯性矩、极惯性矩以及 diaphony 均匀性（`compute_section_properties`）。
- 基于能量等效原理计算 Timoshenko 剪切修正系数（`compute_shear_correction_factor`）。
- 组装平面应力刚度矩阵（`assemble_section_stiffness`），用于等效弹性模量分析。

**关键依赖**：`mesh_utils.diaphony_compute` 用于评估节点均匀性。

### 3. `spectral_dg.py`
**职责**：提供 Chebyshev 谱微分矩阵和基于 Jacobi 多项式的间断 Galerkin 离散化基础。

**核心功能**：
- 构造 Chebyshev 谱微分矩阵（`chebyshev_matrix`），利用负和技巧处理对角元。
- 计算 Jacobi 多项式（`jacobi_polynomial`）和 Vandermonde 矩阵（`vandermonde_1d`）。
- 基于 Jacobi 多项式导数的 1D 微分矩阵（`dmatrix_1d`）。
- Jacobi‑Gauss‑Lobatto 节点生成（`jacobi_gauss_lobatto`）。
- DG lift 算子（`lift_1d`），用于将面通量提升到单元内部。
- 1D DG 空间导数计算（`dg_derivative_1d
