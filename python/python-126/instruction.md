# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 全身 PBPK 药物代谢动力学一体化建模系统

## 项目概述

本项目构建一个面向前沿生物医学问题的全身多器官生理药代动力学（PBPK）建模与仿真系统。它融合了特殊函数计算、随机采样、高维数值积分、偏微分方程离散、随机微分方程模拟、刚性 ODE 求解、几何建模、多项式模型、插值方法、动态规划优化以及数据校验工具等多个模块，用于模拟药物在人体各器官中的分布、代谢与毒性，支持不确定性量化和剂量优化。

项目入口为 `main.py`，它通过调用其余 11 个功能模块，完成一系列计算并输出结果。在后续 benchmark 中，`main.py` 将被保留，其他 `.py` 文件将被删除，需要根据本描述重新实现这些缺失的模块。

## 核心科学问题

基于多尺度随机微分方程、稀疏网格不确定性量化、刚性 ODE 系统与动态规划剂量优化的全身药物分布‑代谢‑毒性一体化建模。

## 项目结构

- `main.py` — 主入口：调用所有模块，展示各模块功能，并执行综合 Monte Carlo 不确定性量化分析。
- `pbpk_special_functions.py` — 特殊函数与结合模型
- `pbpk_random.py` — 随机采样与顺序统计量
- `pbpk_quadrature.py` — 稀疏网格与二维求积
- `pbpk_diffusion.py` — 组织扩散方程与有限差分算子
- `pbpk_stochastic.py` — Feynman‑Kac 随机路径积分
- `pbpk_ode_solver.py` — 刚性 ODE 求解与 PBPK 动力学
- `pbpk_geometry.py` — 器官几何建模与 CVT 采样
- `pbpk_polynomials.py` — 多项式势能与多靶点目标
- `pbpk_interpolation.py` — 插值与浓度‑效应关系
- `pbpk_optimization.py` — 动态规划剂量优化
- `pbpk_utils.py` — 数据校验、统计与数值鲁棒性工具

## 模块详细说明

### 1. `pbpk_special_functions.py`
**职责：** 提供药物‑蛋白结合、非均质扩散等所需的特殊函数。

**核心函数/类：**
- Carlson 对称椭圆积分 `carlson_rf`, `carlson_rd`, `carlson_rc`
- Jacobi 椭圆函数 `jacobi_sncndn`（使用 Landen 变换）
- Gauss 算术‑几何平均 `gauss_agm`
- Jacobi theta 函数 `jacobi_theta`
- 高斯超几何函数 `hyper_2f1`（级数计算，含 Euler 变换）
- 药物‑蛋白结合分数 `drug_protein_binding_fraction`（利用超几何函数）
- 非均质组织有效扩散系数 `effective_diffusion_coefficient`（利用 AGM）

**主要算法：** 对称椭圆积分的迭代倍缩算法，Landen 变换，AGM 迭代，超几何级数求值，Euler 变换处理收敛域。

**依赖：** numpy，内部无模块间依赖。

### 2. `pbpk_random.py`
**职责：** 生成排序随机向量，用于模拟药物到达时间和生理参数采样。

**核心函数：**
- 均匀顺序统计量：指数间距法 `r8vec_uniform_01_sorted_exponential` 和递归乘积法 `r8vec_uniform_01_sorted_product`
- 标准正态逆 CDF `normal_01_cdf_inv`（Wichura 算法 AS 241）
- 正态顺序统计量 `r8vec_normal_01_sorted`（通过均匀统计量 + 逆 CDF）
- 药物到达时间模拟 `sample_drug_arrival_times`（对数正态分布）
- 生理参数 Latin‑Hypercube 风格采样 `sample_physiological_params`

**主要算法：** Dirichlet 分布次序统计量生成，Wichura 分位数逼近，逆 CDF 变换。

**依赖：** numpy，内部无模块间依赖。

### 3. `pbpk_quadrature.py`
**职责：** 高维数值积分，用于参数空间不确定性量化和器官切片浓度积分。

**核心函数：**
- 一维 Clenshaw‑Curtis 节点与权重 `cc_abscissa`, `cc_weights`
- Smolyak 稀疏网格生成 `_generate_sparse_grid_direct` 和积分接口 `sparse_grid_integrate`
- 二维 Felippa 乘积求积规则 `square_felippa_rule`（支持 1/3/5 阶 Gauss‑Legendre）
- 正方形单项式精确积分 `square_monomial_integral`
- 器官切片积分 `integrate_organ_slice`
