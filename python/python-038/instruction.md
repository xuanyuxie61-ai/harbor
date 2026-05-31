# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Jet Clustering and Parton Shower Simulation

该项目实现了一套完整的高能物理蒙特卡洛模拟管线，涵盖从部分子分布函数（PDF）演化、硬散射、部分子簇射、强子化、喷注重建到不确定性量化的全流程。项目基于多个经典数值算法（如 Hilbert 空间填充曲线、元胞自动机、打靶法、多项式混沌展开等），用于模拟 LHC 能量下质子-质子对撞中喷注的生成与性质。

## 文件概览

### `main.py`
主控脚本，按顺序执行以下步骤：
1. 验证所有核心数学库（特殊函数、三对角求解器、求积器、自适应采样、PDF 演化、部分子簇射、强子化、喷注重建、PCE 不确定性）。
2. 通过打靶法求解 DGLAP 方程，获得 PDF 演化结果。
3. 生成硬散射过程初态部分子。
4. 运行角度排序的部分子簇射（基于 Sudakov 否决算法）。
5. 使用 CVT 自适应采样分析部分子相空间密度。
6. 通过元胞自动机强子化模型将部分子转化为强子。
7. 对强子执行 anti‑kT、C/A、kT 喷注重建，并利用背包算法寻找最优子喷注组合。
8. 计算喷注形状变量（推力、球度、展宽）。
9. 利用多项式混沌展开量化 α_s 和 PDF 参数引起的不确定性。
10. 演示高维动量空间积分、三对角扩散方程求解（模拟喷注在 QGP 中的能量损失）以及 Hilbert 空间索引加速。

该脚本仅依赖其他模块提供的接口，不包含底层算法实现。

### `special_functions_qcd.py`
**QCD 特殊函数与物理常数库**
- 定义颜色因子常数（`NC`, `TF`, `CA`, `CF`, `N_F`, `BETA0`, `LAMBDA_QCD`）。
- 实现一阶和二阶 QCD 跑动耦合常数 `alpha_s_1loop`, `alpha_s_2loop`。
- 提供领头阶分裂函数 `p_qq_lo`, `p_qg_lo`, `p_gq_lo`, `p_gg_lo`（包含正则化处理）。
- 计算夸克 Sudakov 形状因子 `sudakov_quark`（通过数值双重积分）。
- 提供正交多项式（Legendre, Chebyshev, Hermite）的递推求值函数。
- 提供广义调和数 `harmonic_sum`、二重对数 `di_log`、以及单圈/双圈尖点反常维度系数。
- 包含一个验证函数 `validate_special_functions`，用于检查正交多项式和 α_s 的合理性。

### `tridiagonal_solver.py`
**三对角线性系统求解器（R83 格式）**
- 采用 R83 压缩存储格式（3×N 阵列）存储对称三对角矩阵。
- 提供共轭梯度法 `r83_cg_solve` 和循环约化法 `r83_cyclic_reduction` 两种求解器。
- 构建一维拉普拉斯离散化矩阵 `build_dif2_r83`。
- 实现隐式欧拉时间步进求解一维扩散方程 `solve_diffusion_1d`（用于模拟部分子级联在介质中的热化）。
- 包含自洽性测试 `test_tridiagonal_solvers`。

### `cubature_integrator.py`
**高维求积与数值积分**
- 一维复合积分 `integrate_1d_composite`（支持梯形和 Simpson 法）。
- 金字塔域（三维）上的 Jaskowiec‑Sukumar 求积规则 `integrate_pyramid`，高精度时回退到张量积 Gauss‑Legendre 求积。
- 超矩形域上的蒙特卡洛积分 `integrate_monte_carlo`。
- 自适应 Simpson 积分 `integrate_adaptive_1d`，适用于具有局部奇点的函数（如 QCD 分裂函数）。
- 测试函数 `test_cubature` 验证积分精度。

### `adaptive_sampling.py`
**自适应空间采样与索引**
- 实现三维 Hilbert 空间填充曲线：`hilbert_h_to_xyz`（索引→坐标）和 `hilbert_xyz_to_h`（坐标→索引），支持任意位数。
- 构建基于 Hilbert 排序的空间索引类 `HilbertSpatialIndex`，可将连续动量空间中的点映射到一维 Hilbert 序，并执行范围查询，用于加速喷注聚类中的最近邻搜索。
- 实现二维 Centroidal Voronoi Tessellation（CVT）的 Lloyd 迭代 `cvt_lloyd_2d`，根据用户提供的密度函数在给定域内生成自适应采样点。
- 测试函数 `test_adaptive_sampling` 验证 Hilbert 往返转换和 CVT 收敛性。

### `dglap_pdf.py`
**DGLAP 演化与部分子分布函数**
- 计算 Mellin 空间 LO 分裂函数 `mellin_moment_splitting`（使用复变量 digamma 函数）。
- 实现 Mellin 空间的 DGLAP 演化 `dglap_mellin_evolve`，通过 2×2 矩阵对角化解析求解单态演化。
-
