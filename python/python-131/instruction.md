# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 浆态床气泡柱反应器 CFD-PBM 耦合模拟器

## 项目概述
本项目实现一个浆态床气泡柱反应器（Slurry Bubble Column Reactor, SBCR）的耦合数值模拟系统。  
系统将两相流体力学、群体平衡方程（PBE）、Fischer-Tropsch 反应动力学与催化剂分布优化集成在一个统一的模拟框架中，支持稳态/准稳态仿真，并可用于敏感性分析和不确定性量化。

模拟器的核心组件包括：
- 结构化圆柱网格生成与边界处理
- 气液两相流场的非线性耦合求解（基于漂移流模型与动量守恒）
- 群体平衡方程的矩方法（QMOM）求解，描述气泡尺寸分布演变
- 反应器内部温度场与物种浓度场的迭代求解
- 催化剂在不同轴向段的最优分布（背包问题与 Diophantine 整数规划）
- 谱方法数值积分、随机入口条件生成、操作时间线与日期计算等辅助模块

程序入口为 `main.py`，通过一次运行即可完成全部子模块的功能验证和完整的耦合模拟，并输出关键结果。

## 文件与模块职责

### `main.py`
- 程序主入口。
- 依次调用所有合成模块，进行单元测试并运行完整的 SBCR 耦合模拟 (`run_all_tests_and_simulation`)。
- 输出网格质量、数值基准误差、流动场、温度场、物种浓度、催化剂优化等汇总结果。

### `reactor_mesh.py`
- **网格生成与边界处理**。
- 提供圆柱坐标下结构化四边形网格的生成 (`generate_cylindrical_mesh`)。
- 提供网格边界线段提取 (`mesh_boundary_segments`) 以及边界分形扰动 (`boundary_perturb`) 以模拟粗糙壁面或分布板。
- 提供网格单元的 Jacobian 行列式计算 (`compute_jacobian_2d`) 和网格质量报告 (`mesh_quality_report`)。

### `spectral_quadrature.py`
- **谱方法与高级数值积分**。
- 提供 Gauss‑Legendre 积分节点/权重的构造 (`legendre_nodes_weights`) 及积分函数。
- 提供 Alpert 混合积分规则 (`alpert_log_integral`)，用于处理对数奇异性。
- 提供 Chebyshev 级数评估 (`chebyshev_eval`) 和系数计算 (`chebyshev_coefficients`)，用于函数逼近。
- 提供基于 Smolyak 构造的稀疏网格积分 (`sparse_grid_gauss_legendre`)，用于高维矩空间积分。

### `nonlinear_solver.py`
- **非线性方程组求解器**。
- 提供定点迭代法 (`fixed_point_iteration`) 和带阻尼线搜索的 Newton 法 (`newton_solver`)。
- 定义气泡柱反应器的局部代数残差 (`reactor_algebraic_residual`) 及其数值 Jacobian，用于求解气含率与液速的耦合关系。

### `momentum_equations.py`
- **动量方程与相间作用模块**。
- 实现 Hartmann 磁流体解析解 (`HartmannFlow` 类)，作为数值基准验证。
- 提供 Schiller‑Naumann 阻力系数、相间动量交换 (`interphase_momentum_exchange`)、浆态有效粘度 (`effective_viscosity_slurry`) 以及简化的一维两流体动量残差计算。

### `population_balance.py`
- **群体平衡方程 (PBE) 模块**。
- 实现气泡成核的 Poisson 过程模拟 (`poisson_nucleation_events`)。
- 提供 Lehr‑Mewes 破裂频率核和 Prince‑Blanch 聚并核。
- 通过 Wheeler 算法 (`wheeler_algorithm`) 从低阶矩恢复高斯积分节点与权重，进而计算矩方程源项 (`moment_source_qmom`)。
- 使用 QMOM 方法对 PBE 进行时间推进 (`qmom_integrate_pbe`)，更新气泡尺寸分布的矩。

### `catalyst_optimization.py`
- **催化剂分布优化模块**。
- 对连续型 0‑1 背包问题进行暴力枚举 (`knapsack_brute_force`)，用于低维催化剂分配。
- 对有界 Diophantine 方程枚举所有整数解 (`diophantine_bounded_solutions`)。
- 基于阿伦尼乌斯动力学和 Fischer‑Tropsch 转化率模型计算各段催化剂的经济价值 (`catalyst_value_per_segment`)。
- 整合以上算法，提供催化剂分布优化函数 (`optimize_catalyst_loading`)，支持暴力枚举与 Diophantine 两种策略。

### `numerical_linear_algebra.py`
- **数值线性代数工具**。
- 提供正交矩阵行列式的 Gower AS 82 算法 (`detq_orthogonal`) 及网格变换正交性检查。
- 提供带阻尼的幂法求主特征向量 (`power_iteration_eigenvector`)，受 PageRank 启发。
- 提供稳态对流‑扩散浓度场的迭代求解器 (`steady_state_concentration_solver`)。
- 提供矩阵谱条件数估计 (`estimate_condition_number`)。

### `stochastic_inlet.py`
- **随机入口条件生成**。
- 基于正态分布扰动生成入口温度、CO/H₂ 摩尔分数、体积流率的统计样本 (`generate_inlet_conditions`)。
- 支持对一维分布施加随机扰动 (`generate_perturbed_profile`)。

### `reactor_operations.py`
- **反应器操作时间线**。
- 提供 Gregorian 日期与 Julian Day Number 的互转、闰年判断等基础日期计算。
- 提供反应器批次循环时间线计算 (`reactor_operation_timeline`) 和考虑计划停机的年操作日历 (`operating_calendar_year`)。

### `cfd_solver.py`
- **CFD‑PBM 耦合求解器主模块**。
- 定义 `SlurryBubbleColumnReactor` 类，封装反应器几何、流场、矩、温度等状态。
- 集成上述所有模块，实现完整的模拟流程：
  1. 生成网格与随机入口条件；
  2. 用 Newton/定点迭代求解气‑液流场；
  3. 用 QMOM 更新气泡尺寸分布矩；
  4. 计算温度剖面与物种浓度分布；
  5. 评估催化剂最优分布；
  6. 执行 Hartmann 基准验证和网格质量检查。
- 提供 `run_simulation` 方法，一次性执行耦合模拟并返回综合结果字典。

## 关键算法与数学思想
- **两相流场耦合**：通过漂移流模型与总体积通量守恒建立代数系统，使用带阻尼 Newton 法求解气含率与液速。
- **群体平衡求解**：采用矩方法 (QMOM)，Wheeler 算法从矩恢复积分节点，计算破裂/聚并源项；源项基于 Lehr 破裂核和 Prince‑Blanch 聚并核。
- **催化剂优化**：将反应器分段，基于转化率模型计算每段催化剂价值，用暴力背包枚举或 Diophantine 整数规划寻找最优分布。
- **反应器热/质耦合**：一维轴向能量平衡和稳态对流‑扩散‑反应方程，通过松弛迭代求解 CO 浓度分布和温度场。
- **数值基准**：使用 Hartmann 磁流体解析解验证动量方程数值方法。

## 模块间依赖关系
- `main.py` 调用所有模块。
- `cfd_solver.py` 是顶层模拟器，依赖于除 `main.py` 外的所有其他模块。
- `population_balance.py` 依赖 `spectral_quadrature.py` 以获取积分规则。
- `nonlinear_solver.py` 被 `cfd_solver.py` 和 `population_balance.py` 调用。
- `catalyst_optimization.py` 相对独立，仅依赖 NumPy。
- `numerical_linear_algebra.py`、`stochastic_inlet.py`、`reactor_operations.py` 被 `cfd_solver.py` 和/或 `main.py` 使用，但彼此基本无横向依赖。

## 项目使用
运行 `python main.py` 即可执行完整测试与模拟流程。无需命令行参数。输出包含网格质量、流场
