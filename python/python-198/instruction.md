# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 通信避免的多项式混沌展开-有限体积法框架 (CA-PCE-FVM)

## 项目概述
本项目实现了一个面向分布式不确定性传播的高性能计算框架。核心思想是将通信避免的 s-step Krylov 子空间方法应用于多项式混沌展开（PCE）与有限体积法（FVM）耦合系统，以加速带有随机边界条件和对流系数的二维无粘 Burgers 方程的求解。项目包含多个独立模块，覆盖网格生成、随机场建模、PCE 基函数、流体求解器、矩阵指数时间积分、通信建模、线性求解器、蒙特卡洛验证、稀疏 I/O、精确解基准和焦散分析等功能。

## 模块说明

### 1. `mesh_geometry.py` – 三角剖分与域分解
负责在单位圆盘上生成结构化三角网格，并提供几何计算与分区工具。
- 核心功能：
  - `generate_disk_triangulation`：按径向层数与角度分段生成节点与三角形单元，同时标记边界节点。
  - 计算三角形面积、重心、网格质量指标（基于形状归一化）。
  - 提取边界边，通过出现次数识别内部/边界边，并将边界边排序为闭合环。
  - 构建节点邻接图；利用递归坐标二分法（RCB）进行域分解，返回每个节点的分区编号。
  - 识别跨两个分区单元的界面节点，用于通信图构建。
- 输入：径向层数、角度分段数；输出：节点坐标、单元列表、边界掩码、分区数组等。

### 2. `random_parameters.py` – 截断正态随机参数
提供截断正态分布的统计建模与采样工具，用于生成随机边界条件或参数场。
- 核心功能：
  - `truncated_normal_ab_pdf`：计算双边截断正态分布的概率密度。
  - `truncated_normal_ab_mean`、`truncated_normal_ab_variance`：解析计算截断正态的均值和方差。
  - `truncated_normal_ab_sample`：通过逆变换采样生成截断正态随机数。
  - `generate_kl_coefficients`：生成 Karhunen-Loève 展开的随机系数，用于参数化随机场（简化的指数协方差模型）。
- 使用 scipy 的误差函数完成 CDF 的解析计算。

### 3. `pce_basis.py` – 多项式混沌基函数
实现概率学家版本的 Hermite 多项式及其正交性积分，并构建 PCE-Galerkin 系统。
- 核心功能：
  - Hermite 多项式递推计算：`hermite_he_prob` 及批量版本。
  - 正交性：双积积分 `he_double_product_integral`（返回阶乘）和三积积分 `he_triple_product_integral`（利用线性化公式计算归一化三阶矩）。
  - `build_pce_galerkin_matrix`：针对随机系数为 `α = μ + σ ξ` 的简单随机 ODE，生成 PCE-Galerkin 耦合矩阵（维度为阶数+1）。
  - Vandermonde 矩阵构造与求解（带条件数保护），以及 Newton 基的广义 Vandermonde 矩阵，为 s-step Krylov 方法提供更好的基。

### 4. `burgers_fvm.py` – 无粘 Burgers 方程有限体积求解器
基于三角网格的二维无粘 Burgers 方程守恒式求解器，支持多种数值通量。
- 核心功能：
  - 数值通量函数：
    - `godunov_flux`：精确求解 Riemann 问题（包含稀疏波与激波处理，带截断防溢出）。
    - `lax_friedrichs_flux`：中心格式加扩散修正，依赖局部最大波速 `α`。
    - `upwind_flux`：简单迎风格式。
  - `build_fvm_operators`：在给定节点和三角形单元的基础上计算单元面积、重心，并建立内部边（附带左/右单元标号、边长、单位法向）与边界边列表。
  - `burgers_fvm_step`：单步时间推进（显式 Euler），通过内部边和边界边（Dirichlet）累积通量贡献，更新所有单元的值。
  - `solve_burgers_fvm`：完整时间积分，包含 CFL 自适应调整和动态安全监控，返回解历史。
- 输入：节点、单元、初始条件函数、时间参数、通量类型等。

### 5. `matrix_exponential_int.py` – 矩阵指数与精确时间积分
基于 scaling-and-squaring 结合 Padé 近似的矩阵指数计算，用于 PCE 系统的高精度时间推进。
- 核心功能：
  - `matrix_exponential_pade`：对任意方阵计算矩阵指数 `exp(A)`，先缩放至范数小于 1，再利用 Padé (q,q) 有理逼近，最后反复平方恢复。
  - `pce_matrix_exponential_step`：单步将 PCE 系数向量乘以 `exp(-A dt)`。
  - `pce_matrix_exponential_integrate`：全时间积分，返回各时刻的 PCE 系数历史。

### 6. `communication_model.py` – 通信模型与延迟分析
建立分布式计算中的通信延迟模型，并通过 TSP 路径优化通信调度。
- 核心功能：
  - 计算点集间的距离矩阵。
  - 贪心 TSP 求解：单起点贪心构造路径，多起点比较选取最短路径。
  - `model_communication_latency`：基于消息大小和模型参数（启动延迟 `α`，传输率 `β`）并加入随机抖动模拟通信时间。
  - 圆盘上均匀随机点对距离的蒙特卡洛估计（对比理论值）。
  - `ca_speedup_theory`：推导 s-step 通信避免方法的理论加速比（计算与通信时间的函数）。
  - `optimize_s_parameter`：在给定的计算/通信时间比下寻找最优聚合步数 `s`。
  - 生成不同拓扑（环形、TSP 优化、全连接）的处理器通信调度表。

### 7. `ca_sstep_solver.py` – 通信避免 s-step Krylov 求解器
实现标准及通信避免的 Arnoldi 过程和 GMRES 求解器，减少正交化中的全局通信。
- 核心功能：
  - `power_basis_arnoldi`：标准 Arnoldi 过程，每次迭代进行一次全正交化，返回 Arnoldi 基和上 Hessenberg 阵。
  - `ca_sstep_arnoldi`：通信避免的 s-step Arnoldi，每 s 步聚合一次矩阵乘幂（局部计算），然后在块内进行 Gram-Schmidt 正交化，仅每 s 步做一次全局内积。支持控制块大小和总维度。
  - `gmres_solve`：闭包 GMRES 求解器，可选择标准模式或 s-step 模式，支持重启，返回解和残差历史。
  - `apply_pce_ca_gmres`：针对块稀疏 PCE-Galerkin 矩阵封装的应用接口（简化为稠密验证）。

### 8. `monte_carlo_uq.py` – 蒙特卡洛不确定性量化验证
基于随机采样的统计验证模块，用于对照 PCE 方法的精度。
- 核心功能：
  - 在单位球和椭圆域内均匀采样（利用 Cholesky 变换）。
  - `monte_carlo_pce_verify`：对原型随机 ODE `du/dt = -α(ξ) u` 进行蒙特卡洛模拟，计算终态统计量，并与 PCE 解析均值对比，计算相对误差。
  - 圆盘上随机距离统计的蒙特卡洛估计。

### 9. `sparse_io.py` – 稀疏矩阵 I/O 与数据结构
提供稀疏矩阵的 CSR 格式转换和 Harwell-Boeing 格式读写，以及构建 PCE 块稀疏矩阵的工具。
- 核心功能：
  - `dense_to_csr` / `csr_to_dense`：稠密矩阵与 CSR 存储格式的相互转换。
  - `write_hb_simple`：将稠密矩阵写入
