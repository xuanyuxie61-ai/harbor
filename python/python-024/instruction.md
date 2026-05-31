# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：太阳耀斑磁重联数值模拟平台（python-024）

## 背景与任务
本项目是一个面向太阳耀斑磁重联物理过程的数值模拟平台，由多个科学计算模块构成。入口文件 `main.py` 已被保留，它通过导入其他模块来组织完整的模拟流程，并调用各个模块的演示函数进行独立验证。

**你的任务**：根据本描述，重新实现所有被 `main.py` 引用的辅助模块文件（除 `main.py` 外的所有 `.py` 文件）。每个模块需提供清晰的接口和基本功能，使得 `main.py` 能够正常导入并执行。

## 需要实现的文件列表

| 文件名 | 主要职责 |
|--------|----------|
| `harris_equilibrium.py` | Harris 电流片平衡态物理场计算、四边形网格生成与插值 |
| `continuation_solver.py` | 伪弧长延拓法追踪非线性系统平衡态分支 |
| `fem_assembler.py` | 一维有限元矩阵组装及稀疏矩阵格式转换 |
| `mhd_stability.py` | MHD 线性撕裂模稳定性分析，包含 Hankel 矩阵与整数 RREF |
| `resistivity_evolution.py` | 反常电阻率的反应‑扩散方程求解及波动模型 |
| `particle_acceleration.py` | 粒子加速的蒙特卡洛模拟及非线性轨道追踪 |
| `field_rotation.py` | 四元数磁场旋转与拓扑对称性分析 |
| `periodic_interpolation.py` | 周期三角插值与极坐标‑笛卡尔数据重排 |
| `wedge_flux.py` | 楔形区域上的精确积分与求积规则验证 |

每个模块还应包含一个无参数的 `demo_*()` 函数（名称与文件对应），供独立测试和 `main.py` 调用，其输出可自由设计，但必须可安全运行。

---

以下各节分别描述每个模块的详细接口与核心逻辑，但不提供完整实现细节。所有模块均依赖 `numpy`，部分模块可能依赖 `scipy` 的子模块（如 `scipy.sparse`、`scipy.linalg`、`scipy.special` 等）。

## 1. `harris_equilibrium.py` — Harris 电流片平衡态与网格

### 类：`HarrisEquilibrium`
用于模拟太阳日冕中经典的 Harris 电流片平衡态，提供各种物理场的一维分布以及二维四边形网格上的场插值。

**构造参数**  
- `B0` : 背景磁场强度（正数）  
- `lambda_cs` : 电流片半厚度（正数）  
- `B_guide` : 引导场大小  
- `p0` : 背景压强  
- `T_plasma` : 等离子体温度（正数）  
- `rho_inf` : 远场背景密度  
- `y_max` : 计算域沿 y 方向的半宽（正数）  

**主要方法**
- `B_field(y)` → 形状为 `(n, 3)` 的 ndarray：由 y 坐标计算磁场向量 `(Bx, By, Bz)`。Bx 遵循双曲正切分布，导出场为零。
- `pressure(y)` → 1‑D array：根据压强平衡条件计算热压强分布。
- `current_density(y)` → `(n, 3)` ndarray：由安培定律得出的电流密度向量，其 z 分量非零。
- `mass_density(y)` → 1‑D array：由状态方程和密度模型给出质量密度。
- `alfven_speed(y)` → 1‑D array：基于局地磁场和密度计算阿尔芬速度。
- `plasma_beta(y)` → 1‑D array：等离子体热压与磁压之比。
- `generate_quadrilateral_mesh(nx, ny)` → `(nodes, elements)`：在二维区域 `[0, 2*y_max] × [-y_max, y_max]` 上生成结构化四边形网格，y 方向可在电流片附近通过非线性变换加密。`nodes` 形状 `(nnodes, 2)`，`elements` 形状 `(nelems, 4)`，元素按逆时针编号。
- `bilinear_interpolate_on_mesh(nodes, elements, field_1d, y_coords)` → 1‑D array：将一维物理场（由 `y_coords` 采样）通过双线性插值映射到网格节点上，返回节点上的场值。
- `compute_reconnection_rate(y, eta, v)` → 1‑D array：基于广义欧姆定律计算重联电场 Ez。
- `magnetic_shear(y)` → 1‑D array：返回磁剪切率 dBx/dy。

**演示函数** `demo_harris()`  
用于快速检查基本物理量在电流片中心的值，无需参数。

### 物理常数
模块内可定义常用国际单位制常数（如真空磁导率、玻尔兹曼常数等）。

---

## 2. `continuation_solver.py` — 伪弧长延拓法

### 类：`ContinuationSolver`
实现伪弧长延拓法，用于跟踪含参数的非线性方程组 \(F(U,\eta)=0\) 的解曲线。采用自适应步长、预测‑校正步骤，并在迭代过程中动态切换延续参数。

**构造参数**  
- `max_iter` : Newton 迭代最大步数  
- `tol` : 收敛容差  
- `h_init` : 初始弧长步长  
- `h_min`, `h_max` : 步长自适应范围  
- `verbose` : 是否打印调试信息  

**核心方法**
- `trace_branch(F, J, x0, param_index=-1, n_steps=50)` → `(xs, params, ps)`  
  `F` 为残差函数（接受 ndarray 返回 ndarray），`J` 为雅可比函数，`x0` 是包含状态变量与参数分量的初始解向量，`param_index` 指定初始延续参数的索引（默认为最后一个分量）。返回解序列列表 `xs`、参数值列表 `params` 以及每步使用的延续参数索引列表 `ps`。  
  内部逻辑：  
  - 计算在当前解处的单位切向量（通过解增广线性系统并归一化），同时选择绝对值最大的分量作为下一个延续参数。  
  - 沿切向做线性预测，随后固定该延续参数进行 Newton 校正。  
  - 若 Newton 未收敛，则缩小步长重试；成功则适当增大步长。  
  - 若步长过小或切向量退化则提前终止。  

该类还包含私有方法 `_newton_step`（执行带固定参数约束的增广 Newton 迭代）和 `_compute_tangent`（计算切向量并选择下一延续参数），无需由外部直接调用。

**演示函数** `demo_mhd_continuation()`  
使用一个简单的三阶多项式系统演示延拓法，追踪 fold 分歧结构。返回解序列和参数序列。

---

## 3. `fem_assembler.py` — 有限元矩阵组装

### 类：`FEM1DAssembler`
处理一维区间上的有限元离散（线性帽子函数，Neumann 边界条件）。

**构造参数**  
- `n_elements` : 剖分单元数（≥2）  
- `domain` : (左端点, 右端点) 元组  

**主要方法**
- `mass_matrix(sparse=True)` : 返回三对角质量矩阵（稀疏或稠密），对角及次对角元素与网格步长 h 相关。
- `stiffness_matrix(sparse=True)` : 返回三对角刚度矩阵。
- `hat_function(x, node_idx)` : 计算指定节点的帽子函数在点 `x` 处的值。
- `project_initial(w0_func)` : 将初始条件函数投影到有限元空间，通过数值积分右端项并解线性系统。
- `solve_steady(K, F, sparse)` : 求解稳态方程 `K u = F`，通过固定第一个节点处理 Neumann 边界的奇异性。

### 类：`
