# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 格点 QCD 强子谱学博士级合成项目

基于 `python-034` 工作区，该项目实现了简化的格点 QCD 模拟流水线，用于从规范场构型出发，通过费米子传播子求解、关联函数构造、变分谱学分析和高阶积分，提取强子质量、衰变常数等物理量，并研究手征动力学和重整化群流。规范群取为 SU(2)，格点规模较小（4³×8），但保留了完整的核心算法链。

目标：保留 `main.py`，删除其余 `.py` 文件，再由其他 agent 按照本描述复现缺失的模块。所有代码应使用 Python 及 `numpy`、`scipy` 等标准科学计算库。

---

## 文件与模块职责

### 1. `lattice_gauge.py` — 格点几何、SU(2) 规范场与热化
- **核心类**：
  - `Lattice`：管理四维周期性晶格，提供坐标与一维索引互转（`site_index`/`index_to_site`），方向相邻点计算 `neighbor`。
  - `GaugeConfig`：存储 4 个方向的 SU(2) 连接变量（形状 `(4, nx, ny, nz, nt, 2, 2)`），提供 link 读写、随机化、plaquette 计算、Wilson 作用量（`wilson_action`）和平均 plaquette。
- **SU(2) 参数化**：通过球极投影将 SU(2) 矩阵映射为 R³ 向量，实现 `su2_stereographic_project` 与 `su2_stereographic_inverse`，便于数值优化。
- **热化**：`ifs_thermalize_gauge` 通过迭代函数系统（IFS）随机更新连接，模拟混沌热化过程。
- **需实现要点**：
  - 四元数与 2×2 复矩阵的互相转换，球极投影公式。
  - 周期边界与邻居索引。
  - Plaquette 计算，SU(2) 矩阵乘积与迹。
  - IFS 热化：随机选择两种更新模式（全局随机与局部扰动）。

### 2. `gauge_dg_update.py` — 间断 Galerkin 谱方法规范场演化
- **目的**：在虚构时间方向上用高阶谱方法对规范场进行冷却/演化，改善算符重叠。
- **核心功能**：
  - `compute_staple`：计算 Wilson 作用量对应的 staple 和。
  - `compute_force`：将 staple 转换为 su(2) 代数的力（反厄米部分）。
  - `dg_gauge_rhs`：由力场构建 right-hand side，引入 upwind 通量以反映空间耦合。
  - `dg_gauge_evolve`：使用五阶低存储 Runge‑Kutta (LSERK) 推进，将力向量通过球极投影更新链路变量。
- **需实现要点**：
  - SU(2) 链路的 staple 计算（所有非重复方向的正向与反向贡献）。
  - 力向量的 R³ 表示及 upwind 通量（力差）的构建。
  - LSERK 系数（可用参考系数）及时间步长控制。
  - 更新时需限制投影向量的大小，避免数值溢出。

### 3. `wilson_flow.py` — Wilson 梯度流平滑
- **目的**：通过虚构时间梯度流进一步抑制规范场的紫外涨落，改善传播子精度。
- **核心算法**：
  - `apply_flow_step_euler_backward`：隐式后向 Euler 推进，采用定点迭代求解力场。
  - `adaptive_midpoint_step`：隐式中点法，先求解中点，再外推至新时刻，可估计局部截断误差。
  - `wilson_flow_run`：集成时间步进，可选择 Euler 或中点法，适应步长。
- **需实现要点**：
  - 力项复用 `gauge_dg_update.compute_force`。
  - 后向 Euler 固定点迭代：用当前力更新投影坐标，阻尼混合，直至收敛。
  - 中点法：在 \(\theta \Delta t\) 处迭代求解中间值，然后外推。

### 4. `matrix_algebra.py` — 对称正定矩阵的打包 Cholesky 分解
- **目的**：用于变分分析中关联矩阵的预处理以及多维高斯采样。
- **核心函数**：
  - `r8pp_fa`：对打包格式存储的上三角矩阵做 Cholesky 分解，返回三角因子 R（打包存储）。
  - `r8pp_sl`：利用 R 求解三角方程组。
  - `dense_to_packed` / `packed_to_dense`：格式转换。
  - `spd_sample`：通过 Cholesky 因子生成多元高斯样本。
- **需实现要点**：
  - 打包格式：仅存上三角，按列优先顺序排列为长度为 `n*(n+1)/2` 的一维数组。
  - 标准 Cholesky 分解算法（按列进行平方根与消去迭代）。
  - 前向/后向替代求解。

### 5. `fermion_solver.py` — Wilson‑Dirac 算符与共轭梯度传播子求解
- **核心类**：`WilsonDiracOperator`，封装 Wilson‑Dirac 矩阵的稀疏应用。
  - 构造函数需格点、规范场、裸质量、边界相位（时间方向反周期，空间周期）。
  - `apply`：计算 D_w ψ，包含对角项和 hopping 项，使用简化的 2×2 gamma 矩阵。
  - `apply_dagger` 及 `apply_hermitian`：通过 γ₅ 厄米性实现。
- **核心函数**：
  - `solve_propagator_cg`：用 CG 求解 D_w S = source，实际求解正定系统 D_w† D_w S = D_w† source，残差归一化与收敛判定。
  - `point_source`：生成 δ 源。
  - `solve_all_propagators`：批量求解。
- **需实现要点**：
  - 简化 Dirac gamma 矩阵（可设为 Pauli 矩阵组合）。
  - Hopping 项的四方向偏移、链接变量乘积，正确处理边界相位。
  - CG 算法：矩阵向量积通过 `apply_hermitian` 实现，内积与残差计算采用复数场向量化。

### 6. `correlator_builder.py` — 强子关联函数构造与分析
- **核心功能**：
  - `lagrange_interpolate` 及 `lagrange_basis_value`：拉格朗日多项式插值。
  - `sech2_soliton`：生成 KdV 型 sech² 波包，用于增强重子算符的重叠。
  - `meson_correlator_pion`：赝标道 π 介子关联函数，对空间求和，利用 γ₅。
  - `baryon_correlator_nucleon`：简化的核子关联函数，结合孤子波包权重。
  - `correlator_effective_mass`：从关联函数计算有效质量。
  - `correlator_interpolated_mass`：插值后求有效质量。
- **需实现要点**：
  - 关联函数收缩：对传播子做简单的自旋缩并（γ₅ 或直接内积）。
  - 孤子波包的离散采样与归一化。
  - 有效质量的对数比计算，需处理近零值。

### 7. `variational_spectrum.py` — 变分能谱提取与优化
- **核心功能**：
  - `calccf` 与 `spline_eval`：Construct Hermite 三次样条（Conte‑de Boor 算法），用于平滑关联函数。
  - `muller_method`：Muller 法求复函数零点。
  - `gevp_solve`：求解广义本征值问题 C(t
