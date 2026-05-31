# PROJECT_194：高性能计算域分解并行有限元求解器 — 博士级合成说明

## 一、项目概述

本项目围绕**高性能计算：域分解并行有限元**这一前沿科学领域，将 15 个种子科研项目的核心算法融合重构为一个面向博士级难度的完整计算框架。项目求解的科学问题是：

> **基于重叠 Schwarz 域分解的高阶谱元离散方法，求解二维不可压缩瞬态 Stokes 方程。**

该问题涵盖了并行计算、偏微分方程数值解、谱方法、自适应时间积分、制造解验证、负载均衡与性能评估等多个高阶研究方向，其理论深度与工程复杂度均达到博士科研水平。

---

## 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法/思想 | 在合成项目中的角色 |
|:---:|--------|--------------|------------------|
| 1 | `987_r8pbl` | 对称正定带状矩阵紧凑存储 (R8PBL) | **子区域直接求解器**：每个子域的局部刚度矩阵以带状 SPD 格式存储，使用 O(n·m²) 的 Cholesky 分解完成局部精确求解 |
| 2 | `258_cvt_metric` | 各向异性 Centroidal Voronoi Tessellation (CVT) | **域分解网格划分**：基于空间变化度量张量的 Lloyd 算法生成负载均衡的子区域划分 |
| 3 | `1391_velocity_verlet` | Velocity Verlet 辛积分 | **二阶 ODE 时间推进**：将 Stokes 方程视为二阶系统时的辛时间积分器 |
| 4 | `567_hypersphere_positive_distance` | 超球面正象限采样与距离统计 | **划分质量评估**：利用超球面距离统计量量化子区域形状质量与负载不均衡度 |
| 5 | `899_polyomino_parity` | 多格拼板奇偶性与整数约束 | **Diophantine 整数负载平衡**：将单元分配建模为带非负约束的线性 Diophantine 方程 |
| 6 | `571_ice_to_medit` | ICE 到 MEDIT 网格格式转换 | **子区域网格拓扑 I/O**：提取子域边界节点与界面拓扑结构 |
| 7 | `510_hb_to_st` | Harwell-Boeing 到稀疏三元组格式转换 | **稀疏刚度矩阵组装**：局部与全局稀疏矩阵的坐标格式存储与格式互转 |
| 8 | `1225_test_partial_digest` | 部分消化问题 (PDP) | **界面节点匹配与重建**：利用距离几何从一维距离信息重建界面节点排序 |
| 9 | `288_diophantine` | 多元线性 Diophantine 方程求解 | **整数负载平衡**：求解 `Σ m_i = N_e` 的整数解以实现单元数均衡分配 |
| 10 | `1227_test_unimodal` | 40 个单峰测试函数 | **线搜索与非线性优化**：Schwarz 迭代中的重叠率优化与下降方向步长搜索 |
| 11 | `1172_stokes_2d_exact` | 二维 Stokes 精确制造解 | **收敛性验证**：提供多项式、三角与 Kovasznay 三种制造解用于数值精度验证 |
| 12 | `688_linpack_bench_backslash` | LINPACK 稠密求解基准 | **子区域求解器性能评估**：MFLOPS、残差范数与归一化残差比 |
| 13 | `1254_tetrahedron_slice_display` | 平面-四面体求交 | **三维子区域界面提取**：计算分割平面与四面体单元的交截面多边形 |
| 14 | `678_line_fekete_rule` | 区间上 Fekete 点与求积权重 | **谱元高阶求积**：Gauss-Lobatto-Legendre (GLL) 节点作为谱元插值/求积点 |
| 15 | `101_blowup_ode` | 有限时间爆破 ODE | **自适应时间步长控制**：监测解的增长率 γ，检测数值爆破并自适应调整 dt |

---

## 三、新增数学物理模型与核心公式

### 3.1 控制方程：不可压缩瞬态 Stokes 方程

在区域 `Ω = [0,1]²` 上，未知量为速度场 `u(x,y,t) = (u, v)` 与压力场 `p(x,y,t)`，控制方程为：

```
ρ ∂u/∂t - ν ∇²u + ∇p = f    在 Ω × (0,T] 内        (动量方程)
∇ · u = 0                     在 Ω × (0,T] 内        (不可压缩条件)
u = 0                         在 ∂Ω × (0,T] 内        (无滑移边界)
u(x,y,0) = u₀(x,y)            在 Ω 内                (初始条件)
```

其中：
- `ρ` 为流体密度（取 1.0）
- `ν` 为运动粘性系数（取 1.0）
- `f = (f_x, f_y)` 为体积力（由制造解导出）

### 3.2 制造解 (Manufactured Solutions)

**多项式制造解**（精确满足无滑移边界与散度为零）：

```
u(x,y) =  2 x² (x-1)² y (2y-1) (y-1)
v(x,y) = -2 x (2x-1) (x-1) y² (y-1)²
p(x,y) = x(1-x)y(1-y)
```

验证：`∂u/∂x + ∂v/∂y = 0` 在 `Ω` 上恒成立。

对应的强制项由代入控制方程得到：

```
f_x = -ν(∂²u/∂x² + ∂²u/∂y²) + ∂p/∂x
f_y = -ν(∂²v/∂x² + ∂²v/∂y²) + ∂p/∂y
```

**三角制造解**：

```
u(x,y) =  sin(πx) cos(πy)
v(x,y) = -cos(πx) sin(πy)
p(x,y) =  sin(πx) sin(πy)
```

### 3.3 各向异性 CVT 域分解能量泛函

给定 SPD 度量张量场 `M(x)`，CVT 能量泛函为：

```
E({z_i}) = Σ_i ∫_{V_i} ρ(x) d_M(x, z_i)² dx
```

其中各向异性距离定义为：

```
d_M(x, y) = √[(x-y)ᵀ M((x+y)/2) (x-y)]
```

Lloyd 迭代通过蒙特卡洛采样近似计算质心：

```
z_i^{new} = (1/|S_i|) Σ_{x∈S_i} x
```

`S_i` 为落入 Voronoi 单元 `V_i` 的采样点集合。

### 3.4 带状 SPD Cholesky 分解

对于带宽为 `m` 的 `n×n` SPD 带状矩阵 `A`，Cholesky 因子 `L` 满足 `A = L Lᵀ`，其元素按列计算：

```
L_{jj} = √(A_{jj} - Σ_{k=max(0,j-m)}^{j-1} L_{jk}²)

L_{ij} = (A_{ij} - Σ_{k=max(0,i-m)}^{j-1} L_{ik} L_{jk}) / L_{jj}
         对于 j < i ≤ min(n-1, j+m)
```

计算复杂度：`O(n m²)` 次浮点运算。

### 3.5 重叠 Schwarz 迭代

全局离散系统：

```
[A   Bᵀ] [u]   [f]
[B   0 ] [p] = [g]
```

对第 `i` 个子域 `Ω_i`（含重叠层），局部残差方程为：

```
[A_i   B_iᵀ] [δu_i]   [r_f^i]
[B_i   0   ] [δp_i] = [r_g^i]
```

其中 `r_f^i = f_i - A_i u_i - B_iᵀ p_i`，`r_g^i = g_i - B_i u_i`。

局部系统通过 Uzawa 迭代求解：

```
u^{k+1} = A_i⁻¹ (r_f - B_iᵀ p^k)
p^{k+1} = p^k + α (B_i u^{k+1} - r_g)
```

步长 `α = 1 / (B_iᵀ A_i⁻¹ B_i)` 由幂迭代预估计。

全局更新采用**加法型 Schwarz**：

```
u^{new} = u^{old} + Σ_i R_iᵀ δu_i
```

重叠区域上的更新取平均值。

### 3.6 谱元 Fekete (GLL) 求积

参考区间 `[-1,1]` 上的 Gauss-Lobatto-Legendre 节点 `ξ_j` 满足：

```
(1 - ξ²) P'_p(ξ) = 0
```

即 `ξ_0 = -1`，`ξ_p = 1`，内部节点为 `P'_p` 的根。

对应权重：

```
w_j = 2 / [p(p+1) P_p(ξ_j)²]
```

满足 `Σ_j w_j = 2`，且精确积分所有次数 `≤ 2p-1` 的多项式。

通过仿射映射 `x = (b-a)/2 · ξ + (a+b)/2` 转换到任意区间 `[a,b]`，权重乘以 `(b-a)/2`。

### 3.7 自适应时间步进与爆破检测

采用半隐式 Euler 格式：

```
(I + dt A) u^{n+1} = u^n + dt f(t^n, u^n)
```

误差估计通过 Richardson 外推：全步解 `u_h` 与两步半步解 `u_{h/2}` 比较：

```
est = ||u_h - u_{h/2}||
```

增长率监测：

```
γ_n = ||u^{n+1}|| / ||u^n||
```

若连续 `max_blowup` 步满足 `γ_n > γ_max`，则判定存在数值爆破风险，将 `dt` 缩减为 `dt/4`；若 `dt < dt_min` 则终止计算。

### 3.8 瞬态 Stokes 分步投影法

第 1 步（预测）：

```
(ρ/dt I + ν A) u* = (ρ/dt) u^n + f^n - Bᵀ p^n
```

第 2 步（压力 Poisson 方程）：

```
-∇² φ = -(1/dt) ∇·u*
```

第 3 步（校正）：

```
u^{n+1} = u* - (dt/ρ) ∇φ
p^{n+1} = p^n + φ
```

### 3.9 Amdahl 并行效率定律

设串行比例为 `s`，并行处理器数为 `p`，则理论加速比与效率为：

```
S(p) = 1 / [s + (1-s)/p]
E(p) = S(p) / p
```

---

## 四、代码文件结构与改造说明

### 文件清单（共 10 个 `.py` 文件）

| 文件名 | 功能 | 改造来源 |
|--------|------|---------|
| `main.py` | 统一入口，零参数运行，串联所有模块 | 新建 |
| `sparse_matrix.py` | 带状 SPD 矩阵存储、Cholesky 分解、稀疏三元组 | `r8pbl` + `hb_to_st` |
| `mesh_partition.py` | 各向异性 CVT 划分、子域边界提取、重叠掩码 | `cvt_metric` + `ice_to_medit` |
| `fekete_quadrature.py` | GLL 节点/权重计算、Legendre 多项式、三角形 Fekete 近似 | `line_fekete_rule` |
| `stokes_manufactured.py` | 三种制造解及其强制项、离散残差计算 | `stokes_2d_exact` |
| `schwarz_solver.py` | 重叠 Schwarz 迭代、局部 Uzawa 求解、Diophantine 整数划分 | `diophantine` + `polyomino_parity` + `linpack_bench_backslash` |
| `time_integrator.py` | 自适应半隐式 Euler、Velocity-Verlet、分步投影法、爆破检测 | `velocity_verlet` + `blowup_ode` |
| `geometry_utils.py` | 超球面采样、平面-四面体求交、PDP 重建、界面匹配 | `hypersphere_positive_distance` + `tetrahedron_slice_display` + `test_partial_digest` |
| `optimization_utils.py` | 黄金分割搜索、回溯线搜索、幂迭代估计特征值 | `test_unimodal` |
| `performance_bench.py` | LINPACK 风格稠密/带状求解基准、Amdahl 效率估计 | `linpack_bench_backslash` |

### 关键改造细节

1. **所有 MATLAB 代码已改写为纯 Python**：去除了 `randn`、`zeros`、`plot3` 等 MATLAB 特有语法，改用 NumPy 实现。
2. **删除全部可视化内容**：`cvt_metric` 原有的 PNG 输出、`tetrahedron_slice_display` 的 3D 绘图、`stokes_2d_exact` 的 gnuplot 输出均已删除，仅保留数值计算核心。
3. **增强数值鲁棒性**：
   - Cholesky 分解中加入 `1e-12` 的扰动保护，防止因舍入误差导致非正定；
   - GLL 节点 Newton 迭代中加入 `1e-15` 的除零保护；
   - 自适应时间步进中设置 `dt_min` 与 `max_steps` 双重安全边界；
   - PDP 重建中使用容差匹配代替精确浮点相等判断。
4. **边界条件处理**：制造解自动满足 `[0,1]²` 上的 Dirichlet 零边界；局部求解器中通过惩罚项 `penalty=1e-8` 近似边界约束。

---

## 五、运行方式

```bash
cd Synthesis-project-python/194_synth_project
python main.py
```

程序零参数运行，内部自动设置所有物理参数、网格规模与时间范围。运行后将依次输出 7 个模块的数值结果，包括：

1. 各向异性 CVT 子域中心与负载均衡质量指标
2. GLL 求积节点/权重及其精度验证
3. Stokes 制造解的散度-free 验证与离散残差
4. 重叠 Schwarz 迭代收敛残差与制造解误差
5. 自适应时间积分结果与 Velocity-Verlet/分步投影对比
6. 几何界面提取、优化与 PDP 重建
7. 子区域求解器性能基准与 Amdahl 并行效率

---

## 六、科学问题的前沿性与难度说明

本项目所求解的**域分解并行谱元 Stokes 求解器**是当前计算流体力学 (CFD) 与高性能计算 (HPC) 交叉领域的前沿课题，其博士级难度体现在：

1. **多物理场耦合**：速度-压力 saddle-point 系统的数值处理涉及 inf-sup 稳定性条件；
2. **高阶谱元离散**：GLL 节点的代数精度与条件数控制需要深入的谱分析知识；
3. **域分解迭代理论**：重叠 Schwarz 方法的收敛性依赖于 Poincaré-Friedrichs 型不等式与粗空间校正；
4. **自适应算法设计**：时间步长的局部截断误差估计与爆破检测涉及非线性分析；
5. **整数优化与负载均衡**：Diophantine 约束下的最优划分属于组合优化与并行计算的交叉难题；
6. **制造解方法论 (MMS)**：通过符号微分构造精确解以严格验证数值代码的正确性，是科学计算软件工程中的标准实践。

---

## 七、参考文献与理论基础

- T. J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*, Dover, 2000.
- J. S. Hesthaven & T. Warburton, *Nodal Discontinuous Galerkin Methods*, Springer, 2008.
- A. Toselli & O. Widlund, *Domain Decomposition Methods — Algorithms and Theory*, Springer, 2005.
- M. G. Larson & F. Bengzon, *The Finite Element Method: Theory, Implementation, and Applications*, Springer, 2013.
- L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000.
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.

---

*本项目由 sci-project-synthesis-python skill 自动合成，所有 15 个种子项目均已真实融入代码与算法流程中，无遗漏、无挂名。*
