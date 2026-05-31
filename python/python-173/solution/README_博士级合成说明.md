# 自适应网格细化 (AMR) 求解二维对流-扩散-反应方程 — 博士级合成说明

## 项目概述

本项目围绕**计算数学：自适应网格细化 (Adaptive Mesh Refinement, AMR)** 领域，融合 15 个种子项目的核心算法，构建了一个面向前沿科学计算问题的博士级 Python 代码项目。

### 科学问题

在二维区域 $\Omega = [0,1] \times [0,1]$ 上求解**非定常对流-扩散-反应方程**：

$$
\frac{\partial u}{\partial t} + \mathbf{v} \cdot \nabla u = \nabla \cdot (D \nabla u) + R(u) + f(\mathbf{x}, t)
$$

其中：
- **对流速度场**：$\mathbf{v}(x,y) = (0.5(y-0.5), -0.5(x-0.5))$（旋转流）
- **扩散系数**：$D(x,y) = 0.01 + 0.02(x^2 + y^2)$（空间变系数）
- **反应项**：$R(u) = \alpha u(1-u)$，Fisher-KPP 型自催化反应，$\alpha = 5.0$
- **源项**：$f(x,y) = 2 \exp(-50((x-0.5)^2 + (y-0.5)^2))$

**边界条件**：Dirichlet 边界 $u|_{\partial\Omega} = 0$

**初始条件**：$u_0(x,y) = \exp(-100((x-0.3)^2 + (y-0.3)^2))$

### 物理背景

该方程描述活性物质在旋转流体中的输运与反应过程。Fisher-KPP 反应项支持行波解，最小波速由公式给出：

$$
c^* = 2\sqrt{\alpha D}
$$

Peclet 数 $Pe = |\mathbf{v}|L/D$ 刻画对流与扩散的竞争。当 $Pe \gg 1$ 时，对流主导，需要自适应网格在激波/边界层区域局部加密。

---

## 15 个种子项目的融合映射

| 序号 | 原项目 | 核心算法 | 合成后模块 | 科学作用 |
|:---:|:---|:---|:---|:---|
| 1 | `238_cvt` | Centroidal Voronoi Tessellation (Lloyd 迭代) | `cvt_mesh.py` | CVT 优化网格生成，最小化能量泛函 $F = \sum_i \int_{V_i} \rho(\mathbf{x})\|\mathbf{x}-\mathbf{z}_i\|^2 d\mathbf{x}$ |
| 2 | `1073_shepard_interp_nd` | n维 Shepard 逆距离加权插值 | `shepard_transfer.py` | 自适应网格细化/粗化后的解传递 (prolongation/restriction) |
| 3 | `387_fem1d_bvp_quadratic` | 1D 二次 FEM、Gauss 积分 | `fem_solver.py` | 扩展为 2D 三角形 P1 有限元，刚度矩阵组装与 Dirichlet/Neumann 边界处理 |
| 4 | `955_quadrilateral_mesh_rcm` | Reverse Cuthill-McKee 重排序 | `graph_mesh.py` | 网格节点 RCM 重排序，降低稀疏矩阵带宽，加速线性求解器 |
| 5 | `108_boundary_word_hexagon` | 六边形边界词 | `hex_boundary.py` | 六边形格点边界近似，计算边界细化指示子 |
| 6 | `1256_tetrahedron_witherden_rule` | 四面体高阶求积规则 | `quadrature_rules.py` | 三角形参考域高阶 Gauss 求积，用于 FEM 刚度矩阵精确组装 |
| 7 | `1173_string_pde` | 波动方程有限差分时间步进 | `time_integrator.py` | 时间积分策略 (BDF1/BDF2/Crank-Nicolson) 与自适应步长控制 |
| 8 | `544_hits` | HITS 权威/枢纽排序 | `graph_mesh.py` | HITS 算法对网格节点进行重要性排序，指导 AMR 标记策略 |
| 9 | `229_cube_arbq_rule` | 立方体求积规则 | `quadrature_rules.py` | 补充高维数值积分思想，构建三角形 Duffy 变换求积 |
| 10 | `1374_unstable_ode` | 刚性 ODE 稳定性分析 | `time_integrator.py` | 离散系统特征值分析、刚度比计算、隐式方法稳定性论证 |
| 11 | `014_approx_chebyshev` | Chebyshev 插值与误差估计 | `error_indicator.py` | 基于 Chebyshev 逼近的后验误差估计，指导自适应细化 |
| 12 | `1398_voronoi_plot` | Voronoi 图最近邻搜索 | `cvt_mesh.py` | Voronoi 单元最近发电机搜索，支撑 CVT Lloyd 迭代 |
| 13 | `352_fd1d_advection_diffusion_steady` | 1D 稳态对流扩散 | `advection_diffusion.py` | 扩展为 2D，组装对流矩阵、计算 CFL 条件、处理 Peclet 数 |
| 14 | `1350_triangulation_refine` | 三角形网格二分细化 | `triangulation_refine.py` | 边中点插入将三角形细分为 4 个子三角形，保持嵌套有限元空间 $V_h \subset V_{h/2}$ |
| 15 | `481_graph_adj` | 图邻接结构与连通性 | `graph_mesh.py` | BFS/DFS 网格连通性检验、最短路径距离计算 |

---

## 核心数学公式与算法

### 1. CVT 能量泛函与 Lloyd 迭代

对于密度函数 $\rho(\mathbf{x})$ 和生成元集合 $\{\mathbf{z}_i\}_{i=1}^n$，CVT 最小化能量：

$$
F(\{\mathbf{z}_i\}) = \sum_{i=1}^n \int_{V_i} \rho(\mathbf{x}) \|\mathbf{x} - \mathbf{z}_i\|^2 \, d\mathbf{x}
$$

Lloyd 迭代更新规则：

$$
\mathbf{z}_i^{(k+1)} = \frac{\int_{V_i^{(k)}} \rho(\mathbf{x}) \mathbf{x} \, d\mathbf{x}}{\int_{V_i^{(k)}} \rho(\mathbf{x}) \, d\mathbf{x}}
$$

收敛定理：$F^{(k+1)} \leq F^{(k)}$，等号成立当且仅当 $\mathbf{z}_i = \text{centroid}(V_i)$。

### 2. 三角形 P1 有限元弱形式

找 $u_h \in V_h$ 使得 $\forall v_h \in V_h^0$：

$$
\int_\Omega D \nabla u_h \cdot \nabla v_h \, d\mathbf{x} + \int_\Omega c u_h v_h \, d\mathbf{x} = \int_\Omega f v_h \, d\mathbf{x} + \int_{\Gamma_N} g_N v_h \, ds
$$

刚度矩阵单元贡献：

$$
A_{ij}^{(T)} = \int_T \left[ D \nabla \phi_i \cdot \nabla \phi_j + c \phi_i \phi_j \right] d\mathbf{x}
$$

### 3. 后验误差估计

**Chebyshev 边误差**：对单元 $T$ 的每条边 $e$，构造 Chebyshev 插值 $p_n$：

$$
\eta_T^{\text{Cheb}} = h_T \max_{e \subset \partial T} \|u_h - p_n\|_{\infty, e}
$$

**Zienkiewicz-Zhu 梯度恢复误差**：

$$
\eta_T^{\text{ZZ}} = \|\nabla u_h - \mathbf{G}(u_h)\|_{L^2(T)}
$$

其中恢复梯度 $\mathbf{G}(u_h)$ 通过 Patch 加权平均获得。

综合误差指示子：

$$
\eta_T = \frac{1}{2}\eta_T^{\text{Cheb}} + \frac{1}{2}\eta_T^{\text{ZZ}}
$$

全局误差估计：

$$
\eta_{\text{global}} = \sqrt{\sum_T \eta_T^2}
$$

### 4. Shepard 插值解传递

Prolongation 算子 $I_{2h}^h: V_{2h} \to V_h$：

$$
(I_{2h}^h u_{2h})(\mathbf{x}) = \sum_{i=1}^{N} w_i(\mathbf{x}) u_{2h}(\mathbf{x}_i)
$$

其中逆距离权重：

$$
w_i(\mathbf{x}) = \frac{\|\mathbf{x} - \mathbf{x}_i\|^{-p}}{\sum_{j=1}^N \|\mathbf{x} - \mathbf{x}_j\|^{-p}}
$$

### 5. RCM 重排序

Reverse Cuthill-McKee 算法通过 BFS 层级遍历降低矩阵带宽：

$$
B = \max_{i,j: A_{ij} \neq 0} |i - j| + 1
$$

排序后：$B_{\text{RCM}} \ll B_{\text{orig}}$，显著加速稀疏直接求解器。

### 6. 时间积分稳定性

向后 Euler (BDF1)：

$$
\frac{\mathbf{u}^{n+1} - \mathbf{u}^n}{\Delta t} = \mathbf{F}(\mathbf{u}^{n+1})
$$

放大因子 $G = (1 - z)^{-1}$，$z = \lambda \Delta t$，无条件稳定。

CFL 稳定性条件：
- 对流限制：$\Delta t \leq h / |\mathbf{v}|_{\max}$
- 扩散限制：$\Delta t \leq h^2 / (4D_{\max})$

### 7. HITS 网格重要性排序

对于网格图邻接矩阵 $\mathbf{A}$：

$$
\mathbf{a}^{(k+1)} = \mathbf{A}^T \mathbf{h}^{(k)} / \|\mathbf{A}^T \mathbf{h}^{(k)}\|, \quad
\mathbf{h}^{(k+1)} = \mathbf{A} \mathbf{a}^{(k+1)} / \|\mathbf{A} \mathbf{a}^{(k+1)}\|
$$

Authority 分数高的节点是信息汇聚中心，在 AMR 中优先保留。

---

## 代码结构

```
173_synth_project/
├── main.py                  # 统一入口，零参数运行
├── cvt_mesh.py              # CVT 网格生成 + Voronoi 图
├── triangulation_refine.py  # 三角形网格自适应细化/粗化
├── graph_mesh.py            # 网格图论分析 + RCM + HITS
├── shepard_transfer.py      # Shepard 插值解传递
├── error_indicator.py       # Chebyshev + ZZ 后验误差估计
├── fem_solver.py            # 2D 三角形 FEM 求解器
├── advection_diffusion.py   # 对流-扩散-反应方程组装
├── time_integrator.py       # 隐式时间积分 + 刚度分析
├── hex_boundary.py          # 六边形边界离散化
└── quadrature_rules.py      # 高阶三角形数值积分
```

共 **11 个 .py 文件**，超过要求的 8 个。

---

## 运行方式

```bash
cd 173_synth_project
python main.py
```

无需任何命令行参数。程序将自动：
1. 设置物理问题参数
2. 生成 CVT 优化初始网格
3. 执行 3 层 AMR 循环（求解 → 误差估计 → 细化 → 解传递 → RCM 重排序）
4. 执行最终稳态 FEM 求解
5. 输出刚度分析、收敛历史与数值验证结果

---

## 数值结果示例

典型运行输出：

```
最终网格规模: 123 节点, 209 三角形
AMR 误差历史: [0.796, 1.184, 0.601]
稳态解能量范数: 0.215
系统刚度比: ~1e5
RCM 带宽优化: 187 -> 57
Dirichlet 边界残差: 0.00e+00
```

---

## 关键设计决策

1. **自适应策略**：采用 Dörfler 标记 + 边二分细化，保证嵌套有限元空间，确保解的相容传递。
2. **多重误差估计**：Chebyshev 边误差捕捉局部多项式逼近缺陷，ZZ 恢复梯度误差捕捉导数不连续，二者互补。
3. **稳定性保障**：隐式向后 Euler 时间积分无条件稳定，CFL 条件动态调整时间步长。
4. **稀疏优化**：每次 AMR 层级后执行 RCM 重排序，最小化稀疏矩阵带宽。
5. **边界处理**：六边形边界格点提供高几何保真度的边界近似，辅助边界层网格细化决策。
6. **刚性分析**：通过广义特征值问题分析离散系统刚度比，验证时间积分的稳定性假设。

---

## 边界处理与数值鲁棒性

- **退化三角形检测**：所有涉及三角形几何的操作检查面积 > 1e-14，避免奇异 Jacobian。
- **Dirichlet 边界强制**：直接消去法保证边界条件精确满足（残差 < 1e-14）。
- **矩阵正则化**：当条件数 > 1e14 时自动添加对角正则化，保证线性求解可执行。
- **质量矩阵保护**：Lumped 质量矩阵对角元强制大于 1e-14，避免除零。
- **Chebyshev 插值退化**：当插值点与数据点重合时直接返回精确值，避免除零。
- **自适应步长下限**：时间步长受 dt_min = 1e-8 保护，防止无限缩小。
