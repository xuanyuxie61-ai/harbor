# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：肿瘤生长微环境多尺度计算建模系统

本项目是一个面向肿瘤生物学的多尺度计算框架，融合了15个种子项目的核心算法，覆盖几何建模、营养传输、细胞动力学、空间竞争、力学分析、谱/有限元方法及数值积分等模块。整个系统由主入口 `main.py` 协调运行，其余 Python 文件各自封装独立物理或数值模块。在后续 Benchmark 中，目标是根据本描述复现除 `main.py` 外的所有模块文件。

## 文件清单与职责

| 文件名 | 职责 |
|--------|------|
| `main.py` | 主入口，依次调用各子模块并打印综合报告。保留此文件不删除。 |
| `tumor_geometry.py` | 肿瘤边界 Bernstein 参数化、Delaunay 三角剖分、边界节点识别、几何度量计算。 |
| `nutrient_diffusion.py` | 径向扩散方程精确/近似解、Clenshaw‑Curtis 与稀疏网格高阶积分、Michaelis‑Menten 消耗及缺氧评估。 |
| `cellular_dynamics.py` | 细胞四态（增殖/静息/凋亡/坏死）的定义、连续时间马尔可夫链群体演化、细胞自动机邻域规则。 |
| `spatial_voronoi.py` | 圆盘内 CVT（Centroidal Voronoi Tessellation）生成子优化、径向生长、贪婪代谢异质性分区。 |
| `mechanical_stress.py` | 双调和算子（薄板弯曲）特征模态求解、冯·米塞斯应力计算、应力诱导凋亡概率与统计指标。 |
| `spectral_solver.py` | Jacobi 正交多项式基、Gauss‑Legendre 求积、谱‑Galerkin 方法求解一维扩散方程。 |
| `nonlinear_coupling.py` | Newton 迭代法（标量及多维）、肿瘤‑营养耦合非线性方程组稳态求解、线搜索与正则化策略。 |
| `sparse_fem.py` | 稀疏三元组→稠密矩阵转换、二维线性元刚度矩阵装配、Dirichlet 边界条件施加、稀疏矩阵‑向量乘及 L2 误差评估。 |
| `quadrature_rules.py` | Gauss‑Legendre 与 Clenshaw‑Curtis 求积节点/权重计算、一维积分封装、治疗响应指数（TRI）与累积氧消耗计算、积分误差估计。 |
| `utils.py` | 外部可控排序、安全除法、sigmoid 函数、参数边界校验、Gini 系数、Morse 势函数。 |

各模块间通过明确的函数接口耦合。例如 `cellular_dynamics.py` 导出的状态常量被主程序及其他模块使用；`main.py` 会利用 `tumor_geometry.py` 生成的网格传递给 `sparse_fem.py` 和 `quadrature_rules.py` 以计算应力场和治疗指数。以下分模块详述其科学背景与核心功能。

---

## 1. `tumor_geometry.py` – 肿瘤几何建模与三角剖分

**背景**：肿瘤边界采用 Bernstein 多项式参数化，能够柔性地描述不规则形状。内部区域需要 Delaunay 三角剖分以便有限元或几何计算。  

**主要函数**：  
- `bernstein_poly_01(n, x)`：计算 n 次 Bernstein 基函数在点 x 处的值（递推实现）。  
- `bernstein_tumor_boundary(control_points, num_samples)`：用控制点生成闭合肿瘤边界点序列。  
- `tumor_surface_triangulation(boundary_points, interior_density)`：在边界内生成均匀内部点，执行简化 Delaunay 剖分，返回节点、三角形和边界节点布尔标记。  
- `detect_boundary_nodes(node_num, triangle_num, triangle_node)`：基于边出现次数判别边界节点（1‑based/0‑based 自适应）。  
- `compute_tumor_area(nodes, triangles)`：网格面积累加。  
- `compute_tumor_perimeter(boundary_points)`：边界点间欧氏距离累加。  

**依赖**：仅使用 `numpy`。

---

## 2. `nutrient_diffusion.py` – 营养扩散与数值积分

**背景**：描述氧气/营养在肿瘤内的扩散‑反应过程，径向假设下使用 Laplace 方程精确解和修正 Bessel 函数解。高阶积分（Clenshaw‑Curtis、稀疏网格）用于计算营养总量与消耗。

**主要函数**：  
- `laplace_radial_2d_exact(x, y, a, b)`：返回二维径向 Laplace 方程精确解及其一、二阶偏导数。  
- `oxygen_diffusion_steady_state_radial(r, R_tumor, C_boundary, D, consumption_rate)`：基于零阶修正 Bessel 函数 I₀ 的稳态氧浓度分布。  
- `clenshaw_curtis_integrate(f, a, b, n)`：n 点 Clenshaw‑Curtis 数值积分。  
- `cc_abscissa(order, i)` 与 `cc_weights(n)`：Clenshaw‑Curtis 节点与权重（内部使用 DCT 关系）。  
- `sparse_grid_monomial_integral(dim, level, exponents)`：二维 Smolyak 稀疏网格单项式积分（一维/二维实现）。  
- `michaelis_menten_consumption(C, rho, Vmax, Km)`：计算 Michaelis‑Menten 消耗速率。  
- `hypoxia_region_fraction(C, threshold)`：低于阈值的区域比例。  
- `integrate_radial_profile(r_vals, f_vals, dim)`：梯形法则径向积分（2D/3D）。

**依赖**：`numpy`，部分函数内部使用 `scipy.special` 中的修正 Bessel 函数（`i0`, `k0`, `i1`）。

---

## 3. `cellular_dynamics.py` – 细胞动力学与细胞自动机

**背景**：细胞状态包含增殖(P)、静息(Q)、凋亡(A)、坏死(N)四种。用连续时间马尔可夫链建模状态转移，并用“Lights Out”风格的细胞自动机模拟空间邻域效应。

**主要函数与常量**：  
- 常量 `STATE_PROLIFERATION`, `STATE_QUIESCENCE`, `STATE_APOPTOSIS`, `STATE_NECROSIS`。  
- `cell_transition_matrix(...)`：根据各通道转移概率构造 4×4 行归一化转移矩阵。  
- `evolve_cell_population_markov(initial_counts, trans_matrix, steps)`：递推计算群体演化历史，使用矩阵转置乘向量。  
- `ca_contact_inhibition_update(cell_grid, nutrient_grid, thresholds)`：8‑邻域统计活细胞数，按接触抑制阈值和营养阈值更新网格细胞状态（N 态不可逆，缺氧触发凋亡/坏死，高密度触发静息，营养充足趋向增殖）。  
- `ca_proliferation_step(cell_grid, empty_probability)`：P 细胞以一定概率占据相邻 A/N 位点。  
- `compute_tumor_cellularity(cell_grid)`：返回四种状态的比例。  
- `compute_doubling_time(population_history)`：对总细胞数进行指数拟合估计倍增时间。

**依赖**：`numpy`。

---

## 4. `spatial_voronoi.py` – Voronoi 镶嵌与空间竞争

**背景**：用 Voronoi 图模拟肿瘤内部细胞空间竞争和生长。CVT 通过 Lloyd 迭代最小化能量泛函，径向生长扩展模拟肿瘤扩张，贪婪算法实现代谢异质性分区。

**主要函数**：  
- `disk_sample_uniform(num_samples, radius, seed)`：圆盘内均匀随机采样（面积加权）。  
- `find_closest(sample_points, generators)`：将采样点分配到最近生成子。  
- `cvt_disk_iterate(radius, ...)`：圆盘约束下 Lloyd 迭代，更新生成子并投影边界点。  
- `initialize_tumor_generators(n_boundary, n_inter
