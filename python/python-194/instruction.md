# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目概述

本项目是一个基于重叠 Schwarz 域分解的高阶谱元有限元求解器，用于求解二维不可压缩 Stokes 方程的瞬态问题。 它集成了多个科学计算组件：各向异性 CVT 域分解与负载均衡、谱元 Fekete/GLL 高阶求积、制造解验证、重叠 Schwarz 迭代求解器、自适应时间步进、几何界面提取与匹配、性能基准测试等。  

程序入口为 `main.py`，该文件调用其他模块协调完整工作流。 其他 `.py` 文件（共 9 个）提供可复用的底层实现，这些文件将在 benchmark 中被移除，需要由 agent 根据本描述和 `main.py` 中的导入与调用重新实现。

---

# 文件职责与关键内容

## fekete_quadrature.py

提供一维和参考三角形上的高阶 Fekete/GLL 求积规则。

- **核心算法**：  
  - 利用 Legendre 多项式的三项递推计算 \(P_n(x)\) 及其导数 \(P'_n(x)\)。  
  - 计算 Gauss-Lobatto-Legendre (GLL) 节点（包含 ±1 端点，内部点为 \(P'_p\) 的根）及对应权重，权重公式与 \(P_p\) 的值有关。  
  - 通过仿射变换将参考区间 \([-1,1]\) 的节点和权重映射到任意线段 \([a,b]\)。  
  - 对于参考三角形，采用近似 Fekete 节点的构造方法：先用 GLL 节点生成张量积点集，再通过 Duffy 变换映射到标准三角形并给出近似权重，最后对权重归一化以保证积分为三角形面积。

- **主要数据结构**：  
  - 节点向量 `xi`（数组）和权重向量 `w`（数组）。  
  - Vandermonde 矩阵（以 Legendre 基在节点上求值）。

- **导出函数**：  
  - `gll_nodes_weights(p)` – 返回 `[-1,1]` 上阶数为 \(p\) 的 GLL 节点与权重。  
  - `fekete_line_rule(a, b, p)` – 返回 \([a,b]\) 上的 Fekete (GLL) 节点与权重，权重按区间长度缩放。  
  - `fekete_triangle_nodes_weights(p)` – 返回近似三角形 Fekete 节点与权重。  
  - 辅助函数 `legendre_polynomial`, `legendre_derivative`, `affine_map`, `vandermonde_1d` 等。

## geometry_utils.py

提供子域界面提取、距离几何、负载均衡质量评估、点集重建等几何工具。

- **核心算法**：  
  - **超球面距离统计**：在正象限单位超球面上均匀采样，计算点对间 Euclidean 距离的蒙特卡洛均值和方差；也提供理论均值公式。  
  - **负载均衡质量**：根据子域体积（或单元数）计算质量指标 \(Q = 1 - \sigma/\mu\)。  
  - **平面–四面体求交**：基于平面与四面体顶点的有向距离进行边切割，收集交点并按角度排序得到凸多边形，可计算其面积。  
  - **偏消化问题 (Partial Digest Problem)**：利用回溯法从一维点集的成对距离重建点坐标。  
  - **界面节点匹配**：对称地计算两个节点集之间在容许误差内的匹配比例，返回匹配得分。

- **主要数据结构**：  
  - 点、向量数组（通常为 numpy 数组）。  
  - 平面参数（点、法向量）。  
  - 多边形顶点列表。

- **导出函数**：  
  - `hypersphere_positive_distance_stats(m, n_samples)` – 估计正象限超球面上随机距离的均值和方差。  
  - `compute_partition_quality(volumes)` – 计算负载均衡质量。  
  - `plane_tetrahedron_intersect(...)` – 返回平面与四面体的交面多边形顶点。  
  - `polygon_area_3d(polygon)` – 计算空间多边形的面积。  
  - `partial_digest_reconstruct(distances, max_coord, tol)` – 从距离集合重建一维点集。  
  - `interface_matching_score(nodes_i, nodes_j, tol)` – 评估两界面节点集的匹配度。

## mesh_partition.py

实现各向异性 Voronoi 镶嵌 (CVT) 的域分解，用于并行有限元的子区域划分。

- **核心算法**：  
  - 定义了几种度量张量：各向同性 (`metric_identity`)、各向异性 (diag(α,1))、边界层自适应度量。  
  - 使用基于度量张量的 anisotropic 距离函数。  
  - Lloyd 迭代：用大量随机样本计算各生成元的 Voronoi 胞腔质心，更新生成元位置，直到最大位移小于给定容差。  
  - 可以根据生成元栅格化子域掩码，支持通过膨胀创建重叠区域（Schwarz 重叠）。  
  - 从两子域掩码的重叠区提取界面节点坐标。

- **主要数据结构**：  
  - 生成元 `generators` 为 (k,2) 数组，子域中心坐标。  
  - 子域掩码为 boolean 网格。  
  - 距离度量函数对象。

- **导出函数**：  
  - `compute_cvt(n_subdomains, ...)` – 执行各向异性 CVT 划分，返回子域中心。  
  - `compute_subdomain_boundaries(...)` – 返回每个子域的 boolean 掩码。  
  - `subdomain_overlap_masks(masks, overlap_layers)` – 扩展掩码以引入重叠。  
  - `extract_interface_nodes(mask_i, mask_j, domain)` – 提取两子域重叠区域内的节点坐标。  
  - 辅助函数 `metric_anisotropic`, `metric_boundary_layer`, `anisotropic_distance`。

## optimization_utils.py

提供单峰函数优化和线搜索方法，用于非线性迭代和参数寻优。

- **核心算法**：  
  - **黄金分割搜索**：在一维区间 \([a,b]\) 上寻找单峰函数的极小值，每次迭代按黄金比缩小区间。  
  - **回溯线搜索**：对于多维目标函数的下降方向，基于 Armijo 条件后退步长。  
  - 还提供了一组单峰测试函数。  
  - 用于 Schwarz 重叠率优化的包装函数。  
  - **幂迭代**：估计对称矩阵的最大特征值（谱半径），用于选择松弛参数。

- **主要数据结构**：  
  - 目标函数和梯度的回调函数。  
  - 数值向量（numpy 数组）。

- **导出函数**：  
  - `golden_section_search(f, a, b, tol, max_iter)` – 返回最小值点和函数值。  
  - `backtracking_line_search(f, grad_f, x, p, ...)` – 返回满足 Armijo 条件的步长。  
  - `optimize_subdomain_overlap(residual_func, overlap_range, tol)` – 优化 Schwarz 重叠分数。  
  - `power_iteration_estimate(A_matvec, n, ...)` – 估计最大特征值。

## performance_bench.py

提供子域直接求解器的性能基准测试，基于 LINPACK 概念。

- **核心算法**：  
  - 稠密系统求解基准：用随机 SPD 矩阵和标准求解器，计算时间和 MFLOPS，并评估归一化残差。  
  - 带状 SPD 矩阵的 Cholesky 求解基准：多次运行，记录平均时间和估计 MFLOPS。  
  - 对不同子域尺寸进行批量基准测试，并打印报告。  
  - 利用 Amdahl 定律估计并行效率。

- **主要数据结构**：  
  - 稠密和带状矩阵对象。  
  - 字典存储基准结果。

- **导出函数**：  
  - `linpack_benchmark_dense(n)` – 返回时间、MFLOPS、归一化残差。
