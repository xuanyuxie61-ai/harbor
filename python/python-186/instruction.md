# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 社交网络传播动力学: 多模态耦合模型项目描述

本项目是一个博士级合成示例，模拟社交网络中信息与传染病传播的耦合过程，融合了图论、常/延迟微分方程、有限元方法、间断Galerkin方法、统计推断和几何概率等多个技术模块。

## 保留文件
- **main.py** — 主入口，按阶段调用各模块，展示完整工作流。评估时将保留此文件，其余所有 `.py` 文件将被删除，由其他 agent 根据本描述重新实现。

## 需要实现的其他模块文件

### 1. data_io.py — 多维数据 I/O 与预处理
提供基本的 3D 点云生成、文件读写、特征归一化和 PCA 降维。

**核心函数 / 类:**
- `generate_helical_point_cloud(n_points, radius, pitch)`  
  生成一个沿 z 轴螺旋上升的 3D 点云，返回形状为 `(n_points, 3)` 的 NumPy 数组。
- `normalize_features(data, method='minmax', axis=0)`  
  对输入数据按指定轴向进行归一化或标准化，支持 `minmax`、`zscore`、`robust` 三种方法。返回归一化后的数据和一个包含所用参数（如最小/最大值、均值/标准差等）的字典。
- `write_xyz_data(filename, points)`  
  将 `(N,3)` 点云写入文本文件，第一行为注释行，随后每行三个浮点数。
- `read_xyz_data(filename)`  
  从上述格式的文件读取点云，返回 `(N,3)` 数组。
- `compute_pca_features(data, n_components=3)`  
  执行主成分分析：中心化数据、计算协方差矩阵、特征分解后取前 `n_components` 个主成分。返回投影后的数据 `(N, n_components)` 和对应主成分的方差贡献率数组。

> 模块边界：仅依赖 NumPy，不涉及项目内其他模块。

---

### 2. epidemic_dynamics.py — 流行病-信息耦合动力学
定义 SEAIHR 仓室模型的右端项、耦合信息反馈的延迟动力学以及积分器。

**核心函数 / 类:**
- `seaihr_ode_rhs(t, y, params)`  
  计算 SEAIHR 模型（S‑E‑A‑I‑H‑R‑D）的导数向量。状态 `y` 顺序为 [S, E, A, I, H, R, D]。参数包含总人口 N、接触率 beta、无症状相对传染性 eta_A、潜伏期逆 sigma、出现症状概率 p_sym、各恢复率/住院率、免疫丧失率 omega 等。右端项描述各仓室间的迁移（例如 S→E 的感染力项、E→A/I 的分流、R→S 的免疫丧失等），并维护总体守恒。
- `coupled_info_epidemic_rhs(t, y, history_func, params)`  
  在 SEAIHR 基础上增加信息变量 `I_info`。利用历史函数获取延迟时刻的感染人数，依据该延迟值和当前 I_info 动态调整有效接触率 beta（负反馈），同时按 Mackey‑Glass 形式演化 I_info。
- `rk4_integrate(rhs_func, y0, t_span, dt, params, history_func=None)`  
  经典四阶 Runge‑Kutta 积分器，适用于无延迟系统。返回时间网格和状态历史矩阵。
- `dde_rk4_integrate(rhs_func, y0, t_span, dt, params, history_func)`  
  延迟微分方程的 RK4 积分器。内部使用已完成的历史数据进行线性插值来近似延迟项，每一子步都构造对应的当前历史函数。
- `compute_reproduction_number(params)`  
  根据 SEAIHR 模型的参数计算基本再生数 R₀ = R₀_sym + R₀_asym，分别对应有症状和无症状传播支（利用概率 p_sym、恢复率等）。

> 模块边界：依赖 NumPy，`dde_rk4_integrate` 内部调用 `seaihr_ode_rhs` 或 `coupled_info_epidemic_rhs`。

---

### 3. network_topology.py — 社交网络拓扑分析
构建带社区结构的加权无向图，并计算多种拓扑指标。

**核心函数 / 类:**
- `construct_social_network(n_nodes, community_structure=True, seed=42)`  
  基于随机块模型（SBM）生成加权邻接矩阵：节点被分配到若干社区，社区内连接概率高，社区间连接概率低。确保生成的图是连通的，若否，则添加最小生成树式的边。
- `floyd_warshall(adj)`  
  动态规划计算全源最短距离矩阵（O(n³)）。
- `network_efficiency(dist)`  
  由距离矩阵计算全局效率（Latora‑Marchiori 度量）。
- `betweenness_centrality(adj)`  
  利用 Brandes 算法计算节点介数中心性。
- `power_method_eigenvector(adj)`  
  用幂法（结合 PageRank 式随机游走加 teleportation）计算主特征值和特征向量，用于中心性分析。
- `clustering_coefficient(adj)`  
  计算每个节点的 Watts‑Strogatz 局部聚类系数。
- `degree_distribution(adj)`  
  返回各节点的度以及归一化度分布概率质量函数。

> 模块边界：仅依赖 NumPy，内部使用辅助函数 `is_connected` 和 `ensure_connected`。

---

### 4. sparse_algebra.py — 稀疏矩阵代数（CCS 格式）
实现压缩列存储（CCS）稀疏矩阵及其基本运算。

**核心类 / 函数:**
- `class SparseCCS`  
  - 属性：`n_rows`, `n_cols`, `colptr`, `rowind`, `values`, `nnz`。  
  - 构造需要提供列指针、行索引和非零值三个数组，内部保证每列行索引升序。  
  - 类方法 `from_dense(dense)`：从稠密矩阵构造 CCS 实例。  
  - 方法 `to_dense()`：还原为稠密矩阵。  
  - 方法 `mv(x)`：稀疏矩阵‑向量乘法 `y = A * x`。  
  - 方法 `mtv(x)`：转置矩阵‑向量乘法 `y = A^T * x`。  
  - 类方法 `network_laplacian(adj)`：从邻接矩阵构造图拉普拉斯 L = D − A。  
  - 方法 `power_iteration_sparse()`：对稀疏矩阵执行幂迭代，返回主特征值和特征向量。

> 模块边界：仅依赖 NumPy。

---

### 5. spatial_mesh.py — 空间异质性建模与有限元采样
生成二维三角网格、网格细化以及基于有限元的函数采样和扩散算子。

**核心函数:**
- `generate_2d_triangular_mesh(nx, ny, x_min, x_max, y_min, y_max)`  
  生成由 `nx×ny` 个矩形各划分为两个三角形形成的结构化网格，返回节点坐标 `(N_nodes, 2)` 和单元连通表 `(N_elem, 3)`。
- `refine_mesh_midpoint(nodes, elements)`  
  中点细分：将每个三角形细分为 4 个子三角形，返回新节点和新单元数组。
- `locate_point_in_mesh(nodes, elements, point)`  
  用重心坐标法在网格中定位包含给定点的单元索引。
- `fem_sample_on_mesh(nodes, elements, node_values, sample_points)`  
  在网格上定义的节点标量场进行线性插值，返回样本点上的函数值。
- `spatial_diffusion_operator(nodes, elements)`  
  组装与线性 T3 单元对应的泊松方程刚度矩阵（梯度内积的离散形式），返回方阵。

> 模块边界：仅依赖 NumPy，内部使用 `triangle_area`、`barycentric_coordinates`、`fem_basis_t3` 等辅助函数
