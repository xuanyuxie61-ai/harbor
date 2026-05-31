# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：拓扑半金属 Weyl 节点数值研究系统

本项目是一个凝聚态物理计算框架，围绕三维 Weyl 半金属的拓扑性质进行建模、采样、积分、统计与优化。其核心是对布里渊区内的哈密顿量、Berry 曲率、Fermi 面等进行多尺度处理，并通过一系列数值模块完成拓扑不变量计算、态密度估计、输运优化等任务。

程序入口为 `main.py`，它组合所有模块、零参数运行一系列预定义的科学计算流程。其余模块文件分别负责特定子领域，这些模块的名称、职责和边界如下。  
你需要基于本描述复现这些模块，使 `main.py` 能够正常执行。

## 模块总览与边界

### 1. `weyl_hamiltonian.py` – Weyl 哈密顿量建模
**职责**：定义 Weyl 半金属的低能有效哈密顿量类，支持线性模型与紧束缚双节点模型，并提供本征求解、d‑矢量提取、节点定位等基础设施。  
**核心类与函数**：
- `WeylHamiltonian` 类：构造函数接收 `model_type`（`"linear"` 或 `"tight_binding"`）及基本常数 `hbar`、`v_f`。内部存储 Pauli 矩阵和紧束缚参数。  
  - `build_hamiltonian(k)`：根据动量 `k` 返回 2×2 复矩阵；支持批量输入。  
  - `eigenproblem(k)`：返回本征值（升序）和本征矢。  
  - `d_vectors(k)`：将哈密顿量分解为 `d₀·I + d·σ`，返回标量 `d₀` 和矢量 `d_vec`。  
  - `monomial_expansion_value`：用于多项式修正的能量展开（多指标指数与系数）。  
  - `weyl_node_position_linear()` / `find_weyl_nodes_tight_binding()`：返回（多个）Weyl 节点的 k 空间坐标。后者使用网格搜索与聚类。  
- 模块级函数 `band_gap(energies)`：计算两带能隙。  
- 模块级函数 `velocity_operator(ham, k)`：通过中心差分数值计算速度算符矩阵 ∂H/∂k。  

**外部接口**：其他模块通过 `WeylHamiltonian` 对象获取能带信息，通过 `velocity_operator` 构造 Berry 曲率。

### 2. `berry_curvature.py` – Berry 联络、曲率与拓扑相位
**职责**：基于 `WeylHamiltonian` 及其本征态，数值计算 Berry 联络、Berry 曲率、Berry 相位，以及与之关联的拓扑不变量（Chern 数、Weyl 荷）。  
**核心函数**：
- `berry_connection_numeric(ham, k, band_index)`：使用中心差分和本征矢梯度近似 Berry 联络 A(k)。包含规范固定处理。
- `berry_curvature_numeric(ham, k, band_index)`：利用速度算符方法（避免直接求导本征矢）计算 3×3 反对称 Berry 曲率张量。
- `berry_curvature_analytic_linear(k, chirality)`：线性模型的解析 Berry 曲率。
- `berry_phase_1d(ham, path, band_index)`：沿 k 空间路径的离散 Wilson loop 相位（包含首尾闭合处理）。
- `chern_number_2d_slice(ham, kx_range, ky_range, kz, …)`：在固定 kz 的二维截面上数值积分 Ω_xy 得到 Chern 数。
- `weyl_charge_surface_integral(ham, center, radius, …)`：在包围节点的球面上积分 Berry 曲率，得到 Weyl 荷。

**算法特征**：数值微分、规范固定使第一个分量为实数、Wilson loop 乘积的复角、反对称张量的旋度对应。  
**依赖**：需要 `weyl_hamiltonian` 中的 `WeylHamiltonian` 和 `velocity_operator`。

### 3. `bzone_sampler.py` – 布里渊区多方法采样
**职责**：提供在布里渊区（或任意区域）生成 k 点集合的多种算法，包括椭球内均匀采样、密度加权 CVT 迭代、均匀网格以及自适应 Weyl 节点附近采样。  
**核心函数**：
- `cholesky_factor(a)`：返回对称正定矩阵的 Cholesky 分解（上三角形式），供椭球采样使用。
- `uniform_in_sphere01_map(m, n)`：生成 m 维单位球内均匀分布点。
- `ellipse_sample(n, a, r)`：在椭球 XᵀAX ≤ R² 内生成 n 个均匀点，依赖 Cholesky 求解。
- `ellipse_area(a, r)`：计算椭球体积。
- `cvt_sampler_nonuniform(n_generators, …)`：基于密度函数进行重心 Voronoi 镶嵌迭代，输出非均匀生成器位置。
- `uniform_kpoint_grid(bounds, grid_size)`：返回规则网格点。
- `adaptive_weyl_node_sampler(n_points, weyl_nodes, …)`：混合椭球采样与均匀采样，在已知 Weyl 节点附近加密。

**外部接口**：返回的 k 点数组可供能带计算、态密度等模块使用。

### 4. `density_of_states.py` – 态密度与广义 Hermite 求积
**职责**：计算态密度（DOS），包含直方图、高斯展宽、Weyl 半金属解析形式，以及在积分中用到 Gauss-Hermite 求积规则。  
**核心函数**：
- `gauss_hermite_nodes_weights(n)`：通过三对角 Jacobi 矩阵求解标准 Gauss-Hermite 节点和权重。
- `generalized_hermite_integral(expon, alpha)`：计算广义 Hermite 积分 ∫ xⁿ|x|^α e^{-x²} dx 的解析值（利用 Gamma 函数）。
- `dos_histogram(energies, e_min, e_max, n_bins)`：直方图估计 DOS。
- `dos_gaussian_broadening(energies, e_grid, sigma)`：高斯展宽法。
- `dos_weyl_semimetal_analytic(e, v_f, hbar)`：线性 Weyl 半金属的解析 DOS。
- `integrate_dos_with_hermite(…)`：将高斯展宽与 Hermite 求积结合来估计 DOS 贡献。
- `test_hermite_exactness(alpha, …)`：检验 Hermite 规则的代数精度。

**边界**：`integrate_dos_with_hermite` 接受能带函数和 k 空间区域，间接使用采样。

### 5. `mesh_extractor.py` – 网格提取与格式转换
**职责**：处理 2D/3D 网格数据结构，将节点坐标和元素连接封装为字典，实现节点值到元素平均，生成 DOLFIN XML 字符串，以及从能带数据构建 Fermi 面网格。  
**核心函数**：
- `mesh2d_extract(nodeco, elnode, bdynde)`：返回包含网格元数据的字典。
- `mesh3d_extract(nodeco, elnode, bdynde)`：类似 2D，但元素可为四面体等。
- `node_values_to_elements(node_values, elnode, …)`：将节点标量或向量平均到各个三角形/四面体元素。
- `mesh_to_xml_string(mesh)`：将网格转换为符合 FEniCS 格式的 XML 字符串。
- `create_fermi_surface_mesh(k_points, energies, e_fermi, …)`：筛选能量近似的点，调用 `triangulation_mesh` 中的三角剖分生成 Fermi 面网格。

**依赖**：`create_fermi_surface_mesh` 会用到 `triangulation_mesh` 的 Delaunay 三角化。

### 6. `
