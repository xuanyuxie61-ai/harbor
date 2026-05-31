# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：多铁性材料磁电耦合畴结构演化模拟系统（python‑017）

本项目是一个面向凝聚态物理的多铁性材料模拟计算框架，整合了 Landau‑Ginzburg‑Devonshire 自由能模型、有限元离散、反应‑扩散动力学、自适应隐式 ODE 积分、蒙特卡洛热采样以及数值优化等多个科学计算模块。项目包含一个主入口文件 `main.py` 和若干支撑模块。在后续的 benchmark 中，你将保留 `main.py`，删除其余 `.py` 文件，需要根据本描述重新实现这些缺失的模块，使 `main.py` 能够正常运行并产生正确的物理模拟结果。

---

## 文件结构与职责概览

| 文件名 | 职责 |
|--------|------|
| `landau_free_energy.py` | 定义多铁性材料参数容器和 Landau 自由能密度、变分导数、热涨落修正等核心物理量 |
| `multiferroic_mesh.py` | 生成二维矩形区域上的 T6（二次六节点）三角形网格，提供基函数计算和高斯积分点生成 |
| `sparse_matrix_utils.py` | 实现 COO 和 CSR 两种稀疏矩阵存储格式，支持基本运算、文件 I/O 和转换 |
| `fem_assembler.py` | 利用网格和稀疏矩阵工具组装扩散刚度矩阵、质量矩阵、反应项，并施加 Dirichlet 边界条件 |
| `pyramid_quadrature.py` | 提供单位金字塔区域的高精度数值积分规则，用于三维梯度项近似 |
| `reaction_diffusion_solver.py` | 基于 FTCS 格式的二维反应‑扩散方程求解器，包含 Fisher‑KPP 和 Allen‑Cahn 等反应项 |
| `adaptive_ode_integrator.py` | 自适应隐式中点法 ODE 积分器，用于刚性问题，含 Newton‑Raphson 迭代和局部截断误差步长控制 |
| `monte_carlo_sampler.py` | Metropolis‑Hastings 蒙特卡洛采样器，用于序参量场热涨落采样及关联函数计算 |
| `domain_optimizer.py` | 基于 Hooke‑Jeeves 直接搜索和 TSP‑descent 邻域搜索的畴结构能量最小化 |
| `hilbert_space_filling.py` | 实现 Hilbert 空间填充曲线，提供坐标转换和网格节点重排序以改善内存局部性 |
| `coupling_dynamics.py` | 顶层耦合模拟器，整合以上模块完成多铁性材料极化‑磁化联合时空演化模拟 |

---

## 模块详细描述

### `landau_free_energy.py`

该模块提供描述多铁性材料（以 BiFeO₃ 为原型）自由能所需的物理模型和数学工具。

**核心类与函数：**

- **`MultiferroicMaterialParams`**：存储温度、居里温度、奈尔温度、各阶 Landau 系数、梯度能系数、磁电耦合系数等材料参数。构造函数接受温度，并根据温度计算与温度相关的系数（如 `alpha1 = alpha0*(T-Tc)`）。包含 `validate` 方法检查参数合理性。
- **`hermite_probabilist(n, x)`**：使用递推公式计算概率学家 Hermite 多项式 `He_n(x)`。返回与输入 `x` 同形状的数组。
- **`landau_free_energy_density(P, M, dPdx, dPdy, dMdx, dMdy, params)`**：计算单点自由能密度，由铁电部分、磁性部分、磁电耦合项、梯度项和高阶交叉项组成。所有参数通过 `params` 对象传入。函数需返回一个标量，若出现非有限值则返回大正惩罚值。
- **`variational_derivative_P(P, M, lapP, params)`** 和 **`variational_derivative_M(P, M, lapM, params)`**：分别计算自由能关于极化矢量 P 和磁化矢量 M 的变分导数（即 TDGL 方程的驱动项）。输入包含局部 P、M 以及对应的 Laplacian 张量（由网格计算提供）。返回 2 元素的一维数组。
- **`thermal_fluctuation_correction(P, M, params, max_hermite_order)`**：利用 Hessian 矩阵的特征值估计热涨落对自由能的谐波修正。内部需要构建 4×4 Hessian 矩阵并对称对角化，基于玻尔兹曼常数和温度计算修正量。

**实现要点：** 需理解 Landau 类型的多项式自由能形式以及梯度能量对序参量空间导数的依赖。Hermite 多项式只用到递推关系，无需考虑正交归一化以外的内容。变分导数公式已在文档中给出数学形式，但精确实现需自行根据物理原理推导。

---

### `multiferroic_mesh.py`

该模块负责生成二维矩形域上的 T6 三角形网格，并为基础有限元操作提供几何支持。

**核心类与函数：**

- **`MultiferroicMesh`**：构造函数接受 `nx, ny` 和四个边界坐标 `xl, xr, yb, yt`。内部自动生成节点坐标（`node_xy`）、6 节点三角形元素连接表（`element_node`）、边界节点标志（`boundary_flags`）以及半带宽（`half_bandwidth`）。节点排列为规则结构化网格，每个矩形单元被划分为两个三角形。提供 `element_area(element)`、`get_element_centroid` 和 `apply_reordering(order)` 方法。
- **`qbf_t6(x, y, element, inode, mesh)`**：评估指定元素内某局部节点所对应 T6 二次基函数的值及其对 x、y 的偏导数。需要根据三个角点建立参考坐标 `(r,s)`，然后依据标准二次基函数公式计算并利用雅可比转换到物理导数。
- **`generate_quadrature_points(mesh, nq)`**：为每个元素生成三角形高斯求积点物理坐标和权重。nq 支持 3 或 7，分别对应不同精度的 Gauss 规则。返回权重数组、每个求积点的 x 和 y 坐标（形状 `(nq, element_num)`）。

**实现要点：** 网格生成必须保证元素编号、节点编号的一致性，以便后续有限元组装能正确索引。基函数计算需正确处理参考坐标的反算和退化情况。

---

### `sparse_matrix_utils.py`

提供稀疏矩阵的基础数据结构，支持 COO/Triad 和 CSR 格式，用于有限元组装和求解。

**核心类：**

- **`SparseMatrixCOO`**：存储 `nrow, ncol, row, col, data`。支持 `add_entry(i, j, v)` 增量添加非零元（允许重复），`nnz()` 返回非零元个数，`rebase` 实现索引重基，`to_dense` 转为稠密矩阵，`to_csr` 转为 CSR 格式，`symmetric_expand` 从下三角恢复全矩阵。还提供类方法 `read_from_triad_file` 和实例方法 `write_to_triad_file` 进行文件读写（Triad 格式：首行可包含维度信息，后续每行 `i j value`）。
- **`SparseMatrixCSR`**：存储 `indptr, indices, data`。提供 `dot(x)` 进行稀疏矩阵‑向量乘法以及 `to_dense`。
- **`coo_to_dense_solve(coo, b)`**：将 COO 矩阵转为稠密后用 NumPy 线性求解器解方程 `Ax=b`，适用于小规模验证。

**实现要点：** 确保索引边界检查、内存分配和格式转换的正确性。Triad 文件读写需处理注释行和可能的格式变化，但主要解析规则为每行三个数字。

---

### `fem_assembler.py`

基于网格和稀疏矩阵，组装有限元离散所需的刚度矩阵和质量矩阵。

**核心类：**

- **`FEMAssembler`**：构造函数接受 `MultiferroicMesh` 对象和求积阶数 `n
