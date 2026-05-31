# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：锂电池电化学-热耦合仿真平台

本项目是一个面向锂离子电池的跨尺度仿真框架，涵盖电化学伪二维（DFN）模型、二维热有限元分析、粒子尺寸分布统计、随机锂离子传输蒙特卡洛模拟、充电协议组合优化以及阻抗谱分析等功能。代码入口为 `main.py`，该文件串联了多个独立模块以完成完整的仿真流程。其余 `*.py` 文件为可复现的缺失模块，需要根据本描述重新实现。

## 模块清单与职责

### 1. `banded_linear_algebra.py` — 带状与 Toeplitz 矩阵线性代数

**职责**：为电化学系统中的带状 Jacobian 矩阵和对称 Toeplitz 矩阵提供高效的存储与求解算法。

**核心类/函数**：

- `BandedMatrix`：存储为 LINPACK 风格紧凑格式的带状矩阵（下带宽 `ml`，上带宽 `mu`）。支持：
  - 按带状索引设置/获取元素（`set_entry`、`get_entry`）。
  - 带部分选主元的 PLU 分解（`plu_factor`），返回分解成功标识。
  - 基于分解结果求解线性方程组（`solve`），支持转置选项。
  - 计算行列式（`determinant`）。
- `SymmetricToeplitzSolver`：输入对称 Toeplitz 矩阵的第一行，提供：
  - 利用 Durbin 算法求解 Yule-Walker 方程（`yule_walker`）。
  - 利用 Levinson 扩展求解一般右端项（`solve_general`）。
  - 快速矩阵向量乘法（`matvec`），利用 Toeplitz 结构实现 O(N^2) 运算。
- 辅助函数：`banded_from_dense`（将稠密矩阵转为带状存储）、`build_tridiagonal_banded`（构建常系数三对角带状矩阵）。

**依赖**：numpy。

---

### 2. `electrochemistry.py` — 伪二维 Doyle-Fuller-Newman (DFN) 电化学模型

**职责**：实现锂离子电池宏观一维和微观径向扩散耦合的电化学核心方程，包括 Butler-Volmer 动力学、固相扩散、电解质输运和电荷守恒。

**核心类**：

- `SolidDiffusionSolver`：球形颗粒径向扩散的有限差分求解器。
  - 构造函数接收颗粒半径 `R`、径向离散点数 `n_r` 和固相扩散系数 `D_s`。
  - 内部构建隐式扩散算子（带状矩阵），结合通量边界条件推进浓度场，并返回表面浓度和平均浓度。
- `MacroscopicElectrochemicalSolver`：一维有限体积求解器，覆盖负极、隔膜、正极区域。
  - 初始化时指定各区域长度、离散点数及初始温度；创建每个网格点的 `SolidDiffusionSolver` 实例。
  - 根据温度更新所有与温度相关的输运参数（通过 `TemperatureDependentProperty` 样条）。
  - 求解电解质浓度方程（隐式矩阵，无通量边界条件），更新 `C_e`。
  - 求解电荷守恒，得到固体/电解质电位分布和局部 Butler-Volmer 反应通量，并据此更新表面浓度。
  - 主步进函数 `step(dt, I_app, T_local)` 返回电压、通量、浓度、电位等字典。

**辅助函数**：多种温度相关材料属性（扩散系数、电导率）的默认样条构建函数；开路电位（OCP）及其熵系数；交换电流密度计算；Butler-Volmer 通量及其逆函数（通过 Muller 求根）。

**依赖**：`banded_linear_algebra`、`numerical_toolkit`（`TemperatureDependentProperty`、Muller 求根）、`quadrature_special`（Gauss-Legendre 节点和权重，用于反应速率积分，但本模块中未直接使用）。

---

### 3. `fem_assembler.py` — 有限元组装与后处理

**职责**：为二维热方程在三角形网格上组装全局刚度矩阵和质量矩阵，支持各向同性热导率。

**核心函数**：

- `assemble_thermal_matrices`：输入节点坐标、单元连接关系、材料热导率字典、区域标签和体积热容，返回全局刚度矩阵 `K` 和质量矩阵 `M`（均为稠密矩阵）。使用参考三角形上的线性形状函数及其导数，通过雅可比映射计算物理梯度，并用三角形单元积分规则累加单元贡献。
- `apply_dirichlet_bc`：通过行替换施加 Dirichlet 边界条件。
- `compute_l2_error`：计算数值解与解析解之间的 L2 误差，使用高阶三角形积分规则。

**内部辅助**：`basis_t3`（参考三角形线性形状函数及导数）、`jacobian_t3`（计算雅可比矩阵和行列式）。

**依赖**：`quadrature_special`（`triangle_unit_rule`）。

---

### 4. `geometry_engine.py` — 几何与域分类

**职责**：定义电池二维截面（多层矩形域）并提供点包含判断。

**核心类**：

- `Polygon2D`：使用复数坐标表示多边形，支持：
  - 基于交叉数算法（crossing-number test）结合包围盒预筛选的 `contains` 方法。
  - 面积和质心计算。
- `BatteryCellGeometry`：由多层矩形区域（负极集流体、负极电极、隔膜、正极电极、正极集流体）以及极耳凸起组成。
  - 构造函数接受各层宽度和总高度，构建各区域的 `Polygon2D` 对象。
  - `classify_point(x, y)` 返回该点所属区域字符串（如 `"neg_elec"`、`"separator"` 等）。
  - `get_all_regions` 返回所有区域标签和对应多边形。

**辅助函数**：`rotate_complex`、`translate_complex`、`reflect
