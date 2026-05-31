# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：光热耦合微腔传感器仿真平台

本项目是一个 Python 仿真平台，用于模拟微环谐振腔中光场与热场的耦合行为，并分析传感器性能。程序入口为 `main.py`，它通过调用多个模块完成从几何建模、网格生成、求解器构造、耦合迭代、本征值分析、代理模型训练到不确定性量化的一整套流程。以下描述各模块 (`*.py`) 的职责与核心接口，agent 需要根据这些说明重新实现缺失的模块，使得 `main.py` 能够正常运行。

---

## 模块清单

### 1. `geometry_mesh.py` —— 微腔几何建模与网格生成

- **`CVV` 类**：实现 Cell‑based Vector of Vectors 数据结构，将变长二维数组压缩为一维存储，通过偏移表实现 O(1) 索引。支持 `iget`/`iset`、`get_row`/`set_row` 等操作。
- **`MicrocavityGeometry` 类**：微环谐振腔的几何模型。  
  - 构造函数接收主半径、截面半径、波导间隙、波导宽度以及各区域折射率。  
  - 提供 `ring_boundary_parametric` 方法，返回截面外边界参数方程 `(x, y)`。  
  - `arc_length_trapezoidal` 用复合梯形法则计算参数曲线弧长。  
  - `generate_cross_section_mesh` 在极坐标下生成结构化三角网格，返回节点坐标、单元连通性和边界标记。  
  - `generate_waveguide_nodes` 生成直波导节点坐标。  
  - `compute_mesh_quality` 计算网格的最小/最大角、面积等质量指标。  
  - 包含 `fem_write_nodes`、`fem_write_elements`、`fem_read_nodes`、`fem_read_elements` 四个方法，实现节点与单元数据的文本读写。

### 2. `quadrature_engine.py` —— 数值积分引擎

- **`SphereQuadrature` 类**：单位球面 S² 上的积分与采样。  
  - `sphere01_area` 返回球面面积。  
  - `monomial_integral_exact` 精确计算单项式球面积分，利用 Gamma
