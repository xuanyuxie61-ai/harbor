# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目 python-036：中微子振荡与质量 Hierarchy 综合分析平台

该项目是一个中微子物理数值模拟框架，包含粒子物理常数、PMNS 矩阵构造、真空与物质中微子哈密顿量、有限元地球密度剖面、蒙特卡洛参数不确定性分析、数值积分、ODE 演化求解、网格质量评估、稀疏迭代本征态分析以及数据验证与 I/O 等模块。主入口为 `main.py`，它依赖所有其他 Python 文件。后续将删除除 `main.py` 外的所有 `.py` 文件，你需要根据本描述重新实现这些缺失模块。

## 文件说明

### 1. constants.py
**职责**：物理常数与中微子振荡标准参数库。

提供中微子物理所需的基本常数（费米耦合常数、单位转换因子、地球物质密度相关常数）、PMNS 混合角（θ₁₂, θ₂₃, θ₁₃）和 CP 破坏相位 δ_CP 的弧度值、质量平方差（Δm²₂₁, Δm²₃₁ 及其反层级版本）、标准模型带电轻子质量、数值积分默认参数以及地球分层密度模型参数。  
提供的函数包括：
- `get_prem_density(radius_ratio)`：根据简化 PREM 模型返回地球径向密度剖面。
- `electron_fraction(radius_ratio)`：返回电子丰度 Y_e。
- `matter_potential_eV(radius_ratio, energy_gev)`：计算中微子在地球物质中的有效物质势（eV）。
- `get_mass_squared_differences(hierarchy)`：返回质量平方差向量（支持 normal/inverted）。
- `get_pmns_angles()` / `get_cp_phase()`：返回混合角和 CP 相位。

### 2. pmns_matrix.py
**职责**：PMNS 矩阵构造与味‑质量本征态转换。

提供标准参数化下 3×3 PMNS 幺正矩阵的构造，以及味基与质量基之间的转换。  
主要函数：
- `rotation_12`、`rotation_13`、`rotation_23`：构造三个旋转矩阵。
- `build_pmns_matrix(theta12, theta23, theta13, delta_cp)`：组合旋转矩阵生成完整 PMNS 矩阵。
- `build_mass_matrix(delta
