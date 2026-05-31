# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：激光-等离子体相互作用多尺度模拟系统

本项目是一个面向惯性约束聚变（ICF）研究的高功率激光-等离子体相互作用数值模拟框架。它将多个独立的科研级算法融合成一个统一的工作流，覆盖从物理常数、空间网格生成、密度剖面构建、射线追踪、能量沉积、色散关系求解、参数空间采样、靶丸几何建模、偏振动力学、泊松方程求解到数据完整性校验的全过程。项目的主入口为 `main.py`，其余所有功能均分散在对应的独立 Python 模块中。后续 benchmark 将保留 `main.py`，删除其他 `.py` 文件，要求你根据本描述重新实现这些缺失模块。

## 文件结构与模块职责

### physics_constants.py
- **职责**：定义全套基础物理常数（电荷、电子质量、真空介电常数、光速、玻尔兹曼常数等），并提供一系列核心物理计算公式。
- **主要函数**：
  - `plasma_frequency(ne)`：计算等离子体频率 ω_p。
  - `critical_density(omega0)`：计算激光临界密度 n_c。
  - `refractive_index(ne, omega0)`：冷等离子体折射率 η。
  - `quiver_velocity(E0, omega0)`：电子抖动速度 v_osc。
  - `ponderomotive_potential(E0, omega0)` 和 `ponderomotive_force_gradient(E0, omega0, grad_E2)`：有质动力相关。
  - `srs_growth_rate(ne, E0, omega0)`：受激拉曼散射线性增长率。
  - `landau_damping_rate(ne, Te, k)`：朗道阻尼率。
  - `coulomb_logarithm(ne, Te, Z)`、`electron_ion_collision_frequency(ne, Te, Z)`：碰撞频率。
  - `inverse_bremsstrahlung_absorption
