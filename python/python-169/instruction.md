# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：7-DOF 冗余机械臂实时轨迹规划与动态避障系统

## 项目目标
本项目实现一个面向 7 自由度冗余机械臂在杂乱 3D 环境中的实时轨迹规划与动态避障系统。系统整合了运动学、动力学、轨迹生成、障碍物建模、构型空间采样、稀疏线性求解、伪谱优化、图搜索、无导数优化、整数规划等多个子模块，并通过一个统一的核心规划器进行编排。最终由 `main.py` 作为入口执行完整的规划管线，并输出多维度的评估指标。

## 文件结构及职责
项目除 `main.py` 外包含以下 Python 源文件，每个文件负责一个独立的算法域：

- `kinematics_dynamics.py` – 机械臂运动学与动力学
- `bernstein_path.py` – Bernstein 多项式与 Bézier 轨迹
- `obstacle_geometry.py` – 障碍物几何与有向距离场
- `configuration_space.py` – 构型空间采样与概率分布
- `sparse_linear_algebra.py` – 稀疏线性代数求解器
- `pseudospectral_control.py` – Gauss-Legendre 伪谱法
- `roadmap_graph.py` – PRM 图与 HITS 重要性排序
- `derivative_free_opt.py` – PRAXIS 无导数优化
- `discrete_planning.py` – 离散规划与整数分配
- `milp_parser.py` – CPLEX MILP 解解析
- `core_planner.py` – 核心规划器，整合上述所有模块

`main.py` 负责实例化 `ManipulatorMotionPlanner` 并调用 `run_full_pipeline()`，输出执行摘要。**后续将删除除 `main.py` 外的所有 `.py` 文件，需要根据本描述重新实现这些文件。**

## 模块详细说明

### 1. kinematics_dynamics.py
**核心功能**：7 自由度机械臂的运动学、动力学、刚性 ODE 积分以及微分逆运动学求解。

**主要类与函数**：
- `ManipulatorKinematics`  
  使用改进 DH 参数描述 7 轴串联机械臂。内部存储 MDH 表（`a, α, d, θ_offset`）和简易的连杆质量/惯量。  
  提供：
  - `forward_kinematics(q)`：计算齐次变换矩阵，返回末端位姿，同时缓存所有连杆变换用于后续雅可比计算。
  - `geometric_jacobian(q)`：返回 6×7 的几何雅可比矩阵，包含线速度与角速度部分。
  - `manipulability_measure(q)`：基于雅可比行列式的可操纵性度量。

- `StiffODEIntegrator`  
  实现 2 阶 L-稳定 SDIRK 方法，用于积分形如 ẏ = f(t, y) 的刚性 ODE。通过 Newton 迭代求解隐式方程，并采用自适应步长控制。  
  提供 `integrate(f, t_span, y0, h0
