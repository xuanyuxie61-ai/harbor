# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：核反应光学模型与统计理论综合计算平台

本项目实现一个用于中子与原子核反应的多模块计算平台，涵盖光学模型分波计算、截面与角分布分析、复合核统计衰变、集体动力学、非线性稳定性、核数据流形学习以及对称性代数等。项目以 `main.py` 为统一入口，调用 12 个辅助模块完成全部计算流程。后续将保留 `main.py`，删除其他 `.py` 文件，需根据本描述重新实现缺失模块。

## 模块总览

| 文件名 | 职责 |
|--------|------|
| `main.py` | 主入口，定义计算流程，调用各模块 |
| `optical_potential.py` | 光学势参数定义与复势场构造 |
| `chebyshev_schrodinger.py` | 径向薛定谔方程求解（Numerov 方法 + Chebyshev 谱微分校验） |
| `s_matrix.py` | S-矩阵处理、截面计算、散射振幅、SVD 分析 |
| `angular_quadrature.py` | Lebedev 球面数值积分规则 |
| `hauser_feshbach.py` | 复合核统计衰变理论（能级密度、衰变宽度、BDF2 衰变链） |
| `collective_dynamics.py` | 核集体运动阻尼振子与巨共振截面 |
| `bifurcation_stability.py` | 非线性稳定性与分岔分析（Logistic 映射、中子增殖） |
| `finite_field_symmetry.py` | GF(2) 多项式代数在核对称性中的应用 |
| `manifold_learning.py` | 核数据流形学习（Sammon 映射、LLE、幻数检测） |
| `nuclear_data_io.py` | 核素数据类、液滴模型质量表、球壳网格 |
| `orthogonality.py` | L² 内积、波函数正交化、Gram‑Schmidt、耦合矩阵元 |
| `special_functions.py` | 高精度特殊函数（Gamma、Coulomb 波函数、球 Bessel 等） |

## 模块详细说明

### `main.py`  
统一入口，零参数可运行。调用所有其他模块完成以下计算章节：  
1. 光学模型：构造 `OpticalPotentialParameters`，分波求解径向薛定谔方程，提取 S-矩阵与相移，计算总截面、弹性截面、反应截面，微分截面角分布，SVD 低秩分析，穿透系数。  
2. Hauser‑Feshbach 统计：复合核形成截面、能级密度、衰变宽度、能量平均截面、宽度涨落修正、BDF2 衰变链演化。  
3. 核数据：生成核素质量表，聚合统计，Q 值计算，球形壳层网格，幻数检测。  
4. 集体动力学：集体质量、恢复力、共振能量与宽度，受迫阻尼振子时间演化，巨共振截面与强度函数积分。  
5. 稳定性分析：Logistic 映射 Lyapunov 指数、分岔分析、中子增殖平衡点稳定性、临界慢化。  
6. 对称性代数：GF(2) 多项式运算、宇称、同位旋、组态表示、时间反演对称性。  
7. 流形学习：Sammon 映射、LLE 嵌入、结合能梯度流。  
8. 特殊函数校验：Gamma 函数、球 Bessel、Coulomb 波函数及相移。  

`main.py` 仅包含流程组织和打印输出，所有计算逻辑均委托给辅助模块。

### `optical_potential.py`  
负责光学势参数封装与势场构建。核心类 `OpticalPotentialParameters` 存储入射粒子类型、靶核质量/电荷数、入射能量、约化质量、波数、几何参数（半径、弥散参数）以及势深参数（实部、虚部体积、虚部表面、自旋‑轨道）。提供 `woods_saxon` 和 `woods_saxon_derivative` 形状函数，`thomas_spin_orbit_factor` 自旋‑轨道径向因子，`coulomb_potential` 均匀带电球库仑势。`build_optical_potential` 组装总
