# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 多维吸积盘流体力学与磁离心喷流模拟项目描述

## 项目概述
本项目实现了一个多物理场吸积盘‑喷流数值模拟框架，围绕 Schwarzschild 黑洞的薄吸积盘，涵盖 Shakura‑Sunyaev 盘结构计算、引力势求解、流体力学时间演化、喷流粒子蒙特卡洛采样、粒子动力学、有限元径向结构方程求解、谱方法角向导数、RBF 场重构、三维高斯求积以及相关的矩阵求解器与网格工具。

主入口文件为 `main.py`，它调用所有其他模块的功能，完成完整的模拟流程并输出诊断信息。你的任务是依据本描述和 `main.py` 中的调用方式，重新实现所有缺失的辅助模块。

## 文件清单与职责
| 文件名 | 职责 |
|--------|------|
| `main.py` | 模拟主入口，零参数运行，串联所有模块，输出物理量汇总。**保留，无需实现**。 |
| `accretion_physics.py` | 吸积盘物理核心：薄盘结构、引力势、喷流判据、光谱、不稳定性、径向速度。 |
| `fem_radial.py` | 一维有限元径向求解：拉格朗日基函数、质量/刚度矩阵组装、FEM 求解器与插值。 |
| `hydrodynamics.py` | 流体力学时间积分：三阶 Runge‑Kutta 积分器、数值梯度/散度、CFL 时间步长。 |
| `matrix_solvers.py` | 稀疏与带状矩阵求解：R8SD 对称对角稀疏矩阵 + CG 法，R8PBL 对称正定带状矩阵 + Cholesky 分解。 |
| `mesh_generation.py` | 网格生成与管理：三角形细分、吸积盘截面网格、三角剖分掩码、FEM 格式转换。 |
| `monte_carlo_transport.py` | 蒙特卡洛采样与传输：楔形体采样/积分、球内均匀采样、喷流粒子初始化、光子能量传输、关联函数。 |
| `particle_dynamics.py` | 粒子动力学：速度 Verlet 积分、对力计算、势函数（正弦平方、Lennard‑Jones）、吸积盘尘埃模型。 |
| `quadrature_rules.py` | 三维数值积分：高斯‑勒让德节点/权重、张量积求积规则、柱坐标求积、精确性测试。 |
| `rbf_interpolation.py` | 径向基函数插值：多种基函数（多二次、高斯、薄板样条等）、权重计算与场重构、RBF 梯度。 |
| `spectral_methods.py` | 谱方法：多项式乘法、切比雪夫/勒让德多项式生成、谱微分矩阵、角向谱导数。 |
| `utils.py` | 通用工具：幻方矩阵构造、归一化权重、球内采样、距离统计、安全除法、值域裁剪。 |

## 模块功能与关键接口

### 1. accretion_physics.py
提供吸积盘物理的核心函数，所有函数均使用 NumPy 进行向量化计算。
- 物理常数：`G_GRAV`, `C_LIGHT`, `M_SUN`, `SIGMA_SB`, `K_BOLTZMANN`, `MP`, `MU`, `GAMMA_AD`。
- `keplerian_angular_velocity(r, M_bh)`：返回开普勒角速度。
- `sound_speed(T, mu, gamma)`：等温声速。
- `scale_height(r, M_bh, T, mu)`：吸积盘标高。
- `shakura_sunyaev_sigma(r, M_dot, M_bh, alpha, mu)`：计算 Shakura‑Sunyaev 薄盘的表面密度、温度和标高，使用 ISCO 半径作内边界。
- `viscous_torque(r, Sigma, M_bh, alpha, mu)`：粘滞力矩。
- `schwarzschild_potential(r, M_bh)`：牛顿引力势。
- `paczynski_wiita_potential(r, M_bh)`：伪牛顿势（Paczynski‑Wiita）。
- `jet_launching_criterion(r, B_z, Omega, M_bh)`：Blandford‑Payne 磁离心喷流判据，返回发射标志、Alfvén 速度和逃逸速度。
- `magnetic_braking_torque(r, B_phi, B_r, Sigma, M_bh)`：磁制动扭矩。
- `disk_spectrum_nu(nu_freq, r_in, r_out, M_dot, M_bh)`：多色黑体光谱。
- `disk_instability_criterion(...)`：热不稳定性判据，返回不稳定标志及特征时标。
- `compute_radial_velocity(Sigma, r, M_bh, alpha, M_dot)`：径向吸积速度。

### 2. fem_radial.py
实现一维
