# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：心脏电生理与心律失常模拟

本项目是一个 Python 科学计算项目，用于模拟心脏组织的电生理活动，分析心律失常机制。项目整合了离子通道动力学、组织反应扩散方程、网格生成、随机采样、数值积分、线性代数求解等多个子模块，提供从单细胞到二维组织层面的完整模拟流程。主入口为 `main.py`，其余各 `.py` 文件为被依赖的模块，复现时需保持与 `main.py` 的接口一致。

## 文件清单与职责

- **`main.py`**：主入口，零参数运行，依次执行 9 个阶段，验证各模块功能并运行完整模拟，最终输出心律失常指标。该文件保留，无需复现。
- **`utils.py`**：通用工具与组合数学模块，提供斯特林数、贝尔数、离子通道状态枚举、格雷码生成、数值误差分析（相对误差、收敛率、截断误差估算、灾难性抵消测试）等基础功能。
- **`linear_algebra_core.py`**：线性代数核心求解模块，实现带状对称正定矩阵的共轭梯度法（CG）、幂法求主特征值、稳定性特征值分析、二维泊松方程的 CG 求解，以及拉普拉斯算子的带状矩阵构建。
- **`numerical_integration.py`**：高维数值积分模块，提供四边形 Witherden 求积规则（精度 p=1,3,5,7,9 等）、任意四边形上的仿射映射积分、一维/二维蒙特卡洛积分，以及函数矩的计算。
- **`stochastic_sampler.py`**：准随机序列与随机采样模块，实现 Niederreiter 基2低差异序列生成、Hammersley 序列，用于心肌电导率参数的均匀采样和多边形面积估计，并提供差异度计算。
- **`mesh_generator.py`**：心脏组织网格生成模块，基于 Centroidal Voronoi Tessellation (CVT) 迭代在给定多边形区域内生成非结构化节点，进行点在多边形内测试（射线投射法），并实现二维散乱数据的 Shepard 插值。
- **`ion_channel_dynamics.py`**：离子通道动力学与随机噪声模块，包含 Hodgkin-Huxley 类型门控速率常数计算、门控变量更新、离子电流计算（Na⁺, Ca²⁺, K⁺ 等），单细胞动作电位模拟，Aliev-Panfilov 简化反应项，1/f^α 有色噪声生成，以及 Squircle ODE 积分（用于守恒量测试）。
- **`tissue_reaction_diffusion.py`**：组织反应扩散方程求解模块，实现各向异性扩散张量构建、各向同性与各向异性五点差分拉普拉斯算子、前向欧拉/Crank-Nicolson/ADI 时间步进、纤维角度场生成，以及将上述组合成完整的反应扩散方程求解器（Monodomain 模型）。
- **`electrophysiology_simulator.py`**：电生理模拟集成器，提供刺激区域掩码、方波刺激、波前速度计算、折返活动检测、动作电位时程（APD）与有效不应期（ERP）估算、波长计算、心律失常风险指数，以及高层次的全模拟函数，整合其他模块执行完整仿真并输出分析结果。

## 模块间依赖关系

- `main.py` 直接导入并使用所有其他模块。
- `electrophysiology_simulator.py` 在 `run_full_simulation` 中导入 `ion_channel_dynamics`、`tissue_reaction_diffusion`、`mesh_generator`、`linear_algebra_core`。
- `tissue_reaction_diffusion.py` 在 `crank_nicolson_step` 中导入 `linear_algebra_core` 的 CG 求解器。
- 其余模块相对独立，但 `mesh_generator` 的 `polygon_contains_point` 被 `stochastic_sampler` 的 `estimate_area_qmc` 间接使用（通过独立的射线投射实现，但思想一致）。
- 所有模块均依赖 `numpy` 和 `math`。

## 主要算法与数值方法

### 离子通道与单细胞动力学
- **门控变量**：速率常数 α, β 由分段函数给出（基于膜电位），采用解析积分更新 `x_{n+1} = x_∞ + (x_n - x_∞) exp(-dt/
