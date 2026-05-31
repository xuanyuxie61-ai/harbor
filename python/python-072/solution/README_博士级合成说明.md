# 计算流体力学：多相流界面追踪与相变 —— 博士级合成说明

## 项目概述

本项目基于 **15 个科研种子项目**的核心算法，在 **计算流体力学：多相流界面追踪与相变** 领域内，融合构建了一个面向二元合金凝固过程的博士级耦合数值模拟系统。

核心科学问题：
> **基于相场-Navier-Stokes 耦合模型，模拟二元合金在过冷熔体中的凝固过程，追踪液固界面的演化，分析温度场、浓度场与速度场的耦合发展，以及 Mullins-Sekerka 界面失稳与枝晶形态的形成机制。**

---

## 核心物理模型与数学公式

### 1. Allen-Cahn 相场方程

序参量 $\phi \in [-1, 1]$ 描述液固两相，其中 $\phi = +1$ 为固相，$\phi = -1$ 为液相：

$$
\tau \frac{\partial \phi}{\partial t} = \varepsilon^2 \nabla^2 \phi - W'(\phi) - \lambda_T (1-\phi^2)^2 (T - T_M) - \lambda_C (1-\phi^2)^2 (C - C_e)
$$

其中双阱势函数及其导数为：

$$
W(\phi) = \frac{1}{4}(\phi^2 - 1)^2, \quad W'(\phi) = \phi^3 - \phi = \phi(\phi^2 - 1)
$$

界面能密度泛函：

$$
f_{\text{int}} = \frac{\varepsilon^2}{2} |\nabla \phi|^2 + W(\phi), \quad E_{\text{int}} = \int_\Omega f_{\text{int}} \, d\Omega
$$

固相分数插值函数：

$$
h(\phi) = \frac{1}{2}(1 + \phi), \quad h \in [0, 1]
$$

### 2. Navier-Stokes 方程（不可压流体）

连续性方程：

$$
\nabla \cdot \mathbf{v} = 0
$$

动量方程：

$$
\rho \left( \frac{\partial \mathbf{v}}{\partial t} + \mathbf{v} \cdot \nabla \mathbf{v} \right) = -\nabla p + \mu \nabla^2 \mathbf{v} + \mathbf{F}_\sigma
$$

界面张力体积力（基于相场 CSF 模型）：

$$
\mathbf{F}_\sigma = \frac{3\sigma}{2\sqrt{2}\varepsilon} W'(\phi) \nabla \phi
$$

采用 **投影法（Projection Method）** 分两步求解：
1. **预测步**：计算中间速度 $\mathbf{v}^*$
2. **投影步**：求解压力泊松方程 $-\nabla^2 p = -\frac{\rho}{\Delta t} \nabla \cdot \mathbf{v}^*$，修正速度满足不可压条件

### 3. 温度对流-扩散方程

$$
\frac{\partial T}{\partial t} + \mathbf{v} \cdot \nabla T = \alpha_T \nabla^2 T + \frac{L_f}{c_p} \frac{\partial h}{\partial t}
$$

其中相变潜热源项：

$$
Q_{\text{latent}} = \frac{L_f}{c_p} \frac{h(\phi^{n+1}) - h(\phi^n)}{\Delta t}
$$

### 4. 浓度对流-扩散方程

$$
\frac{\partial C}{\partial t} + \mathbf{v} \cdot \nabla C = \nabla \cdot \left( D(\phi) \nabla C \right) + Q_C
$$

变系数扩散系数：

$$
D(\phi) = D_s \cdot h(\phi) + D_l \cdot (1 - h(\phi))
$$

溶质排出源项（凝固时溶质被排斥到液相）：

$$
Q_C = C \frac{(1 - k_p)}{(k_p + (1-k_p)h)} \frac{\partial h}{\partial t}
$$

### 5. Gibbs-Thomson 过冷度关系

总过冷度：

$$
\Delta T_{\text{total}} = \Delta T_{\text{thermal}} + \Delta T_{\text{solutal}} - \Gamma \kappa
$$

其中：
- 热过冷度：$\Delta T_{\text{thermal}} = T_M - T$
- 溶质过冷度：$\Delta T_{\text{solutal}} = -m_L (C - C_e)$
- 毛细长度：$\Gamma = \gamma / \Delta S_f$
- 界面曲率：$\kappa = \nabla \cdot \left( \frac{\nabla \phi}{|\nabla \phi|} \right)$

### 6. Mullins-Sekerka 不稳定性判据

枝晶尖端稳定性参数：

$$
\sigma^* = \frac{2 D_l d_0 V}{\lambda^2}
$$

其中 $d_0 = \Gamma / \Delta T_0$ 为毛细长度，$V$ 为特征速度，$\lambda$ 为扰动波长。当 $\sigma^* < 0.025$ 时，界面失稳形成枝晶。

---

## 文件结构与种子项目映射

本项目共包含 **12 个 Python 文件**，每个文件均融合了至少一个种子项目的核心算法：

### `phase_field_core.py` — 相场方程核心
- **融入种子**: 464_gen_hermite_exactness（高斯核函数思想用于界面能计算）
- 实现 Allen-Cahn 相场方程的离散与求解
- 包含双阱势、界面法向量/曲率、界面能密度泛函计算

### `navier_stokes_solver.py` — NS 方程求解器
- **融入种子**: 875_poisson_1d（压力泊松方程的 Gauss-Seidel 迭代求解思想）
- 投影法求解不可压 NS 方程
- 连续表面力（CSF）模型计算界面张力

### `fem_2d_serene.py` — 二维 Serendipity 有限元
- **融入种子**: 402_fem2d_bvp_serene（核心 FEM 框架完整迁移）
- 8 节点 serendipity 四边形单元
- 3×3 Gauss-Legendre 数值积分
- Dirichlet 边界条件处理与 L2 误差估计

### `thermal_transport.py` — 热质传输方程
- **融入种子**: 060_axon_ode（Hodgkin-Huxley 离子通道动力学思想 → 相变潜热释放动力学）
- 温度场与浓度场对流-扩散方程
- 变系数扩散项（守恒型差分格式）
- 潜热源项与溶质排出源项

### `interface_tracking.py` — 界面追踪与几何分析
- **融入种子**: 559_hypercube_integrals（积分思想用于界面面积计算）
- 界面等值线提取、法向量与曲率计算
- 界面面积（长度）、形态学数、尖端速度诊断

### `numerical_quadrature.py` — 数值求积与积分检验
- **融入种子**: 464_gen_hermite_exactness（广义 Gauss-Hermite 求积精确度检验）
- **融入种子**: 559_hypercube_integrals（超立方体单项式积分）
- **融入种子**: 1207_test_int（多种求积方法：Simpson、梯形、复合 Gauss-Legendre）
- Gauss-Hermite/Legendre 求积、Monte Carlo 积分、求积精确度检验

### `stability_analysis.py` — 稳定性分析与特征值
- **融入种子**: 203_companion_matrix（Hermite/Chebyshev 伴矩阵特征值计算）
- **融入种子**: 697_log_norm（矩阵对数范数 L1/L2/L∞）
- **融入种子**: 700_logistic_bifurcation（非线性分叉分析与 Lyapunov 指数）
- Jacobian 线性稳定性分析、CFL 条件、Mullins-Sekerka 稳定性参数

### `stochastic_perturbation.py` — 随机扰动与热噪声
- **融入种子**: 029_asa053（Wishart 分布 / Marsaglia 极坐标法正态随机数）
- **融入种子**: 060_axon_ode（门控变量随机动力学 → 相变激活度模型）
- 热涨落噪声（白噪声/有色噪声）、随机 Allen-Cahn 方程
- Fluctuation-Dissipation 定理应用

### `time_integrator.py` — 时间积分器
- **融入种子**: 831_ode_trapezoidal（梯形法隐式 ODE 求解器，Picard 迭代）
- **融入种子**: 1138_spring_double_ode（双弹簧耦合 ODE 系统 → 两相界面振动模型）
- 显式 Euler、梯形法、RK4、自适应 RK45
- 相场专用显式-隐式混合时间步进器

### `mesh_adaptation.py` — 自适应网格细化
- **融入种子**: 156_change_dynamic（动态规划思想用于最优网格节点分布）
- **融入种子**: 1358_trinity（三角形网格拓扑结构）
- 基于相场梯度的误差指示器、界面聚焦加密
- 动态规划网格优化、三角形网格质量评估

### `spectral_solver.py` — 谱方法求解器
- **融入种子**: 1085_sine_transform（离散正弦变换 DST）
- **融入种子**: 875_poisson_1d（Gauss-Seidel 迭代求解 Poisson 方程）
- DST 快速求解 1D/2D Poisson 方程与热方程
- 谱域指数衰减时间推进

### `main.py` — 统一入口
- 零参数运行，执行完整的相场-NS 耦合模拟
- 集成所有模块的数值验证测试
- 输出界面面积、形态学数、尖端速度、Mullins-Sekerka 稳定性分析等诊断

---

## 运行方式

```bash
cd Synthesis-project-python/072_synth_project
python main.py
```

程序将自动执行：
1. **数值方法验证**（10 项核心算法测试）
2. **主模拟**：二元合金凝固过程（相场-NS 耦合）
3. **最终诊断**：界面演化统计与稳定性分析

---

## 关键数值方法

| 方法 | 应用 | 文件 |
|------|------|------|
| Allen-Cahn 相场方程 + RK4 时间积分 | 界面演化 | phase_field_core.py, time_integrator.py |
| 投影法（Projection Method）| 不可压 NS 方程 | navier_stokes_solver.py |
| 变系数守恒型差分 | 热质传输 | thermal_transport.py |
| Serendipity FEM + Gauss-Legendre 积分 | 偏微分方程离散 | fem_2d_serene.py |
| 离散正弦变换（DST）| Poisson/热方程快速求解 | spectral_solver.py |
| Gauss-Seidel 迭代 | 压力泊松方程 | spectral_solver.py |
| 伴矩阵特征值 | 多项式求根/稳定性分析 | stability_analysis.py |
| 矩阵对数范数 | 数值稳定性估计 | stability_analysis.py |
| 动态规划 | 网格节点最优分配 | mesh_adaptation.py |
| Marsaglia 极坐标法 | 热噪声生成 | stochastic_perturbation.py |

---

## 边界处理与数值鲁棒性

1. **相场限制器**：$\phi \in [-1.2, 1.2]$，防止非物理溢出
2. **温度/浓度限制器**：物理合理范围截断
3. **RHS 变化率限制**：温度场和浓度场右端项施加 $[-10/\Delta t, 10/\Delta t]$ 限制
4. **Neumann 边界条件**：相场、温度、浓度采用零法向梯度边界
5. **Dirichlet 边界条件**：速度场无滑移边界，FEM 边界值固定
6. **CFL 稳定性检查**：时间步长自动适配扩散限制
7. **Jacobian 退化检测**：FEM 单元 Jacobian 行列式下限保护
8. **除零保护**：梯度模长、分母等关键量设置最小阈值 $10^{-12}$

---

## 科学意义

本项目合成的代码系统能够：
- 模拟二元合金凝固过程中的液固界面演化
- 分析热扩散、溶质扩散与流体运动对界面形态的影响
- 评估 Mullins-Sekerka 界面失稳条件
- 计算枝晶尖端速度与形态学特征数
- 为材料科学中的凝固微观组织预测提供数值工具

---

## 技术规格

- **语言**: Python 3
- **依赖**: NumPy, SciPy（仅 gamma 函数与 convolve2d）
- **网格**: 41×41（可调整）
- **时间步长**: 自适应限制
- **无可视化代码**
