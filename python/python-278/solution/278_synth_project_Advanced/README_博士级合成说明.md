# PROJECT 278 — 计算材料：相图计算与 CALPHAD 建模
## 高阶有限差分与稳定性分析（小规模可复现实验）

> **博士级合成项目说明文档**  
> 科学领域：计算材料学 / CALPHAD 热力学 / 相场动力学  
> 目标体系：Fe-C 二元合金  
> 运行方式：`python3 main.py`（零参数，统一入口）  
> Python 文件数：16 个

---

## 1. 项目科学问题概述

本项目实现一个**完整的 CALPHAD（CALculation of PHAse Diagrams）工作流**，
围绕 Fe-C 二元合金系统展开，涵盖：

1. **非均匀成分网格生成**（CVT / Lloyd 算法）
2. **多相 Gibbs 能曲面计算**（Sublattice 模型 + Inden-Hillert-Jarl 铁磁性贡献）
3. **分段常数相指示函数与 Gibbs 能景观重构**
4. **Newton-Maehly 消去法求解两相化学势平衡**（结合 LU 分解）
5. **相边界分岔检测**（Jacobian 行列式跳变 + 边界碰撞图）
6. **高阶紧致差分 Cahn-Hilliard 方程求解**（Backward Euler + Picard）
7. **刚性 ODE 积分器**（隐式 Newton-Raphson，自适应步长）
8. **Lyapunov 指数估计与线性稳定性色散关系**
9. **Allen-Cahn 相变动力学**（隐式中点法）
10. **高斯整数螺旋扫描**（成分-温度空间探索）
11. **蒙特卡罗不确定性传播**（超球面采样 + 赌徒破产参数存活分析）
12. **MOEA/D 多目标 CALPHAD 参数优化**（PBI + Chebyshev 自适应标量化）
13. **统计学敏感性分析**（Welch t / Clopper-Pearson / Wilcoxon / Tipping-point）

核心公式体系涉及：

- **Redlich-Kister-Muggiano 多项式**：$G^E = x_A x_B \sum_n L^{(n)}(T) (x_A - x_B)^n$
- **Cahn-Hilliard 方程**：$\frac{\partial c}{\partial t} = \nabla \cdot \left[ M(c) \nabla \left( \frac{\partial g}{\partial c} - 2\kappa \nabla^2 c \right) \right]$
- **化学势平衡条件**：$\mu_i^\alpha(x^\alpha, T) = \mu_i^\beta(x^\beta, T)$
- **Spinodal 条件**：$\frac{\partial^2 G_m}{\partial x_C^2} = 0$
- **线性稳定性色散关系**：$\omega(k) = -M k^2 \left( \frac{\partial^2 g}{\partial c^2} + 2\kappa k^2 \right)$

---

## 2. 种子项目到科学问题的映射

| # | 种子项目 | 核心算法 | CALPHAD 角色 |
|---|----------|----------|--------------|
| 1 | 689_linpack_d | 稠密线性代数（LU/Cholesky 分解） | Newton 步求解 2×2 化学势 Jacobian |
| 2 | 1085_border-collision-bifurcations | 分段光滑映射的边界碰撞分岔 | 相边界追踪与切线分岔检测 |
| 3 | 259_cvt_square_nonuniform | 重心 Voronoi 非均匀网格（Lloyd） | 成分空间自适应离散化 |
| 4 | 801_newton_maehly | Newton-Maehly 消去法多项式求根 | 多变量相平衡方程组求解 |
| 5 | 841_ozone_ode | 刚性多组分 ODE（臭氧化学） | CH 方程刚性时间积分 |
| 6 | 555_hyperball_positive_distance | 高维正超球面采样 | CALPHAD 参数不确定性传播 |
| 7 | 064_backward_euler_fixed | 定点迭代 Backward Euler | Cahn-Hilliard 隐式时间推进 |
| 8 | 450_gamblers_ruin_simulation | 赌徒破产随机游走 | CALPHAD 参数存活概率分析 |
| 9 | 923_pwc_plot_1d | 分段常数（piecewise constant）表示 | 相指示函数与 Gibbs 景观重构 |
| 10 | 1219_MOEA-D-Container-Routing | MOEA/D 多目标分解进化算法 | CALPHAD 参数拟合（精度 + 鲁棒性） |
| 11 | 828_ode_midpoint | 隐式中点法 | Allen-Cahn 相变动力学 |
| 12 | 456_gaussian_prime_spiral | 高斯整数螺旋轨迹 | 温度-成分二维相空间扫描 |
| 13 | 1204_lle-chaos-demos | KNN Lyapunov 指数估计 | 相场动力学稳定性量化 |
| 14 | 100_blood_pressure_ode | 顺应性/阻力切换 ODE | 相变动力学切换模型 |
| 15 | 1100_Exercise-Prescription-System | 统计亚组分析 / 敏感性分析 | CALPHAD 参数 tipping-point 分析 |

---

## 3. 核心数学物理模型

### 3.1 CALPHAD Sublattice 模型（替代式相）

对 FCC/BCC/LIQUID 相，摩尔 Gibbs 能为：

$$
G_m(x_C, T) = (1-x_C) G^\circ_{Fe}(T) + x_C G^\circ_C(T)
+ RT[(1-x_C)\ln(1-x_C) + x_C \ln x_C]
+ x_C(1-x_C) \sum_{n=0}^{2} L^{(n)}_{Fe-C}(T) (2x_C - 1)^n
$$

其中 Redlich-Kister 交互参数温度依赖：

$$
L^{(n)}(T) = a_n + b_n T + c_n T \ln T + d_n T^2 + e_n T^{-1} + f_n T^7 + g_n T^{-9}
$$

### 3.2 Inden-Hillert-Jarl 铁磁贡献

$$
G^{mag} = RT \ln(\beta + 1) \cdot f(\tau), \quad \tau = T / T_C
$$

对 $\tau < 1$（有序）：

$$
f(\tau) = 1 - \frac{1}{A}\left[ \frac{7(1-\tau)}{15p}
+ \frac{\tau^{3/p}}{30} + \frac{\tau^{9/p}}{150} + \frac{\tau^{15/p}}{400} \right]
$$

对 $\tau > 1$（无序）：

$$
f(\tau) = -\frac{1}{A}\left[ \frac{\tau^{-5/p}}{30}
+ \frac{\tau^{-15/p}}{150} + \frac{\tau^{-25/p}}{400} \right]
$$

其中 $A = \frac{518}{1125} + \frac{11692}{15975}\left(\frac{1}{p} - 1\right)$。

### 3.3 高阶紧致差分算子

采用 Padé 型紧致差分：

$$
\frac{1}{6} f'_{i-1} + \frac{2}{3} f'_i + \frac{1}{6} f'_{i+1}
= \frac{f_{i+1} - f_{i-1}}{2h}
$$

此为 $O(h^4)$ 精度的一阶导数紧致格式。四阶导数采用：

$$
\delta^4_x f_i = \frac{f_{i-2} - 4f_{i-1} + 6f_i - 4f_{i+1} + f_{i+2}}{h^4}
$$

### 3.4 Cahn-Hilliard Backward Euler + Picard

半离散化：

$$
\frac{c^{n+1} - c^n}{\Delta t}
= D_h \left[ M(c^{n+1}) D_h \left( g'(c^{n+1}) - 2\kappa D_h^2 c^{n+1} \right) \right]
$$

Picard 迭代线性化：

$$
c^{(k+1)} = c^n + \Delta t \cdot D_h \left[ M(c^{(k)}) D_h \left( g'(c^{(k)}) - 2\kappa D_h^2 c^{(k+1)} \right) \right]
$$

### 3.5 Spinodal 与线性稳定性

Spinodal 边界条件：

$$
\frac{\partial^2 G_m}{\partial x_C^2} = RT \left( \frac{1}{x_C} + \frac{1}{1-x_C} \right)
+ \frac{d^2}{dx^2}[x(1-x)L(x)] = 0
$$

Cahn-Hilliard 线性化色散关系：

$$
\omega(k) = -M k^2 \left( \frac{\partial^2 g}{\partial c^2} + 2\kappa k^2 \right)
$$

临界波数：$k_c = \sqrt{-\frac{g''}{2\kappa}}$（当 $g'' < 0$）  
最快增长波数：$k_m = k_c / \sqrt{2}$

### 3.6 Newton-Maehly 消去法

原始 Maehly 消去（用于多项式求根）：

$$
x_k^{new} = x_k - \frac{p(x_k)}{p'(x_k) - \sum_{j \neq k} \frac{1}{x_k - x_j}}
$$

推广到 2×2 相平衡：

$$
\delta = -J^{-1} F, \quad
\delta_1' = \delta_1 + \epsilon \frac{\delta_2}{x_C^\alpha - x_C^\beta}, \quad
\delta_2' = \delta_2 + \epsilon \frac{\delta_1}{x_C^\beta - x_C^\alpha}
$$

其中 $F = (\mu_{Fe}^\alpha - \mu_{Fe}^\beta, \mu_C^\alpha - \mu_C^\beta)^T$。

### 3.7 MOEA/D 标量化

**PBI（Penalty-based Boundary Intersection）**：

$$
g^{pbi}(x|w, z^*) = d_1 + \theta d_2, \quad
d_1 = \frac{|(f(x) - z^*)^T w|}{\|w\|}, \quad
d_2 = \|f(x) - z^* - d_1 \hat{w}\|
$$

**Chebyshev 标量化**：

$$
g^{cheb}(x|w, z^*) = \max_i \{ w_i |f_i(x) - z_i^*| \}
$$

### 3.8 Lyapunov 指数 KNN 估计

重构延迟坐标：$\mathbf{y}_i = (x_i, x_{i+\tau}, \ldots, x_{i+(m-1)\tau})$，  
对每个参考点 $\mathbf{y}_i$ 找 K 近邻 $\mathbf{y}_{j}$，追踪误差演化：

$$
\lambda \approx \frac{1}{K \Delta t \cdot H} \sum_{i} \sum_{k=1}^{H}
\ln \frac{\|\mathbf{y}_{i+k} - \mathbf{y}_{nn(i)+k}\|}{\|\mathbf{y}_i - \mathbf{y}_{nn(i)}\|}
$$

---

## 4. 项目文件清单

| 文件名 | 行数 | 功能 |
|--------|------|------|
| `main.py` | 637 | 统一入口，13 个计算步骤编排 |
| `calphad_fec_constants.py` | 286 | Fe-C 热力学数据库 + 辅助函数 |
| `gibbs_energy_calphad.py` | 439 | 多相 Gibbs 能曲面、化学势、二阶导 |
| `high_order_fd.py` | 339 | 紧致差分算子、von Neumann 因子 |
| `newton_maehly_equilibrium.py` | 356 | Newton-Maehly 相平衡、LU 分解 |
| `backward_euler_calphad.py` | 334 | Backward Euler + Picard CH 求解 |
| `cvt_composition_mesh.py` | 211 | CVT 非均匀成分网格（Lloyd） |
| `monte_carlo_hyperball.py` | 351 | 超球采样、成对距离、MC 传播 |
| `calphad_parameter_optimizer.py` | 469 | MOEA/D 多目标 CALPHAD 优化 |
| `lyapunov_phase_stability.py` | 338 | LLE 估计、色散关系、spinodal 搜索 |
| `bifurcation_detection.py` | 334 | 切线分岔、边界碰撞、Jacobian 检测 |
| `ode_midpoint_phasefield.py` | 264 | Allen-Cahn 中点法相变动力学 |
| `piecewise_constant_phase.py` | 266 | 分段常数相指示、Gibbs 景观光滑化 |
| `gaussian_spiral_scan.py` | 278 | 高斯整数螺旋扫描、相边界追踪 |
| `clinical_statistics_calphad.py` | 480 | Welch / Clopper-Pearson / Wilcoxon / Tipping |
| `stiff_ode_cahn_hilliard.py` | 345 | 刚性 CH 积分器、Jacobian 特征值估计 |

**总计**：16 个 Python 文件，约 5727 行代码。

---

## 5. 运行方式

### 环境要求

- Python 3.8+
- NumPy（仅需 numpy）

### 执行命令

```bash
cd /mnt/data/zpy/sci-swe/source\ code/Synthesis-project-python/278_synth_project/278_synth_project_Advanced
python3 main.py
```

### 预期输出

程序将依次执行 13 个步骤，每步打印：

- 当前步骤标题与主要计算结果
- 物理量（Gibbs 能、化学势、波数、Lyapunov 指数等）
- 收敛状态、残差、迭代次数

总运行时间约 **30-60 秒**（取决于硬件），最终打印：

```
所有 13 个计算步骤均成功完成
种子项目映射:
  689_linpack_d        → LU 分解 (Step 4)
  ...
  1100_Exercise        → 统计检验 (Step 13)
```

---

## 6. 科学问题落地说明

### 6.1 本项目解决的科学问题

本项目实现**小尺度可复现的 Fe-C 二元合金 CALPHAD 全流程**，可回答：

1. 在给定温度下，Fe-C 合金的 FCC/BCC/LIQUID 相平衡成分是什么？
2. 该成分处于 spinodal 区还是亚稳区？
3. Spinodal 分解的最快增长波长是多少？
4. 相场动力学的 Lyapunov 稳定性如何？
5. 当 CALPHAD 参数存在实验误差时，相图预测的不确定性如何传播？
6. 通过 MOEA/D 多目标优化，如何同时保证拟合精度与参数鲁棒性？

### 6.2 博士级难度体现

- **多模型耦合**：Sublattice + IHJ 铁磁 + RK 多项式 + CH 相场
- **高阶数值方法**：紧致差分（4 阶）+ Backward Euler + Picard + 隐式中点
- **稳定性与分岔分析**：色散关系、Lyapunov 指数、Jacobian 行列式跳变
- **多目标优化**：MOEA/D + SBX + 多项式变异 + 自适应 PBI/Chebyshev
- **统计推断**：Welch / Clopper-Pearson / Wilcoxon / Tipping-point 全套

### 6.3 边界处理与数值鲁棒性

- 所有 `np.log` 调用前使用 `np.clip(x, 1e-15, 1 - 1e-15)` 防止 `ln(0)`
- LU 分解检测近奇异矩阵（`|pivot| < 1e-30`）并回退到 `numpy.linalg.solve`
- Newton 步带阻尼：`damp = min(1, 0.5 / max(|delta|, 0.5))`
- Maehly 消去因子容差：`|x_a - x_b| < NM_DEF_TOL` 时置零
- 超球采样严格限制在正象限，符合成分物理约束
- MOEA/D 变量边界：`lb ≤ theta ≤ ub` 通过 `np.clip` 强制

### 6.4 可复现性

- 所有随机源使用固定种子 `MC_RANDOM_SEED = 278`
- 小规模参数（MOEA pop=12, gen=6；Newton 80 iter；MC 500 samples）保证快速复现
- 实验数据内嵌在代码中，无需外部文件

---

## 7. 可视化内容处理

本项目已按规则**删除所有可视化代码**（matplotlib、plt.show、plt.plot 等）。
所有结果以纯文本数值形式打印到标准输出。

---

## 8. 总结

本项目成功将 15 个来源各异（数值代数、动力系统、ODE、随机模拟、进化计算、统计检验等）的科研代码种子项目，融合为一个**博士级 CALPHAD 计算材料工作流**。每个种子项目都在新架构中承担不可替代的物理角色，算法与公式深度耦合到相图计算与相场动力学的领域语义中。

代码具备工程级鲁棒性、可复现性、零参数可运行性，并通过 `python3 main.py` 一键完成从网格生成到统计推断的完整科学计算流程。
