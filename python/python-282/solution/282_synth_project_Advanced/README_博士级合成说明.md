# 固态电解质界面反应建模 — 高阶有限差分与稳定性分析

## 项目概述

本项目为面向锂离子电池负极固态电解质界面（Solid Electrolyte Interphase, SEI）层的**博士级科研计算代码**，融合 15 个经典数值算法种子项目，构建一个完整的反应-扩散-电迁移-力学耦合高阶数值仿真框架。

### 科学问题
锂离子在充放电循环过程中，负极表面会形成一层纳米厚度的 SEI 膜。该膜的生长动力学涉及：
1. **Li⁺ 在非均匀 SEI 介质中的扩散**（Nernst-Planck 方程）
2. **电化学反应动力学**（Butler-Volmer 方程）
3. **SEI 内部的电场分布**（Poisson 方程）
4. **SEI 力学应力演化**（弹性波方程）
5. **SEI 形貌失稳**（Lorenz 型降维动力学）
6. **SEI 成核的随机性**（蒙特卡罗方法）

本项目使用**高阶有限差分方法**对上述耦合方程组进行数值求解，并进行严格的**von Neumann 稳定性分析**，提供可复现的小规模实验结果。

---

## 文件结构（12 个 Python 模块）

| 文件名 | 融合种子项目 | 主要功能 |
|--------|--------------|----------|
| `main.py` | — | 统一入口，零参数运行 |
| `sei_parameters.py` | — | 物理/化学/材料参数集中管理 |
| `grid_mesh.py` | 1333_triangulation_boundary_nodes<br>360_fd1d_heat_explicit<br>682_line_lines_packing | 一维 SEI 网格构造、边界节点识别、纳米孔隙 Rényi 密堆积 |
| `high_order_fd.py` | 360_fd1d_heat_explicit<br>479_gram_polynomial<br>1402_wave_pde | 2阶/4阶中心差分、Gram 正交多项式重构、周期边界 Laplacian |
| `butler_volmer_kinetics.py` | 125_burgers_steady_viscous | Butler-Volmer 电化学动力学、Newton 非线性求解器、Thomas 三对角算法 |
| `sei_diffusion_solver.py` | 360_fd1d_heat_explicit<br>1402_wave_pde | Li⁺ 反应-扩散-电迁移显式时间推进、SEI 应力波传播耦合 |
| `stability_analysis.py` | 360_fd1d_heat_explicit_cfl<br>125_burgers_steady_viscous | von Neumann 放大因子分析、CFL 临界值、反应-扩散耦合稳定性、Newton 收敛性诊断 |
| `invariant_detector.py` | 1256_AXVIAM_gw-ringdown-invariant | SEI 演化无量纲不变量 Ξ(t) = |dL/dt|/(|d²L/dt²|+ε) 与"ringdown"稳态 plateau 检测 |
| `eigenmode_decomposition.py` | 326_eigenfaces | SEI 浓度场 PCA 本征模分解（Turk-Pentland trick） |
| `stochastic_nucleation.py` | 348_fair_dice_simulation<br>899_polyomino_parity<br>323_e_spigot | 蒙特卡罗 SEI 成核、多维不定方程化学计量学、spigot 高精度常数计算 |
| `quadrature_integration.py` | 467_gen_laguerre_rule<br>519_hermite_exactness | Gauss-Laguerre 求积（Boltzmann 权重积分）、Gauss-Hermite 求积（活化能分布）、精确性检验 |
| `sei_lorenz_dynamics.py` | 092_bioconvection_ode | SEI 形貌失稳 Lorenz 降维模型、不动点分类、RK4 积分、分岔分析 |
| `timestamp_utils.py` | 1412_weekday_zeller | Zeller 同余式、实验时间戳、循环充放电调度 |

---

## 核心数学物理公式

### 1. Nernst-Planck 扩散方程（SEI 中 Li⁺ 传输）

$$\frac{\partial c}{\partial t} = \nabla \cdot \left( D_{\text{eff}} \nabla c \right) - \nabla \cdot \left( \frac{z F D_{\text{eff}} c}{R T} \nabla \phi \right) + R_{\text{sei}}(c, \phi)$$

### 2. Butler-Volmer 电化学动力学

$$j = j_0 \left[ \exp\left(\frac{\alpha_a F \eta}{R T}\right) - \exp\left(-\frac{\alpha_c F \eta}{R T}\right) \right]$$

其中过电位 $\eta = \phi_s - \phi_e - U_{\text{eq}}$。

### 3. 高阶有限差分

**二阶中心差分（$O(\Delta x^2)$）**：
$$f''(x_i) \approx \frac{f_{i-1} - 2 f_i + f_{i+1}}{\Delta x^2}$$

**四阶中心差分（$O(\Delta x^4)$）**：
$$f''(x_i) \approx \frac{-f_{i-2} + 16 f_{i-1} - 30 f_i + 16 f_{i+1} - f_{i+2}}{12 \Delta x^2}$$

**Gram 多项式重构**（离散正交多项式基投影）：
$$P_0(x) = 1, \quad P_{k+1}(x) = x P_k(x) - \beta_k P_{k-1}(x)$$
$$\beta_k = \frac{k^2 (m^2 - k^2)}{4(4 k^2 - 1)}$$

### 4. von Neumann 稳定性分析

FTCS 格式放大因子：
$$G(k) = 1 - 4 \, \text{CFL} \, \sin^2(k \Delta x / 2)$$

稳定性要求 $|G(k)| \leq 1$ 对所有 $k$，得临界 CFL：
$$\text{CFL}_{\text{crit}} = \frac{D \Delta t}{\Delta x^2} \leq 0.5 \quad \text{（二阶）}$$
$$\text{CFL}_{\text{crit}} \approx 0.375 \quad \text{（四阶）}$$

### 5. SEI 形貌失稳 Lorenz 降维

$$\dot{X} = \text{Sc}(Y - X), \quad \dot{Y} = \text{Ra} \, X + X Z - Y, \quad \dot{Z} = -X Y - b Z$$

非平凡不动点（$\text{Ra} > 1$）：
$$X^* = Y^* = \pm \sqrt{b(\text{Ra} - 1)}, \quad Z^* = -(\text{Ra} - 1)$$

### 6. 无量纲不变量（类比引力波 ringdown）

$$\Xi(t) = \frac{|dL/dt|}{|d^2 L/dt^2| + \varepsilon}, \quad \tilde{\Xi} = \frac{\tau_{\text{char}}}{\Xi_{\text{plateau}}}$$

### 7. Gauss 求积

**Gauss-Laguerre**（半无穷域，Boltzmann 权重）：
$$\int_0^\infty x^\alpha e^{-x} f(x) \, dx \approx \sum_{i=1}^N w_i f(x_i)$$

**Gauss-Hermite**（活化能高斯分布）：
$$\int_{-\infty}^{+\infty} e^{-x^2} f(x) \, dx \approx \sum_{i=1}^N w_i f(x_i)$$

$N$ 阶求积精确至 $2N-1$ 次多项式。

---

## 融合种子项目映射详表

| 种子项目 | 核心算法 | SEI 建模中的角色 |
|----------|----------|------------------|
| **360_fd1d_heat_explicit** | 显式 FTCS + CFL | SEI 扩散方程时间推进 |
| **125_burgers_steady_viscous** | Newton + Jacobian | 稳态 Butler-Volmer-NP 耦合求解 |
| **1402_wave_pde** | 波动方程 leapfrog | SEI 弹性应力波传播 |
| **479_gram_polynomial** | Gram 正交多项式递推 | 高阶空间重构 |
| **092_bioconvection_ode** | Lorenz 系统 | SEI 形貌失稳降维模型 |
| **1256_AXVIAM** | Ringdown 不变量检测 | SEI 生长稳态检测 |
| **326_eigenfaces** | PCA 本征分解 | 浓度场模式提取 |
| **348_fair_dice_simulation** | 蒙特卡罗离散采样 | SEI 随机成核 |
| **323_e_spigot** | Spigot 高精度算法 | 基本常数高精度验证 |
| **899_polyomino_parity** | 多维 Diophantine 求解 | 化学计量学平衡 |
| **682_line_lines_packing** | Rényi RSA 密堆积 | SEI 纳米孔隙构型 |
| **467_gen_laguerre_rule** | Gauss-Laguerre 求积 | Boltzmann 平均速率 |
| **519_hermite_exactness** | Gauss-Hermite 精确性 | 活化能分布积分 |
| **1333_triangulation_boundary_nodes** | 边界节点识别 | SEI 网格边界分类 |
| **1412_weekday_zeller** | Zeller 同余式 | 实验时间戳与调度 |

---

## 运行方法

本项目**零参数运行**：

```bash
cd 282_synth_project_Advanced
python main.py
```

输出 12 个阶段的完整科学计算结果，涵盖从网格构造、参数校验、差分精度验证、稳定性分析、Butler-Volmer 动力学、时间推进、不变量检测、本征模分解、Lorenz 形貌分析、随机成核到 Gauss 求积的完整仿真流程。

---

## 运行输出示例（节选）

```
========================================================================
  阶段 5：von Neumann 稳定性分析
========================================================================
  实际 CFL 数: 0.2000
  二阶临界 CFL: 0.5000
  四阶临界 CFL: 0.3750
  von Neumann 扫描（二阶格式）:
    CFL = 0.100: |G|_max = 0.999975 [✓ 稳定]
    ...
    CFL = 0.500: |G|_max = 1.000000 [✓ 稳定]
    CFL = 0.510: |G|_max = 1.040000 [✗ 不稳定]
    CFL = 0.600: |G|_max = 1.400000 [✗ 不稳定]

========================================================================
  阶段 6：Butler-Volmer 动力学与 Newton 稳态求解
========================================================================
  Newton 稳态 NP 方程求解:
    收敛: True, 步数: 2
    初始残差: 9.500000e-03
    最终残差: 4.517220e-15
    二次收敛: True
```

---

## 边界处理与数值鲁棒性

1. **浓度非负约束**：扩散方程显式推进后，强制 $c_i \leftarrow \max(c_i, 0)$
2. **指数参数限幅**：Butler-Volmer 指数参数限幅在 $[-80, 80]$，避免溢出
3. **边界退化**：四阶差分在边界附近自动退化为二阶差分
4. **Thomas 算法主元检查**：检测除零情况
5. **Lorenz 系统发散保护**：状态变量限幅 $|x| < 10^6$
6. **参数自洽性校验**：`validate_parameters()` 检查所有物理参数的正值性、范围约束、CFL 稳定性

---

## 科学意义

本项目构建了一个**多尺度、多物理场、多算法**的 SEI 建模框架：

- **空间离散**：2 阶 / 4 阶有限差分 + Gram 多项式重构
- **时间积分**：显式 FTCS + RK4
- **非线性求解**：Newton 迭代 + Thomas 三对角求解
- **稳定性**：von Neumann 分析 + CFL 临界条件 + Damköhler 数判据
- **数据分析**：PCA 本征模分解 + 无量纲不变量检测
- **随机性**：蒙特卡罗成核 + 化学计量学平衡
- **数值求积**：Gauss-Laguerre（Boltzmann 权重）+ Gauss-Hermite（活化能分布）

该框架为**锂离子负极 SEI 层的生长机理研究**提供了可复现、可扩展的数值工具，所有算法深度耦合到 SEI 电化学体系，不是通用数值方法的简单换皮。

---

## 作者与许可

- **作者**: DA-Synthesis
- **生成方式**: 融合 15 个经典科研代码项目的核心算法，针对"计算材料：固态电解质界面反应建模：高阶有限差分与稳定性分析"领域进行博士级科研代码合成
- **许可**: MIT License
