# README_博士级合成说明.md

## 量子光源纠缠态产生全数值模拟平台 (QEPSS)

**指定科学领域**：光学工程 —— 量子光源纠缠态产生  
**合成语言**：Python 3  
**项目路径**：`/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/105_synth_project`

---

## 一、项目概述

本项目将 15 个独立的科研算法代码项目，融合重构为一个面向**准相位匹配 (QPM) 周期性极化晶体中 Type-II 自发参量下转换 (SPDC) 过程**的博士级全数值模拟平台。平台涵盖从晶体色散物理、泵浦光非线性传播、量子态刚性演化、联合光谱振幅计算、动量空间高维积分、横向模式分析、离散参数优化、级联网络耦合到纠缠度量评估的完整科学计算流程。

---

## 二、原项目到科学问题的映射

| 原项目编号 | 原核心算法 | 在合成项目中的角色 | 映射关系 |
|-----------|-----------|-------------------|---------|
| 285_digraph_adj | 有向图邻接矩阵 | `network_coupling.py` | 级联多段晶体中光子模式的有向耦合网络，邻接矩阵描述信号/闲置通道间的线性光学耦合与非线性增益转移。 |
| 013_approx_bernstein | Bernstein 多项式逼近 | `mode_analysis.py` | 用于光滑逼近泵浦光谱包络与相位匹配函数，避免 Runge 现象，保证频域边界稳定性。 |
| 124_burgers_exact | Burgers 方程精确解 (Cole-Hopf + Hermite 求积) | `pump_propagation.py` | 泵浦光在晶体中的慢变包络传播满足修正 Burgers-type 对流-扩散-非线性方程，精确解思想用于验证数值传播结果。 |
| 1179_subset_sum_backtrack | 子集和回溯算法 | `parameter_optimizer.py` | 在离散功率预算分配与工艺参数组合中，回溯搜索满足泵浦功率约束的最优子集。 |
| 743_mcnuggets_diophantine | 多元非负丢番图方程 | `parameter_optimizer.py` | 温度-长度乘积的量化约束转化为多元丢番图方程，计数并枚举所有可行的离散工艺参数组合。 |
| 931_pyramid_felippa_rule | 金字塔单元 48 点高阶求积 | `phase_space_integral.py` | 三维动量空间 (:math:`k_x, k_y, k_z`) 中的锥形积分域用金字塔形参考单元离散，48 点 Felippa 规则保证多项式精确性。 |
| 387_fem1d_bvp_quadratic | 1D 二次元有限元 BVP | `pump_propagation.py` | 泵浦包络方程在传播方向 :math:`z` 上采用二次 Lagrange 有限元与 Galerkin 投影，3 点 Gauss-Legendre 求积组装刚度矩阵与载荷向量。 |
| 1041_robertson_ode | Robertson 刚性 ODE 系统 | `quantum_evolution.py` | SPDC 三波耦合的平均场方程具有多时间尺度刚性特征（类似 Robertson 化学反应系统），采用隐式后向欧拉 + Newton-Raphson 迭代保证稳定性。 |
| 1176_subset | Gray 码子集枚举 | `parameter_optimizer.py` | 在离散参数空间搜索时，Gray 码枚举使相邻评估仅改变一个参数，减少缓存失效并加速目标函数评估。 |
| 881_polpak | 球谐函数 / Jacobi / Legendre 多项式 | `mode_analysis.py` | 纠缠光子的横向空间模式用归一化连带 Legendre 函数与球谐函数展开，Jacobi 多项式用于非对称横向模式基底。 |
| 063_backward_euler | 后向欧拉隐式 ODE 求解 | `quantum_evolution.py` | 直接作为 SPDC 刚性系统的时间积分器，每步求解非线性残差方程 :math:`R(y_{n+1})=0`。 |
| 1082_sinc | 归一化 sinc 函数 | `joint_spectrum.py` | 有限晶体长度下的相位匹配函数核心为 :math:`\text{sinc}(\Delta k L / 2\pi)`，决定联合光谱振幅 (JSA) 的频谱结构。 |
| 654_lattice_rule | Fibonacci 格点积分规则 | `phase_space_integral.py` | 周期化动量空间中的光滑被积函数采用 Fibonacci 格点规则，收敛速率优于普通 Monte Carlo。 |
| 1312_triangle_monte_carlo | 三角形域 Monte Carlo 积分 | `phase_space_integral.py` | 横向动量 :math:`(k_x, k_y)` 的允许区域为三角形波导模式截面，使用均匀随机采样计算耦合效率。 |
| 337_eros | Gauss 消元 / PLU 分解 | `linear_solver.py` | FEM 组装得到的复线性系统以及 Newton-Raphson 迭代中的 Jacobian 求解，均采用带部分主元的高斯消元与 Doolittle PLU 分解。 |

---

## 三、新增数学物理模型与核心公式

### 3.1 准相位匹配 SPDC 哈密顿量与三波耦合

在相互作用绘景中，Type-II SPDC 过程的有效哈密顿量为

```
H_int = i ħ κ(t) (a_s^† a_i^† a_p - a_s a_i a_p^†)
```

对应的平均场 c-number 方程组为

```
dy_1/dt = -γ_s/2 y_1 + κ* y_2* y_3 + f_1(t)
dy_2/dt = -γ_i/2 y_2 + κ* y_1* y_3 + f_2(t)
dy_3/dt = -γ_p/2 y_3 - κ y_1 y_2 + f_3(t)
```

其中 :math:`y_1=\langle a_s\rangle, y_2=\langle a_i\rangle, y_3=\langle a_p\rangle`，
:math:`\kappa(t)` 为时变耦合强度，:math:`f_j(t)` 为噪声驱动。

### 3.2 泵浦传播修正 Burgers-type 方程

慢变包络 :math:`A_p(z)` 满足

```
dA_p/dz + (i/2k_p) d²A_p/dz² = -α_p/2 A_p - i γ(z) |A_p|² A_p + S_SPDC(z)
```

二次元有限元离散后得到复线性系统，拆分为实部/虚部 :math:`2n \times 2n` 求解。

### 3.3 联合光谱振幅 (JSA) 与 Schmidt 分解

```
f(ω_s, ω_i) = α(ω_s + ω_i) · Φ(ω_s, ω_i)
Φ(ω_s, ω_i) = sinc(Δk(ω_s, ω_i) L / 2π)
```

相位失配 :math:`\Delta k = k_p(ω_s+ω_i) - k_s(ω_s) - k_i(ω_i) - 2π/Λ`。

Schmidt 分解：

```
f(ω_s, ω_i) = Σ_n √λ_n u_n(ω_s) v_n(ω_i)
```

有效 Schmidt 数 :math:`K = 1 / \sum_n \lambda_n^2`，纯度 :math:`\mathcal{P} = 1/K`。

### 3.4 纠缠度量

- **Concurrence**：:math:`C = \sqrt{2(1 - \mathcal{P})}`
- **纠缠熵**：:math:`S = -\sum_n \lambda_n \log_2 \lambda_n`
- **HOM 可见度**：:math:`V = (R_{\max} - R_{\min}) / (R_{\max} + R_{\min})`
- **CHSH 参数**：:math:`S_{\text{CHSH}} = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|`
- **态保真度**：:math:`F = |\langle \Psi_{\text{target}} | \Psi \rangle|^2`

### 3.5 高维动量空间积分

- **Fibonacci 格点规则**：二维周期积分，点数 :math:`F_m`，收敛 :math:`O(N^{-1})`。
- **Felippa 金字塔 48 点规则**：:math:`\int_{\text{pyramid}} f(x,y,z) \, dV \approx \sum_{i=1}^{48} w_i f(x_i,y_i,z_i)`，精确到 15 次多项式。
- **三角形 Monte Carlo**：:math:`\int_T f \, dA \approx |T| \cdot \frac{1}{N} \sum_{j=1}^N f(x_j,y_j)`。

### 3.6 离散参数优化

- **丢番图方程**：:math:`a_1 x_1 + \dots + a_d x_d = b`，枚举所有非负整数解。
- **子集和回溯**：:math:`\sum_{i \in I} v_i = S`，深度优先搜索所有满足约束的子集。
- **Gray 码枚举**：相邻子集仅改变一个元素，状态转移 :math:`a_{\text{next}} = a \oplus e_{i_{\text{add}}}`。

---

## 四、文件结构与实现路径

### 4.1 模块清单（共 10 个 .py 文件 + main.py）

| 文件名 | 功能 | 核心算法来源 |
|-------|------|-------------|
| `linear_solver.py` | 带部分主元高斯消元、PLU 分解、条件数估计 | 337_eros |
| `mode_analysis.py` | 球谐函数、Jacobi 多项式、Bernstein 逼近 | 881_polpak, 013_approx_bernstein |
| `pump_propagation.py` | 1D 二次元 FEM 求解泵浦传播、Burgers 精确解 | 387_fem1d_bvp_quadratic, 124_burgers_exact |
| `quantum_evolution.py` | SPDC 刚性 ODE 后向欧拉 + Newton 迭代 | 1041_robertson_ode, 063_backward_euler |
| `joint_spectrum.py` | JSA 计算、相位匹配、sinc 函数、Schmidt 分解 | 1082_sinc |
| `phase_space_integral.py` | 格点规则、金字塔求积、三角形 Monte Carlo | 931_pyramid_felippa_rule, 654_lattice_rule, 1312_triangle_monte_carlo |
| `parameter_optimizer.py` | 丢番图求解、子集和回溯、Gray 码枚举、离散优化 | 743_mcnuggets_diophantine, 1179_subset_sum_backtrack, 1176_subset |
| `network_coupling.py` | 级联晶体有向图邻接矩阵、转移矩阵、传递闭包 | 285_digraph_adj |
| `entanglement_metrics.py` | Concurrence、纠缠熵、HOM、保真度、CHSH | 新增 |
| `main.py` | 统一入口，零参数运行，完整流程调度 | 所有模块 |

### 4.2 边界处理与数值鲁棒性

- **维度检查**：所有模块对输入矩阵/向量维度进行严格校验，非方阵、非奇数节点、负长度等立即抛出 `ValueError`。
- **奇异检测**：高斯消元与 PLU 分解中设置 `tol = eps · n · max|A_{ij}|`，无主元时明确报错。
- **非负约束**：光子数、概率、Schmidt 系数等物理量通过 `np.clip` 与 `np.maximum` 保证非负。
- **复系统实部/虚部分离**：FEM 与 Newton-Raphson 中的复线性系统统一拆分为 :math:`2n \times 2n` 实系统求解，避免复数运算库的数值不稳定性。
- **刚性稳定性**：后向 Euler 的隐式性质保证无论步长多大，数值解均稳定（L-stable）。

---

## 五、合成后的项目能够解决的科学问题

1. **准相位匹配晶体设计优化**：在离散工艺窗口内，搜索使纠缠纯度最大的极化周期、晶体长度与工作温度组合。
2. **联合光谱振幅预测**：基于 Sellmeier 色散模型，数值计算 Type-II SPDC 的 JSA，评估 Schmidt 数与态纯度。
3. **泵浦光非线性传播分析**：考虑周期性极化结构、自相位调制与吸收损耗，预测晶体内部泵浦场分布。
4. **多段级联光源性能评估**：通过有向图网络模型，分析多段晶体级联后的光子数演化与模式传递特性。
5. **纠缠质量全面表征**：计算 Concurrence、纠缠熵、HOM 干涉可见度、CHSH 参数与态保真度，判断光源是否可用于量子通信与量子计算。
6. **高维动量空间耦合效率积分**：使用格点规则、高阶求积与 Monte Carlo 三种方法交叉验证，确保数值可靠性。

---

## 六、运行方式

### 环境要求
- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7

### 安装依赖
```bash
pip install numpy scipy
```

### 运行
```bash
cd /mnt/data/zpy/sci-swe/source code/Synthesis-project-python/105_synth_project
python main.py
```

程序将自动执行以下 10 个科学计算阶段，无需任何输入参数：
1. 晶体色散模型建立
2. 泵浦光 FEM 非线性传播
3. SPDC 量子态刚性 ODE 演化
4. 联合光谱振幅与 Schmidt 分解
5. 动量空间高维积分（三种方法交叉验证）
6. 横向模式与光谱包络分析
7. 离散参数优化（丢番图 + 回溯 + Gray 码）
8. 级联晶体网络耦合
9. 纠缠度量综合评估
10. 线性求解器数值验证

运行结束后，终端输出所有关键物理量的数值结果与综合摘要。

---

## 七、质量检查清单

- [x] 原目录未被修改
- [x] 合成后的项目为 Python 语言
- [x] 新目录完整包含合成后的项目
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] 15 个输入项目均已真实融入，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
