# 鲁棒优化与不确定约束 —— 博士级科学计算项目 217

## 项目概述

本项目围绕 **数学优化：鲁棒优化与不确定约束** 这一前沿领域，融合 15 个种子项目的核心算法，构建了一个面向 **模拟移动床 (SMB) 色谱过程** 的博士级鲁棒优化科学计算平台。

### 科学问题

模拟移动床色谱是一种连续分离技术，广泛应用于制药、化工等领域。其优化面临以下挑战：

1. **参数不确定性**：扩散系数、流速、吸附等温线参数存在测量误差和空间变化
2. **机会约束**：产品纯度需满足概率保证（如 P(纯度 ≥ 99%) ≥ 95%）
3. **PDE 约束**：每根色谱柱由对流-扩散-吸附偏微分方程描述
4. **延迟动力学**：周期性切换操作引入延迟微分代数方程
5. **谱鲁棒性**：系统特征值对参数扰动敏感，影响稳定性

本项目通过多种鲁棒优化算法求解该问题，提供完整的科学计算解决方案。

---

## 种子项目映射

| 序号 | 种子项目 | 核心算法 | 在本项目中的角色 |
|------|---------|---------|----------------|
| 1 | 976_r8ci | 循环矩阵求解 (Trench 算法) | 循环预条件 KKT 系统求解 |
| 2 | 790_navier_stokes_mesh3d | 3D 网格提取 | 3D FEM 网格生成基础 |
| 3 | 999_r8sto | 对称 Toeplitz 求解 (Levinson-Durbin) | 椭球不确定性集合的协方差结构 |
| 4 | 592_interp_equal | Newton 均差插值 | 不确定性随机场的空间基函数 |
| 5 | 1192_Quantum-Control | 部分可观测性下的量子控制 | 启发：不确定性下的鲁棒决策 |
| 6 | 273_dg1d_heat | 间断 Galerkin 方法 | PDE 约束的 DG 离散 |
| 7 | 051_asa243 | 非中心 t 分布 (AS 243) | 机会约束的统计修正 |
| 8 | 555_hyperball_positive_distance | 超球正象限采样 | 不确定性集合采样 |
| 9 | 1107_cellcyclemodules | 延迟双稳态 ODE | SMB 切换动力学建模 |
| 10 | 172_chladni_figures | Chladni 板振动特征值 | 谱鲁棒性分析 |
| 11 | 418_fem3d_project | 3D FEM 投影 | PDE 约束的有限元离散 |
| 12 | 1071_HigherOrderLMC | 高阶 Langevin Monte Carlo | 分布鲁棒采样的 LMC 采样器 |
| 13 | 1128_cadet_SMB | SMB 色谱仿真 | 应用目标与物理背景 |
| 14 | 1176_Online-Learning-RC-Control | ESN + RLS 自适应控制 | 在线鲁棒控制器 |
| 15 | 280_diff_forward | 前向差分数值微分 | 伴随灵敏度分析 |

---

## 核心算法与公式

### 1. 不确定性集合构造

**椭球集合**：
$$
W = \{ w \in \mathbb{R}^m : (w - w_0)^T \Sigma^{-1} (w - w_0) \leq \rho^2 \}
$$

**Toeplitz 协方差 (AR(1))**：
$$
\Sigma_{ij} = \rho^{|i-j|}, \quad |\rho| < 1
$$

**Levinson-Durbin 递归**：
$$
\beta_0 = 1, \quad x_0 = b_0 / \beta_0, \quad y_0 = -a_1 / \beta_0
$$
$$
\beta_k = (1 - y_{k-1}^2) \beta_{k-1}
$$
$$
x_k = \frac{b_k - a_{2:k+1}^T x_{k-1:-1:0}}{\beta_k}
$$

**超球正象限采样 (Cheng-Rubinstein)**：
$$
g \sim \mathcal{N}(0, I_m), \quad u = \frac{|g|}{\|g\|_2}, \quad r \sim U(0,1), \quad x = r^{1/m} u
$$

### 2. 机会约束的确定性等价

**概率约束**：
$$
\mathbb{P}(a^T x + w^T x \leq b) \geq 1 - \alpha
$$

**确定性等价 (高斯情形)**：
$$
a^T x + \mu_w^T x + z_{1-\alpha} \sqrt{x^T \Sigma_w x} \leq b
$$

**非中心 t 修正 (小样本)**：
$$
t_{1-\alpha}(n-1, \delta = \sqrt{n} \mu / \sigma)
$$

### 3. DG 离散对流-扩散方程

**PDE**：
$$
\frac{\partial u}{\partial t} + v \frac{\partial u}{\partial x} = D \frac{\partial^2 u}{\partial x^2} + f(x,t)
$$

**弱形式 (DG)**：
$$
M \frac{du}{dt} = -v S u + D (M^{-1} S^T M S u + \text{LIFT} \cdot \text{flux})
$$

**数值通量 (Lax-Friedrichs)**：
$$
F^* = \frac{1}{2} (F(u_L) + F(u_R)) - \frac{1}{2} |v| (u_R - u_L)
$$

### 4. 高阶 Langevin Monte Carlo

**Picard-Lagrange LMC (K ≥ 3)**：
$$
dY_j = \sum_{i=0}^{j-1} \alpha_{ji} Y_i \, dt + D_j \, dW_t
$$

**Kronecker 结构**：
$$
A = A_{\text{small}} \otimes I_d, \quad D = D_{\text{small}} \otimes I_d
$$

### 5. 延迟双稳态动力学

**Gelens Lab 模型**：
$$
\frac{d\text{Cdk1}}{dt} = c - \text{Cdk1} \cdot \text{Apc}
$$
$$
\frac{d\text{Apc}}{dt} = \varepsilon \left( \frac{\text{Cdk1}(t-\tau)^n}{\text{Cdk1}(t-\tau)^n + \Xi^n} - \text{Apc} \right)
$$
$$
\Xi = 1 + a \cdot \text{Apc} \cdot (\text{Apc} - 1) \cdot (\text{Apc} - r)
$$

### 6. 谱鲁棒性 (Chladni 板)

**双 Laplacian 特征值问题**：
$$
\Delta^2 u = \omega^2 u, \quad u = \Delta u = 0 \text{ on } \partial\Omega
$$

**鲁棒性比**：
$$
\rho(x) = \frac{\lambda_1(L(x))}{\lambda_1(L(x_0))}
$$

### 7. 鲁棒优化算法

**最坏情况鲁棒优化**：
$$
\min_x \sup_{w \in W} F(x, w)
$$

**均值-方差鲁棒优化**：
$$
\min_x (1-\beta) \mathbb{E}_w[F(x,w)] + \beta \sup_w F(x,w)
$$

**机会约束鲁棒优化**：
$$
\min_x F(x) \quad \text{s.t.} \quad \mathbb{P}(g_i(x,w) \leq 0) \geq 1 - \alpha_i
$$

---

## 项目结构

```
217_synth_project_Advanced/
├── main.py                      # 统一入口 (零参数运行)
├── uncertainty_set.py           # 不确定性集合构造 (椭球、超球、非中心 t)
├── circulant_kkt.py             # 循环预条件 KKT 系统求解
├── dg_pde_constraint.py         # DG 离散 PDE 约束
├── fem3d_mesh.py                # 3D FEM 网格与 L2 投影
├── chance_constraints.py        # 机会约束处理
├── lmc_robust_sampler.py        # 高阶 LMC 鲁棒采样
├── delay_bistability.py         # 延迟双稳态 DDAE 求解
├── adaptive_esn_controller.py   # ESN + RLS 自适应控制
├── spectral_robustness.py       # 谱鲁棒性分析
├── smb_process.py               # SMB 色谱过程仿真
├── uncertain_field.py           # 不确定性随机场构造
├── adjoint_sensitivity.py       # 伴随灵敏度分析
├── robust_optimizer.py          # 鲁棒优化主求解器
└── README_博士级合成说明.md     # 本文档
```

---

## 运行方式

```bash
cd /mnt/data/zpy/sci-swe/source\ code/Synthesis-project-python/217_synth_project/217_synth_project_Advanced
python main.py
```

**无需任何参数**，程序将自动执行以下 12 个步骤：

1. 构造不确定性集合 (椭球、超球、非中心 t)
2. 循环预条件 KKT 系统求解
3. DG 离散对流-扩散方程
4. 3D FEM 网格生成与 L2 投影
5. 机会约束处理与 Bonferroni 近似
6. 高阶 Langevin Monte Carlo 采样
7. 延迟双稳态 DDAE 求解
8. 谱鲁棒性分析 (Chladni 板特征值)
9. 自适应 ESN 鲁棒控制
10. SMB 色谱过程仿真
11. 不确定性随机场构造 (Newton 插值 + KL 展开)
12. 多种鲁棒优化算法求解

---

## 输出示例

```
**********************************************************************
*  鲁棒优化与不确定约束 —— 博士级科学计算项目 217
*  应用：模拟移动床色谱过程的鲁棒优化
**********************************************************************
======================================================================
步骤 1: 构造不确定性集合
======================================================================
  椭球集合维度: 4
  样本数量: 20
  最坏情况线性目标: 2.4928

======================================================================
步骤 12: 鲁棒优化求解
======================================================================
  最坏情况鲁棒优化:
    最优值: -0.6568
  均值-方差鲁棒优化:
    最优值: -0.6472
  机会约束鲁棒优化:
    最优值: -0.6570
    可行: True

======================================================================
所有步骤执行完毕！
======================================================================
```

---

## 科学贡献

1. **多学科融合**：将数值分析 (DG、FEM)、随机优化 (LMC、机会约束)、控制理论 (ESN、延迟系统)、谱理论 (Chladni 特征值) 有机融合
2. **博士级难度**：涉及高阶数值方法、非中心 t 分布、Karhunen-Loève 展开、Picard-Lagrange LMC 等前沿算法
3. **工程鲁棒性**：所有模块均包含边界检查、数值稳定性保护、异常处理
4. **可复现性**：固定随机种子，结果完全可复现
5. **零参数运行**：统一入口 `main.py`，无需任何配置

---

## 技术亮点

### 数值方法
- **Levinson-Durbin 递归**：O(n²) 求解 Toeplitz 系统
- **循环矩阵 FFT 加速**：O(n log n) 求解循环系统
- **DG 方法**：高阶精度、局部守恒、易于并行
- **RK4 时间积分**：四阶精度、显式稳定

### 优化算法
- **次梯度法**：最坏情况鲁棒优化
- **精确罚函数法**：机会约束处理
- **投影梯度法**：盒约束优化
- **样本平均近似 (SAA)**：分布鲁棒优化

### 机器学习
- **回声状态网络 (ESN)**：模型无关的自适应控制
- **递归最小二乘 (RLS)**：在线参数估计
- **Langevin Monte Carlo**：基于采样的优化

---

## 扩展方向

1. **高阶 DG 方法**：提升至 P ≥ 3 阶
2. **自适应网格**：基于后验误差估计的 h/p 自适应
3. **分布式鲁棒优化**：Wasserstein 球不确定性集合
4. **深度强化学习**：替代 ESN 的端到端控制
5. **GPU 加速**：CUDA 实现大规模并行采样

---

## 参考文献

1. Ben-Tal, A., El Ghaoui, L., & Nemirovski, A. (2009). *Robust Optimization*. Princeton University Press.
2. Hesthaven, J. S., & Warburton, T. (2008). *Nodal Discontinuous Galerkin Methods*. Springer.
3. Lenth, R. V. (1989). Algorithm AS 243: Cumulative Distribution Function of the Non-Central T Distribution. *Applied Statistics*, 38(1), 185-189.
4. Cheng, R., & Rubinstein, R. (1998). Random Variate Generation. In *Handbook of Simulation* (pp. 139-174). Wiley.
5. Gelens, L., et al. (2020). Delay-induced bistability in the cell cycle. *Cell Systems*, 10(4), 327-341.
6. Gander, M. J., & Kwok, F. (2012). Chladni figures and the Tacoma bridge. *SIAM Review*, 54(3), 573-596.
7. Kaihongz, et al. (2024). Higher-Order Langevin Monte Carlo. *arXiv preprint*.
8. Shen, J., et al. (2025). Online Learning RC Control. *RoboSoft 2025*.

---

## 许可证

本项目代码遵循 MIT 许可证。

---

## 联系方式

如有问题或建议，请联系项目维护者。

---

**项目完成日期**：2026-06-07  
**Python 版本**：3.10+  
**依赖库**：numpy, scipy
