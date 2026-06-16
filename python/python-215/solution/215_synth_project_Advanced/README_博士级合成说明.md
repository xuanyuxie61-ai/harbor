# 催化反应器多目标拓扑优化 — 博士级科学计算合成项目

## PROJECT_215 合成说明

**科学领域**: 数学优化 — 多目标优化与 Pareto 前沿
**应用问题**: 一维催化反应器的功能梯度材料拓扑优化
**难度级别**: 博士级前沿科学计算

---

## 一、科学问题描述

本项目解决的是一个 **多物理场耦合催化反应器的拓扑优化** 问题。
设计目标是确定一维反应器沿流向的材料密度分布 ρ(x) ∈ [0,1]ⁿ，
在体积分数约束下，**同时优化三个相互竞争的目标**：

1. **f₁(ρ)**: 最小化结构柔度 C = **uᵀKu** （最大化刚度）
2. **f₂(ρ)**: 最小化热应力方差 Var[σ_th] （均匀温度分布）
3. **f₃(ρ)**: 最大化生化转化率 η = 1 − c_out/c_in

### 多目标优化的核心数学

**Pareto 支配**:
$$\mathbf{x} \prec \mathbf{y} \iff \forall i: f_i(\mathbf{x}) \le f_i(\mathbf{y}) \wedge \exists j: f_j(\mathbf{x}) < f_j(\mathbf{y})$$

**非支配排序**: 将种群按 Pareto 支配关系分层 F₁, F₂, ...

**拥挤距离**:
$$CD(i) = \sum_{k=1}^{M} \frac{f_k^{(i+1)} - f_k^{(i-1)}}{f_k^{\max} - f_k^{\min}}$$

**超体积指标 (HV)** — 综合收敛性与多样性:
$$HV = \sum_{i=0}^{|PF|-1} (f_1^{ref} - f_1^{(i)}) \cdot (f_2^{(i)} - f_2^{(i+1)})$$

---

## 二、物理模型与核心公式

### 2.1 SIMP 材料插值（惩罚模型）

$$E(\rho) = E_{\min} + \rho^p (E_0 - E_{\min}), \quad p = 3$$
$$\kappa(\rho) = \kappa_{\min} + \rho^p (\kappa_0 - \kappa_{\min})$$
$$D(\rho) = D_{\min} + \rho^p (D_0 - D_{\min})$$

### 2.2 稳态热传导（PDE 约束）

$$-\nabla \cdot (\kappa(\rho) \nabla T) = Q(\rho, T) \quad \text{in } \Omega$$

弱形式 (Galerkin):
$$\int_\Omega \kappa(\rho) \nabla T \cdot \nabla w \, dV = \int_\Omega Q w \, dV + \int_{\Gamma_N} q_N w \, dS$$

### 2.3 生化反应动力学（Michaelis-Menten + Arrhenius）

$$r(c, T) = k_0 \exp\left(-\frac{E_a}{R_g T}\right) \cdot \frac{c}{K_m + c}$$

化学计量矩阵:
$$S = \begin{bmatrix} -a & 0 \\ -b & 0 \\ 1 & -1 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{r} = \begin{bmatrix} r_{\max} \frac{c}{K_c+c} \frac{n}{K_n+n} p \\ \varepsilon p \end{bmatrix}$$

### 2.4 热应力

$$\sigma_{th} = E(\rho) \alpha(\rho) (T - T_{ref})$$

### 2.5 结构柔度

$$C(\rho) = \mathbf{u}^T K(\rho) \mathbf{u}$$

### 2.6 Fresnel 波场品质因子

$$U(x') = \int A(x) e^{i\varphi(x)} e^{i\pi(x'-x)^2/(\lambda z)} dx, \quad \varphi = \pi\rho(x)$$
$$Q = \frac{I_{focus}}{I_{total}}$$

### 2.7 高阶数值积分

**金字塔域** (Felippa 2004):
$$\int_\Omega x^a y^b z^c dV = \frac{2}{a+1} \cdot \frac{2}{b+1} \cdot B(c+1, a+b+3)$$

**圆盘域**:
$$\int_{D_R} x^m y^n dA = \frac{R^{m+n+2}}{m+n+2} \cdot 2B\left(\frac{m+1}{2}, \frac{n+1}{2}\right)$$

### 2.8 随机鲁棒性（OU 过程）

$$dx = \theta(\mu - x) dt + \sigma dW$$
$$E[x(t)] = \mu + (x_0 - \mu) e^{-\theta t}, \quad \text{Var}[x(t)] = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$$

鲁棒目标:
$$F_{robust} = \mu_F + \beta \sigma_F$$

### 2.9 gPC 多项式混沌展开

$$F(\xi) \approx \sum_{k=0}^p c_k He_k(\xi), \quad c_k = \frac{1}{k!} E[F(\xi) He_k(\xi)]$$

### 2.10 保真度 Witness（类比量子保真度）

$$F_w = \prod_{i=1}^{M} \left(1 - \frac{f_i - f_i^*}{f_i^{nadir} - f_i^*}\right)$$

---

## 三、种子项目到科学问题的映射

| 种子项目 | 核心算法 | 在合成项目中的角色 |
|---------|---------|-------------------|
| **931_pyramid_felippa_rule** | 金字塔域 Felippa 高阶求积 | `quadrature_engine.py`: 3D 域数值积分 |
| **946_quad2d** | 2D 矩形域 Gauss 求积 | `quadrature_engine.py`: 2D Gauss-Legendre 张量积 |
| **294_disk_integrals** | 圆盘域单项式解析积分 | `quadrature_engine.py`: 圆盘域积分与采样 |
| **1221_WillemWybo_SGF_formalism** | Spike-generation formalism | `coupled_pde_system.py`: 催化剂退化脉冲建模 |
| **910_prime** | Eratosthenes 素数筛法 | `cvt_design_sampler.py`: Halton 低差异序列基底 |
| **839_ornstein_uhlenbeck** | OU 过程 Euler-Maruyama 求解 | `stochastic_robustness.py` + `adaptive_rk_integrator.py`: 材料不确定性 |
| **091_biochemical_nonlinear_ode** | 生化 ODE (Michaelis-Menten) | `coupled_pde_system.py`: 反应动力学核心 |
| **737_matrix_analyze** | 矩阵结构分析 (对称/带状/Cholesky) | `coupled_pde_system.py`: FEM 刚度矩阵诊断 |
| **906_pram_view** | PRAM 并行分块配置 | `pram_parallel.py`: 种群并行评估协调 |
| **972_r8but** | Butcher 表 (RK 系数定义) | `adaptive_rk_integrator.py`: RK4/Dormand-Prince RK45 |
| **909_predator_prey_ode_period** | 捕食-被捕食竞争动力学 | `pareto_core.py`: 多目标竞争的 Pareto 支配关系 |
| **385_fem1d_approximate** | 一维有限元组装与加权逼近 | `coupled_pde_system.py`: 1D FEM 热传导求解 |
| **246_cvt_1d_sampling** | Lloyd 算法采样版 CVT | `cvt_design_sampler.py`: 设计空间最优采样 |
| **448_fresnel** | Fresnel 余弦/正弦积分 | `cvt_design_sampler.py`: 波场相位编码目标 |
| **1109_marekgluza_Fidelity_witnesses** | 量子保真度 witness | `stochastic_robustness.py`: Pareto 前沿质量度量 |

---

## 四、项目结构与文件说明

```
215_synth_project_Advanced/
├── main.py                        # 统一入口 (零参数可运行)
├── pareto_core.py                 # Pareto 支配、非支配排序、拥挤距离、HV
├── quadrature_engine.py           # 高阶数值积分 (金字塔/圆盘/2D Gauss)
├── adaptive_rk_integrator.py      # Butcher 表 + 自适应 RK45 + OU 过程
├── coupled_pde_system.py          # 热-化学-力学 PDE 耦合 + SIMP 材料
├── stochastic_robustness.py       # 随机鲁棒性 + gPC + 保真度 witness
├── cvt_design_sampler.py          # CVT 采样 + Halton 序列 + Fresnel 积分
├── objective_functions.py         # 三目标评估器 (柔度/热应力/转化率)
├── nsga2_optimizer.py             # NSGA-II (SBX 交叉 + 多项式变异)
├── pram_parallel.py               # PRAM 并行评估协调
└── README_博士级合成说明.md        # 本文档
```

共计 **10 个 Python 文件**，每个文件都对应 1-3 个种子项目的核心算法。

---

## 五、运行方法

### 零参数直接运行

```bash
cd 215_synth_project_Advanced
python main.py
```

### 输出内容

1. **Phase 1**: 基础数值模块验证（积分精度、ODE 精度、矩阵分析、CVT 收敛、素数/Halton、Fresnel）
2. **Phase 2**: 多目标问题构建与单点评估
3. **Phase 3**: NSGA-II 进化求解 Pareto 前沿
4. **Phase 4**: Pareto 前沿质量分析（间距指标、超体积、保真度）
5. **Phase 5**: 鲁棒性验证与 gPC 不确定性展开

### 典型输出示例

```
  Pareto 前沿大小: 30
  理想点 (utopia): [ 1.0014  0.1806 -1.    ]
  Nadir 点:        [ 1.0776  0.3626 -1.    ]
  平均保真度:      0.2479
  间距指标:        0.0022
  最优转化率:      1.0000
  总计算时间: ~2 秒
```

---

## 六、算法独特性说明

本项目的核心方法论与通用多目标优化的区别：

1. **PDE 约束的多目标拓扑优化**: 不是简单的目标函数黑箱优化，每个设计评估都要求解一组耦合 PDE（热传导 + 化学反应 + 弹性力学）。

2. **三物理场竞争**: 三个目标分别源于力学、热学、化学三个物理场，彼此存在根本性冲突（提高刚度需要实心结构，均匀温度需要高导热路径，高转化率需要催化材料分布），这构成了真正的 Pareto 前沿。

3. **随机鲁棒性**: 使用 Ornstein-Uhlenbeck 过程建模制造误差，通过 gPC 展开和 Monte Carlo 采样量化不确定性对 Pareto 前沿的影响。

4. **跨尺度科学公式**: 从微观的 Michaelis-Menten 反应动力学、Arrhenius 温度依赖，到宏观的 FEM 结构分析，再到 Fresnel 波场品质因子，形成完整的跨尺度科学计算链条。

5. **独特算法组合**: NSGA-II 的非支配排序映射捕食-被捕食竞争动态、CVT 最优采样映射 Voronoi 几何、Halton 低差异序列映射素数理论、PRAM 并行分块映射并行计算模型 — 这种组合在已合成的项目中是独一无二的。

---

## 七、数值鲁棒性设计

- **边界处理**: 密度变量裁剪至 [0.01, 0.99] 避免 SIMP 奇异性
- **矩阵病态**: 通过条件数监测 + Cholesky 分解验证正定性
- **ODE 刚性**: 自适应 RK45 步长控制，局部误差容限 10⁻⁶~10⁻⁸
- **浮点容差**: Pareto 支配比较使用 ε = 10⁻¹² 防止抖动
- **除零保护**: 所有归一化操作分母添加 ε
- **非负约束**: 浓度、温度、密度等物理量强制非负

---

## 八、可扩展方向

1. **2D/3D 扩展**: 将 1D FEM 升级为 2D 四节点四边形单元
2. **更多目标**: 加入制造约束、频率响应等目标形成 4-5 目标优化
3. **代理模型**: 使用 Kriging/RBF 代理加速高维优化
4. **决策支持**: 基于 TOPSIS/熵权法从 Pareto 前沿中选出最终设计
5. **实验验证**: 将计算得到的最优密度分布与 3D 打印样品对比

---

*本项目由 Sci-Project-Synthesis (DA workflow) 自动生成*
*融合 15 个种子项目的核心算法，围绕多目标优化与 Pareto 前沿构建*
