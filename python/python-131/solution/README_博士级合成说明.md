# 浆态床气泡柱反应器 CFD-PBM 耦合模拟系统 — 博士级合成说明

## 一、项目概述

本项目将 **15 个独立科研代码项目** 的核心算法融合重构，面向**化学工程：多相反应器 CFD 模拟**领域，构建了一个前沿博士级计算项目——**浆态床气泡柱反应器（Slurry Bubble Column Reactor, SBCR）CFD-PBM 耦合模拟器**。

该反应器用于 Fischer-Tropsch 合成等重质烃生产过程，涉及气-液-固三相流动、群体平衡方程（PBE）、反应动力学与催化剂优化的强耦合问题。项目包含 11 个 Python 模块，统一入口 `main.py` 零参数可运行。

---

## 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 合成后角色 |
|:---:|--------|----------|-----------|
| 1 | `807_nonlin_fixed_point` | 定点迭代 / Newton 法 | `nonlinear_solver.py`：反应器气含率-液速耦合代数方程组的非线性求解 |
| 2 | `665_legendre_rule` | Gauss-Legendre 积分 / Jacobi 矩阵 / IMTQLX | `spectral_quadrature.py`：PBE 矩计算、反应速率体积平均的高精度数值积分 |
| 3 | `034_asa082` | 正交矩阵行列式 detq | `numerical_linear_algebra.py`：网格坐标变换 Jacobian 的正交性校验 |
| 4 | `879_poisson_simulation` | Poisson 过程 / 指数等待时间 | `population_balance.py`：气泡成核事件的随机模拟 |
| 5 | `623_knapsack_brute` | 0-1 背包暴力枚举 | `catalyst_optimization.py`：反应器各段催化剂装载量的最优分配 |
| 6 | `762_mhd_exact` | Hartmann MHD 精确解 | `momentum_equations.py`：动量方程数值验证基准 + 相间力模型 |
| 7 | `1362_truncated_normal_sparse_grid` | Smolyak 稀疏网格积分 | `spectral_quadrature.py`：高维 PBE 矩空间积分 |
| 8 | `117_brc_data` | 正态分布随机数据生成 | `stochastic_inlet.py`：入口温度/流量/浓度的随机扰动生成 |
| 9 | `163_chebyshev_series` | Clenshaw 算法求 Chebyshev 级数 | `spectral_quadrature.py`：温度/浓度分布的谱逼近 |
| 10 | `446_fractal_coastline` |  coastline 分形扰动 | `reactor_mesh.py`：反应器壁面/分布板边界的粗糙度几何建模 |
| 11 | `004_alpert_rule` | Alpert 混合 Gauss-Trapezoidal 积分 | `spectral_quadrature.py`：PBE 破裂核在 V→0 处奇异性的端点修正积分 |
| 12 | `844_pagerank` | 幂法 / Google 矩阵特征向量 | `numerical_linear_algebra.py`：稳态物种浓度场的迭代收敛 |
| 13 | `289_diophantine_nd` | 有界非负 Diophantine 整数解 | `catalyst_optimization.py`：离散催化剂颗粒数的最优整数规划 |
| 14 | `135_calpak` | Julian Day / Gregorian 日期转换 | `reactor_operations.py`：反应器批次操作时间线与运行日历 |
| 15 | `753_mesh_boundary` | 多边形网格边界提取 | `reactor_mesh.py`：圆柱结构化网格的边界段提取与序列化 |

---

## 三、核心科学公式体系

### 3.1 两流体动量方程（Euler-Euler 模型）

气相：
```
∂(α_g ρ_g u_g)/∂t + ∇·(α_g ρ_g u_g u_g) = -α_g ∇p + ∇·τ_g + M_gl + α_g ρ_g g
```

液相：
```
∂(α_l ρ_l u_l)/∂t + ∇·(α_l ρ_l u_l u_l) = -α_l ∇p + ∇·τ_l - M_gl + α_l ρ_l g
```

相间动量交换：
```
M_gl = M_D + M_VM + M_L
M_D = (3/4) α_g (ρ_l / d_b) C_D |u_g - u_l| (u_g - u_l)
M_VM = C_VM α_g ρ_l (D u_g/Dt - D u_l/Dt)
M_L = C_L α_g ρ_l (u_g - u_l) × (∇ × u_l)
```

Schiller-Naumann 阻力系数：
```
C_D = (24/Re_p)(1 + 0.15 Re_p^{0.687})   (Re_p < 1000)
C_D = 0.44                               (Re_p ≥ 1000)
Re_p = ρ_l |u_g - u_l| d_b / μ_l
```

### 3.2 Zuber-Findlay 漂移流模型

```
u_g/α_g = C_0 · j + u_∞
j = α_g · u_g + (1 - α_g) · u_l
```

其中 `C_0 = 1.2` 为分布系数，`u_∞ = 0.23 m/s` 为气泡终端上升速度。

### 3.3 群体平衡方程（PBE）

```
∂f(V; x,t)/∂t + ∇·[u_b(V) f] = B_B(V) - D_B(V) + B_C(V) - D_C(V)
```

破裂项：
```
B_B(V) = ∫_V^∞ ν(V') g(V') f(V') dV'
D_B(V) = g(V) f(V)
g(V) = C_B √(σ / (ρ_l d_eq³)),   d_eq = (6V/π)^{1/3}
```

聚并项：
```
B_C(V) = 1/2 ∫_0^V Q(V-V', V') f(V-V') f(V') dV'
D_C(V) = f(V) ∫_0^∞ Q(V, V') f(V') dV'
Q(V_i, V_j) = ω(V_i, V_j) · h(V_i, V_j)
```

Prince-Blanch 聚并效率：
```
h = exp(-t_contact / t_drainage)
t_contact = (r_i + r_j)^{2/3} / ε^{1/3}
t_drainage = √(ρ_l r_eq³ / (16σ)) · ln(h_0 / h_f)
```

### 3.4 矩方法（QMOM）

```
m_k = ∫_0^∞ V^k f(V) dV
∂m_k/∂t + ∇·(u_b m_k) = S̄_k
```

 Wheeler 算法由矩构造 Gaussian 积分节点 `{ξ_i}` 与权重 `{w_i}`：
```
S̄_k ≈ Σ_i w_i [B_B(ξ_i) - D_B(ξ_i) + B_C(ξ_i) - D_C(ξ_i)] ξ_i^k
```

### 3.5 Fischer-Tropsch 反应动力学

简化双曲型速率：
```
r_FT = η_eff · k_FT · exp(-E_a / RT) · C_CO
```

其中 `η_eff ≈ 0.08` 为催化剂有效因子（Thiele 模数与内扩散限制）。

### 3.6 能量平衡（稳态一维）

```
ρ_m C_p (u_l · dT/dz) = ∇·(k_eff ∇T) + (-ΔH) r_FT - q_cool
q_cool = h_cool (T - T_in)
```

### 3.7 Hartmann 磁流体力学基准解

用于验证动量方程数值实现的解析基准：
```
u(y) = (G Re / Ha) / tanh(Ha) · [1 - cosh(Ha·y) / cosh(Ha)]
b(y) = (G/S) · [sinh(Ha·y) / sinh(Ha) - y]
p(x,y) = -G·x - 0.5·S·b(y)²
```

验证残差：
```
u'' + Re·S·b' = -G·Re
b'' + Rm·u' = 0
```

### 3.8 催化剂优化模型

连续型背包：
```
max Σ v_i x_i
s.t. Σ w_i x_i ≤ K,   x_i ∈ {0,1}
v_i = ΔX_i · Y_{C5+} · ρ_wax
ΔX_i = 1 - exp(-k_i W_i / Q)
k_i = k_0 exp(-E_a / (R T_i))
```

离散型 Diophantine：
```
a_1 x_1 + a_2 x_2 + ... + a_n x_n = b
0 ≤ x_i ≤ m_i,   x_i ∈ ℤ
```

### 3.9 谱方法公式

Gauss-Legendre 积分：
```
∫_a^b f(x) dx ≈ Σ_{i=1}^n w_i f(x_i)
w_i = 2 / [(1-x_i²)(P_n'(x_i))²]
```

Chebyshev 级数（Clenshaw 递推）：
```
f(x) = Σ_{k=0}^{N-1} c_k T_k(x)
b_0 = c_N, b_1 = 0, b_2 = 0
b_k = c_k - b_{k+2} + 2x b_{k+1}   (k = N-1,...,0)
f(x) = (b_0 - b_2)/2
```

稀疏网格（Smolyak）：
```
A(q,d) = Σ_{q-d+1 ≤ |l|_1 ≤ q} (-1)^{q-|l|_1} C(d-1, q-|l|_1) (U^{l_1} ⊗ ... ⊗ U^{l_d})
```

---

## 四、文件结构与改造说明

```
131_synth_project/
├── main.py                      # 统一入口，零参数运行完整流程
├── reactor_mesh.py              # 网格生成 + 边界提取 + 分形扰动 + Jacobian 质量
├── spectral_quadrature.py       # Legendre / Alpert / Chebyshev / 稀疏网格积分
├── nonlinear_solver.py          # 定点迭代 + 阻尼 Newton + 反应器代数残差
├── momentum_equations.py        # Hartmann 基准 + 相间力 + 有效粘度
├── population_balance.py        # Poisson 成核 + 破裂/聚并核 + QMOM
├── catalyst_optimization.py     # 背包优化 + Diophantine 整数规划
├── numerical_linear_algebra.py  # detq + 幂法 + 稳态浓度求解 + 条件数
├── stochastic_inlet.py          # 随机入口条件生成 + 分布扰动
├── reactor_operations.py        # Julian Day / Gregorian 转换 + 操作日历
└── cfd_solver.py                # CFD-PBM 耦合求解器主程序
```

### 改造要点

1. **nonlinear_solver.py**：将原 MATLAB 的字符串解析式函数改写为 Python 的 lambda/callable 接口，新增物理边界约束（`bounds`）与发散检测机制。
2. **spectral_quadrature.py**：将 IQPACK 的 Jacobi 矩阵构造从 MATLAB 移植为 NumPy 的 `eigh` 特征值分解；Alpert 规则直接嵌入 10 组预计算节点权重表；稀疏网格参考 `nwspgr.m` 的 Smolyak 组合逻辑，使用 `_get_sequences` 生成正整数层数向量。
3. **momentum_equations.py**：将 Hartmann 流的参数从全局变量改为类封装 `HartmannFlow`，新增相间动量交换的三项分解（Drag/Virtual Mass/Lift）。
4. **population_balance.py**：将 Poisson 等待时间模拟从 `rand` / `cumsum` 改写为 `numpy.random.default_rng` 的 `exponential` 方法；QMOM 中 `wheeler_algorithm` 对 2 节点情形采用解析公式替代 Hankel 矩阵，提升数值稳定性。
5. **reactor_mesh.py**：将 `mesh_boundary` 的 MATLAB 排序-匹配逻辑改写为 Python 字典统计，并新增边界环的重排算法；`coastline_perturb` 的 `circshift` 用 `np.roll` 实现。
6. **numerical_linear_algebra.py**：`detq_orthogonal` 保留 Gower AS 82 的逐步化简逻辑，添加除零保护；幂法引入 Google 阻尼因子，与稳态浓度求解的松弛迭代统一。
7. **cfd_solver.py**：将所有模块耦合为 `SlurryBubbleColumnReactor` 类，实现流动场→PBE→温度场→浓度场→催化剂优化的顺序耦合，并加入边界处理（NaN/Inf 保护、Jacobian 负值检查、温度烧结限制）。

---

## 五、合成后项目能解决的科学问题

1. **多相反应器流动-反应耦合模拟**：预测气含率、液速、压力沿轴向的分布，评估相间动量交换对流动结构的调控作用。
2. **气泡尺寸分布演化**：通过 PBE-QMOM 计算 Sauter 平均直径 `d_32` 与比表面积 `a_i`，为传质模型提供动态输入。
3. **反应器热安全分析**：基于能量平衡计算轴向温升，判断是否存在催化剂烧结风险（温度上限 623 K）。
4. **催化剂装载优化**：在给定总催化剂质量约束下，通过背包问题与 Diophantine 整数规划，求解最优分段装载方案。
5. **入口条件不确定性量化**：利用随机入口数据生成，评估温度/流量波动对转化率的敏感性。
6. **数值方法验证**：以 Hartmann 精确解为基准，定量检验动量方程离散化的精度（残差 < 1e-3）。

---

## 六、运行方式

```bash
cd Synthesis-project-python/131_synth_project
python main.py
```

无需任何命令行参数。程序自动执行：
1. 网格生成与质量检验
2. 谱方法数值积分验证
3. 非线性求解器测试
4. Hartmann MHD 基准验证
5. 群体平衡与 Poisson 成核测试
6. 催化剂优化测试
7. 数值线性代数工具测试
8. 随机入口条件生成
9. 反应器操作时间线计算
10. 完整耦合 CFD-PBM 模拟
11. 结果汇总输出

---

## 七、质量检查清单

- [x] 原目录未被修改
- [x] 合成后项目为 Python 语言
- [x] 新目录完整包含合成后的 11 个 `.py` 文件
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] 15 个输入项目均已真实融入，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性（NaN/Inf 保护、物理约束 clip、Jacobian 正则化）
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 无可视化代码
