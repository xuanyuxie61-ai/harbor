# README：博士级合成说明

## 项目概述

**项目名称**：三维热盐环流-浮游生物生态耦合系统的多尺度数值模拟与参数反演  
**科学领域**：海洋科学 —— 热盐环流（Thermohaline Circulation）与海洋生态耦合  
**合成语言**：Python 3  
**合成目录**：`/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/051_synth_project`

本项目将 15 个独立科研代码项目的核心算法融合为一个面向海洋科学前沿问题的博士级计算框架，用于模拟经向翻转环流（MOC）与营养盐-浮游植物-浮游动物-碎屑（NPZD）生态系统的多尺度耦合动力学，并集成不确定性量化、参数标定、最优观测路径规划与数据压缩等高级功能。

---

## 一、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 合成后角色 | 所在文件 |
|------|--------|----------|------------|----------|
| 1 | `1281_toms1040` | Sparse Grids Matlab Kit（稀疏网格工具包） | 高维不确定性量化中多指标集的管理与 Smolyak 稀疏网格构造 | `sparse_grid_cubature.py` |
| 2 | `1053_sandia_cubature` | Stroud 多维求积规则（CN:5-1 等） | 海洋生态参数敏感性分析中的高维数值积分 | `sparse_grid_cubature.py` |
| 3 | `797_nelder_mead` | Nelder-Mead 单纯形优化 | NPZD 生态系统参数（V_max, K_N 等）的自动标定 | `optimization_calibration.py` |
| 4 | `999_r8sto` | 对称 Toeplitz 矩阵逆（Gohberg-Semencul） | 海洋时空相关分析中协方差矩阵的快速求逆 | `matrix_solvers.py` |
| 5 | `869_pic` | Particle-in-Cell（PIC）云-in-单元权重方案 | 拉格朗日营养盐微团在热盐环流中的输运追踪 | `particle_transport.py` |
| 6 | `1364_tsp_descent` | TSP 下降启发式（转置与反转邻域搜索） | 自主水下航行器（AUV）最优采样站部署路径规划 | `optimization_calibration.py` |
| 7 | `1101_sparse_count` | 整数组合枚举（comp_next） | 稀疏网格多指标集的遍历与节点计数 | `sparse_grid_cubature.py` |
| 8 | `184_circle_segment` | 圆缺几何（面积、形心、高度反解） | 曲边海湾海岸线的边界积分与形心计算 | `coastal_geometry.py` |
| 9 | `986_r8ncf` | NCF 稀疏矩阵存储与运算 | 海洋网格上泊松/扩散算子的高效稀疏矩阵-向量乘法 | `matrix_solvers.py` |
| 10 | `814_norm_loo` | L∞ 范数估计 | 模拟输出场（流函数、涡度）的稳定性监测与最大偏差定位 | `uncertainty_quantification.py` |
| 11 | `1137_spquad` | Clenshaw-Curtis 稀疏网格求积 | 生态初级生产力高维参数空间的 Smolyak 积分 | `sparse_grid_cubature.py` |
| 12 | `743_mcnuggets_diophantine` | N 维非负丢番图方程枚举 | 离散氮预算在多个生态功能组间的整数优化分配 | `discrete_allocation.py` |
| 13 | `085_bicg` | 双共轭梯度法（BiCG） | 流函数-涡度泊松方程及大型稀疏线性系统求解 | `matrix_solvers.py` |
| 14 | `278_dictionary_code` | 字典编码/解码 | 长时间海洋模拟状态快照的无损压缩归档 | `discrete_allocation.py` |
| 15 | `659_legendre_exactness` | Legendre 求积精确性检验 | 验证稀疏网格对生态通量积分的代数精度 | `uncertainty_quantification.py` |

---

## 二、新增数学物理模型与核心公式

### 2.1 热盐环流动力学（Thermohaline Circulation）

采用二维经向平面上的流函数-涡度方程组描述 MOC：

**泊松关系**：
```
∇²ψ = -ω
```

**涡度方程**（含斜压项与涡粘耗散）：
```
∂ω/∂t + J(ψ, ω) = ν∇²ω + (g/ρ₀)(∂ρ/∂x)
```
其中 `J(A,B) = A_x B_z - A_z B_x` 为雅可比算子。

**温度平流-扩散方程**：
```
∂T/∂t + J(ψ, T) = κ_T∇²T + Q_T(x,z)
```

**盐度平流-扩散方程**：
```
∂S/∂t + J(ψ, S) = κ_S∇²S + Q_S(x,z)
```

**状态方程**（线性近似）：
```
ρ = ρ₀[1 - α(T - T_ref) + β(S - S_ref)]
```

**斜压项展开**：
```
(g/ρ₀)(∂ρ/∂x) = -gα(∂T/∂x) + gβ(∂S/∂x)
```

数值离散：
- 空间：二阶中心差分（Arakawa 守恒型雅可比离散）
- 时间：Adams-Bashforth 二阶（平流项）+ 显式欧拉（扩散与斜压项）
- 边界：刚性 lid（ψ = 0），无滑移壁面（Dirichlet），T/S 恢复型边界

### 2.2 NPZD 生态系统动力学

**营养盐 N**（mmol N/m³）：
```
∂N/∂t + u·∇N = κ_N∇²N - U(N,I)·P + γ·g·Z + r_D·D
```

**浮游植物 P**：
```
∂P/∂t + u·∇P = κ_P∇²P + U(N,I)·P - m_P·P - G(P)·Z
```

**浮游动物 Z**：
```
∂Z/∂t + u·∇Z = κ_Z∇²Z + β·G(P)·Z - m_Z·Z - g·Z
```

**碎屑 D**：
```
∂D/∂t + u·∇D = κ_D∇²D + (1-β)·G(P)·Z + m_P·P + m_Z·Z - r_D·D - w_s·(∂D/∂z)
```

**Monod-Michaelis 光限制生长率**：
```
U(N,I) = V_max · [N/(K_N + N)] · [I/√(I_opt² + I²)]
```

**Holling II 型摄食函数**：
```
G(P) = g_max · P/(K_P + P)
```

**Beer-Lambert 光衰减**：
```
I(z) = I₀ · exp(-k_w·z - k_c·∫₀^z P(z') dz')
```

### 2.3 拉格朗日粒子输运（PIC 方法）

粒子运动方程：
```
dx_p/dt = u(x_p, z_p),   dz_p/dt = w(x_p, z_p)
```

云-in-单元（CIC）双线性权重投影：
```
Q_grid(i,j) = Σ_p w_p(i,j) · q_p
```
其中 `w_p(i,j)` 为粒子 p 对节点 (i,j) 的双线性权重。

### 2.4 稀疏网格不确定性量化

**Smolyak 公式**：
```
A(q,d) = Σ_{|i|≤q} (Δ^{i₁}⊗...⊗Δ^{i_d}) f
```
其中 `Δ^i = U^i - U^{i-1}`。

**Clenshaw-Curtis 一维节点**：
```
x_k = cos(kπ/n),  k = 0,...,n
```

**权重 FFT 算法（Waldvogel）**：
```
c = [2, 0, 2/(1-4), 0, 2/(1-16), ...]
w = 2·Re{IFFT([c; c_rev])} 的首尾缩放
```

**Sobol' 一阶敏感性指数（PCE 近似）**：
```
S_i = (1/V(Y)) Σ_{α: α_i>0} c_α²
```

### 2.5 参数标定与优化

**Nelder-Mead 单纯形法**：
反射：`x_r = x̄ + ρ(x̄ - x_worst)`  
扩张：`x_e = x̄ + ρξ(x̄ - x_worst)`  
外部收缩：`x_c = x̄ + ργ(x̄ - x_worst)`  
内部收缩：`x_c = x̄ - γ(x̄ - x_worst)`  
压缩：`x_i = x_best + σ(x_i - x_best)`  
标准参数：`ρ=1, ξ=2, γ=0.5, σ=0.5`

**TSP 目标函数**（AUV 路径）：
```
min Σ_{k=1}^n d(p_k, p_{k+1}),  p_{n+1} = p_1
```

### 2.6 曲边海岸几何

**圆缺面积**：
```
θ = 2·arcsin(√(R² - (R-h)²)/R)
A = R²(θ - sin θ)/2
```

**圆缺形心距弦距离**：
```
d = 4R·sin³(θ/2) / [3(θ - sin θ)]
```

### 2.7 离散资源分配

**N 维丢番图方程**：
```
a₁N₁ + a₂N₂ + ... + a_k N_k = b,   N_i ∈ ℤ_{≥0}
```

**字典编码压缩率**：
```
CR = (原始字符数) / (编码后索引数 × 索引位宽 + 字典大小)
```

### 2.8 线性代数工具

**BiCG 迭代格式**：
```
x_{k+1} = x_k + α_k p_k
r_{k+1} = r_k - α_k A p_k
r̃_{k+1} = r̃_k - α_k A^T p̃_k
α_k = (r_k, r̃_k) / (A p_k, p̃_k)
```

**Toeplitz 逆（Gohberg-Semencul）**：
对正定对称 Toeplitz 矩阵 T_n，先求解 Yule-Walker 方程 T_{n-1}v = -a，
再由 Levinson-Durbin 递推构造逆矩阵元素。

---

## 三、文件结构说明

```
051_synth_project/
├── main.py                          # 统一入口，零参数运行
├── thermohaline_circulation.py      # 热盐环流模型（流函数-涡度-温度-盐度）
├── ecosystem_dynamics.py            # NPZD 生态系统与生物地球化学循环
├── particle_transport.py            # 拉格朗日粒子追踪（PIC/CIC 方案）
├── matrix_solvers.py                # BiCG、Toeplitz 逆、NCF 稀疏矩阵、泊松 stencil
├── sparse_grid_cubature.py          # Smolyak 稀疏网格、Stroud 规则、Legendre 精确性检验
├── optimization_calibration.py      # Nelder-Mead 参数标定 + TSP 采样路径优化
├── coastal_geometry.py              # 圆缺几何、曲边域积分、海岸边界长度
├── uncertainty_quantification.py    # L∞ 范数、Sobol' 指数、GCI 收敛指标
└── discrete_allocation.py           # 丢番图资源分配、字典编码状态压缩
```

共 **10 个 .py 文件**（含 main.py），满足至少 8 个的要求。

---

## 四、合成后的项目能够解决什么科学问题

1. **热盐环流-生态耦合模拟**：在二维经向截面上自洽求解物理环流与生物地球化学过程的耦合演化，研究营养盐上涌、初级生产力分布及其对环流强度变化的响应。

2. **拉格朗日输运追踪**：利用 PIC 方法追踪离散营养盐微团在复杂速度场中的路径，量化跨等密度面通量与滞留时间。

3. **高维参数不确定性量化**：通过稀疏网格 Smolyak 求积与多项式混沌展开（PCE），评估生态参数（V_max, K_N, g_max 等）对初级生产力预测的不确定性贡献。

4. **自动参数标定**：基于 Nelder-Mead 优化，将模型输出与观测初级生产力匹配，反演最优生态参数组合。

5. **最优观测网络设计**：将 AUV 采样站部署建模为 TSP，通过下降启发式搜索最小化总航程的观测路径。

6. **复杂海岸区域积分**：对海湾、半岛等曲边几何区域进行高精度面积与通量积分，支持近岸生态监测。

7. **离散资源管理与数据压缩**：利用丢番图方程实现生态氮预算的整数优化分配，利用字典编码压缩海量模拟状态快照。

---

## 五、合成后的项目如何运行

### 环境要求
- Python 3.8+
- NumPy（唯一外部依赖）

### 运行方式
在终端中执行：
```bash
cd "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/051_synth_project"
python main.py
```

无需任何命令行参数。程序将自动执行：
1. 初始化热盐环流与 NPZD 生态系统
2. 运行 20 步耦合时间积分（每步 12 小时）
3. 验证稀疏线性代数模块（BiCG、Toeplitz、NCF）
4. 执行稀疏网格不确定性量化（2 维参数空间）
5. 运行 Nelder-Mead 参数标定
6. 求解 TSP 最优采样路径（12 个站点）
7. 计算曲边海岸几何与区域积分
8. 执行离散氮预算分配与状态字典编码
9. 输出稳定性、收敛性与敏感性分析结果

### 预期输出
整个流程约 2~3 秒完成，终端将打印各模块的计算结果摘要，包括：
- 初级生产力（PP）时间序列
- 流函数 L∞ 范数
- BiCG 残差、Toeplitz 逆误差
- 稀疏网格积分期望值与节点数
- 标定最优参数、TSP 路径成本
- 圆缺面积、生物量积分、海岸边界长度
- 氮预算分配方案、字典压缩比
- Sobol' 一阶敏感性指数

---

## 六、代码鲁棒性与边界处理设计

本项目在以下方面体现了高工程复杂度与数值鲁棒性：

1. **非负约束**：所有生态浓度（N, P, Z, D）在每次子步后强制 clip 到 [0, max]，防止无界增长。
2. **NaN/Inf 清洗**：在关键场更新后使用 `np.where(np.isfinite(...), ..., default)` 替换异常值。
3. **BiCG 回退机制**：若迭代求解出现非有限值，自动回退至直接法（`numpy.linalg.solve`）。
4. **粒子边界反射**：底边界与海面采用反射条件，侧边界采用开放流入-流出并重新注入。
5. **生物子步（substepping）**：生态反应时间尺度远小于物理平流，自动细分时间步以满足稳定性。
6. **参数边界检查**：Nelder-Mead 标定中对 V_max、K_N 等施加正下界。
7. **组合枚举安全**：`comp_next` 与 `diophantine_nd_nonnegative` 对空解、负预算等边界情况返回空数组。
8. **GCI 除零保护**：收敛指标计算中对所有可能出现零除的分母设置最小阈值。

---

## 七、科学难度说明

本项目的科学计算难度达到博士级，体现在：

- **多物理场耦合**：同时求解非线性涡度方程、两个标量平流-扩散方程与四个耦合的常微分-偏微分生物反应方程。
- **多尺度挑战**：物理平流尺度（10⁶ m）与生物反应尺度（10⁻³ m）相差 9 个数量级，需采用子步技术。
- **高维不确定性量化**：2~4 维参数空间的稀疏网格求积涉及组合数学、正交多项式理论与快速 FFT 权重算法。
- **反问题与优化**：参数标定为非凸反问题，TSP 为 NP-hard 组合优化。
- **复杂几何处理**：圆缺几何的非线性反解（由面积求弓高）需要牛顿迭代。
- **大规模稀疏线性代数**：BiCG 求解数十万元未知数的泊松方程，Toeplitz 逆利用特殊结构将复杂度从 O(n³) 降至 O(n²)。
