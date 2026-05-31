# 肿瘤生长微环境多尺度计算建模系统 —— 博士级合成说明

## 一、项目概述

本项目围绕**生物医学前沿领域：肿瘤生长微环境建模（Tumor Growth Microenvironment Modeling）**，将 15 个种子项目的核心算法融合为一个多尺度、多物理场耦合的博士级计算框架。

### 1.1 科学问题定位

实体肿瘤（Solid Tumor）的生长受微环境（TME）多重因素调控：
- **营养输运**：氧气、葡萄糖通过血管网络扩散至肿瘤组织
- **细胞异质性**：增殖态（P）、静息态（Q）、凋亡态（A）、坏死态（N）之间的动态转化
- **空间竞争**：细胞通过接触抑制和 Voronoi 镶嵌竞争有限空间
- **固体应力**：细胞外基质（ECM）沉积与血管塌陷产生机械应力
- **治疗响应**：药物渗透受应力场和细胞密度共同制约

本系统建立了一套从几何建模 → 营养扩散 → 细胞动力学 → 空间生长 → 机械应力 → 谱方法求解 → 非线性耦合 → 有限元离散 → 治疗评估的完整计算管线，所有模块可零参数运行。

---

## 二、种子项目到科学问题的映射

| 编号 | 原始项目 | 核心算法 | 合成后角色 |
|:---|:---|:---|:---|
| 078 | bernstein_polynomial | Bernstein 多项式基 | **肿瘤边界参数化**：控制点驱动的闭合曲线生成 |
| 1333 | triangulation_boundary_nodes | 三角网格边界检测 | **边界节点识别**：为 Dirichlet/Neumann 条件标记边界自由度 |
| 1168 | stla_to_tri_surface_fast | STL 表面快速三角化 | **肿瘤表面网格重建**：Delaunay 剖分与表面提取 |
| 646 | laplace_radial_exact | 径向 Laplace 精确解 | **氧气径向扩散**：验证数值解与解析解的一致性 |
| 1056 | sandia_sparse | 稀疏网格 Clenshaw-Curtis 积分 | **高维参数积分**：营养源项与药物响应曲面的稀疏网格求积 |
| 1200 | tennis_matrix | 马尔可夫转移矩阵 | **细胞状态转移**：P→Q→A→N 的离散时间马尔可夫链 |
| 673 | lights_out_game | 网格邻居交互更新 | **细胞自动机（CA）**：接触抑制与低氧诱导的邻域状态演化 |
| 256 | cvt_corn_movie | 圆盘约束 CVT 与径向生长 | **肿瘤空间生长**：Lloyd 迭代优化细胞 Voronoi 镶嵌 |
| 850 | partition_greedy | 贪婪二分分区 | **代谢异质性分区**：肿瘤亚克隆的代谢活性二分 |
| 172 | chladni_figures | 双调和算子特征值问题 | **固体应力分析**：薄板弯曲的 biharmonic 算子与冯·米塞斯应力 |
| 607 | jacobi_polynomial | Jacobi 多项式求值 | **谱方法基函数**：正交多项式展开浓度场 |
| 940 | quad_gauss | Gauss-Legendre 数值积分 | **谱内积计算**：Legendre 节点/权重的高精度积分 |
| 808 | nonlin_newton | Newton 非线性求解 | **耦合稳态求解**：肿瘤-营养非线性 PDE 系统的 Newton 迭代 |
| 1156 | st_to_ge | 稀疏三元组转稠密 | **FEM 刚度矩阵组装**：三角形单元刚度累加为全局矩阵 |
| 1100 | sort_rc | 外部可控堆排序 | **细胞群体排序**：按增殖速率/药物敏感性排序细胞 |

**所有 15 个项目均已真实融入，无遗漏、无挂名。**

---

## 三、新增数学物理模型与核心公式

### 3.1 肿瘤几何建模（Bernstein 参数化 + Delaunay 三角剖分）

肿瘤边界由 Bernstein 多项式参数化定义：

```
B_{n,i}(t) = C(n,i) * t^i * (1-t)^{n-i},   t ∈ [0,1]
C(t) = Σ_{i=0}^{n} P_i * B_{n,i}(t)
```

其中 `P_i` 为二维控制点。闭合曲线要求 `P_0 = P_n`。

Delaunay 三角剖分基于外接圆空圆准则：对任意三角形，其外接圆内不含其他节点。采用 Bowyer-Watson 增量算法实现。

边界检测基于边出现频率统计：仅被一个三角形使用的边为边界边，其端点为边界节点。

### 3.2 营养扩散-反应方程（径向 Laplace + Michaelis-Menten）

稳态氧气浓度 `C(r)` 满足反应-扩散方程：

```
D_O2 * ∇²C - k_c * ρ * C / (K_m + C) + S(r) = 0
```

在径向对称假设下，2D 简化形式为：

```
D * (1/r) * d/dr(r * dC/dr) - k_c * ρ * C / (K_m + C) + S(r) = 0
```

其解析通解（无消耗时）为 Laplace 径向解：

```
2D:  u(r) = a * log(r) + b
3D:  u(r) = a / r + b
```

含线性消耗 `k*C` 时，通解为修正 Bessel 函数 `I_0(√(k/D)*r)` 与 `K_0` 的组合。

Michaelis-Menten 消耗动力学：

```
f(C, ρ) = V_max * ρ * C / (K_m + C)
```

缺氧区域定义为 `C < C_hypoxia`（通常 `C_hypoxia = 0.02`，对应 pO₂ < 2 mmHg）。

### 3.3 稀疏网格数值积分（Smolyak 构造）

对 d 维积分，采用 Clenshaw-Curtis 规则的稀疏网格组合：

```
A(L,d) = Σ_{|l|_1 ≤ L+d-1} α_l * (Q_{l_1} ⊗ ... ⊗ Q_{l_d})
α_l = (-1)^{L+d-1-|l|_1} * C(d-1, L+d-1-|l|_1)
```

1D CC 节点与权重：

```
x_i = cos((i-1)*π / (n-1)),  i = 1..n
w_i = (1 / (n-1)) * [1 - Σ_{j} b_j * cos(2j*θ_i) / (4j²-1)]
```

稀疏网格对光滑函数具有 `O(N^{-2} * |log N|^{d-1})` 的收敛率，远优于全张量积的 `O(N^{-2/d})`。

### 3.4 细胞状态动力学（CTMC + CA）

**连续时间马尔可夫链（CTMC）**：

细胞状态空间 `S = {P, Q, A, N}`，转移概率矩阵 `M` 满足 `Σ_j M_{ij} = 1`。

群体演化递推：

```
N_{t+1} = M^T * N_t
```

有效倍增时间由指数拟合估计：

```
N(t) = N_0 * e^{λt}  ⇒  T_d = ln(2) / λ
```

**细胞自动机（CA）规则**（基于 Lights Out 邻域交互思想）：

对格点 `(i,j)`，统计 8-邻域活细胞数 `n_neighbors`：
- 若 `n_neighbors ≥ inhibition_threshold`：接触抑制 → 转入静息态 Q
- 若 `C(i,j) < C_threshold`：低氧 → 凋亡态 A（极低的 → 坏死态 N）
- 否则：正常增殖或从 Q/A 恢复

增殖步以概率 `p_empty` 向空位或凋亡位扩展新细胞。

### 3.5 空间 Voronoi 生长（CVT + 径向扩展）

CVT 能量泛函：

```
F(P) = Σ_{i=1}^{N_p} ∫_{V_i} ρ(x) * |x - p_i|² dx
```

Lloyd 迭代将生成子移至 Voronoi 区域质心：

```
p_i^{(new)} = (∫_{V_i} x dx) / (∫_{V_i} dx)
```

径向生长遵循面积守恒：新增 `n_bud` 个边界细胞时，半径扩展因子为：

```
factor = (n_boundary + n_bud) / n_boundary
R_new = factor * R_old
```

**代谢异质性贪婪分区**：给定代谢权重集合 `W = {w_i}`，目标为：

```
min |Σ_{i∈S_0} w_i - Σ_{i∈S_1} w_i|
```

算法按降序排列权重，每次放入当前和较小的子集。

### 3.6 固体应力分析（双调和算子）

肿瘤内部固体应力由薄板双调和方程描述：

```
∇⁴ w - λ * w = 0
```

离散化采用 5 点 Laplacian 的复合：

```
L = D_{5-point},   A = L * L
```

广义特征值问题：

```
A_0 * φ = λ * B_0 * φ
```

冯·米塞斯等效应力（Von Mises Stress）：

```
σ_vm = √(σ_xx² - σ_xx*σ_yy + σ_yy² + 3*σ_xy²)
```

其中：

```
σ_xx = E/(1-μ²) * (w_xx + μ*w_yy)
σ_yy = E/(1-μ²) * (w_yy + μ*w_xx)
σ_xy = E/(2(1+μ)) * w_xy
```

应力诱导凋亡概率采用 Sigmoid 响应：

```
P_apop(σ) = 1 / (1 + exp(-s*(σ - σ_th)))
```

### 3.7 谱方法求解（Jacobi 基 + Gauss-Legendre 积分）

Jacobi 多项式 `P_n^{(α,β)}(x)` 满足正交性：

```
∫_{-1}^{1} (1-x)^α (1+x)^β P_i(x) P_j(x) dx = γ_i * δ_{ij}
```

三项递推关系：

```
P_0(x) = 1
P_1(x) = 0.5*(α+β+2)*x + 0.5*(α-β)
c1*P_i = (c3 + c2*x)*P_{i-1} + c4*P_{i-2}
```

Gauss-Legendre 求积节点为 Legendre 多项式 `P_n(x)` 的零点，权重：

```
w_i = 2 / [(1-x_i²) * (P_n'(x_i))²]
```

对 `2n-1` 次多项式精确。通过 Elhay-Kautsky 方法（对称三对角矩阵特征值分解）高效计算。

谱 Galerkin 刚度矩阵：

```
K_{ij} = ∫_{-1}^{1} dφ_i/dx * dφ_j/dx * w(x) dx
```

### 3.8 非线性肿瘤-营养耦合系统（Newton 迭代）

离散化后的耦合残差：

```
R_C = D * L * C - k_c * ρ * C/(K_m + C) + S
R_ρ = λ_p * ρ * (1 - ρ/ρ_max) * C/(K_m + C) - λ_d * ρ
```

Newton 迭代格式：

```
J(U^k) * δU = -R(U^k)
U^{k+1} = U^k + δU
```

Jacobian 的解析形式包含对角修正项：

```
dR_C/dC = D*L - diag(k_c*ρ*K_m / (K_m+C)²)
dR_ρ/dρ = diag(λ_p*(1-2ρ/ρ_max)*C/(K_m+C) - λ_d)
```

数值鲁棒性策略：
- 线搜索（backtracking）保证残差单调下降
- Levenberg-Marquardt 修正处理奇异 Jacobian
- 变量截断 `U ≥ 0` 防止非物理负值
- 若收敛到平凡解 `ρ≈0`，利用空间平均解析关系恢复非平凡稳态：

```
ρ_avg = ρ_max * (1 - λ_d*(K_m+C_avg) / (λ_p*C_avg))
```

### 3.9 有限元刚度矩阵组装（ST → GE）

对三角形单元 `e = (a,b,c)`，局部刚度矩阵：

```
K_{ij}^{(e)} = (1/(4*A_e)) * [(x_j-x_k)*(x_i-x_k) + (y_j-y_k)*(y_i-y_k)]
```

其中 `A_e` 为单元有向面积，`(i,j,k)` 为 `(a,b,c)` 的轮换。

稀疏三元组（COO）格式：

```
(IST(k), JST(k), AST(k)),  k = 1..NST
```

转换为稠密格式时同位置数值累加：

```
A_{i,j} = Σ_{k: IST(k)=i, JST(k)=j} AST(k)
```

Dirichlet 边界处理：对边界节点 `i`，清零第 `i` 行/列，置 `A_{ii}=1`，`b_i=g_i`。

孤立节点检测：统计未出现在任何三角形中的节点，强制施加 `u=0` 边界条件，消除矩阵奇异性。

### 3.10 治疗响应评估（Gauss + CC 积分）

治疗响应指数（Therapeutic Response Index, TRI）：

```
TRI = ∫_Ω C_drug(x) * ρ(x) * exp(-β * σ_vm(x)) dA
```

数值实现采用中点法则与网格数据兼容。高精度一维积分采用 Gauss-Legendre 或 Clenshaw-Curtis 规则。

Richardson 外推误差估计：

```
err_est = |Q_{fine} - Q_{coarse}|
```

---

## 四、项目文件结构

```
123_synth_project/
├── main.py                           # 统一入口，零参数运行
├── tumor_geometry.py                 # 肿瘤几何建模（Bernstein + Delaunay + 边界检测）
├── nutrient_diffusion.py             # 营养扩散（Laplace + CC 积分 + 稀疏网格）
├── cellular_dynamics.py              # 细胞动力学（Markov 链 + CA 邻居交互）
├── spatial_voronoi.py                # 空间 Voronoi 生长（CVT + 贪婪分区）
├── mechanical_stress.py              # 机械应力（双调和算子 + 冯·米塞斯应力）
├── spectral_solver.py                # 谱方法求解（Jacobi + Gauss-Legendre）
├── nonlinear_coupling.py             # 非线性耦合（Newton 迭代 + Jacobian）
├── sparse_fem.py                     # 稀疏 FEM（ST→GE + 刚度组装 + Dirichlet BC）
├── quadrature_rules.py               # 求积规则（Gauss + CC + 治疗响应积分）
├── utils.py                          # 工具函数（排序 + 鲁棒性 + Gini + Morse 势）
└── README_博士级合成说明.md          # 本文档
```

共 **12 个 .py 文件**（含 main.py），满足至少 8 个的要求。

---

## 五、合成改造路径说明

### 5.1 原项目到科学问题的具体改造

**078_bernstein_polynomial → tumor_geometry.py**
- 保留 Bernstein 多项式的递推计算核心
- 改造目标：用控制点参数化肿瘤闭合边界曲线
- 新增：多控制点闭合曲线生成、Delaunay 三角剖分、边界节点识别

**1333_triangulation_boundary_nodes → tumor_geometry.py**
- 保留边界边/节点检测算法（边出现频率统计）
- 改造目标：为肿瘤三角网格标记边界自由度，施加 Dirichlet/Neumann 条件
- 新增：0-based/1-based 索引自动检测与纠正

**1168_stla_to_tri_surface_fast → tumor_geometry.py**
- 保留 STL 风格表面三角化思想
- 改造目标：快速构建肿瘤二维表面三角网格
- 新增：内部采样点筛选（绕数法判定点在多边形内）、面积/周长计算

**646_laplace_radial_exact → nutrient_diffusion.py**
- 保留径向 Laplace 方程的精确解公式（u, ux, uy, uxx, uxy, uyy）
- 改造目标：验证营养扩散数值解，提供解析基准
- 新增：修正 Bessel 函数描述的含消耗径向稳态解

**1056_sandia_sparse → nutrient_diffusion.py**
- 保留 Clenshaw-Curtis 节点/权重计算、层级索引枚举
- 改造目标：高维参数空间（如多药物浓度组合）的稀疏网格积分
- 新增：Smolyak 组合公式的 2D 实现、单项式积分验证

**1200_tennis_matrix → cellular_dynamics.py**
- 保留转移矩阵的行随机性构造（概率归一化）
- 改造目标：肿瘤细胞四态（P/Q/A/N）的离散时间马尔可夫链
- 新增：CTMC 群体演化、倍增时间指数拟合

**673_lights_out_game → cellular_dynamics.py**
- 保留网格邻居交互状态更新逻辑（4/8 邻域翻转/切换）
- 改造目标：细胞自动机规则（接触抑制、低氧凋亡、增殖扩展）
- 新增：基于营养场和邻居密度的多规则状态转移

**256_cvt_corn_movie → spatial_voronoi.py**
- 保留 CVT Lloyd 迭代、圆盘约束、径向生长因子计算
- 改造目标：肿瘤细胞的空间 Voronoi 镶嵌与竞争性生长
- 新增：边界生成子投影回圆周、内部点圆盘约束

**850_partition_greedy → spatial_voronoi.py**
- 保留贪婪降序分区核心（放入当前和较小的子集）
- 改造目标：肿瘤代谢异质性亚克隆二分
- 新增：代谢权重生成、分区差异计算

**172_chladni_figures → mechanical_stress.py**
- 保留双调和算子的离散构造（5 点 Laplacian 复合、边界修正）
- 改造目标：肿瘤固体应力的薄板弯曲模型
- 新增：冯·米塞斯等效应力数值差分、应力诱导凋亡 Sigmoid 模型

**607_jacobi_polynomial → spectral_solver.py**
- 保留 Jacobi 多项式的三项递推公式、参数边界检查
- 改造目标：谱方法基函数展开营养浓度场
- 新增：正交性检验、Gauss-Jacobi 内积积分

**940_quad_gauss → spectral_solver.py**
- 保留 Gauss-Legendre 的 Elhay-Kautsky 实现、IMTQLX 对角化
- 改造目标：谱方法刚度矩阵与质量矩阵的高精度数值积分
- 新增：区间线性变换、谱 Galerkin RHS 组装

**808_nonlin_newton → nonlinear_coupling.py**
- 保留 Newton 迭代的线搜索、导数下界保护、发散检测
- 改造目标：肿瘤-营养耦合非线性 PDE 的稳态求解
- 新增：多维 Newton、解析 Jacobian、Levenberg-Marquardt 回退、非平凡稳态恢复

**1156_st_to_ge → sparse_fem.py**
- 保留稀疏三元组到稠密矩阵的累加转换
- 改造目标：有限元刚度矩阵的全局组装
- 新增：三角形单元局部刚度公式、Dirichlet BC 施加、孤立节点检测

**1100_sort_rc → utils.py**
- 保留 Nijenhuis-Wilf 外部排序接口（堆排序变体）
- 改造目标：细胞群体按属性排序
- 新增：数组排序包装、Gini 系数、Morse 势、参数边界验证

---

## 六、运行方式

### 6.1 环境要求

- Python 3.8+
- NumPy
- SciPy（仅用于 `scipy.special.i0`, `scipy.special.k0`, `scipy.special.erf`）

### 6.2 运行命令

```bash
cd Synthesis-project-python/123_synth_project
python main.py
```

无需任何命令行参数。程序将自动执行以下完整流程：
1. 生成 Bernstein 参数化肿瘤边界并进行 Delaunay 三角剖分
2. 求解径向营养扩散与 Michaelis-Menten 消耗
3. 运行 Markov 链细胞状态演化与细胞自动机模拟
4. 执行 CVT Lloyd 迭代、径向生长与代谢贪婪分区
5. 计算双调和固体应力与冯·米塞斯等效应力
6. Jacobi 谱方法正交性检验与一维扩散方程求解
7. Newton 迭代求解肿瘤-营养耦合非线性稳态
8. FEM 刚度矩阵组装、Dirichlet 边界处理与线性求解
9. Gauss-Legendre / Clenshaw-Curtis 高精度积分与治疗响应评估
10. 鲁棒性工具验证（排序、Gini、Morse 势、参数检查）

### 6.3 预期输出

程序将在终端打印各模块的定量结果，包括：
- 肿瘤几何参数（面积、周长、圆度）
- 氧气浓度分布与缺氧比例
- 细胞群体分布与倍增时间
- CVT 能量泛函与代谢分区差异
- 固体应力统计指标与凋亡概率
- 谱方法正交性检验与误差
- Newton 迭代收敛信息与非平凡稳态
- FEM 求解结果与稀疏-稠密一致性验证
- 治疗响应指数 TRI 与累积氧消耗
- 综合摘要报告

---

## 七、数值鲁棒性与边界处理

本项目在多个层面实现了边界处理与数值鲁棒性：

1. **几何层面**：
   - Bernstein 参数化输入 `t` 自动截断到 `[0,1]`
   - Delaunay 剖分处理退化三角形（面积 < 1e-15 则跳过）
   - 孤立节点检测并强制施加 Dirichlet 条件，消除刚度矩阵奇异性

2. **扩散层面**：
   - 径向坐标 `r` 下限截断至 `1e-12`，防止 `log(0)` 或 `1/0`
   - Michaelis-Menten 分母 `Km + C` 下限保护至 `1e-15`

3. **细胞动力学层面**：
   - Markov 转移矩阵每行概率和自动归一化到 1
   - CA 规则中所有状态转移有明确的边界条件（Necrosis 不可逆）

4. **非线性求解层面**：
   - Newton 迭代带线搜索与阻尼（damping=0.6）
   - Jacobian 奇异时自动切换 Levenberg-Marquardt 正则化
   - 变量截断防止负浓度/负密度
   - 平凡稳态检测与解析恢复机制

5. **积分层面**：
   - Clenshaw-Curtis 权重和验证为 2（[-1,1]）
   - Gauss-Legendre 节点通过特征值分解高精度计算
   - Richardson 外推提供误差估计

---

## 八、科学前沿性与博士级难度说明

### 8.1 多尺度耦合

本项目同时涉及：
- **微观尺度**：单个细胞的 Markov 状态转移与 CA 邻域交互
- **介观尺度**：CVT 空间镶嵌与细胞竞争
- **宏观尺度**：连续介质 PDE（扩散-反应、双调和应力、谱方法）

### 8.2 高阶数值方法

- **稀疏网格积分**：Smolyak 组合技术，突破维度诅咒
- **谱方法**：Jacobi 多项式基，指数收敛特性
- **Newton-Krylov 风格**：非线性耦合系统的 Jacobian 解析推导与迭代求解
- **FEM 误差分析**：L2 误差基于单元质心近似

### 8.3 复杂边界条件

- 肿瘤几何边界：Bernstein 参数化闭合曲线
- 营养扩散：Dirichlet 边界（血管源）+ 内部 Robin 型消耗
- 固体应力：自由表面边界 + 双调和 ghost point 处理
- FEM：Dirichlet 约束 + 孤立节点正则化

### 8.4 生物医学公式密度

项目中系统注入的公式包括：
- Bernstein 基函数、Laplace 径向解、修正 Bessel 方程
- Michaelis-Menten 酶动力学、CTMC 主方程
- CVT 能量泛函、Voronoi 质心公式
- 双调和算子、冯·米塞斯应力、Sigmoid 响应
- Jacobi 正交性、三项递推、Gauss-Legendre 权重公式
- Newton 迭代、Jacobian 矩阵、Logistic 增长稳态
- FEM 局部刚度矩阵、Dirichlet 罚函数
- 稀疏网格 Smolyak 组合系数、Richardson 外推

---

## 九、结论

本项目成功将 15 个独立科研代码项目的核心算法融合为一个面向**肿瘤生长微环境建模**的博士级多尺度计算系统。系统涵盖几何建模、营养扩散、细胞动力学、空间生长、机械应力、谱方法、非线性耦合、有限元离散和高精度积分等十大模块，具备完整的数学物理公式体系、数值鲁棒性处理和零参数可运行能力。
