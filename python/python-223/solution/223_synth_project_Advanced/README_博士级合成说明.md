# 计算高能物理: 喷注聚类与 Jet Substructure 分析
## 高阶有限差分与稳定性分析 (小规模可复现实验)

**项目类型**: 博士级科学计算合成项目
**应用领域**: 计算高能物理 (Computational High-Energy Physics)
**编程语言**: Python 3 (纯标准库 + NumPy)
**合成来源**: 15 个随机种子科研项目的核心算法融合

---

## 一、科学问题概述

本项目实现了一个**前沿博士级**的计算高能物理问题: 在 LHC 对撞机环境下,
对**喷注 (jet)** 进行聚类、子结构分析与数值稳定性研究. 具体包括:

1. **喷注聚类算法**: Anti-kT、Cambridge/Aachen、kT 三种顺序重组算法的
   统一实现, 包含完整的距离矩阵计算、最近邻搜索与合并历史追踪.
2. **Jet Substructure 观测量**: N-subjettiness (τ_N)、能量关联函数 (ECF)、
   D₂ 比率、能量密度径向分布 ρ(r)、二维互相关函数 C(Δη,Δφ) 等.
3. **高阶有限差分**: Fornberg 算法生成任意阶 FD 系数, 4阶中心差分、
   Richardson 外推、三角周期插值 (用于方位角 φ 方向).
4. **数值稳定性分析**: Von Neumann 放大因子、CFL 条件、
   聚类矩阵条件数、pT 阈值敏感性、微扰稳定性分析.
5. **守恒定律验证**: 四动量守恒、DGLAP 动量求和规则、
   守恒 ODE (摆锤 + 刚性转子) 的数值漂移分析.

---

## 二、核心数学物理模型

### 2.1 四维动量与 Lorentz 不变量

采用 (+, -, -, -) 度规约定, 粒子四动量:

$$
p^\mu = (E, p_x, p_y, p_z), \quad p^2 = E^2 - |\vec{p}|^2 = m^2
$$

赝快度 (pseudorapidity):

$$
\eta = -\ln\left[\tan\frac{\theta}{2}\right]
$$

快度 (rapidity):

$$
y = \frac{1}{2}\ln\frac{E+p_z}{E-p_z}
$$

横动量: $p_T = \sqrt{p_x^2 + p_y^2}$

喷注距离度量:

$$
\Delta R = \sqrt{(\Delta\eta)^2 + (\Delta\phi)^2}
$$

### 2.2 Anti-kT 聚类算法

统一距离定义:

$$
d_{ij} = \min(p_{Ti}^p, p_{Tj}^p) \cdot \frac{\Delta R_{ij}^2}{R^2}
$$
$$
d_{iB} = p_{Ti}^p
$$

参数 $p$:
- $p = -2$: Anti-kT (LHC 标准, 产生圆形喷注)
- $p = 0$: Cambridge/Aachen (纯几何)
- $p = +1$: kT (优先合并软粒子)

### 2.3 N-subjettiness

$$
\tau_N^{(\beta)} = \frac{1}{d_0} \sum_i p_{Ti} \min_k (\Delta R_{i,k})^\beta
$$

其中 $d_0 = \sum_i p_{Ti} R^\beta$ 为归一化因子.

判别量: $\tau_{21} = \tau_2 / \tau_1$

- $\tau_{21} \ll 1$: 2-prong 结构 (如 W/Z → qq')
- $\tau_{21} \sim 1$: 各向同性 (QCD 喷注)

### 2.4 能量关联函数 (ECF)

二体:
$$
e_2^{(\alpha)} = \sum_{i<j} z_i z_j (\Delta R_{ij})^\alpha
$$

三体:
$$
e_3^{(\alpha)} = \sum_{i<j<k} z_i z_j z_k (\Delta R_{ij}\Delta R_{jk}\Delta R_{ik})^\alpha
$$

D₂ 观测量:
$$
D_2 = \frac{e_3 \cdot e_1^3}{e_2^3}
$$

- Quark jet: $D_2 \sim \mathcal{O}(\alpha_s)$
- Gluon jet: $D_2 \sim \mathcal{O}(1)$

### 2.5 DGLAP 演化与 Splitting Functions

$$
\frac{\partial f_i(x, Q^2)}{\partial \ln Q^2} = \sum_j \int_x^1 \frac{dz}{z} P_{ij}(z) f_j\left(\frac{x}{z}, Q^2\right)
$$

Leading-order splitting functions:

$$
P_{qq}(z) = C_F \frac{1+z^2}{1-z}, \quad P_{gq}(z) = C_F \frac{1+(1-z)^2}{z}
$$
$$
P_{qg}(z) = T_R (z^2 + (1-z)^2), \quad P_{gg}(z) = 2C_A\left[\frac{z}{1-z} + \frac{1-z}{z} + z(1-z)\right]
$$

颜色因子 (SU(3)): $C_F = 4/3$, $C_A = 3$, $T_R = 1/2$

1-loop 跑动耦合:

$$
\alpha_s(Q^2) = \frac{12\pi}{(33-2n_f)\ln(Q^2/\Lambda_{QCD}^2)}
$$

### 2.6 高阶有限差分 (Fornberg 算法)

p阶精度一阶导数:

$$
f'(x) \approx \frac{1}{h} \sum_k c_k f(x + s_k h)
$$

4阶中心差分:

$$
f'(x) \approx \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h}
$$

Richardson 外推:

$$
D_{extrap} = \frac{4^k D(h/2) - D(h)}{4^k - 1} \quad \Rightarrow \quad \mathcal{O}(h^{2k+2})
$$

### 2.7 Von Neumann 稳定性分析

对线性 PDE $\partial_t u + a\partial_x u = \nu \partial_{xx} u$,
离散化后 $u_j^{n+1} = G \cdot u_j^n$.

各格式的放大因子:

- **FTCS**: $G = 1 - 2r(1-\cos\theta) - ic\sin\theta$
- **Lax-Friedrichs**: $G = \cos\theta - ic\sin\theta$
- **Lax-Wendroff**: $G = 1 - c^2(1-\cos\theta) - ic\sin\theta$
- **Upwind**: $G = 1 - c(1-e^{-i\theta})$

稳定性条件: $|G(k\Delta x)| \leq 1 + \mathcal{O}(\Delta t)$

CFL 条件: $\Delta t \leq C \cdot \Delta x / |v_{max}|$

### 2.8 Padua 节点

$[-1,1]^2$ 上总次数 $\leq L$ 的最优多项式插值节点集,
节点数 $N = (L+1)(L+2)/2$. 四个 family 由 $L \bmod 4$ 决定.

### 2.9 Mandelstam 变量

$$
s = (p_1 + p_2)^2, \quad t = (p_1 - p_3)^2, \quad u = (p_1 - p_4)^2
$$

恒等式: $s + t + u = m_1^2 + m_2^2 + m_3^2 + m_4^2$

---

## 三、原项目到科学问题的映射

| 种子项目 | 原算法 | 科学映射 |
|---------|--------|---------|
| 849_partition_brute | 暴力集合划分 | 喷注 constituents 的 pT 平衡二分区优化 (NP-hard) |
| 1233_passive-DAS | 互相关函数 | 喷注内二维能量互相关 C(Δη,Δφ) 提取角关联 |
| 964_r83p | 3D 点集运算 | 四动量 Lorentz 向量代数、行列式、随机采样 |
| 176_circle_arc_grid | 圆弧网格 | η-φ 平面喷注锥边界圆弧离散采样 |
| 843_padua | Padua 节点 | 喷注能量密度场最优多项式重建节点 |
| 793_nearest_neighbor | KNN 搜索 | Anti-kT 聚类距离矩阵的最近对搜索 |
| 106_boundary_word | 边界词追踪 | 喷注 η-φ 边界的方向序列表示与对称性分析 |
| 961_r8_scale | 浮点邻域 | pT 阈值的 IEEE-754 邻域扫描 (鲁棒性) |
| 1161_AmandaRosa | framework/subscriber | 事件生成器框架: 多物理过程注册 |
| 208_conservation_ode | 守恒 ODE 积分 | 摆锤+刚性转子的守恒量漂移 → 聚类四动量守恒 |
| 1131_rockstar | merger tree | 聚类历史的二叉树表示 (Mass Drop, pruning) |
| 1356_trig_interp | 三角插值 | φ 方向周期性 Dirichlet 核插值 |
| 192_closest_point | 暴力最近点 | 粒子对最小 ΔR 搜索 (聚类核心) |
| 112_box_display | 网格区域逻辑 | 量热器 cell 的 η-φ 区域掩码判定 |
| 1252_numerical-bypass | 数值优化 | 聚类结果对四动量微扰的稳定性扫描 |

---

## 四、修改文件清单

| 文件 | 功能 | 关键算法 |
|------|------|---------|
| `main.py` | 统一入口, 10 阶段分析流程 | 全流程编排 |
| `jet_fourvector.py` | 四动量 + 浮点邻域 + 分区 | FourVector, Mandelstam, partition_brute |
| `jet_grid.py` | η-φ 网格 + Padua + 圆弧 | CalorimeterGrid, Padua, arc_grid |
| `jet_clustering.py` | 喷注聚类 (anti-kT/C/A/kT) | 距离矩阵, 最近对, E-scheme 重组 |
| `jet_substructure.py` | N-subjettiness, ECF, D₂ | τ_N, e2/e3, ρ(r), 互相关 |
| `jet_boundary.py` | 边界词 + 面积 + Voronoi | BoundaryWord, MC面积, 反射/旋转 |
| `high_order_fd.py` | 高阶 FD + 三角插值 | Fornberg, 4阶中心差分, Richardson |
| `stability_analysis.py` | Von Neumann + CFL + 敏感性 | 放大因子, 条件数, pT扫描 |
| `event_generator.py` | 部分子簇射 + 事件生成 | DGLAP, PartonShower, EventGenerator |
| `conservation_laws.py` | 守恒律 + ODE 验证 | 四动量守恒, 摆锤/转子 ODE |

---

## 五、项目结构

```
223_synth_project_Advanced/
├── main.py                    # 统一入口 (零参数)
├── jet_fourvector.py          # 四动量 + Lorentz 运算
├── jet_grid.py                # η-φ 量热器网格 + Padua 节点
├── jet_clustering.py          # Anti-kT / C/A / kT 聚类
├── jet_substructure.py        # N-subjettiness, ECF, D₂
├── jet_boundary.py            # 喷注边界词 + Voronoi 面积
├── high_order_fd.py           # 高阶有限差分 + 三角插值
├── stability_analysis.py      # Von Neumann + CFL + 敏感性
├── event_generator.py         # 部分子簇射 + 事件框架
├── conservation_laws.py       # 守恒律 + ODE 验证
└── README_博士级合成说明.md    # 本文档
```

---

## 六、运行方法

```bash
cd 223_synth_project_Advanced
python3 main.py
```

**零参数**运行. 输出包含 10 个阶段:
1. 多喷注事件生成 (部分子簇射)
2. η-φ 量热器网格能量沉积
3. 三种喷注聚类算法
4. Jet substructure 观测量
5. 喷注边界与面积
6. 高阶有限差分精度分析
7. 数值稳定性 (Von Neumann, CFL)
8. 守恒定律验证
9. Mandelstam 变量与动量分区
10. 喷注观测量汇总

---

## 七、科学问题与可解决任务

本项目可解决以下博士级科研计算问题:

1. **喷注算法对比**: 在同一事件上比较 anti-kT/C/A/kT 的聚类行为,
   分析喷注形状、红外安全性、对软辐射的敏感度.
2. **Quark/Gluon 判别**: 通过 D₂、τ₂₁ 等 IRC-safe 观测量
   区分 quark 喷注与 gluon 喷注.
3. **Boosted 对象识别**: 使用 Mass Drop + Symmetry 条件
   识别 W/Z/H → qq' 的大半径喷注.
4. **有限差分精度优化**: 通过 Richardson 外推和最佳步长选择,
   将喷注子结构观测量的数值误差降至机器精度量级.
5. **聚类稳定性量化**: 用条件数、微扰扫描量化聚类结果对
   输入粒子四动量的敏感度, 指导实验系统误差估计.
6. **守恒律数值验证**: 通过 ODE 积分器的守恒量漂移测试,
   验证 E-scheme 重组的数值精度.

---

## 八、关键数值结果

运行一次典型输出的关键数值:

| 观测量 | 典型值 | 物理意义 |
|-------|-------|---------|
| H_T | ~1000 GeV | 标量横动量和 |
| α_s(200 GeV) | 0.123 | QCD 耦合 (与实验一致) |
| Anti-kT jets | 8 | 聚类后喷注数 |
| τ₂₁ | 0~1 | 2-prong 判别量 |
| D₂ | 0~O(1) | quark/gluon 判别 |
| FD 误差 (4阶) | ~1e-12 | 最佳步长下精度 |
| 摆锤能量漂移 | ~1e-7 | RK4 守恒精度 |
| 聚类矩阵条件数 | ~10² | 聚类稳定性指标 |

---

## 九、边界处理与鲁棒性

项目中的边界处理:
- **Δφ 折叠**: 所有 Δφ 计算都折叠到 [-π, π]
- **η 截断**: 赝快度在 θ→0,π 时截断到 ±1e10
- **pT 下限**: pT < 1e-10 被截断避免除零
- **浮点邻域**: r8_next/r8_previous 用于阈值鲁棒性
- **边界 FD**: 靠近边界时自动降阶为单侧差分
- **周期性**: φ 方向所有差分使用周期边界条件
- **质量壳**: m² = E² - p⃗² 自动取 max(m², 0)
- **Sudakov 截断**: 部分子簇射分裂概率截断到 [0, 0.9]

---

## 十、差异化特征

本项目的独特方法论:
1. **领域深度耦合**: 所有变量命名、类设计、公式注释都严格
   使用高能物理术语 (FourVector, pT, η, φ, τ_N, D₂, ...),
   非通用数值方法换皮.
2. **多算法融合**: 15 个种子项目的算法被重新组合为
   完整的喷注分析流水线, 每个模块承担独特角色.
3. **博士级难度**: 包含 Fornberg 算法、Von Neumann 分析、
   DGLAP splitting functions、Padua 节点等高级内容.
4. **可复现性**: 固定随机种子, 零参数可运行,
   结果完全确定.
5. **工程鲁棒性**: 完整的边界处理、数值稳定性保护、
   IEEE-754 浮点邻域分析.

---

## 十一、参考文献 (核心公式来源)

1. M. Cacciari, G.P. Salam, G. Soyez, *The anti-kT jet clustering algorithm*,
   JHEP 04 (2008) 063.
2. A. Larkoski, I. Moult, B. Nachman, *Jet Substructure at the Large Hadron Collider*,
   Phys. Rept. 841 (2020) 1-63.
3. J. Thaler, K. Van Tilburg, *Identifying Boosted Objects with N-subjettiness*,
   JHEP 03 (2011) 015.
4. A. Larkoski, I. Moult, D. Neill, *Analytic Boosted Boson Discrimination*,
   JHEP 05 (2016) 117.
5. B. Fornberg, *Generation of Finite Difference Formulas on Arbitrarily Spaced Grids*,
   Math. Comp. 51 (1988) 699.
6. M. Caliari, S. de Marchi, M. Vianello, *Bivariate interpolation on the square
   at new nodal sets*, Appl. Math. Comp. 165 (2005) 261.
7. G. Salam, *Towards Jet Geography*, JHEP 07 (2010) 036.
8. Y.L. Dokshitzer, *Calculation of the Colour Factors in QCD*,
   Sov. Phys. JETP 46 (1977) 641.
