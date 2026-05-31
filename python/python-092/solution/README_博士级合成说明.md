# 室内声场射线追踪与模态分析 —— 博士级合成说明

## 一、项目概述

本项目围绕**声学工程：室内声场射线追踪与模态分析**展开，融合15个科研代码项目的核心算法，构建了一个面向三维封闭空间声场仿真的**混合射线-有限元模态分析与贝叶斯参数反演**系统。项目解决的核心科学问题是：

> **如何在高维参数不确定性的条件下，对复杂几何封闭空间的声场进行从低频模态控制区到高频扩散区的全频段预测，并通过贝叶斯推断量化吸声参数的后验不确定性？**

项目统一入口为 `main.py`，零参数可直接运行，完成从房间几何定义、网格生成、有限元求解、模态分析、蒙特卡洛射线追踪、边缘衍射计算、反射图构建到贝叶斯参数反演的完整流程。

---

## 二、原项目到科学问题的映射

以下15个输入科研项目均在合成项目中承担**真实算法角色**，无任何挂名：

| 原项目 | 核心算法 | 在合成项目中的科学角色 |
|--------|---------|----------------------|
| `845_pagerank2` | 稀疏邻接图构造 | **反射转移图构建**：将房间表面视为图节点，射线反射视为边，构造稀疏马尔可夫转移矩阵，并求解稳态能量分布（PageRank类比） |
| `402_fem2d_bvp_serene` | 2D FEM serendipity元求解 | **3D Helmholtz FEM求解**：将serendipity元的基函数构造思想升维至3D线性四面体(P1)元，组装刚度矩阵K与质量矩阵M，求解声压场 |
| `1432_zero_rc` | Brent反向通信求根 | **特征频率精细化搜索**：在模态分析中用于搜索特征方程 `det(K - λM) = 0` 的根，实现Brent混合策略（逆二次插值+二分法） |
| `068_ball_integrals` | 单位球单项式精确积分与采样 | **源辐射指向性积分**：利用Gamma函数精确公式计算球面上声源辐射模式的积分；`ball01_sample`用于球形麦克风阵列采样 |
| `998_r8st` | 稀疏COO格式与CG/Jacobi求解器 | **大规模稀疏系统求解**：实现COO格式稀疏矩阵-向量乘法、共轭梯度法(CG)和Jacobi迭代，用于FEM组装后的线性系统求解 |
| `325_edge` | 阶跃/边缘检测测试函数 | **边缘衍射系数建模**：利用edge函数的阶跃不连续思想，在Keller GTD理论中计算边缘绕射系数，处理阴影边界奇点 |
| `1316_triangle_symq_rule` | 三角形对称高阶求积 | **表面声能积分**：在房间三角形表面patch上使用高阶对称求积规则（精度5，7点规则）积分声压与质点速度 |
| `475_gmsh_to_fem` | GMSH网格格式转换 | **网格数据I/O处理**：将DistMesh生成的节点-单元数据以标准文本格式读写，衔接网格生成与FEM求解器 |
| `477_gpl_display` | 网格/曲线数据解析 | **房间表面结构化数据解析**：解析房间各墙面、天花、地板的三角形patch数据，供射线追踪和求积使用 |
| `1351_triangulation_refine_local` | 局部边二分加密 | **边界附近网格自适应加密**：在墙面边界附近（`|fd(p)| < 2h0`）对四面体进行edge bisection细分，提高近壁声压梯度精度 |
| `309_distmesh_3d` | 3D距离函数网格生成 | **房间四面体网格生成**：基于Persson-Strang力平衡松弛算法，使用有向距离函数(SDF)生成非结构化四面体网格 |
| `1097_sobol` | Sobol低差异序列 | **射线方向低差异采样**：用类Sobol序列在球面上均匀采样射线方向，替代纯随机采样以降低蒙特卡洛方差 |
| `989_r8po` | SPD矩阵Cholesky分解 | **模态分析预处理**：对FEM质量矩阵M进行Cholesky分解 `M = R^T R`，将广义特征值问题化为标准形式 |
| `319_dream` | DREAM差分进化MCMC | **贝叶斯吸声系数反演**：多条链并行演化，差分进化生成候选，Metropolis-Hastings接受，Gelman-Rubin诊断收敛 |
| `683_line_monte_carlo` | 线段上的蒙特卡洛与遍历采样 | **射线参数化路径采样**：使用黄金比例低差异序列沿射线传播路径进行1D参数化采样，用于能量累积计算 |

---

## 三、核心数学物理模型与公式

### 3.1 Helmholtz 方程（声波传播控制方程）

三维时谐声压场满足Helmholtz方程：

```
∇² p + k² p = -f(ω) δ(r - r_s)
```

其中：
- `p(r, ω)` 为复声压 [Pa]
- `k = ω / c = 2πf / c` 为波数 [rad/m]
- `c = 343 m/s` 为空气中声速（20°C）
- `ρ_0 = 1.21 kg/m³` 为空气密度
- `f(ω)` 为声源强度
- `δ(r - r_s)` 为点声源位置

### 3.2 有限元弱形式与离散化

乘以测试函数 `q` 并在域 `Ω` 上积分，应用Green第一恒等式：

```
∫Ω ∇p · ∇q dV - k² ∫Ω p q dV - ∫Γ (q ∂p/∂n) dS = ∫Ω f q dV
```

采用P1四面体元离散 `p ≈ Σ N_i(x) p_i`，得到线性系统：

```
(K - k² M) p = F
```

刚度矩阵与质量矩阵的单元贡献：

```
K_{ij}^{(e)} = ∫Ω_e ∇N_i · ∇N_j dV = V_e · (∇N_i)·(∇N_j)
M_{ij}^{(e)} = ∫Ω_e N_i N_j dV = V_e / 20  (i≠j),  V_e / 10  (i=j)
```

其中 `V_e = |det(J)| / 6` 为四面体体积，`J = [v_1-v_0, v_2-v_0, v_3-v_0]` 为Jacobian。

### 3.3 模态分析与特征值问题

封闭空间的声学模态满足广义特征值问题：

```
K φ = λ M φ,   λ = (ω/c)²
```

对于长方体房间（刚性壁，Neumann边界），解析解为：

```
f_{l,m,n} = (c/2) · √((l/L_x)² + (m/L_y)² + (n/L_z)²)
```

**Schroeder频率**（模态区与扩散区的分界）：

```
f_s = 2000 · √(T60 / V)   [Hz]
```

**模态密度**（单位频率内的模态数）：

```
n(f) = 4πVf² / c³ + πSf / (2c²) + L / (8c)
```

**模态重叠因子**：

```
MOF = n(f) · η · f
```

当 `MOF > 1` 时，声场进入扩散区。

### 3.4 Sabine & Eyring 混响时间

**Sabine公式**（统计声学）：

```
T_{60} = 0.161 · V / A
A = Σ S_i α_i
```

**Eyring公式**（更精确）：

```
T_{60} = 0.161 · V / (-S · ln(1 - α_{avg}))
```

### 3.5 射线追踪与能量衰减

射线传播方程：

```
r(t) = r_0 + c · t · d̂
```

镜面反射定律：

```
d̂_{ref} = d̂ - 2(d̂ · n̂) n̂
```

能量衰减：

```
E_{n+1} = E_n · (1 - α_i)
```

**平均自由程**：

```
Λ = 4V / S
```

### 3.6 边缘衍射（Keller GTD）

衍射射线位于**Keller锥**上，满足：

```
d̂_{dif} · t̂ = ± d̂_{inc} · t̂
```

其中 `t̂` 为边缘切向单位向量。

硬边衍射系数（简化模型）：

```
D = -e^{-iπ/4} / (2n√(2πk)) · [cot((π+Φ)/2n) + cot((π-Φ)/2n)]
```

其中 `Φ = φ_{inc} + φ_{dif}`，`n=2`（直边）。

### 3.7 贝叶斯参数反演

**贝叶斯定理**：

```
p(α | D) ∝ p(D | α) · p(α)
```

**似然函数**（高斯噪声假设）：

```
ln L = -0.5 · Σ_i ((T_{60}^{pred}(α) - T_{60,i}^{obs}) / σ_i)² + const
```

**DREAM差分进化候选生成**：

```
z_p = z_i + (1 + γ) · J · Σ_{k=1}^{δ} (z_{a_k} - z_{b_k}) + ε
```

**Gelman-Rubin收敛诊断**：

```
R = √((W + B/n) / W)
```

`W` 为链内方差，`B` 为链间方差，当 `R < 1.2` 时认为收敛。

### 3.8 单位球单项式精确积分

基于Folland公式，单位球上单项式的精确体积分：

```
I = 2 · Γ((e_1+1)/2) · Γ((e_2+1)/2) · Γ((e_3+1)/2) / Γ(Σ(e_i+1)/2) · r^s / s
```

其中 `s = Σ e_i + 3`。若任一指数为奇数，则 `I = 0`（对称性）。

### 3.9 三角形对称求积

单位三角形上的高阶对称求积（Stroud规则，精度5，7点）：

```
∫_T f dA ≈ |T| · Σ_{i=1}^{7} w_i f(ξ_i)
```

---

## 四、文件结构与功能

```
092_synth_project/
├── main.py                     # 统一入口，零参数运行
├── room_geometry.py            # 房间SDF、表面提取、法向量、面积计算
├── mesh_generator.py           # 3D DistMesh生成、局部加密、质量评估
├── sparse_linalg.py            # COO稀疏矩阵、CG/Jacobi求解、Cholesky分解
├── fem_acoustics.py            # Helmholtz FEM组装、求解、SPL/声强计算
├── modal_analysis.py           # 解析模态、逆迭代、Rayleigh商、模态密度
├── quadrature_rules.py         # 三角形求积、球面积分、线段低差异采样
├── ray_tracer.py               # Sobol射线采样、追踪、EDC/T60、反射图
├── edge_diffraction.py         # Keller锥、衍射系数、房间边缘检测
├── bayesian_calibration.py     # DREAM MCMC、贝叶斯反演、不确定性传播
├── utils.py                    # 工具函数：统计、回归、安全除法
└── README_博士级合成说明.md    # 本文档
```

---

## 五、运行方式

```bash
cd Synthesis-project-python/092_synth_project
python main.py
```

程序将自动执行以下流程并输出结果：
1. 房间几何定义与表面积计算
2. 四面体网格生成（~434节点，~296四面体）
3. 125Hz FEM声压场求解
4. 前10阶解析模态频率与Schroeder频率
5. 2000条射线蒙特卡洛追踪与T60估计
6. 12条边缘的GTD衍射场计算
7. 6表面反射转移图与稳态分布
8. DREAM MCMC贝叶斯吸声系数反演
9. T60预测的不确定性传播
10. 球面积分与三角形求积验证

---

## 六、科学难度与工程鲁棒性

### 6.1 科学难度
- **跨频段建模**：同时处理低频模态区（FEM）与高频扩散区（射线追踪），覆盖Schroeder频率上下
- **高阶数值方法**：P1 FEM + 高阶三角形求积 + 低差异序列蒙特卡洛 + CG迭代求解
- **多物理场耦合**：声波传播、壁面反射/散射、边缘衍射、能量衰减的统一框架
- **贝叶斯不确定性量化**：使用DREAM MCMC进行参数反演，输出后验分布与置信区间

### 6.2 工程鲁棒性
- **边界处理**：所有除法均检查零值；三角函数在奇点附近添加正则化（eps=0.1）
- **数值稳定性**：FEM矩阵添加 `1e-10` 正则化；CG求解设置残差容差；Cholesky分解检查正定性
- **参数约束**：贝叶斯采样使用折叠法（folding）限制吸声系数在 `[0.01, 0.99]`
- **退化检测**：四面体体积/质量检查；Jacobian行列式检查；模态频率非负检查

---

## 七、修改记录

| 修改项 | 说明 |
|--------|------|
| 语言迁移 | 所有15个原项目（MATLAB）迁移至Python 3 |
| 维度升级 | `fem2d_bvp_serene`（2D Q8元）→ `fem_acoustics.py`（3D P1四面体元） |
| 升维重构 | `distmesh_3d`（通用3D网格）→ 带柱子shoebox房间专用几何 |
| 算法融合 | `sobol` + `line_monte_carlo` → 球面方向采样 + 路径参数化采样 |
| 图论应用 | `pagerank2` 的稀疏图构造 → 声学反射转移概率矩阵 |
| MCMC反演 | `dream` 的差分进化 → 吸声系数贝叶斯校准 |
| 删除可视化 | 所有plot、trimesh、patch等可视化代码已移除 |

---

## 八、参考文献与理论基础

1. **Helmholtz方程有限元**: Ihlenburg, F. *Finite Element Analysis of Acoustic Scattering*. Springer, 1998.
2. **Sabine混响理论**: Sabine, W.C. *Collected Papers on Acoustics*. Harvard University Press, 1922.
3. **Keller几何衍射理论**: Keller, J.B. "Geometrical Theory of Diffraction." *J. Opt. Soc. Am.* 52(2), 1962.
4. **DistMesh算法**: Persson, P.O. & Strang, G. "A Simple Mesh Generator in MATLAB." *SIAM Review* 46(2), 2004.
5. **DREAM MCMC**: Vrugt, J.A. et al. "Accelerating Markov Chain Monte Carlo Simulation by Differential Evolution." *Int. J. Nonlin. Sci. Num.* 10(3), 2009.
6. **Sobol序列**: Antonov, I.A. & Saleev, V.M. "An Economic Method of Computing LPτ-Sequences." *USSR Comput. Math. Math. Phys.* 19, 1979.
7. **三角形对称求积**: Xiao, H. & Gimbutas, Z. "A Numerical Algorithm for the Construction of Efficient Quadrature Rules in Two and Higher Dimensions." *Computers & Mathematics with Applications* 59, 2010.
8. **Folland球面积分**: Folland, G.B. "How to Integrate a Polynomial Over a Sphere." *American Mathematical Monthly* 108(5), 2001.
