# 博士级合成说明：环形域上空间耦合反应-扩散系统的贝叶斯层次推断

## 1. 项目概述

本项目围绕 **数据科学：贝叶斯推断与 MCMC 采样** 领域，将 15 个种子科研代码项目的核心算法融合为一个前沿博士级自然科学计算问题：

> **环形心肌组织切片上空间耦合 FitzHugh-Nagumo 反应-扩散系统的贝叶斯层次校准**

我们在二维环形域（annulus）上建立了一个耦合物理模型：
- **局部反应**：每个空间位置服从 FitzHugh-Nagumo（FHN）可激发动媒体模型，描述细胞膜电位动力学；
- **空间传播**：Helmholtz 波动方程的精确 Bessel 解提供稳态空间调制场；
- **空间异质性**：电导率（刺激强度）通过 FEM 拉格朗日基函数在参考三角形上展开，形成角向异质场；
- **推断目标**：从带噪声的稀疏空间观测中，利用自定义 MCMC 采样器联合推断 FHN 参数、Helmholtz 振幅、空间场系数及观测噪声方差。

---

## 2. 种子项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 在合成项目中的真实角色 |
|:---:|:---|:---|:---|
| 1 | `371_fem_basis` | 参考三角形上任意次 Lagrange 基函数 | `spatial_domain.py`：利用 Kronecker-delta 性质的二维 Lagrange 多项式 `L_{i,j,k}(x,y)` 对环形扇区的空间刺激场进行谱展开 |
| 2 | `651_latin_edge` | 拉丁超立方边缘采样 | `utils.py`：为 MCMC 链的 4 维核心参数提供分层初始化，避免随机聚集 |
| 3 | `1149_square_monte_carlo` | 单位正方形上的 Monte Carlo 积分 | `bayesian_quadrature.py`：对参数空间中的 `(a,b)` 二维切片进行 Bayes 模型证据的随机积分 |
| 4 | `1327_triangle01_monte_carlo` | 单位三角形上的 Monte Carlo 积分 | `bayesian_quadrature.py`：对 Dirichlet/单纯形约束的混合权重进行面积加权的 Monte Carlo 积分 |
| 5 | `008_annulus_grid` | 环形域网格生成（笛卡尔/Fibonacci） | `spatial_domain.py`：在 `[r1,r2]` 环形域内生成极坐标/Fibonacci 黄金角螺旋观测网格 |
| 6 | `047_asa183` | Wichmann-Hill 与 L'Ecuyer 组合 PRNG | `prng_asa183.py`：为 Metropolis-Hastings 接受-拒绝步、Gibbs 更新、Box-Muller 正态采样提供独立随机流 |
| 7 | `178_circle_distance` | 单位圆上弦长分布的几何概率 | `spatial_domain.py`/`utils.py`：计算圆上弦长 `pdf(d)=1/(π√(1-d²/4))` 的精确均值 `4/π` 与方差 `2-16/π²`，用于验证空间相关核的解析性质 |
| 8 | `515_helmholtz_exact` | 圆膜 Helmholtz 方程精确分离变量解 | `forward_models.py`：通过 Bessel 函数 `J_n` 的零点与 Halley 迭代求根，计算 `Z(r,θ)=γ·J_n(ρ·r/a)·(α·cos(nθ)+β·sin(nθ))` |
| 9 | `964_r83p` | 周期三对角矩阵的 Schur 补直接求解 | `periodic_solver.py`：对环形图上的 Gauss-Markov 随机场（GMRF）精度矩阵 `Q` 进行 R83P 格式的 LU 分解，批量求解 `Q·Σ=I` 得到精确协方差 |
| 10 | `683_line_monte_carlo` | 一维 Monte Carlo 与黄金比率遍历采样 | `bayesian_quadrature.py`：对 `γ` 参数进行一维边缘似然积分，分别使用均匀随机和黄金比率低差异序列 `(shift+j·φ) mod 1` |
| 11 | `435_fitzhugh_nagumo_ode` | FHN 可激发动媒体 ODE 右端项 | `forward_models.py`：定义 `dv/dt = v-v³/3-w+d`, `dw/dt=(v+a-bw)/c` 的向量场 |
| 12 | `657_least_squares_approximant` | SVD 阈值伪逆最小二乘多项式逼近 | `surrogate.py`：构建 Vandermonde 矩阵 `V`，通过 `lstsq` 与 SVD 正则化（截断小奇异值）拟合 6 次多项式代理模型，加速 FHN 稳态电压评估 |
| 13 | `343_euler` | 前向 Euler 显式时间步进 | `forward_models.py`：`y_{i+1}=y_i+Δt·f(t_i,y_i)` 用于从静息态 `(0,0)` 积分 FHN 系统至稳态 |
| 14 | `133_calendar_nyt` | Julian 日期与 NYT 卷期号的复杂历法转换 | `utils.py`：为合成实验批次生成基于 Julian Ephemeris Date 的历史卷期索引，保留 33 条历法修正的简化核心逻辑 |
| 15 | `1372_unicycle` | 单圈置换（unicycle）的字典序枚举 | `utils.py`/`mcmc_sampler.py`：利用单圈置换的循环移位（cyclic shift）作为 MCMC 旋转提案，保证环形域上空间场的旋转不变性 |

---

## 3. 核心数学物理模型与公式

### 3.1 空间域与 FEM 基函数展开

环形域划分为 `n=4` 个角向扇区，观测点位于外缘中点，极角
```
θ_j = 2π·j / n,   j = 0,…,n-1
```
每个扇区映射到参考三角形（顶点 `(0,0),(1,0),(0,1)`）。在外缘中点 `(0.5,0.5)` 处，度数为 `D=1` 的 Lagrange 基函数值为：
```
L_{1,0,0}(0.5,0.5) = 0.5,   L_{0,1,0}(0.5,0.5) = 0.5,   L_{0,0,1}(0.5,0.5) = 0
```
因此第 `j` 个观测点的刺激强度为
```
d_j = d_0 + 0.5·c_j + 0.5·c_{j+1 mod n}
```
其中 `c=(c_0,…,c_{n-1})` 为 FEM 基权重向量。

### 3.2 FitzHugh-Nagumo 反应动力学

局部膜电位 `y=(v,w)` 满足
```
dv/dt = v - v³/3 - w + d_j
dw/dt = (v + a - b·w) / c_param
```
采用显式 Euler 方法从静息初值 `y(0)=(0,0)` 积分至 `t=20`，步长 `Δt=20/30`。稳态电压记为 `v_∞(a,b,c_param,d_j)`。

### 3.3 Helmholtz 波动方程精确解

圆膜半径 `a_disk=1.0`，边界条件 `Z(a_disk,θ)=0`。分离变量后得到
```
Z(r,θ) = γ · J_n( ρ(m,n)·r / a_disk ) · [ α·cos(nθ) + β·sin(nθ) ]
```
其中 `ρ(m,n)` 为 `J_n` 的第 `m` 个正零点，通过 Halley 迭代求得：
```
x_{k+1} = x_k - 2·a·x_k·(n·a - b·x_k) / [ 2·b²·x_k² - a·b·x_k·(4n+1) + (n(n+1)+x_k²)·a² ]
```
这里 `a=J_n(x_k)`, `b=J_{n+1}(x_k)`。初始猜测由对 `n∈[0,10000]` 的最小二乘拟合公式提供。

### 3.4 前向模型与似然

第 `j` 个观测点的预测值为
```
μ_j(θ) = v_∞(a,b,c_param,d_j) + γ · J_0( ρ(1,0)·r / a_disk )
```
观测模型为独立高斯噪声
```
y_j | θ ~ N( μ_j(θ), σ² )
```
对数似然
```
ℓ(θ) = - (n_obs/2)·log(2πσ²) - (1/(2σ²))·Σ_{j=1}^{n_obs} (y_j - μ_j(θ))²
```

### 3.5 GMRF 先验与周期三对角求解

空间系数 `c` 的先验为环形图上的高斯马尔可夫随机场，精度矩阵 `Q` 为周期三对角（R83P）格式：
```
Q_{ii} = 2τ + δ,   Q_{i,i+1}=Q_{i,i-1}=-τ,   Q_{1,n}=Q_{n,1}=-τ
```
其中 `τ=100.0` 为精度参数，`δ=1.0` 为稳定性 nugget。`Q` 以 3×n 的 R83P 数组存储：
```
a[0,0]=A_{n,1},  a[0,1:]=superdiag,  a[1,:]=diag,  a[2,:-1]=subdiag,  a[2,-1]=A_{1,n}
```
通过 Schur 补分解求解 `Q·Σ=I`：
```
Σ = [ A1  A2 ]^{-1} = [ I  -A1^{-1}A2 ] [ A1^{-1}       0       ]
    [ A3  A4 ]        [ 0       I      ] [ 0    (A4-A3·A1^{-1}·A2)^{-1} ]
```
其中 `A1` 为前 `(n-1)×(n-1)` 三对角块，使用非周期三对角分解 `r83_np_fa` / `r83_np_sl` 求解。

### 3.6 多项式代理模型（Surrogate）

为加速似然计算，对固定 `a=0.7,b=0.8` 下的稳态电压 `v_∞(d)` 构建 1D 多项式代理。在 `n_train=15` 个 Latin 超立方采样点上计算精确 Euler 解，然后拟合阶数为 `m=6` 的最小二乘多项式：
```
min_c ‖ V·c - y ‖_2
```
其中 `V` 为 Vandermonde 矩阵，`c` 通过 `lstsq` 与 SVD 伪逆（截断阈值 `√ε`）双重求解以确保数值鲁棒性。

### 3.7 MCMC 采样器

采用 **自适应随机游走 Metropolis-within-Gibbs**：
- **RWMH**：对 `(a,b,γ,d_0,c_0,…,c_3)` 提出高斯随机游走在后验中探索；
- **Gibbs**：`log σ` 的对数尺度条件更新；
- **Unicycle 旋转提案**：每 10 步以 `shift∈{1,2,3}` 对 `c` 做循环移位 `c_j → c_{j+shift mod n}`，保证环形域旋转不变性。

接受概率
```
α = min{ 1,  exp[ log p(θ') + log ℓ(θ') - log p(θ) - log ℓ(θ) ] }
```
随机数由 Wichmann-Hill（周期 `~7×10^{12}`）与 Box-Muller 变换生成。

### 3.8 Bayesian 积分（模型证据）

对边缘似然 `Z = ∫ L(θ)·π(θ) dθ` 进行分域 Monte Carlo 估计：
- **线积分**（一维 `γ`）：黄金比率遍历序列 `x_j = (shift + j·φ) mod 1`，`φ=(1+√5)/2`；
- **正方形积分**（二维 `(a,b)`）：均匀随机采样于 `[0,1]²`；
- **三角形积分**（单纯形权重）：利用指数间距 `e_1,e_2~Exp(1)` 在单位单纯形上均匀采样。

---

## 4. 文件结构与改造路径

| 文件名 | 来源/合成方式 | 核心功能 |
|:---|:---|:---|
| `main.py` | 新建 | 统一零参数入口，执行完整贝叶斯推断管线并输出后验摘要 |
| `prng_asa183.py` | 移植自 `047_asa183` | Wichmann-Hill 与 L'Ecuyer 组合 PRNG，含 Box-Muller 正态生成 |
| `periodic_solver.py` | 移植自 `964_r83p` | 非周期/周期三对角矩阵的 LU 分解与线性求解（R83P 格式） |
| `spatial_domain.py` | 移植自 `008_annulus_grid` + `371_fem_basis` + `178_circle_distance` | 环形网格生成、FEM Lagrange 基函数评估、圆上弦长统计 |
| `forward_models.py` | 移植自 `435_fitzhugh_nagumo_ode` + `343_euler` + `515_helmholtz_exact` | FHN 右端项、Euler 积分、Bessel 零点 Halley 迭代、Helmholtz 精确解 |
| `surrogate.py` | 移植自 `657_least_squares_approximant` | SVD 正则化最小二乘多项式拟合与 1D 代理模型构建 |
| `bayesian_quadrature.py` | 移植自 `683_line_monte_carlo` + `1149_square_monte_carlo` + `1327_triangle01_monte_carlo` | 线/正方形/三角形域上的 Monte Carlo 积分（含黄金比率遍历采样） |
| `utils.py` | 移植自 `651_latin_edge` + `133_calendar_nyt` + `1372_unicycle` + `178_circle_distance` | 拉丁超立方、NYT 历法转换、单圈置换枚举、圆上 Monte Carlo 统计 |
| `mcmc_sampler.py` | 新建（融合多个种子算法） | 自适应 RWMH + Gibbs + Unicycle 旋转提案的完整采样内核 |
| `inference_engine.py` | 新建（融合所有模块） | 数据生成、GMRF 先验构建、代理训练、MCMC 执行、Bayesian 积分、后验汇总 |

**删除的内容**：所有原始种子项目中的可视化代码（`plot`、`display`、`histogram` 等）均已删除，仅保留核心数值算法。

---

## 5. 合成项目解决的科学问题

本项目解决的是一个 **心脏电生理学中的逆问题**：

> 给定环形心肌组织切片上的稀疏、带噪声电压观测，如何联合推断局部细胞膜动力学参数（FHN）、稳态波传播模式（Helmholtz 振幅）以及空间异质性刺激场（FEM 基系数）？

具体科学价值包括：
1. **多物理场耦合建模**：将反应动力学（FHN）与波动方程（Helmholtz）在几何非平凡的环形域上耦合；
2. **空间统计先验**：利用周期图结构上的 GMRF 对空间异质性施加平滑性先验，并通过周期三对角直接求解器高效计算；
3. **计算加速**：多项式代理模型将大量前向 ODE 积分替换为多项式求值，显著降低 MCMC 计算成本；
4. **旋转不变性**：通过 unicycle 置换提案保证推断结果不依赖于角向观测坐标系的选择；
5. **模型证据计算**：利用不同域上的 Monte Carlo 积分对竞争假设（不同参数子空间）进行定量比较。

---

## 6. 运行方式

```bash
cd Synthesis-project-python/182_synth_project
python main.py
```

无需任何命令行参数。`main.py` 自动完成：
1. 合成数据生成（8 个角向观测点）；
2. GMRF 先验与周期三对角协方差计算；
3. 6 次多项式代理模型训练；
4. Latin 超立方初始化 + 500 步自适应 MCMC；
5. 1D/2D/单纯形 Bayesian 积分；
6. 后验均值、标准差、模型证据输出。

运行时间约 5–15 秒（取决于 CPU）。

---

## 7. 数值鲁棒性与边界处理

- **PRNG 种子边界**：Wichmann-Hill 种子严格限制在 `[1,30268]×[1,30306]×[1,30322]`，L'Ecuyer 种子限制在 `[1,2147483562]×[1,2147483398]`；
- **FEM 基函数**：对 `cijk=0` 的情况返回 `0.0`，避免除零；
- **Bessel 零点**：Halley 迭代设置最大步数 `100` 与相对容差 `1e4·eps`，不收敛时保留最后迭代值；
- **周期三对角**：在 Schur 补 `work4` 上检查 `|work4|<1e-15`，防止奇异矩阵导致崩溃；
- **GMRF 稳定性**：在精度矩阵对角线上加入 `δ=1.0` 的 nugget，消除零特征值；
- **积分鲁棒性**：线/正方形/三角形采样均对除零（如指数间距和为 0）做 `max(s, 1e-15)` 截断；
- **MCMC 边界**：所有参数均有硬边界（Uniform 截断），`log σ` 限制在 `σ∈[1e-12,10]`；
- **代理模型外推**：对 `d` 做 `clamp` 到训练区间 `[-0.3,0.3]`，避免 Runge 现象导致的外推爆炸。

---

## 8. 结论

本项目将 15 个独立科研代码项目（涵盖随机数生成、有限元、Monte Carlo、ODE、PDE、线性代数、最小二乘、历法、组合数学）有机融合为一个统一的前沿科学计算框架。所有原始算法均承担真实数值角色，代码可直接零参数运行，具备完整的边界处理、数值鲁棒性与丰富的数学物理公式体系。
