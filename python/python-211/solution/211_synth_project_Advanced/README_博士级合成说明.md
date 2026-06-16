# PROJECT_211: 多尺度无约束非线性优化
## 量子-经典混合能量景观全局寻优 —— 博士级科学计算项目

**项目领域**: 数学优化 · 无约束非线性优化  
**难度级别**: 博士级  
**语言**: Python 3.x  
**依赖**: numpy, scipy  
**种子项目**: 15 个科研代码融合

---

## 一、项目概述

本项目围绕**无约束非线性优化**这一核心数学领域,融合 15 个种子项目的核心算法,构造了一个前沿博士级科学计算问题:

> **问题定义**: 给定由量子光学态密度、关联随机场、Chebyshev 代理模型共同定义的复杂多模态目标函数 V(x), 使用一系列高级优化方法 (BFGS, L-BFGS, 延拓法, ODE 梯度流, 模拟退火, 代理优化) 找到全局极小点.

该问题涵盖了:
- **非凸多模态优化** (能量景观含大量局部极小)
- **病态条件优化** (Hessian 条件数可达 10⁴ 以上)
- **昂贵函数优化** (单次评估成本高, 需代理模型)
- **随机地景优化** (相关随机场引入的不确定性)
- **物理约束优化** (量子光学 + 种群动力学中的守恒律)

---

## 二、种子项目到科学问题的映射

| # | 种子项目 | 核心算法 | 在本项目中的角色 |
|---|---------|---------|----------------|
| 1 | 1286_Shukti042_dreamrelation | LoRA权重加载, 余弦相似度 | 特征匹配优化目标 (cosine_similarity_loss) |
| 2 | 548_human_mesh2d | 2D三角网格, Delaunay | 优化域离散化 (delaunay_triangulation_2d) |
| 3 | 210_continuation | 延拓法, Newton校正 | 同伦路径跟踪 (continuation.py) |
| 4 | 1242_ce335805_PhotonDosReference | 光子态密度, 介电函数 | 量子光学势 (DielectricModel, SPhP) |
| 5 | 382_fem_to_xml | FEM网格序列化 | 优化域数据表示 (mesh_to_dict/json) |
| 6 | 220_correlation | 相关函数族, FFT采样 | 随机场构造 (CorrelatedRandomField) |
| 7 | 163_chebyshev_series | Chebyshev级数, Clenshaw | 代理模型核心 (chebyshev_surrogate.py) |
| 8 | 881_polpak | 特殊函数 (Bernoulli, Beta, AGM) | 数学基础库 (special_functions.py) |
| 9 | 572_ill_bvp | 病态问题处理 | 收敛诊断 (condition_number) |
| 10 | 764_midpoint | 隐式中点法 | 辛优化积分器 (implicit_midpoint) |
| 11 | 1363_tsp_brute | 排列枚举 | 全局搜索 (brute_force_tsp) |
| 12 | 234_cube_integrals | 单位立方体积分 | Monte Carlo验证 (monte_carlo_integral) |
| 13 | 337_eros | 高斯消元, PLU | 线性代数核心 (GaussianElimination) |
| 14 | 1434_zombie_ode | ODE守恒量 | 种群动力学优化 (PopulationDynamics) |
| 15 | 371_fem_basis | 重心基函数 | 有限元基 (fem_basis_1d/2d/3d/md) |

---

## 三、核心数学公式

### 3.1 优化基本框架

**无约束优化问题**:
```
min_{x∈ℝⁿ} f(x)
```

**一阶必要条件** (Fermat): `∇f(x*) = 0`  
**二阶充分条件**: `∇f(x*) = 0` 且 `∇²f(x*)` 正定

### 3.2 BFGS 拟 Newton 更新

```
H_{k+1} = (I - ρ_k s_k y_kᵀ) H_k (I - ρ_k y_k s_kᵀ) + ρ_k s_k s_kᵀ
```
其中 `s_k = x_{k+1} - x_k`, `y_k = ∇f_{k+1} - ∇f_k`, `ρ_k = 1/(y_kᵀ s_k)`

### 3.3 Wolfe 线搜索条件

- 充分下降: `f(x_k + αp_k) ≤ f(x_k) + c₁α∇f_kᵀ p_k`
- 曲率条件: `|∇f(x_k + αp_k)ᵀ p_k| ≤ c₂|∇f_kᵀ p_k|`

### 3.4 Chebyshev 级数与 Clenshaw 算法

```
f(x) ≈ Σ c_k T_k(x),  x ∈ [-1,1]
c_k = (2/n) Σ f(x_j) T_k(x_j),  x_j = cos(π(j+½)/n)

Clenshaw 递推:
  b_{N+2} = b_{N+1} = 0
  b_k = 2x·b_{k+1} - b_{k+2} + c_k,  k = N, ..., 1
  f(x) = x·b_1 - b_2 + c_0
```

### 3.5 同伦延拓

```
H(x,t) = (1-t)·G(x) + t·F(x),  t ∈ [0,1]
```
从 `t=0` (易解问题 G) 跟踪到 `t=1` (目标问题 F).

预测步 (Euler): `x₁ = x₀ + h·T₀`, T₀ 为切向量 (通过 SVD 求 null(J))  
校正步 (Newton): 解 `H(x, t_k) = 0`, 固定参数分量

### 3.6 隐式中点法 (辛积分)

```
y_{n+1} = y_n + h·f(t_n + h/2, (y_n + y_{n+1})/2)
```
性质: 二阶精度 O(h²), 辛几何 (保持 Hamilton 结构), A-稳定

### 3.7 梯度流 ODE

```
ẋ = -∇f(x)  ⇒  d/dt f(x(t)) = -‖∇f(x)‖² ≤ 0
```

### 3.8 量子光学势

```
ε(ω) = ε_∞ · (ω²_LO - ω²) / (ω²_TO - ω²)     (Lorentz 介电函数)
ω_SPhP = √((ε_∞ ω²_LO + ω²_TO) / (ε_∞ + 1))  (SPhP 频率)
ρ_surf(ω,z) = (ω²_LO - ω²_TO)/(8π ω²_LO z³)   (近场 DOS)
```

### 3.9 相关函数族

```
球面:  C(ρ) = 1 - 1.5·ρ̂ + 0.5·ρ̂³,  ρ̂ = min(|ρ|/ρ₀, 1)
指数:  C(ρ) = exp(-|ρ|/ρ₀)
高斯:  C(ρ) = exp(-ρ²/(2ρ₀²))
Matérn: C(ρ) = 2^{1-ν}/Γ(ν) · (√(2ν)|ρ|/ρ₀)^ν · K_ν(...)
Bessel: C(ρ) = J₀(|ρ|/ρ₀)
```

---

## 四、项目结构

```
211_synth_project_Advanced/
├── main.py                    # 统一入口 (14 个演示模块)
├── special_functions.py       # 特殊函数库 (Bernoulli, Chebyshev, AGM, ...)
├── mesh_basis.py              # 有限元网格与基函数
├── correlation_field.py       # 相关函数与随机场
├── linalg_solver.py           # 高斯消元、PLU、共轭梯度
├── quasi_newton.py            # BFGS, L-BFGS, Wolfe 线搜索
├── continuation.py            # 延拓法、同伦全局优化
├── ode_optimizers.py          # ODE 梯度流 (中点、RK4、Heavy-Ball、Nesterov)
├── quantum_objective.py       # 量子光学目标 + 标准测试函数
├── global_search.py           # 全局搜索 (LHS, SA, 多起始点)
├── chebyshev_surrogate.py     # Chebyshev 代理模型
├── test_problems.py           # 标准测试问题集
└── README_博士级合成说明.md   # 本文件
```

**共 12 个 .py 文件 + 1 个 README**

---

## 五、各模块详解

### 模块 1: 特殊函数库 (`special_functions.py`)
- Chebyshev 多项式 T_n, U_n (递推求值)
- Chebyshev 级数 Clenshaw 算法 (0/1/2 阶导数)
- Bernoulli 数 B_n (递推)
- Beta 函数 B(a,b) (通过对数 Gamma)
- AGM 算术几何平均 (Gauss 迭代)
- 完全椭圆积分 K(k), E(k)
- Lerch 超越函数 Φ(z,s,a)
- Pochhammer 符号、Hermite 多项式、Möbius 函数、除数函数 σ(n)

### 模块 2: 有限元网格与基函数 (`mesh_basis.py`)
- 2D 边界点生成 (圆形/方形/人体轮廓)
- Delaunay 三角剖分 (scipy 或 Bowyer-Watson)
- 三角形面积、雅可比、重心坐标
- 1D/2D/3D/MD Lagrange 基函数
- 网格 JSON 序列化
- 三角形 Gauss 求积规则 (1/3/6/7 点)
- 均匀网格细化 (中点细分)

### 模块 3: 相关函数与随机场 (`correlation_field.py`)
- 7 种相关函数: 球面、线性、指数、高斯、Matérn、幂律、Bessel
- Toeplitz 相关矩阵构造
- Cholesky 采样 (O(n³))
- FFT 循环嵌入采样 (O(n log n), Dietrich & Newsam)
- CorrelatedRandomField 对象 (求值 + 梯度)

### 模块 4: 线性代数核心 (`linalg_solver.py`)
- GaussianElimination 类 (带部分主元)
- 前向消元 + 回代
- 行列式、逆矩阵计算
- PLU 分解
- 条件数估计 (∞-范数和 2-范数)
- Cholesky 分解 (SPD)
- 共轭梯度法 (Hestenes-Stiefel)
- Hessian 修正 (特征值修正)

### 模块 5: 拟 Newton 法 (`quasi_newton.py`)
- Armijo 回溯线搜索
- 强 Wolfe 线搜索 (Nocedal-Wright Alg. 3.5)
- 纯 Newton 法 (含 Hessian 修正)
- BFGS 秩-2 更新
- L-BFGS 两循环递归 (O(mn) 存储)
- 梯度下降 (最速下降)
- 余弦相似度损失 (源自 1286)
- 收敛诊断与收敛阶估计

### 模块 6: 延拓法 (`continuation.py`)
- 切向量计算 (SVD null 空间)
- Newton 校正步 (增广系统)
- 单步延拓 (预测-校正)
- 自然参数延拓
- 伪弧长延拓 (处理折返/分岔)
- 同伦全局优化 (多阶段跟踪)

### 模块 7: ODE 优化器 (`ode_optimizers.py`)
- 隐式中点法 (辛积分)
- 显式中点法 (RK2)
- RK4 经典四阶
- Heavy-Ball 优化 (Polyak)
- Nesterov 加速梯度
- ODE 梯度流优化器
- 守恒量监控 (源自 1434)
- PopulationDynamics (SZR 模型)

### 模块 8: 量子光学目标 (`quantum_objective.py`)
- DielectricModel (Lorentz 介电函数)
- 光子体/表面态密度
- Casimir 有效质量
- QuantumOpticalObjective (腔+声子+关联势)
- Rosenbrock, Rastrigin, Ackley 函数及梯度/Hessian
- 单位立方体单调积分、Monte Carlo 积分

### 模块 9: 全局搜索 (`global_search.py`)
- 全排列枚举 (源自 1363)
- TSP 暴力求解
- 拉丁超立方采样 (LHS)
- 多起始点局部优化
- 模拟退火 (Metropolis)
- 随机搜索 + 局部精化
- 粗到细网格搜索

### 模块 10: Chebyshev 代理 (`chebyshev_surrogate.py`)
- ChebyshevSurrogate 一维代理
- MultivariateChebyshevSurrogate 多维张量积
- 自适应代理优化
- 自适应阶数选择
- 截断误差估计

### 模块 11: 标准测试问题 (`test_problems.py`)
- Sphere, Rosenbrock, Rastrigin, Beale, Himmelblau, Powell, Trid, Schwefel
- 精确梯度与 Hessian (部分)
- 测试套件运行器

---

## 六、运行方法

```bash
cd 211_synth_project_Advanced
python main.py
```

**零参数运行**: 不需要任何输入参数, 直接运行 `main.py` 即可执行所有 14 个演示模块.

**预期输出**:
- 特殊函数验证 (精度 1e-14 级别)
- 网格生成与基函数验证
- 随机场采样 (Cholesky + FFT)
- 线性代数测试 (Gauss 消元、PLU、CG)
- 8 个标准测试问题的 BFGS 基准
- 5 种优化器的性能对比
- 同伦全局优化路径
- ODE 梯度流 + 守恒量监控
- 量子光学目标优化
- Rastrigin 2D 全局搜索 (多起始点 + 退火 + 网格)
- Chebyshev 代理优化 (16 次评估达到高精度)
- SZR 种群动力学参数优化
- 余弦相似度特征匹配
- 病态问题收敛分析 (条件数 10 到 10000)

---

## 七、科学问题与应用

本项目解决的核心科学问题:

### 7.1 量子光学腔体参数优化
通过介电函数和光子态密度构造的物理势, 优化腔体参数以最小化系统能量. 应用于:
- 表面极化激元 (SPhP) 共振频率调谐
- Casimir 效应有效质量计算
- 纳米光子腔设计

### 7.2 种群动力学参数辨识
基于 SZR (Susceptible-Zombie-Removed) 模型, 通过优化终态僵尸数量反演传播参数 (α, β, γ, δ). 此框架同样适用于:
- 传染病模型 (SIR/SEIR) 参数拟合
- 生态种群模型参数辨识
- 化学反应动力学速率常数估计

### 7.3 复杂能量景观全局寻优
结合延拓法、多起始点、模拟退火、代理模型等策略, 在含大量局部极小的多模态景观中寻找全局最优. 应用于:
- 分子构象搜索
- 蛋白质折叠能量面
- 材料科学中的相空间探索

---

## 八、算法创新点

1. **多策略融合**: 将局部优化 (BFGS/L-BFGS)、全局探索 (SA/LHS/多起始点)、同伦跟踪 (continuation) 有机结合
2. **辛积分优化**: 利用隐式中点法的辛几何性质, 在梯度流积分中保持能量耗散结构
3. **代理驱动**: Chebyshev 代理模型大幅减少昂贵函数评估 (从 ~1000 次降到 ~16 次)
4. **病态诊断**: 实时监控 Hessian 条件数, 动态修正以确保 Newton 方向下降
5. **物理约束**: 在 SZR 模型中保持守恒量 S+Z+R=const, 保证物理一致性

---

## 九、数值鲁棒性与边界处理

- **除零保护**: 所有除法运算前检查分母 (阈值 1e-300)
- **溢出防护**: 大指数运算使用 `np.clip` 或对数域计算
- **病态系统**: Hessian 非正定时自动修正 (特征值抬升)
- **NaN 处理**: 优化过程中检测 NaN, 自动回退到安全点
- **参数边界**: 通过 `np.clip` 将参数限制在物理合理范围内
- **收敛诊断**: 同时检查梯度范数和函数值变化, 避免假收敛

---

## 十、参考文献与公式来源

1. Nocedal & Wright, *Numerical Optimization*, 2nd ed., Springer, 2006.
2. Clenshaw, *Chebyshev series for mathematical functions*, 1962.
3. Maess, *Vorlesungen ueber Numerische Mathematik II*, 1984.
4. Dietrich & Newsam, *Fast and exact simulation of stationary Gaussian processes*, SIAM J. Sci. Comput., 1997.
5. Munz et al., *When Zombies Attack!: Mathematical Modelling*, 2009.
6. Su, Boyd, Candès, *A differential equation for Nesterov's acceleration*, 2014.
7. Polyak, *Some methods of speeding up the convergence of iteration methods*, 1964.
8. Abrahamsen, *A Review of Gaussian Random Fields and Correlation Functions*, 1997.
9. Burkardt et al., *POLPAK, TEST_TRI, FEM_BASIS* 等数值库.

---

## 十一、扩展方向

- 加入 **L-BFGS-B** 处理有约束优化
- 引入 **信赖域方法** (trust region)
- 扩展 **随机优化** (SGD, Adam, 自然梯度)
- 实现 **贝叶斯优化** (高斯过程代理)
- 加入 **自动微分** (JAX/PyTorch) 提高梯度精度
- 实现 **分布式优化** (MPI/CUFI)

---

**项目完成日期**: 2026-06-07  
**作者**: 自动合成系统  
**许可**: 仅供学术研究与教学使用
