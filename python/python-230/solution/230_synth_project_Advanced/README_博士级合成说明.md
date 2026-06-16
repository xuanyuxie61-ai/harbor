# PROJECT 230: 计算高能物理 Profile Likelihood 与 CLs 上限设定

## 高阶有限差分与稳定性分析 (小规模可复现实验)

### 博士级科研合成项目

本项目将 15 个跨领域科研种子项目的核心算法融合为一个完整的
**高能物理统计推断框架**, 聚焦于 LHC 型粒子物理实验中的
**Profile Likelihood Ratio 检验** 与 **CLs 上限设定** 的数值实现。

---

## 科学问题描述

在 LHC 等高能物理实验中, 寻找新物理信号 (如 Higgs 玻色子、超对称粒子)
时, 需要从大量背景事件中区分微弱的信号贡献。核心统计问题:

1. **信号强度上限设定**: 给定观测数据, 求信号强度 μ 的 95% 置信水平上限
2. **Profile Likelihood**: 在高维 nuisance 参数空间中最大化似然函数
3. **有限差分数值稳定性**: 高阶有限差分在不同步长下的行为分析
4. **可复现实验**: 小规模实验设计, 使所有数值结果可被独立验证

### 数学模型

**Poisson × Gaussian 约束似然**:

```
L(μ, θ) = Π_i Poisson(n_i | ν_i(μ, θ)) · Π_k Gaussian(θ_k | 0, 1)

其中:
  ν_i(μ, θ) = μ · s_i(θ) + b_i(θ)
  s_i(θ) = ε_i(a) · s_i^0 · (1 + Σ_k γ_{ik} θ_k)
  b_i(θ) = b_i^0 · Π_k (1 + δ_{ik} θ_k)^{κ_{ik}}
```

**Profile Likelihood Ratio**:

```
λ(μ) = L(μ, θ̂_μ) / L(μ̂, θ̂)
q_μ  = -2 ln λ(μ)   (one-sided)
```

**CLs 比率**:

```
CLs(μ) = p_{s+b}(μ) / (1 - p_b(μ))
       = [1 - Φ(√q_μ + μ/σ)] / Φ(√q_μ)
```

上限条件: CLs(μ_up) = 0.05 (95% CL)

---

## 种子项目融合映射

| # | 种子项目 | 原领域 | 在本项目中的角色 |
|---|---------|-------|------------------|
| 1 | 1220 (Unruh Effect) | 黑洞热力学 | 探测器效率的 Unruh 温度修正: ε(a) = ε₀[1 - (T_U/T_beam)²c₂ + (T_U/T_beam)⁴c₄] |
| 2 | 846 (Paraheat FEM) | 参数化热传导 | Nuisance 参数空间的有限元积分: 分片线性基函数 + Gauss 求积 |
| 3 | 1171 (MAGNet) | 算子学习 | 网格采样式 profiling 初始化策略 |
| 4 | 1297 (FormalCellular) | 形式化验证 | CLs 在离散 μ 网格上的 CDF 式扫描 |
| 5 | 1020 (SoleFlip) | 量化攻击 | 自适应蒙特卡洛采样 (量化退火策略) |
| 6 | 752 (Mesh Bandwidth) | 有限元网格 | Profiling 依赖图的稀疏带宽分析 |
| 7 | 285 (Digraph Adj) | 图论 | Nuisance 依赖图的邻接矩阵 → 连通分量分解 |
| 8 | 658 (Lebesgue) | 数值分析 | Lebesgue 常数 → 有限差分节点稳定性度量 |
| 9 | 1435 (Zoomin) | 数值求根 | Halley (三次) / Brent / Chebyshev 高阶方法用于 profiling |
| 10 | 807 (Fixed Point) | 不动点迭代 | Profiling 的鲁棒不动点精化 |
| 11 | 671 (Game of Life) | 细胞自动机 | 离散状态空间 {−2,−1,0,1,2}^n_nuis 的初始化扫描 |
| 12 | 1225 (VSC HEOM) | 量子耗散 | 谱分解思想 → 质量分辨率 Voigt 展宽 |
| 13 | 229 (Cube ARBQ) | 数值求积 | 3D 立方体高阶求积规则 (degree ≤ 15) |
| 14 | 166 (Chebyshev2) | 数值求积 | Gauss-Chebyshev Type 2 精确度检验 |
| 15 | 159 (Chebyshev) | 多项式插值 | Profile likelihood 的 Chebyshev 快速插值 |

---

## 核心算法与公式

### 1. 高阶有限差分 (finite_diff.py, richardson.py)

**Fornberg 算法** (任意节点、任意阶导数权重):

```
f^{(m)}(x₀) ≈ Σ_j w_j f(z_j)

递推构造:
  d[0,0] = 1
  for i = 1..n:
    c₂ = Π_{j<i} (z_i - z_j)
    for k = min(i,m)..1:
      d[i,k] = [(z_i - x₀) d[i-1,k] - k d[i-1,k-1]] / (z_i - z_j)
```

**中心差分模板** (精度 O(h²), O(h⁴), O(h⁶)):

```
一阶 O(h²): f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
一阶 O(h⁴): f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)
二阶 O(h⁴): f''(x) ≈ [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / (12h²)
```

**Richardson 外推** (Romberg 表):

```
T[k,j] = [r^{p_j} T[k,j-1] - T[k-1,j-1]] / (r^{p_j} - 1)
p_j = 2j  (中心差分偶数阶误差)
r = 2     (步长减半)
```

**复步微分** (无相消误差):

```
f'(x) ≈ Im[f(x + ih)] / h   (h ~ O(1) 即可达机器精度)
```

### 2. 数值稳定性分析 (stability.py)

**Lebesgue 常数** (节点稳定性):

```
Λ_n = max_x Σ_j |l_j(x)|
l_j(x) = Π_{k≠j} (x - x_k) / (x_j - x_k)

等距节点: Λ_n ~ 2^{n+1} / (e n ln n)   → 指数发散
Chebyshev: Λ_n ~ (2/π) ln n            → 对数稳定
```

**差分算子条件数**:

```
κ(W) = Σ_j |w_j|  (与步长无关的固有稳定性度量)
||D_h||_∞ = κ(W) / h^m  (步长依赖的放大因子)
```

**最优步长**:

```
h* ≈ ε_mach^{1/(p+m)}

对 p=4 (四阶差分), m=1 (一阶导): h* ≈ ε^{1/5} ≈ 6.5×10⁻⁴
```

### 3. Profile Likelihood Profiling (profiling_optimizer.py)

**Halley 方法** (三次收敛, seed 1435):

```
θ ← θ - (f/f') / (1 - 0.5 f·f''/(f')²)

对对角 Hessian:
  f_k  = ∂NLL/∂θ_k
  f'_k = ∂²NLL/∂θ_k²
  f''_k ≈ [g_k(θ+h) - g_k(θ-h)] / (2h)
```

**不动点迭代** (seed 807):

```
G(θ) = θ - D⁻¹ · ∇NLL(θ),  D = diag(Hessian)
θ^{(n+1)} = (1-α) θ^{(n)} + α · G(θ^{(n)})
```

**依赖图分解** (seed 285):

```
A_{jk} = 1 若 ∃ bin i 使得 θ_j, θ_k 同时影响 ν_i
分解为连通分量 → 块对角结构 → 独立子问题
```

### 4. CLs 上限设定 (asymptotic.py, cls_calculator.py)

**渐近公式** (Cowan et al. 2012):

```
p_{s+b} = 1 - Φ(√q_μ + μ/σ)
p_b     = 1 - Φ(√q_μ)
CLs     = p_{s+b} / (1 - p_b)

σ = μ / √(q_{μ,A})
```

**Asimov 数据集**:

```
n_i^{A} = ν_i(μ', θ̂_{μ'})

Asimov q_{μ,A} = 2 Σ_i [ν_i(μ) - ν_i(μ') - n_i^A ln(ν_i(μ)/ν_i(μ'))]
```

**蒙特卡洛 CLs** (seed 1020 自适应采样):

```
对 b = 1..N_toys:
  n_toy ~ Poisson(ν(μ, θ̂_μ))  或  Poisson(ν(0, θ̂_0))
  q_μ^{(b)} = profile_likelihood_ratio(n_toy, μ)

p_{s+b} ≈ (1/N) Σ I(q_μ^{(b)} ≥ q_μ^{obs})

自适应: N(μ) = N_min + (N_max - N_min) · (1 + cos(π·d/Δ)) / 2
```

### 5. 数值工具 (numerical_tools.py)

**Chebyshev 插值** (seed 159):

```
c_k = (2/n) Σ_j f(x_j) T_k(x_j)
x_j = cos((2j-1)π/(2n))  (Chebyshev 零点)
f(x) ≈ c_0/2 + Σ_{k=1}^{n-1} c_k T_k(x)

Clenshaw 递推: d_k = 2x d_{k+1} - d_{k+2} + c_k
```

**Gauss-Chebyshev Type 2** (seed 166):

```
∫_{-1}^1 f(x) √(1-x²) dx ≈ (π/(n+1)) Σ_j sin²(jπ/(n+1)) f(x_j)
x_j = cos(jπ/(n+1))

精确度: 对 2n-1 次多项式精确
```

**3D 立方体求积** (seed 229):

```
∫_{[-1,1]³} f(x,y,z) dV ≈ Σ_{i,j,k} w_i w_j w_k f(x_i, y_j, z_k)

Gauss-Legendre 节点: x_i = ±√(3/5), 0  (3点)
权重: w_i = 5/9, 8/9, 5/9
```

---

## 项目结构

```
230_synth_project_Advanced/
├── main.py                    # 统一入口 (零参数运行)
├── physical_model.py          # 物理模型: H→γγ 双光子搜索
│                               # + Unruh 温度修正 (seed 1220)
│                               + Voigt 分辨率展宽 (seed 1225)
├── likelihood.py              # Profile Likelihood 构造
│                               # + Poisson × Gaussian NLL
│                               + 解析梯度与 Hessian
├── finite_diff.py             # 高阶有限差分
│                               # + Fornberg 算法
│                               + O(h²/⁴/⁶) 中心差分
│                               + 复步微分
├── richardson.py              # Richardson 外推
│                               # + Romberg 表
│                               + 自适应步长选择
├── stability.py               # 稳定性分析
│                               # + Lebesgue 常数 (seed 658)
│                               + 条件数 / 最优步长
│                               + 稳定性评分
├── profiling_optimizer.py     # Nuisance Profiling
│                               # + Halley 三次收敛 (seed 1435)
│                               + 不动点迭代 (seed 807)
│                               + 依赖图分解 (seed 285)
│                               + 离散初始化 (seed 671)
├── asymptotic.py              # 渐近公式
│                               # + Cowan et al. 2012
│                               + Asimov 数据集
│                               + Φ, Φ⁻¹, CLs
├── cls_calculator.py          # CLs 计算器
│                               # + 蒙特卡洛玩具实验
│                               + 自适应采样 (seed 1020)
├── scan.py                    # μ 扫描
│                               # + Profile Likelihood 扫描
│                               + Brent 上限定位
│                               + 期望限 Band
├── numerical_tools.py         # 数值工具
│                               # + Chebyshev 插值 (seed 159)
│                               + Gauss-Chebyshev-2 求积 (seed 166)
│                               + 3D 立方体求积 (seed 229)
│                               + FEM 参数积分 (seed 846)
└── README_博士级合成说明.md   # 本文件
```

---

## 运行方法

### 零参数运行

```bash
cd 230_synth_project_Advanced
python main.py
```

运行将自动执行 8 个模块:
1. 物理模型构建 (H → γγ 5-bin 模型)
2. 似然函数与全局拟合
3. 高阶有限差分与 Richardson 外推
4. 数值稳定性分析
5. Nuisance Profiling 优化器
6. 渐近 CLs 与 μ 扫描
7. 蒙特卡洛 CLs 验证
8. 数值工具验证

预期耗时: ~8 秒 (单核, 普通笔记本)

### 输出示例

```
========================================================================
  模块 6: 渐近 CLs 与 mu 扫描
========================================================================
Asimov q_{mu=1,A} = 7.0874
估计 sigma = 0.3756

95% CL 观测上限: mu_up = 0.3000 (converged)

期望限 (Asimov b-only):
  Median: 0.3176
  +1sigma band: 0.6932
  -1sigma band: 0.0000
```

---

## 可复现性保证

1. **固定随机种子**: `np.random.seed(230)`, `default_rng(seed=230)`
2. **小规模实验**: 5-bin 模型, 2 个 nuisance 参数, 可手算验证
3. **解析校验**: 每个模块的 `__main__` 块提供独立单元测试
4. **收敛监控**: 所有迭代算法输出收敛信息与残差
5. **边界处理**: 所有对数、除法、开方操作均有保护

---

## 关键公式速查

### Profile Likelihood Ratio

```
λ(μ) = L(μ, θ̂_μ) / L(μ̂, θ̂)
-2 ln λ(μ) = 2 [NLL(μ, θ̂_μ) - NLL(μ̂, θ̂)]
q_μ = -2 ln λ(μ)  if μ̂ ≤ μ, else 0
```

### Asimov 显著度

```
Median Z_{exp} = √(q_{μ,A}) = μ / σ
σ ≈ μ / √(q_{μ,A})
```

### CLs 上限的渐近解

```
μ_up ≈ σ [z_α + √(z_α² + 4q_{μ,A})] / 2

z_{0.05} = Φ⁻¹(0.95) ≈ 1.645
```

### 有限差分误差平衡

```
E(h) = C_T h^p + C_R ε / h^m
h* = (m C_R ε / (p C_T))^{1/(p+m)}
E_min ~ ε^{p/(p+m)}
```

---

## 依赖

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.6

无需其他外部依赖, 无需 GPU, 无需可视化库。

---

## 物理参数参考

| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| Higgs 质量 | m_H | 125.09 | GeV |
| Higgs 宽度 (SM) | Γ_H | 4.07×10⁻³ | GeV |
| LHC 质心能 | √s | 13000 | GeV |
| Run-2 积分亮度 | L | 139 | fb⁻¹ |
| 质子质量 | m_p | 938.27 | MeV |
| ℏc | ℏc | 197.33 | MeV·fm |

---

## 博士级难度体现

1. **多尺度耦合**: Unruh 温度修正 (~10⁻³ GeV) 与 Higgs 质量 (~125 GeV) 相差 5 个数量级
2. **非凸优化**: Profile likelihood 在 nuisance 参数空间存在多个局部极小
3. **病态条件**: 高阶有限差分矩阵条件数随阶数指数增长
4. **渐近失效**: 小统计量 (n_i ~ 1-15) 时渐近公式需要修正
5. **多维积分**: Nuisance 空间 FEM 积分随维度指数增长 (维度灾难)
6. **稳定性诊断**: 需要综合 Lebesgue 常数、条件数、收敛速率多维指标

---

## 作者与许可

本项目为科研教学合成项目, 融合 15 个开源科研项目的核心算法思想。
代码遵循 MIT 许可, 仅供学术研究使用。
