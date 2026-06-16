# 计算高能物理: 异常事件检测与新物理信号搜索
## 高阶有限差分与稳定性分析 (小规模可复现实验)

---

## 一、科学问题描述

本项目解决一个前沿博士级科学计算问题: **在1+1维标量场论框架下, 通过高阶有限差分方法求解Klein-Gordon方程, 分析数值格式的Von Neumann稳定性, 模拟包含BSM (Z'共振) 信号的Drell-Yan对撞事件, 经探测器模拟后, 使用Profile Likelihood Ratio、核密度异常检测和主动学习参数空间搜索等方法, 发现超出标准模型的新物理信号。**

### 核心物理模型

**1+1维标量场论**

作用量:
```
S = ∫ dt dx [½(∂φ/∂t)² - ½(∂φ/∂x)² - V(φ)]
```

势能:
```
V(φ) = ½m²φ² + (λ/4!)φ⁴ + μ³φ
```

其中:
- m: 标量场质量 (GeV)
- λ: φ⁴自耦合常数
- μ³: BSM tadpole项 (显式对称性破缺)

**运动方程 (Klein-Gordon方程)**:
```
∂²φ/∂t² - ∂²φ/∂x² + m²φ + (λ/6)φ³ + μ³ = 0
```

---

## 二、种子项目到科学问题的映射 (15个项目全部融入)

| 种子项目 | 原始算法 | 在本项目中的角色 | 实现文件 |
|---------|---------|---------------|---------|
| 270_dfield9 | 方向场/ODE积分 | 时间演化积分器 (初值问题) | `field_evolution.py` |
| 982_r8ge_np | 无主元LU分解 | 隐式格式线性系统求解 | `finite_difference.py` |
| 1000_dasayan05 | Brownian运动/混沌 | 部分子簇射的随机游动模拟 | `event_generator.py` |
| 095_bisection_integer | 整数二分法 | 连续二分法搜索共振质量 | `resonance_finder.py` |
| 1097_catniplab | 吸引子/Lyapunov谱 | 放大因子分析与稳定性盆地 | `stability_analysis.py` |
| 867_persistence | 持久变量/在线统计 | Welford递推均值/方差 | `anomaly_detector.py` |
| 508_hb_to_mm | 稀疏矩阵格式转换 | 差分模板CSR稀疏索引构建 | `lattice_config.py` |
| 034_asa082 | 正交矩阵行列式 | 稳定性矩阵谱分析 | `stability_analysis.py` |
| 1143_ALRLDA | RLDA/主动学习 | Fisher信息驱动的参数扫描 | `active_learning.py` |
| 1206_EMIT_SIM | 反应扩散PDE | 格点场的时间步进演化 | `field_evolution.py` |
| 470_gl_fast_rule | Gauss-Legendre正交 | 高精度数值积分 (截面计算) | `special_functions.py` |
| 501_hand_area | Monte Carlo面积 | 相空间体积的MC积分 | `monte_carlo.py` |
| 002_advection_pde | 对流PDE/CFL | 差分格式的CFL稳定性约束 | `lattice_config.py`, `stability_analysis.py` |
| 443_fn | 特殊函数库 (FNLIB) | 散射振幅与截面所需特殊函数 | `special_functions.py` |
| 1289_BABILong | 注意力机制/噪声注入 | 探测器噪声建模与注意力加权评分 | `detector_sim.py`, `anomaly_detector.py` |

---

## 三、核心数学物理公式

### 3.1 高阶有限差分

**4阶中心差分** (二阶导数):
```
D²φ_i ≈ (-φ_{i+2} + 16φ_{i+1} - 30φ_i + 16φ_{i-1} - φ_{i-2}) / (12h²)
```

截断误差: O(h⁴)

**修正波数**:
```
k̃²(k) = [30 - 32cos(kh) + 2cos(2kh)] / (12h²)
```

### 3.2 Von Neumann 稳定性分析

**放大因子** (显式格式):
```
G(k) = (1 - s/2) ± √[(1 - s/2)² - 1]
```
其中 s = dt²(k̃² + m²) 为广义Courant数.

**稳定性条件**: |G(k)| ≤ 1, 等价于
```
dt ≤ dt_max = 2h / √(σ_max + (mh)²)
```

对4阶差分: σ_max = 64/12, 因此 dt_max ≈ 0.866h (m=0).

**Newmark-β隐式格式** (β=0.25, γ=0.5):
无条件稳定, 二阶精度.

### 3.3 色散关系

格点上的色散关系:
```
ω²(k) = k̃²(k) + m²
```

群速度:
```
v_g(k) = dω/dk
```

### 3.4 Breit-Wigner共振

相对论性Breit-Wigner分布:
```
BW(M; M₀, Γ) = (2M₀Γ/π) / [(M² - M₀²)² + M₀²Γ²]
```

### 3.5 QCD跑动耦合常数

1-loop跑动:
```
α_s(Q) = α_s(MZ) / [1 + b₀α_s(MZ)ln(Q²/MZ²)/(2π)]
b₀ = 11 - 2n_f/3
```

### 3.6 相空间体积

二体相空间:
```
Φ₂(s; m₁, m₂) = (π/2s) √λ(s, m₁², m₂²)
```
其中 λ(a,b,c) = a²+b²+c²-2ab-2ac-2bc (Källén函数).

### 3.7 Profile Likelihood Ratio

检验统计量:
```
q₀ = -2 ln[L(μ=0,θ̂̂) / L(μ̂,θ̂)]
```

渐近显著性: Z = √q₀ (单位: σ).

### 3.8 Asimov预期显著性

```
Z_A = √[2((s+b)ln(1+s/b) - s)]
```

---

## 四、项目结构

```
235_synth_project_Advanced/
├── main.py                  # 统一入口 (零参数可运行)
├── lattice_config.py        # 格点配置 (种子: 270+002+508)
├── finite_difference.py     # 高阶差分算子 (种子: 982+002+095)
├── stability_analysis.py    # Von Neumann稳定性 (种子: 1097+034)
├── special_functions.py     # 特殊函数库 (种子: 443+470)
├── field_evolution.py       # 场时间演化 (种子: 1206+270)
├── event_generator.py       # 事件生成器 (种子: 1000+501)
├── anomaly_detector.py      # 异常检测 (种子: 1143+867)
├── resonance_finder.py      # 共振搜索 (种子: 095+1097)
├── monte_carlo.py           # MC积分 (种子: 501+1000)
├── detector_sim.py          # 探测器模拟 (种子: 1289)
├── active_learning.py       # 主动学习 (种子: 1143)
└── README_博士级合成说明.md  # 本文档
```

共12个.py文件, 1个README.

---

## 五、运行方法

```bash
cd 235_synth_project_Advanced
python main.py
```

零参数运行, 无需任何输入. 预计运行时间 < 1秒.

### 输出流程

1. **Phase 0**: 物理常数与实验配置
2. **Phase 1**: 高阶有限差分稳定性分析 (CFL条件、色散关系、放大因子)
3. **Phase 2**: 格点场论时间演化 (Leapfrog积分、能量守恒检验)
4. **Phase 3**: 特殊函数库验证 (Gamma、erf、BW、α_s、GL正交、相空间)
5. **Phase 4**: 事件生成与探测器模拟 (Drell-Yan、接受度、效率)
6. **Phase 5**: 异常事件检测 (Welford统计、χ²、Profile LR、KDE、注意力加权)
7. **Phase 6**: 主动学习参数空间探索 (Fisher信息、Sherman-Morrison)
8. **Phase 7**: 二分法共振质量搜索
9. **Phase 8**: Monte Carlo相空间积分

---

## 六、关键结果

### 稳定性分析
- 4阶差分: dt_max ≈ 0.086 (h=0.1, m=2)
- CFL安全因子0.8下: dt_safe ≈ 0.069, CFL数 ≈ 0.80
- 放大因子 max|G| = 1.0 (稳定)

### 场演化
- Leapfrog 100步: 能量漂移 |ΔE|/E₀ ≈ 8×10⁻⁴ (良好守恒)

### 异常检测
- Profile Likelihood: 计算局部与全局p值
- 核密度异常检测: 识别最异常事件
- 注意力加权: 自动学习特征重要性

### 主动学习
- 14个评估点 (6初始+8迭代)
- Fisher信息矩阵驱动采样
- Sherman-Morrison更新误差 < 10⁻¹⁵

---

## 七、边界处理与数值鲁棒性

1. **CFL自动检查**: 时间步长自动调整至稳定域内
2. **差分矩阵对角占优检验**: LU分解前检查主元
3. **除零保护**: 所有除法操作均添加 ε 保护
4. **对数保护**: ln(x) 中 x 限制为 x ≥ ε
5. **平方根保护**: √x 中 x 限制为 x ≥ 0
6. **归一化保护**: 所有归一化操作添加最小值限制
7. **迭代收敛监控**: Newton-Raphson 设最大迭代数和残差阈值

---

## 八、创新点与独特性

1. **领域深度耦合**: 算法结构完全围绕格点场论+对撞机物理构建
2. **高阶差分+稳定性分析**: 系统实现2/4/6阶差分与完整Von Neumann分析
3. **端到端流水线**: 从格点配置→场演化→事件生成→探测器→异常检测
4. **多方法交叉验证**: χ²、Profile LR、KDE、注意力加权相互印证
5. **主动学习加速**: Fisher信息驱动的BSM参数空间高效探索

---

## 九、依赖

- Python ≥ 3.7
- NumPy (标准科学计算)
- 无其他第三方依赖

---

*本项目由15个种子科研项目的核心算法融合而成, 面向计算高能物理前沿问题.*
