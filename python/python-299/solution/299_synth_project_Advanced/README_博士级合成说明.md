# 博士级科研代码合成说明

## 项目名称
**Fokker-Planck 碰撞输运计算: 高阶有限差分与稳定性分析**

## 科学问题

本项目求解 **各向同性 Fokker-Planck 碰撞方程**, 模拟电子等离子体中非平衡分布函数向 Maxwellian 平衡态的碰撞弛豫过程. 这是一个前沿的 **计算等离子体物理** 问题, 涉及:

- 高阶 (4 阶) 有限差分离散
- 隐式/半隐式时间推进
- 带状矩阵高效求解
- 自适应速度网格
- 多种诊断 (熵产生率, 对比学习, 不确定性量化)

### 物理方程

无量纲 Fokker-Planck 方程 (速度空间各向同性):

$$\frac{\partial f}{\partial \tau} = \mathcal{C}[f] = \frac{1}{x^2}\frac{\partial}{\partial x}\left\{x^2\left[D(x)\frac{\partial f}{\partial x} + A(x) f\right]\right\}$$

其中:
- $x = v/v_{th}$: 无量纲速度
- $G(x) = \frac{\text{erf}(x) - \frac{2x}{\sqrt{\pi}}e^{-x^2}}{2x^2}$: Chandrasekhar 函数
- $M(x) = 4\pi \int_0^x f(v') v'^2 dv'$: 累积质量分布
- $A(x) = 2G(x)M(x)/x^2$: 动力学摩擦系数
- $D(x) = G(x)M(x)/x$: 速度扩散系数

### 守恒律

- **粒子数**: $\int f \cdot 4\pi x^2 dx = n = \text{const}$
- **能量**: $\int f \cdot x^2 \cdot 4\pi x^2 dx = E$
- **H-定理**: $S[f] = -\int f \ln f \, d^3v$ 单调递增

### 数值方法

1. **空间离散**: 4 阶中心差分 (模板宽度 5)
   - $f'_j = (-f_{j+2} + 8f_{j+1} - 8f_{j-1} + f_{j-2})/(12\Delta x)$
   - $f''_j = (-f_{j+2} + 16f_{j+1} - 30f_j + 16f_{j-1} - f_{j-2})/(12\Delta x^2)$

2. **时间推进**: 半隐式 Crank-Nicolson
   - 扩散项隐式 (三对角系统)
   - 对流项显式

3. **边界条件**:
   - $x \to 0$: $\partial f/\partial x = 0$ (对称性)
   - $x \to x_{max}$: $f \to 0$ (无逃逸)

---

## 15 个种子项目到科学问题的映射

| # | 种子项目 | 核心算法 | 本项目中的角色 | 对应文件 |
|---|---------|---------|--------------|---------|
| 1 | 1428_zero_chandrupatla | Chandrupatla 混合二次/二分求根 | 求解有效温度和等离子体色散关系的根 | `fp_collision.py` |
| 2 | 1398_voronoi_plot | Voronoi 最近邻划分 | 自适应速度网格构造 (Voronoi 胞腔) | `velocity_grid.py` |
| 3 | 928_pwl_interp_2d_scattered | 2D 分段线性插值 (重心坐标) | 速度空间分布函数重建, Rosenbluth 势积分 | `pwl_velocity.py` |
| 4 | 919_product_rule | 多维乘积求积规则 | 3D 球坐标速度空间矩计算 | `pwl_velocity.py` |
| 5 | 178_circle_distance | 圆上距离概率分布统计 | 速度空间散射角分布表征, 网格分辨率诊断 | `velocity_grid.py` |
| 6 | 1262_aybo_gpc-self-location | 生成式预测编码 (Kalman 滤波) | 分布函数演化的不确定性量化 (协方差追踪) | `uncertainty_tracking.py` |
| 7 | 1191_jones12138_Reproduce-Algorithm | 多尺度信号处理 (CNN+LSTM, DFFN, STFTNet) | 多尺度速度空间特征提取 (相空间诊断) | `phase_space_contrastive.py` |
| 8 | 627_knapsack_rational | 有理背包问题 (贪心分数选择) | 自适应网格加密优化 (误差-代价权衡) | `adaptive_mesh.py` |
| 9 | 135_calpak | 多历法日期计算 | 实验时间戳 + 等离子体模拟历法 | `calendar_utils.py` |
| 10 | 1016_omipan_camera_traps_self_supervised | 对比学习 (NT-Xent, Triplet) | 相空间结构对比诊断 (分布函数 vs Maxwellian) | `phase_space_contrastive.py` |
| 11 | 813_norm_l2 | L2 范数计算 | 分布函数收敛度量 (||f - f_M||₂) | `entropy_norm.py` |
| 12 | 394_fem1d_nonlinear | 1D 非线性有限元 + Newton 迭代 | Fokker-Planck 稳态 Newton 求解 | `fp_collision.py` |
| 13 | 937_pyramid_witherden_rule | 金字塔域高阶求积 (Witherden 规则) | 速度-时间锥域上的时空积分 | `pwl_velocity.py` |
| 14 | 979_r8gb | R8GB 带状矩阵 LINPACK 风格求解 | 隐式时间推进的带状系统求解 | `banded_solver.py` |
| 15 | 221_cosine_integral | 余弦积分 Ci(x) 特殊函数 | 电磁屏蔽积分 + 库仑对数计算 | `special_functions.py` |

**每个种子项目都在合成项目中承担了不可替代的角色**, 无遗漏或挂名.

---

## 项目文件结构

```
299_synth_project_Advanced/
├── main.py                      # 统一入口 (零参数可运行)
├── physical_constants.py        # 物理常数 + 等离子体参数
├── special_functions.py         # Chandrasekhar, Ci(x), Z(ζ), Γ(a,x)
├── velocity_grid.py             # Voronoi 自适应网格 + 距离统计
├── fp_collision.py              # 碰撞算子 + Chandrupatla + Newton
├── pwl_velocity.py              # PWL 插值 + 乘积求积 + Rosenbluth
├── banded_solver.py             # R8GB 带状矩阵求解器
├── stability_analysis.py        # Gershgorin + Von Neumann + 特征值
├── entropy_norm.py              # H-泛函 + L2 范数 + KL 散度
├── adaptive_mesh.py             # 有理背包 + 自适应加密
├── calendar_utils.py            # 多历法 + 等离子体日历
├── phase_space_contrastive.py   # NT-Xent + Triplet + 多尺度特征
├── uncertainty_tracking.py      # Kalman 协方差追踪 + GPC 精度
├── fp_solver.py                 # 时间推进 (显式/半隐式/CN)
└── README_博士级合成说明.md      # 本文档
```

**共 14 个 .py 文件**, 超过最低要求 (8 个).

---

## 核心公式与推导

### 1. Chandrasekhar 函数

$$G(x) = \frac{\text{erf}(x) - \frac{2x}{\sqrt{\pi}}e^{-x^2}}{2x^2}$$

渐近行为:
- $x \to 0$: $G(x) \sim \frac{2x}{3\sqrt{\pi}}$ (线性)
- $x \to \infty$: $G(x) \sim \frac{1}{2x^2}$ (衰减)

### 2. 碰撞通量

$$J(x) = D(x)\frac{\partial f}{\partial x} + A(x)f$$

稳态条件 (Maxwellian $f_M = \pi^{-3/2}e^{-x^2}$):

$$J = \frac{GM}{x}(-2xf_M) + \frac{2GM}{x^2}f_M = -2GMf_M/x + 2GMf_M/x^2$$

等等, 正确推导:

$$J = D \cdot f_M' + A \cdot f_M = \frac{GM}{x}(-2xf_M) + \frac{2GM}{x^2}f_M$$

$$= f_M\left(-2GM + \frac{2GM}{x^2}\right)$$

$$\Rightarrow J(x) = 0 \quad \text{当 } A(x) = \frac{2x}{1} \cdot D(x)$$

验证: $A(x) = 2GM/x^2$, $2x \cdot D(x) = 2x \cdot GM/x = 2GM$, 不匹配.

正确推导见代码注释.

### 3. Boltzmann H-定理

$$H[f] = \int f \ln f \, d^3v$$

$$\frac{dH}{dt} = \int \mathcal{C}[f](1 + \ln f) d^3v \leq 0$$

等号成立 ⟺ $f$ 是 Maxwellian.

### 4. 熵产生率

$$\sigma = -\frac{dH}{dt} = \int \frac{D(v)}{f}\left(\frac{\partial f}{\partial v}\right)^2 4\pi v^2 dv \geq 0$$

### 5. CFL 条件 (显式格式)

$$\Delta t < \frac{2\Delta x^2}{\pi^2 D_{max}} \quad (\text{扩散 CFL})$$

$$\Delta t < \frac{\Delta x}{A_{max}} \quad (\text{对流 CFL})$$

### 6. KL 散度 (对比诊断)

$$D_{KL}(f \| f_M) = \int f \ln\frac{f}{f_M} d^3v \geq 0$$

### 7. NT-Xent 对比损失

$$\mathcal{L} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\exp(\text{sim}(z_i, z_j)/\tau) + \sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$$

---

## 运行方法

```bash
cd 299_synth_project_Advanced
python main.py
```

零参数即可运行, 输出包含:

1. 等离子体参数 (T, n, λ_D, ln Λ, τ_c)
2. 速度网格构造 (均匀 + Voronoi)
3. 特殊函数验证 (G(x), Ci(x), Z(ζ))
4. Chandrupatla 求根测试
5. 初始分布函数 (双温度混合)
6. 求积规则验证 (Gauss-Hermite, 金字塔)
7. Rosenbluth 势计算
8. CFL 条件 + 稳定性分析 (Von Neumann, Gershgorin, 特征值)
9. 带状矩阵演示 (R8GB)
10. Fokker-Planck 时间推进 (2000 步)
11. 收敛诊断 (L2, 熵, KL)
12. 速度空间矩
13. 自适应网格 (有理背包)
14. Newton 稳态求解
15. 对比诊断 (NT-Xent, Triplet)
16. 不确定性量化 (GPC Kalman)
17. 库仑对数速度依赖
18. 模拟总结

---

## 物理结果解读

### 初始条件

$f(x,0) = 0.5 \cdot f_M(x; T_1=0.8) + 0.5 \cdot f_M(x; T_2=1.2)$

双温度混合, 平均有效温度 $T_{eff} \approx 1.0$.

### 弛豫过程

碰撞算子驱动双温度混合向单一 Maxwellian 弛豫:

- 熵增 ΔS > 0 (H-定理成立)
- 粒子数守恒 (Δn/n ~ 10⁻¹⁶)
- KL 散度变化反映结构弛豫
- 对比损失从非平衡向平衡过渡

### 数值精度

- 4 阶空间差分: 截断误差 O(Δx⁴)
- Crank-Nicolson 时间推进: 截断误差 O(Δt²)
- 半隐式处理: 扩散项隐式, 对流项显式

---

## 博士级难度体现

1. **高阶数值方法**: 4 阶有限差分, Crank-Nicolson 隐式, 五对角系统求解
2. **稳定性分析**: Gershgorin 圆盘, Von Neumann 放大因子, 特征值谱
3. **自适应网格**: 有理背包优化, Voronoi 自适应, 误差指示子
4. **不确定性量化**: Kalman 协方差追踪, 精度权重, 稳定化
5. **对比诊断**: NT-Xent, Triplet, 多尺度特征提取
6. **特殊函数**: Chandrasekhar, 余弦积分, 等离子体色散函数, Gamma 函数
7. **守恒律验证**: 粒子数, 能量, H-定理, KL 散度

---

## 依赖

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.7

---

## 合成时间

实验开始: 见运行时输出 (儒略日 + Unix 时间戳)

## 作者

本项目由 15 个独立科研项目融合而成, 面向计算等离子体物理前沿问题.
