# 重离子碰撞：椭圆流与初始条件涨落建模
# Heavy-Ion Collision: Elliptic Flow & Initial Condition Fluctuation Modeling

## 博士级科学计算合成项目说明

---

## 一、项目概述

本项目是一个面向**前沿核物理**的博士级计算项目，模拟**相对论重离子碰撞**（如 RHIC/LHC 的 Au+Au 或 Pb+Pb 碰撞）中**夸克-胶子等离子体 (QGP)** 的形成与演化过程。核心科学问题是：

> **初始核几何涨落如何通过粘滞流体力学演化，产生实验观测到的椭圆流 v₂ 及高阶流谐波 vₙ？**

项目融合了 15 个种子科研项目的核心算法，在指定领域内构建了完整的"初始条件 → 数值离散 → 流体力学演化 → 可观测量提取 → 统计推断 → 参数优化"科学计算链条。

---

## 二、科学问题背景

### 2.1 物理背景

- **相对论重离子碰撞**：在金-金 (Au+Au, √s = 200 GeV) 碰撞中，碰撞区域能量密度可达核物质密度的 100 倍以上，形成夸克-胶子等离子体 (QGP)。
- **集体流与椭圆流**：QGP 在压强梯度驱动下集体膨胀，空间各向异性转化为动量各向异性，表现为粒子角分布中的**椭圆流 v₂**。
- **初始条件涨落**：核子位置的量子涨落导致初始几何不仅具有椭圆偏心率 ε₂，还产生三角 ε₃、四极 ε₄ 等高阶涨落。
- **输运系数**：QGP 的剪切粘滞系数 η/s 接近 KSS 下界 1/(4π)，是已知"最完美"的流体。

### 2.2 核心方程

**能量-动量张量守恒**：
$$\partial_\mu T^{\mu\nu} = 0$$

**含粘性修正的能量-动量张量**：
$$T^{\mu\nu} = (\varepsilon + P + \Pi) u^\mu u^\nu - (P + \Pi) g^{\mu\nu} + \pi^{\mu\nu}$$

**Israel-Stewart 弛豫方程**：
$$\tau_\pi \Delta^{\mu\alpha}\Delta^{\nu\beta} D\pi_{\alpha\beta} + \pi^{\mu\nu} = 2\eta\sigma^{\mu\nu} + \cdots$$

**流谐波分解**：
$$\frac{dN}{d\phi} \propto 1 + 2\sum_{n=1}^{\infty} v_n \cos(n(\phi - \Psi_n))$$

---

## 三、种子项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 本项目中的角色 | 对应模块 |
|------|----------|----------|----------------|----------|
| 1 | 834_opt_golden | 黄金分割搜索 | 优化 η/s 与展宽参数 σ | `golden_optimizer.py` |
| 2 | 916_prism_jaskowiec_rule | 棱柱求积法则 | 3D 时空积分（横向+纵向快度） | `quadrature.py` |
| 3 | 1218_uw-mad-dash_bagpipe | 分布式分块归约 | Q-vectors 分块并行归约 | `flow_harmonics.py` |
| 4 | 468_geometry | 计算几何原语 | Wood-Saxon 分布、偏心率、参与者平面 | `nuclear_geometry.py` |
| 5 | 286_digraph_arc | 有向弧表示 | 网格因果连接性（光锥结构） | `nuclear_geometry.py` |
| 6 | 1309_triangle_interpolate | 三角形重心插值 | 子网格插值 | `nuclear_geometry.py` |
| 7 | 871_plasma_matrix | 等离子体矩阵组装 | 输运系数矩阵（剪切/体积粘滞张量） | `initial_conditions.py` |
| 8 | 1194_principal-agent-hypothesis | 假设检验 | v₂ 显著性 t 检验、置信区间 | `hypothesis_test.py` |
| 9 | 410_fem2d_predator_prey_fast | 快速 2D FE 模板 | 2D 横向拉普拉斯/梯度算子 | `high_order_fd.py`, `viscous_hydro.py` |
| 10 | 846_paraheat_functional | 抛物热方程泛函 | 粘性修正的扩散项、稳定性边界 | `initial_conditions.py`, `stability_analysis.py` |
| 11 | 991_r8pp | 压缩对称存储 | 粘滞应力 π^μν 的压缩矩阵存储 | `linear_solver.py` |
| 12 | 473_gmres | GMRES 迭代求解 | 隐式粘性修正方程求解 | `linear_solver.py` |
| 13 | 1316_triangle_symq_rule | 三角形对称求积 | 角向傅里叶积分、多边形积分 | `quadrature.py`, `flow_harmonics.py` |
| 14 | 1073_balloon-dynamics | 径向膨胀动力学 | 自适应 CFL、火球膨胀演化 | `stability_analysis.py` |
| 15 | 1099_amsontag_mcpse | 蒙特卡罗事件抽样 | 核子位置抽样、碰撞计数 | `initial_conditions.py` |

---

## 四、文件结构与模块说明

```
240_synth_project_Advanced/
├── main.py                  # 统一入口，零参数运行
├── physics_constants.py     # 物理常数与单位换算
├── nuclear_geometry.py      # 核几何：Wood-Saxon、偏心率、参与者平面
├── initial_conditions.py    # MC Glauber 初始条件
├── high_order_fd.py         # 高阶有限差分模板（2/4/6/8 阶）
├── equation_of_state.py     # 格点 QCD 状态方程（Wuppertal-Budapest）
├── viscous_hydro.py         # 2+1D 相对论粘滞流体力学
├── flow_harmonics.py        # 流谐波 vₙ 提取与事件平面法
├── quadrature.py            # 对称求积法则（三角形、棱柱、高斯）
├── linear_solver.py         # GMRES、Jacobi、压缩矩阵存储
├── stability_analysis.py    # von Neumann 分析、CFL 条件、色散关系
├── golden_optimizer.py      # 黄金分割搜索优化器
├── hypothesis_test.py       # 统计假设检验（t、F、KS、χ²）
└── README_博士级合成说明.md  # 本文档
```

### 4.1 模块详情

#### `physics_constants.py`
- 自然单位制：ℏ = c = k_B = 1
- 核物理常数：核子质量、π/K 介子质量
- Wood-Saxon 参数：Au (R=6.38 fm, a=0.535 fm)、Pb (R=6.62 fm, a=0.546 fm)
- QCD 临界温度：T_c = 154 MeV
- Stefan-Boltzmann 常数：σ_SB = (π²/30)(16 + 21N_f/2)

#### `nuclear_geometry.py`
- Wood-Saxon 密度：ρ(r) = ρ₀ / (1 + exp((r-R)/a))
- 核厚度函数 T_A(b) = ∫dz ρ_A(√(b²+z²))
- 参与者偏心率 ε_n e^{inΦ_n} = -⟨r^n e^{inφ}⟩ / ⟨r^n⟩
- 三角形重心插值、对称求积

#### `initial_conditions.py`
- Monte Carlo Glauber 事件生成
- 高斯核-核碰撞判定：d_cut = √(σ_NN/π)
- 能量密度展宽：ε(x,y) = Σ w_i G(x-x_i, y-y_i; σ)
- 涨落关联函数 C(r) = ⟨δε(x)δε(x+r)⟩ / ⟨δε²⟩
- 输运矩阵组装（稀疏带状）

#### `high_order_fd.py`
- 2/4/6/8 阶中心差分模板
- 拉普拉斯算子 ∇² = ∂²/∂x² + ∂²/∂y²
- 梯度 ∇f、散度 ∇·F、旋度 ∇×F
- 双调和算子 ∇⁴
- 紧致 (Padé) 差分格式
- Thomas 算法三对角求解

#### `equation_of_state.py`
- Wuppertal-Budapest 格点 QCD 参数化：
  - P(T)/T⁴ = A₁ exp(-A₂ T/T₀) + S₁ tanh((T-T₀)/S₂) + S₃
- ε(T) = T dP/dT - P
- s(T) = dP/dT
- c_s²(T) = (dP/dT) / (dε/dT)
- η/s(T) 参数化：最小值在 T_c，向两侧二次增长
- ζ/s(T) 参数化：T_c 附近高斯峰
- 弛豫时间 τ_π = 5η/s

#### `viscous_hydro.py`
- 理想流体 T^μν = (ε+P)u^μ u^ν - P g^μν
- 剪切应力 π^μν = 2η σ^μν
- 体积粘滞 Π = -ζ (∂·u)
- RK2 (Heun) 时间积分
- Israel-Stewart 弛豫方程简化实现
- Cooper-Frye 冻结-out 积分

#### `flow_harmonics.py`
- Q_n = Σ w_k e^{inφ_k}
- 事件平面法：v_n = ⟨cos(n(φ - Ψ_n))⟩ / R_n
- 双粒子关联 V_nΔ = ⟨cos(n(φ₁ - φ₂))⟩
- 标量积法、分块并行归约
- 对称求积傅里叶系数

#### `quadrature.py`
- 高斯-勒让德 1D/2D 求积
- 三角形对称求积（1-5 阶）
- 棱柱求积（Jaskowiec 规则）
- 多项式积分精确验证

#### `linear_solver.py`
- GMRES (Arnoldi + Givens 旋转)
- Jacobi 迭代、Gauss-Seidel 迭代
- CG 共轭梯度法
- 压缩对称矩阵存储 r8pp
- Thomas 三对角算法

#### `stability_analysis.py`
- CFL 条件：Δt ≤ C min(Δx,Δy) / v_max
- von Neumann 放大因子 g(k)
- 谱半径 ρ(∇²)
- 抛物型稳定性界限
- 自适应时间步长
- 数值色散关系

#### `golden_optimizer.py`
- 黄金分割搜索（一维极小化）
- 参数优化：η/s、σ、T_freeze
- 2D 网格搜索 + 精化

#### `hypothesis_test.py`
- 单样本 t 检验：H₀: v₂ = 0
- F 检验：方差齐性
- KS 检验：分布比较
- χ² 拟合优度
- Bootstrap 置信区间
- 单因素方差分析

---

## 五、核心公式汇总

### 5.1 初始条件

**Wood-Saxon 核密度**：
$$\rho(r) = \frac{\rho_0}{1 + \exp\left(\frac{r - R}{a}\right)}$$

**参与者偏心率**：
$$\varepsilon_n e^{i n \Phi_n} = -\frac{\sum_k w_k r_k^n e^{i n \phi_k}}{\sum_k w_k r_k^n}$$

**高斯展宽能量密度**：
$$\varepsilon(x, y) = \sum_i w_i \frac{1}{2\pi\sigma^2} \exp\left(-\frac{(x-x_i)^2 + (y-y_i)^2}{2\sigma^2}\right)$$

### 5.2 流体力学

**守恒方程**：
$$\partial_\tau \varepsilon = -\partial_x(w v_x) - \partial_y(w v_y)$$
$$\partial_\tau(w v_x) = -\partial_x(w v_x^2 + P) - \partial_y(w v_x v_y)$$

**剪切张量**：
$$\sigma^{\mu\nu} = \frac{1}{2}(\partial^\mu u^\nu + \partial^\nu u^\mu) - \frac{1}{3}\Delta^{\mu\nu}(\partial \cdot u)$$

### 5.3 数值方法

**4 阶中心差分（一阶导数）**：
$$f'(x) \approx \frac{f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)}{12h}$$

**6 阶中心差分（二阶导数）**：
$$f''(x) \approx \frac{2f(x-3h) - 27f(x-2h) + 270f(x-h) - 490f(x) + 270f(x+h) - 27f(x+2h) + 2f(x+3h)}{180h^2}$$

**CFL 稳定性条件**：
$$\Delta t \leq C \frac{\min(\Delta x, \Delta y)}{|v| + c_s}$$

**Von Neumann 放大因子**：
$$g(k) = 1 - i \cdot \text{CFL} \cdot \sum_m a_m e^{i m k \Delta x}$$

### 5.4 统计推断

**t 统计量**：
$$t = \frac{\bar{v}_2}{s / \sqrt{n}}$$

**χ² 拟合优度**：
$$\chi^2 = \sum_i \frac{(v_{2,\text{model}} - v_{2,\text{data}})^2}{\sigma_i^2}$$

---

## 六、运行说明

### 6.1 环境要求

- Python 3.8+
- NumPy
- SciPy

### 6.2 运行命令

```bash
cd /mnt/data/zpy/sci-swe/source\ code/Synthesis-project-python/240_synth_project/240_synth_project_Advanced
python main.py
```

**零参数运行**：直接执行 `python main.py` 即可运行完整模拟流程。

### 6.3 输出内容

程序输出包含 12 个主要部分：
1. MC Glauber 初始条件
2. 能量密度与温度分布
3. 高阶有限差分算子验证
4. 稳定性分析（CFL、von Neumann）
5. 流体力学演化
6. 粘性修正
7. 流谐波提取（v₂, Ψ₂）
8. 统计假设检验
9. 参数优化
10. 数值积分验证
11. 线性求解器测试
12. 涨落关联分析

---

## 七、博士级科学难度体现

### 7.1 物理深度

- **格点 QCD 状态方程**：使用 Wuppertal-Budapest 2+1 味物理夸克质量参数化
- **Israel-Stewart 二阶粘性流体力学**：超越 Navier-Stokes 的因果理论
- **KSS 界与 QGP 输运系数**：η/s ≥ 1/(4π) 的 AdS/CFT 对偶结果

### 7.2 数值难度

- **高阶有限差分**：2/4/6/8 阶精度模板，含紧致格式
- **稳定性分析**：von Neumann 分析、CFL 自适应、谱半径计算
- **隐式求解**：GMRES 迭代、压缩矩阵存储、Thomas 算法

### 7.3 统计严谨性

- **假设检验体系**：t 检验、F 检验、KS 检验、χ² 检验
- **置信区间**：参数法与 Bootstrap 法
- **参数优化**：黄金分割搜索与 χ² 拟合

### 7.4 工程复杂性

- **边界条件处理**：outflow / reflecting 两种模式
- **数值鲁棒性**：所有除法含 ε 保护，指数函数裁剪
- **模块化设计**：15 个独立模块，职责清晰

---

## 八、创新点与独特性

1. **领域深度耦合**：所有变量命名、物理量纲、参数范围均严格对应重离子碰撞物理
2. **完整科学链条**：从核子抽样到流谐波统计推断的端到端模拟
3. **高阶数值方法**：首次在此类教学项目中实现 8 阶紧致差分与 von Neumann 稳定性分析
4. **统计物理融合**：将假设检验与物理参数优化有机结合
5. **无可视化依赖**：纯数值计算输出，适合大规模参数扫描与 HPC 部署

---

## 九、可扩展方向

1. **3+1D 演化**：加入纵向快度 η_s 依赖
2. **强子级联**：UrQMD/SMASH 后处理
3. **喷注淬火**：高能部分子能量损失
4. **电磁场效应**：初始强磁场对带电粒子流的影响
5. **机器学习代理模型**：用神经网络加速参数空间扫描

---

## 十、参考文献

1. Heinz, U., & Snellings, R. (2013). Collective flow and viscosity in relativistic heavy-ion collisions. *Annual Review of Nuclear and Particle Science*, 63, 123-151.
2. Romatschke, P., & Romatschke, U. (2019). *Relativistic Fluid Dynamics In and Out of Equilibrium*. Cambridge University Press.
3. Borsányi, S., et al. (2014). QCD equation of state to O(μ_B^4) from lattice QCD. *Physical Review D*, 92(1), 014505.
4. Kovchegov, Y. V., & Levin, E. (2012). *Quantum Chromodynamics at High Energy*. Cambridge University Press.
5. Luzum, M., & Romatschke, P. (2008). Conformal relativistic viscous hydrodynamics. *Physical Review C*, 78(3), 034915.
6. Policastro, A., Son, D. T., & Starinets, A. O. (2002). Shear viscosity of strongly coupled N=4 supersymmetric Yang-Mills plasma. *Physical Review Letters*, 87(8), 081601.
7. Jaskowiec, J. A. (2016). High-degree cubature rules for prism finite elements. *Journal of Computational and Applied Mathematics*.
8. Burkardt, J. (2019). GEOMETRY library: computational geometry routines.

---

## 十一、许可证

本项目代码遵循 MIT 许可证。

---

**项目完成日期**：2026 年 6 月  
**科学领域**：核物理 / 高能物理 / 计算物理  
**难度等级**：博士级前沿科学计算
