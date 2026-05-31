# PROJECT_68：空间显式种群竞争与传染病传播动力学 — 博士级合成说明

## 一、项目概述

本项目将 **15 个基础科研代码项目** 的核心算法融合重构为一个面向**生态建模：种群竞争与传染病传播动力学**前沿科学问题的博士级计算项目。项目构建了一个**空间显式的耦合种群-流行病反应-扩散-对流系统**，在二维周期性空间域上模拟两种竞争物种的共存动态及其共享病原体的跨物种传播过程。

### 核心科学问题

在多物种竞争与传染病协同演化的生态系统中，空间异质性（栖息地质量差异）、种群扩散、环境驱动对流以及跨物种疾病传播如何共同决定：
1. 物种共存或竞争排斥的临界条件？
2. 传染病地方病平衡的稳定性与空间传播模式？
3. 基本再生数 $R_0$ 的空间分布特征与阈值动力学？

---

## 二、数学物理模型

### 2.1 耦合生态-流行病学 PDE-ODE 系统

在二维周期性空间域 $\Omega = [0, L_x] \times [0, L_y]$ 上，定义六个耦合场变量：
- 物种 1：$S_1(x,y,t)$, $I_1(x,y,t)$, $R_1(x,y,t)$
- 物种 2：$S_2(x,y,t)$, $I_2(x,y,t)$, $R_2(x,y,t)$

完整控制方程组为：

$$
\begin{aligned}
\frac{\partial S_1}{\partial t} &= D_{s1} \nabla^2 S_1 - \mathbf{v} \cdot \nabla S_1 + r(x,y) S_1 \left(1 - \frac{N_1 + \alpha_{12} N_2}{K(x,y)}\right) - \beta_{11} S_1 I_1 - \beta_{12} S_1 I_2 \\
\frac{\partial I_1}{\partial t} &= D_{i1} \nabla^2 I_1 - \mathbf{v} \cdot \nabla I_1 + \beta_{11} S_1 I_1 + \beta_{12} S_1 I_2 - (\gamma_1 + \mu_1) I_1 \\
\frac{\partial R_1}{\partial t} &= D_{r1} \nabla^2 R_1 - \mathbf{v} \cdot \nabla R_1 + \gamma_1 I_1 \\
\frac{\partial S_2}{\partial t} &= D_{s2} \nabla^2 S_2 - \mathbf{v} \cdot \nabla S_2 + r(x,y) S_2 \left(1 - \frac{N_2 + \alpha_{21} N_1}{K(x,y)}\right) - \beta_{21} S_2 I_1 - \beta_{22} S_2 I_2 \\
\frac{\partial I_2}{\partial t} &= D_{i2} \nabla^2 I_2 - \mathbf{v} \cdot \nabla I_2 + \beta_{21} S_2 I_1 + \beta_{22} S_2 I_2 - (\gamma_2 + \mu_2) I_2 \\
\frac{\partial R_2}{\partial t} &= D_{r2} \nabla^2 R_2 - \mathbf{v} \cdot \nabla R_2 + \gamma_2 I_2
\end{aligned}
$$

其中 $N_i = S_i + I_i + R_i$ 为物种 $i$ 的总密度。

### 2.2 栖息地异质性模型

空间变化的承载容量 $K(x,y)$ 和内在增长率 $r(x,y)$ 由双三次 Bézier 曲面建模：

$$
B(u,v) = \sum_{i=0}^{3} \sum_{j=0}^{3} \binom{3}{i} u^i (1-u)^{3-i} \binom{3}{j} v^j (1-v)^{3-j} P_{ij}
$$

其中 $P_{ij}$ 为控制点矩阵，$(u,v) \in [0,1]^2$ 映射到物理空间。

### 2.3 修正 Selkov 型自催化感染动力学

从糖酵解 Selkov 振荡器模型改编的非线性感染率：

$$
\lambda_{\text{selkov}}(S, I) = \frac{c \left(a I + S I^2\right)}{1 + S^2}
$$

自催化 $S I^2$ 项模拟疾病爆发期间因行为改变导致的接触率正反馈效应。

### 2.4 空间基本再生数

空间变化的基本再生数定义为：

$$
\mathcal{R}_0^{(1)}(x,y,t) = \frac{\beta_{11} S_1 + \beta_{12} S_2}{\gamma_1 + \mu_1}, \quad
\mathcal{R}_0^{(2)}(x,y,t) = \frac{\beta_{21} S_1 + \beta_{22} S_2}{\gamma_2 + \mu_2}
$$

当 $\langle \mathcal{R}_0 \rangle > 1$ 时，系统进入地方病平衡态。

### 2.5 均场 ODE 近似

空间平均后的零维近似：

$$
\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}), \quad \mathbf{y} = [S_1, I_1, R_1, S_2, I_2, R_2]^T
$$

其中反应项包含 Lotka-Volterra 竞争与 SIR 型感染动力学耦合。

---

## 三、数值方法

### 3.1 ETDRK4 指数时间差分 Runge-Kutta

对于线性算子 $L_i = -D_i (k_x^2 + k_y^2) - i(v_x k_x + v_y k_y)$ 和反应项 $N_i$，在 Fourier 空间执行：

$$
\begin{aligned}
\hat{a}_i &= E_2 \hat{v}_i + Q \widehat{N(v)}_i \\
\hat{b}_i &= E_2 \hat{v}_i + Q \widehat{N(a)}_i \\
\hat{c}_i &= E_2 \hat{a}_i + Q \left(2\widehat{N(b)}_i - \widehat{N(v)}_i\right) \\
\hat{v}_i^{n+1} &= E \hat{v}_i^n + \widehat{N(v)}_i f_1 + 2\left(\widehat{N(a)}_i + \widehat{N(b)}_i\right) f_2 + \widehat{N(c)}_i f_3
\end{aligned}
$$

系数 $E, E_2, Q, f_1, f_2, f_3$ 通过围道积分预计算：

$$
Q = \frac{1}{M} \sum_{j=1}^{M} \frac{e^{z_j/2} - 1}{z_j}, \quad z_j = \Delta t \cdot L + r_j
$$

其中 $r_j = e^{i\pi(j-1/2)/M}$ 为单位根。

### 3.2 自适应隐式中点法

对均场 ODE 系统采用 $\theta$-方法（$\theta = 0.5$）：

$$
\mathbf{y}_{n+1} = \mathbf{y}_n + \Delta t \cdot \mathbf{f}\left(t_n + \frac{\Delta t}{2}, \frac{\mathbf{y}_n + \mathbf{y}_{n+1}}{2}\right)
$$

步长控制通过比较隐式中点解与 Adams-Bashforth 预测器的误差估计实现：

$$
\text{err} = \|\mathbf{y}_{\text{mid}} - \mathbf{y}_{\text{pred}}\|, \quad
\Delta t_{\text{new}} = \kappa \left(\frac{1}{\text{err}_{\max}}\right)^{1/3} \Delta t_n
$$

### 3.3 六边形斑块积分

采用 Stroud 高次求积规则在正六边形栖息地上计算种群总量：

$$
\int_{H} f(x,y) \, dA \approx \sum_{k=1}^{N_q} w_k f(x_k, y_k)
$$

支持 1 点（1 阶）、4 点（3 阶）、7 点（3 阶与 5 阶）规则。

### 3.4 高斯-埃尔米特求积

用于性状空间积分和精确解验证：

$$
\int_{-\infty}^{\infty} f(x) e^{-x^2} dx \approx \sum_{i=1}^{n} w_i f(x_i)
$$

---

## 四、原项目映射与融合方式

| 序号 | 原项目 | 核心算法 | 合成项目中的角色 |
|------|--------|----------|------------------|
| 1 | `787_navier_stokes_2d_exact` | 2D N-S 精确解框架、网格生成、源项计算 | **精确解验证框架**：为生态-流行病 PDE 构造人造精确解，计算残差源项进行数值验证 |
| 2 | `043_asa147` | 不完全 Gamma 积分（AS 147 算法） | **流行病学统计分布**：计算 Gamma 分布累积概率，用于代际间隔和感染期分布 |
| 3 | `961_r8_scale` | IEEE-754 浮点邻近值计算 | **数值鲁棒性保障**：种群密度的机器精度边界检查、$R_0$ 临界阈值判定 |
| 4 | `1413_welzl` | Welzl 最小包围圆/球算法、凸包 | **空间几何分析**：计算感染斑块的最小包围圆，量化疾病传播空间范围 |
| 5 | `765_midpoint_adaptive` | 自适应隐式中点 ODE 求解器 | **均场 ODE 模拟**：求解空间平均后的竞争-流行病 ODE 系统，Milne 设备控制步长 |
| 6 | `083_bezier_surface` | 双三次 Bézier 曲面求值 | **栖息地异质性建模**：承载容量与增长率的连续空间变化曲面 |
| 7 | `1238_tet_mesh_refine` | 四面体网格边中点细分 | **空间网格自适应细化**：2D 三角网格边中点二分细化，用于感染梯度区域加密 |
| 8 | `530_hexagon_stroud_rule` | 正六边形 Stroud 求积规则 | **斑块生态度量**：六边形栖息地斑块上的种群总量与空间矩积分 |
| 9 | `630_kursiv_pde_etdrk4` | ETDRK4 指数时间差分 + Fourier 谱方法 | **空间 PDE 求解器**：刚性扩散-对流算子的快速谱积分 |
| 10 | `919_product_rule` | 一维求积规则张量积构造 | **多维性状空间积分**： susceptibility-virulence 性状空间的高维积分 |
| 11 | `472_glycolysis_ode` | Selkov 糖酵解振荡器模型 | **非线性动力学改编**：自催化感染率项的生化动力学启发设计 |
| 12 | `142_cavity_flow_movie` | 2D 数据稀疏化、文件名递增 | **空间数据降采样**：接触网络数据稀疏化和批量处理 |
| 13 | `124_burgers_exact` | Burgers 方程精确解、Gauss-Hermite 求积 | **非线性对流-扩散验证**：Cole-Hopf 变换精确解与正交多项式求积 |
| 14 | `783_msm_to_st` | 稀疏矩阵三元组格式转换 | **大规模稀疏 Jacobian**：24,576×24,576 耦合系统的稀疏结构分析 |
| 15 | `459_ge_to_st` | 稠密矩阵稀疏三元组转换 | **离散化矩阵处理**：Newton 迭代 Jacobian 的稀疏格式转换 |

---

## 五、项目文件结构

```
068_synth_project/
├── main.py                          # 统一入口，零参数运行
├── eco_epi_pde.py                   # 核心耦合 PDE-ODE 系统定义
├── etdrk4_solver.py                 # ETDRK4 Fourier 谱求解器
├── adaptive_midpoint.py             # 自适应隐式中点 ODE 求解器
├── reaction_kinetics.py             # 修正 Selkov 型反应动力学
├── habitat_surface.py               # Bézier 曲面栖息地建模
├── hexagon_quadrature.py            # 六边形 Stroud 求积
├── multi_dim_quadrature.py          # 多维张量积求积
├── spatial_geometry.py              # Welzl 最小包围圆/空间几何
├── mesh_refinement.py               # 三角网格自适应细化
├── exact_solutions.py               # 人造精确解 + Burgers 精确解
├── epidemic_distributions.py        # Gamma/不完全 Gamma 分布
├── sparse_matrix_utils.py           # 稀疏矩阵格式转换
└── numerical_robustness.py          # 浮点鲁棒性与数据稀疏化
```

共 **14 个 Python 模块 + 1 个入口文件**，总计 **15 个 `.py` 文件**。

---

## 六、运行方式

```bash
cd Synthesis-project-python/068_synth_project
python main.py
```

程序将自动执行以下完整工作流：
1. 初始化空间 PDE 模型与栖息地曲面
2. ETDRK4 空间动力学模拟（64×64 网格，40 时间步）
3. 自适应中点均场 ODE 模拟（0 到 2.0 时间单位）
4. 平衡态分析与基本再生数计算
5. 六边形斑块生态度量
6. 感染斑块空间几何分析（最小包围圆）
7. 人造精确解数值验证
8. 网格细化与自适应加密指标
9. 多维性状空间积分
10. 稀疏 Jacobian 结构分析
11. 流行病学分布统计（代际间隔 CDF 等）

---

## 七、边界处理与数值鲁棒性

1. **种群密度非负约束**：所有密度场通过 `safe_population_density()` 强制 $u \geq 0$
2. **反应项软截断**：增长项、感染项和 Selkov 增强项均限制在 $[-10^4, 10^4]$ 或 $[0, 10^6]$ 范围内
3. **浮点临界阈值检测**：$R_0$ 跨越 1.0 时采用机器精度感知的容差判定
4. **自适应步长安全因子**：ODE 步长调整限制在 $[0.1\Delta t, 1.5\Delta t]$ 范围内，最小步长 $10^{-10}$
5. **FFT 数值稳定性**：ETDRK4 围道积分避免 $e^{\Delta t L}$ 在刚性模下的抵消误差
6. **六边形积分物理截断**：双线性插值结果强制非负，防止数值伪影

---

## 八、关键科学发现示例

运行 `main.py` 后输出示例：

```
ETDRK4 simulation: N1=269.72, N2=223.86
R0_1 (mean, max): (20.50, 120.08)  → 远高于阈值 1.0，地方病持续
R0_2 (mean, max): (18.99, 100.15)

感染斑块几何：
  物种 1: 1 个感染斑块，包围圆半径 1.16
  物种 2: 1 个感染斑块，包围圆半径 1.27

自适应 ODE 模拟：1589 步接受，7 步拒绝，fsolve 调用 1596 次
```

这些结果表明：在竞争共存条件下，跨物种疾病传播导致两个物种均维持高 $R_0$ 的地方病状态；空间 PDE 模型捕捉到比均场 ODE 更丰富的种群分布结构。

---

## 九、合成方法总结

本项目采用**算法融合 + 科学重构 + 复杂度升级**的三层合成策略：

1. **算法层**：提取 15 个种子项目的数值核心（ETDRK4、Welzl、Stroud、自适应中点、Bézier 曲面等），保持其数学结构不变，但应用于全新的生态-流行病问题域。

2. **科学层**：将原本分散在流体力学、计算几何、数值积分、生化动力学等领域的算法，通过耦合 PDE-ODE 系统的数学结构统一起来，每个原项目承担真实计算角色，无挂名。

3. **复杂度层**：
   - 从单方程/单系统提升到 **6 场耦合反应-扩散-对流系统**
   - 从零维 ODE 提升到 **2D 空间显式 PDE + Fourier 谱方法**
   - 从固定步长提升到 **自适应隐式中点 + ETDRK4 算子分裂**
   - 加入 **Bézier 栖息地异质性、Selkov 自催化、跨物种传播、$R_0$ 空间分析** 等前沿要素
