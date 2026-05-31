# 燃烧科学：爆轰波结构与传播 — 博士级合成说明

## 一、项目概述

本项目基于 **15 个科研代码项目**的核心算法与数据结构，融合重构为一个面向**燃烧科学：爆轰波结构与传播**的博士级科学计算系统。项目使用 Python 语言实现，包含 11 个 `.py` 模块文件与 1 个统一入口 `main.py`，零参数即可运行完整计算流程。

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法/数据结构 | 在合成项目中的角色 |
|---|---|---|
| `489_grf_display` | GRF 图文件读取、节点-边数据结构 | `reaction_network_graph.py`：H₂-O₂ 燃烧反应网络的图表示与连通性分析 |
| `695_local_min_rc` | Brent 法局部最小化（Golden Section + 抛物线插值） | `cj_condition_optimizer.py`：CJ 爆轰速度的精确优化求解 |
| `1432_zero_rc` | Brent 法根查找（二分/割线/逆二次插值） | `cj_condition_optimizer.py`：Rayleigh 线与 Hugoniot 曲线交点条件求解 |
| `861_pendulum_nonlinear_ode` | 非线性 ODE 参数管理、右端项结构 | `reaction_kinetics.py`：反应 Euler 方程状态向量与通量计算；`znd_structure.py`：ZND 结构 ODE 系统 |
| `978_r8crs` | 稀疏矩阵 CRS 格式、矩阵-向量乘法 | `euler_reactive_solver.py`：隐式推进的 Jacobian 稀疏存储与运算验证 |
| `836_opt_quadratic` | 二次插值法求临界点 | `cj_condition_optimizer.py`：CJ 速度初值粗略定位 |
| `1133_spinterp` | 稀疏网格分层插值（Clenshaw-Curtis） | `sparse_grid_chemistry.py`：高维化学流形（T, p, φ, xᵣ）稀疏网格近似 |
| `1151_square_symq_rule` | 单位正方形对称求积规则 | `thermal_quadrature.py`：反应区释热率空间积分的高精度求积 |
| `1092_snakes_and_ladders_simulation` | 蒙特卡洛批量统计与批次分析 | `monte_carlo_ignition.py`：湍流点火概率的分批次蒙特卡洛评估 |
| `373_fem_basis_t3_display` | T3 线性三角形基函数与面积计算 | `adaptive_mesh.py`：爆轰波前自适应网格的三角形基函数验证 |
| `331_ellipse_monte_carlo` | 椭圆内 Cholesky 采样 | `monte_carlo_ignition.py`：局部热点椭圆区域的随机采样 |
| `261_cvt_square_uniform` | CVT（Centroidal Voronoi Tessellation）迭代 | `adaptive_mesh.py`：波前区域节点重分布与网格优化 |
| `711_mandelbrot_area` | 逃逸时间迭代与区域占比统计 | `monte_carlo_ignition.py`：临界热点核的逃逸/点火时间分析 |
| `315_double_well_ode` | 双阱势能、Hamiltonian 动力学 | `stability_analysis.py`：反应区扰动势能的线性稳定性分析；`znd_structure.py`：化学能势阱驱动的反应演化 |
| `1180_subset_sum_brute` | 组合搜索与二进制枚举 | `reaction_network_graph.py`：反应循环检测中的路径枚举与回溯 |

**每一个输入项目均已真实融入，无遗漏、无挂名。**

---

## 三、新增数学物理模型与核心公式

### 3.1 反应 Euler 方程组

二维可压缩反应流的质量、动量、能量与反应进度守恒方程：

$$
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}}{\partial x} + \frac{\partial \mathbf{G}}{\partial y} = \boldsymbol{\omega}(\mathbf{U})
$$

其中守恒量与通量分别为：

$$
\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ E \\ \rho\lambda \end{pmatrix}, \quad
\mathbf{F} = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho uv \\ (E+p)u \\ \rho\lambda u \end{pmatrix}, \quad
\mathbf{G} = \begin{pmatrix} \rho v \\ \rho uv \\ \rho v^2 + p \\ (E+p)v \\ \rho\lambda v \end{pmatrix}
$$

总比能 $E$ 与比内能 $e$ 的关系：

$$
E = \rho e + \frac{1}{2}\rho(u^2+v^2)
$$

### 3.2 化学反应源项（Arrhenius 律）

反应进度 $\lambda \in [0,1]$ 的演化由 Arrhenius 动力学控制：

$$
\frac{\mathrm{d}\lambda}{\mathrm{d}t} = -k(T)\,(1-\lambda)^n, \quad
k(T) = A\,\exp\!\left(-\frac{E_a}{RT}\right)
$$

化学源项向量：

$$
\boldsymbol{\omega} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ \rho Q\,k(T)\,(1-\lambda)^n \\ -\rho\,k(T)\,(1-\lambda)^n \end{pmatrix}
$$

其中 $Q$ 为单位质量化学反应释热，$E_a$ 为活化能，$A$ 为指前因子，$R$ 为通用气体常数。

### 3.3 理想气体状态方程（反应流体）

$$
p = (\gamma - 1)\,\rho\,\bigl[e - (1-\lambda)Q\bigr]
$$

温度由状态方程导出：

$$
T = \frac{p}{\rho\,R_{\text{specific}}}, \quad R_{\text{specific}} = \frac{R}{W_{\text{mol}}}
$$

声速：

$$
a = \sqrt{\frac{\gamma p}{\rho}}
$$

### 3.4 Rankine-Hugoniot 激波关系

对于以速度 $D$ 传播的激波，激波前后压强比与密度比为：

$$
\frac{p_2}{p_1} = 1 + \frac{2\gamma}{\gamma+1}\,(M^2 - 1), \quad
\frac{\rho_2}{\rho_1} = \frac{(\gamma+1)M^2}{(\gamma-1)M^2 + 2}
$$

其中 $M = D/a_0$ 为激波马赫数，$a_0 = \sqrt{\gamma p_0/\rho_0}$ 为未燃气声速。

### 3.5 Chapman-Jouguet (CJ) 爆轰速度

理想气体简化公式：

$$
D_{\mathrm{CJ}}^2 = 2(\gamma^2 - 1)\,Q + \frac{\gamma p_0}{\rho_0}
$$

更一般地，CJ 条件要求 Rayleigh 线与 Hugoniot 曲线相切：

$$
\frac{p - p_0}{v_0 - v} = -\rho_0^2 D_{\mathrm{CJ}}^2, \quad v = \frac{1}{\rho}
$$

Hugoniot 曲线（含释热修正）：

$$
p_H(v) = p_0\,\frac{\frac{\gamma+1}{\gamma-1}\frac{v_0}{v} - 1}{\frac{\gamma+1}{\gamma-1} - \frac{v_0}{v}} + \frac{2\rho_0 Q/(\gamma+1)}{1 - \frac{\gamma-1}{\gamma+1}\frac{v}{v_0}}
$$

### 3.6 ZND 爆轰结构

在随爆轰波以速度 $D$ 运动的坐标系 $\xi = x - Dt$ 中，控制方程简化为：

$$
\rho(u-D) = \dot{m} = \text{const}, \quad
p + \dot{m}(u-D) = \text{const}, \quad
h + \frac{1}{2}(u-D)^2 = \text{const}
$$

反应方程：

$$
\frac{\mathrm{d}\lambda}{\mathrm{d}\xi} = -\frac{A}{D}\,\exp\!\left(-\frac{E_a}{RT}\right)(1-\lambda)^n
$$

### 3.7 线性稳定性分析

对 ZND 剖面施加小扰动 $\mathbf{U}(x,t) = \mathbf{U}_0(x) + \varepsilon\,\mathbf{U}'(x)\,e^{\sigma t}$，得到局部 Jacobian 矩阵 $J$，其特征值 $\sigma = \alpha + i\omega$ 决定稳定性：

- 若 $\alpha > 0$：不稳定模态，爆轰波头将发生振荡；
- 若 $\alpha < 0$：稳定模态；
- 振荡频率 $f = |\omega|/(2\pi)$。

### 3.8 稀疏网格插值

对 $d$ 维函数 $f(\mathbf{y})$，稀疏网格使用 Clenshaw-Curtis 节点：

$$
x_j^{(l)} = \cos\!\left(\frac{\pi j}{2^l}\right), \quad j = 0,\dots,2^l
$$

层级差值（hierarchical surplus）递推构建近似：

$$
f_l(\mathbf{y}) = f_{l-1}(\mathbf{y}) + \sum_{\mathbf{i} \in I_l^{\text{new}}} w_{\mathbf{i}}\,\varphi_{\mathbf{i}}(\mathbf{y})
$$

### 3.9 热力学积分求积

反应区总释热率的空间积分：

$$
\dot{q}_{\text{total}} = \iint_\Omega Q\,\rho\,k(T)\,(1-\lambda)^n \,\mathrm{d}x\,\mathrm{d}y
$$

采用 $[-1,1]^2$ 上的对称 Gauss-Legendre 求积：

$$
\iint_{[-1,1]^2} g(x,y)\,\mathrm{d}x\,\mathrm{d}y \approx \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j\,g(x_i, x_j)
$$

### 3.10 蒙特卡洛点火概率

局部点火条件：

$$
k_{\text{eff}} = A\,\exp\!\left(-\frac{E_a}{RT}\right)\,\phi_{\text{eff}} > k_{\text{ign}}
$$

其中当量比修正：

$$
\phi_{\text{eff}} = \exp\!\left[-\frac{1}{2}\left(\frac{\phi-1}{0.3}\right)^2\right]
$$

---

## 四、文件结构与修改说明

```
157_synth_project/
├── main.py                         # 统一入口，零参数运行
├── combustion_utils.py             # 物理常数、边界检查、公式计算、Cholesky分解
├── reaction_kinetics.py            # 反应状态类、Euler通量、化学反应源项
├── znd_structure.py                # ZND一维结构ODE求解器
├── euler_reactive_solver.py        # 二维反应Euler方程RK3-TVD求解器、CRS稀疏矩阵
├── sparse_grid_chemistry.py        # 高维化学流形稀疏网格插值
├── thermal_quadrature.py           # 热力学积分对称求积
├── adaptive_mesh.py                # 自适应三角形网格生成、CVT迭代、T3基函数
├── monte_carlo_ignition.py         # 椭圆采样、点火概率MC、临界核逃逸时间
├── cj_condition_optimizer.py       # CJ条件优化（Brent根查找/最小化、二次插值）
├── reaction_network_graph.py       # 反应网络图、BFS最短路径、循环检测
├── stability_analysis.py           # ZND线性稳定性分析与特征值分解
└── README_博士级合成说明.md         # 本文档
```

---

## 五、合成项目解决的科学问题

1. **CJ/DCJ 爆轰速度精确求解**：通过 Brent 根查找与局部最小化，在 Hugoniot 曲线与 Rayleigh 线的交点约束下确定爆轰速度，避免简化公式的系统误差。
2. **ZND 爆轰结构解析**：一维反应区中密度、速度、压强与反应进度的空间剖面计算，提取诱导区长度与半反应长度等关键特征尺度。
3. **二维可压缩反应流数值模拟**：使用 TVD-RK3 时间推进与 Lax-Friedrichs 数值通量，捕捉爆轰波在二维平面上的传播与化学反应耦合。
4. **高维化学流形快速评估**：利用稀疏网格插值，在四维参数空间（温度、压强、当量比、残余气体分数）中快速近似复杂反应速率。
5. **反应区热力学积分**：高精度对称求积计算总释热率与平均温度，为爆轰波能量收支提供定量依据。
6. **波前自适应网格生成**：基于 CVT 迭代与自适应密度函数，在爆轰反应区自动生成高分辨率三角形网格。
7. **湍流点火概率评估**：蒙特卡洛采样结合批次统计，估算局部热点在随机温度、压强与当量比涨落下的点火概率。
8. **临界热点核分析**：通过逃逸时间迭代，判断不同尺寸热点是否能在有限时间内自点燃。
9. **反应网络路径分析**：图论方法分析 H₂-O₂ 链式反应中的最短路径、反应循环与网络聚类特性。
10. **线性稳定性分析**：基于 ZND 剖面构建全局稳定性矩阵，识别不稳定振荡模态并估算爆轰头脉动频率。

---

## 六、运行方式

```bash
cd Synthesis-project-python/157_synth_project
python main.py
```

无需任何命令行参数。程序将自动执行以下 10 个计算模块并输出结果：

1. CJ/DCJ 爆轰条件优化求解
2. ZND 爆轰结构一维求解
3. 二维可压缩反应 Euler 方程数值模拟
4. 稀疏网格高维化学流形插值
5. 热力学积分高精度对称求积
6. 爆轰波前自适应网格生成
7. 蒙特卡洛点火概率与临界核分析
8. H₂-O₂ 燃烧反应网络图分析
9. 爆轰波线性稳定性分析
10. 稀疏矩阵 CRS 格式验证

---

## 七、数值鲁棒性与边界处理

- **正定性检查**：所有密度、压强、温度输入均通过 `check_positive` 与 `check_nonnegative` 校验。
- **截断保护**：反应进度 $\lambda$ 严格截断至 $[0,1]$；温度下限设为 $10^{-6}\,\mathrm{K}$ 防止除零。
- **CFL 约束**：二维 Euler 求解器采用自动 CFL 时间步长计算，确保显式格式的线性稳定性。
- **指数溢出防护**：Arrhenius 指数项在指数小于 $-700$ 时返回 $0$，大于 $700$ 时返回极大值，避免 `np.exp` 溢出。
- **稀疏矩阵维度校验**：CRS 矩阵-向量乘法前检查向量长度，不匹配时抛出清晰异常。

---

## 八、前沿性与博士级难度说明

本项目涵盖的爆轰波问题涉及**非线性双曲守恒律与刚性化学反应源项的耦合**（反应 Euler 方程）、**高维参数空间的稀疏网格逼近**（维数灾难的缓解）、**多尺度结构分析**（ZND 反应区尺度与流体力学尺度的分离），以及**图论与随机过程在燃烧化学中的应用**。这些问题的综合求解在计算燃烧学领域属于典型的博士研究课题，需要深厚的偏微分方程、数值分析、反应动力学与概率统计知识。
