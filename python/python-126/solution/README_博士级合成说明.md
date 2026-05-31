# PROJECT_126：全身 PBPK 药物代谢动力学一体化建模系统

## 一、项目概述

本项目围绕 **生物医学：药物代谢动力学 PBPK（Physiologically Based Pharmacokinetic）模型** 展开，将 15 个种子科研代码项目的核心算法融合重构为一个面向前沿科学问题的博士级 Python 计算系统。

**核心科学问题**：
> 基于多尺度随机微分方程、稀疏网格不确定性量化、刚性 ODE 系统与动态规划剂量优化的全身药物分布-代谢-毒性一体化建模。

项目实现了从分子尺度的随机扩散、组织尺度的有限差分离散、器官尺度的几何建模，到全身尺度的多 compartment 刚性动力学、高维参数不确定性量化以及最优给药策略的完整计算链路。

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 在 PBPK 中的科学角色 |
|---|---|---|
| 648_laplacian_matrix | 1D Laplacian 有限差分、特征值/特征向量显式公式 | 组织内药物扩散的空间离散化与稳态浓度求解 |
| 424_feynman_kac_3d | Feynman-Kac 公式、3D Monte Carlo 随机游走 | 药物分子在组织中的布朗运动与边界吸收概率 |
| 1213_test_interp_fun | 经典插值测试函数（Runge、振荡、分段） | 药时曲线重构与浓度-效应（PD）非线性关系建模 |
| 1283_tough_ode | Hairer 刚性 ODE benchmark | 多 compartment 刚性系统求解器验证 |
| 823_obj_to_tri_surface | OBJ 三角剖分（fan triangulation） | 器官 3D 几何近似与表面积/体积计算 |
| 116_box_plot | 离散化 RGB 数据分箱 | 药物浓度分布的统计分箱与直方图分析 |
| 1378_usa_cvt_geo | Lloyd 算法 CVT（Centroidal Voronoi Tessellation） | 器官内部最优采样点生成（Monte Carlo 积分） |
| 1007_random_sorted | 顺序统计量生成（指数间距、逆 CDF） | 药物分子到达时间随机分布与生理参数 Latin-Hypercube 采样 |
| 898_polynomials | 多元多项式 benchmark（Rosenbrock、Himmelblau 等） | 药物-受体结合势能面与多靶点优化目标函数 |
| 551_hyper_2f1 | 高斯超几何函数 $_2F_1$ | 药物-血浆蛋白结合分数的解析表达式 |
| 327_elfun | Carlson 椭圆积分、Jacobi 椭圆函数、AGM | 非均质组织中各向异性有效扩散系数计算 |
| 704_luhn | Luhn 校验算法 | 临床试验患者 ID / 药物批号的数据完整性校验 |
| 1103_sparse_grid_cc | Smolyak 稀疏网格 Clenshaw-Curtis 求积 | 高维生理参数空间的不确定性量化（UQ） |
| 624_knapsack_dynamic | 0/1 背包动态规划 | 多器官剂量分配与给药时间表的优化 |
| 1144_square_felippa_rule | 2D Felippa 乘积求积规则 | 器官切片上药物浓度的二维面积分 |

---

## 三、新增数学物理模型与核心公式

### 3.1 组织扩散方程（基于 Laplacian 离散化）

组织内药物稳态分布满足带有清除的扩散方程：

$$
-D_{\text{eff}} \nabla^2 C(x) + k_{\text{cl}} C(x) = 0, \quad x \in (0, L)
$$

边界条件：$C(0) = C_{\text{influx}}$，$C(L) = 0$（Dirichlet-Dirichlet）。

解析解：

$$
C(x) = C_{\text{influx}} \frac{\sinh\bigl(\alpha (L - x)\bigr)}{\sinh(\alpha L)}, \quad \alpha = \sqrt{\frac{k_{\text{cl}}}{D_{\text{eff}}}}
$$

有限差分离散化（1D DD，$n$ 个内部节点，步长 $h = L/(n+1)$）：

$$
A = \frac{1}{h^2} \begin{pmatrix}
2 & -1 & & \\
-1 & 2 & -1 & \\
& \ddots & \ddots & \ddots \\
& & -1 & 2
\end{pmatrix}
$$

特征值与特征向量的显式公式：

$$
\lambda_j = \frac{4}{h^2} \sin^2\!\left(\frac{j\pi}{2(n+1)}\right), \quad
V_{ij} = \sqrt{\frac{2}{n+1}} \sin\!\left(\frac{ij\pi}{n+1}\right)
$$

### 3.2 非均质组织有效扩散系数（基于 AGM 与椭圆积分）

生物组织具有纤维各向异性，主轴扩散系数为 $D_{\parallel}$ 和 $D_{\perp}$。旋转到纤维取向角 $\theta$ 的坐标系：

$$
D_{11} = D_{\parallel} \cos^2\theta + D_{\perp} \sin^2\theta, \quad
D_{22} = D_{\parallel} \sin^2\theta + D_{\perp} \cos^2\theta
$$

有效扩散系数通过 Gauss 算术-几何平均（AGM）计算：

$$
D_{\text{eff}}(\theta) = \frac{\pi}{4 \cdot \text{AGM}\bigl(D_{11}^{-1/2}, D_{22}^{-1/2}\bigr)}
$$

其中 AGM 迭代：

$$
a_{k+1} = \frac{a_k + b_k}{2}, \quad b_{k+1} = \sqrt{a_k b_k}
$$

### 3.3 Feynman-Kac 随机路径积分

3D 椭圆型边界值问题：

$$
\frac{1}{2} \Delta U - V(\mathbf{x}) U = 0 \quad \text{in } \Omega, \qquad
U = g(\mathbf{x}) \quad \text{on } \partial\Omega
$$

Feynman-Kac 公式给出概率表示：

$$
U(\mathbf{x}_0) = \mathbb{E}\left[ \exp\!\left(-\int_0^{\tau} V(\mathbf{X}_s)\, ds\right) g(\mathbf{X}_{\tau}) \right]
$$

其中 $\tau$ 为首次退出时间，$\mathbf{X}_t$ 为 3D 布朗运动，离散化采用 Euler-Maruyama 方案：

$$
\mathbf{X}_{k+1} = \mathbf{X}_k + \sqrt{3h}\, \boldsymbol{\xi}_k, \quad \boldsymbol{\xi}_k \in \{ \pm\mathbf{e}_1, \pm\mathbf{e}_2, \pm\mathbf{e}_3 \}
$$

### 3.4 药物-蛋白结合分数（超几何函数解析解）

$n$ 个结合位点的药物-蛋白结合，未结合分数 $f_u$：

$$
f_u = \frac{1}{{_2F_1}\bigl(1, n; n+1; -K_a C_p\bigr)}
$$

其中 $_2F_1(a,b;c;z)$ 为高斯超几何函数：

$$
{_2F_1}(a,b;c;z) = \sum_{k=0}^{\infty} \frac{(a)_k (b)_k}{(c)_k} \frac{z^k}{k!}
$$

$(a)_k = a(a+1)\cdots(a+k-1)$ 为 Pochhammer 符号。

### 3.5 多 Compartment PBPK 刚性 ODE 系统

7-compartment 系统（动脉血、肝、肾、肌肉、脂肪、肿瘤、静脉血）：

$$
V_i \frac{dC_i}{dt} = Q_i \left(C_{\text{art}} - \frac{C_i}{K_{p,i}}\right) - \text{CL}_i C_i + \text{Input}_i(t)
$$

**肝脏 Michaelis-Menten 代谢**：

$$
\text{Met Rate} = \frac{V_{\max} C_{\text{liv,free}}}{K_m + C_{\text{liv,free}}}, \quad
C_{\text{liv,free}} = f_u \frac{C_{\text{liv}}}{K_{p,\text{liv}}}
$$

**肾脏 GFR 清除**：

$$
\text{GFR Clear} = \text{GFR} \cdot C_{\text{kid,free}}
$$

**口服吸收（一级动力学）**：

$$
\text{Input}(t) = D_{\text{ose}} \cdot k_a \cdot e^{-k_a t}
$$

该系统具有时间尺度分离（肝脏分钟级 vs 脂肪小时级），属于刚性系统，采用 Rosenbrock 线性隐式方法求解：

$$
\bigl(I - h\gamma J(t_n, y_n)\bigr) k = f(t_n, y_n), \quad
y_{n+1} = y_n + h k, \quad \gamma = 1.0
$$

### 3.6 Smolyak 稀疏网格 Clenshaw-Curtis 不确定性量化

$d$ 维 Smolyak 稀疏网格求积：

$$
Q_L^{(d)} f = \sum_{|i|_1 \leq L+d-1} (-1)^{L+d-1-|i|_1} \binom{d-1}{L+d-1-|i|_1}
\bigl(U_{i_1} \otimes \cdots \otimes U_{i_d}\bigr) f
$$

其中 $i_k \geq 1$，$U_{i}$ 为 1D Clenshaw-Curtis 规则（$m_i = 2^{i-1}+1$ 个嵌套 Chebyshev 节点）。

1D 节点：$x_j = \cos(j\pi/n)$，$j=0,\dots,n$，$n=2^{i-1}$。

权重通过 Chebyshev 基线性系统求解：

$$
\sum_{j=0}^{n} w_j T_k(x_j) = \int_{-1}^{1} T_k(x)\, dx, \quad k=0,\dots,n
$$

### 3.7 动态规划剂量优化

**多器官剂量分配**（0/1 背包框架）：

$$
\max \sum_i \text{sensitivity}_i \cdot d_i \quad
\text{s.t.} \quad \sum_i d_i \leq D_{\text{total}}, \quad
\sum_i \text{toxicity}_i \cdot d_i \leq T_{\max}
$$

**给药时间表优化**：

$$
\max \sum_t \bigl[ E(d_t) - \lambda \cdot \text{Tox}(d_t) \bigr] \quad
\text{s.t.} \quad t_{j+1} - t_j \geq \Delta t_{\min}
$$

### 3.8 经典插值测试函数（药时曲线重构）

**Runge 函数**（测试外推稳定性）：

$$
f(x) = \frac{1}{1+x^2}
$$

**高振荡函数**（模拟快速 PK 变化）：

$$
f(x) = \sqrt{x(1-x)} \sin\!\left(\frac{2.1\pi}{x+0.05}\right)
$$

**Hill 方程药效模型**：

$$
E(C) = E_0 + \frac{E_{\max} C^n}{C_{50}^n + C^n}
$$

---

## 四、项目文件结构与改造方法

### 4.1 文件清单（共 13 个 Python 文件）

| 文件名 | 对应种子项目 | 科学功能 |
|---|---|---|
| `main.py` | — | 统一入口，零参数运行，调用 12 个模块完成全流程 |
| `pbpk_special_functions.py` | 327 + 551 | Carlson 椭圆积分、Jacobi 函数、AGM、超几何 $_2F_1$、药物结合分数 |
| `pbpk_random.py` | 1007 | 均匀/正态顺序统计量、Wichura 逆 CDF、药物到达时间采样 |
| `pbpk_quadrature.py` | 1103 + 1144 | Smolyak 稀疏网格 CC、Felippa 2D 乘积规则、高维 UQ |
| `pbpk_diffusion.py` | 648 | 1D/3D Laplacian 矩阵、特征分解、稳态扩散求解 |
| `pbpk_stochastic.py` | 424 | 3D 随机游走、Feynman-Kac Monte Carlo、器官吸收概率 |
| `pbpk_ode_solver.py` | 1283 | RK4、隐式梯形、Rosenbrock 方法、7-compartment PBPK 刚性系统 |
| `pbpk_geometry.py` | 823 + 1378 | 椭球三角剖分、CVT Lloyd 算法、多器官几何配置 |
| `pbpk_polynomials.py` | 898 | Rosenbrock/Himmelblau landscape、药物-受体势能、酶动力学 |
| `pbpk_interpolation.py` | 1213 | Runge/Bernstein/振荡函数、Chebyshev/Lagrange/分段线性插值、PD 曲线 |
| `pbpk_optimization.py` | 624 | 0/1 背包 DP、剂量分配优化、给药时间表优化、药物组合优化 |
| `pbpk_utils.py` | 704 + 116 | Luhn 校验、浓度分箱统计、安全运算、生理常数、单位转换 |

### 4.2 改造方法说明

1. **算法迁移**：将原 MATLAB 代码的核心数值思想（Landen 变换、AGM 迭代、稀疏网格组合、背包 DP 等）用 Python + NumPy 重新实现。
2. **科学注入**：每个模块都新增了与 PBPK 直接相关的物理/化学模型（Michaelis-Menten、GFR、Hill 方程、Fick 定律、对流-扩散方程等）。
3. **工程强化**：添加了全面的边界检查（非负浓度、正扩散系数、有效步长等）、数值鲁棒性工具（safe_divide、safe_exp、softplus、clip_concentration）。
4. **删除可视化**：所有与绘图、图像显示相关的代码均已移除，仅保留数值计算与文本输出。

---

## 五、合成项目解决的科学问题

1. **组织内药物空间分布预测**：通过有限差分离散化与特征值分解，求解带有清除项的稳态扩散方程，预测药物在组织深处的穿透深度。
2. **分子尺度随机输运评估**：利用 Feynman-Kac 公式计算药物分子在 3D 器官几何中的保留概率与靶向 hit 概率。
3. **全身药代动力学瞬态模拟**：求解 7-compartment 刚性 ODE 系统，预测口服给药后各器官的药时曲线、Cmax、Tmax、AUC。
4. **高维参数不确定性量化**：使用 Smolyak 稀疏网格对生理参数（体重、心输出量、游离分数等）进行 Monte Carlo 采样，评估药物暴露量的统计分布。
5. **最优给药策略设计**：通过动态规划优化剂量在多器官间的分配、给药时间表以及多药联用组合，在疗效与毒性之间取得平衡。
6. **药物-蛋白相互作用分析**：利用超几何函数解析计算游离药物分数，评估血浆蛋白结合对药效的影响。

---

## 六、如何运行

```bash
cd Synthesis-project-python/126_synth_project
python main.py
```

程序零参数运行，自动执行以下 12 个科学计算模块：
1. 特殊函数验证（Carlson RF、Jacobi sn/cn/dn、AGM、$_2F_1$）
2. 随机采样与顺序统计量
3. 稀疏网格与二维求积验证
4. 有限差分 Laplacian 与稳态扩散
5. Feynman-Kac 1D/3D Monte Carlo
6. 刚性 ODE 求解与 PBPK 瞬态动力学
7. 器官几何建模与 CVT 采样
8. 多项式势能与多靶点目标函数
9. 插值测试与 PD 响应计算
10. 动态规划剂量优化
11. 数据完整性校验与数值鲁棒性
12. 综合 Monte Carlo 不确定性量化

---

## 七、质量检查清单

- [x] 原目录未被修改
- [x] 合成项目为 Python 语言
- [x] 新目录完整包含合成项目（13 个 .py 文件）
- [x] 只有一个博士级科学计算问题（全身 PBPK 建模）
- [x] **15 个输入项目均已真实融入**，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成
- [x] 无可视化代码
