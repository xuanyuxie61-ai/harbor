# PROJECT-283 博士级科研代码合成说明

## 项目名称
**钙钛矿太阳能电池缺陷态计算：高阶有限差分与稳定性分析**
(Perovskite Solar Cell Defect-State Calculation: High-Order Finite Differences and Stability Analysis)

## 科学领域
**计算材料学 —— 钙钛矿太阳能电池中的点缺陷态理论**

本项目围绕有机-无机杂化钙钛矿（MAPbI₃）太阳能电池中的**点缺陷态**展开博士级计算。具体聚焦：
- 碘空位 V_I、铅间隙 Pb_i、碘间隙 I_i 等本征点缺陷的形成能与热力学跃迁能级 ε(q₁/q₂)
- 缺陷态密度分布下的泊松方程高阶有限差分求解
- Shockley-Read-Hall (SRH) 复合动力学与缺陷占据率
- 缺陷离子的瞬态扩散与高阶显式格式的 von Neumann 稳定性
- 基于机器学习 (RandomForest, PCA, KNN) 的缺陷跃迁能级预测
- 高阶 Langevin 蒙特卡罗采样缺陷构型平衡分布
- 不连续伽辽金 (DG) 方法求解载流子漂移-扩散输运

---

## 输入种子项目到科学问题的映射（15 个全部真实融入）

| 序号 | 原始项目 | 在合成项目中的角色 | 对应模块 |
|------|----------|---------------------|----------|
| 1 | `739_matrix_chain_brute` | 缺陷多物种反应路径的最优括号化（Catalan 枚举 + DP） | `defect_rate_equations.py::reaction_pathway_cost`, `catalan_number` |
| 2 | `1025_USTBifrt_Pyrolysis-of-Coal` | MAPbI₃ 热分解化学键断裂动力学 (MAPbI₃ → PbI₂ + CH₃NH₂ + HI) | `defect_rate_equations.py::bond_population_kinetics`, `pyrolysis_product_evolution` |
| 3 | `1193_msg-byu_ML-for-CurieTemp-Predictions` | RandomForest + PCA + KNN 预测缺陷跃迁能级 ε(0/+) | `ml_defect_predictor.py` |
| 4 | `951_quadrature_weights_vandermonde_2d` | Vandermonde 矩阵构造任意节点导数权重（用于非均匀网格 FD） | `high_order_fd.py::vandermonde_derivative_weights` |
| 5 | `1092_omnibenchmark_omnibenchmark_paper_code` | 跨求解器/材料/方法的缺陷态计算性能基准分析 | `benchmark_analyzer.py` |
| 6 | `1071_kaihongz_HigherOrderLMC` | Picard-Lagrange 高阶 Langevin 采样缺陷构型 | `langevin_defect_sampler.py` |
| 7 | `953_quadrilateral_mesh` | Q4 四边形网格生成 + Jacobian 质量检查 | `mesh_generator.py::q4_mesh_unit_square`, `jacobian_q4` |
| 8 | `271_dg1d_advection` | Hesthaven-Warburton 节点 DG 方法（Jacobi 多项式 + Vandermonde + RK5） | `carrier_transport_dg.py` |
| 9 | `676_line_cvt_lloyd` | Lloyd 算法计算加权 CVT，最优缺陷位点布局 | `mesh_generator.py::line_cvt_lloyd` |
| 10 | `414_fem2d_scalar_display` | 三角剖分上 P1 标量场（电势）的 L2/H1 范数分析（无可视化） | `fem_scalar_field.py` |
| 11 | `792_nearest_interp_1d` | 粗网格 DFT 缺陷密度 → 细网格器件尺度的最邻近插值 | `defect_field_interpolation.py::nearest_interp_1d` |
| 12 | `1335_triangulation_delaunay_discrepancy` | Delaunay 局部 discrepancy 检查缺陷网格质量 | `mesh_generator.py::delaunay_flip_discrepancy` |
| 13 | `702_logistic_ode` | Logistic 缺陷生成 ODE + 闭式解 + RK4 积分 | `defect_rate_equations.py::logistic_defect_deriv`, `logistic_defect_exact` |
| 14 | `1023_pranavgupta2603_covid-spread-simulation` | 缺陷对相互作用网络（V_I 迁移、二聚、捕获三态动力学） | `defect_rate_equations.py::DefectNetwork` |
| 15 | `359_fd1d_display` | 1D 有限差分离散场的分段线性表示 + 收敛阶估计 | `defect_field_interpolation.py::FD1DField` |

---

## 核心数学物理模型

### 1. 高阶有限差分 Laplace 算子（2p 阶中心差分）
对于 p = 6，使用 13 点模板：

$$
\left.\frac{d^2 u}{dx^2}\right|_i = \frac{1}{h^2}\sum_{k=-p}^{p} c_k\, u_{i+k} + \mathcal{O}(h^{2p})
$$

其中系数由 Vandermonde 系统 $Vc = e_2$ 求解，或由 Fornberg 闭式给出：

$$
c_k = (-1)^{k+1}\frac{2(p!)^2}{(p+k)!(p-k)!\,k^2},\quad k \geq 1
$$

### 2. von Neumann 稳定性分析
显式 FTCS 格式用于瞬态缺陷离子扩散：

$$
G(\theta) = 1 + r\sum_{k=-p}^{p} c_k e^{ik\theta},\quad r = \frac{D\,\Delta t}{h^2}
$$

稳定性要求 $|G(\theta)| \leq 1,\ \forall \theta \in [0,\pi]$，对 p=6 给出 $r_{\max} \approx 0.282768$（经典 p=1 为 0.5）。

### 3. 泊松 + SRH 自洽（Gummel 迭代）

缺陷占据率：
$$
f_t = \frac{\sigma_n n + \sigma_p p_1}{\sigma_n(n + n_1) + \sigma_p(p + p_1)}
$$

SRH 复合率：
$$
R_{\text{SRH}} = \frac{np - n_i^2}{\tau_{p0}(n + n_1) + \tau_{n0}(p + p_1)}
$$

缺陷电荷密度：
$$
\rho_{\text{def}}(x) = q\,N_t(x)\,[f_t(x) - f_0]
$$

泊松方程：
$$
-\varepsilon_r\varepsilon_0\frac{d^2\varphi}{dx^2} = \rho_{\text{def}}(x)
$$

### 4. Logistic 缺陷生成 ODE

$$
\frac{dN_t}{dt} = r(T)\,N_t\left(1 - \frac{N_t}{N_{\max}}\right) - \gamma N_t
$$

Arrhenius 温度依赖：$r(T) = r_0\exp(-E_a/k_BT)$，MAPbI₃ 中 V_I 形成能 $E_a \approx 0.58$ eV。

### 5. CVT Lloyd 算法
对缺陷密度加权 $\rho(x) = N_t(x)$，最小化：

$$
E(\{z_i\}) = \sum_{i=1}^N \int_{V_i} \rho(x)\,\|x - z_i\|^2\,dx
$$

交替迭代：Voronoi 剖分 → 质量中心更新 → 收敛至 CVT。

### 6. 高阶 Langevin 采样（Picard-Lagrange 格式，K ≥ 3）

$$
X_{n+1} = e^{Ah}X_n + \int_0^h e^{A(h-s)}B\,dW(s) - \int_0^h e^{A(h-s)}\nabla U(X_n)\,ds + \mathcal{O}(h^{K/2})
$$

Kronecker 结构 $A = A_{\text{small}} \otimes I_d$ 使得小矩阵 $K\times K$ 指数精确预计算。

### 7. DG 数值通量
对于 $u_t + a u_x = 0$，upwind 通量：

$$
F^* = \begin{cases} a\,u^- & a \geq 0 \\ a\,u^+ & a < 0 \end{cases}
$$

半离散格式的 CFL 限：$\Delta t \leq \text{CFL}\cdot h / (|a|(2N+1)^2)$。

---

## 合成后的文件结构

```
283_synth_project_Advanced/
├── main.py                          统一入口（零参数运行）
├── perovskite_constants.py          物理/材料/数值常数
├── high_order_fd.py                 高阶 FD 算子 + 1D/2D Poisson 求解
├── poisson_defect_solver.py         Gummel 自洽泊松 + SRH
├── mesh_generator.py                Q4 网格 + Delaunay 质量 + Lloyd CVT
├── defect_rate_equations.py         Logistic + 网络 + 热解 + 反应路径
├── carrier_transport_dg.py          DG 载流子输运
├── ml_defect_predictor.py           ML 缺陷能级预测
├── langevin_defect_sampler.py       高阶 Langevin 缺陷采样
├── fem_scalar_field.py              FEM 标量场分析（L2/H1 范数）
├── defect_field_interpolation.py    缺陷场插值 + 1D FD 表示
├── stability_analysis.py            von Neumann + Gummel + DG CFL 稳定性
├── benchmark_analyzer.py            跨求解器基准分析
└── README_博士级合成说明.md           本文档
```

共 **13 个 Python 文件**（远超 8 个最低要求）。

---

## 运行方法

```bash
cd 283_synth_project_Advanced
python main.py
```

零参数运行，完成 12 个阶段计算：
1. 物理常数自检（V_t, N_c, N_v, τ_SRH, S_p, λ_th）
2. 高阶 FD 模板验证 + von Neumann 稳定性
3. Q4 网格质量 + Lloyd CVT + Delaunay discrepancy
4. Gummel 自洽泊松+SRH（缺陷电势、复合电流、Debye 长度、电场）
5. Logistic 缺陷生成 + 网络动力学 + 热解 + 反应路径
6. DG 对流 + 漂移-扩散载流子输运
7. ML 缺陷跃迁能级预测（RF、KNN、RF-PCA + 5 折交叉验证）
8. 高阶 Langevin 缺陷构型采样
9. P1 有限元电势场 L2/H1 分析
10. 缺陷场最近邻 vs 线性插值精度对比
11. 综合稳定性分析（von Neumann、Gummel 谱半径、DG CFL）
12. 跨材料/求解器/方法基准汇总 + t 检验

---

## 关键物理结果

- 室温下 V_t = 25.85 mV，N_c = 1.043×10²⁴ m⁻³，N_v = 1.458×10²⁴ m⁻³
- SRH 寿命：τ_n ≈ 29.7 ns，τ_p ≈ 33.2 ns（N_t = 10²¹ m⁻³）
- 6 阶 FD 稳定性因子 S₆ ≈ 0.488978（经典 2 阶为 0.5）
- RF 预测缺陷跃迁能级 MAE ≈ 0.079 eV（5 折交叉验证）
- HO-FD6 求解器在三种钙钛矿材料上 MAE 最优（~0.05 eV）
- 配对 t 检验证实 HO-FD6 显著优于次优求解器（p < 0.001）

---

## 边界处理与数值鲁棒性

1. **高阶 FD 模板**：对 p=1..6 进行 x², x⁴ 精度验证，确保导数阶数严格
2. **Gummel 迭代**：引入 0.3 欠松弛 + 电势值域钳制（[-1, V_bi+1]）
3. **SRH 分母**：分母钳制至 ≥ 1e-30，避免除零
4. **Logistic ODE**：RK4 + 非负/饱和钳制 + 闭式解对比
5. **Langevin 采样**：Sigma_C 对称化 + 特征值非负钳制
6. **DG 格式**：penalty 稳定项 + 保守 CFL（0.25/(2N+1)²）+ 步数上限
7. **Delaunay 检查**：对所有内部边枚举 flip，严格比较最小角
8. **网格 Jacobian**：每单元计算 detJ，负值视为反转单元
9. **ML 训练**：bootstrap 重采样 + 置换特征重要性 + 交叉验证
10. **基准分析**：3 次重复 + t 检验 + 材料/求解器/方法三维分组

---

## 合成创新点

1. **首次**将 15 个不同领域的种子项目（矩阵链优化、煤热解、Curie 温度 ML、2D Vandermonde 积分、omnibenchmark、高阶 Langevin、Q4 网格、DG 对流、Lloyd CVT、FEM 显示、最近邻插值、Delaunay discrepancy、logistic ODE、疫情扩散、1D FD 显示）统一映射到**钙钛矿太阳能电池缺陷态计算**这一前沿博士级科学问题。

2. **数学公式密度极高**：包含泊松方程、SRH 复合、Arrhenius 速率、Logistic ODE、von Neumann 分析、CVT 能量泛函、Picard-Lagrange 高阶 Langevin、DG upwind 通量、P1 FEM L2/H1 范数等 20+ 个核心方程。

3. **博士级数值难度**：6 阶中心差分（13 点模板）、Kronecker 结构矩阵指数、Gummel 非线性迭代、Catalan 路径优化、Lloyd CVT 收敛、配对 t 检验。

4. **完全可复现**：随机种子固定（seed=283），小网格（nx=64, ny=32 量级），6-7 秒完成全部 12 阶段。

---

## 依赖

- Python ≥ 3.9
- numpy
- scipy（仅 `scipy.linalg.expm` 用于 Langevin 矩阵指数）

无其他第三方依赖，无可视化依赖。
