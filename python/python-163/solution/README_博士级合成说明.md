# 地热储层热流固耦合（THM）博士级合成项目

## 一、项目概述

本项目围绕**能源系统：地热储层热流固耦合**这一前沿科学领域，基于15个输入种子项目的核心算法，融合构建了一个面向增强型地热系统（EGS）的博士级热-流-固全耦合数值模拟框架。

### 核心科学问题

模拟深部地热储层在流体注入-采出过程中的**热-流-固（Thermal-Hydro-Mechanical, THM）全耦合演化**，包括：
- **热传递**：多孔介质中热传导与对流输运的耦合
- **流体流动**：达西定律与Forchheimer非达西修正
- **岩石力学**：孔隙弹性变形与热应力耦合
- **裂缝演化**：马尔可夫链模型描述裂缝开度动态变化
- **不确定性量化**：蒙特卡洛方法评估参数不确定性

---

## 二、原项目到科学问题的映射

| 编号 | 原项目名称 | 核心算法 | 在合成项目中的角色 |
|------|-----------|---------|------------------|
| 595 | interp_spline_data | 三次样条插值（not-a-knot） | `spline_properties.py`：温度依赖的岩石热导率、流体粘度的样条插值 |
| 837 | opt_sample | 随机采样估计函数极值 | `stochastic_fields.py`：Latin Hypercube参数空间采样、蒙特卡洛采样器 |
| 532 | hexahedron_witherden_rule | 六面体高精度求积规则 | `quadrature_rules.py`：Witherden-Vincent六面体求积，用于3D单元积分 |
| 1026 | risk_matrix | 风险转移矩阵（马尔可夫链） | `risk_fracture.py`：裂缝网络连通性演化的马尔可夫链模型 |
| 1169 | stochastic_diffusion | 2D随机扩散系数场 | `stochastic_fields.py`：基于KL展开的随机渗透率场、随机热扩散系数 |
| 630 | kursiv_pde_etdrk4 | ETD-RK4指数时间差分法 | `etdrk4_solver.py`：刚性热对流扩散方程的谱/ETD-RK4时间步进 |
| 1298 | triangle_analyze | 三角形几何质量分析 | `mesh_geometry.py`：三角形网格质量度量（面积、内切圆、外接圆、质量因子） |
| 718 | matlab_commandline | 命令行文件脚本 | `io_utils.py`：模拟数据的格式化输出与参数表生成 |
| 232 | cube_felippa_rule | 三维立方体高斯求积 | `quadrature_rules.py`：1D~5D乘积型高斯-勒让德求积规则 |
| 941 | quad_monte_carlo | 蒙特卡洛数值积分 | `monte_carlo_uq.py`：蒙特卡洛不确定度量化、热采出期望值估计 |
| 546 | house_data | 多边形顶点数据 | `mesh_geometry.py`：储层边界多边形表示与三角剖分 |
| 644 | lambert_w | Lambert W函数逼近 | `lambert_flow.py`：井筒压力降模型中的超越方程求解 |
| 338 | errors | 数值误差与收敛分析 | `convergence_analysis.py`：L2/L∞误差、收敛速率、Richardson外推、Newton-Raphson |
| 209 | conte_deboor | 离散正交多项式拟合 | `orthogonal_fit.py`：渗透率场与热导率场的正交多项式基表示 |
| 1234 | tet_mesh_q2l | 二次-线性四面体网格转换 | `mesh_geometry.py`：二次四面体剖分为8个线性四面体 |

---

## 三、新增数学物理模型与核心公式

### 3.1 热传递方程

```
(ρc)_eff ∂T/∂t + ρ_f c_f q·∇T = ∇·(λ_eff ∇T) + Q_T
```

其中有效热容 `(ρc)_eff = φ ρ_f c_f + (1-φ) ρ_r c_r`，有效热导率 `λ_eff = φ λ_f + (1-φ) λ_r`。

### 3.2 达西定律与Forchheimer修正

```
q = -(k/μ)(∇p + ρ_f g ∇z)
```

井筒压力降使用Blasius关联式与Forchheimer项：

```
Δp = f (L/D) (ρ v²)/2 + β_F ρ v² L
```

### 3.3 孔隙弹性力学平衡

```
∇·σ + ρ_b g = 0
σ = C:ε - α p I - β K_T (T - T_0) I
```

平面应变条件下的应力-应变关系：

```
σ_xx = (λ+2μ) ε_xx + λ ε_zz - α p - β K_T ΔT
σ_zz = λ ε_xx + (λ+2μ) ε_zz - α p - β K_T ΔT
σ_xz = 2μ ε_xz
```

### 3.4 ETD-RK4时间步进（谱方法）

对于傅里叶空间中的方程 `∂v̂/∂t = L v̂ + N(v̂)`：

```
v̂_{n+1} = e^{Δt L} v̂_n + f_1 N_1 + 2 f_2 (N_2 + N_3) + f_3 N_4
```

系数 `Q, f_1, f_2, f_3` 通过围道积分计算：

```
Q = Δt · Re[ (1/M) Σ_m (e^{z_m/2} - 1) / z_m ]
f_1 = Δt · Re[ (1/M) Σ_m (-4 - z_m + e^{z_m}(4 - 3z_m + z_m²)) / z_m³ ]
```

### 3.5 对数正态随机渗透率场（Karhunen-Loève展开）

```
ln k(x,ω) = ln k_0 + Σ_m √λ_m φ_m(x) ξ_m(ω)
```

### 3.6 Lambert W函数在井筒模型中的应用

求解 `p e^{βp} = C` 的超越方程：

```
p = W(β C) / β
```

### 3.7 裂缝三次定律

```
k_eff = k_m + (ρ_f g / 12μ) Σ_i a_i³ b_i
```

---

## 四、项目文件结构

```
163_synth_project/
├── main.py                          # 统一入口，零参数运行
├── thm_model.py                     # THM控制方程与物性参数
├── spline_properties.py             # 三次样条插值物性（595）
├── orthogonal_fit.py                # 正交多项式拟合（209）
├── stochastic_fields.py             # 随机渗透率/扩散场（1169, 837）
├── mesh_geometry.py                 # 网格生成与质量分析（1298, 1234, 546）
├── quadrature_rules.py              # 3D求积规则（532, 232）
├── etdrk4_solver.py                 # ETD-RK4热输运求解器（630）
├── lambert_flow.py                  # Lambert W井筒模型（644）
├── risk_fracture.py                 # 裂缝马尔可夫演化（1026）
├── convergence_analysis.py          # 误差与收敛分析（338）
├── poroelastic_solver.py            # 孔隙弹性力学求解器
├── monte_carlo_uq.py                # 蒙特卡洛不确定度量化（941）
├── io_utils.py                      # 数据I/O（718）
└── README_博士级合成说明.md          # 中文说明文档
```

---

## 五、运行方法

```bash
cd /mnt/data/zpy/sci-swe/source code/Synthesis-project-python/163_synth_project
python main.py
```

程序将自动执行：
1. 参数初始化与物性样条构建
2. 随机渗透率场与扩散场生成
3. 正交多项式拟合
4. 储层网格生成与质量分析
5. 六面体/立方体高精度数值积分测试
6. ETD-RK4热输运时间步进（2D有限差分 + 1D谱方法）
7. 孔隙弹性力学平衡求解（Gauss-Seidel迭代）
8. 裂缝马尔可夫演化、Lambert W井筒模型、蒙特卡洛UQ
9. 结果文件输出到 `./thm_output/`

---

## 六、科学难度说明

本项目涉及的数学物理难度达到博士研究水平：

1. **多物理场耦合**：热、流、固三场强耦合，涉及非线性偏微分方程组
2. **高阶数值方法**：ETD-RK4指数时间差分法（针对刚性问题）、谱方法、高斯求积
3. **随机偏微分方程**：Karhunen-Loève展开描述随机场，蒙特卡洛采样
4. **孔隙弹性力学**：Biot理论框架下的位移-压力-温度耦合
5. **超越方程求解**：Lambert W函数在井筒非线性模型中的解析应用
6. **收敛性分析**：Richardson外推、Newton-Raphson、多重迭代器监控
7. **网格质量理论**：三角形/四面体几何质量度量与剖分算法
