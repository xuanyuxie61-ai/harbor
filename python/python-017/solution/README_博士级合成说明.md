# 多铁性材料磁电耦合畴结构演化模拟 —— 博士级合成说明

## 一、项目概述

本项目围绕**凝聚态物理：多铁性材料耦合机制**这一前沿科学领域，基于 15 个输入科研代码项目的核心算法，融合构建了一个面向博士级难度的前沿科学计算系统。

**科学问题**：以 BiFeO₃ 为原型，模拟多铁性材料中**铁电极化 (P)** 与**磁化 (M)** 的耦合时空演化，包括：
- Landau-Ginzburg-Devonshire (LGD) 自由能驱动的畴结构形成
- 时间依赖 Ginzburg-Landau (TDGL) 方程的数值求解
- 磁电耦合效应的定量分析与关联长度计算
- 热涨落对畴壁动力学的蒙特卡洛修正
- 有限元离散化与自适应时间步进

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 在合成项目中的角色 |
|--------|----------|-------------------|
| **1158_st_to_mm** | 稀疏矩阵 ST↔MM 格式转换、索引重基 | `sparse_matrix_utils.py`：COO/CSR 稀疏矩阵存储、Triad I/O、格式转换，为有限元刚度矩阵提供高效后端 |
| **754_mesh_display** | 网格节点/元素读取、索引基检测 | `multiferroic_mesh.py`：T6 二次元网格生成、边界标记、带宽计算 |
| **765_midpoint_adaptive** | 自适应隐式中点法、LTE 步长控制 | `adaptive_ode_integrator.py`：TDGL 方程的自适应时间步进，Newton-Raphson 求解非线性阶段方程 |
| **535_hilbert_curve** | Hilbert 空间填充曲线 d2xy/xy2d | `hilbert_space_filling.py`：网格节点 Hilbert 重排序，优化稀疏矩阵内存局部性 |
| **1364_tsp_descent** | TSP 下降法（转置/反转邻域搜索） | `domain_optimizer.py`：畴结构邻域搜索优化，反转/转置型扰动寻找能量更低构型 |
| **400_fem2d_bvp_linear** | 2D FEM 线性元、高斯积分、刚度组装 | `fem_assembler.py`：二次元刚度矩阵与质量矩阵组装、Dirichlet 边界处理 |
| **1266_toms178** | Hooke-Jeeves 直接搜索优化 | `domain_optimizer.py`：无导数多变量优化 Landau 自由能参数，best_nearby 子程序 |
| **1088_slap_io** | SLAP Triad 稀疏矩阵 I/O | `sparse_matrix_utils.py`：Triad 格式稀疏矩阵读写，头信息解析 |
| **1092_snakes_and_ladders** | 随机游走蒙特卡洛模拟 | `monte_carlo_sampler.py`：Metropolis 热涨落采样，随机游走型畴壁运动 |
| **934_pyramid_jaskowiec_rule** | 金字塔区域高精度对称求积 | `pyramid_quadrature.py`：Jaskowiec-Sukumar 规则，用于三维自由能体积分（厚度方向映射） |
| **449_full_deck_simulation** | 批量蒙特卡洛统计 | `monte_carlo_sampler.py`：批量 MC 统计（能量均值/方差、磁化率、比热） |
| **292_disk_distance** | 单位圆盘均匀采样、距离统计 | `monte_carlo_sampler.py`：圆盘型热扰动位移采样，关联长度标定 |
| **434_fisher_pde_ftcs** | Fisher-KPP 方程 FTCS 求解 | `reaction_diffusion_solver.py`：反应扩散方程求解器，应用于 P/M 的非线性时空演化 |
| **408_fem2d_poisson_rectangle** | T6 二次元泊松方程 FEM | `multiferroic_mesh.py` + `fem_assembler.py`：grid_t6 网格生成、qbf 基函数、bandwidth 计算 |
| **523_hermite_product_display** | Hermite 多项式递推与正交性 | `landau_free_energy.py`：概率/归一化 Hermite 多项式，热涨落自由能修正的谱展开 |

---

## 三、新增数学物理模型与核心公式

### 3.1 Landau-Ginzburg-Devonshire 自由能密度

多铁性材料的总自由能密度由铁电部分、磁性部分、磁电耦合与梯度能组成：

$$
f = f_P + f_M + f_c + f_{\text{grad}}
$$

**铁电 LGD 展开**：
$$
f_P = \alpha_1 (P_x^2 + P_y^2) + \alpha_{11} (P_x^2 + P_y^2)^2 + \alpha_{12} P_x^2 P_y^2
$$

其中 α₁ = α₀(T − T_c) 遵循居里-外斯定律，T_c = 1103 K 为铁电居里温度。

**磁性 Landau 展开**：
$$
f_M = \beta_1 (M_x^2 + M_y^2) + \beta_{11} (M_x^2 + M_y^2)^2 + \beta_{12} M_x^2 M_y^2
$$

其中 β₁ = β₀(T − T_N)，T_N = 643 K 为奈尔温度。

**磁电耦合能**（交换收缩型，反映 P 与 M 的正交耦合）：
$$
f_c = \gamma (P_x M_y - P_y M_x)^2
$$

**梯度能与高阶交叉项**：
$$
f_{\text{grad}} = \frac{g_{11}}{2}\left[\left(\frac{\partial P_x}{\partial x}\right)^2 + \left(\frac{\partial P_y}{\partial y}\right)^2\right] + \frac{g_{12}}{2}\left[\left(\frac{\partial P_x}{\partial y}\right)^2 + \left(\frac{\partial P_y}{\partial x}\right)^2\right] + g_{44} \frac{\partial P_x}{\partial y}\frac{\partial P_y}{\partial x} + \frac{A_{11}}{2}\left[\left(\frac{\partial M_x}{\partial x}\right)^2 + \left(\frac{\partial M_y}{\partial y}\right)^2\right] + \eta\left(\frac{\partial P_x}{\partial x}\frac{\partial M_y}{\partial y} - \frac{\partial P_y}{\partial y}\frac{\partial M_x}{\partial x}\right)
$$

### 3.2 时间依赖 Ginzburg-Landau (TDGL) 方程

序参量的非平衡演化由自由能变分驱动：

$$
\frac{\partial \mathbf{P}}{\partial t} = -\Gamma_P \frac{\delta F}{\delta \mathbf{P}} + \boldsymbol{\xi}_P
$$

$$
\frac{\partial \mathbf{M}}{\partial t} = -\Gamma_M \frac{\delta F}{\delta \mathbf{M}} + \boldsymbol{\xi}_M
$$

其中 Γ_P, Γ_M 为动力学系数，ξ 为热噪声项。

**变分导数**（以 P_x 为例）：

$$
\frac{\delta F}{\delta P_x} = \alpha_1 P_x + 4\alpha_{11} P_x (P_x^2+P_y^2) + 2\alpha_{12} P_x P_y^2 + 2\gamma (P_x M_y - P_y M_x) M_y - g_{11}\frac{\partial^2 P_x}{\partial x^2} - g_{12}\frac{\partial^2 P_x}{\partial y^2}
$$

### 3.3 Hermite 多项式热涨落修正

对序参量场 Q 的热涨落，投影到概率 Hermite 多项式基 He_n(Q/σ)，满足正交性：

$$
\int_{-\infty}^{+\infty} e^{-Q^2/(2\sigma^2)} \text{He}_m(Q/\sigma) \text{He}_n(Q/\sigma) \, dQ = \sqrt{2\pi}\,\sigma \, n! \, \delta_{mn}
$$

自由能 Hessian 矩阵 H 的特征值 λ_i 给出谐波近似下的热修正：

$$
\Delta f = \frac{k_B T}{2} \sum_i \ln\lambda_i
$$

### 3.4 磁电响应系数（涨落-耗散定理）

$$
\alpha_{\text{ME}} = \frac{1}{V} \int \mathbf{P} \cdot \mathbf{M} \, dV \, / \, (|\langle\mathbf{P}\rangle| \, |\langle\mathbf{M}\rangle|)
$$

### 3.5 有限元离散化

采用 T6（六节点二次三角形）有限元，基函数 N_i 在参考单元 (r,s) 上满足：

$$
N_1 = 2(1-r-s)(0.5-r-s), \quad N_2 = 2r(r-0.5), \quad N_3 = 2s(s-0.5)
$$

刚度矩阵组装公式：

$$
K_{ij} = \sum_e \sum_q w_q |J_e| D_e \left( \frac{\partial N_i}{\partial x}\frac{\partial N_j}{\partial x} + \frac{\partial N_i}{\partial y}\frac{\partial N_j}{\partial y} \right)
$$

### 3.6 FTCS 稳定性条件

二维扩散方程的显式 FTCS 格式稳定性要求：

$$
\Delta t \leq \frac{1}{2D}\left(\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}\right)^{-1} = \frac{\Delta x^2 \Delta y^2}{2D(\Delta x^2 + \Delta y^2)}
$$

### 3.7 自适应隐式中点法

阶段方程：

$$
\mathbf{Y} = \mathbf{y}_n + \frac{\tau}{2} f\left(t_n + \frac{\tau}{2}, \mathbf{Y}\right)
$$

Newton 迭代：

$$
\mathbf{Y}^{(k+1)} = \mathbf{Y}^{(k)} - \left(\mathbf{I} - \frac{\tau}{2} \mathbf{J}\right)^{-1} g(\mathbf{Y}^{(k)}), \quad g(\mathbf{Y}) = \mathbf{Y} - \mathbf{y}_n - \frac{\tau}{2} f(t_{\text{mid}}, \mathbf{Y})
$$

步长更新（基于嵌入法 LTE 估计）：

$$
\tau_{\text{new}} = \kappa \tau_n \left(\frac{1}{\|\text{LTE}\|}\right)^{1/3}, \quad \kappa = 0.85
$$

---

## 四、文件结构与修改说明

### 4.1 代码文件清单（12 个 .py 文件）

| 文件名 | 功能说明 | 融入的原项目 |
|--------|---------|-------------|
| `main.py` | 统一入口，零参数运行完整模拟流程 | 全部 |
| `landau_free_energy.py` | LGD 自由能模型、变分导数、Hermite 热修正 | 523_hermite_product_display |
| `multiferroic_mesh.py` | T6 二次元网格生成、节点坐标、元素连通 | 754_mesh_display, 408_fem2d_poisson_rectangle |
| `sparse_matrix_utils.py` | COO/CSR 稀疏矩阵、Triad I/O、格式转换 | 1158_st_to_mm, 1088_slap_io |
| `fem_assembler.py` | 刚度矩阵/质量矩阵组装、高斯积分、Dirichlet 边界 | 400_fem2d_bvp_linear, 408_fem2d_poisson_rectangle |
| `reaction_diffusion_solver.py` | FTCS 反应扩散求解器、Fisher-KPP/Allen-Cahn | 434_fisher_pde_ftcs |
| `adaptive_ode_integrator.py` | 自适应隐式中点法、Newton-Raphson、LTE 步长控制 | 765_midpoint_adaptive |
| `domain_optimizer.py` | Hooke-Jeeves 直接搜索、TSP-Descent 邻域搜索 | 1266_toms178, 1364_tsp_descent |
| `monte_carlo_sampler.py` | Metropolis 采样、圆盘扰动、批量统计、关联函数 | 1092_snakes_and_ladders, 449_full_deck_simulation, 292_disk_distance |
| `hilbert_space_filling.py` | Hilbert 曲线排序、节点重排序优化局部性 | 535_hilbert_curve |
| `pyramid_quadrature.py` | Jaskowiec-Sukumar 金字塔求积规则 | 934_pyramid_jaskowiec_rule |
| `coupling_dynamics.py` | 多铁性耦合动力学核心：TDGL 演化、磁电系数计算 | 整合全部 |

### 4.2 关键修改与增强

1. **删除所有可视化代码**：原项目中的 `plot`、`surface`、`print('-dpng')` 等全部移除，仅保留数值结果输出。
2. **MATLAB → Python 迁移**：所有算法从 MATLAB/Octave 翻译为 Python/NumPy，保留核心数值逻辑。
3. **边界处理与数值鲁棒性**：
   - 稀疏矩阵索引越界检查
   - FTCS 稳定性条件自动判定
   - Newton 迭代奇异正则化
   - NaN/Inf 检测与回退机制
   - 物理约束截断（P ∈ [-1,1], M ∈ [-5e5, 5e5]）
4. **科学公式注入**：代码中内嵌了 LGD 自由能、TDGL 方程、变分导数、Hermite 正交性、涨落-耗散定理、有限元弱形式等大量公式。

---

## 五、合成后的项目能解决什么科学问题

1. **多铁性畴结构演化模拟**：给定材料参数和温度，模拟铁电畴与磁畴的耦合时空演化。
2. **磁电耦合系数定量计算**：通过极化-磁化关联函数计算磁电响应系数 α_ME 的温度依赖。
3. **畴壁动力学分析**：追踪畴壁位置、计算关联长度，分析热涨落对畴壁钉扎/退钉扎的影响。
4. **稳态畴结构预测**：结合 Hooke-Jeeves 与 TSP-Descent 优化算法，寻找 Landau 自由能的局域极小值，预测稳态畴构型。
5. **热涨落效应评估**：通过 Metropolis 蒙特卡洛采样，评估有限温度下序参量的热噪声修正。

---

## 六、如何运行

```bash
cd 017_synth_project
python main.py
```

程序零参数运行，自动执行：
1. Landau 自由能模型验证
2. 有限元网格生成与 Hilbert 重排序
3. 刚度矩阵组装与线性系统求解验证
4. 金字塔数值积分精度验证
5. Fisher-KPP / Allen-Cahn 反应扩散验证
6. 自适应隐式中点法 ODE 积分验证
7. 蒙特卡洛圆盘距离统计验证
8. Hooke-Jeeves + TSP-Descent 优化验证
9. 多铁性磁电耦合主模拟（32×32 网格，100 步 FTCS 演化）
10. 蒙特卡洛热涨落采样
11. 稀疏矩阵 I/O 验证

---

## 七、质量检查清单

- [x] 原目录未被修改
- [x] 合成后的项目为 Python 语言
- [x] 新目录完整包含合成后的项目（12 个 .py + 1 个 README）
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] **15 个输入项目均已真实融入**，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 无可视化代码残留
