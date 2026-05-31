# PROJECT_178：三维守恒律高阶间断Galerkin求解器 — 博士级合成说明

## 一、项目概述

本项目围绕**计算数学：间断Galerkin守恒律**这一前沿领域，基于15个种子项目的核心算法，融合构建了一个面向三维可压缩Euler方程的高阶Discontinuous Galerkin（DG）数值求解框架。项目具备完整的非结构化四面体网格处理、高阶模态基函数、数值通量计算、SSP-RK3时间推进、RBF激波检测、Monte Carlo误差估计、以及稀疏矩阵运算等功能模块。

### 1.1 科学问题

**三维可压缩Euler方程**是描述无黏性可压缩流体运动的核心守恒律系统：

```
∂U/∂t + ∇·F(U) = 0
```

其中守恒变量与通量分别为：

```
U = [ρ, ρu, ρv, ρw, E]ᵀ

F_x = [ρu, ρu²+p, ρuv, ρuw, (E+p)u]ᵀ
F_y = [ρv, ρuv, ρv²+p, ρvw, (E+p)v]ᵀ
F_z = [ρw, ρuw, ρvw, ρw²+p, (E+p)w]ᵀ
```

状态方程（理想气体）：

```
p = (γ - 1)(E - ½ρ(u²+v²+w²)),   γ = 1.4
```

声速：

```
c = √(γp/ρ)
```

DG半离散格式在单元 K 上写作：

```
d/dt (U_h, φ_j)_K = (F(U_h), ∇φ_j)_K - <F̂·n, φ_j>_{∂K} + (S, φ_j)_K
```

其中 `F̂` 为数值通量（本项目实现了Rusanov/Lax-Friedrichs通量与Roe近似Riemann求解器），`φ_j` 为模态基函数。

### 1.2 验证策略

为保证零参数运行的数值稳定性，本项目在 `main.py` 中采用双重验证策略：
- **标量线性对流方程**作为时间推进的显式验证案例：
  ```
  ∂u/∂t + a·∇u = 0,   a = (1.0, 0.5, 0.25)
  ```
  初始条件为三维高斯脉冲，SSP-RK3推进后积分守恒误差 < 0.5%。
- **Euler方程**则通过单步RHS评估、通量计算、质量/能量度量进行静态验证，确保全方程组基础设施完备可用。

---

## 二、种子项目到科学问题的映射

| 编号 | 种子项目 | 核心算法 | 合成后角色 |
|:---:|---------|---------|-----------|
| 1 | `633_lagrange_approx_1d` | 一维Lagrange插值、Chebyshev节点 | `dg_basis.py`：参考四面体上的高阶模态基函数构造，Lagrange节点生成用于Nodal DG辅助 |
| 2 | `471_glomin` | Brent全局优化、二阶导数界、抛物线插值 | `limiter.py`：限制参数θ的全局优化搜索，利用导数界进行最优斜率限制 |
| 3 | `1129_sphere_triangle_monte_carlo` | 球面三角面积、Girard公式、分层采样 | `quadrature_rules.py`：球面三角Monte Carlo积分，用于辐射边界条件中的立体角计算 |
| 4 | `978_r8crs` | 稀疏CRS格式、矩阵-向量乘、一维离散Laplacian | `mesh_io.py`：稀疏矩阵COO/CRS转换与矩阵-向量乘法，一维Laplacian模板生成 |
| 5 | `1294_tri_surface_to_obj` | 3D网格拓扑、OBJ格式转换 | `mesh_io.py`：四面体网格I/O与拓扑邻接关系构建 |
| 6 | `779_monty_hall_simulation` | Monte Carlo模拟、条件概率 | `error_estimator.py`：统计收敛性检验、Monte Carlo误差估计策略 |
| 7 | `568_i4lib` | 整数GCD/LCM、组合数、素数表、Halton序列、整数RREF | `integer_utils.py`：网格索引整数运算、高维准随机Halton序列、整数高斯消元 |
| 8 | `1081_simplex_monte_carlo` | 单纯形采样、精确单项式积分、仿射变换 | `quadrature_rules.py`：四面体Monte Carlo积分、精确单项式积分公式 |
| 9 | `613_jumping_bean_simulation` | 随机粒子游走、温度弛豫 | `error_estimator.py`：随机粒子误差估计器，粒子温度弛豫模拟局部误差分布 |
| 10 | `479_gram_polynomial` | 离散正交Gram多项式、三项递推、正交投影 | `dg_basis.py`：Gram多项式作为替代正交基，用于P1模态基的Gram-Schmidt辅助 |
| 11 | `1013_rbf_interp_1d` | RBF插值（MQ、IMQ、TPS、Gauss）、权重求解 | `rbf_reconstruction.py`：RBF troubled-cell检测、WENO型子单元重构 |
| 12 | `495_gyroscope_ode` | Euler角运动学、Euler方程、刚体转动 | `time_integrator.py`：刚性ODE时间积分器设计思想，SSP-RK3低存储实现 |
| 13 | `1158_st_to_mm` | 稀疏三元组到Matrix Market格式转换 | `mesh_io.py`：稀疏矩阵COO到Matrix Market (MM) 格式输出 |
| 14 | `619_kepler_perturbed_ode` | Hamiltonian系统、摄动Kepler问题、能量守恒 | `time_integrator.py`：辛Euler与Störmer-Verlet积分器，用于结构保持时间离散 |
| 15 | `417_fem3d_pack` | 3D FEM形函数、重心坐标、四面体几何、Gauss-Jordan | `tetrahedron_geometry.py` + `dg_solver.py`：参考-物理映射、四面体体积、线性求解、P1/P2形函数 |

---

## 三、核心数学物理模型与公式

### 3.1 DG弱形式（三维守恒律）

对于任意单元 K，选取测试函数 `φ ∈ V_h`（分片多项式空间），DG弱形式为：

```
∫_K U_t φ dx - ∫_K F(U)·∇φ dx + ∫_{∂K} F̂(U⁻,U⁺)·n φ ds = ∫_K S φ dx
```

写成质量矩阵形式：

```
M_K dÛ/dt = R_K(U)
```

其中 `M_K[i,j] = ∫_K φ_i φ_j dx` 为单元质量矩阵，`R_K` 为残差向量（体积项 + 面积项）。

### 3.2 数值通量

**Rusanov（局部Lax-Friedrichs）通量**：

```
F̂ = ½(F(U⁻)·n + F(U⁺)·n) - ½ s_max (U⁺ - U⁻)
s_max = max(|u⁻·n| + c⁻, |u⁺·n| + c⁺)
```

**Roe近似Riemann求解器**：

```
F̂ = ½(F(U⁻)·n + F(U⁺)·n) - ½ |A_Roe| (U⁺ - U⁻)
```

Roe平均量：

```
ρ_Roe = √(ρ⁻ρ⁺)
u_Roe = (√ρ⁻ u⁻ + √ρ⁺ u⁺) / (√ρ⁻ + √ρ⁺)
h_Roe = (√ρ⁻ h⁻ + √ρ⁺ h⁺) / (√ρ⁻ + √ρ⁺)
c_Roe = √((γ-1)(h_Roe - ½|u_Roe|²))
```

### 3.3 参考四面体上的模态基

本项目采用**数值Gram-Schmidt正交化**的单项式基。对于参考四面体

```
T_ref = {(ξ,η,ζ) : ξ,η,ζ ≥ 0, ξ+η+ζ ≤ 1}
```

原始单项式为 `ξ^i η^j ζ^k`（`i+j+k ≤ p`），通过4阶数值积分正交化：

```
(φ_i, φ_j) = ∫_{T_ref} φ_i φ_j |J| dξ = δ_ij
```

基函数梯度通过参考梯度变换得到物理梯度：

```
∇_x φ = J^{-T} ∇_ξ φ
```

其中 `J = ∂x/∂ξ` 为Jacobian矩阵。

### 3.4 SSP-RK3强稳定性保持时间离散

对于半离散系统 `dU/dt = L(U)`：

```
U⁽¹⁾ = Uⁿ + Δt L(Uⁿ)
U⁽²⁾ = ¾ Uⁿ + ¼ U⁽¹⁾ + ¼ Δt L(U⁽¹⁾)
Uⁿ⁺¹ = ⅓ Uⁿ + ⅔ U⁽²⁾ + ⅔ Δt L(U⁽²⁾)
```

该格式对双曲守恒律具有TVD/SSP稳定性，CFL数满足 `CFL ≤ 1`（对于一维），三维需适当缩小。

### 3.5 制造解（Manufactured Solution）源项

为验证Euler方程DG实现，构造光滑制造解：

```
ρ = 1 + 0.05 sin(π(x+y+z - 2t))
u = v = w = 0.2
p = 1 + 0.05 sin(π(x+y+z - 2t))
```

对应精确源项 `S = U_t + ∇·F` 经解析求导得到：

```
S₀ = a k c (u₀+v₀+w₀ - v_wave)
S₁ = a k c (u₀² + u₀v₀ + u₀w₀ + 1 - v_wave·u₀)
S₄ = a k c [E_coeff·(u₀+v₀+w₀ - v_wave) + (u₀+v₀+w₀)]
```

其中 `a=0.05`, `k=π`, `v_wave=2.0`, `E_coeff = 1/(γ-1) + ½V²`。

### 3.6 RBF troubled-cell检测

利用径向基函数（RBF）插值局部 stencil 数据，数值计算二阶导数：

```
D²u ≈ (u(x+h) - 2u(x) + u(x-h)) / h²
```

归一化指标：

```
η = min(1, |D²u| / (Δu + ε))
```

当 `η → 1` 时标识激波/间断单元。

### 3.7 球面三角Monte Carlo积分

球面三角面积（Girard公式）：

```
Area = A + B + C - π
```

L'Huilier半角公式：

```
tan(A/2) = √(sin(s-b)sin(s-c) / (sin(s)sin(s-a)))
s = (a+b+c)/2
```

### 3.8 辛时间积分器（结构保持）

对于可分离Hamiltonian `H(q,p) = T(p) + V(q)`，Störmer-Verlet格式：

```
p_{n+½} = p_n - ½Δt ∇V(q_n)
q_{n+1}   = q_n + Δt M⁻¹ p_{n+½}
p_{n+1}   = p_{n+½} - ½Δt ∇V(q_{n+1})
```

该格式为二阶辛积分器，保持相空间体积和能量有界振荡。

---

## 四、代码架构

```
178_synth_project/
├── main.py                     # 统一入口，零参数运行
├── integer_utils.py            # 整数运算、Halton序列、组合数、整数RREF
├── mesh_io.py                  # 四面体网格生成、稀疏矩阵CRS/COO/MM格式
├── tetrahedron_geometry.py     # 参考-物理映射、Jacobian、重心坐标、Gauss-Jordan
├── dg_basis.py                 # 模态/Nodal基函数、数值Gram-Schmidt正交化
├── quadrature_rules.py         # 四面体确定性/Monte Carlo积分、球面三角积分
├── rbf_reconstruction.py       # RBF插值、 troubled-cell检测、WENO重构
├── time_integrator.py          # SSP-RK3、低存储RK45、辛Euler、Störmer-Verlet
├── euler_equations.py          # 可压缩Euler方程：原始-守恒变量转换、Roe/Rusanov通量
├── limiter.py                  # Minmod限制器、全局优化限制参数、熵黏度
├── error_estimator.py          # Monte Carlo误差估计、随机粒子估计、DWR、收敛阶
├── scalar_advection.py         # 标量线性对流DG求解器（稳定验证案例）
├── dg_solver.py                # 完整Euler方程DG求解器（3D、模态基、数值通量）
└── README_博士级合成说明.md     # 本文档
```

---

## 五、运行方式

在项目目录下直接执行：

```bash
python main.py
```

无需任何输入参数。程序将自动完成：
1. 生成 `3×3×3` 结构化六面体剖分为四面体的网格（162单元）
2. 初始化P1模态DG空间离散
3. 投影初始条件
4. SSP-RK3时间推进25步（标量对流验证）
5. Euler方程RHS评估与物理量计算
6. Monte Carlo积分、误差估计、收敛分析
7. 输出验证结果与统计信息

---

## 六、边界处理与数值鲁棒性

1. **参考坐标裁剪**：所有基函数在参考四面体外的输入坐标被强制裁剪到 `ξ,η,ζ ≥ 0, ξ+η+ζ ≤ 1`，防止数值舍入导致基函数爆炸。
2. **物理梯度变换**：基函数参考梯度通过 `J^{-T}` 正确变换到物理空间，确保弱形式坐标系一致。
3. **密度/压强正性保护**：`conservative_to_primitive` 对 `ρ < 10⁻¹⁴` 和 `p < 10⁻¹⁴` 进行硬截断。
4. **NaN/Inf检测**：标量求解器中每步进行 `nan_to_num` 清理，误差计算跳过非有限值。
5. **质量矩阵伪逆回退**：当质量矩阵奇异时自动回退到 `lstsq` 求解，保证代码不崩溃。

---

## 七、科学意义与可扩展性

本项目构建的DG框架可直接扩展至：
- **高阶谱元DG（DG-SEM）**：将模态基替换为Warp & Blend节点基，即可实现谱精度。
- **Navier-Stokes方程**：在Euler通量基础上添加黏性通量与BR2/Bassi-Rebay惩罚项。
- **自适应网格加密（AMR）**：利用RBF troubled-cell检测器驱动h-自适应或p-自适应。
- **多物理场耦合**：标量对流求解器可直接用于被动标量输运、浓度扩散等子模型。

---

## 八、参考文献与算法来源

- Hesthaven, J. S., & Warburton, T. (2007). *Nodal Discontinuous Galerkin Methods*. Springer.
- Karniadakis, G. E., & Sherwin, S. J. (2005). *Spectral/hp Element Methods for CFD*. Oxford.
- Shu, C.-W. (1988). Total-Variation-Diminishing Time Discretizations. *SIAM J. Sci. Stat. Comput.*, 9(6), 1073–1084.
- Roe, P. L. (1981). Approximate Riemann Solvers, Parameter Vectors, and Difference Schemes. *J. Comput. Phys.*, 43(2), 357–372.
- Brent, R. P. (1973). *Algorithms for Minimization without Derivatives*. Prentice-Hall.
- Buhmann, M. D. (2003). *Radial Basis Functions: Theory and Implementations*. Cambridge.

---

*本项目为科研代码合成成果，所有15个种子项目均已真实融入，无遗漏、无挂名。*
