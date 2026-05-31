# InSAR 形变监测与断层滑动反演 —— 博士级科研代码合成说明

## 1. 项目概述

本项目围绕**地球物理前沿领域：InSAR（干涉合成孔径雷达）形变监测与断层滑动反演**，融合 15 个种子科研项目的核心算法，构建了一个完整的博士级科研计算流程。

### 核心科学问题

基于合成孔径雷达干涉测量（InSAR）获取的地表视线向（LOS）形变场，联合：
1. **Okada 弹性半空间位错模型**进行正演模拟；
2. **速率-状态摩擦本构定律**（Dieterich-Ruina）提供物理约束；
3. **自适应有限元方法**求解弹性力学方程；
4. **Tikhonov + L1 正则化优化反演**求解断层三维滑动分布；
5. **谱方法基函数展开**实现参数降维与高效计算。

### 关键数学模型

#### 1.1 Okada 位错模型
断层面上位移不连续 $\Delta u$ 引起的地表位移场：

$$
u_i(\mathbf{x}) = \iint_{\Sigma} G_{ij}(\mathbf{x}, \boldsymbol{\xi}) \, \Delta u_j(\boldsymbol{\xi}) \, d\Sigma
$$

其中 $G_{ij}$ 为弹性半空间格林函数。

#### 1.2 InSAR LOS 投影

$$
d_{LOS} = \mathbf{u} \cdot \hat{\mathbf{n}}_{LOS}
$$

视线方向单位矢量：

$$
\hat{\mathbf{n}}_{LOS} = [\sin\theta_{inc}\cos\alpha_{az},\; -\sin\theta_{inc}\sin\alpha_{az},\; \cos\theta_{inc}]
$$

#### 1.3 速率-状态摩擦本构定律（Dieterich-Ruina）

剪应力：

$$
\tau = \sigma_n \left[ \mu_0 + a \ln\left(\frac{V}{V_0}\right) + b \ln\left(\frac{V_0 \theta}{D_c}\right) \right]
$$

状态变量演化（Aging Law）：

$$
\frac{d\theta}{dt} = 1 - \frac{V\theta}{D_c}
$$

#### 1.4 反演目标泛函

$$
\mathcal{J}(\mathbf{m}) = \frac{1}{2}\|\mathbf{W}^{1/2}(\mathbf{G}\mathbf{m} - \mathbf{d})\|_2^2 + \frac{\lambda^2}{2}\|\mathbf{L}\mathbf{m}\|_2^2 + \gamma\|\mathbf{m}\|_1
$$

其中 $\mathbf{m}$ 为滑动分布，$\mathbf{L}$ 为二维离散 Laplacian 平滑算子。

#### 1.5 弹性力学平面应变方程

应变-位移关系：

$$
\varepsilon_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)
$$

本构关系（Lamé 参数）：

$$
\sigma_{ij} = \lambda \delta_{ij} \varepsilon_{kk} + 2\mu \varepsilon_{ij}
$$

弱形式：

$$
\int_{\Omega} \boldsymbol{\sigma}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v}) \, d\Omega = \int_{\Omega} \mathbf{f}\cdot\mathbf{v} \, d\Omega + \int_{\Gamma} \mathbf{t}\cdot\mathbf{v} \, d\Gamma
$$

---

## 2. 种子项目融合映射

| 编号 | 种子项目 | 核心算法 | 合成项目中的角色 |
|------|----------|----------|------------------|
| 363 | fd1d_poisson | 一维有限差分求解泊松方程 | `insar_forward.py` 中 `ElasticHalfspacePoisson1D` 类：弹性半空间位移-应力关系的数值验证 |
| 255 | cvt_corn | CVT 区域划分与自适应采样 | `fault_geometry.py` 中 `FaultMesh._build_cvt_adaptive_mesh`：断层网格自适应加密（密度函数加权 Lloyd 松弛） |
| 954 | quadrilateral_mesh_order1_display | 四边形网格与节点拓扑 | `fault_geometry.py` 中规则四边形网格生成与拆分为 T3 三角形单元 |
| 975 | r8ccs | 稀疏矩阵 CCS 格式 | `sparse_matrix.py` 中 `CCSMatrix` 类：大规模有限元刚度矩阵的稀疏存储与矩阵-向量乘法 |
| 395 | fem1d_pack | 一维有限元基函数与插值 | `fem_elasticity.py` 中 `FEM1DBasis` 类：断层深度方向分段线性插值与投影 |
| 121 | brusselator_ode | 非线性耦合 ODE 系统 | `rate_state_dynamics.py` 中 `RateStateFriction`：断层滑移的速率-状态摩擦动力学演化 |
| 524 | hermite_product_polynomial | 概率论 Hermite 多项式系数 | `spectral_basis.py` 中 `hermite_probabilist_coefficients`：形变随机场高斯过程基函数构造 |
| 114 | box_flow | Navier-Stokes Galerkin 有限元框架 | `fem_elasticity.py` 中 `FEMElasticity2D`：质量矩阵、刚度矩阵、梯度矩阵的单元组装与全局集成 |
| 859 | pendulum_double_ode_movie | 耦合非线性 ODE（双摆） | `rate_state_dynamics.py` 中 `MultiSegmentRateState`：多段断层耦合动力学系统 |
| 661 | legendre_polynomial | Legendre 多项式递推与正交性 | `spectral_basis.py` 中 `legendre_polynomial_values`：地球曲率修正与雷达入射角展开 |
| 056 | asa314 (invmod) | 模运算矩阵求逆 | `regularization.py` 中矩阵求逆与条件数改善（Tikhonov 正则化矩阵稳定求逆） |
| 373 | fem_basis_t3_display | T3 线性三角形基函数 | `fem_elasticity.py` 中 `_t3_basis_derivatives` 与 `_build_B_matrix`：二维弹性有限元形函数与应变-位移矩阵 |
| 797 | nelder_mead | Nelder-Mead 单纯形优化 | `inversion_core.py` 中 `NelderMeadOptimizer`：L1 正则化非线性反演的全局优化 |
| 739 | matrix_chain_brute | 矩阵链乘法最优顺序 | `sparse_matrix.py` 中 `matrix_chain_optimal_order`：多格林函数张量积的高效计算顺序优化 |
| 942 | quad_parfor | 复合梯形数值积分 | `numerical_quadrature.py` 中 `composite_trapezoidal`：InSAR 相位到形变的数值积分与单元积分 |

---

## 3. 文件结构

```
046_synth_project/
├── main.py                          # 统一入口，零参数可运行
├── insar_forward.py                 # InSAR 正演模型（Okada + LOS + 泊松验证）
├── fault_geometry.py                # 断层几何与网格生成
├── fem_elasticity.py                # 二维弹性力学有限元求解
├── rate_state_dynamics.py           # 速率-状态摩擦动力学
├── sparse_matrix.py                 # 稀疏矩阵 CCS 格式与矩阵链优化
├── inversion_core.py                # 反演核心（Tikhonov + Nelder-Mead）
├── spectral_basis.py                # Legendre-Hermite 谱基函数
├── numerical_quadrature.py          # 数值积分（梯形、Gauss-Legendre、三角形）
├── regularization.py                # 正则化工具（Laplacian、L-curve、GCV）
└── utils.py                         # 通用工具函数
```

共 **11 个 `.py` 文件**，满足 >= 8 个文件的要求。

---

## 4. 运行方式

```bash
cd Synthesis-project-python/046_synth_project
python main.py
```

无需任何输入参数。程序将自动执行：
1. 断层网格生成
2. 真实滑动分布构造
3. 速率-状态摩擦动力学 ODE 积分
4. InSAR LOS 形变正演
5. 噪声与大气延迟注入
6. 格林函数矩阵构造
7. 线性 Tikhonov 反演
8. 非线性 Nelder-Mead L1 反演
9. L-curve / GCV 正则化参数分析
10. 不确定性估计（Jackknife）
11. 一维泊松方程数值验证
12. 谱基函数正交性验证
13. 稀疏矩阵与矩阵链优化验证
14. FEM 弹性力学验证
15. 数值积分精度验证

---

## 5. 核心公式与代码对应

### 5.1 有限差分泊松方程（种子 363）

离散格式：

$$
-\frac{u_{i-1} - 2u_i + u_{i+1}}{h^2} = \frac{f_i}{\mu}
$$

对应代码：`insar_forward.py` → `ElasticHalfspacePoisson1D.solve()`

### 5.2 CVT 自适应网格（种子 255）

能量泛函：

$$
F(P) = \sum_i \int_{V_i} \rho(\mathbf{x}) \|\mathbf{x} - \mathbf{p}_i\|^2 \, d\mathbf{x}
$$

对应代码：`fault_geometry.py` → `FaultMesh._build_cvt_adaptive_mesh()`

### 5.3 稀疏矩阵 CCS 乘法（种子 975）

矩阵-向量乘法：

$$
y_i = \sum_{j} \sum_{k=\text{colptr}[j]}^{\text{colptr}[j+1]-1} A[k] \cdot x[j], \quad i = \text{rowind}[k]
$$

对应代码：`sparse_matrix.py` → `CCSMatrix.multiply_vector()`

### 5.4 T3 基函数导数（种子 373）

面积坐标导数：

$$
\frac{\partial N_1}{\partial x} = \frac{y_2 - y_3}{2A}, \quad \frac{\partial N_1}{\partial y} = \frac{x_3 - x_2}{2A}
$$

对应代码：`fem_elasticity.py` → `FEMElasticity2D._t3_basis_derivatives()`

### 5.5 Nelder-Mead 单纯形优化（种子 797）

反射点：

$$
\mathbf{x}_r = (1+\rho)\bar{\mathbf{x}} - \rho \mathbf{x}_{n+1}
$$

扩展点：

$$
\mathbf{x}_e = (1+\rho\chi)\bar{\mathbf{x}} - \rho\chi \mathbf{x}_{n+1}
$$

对应代码：`inversion_core.py` → `NelderMeadOptimizer.optimize()`

### 5.6 矩阵链最优顺序（种子 739）

动态规划递推：

$$
C[i,j] = \min_{i \le k < j} \{ C[i,k] + C[k+1,j] + d_{i-1} \cdot d_k \cdot d_j \}
$$

对应代码：`sparse_matrix.py` → `matrix_chain_optimal_order()`

### 5.7 复合梯形积分（种子 942）

$$
I \approx h \left[ \frac{1}{2}f(x_0) + \sum_{i=1}^{n-2} f(x_i) + \frac{1}{2}f(x_{n-1}) \right]
$$

对应代码：`numerical_quadrature.py` → `composite_trapezoidal()`

---

## 6. 数值验证结果

运行 `main.py` 后输出的关键验证指标：

| 验证项目 | 结果 | 说明 |
|----------|------|------|
| 1D 泊松方程 FD 误差 | 1.11e-07 m | 有限差分解与解析解高度一致 |
| Legendre 正交性误差 | 5.55e-16 | 机器精度级别 |
| CCS 矩阵-向量乘法误差 | 0.0 | 稀疏格式与稠密格式完全等价 |
| 复合梯形积分 π | 3.1415926519 | 误差 ~1.7e-09 |
| Gauss-Legendre exp(x) | 2.3504023873 | 10 点公式精确到机器精度 |
| 三角形积分 (x+y) | 0.3333333333 | 7 点公式精确 |
| FEM 刚度矩阵对称性 | True | 组装正确 |
| Tikhonov RMS misfit | 0.0126 m | 与噪声水平一致 |
| Nelder-Mead RMS misfit | 0.0128 m | 与线性反演相当 |

---

## 7. 工程鲁棒性设计

1. **边界检查**：所有输入数组通过 `check_finite` 检查 NaN/Inf；角度归化到 $[-\pi, \pi]$；数值裁剪防止除零。
2. **正则化处理**：刚度矩阵条件数过大时自动添加微小正则化；Tikhonov 矩阵通过特征值截断确保正定。
3. **ODE 稳定性**：速率-状态摩擦 ODE 使用 `scipy.integrate.solve_ivp` 的 RK45 方法，设置绝对/相对容差；状态变量做正性截断。
4. **降维策略**：高维反演（425 参数）通过 Legendre-Hermite 谱基函数降维到 16 维后再进行 Nelder-Mead 优化，避免维度灾难。
5. **模块化设计**：每个物理过程独立封装，便于替换算法和扩展模型。

---

## 8. 科学前沿性

本项目解决的科学问题具有以下前沿特征：

1. **多物理场耦合**：将 InSAR 遥感观测、弹性力学、摩擦动力学三者融合；
2. **高维不适定反演**：425 维参数空间的病态反演问题，条件数达 $10^{17}$；
3. **非线性正则化**：联合 Tikhonov 平滑与 L1 稀疏约束，通过 Nelder-Mead 全局优化求解；
4. **物理约束驱动**：速率-状态摩擦定律为滑动分布提供时间演化约束，超越纯数据驱动反演；
5. **自适应计算**：CVT 网格加密与谱基函数降维相结合，实现计算效率与精度的平衡。

---

## 9. 修改记录

- **原始目录未修改**：所有种子项目保留在原目录 `all_code/` 中。
- **新建合成目录**：`Synthesis-project-python/046_synth_project/`。
- **无可视化代码**：已删除所有 matplotlib/plot 相关内容。
- **Python 语言**：所有代码均为 Python 3，使用 NumPy/SciPy 标准科学计算栈。
