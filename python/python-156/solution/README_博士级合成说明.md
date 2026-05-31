# README_博士级合成说明.md

## 湍流燃烧火焰面模型数值模拟系统

**科学领域**: 燃烧科学 — 湍流燃烧火焰面模型 (Turbulent Combustion Flamelet Model)

**项目编号**: PROJECT_156

**合成日期**: 2026-05-04

---

## 一、项目概述

本项目将 15 个原始科研代码项目的核心算法融合重构，面向**湍流非预混燃烧的稳态层流火焰面模型（Steady Laminar Flamelet Model, SLFM）**这一前沿博士级科学计算问题，构建了一套完整的 Python 数值模拟系统。

火焰面模型由 Peters (1986) 提出，是当今湍流燃烧大涡模拟（LES）和雷诺平均（RANS）中描述有限速率化学效应的主流方法。其核心思想是：在湍流非预混燃烧中，化学反应发生在极薄的层流火焰面内，而湍流的作用是将这些火焰面拉伸和折叠。通过引入**混合分数** $Z$ 作为独立变量，将多维时空问题降维到一维混合分数空间中求解。

---

## 二、核心物理模型与数学公式

### 2.1 稳态火焰面方程

在混合分数空间 $Z \in [0, 1]$ 中，温度 $T(Z)$ 与组分质量分数 $Y_k(Z)$ 满足：

$$\rho(Z) \frac{\chi(Z)}{2} \frac{d^2 T}{dZ^2} + \dot{\omega}_T(T, Y_k) = 0 \quad \cdots (1)$$

$$\rho(Z) \frac{\chi(Z)}{2} \frac{d^2 Y_k}{dZ^2} + \dot{\omega}_k(T, Y_k) = 0 \quad \cdots (2)$$

其中：
- $\rho(Z)$ 为混合气体密度（理想气体状态方程）
- $\chi(Z) = 2D|\nabla Z|^2$ 为**标量耗散率**（scalar dissipation rate）
- $\dot{\omega}_T, \dot{\omega}_k$ 为温度与组分的化学反应源项

### 2.2 标量耗散率分布

采用 Peters (1984) 提出的反误差函数形式：

$$\chi(Z) = \chi_{st} \frac{\exp\{-2[\text{erf}^{-1}(2Z-1)]^2\}}{\exp\{-2[\text{erf}^{-1}(2Z_{st}-1)]^2\}} \quad \cdots (3)$$

其中 $\chi_{st}$ 为化学计量点标量耗散率，$Z_{st}$ 为化学计量混合分数。

### 2.3 理想气体状态方程

$$p = \frac{\rho R_u T}{W(Z)} \quad \cdots (4)$$

混合气体平均分子量：

$$W(Z) = \frac{1}{\frac{Z}{W_F} + \frac{1-Z}{W_O}} \quad \cdots (5)$$

### 2.4 Arrhenius 一步总包反应速率

$$\dot{\omega}_F = A \rho^2 Y_F Y_O \exp\left(-\frac{E_a}{R_u T}\right) \quad \cdots (6)$$

### 2.5 点火与熄火极限（Liñán 理论）

通过渐近分析导出临界 Damköhler 数 $Da_{cr}$ 满足的特征多项式：

$$P(Da) = Da^d + c_{d-1}Da^{d-1} + \cdots + c_1 Da + c_0 = 0 \quad \cdots (7)$$

Zel'dovich 数：

$$Ze = \frac{\beta(1-\sigma)}{2}, \quad \beta = \frac{E_a}{R_u T_{st}}, \quad \sigma = \frac{T_{ox}}{T_{ad}} \quad \cdots (8)$$

### 2.6 Darrieus-Landau 不稳定性增长率

$$\sigma_{DL} = S_L k \left[ \frac{\rho_u}{\rho_u+\rho_b}\sqrt{\frac{\rho_u+\rho_b}{\rho_b} + \frac{(\rho_u-\rho_b)^2}{\rho_b^2}} - \frac{\rho_u}{\rho_b} \right] \quad \cdots (9)$$

### 2.7 Markstein 曲率修正

局部法向火焰速度：

$$S_n = S_L (1 - L_M \kappa) \quad \cdots (10)$$

Markstein 长度（Clavin & Williams, 1982）：

$$L_M = \delta_L \frac{Le \ln(1/Le)}{Le - 1} \quad (Le \neq 1) \quad \cdots (11)$$

### 2.8 火焰前锋皱褶因子

湍流火焰速度与层流火焰速度之比：

$$\frac{S_T}{S_L} = \frac{A_T}{A_L} = \text{area ratio} \quad \cdots (12)$$

---

## 三、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 合成后模块 | 科学角色 |
|:---:|:---|:---|:---|:---|
| 1 | `1404_wdk` | Weierstrass-Durand-Kerner 多项式求根 | `ignition_polynomial.py` | 求解点火/熄火极限的特征多项式根，确定临界 Damköhler 数 |
| 2 | `316_doughnut_exact` | Doughnut ODE 精确解 | `exact_benchmark.py` | 提供 Manufactured Solution 和解析基准解，用于验证 FEM/FDM 求解器精度 |
| 3 | `392_fem1d_heat_steady` | 1D 稳态热方程 FEM (线性基函数) | `fem_thermal_solver.py` | 求解火焰面能量方程的稳态温度分布 |
| 4 | `093_bird_egg` | 通用蛋形曲线 | `flame_front_shape.py` | 参数化火焰前锋皱褶几何形状，计算皱褶表面积 |
| 5 | `387_fem1d_bvp_quadratic` | 1D BVP 二次 FEM | `fem_quadratic_solver.py` | 高阶有限元求解组分质量分数输运方程 |
| 6 | `1005_randlc` | LCG 伪随机数生成 | `turbulent_random_field.py` | 生成符合 Kolmogorov 谱的湍流速度脉动场 |
| 7 | `752_mesh_bandwidth` | 网格带宽分析 | `matrix_bandwidth.py` | 分析火焰面方程离散后刚度矩阵的带宽与稀疏存储需求 |
| 8 | `725_matlab_map` | Voronoi 图生成 | `spatial_partition.py` | 空间 Voronoi 分区用于并行计算域分解 |
| 9 | `1274_toms577` | Carlson 不完全椭圆积分 RF | `elliptic_special.py` | 计算火焰曲率相关的特殊函数和 Markstein 长度 |
| 10 | `309_distmesh_3d` | 3D 距离函数网格生成 | `mesh3d_generator.py` | 生成圆柱形燃烧室的 3D 四面体非结构网格 |
| 11 | `702_logistic_ode` | Logistic ODE | `arrhenius_kinetics.py` | 构造 Arrhenius 反应速率方程和进展变量演化模型 |
| 12 | `158_change_polynomial` | 找零钱多项式算法 | `stoichiometric_polynomial.py` | 化学计量路径的组合计数与反应机理复杂度分析 |
| 13 | `1049_rubber_band_ode` | 分段非线性 ODE | `flame_instability.py` | 火焰前锋 Darrieus-Landau 热扩散不稳定性动力学 |
| 14 | `358_fd1d_bvp` | 1D BVP 有限差分 | `fd_scalar_solver.py` | 有限差分法求解标量耗散率修正方程 |
| 15 | `600_isbn` | ISBN 校验和算法 | `conservation_validator.py` | 质量守恒与能量守恒的加权校验和验证 |

---

## 四、代码文件结构

```
156_synth_project/
├── main.py                          # 统一入口，零参数可运行
├── flamelet_core.py                 # 核心物理模型与参数定义
├── fem_thermal_solver.py            # FEM 线性单元温度场求解器 (原392)
├── fem_quadratic_solver.py          # FEM 二次单元组分场求解器 (原387)
├── fd_scalar_solver.py              # FDM 标量耗散率求解器 (原358)
├── mesh3d_generator.py              # 3D 燃烧室网格生成器 (原309)
├── turbulent_random_field.py        # 湍流随机脉动场生成器 (原1005)
├── ignition_polynomial.py           # 点火多项式与 WDK 求根 (原1404)
├── flame_front_shape.py             # 火焰前锋几何形状 (原093)
├── arrhenius_kinetics.py            # Arrhenius 反应动力学 (原702)
├── flame_instability.py             # 火焰不稳定性动力学 (原1049)
├── elliptic_special.py              # 椭圆积分与曲率效应 (原1274)
├── spatial_partition.py             # Voronoi 空间分区 (原725)
├── matrix_bandwidth.py              # 矩阵带宽分析 (原752)
├── stoichiometric_polynomial.py     # 化学计量路径计数 (原158)
├── conservation_validator.py        # 守恒律校验 (原600)
├── exact_benchmark.py               # 精确解验证基准 (原316)
└── README_博士级合成说明.md          # 本说明文档
```

**共 18 个文件（17 个 .py + 1 个 .md）。**

---

## 五、运行方式

```bash
cd Synthesis-project-python/156_synth_project
python main.py
```

程序无需任何命令行参数，直接运行即可完成从参数初始化、网格离散、
FEM/FDM 求解、湍流脉动生成、点火分析、火焰几何计算、3D 网格生成、
带宽分析、守恒校验到误差评估的完整流程。

---

## 六、数值方法与工程鲁棒性

1. **非线性迭代**：温度场和组分场均采用 Picard 迭代，带收敛容差控制；
2. **边界处理**：所有物理量均做 clip 处理，防止温度/质量分数越界；
3. **指数溢出保护**：Arrhenius 指数项限制在 [-700, 700]；
4. **除零保护**：所有分母加入最小阈值（1.0e-12 ~ 1.0e-30）；
5. **网格质量检查**：3D 网格生成中自动剔除外部四面体；
6. **守恒校验**：运行结束时自动验证质量守恒和能量守恒；
7. **多求解器交叉验证**：FEM 温度场与 Manufactured Solution 和高斯渐近解对比。

---

## 七、科学问题与创新点

本项目解决的核心科学问题是：**在湍流非预混燃烧条件下，如何准确预测层流火焰面在混合分数空间中的温度与组分分布，并评估点火、熄火、不稳定性等极限行为。**

主要创新点：
- 将 15 个独立科研代码的核心算法系统性地融合为一个统一的火焰面模拟框架；
- 引入 WDK 多项式求根方法进行点火/熄火极限的精确分析；
- 结合湍流随机脉动场与标量耗散率脉动，实现火焰面-湍流耦合效应的简化建模；
- 通过蛋形曲线参数化火焰前锋几何，定量计算皱褶因子对湍流火焰速度的贡献；
- 利用不完全椭圆积分精确计算曲率效应和 Markstein 修正；
- 3D 网格生成与带宽分析为后续全尺度 LES/RANS 耦合计算提供前处理基础。
