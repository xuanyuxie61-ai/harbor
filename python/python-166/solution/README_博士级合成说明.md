# 软体机器人运动学建模 — 博士级科研代码合成项目

## 1. 项目概述

本项目围绕**机器人学：软体机器人运动学建模**这一前沿科学领域，将15个种子科研项目的核心算法融合为一个完整的博士级计算框架。软体机器人由超弹性材料（如硅胶、离子聚合物金属复合材料IPMC）构成，其连续体变形涉及几何非线性、材料非线性与化学-力学耦合，传统刚性机器人理论无法直接适用。本项目基于**Cosserat杆理论**，构建了从横截面分析、前向/逆向运动学、动力学到降阶模型的完整计算链条。

## 2. 科学问题与数学模型

### 2.1 Cosserat杆理论

软体机器人可建模为具有可变形截面的细长弹性杆。设中心线位置为 $\mathbf{r}(s,t) \in \mathbb{R}^3$，截面姿态为正交矩阵 $\mathbf{R}(s,t) \in SO(3)$，其中 $s \in [0,L]$ 为弧长参数。

**应变度量:**
- 线应变向量: $\mathbf{v} = \mathbf{R}^T \mathbf{r}'$
- 曲率向量: $\mathbf{u} = \left( \mathbf{R}^T \mathbf{R}' \right)^\vee$

其中 $(\cdot)^\vee$ 为 $so(3) \to \mathbb{R}^3$ 的vee映射（hat映射的逆）。

**运动学约束:**
$$\dot{\mathbf{r}} = \mathbf{R}\mathbf{q}, \quad \dot{\mathbf{R}} = \mathbf{R}\hat{\mathbf{w}}$$

其中 $\mathbf{q}$ 为线速度，$\mathbf{w}$ 为角速度，$\hat{\cdot}$ 为向量到反对称矩阵的hat映射。

**动力学平衡方程:**
$$\begin{aligned}
\rho A \ddot{\mathbf{r}} &= \mathbf{n}' + \tilde{\mathbf{f}} \\
\mathbf{I}\dot{\mathbf{w}} + \mathbf{w} \times (\mathbf{I}\mathbf{w}) &= \mathbf{m}' + \mathbf{r}' \times \mathbf{n} + \tilde{\boldsymbol{\tau}}
\end{aligned}$$

其中 $\mathbf{n}, \mathbf{m}$ 为截面内力/内矩，$\rho$ 为密度，$A$ 为截面积，$\mathbf{I}$ 为截面惯性张量。

**线性化本构关系:**
$$\mathbf{n} = \begin{bmatrix} GA_s v_1 \\ GA_s v_2 \\ EA(v_3 - 1) \end{bmatrix}, \quad
\mathbf{m} = \begin{bmatrix} EI_{xx} u_1 \\ EI_{yy} u_2 \\ GJ u_3 \end{bmatrix}$$

### 2.2 超弹性本构模型

**Neo-Hookean模型:**
$$W = \frac{\mu}{2}(I_1 - 3) - \mu \ln J + \frac{K_{bulk}}{2}(\ln J)^2$$

其中 $I_1 = \text{tr}(\mathbf{C})$，$\mathbf{C} = \mathbf{F}^T\mathbf{F}$，$J = \det(\mathbf{F})$。

第一Piola-Kirchhoff应力:
$$\mathbf{P} = \mu(\mathbf{F} - \mathbf{F}^{-T}) + K_{bulk}\ln(J)\mathbf{F}^{-T}$$

**Mooney-Rivlin模型:**
$$W = C_{10}(I_1 - 3) + C_{01}(I_2 - 3) + \frac{K_{bulk}}{2}(\ln J)^2$$

其中 $I_2 = \frac{1}{2}[\text{tr}(\mathbf{C})^2 - \text{tr}(\mathbf{C}^2)]$。

**化学-力学耦合:**
软体材料（如IPMC）的弹性模量受化学状态影响:
$$E_{eff} = E_0 (1 + \gamma c_{ion}) \exp(-\beta_{chem}|pH - 7|)$$

其中 $c_{ion}$ 为离子浓度，$pH$ 为酸碱度。

### 2.3 双调和方程（薄板弯曲验证）

横截面/薄板弯曲满足双调和方程:
$$\nabla^4 W = W_{xxxx} + 2W_{xxyy} + W_{yyyy} = R$$

本项目提供三族精确制造解用于数值验证:

**族1（双曲-三角可分离）:**
$$W = [a\cosh(gX) + b\sinh(gX) + cX\cosh(gX) + dX\sinh(gX)] \cdot [e\cos(gY) + f\sin(gY)]$$

**族2（三角-双曲互换）:**
$$W = [a\cos(gX) + b\sin(gX) + cX\cos(gX) + dX\sin(gX)] \cdot [e\cosh(gY) + f\sinh(gY)]$$

**族3（径向对数，奇异性）:**
$$W = aR^2\ln R + bR^2 + c\ln R + d, \quad R = \sqrt{(X-e)^2 + (Y-f)^2}$$

薄板弯曲应变能:
$$U = \frac{1}{2}D \iint \left[ (\nabla^2 W)^2 - 2(1-\nu)\left( \frac{\partial^2 W}{\partial x^2}\frac{\partial^2 W}{\partial y^2} - \left(\frac{\partial^2 W}{\partial x\partial y}\right)^2 \right) \right] dxdy$$

其中 $D = \frac{Eh^3}{12(1-\nu^2)}$ 为弯曲刚度。

### 2.4 Selkov糖酵解化学动力学

用作化学驱动的代谢模型:
$$\begin{aligned}
\frac{du}{dt} &= -u + av + u^2v \\
\frac{dv}{dt} &= b - av - u^2v
\end{aligned}$$

参数 $a,b$ 控制系统的振荡行为（Hopf分岔）。

### 2.5 锯齿波驱动

仿生周期性肌肉激励:
$$f(t) = \text{mod}(t + \omega\pi, 2\omega\pi) - \omega\pi$$

周期 $T = 2\omega\pi$，幅值 $\omega\pi$。

## 3. 种子项目映射关系

| 编号 | 种子项目 | 核心算法 | 在合成项目中的角色 |
|------|----------|----------|-------------------|
| 1 | 970_r8blt | 带状下三角矩阵紧凑存储与快速求解 | Cosserat杆有限元刚度矩阵的前代求解与矩阵-向量乘法 (`cosserat_core.py`) |
| 2 | 680_line_grid | 1D网格5种居中方案生成 | 中心线弧长参数离散化 (`mesh_utils.py`) |
| 3 | 344_exactness | 高斯求积规则精确性测试 | 横截面属性积分的高斯-勒让德求积 (`section_fem.py`) |
| 4 | 375_fem_basis_t6_display | T6二次三角形形函数及导数 | 截面有限元应力分析的形函数计算 (`section_fem.py`) |
| 5 | 273_dg1d_heat | 1D热方程DG方法（Jacobi多项式、Vandermonde、lift算子、低存储RK） | 空间高阶离散化 (`spectral_dg.py`) |
| 6 | 1184_svd_basis | SVD降阶基提取（POD/ROM） | 软体形状快照的POD降阶 (`rom_pod.py`) |
| 7 | 161_chebyshev_matrix | Chebyshev谱微分矩阵 | 中心线曲率与导数的高精度谱计算 (`spectral_dg.py`, `cosserat_core.py`) |
| 8 | 087_biharmonic_exact | 双调和方程3族精确解 | 薄板弯曲变形的制造解验证 (`validation_biharmonic.py`) |
| 9 | 156_change_dynamic | 动态规划硬币找零 | 多段构型空间最优路径规划 (`path_planner.py`) |
| 10 | 1059_sawtooth_ode | 锯齿波驱动谐振子ODE | 周期性仿生驱动的动态响应 (`dynamics_solver.py`) |
| 11 | 276_diaphony | 点集Diaphony均匀性度量 | 横截面采样点质量评估 (`mesh_utils.py`) |
| 12 | 472_glycolysis_ode | Selkov糖酵解ODE | 化学-力学耦合驱动模型 (`hyperelastic_law.py`, `dynamics_solver.py`) |
| 13 | 928_pwl_interp_2d_scattered | 散乱数据2D分段线性插值（Delaunay+行走搜索） | 传感器散乱读数的形状重构 (`inverse_kinematics.py`) |
| 14 | 138_cauchy_method | Cauchy(theta)ODE单步法 | Cosserat杆动力学的时间积分 (`dynamics_solver.py`) |
| 15 | 548_human_mesh2d | 2D约束Delaunay三角剖分 | 软体机器人横截面网格生成 (`mesh_utils.py`) |

## 4. 文件结构

```
166_synth_project/
├── main.py                          # 统一入口，零参数可运行
├── mesh_utils.py                    # 网格生成、Diaphony采样质量评估
├── section_fem.py                   # T6截面有限元、高斯求积、截面属性
├── spectral_dg.py                   # Chebyshev谱微分、Jacobi多项式、DG算子
├── hyperelastic_law.py              # Neo-Hookean/Mooney-Rivlin本构、化学耦合
├── dynamics_solver.py               # Cauchy(theta)积分、锯齿波驱动、动力学
├── cosserat_core.py                 # Cosserat杆运动学核心、带状矩阵求解
├── inverse_kinematics.py            # 逆运动学、散乱数据PWL/RBF插值
├── path_planner.py                  # 动态规划路径规划
├── rom_pod.py                       # SVD/POD降阶模型
├── validation_biharmonic.py         # 双调和方程精确解与验证
└── README_博士级合成说明.md          # 本说明文档
```

## 5. 运行方法

```bash
cd /mnt/data/zpy/sci-swe/source\ code/Synthesis-project-python/166_synth_project
python main.py
```

程序将依次运行9个模块，输出各模块的关键计算结果到控制台。无需任何输入参数。

## 6. 关键技术特点

1. **高阶数值方法**: Chebyshev谱微分、Jacobi-Gauss-Lobatto点、DG lift算子
2. **化学-力学耦合**: 将糖酵解ODE与超弹性本构实时耦合
3. **多尺度建模**: 从微观截面FEM到宏观Cosserat杆再到POD降阶
4. **边界处理与鲁棒性**: 所有数值操作包含奇异值保护、正则化、截断
5. **无可视化**: 纯数值计算，符合高性能科学计算规范
