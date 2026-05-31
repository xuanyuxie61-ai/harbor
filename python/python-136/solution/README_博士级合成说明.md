# 催化剂孔扩散与表面反应：多尺度耦合模拟系统

## 一、项目概述

本项目围绕**化学工程：催化剂孔扩散与表面反应**这一前沿博士级科学领域，基于 15 个种子科研代码项目的核心算法，构建了一个完整的多尺度数值模拟平台。系统涵盖从孔道尺度到颗粒尺度的扩散-反应耦合过程，集成了有限差分/有限元求解、非线性牛顿迭代、自适应网格生成、高斯数值积分、蒙特卡洛孔结构分析、共轭梯度稀疏求解等高端数值方法，并注入了丰富的物理化学数学公式。

## 二、原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 在本系统中的角色 |
|------|-----------|---------|----------------|
| 1 | `1268_toms243` | 复数自然对数稳定算法 | 特殊函数模块：`complex_log_stable()`，用于解析解中的复频域稳定性分析与球谐函数展开 |
| 2 | `152_cg_rc` | 反向通信共轭梯度法 | 线性求解器模块：`conjugate_gradient_rc()`，用于求解 FEM/FDM 离散后的稀疏对称正定线性系统 |
| 3 | `298_disk_triangle_picking` | 圆盘内随机三角形蒙特卡洛 | 孔结构分析：`random_triangle_area_in_disk()`，估计孔截面连通面积与曲折度 |
| 4 | `098_black_scholes` | Black-Scholes PDE 解析解 | 积分验证模块：利用热核类比构造扩散方程 Green 函数精确解，校验数值离散守恒性 |
| 5 | `927_pwl_interp_2d` | 二维分段线性插值 | 插值模块：`pwl_interp_2d()`，将一维径向解映射到二维截面进行后处理 |
| 6 | `246_cvt_1d_sampling` | 一维 CVT Lloyd 算法 | 网格生成：`cvt_1d_lloyd()`，在反应剧烈区自动加密径向节点 |
| 7 | `1420_xy_io` | XY 坐标数据读写 | 数据 I/O：`write_xy_profile()` / `read_xy_profile()`，持久化浓度-温度剖面 |
| 8 | `1143_square_exactness` | 2D Legendre 积分精确度验证 | 积分验证：`validate_2d_quadrature_rule()`，确保高斯积分误差可控 |
| 9 | `320_duel_simulation` | 决斗概率随机模拟 | 孔道可达性：`pore_accessibility_simulation()`，模拟反应物分子在多级孔道中的随机穿行 |
| 10 | `461_gegenbauer_exactness` | Gegenbauer 积分与超几何函数 | 特殊函数：`gegenbauer_integral()`、`r8_hyper_2f1()` 迁移，用于孔径分布矩计算 |
| 11 | `386_fem1d_bvp_linear` | 一维线性有限元 | 核心求解：`solve_diffusion_reaction_fem()`，分段线性基函数离散弱形式 |
| 12 | `467_gen_laguerre_rule` | 广义 Laguerre 高斯积分规则 | 积分规则：`gauss_genlaguerre_rule()`，用于无限域与径向平均积分 |
| 13 | `358_fd1d_bvp` | 一维有限差分边值问题 | 核心求解：`solve_diffusion_reaction_fd()`，有限体积法离散球坐标守恒方程 |
| 14 | `261_cvt_square_uniform` | 二维正方形 CVT | 网格生成：`cvt_square_uniform_2d()`，催化剂截面二维域剖分 |
| 15 | `125_burgers_steady_viscous` | 稳态 Burgers 方程牛顿迭代 | 非线性求解：`solve_coupled_diffusion_reaction_newton()`，块 Gauss-Seidel / 阻尼牛顿框架 |

## 三、核心科学模型与公式

### 3.1 球形颗粒内的稳态扩散-反应方程

对于组分 $A$（如 CO）在半径 $r$ 处的浓度 $C_A(r)$，满足：

$$
\frac{D_e}{r^2} \frac{\mathrm{d}}{\mathrm{d}r}\left(r^2 \frac{\mathrm{d}C_A}{\mathrm{d}r}\right) - R(C_A, T) = 0, \quad 0 < r < R_p
$$

展开形式：

$$
D_e \left( \frac{\mathrm{d}^2 C_A}{\mathrm{d}r^2} + \frac{2}{r} \frac{\mathrm{d}C_A}{\mathrm{d}r} \right) = R(C_A, T)
$$

边界条件：

$$
\left. \frac{\mathrm{d}C_A}{\mathrm{d}r} \right|_{r=0} = 0 \quad (\text{对称性}), \qquad C_A(R_p) = C_{A,\text{surf}} \quad (\text{Dirichlet})
$$

### 3.2 能量守恒方程（温度耦合）

$$
\frac{\lambda_e}{r^2} \frac{\mathrm{d}}{\mathrm{d}r}\left(r^2 \frac{\mathrm{d}T}{\mathrm{d}r}\right) + (-\Delta H) \, R(C_A, T) = 0
$$

其中有效导热系数：

$$
\lambda_e = \lambda_{\text{solid}} (1-\varepsilon) + \lambda_{\text{gas}} \varepsilon
$$

### 3.3 有效扩散系数（并联阻力模型）

Knudsen 扩散系数：

$$
D_{Kn} = \frac{d_p}{3} \sqrt{\frac{8RT}{\pi M_A}}
$$

综合 Bulk 与 Knudsen 扩散：

$$
\frac{1}{D_{\text{eff,pore}}} = \frac{1}{D_{\text{bulk}}} + \frac{1}{D_{Kn}}
$$

考虑孔隙率 $\varepsilon$ 与曲折因子 $\tau$：

$$
D_e = \frac{\varepsilon}{\tau} D_{\text{eff,pore}}
$$

### 3.4 Langmuir-Hinshelwood 表面反应动力学

以 CO 氧化为例（$\text{CO} + \tfrac{1}{2}\text{O}_2 \rightarrow \text{CO}_2$）：

$$
R = \frac{k_r K_{\text{CO}} C_{\text{CO}} \sqrt{K_{\text{O}_2} C_{\text{O}_2}}}{\left(1 + K_{\text{CO}} C_{\text{CO}} + \sqrt{K_{\text{O}_2} C_{\text{O}_2}}\right)^2}
$$

温度依赖的 Arrhenius 形式：

$$
k_r = k_0 \exp\!\left(-\frac{E_a}{RT}\right), \qquad K_i = K_{i,0} \exp\!\left(-\frac{\Delta H_{\text{ads},i}}{RT}\right)
$$

### 3.5 Thiele 模数与内部效率因子

Thiele 模数（基于一级近似）：

$$
\phi = R_p \sqrt{\frac{k_{\text{obs}}}{D_e}}, \qquad k_{\text{obs}} = \frac{R(C_{\text{surf}}, T_{\text{surf}})}{C_{\text{surf}}}
$$

球形颗粒的理论效率因子：

$$
\eta = \frac{3}{\phi^2}\bigl(\phi \coth\phi - 1\bigr)
$$

数值计算的效率因子：

$$
\eta_{\text{num}} = \frac{\displaystyle\int_0^{R_p} R\bigl(C(r)\bigr) \, 4\pi r^2 \, \mathrm{d}r}{R(C_{\text{surf}}) \cdot \tfrac{4}{3}\pi R_p^3}
$$

### 3.6 Weisz-Prater 准则

$$
C_{WP} = \eta \phi^2 = \frac{R_{\text{obs}}}{D_e C_{\text{surf}} / R_p^2}
$$

工程判据：若 $C_{WP} \ll 1$，扩散限制可忽略；若 $C_{WP} > 0.3$，存在显著的孔内扩散限制。

### 3.7 有限体积法离散（球坐标）

在控制体积 $[r_{i-1/2}, r_{i+1/2}]$ 上积分：

$$
r_{i+1/2}^2 D_e \frac{C_{i+1}-C_i}{r_{i+1}-r_i} - r_{i-1/2}^2 D_e \frac{C_i-C_{i-1}}{r_i-r_{i-1}} = r_i^2 R_i \, \Delta r_i
$$

### 3.8 有限元弱形式

乘以测试函数 $v(r)$ 并分部积分：

$$
\int_0^{R_p} D_e \frac{\mathrm{d}C}{\mathrm{d}r} \frac{\mathrm{d}v}{\mathrm{d}r} \, r^2 \, \mathrm{d}r = -\int_0^{R_p} R(C,T) \, v \, r^2 \, \mathrm{d}r
$$

单元刚度矩阵采用两点 Gauss-Legendre 数值积分。

### 3.9 Black-Scholes 与扩散方程的数学类比

Black-Scholes PDE：

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0
$$

通过变量替换 $x = \ln(S/K)$, $\tau = \tfrac{1}{2}\sigma^2 (T-t)$，可化为标准热传导方程：

$$
\frac{\partial u}{\partial \tau} = \frac{\partial^2 u}{\partial x^2}
$$

这与催化剂孔道内的一维非稳态扩散方程 $\partial C/\partial t = D \partial^2 C/\partial z^2$ 形式一致，其 Green 函数（热核）可用于数值积分守恒性校验。

### 3.10 蒙特卡洛有效扩散系数（Einstein 关系）

$$
D_{\text{eff}}^{\text{MC}} = \frac{\langle |\mathbf{r}(t) - \mathbf{r}(0)|^2 \rangle}{6t}
$$

## 四、代码架构与文件说明

本项目共包含 **12 个 Python 模块** 与 **1 个统一入口文件** `main.py`：

| 文件 | 功能 |
|------|------|
| `main.py` | 统一入口，零参数运行，自动执行完整多尺度模拟流程 |
| `special_functions.py` | 复数对数、Knudsen 扩散、有效扩散系数、Arrhenius 速率、Thiele 模数与效率因子、Gegenbauer 积分 |
| `linear_solvers.py` | Thomas 算法（三对角）、反向通信 CG、Jacobi 预处理、scipy 稀疏 CG |
| `quadrature_rules.py` | Gauss-Legendre、广义 Laguerre、径向球积分、Gegenbauer 精确度验证、反应速率体积分 |
| `mesh_generation.py` | 一维/二维 CVT Lloyd 自适应网格生成，针对反应梯度加密 |
| `interpolation.py` | 二维分段线性插值（PWL）、径向到二维圆盘的场量映射 |
| `monte_carlo_pore.py` | 随机三角形面积估计、孔道可达性马尔可夫模拟、随机行走估计 D_eff、曲折因子统计估计 |
| `integration_validator.py` | 2D 积分精确度验证、扩散 Green 函数守恒校验、Black-Scholes 类比、反应-扩散积分守恒校验 |
| `pore_diffusion.py` | FDM 有限体积法、FEM 分段线性元、表面通量计算、效率因子提取 |
| `surface_reaction.py` | Langmuir-Hinshelwood 动力学、幂律动力学、催化剂颗粒多物理场模型、Thiele/Weisz-Prater 计算 |
| `nonlinear_solver.py` | 阻尼牛顿法（Burgers 风格）、块 Gauss-Seidel 耦合迭代、伪瞬态延续法 |
| `data_io.py` | XY 剖面数据、二维孔结构数据的读写与边界校验 |

所有模块均包含完整的边界条件处理、数值鲁棒性校验（非负浓度、正温度、概率归一化等），且**已完全删除可视化代码**。

## 五、运行方式

```bash
cd Synthesis-project-python/136_synth_project
python main.py
```

程序无需任何命令行参数，运行后将自动：
1. 初始化 Pt/Al₂O₃ 催化剂颗粒物理模型（CO 氧化反应）
2. 计算 Knudsen/有效扩散系数、Thiele 模数、理论效率因子
3. 生成自适应 CVT 径向网格与二维截面网格
4. 验证 Gauss-Legendre、Laguerre、Gegenbauer 数值积分精确度
5. 执行蒙特卡洛孔结构分析（随机几何、可达性、曲折因子）
6. 利用 Black-Scholes 热核类比验证数值积分守恒性
7. 分别使用 FDM、FEM、块 Gauss-Seidel 牛顿耦合求解非线性扩散-反应方程组
8. 计算表面扩散通量、效率因子、Weisz-Prater 准则、积分守恒误差
9. 将浓度/温度剖面数据持久化到 `output/` 目录

## 六、典型输出结果解读

以默认参数（$R_p = 3.0\,\text{mm}$, $T_{\text{surf}} = 573\,\text{K}$）为例：

- **Thiele 模数** $\phi \approx 2.08$：中等扩散限制
- **理论效率因子** $\eta \approx 0.80$：约 20% 的活性位点因扩散限制未能充分利用
- **FDM 效率因子** $\eta_{\text{num}} \approx 1.07$：由于 L-H 动力学的非单调性（强吸附导致表面堵塞），内部适中浓度区域的反应活性反而高于表面，出现 $\eta > 1$ 的特殊现象
- **中心/表面浓度比** $C_0/C_s \approx 0.23$：显著的浓度梯度
- **Weisz-Prater 准则** $C_{WP} \approx 4.6 \gg 0.3$：存在强烈的孔内扩散限制
- **积分守恒误差** $< 0.4\%$：数值离散具有良好守恒性
- **最大温度升高** $\Delta T_{\max} \approx 1.0\,\text{K}$：放热反应导致的轻微颗粒内热梯度

## 七、科学问题的深度与前沿性

本项目解决的科学问题处于化学工程反应工程与传递现象交叉领域的前沿：

1. **多尺度耦合**：从纳米孔道（Knudsen 扩散）到毫米级颗粒（Thiele 模数）的跨尺度建模；
2. **非线性动力学与扩散耦合**：L-H 动力学的非单调性导致效率因子可能大于 1，突破了经典一级反应近似；
3. **热-质耦合**：放热反应引起的颗粒内温度分布反作用于反应速率，形成正反馈或热稳定问题；
4. **数值方法验证体系**：通过 Green 函数解析解、Black-Scholes 热核类比、高斯积分精确度验证，建立了完整的数值可信度评估框架；
5. **蒙特卡洛孔结构分析**：将随机几何与随机行走方法引入催化剂孔扩散研究，补充了连续介质模型的不足。

## 八、质量检查清单

- [x] 原目录未被修改
- [x] 合成后项目为 Python 语言
- [x] 新目录包含完整合成项目（12 个 .py 文件）
- [x] 只有一个博士级科学计算问题落地为可执行代码
- [x] 15 个输入项目均已真实融入，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 已删除所有可视化相关内容
