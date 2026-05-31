# PROJECT_162：锂电池电化学-热耦合全尺度仿真平台

## 1. 项目概述

本项目基于 **15 个科研代码种子项目** 的核心算法，在 **能源系统：锂电池电化学热耦合** 领域内融合重构为一个前沿博士级科学计算问题。

### 1.1 科学问题定义

锂电池的失效机理与性能衰减本质上是**电化学反应-离子传输-热传导**三场强耦合的非线性问题。本项目构建了一个**伪二维 Doyle-Fuller-Newman (DFN) 电化学模型**与**二维热有限元模型**的耦合求解框架，涵盖：

- **介观尺度**：1D 宏观电极-隔膜-电极三明治结构中的电荷守恒与电解液传输
- **微观尺度**：球形活性颗粒中的径向固相扩散（Fick 第二定律）
- **介观-微观耦合**：Butler-Volmer 电极动力学连接固相表面浓度与宏观电流
- **热尺度**：2D 非结构化三角形网格上的瞬态热传导有限元求解
- **电化学-热耦合**：温度依赖的材料物性（Arrhenius 律）、热源反馈
- **统计与优化**：粒子尺寸分布（PSD）聚类、充电协议组合优化、随机传输蒙特卡洛分析

### 1.2 核心控制方程

#### 固相扩散（球形颗粒，径向坐标 $r$）

$$
\frac{\partial C_s}{\partial t} = \frac{1}{r^2} \frac{\partial}{\partial r}\left( D_s(T) \, r^2 \frac{\partial C_s}{\partial r} \right)
$$

边界条件：
$$
\left. \frac{\partial C_s}{\partial r}\right|_{r=0} = 0, \quad -D_s \left. \frac{\partial C_s}{\partial r}\right|_{r=R} = \frac{j_{BV}}{F}
$$

其中 $D_s(T) = D_{s,\text{ref}} \exp\left(-\frac{E_a}{R_g}\left(\frac{1}{T}-\frac{1}{T_{\text{ref}}}\right)\right)$ 为 Arrhenius 温度依赖固相扩散系数。

#### 电解液浓度传输（1D 宏观坐标 $x$）

$$
\varepsilon_e \frac{\partial C_e}{\partial t} = \frac{\partial}{\partial x}\left( D_{e,\text{eff}}(T) \frac{\partial C_e}{\partial x} \right) + \frac{1-t_+}{F} a_s j_{BV}
$$

#### 电荷守恒（固相电势 $\phi_s$）

$$
\frac{\partial}{\partial x}\left( \sigma_{s,\text{eff}} \frac{\partial \phi_s}{\partial x} \right) = a_s F j_{BV}
$$

#### 电荷守恒（电解液电势 $\phi_e$）

$$
\frac{\partial}{\partial x}\left( \kappa_{\text{eff}}(T) \frac{\partial \phi_e}{\partial x} \right) + \frac{\partial}{\partial x}\left( \kappa_{D,\text{eff}} \frac{\partial \ln C_e}{\partial x} \right) = -a_s F j_{BV}
$$

#### Butler-Volmer 电极动力学

$$
j_{BV} = j_0 \left[ \exp\left(\frac{\alpha_a F \eta}{R_g T}\right) - \exp\left(-\frac{\alpha_c F \eta}{R_g T}\right) \right]
$$

其中交换电流密度：
$$
j_0 = k_{\text{ref}} \sqrt{C_e C_{s,\text{surf}} (C_{s,\max} - C_{s,\text{surf}})} \, \exp\left(-\frac{E_a}{R_g}\left(\frac{1}{T}-\frac{1}{T_{\text{ref}}}\right)\right)
$$

过电势：
$$
\eta = \phi_s - \phi_e - U_{\text{ocp}}(C_{s,\text{surf}}) - I R_{\text{film}}
$$

#### 二维瞬态热传导

$$
\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (\kappa_T \nabla T) + Q_{\text{gen}}
$$

热源 $Q_{\text{gen}}$ 包含：
- 欧姆热：$q_{\text{ohm}} = \sigma_{\text{eff}} |\nabla \phi|^2$
- 不可逆反应热：$q_{\text{react}} = \eta \, j_{BV} \, a_s F$
- 可逆熵热：$q_{\text{rev}} = I T \frac{\partial U_{\text{ocp}}}{\partial T} \frac{a_s}{3F}$

#### 时间离散（Crank-Nicolson）

$$
\left(M + \frac{\Delta t}{2} K\right) T^{n+1} = \left(M - \frac{\Delta t}{2} K\right) T^n + \Delta t \, Q^n
$$

其中 $M$ 为质量矩阵，$K$ 为刚度矩阵。

---

## 2. 种子项目到科学问题的映射

| 种子项目 | 核心算法 | 合成后角色 |
|---------|---------|-----------|
| **979_r8gb** | 一般带状矩阵 PLU 分解与求解 | 电化学固相扩散有限差分离散后的带状线性系统求解器；热 FEM 刚度矩阵的带状存储与求解 |
| **1363_tsp_brute** | Trotter 算法全排列枚举 | 充电协议电流步序的暴力搜索优化；电流-时长配对的最优排列 |
| **065_ball_and_stick_display** | Lax-Wendroff 有限差分格式 | 热传导时间推进的 Crank-Nicolson 稳定性分析思想；有限差分 stencil 设计 |
| **757_mesh2d** | 2D 非结构化三角网格生成、Laplacian 光滑、质量评估 | 电池横截面几何的三角网格生成；边界保持的 Laplacian 网格光滑 |
| **1117_sphere_fibonacci_grid** | Fibonacci 螺旋准均匀球面点分布 | 2D 域内准均匀种子点生成（自适应）；颗粒表面离散化思想 |
| **1267_toms179** | 不完全 Beta 函数、对数 Gamma | 粒子尺寸分布的置信区间统计；PSD 混合模型中的 Beta 分布 CDF 反演 |
| **039_asa113** | Transfer/Swap 聚类优化 | 粒子尺寸分布分类；充电协议电流段聚类 |
| **470_gl_fast_rule** | Bogaert 迭代自由 Gauss-Legendre 求积 | FEM 单元积分高阶求积；Butler-Volmer 反应率在颗粒表面的数值积分 |
| **209_conte_deboor** | 样条插值、FFT、Muller 根求解、LU、RK2 | 温度依赖物性的三次样条插值；阻抗谱 FFT 分析；过电势 Muller 反解；ODE 时间积分 |
| **424_feynman_kac_3d** | Feynman-Kac 公式、布朗运动蒙特卡洛 | 颗粒内锂离子随机传输建模；首达时间蒙特卡洛估计；浓度涨落分析 |
| **1194_t_puzzle_gui** | 复数几何、点在多边形内测试 | 电池几何区域的复数表示；电极/隔膜/集流体区域分类的点在多边形测试 |
| **412_fem2d_project_function** | FEM L² 投影、质量矩阵组装、Dirichlet BC | 2D 热 FEM 刚度/质量矩阵组装；边界条件处理；L² 误差评估 |
| **999_r8sto** | 对称 Toeplitz Durbin/Levinson 求解 | 径向扩散周期算子的快速 Toeplitz 矩阵-向量乘；阻抗计算中的卷积型算子 |
| **1291_treepack** | Prüfer 编码、树枚举、Catalan 数 | 粒子聚类层次树的编码/解码；微观结构组合枚举 |
| **407_fem2d_pack** | T3 形函数、参考-物理映射、三角形求积 | 热 FEM 的 P1 线性三角形形函数；Jacobi 映射；参考三角形高阶求积规则 |

---

## 3. 项目文件结构

```
162_synth_project/
├── main.py                        # 统一入口，零参数运行
├── electrochemistry.py            # DFN 电化学核心（固相扩散、电解液、Butler-Volmer）
├── thermal_fem.py                 # 2D 瞬态热有限元求解器（Crank-Nicolson）
├── mesh_generator.py              # 电池几何网格生成与光滑
├── geometry_engine.py             # 几何引擎（多边形、区域分类）
├── fem_assembler.py               # FEM 矩阵组装（刚度、质量、Dirichlet BC）
├── banded_linear_algebra.py       # 带状矩阵 PLU / Toeplitz Durbin 求解器
├── quadrature_special.py          # Gauss-Legendre / 三角形求积 / 特殊函数
├── numerical_toolkit.py           # 样条、FFT、根求解、LU、RK2 等数值工具箱
├── stochastic_li_transport.py     # Feynman-Kac 蒙特卡洛随机传输
├── particle_distribution.py       # PSD 建模、聚类、层次树、有效扩散统计
├── protocol_optimizer.py          # 充电协议组合优化（TSP/贪心/聚类）
└── README_博士级合成说明.md       # 本文档
```

共 **12 个 .py 文件**（不含 `__pycache__`），满足至少 8 个的要求。

---

## 4. 运行方式

```bash
cd 162_synth_project
python main.py
```

无需任何参数。程序自动执行以下 8 个阶段：

1. **几何与网格**：生成 861 节点、1600 三角单元的电池横截面网格
2. **PSD 建模**：500 颗粒的对数正态分布 + Transfer/Swap 聚类
3. **DFN 电化学**：20 步时间推进，输出电压与表面浓度
4. **热 FEM 耦合**：20 步 Crank-Nicolson 热传导，含电化学热源反馈
5. **随机传输**：Feynman-Kac 蒙特卡洛（200 路径）+ 首达时间 + 电解液随机游走
6. **协议优化**：4 电流等级的暴力搜索 + 贪心热约束协议
7. **阻抗谱 FFT**：Cooley-Tukey FFT 计算复阻抗 $Z(f)$
8. **数值验证**：对数 Gamma、不完全 Beta、Gauss-Legendre、带状矩阵、Toeplitz、Muller、RK2

---

## 5. 关键科学公式与代码对应

### 5.1 固相扩散的有限差分离散

在 `electrochemistry.py` 的 `SolidDiffusionSolver` 中，球坐标下的隐式离散为：

$$
\frac{C_i^{n+1} - C_i^n}{\Delta t} = \frac{D_s}{r_i^2 \Delta r^2} \left[ r_{i+1/2}^2 (C_{i+1}^{n+1} - C_i^{n+1}) - r_{i-1/2}^2 (C_i^{n+1} - C_{i-1}^{n+1}) \right]
$$

整理后形成三对角带状系统 $A C^{n+1} = b$，使用 `BandedMatrix` 存储并以 PLU 分解求解。

### 5.2 Butler-Volmer 过电势反解

在 `electrochemistry.py` 中，已知目标电流密度 $j_{\text{target}}$，求解过电势 $\eta$：

$$
j_{\text{target}} = j_0 \left[ e^{\alpha_a F \eta / RT} - e^{-\alpha_c F \eta / RT} \right]
$$

使用 `numerical_toolkit.muller_root`（Muller 二次插值法）在区间 $[-0.5, 0.5]$ V 内反解 $\eta$，残差 $<10^{-12}$。

### 5.3 FEM 刚度矩阵组装

在 `fem_assembler.py` 中，参考三角形上的 P1 形函数：

$$
N_1 = 1 - \xi - \eta, \quad N_2 = \xi, \quad N_3 = \eta
$$

物理梯度通过仿射映射 Jacobi 矩阵转换：

$$
\nabla N_i = J^{-T} \begin{bmatrix} \partial N_i / \partial \xi \\ \partial N_i / \partial \eta \end{bmatrix}, \quad J = \begin{bmatrix} x_2-x_1 & x_3-x_1 \\ y_2-y_1 & y_3-y_1 \end{bmatrix}
$$

单元刚度矩阵：

$$
K_e = \sum_{q} w_q |\det J| \, \kappa_T \, (\nabla N) (\nabla N)^T
$$

### 5.4 特殊函数验证

- **对数 Gamma**：Stirling 渐近展开至 $1/y^9$ 阶
  $$
  \ln\Gamma(y) \approx \left(y-\frac{1}{2}\right)\ln y - y + \frac{1}{2}\ln(2\pi) + \frac{1}{12y} - \frac{1}{360y^3} + \cdots
  $$
  代码中 `log_gamma(5.0)` 输出 `3.178054`，与理论值 $\ln 24$ 完全一致。

- **不完全 Beta 比率**：级数展开 + 对称变换
  $$
  I_x(p,q) = \frac{x^p (1-x)^q}{p B(p,q)} \sum_{k=0}^{\infty} \frac{(p+q)_k}{(p+1)_k} x^k
  $$

- **Gauss-Legendre 求积**：Bogaert 渐近法（Bessel 零点 + Chebyshev 修正），$N>100$ 时免迭代。

### 5.5 温度依赖物性样条

在 `numerical_toolkit.py` 中，任意物性 $P(T)$ 通过三次样条插值：

$$
S_i(T) = a_i + b_i (T-T_i) + c_i (T-T_i)^2 + d_i (T-T_i)^3
$$

边界条件为自然样条（$S'' = 0$ 在端点），用于 $D_s(T)$、$\kappa(T)$、$D_e(T)$ 等。

---

## 6. 边界处理与数值鲁棒性

1. **浓度截断**：所有固相/液相浓度严格截断至 $[0, C_{\max}]$，防止物性计算中出现负数或零
2. **温度截断**：热求解中温度截断至 $[T_{\text{ambient}}, 400]$ K，防止数值溢出
3. **零值保护**：所有除法运算均添加 $10^{-18}$ 量级的正则化（如 `D_s + 1e-18`）
4. **矩阵奇异 fallback**：`BandedMatrix` 在 PLU 分解失败时自动回退到 NumPy 稠密求解
5. **Beta 函数定义域检查**：`incomplete_beta_ratio` 对 $x \notin [0,1]$ 或 $p,q \le 0$ 返回 `nan`
6. **Muller 根求解**：自动处理判别式为负（截断至 0）和除零保护

---

## 7. 合成改造说明

### 7.1 新增内容

- **物理模型层**：完整的 DFN 方程组、Butler-Volmer 动力学、Arrhenius 温度依赖、Bruggeman 有效扩散、熵热模型
- **FEM 层**：从 MATLAB `fem2d_pack` 和 `fem2d_project_function` 移植并扩展为 2D 瞬态热求解器，含 Crank-Nicolson 时间离散
- **几何层**：从 `t_puzzle_gui` 的复数几何和 `mesh2d` 的点在多边形测试，构建电池专用几何引擎
- **统计层**：从 `asa113` 的 transfer/swap 聚类和 `toms179` 的特殊函数，构建 PSD 分析与置信区间框架
- **优化层**：从 `tsp_brute` 的排列枚举，构建充电协议组合优化
- **随机层**：从 `feynman_kac_3d` 的布朗运动模拟，构建锂离子颗粒内传输与首达时间分析

### 7.2 删除内容

- 所有可视化代码（`plot3`、`surf`、`fill`、`uicontrol` 等）已全部删除
- `t_puzzle_gui` 的交互式拖拽、旋转、翻转等游戏逻辑已删除，仅保留复数几何与点在多边形测试核心
- `ball_and_stick_display` 的动画与 PNG 导出已删除，仅保留有限差分 stencil 设计思想

---

## 8. 结论

本项目成功将 15 个独立科研代码项目融合为一个面向**锂电池电化学热耦合**的博士级综合计算平台。代码具备：

- 完整的物理-数学模型体系（DFN + 2D 热 FEM + 随机传输 + 优化）
- 高阶数值方法（PLU、Crank-Nicolson、Muller、FFT、样条、Monte Carlo）
- 严格的边界处理与数值鲁棒性
- 零参数可运行的统一入口 `main.py`
- 详细的中文科学文档

所有核心模块均通过 `main.py` 实际运行验证，无报错。
