# 暗物质直接探测信号模拟系统 —— 博士级合成说明

## 1. 项目概述

本项目是一个面向**粒子物理：暗物质直接探测信号模拟**的综合性科研计算框架。基于 15 个种子科研代码项目的核心算法，融合构建了一个涵盖探测器物理、WIMP（弱相互作用大质量粒子）散射理论、蒙特卡洛事件模拟、信号处理、年度调制分析、事件重建与统计推断的完整模拟链。

### 1.1 科学问题

暗物质占宇宙物质总量的约 27%，但其本质仍是粒子物理学中最大的未解之谜之一。直接探测实验通过测量 WIMP 与靶核的弹性散射产生的微弱核反冲信号来寻找暗物质。本项目的核心科学问题是：

> **在存在显著背景噪声的条件下，如何从低温固体探测器（如锗探测器）的实验数据中，利用年度调制效应和多元判别分析提取 WIMP 散射信号，并对 WIMP-核子散射截面给出严格的统计上限？**

该问题涉及核物理、天体物理、统计学和数值计算等多个前沿交叉领域，计算难度达到博士级水平。

---

## 2. 原项目到科学问题的映射

| 序号 | 原项目 | 核心算法/思想 | 在合成项目中的角色 |
|:---:|:---|:---|:---|
| 1 | `398_fem1d_sample` | 1D FEM 函数求值与插值 | `detector_field.py` 中用于电势插值到任意查询点 |
| 2 | `691_lissajous` | 参数化振荡曲线 | `annual_modulation.py` 中将调制信号表示为 Lissajous 参数曲线，用于多探测器相位分析 |
| 3 | `001_aberth` | Aberth-Ehrlich 复多项式同时求根 | `signal_formation.py` 中分析 CR-RCⁿ 成形器传递函数的复平面极点 |
| 4 | `189_clock_solitaire_simulation` | 蒙特卡洛状态机模拟 | `monte_carlo_generator.py` 中 WIMP 事件的随机状态采样引擎 |
| 5 | `154_chain_letter_tree` | 层次聚类与距离矩阵 | `event_reconstruction.py` 中事件层次聚类与背景甄别 |
| 6 | `1303_triangle_fekete_rule` | 三角形 Fekete 高斯求积 | `detector_geometry.py` 中探测器表面数值积分 |
| 7 | `447_freefem_msh_io` | FreeFEM++ 网格 I/O | `detector_geometry.py` 中网格读写与格式转换 |
| 8 | `772_mm_to_st` | 稀疏矩阵格式转换 | `sparse_matrix_utils.py` 中 COO/CSR/MM/ST 格式互转 |
| 9 | `827_ode_euler_system` | 显式 Euler ODE 系统求解 | `particle_transport.py` 中电子漂移轨迹积分 |
| 10 | `925_pwl_approx_1d` | 分段线性近似与插值 | `signal_formation.py` 中探测器响应函数 PWL 拟合 |
| 11 | `226_craps_simulation` | 概率分支蒙特卡洛 | `monte_carlo_generator.py` 中多源背景事件的概率分支产生 |
| 12 | `519_hermite_exactness` | Gauss-Hermite 求积 | `utils.py` 中速度分布矩计算的核心数值积分工具 |
| 13 | `471_glomin` | Brent 全局最小化 | `statistical_analysis.py` 中似然函数全局优化 |
| 14 | `472_glycolysis_ode` | 耦合 ODE 系统（Sel'kov 模型） | `particle_transport.py` 中闪烁光-电离电子耦合脉冲 ODE |
| 15 | `968_r85` | 五对角矩阵求解 | `detector_field.py` 中扩散方程离散化后的 R85 格式直接求解 |

---

## 3. 核心数学物理模型与公式

### 3.1 WIMP-核子弹性散射微分率

微分事件率的标准公式（Lewin & Smith, 1996）：

$$
\frac{dR}{dE_R} = \frac{\rho_0 \sigma_0 A^2 F^2(E_R)}{2 m_\chi \mu_{\chi N}^2} \cdot \eta(v_{\min})
$$

其中：
- $\rho_0 = 0.3\,\text{GeV/cm}^3$：本地暗物质密度
- $\sigma_0$：WIMP-核子散射截面
- $A$：靶核质量数
- $F^2(E_R)$：Helm 核形状因子
- $m_\chi$：WIMP 质量
- $\mu_{\chi N} = \frac{m_\chi m_N}{m_\chi + m_N}$：约化质量
- $\eta(v_{\min}) = \int_{v_{\min}}^{\infty} \frac{f(v)}{v}\,dv$：速度积分

### 3.2 Helm 核形状因子

$$
F^2(q) = \left[ \frac{3 j_1(q R_n)}{q R_n} \right]^2 \exp\left[ -(qs)^2 \right]
$$

其中 $j_1(x) = \frac{\sin x}{x^2} - \frac{\cos x}{x}$ 为一阶球贝塞尔函数，
核参数为：

$$
R_n = \sqrt{c^2 + \frac{7}{3}\pi^2 a^2 - 5s^2}, \quad
\begin{cases}
c = 1.23 A^{1/3} - 0.60\,\text{fm} \\
a = 0.52\,\text{fm} \\
s = 0.90\,\text{fm}
\end{cases}
$$

### 3.3 最小反冲速度

$$
v_{\min} = \sqrt{\frac{m_N E_R}{2 \mu_{\chi N}^2}}
$$

### 3.4 截断 Maxwell-Boltzmann 速度分布

$$
f(v) = \frac{1}{N_{\rm esc}} \cdot \frac{1}{\sqrt{\pi} v_0}
\left\{ \exp\left[ -\frac{(v + v_e)^2}{v_0^2} \right]
- \exp\left[ -\frac{v_{\rm esc}^2}{v_0^2} \right] \right\}
$$

其中 $N_{\rm esc}$ 为归一化常数，$v_0 = 220\,\text{km/s}$，$v_e = 232\,\text{km/s}$，$v_{\rm esc} = 544\,\text{km/s}$。

### 3.5 年度调制

$$
S(t) = S_0 + S_m \cos\left[ \frac{2\pi (t - t_0)}{T} \right]
$$

典型调制分数 $S_m / S_0 \approx 0.03\text{–}0.07$，峰值在 $t_0 \approx 152$ 天（6 月 2 日）。

### 3.6 Lindhard Quenching Factor

$$
Q(E_R) = \frac{k \cdot g(\epsilon)}{1 + k \cdot g(\epsilon)}
$$

其中：

$$
\epsilon = 11.5 \, Z^{-7/3} \, E_R\,[\text{keV}], \quad
k = 0.133 \, Z^{2/3} \, A^{-1/2}, \quad
g(\epsilon) = 3\epsilon^{0.15} + 0.7\epsilon^{0.6} + \epsilon
$$

### 3.7 一维泊松方程（FEM 弱形式）

$$
\int_0^L \epsilon \frac{d\phi}{dz} \frac{d\psi}{dz}\,dz = \int_0^L \rho \psi\,dz
$$

采用 P1 有限元离散，单元刚度矩阵：

$$
K^e = \frac{\epsilon_e}{h_e} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
$$

### 3.8 电子漂移 ODE

$$
\frac{d\vec{r}}{dt} = \mu_e \vec{E}(\vec{r})
$$

显式 Euler 离散：$\vec{r}_{n+1} = \vec{r}_n + h \mu_e \vec{E}(\vec{r}_n)$。

### 3.9 闪烁-电离耦合 ODE（改进 Sel'kov 模型）

$$
\begin{cases}
\displaystyle\frac{dP}{dt} = -\gamma_P P + \alpha_Q Q \cdot E_{\rm dep} \\[8pt]
\displaystyle\frac{dQ}{dt} = \beta_R - \alpha_Q Q \cdot E_{\rm dep} - \kappa_{PQ} P \cdot Q
\end{cases}
$$

### 3.10 CR-RCⁿ 成形器脉冲响应

$$
h(t) = A \left(\frac{t}{\tau_{RC}}\right)^n \exp\left(-\frac{t}{\tau_{RC}}\right)
\left[ 1 - \exp\left(-\frac{t}{\tau_{CR}}\right) \right]
$$

### 3.11 Fisher 线性判别

寻找投影方向 $\vec{w}$ 使类间散度与类内散度之比最大：

$$
J(\vec{w}) = \frac{\vec{w}^T S_B \vec{w}}{\vec{w}^T S_W \vec{w}}, \quad
\vec{w} \propto S_W^{-1} (\vec{\mu}_s - \vec{\mu}_b)
$$

### 3.12 Poisson 轮廓似然比

$$
q_\mu = -2 \ln \frac{\mathcal{L}(\mu, \hat{\hat{\theta}})}{\mathcal{L}(\hat{\mu}, \hat{\theta})}
$$

90% CL 上限对应 $q_\mu = 2.70$（1 自由度大样本极限）。

### 3.13 Gauss-Hermite 求积

$$
\int_{-\infty}^{\infty} e^{-x^2} f(x)\,dx \approx \sum_{i=1}^{n} w_i f(x_i)
$$

节点与权重通过 Jacobi 矩阵特征值（Golub-Welsch 算法）计算。

### 3.14 Fekete 三角形求积

$$
\int_{\hat{T}} g(\hat{\vec{x}})\,d\hat{A} \approx \sum_i w_i g(\hat{\vec{x}}_i)
$$

参考三角形 $\hat{T}$ 的顶点为 $(0,0), (1,0), (0,1)$，面积 $= 1/2$。

### 3.15 Aberth-Ehrlich 同时求根

迭代公式：

$$
z_i^{(k+1)} = z_i^{(k)} - \frac{p(z_i)/p'(z_i)}{1 - \frac{p(z_i)}{p'(z_i)} \sum_{j \neq i} \frac{1}{z_i - z_j}}
$$

### 3.16 五对角矩阵直接求解

R85 格式存储的 $5 \times n$ 带状矩阵，采用不选主元高斯消去：

$$
Ax = b, \quad A_{ij} = 0 \text{ if } |i-j| > 2
$$

前向消去 + 回代，时间复杂度 $O(n)$。

---

## 4. 文件结构

```
037_synth_project/
├── main.py                          # 统一入口，零参数运行
├── utils.py                         # 数学工具、物理常数、特殊函数
├── wimp_physics.py                  # WIMP 散射物理、微分率、形状因子
├── detector_geometry.py             # 探测器几何、三角网格、Fekete 求积
├── detector_field.py                # 1D FEM 电场求解、R85 五对角求解器
├── particle_transport.py            # 电子漂移 ODE、闪烁脉冲 ODE、QF 模型
├── signal_formation.py              # 信号成形、Aberth 极点分析、PWL 近似
├── monte_carlo_generator.py         # MC 事件产生、探测效率、能量分辨率
├── annual_modulation.py             # 年度调制曲线、Lissajous 分析、显著性
├── event_reconstruction.py          # Fisher 判别、层次聚类、背景抑制评估
├── statistical_analysis.py          # 轮廓似然、CL 上限、灵敏度曲线、全局优化
├── sparse_matrix_utils.py           # 稀疏矩阵 COO/CSR/MM/ST 格式转换
└── README_博士级合成说明.md          # 本文档
```

共 **12 个 .py 文件**，满足项目要求。

---

## 5. 运行方式

```bash
cd 037_synth_project
python main.py
```

无需任何命令行参数。程序将自动执行完整的模拟链，并在终端输出各阶段的物理量和统计结果。

### 5.1 运行时间

典型运行时间约 **5–15 秒**（取决于 CPU 性能）。

### 5.2 输出说明

运行后会依次输出：
1. 探测器与物理参数配置
2. 三角网格几何信息
3. FEM 电场分布
4. WIMP 微分散射率与 Helm 形状因子
5. 蒙特卡洛生成的事件统计
6. 电子漂移与闪烁脉冲模拟
7. 信号成形与极点分析
8. 年度调制拟合结果
9. Fisher 判别与背景抑制
10. 轮廓似然上限与灵敏度曲线
11. 稀疏矩阵性能测试

---

## 6. 工程鲁棒性设计

### 6.1 边界处理

- **能量边界**：`differential_rate` 中 $E_R \leq 0$ 时返回 0；`helm_form_factor` 中 $q \to 0$ 时采用泰勒展开避免除零。
- **速度边界**：`velocity_distribution_mb` 中 $v > v_{\rm esc} + v_e$ 时强制置零。
- **ODE 非负约束**：`ScintillationODESystem.solve_euler` 中每步强制 $P, Q \geq 0$。
- **矩阵奇异处理**：`r85_np_fs` 中主元过小时替换为极小值，防止除零崩溃。
- **数组越界**：`r8vec_bracket4` 对查询点超出范围的情况返回安全索引。

### 6.2 数值稳定性

- Gauss-Hermite 节点通过对称三对角矩阵特征值法计算，数值稳定。
- Aberth-Ehrlich 求根使用 Cauchy 半径初始化，收敛可靠。
- 扩散方程 R85 求解器引入正则化项保证可解性。
- Fisher 判别中类内散度矩阵 $S_W$ 加入 $10^{-6} I$ 避免奇异。

### 6.3 可复现性

- 所有随机过程使用 `ReproducibleRNG`（Park-Miller LCG），种子固定为 `20240503`。
- NumPy 的 `np.random` 仅在噪声添加和聚类测试中使用，不影响主流程。

---

## 7. 科学前沿性

本项目融合的前沿科学计算内容：

1. **核物理**：Helm 形状因子的动量依赖、Lindhard 淬灭因子
2. **天体物理**：截断 Maxwell-Boltzmann 速度分布、年度调制效应
3. **数值方法**：FEM 泊松求解、Gauss-Hermite 求积、Fekete 三角形积分
4. **信号处理**：CR-RCⁿ 成形器、传递函数极点分析、PWL 响应近似
5. **统计推断**：轮廓似然比、CLₛ 方法、灵敏度曲线
6. **机器学习辅助**：Fisher 线性判别、层次聚类背景甄别
7. **高性能计算**：稀疏矩阵 CSR 格式、五对角直接求解器

---

## 8. 参考文献

1. Lewin, J. D., & Smith, P. F. (1996). Astroparticle Physics, 6, 87.
2. Helm, R. H. (1956). Phys. Rev. 104, 1466.
3. Lindhard, J., et al. (1963). Mat. Fys. Medd. Dan. Vid. Selsk. 33, 1.
4. Drukier, A. K., Freese, K., & Spergel, D. N. (1986). Phys. Rev. D 33, 3495.
5. Brent, R. P. (1973). Algorithms for Minimization Without Derivatives.
6. Taylor, M. A., Wingate, B. A., & Vincent, R. E. (2000). SIAM J. Numer. Anal. 38, 1707.
7. Cheney, W., & Kincaid, D. (1985). Numerical Mathematics and Computing.
8. Aberth, O. (1973). Math. Comp. 27, 339.
9. Cowan, G., et al. (2011). Eur. Phys. J. C 71, 1554.
10. Bernabei, R., et al. (2000). Phys. Lett. B 480, 23.

---

*文档生成时间：2025 年 5 月*
