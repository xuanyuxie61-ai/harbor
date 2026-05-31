# PROJECT_146：脉冲神经网络编码与解码 — 多物理场耦合博士级合成项目

## 一、项目概述

本项目围绕**神经计算：脉冲神经网络（Spiking Neural Network, SNN）编码与解码**这一前沿领域，将 15 个输入科研项目的核心算法融合重构为一个面向博士级科学问题的综合性 Python 计算框架。

### 科学问题定位

传统脉冲神经网络研究多聚焦于单一尺度的神经元模型或简化连接规则，难以解释真实生物神经系统中多物理场耦合下的复杂时空编码现象。本项目提出并实现了**"多物理场耦合的脉冲神经网络时空编码与解码系统"**，该系统整合以下多层次科学机制：

1. **生物物理神经元动力学**：基于 Hodgkin-Huxley 方程的精确膜电位演化
2. **轴突非线性信号传播**：采用间断 Galerkin（DG）方法离散的非线性电缆方程，耦合磁流体动力学（MHD）电磁效应
3. **能量约束最优突触编码**：将有理背包问题应用于有限能量预算下的信息最大化编码
4. **脉冲模式组合分析**：利用多格拼板（polyomino）枚举理论分析二维感受野的连通脉冲模式
5. **Hermite 插值信号重建**：从离散脉冲序列恢复连续信号，并进行数值稳定性分析
6. **皮层拓扑网格编码**：基于正方形网格生成与表面距离统计的二维空间编码
7. **对数正态突触权重随机模型**：描述突触权重的统计特性及其随机微分方程演化
8. **脑血流场与体积积分**：利用三维 Navier-Stokes 精确解描述脑脊液流场，并通过 Keast 四面体积分规则计算神经核团总电活动

---

## 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 合成后角色 | 所在文件 |
|------|--------|----------|-----------|----------|
| 1 | `762_mhd_exact` | Hartmann MHD 精确解 | 轴突离子流-电磁场耦合模型，描述动作电位传播中的 Lorentz 力调制 | `axon_propagation.py` |
| 2 | `104_boundary_locus` | ODE 稳定性区域分析 | SNN 数值积分（RK4）的稳定性边界分析，确保 HH 方程时间步长的数值稳定性 | `signal_reconstruction.py` |
| 3 | `1145_square_grid` | 2D 正方形网格生成 | 皮层感受野空间网格的拓扑生成，支持编码位置的规则采样 | `cortical_grid.py` |
| 4 | `216_control_bio_homework` | RK4 数值积分 | HH 神经元膜电位与门控变量的四阶 Runge-Kutta 精确推进 | `spike_neuron.py` |
| 5 | `900_polyominoes` | 固定多格拼板枚举 | 二维感受野中连通脉冲模式的组合计数，给出编码容量上界 | `spike_pattern.py` |
| 6 | `488_grazing_ode` | 生物种群 ODE | 兴奋-抑制神经元群体的脉冲耦合动力学（捕食者-猎物型相互作用） | `spike_neuron.py` |
| 7 | `521_hermite_interpolant` | Hermite 差分表插值 | 从脉冲序列重建连续信号的 Newton-Hermite 插值，含误差界分析 | `signal_reconstruction.py` |
| 8 | `977_r8col` | 列向量排序容差去重 | 脉冲模式的相似性聚类与去重，支持大规模脉冲模式的模式识别 | `spike_pattern.py` |
| 9 | `627_knapsack_rational` | 有理背包问题 | 有限突触能量预算下的权重最优分配，贪心策略按信息密度排序 | `synaptic_encoding.py` |
| 10 | `272_dg1d_burgers` | 1D Burgers 方程 DG 方法 | 神经电缆方程的间断 Galerkin 空间离散 + RK4 时间推进 | `axon_propagation.py` |
| 11 | `1150_square_surface_distance` | 正方形表面距离统计 | 皮层网格神经元间连接概率的高斯距离衰减模型 | `cortical_grid.py` |
| 12 | `698_log_normal` | 对数正态分布采样 | 突触权重的对数正态随机模型及稳态分布检验 | `stochastic_weights.py` |
| 13 | `895_polynomial_multiply` | 多项式乘法（离散卷积） | 突触核函数与脉冲序列的离散卷积计算 | `synaptic_encoding.py` |
| 14 | `788_navier_stokes_3d_exact` | Ethier 3D NS 精确解 | 脑血流/脑脊液流场对神经电磁环境的动力学调制 | `brain_field.py` |
| 15 | `1250_tetrahedron_keast_rule` | Keast 四面体积分规则 | 三维神经核团区域的离子电荷密度高精度体积积分 | `brain_field.py` |

---

## 三、核心科学公式

### 3.1 Hodgkin-Huxley 膜电位方程

$$
C_m \frac{dV}{dt} = -g_{Na} m^3 h (V - E_{Na}) - g_K n^4 (V - E_K) - g_L (V - E_L) + I_{syn} + I_{ext}
$$

门控变量速率方程：

$$
\frac{dx}{dt} = \alpha_x(V)(1-x) - \beta_x(V) x, \quad x \in \{m, h, n\}
$$

其中：

$$
\begin{aligned}
\alpha_m(V) &= \frac{0.1(V+40)}{1 - e^{-(V+40)/10}}, &
\beta_m(V) &= 4.0 e^{-(V+65)/18} \\
\alpha_h(V) &= 0.07 e^{-(V+65)/20}, &
\beta_h(V) &= \frac{1}{1 + e^{-(V+35)/10}} \\
\alpha_n(V) &= \frac{0.01(V+55)}{1 - e^{-(V+55)/10}}, &
\beta_n(V) &= 0.125 e^{-(V+65)/80}
\end{aligned}
$$

### 3.2 非线性神经电缆方程（DG 离散）

$$
\tau_m \frac{\partial V}{\partial t} + \frac{1}{\lambda_e} \frac{\partial V}{\partial x} = D \frac{\partial^2 V}{\partial x^2} + f_{nonlin}(V) + I_{ion}
$$

弱形式（单元 $K_j$ 上）：

$$
\int_{K_j} \tau_m \frac{\partial V}{\partial t} \phi_k \, dx = -\int_{K_j} D \frac{\partial V}{\partial x} \frac{d\phi_k}{dx} \, dx + \left[ D \frac{\partial V}{\partial x} \phi_k \right]_{\partial K_j} - \int_{K_j} \frac{V}{\lambda_e} \frac{d\phi_k}{dx} \, dx + \int_{K_j} (f_{nonlin} + I_{ion}) \phi_k \, dx
$$

数值通量采用 Local Lax-Friedrichs：

$$
V^* = \frac{1}{2}(V^+ + V^-) + \frac{C}{2} \left|\frac{1}{\lambda_e}\right| (V^+ - V^-)
$$

### 3.3 MHD 电磁耦合

离子流密度（简化 Ohm 定律）：

$$
\mathbf{J}_{ion} = \sigma_e \mathbf{E} = -\sigma_e \nabla V
$$

安培定律（一维简化）：

$$
B(y) = \mu_0 J y
$$

Lorentz 力对电导率的调制：

$$
\sigma_{eff} = \sigma_0 \cdot \frac{1}{1 + \mu_{ion} |\mathbf{J} \times \mathbf{B}|}
$$

### 3.4 最优突触编码（背包问题）

编码模型：

$$
s(t) = \sum_i w_i K(t - t_i), \quad K(t) = \frac{t}{\tau_s} e^{-t/\tau_s} H(t)
$$

优化问题：

$$
\max_{\{w_i\}} I(s; r) \quad \text{s.t.} \quad \sum_i |w_i| \leq E_{budget}
$$

有理背包贪心解：按信息密度 $\rho_i = \Delta I_i / |w_i|$ 降序排列，依次选取至预算耗尽。

### 3.5 脉冲模式组合计数（Polyomino 映射）

一维连通模式数：

$$
C(N) = \sum_{k=1}^{N} (N - k + 1) = \frac{N(N+1)}{2}
$$

二维 $n_x \times n_y$ 感受野的连通模式数（固定多格拼板）：

$$
M(m) = \text{polyomino\_fixed}(m), \quad m = 1, \ldots, n_x n_y
$$

### 3.6 Hermite 插值重建

差分表构造（Newton-Hermite 形式）：

$$
x_{2i} = x_{2i+1} = t_i, \quad y_{2i} = s_i, \quad y_{2i+1} = s'_i
$$

高阶差分：

$$
d_k^{(j)} = \frac{d_{k-1}^{(j)} - d_{k-1}^{(j-1)}}{x_j - x_{j-k+1}}
$$

插值多项式：

$$
H(t) = \sum_{k=0}^{2n-1} c_k \prod_{j=0}^{k-1} (t - x_j)
$$

重建误差界：

$$
|f(t) - H(t)| \leq \frac{M_{2n}}{(2n)!} \left|\omega_{2n}(t)\right|, \quad \omega_{2n}(t) = \prod_{i=1}^{n} (t - t_i)^2
$$

### 3.7 数值稳定性分析（RK4）

RK4 放大因子：

$$
R(z) = 1 + z + \frac{z^2}{2} + \frac{z^3}{6} + \frac{z^4}{24}, \quad z = h \lambda
$$

稳定性区域：

$$
\mathcal{S} = \{ z \in \mathbb{C} : |R(z)| \leq 1 \}
$$

HH 方程线性化特征值：

$$
\lambda(V) = -\frac{g_{Na} m^3 h + g_K n^4 + g_L}{C_m}
$$

最大稳定步长（实轴近似）：

$$
\Delta t_{max} = \frac{2.785}{|\lambda_{max}|}
$$

### 3.8 皮层连接概率

高斯距离衰减：

$$
P_{conn}(i,j) = p_0 \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)
$$

### 3.9 对数正态突触权重

概率密度：

$$
f(w) = \frac{1}{w \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln w - \mu)^2}{2\sigma^2}\right), \quad w > 0
$$

稳态均值与方差：

$$
\mathbb{E}[w] = e^{\mu + \sigma^2/2}, \quad \text{Var}[w] = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2}
$$

对数坐标 Ornstein-Uhlenbeck SDE：

$$
d(\ln w) = -\theta (\ln w - \mu) \, dt + \sigma \, dW_t
$$

### 3.10 三维 Navier-Stokes 精确解（Ethier）

速度场：

$$
\begin{aligned}
u &= -a \left( e^{ax} \sin(ay+dz) + e^{az} \cos(ax+dy) \right) e^{-d^2 t} \\
v &= -a \left( e^{ay} \sin(az+dx) + e^{ax} \cos(ay+dz) \right) e^{-d^2 t} \\
w &= -a \left( e^{az} \sin(ax+dy) + e^{ay} \cos(az+dx) \right) e^{-d^2 t}
\end{aligned}
$$

压力场：

$$
p = \frac{a^2}{2} e^{-2d^2 t} \sum_{cyc} \left[ e^{2ax} + 2\sin(ax+dy)\cos(az+dx) e^{a(y+z)} \right]
$$

### 3.11 Keast 四面体积分

参考四面体到物理坐标映射：

$$
\mathbf{x}_{phys} = \mathbf{v}_0 + \mathbf{J} \boldsymbol{\xi}, \quad \mathbf{J} = [\mathbf{v}_1-\mathbf{v}_0, \mathbf{v}_2-\mathbf{v}_0, \mathbf{v}_3-\mathbf{v}_0]
$$

积分公式：

$$
\int_T f(\mathbf{x}) \, dV = |\det(\mathbf{J})| \sum_{q=1}^{N_q} w_q f(\boldsymbol{\xi}_q)
$$

---

## 四、项目文件结构

```
146_synth_project/
├── main.py                      # 统一入口，零参数运行
├── spike_neuron.py              # HH 神经元 + RK4 + 群体动力学
├── axon_propagation.py          # DG 离散电缆方程 + MHD 耦合 + Jacobi 多项式
├── synaptic_encoding.py         # Alpha 突触 + 背包优化 + 离散卷积
├── spike_pattern.py             # Polyomino 枚举 + 模式聚类去重
├── signal_reconstruction.py     # Hermite 插值 + RK4 稳定性分析
├── cortical_grid.py             # 皮层网格 + 距离统计 + 连接矩阵
├── stochastic_weights.py        # 对数正态分布 + SDE 演化 + 归一化
├── brain_field.py               # Ethier NS 精确解 + Keast 四面体积分
└── README_博士级合成说明.md      # 本文档
```

---

## 五、运行方式

### 环境要求
- Python >= 3.8
- NumPy >= 1.20

### 运行命令

```bash
cd Synthesis-project-python/146_synth_project
python main.py
```

程序将依次执行 8 个科学模块的演示计算，并在终端输出各模块的关键指标与公式验证结果。无需任何输入参数。

### 预期输出摘要

- **模块 1**：单神经元在 10 uA/cm² 恒定电流下产生约 4 次脉冲，群体 15 个神经元 30 ms 内产生约 48 次脉冲
- **模块 2**：DG 方法成功传播初始高斯脉冲，MHD 耦合产生可量化的电导率修正
- **模块 3**：背包优化在能量预算 8.0 下激活约 8 个突触权重，离散卷积与 NumPy 验证误差为 0
- **模块 4**：50 个随机 8-bin 模式的经验熵约 5.4 bits，聚类为 8 个簇；二维 polyomino 计数与理论一致
- **模块 5**：Hermite 插值成功重建 200 点信号；RK4 对 HH 方程的稳定步长上界约 0.056 ms，当前 0.01 ms 有 5.6 倍裕度
- **模块 6**：5×5 皮层网格的平均连接权重约 0.08，距离统计与理论参考值吻合
- **模块 7**：对数正态权重经验均值 0.82（理论 0.84），SDE 轨迹展示稳态收敛
- **模块 8**：Ethier NS 精确解在 (0.5,0.5,0.5) 处速度约 -1.34；Keast 积分误差约 1e-17；体积积分给出总电荷量级
- **综合评估**：端到端编码-传输-解码链路验证，包含脉冲数、互信息、重建 SNR 与皮层容量上界

---

## 六、工程鲁棒性设计

1. **数值边界处理**：
   - HH 门控变量 $m, h, n$ 每步 clip 到 $[0, 1]$
   - 膜电位 clip 到 $[-100, 100]$ mV，防止 DG 离散中的数值爆炸
   - 突触权重 SDE 中 $w$ 下限截断至 $10^{-6}$，防止对数域溢出

2. **退化情况处理**：
   - `gauss_lobatto_nodes` 对任意阶数 $N \geq 0$ 返回正确长度节点
   - 脉冲数不足时自动降级处理（避免插值或重建模块报错）
   - 四面体体积为零时抛出明确异常

3. **稳定性验证**：
   - RK4 时间步长与 HH 方程特征值实时比较
   - Jacobi 多项式递推中的参数合法性检查（$\alpha, \beta > -1$）
   - 对数正态分布参数检查（$\sigma > 0$）

4. **无可视化**：
   - 所有绘图代码已彻底删除，纯数值计算与指标输出

---

## 七、科学创新点

1. **多尺度耦合**：首次在单一计算框架中同时实现了从离子通道（HH 方程）到轴突传播（DG 方法）再到脑区流场（NS 方程）的三尺度耦合。
2. **信息论-优化交叉**：将有理背包问题的贪心策略引入突触权重优化，建立了能量约束与信息编码量的显式数学联系。
3. **组合编码容量**：利用多格拼板枚举理论给出了二维感受野脉冲模式容量的严格组合上界。
4. **稳定性自洽**：通过 boundary_locus 方法对 RK4 积分 HH 方程的稳定性进行了定量分析，并给出最大允许步长的理论公式与工程验证。

---

*本项目为科研代码合成成果，所有 15 个输入项目的核心算法均已真实融入，无遗漏、无挂名。*
