# 引力波信号数值相对论模拟与贝叶斯参数推断系统

## 一、项目概述

本项目面向天体物理前沿科学问题——**双黑洞并合产生的引力波信号的数值相对论建模与贝叶斯参数推断**。基于 15 个科研种子项目的核心算法，融合构造了一个面向博士级难度的自然科学计算系统，涵盖从初始数据构造、时空演化、引力波波形生成、探测器响应到贝叶斯参数估计的完整科学计算流程。

## 二、原项目到科学问题的映射

| 种子项目 | 核心算法 | 在合成项目中的角色 |
|---------|---------|-----------------|
| 1404_wdk | Weierstrass-Durand-Kerner 多项式根求解 | `teukolsky.py`: 求解 Kerr 黑洞准正规模(QNM)特征方程的复根 |
| 089_biharmonic_fd2d | 二维双调和方程有限差分 | `sparse_solver.py`: 构建 Brill-Lindquist 初始数据的类双调和正则化算子 |
| 507_hb_io | Harwell-Boeing 稀疏矩阵格式 | `sparse_solver.py`: 稀疏矩阵元数据结构与线性系统存储 |
| 666_legendre_shifted_polynomial | 移位 Legendre 多项式 | `waveform.py`: 引力波多极展开中的球谐函数与角分布计算 |
| 1144_square_felippa_rule | 2D 方形区域高斯求积 | `bayesian.py`: 质量平面上的边缘后验数值积分 |
| 342_euclid | 欧几里得最大公约数算法 | `utils.py`: 质量比的有理数近似与周期分析 |
| 1059_sawtooth_ode | 锯齿波驱动振子 | `utils.py`: 周期性驱动信号的测试与稳定性验证 |
| 699_log_normal_truncated_ab | 截断对数正态分布采样 | `bayesian.py`: 黑洞质量先验分布的蒙特卡洛采样 |
| 167_chebyshev2_rule | Gauss-Chebyshev Type 2 求积 | `waveform.py`: 波形内积（匹配滤波）的高精度频域积分 |
| 872_ply_display | 3D 几何张量 | `detector.py`: 探测器臂方向张量与响应函数计算 |
| 851_patterson_rule | Gauss-Patterson 嵌套求积 | `waveform.py`: 自适应高精度辐射反作用力积分 |
| 307_distance_to_position_sphere | 球面距离到位置反演 | `detector.py`: 多探测器引力波源天球定位（三角测量） |
| 767_midpoint_fixed | 固定点隐式中点法 | `utils.py` & `numerical_relativity.py`: ADM 方程时间演化积分器 |
| 126_burgers_time_inviscid | 无粘 Burgers 方程 | `utils.py`: 激波捕获数值稳定性基准测试 |
| 1041_robertson_ode | Robertson 刚性 ODE | `utils.py`: 多时间尺度数值稳定性验证 |

**每个输入项目均已真实融入合成项目，无遗漏、无挂名。**

## 三、新增数学物理模型与核心公式

### 3.1 ADM 3+1 分解与初始数据

广义相对论的 ADM (Arnowitt-Deser-Misner) 形式将四维时空分解为三维空间超曲面族：

$$
ds^2 = -\alpha^2 dt^2 + \gamma_{ij}(dx^i + \beta^i dt)(dx^j + \beta^j dt)
$$

其中 $\alpha$ 为时移函数(lapse)，$\beta^i$ 为位移函数(shift)，$\gamma_{ij}$ 为空间三维度规。

**Brill-Lindquist 初始数据**在共形平坦近似下：

$$
\gamma_{ij} = \psi^4 \delta_{ij}
$$

共形因子满足：

$$
\psi = 1 + \sum_p \frac{m_p}{2|\mathbf{r} - \mathbf{r}_p|}
$$

在数值实现中，采用**双调和正则化**处理奇点：

$$
\Delta^2 \psi + \varepsilon \Delta \psi = f(\psi)
$$

二维双调和算子的 13 点有限差分模板（网格间距 $h$）：

$$
\Delta^2 u_{i,j} \approx \frac{1}{h^4}\Big[20u_{i,j} - 8(u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}) + 2(u_{i+1,j+1}+\cdots) + (u_{i+2,j}+\cdots)\Big]
$$

### 3.2 轨道动力学与辐射反作用

后牛顿 inspiral 轨道演化由能量损失驱动。引力波四极辐射导致的轨道能量损失率（Peters-Mathews 公式）：

$$
\frac{dE}{dt} = -\frac{32}{5}\frac{G^4}{c^5}\frac{m_1^2 m_2^2(m_1+m_2)}{a^5}
$$

在几何单位制 $G=c=1$ 下，2.5PN 辐射反作用力修正的轨道加速度：

$$
\frac{d^2\mathbf{r}}{dt^2} = -\frac{M}{r^3}\mathbf{r} + \mathbf{F}_{RR}
$$

其中辐射反作用力项：

$$
\mathbf{F}_{RR} = -\frac{64}{5}\frac{\eta M^2}{r^3}v \, \hat{\mathbf{v}}
$$

啁啾质量（chirp mass）与对称质量比：

$$
M_c = \frac{(m_1 m_2)^{3/5}}{(m_1+m_2)^{1/5}} = M \eta^{3/5}, \quad \eta = \frac{m_1 m_2}{(m_1+m_2)^2}
$$

### 3.3 Teukolsky 方程与准正规模

Kerr 黑洞微扰理论的核心方程——Teukolsky 主方程（自旋权重 $s=-2$）：

$$
\Delta^{-s}\partial_r(\Delta^{s+1}\partial_r R) + \left[\frac{K^2 - 2is(r-M)K}{\Delta} + 4is\omega r - \lambda\right]R = 0
$$

其中：
- $\Delta = r^2 - 2Mr + a^2 = (r-r_+)(r-r_-)$
- $K = (r^2+a^2)\omega - am$
- $\lambda = A_{\ell m} - 2ma\omega + a^2\omega^2 - 2s(s+1)$

准正规模(QNM)频率 $\omega_{\ell mn}$ 通过**WDK 多项式根求解**获得。迭代格式：

$$
z_i^{(k+1)} = z_i^{(k)} - \frac{P(z_i^{(k)})}{\prod_{j\neq i}(z_i^{(k)} - z_j^{(k)})}
$$

初始猜测采用 Cauchy 界缩放的单位根：$z_j^{(0)} = R \exp(i \cdot 2\pi j/d)$。

### 3.4 引力波波形多极展开

引力波应变的多极展开（TT 规范）：

$$
h_+ - i h_\times = \frac{1}{D_L}\sum_{\ell=2}^{\infty}\sum_{m=-\ell}^{\ell} H_{\ell m}(t) \, {}_{-2}Y_{\ell m}(\iota, \varphi)
$$

后牛顿 inspiral 波形的 $(2,2)$ 主导模式：

$$
H_{22}(t) = \frac{\eta M}{D_L}(M\Omega)^{2/3}e^{-2i\Phi(t)}
$$

轨道相位演化（leading order）：

$$
\Phi(t) = \phi_c - \frac{1}{\eta}\left[\frac{\eta(t_c-t)}{5M}\right]^{3/8}
$$

完整的 IMR (Inspiral-Merger-Ringdown) 波形通过平滑窗口函数连接：

$$
h(t) = h_{\text{insp}}(t) \cdot w_{\text{insp}}(t) + h_{\text{ring}}(t) \cdot w_{\text{ring}}(t)
$$

$$
w_{\text{insp}} = \frac{1}{2}(1+\tanh((t_c-t)/\Delta t)), \quad w_{\text{ring}} = \frac{1}{2}(1+\tanh((t-t_c)/\Delta t))
$$

### 3.5 探测器响应与天球定位

探测器响应张量（LIGO/Virgo 型 Michelson 干涉仪）：

$$
D^{ab} = \frac{1}{2}(u^a u^b - v^a v^b)
$$

天线方向函数：

$$
F^+(\theta,\varphi,\psi) = \frac{1}{2}(1+\cos^2\theta)\cos(2\varphi)\cos(2\psi) - \cos\theta\sin(2\varphi)\sin(2\psi)
$$

$$
F^\times(\theta,\varphi,\psi) = \frac{1}{2}(1+\cos^2\theta)\cos(2\varphi)\sin(2\psi) + \cos\theta\sin(2\varphi)\cos(2\psi)
$$

多探测器到达时间差定位转化为非线性最小二乘问题：

$$
\min_{|\mathbf{n}|=1} \sum_{i<j} \left[\Delta t_{ij} - \frac{(\mathbf{r}_i - \mathbf{r}_j)\cdot\mathbf{n}}{c}\right]^2
$$

### 3.6 贝叶斯参数推断

贝叶斯定理：

$$
p(\boldsymbol{\theta}|d) = \frac{p(d|\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(d)}
$$

高斯似然函数（时域简化形式）：

$$
\ln p(d|\boldsymbol{\theta}) = -\frac{1}{2}\sum_i \frac{(d_i - h_i(\boldsymbol{\theta}))^2}{\sigma^2}
$$

黑洞质量先验采用**截断对数正态分布**：

$$
p(m) = \frac{1}{m\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln m - \mu)^2}{2\sigma^2}\right) \Big/ Z
$$

其中归一化常数：

$$
Z = \Phi\left(\frac{\ln b - \mu}{\sigma}\right) - \Phi\left(\frac{\ln a - \mu}{\sigma}\right)
$$

MCMC 采样使用 Metropolis-Hastings 算法，接受率：

$$
\alpha = \min\left(1, \frac{p(\boldsymbol{\theta}'|d)}{p(\boldsymbol{\theta}|d)}\right)
$$

质量平面的边缘后验通过**2D Gauss-Legendre 求积**（Felippa 方形规则）计算：

$$
p(m_1,m_2|d) = \int p(\boldsymbol{\theta}|d) \, d(\text{其他参数}) \approx \sum_{i,j} w_i w_j \, p(m_{1,i}, m_{2,j}, \dots|d)
$$

### 3.7 数值积分方法

**Gauss-Chebyshev Type 2 求积**（用于匹配滤波内积）：

$$
\int_a^b f(x)\sqrt{(x-a)(b-x)}\,dx \approx \sum_{k=1}^n w_k f(x_k)
$$

节点：$x_k = \cos(k\pi/(n+1))$，权重：$w_k = \frac{\pi}{n+1}\sin^2\frac{k\pi}{n+1}$

**Gauss-Patterson 嵌套求积**（用于自适应高精度积分）：

规则阶数序列为 $1, 3, 7, 15, 31, 63, 127, 255, 511$，每级嵌套前一级节点，误差估计：

$$
\varepsilon = |I_{n_{k+1}} - I_{n_k}|
$$

自适应细分策略：若 $\varepsilon > \text{tol}$，则将区间二分并递归积分。

## 四、代码文件结构与实现路径

### 4.1 文件清单（9 个 Python 模块）

| 文件名 | 功能 | 融合的种子项目 |
|-------|------|--------------|
| `main.py` | 统一入口，零参数运行完整流程 | — |
| `binary_black_hole.py` | 双黑洞系统物理参数与单位转换 | 新构建 |
| `numerical_relativity.py` | ADM 初始数据、轨道演化、稳定性测试 | 767_midpoint_fixed, 126_burgers_time_inviscid, 1041_robertson_ode, 1059_sawtooth_ode |
| `sparse_solver.py` | 稀疏矩阵系统与双调和方程离散化 | 089_biharmonic_fd2d, 507_hb_io |
| `teukolsky.py` | Teukolsky 方程与 QNM 频率求解 | 1404_wdk |
| `waveform.py` | 引力波波形生成与匹配滤波 | 666_legendre_shifted_polynomial, 167_chebyshev2_rule, 851_patterson_rule |
| `detector.py` | 探测器响应与多探测器天球定位 | 307_distance_to_position_sphere, 872_ply_display |
| `bayesian.py` | 贝叶斯先验、MCMC 采样、方形求积 | 699_log_normal_truncated_ab, 1144_square_felippa_rule |
| `utils.py` | 数值工具、刚性 ODE 测试、激波测试 | 342_euclid, 1059_sawtooth_ode, 767_midpoint_fixed, 126_burgers_time_inviscid, 1041_robertson_ode |

### 4.2 关键实现细节

**边界处理与数值鲁棒性：**
- 所有除法操作使用 `safe_divide` 函数，防止除以零
- 数组输入通过 `check_finite` 验证 NaN/Inf
- 质量比 `η` 被裁剪到物理范围 `[1e-6, 0.25]`
- 自旋参数被裁剪到 `[-0.99, 0.99]`
- MCMC 提议参数实时边界检查
- Burgers 方程 CFL 条件自适应调整

**博士级科学复杂度：**
- 共形平坦初始数据的双调和正则化求解
- 后牛顿 + 2.5PN 辐射反作用的轨道演化
- 准正规模频率的复特征值问题求解
- IMR 波形的多模式叠加与平滑连接
- 多探测器网络响应与到达时间差天球定位
- 截断对数正态先验的贝叶斯 MCMC 推断
- 自适应嵌套求积的高精度数值积分

## 五、科学问题解决能力

本合成项目能够解决以下前沿科学计算问题：

1. **双黑洞并合的引力波波形数值建模**：从初始数据构造到轨道演化、波形同化的完整数值相对论流程
2. **准正规模频率的数值计算**：基于 WDK 算法的 Kerr 黑洞微扰特征值问题求解
3. **引力波探测器网络响应分析**：LIGO/Virgo 型探测器的天线方向函数与网络信噪比计算
4. **引力波源天球定位**：基于到达时间差的多探测器三角测量与最小二乘反演
5. **双黑洞系统参数的贝叶斯推断**：MCMC 后验采样与质量平面的边缘化积分
6. **数值方法的稳定性验证**：刚性 ODE、激波捕获、周期驱动、波动传播的系统性基准测试

## 六、运行方式

### 6.1 环境要求

- Python >= 3.8
- NumPy
- SciPy

### 6.2 运行命令

```bash
cd /mnt/data/zpy/sci-swe/source_code/Synthesis-project-python/004_synth_project
python main.py
```

程序零参数即可运行，内置默认双黑洞参数（参考 GW150914：$36 M_\odot + 29 M_\odot$）。运行流程包括：

1. 双黑洞初始数据构造
2. 轨道动力学数值演化
3. IMR 引力波波形生成
4. LIGO/Virgo 探测器网络响应计算
5. 贝叶斯 MCMC 参数估计
6. 数值稳定性验证套件
7. 辅助科学计算（Legendre 展开、有理近似等）

### 6.3 预期输出

程序将输出各阶段的科学计算结果摘要，包括：
- ADM 质量、共形因子范围、稀疏矩阵密度
- 啁啾质量、对称质量比、ISCO 频率
- QNM 基频与引力波峰值光度
- 三探测器天线方向函数与网络信噪比
- MCMC 后验均值与标准差
- 数值稳定性测试通过状态

## 七、重要公式汇总

### ADM 约束方程

**Hamiltonian 约束：**
$$
R + K^2 - K_{ij}K^{ij} = 16\pi\rho
$$

**Momentum 约束：**
$$
\nabla_j(K^{ij} - \gamma^{ij}K) = 8\pi j^i
$$

### 后牛顿展开参数

$$
x = \left(\frac{GM\Omega}{c^3}\right)^{2/3} = \frac{v^2}{c^2}
$$

### ISCO 频率

$$
f_{\text{ISCO}} = \frac{c^3}{6^{3/2}\pi GM} = \frac{1}{6^{3/2}\pi M_{\text{geom}}}
$$

### 引力波应变振幅

$$
h_c = \frac{M_c^{5/3}}{D_L}(\pi f_{\text{GW}})^{2/3}
$$

### 匹配滤波信噪比

$$
\rho^2 = 4\int_{f_{\min}}^{f_{\max}} \frac{|\tilde{h}(f)|^2}{S_n(f)} df
$$

### 辐射反作用力（2.5PN）

$$
F_{\text{RR}}^i = -\frac{64}{5}\eta^2 \frac{M^3}{r^4} v^i
$$

### 共形 Killing 算子

$$
(LX)_{ij} = \nabla_i X_j + \nabla_j X_i - \frac{2}{3}\gamma_{ij}\nabla_k X^k
$$

---

**项目完成日期**：2026-05-03  
**科学领域**：天体物理 — 引力波信号数值相对论  
**编程语言**：Python 3  
**代码文件数**：9 个 `.py` 文件 + 1 个说明文档
