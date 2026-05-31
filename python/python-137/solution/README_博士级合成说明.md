# PROJECT_137：结晶过程成核与生长动力学

## 博士级科研代码合成说明文档

---

## 一、项目概述

本项目为**化学工程**领域的前沿博士级科研代码合成项目，聚焦于**结晶过程中成核与生长动力学的多尺度建模与贝叶斯参数推断**。项目基于 15 个种子科研代码项目的核心算法，构建了一个涵盖人口平衡方程 (PBE) 求解、混沌混合建模、稀疏网格不确定性量化 (UQ) 和 DREAM MCMC 贝叶斯推断的完整计算框架。

### 核心科学问题

工业结晶器中，晶体尺寸分布 (CSD) 的演化受多种耦合机制控制：
- **成核动力学**：初级均相/非均相成核遵循经典成核理论 (CNT)，二级成核与搅拌和悬浮密度相关
- **生长动力学**：尺寸依赖生长、温度活化、扩散-表面集成联合控制、BCF 螺旋位错生长
- **程序冷却**：线性、锯齿波、最优多项式冷却曲线控制过饱和度轨迹
- **混沌混合**：搅拌流场的混沌特性导致局部过饱和度非周期涨落，显著增强有效成核率
- **参数不确定性**：动力学参数具有实验不确定性，需通过贝叶斯推断反演

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 合成后角色 | 融入文件 |
|:---|:---|:---|:---|
| `583_image_quantization` | K-Means 聚类 | CSD 连续分布离散化为有限尺寸类 | `csd_analysis.py` |
| `585_image_sample` | 坐标采样与格式化 I/O | 结构化科学数据导出 (r8vec2_write) | `data_io.py` |
| `319_dream` | DREAM MCMC 贝叶斯推断 | 结晶动力学参数的贝叶斯后验采样 | `mcmc_inference.py` |
| `631_l4lib` | Park-Miller LCG 布尔生成 | 随机成核事件的离散泊松过程模拟 | `nucleation_model.py` |
| `1409_wedge_integrals` | 楔形区域精确积分 | 多组分溶解度空间积分验证 | `simplex_sampling.py` |
| `1066_set_theory` | 整数向量格式化打印 | 尺寸类索引集管理与输出 | `data_io.py` |
| `436_flame_exact` | Lambert W 函数 | 尺寸依赖生长律的解析求解 | `special_functions.py` |
| `1059_sawtooth_ode` | 锯齿波生成与驱动振荡器 | 锯齿波程序冷却曲线 | `cooling_profile.py` |
| `448_fresnel` | Fresnel 积分 | 激光衍射粒度分析 (Fraunhofer/Fresnel) | `special_functions.py` |
| `818_normal_ode` | 高斯 PDF 与简单 ODE | PBE 初始高斯分布与矩量初始化 | `population_balance.py` |
| `1248_tetrahedron_integrals` | 四面体精确积分 | 四组分相图空间积分 | `simplex_sampling.py` |
| `685_line_nco_rule` | Newton-Cotes Open 求积 | PBE 矩方程中沿特征线的数值积分 | `population_balance.py` |
| `168_chen_ode` | Chen 混沌吸引子 | 结晶器混沌混合模型 | `chaotic_mixing.py` |
| `224_cplex_solution_read` | XML/结构化数据解析 | 优化结果与参数向量的结构化解析 | `data_io.py` |
| `1103_sparse_grid_cc` | Smolyak 稀疏网格 + CC 求积 | 高维参数不确定性量化 | `sparse_grid_uq.py` |

---

## 三、核心数学物理模型与公式

### 3.1 人口平衡方程 (Population Balance Equation)

结晶过程中晶体尺寸分布 $f(L,t)$ 的时空演化由以下 PBE 描述：

$$
\frac{\partial f(L,t)}{\partial t} + \frac{\partial [G(L,t,\sigma) \cdot f(L,t)]}{\partial L} = B(\sigma,t) \cdot \delta(L - L_0)
$$

其中：
- $f(L,t)$：晶体尺寸分布函数，单位 $\#/(\text{m}^3 \cdot \text{m})$
- $G(L,t,\sigma)$：尺寸依赖生长速率，单位 $\text{m/s}$
- $B(\sigma,t)$：总成核率，单位 $\#/(\text{m}^3 \cdot \text{s})$
- $L_0$：临界核尺寸，单位 $\text{m}$
- $\sigma = (c - c_{\text{sat}})/c_{\text{sat}}$：过饱和度

### 3.2 矩方程 (Method of Moments)

定义第 $j$ 阶矩：

$$
\mu_j(t) = \int_0^{\infty} L^j f(L,t) \, dL
$$

矩方程的封闭形式：

$$
\frac{d\mu_j}{dt} = j \int_0^{\infty} L^{j-1} G(L,\sigma) f(L,t) \, dL + B(\sigma,t) \cdot L_0^j
$$

对于尺寸依赖生长 $G(L) = G_0(1+\alpha L)^\beta$，特征线法给出：

$$
\frac{dL}{dt} = G(L), \quad \frac{df}{dt} = -f \frac{\partial G}{\partial L}
$$

### 3.3 经典成核理论 (CNT)

初级均相成核率：

$$
B_{\text{prim}} = A \exp\left(-\frac{\Delta G^*}{k_B T}\right)
$$

临界成核能：

$$
\Delta G^* = \frac{16\pi \gamma^3 v_m^2}{3(k_B T \ln S)^2}
$$

其中 $S = 1 + \sigma$ 为过饱和比，$\gamma$ 为表面能，$v_m$ 为分子体积，$k_B$ 为玻尔兹曼常数。

临界核半径：

$$
r^* = \frac{2\gamma v_m}{k_B T \ln S}
$$

### 3.4 二级成核

$$
B_{\text{sec}} = k_b \cdot \sigma^b \cdot M_T^j
$$

其中 $M_T = \rho_c k_v \mu_3$ 为悬浮密度。

### 3.5 生长动力学模型

#### (a) 温度活化幂律生长

$$
G = k_{g0} \exp\left(-\frac{E_g}{RT}\right) \sigma^g
$$

#### (b) 尺寸依赖生长 ($\Delta L$-law)

$$
G(L,\sigma,T) = k_{g0} \exp\left(-\frac{E_g}{RT}\right) \sigma^g (1 + \alpha L)^\beta
$$

#### (c) 扩散-表面集成联合控制 (Two-step model)

$$
\frac{1}{G} = \frac{1}{k_d} + \frac{1}{k_r \sigma^{g_r}}
$$

#### (d) BCF 螺旋位错生长

$$
G = A_{\text{BCF}} \exp\left(-\frac{E_{\text{act}}}{RT}\right) \sigma^2 \tanh\left(\frac{B_{\text{BCF}}}{\sigma}\right)
$$

### 3.6 溶解度与过饱和度

van't Hoff 方程：

$$
\ln c_{\text{sat}}(T) = -\frac{\Delta H_{\text{diss}}}{RT} + \frac{\Delta S_{\text{diss}}}{R}
$$

过饱和度：

$$
\sigma = \frac{c - c_{\text{sat}}(T)}{c_{\text{sat}}(T)}
$$

### 3.7 质量平衡

溶质消耗速率：

$$
\frac{dc}{dt} = -3\rho_c k_v \int_0^{\infty} L^2 G(L,\sigma) f(L,t) \, dL = -3\rho_c k_v \mu_2 G_{\text{eff}}
$$

### 3.8 混沌混合模型 (Chen 吸引子)

将 Chen 混沌系统映射为局部过饱和度波动：

$$
\begin{cases}
\dfrac{dx}{dt} = a(y - x) \\[8pt]
\dfrac{dy}{dt} = (c - a)x - xz + cy \\[8pt]
\dfrac{dz}{dt} = xy - bz
\end{cases}
$$

映射关系：$\Delta T \propto x$，$\Delta c \propto y$，局部过饱和度：

$$
\sigma_{\text{local}} = \sigma_{\text{base}} + k_c \Delta c - k_T \Delta T
$$

### 3.9 DREAM MCMC 贝叶斯推断

对数后验：

$$
\ln p(\theta | D) = \ln L(D|\theta) + \ln \pi(\theta)
$$

高斯似然：

$$
\ln L = -\frac{N}{2}\ln(2\pi\sigma_n^2) - \frac{1}{2\sigma_n^2}\sum_{i=1}^N (D_i - M_i(\theta))^2
$$

DE 提议分布：

$$
z_p = z_{\text{current}} + (1+\eta) \cdot \gamma \cdot \sum_{\text{pairs}} (z_a - z_b) + \varepsilon
$$

Metropolis-Hastings 接受准则：

$$
\alpha = \min\left(1, \exp\left[\ln p(\theta_p|D) - \ln p(\theta_{\text{old}}|D)\right]\right)
$$

Gelman-Rubin 收敛诊断：

$$
\hat{R} = \sqrt{\frac{\frac{n-1}{n}W + \frac{1}{n}B}{W}}
$$

### 3.10 稀疏网格 Smolyak 不确定性量化

Smolyak 构造：

$$
Q_L^{(d)} f = \sum_{|\ell|_1 \leq L} (-1)^{L - |\ell|_1} \binom{d-1}{L - |\ell|_1} \bigotimes_{i=1}^d Q_{\ell_i}^{(1)} f
$$

其中 $Q_{\ell}^{(1)}$ 为 1D Clenshaw-Curtis 求积规则，节点为 $x_i = \cos\frac{(i-1)\pi}{n-1}$，权重由余弦级数计算。

### 3.11 Lambert W 解析解

对于尺寸依赖生长律 $G(L) = G_0 L / (1 + \alpha L)$，特征线方程的隐式解可用 Lambert W 函数显式写出：

$$
L(t) = \frac{1}{\alpha} W\left(\alpha L_0 \exp(\alpha L_0 + G_0 t)\right)
$$

### 3.12 Fresnel 衍射粒度分析

Fresnel 积分：

$$
C(x) = \int_0^x \cos\left(\frac{\pi t^2}{2}\right) dt, \quad S(x) = \int_0^x \sin\left(\frac{\pi t^2}{2}\right) dt
$$

Fraunhofer 远场衍射光强：

$$
I(\theta) \propto \left[\frac{2J_1(ka\sin\theta)}{ka\sin\theta}\right]^2
$$

其中 $k = 2\pi/\lambda$ 为波数，$a$ 为颗粒半径，$J_1$ 为一阶贝塞尔函数。

### 3.13 单纯形积分与 Dirichlet 采样

$d$ 维标准单纯形上的均匀采样：

$$
x_i = \frac{E_i}{\sum_{j=1}^{d+1} E_j}, \quad E_i \sim \text{Exp}(1)
$$

单位四面体精确单项式积分：

$$
\int_T x^{e_1} y^{e_2} z^{e_3} \, dV = \frac{e_1! \, e_2! \, e_3!}{(e_1 + e_2 + e_3 + 3)!}
$$

---

## 四、项目文件结构

```
137_synth_project/
├── main.py                    # 统一入口，零参数运行
├── special_functions.py       # Lambert W、Fresnel 积分、衍射计算
├── simplex_sampling.py        # 单纯形采样、Dirichlet 分布、精确积分
├── cooling_profile.py         # 冷却曲线、溶解度、过饱和度计算
├── chaotic_mixing.py          # Chen 混沌吸引子、混合增强成核
├── nucleation_model.py        # CNT 成核、二级成核、随机事件模拟
├── growth_kinetics.py         # 幂律/尺寸依赖/两步/BCF 生长模型
├── population_balance.py      # PBE 矩方法求解器、NCO 求积
├── csd_analysis.py            # K-Means 离散化、衍射反演、统计矩
├── sparse_grid_uq.py          # Smolyak 稀疏网格、Clenshaw-Curtis UQ
├── mcmc_inference.py          # DREAM MCMC、Gelman-Rubin 诊断
├── data_io.py                 # 结构化数据 I/O、格式化输出
└── README_博士级合成说明.md    # 本文档
```

---

## 五、运行说明

### 环境要求
- Python >= 3.9
- NumPy
- SciPy

### 运行方式
```bash
cd 137_synth_project
python main.py
```

程序将自动执行以下 10 个计算模块：
1. 特殊函数验证 (Lambert W、Fresnel、Fraunhofer)
2. 程序冷却曲线与过饱和度分析
3. 混沌混合与过饱和度涨落模拟
4. 成核与生长动力学模型比较
5. 人口平衡方程矩方法求解
6. CSD 分析与 K-Means 离散化
7. 多组分溶解度空间积分
8. 稀疏网格不确定性量化
9. DREAM MCMC 贝叶斯参数推断
10. 数据输入输出与结果管理

---

## 六、算法复杂度与数值鲁棒性

### 6.1 复杂度分析

| 模块 | 时间复杂度 | 空间复杂度 | 关键瓶颈 |
|:---|:---|:---|:---|
| PBE 矩求解 | $O(N_t \cdot N_{\text{fev}})$ | $O(N_{\text{moments}})$ | ODE 刚性步长控制 |
| 稀疏网格 UQ | $O(N_{\text{points}} \cdot C_f)$ | $O(N_{\text{points}})$ | 维数灾难的缓解：$N \sim 2^L L^{d-1}/(d-1)!$ |
| DREAM MCMC | $O(N_{\text{chains}} \cdot N_{\text{gen}} \cdot C_f)$ | $O(N_{\text{chains}} \cdot N_{\text{gen}} \cdot d)$ | 前向模型评估次数 |
| K-Means 离散化 | $O(K \cdot N_{\text{samples}} \cdot I_{\text{max}})$ | $O(N_{\text{samples}})$ | 迭代收敛 |

### 6.2 边界处理与数值鲁棒性

- **过饱和度截断**：$\sigma < 0$ 时强制为 0，防止非物理的溶解成核
- **温度正定性**：$T \leq 0$ 时替换为 $10^{-6}$ K，避免 van't Hoff 方程发散
- **对数参数空间**：MCMC 在对数尺度上采样，保证参数严格为正
- **边界反射**： proposals 超出边界时进行反射处理
- **零值保护**：所有除法运算均检查分母是否接近零
- **Gelman-Rubin 诊断**：自动检测链的收敛性
- **空簇处理**：K-Means 中遇到空簇时重新随机初始化

---

## 七、合成改造路径总结

### 改造原则
1. **每个种子项目均承担真实科学计算角色**，无遗漏、无挂名
2. **删除所有可视化代码**，仅保留数值计算与数据输出
3. **统一 Python 语言实现**，所有 MATLAB 原代码已重写为 Python
4. **大量注入科学公式**，确保公式-算法-代码三者一致
5. **工程代码具备边界处理和数值鲁棒性**

### 核心创新点
- 首次将 **Chen 混沌吸引子** 映射为结晶器局部过饱和度波动模型
- 将 **DREAM MCMC** 与 **矩方法 PBE 求解器** 耦合，实现动力学参数的贝叶斯反演
- 将 **Smolyak 稀疏网格** 应用于结晶动力学的高维不确定性量化
- 将 **Lambert W 函数** 引入尺寸依赖生长律的解析求解
- 将 **Fresnel 衍射理论** 与 **K-Means 聚类** 结合用于 CSD 的实验反演与离散化

---

## 八、参考文献与理论来源

1. Randolph, A.D. & Larson, M.A. (1988). *Theory of Particulate Processes*, 2nd Ed. Academic Press.
2. Kashchiev, D. (2000). *Nucleation: Basic Theory with Applications*. Butterworth-Heinemann.
3. Vetter, T. et al. (2013). "Modeling nucleation and crystal growth kinetics." *Crystal Growth & Design*.
4. ter Braak, C.J.F. (2006). "A Markov Chain Monte Carlo version of the Metropolis-Hastings algorithm." *Statistics and Computing*.
5. Gerstner, T. & Griebel, M. (1998). "Numerical integration using sparse grids." *Numerical Algorithms*.
6. Chen, G. & Ueta, T. (1999). "Yet another chaotic attractor." *International Journal of Bifurcation and Chaos*.
7. Corless, R.M. et al. (1996). "On the Lambert W function." *Advances in Computational Mathematics*.

---

*本项目为自动化科研代码合成产物，面向化学工程结晶动力学的前沿科学计算问题。*
