# PROJECT 233 — 计算高能物理：顶夸克质量测量统计建模
## 高阶有限差分与稳定性分析（小规模可复现实验）

---

## 一、项目概述

本项目是一个**博士级计算高能物理 (Computational High Energy Physics) 研究平台**，
实现了在 LHC (大型强子对撞机) pp 碰撞实验中，通过 $t\bar{t}$ ne变质量谱
提取顶夸克 (top quark) pole 质量的完整统计分析流程。

**科学目标**: 从 $t\bar{t}$ ne变质量分布 $d\sigma/dM_{t\bar{t}}$ 中提取
顶夸克 pole 质量 $m_t^{\text{pole}}$，精度达到 $\delta m_t \sim 0.5$ GeV。

**实验条件**:
- 质子-质子碰撞，质心系能量 $\sqrt{s} = 13$ TeV (LHC Run 2)
- 积分光度 $\mathcal{L} = 139$ fb$^{-1}$ (ATLAS/CMS 全数据集)
- 衰变道：半轻道 $t\bar{t} \to b\ell\nu \, \bar{b}jj$

**方法论**:
- **NRQCD 阈值截面**: 非相对论 QCD 格林函数方法计算 $t\bar{t}$ 阈值产生
- **轮廓似然比 (PLR)**: 统计推断框架，profile 掉 nuisance 参数
- **高阶有限差分**: 4 阶中心差分 + Richardson 外推计算参数灵敏度
- **非线性动力学**: Duffing/Anishchenko 型拟合动力学 + Floquet 稳定性

---

## 二、种子项目到科学问题的映射 (全部 15 个)

| # | 种子项目 | 核心算法 | 在本项目中的角色 |
|---|---------|---------|----------------|
| 1 | `392_fem1d_heat_steady` | FEM 弱形式 + 三对角组装 | Schrödinger 方程 Numerov 求解 ($\S$`spectral_function.py`) |
| 2 | `623_knapsack_brute` | 二进制暴力枚举 + 子集搜索 | 喷注-部分子配对组合优化 ($\S$`knapsack_combinatoric.py`) |
| 3 | `142_cavity_flow_movie` | Navier-Stokes + Poisson 压力求解 | 探测器响应的扩散核模型 ($\S$`detector_response.py`) |
| 4 | `322_duffing_ode` | Duffing 振荡器 RK4 求解 | 拟合参数非线性动力学 ($\S$`nonlinear_ode_solvers.py`) |
| 5 | `498_hammersley` | Hammersley 准随机序列生成 | $t\bar{t}$ 相空间 QMC 采样 ($\S$`hammersley_phase_space.py`) |
| 6 | `566_hypersphere_monte_carlo` | 高维球面均匀采样 | 多维 nuisance 参数积分 ($\S$`hammersley_phase_space.py`) |
| 7 | `237_cuda_loop` | GPU 线程索引 + striping loop | 并行核响应计算 ($\S$`detector_response.py`) |
| 8 | `776_monomial_symmetrize` | 多项式置换对称化 | 不可区分 jet 的对称可观测量 ($\S$`monomial_symmetry.py`) |
| 9 | `006_anishchenko_ode` | Heaviside 开关 + 非线性 ODE | 拟合收敛模式监测 ($\S$`time_delay_stability.py`) |
| 10 | `1085_PierceRyan` | 边界碰撞分岔 + Floquet 乘子 | 迭代稳定性分析 ($\S$`time_delay_stability.py`) |
| 11 | `505_hankel_inverse` | Hankel 矩阵逆 + 结构化求解 | 动量空间 ↔ 坐标空间变换 ($\S$`hankel_momentum_transform.py`) |
| 12 | `885_polygon_grid` | 多边形扇形三角化 + 重心坐标 | 六边形参数扫描网格 ($\S$`polygon_phase_grid.py`) |
| 13 | `945_quad_trapezoid` | 复合梯形积分规则 | 似然函数数值积分 ($\S$`profile_likelihood.py`) |
| 14 | `300_disk01_integrands` | 圆盘积分 + 三角幂递推 | 角度积分 + 精确参考值 ($\S$`spectral_function.py`) |
| 15 | `1028_nereagurru` | 盘演化 ODE + 事件检测 | 阈值格林函数传播子 ($\S$`spectral_function.py`) |

---

## 三、核心物理模型与公式

### 3.1 $t\bar{t}$ 阈值产生截面 (NRQCD)

在阈值区域 $\sqrt{\hat{s}} \approx 2m_t$，顶夸克对速度 $v \sim \alpha_s \ll 1$，
需要重求库仑胶子交换 $(\alpha_s/v)^n$ 到所有阶。

**Schrödinger 方程** (S 波):
$$\left[-\frac{\nabla^2}{m_t} + V(r)\right] u(r) = m_t(E + i\Gamma_t)\, u(r)$$

**QCD 静态势** (坐标空间，包含高阶修正):
$$V(r) = -\frac{C_F\, \alpha_V(1/r)}{r}, \quad C_F = \frac{N_c^2-1}{2N_c} = \frac{4}{3}$$

**V-scheme 耦合** (微扰展开):
$$\alpha_V(q) = \alpha_s(q)\left[1 + a_1\frac{\alpha_s}{4\pi} + a_2\left(\frac{\alpha_s}{4\pi}\right)^2 + \cdots\right]$$

**Sommerfeld 增强因子** (库仑重求和):
$$S(\beta) = \frac{z}{1 - e^{-z}}, \quad z = \frac{\pi C_F \alpha_s}{\beta}$$

**格林函数虚部** (与截面成正比):
$$\text{Im}\, G(E+i\Gamma_t;\, 0, 0) \propto \sigma_{t\bar{t}}^{\text{threshold}}(\hat{s})$$

**Numerov 方法** (6 阶精度):
$$u_{n+1} = \frac{2\left(1 - \frac{5h^2 f_n}{12}\right) u_n - \left(1 + \frac{h^2 f_{n-1}}{12}\right) u_{n-1}}{1 + \frac{h^2 f_{n+1}}{12}}$$

### 3.2 强耦合常数跑动 (5-loop β 函数)

$$\frac{d\alpha_s}{d\ln\mu^2} = \beta(\alpha_s) = -\sum_{i=0}^{4} \beta_i \left(\frac{\alpha_s}{4\pi}\right)^{i+1}$$

其中 (对 $n_f=5$):
- $\beta_0 = \frac{11N_c - 2n_f}{3} = \frac{23}{3}$
- $\beta_1 = \frac{34N_c^2 - 10N_c n_f - 3\frac{N_c^2-1}{N_c}n_f}{12} = \frac{29}{3}$

### 3.3 顶夸克衰变宽度 (NLO QCD)

$$\Gamma_t = \frac{G_F m_t^3}{8\pi\sqrt{2}} |V_{tb}|^2 (1-w)^2(1+2w)\left(1 + \delta_{\text{QCD}}\right)$$

其中 $w = (m_W/m_t)^2$, QCD 修正:
$$\delta_{\text{QCD}} = -\frac{2\alpha_s(m_t)}{3\pi}\left(\frac{\pi^2}{3} - \frac{5}{4}\right) + \delta_{\text{NNLO}}$$

### 3.4 轮廓似然比 (Profile Likelihood Ratio)

**扩展似然函数**:
$$L(m_t, \boldsymbol{\theta}) = \prod_{i=1}^{N_{\text{bins}}} \text{Poisson}(n_i \mid \mu s_i(\boldsymbol{\theta}) + b_i(\boldsymbol{\theta})) \times \prod_k \pi_k(\theta_k)$$

**轮廓似然比**:
$$\lambda(m_t) = \frac{L(m_t, \hat{\hat{\boldsymbol{\theta}}}(m_t))}{L(\hat{m}_t, \hat{\boldsymbol{\theta}})}$$

**检验统计量**:
$$q(m_t) = -2\ln\lambda(m_t) \sim \chi^2_1 \quad (\text{Wilks 定理})$$

**置信区间**:
- 68% CL: $q(m_t) < 1.0$
- 95% CL: $q(m_t) < 3.84$

### 3.5 高阶有限差分

**4 阶中心差分**:
$$f'(x) = \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h} + O(h^4)$$

**6 阶中心差分**:
$$f'(x) = \frac{f(x+3h) - 9f(x+2h) + 45f(x+h) - 45f(x-h) + 9f(x-2h) - f(x-3h)}{60h} + O(h^6)$$

**Richardson 外推** (Romberg 表):
$$T[i,j] = \frac{2^p\, T[i,j-1] - T[i-1,j-1]}{2^p - 1}$$

**Fornberg 算法** (任意网格权重):
$$c_{m,n,j} = \frac{(x_n - x^*)c_{m,n-1,j} - m\, c_{m-1,n-1,j}}{x_n - x_j}$$

### 3.6 Duffing 型拟合动力学

将质量拟合过程建模为非线性动力系统:
$$m'' + \delta m' + \alpha(m - m_{\text{target}}) + \beta(m - m_{\text{target}})^3 = \gamma\cos(\omega t)$$

**Floquet 乘子**: $\mu_k = \text{eig}(J)$, $J = \partial F/\partial m$
- $|\mu| < 1$: 稳定不动点
- $|\mu| > 1$: 不稳定
- $|\mu| = 1$: 分岔点

### 3.7 Hammersley 准随机序列

**星差异界**: $D_N^* \leq C_d (\ln N)^{d-1}/N$

**Radical inverse** (base $p$):
$$\phi_p(i) = \sum_{k=0}^{\infty} a_k\, p^{-(k+1)}, \quad i = \sum_k a_k p^k$$

**Koksma-Hlawka 不等式**:
$$|Q_N(f) - I(f)| \leq V(f) \cdot D_N^*$$

### 3.8 Collins-Soper 横动量重求和

$$\frac{d\sigma}{dq_T^2} = \frac{1}{2\pi}\int_0^\infty b\, db\, J_0(q_T b)\, W(b, m_t, \mu)$$

**$b^*$ 处方**:
$$b^* = \frac{b}{\sqrt{1 + b^2/b_{\max}^2}}$$

**Sudakov 因子**:
$$W(b) = \exp\left(-S_{\text{pert}}(b^*, \mu) - S_{\text{NP}}(b)\right)$$

---

## 四、代码结构

```
233_synth_project_Advanced/
├── main.py                      # 统一入口 (零参数可运行)
├── topmass_constants.py         # 物理常数, SM 参数, PDF, α_s 跑动
├── spectral_function.py         # 谱函数: NRQCD 格林函数 + 阈值截面
├── finite_diff_operators.py     # 高阶有限差分: 4th/6th 阶 + Richardson + Fornberg
├── hammersley_phase_space.py    # Hammersley QMC 相空间采样
├── detector_response.py         # 探测器响应: Navier-Stokes 扩散核
├── polygon_phase_grid.py        # 多边形网格: 六边形参数扫描
├── hankel_momentum_transform.py # Hankel 变换 + CSS 重求和
├── monomial_symmetry.py         # 多项式对称化: 置换不变可观测量
├── time_delay_stability.py      # Floquet 稳定性 + 收敛监测
├── knapsack_combinatoric.py     # 组合优化: 喷注配对暴力枚举
├── profile_likelihood.py        # 轮廓似然: PLR 扫描 + 置信区间
├── nonlinear_ode_solvers.py     # 非线性 ODE: RK45 自适应 + Duffing 动力学
├── top_mass_pipeline.py         # 完整分析流水线 (10 步)
└── README_博士级合成说明.md      # 本文档
```

---

## 五、运行方法

```bash
cd 233_synth_project_Advanced
python main.py
```

**无需任何参数**，程序将自动执行完整的分析流水线：

1. **物理参数初始化**: 计算 $\Gamma_t$, $\alpha_s(m_t)$, Sommerfeld 增强
2. **信号模板**: NRQCD 阈值截面 + PDF 部分子光度
3. **背景模型**: 指数参数化 $W$+jets/QCD 多喷注
4. **探测器卷积**: Navier-Stokes 型扩散核展宽
5. **伪数据生成**: Hammersley QMC 采样 + Poisson 涨落
6. **轮廓似然扫描**: 在 $m_t \in [167, 178]$ GeV 扫描
7. **质量拟合**: 梯度上升迭代 + Floquet 稳定性
8. **有限差分灵敏度**: 4 阶 + Richardson 外推 + 自适应步长
9. **稳定性分析**: Duffing/Anishchenko 动力学 + Lyapunov 指数
10. **组合优化**: 喷注配对暴力枚举 + 对称可观测量

---

## 六、关键输出

程序输出包括：
- **最佳拟合质量** $\hat{m}_t$ (PLR 方法和迭代方法)
- **68%/95% 置信区间**
- **强耦合常数** $\alpha_s(m_t)$
- **顶夸克宽度** $\Gamma_t$
- **有限差分导数** $d\sigma/dm_t$ (4 阶和 Richardson)
- **Floquet 乘子** (稳定性判据)
- **Lyapunov 指数** (混沌检测)
- **Hankel 矩阵条件数** (矩方法稳定性)

---

## 七、边界处理与数值鲁棒性

1. **$\alpha_s$ 跑动**: 自动截断到物理范围 $[10^{-6}, 4\pi \times 0.9]$
2. **顶夸克宽度**: 运动学禁止 ($m_t < m_W$) 时返回极小值
3. **Numerov 求解**: 每步重归一化防止指数溢出 + NaN/Inf 保护
4. **Hankel 矩阵**: 条件数检测，超过 $10^{12}$ 使用伪逆
5. **轮廓似然**: 期望值下限保护 (`max(expected, 1e-10)`)
6. **RK45 自适应**: 20 次拒绝后强制接受 + 步长上下限
7. **组合优化**: 非法分配给予大惩罚 ($10^8$)
8. **所有物理函数**: 对零/负输入做显式边界检查

---

## 八、依赖

- `numpy` (数值计算)
- `scipy` (特殊函数: `gammaln`, `jv`, `erf`, `comb`, `chi2`)

---

## 九、科学意义

本项目展示的**方法论创新**：
1. 将**非线性动力学理论** (Duffing/Anishchenko/Floquet) 引入实验粒子物理的参数拟合稳定性分析
2. 使用**高阶有限差分 + Richardson 外推**实现参数灵敏度的机器精度计算
3. **Hammersley QMC** 相比传统 MC 将相空间积分误差从 $O(N^{-1/2})$ 降至 $O((\ln N)^5/N)$
4. **Navier-Stokes 扩散核**类比探测器分辨率，提供物理 motivated 的响应函数参数化
5. **组合优化暴力枚举**确保小事件样本中的全局最优 jet 配对

这些方法的结合使得本项目达到了**博士级计算高能物理**的研究水平。
