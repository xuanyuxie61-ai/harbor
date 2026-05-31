# 神经计算：强化学习与最优控制 — 博士级合成项目说明

## 一、项目概述

本项目围绕**神经计算：强化学习与最优控制**这一前沿科学领域，基于15个种子科研代码项目的核心算法，融合构建了一个面向博士级难度的Python科研计算项目。

### 核心科学问题

**基于随机微分方程（SDE）的神经群体最优控制与深度强化学习**

考虑由扩展Wilson-Cowan神经质量模型描述的神经元群体平均场动力学：

$$
\begin{aligned}
\tau_e \frac{dE}{dt} &= -E + S_e\big(a_{ee}E - a_{ei}I + P + u(t)\big) + \sigma_e \xi_e(t) \\
\tau_i \frac{dI}{dt} &= -I + S_i\big(a_{ie}E - a_{ii}I + Q\big) + \sigma_i \xi_i(t)
\end{aligned}
$$

其中 $S(x) = \frac{1}{1+e^{-k(x-\theta)}}$ 为S型激活函数，$u(t)$ 为外部控制输入（模拟经颅电刺激），$\xi(t)$ 为突触噪声的Wiener过程。

**控制目标**：寻找最优策略 $u^*(t) = \pi^*(E(t), I(t))$，使得代价泛函最小：

$$
J(\pi) = \mathbb{E}\left[ \int_0^T \big( (y-y_{\text{target}})^T Q (y-y_{\text{target}}) + R u^2 \big) dt + (y(T)-y_{\text{target}})^T P (y(T)-y_{\text{target}}) \right]
$$

该问题可通过**Hamilton-Jacobi-Bellman (HJB) 方程**精确刻画，也可通过**深度强化学习（Actor-Critic）**近似求解。

---

## 二、15个种子项目的融合映射

| 序号 | 种子项目 | 核心算法 | 融合角色 |
|:---:|:---|:---|:---|
| 1 | `797_nelder_mead` | Nelder-Mead单纯形无梯度优化 | `policy_optimizer.py`：策略网络参数的直接搜索优化，避免SDE导致的非光滑目标 |
| 2 | `774_monoalphabetic` | 单字母置换编码/解码 | `state_space_tools.py`：状态空间离散索引的置换编码，用于Q-table紧凑存储 |
| 3 | `1199_tec_to_vtk` | 网格数据格式转换 | `state_space_tools.py`：状态轨迹的结构化序列化/反序列化 |
| 4 | `1230_tet_mesh` | 四面体网格剖分与质量评估 | `hjb_fem_solver.py`：三维状态空间四面体剖分（Kuhn剖分）及网格质量计算 |
| 5 | `934_pyramid_jaskowiec_rule` | 金字塔区域高精度数值积分 | `quadrature_engine.py`：Jaskowiec-Sukumar对称求积规则，用于高维期望计算 |
| 6 | `863_pendulum_ode_period` | 单摆ODE与周期分析 | `neural_mass_dynamics.py`：神经振荡器平衡点分析、Jacobian线性化、周期估计 |
| 7 | `824_octopus` | Octave环境检测 | `numeric_utils.py`：Python数值计算环境检测与稳定性断言 |
| 8 | `258_cvt_metric` | 变度量CVT空间剖分 | `state_space_tools.py`：变度量（Fisher信息矩阵）CVT Lloyd迭代，状态空间最优量化 |
| 9 | `738_matrix_assemble_parfor` | 大规模矩阵组装 | `hjb_fem_solver.py`：有限元质量矩阵与刚度矩阵的单元级组装 |
| 10 | `085_bicg` | 双共轭梯度法 | `hjb_fem_solver.py`：求解HJB离散化后的非对称线性系统 |
| 11 | `059_autocatalytic_ode` | 自催化反应ODE | `neural_mass_dynamics.py`：自催化动力学映射为神经元兴奋-抑制耦合机制 |
| 12 | `1063_sde` | 随机微分方程数值方法 | `sde_integrator.py`：Euler-Maruyama、Milstein、随机显式中点法及均方稳定性分析 |
| 13 | `094_bisection` | 二分法求根 | `numeric_utils.py`：Bang-Bang最优控制切换时间求解、Pontryagin原理应用 |
| 14 | `898_polynomials` | 多项式基准测试函数 | `numeric_utils.py`：Rosenbrock、Himmelblau函数用于验证优化器收敛性能 |
| 15 | `766_midpoint_explicit` | 显式中点法 | `sde_integrator.py`：随机显式中点法（Stratonovich SDE）；`policy_optimizer.py`：策略评估轨迹积分 |

---

## 三、核心数学物理模型与公式

### 3.1 Wilson-Cowan神经质量模型

兴奋性与抑制性神经元群体的平均场方程：

$$
\tau_e \dot{E} = -E + S_e(a_{ee}E - a_{ei}I + P + u), \quad S_e(x) = \frac{1}{1+e^{-k_e(x-\theta_e)}}
$$

$$
\tau_i \dot{I} = -I + S_i(a_{ie}E - a_{ii}I + Q), \quad S_i(x) = \frac{1}{1+e^{-k_i(x-\theta_i)}}
$$

### 3.2 Hamilton-Jacobi-Bellman方程

值函数 $V(x,t)$ 满足的后向抛物型PDE：

$$
\frac{\partial V}{\partial t} + \min_u \left\{ L(x,u) + \nabla V \cdot f(x,u) + \frac{1}{2}\text{Tr}\big[\sigma(x)\sigma(x)^T \nabla^2 V\big] \right\} = 0
$$

终端条件：$V(x,T) = \Phi(x)$

有限元+隐式欧拉离散：

$$
(M + \Delta t A) V^n = M V^{n+1} + \Delta t \min_u \big[ L(x,u) \big]
$$

### 3.3 Actor-Critic强化学习

**Critic更新（时序差分）**：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \quad w \leftarrow w + \alpha_w \delta_t \nabla_w V(s_t)
$$

**Actor更新（策略梯度）**：

$$
\theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

值函数RBF近似：

$$
V_w(s) = w_0 + \sum_{i=1}^{N} w_i \exp\left(-\frac{\|s-c_i\|^2}{2\sigma^2}\right)
$$

高斯策略：

$$
\pi_\theta(a|s) = \mathcal{N}\big(\mu_\theta(s), \sigma_{\text{policy}}^2\big), \quad \mu_\theta(s) = a_{\max} \tanh(\theta^T \phi(s))
$$

### 3.4 Nelder-Mead单纯形优化

单纯形变换操作：
- **反射**：$x_r = \bar{x} + \rho(\bar{x} - x_{\text{worst}})$
- **扩展**：$x_e = \bar{x} + \xi(\bar{x} - x_{\text{worst}})$
- **外收缩**：$x_c = \bar{x} + \gamma(\bar{x} - x_{\text{worst}})$
- **内收缩**：$x_{ci} = \bar{x} - \gamma(\bar{x} - x_{\text{worst}})$
- **收缩**：$x_i = x_{\text{best}} + \sigma(x_i - x_{\text{best}})$

标准参数：$\rho=1, \xi=2, \gamma=0.5, \sigma=0.5$

### 3.5 SDE数值方法的强收敛阶

**Euler-Maruyama**：

$$
X_j = X_{j-1} + f(X_{j-1})\Delta t + g(X_{j-1})\Delta W_j, \quad \mathbb{E}[|X_T - X_T^h|] \leq C h^{1/2}
$$

**Milstein**：

$$
X_j = X_{j-1} + f\Delta t + g\Delta W_j + \frac{1}{2} g g' (\Delta W_j^2 - \Delta t), \quad \mathbb{E}[|X_T - X_T^h|] \leq C h
$$

**随机显式中点法**：

$$
Y_m = Y_{j-1} + \frac{1}{2}\Delta t f(Y_{j-1}) + \frac{1}{2} g(Y_{j-1}) \Delta W_j
$$

$$
Y_j = Y_{j-1} + \Delta t f(Y_m) + g(Y_m) \Delta W_j
$$

### 3.6 均方稳定性

对线性测试方程 $dX = \lambda X dt + \mu X dW$，Euler-Maruyama的均方稳定条件：

$$
|1 + \lambda\Delta t|^2 + |\mu|^2 \Delta t < 1
$$

等价于：

$$
\Delta t < -\frac{2\lambda + \mu^2}{\lambda^2} \quad (\lambda < 0)
$$

### 3.7 变度量CVT

Voronoi区域在度量 $A(x)$ 下的距离：

$$
d_A(x, z)^2 = (x-z)^T A\left(\frac{x+z}{2}\right)(x-z)
$$

Lloyd迭代更新生成元为Voronoi区域的质心：

$$
z_i^{(k+1)} = \frac{\int_{V_i} x \, dx}{\int_{V_i} dx}
$$

### 3.8 Jaskowiec-Sukumar金字塔求积

在参考金字塔 $P = \{(x,y,z): -(1-z)\leq x,y\leq 1-z, 0\leq z\leq 1\}$ 上：

$$
\int_P f(x,y,z)\,dV \approx \sum_{k=1}^{n} w_k f(x_k, y_k, z_k)
$$

### 3.9 Lyapunov指数与指数稳定性

最大Lyapunov指数：

$$
\lambda_{\max} = \lim_{t\to\infty} \frac{1}{t} \ln\frac{\|\delta x(t)\|}{\|\delta x(0)\|}
$$

受控系统的指数衰减速率通过线性回归估计：

$$
\ln\|y(t) - y_{\text{eq}}\| \approx \ln C - \lambda t
$$

---

## 四、项目文件结构

```
149_synth_project/
├── main.py                       # 统一入口，零参数可运行
├── neural_mass_dynamics.py       # 神经质量模型动力学、代价函数
├── sde_integrator.py             # SDE数值积分（EM, Milstein, 随机中点）
├── hjb_fem_solver.py             # HJB方程有限元求解器 + BiCG
├── policy_optimizer.py           # Nelder-Mead策略优化 + 反馈策略
├── state_space_tools.py          # CVT离散化 + 状态编码 + 数据序列化
├── quadrature_engine.py          # 金字塔规则 + Gauss-Hermite + Monte Carlo
├── rl_agent.py                   # Actor-Critic强化学习智能体
├── numeric_utils.py              # 二分法 + 基准函数 + 环境检测
├── convergence_analyzer.py       # 收敛阶 + Lyapunov指数 + 稳定性扫描
└── README_博士级合成说明.md      # 本文档
```

---

## 五、运行方式

```bash
cd 149_synth_project
python main.py
```

无需任何命令行参数。程序将自动执行：
1. 环境检测
2. 神经动力学验证
3. 三种SDE数值积分方法对比
4. HJB有限元求解
5. Actor-Critic强化学习训练（30回合）
6. Nelder-Mead策略优化
7. 多种数值积分验证
8. CVT状态空间离散化
9. 收敛性与稳定性分析
10. 综合结果汇总

---

## 六、边界处理与数值鲁棒性

- **Sigmoid溢出保护**：当指数参数超过±700时截断
- **控制输入饱和**：生理约束 $|u| \leq 5$ mA/cm²
- **状态硬约束**：神经元活动率限制在 $[-0.1, 1.1]$ 区间
- **除零保护**：`safe_divide` 函数
- **数值稳定性断言**：所有关键数组通过 `assert_numeric_stability` 检查
- **BiCG breakdown保护**：当 $|\rho| < 10^{-30}$ 时自动终止并回退
- **单纯形坍塌保护**：当单纯形收缩到机器精度时注入微扰动
- **Monte Carlo异常值剔除**：去除5%-95%之外的极端样本

---

## 七、科学难度说明

本项目涉及以下博士级科学计算内容：

1. **随机偏微分方程（SPDE）数值求解**：HJB方程作为后向Kolmogorov方程的推广，其有限元离散涉及非对称 convection-diffusion 系统
2. **随机分析**：Milstein方法的Itô修正项、Stratonovich积分、均方稳定性理论
3. **非凸优化**：Nelder-Mead在策略空间中的全局搜索、Rosenbrock基准验证
4. **泛函分析**：变度量空间中的最优量化（CVT）、值函数的RBF逼近
5. **动力系统**：Lyapunov指数计算、平衡点分岔分析、神经振荡周期估计
6. **概率数值方法**：高斯求积、金字塔对称规则、Monte Carlo方差分析
7. **强化学习理论**：策略梯度定理、Actor-Critic的TD误差与优势函数
8. **最优控制理论**：Pontryagin极大值原理、Bang-Bang切换、LQR代价泛函
