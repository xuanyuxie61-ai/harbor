# PROJECT_189: 强化学习策略梯度 —— 非线性振荡网络最优控制

## 一、项目概述

本项目是一个**博士级**的 Python 科研代码合成项目，围绕**数据科学：强化学习策略梯度**领域展开。项目将 15 个种子项目的核心算法融合为一个面向前沿科学问题的完整计算框架，用于解决**受迫非线性振荡网络的最优控制问题**。

### 核心科学问题

设计数据驱动的策略梯度算法，控制一个具有锯齿波周期强迫与放牧型非线性耦合的动力学系统，使其跟踪由贝塞尔函数描述的准周期参考轨迹，同时满足物理约束并最小化控制能量。

---

## 二、原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 在合成项目中的角色 |
|------|-----------|---------|------------------|
| 1 | 1059_sawtooth_ode | 锯齿波驱动 ODE | **环境动力学核心**：`dynamical_system.py` 中的 `sawtooth_wave(t)` 提供周期强迫项 $S(t) = 2(\text{frac}(\omega_s t/2\pi) - 0.5)$ |
| 2 | 488_grazing_ode | 放牧捕食者-猎物模型 | **环境动力学核心**：`grazing_coupling(x1, x3)` 实现振荡子与生态模块间的状态依赖非线性耦合 |
| 3 | 326_eigenfaces | PCA / Turk-Pentland 技巧 | **状态表示**：`spectral_basis.py` 中的 `pca_vectors()` 将高维观测投影到低维子空间，降低策略学习复杂度 |
| 4 | 1084_sine_integral | 正弦积分 Si(x) | **奖励塑形**：`dynamical_system.py` 的奖励函数中使用 $R(s,a) \propto \text{Si}(\|s\|)$ 提供软饱和特性 |
| 5 | 664_legendre_product_polynomial | Legendre 乘积多项式 | **谱基函数逼近**：`policy_network.py` 和 `value_approximator.py` 使用多元 Legendre 多项式作为策略和值函数的基函数 |
| 6 | 1267_toms179 | 不完全 Beta 函数 | **信任区域约束**：`constrained_optimizer.py` 中使用 $I_x(p,q)$ 计算策略更新的置信概率 |
| 7 | 1006_random_data | Brownian 运动 | **探索噪声**：`stochastic_processes.py` 中的 `ornstein_uhlenbeck_process()` 为策略提供时间相关探索噪声 |
| 8 | 1411_weekday | Julian Date / 时间计算 | **学习率调度**：`constrained_optimizer.py` 中的余弦退火调度器借鉴周期性时间编码思想 |
| 9 | 339_eternity | Eternity 拼图 LP 建模 | **约束优化**：`constrained_optimizer.py` 中的 `lp_action_projection()` 将动作投影到线性约束可行域 |
| 10 | 964_r83p | 周期三对角矩阵求解 | **Fisher 矩阵快速运算**：`linear_algebra.py` 中的 R83P 求解器用于自然梯度中的特殊结构线性系统 |
| 11 | 823_obj_to_tri_surface | 3D 网格三角化 | **状态空间划分**：`mesh_geometry.py` 中的 `StateSpaceTriangulation` 对状态空间进行 Delaunay 三角剖分 |
| 12 | 819_normal01_multivariate_distance | 多元正态距离统计 | **状态相似性度量**：`stochastic_processes.py` 中的高斯核矩阵用于核化优势估计 |
| 13 | 081_besselzero | 贝塞尔零点 / Halley 迭代 | **特征模态分析**：`special_functions.py` 中的 `bessel_zero()` 用于系统共振频率提取与谱滤波 |
| 14 | 1048_rref2 | RREF 行简化阶梯形 | **鲁棒线性求解**：`linear_algebra.py` 中的 `rref_solve()` 用于亏秩最小二乘问题 |
| 15 | 1262_toeplitz_cholesky | Toeplitz Cholesky 分解 | **协方差采样**：`linear_algebra.py` 中的 `sample_from_toeplitz_covariance()` 用于高斯策略的相关动作噪声生成 |

---

## 三、新增数学物理模型与核心公式

### 3.1 受控非线性动力学系统

状态方程融合锯齿波谐振子与放牧生态模型：

$$
\begin{aligned}
\dot{x}_1 &= x_2 + u_1 \\
\dot{x}_2 &= -\omega_0^2 x_1 + S(t) + F_{\text{graze}}(x_1, x_3) + u_2 \\
\dot{x}_3 &= r_1 x_3 \left(1 - \frac{x_3}{k}\right) - c_1 x_4 \left(1 - e^{-d_1 x_3}\right) + u_3 \\
\dot{x}_4 &= -a x_4 + c_2 x_4 \left(1 - e^{-d_2 x_3}\right) + u_4
\end{aligned}
$$

其中锯齿波 $S(t)$ 的 Fourier 展开为：

$$
S(t) = -\frac{2}{\pi} \sum_{n=1}^{\infty} \frac{(-1)^n}{n} \sin(n \omega_s t)
$$

放牧耦合项：

$$
F_{\text{graze}}(x_1, x_3) = -\gamma \frac{x_1 x_3}{1 + x_3^2}
$$

### 3.2 奖励函数（含特殊函数）

$$
R(s, a) = -\frac{1}{2}\|s\|^2 - \frac{1}{10}\|a\|^2 + \frac{1}{2} \text{Si}(\|s\|) \exp\left(-\frac{\|a\|^2}{4}\right)
$$

其中 $\text{Si}(x) = \int_0^x \frac{\sin t}{t} dt$ 为正弦积分，提供对小幅状态的额外软饱和鼓励。

### 3.3 谱基策略参数化

均值函数使用多元 Legendre 乘积多项式展开：

$$
\mu_i(s) = \sum_{|\alpha| \leq p} \theta_{i,\alpha} P_\alpha(\phi(s)), \quad
P_\alpha(z) = \prod_{j=1}^{d} P_{\alpha_j}(z_j)
$$

其中 $P_n(x)$ 满足 Bonnet 递推关系：

$$
(n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
$$

正交性：

$$
\int_{-1}^{1} P_n(x) P_m(x) \, dx = \frac{2}{2n+1} \delta_{nm}
$$

### 3.4 自然策略梯度

参数更新方向为 Fisher 信息矩阵的逆作用在标准梯度上：

$$
\Delta \theta = F(\theta)^{-1} \nabla_\theta J(\theta)
$$

Fisher 矩阵的定义：

$$
F(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta}
\left[ \nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T \right]
$$

通过共轭梯度法 (CG) 近似求解 $F x = g$，避免 $O(d^3)$ 的显式求逆。

### 3.5 广义优势估计 GAE($\lambda$)

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### 3.6 信任区域约束（不完全 Beta）

策略更新的 KL 散度约束：

$$
D_{KL}\left(\pi_{\theta_{\text{old}}} \| \pi_{\theta_{\text{new}}}\right) \leq \delta
$$

置信概率通过不完全 Beta 函数计算：

$$
P\left(D_{KL} \leq \delta\right) \approx I_{\frac{\delta}{\sigma^2 + \delta}}\left(\frac{d}{2}, \frac{N-d}{2}\right)
$$

其中 $I_x(p,q) = \frac{B(x;p,q)}{B(p,q)}$ 为正则化不完全 Beta 函数。

### 3.7 参考轨迹（贝塞尔函数）

准周期参考轨迹基于零阶贝塞尔函数：

$$
x_1^*(t) = J_0(\omega_0 t) \cos(\omega_s t), \quad
x_3^*(t) = 0.5 + 0.3 \sin(0.5t)
$$

$J_0(\omega_0 t)$ 的零点对应系统的反共振点，设计控制器需避开这些频率。

---

## 四、项目文件结构

```
189_synth_project/
├── main.py                         # 统一入口，零参数可运行
├── special_functions.py            # Si(x), Bessel零点, 不完全Beta
├── linear_algebra.py               # RREF, R83P, Toeplitz Cholesky
├── stochastic_processes.py         # Brownian运动, OU噪声, 高斯核
├── spectral_basis.py               # PCA, Legendre乘积多项式
├── dynamical_system.py             # 受控非线性振荡环境
├── policy_network.py               # 谱基高斯策略网络
├── value_approximator.py           # 谱最小二乘价值函数 + GAE
├── natural_gradient.py             # Fisher矩阵, CG, NPG优化器
├── constrained_optimizer.py        # LP投影, 信任区域, 学习率调度
├── mesh_geometry.py                # Delaunay三角剖分, 自适应加密
├── policy_gradient_core.py         # REINFORCE + NPG 训练循环
└── README_博士级合成说明.md        # 本文档
```

---

## 五、运行方式

```bash
cd /mnt/data/zpy/sci-swe/source\ code/Synthesis-project-python/189_synth_project
python main.py
```

程序将自动执行：
1. **数值验证阶段**：验证所有核心科学计算模块（Si, Bessel, Beta, RREF, Toeplitz, PCA, Legendre, Brownian, 信任区域）
2. **策略梯度训练阶段**：50 次迭代训练 REINFORCE + 自然梯度控制器
3. **测试评估阶段**：5 条确定性测试轨迹 + 参考轨迹跟踪误差分析
4. **状态空间网格分析阶段**：Delaunay 三角剖分与插值验证

---

## 六、科学意义与应用前景

本项目所研究的问题具有以下前沿科学意义：

1. **等离子体约束控制**：托卡马克装置中的粒子轨道可用受迫非线性振荡描述，锯齿波对应等离子体不稳定性周期，策略梯度方法可实时优化磁场位形。

2. **海洋生态-气候耦合**：放牧模型描述浮游生物-鱼类相互作用，季节强迫（锯齿波类比）驱动生态系统，最优捕捞策略需考虑非线性反馈。

3. **机械振动抑制**：旋转机械中的转子动力学具有周期边界条件（R83P 结构），自然梯度方法比传统 PID 控制更适合高维、强耦合系统。

4. **谱方法在 RL 中的理论保证**：Legendre 基函数的正交性为值函数逼近提供了 $L^2$ 收敛保证，相比神经网络具有更好的可解释性和样本效率。

---

## 七、边界处理与数值鲁棒性

- **ODE 积分**：采用四阶 Runge-Kutta (RK4) 方法，状态超出边界时强制截断到 $[-10, 10]^4$。
- **矩阵求逆**：RREF 求解器自动处理亏秩系统；Toeplitz Cholesky 在矩阵非正定时自动加正则化偏移。
- **梯度爆炸**：策略梯度计算中使用范数截断 ($\|\nabla J\| > 1$ 时归一化)。
- **KL 散度越界**：自然梯度更新后执行线搜索，逐步缩小步长直到满足信任区域约束。
- **数值溢出**：不完全 Beta 函数计算中使用指数缩放 (aleps 机制) 防止中间结果下溢。
