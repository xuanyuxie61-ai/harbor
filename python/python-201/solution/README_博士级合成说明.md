# 随机 Cahn-Hilliard 相场方程的多元素广义多项式混沌不确定性量化

## PROJECT 201: 博士级科学合成项目

---

## 一、科学问题概述

本项目实现了一个前沿博士级科学计算问题：**随机 Cahn-Hilliard 相场方程的多元素广义多项式混沌 (Multi-Element generalized Polynomial Chaos, ME-gPC) 不确定性量化与全局灵敏度分析**。

### 物理背景

二元合金的旋节分解 (spinodal decomposition) 是材料科学中的核心相变过程。Cahn-Hilliard 方程描述了这一过程：

$$\frac{\partial c}{\partial t} = \nabla \cdot \left( M(\boldsymbol{\xi}) \nabla \mu \right)$$

$$\mu = \frac{\delta \mathcal{F}}{\delta c} = \gamma(\boldsymbol{\xi})(c^3 - c) - \kappa(\boldsymbol{\xi}) \nabla^2 c$$

其中自由能泛函为：

$$\mathcal{F}[c] = \int_\Omega \left[ \frac{\gamma}{4}(c^2 - 1)^2 + \frac{\kappa}{2}|\nabla c|^2 \right] d\mathbf{x}$$

在实际应用中，以下参数存在不可避免的不确定性：
- **梯度能量系数** $\kappa(\boldsymbol{\xi})$：控制界面张力，高斯分布
- **双阱势参数** $\gamma(\boldsymbol{\xi})$：控制相分离驱动力，均匀分布
- **迁移率** $M(\boldsymbol{\xi})$：控制扩散速率，截断高斯分布

### 数学框架

**多元素广义多项式混沌 (ME-gPC) 展开：**

$$c(\mathbf{x}, t, \boldsymbol{\xi}) \approx \sum_{e=1}^{E} \sum_{k=0}^{P_e} \hat{c}_k^{(e)}(\mathbf{x}, t) \Psi_k^{(e)}(\boldsymbol{\xi}), \quad \boldsymbol{\xi} \in \Xi_e$$

其中 $\{\Xi_e\}_{e=1}^E$ 是随机空间的非重叠分解，$\Psi_k^{(e)}$ 是元素 $e$ 内的局部正交多项式基。

**随机 Galerkin 投影：**

将随机 PDE 投影到多项式基上，得到确定性耦合系统：

$$\sum_{k=0}^{P} \left\langle \mathcal{L}(\hat{c}_k \Psi_k), \Psi_j \right\rangle_w = \left\langle f, \Psi_j \right\rangle_w, \quad j = 0, \ldots, P$$

**三重乘积耦合张量：**

$$C_{ijk} = \langle \Psi_i \Psi_j, \Psi_k \rangle_w = \int \Psi_i(\boldsymbol{\xi}) \Psi_j(\boldsymbol{\xi}) \Psi_k(\boldsymbol{\xi}) w(\boldsymbol{\xi}) d\boldsymbol{\xi}$$

**非线性项的 Galerkin 投影 (c³)：**

$$(c^3)_j \approx \sum_{i,k,l} \hat{c}_i \hat{c}_k \hat{c}_l \langle \Psi_i \Psi_k \Psi_l, \Psi_j \rangle$$

---

## 二、项目结构与文件说明

```
201_synth_project_Advanced/
├── main.py                    # 统一入口 (零参数运行)
├── config.py                  # 全局配置管理
├── measure.py                 # 概率测度、Gauss求积、Owen T函数
├── utils.py                   # 数值工具 (二分法、稀疏矩阵、谱微分)
├── grid.py                    # 物理/随机空间网格生成
├── polynomial_basis.py        # 正交多项式基函数构造
├── multi_element.py           # 多元素gPC分解与耦合
├── sparse_grid.py             # Smolyak自适应稀疏网格
├── galerkin.py                # 随机Galerkin投影
├── cahn_hilliard.py           # 随机Cahn-Hilliard方程求解
├── refinement.py              # 自适应细化策略
├── neural_closure.py          # Port-Hamiltonian神经闭合模型
├── multifidelity.py           # 多保真度控制变量方法
├── statistics.py              # 统计量与Sobol灵敏度
├── mcmc_bayes.py              # MCMC贝叶斯推断
└── README_博士级合成说明.md    # 本文件
```

### 共 14 个 Python 文件 + 1 个 README

---

## 三、15个种子项目的融入映射

| 种子项目 | 核心算法 | 在本项目中的角色 |
|---------|---------|----------------|
| **096_bisection_min** | 二分法求单峰函数最小值 | 二分法搜索满足目标误差的最优PCE阶数 |
| **1050_LaM-SLidE** | 编码器-解码器、潜空间扩散模型 | 编码器-解码器架构用于跨保真度映射和神经闭合特征提取 |
| **986_r8ncf** | 稀疏矩阵COO坐标格式 | 随机Galerkin系统的稀疏矩阵存储与装配 |
| **1434_zombie_ode** | 耦合ODE系统(SIR模型) | 多元素之间的界面耦合机制，类似分区ODE耦合 |
| **680_line_grid** | 一维线段网格生成 | 物理空间网格和随机空间求积网格的基础生成器 |
| **738_matrix_assemble_parfor** | 并行矩阵装配 | 随机Galerkin耦合系统的块矩阵并行装配 |
| **932_pyramid_grid** | 三维金字塔层级网格 | Smolyak稀疏网格的层级结构组织 |
| **068_ball_integrals** | 高维精确积分(Gamma函数) | 多维概率空间上的正交多项式内积计算 |
| **1246_Gaulios** | 有限-无穷horizon Riccati耦合 | 多保真度模型的桥接：低保真→高保真的信息传递 |
| **211_continuity_exact** | 无散度速度场构造 | Cahn-Hilliard方程的质量守恒约束验证 |
| **033_asa076** | 正态CDF + Owen T函数 | 失效概率计算和二元正态累积概率 |
| **1298_Port-Hamiltonian** | Port-Hamiltonian神经网络 | 能量保持约束的神经闭合模型：确保dF/dt≤0 |
| **787_navier_stokes_2d_exact** | 2D Navier-Stokes精确解 | Cahn-Hilliard求解器的验证框架(能量耗散、守恒律) |
| **1180_HII-galaxy** | MCMC采样 + 贝叶斯证据 | PCE代理加速的MCMC后验推断和模型选择 |
| **382_fem_to_xml** | 有限元网格格式转换 | 物理计算域的网格生成与元数据管理 |

---

## 四、核心数学公式与算法

### 4.1 正交多项式三递推关系

$$P_{n+1}(\xi) = (\xi - \alpha_n) P_n(\xi) - \beta_n P_{n-1}(\xi)$$

- **Hermite** (高斯测度): $\alpha_k = \mu$, $\beta_k = k\sigma^2$
- **Legendre** (均匀测度): $\alpha_k = \frac{a+b}{2}$, $\beta_k = \frac{(b-a)^2 k^2}{4k^2-1}$
- **Jacobi** (Beta测度): 由参数 $\alpha, \beta$ 确定的递推系数

### 4.2 Gauss 求积 (Golub-Welsch 算法)

构造 Jacobi 矩阵 $J$ 的对称三对角形式，其特征值 = 求积节点，特征向量第一行的平方 = 求积权重。

$$\int f(\xi) w(\xi) d\xi \approx \sum_{k=1}^{N_q} w_k f(\xi_k)$$

### 4.3 Smolyak 稀疏网格

$$A(q, d) = \sum_{(-1)^{q-|\mathbf{i}|}} \binom{d-1}{q-|\mathbf{i}|} (Q^{i_1} \otimes \cdots \otimes Q^{i_d})$$

将求积点数从 $O(n^d)$ 降至 $O(n (\log n)^{d-1})$。

### 4.4 自适应细化的 Dörfler 标记策略

选择最小标记集合 $S$ 使得：

$$\sum_{e \in S} \eta_e^2 \geq \theta \sum_{e=1}^{E} \eta_e^2$$

### 4.5 Sobol 全局灵敏度指数

**一阶指数:**
$$S_i = \frac{\text{Var}_{\xi_i}[\mathbb{E}[u|\xi_i]]}{\text{Var}[u]} = \frac{\sum_{k \in \mathcal{S}_i} \hat{u}_k^2}{\sum_{k=1}^{P} \hat{u}_k^2}$$

**总效应指数:**
$$S_{Ti} = 1 - \frac{\sum_{k: \alpha_k^{(i)}=0} \hat{u}_k^2}{\sum_{k=1}^{P} \hat{u}_k^2}$$

### 4.6 Port-Hamiltonian 能量约束

$$\frac{d\mathcal{F}}{dt} = -\int_\Omega M(\boldsymbol{\xi}) |\nabla \mu|^2 d\mathbf{x} \leq 0$$

神经闭合模型的修正被结构化为：$\hat{\tau} = -R_{nn} \nabla H(\hat{c})$，其中 $R_{nn} \geq 0$ 通过 softplus 参数化保证。

### 4.7 多保真度控制变量估计

$$\hat{\mu}_{CV} = \hat{\mu}_L + \beta^* (\hat{\mu}_H - \hat{\mu}_L)$$

$$\beta^* = \frac{\text{Cov}[f_H, f_L]}{\text{Var}[f_L]}$$

### 4.8 Owen T 函数与失效概率

$$T(h, a) = \frac{1}{2\pi} \int_0^a \frac{\exp(-h^2(1+t^2)/2)}{1+t^2} dt$$

$$P_f = P(g(\boldsymbol{\xi}) > \text{threshold}) \approx \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}_{g(\boldsymbol{\xi}_i) > \tau}$$

---

## 五、计算流程

`main.py` 按以下 10 个阶段顺序执行：

1. **配置初始化**: 创建测度、基函数、物理参数等全局配置
2. **基函数与求积**: 构造正交多项式基、Smolyak 稀疏网格、二分法搜索最优阶数
3. **Galerkin 投影**: 计算三重乘积耦合张量、装配随机 Galerkin 系统
4. **自适应细化**: 多元素分解、Dörfler 标记、hp-自适应策略
5. **Cahn-Hilliard 求解**: Fourier 谱方法 + 半隐式时间积分 + 配点法 UQ
6. **神经闭合训练**: Port-Hamiltonian 约束的 MLP 训练
7. **多保真度分析**: 控制变量估计 + 非线性修正模型
8. **统计分析**: 均值、方差、Sobol 指数、PDF、分位数
9. **MCMC 推断**: PCE 代理加速的 Metropolis-Hastings 采样 + 失效概率
10. **综合报告**: 输出完整的计算结果摘要

---

## 六、运行方法

```bash
cd 201_synth_project_Advanced
python3 main.py
```

**零参数运行**，所有配置已在 `config.py` 中预设。

**依赖**: 仅需要 `numpy` 和 `scipy` (标准科学计算库)。

---

## 七、合成后项目能解决的科学问题

1. **随机相场方程的不确定性传播**: 量化输入参数的不确定性如何影响相分离动力学
2. **全局灵敏度分析**: 识别对输出变异性贡献最大的不确定参数
3. **失效概率估计**: 计算相分离时间超过临界阈值的概率
4. **高效 UQ**: 通过 ME-gPC + 多保真度策略，以最小计算代价达到目标精度
5. **物理一致的代理模型**: Port-Hamiltonian 约束确保代理模型遵循热力学第二定律
6. **贝叶斯参数推断**: 从观测数据反推不确定参数的后验分布

---

## 八、关键数值特性

- **数值鲁棒性**: 所有除法操作使用安全除零保护，矩阵条件数监控
- **边界处理**: 浓度场限制在物理范围 $[-1.1, 1.1]$，求积节点限制在支撑集内
- **守恒律验证**: 质量守恒和能量耗散律的定量验证
- **自适应精度**: 基于后验误差估计的自适应网格细化
- **可重复性**: 固定随机种子确保计算可重复
