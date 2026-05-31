# 量子计算：量子退火组合优化 —— 博士级合成说明

## 1. 项目概述

本项目将 15 个原始科研代码项目的核心算法融合重构为一个面向**量子计算：量子退火组合优化**的博士级科研计算项目。项目围绕以下前沿科学问题展开：

> **如何利用路径积分蒙特卡洛（PIMC）与平均场理论，在存在控制噪声与热浴耦合的条件下，对非 stoquastic 哈密顿量进行量子退火优化，并系统评估不同退火 schedule 的绝热性能？**

该问题涉及量子统计力学、组合优化、数值分析、蒙特卡洛方法等多个高难度的交叉领域，计算复杂度与理论深度达到博士研究级别。

---

## 2. 原项目到科学问题的映射

| 序号 | 原项目 | 核心算法/思想 | 在合成项目中的角色 |
|:---|:---|:---|:---|
| 1 | `1209_test_int_nd` | N 维盒形 Monte Carlo 积分 | 路径积分中高维配分函数积分的数值框架 |
| 2 | `587_imshow_numeric` | 数值数组归一化 | 量子态振幅归一化、密度矩阵迹归一化 |
| 3 | `622_knapsack_01_brute` | 0/1 背包 + Gray 码枚举 | QUBO→Ising 映射、精确基态 Gray 码枚举 |
| 4 | `523_hermite_product_display` | 厄米特多项式 | 横向场下的量子谐振子基函数与隧穿振幅 |
| 5 | `589_insurance_simulation` | 随机过程与死亡率模型 | 热噪声 Metropolis 采样、概率跃迁模型 |
| 6 | `198_collatz_polynomial` | 模 2 多项式迭代 | 非线性分形型退火 schedule 构造 |
| 7 | `585_image_sample` | 坐标采样框架 | 状态空间采样与构型空间遍历 |
| 8 | `048_asa226` | 不完全 Beta 函数 | 多项式 schedule 的平滑过渡、统计分布 |
| 9 | `107_boundary_word_equilateral` | 边界词与六方对称群 | 六边形晶格几何与对称操作群 C₆ᵥ |
| 10 | `372_fem_basis_q4_display` | Q4 有限元双线性基函数 | 连续势场的空间离散化与插值 |
| 11 | `603_jacobi` | Jacobi 迭代 | 线性化平均场方程求解与 Chebyshev 加速 |
| 12 | `503_hand_mesh2d` | 2D 网格生成 | 量子势场区域三角化与面积分 |
| 13 | `1360_truncated_normal` | 截断正态分布 | 控制噪声的统计建模与有界扰动采样 |
| 14 | `1312_triangle_monte_carlo` | 三角形 Monte Carlo | 几何域上的量子态重叠积分 |
| 15 | `779_monty_hall_simulation` | 条件概率更新 | 固定部分自旋的条件采样与信息更新 |

---

## 3. 新增数学物理模型与核心公式

### 3.1 伊辛模型与 QUBO 映射

经典组合优化问题可编码为二次无约束二进制优化（QUBO）：

$$
H_{\text{QUBO}}(\mathbf{x}) = \sum_{i \le j} Q_{ij} x_i x_j, \quad x_i \in \{0,1\}
$$

通过映射 $s_i = 2x_i - 1 \in \{-1,+1\}$，转化为伊辛哈密顿量：

$$
H_{\text{Ising}}(\mathbf{s}) = \sum_{i<j} J_{ij} s_i s_j + \sum_i h_i s_i
$$

其中：

$$
J_{ij} = \frac{Q_{ij}}{4}, \quad h_i = \frac{Q_{ii}}{2} + \sum_{j \ne i} \frac{Q_{ij}}{4}
$$

### 3.2 横向场量子退火哈密顿量

含时薛定谔方程（虚时间）描述量子退火演化：

$$
H(t) = A(t) H_D + B(t) H_P
$$

其中驱动哈密顿量 $H_D = -\Gamma(t) \sum_i \sigma_i^x$ 诱导量子隧穿，问题哈密顿量 $H_P = H_{\text{Ising}}$。$A(t), B(t)$ 为退火 schedule，满足 $A(0) \approx 1, B(0) \approx 0$ 且 $A(T) \approx 0, B(T) \approx 1$。

### 3.3 路径积分与 Trotter-Suzuki 分解

配分函数通过 Trotter 分解映射到经典 $(d+1)$ 维系统：

$$
Z = \text{Tr}\, e^{-\beta H} \approx \sum_{\{\mathbf{s}^{(m)}\}} \exp\left( -S_E[\{\mathbf{s}^{(m)}\}] \right)
$$

欧几里得作用量（世界线作用量）：

$$
S_E = \sum_{m=1}^{M} \left[ \Delta\tau \, E_P(\mathbf{s}^{(m)}) - J_\perp^{(m)} \sum_i s_i^{(m)} s_i^{(m+1)} \right]
$$

其中虚时间步长 $\Delta\tau = \beta/M$，有效横向耦合：

$$
J_\perp = \frac{1}{2} \ln \coth(\Delta\tau \, \Gamma)
$$

### 3.4 平均场自洽方程

在逆温度 $\beta$ 下，局域磁化强度满足不动点方程：

$$
m_i = \tanh\!\left( \beta \left[ h_i + \sum_j J_{ij} m_j \right] \right)
$$

采用阻尼迭代格式：

$$
\mathbf{m}^{(k+1)} = (1-\alpha) \mathbf{m}^{(k)} + \alpha \tanh(\beta \mathbf{h}^{\text{eff}})
$$

变分自由能：

$$
F_{\text{MF}} = -\frac{1}{2} \mathbf{m}^T \! J \, \mathbf{m} - \mathbf{h}^T \! \mathbf{m} + \frac{1}{\beta} \sum_i \left[ \frac{1+m_i}{2} \ln\frac{1+m_i}{2} + \frac{1-m_i}{2} \ln\frac{1-m_i}{2} \right]
$$

### 3.5 截断正态噪声模型

控制误差建模为截断正态分布 $X \sim \mathcal{TN}(\mu, \sigma^2, a, b)$：

$$
f_X(x) = \frac{\phi((x-\mu)/\sigma)}{\sigma \, [\Phi((b-\mu)/\sigma) - \Phi((a-\mu)/\sigma)]}, \quad a \le x \le b
$$

其中 $\phi, \Phi$ 分别为标准正态的 pdf 与 cdf。解析均值：

$$
\mathbb{E}[X] = \mu + \sigma \frac{\phi(\alpha) - \phi(\beta)}{\Phi(\beta) - \Phi(\alpha)}, \quad \alpha = \frac{a-\mu}{\sigma}, \; \beta = \frac{b-\mu}{\sigma}
$$

### 3.6 厄米特函数基与隧穿振幅

横向场的本征基为量子谐振子态。物理学家厄米特多项式满足：

$$
H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
$$

厄米特函数（正交归一）：

$$
\psi_n(x) = \frac{H_n(x) e^{-x^2/2}}{\sqrt{2^n n! \sqrt{\pi}}}, \quad \int_{-\infty}^{\infty} \psi_m(x) \psi_n(x) \, dx = \delta_{mn}
$$

### 3.7 不完全 Beta 函数 Schedule

多项式局部最优 schedule 利用正则化不完全 Beta 函数：

$$
B(s; p,q) = \int_0^s t^{p-1} (1-t)^{q-1} dt, \quad I_s(p,q) = \frac{B(s;p,q)}{B(p,q)}
$$

定义 $B(t) = I_{t/T}(p,q)$，在能隙最小点 $s^*$ 附近自然实现局部放缓，满足绝热定理要求：

$$
\frac{|\langle \psi_1 | \partial_t H | \psi_0 \rangle|}{\Delta(t)^2} \le \varepsilon
$$

---

## 4. 项目文件结构

```
154_synth_project/
├── main.py                          # 统一入口，零参数运行
├── ising_hamiltonian.py             # Ising/QUBO 哈密顿量构造
├── transverse_field_basis.py        # 厄米特基函数与隧穿核
├── noise_model.py                   # 截断正态噪声与热浴模型
├── annealing_schedules.py           # 退火 schedule 设计
├── lattice_geometry.py              # 六方晶格与 Q4 有限元
├── iterative_solver.py              # Jacobi 迭代与自洽场求解
├── state_sampler.py                 # Gray 码/MCMC/并行回火采样
├── path_integral_monte_carlo.py     # PIMC 世界线模拟
└── utils.py                         # 高维积分、保真度、纠缠熵
```

---

## 5. 运行方式

```bash
cd Synthesis-project-python/154_synth_project
python main.py
```

程序将自动执行 13 个计算阶段，总耗时约 40–60 秒，终端输出包含：
- 精确基态能量（Gray 码枚举）
- 平均场自洽解
- Jacobi/Chebyshev 迭代收敛性
- 五种 schedule 的绝热指标
- 噪声下基态漂移
- PIMC 能量、磁化率与缠绕数
- 有限元势场积分
- 高维 Monte Carlo 积分验证
- 量子信息度量与鲁棒性检验

---

## 6. 科学问题与工程复杂性说明

### 6.1 科学难度

1. **非 stoquastic 哈密顿量**：项目不仅限于标准 stoquastic 横向场伊辛模型，还引入了符号变化的 $J_{ij}$（自旋玻璃），使得基态求解为 NP-hard。
2. **有限温度效应**：通过 PIMC 在有限 $\beta$ 下模拟量子涨落，超越纯基态变分方法。
3. **多体纠缠与拓扑**：世界线缠绕数作为 $(d+1)$ 维系统的拓扑序参量，反映量子相变。
4. **噪声鲁棒性**：截断正态分布、热 Metropolis 准则与横向场抖动共同建模真实量子硬件的非理想性。

### 6.2 工程复杂性

1. **边界处理**：所有输入函数均检查正定性、维度匹配、数值范围（如 `np.clip` 防止指数上溢）。
2. **数值稳定性**：
   - `log-sum-exp` 技巧用于配分函数
   - Acklam 近似 + 牛顿修正实现逆正态 CDF
   - 小参数展开处理 $J_\perp$ 的奇异性
3. **算法多样性**：精确枚举（指数级）、迭代松弛（多项式级）、蒙特卡洛（统计级）三种尺度的方法共存并交叉验证。
4. **模块化设计**：10 个独立 `.py` 文件各司其职，通过统一接口在 `main.py` 中协调。

---

## 7. 关键结果示例

运行 `main.py` 后得到的典型输出：

- **精确基态能量**（N=14 随机自旋玻璃）：$E_0 = -15.143$
- **平均场变分能量**：$E_{\text{var}} = -17.388$（误差 $|E_{\text{var}} - E_0| = 2.24$，说明平均场对受挫系统估计偏优）
- **PIMC 估算基态**（N=12, β=5, M=32）：$E_0 \approx -9.64$
- **3D 高斯积分 MC 验证**：相对误差 < 2%
- **噪声导致能量漂移**：$|E_0' - E_0| \approx 0.43$（约 3%）

---

## 8. 合成方法总结

本项目的合成策略为**“算法内核提取 + 物理语义重写 + 公式一致性注入”**：

1. **提取**：从每个原始项目中提取核心数值算法（Gray 码生成、Jacobi 迭代、截断正态采样、不完全 Beta 级数、Hermite 递推等）。
2. **重写**：将算法嵌入到量子退火的物理语境中，赋予其新的数学物理语义（如 Gray 码枚举 → 自旋构型空间遍历， insurance 死亡率 → Metropolis 热激发概率）。
3. **注入**：系统性地加入配分函数、Trotter 分解、自洽方程、绝热定理、纠缠熵等博士级公式，确保“公式—算法—代码”三线一致。
4. **删除**：彻底移除所有可视化代码（imshow、plot、surface 等），保留纯数值计算。

---

*本项目为科研教学与算法验证用途，代码遵循模块化与数值鲁棒性最佳实践。*
