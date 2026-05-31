# 量子行走搜索算法 — 博士级科研代码合成说明

## 项目概述

本项目围绕**量子计算：量子行走搜索算法**这一前沿科学领域，融合15个种子科研代码项目的核心算法与数据结构，构建了一个面向多种复杂图结构（1D链、2D网格、六边形晶格、三角剖分区域、超立方体、球面立方网格）的量子行走搜索算法综合计算框架。

量子行走（Quantum Walk）是量子计算中的核心动力学模型，是经典随机行走的量子力学推广。在搜索问题中，量子行走算法能够在特定图结构上实现相对于经典算法的**二次加速**甚至更快的搜索效率。本项目不仅实现了离散时间硬币量子行走（DTQW）和连续时间量子行走（CTQW）的核心演化引擎，还系统整合了高维数值积分、稀疏网格参数优化、谱分析、三对角/Toeplitz/Vandermonde专用矩阵求解器等高级数值方法。

---

## 文件结构

| 文件 | 功能说明 | 融合种子项目 |
|------|---------|------------|
| `main.py` | 统一入口，零参数运行全部14组实验 | 全部 |
| `utils.py` | 数值工具、Hadamard/Grover硬币、Laplacian、网格邻居 | - |
| `matrix_solvers.py` | Toeplitz、Vandermonde、三对角（CG/循环约化/Jacobi/GS）、反向通信CG、Wathen矩阵 | r8to, r8vm, r83, cg_rc |
| `geometry_mesh.py` | 2D三角剖分、球面立方网格、六边形晶格与Stroud求积规则 | image_mesh2d, sphere_cubed_grid, hexagon_stroud_rule |
| `lattice_states.py` | Diophantine约束状态枚举、Clenshaw-Curtis稀疏网格、概率landscape随机采样 | diophantine_nd, cc_display, levels |
| `quantum_operators.py` | 硬币算符（Hadamard/Grover/Fourier）、平移算符、Oracle、Hamiltonian、Chebyshev传播子 | pwc_plot_2d |
| `quantum_walk_core.py` | DTQW、CTQW、多维量子行走、搜索增强量子行走核心类 | - |
| `search_algorithm.py` | 2D网格空间搜索、超立方体搜索、六边形晶格搜索、谱分析、多目标搜索 | - |
| `numerical_quadrature.py` | 六边形Stroud求积、梯形/Simpson积分、Steinerberger精确积分测试 | hexagon_stroud_rule, area_under_curve, steinerberger |
| `optimization_diagnostics.py` | Newton-Raphson优化、随机等高线采样、Coupon Collector覆盖时间、收敛率分析 | nonlin_newton, levels, full_deck_simulation |

---

## 核心数学物理模型与公式

### 1. 离散时间量子行走（DTQW）

硬币-位置复合Hilbert空间 $\mathcal{H} = \mathcal{H}_C \otimes \mathcal{H}_P$，维度为 $d \times N$。

**单步演化算符：**
$$\hat{W} = \hat{S} \cdot (\hat{C} \otimes \hat{I}_P)$$

- **Hadamard硬币**（维度 $d=2$）：
$$\hat{C}_H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

- **Grover扩散硬币**（维度 $d$）：
$$\hat{C}_G = 2|\psi\rangle\langle\psi| - \hat{I}_d, \quad |\psi\rangle = \frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}|i\rangle$$

- **平移算符**（1D周期边界）：
$$\hat{S}|c, x\rangle = |c, x + (-1)^c \mod N\rangle$$

### 2. 连续时间量子行走（CTQW）

**Hamiltonian**（图Laplacian形式）：
$$\hat{H} = \gamma \hat{L} + \hat{V}$$

其中图Laplacian为：
$$L_{ij} = \begin{cases} \deg(i) & i = j \\ -1 & (i,j) \in E \\ 0 & \text{otherwise} \end{cases}$$

**时间演化算符：**
$$\hat{U}(t) = e^{-i\hat{H}t} = \sum_k e^{-i E_k t}|E_k\rangle\langle E_k|$$

**Chebyshev多项式近似传播子**（用于稀疏Hamiltonian）：
$$e^{-iHt} \approx \sum_{k=0}^{K} c_k(t) T_k(\tilde{H}), \quad \tilde{H} = \frac{H}{E_{\max}}$$

其中 $c_k(t) = (-i)^k J_k(E_{\max} t)$，$J_k$ 为Bessel函数。

### 3. 量子行走搜索算法

**Oracle算符**（标记顶点相位翻转）：
$$\hat{O} = \hat{I} - (1 - e^{i\phi})\sum_{v \in M}|v\rangle\langle v|$$

**搜索演化：**
$$|\psi(t+1)\rangle = \hat{W}\hat{O}|\psi(t)\rangle$$

**理论最优步数**（$d$-正则图，$M$个标记点）：
$$T^* \approx \frac{\pi}{4}\sqrt{\frac{N}{M \cdot d}}$$

**二次加速比：**
$$\text{Speedup} = \frac{O(N/M)}{O(\sqrt{N/(M \cdot d)})} = O\left(\sqrt{\frac{N \cdot d}{M}}\right)$$

### 4. 谱分析与临界gamma

对于CTQW搜索，Childs-Goldstone临界值为：
$$\gamma_c = \frac{1}{N}\sum_{k>0}\frac{1}{\lambda_k}$$

其中 $\lambda_k$ 为图Laplacian的非零特征值。谱隙决定绝热演化时间：
$$T_{\text{ad}} = \frac{\pi}{2\Delta E}$$

### 5. 六边形Stroud数值积分

对于单位正六边形上的积分，采用7点5次精度Stroud规则：
$$\iint_{\text{hex}} f(x,y)\,dx\,dy \approx \sum_{i=1}^{7} w_i f(x_i, y_i)$$

**精确单式积分**（Steger多边形矩公式）：
$$\nu_{pq} = \sum_{\text{edge } (i,j)} \frac{x_i y_j - x_j y_i}{(p+q+2)(p+q+1)\binom{p+q}{p}} s_{pq}$$

其中 $s_{pq}$ 为边上双求和项。

### 6. Steinerberger病态函数

用于验证数值积分精度：
$$f(n, x) = \sum_{k=1}^{n} \frac{|\sin(\pi k x)|}{k}$$

**精确积分：**
$$\int_0^1 f(n, x)\,dx = \frac{2}{\pi}H_n, \quad H_n = \sum_{i=1}^{n}\frac{1}{i}$$

### 7. Coupon Collector覆盖时间

经典随机游走覆盖 $n$ 个顶点的期望时间：
$$E[T_{\text{cover}}] = n H_n = n\sum_{i=1}^{n}\frac{1}{i}$$

方差：
$$\text{Var}(T) = n\sum_{i=2}^{n}\frac{i-1}{(n-i+1)^2}$$

量子搜索时间对比：
$$T_{\text{quantum}} = O\left(\sqrt{\frac{n}{d}}\right)$$

### 8. Diophantine约束状态枚举

高维量子行走中，守恒律约束下的允许状态满足：
$$a_1 x_1 + a_2 x_2 + \cdots + a_n x_n = b, \quad 0 \leq x_i \leq m_i$$

采用递归回溯算法枚举所有非负整数解。

### 9. Clenshaw-Curtis稀疏网格

一维CC节点：
$$x_j = \cos\left(\frac{n-j}{n-1}\pi\right), \quad j = 0, \ldots, n-1$$

多维稀疏网格基于层级约束：
$$\sum_{d=1}^{D}\ell_d \leq L_{\max}$$

加权各向异性约束：
$$\sum_{d=1}^{D}\alpha_d \ell_d \leq L_{\max}$$

### 10. Newton-Raphson参数优化

寻找最优硬币角度 $\theta^*$ 使得搜索成功概率最大：
$$\theta_{k+1} = \theta_k - \frac{f'(\theta_k)}{f''(\theta_k)}$$

收敛判据：$|f(\theta)| < \varepsilon$，$|f'(\theta)| > 10^{-8}$，且函数值不发散。

---

## 15个种子项目的融合映射

| 序号 | 种子项目 | 核心算法 | 在合成项目中的角色 |
|------|---------|---------|------------------|
| 1 | `289_diophantine_nd` | N维Diophantine方程求解 | `lattice_states.py`：高维量子行走状态空间的守恒律约束枚举 |
| 2 | `1000_r8to` | Toeplitz矩阵紧凑存储与$O(N^2)$求解 | `matrix_solvers.py`：1D量子行走平移算符的Toeplitz结构高效求解 |
| 3 | `1004_r8vm` | Vandermonde矩阵紧凑存储与快速求解 | `matrix_solvers.py`：量子态振幅从本征值数据的高精度谱重构 |
| 4 | `143_cc_display` | 多维Clenshaw-Curtis网格生成 | `lattice_states.py`：量子行走参数空间的稀疏网格采样与优化 |
| 5 | `580_image_mesh2d` | 2D边界三角剖分 | `geometry_mesh.py`：复杂几何区域上的空间量子行走图结构生成 |
| 6 | `808_nonlin_newton` | Newton-Raphson非线性求根 | `optimization_diagnostics.py`：最优硬币角度与临界gamma的数值优化 |
| 7 | `924_pwc_plot_2d` | 二维分段常数函数 | `quantum_operators.py`：分段常数势场下的CTQW Hamiltonian构造 |
| 8 | `962_r83` | 三对角矩阵CG/循环约化/Jacobi/GS求解 | `matrix_solvers.py`：1D量子行走稳态方程的专用高效求解器族 |
| 9 | `530_hexagon_stroud_rule` | 正六边形Stroud求积与精确矩 | `geometry_mesh.py`/`numerical_quadrature.py`：六边形晶格量子观测量的高精度数值积分 |
| 10 | `152_cg_rc` | 反向通信CG + Wathen有限元测试矩阵 | `matrix_solvers.py`：大规模稀疏量子算符系统的迭代求解 |
| 11 | `667_levels` | 随机等高线采样 | `lattice_states.py`/`optimization_diagnostics.py`：量子概率landscape的随机水平分析 |
| 12 | `017_area_under_curve` | 曲线下面积积分 | `numerical_quadrature.py`：量子概率时间演化曲线的数值积分 |
| 13 | `1112_sphere_cubed_grid` | 球面立方网格生成 | `geometry_mesh.py`：球面拓扑上的量子行走状态空间离散化 |
| 14 | `449_full_deck_simulation` | Coupon Collector问题 | `optimization_diagnostics.py`：经典vs量子覆盖时间的理论对比分析 |
| 15 | `1161_steinerberger` | Steinerberger病态函数与精确积分 | `numerical_quadrature.py`：数值积分算法的高精度验证基准 |

---

## 运行方式

```bash
cd "Synthesis-project-python/155_synth_project"
python main.py
```

程序无需任何命令行参数，自动执行14组实验，涵盖：
1. 1D量子行走与三对角/Toeplitz求解器对比
2. Vandermonde谱重构
3. 2D网格与三角剖分区域空间搜索
4. 六边形晶格搜索与六边形数值积分
5. 球面立方网格CTQW
6. 超立方体高维搜索与Diophantine约束
7. CC稀疏网格参数采样
8. Newton法参数优化
9. Coupon Collector覆盖时间分析
10. Steinerberger病态函数积分测试
11. 反向通信CG与Wathen矩阵
12. 谱分析与搜索复杂度
13. 多维格子量子行走
14. 分段常数势场下的能谱分析

---

## 技术特点

- **边界处理与数值鲁棒性**：所有算符构造均进行厄米性检验、归一化保护、零除防护；矩阵求解器包含奇异性检测与退化回退。
- **无可视化**：严格删除所有图形绘制代码，纯数值计算输出。
- **工程复杂性**：涵盖10个Python模块，超2500行代码，融合组合枚举、稀疏网格、谱分析、多类专用矩阵求解、反向通信迭代、有限元矩阵生成等高难度数值方法。
- **公式-算法-代码一致性**：每个核心公式均在代码中有直接对应实现，并在中文文档中给出完整推导关系。
