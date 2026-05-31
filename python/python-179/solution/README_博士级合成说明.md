# 博士级合成项目 179：低秩矩阵近似与张量分解

## 1. 项目概述

本项目围绕**计算数学：低秩矩阵近似与张量分解**这一前沿科学领域，将 15 个原始科研代码项目的核心算法融合为一个统一的博士级计算框架。

### 1.1 合成的科学问题

**问题名称**：参数化反应-扩散-形态演化系统的高维响应面低秩张量近似

在计算数学与科学工程领域，高维参数化偏微分方程（Parametric PDE）的求解是核心挑战之一。考虑如下一维轴对称参数化反应-扩散方程：

$$
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + R(u; \theta), \quad x \in \Omega(B,L,w,D_{\text{egg}}), \quad t \in [0, T]
$$

带有齐次 Neumann 边界条件：

$$
\left.\frac{\partial u}{\partial n}\right|_{\partial\Omega} = 0
$$

其中参数向量 $\theta = (k_1, k_2, \mu, \alpha, B, L, w, D_{\text{egg}})$ 涵盖化学反应速率、非线性振荡强度、形态几何参数等多物理场变量。对参数网格上的每一个离散点执行 FEM 求解，得到的高维解张量为：

$$
\mathcal{U} \in \mathbb{R}^{n_x \times n_t \times n_{k_1} \times n_{k_2} \times n_\mu \times n_\alpha \times n_B \times n_w \times n_{D_{\text{egg}}}}
$$

本项目的核心目标是利用**张量列车（Tensor Train, TT）分解**对该高维张量进行低秩近似压缩，将存储复杂度从 $O(n^d)$ 降至 $O(d \cdot n \cdot r^2)$，同时保持可控的近似误差。

---

## 2. 原项目到科学问题的映射

| 原项目 | 核心算法/概念 | 在合成项目中的角色 |
|--------|--------------|-------------------|
| **502_hand_data** | 2D 形状采集与多边形数据 | 生成手部轮廓的 Fourier 描述子低秩近似，用于复杂边界生成与 SVD 主成分分析 |
| **738_matrix_assemble_parfor** | Hilbert 矩阵并行组装 | 构造低秩测试矩阵，验证随机化 SVD 的近似精度；Hilbert 矩阵奇异值指数衰减特性是低秩近似的典范 Benchmark |
| **1018_reaction_twoway_ode** | 双向线性化学反应 ODE | 扩展为空间分布反应源项 $R_{\text{two-way}}(u) = -k_1 u + k_2(1-u)$，描述化学动力学 |
| **713_maple_area** | MC/QMC 面积估计 | 扩展为高维 Monte Carlo 与 Quasi-Monte Carlo 积分器，用于张量 Frobenius 范数估计与误差分析 |
| **513_hello** | 系统初始化 | 升级为科研级日志系统 `Logger` 与数值安全工具（安全求逆、鲁棒平方根、边界裁剪） |
| **962_r83** | 三对角矩阵求解器（CG/循环约化/GS/Jacobi） | FEM 离散化后得到的三对角质量/刚度矩阵的求解核心，支持 IMEX 时间步进 |
| **198_collatz_polynomial** | GF(2) 上 Collatz 多项式序列 | 构造 Hankel 张量，验证张量秩与多项式递推长度之间的关系（Prony 方法） |
| **045_asa159** | 随机列联表生成 | 为 NMF/NTF 提供满足边际约束的非负随机初始化，避免零因子问题 |
| **377_fem_neumann** | 一维 FEM 反应扩散 | 核心空间离散化模块：分段线性 hat 函数、质量矩阵 $M$、刚度矩阵 $K$、Neumann 边界处理 |
| **1386_vanderpol_ode** | 范德波尔振子 | 扩展为范德波尔型非线性反应项 $R_{\text{vdp}}(u) = \mu(1-u^2)u$，与化学动力学混合构成复杂源项 |
| **007_annulus_distance** | 环形域随机采样 | 随机化 SVD 的范围寻找（Range Finder）中的几何采样策略，用于矩阵草图 |
| **1318_triangle_symq_rule_original** | 三角形对称求积规则 | 二维截面泛函积分的三角形 Gauss 求积实现 |
| **769_mm_io** | Matrix Market I/O | 张量坐标格式（COO）的读写扩展，支持对称矩阵半存储 |
| **1048_rref2** | RREF 计算与秩判定 | 张量展开的数值秩分析、Multilinear Rank 与 TT-rank 估计 |
| **093_bird_egg** | 鸟类蛋形参数化公式 | 通用蛋形公式生成参数化计算域，Chebyshev 节点分布优化插值稳定性 |

---

## 3. 新增数学物理模型与核心公式

### 3.1 参数化反应-扩散方程

弱形式（Galerkin 投影）：求 $u \in H^1(\Omega)$ 使得对所有测试函数 $v \in H^1(\Omega)$ 有

$$
\left(\frac{\partial u}{\partial t}, v\right)_{L^2} + D \left(\nabla u, \nabla v\right)_{L^2} = \left(R(u;\theta), v\right)_{L^2}
$$

其中 $R(u;\theta)$ 为混合反应源项：

$$
R(u; k_1, k_2, \mu, \alpha) = \alpha \cdot \underbrace{(-k_1 u + k_2(1-u))}_{\text{线性化学动力学}} + (1-\alpha) \cdot \underbrace{\mu(1-u^2)u}_{\text{范德波尔非线性振荡}}
$$

### 3.2 FEM 离散化

对一维均匀/非均匀网格 $\{x_i\}_{i=0}^{N-1}$，分段线性 hat 函数 $\phi_i$ 构成有限维子空间 $V_h = \text{span}\{\phi_i\}$。单元局部矩阵为：

$$
M_{\text{loc}}^{(i)} = \frac{h_i}{6} \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}, \quad
K_{\text{loc}}^{(i)} = \frac{D}{h_i} \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}
$$

组装后得到全局三对角矩阵，以紧凑 R83 格式存储：

$$
A = \begin{pmatrix} a_{-1}^{(1)} & a_{-1}^{(2)} & \cdots & a_{-1}^{(N-1)} \\ a_0^{(0)} & a_0^{(1)} & \cdots & a_0^{(N-1)} \\ a_1^{(0)} & a_1^{(1)} & \cdots & a_1^{(N-2)} \end{pmatrix}
$$

### 3.3 IMEX 时间离散化

采用一阶隐式-显式 Euler 格式（扩散隐式、反应显式）：

$$
(M + \Delta t \, D K) \, u^{n+1} = M u^n + \Delta t \, M R(u^n)
$$

左侧 $(M + \Delta t D K)$ 为对称正定三对角矩阵，使用共轭梯度法（CG）求解，理论上 $N$ 步内精确收敛。

### 3.4 张量列车（TT）分解

对 $d$ 阶张量 $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d}$，TT 分解表示为：

$$
\mathcal{A}(i_1, i_2, \ldots, i_d) = \sum_{\alpha_0,\ldots,\alpha_d} G_1(i_1)_{\alpha_0,\alpha_1} G_2(i_2)_{\alpha_1,\alpha_2} \cdots G_d(i_d)_{\alpha_{d-1},\alpha_d}
$$

其中 $r_0 = r_d = 1$，$r_k$ 为 TT-ranks，核心张量 $G_k \in \mathbb{R}^{r_{k-1} \times n_k \times r_k}$。

**TT-SVD 算法（Oseledets, 2011）**：逐次对左展开矩阵执行截断 SVD：

$$
A^{(k)} = U_k \Sigma_k V_k^T, \quad r_k = \min\{j : \sigma_j > \varepsilon \|\Sigma_k\|_F\}
$$

**误差界**：

$$
\|\mathcal{A} - \mathcal{A}_{\text{TT}}\|_F^2 \leq \sum_{k=1}^{d-1} \sum_{j > r_k} \sigma_j^2(A^{(k)})
$$

### 3.5 随机化 SVD

对矩阵 $A \in \mathbb{R}^{m \times n}$，随机化范围寻找（Halko, Martinsson, Tropp, 2011）：

1. 生成高斯随机矩阵 $\Omega \in \mathbb{R}^{n \times (k+p)}$
2. 执行 Power Iteration：$Y = (AA^T)^q A \Omega$（$q=2$）
3. QR 分解：$Y = QR$，$Q$ 张成近似范围
4. 小矩阵 SVD：$B = Q^T A = U_B \Sigma V^T$
5. 输出 $U = Q U_B$

误差界（$p \geq 2$）：

$$
\mathbb{E}\|A - QQ^T A\| \leq \left[1 + 4\frac{\sqrt{(k+p)p}}{p-1}\right] \sigma_{k+1} + \cdots
$$

### 3.6 Hankel 张量与 Prony 方法

对序列 $\{s_k\}_{k=0}^{N-1}$，$d$ 阶 Hankel 张量定义为：

$$
\mathcal{H}_{i_1, i_2, \ldots, i_d} = s_{i_1 + i_2 + \cdots + i_d}
$$

若序列满足 $p$ 阶线性递推 $s_{k+p} = \sum_{j=0}^{p-1} c_j s_{k+j}$，则其 Hankel 矩阵的秩不超过 $p$。Collatz 多项式序列在 GF(2) 上的截断提供了构造低秩 Hankel 张量的非平凡例子。

### 3.7 Quasi-Monte Carlo 积分

对 $d$ 维积分，Hammersley 低差异序列的星差异满足：

$$
D_n^* = O\left(\frac{(\log n)^{d-1}}{n}\right)
$$

远优于标准 Monte Carlo 的 $O(1/\sqrt{n})$，在中等维度（$d \leq 10$）下具有显著优势。

### 3.8 通用蛋形参数化公式

基于 Narushin et al. (2021) 的通用蛋形公式，半高度轮廓为：

$$
y(x) = \frac{B}{2} \sqrt{\frac{L^2 - 4x^2}{L^2 + 8wx + 4w^2}}
$$

其中 $B$ 为最大宽度，$L$ 为长度，$w$ 为最大宽度偏移量。Chebyshev 节点分布：

$$
x_i = \frac{a+b}{2} + \frac{b-a}{2} \cos\left(\frac{2i+1}{2n}\pi\right), \quad i = 0, \ldots, n-1
$$

最小化多项式插值的 Lebesgue 常数，抑制 Runge 现象。

---

## 4. 合成项目文件结构

```
179_synth_project/
├── main.py                           # 统一入口，零参数运行
├── system_utils.py                   # 日志、数值安全、边界处理
├── tensor_io.py                      # 张量 Matrix Market I/O（原 769_mm_io）
├── domain_generator.py               # 参数化域生成（原 502_hand_data + 093_bird_egg）
├── tridiagonal_solver.py             # R83 格式求解器（原 962_r83）
├── fem_discretization.py             # 一维 FEM 离散化（原 377_fem_neumann + 1318_triangle_symq_rule）
├── reaction_kinetics.py              # 反应动力学源项（原 1018_reaction_twoway_ode + 1386_vanderpol_ode）
├── randomized_sketching.py           # 随机采样与低秩近似（原 007_annulus_distance + 738_matrix_assemble_parfor）
├── quadrature_integrator.py          # 高维数值积分（原 713_maple_area）
├── rank_analysis.py                  # RREF 秩分析与 Hankel 张量（原 1048_rref2 + 198_collatz_polynomial）
├── nmf_initializer.py                # 非负张量初始化（原 045_asa159）
├── tensor_train_decomposition.py     # TT 分解核心算法
├── pde_solver.py                     # 参数化 PDE 求解器与解张量构建
├── README_博士级合成说明.md          # 本说明文档
├── solution_tensor_coordinate.mm     # 输出张量切片（Matrix Market 坐标格式）
└── fem_mass_matrix_symmetric.mm      # 输出对称 FEM 质量矩阵
```

---

## 5. 运行方式

```bash
cd Synthesis-project-python/179_synth_project
python main.py
```

程序将自动执行以下 12 个步骤：

1. **系统初始化**：配置机器精度、数值秩阈值与日志系统
2. **参数化域生成**：基于蛋形公式与手部轮廓 Fourier 描述子生成计算域
3. **三对角求解器验证**：CG、循环约化、Jacobi、Gauss-Seidel 四种算法的精度对比
4. **随机化 SVD 验证**：Hilbert 矩阵低秩近似与环形域随机采样
5. **反应动力学验证**：双向反应精确解与范德波尔非线性项
6. **FEM 离散化验证**：质量/刚度矩阵组装与 $L^2$ 范数计算
7. **高维积分验证**：3D Gauss 积分的 Monte Carlo vs Quasi-Monte Carlo 对比
8. **秩分析**：Hilbert 矩阵 RREF 秩、Collatz Hankel 张量 multilinear rank 与 TT-rank
9. **非负初始化**：列联表启发式 NMF 因子生成
10. **参数化 PDE 批量求解**：在 288 组参数组合上求解反应-扩散方程
11. **TT 压缩**：对 9 维解张量执行 TT-SVD 分解，输出压缩比与近似误差
12. **结果输出**：Matrix Market 格式文件导出

---

## 6. 边界处理与数值鲁棒性

本项目在以下层面实现了工程级边界处理：

- **参数边界裁剪**：`clip_to_range` 确保所有物理参数（蛋形 $B, w$，反应速率 $k_1, k_2$ 等）处于合法区间
- **安全求逆**：`safe_inv` 对接近零的元素返回 0 而非 `inf`，避免数值爆炸
- **鲁棒平方根**：`robust_sqrt` 对负数输入返回 0，防止复数污染实数计算
- **FEM 非负约束**：时间步进后对浓度 $u$ 执行 `[0, 1]` 裁剪，保证物理意义
- **数值秩阈值**：基于 $\sqrt{\varepsilon}$ 的阈值判定，自动适应双精度浮点环境
- **TT 秩自适应**：TT-SVD 根据奇异值衰减自动确定每层的截断秩，避免过拟合或欠拟合

---

## 7. 科学意义与应用前景

本项目合成的计算框架可直接应用于以下前沿科学问题：

- **参数化 PDE 的 surrogate modeling**：通过 TT 分解压缩高维参数空间响应面，实现实时查询与不确定性量化（UQ）
- **多尺度反应-扩散系统的模型降阶**：结合 POD（本征正交分解）与 TT，构建从微观化学动力学到宏观形态演化的跨尺度耦合模型
- **生物形态计量学的低秩表示**：蛋形参数化与手部轮廓的 Fourier-SVD 方法可扩展至任意生物形状的分类与生成
- **张量网络在量子化学中的应用**：TT 分解的核心算法（左/右正交化、ALS 优化）与量子多体物理中的 DMRG 方法数学等价，具备向量子化学转移的潜力

---

## 8. 质量检查清单

- [x] 原目录未被修改
- [x] 合成后项目为 Python 语言
- [x] 新目录完整包含合成项目（13 个 `.py` 文件 + 文档）
- [x] 统一的博士级科学计算问题已落地为可执行代码
- [x] **15 个输入项目均已真实融入**，无遗漏、无挂名
- [x] `main.py` 零参数运行通过且无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档包含大量数学公式与清晰推导
- [x] 中文说明文档已生成
- [x] 无可视化代码
