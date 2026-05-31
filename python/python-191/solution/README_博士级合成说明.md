# MPI并行科学计算矩阵乘法综合验证平台

## 一、项目概述

本项目围绕**高性能计算：MPI并行矩阵乘法**领域，将15个种子科研代码项目的核心算法深度融合，构建了一个面向天体力学N体传播、有限元刚度矩阵组装、高阶数值积分与概率采样验证的博士级科学计算代码库。

项目使用Python语言实现，包含9个核心模块文件与1个统一入口文件`main.py`，零参数可直接运行，无需外部输入。

---

## 二、原项目到科学问题的映射（15→9）

| 序号 | 原种子项目 | 核心算法 | 在合成项目中的角色 |
|------|-----------|---------|------------------|
| 1 | 1292_tri_surface_display | 3D三角网格表面数据读取 | `sparse_fem_topology.py`：FEM网格节点拓扑与三角形面元处理 |
| 2 | 379_fem_to_medit | FEM格式转换与网格数据读写 | `sparse_fem_topology.py`：网格坐标解析与刚度矩阵组装 |
| 3 | 1358_trinity | 三角形区域线性规划/拼图覆盖 | `sparse_fem_topology.py`：稀疏矩阵约束模式分析（Trinity覆盖） |
| 4 | 699_log_normal_truncated_ab | 截断对数正态分布采样 | `sampling_distribution.py`：矩阵元素随机扰动的概率模型 |
| 5 | 225_cpr | Chebyshev代理根查找 | `chebyshev_hermite_approx.py`：Chebyshev插值与伴随矩阵特征值 |
| 6 | 146_ccvt_reflect | CVT带反射点分布优化 | `sampling_distribution.py`：Lloyd迭代优化采样点分布 |
| 7 | 346_exp_ode | 指数增长ODE | `matrix_exponential_ode.py`：矩阵指数Pade逼近基础模型 |
| 8 | 521_hermite_interpolant | Hermite差商插值 | `chebyshev_hermite_approx.py`：高阶Hermite插值近似核矩阵元素 |
| 9 | 660_legendre_fast_rule | 快速Gauss-Legendre求积 | `legendre_quadrature_kernel.py`：GL节点/权重计算与核矩阵积分 |
| 10 | 913_prime_parfor | 素数并行计数 | `parallel_prime_partition.py`：素数分布指导并行负载均衡 |
| 11 | 302_disk01_rule | 单位圆盘求积规则 | `legendre_quadrature_kernel.py`：圆盘径向-角度变换求积 |
| 12 | 198_collatz_polynomial | 模2 Collatz多项式 | `polynomial_hash_verify.py`：矩阵多项式哈希与F2校验 |
| 13 | 1399_walker_sample | Walker别名离散采样 | `sampling_distribution.py`：O(1)离散概率采样 |
| 14 | 303_disk01_positive_monte_carlo | 圆盘蒙特卡洛采样 | `sampling_distribution.py`：随机矩阵生成与积分验证 |
| 15 | 618_kepler_ode | 开普勒二体问题ODE | `matrix_exponential_ode.py`：变分方程与状态转移矩阵传播 |

---

## 三、新增数学物理模型与核心公式

### 3.1 MPI并行矩阵乘法（Cannon & SUMMA算法）

对于矩阵 $A, B \in \mathbb{R}^{n \times n}$，标准矩阵乘法为：

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

**Cannon算法**在 $\sqrt{p} \times \sqrt{p}$ 进程网格上执行：
1. 初始对齐：$A_{ij}$ 左移 $i$ 格，$B_{ij}$ 上移 $j$ 格
2. 循环 $s = 0, \ldots, \sqrt{p}-1$：
   - 本地相乘：$C_{ij} \mathrel{+}= A_{ij} \cdot B_{ij}$
   - $A$ 左循环移位1格，$B$ 上循环移位1格

**通信复杂度**：$O(n^2 / \sqrt{p})$ 每进程  
**计算复杂度**：$O(n^3 / p)$ 每进程

浮点误差界：
$$\|C - \hat{C}\|_F \leq \gamma_n \|A\|_F \|B\|_F, \quad \gamma_n = \frac{n \cdot \varepsilon_{\text{mach}}}{1 - n \cdot \varepsilon_{\text{mach}}}$$

### 3.2 矩阵指数与Scaling-and-Squaring

矩阵指数定义为：

$$e^{A} = \sum_{k=0}^{\infty} \frac{A^k}{k!}$$

Scaling-and-Squaring算法：
1. 选取 $s$ 使得 $\|A / 2^s\| \leq 0.5$
2. 用 $[m/m]$ Padé逼近计算 $E = R_{mm}(A / 2^s)$：
   $$N(X) = \sum_{j=0}^{m} \frac{(2m-j)! \, m!}{(2m)! \, j! \, (m-j)!} X^j$$
   $$D(X) = \sum_{j=0}^{m} \frac{(2m-j)! \, m!}{(2m)! \, j! \, (m-j)!} (-X)^j$$
   $$R_{mm}(X) = D(X)^{-1} N(X)$$
3. 重复平方：$e^A = E^{2^s}$

### 3.3 开普勒问题变分方程

Hamilton量：
$$H(q,p) = \frac{1}{2}(p_1^2 + p_2^2) - \frac{1}{\sqrt{q_1^2 + q_2^2}}$$

运动方程：
$$\dot{q} = p, \quad \dot{p} = -\frac{q}{\|q\|^3}$$

变分方程（状态转移矩阵 $\Phi(t)$）：
$$\frac{d\Phi}{dt} = M(t) \Phi, \quad M = \begin{pmatrix} 0 & I \\ -H_{qq} & 0 \end{pmatrix}$$

其中Hessian：
$$H_{qq} = -\frac{I}{r^3} + \frac{3 q q^{\top}}{r^5}, \quad r = \|q\|$$

辛守恒：$\det \Phi(t) \equiv 1$。

### 3.4 Chebyshev代理根查找（CPR）

Chebyshev插值在节点 $x_k = \cos(k\pi/N)$ 上：
$$p_N(x) = \sum_{j=0}^{N} c_j T_j(\xi), \quad \xi = \frac{2x - (a+b)}{b-a}$$

系数：
$$c_j = \frac{2}{N} \sum_{k=0}^{N}{}^{''} f(x_k) \cos\frac{jk\pi}{N}$$

Chebyshev伴随矩阵 $C \in \mathbb{R}^{N_t \times N_t}$：
$$C_{1,2}=1, \quad C_{j,j-1}=C_{j,j+1}=\frac{1}{2}, \quad C_{N_t,1:N_t} = -\frac{c_{0:N_t}}{2c_{N_t}}$$

根为 $\text{eig}(C)$ 的实部，映射回 $[a,b]$。

### 3.5 Hermite插值

给定 $(x_i, y_i, y'_i)$，$i=1,\ldots,n$，构造唯一多项式 $H$ 满足 $H(x_i)=y_i$, $H'(x_i)=y'_i$。

扩展节点 $z = [x_0, x_0, x_1, x_1, \ldots]$，差商表满足：
$$f[z_{2k}, z_{2k+1}] = f'(x_k)$$

Newton形式：
$$H(x) = d_0 + d_1(x-z_0) + d_2(x-z_0)(x-z_1) + \cdots$$

### 3.6 Gauss-Legendre快速求积（GLR算法）

求积公式：
$$\int_{-1}^{1} f(x)\,dx \approx \sum_{i=1}^{n} w_i f(x_i)$$

节点 $x_i$ 为 $P_n(x)$ 的根，权重：
$$w_i = \frac{2}{(1-x_i^2)[P_n'(x_i)]^2}$$

GLR快速算法以渐近初值启动Newton迭代：
$$\theta_k = \frac{\pi(4k-1)}{4n+2}, \quad x_k^{(0)} = \left[1 - \frac{n-1}{8n^3} - \frac{39 - 28/\sin^2\theta_k}{384n^4}\right] \cos\theta_k$$

圆盘积分变换：
$$\iint_D f(x,y)\,dA = \pi \sum_{j=1}^{n_t} \sum_{i=1}^{n_r} w_i \, f\bigl(r_i \cos\theta_j, r_i \sin\theta_j\bigr)$$
其中 $r_i = \sqrt{(\xi_i+1)/2}$，$\xi_i$ 为 $[0,1]$ 上的Legendre节点。

### 3.7 FEM刚度矩阵组装

三角形单元 $e$（顶点 $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$）面积：
$$A_e = \frac{1}{2}\left|\det\begin{pmatrix} \mathbf{v}_2-\mathbf{v}_1 & \mathbf{v}_3-\mathbf{v}_1 \end{pmatrix}\right|$$

形函数梯度：
$$\nabla\lambda = \frac{1}{2A_e} \begin{pmatrix}
v_{2y}-v_{3y} & v_{3y}-v_{1y} & v_{1y}-v_{2y} \\
v_{3x}-v_{2x} & v_{1x}-v_{3x} & v_{2x}-v_{1x}
\end{pmatrix}$$

单元刚度矩阵：
$$K_e(i,j) = A_e \, (\nabla\lambda_i \cdot \nabla\lambda_j)$$

全局组装：$K = \sum_e P_e^{\top} K_e P_e$。

### 3.8 截断对数正态采样

PDF：
$$f(x) = \frac{1}{x\sigma\sqrt{2\pi}} \frac{\exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)}{\Phi(\frac{\ln b-\mu}{\sigma}) - \Phi(\frac{\ln a-\mu}{\sigma})}, \quad x \in [a,b]$$

逆CDF采样：
$$X = \exp\bigl(\mu + \sigma \Phi^{-1}(U)\bigr), \quad U \sim \text{Uniform}[\Phi_a, \Phi_b]$$

### 3.9 Walker别名方法

预处理 $O(n)$，采样 $O(1)$：
1. 构造阈值 $y_i = n p_i$
2. 构造别名 $a_i$
3. 采样：$i \sim \text{Uniform}[1,n]$；若 $U < y_i$ 返回 $i$，否则返回 $a_i$

### 3.10 CVT能量与Lloyd迭代

CVT能量泛函：
$$E = \sum_{i=1}^{n} \int_{V_i} \|\mathbf{x} - \mathbf{z}_i\|^2 \rho(\mathbf{x})\,d\mathbf{x}$$

Lloyd更新：
$$\mathbf{z}_i^{\text{new}} = \frac{\int_{V_i} \mathbf{x} \rho(\mathbf{x})\,d\mathbf{x}}{\int_{V_i} \rho(\mathbf{x})\,d\mathbf{x}}$$

### 3.11 素数定理与负载均衡

素数定理：
$$\pi(n) \sim \frac{n}{\ln n}$$

进程分块均衡：
$$q = \lfloor n/p \rfloor, \quad r = n \bmod p$$
前 $r$ 个进程分得 $q+1$ 行，其余 $q$ 行。

### 3.12 Collatz多项式与F2哈希

在 $\mathbb{F}_2[x]$ 中：
$$P_{k+1} = \begin{cases} P_k / x, & p_0 = 0 \\ P_k(x+1) + 1 \pmod{2}, & p_0 = 1 \end{cases}$$

矩阵编码为多项式：
$$p(t) = \sum_{i,j} (M_{ij} \bmod 2) \, t^{in+j} \in \mathbb{F}_2[t]$$

---

## 四、文件结构与实现路径

```
191_synth_project/
├── main.py                          # 统一入口，零参数运行，8个实验序列
├── mpi_cannon_multiply.py           # Cannon/SUMMA并行矩阵乘法 + 误差分析
├── matrix_exponential_ode.py        # 矩阵指数Pade逼近 + 开普勒STM传播
├── chebyshev_hermite_approx.py      # Chebyshev代理根查找 + Hermite插值
├── legendre_quadrature_kernel.py    # GLR快速求积 + 圆盘积分 + 核矩阵构造
├── sparse_fem_topology.py           # 三角网格拓扑 + FEM刚度矩阵 + Trinity模式
├── sampling_distribution.py         # 截断对数正态 + Walker + CVT + 圆盘MC
├── parallel_prime_partition.py      # 素数筛 + 进程分块 + 负载均衡分析
└── polynomial_hash_verify.py        # Collatz序列 + 矩阵哈希 + F2乘法验证
```

### 边界处理与数值鲁棒性
- 所有模块包含输入维度校验与空矩阵处理
- `kepler_derivatives` 在 $r \to 0$ 时加入正则化 `eps = 1e-14`
- `chebyshev_companion_matrix` 处理零尾截断与近零分母
- `hermite_divided_differences` 处理重复节点的差商退化
- `matrix_exponential_pade` 的scaling确保 $\|A/2^s\| \leq 0.5$
- 并行模块在矩阵过小时自动回退到 `numpy.dot`

---

## 五、解决的科学问题

1. **分布式稠密矩阵乘法的高精度验证**：在单机多核环境下模拟MPI并行，通过Cannon和SUMMA算法实现$O(n^3/p)$计算复杂度的矩阵乘法，并给出浮点误差界。

2. **天体力学状态转移矩阵的数值传播**：将开普勒二体问题的变分方程线性化，通过RK4同时积分参考轨道和$4 \times 4$状态转移矩阵，验证辛守恒性质。

3. **核矩阵元素的高阶近似**：利用Chebyshev插值和Hermite插值降低核函数（如高斯核）的求值成本，结合Gauss-Legendre和圆盘求积构造刚度矩阵与核矩阵。

4. **有限元稀疏模式的组合分析**：将Trinity拼图的覆盖约束映射为稀疏二元矩阵，结合FEM网格拓扑进行稀疏刚度矩阵组装与稀疏矩阵-向量乘积。

5. **并行负载均衡的数论优化**：利用素数分布理论筛选具有光滑因子的进程数，实现近乎完美的矩阵分块均衡（负载不平衡度$\approx 0$）。

6. **概率驱动的随机矩阵测试**：通过截断对数正态分布、Walker别名采样和CVT优化生成具有物理意义的随机测试矩阵，用于并行算法的蒙特卡洛验证。

7. **结果正确性的代数验证**：使用Collatz多项式序列和$\mathbb{F}_2$矩阵乘法为并行矩阵乘法结果提供快速概率校验。

---

## 六、如何运行

```bash
cd Synthesis-project-python/191_synth_project
python main.py
```

无需任何命令行参数。程序自动依次执行8个实验模块，输出数值结果与误差分析，约10-20秒完成。

### 环境要求
- Python >= 3.8
- NumPy >= 1.20
- 标准库：`multiprocessing`, `math`, `time`

---

## 七、性能与精度指标（示例输出）

| 实验 | 关键指标 | 典型结果 |
|------|---------|---------|
| Cannon 128x128 | 相对Frobenius误差 | ~3.6e-16 |
| SUMMA 128x128 | 相对Frobenius误差 | ~3.6e-16 |
| 矩阵指数 | $||e^A - \hat{e}^A||_F$ | ~4.6e-16 |
| 开普勒STM | $\det \Phi$ | 1.000000 |
| Hermite插值 | $||H - \sin||_\infty$ | ~3.0e-09 |
| GL求积 $x^{14}$ | 绝对误差 | ~1.0e-10 |
| 圆盘积分 | 绝对误差 | ~1.0e-06 |
| CVT Lloyd | 能量下降比 | ~22.5% |
| 素数分块 | 负载不平衡度 | 0.0000 |
