# 蛋白质折叠自由能景观的多尺度数值分析系统

## 项目概述

本项目将 **15 个科研代码项目** 的核心算法融合重构，构建了一个面向 **分子动力学：蛋白质折叠自由能景观** 的博士级科研计算系统。

科学问题核心：
> 对一个粗粒化蛋白质模型（12 残基链），从多尺度数值方法角度系统分析其折叠自由能景观（Free Energy Landscape, FEL），包括亚稳态识别、折叠路径分析、过渡态判定、动力学速率估计及溶剂化效应评估。

---

## 一、原项目到科学问题的映射

| 原项目 | 核心算法 | 在合成项目中的角色 |
|--------|---------|-------------------|
| `759_mesh2d_write` | 2D 网格数据 I/O | 反应坐标空间 (Q, RMSD) 的网格生成与导出 |
| `395_fem1d_pack` | 1D FEM / Lagrange 基函数 / Gauss-Legendre 积分 | FEM 求解器的基函数与数值积分组件 |
| `896_polynomial_resultant` | Sylvester 矩阵 / 结式计算 | 势能面多项式约束的临界点与分岔分析 |
| `422_feynman_kac_1d` | Feynman-Kac 路径积分 / 随机游走 | 折叠概率与平均首通时间 (MFPT) 的蒙特卡洛估计 |
| `159_chebyshev` | Chebyshev 插值 / Clenshaw 递推 | 自由能剖面的高精度谱逼近与导数计算 |
| `1126_sphere_quad` | 正二十面体细分 / 球面积分 | 溶剂分子取向分布与 NMR 序参数计算 |
| `396_fem1d_pmethod` | p-version FEM / 正交多项式基 | 稳态 Smoluchowski / Fokker-Planck 方程的高阶求解 |
| `043_asa147` | 不完全 Gamma 函数级数展开 | 亚稳态停留时间分布的统计建模与卡方检验 |
| `232_cube_felippa_rule` | 3D 张量积 Gauss-Legendre 积分 | 构象空间局部配分函数的高精度体积积分 |
| `1231_tet_mesh_boundary` | 四面体网格边界提取 | 自由能盆地边界的自动识别与表面积计算 |
| `850_partition_greedy` | 贪心整数划分 | 并行 MD 负载均衡与蛋白质结构域自动划分 |
| `509_hb_to_msm` | Harwell-Boeing 稀疏矩阵解析 | 弹性网络 Hessian 矩阵的格式兼容层 |
| `997_r8ss` | 对称天际线矩阵存储与运算 | Kirchhoff 矩阵的高效稀疏存储与矩阵-向量乘法 |
| `308_distmesh` | 距离函数网格生成 / Delaunay 优化 | 反应坐标空间的自适应三角网格生成 |
| `1076_sigmoid` | Sigmoid 高阶导数幂级数展开 | 势能平滑截断与介电常数连续过渡函数 |

---

## 二、新增数学物理模型与核心公式

### 2.1 粗粒化蛋白质模型（Go-like 势）

势能函数由三部分组成：

$$
V(\mathbf{r}) = V_{\text{bond}} + V_{\text{native}} + V_{\text{excluded}}
$$

其中：

$$
V_{\text{bond}} = \sum_{i=1}^{N-1} \frac{k_{\text{bond}}}{2} \left( |\mathbf{r}_{i+1} - \mathbf{r}_i| - d_0 \right)^2
$$

$$
V_{\text{native}} = \sum_{\substack{|i-j|>2 \\ d_{ij}^{\text{native}} < r_{\text{cut}}}} \frac{k_{\text{native}}}{2} \left( |\mathbf{r}_i - \mathbf{r}_j| - d_{ij}^{\text{native}} \right)^2
$$

$$
V_{\text{excluded}} = \sum_{\substack{\text{non-native} \\ |\mathbf{r}_i - \mathbf{r}_j| < r_0}} k_{\text{repulse}} (r_0 - |\mathbf{r}_i - \mathbf{r}_j|)^2
$$

### 2.2 反应坐标

**天然接触分数 Q**：

$$
Q = \frac{1}{M} \sum_{(i,j) \in \mathcal{C}} \mathbb{I}\left( |\mathbf{r}_i - \mathbf{r}_j| < \gamma \cdot d_{ij}^{\text{native}} \right)
$$

其中 $\mathcal{C}$ 为天然接触对集合，$M = |\mathcal{C}|$，$\gamma$ 为相对容差。

**均方根偏差 RMSD**：

$$
\text{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} |\mathbf{r}_i - \mathbf{r}_i^{\text{native}}|^2}
$$

**回转半径 Rg**：

$$
R_g^2 = \frac{1}{M_{\text{tot}}} \sum_{i=1}^{N} m_i |\mathbf{r}_i - \mathbf{r}_{\text{cm}}|^2
$$

### 2.3 自由能景观

沿反应坐标 $x$（如 $Q$）的自由能剖面：

$$
F(x) = -k_B T \ln P(x) + C
$$

其中 $P(x)$ 为概率密度，通过构象系综的直方图估计得到。

### 2.4 Chebyshev 谱插值

将自由能函数展开为 Chebyshev 级数：

$$
F(x) \approx \sum_{k=0}^{n-1} c_k T_k\left( \frac{2x - a - b}{b - a} \right) - \frac{1}{2} c_0
$$

节点（第一类零点）：

$$
x_j = \cos\left( \frac{\pi(2j-1)}{2n} \right), \quad j = 1, \dots, n
$$

系数：

$$
c_k = \frac{2}{n} \sum_{j=1}^{n} F(x_j) \cos\left( \frac{\pi k (2j-1)}{2n} \right)
$$

### 2.5 稳态 Smoluchowski 方程（Fokker-Planck）

一维稳态概率密度 $p(x)$ 满足：

$$
\frac{d}{dx} \left[ D \frac{dp}{dx} + \frac{D}{k_B T} p(x) \frac{dF}{dx} \right] = 0
$$

边界条件：$p(x_{\min}) = p_L$，$p(x_{\max}) = p_R$。

采用线性有限元 (P1) 离散，单元刚度矩阵通过 Gauss-Legendre 数值积分计算：

$$
\int_{-1}^{1} f(\xi) \, d\xi \approx \sum_{q=1}^{n_g} w_q f(\xi_q)
$$

### 2.6 Fokker-Planck 特征值问题

求解广义特征值问题：

$$
\mathbf{K} \mathbf{u} = \lambda \mathbf{M} \mathbf{u}
$$

其中 $\mathbf{K}$ 为刚度矩阵，$\mathbf{M}$ 为质量矩阵。最小非零特征值 $\lambda_1$ 对应折叠/解折叠的速率（Kramers 理论的谱近似）。

### 2.7 Feynman-Kac 路径积分

折叠概率的 Feynman-Kac 表示：

$$
P_{\text{fold}}(x_0) = \mathbb{E}\left[ \exp\left( -\int_0^{\tau} \frac{F(X_s)}{k_B T} \, ds \right) \cdot \mathbb{I}\{X_{\tau} = \text{folded}\} \right]
$$

离散随机游走（Euler-Maruyama）：

$$
X_{t+h} = X_t + \sqrt{2Dh} \, Z, \quad Z \sim \mathcal{N}(0,1)
$$

### 2.8 Kramers 速率理论

对于双势阱系统，折叠速率：

$$
k_{\text{Kramers}} = \frac{\omega_m |\omega_b|}{2\pi} \exp\left( -\frac{\Delta F}{k_B T} \right)
$$

其中 $\Delta F$ 为势垒高度，$\omega_m$ 为势阱底部曲率，$\omega_b$ 为势垒顶部曲率（$\omega_b = \sqrt{|F''(x_{\max})|}$）。

### 2.9 弹性网络模型（ENM）

Kirchhoff 矩阵（Gaussian Network Model）：

$$
\Gamma_{ij} = \begin{cases}
-k & \text{if } i \neq j \text{ and } |\mathbf{r}_i - \mathbf{r}_j| < r_c \\
\sum_{l \neq i} k_{il} & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}
$$

均方涨落（MSF）：

$$
\langle \Delta r_i^2 \rangle = k_B T \left[ \boldsymbol{\Gamma}^{-1} \right]_{ii}
$$

### 2.10 多项式结式与临界点分析

Sylvester 矩阵：

$$
\mathbf{S}(p, q) = \begin{bmatrix}
p_m & p_{m-1} & \cdots & p_0 & & \\
& p_m & \cdots & p_1 & p_0 & \\
& & \ddots & & & \ddots \\
q_n & q_{n-1} & \cdots & q_0 & & \\
& q_n & \cdots & q_1 & q_0 & \\
& & \ddots & & & \ddots
\end{bmatrix}
$$

结式：

$$
\text{Res}(p, q) = \det(\mathbf{S})
$$

$\text{Res}(p, q) = 0 \iff p$ 与 $q$ 有公共根。

### 2.11 球面积分

正二十面体细分球面积分。球面三角形面积（Girard 公式 / L'Huilier 定理）：

$$
A = 4 \arctan\sqrt{ \tan\frac{s}{2} \tan\frac{s-a}{2} \tan\frac{s-b}{2} \tan\frac{s-c}{2} }
$$

其中 $a, b, c$ 为球面边长（中心角），$s = (a+b+c)/2$。

NMR 序参数：

$$
S^2 = \frac{1}{2} \langle 3\cos^2\theta - 1 \rangle = \frac{1}{4\pi} \int_{S^2} P_2(\cos\theta) \, d\Omega
$$

### 2.12 三维构象空间积分

局部配分函数：

$$
Z_V = \int_V \exp\left( -\frac{U(\mathbf{r})}{k_B T} \right) d^3r
$$

采用张量积 Gauss-Legendre 规则：

$$
\int_{[a,b]^3} f(x,y,z) \, dxdydz \approx \sum_{i,j,k} w_i w_j w_k \cdot \frac{V}{8} \cdot f(x_i, y_j, z_k)
$$

### 2.13 Sigmoid 平滑截断

平滑截断函数：

$$
S(r) = \sigma\left( \frac{r_c - r}{w} \right) = \frac{1}{1 + \exp\left( -\frac{r_c - r}{w} \right)}
$$

高阶导数展开（McKenna 定理）：

$$
\sigma^{(n)}(x) = \sum_{j=1}^{n+1} c_{n,j} \, \sigma(x)^j, \quad c_{n,k} = \sum_{j=0}^{k} (-1)^j (j+1)^n \binom{k}{j}
$$

介电常数连续过渡：

$$
\varepsilon(r) = \varepsilon_{\text{in}} + (\varepsilon_{\text{out}} - \varepsilon_{\text{in}}) \cdot S(r)
$$

### 2.14 不完全 Gamma 函数

归一化下不完全 Gamma 函数（Algorithm AS 147）：

$$
P(x, p) = \frac{x^p e^{-x}}{\Gamma(p+1)} \sum_{n=0}^{\infty} \frac{x^n}{(p+1)(p+2)\cdots(p+n)}
$$

级数迭代：$c_0 = 1$，$c_n = c_{n-1} \cdot x / (p+n)$，直到 $|c_n / \sum| < \varepsilon$。

停留时间分布（Gamma 分布）：

$$
f(t; \alpha, \beta) = \frac{t^{\alpha-1} e^{-t/\beta}}{\beta^\alpha \Gamma(\alpha)}
$$

### 2.15 贪心划分

负载均衡贪心算法：

$$
x_j = \begin{cases} 0 & \text{if } S_0 < S_1 \\ 1 & \text{otherwise} \end{cases}
$$

时间复杂度：$O(N \log N)$。

---

## 三、项目文件结构

```
111_synth_project/
├── main.py                         # 统一入口，零参数运行
├── reaction_coordinates.py         # 反应坐标计算与网格 I/O
├── chebyshev_pes.py               # Chebyshev 谱插值势能面
├── fem1d_pmethod_solver.py        # p-version FEM 求解扩散方程
├── feynman_kac_integrator.py      # Feynman-Kac 路径积分
├── sphere_quad.py                 # 球面积分与取向采样
├── cube_integrator.py             # 3D 构象空间高斯积分
├── tet_mesh_surface.py            # 四面体网格边界提取
├── distmesh_generator.py          # 距离函数网格生成
├── sparse_hessian.py              # 稀疏 Hessian / ENM / NMA
├── polynomial_analysis.py         # 多项式结式临界分析
├── sigmoid_switch.py              # Sigmoid 平滑截断
├── gamma_stats.py                 # 不完全 Gamma 统计
├── partition_optimizer.py         # 贪心划分优化
├── README_博士级合成说明.md        # 本文档
└── output/                         # 输出目录（运行后生成）
    ├── free_energy_profile.txt
    ├── smoluchowski_steady.txt
    ├── fp_eigenvalues.txt
    ├── feynman_kac_fold_prob.txt
    ├── sigmoid_switch.txt
    ├── mean_square_fluctuation.txt
    ├── rc_grid_nodes.txt
    ├── rc_grid_elements.txt
    ├── distmesh_nodes.txt
    ├── distmesh_elements.txt
    └── summary_report.txt
```

---

## 四、合成后的项目能够解决什么科学问题

1. **自由能景观重构**：从构象系综计算沿任意反应坐标（Q、RMSD、Rg）的一维/二维自由能面。
2. **扩散-反应动力学**：求解 Smoluchowski / Fokker-Planck 方程，获得稳态概率密度和特征速率。
3. **路径积分验证**：通过 Feynman-Kac 蒙特卡洛路径积分独立验证折叠概率和首通时间。
4. **过渡态与速率分析**：提取势垒参数，应用 Kramers 理论估计折叠速率常数。
5. **弹性网络正常模式**：构建 Kirchhoff 矩阵，计算低频集体运动模式和残基涨落。
6. **势能面代数拓扑**：用多项式结式检测势能面临界点、分岔点和势阱合并。
7. **溶剂化效应评估**：球面积分计算取向分布和 NMR 序参数；Sigmoid 介电函数模拟隐式溶剂。
8. **构象空间热力学**：3D 高斯积分计算局部配分函数，评估亚稳态相对稳定性。
9. **结构域识别**：贪心划分算法自动识别蛋白质结构域和负载均衡分配。
10. **停留时间统计**：不完全 Gamma 函数建模亚稳态寿命分布，支持卡方显著性检验。

---

## 五、如何运行

### 环境要求
- Python >= 3.8
- NumPy
- SciPy

### 运行命令

```bash
cd 111_synth_project
python main.py
```

程序无需任何输入参数，运行后将：
1. 构建 12 残基粗粒化蛋白质模型
2. 执行 Metropolis-Hastings 构象采样（500 样本）
3. 依次调用所有数值分析模块
4. 在 `output/` 目录输出数据文件和摘要报告

### 预期运行时间
- 普通 CPU：约 10–20 秒

---

## 六、关键设计决策与鲁棒性处理

1. **数值稳定性**：
   - Chebyshev 插值使用 Clenshaw 递推避免 Runge 效应
   - 不完全 Gamma 函数采用对数形式防止溢出
   - Sigmoid 高阶导数使用幂级数展开避免数值微分不稳定

2. **边界处理**：
   - 布朗运动随机游走的软边界反射
   - FEM 的 Dirichlet 边界条件严格施加
   - 概率密度截断保护（`np.maximum(p, 1e-12)`）

3. **工程鲁棒性**：
   - 所有模块独立可测试
   - 输入参数合法性校验（维度匹配、正定性等）
   - Kramers 速率提取的失败保护（异常时输出警告而非崩溃）

4. **无可视化**：
   - 所有输出均为文本/数据文件，删除全部图形绘制代码

---

*本项目为博士级科学计算合成项目，代码复杂度与数学深度均达到前沿科研水准。*
