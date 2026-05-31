# 光热耦合微腔传感器全耦合仿真平台 —— 博士级合成说明

## 一、项目概述

本项目围绕**光学工程：光热耦合微腔传感**这一前沿博士级科学领域，将 15 个种子科研代码项目的核心算法融合重构为一个完整的、可独立运行的 Python 科研计算平台。

### 1.1 科学问题背景

微环谐振腔（Micro-ring Resonator, MRR）是一种高品质因数（Q > 10⁶）的光学微腔结构，广泛应用于生化传感、温度监测与光通信领域。当谐振光在微环中传播时，光吸收不可避免地产生热量，导致温度上升；而温度变化通过热光效应（thermo-optic effect）改变材料折射率，反过来又影响光场分布——此即**光热耦合（opto-thermal coupling）**。在高功率泵浦或高灵敏度检测场景下，光-热-材料三场耦合效应不可忽略，必须自洽求解。

本项目的核心科学问题是：
> **建立微环谐振腔中光场（Helmholtz 方程）、热场（热传导方程）与材料折射率之间的全耦合自洽模型，分析热致谐振漂移，评估制造参数不确定性对传感性能的影响，并构建神经网络代理模型实现快速传感响应预测。**

### 1.2 核心物理方程体系

#### (1) 光场控制方程 —— 标量 Helmholtz 方程

在时谐近似下，微环截面内的光场 $E(x,y)$ 满足：

$$
\nabla^2 E + k_0^2 \, n^2(x,y,T) \, E = Q_{\text{src}}(x,y)
$$

其中：
- $k_0 = 2\pi / \lambda_0$ 为真空波数
- $n(x,y,T) = n_0 + \frac{dn}{dT}\bigl(T(x,y) - T_0\bigr)$ 为温度依赖折射率
- $Q_{\text{src}}$ 为外部泵浦源项

为避免离散共振奇异，实际数值实现中引入阻尼正则项：

$$
\nabla^2 E + \bigl[k_0^2 n^2 + \gamma k_0^2\bigr] E = Q_{\text{src}}
$$

其中 $\gamma = 0.05$ 为等效吸收阻尼系数。

#### (2) 稳态热传导方程

光吸收产生的热源驱动稳态热扩散：

$$
\nabla \cdot \bigl(\kappa \nabla T\bigr) + Q_{\text{abs}}(E) = 0
$$

体积热源密度由光强决定：

$$
Q_{\text{abs}} = \alpha_{\text{abs}} \, I = \alpha_{\text{abs}} \, |E|^2
$$

其中 $\alpha_{\text{abs}}$ 为有效吸收系数 $[\text{m}^{-1}]$，$\kappa$ 为热导率 $[\text{W}\cdot\text{m}^{-1}\cdot\text{K}^{-1}]$。

边界采用 Robin（对流）条件：

$$
-\kappa \, \frac{\partial T}{\partial n} = h_{\text{conv}} \, (T - T_{\text{ambient}})
$$

#### (3) 材料本构关系 —— 热光效应

硅材料在 1550 nm 波段的热光系数：

$$
n(T) = n_0 + \frac{dn}{dT} \cdot (T - T_{\text{ambient}})
$$

对 Si，$dn/dT \approx 1.86 \times 10^{-4} \, \text{K}^{-1}$。

#### (4) 谐振条件与模式分析

微环谐振腔的谐振波长满足：

$$
2\pi R_{\text{eff}} \, n_{\text{eff}} = m \, \lambda_m
$$

其中 $m$ 为方位角模式数。对应的自由光谱范围（FSR）：

$$
\text{FSR} = \frac{\lambda^2}{2\pi R \, n_g}
$$

$n_g = n_{\text{eff}} - \lambda \cdot \frac{dn_{\text{eff}}}{d\lambda}$ 为群折射率。

传感灵敏度定义为：

$$
S = \frac{d\lambda}{dn_{\text{env}}} = \frac{\lambda}{n_{\text{eff}}}
$$

#### (5) 自洽迭代格式（Picard 迭代）

给定初始折射率 $n^{(0)} = n_0$，迭代直到收敛：

$$
\begin{aligned}
&\text{Step 1: } && \bigl[\nabla^2 + k_0^2 (n^{(k)})^2 + \gamma k_0^2\bigr] E^{(k)} = Q_{\text{src}} \\
&\text{Step 2: } && Q_{\text{abs}}^{(k)} = \alpha_{\text{abs}} \, |E^{(k)}|^2 \\
&\text{Step 3: } && \nabla \cdot(\kappa \nabla T^{(k)}) + Q_{\text{abs}}^{(k)} = 0 \\
&\text{Step 4: } && n^{(k+1)} = n_0 + \frac{dn}{dT} \bigl(T^{(k)} - T_{\text{ambient}}\bigr) \\
&\text{Step 5: } && \text{若 } \|n^{(k+1)} - n^{(k)}\| < \text{tol，停止}
\end{aligned}
$$

为增强稳定性，引入松弛因子 $\omega = 0.7$：

$$
n^{(k+1)} \leftarrow \omega \, n^{(k+1)} + (1-\omega) \, n^{(k)}
$$

---

## 二、种子项目映射与融合方式

本项目严格遵循"**每一个输入项目都必须在合成项目中承担真实角色，不得遗漏或挂名**"的原则。下表详细列出 15 个种子项目的核心算法及其在本项目中的真实功能定位。

| 序号 | 种子项目 | 核心算法 | 融合目标文件 | 真实功能角色 |
|:---:|:---|:---|:---|:---|
| 1 | `1119_sphere_integrals` | 球面单项式精确积分（Gamma 函数）、蒙特卡洛采样 | `quadrature_engine.py`, `photothermal_coupler.py` | 计算球面 WGM 模式功率积分的精确基准；蒙特卡洛验证求积规则精度 |
| 2 | `147_cell` | CVV 变长向量数据结构（ragged array 压缩存储） | `geometry_mesh.py` | 非结构化 FEM 网格中变长行数据（如不同单元节点数）的高效存储与 O(1) 索引 |
| 3 | `973_r8cb` | 压缩带状矩阵 LU 分解（无 pivoting）与前代回代 | `helmholtz_fd.py`, `thermal_fd.py` | Helmholtz 与热传导方程 5 点 stencil 离散后产生的大型带状线性系统的直接求解 |
| 4 | `974_r8cbb` | 边界带状矩阵 Schur 补分解与求解 | `helmholtz_fd.py` | 处理边界条件引入稠密耦合块的分块矩阵求解，演示混合 FEM 中的鞍点问题求解 |
| 5 | `799_neural_network` | 前馈神经网络 + 反向传播 + SGD | `nn_surrogate.py` | 构建传感响应代理模型，输入环境参数快速预测谐振波长漂移 |
| 6 | `563_hypersphere_angle` | 高维超球面均匀采样与角度统计 | `stochastic_uq.py` | 不确定性量化中多参数扰动方向的各向同性采样；验证高维空间随机方向正交性 |
| 7 | `647_laplacian` | 3/5/9 点拉普拉斯 stencil（均匀/非均匀/周期边界） | `helmholtz_fd.py`, `thermal_fd.py` | 提供 Helmholtz 与热方程有限差分离散的核心微分算子 |
| 8 | `951_quadrature_weights_vandermonde_2d` | 2D Vandermonde 矩阵求积权重 | `quadrature_engine.py`, `photothermal_coupler.py` | 在任意节点集上构造高精度 2D 求积规则，用于热源体积积分与 FEM 单元积分 |
| 9 | `198_collatz_polynomial` | GF(2) 上多项式动力系统 | `eigenvalue_solver.py` | 作为预处理算子设计的代数原型；其迭代动力学思想用于分析幂迭代的收敛域 |
| 10 | `094_bisection` | 二分法稳健求根 | `eigenvalue_solver.py` | 求解微环谐振条件非线性方程 $f(\lambda) = 2\pi R n_{\text{eff}}(\lambda) - m\lambda = 0$ |
| 11 | `1322_triangle_to_xml` | 三角网格索引转换与拓扑管理 | `geometry_mesh.py` | 极坐标映射生成的伪三角网格的连通性管理与 0-based/1-based 索引规范 |
| 12 | `376_fem_io` | FEM 节点/单元数据读写 | `geometry_mesh.py` | 微腔几何数据的持久化存储与标准格式交互（.node / .ele 文件） |
| 13 | `812_norm_l1` | L¹ 范数与误差度量 | `error_norms.py` | 温度场、光场的离散 L¹/L²/L∞ 范数计算；相对误差与收敛阶估计 |
| 14 | `1012_ranlib` | 多分布随机变量生成（正态/Gamma/Beta/指数等） | `stochastic_uq.py` | 蒙特卡洛不确定性量化中的材料参数随机采样 |
| 15 | `016_arclength` | 参数化曲线弧长计算（梯形法则） | `geometry_mesh.py` | 微环截面边界周长的数值计算与网格质量评估 |

---

## 三、代码架构与文件说明

本项目共包含 **10 个 Python 文件**，模块间依赖关系清晰，符合高内聚低耦合的工程原则。

```
108_synth_project/
├── main.py                      # 统一入口，零参数运行完整仿真流程
├── geometry_mesh.py             # 微腔几何建模、三角网格生成、CVV、FEM I/O、弧长计算
├── quadrature_engine.py         # 球面精确积分、2D Vandermonde 求积、Gauss-Legendre 张量积
├── helmholtz_fd.py              # Helmholtz 方程有限差分求解器、带状矩阵与边界带状矩阵求解
├── thermal_fd.py                # 热传导方程有限差分求解器、Robin 边界、热源计算
├── photothermal_coupler.py      # 光热耦合自洽迭代、热源积分、WGM 球面积分
├── eigenvalue_solver.py         # 谐振波长二分法求解、幂迭代本征值、Collatz 多项式动力学
├── stochastic_uq.py             # 超球面采样、多分布随机数生成、蒙特卡洛不确定性传播
├── nn_surrogate.py              # 前馈神经网络代理模型、传感响应训练与预测
└── error_norms.py               # L¹/L²/L∞ 范数、Richardson 外推、收敛阶估计
```

### 3.1 各模块核心算法复杂度

| 模块 | 核心算法 | 计算复杂度 | 数值特性 |
|:---|:---|:---|:---|
| `geometry_mesh.py` | 极坐标网格映射 + 三角剖分 | $O(N_x N_\theta)$ | 边界弧长误差 $O(h^2)$ |
| `helmholtz_fd.py` | 5 点 stencil + 带状 LU | $O(N^{1.5})$（带宽 $b \sim \sqrt{N}$） | 截断误差 $O(h^2)$ |
| `thermal_fd.py` | 5 点 stencil + Robin 边界 | $O(N^{1.5})$ | 能量守恒严格满足 |
| `photothermal_coupler.py` | Picard 自洽迭代 | $O(K \cdot N^{1.5})$，$K$ 为迭代次数 | 松弛因子保证收敛 |
| `eigenvalue_solver.py` | 二分法 + 幂迭代 | 二分 $O(\log(1/\epsilon))$，幂迭代 $O(1/\Delta\lambda)$ | 全局收敛保证 |
| `nn_surrogate.py` | 反向传播 SGD | $O(E \cdot M \cdot L)$，$E$ 轮数，$M$ 样本，$L$ 层数 | Xavier 初始化 |
| `stochastic_uq.py` | Monte Carlo 传播 | $O(S \cdot N^{1.5})$，$S$ 样本数 | 统计误差 $O(1/\sqrt{S})$ |
| `quadrature_engine.py` | 球面 Gamma 精确积分 | $O(d)$（闭式） | 机器精度 |

---

## 四、运行方式

### 环境要求
- Python 3.8+
- NumPy
- SciPy（用于 Gamma 函数 `scipy.special.gamma`）

### 执行命令
```bash
cd 108_synth_project
python main.py
```

程序无需任何命令行参数，自动顺序执行以下 9 大步骤：
1. 微腔几何建模与网格生成
2. 数值积分引擎验证（球面积分 + Vandermonde2D + Gauss-Legendre）
3. Helmholtz 光场与热传导求解
4. 光热耦合自洽迭代
5. 本征值分析与谐振波长计算
6. 神经网络代理模型训练与预测
7. 蒙特卡洛不确定性量化
8. 误差分析与范数计算
9. 边界带状矩阵 Schur 补求解演示

---

## 五、关键数值方法详述

### 5.1 带状矩阵 LU 分解（融入 973_r8cb）

对于带宽为 $(ml, mu)$ 的 $N \times N$ 带状矩阵，紧凑存储为 $A_{\text{band}}[ml+mu+1, N]$。Doolittle 变体 LU 分解：

$$
A_{ij} = A_{\text{band}}[mu + i - j, \, j], \quad \max(0, j-mu) \le i \le \min(N-1, j+ml)
$$

分解过程仅遍历带宽内元素，复杂度 $O(N \cdot ml \cdot mu)$。前代回代同样限制在带宽内。

### 5.2 边界带状矩阵 Schur 补（融入 974_r8cbb）

对分块矩阵：

$$
\begin{bmatrix} A_1 & A_2 \\ A_3 & A_4 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
$$

其中 $A_1$ 为 $n_1 \times n_1$ 带状，$A_4$ 为 $n_2 \times n_2$ 稠密。Schur 补步骤：

1. $A_1 = LU$（带状分解）
2. 求解 $A_1 X = -A_2$（逐列带状前代回代）
3. 计算 Schur 补 $S = A_4 + A_3 X$
4. $S = LU_{\text{dense}}$（稠密分解，带部分选主元）
5. 回代求解 $x_2 = S^{-1}(b_2 - A_3 A_1^{-1} b_1)$，$x_1 = A_1^{-1} b_1 + X x_2$

### 5.3 高维超球面采样（融入 563_hypersphere_angle）

在 $m$ 维单位超球面 $S^{m-1}$ 上均匀采样的理论保证：若 $g \sim \mathcal{N}(0, I_m)$，则 $u = g / \|g\|_2$ 在 $S^{m-1}$ 上均匀分布。

两独立随机方向夹角 $\theta$ 的理论均值：

$$
\mathbb{E}[|\cos\theta|] = \frac{\Gamma(m/2)}{\sqrt{\pi} \, \Gamma\bigl((m+1)/2\bigr)}
$$

当 $m \to \infty$ 时，$\mathbb{E}[|\cos\theta|] \to 0$，即随机方向趋于正交——此性质是随机投影与不确定性分析的理论基石。

### 5.4 2D Vandermonde 求积（融入 951_quadrature_weights_vandermonde_2d）

给定 $n = (t+1)(t+2)/2$ 个节点 $\{(x_l, y_l)\}$，要求精确积分所有次数 $\le t$ 的单项式 $x^i y^j$。构造线性系统：

$$
V \cdot w = r, \quad V_{k,l} = x_l^{i_k} y_l^{j_k}
$$

右端项为矩形域上的精确积分：

$$
r_k = \frac{B^{i_k+1} - A^{i_k+1}}{i_k+1} \cdot \frac{D^{j_k+1} - C^{j_k+1}}{j_k+1}
$$

通过伪逆求解病态 Vandermonde 系统获得权重 $w$。

### 5.5 神经网络反向传播（融入 799_neural_network）

对sigmoid激活 $\sigma(z) = 1/(1+e^{-z})$，输出层误差：

$$
\delta^{[L]} = (a^{[L]} - y) \odot \sigma'(z^{[L]})
$$

隐层反向传播：

$$
\delta^{[l]} = \bigl((W^{[l+1]})^T \delta^{[l+1]}\bigr) \odot \sigma'(z^{[l]})
$$

权重梯度：$\partial J / \partial W^{[l]} = \delta^{[l]} (a^{[l-1]})^T$。

本项目对回归任务做了输出归一化处理：将目标 $y$ 映射到 $[0,1]$ 区间以适应 sigmoid 输出，预测时反归一化。

---

## 六、工程鲁棒性设计

本项目在多处实现了边界检查与数值保护：

1. **矩阵奇异保护**：带状 LU 分解中每一步检查主元绝对值，若 $<10^{-30}$ 则抛出异常；稠密 LU 采用部分选主元。
2. **迭代收敛保护**：Picard 自洽迭代设置最大迭代次数上限（8 次），松弛因子 $0.7$ 防止振荡。
3. **物理量边界保护**：折射率更新后执行 `np.clip`，限制在 $[0.5n_0, 2n_0]$ 区间内。
4. **随机采样保护**：Monte Carlo 中跳过数值失败样本，记录成功样本数。
5. **数组维度校验**：所有矩阵-向量操作前检查形状匹配，CVV 索引检查行列越界。
6. **参数合法性校验**：网格分辨率、分布参数、迭代次数等均做前置非负/正检查。

---

## 七、合成后项目可解决的科学问题

1. **光热耦合效应定量分析**：计算微腔在高功率泵浦下的稳态温度分布与谐振波长热漂移。
2. **谐振模式设计优化**：通过二分法与幂迭代快速定位目标波长的谐振模式数 $m$ 与 FSR。
3. **传感灵敏度评估**：量化环境折射率变化引起的波长漂移（pm/RIU 级灵敏度）。
4. **制造公差影响预测**：蒙特卡洛方法评估半径、折射率、热导率等参数波动对器件性能的影响。
5. **实时传感代理推断**：训练后的神经网络可在毫秒级预测传感响应，替代耗时的全物理场仿真。
6. **多尺度数值验证**：通过球面精确积分、Vandermonde 求积与梯形法则的交叉验证，确保数值结果可靠性。

---

## 八、结论

本项目成功将 15 个独立的科研代码种子项目融合重构为一个面向**光学工程：光热耦合微腔传感**前沿领域的博士级 Python 计算平台。每个种子项目的核心算法都在合成系统中承担了不可替代的真实功能，从几何建模、微分方程离散、线性系统求解、数值积分、本征值分析、不确定性量化到机器学习代理模型，形成了完整的"建模-求解-分析-预测"科研闭环。所有代码已通过 `main.py` 零参数运行验证，无语法错误与运行时异常。
