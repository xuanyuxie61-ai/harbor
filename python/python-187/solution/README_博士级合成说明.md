# Thermodynamic-Geometric Collaborative Filtering (TGCF)
# 热力学-几何协同过滤系统 —— 博士级合成说明

---

## 1. 项目概述

本项目将 **15 个科研代码种子项目** 的核心算法融合为一个面向 **数据科学：推荐系统协同过滤** 的前沿博士级计算平台。项目以**热力学-几何耦合框架**统一建模推荐系统，将用户与物品视为高维潜空间中的粒子，评分视为能量交换，信息扩散服从偏微分方程，不确定性通过广义多项式混沌量化。

---

## 2. 科学问题定义

### 2.1 核心科学问题

**问题**：如何在高度稀疏、噪声污染、有界且随时间演化的用户-物品交互数据上，构建具备不确定性量化、物理可解释性和数值鲁棒性的协同过滤预测模型？

### 2.2 数学建模框架

将推荐系统建模为**连续介质热力学系统**：

- **状态空间**：用户-物品潜空间 $\mathcal{M} \subset \mathbb{R}^d$，其中 $d \ll \min(N_{\text{users}}, N_{\text{items}})$
- **场变量**：评分密度函数 $u(x, y, t)$，定义在流形 $\mathcal{M}$ 上
- **控制方程**：

$$
\frac{\partial u}{\partial t} - \alpha \nabla^2 u + k(x,y,t) \cdot u = f(x,y,t), \quad (x,y) \in \Omega
$$

$$
u(x,y,t) = g(x,y,t), \quad (x,y) \in \partial \Omega_D
$$

$$u(x,y,0) = h(x,y)
$$

其中：
- $\alpha$ 为**热扩散系数**，控制信息传播速度
- $k$ 为反应系数，建模信息衰减
- $f$ 为热源项，表示外部评分注入
- Dirichlet 边界条件 $g$ 对应已知观测评分

- **时间演化**：用户偏好受多频率不可公约周期驱动，服从**准周期动力学**

$$
\frac{d^4 y}{dt^4} + (\pi^2 + 1) \frac{d^2 y}{dt^2} + \pi^2 y = 0
$$

特征根 $r = \pm i, \pm i\pi$ 不可公约，解为准周期函数 $y(t) = \cos(t) + \cos(\pi t)$。

- **不确定性量化**：采用 **Laguerre 广义多项式混沌（gPC）**

$$
\hat{Y} = \sum_{k=0}^{P} c_k L_k(\xi), \quad \xi \sim \text{Exp}(1)
$$

$$
\text{Var}[\hat{Y}] = \sum_{k=1}^{P} c_k^2 \langle L_k, L_k \rangle_w
$$

---

## 3. 种子项目映射与改造方法

| 序号 | 种子项目 | 核心算法 | 在合成项目中的角色 | 改造文件 |
|:---:|:---|:---|:---|:---|
| 1 | `959_quasiperiodic_ode` | 四阶准周期 ODE | 用户偏好时间演化动力学 | `quasiperiodic_dynamics.py` |
| 2 | `347_faces_average` | 图像平均 | 用户分组潜向量聚合 | `aggregate_stats.py` |
| 3 | `642_laguerre_product` | Laguerre 正交多项式与 Gauss-Laguerre 求积 | 预测不确定性 gPC 量化 | `laguerre_chaos.py` |
| 4 | `219_cordic` | CORDIC 算法（sin/cos, exp, log, sqrt） | 快速嵌入式相似度计算 | `cordic_engine.py` |
| 5 | `419_fem3d_sample` | 3D FEM 四面体基函数 | 潜空间四面体插值 | `fem_embedding.py` |
| 6 | `403_fem2d_heat` | 2D 热方程 FEM 求解 | 评分信息扩散模型 | `heat_diffusion.py` |
| 7 | `118_brc_naive` | 流式聚合统计 | 用户分组运行统计 | `aggregate_stats.py` |
| 8 | `181_circle_monte_carlo` | 圆上蒙特卡洛积分 | 潜空间球面采样与积分 | `geometric_sampler.py` |
| 9 | `337_eros` | Gauss 消元、PLU、行列式、逆矩阵 | 隐因子线性系统求解 | `gaussian_solver.py` |
| 10 | `1254_tetrahedron_slice_display` | 四面体-平面相交、3D 面积 | 潜空间置信区域计算 | `geometric_sampler.py` |
| 11 | `669_levenshtein_matrix` | Levenshtein 编辑距离 | 冷启动物品文本相似性 | `string_similarity.py` |
| 12 | `536_hilbert_curve_3d` | 3D Hilbert 空间填充曲线 | 局部敏感哈希 LSH | `hilbert_hashing.py` |
| 13 | `1360_truncated_normal` | 截断正态分布采样、均值、方差 | 有界评分分布建模 | `truncated_distribution.py` |
| 14 | `1331_triangulation_boundary` | 三角剖分边界边检测 | 利基社区边界识别 | `triangulation_boundary.py` |
| 15 | `182_circle_positive_distance` | 正象限圆距离统计 | 推荐多样性度量 | `geometric_sampler.py` |

---

## 4. 新增数学物理模型与核心公式

### 4.1 热传导方程离散化（后向 Euler）

$$
\frac{u^{n+1} - u^n}{\Delta t} - \alpha \nabla^2 u^{n+1} = f^{n+1}
$$

整理得线性系统：

$$
(I - \alpha \Delta t \, L) \, u^{n+1} = u^n + \Delta t \, f^{n+1}
$$

其中 $L$ 为五点差分格式的离散拉普拉斯算子：

$$
(Lu)_{i,j} = \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2}
$$

### 4.2 有限元四面体体积坐标

对四面体 $T = \{v_1, v_2, v_3, v_4\}$，点 $p$ 的体积坐标（barycentric coordinates）：

$$
\lambda_i(p) = \frac{V_i(p)}{V}, \quad V = \frac{1}{6} \det\begin{bmatrix} v_2-v_1 & v_3-v_1 & v_4-v_1 \end{bmatrix}
$$

其中 $V_i(p)$ 是将 $v_i$ 替换为 $p$ 后的子四面体体积。插值公式：

$$
\tilde{u}(p) = \sum_{i=1}^{4} u_i \lambda_i(p)
$$

满足 $\sum \lambda_i = 1$，$\lambda_i \geq 0$，以及插值精确性 $\tilde{u}(v_i) = u_i$。

### 4.3 Laguerre 广义多项式混沌

Laguerre 多项式递推：

$$
L_0(x) = 1, \quad L_1(x) = 1 - x
$$

$$
(n+1) L_{n+1}(x) = (2n + 1 - x) L_n(x) - n L_{n-1}(x)
$$

Gauss-Laguerre 求积规则：

$$
\int_0^{\infty} e^{-x} f(x) \, dx \approx \sum_{k=1}^{N} w_k f(x_k)
$$

其中节点 $\{x_k\}$ 为 $L_N(x)$ 的根，权重：

$$
w_k = \frac{\prod_{j=2}^{N} c_j}{L'_N(x_k) \, L_{N-1}(x_k)}, \quad c_j = (j-1)^2
$$

### 4.4 CORDIC 算法

旋转模式伪旋转矩阵：

$$
\begin{bmatrix} x_{i+1} \\ y_{i+1} \end{bmatrix} = \begin{bmatrix} 1 & -\sigma_i 2^{-i} \\ \sigma_i 2^{-i} & 1 \end{bmatrix} \begin{bmatrix} x_i \\ y_i \end{bmatrix}
$$

$$
z_{i+1} = z_i - \sigma_i \arctan(2^{-i}), \quad \sigma_i = \text{sign}(z_i)
$$

幅值修正因子：

$$
K = \prod_{i=0}^{N-1} \cos(\arctan(2^{-i})) = \prod_{i=0}^{N-1} \frac{1}{\sqrt{1 + 2^{-2i}}}
$$

### 4.5 截断正态分布

有界评分 $X \in [a, b]$ 的截断正态密度：

$$
f(x) = \frac{\phi((x-\mu)/\sigma)}{\sigma \, [\Phi(\alpha_b) - \Phi(\alpha_a)]}, \quad \alpha_a = \frac{a-\mu}{\sigma}, \quad \alpha_b = \frac{b-\mu}{\sigma}
$$

均值与方差：

$$
\mathbb{E}[X] = \mu + \sigma \frac{\phi(\alpha_a) - \phi(\alpha_b)}{\Phi(\alpha_b) - \Phi(\alpha_a)}
$$

$$
\text{Var}[X] = \sigma^2 \left[ 1 + \frac{\alpha_a \phi(\alpha_a) - \alpha_b \phi(\alpha_b)}{\Phi(\alpha_b) - \Phi(\alpha_a)} - \left( \frac{\phi(\alpha_a) - \phi(\alpha_b)}{\Phi(\alpha_b) - \Phi(\alpha_a)} \right)^2 \right]
$$

### 4.6 PLU 分解

$$
PA = LU
$$

求解 $Ax = b$：

$$
Ly = Pb \quad \text{(前代)}, \quad Ux = y \quad \text{(回代)}
$$

行列式：

$$
\det(A) = (-1)^{\text{swap\_count}} \prod_{i} U_{ii}
$$

### 4.7 Hilbert 空间填充曲线 LSH

3D Hilbert 映射保持局部性：

$$
|H(p) - H(q)|^{1/d} \propto \|p - q\|
$$

其中 $H = \text{xyz\_to\_h}(x, y, z, r)$，$r$ 为曲线阶数。

### 4.8 Levenshtein 编辑距离

动态规划递推：

$$
d[i,j] = \min \begin{cases}
d[i-1, j] + 1 & \text{(删除)} \\
d[i, j-1] + 1 & \text{(插入)} \\
d[i-1, j-1] + \mathbb{1}_{s_i \neq t_j} & \text{(替换/匹配)}
\end{cases}
$$

### 4.9 圆上单项式积分（解析公式）

$$
I(e_1, e_2) = \oint_{S^1} x^{e_1} y^{e_2} \, ds = \frac{2 \, \Gamma(\frac{e_1+1}{2}) \, \Gamma(\frac{e_2+1}{2})}{\Gamma(\frac{e_1+e_2+2}{2})}
$$

当 $e_1$ 或 $e_2$ 为奇数时，$I = 0$（奇函数对称性）。

---

## 5. 项目架构与文件说明

```
187_synth_project/
├── main.py                          # 统一入口，零参数运行
├── quasiperiodic_dynamics.py        # 准周期 ODE 偏好动力学
├── laguerre_chaos.py                # Laguerre gPC 不确定性量化
├── cordic_engine.py                 # CORDIC 快速计算引擎
├── fem_embedding.py                 # 3D FEM 潜空间插值
├── heat_diffusion.py                # 2D 热传导信息扩散
├── gaussian_solver.py               # 高斯消元与 PLU 分解
├── geometric_sampler.py             # 几何蒙特卡洛采样
├── hilbert_hashing.py               # Hilbert 曲线 LSH
├── string_similarity.py             # Levenshtein 文本相似性
├── truncated_distribution.py        # 截断正态评分模型
├── triangulation_boundary.py        # 三角剖分边界检测
└── aggregate_stats.py               # 聚合统计引擎
```

---

## 6. 运行方式

### 6.1 环境要求

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7

### 6.2 执行命令

```bash
cd 187_synth_project
python main.py
```

无需任何命令行参数，程序将自动：
1. 生成合成评分数据
2. 执行热扩散、FEM 插值、准周期演化、CORDIC 计算、Laguerre 混沌分析等全部流程
3. 输出各环节科学指标与最终预测精度（MAE/RMSE）

---

## 7. 数值鲁棒性设计

| 模块 | 边界条件处理 |
|:---|:---|
| `quasiperiodic_dynamics.py` | 空输入返回空数组；时间步长为 0 时保持状态；角度自动归一化到 $[-\pi, \pi]$ |
| `laguerre_chaos.py` | max_degree < 0 时抛出异常；求积阶数 < 1 返回空；方差下限截断为 0 |
| `cordic_engine.py` | exp 输入 > 700 返回 inf；log 输入 ≤ 0 返回 -inf；sqrt 负输入返回 nan |
| `fem_embedding.py` | 全缺失行/列使用全局均值；点数 < 4 退化为最近邻；退化四面体体积下限保护 |
| `heat_diffusion.py` | 全 NaN 矩阵返回常数矩阵；Jacobi 迭代最大 50 步、容差 1e-5；结果截断 [1,5] |
| `gaussian_solver.py` | 零主元跳过处理；非方阵抛出异常；奇异矩阵回退伪逆；解形状自动调整 |
| `geometric_sampler.py` | 空输入返回空数组/0；样本数 < 10 自动提升；3 点以下面积退化为 0 |
| `hilbert_hashing.py` | 坐标越界自动取模；空向量列表返回空数组 |
| `string_similarity.py` | 空字符串距离为另一字符串长度；非字符串输入强制转换 |
| `truncated_distribution.py` | 采样结果严格截断 [a,b]；归一化常数 Z 下限保护为 1e-12 |
| `triangulation_boundary.py` | 点数 < 3 返回所有点；Delaunay 失败时返回极值点；边不连通时返回多条路径 |
| `aggregate_stats.py` | NaN 值自动过滤；单元素组标准差设为 0；空组返回默认值 |

---

## 8. 实验结果示例

在 80 用户 × 60 物品、观测密度约 15% 的合成数据上运行结果：

```
训练集 MAE:  0.2176
训练集 RMSE: 0.2505
测试集 MAE:  0.2357
测试集 RMSE: 0.2650
95% 预测区间覆盖率: 100.0%
```

关键科学指标：
- 准周期 ODE 数值解与精确解 L2 误差: $\sim 10^{-1}$（RK4，非均匀步长）
- CORDIC cos(π/6) 精度: 误差 $\sim 10^{-8}$（24 次迭代）
- Laguerre 指数加权积矩阵范数: 265.30
- 圆上 $x^2 y^2$ 积分: 解析值 0.785398，蒙特卡洛估计 0.785785
- Hilbert LSH 近邻对: 400
- 利基社区边界节点: 7 / 80

---

## 9. 创新点与科学贡献

1. **跨学科融合**：首次将热传导方程、准周期动力学、广义多项式混沌、有限元方法、CORDIC 算法、空间填充曲线、截断正态分布、三角剖分边界检测等 8 大学科方法统一应用于协同过滤。

2. **物理可解释性**：评分预测不再是纯黑箱矩阵分解，而是具有热力学-几何物理解释的连续场演化过程。

3. **不确定性量化**：通过 Laguerre gPC 给出预测的统计置信区间，而非点估计。

4. **时间演化建模**：引入准周期 ODE 刻画用户偏好的多频率竞争演化。

5. **冷启动处理**：结合 Levenshtein 文本相似性、截断正态分布和 FEM 插值，从多模态信息中缓解冷启动。

---

## 10. 参考文献与算法来源

本项目算法直接源于 John Burkardt 的 15 个科研代码项目，经系统性重构、跨学科映射和 Python 化改造后形成统一的推荐系统科学计算平台。所有核心数值方法均保留了原始算法的数学结构，同时注入了推荐系统领域的物理建模与工程适配。
