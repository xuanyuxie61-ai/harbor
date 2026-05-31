# 三维阻挫自旋玻璃基态与动力学相变的全栈计算框架

## 项目概述

本项目围绕**凝聚态物理：自旋玻璃与阻挫磁性**领域，将 15 个种子科研代码项目的核心算法融合重构为一个面向前沿科学问题的博士级 Python 计算框架。

### 核心科学问题

在具有几何阻挫（geometric frustration）与键无序（bond disorder）的磁性晶格中，经典 Heisenberg 自旋系统的基态求解、能量景观结构、自旋动力学弛豫以及磁畴演化是凝聚态物理中最具挑战性的问题之一。本项目构建了一个从**晶格构造** → **交换矩阵生成** → **基态优化** → **特征值分析** → **自旋动力学积分** → **磁畴统计** → **自适应有限元**的全链路计算 pipeline，能够定量分析阻挫磁体的以下关键物理量：

- 基态能量与亚稳态分布
- 自旋波色散关系与关联长度
- Landau-Lifshitz-Gilbert (LLG) 动力学轨迹
- 磁畴尺寸分布与构型熵
- Ginzburg-Landau 序参量的空间分布

---

## 原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 合成后角色 |
|:---:|--------|---------|-----------|
| 1 | `540_histogram_display` | 多列数据直方图读取与统计 | `utils.py` 中 `histogram_stats_1d`，用于自旋取向分布的能量直方图统计 |
| 2 | `694_local_min` | Brent 方法（黄金分割 + 抛物线插值）局部最小值搜索 | `energy_landscape.py` 中 `local_min_brent`，用于单自旋线搜索能量极小 |
| 3 | `121_brusselator_ode` | Brusselator 非线性 ODE 系统 | `spin_dynamics.py` 中 `brusselator_like_spin_pump`，模拟非线性自旋泵浦 |
| 4 | `1405_web_matrix` | 幂法求图的主导特征向量 | `eigen_analysis.py` 中 `power_method`，分析交换矩阵的占优本征模式 |
| 5 | `648_laplacian_matrix` | 1D 离散拉普拉斯矩阵（多种边界条件） | `exchange_matrix.py` 中 `exchange_laplacian_1d/2d`，构造交换相互作用核 |
| 6 | `827_ode_euler_system` | 显式欧拉法求解 ODE 系统 | `spin_dynamics.py` 中 `euler_integrate_llg`，显式积分 LLG 方程 |
| 7 | `433_fisher_exact` | Fisher-KPP 方程精确行波解 | `spin_dynamics.py` 中 `fisher_kpp_domain_wall_exact`，描述磁畴壁孤子传播 |
| 8 | `1306_triangle_histogram` | 单位三角形内子区域直方图 | `utils.py` / `domain_analysis.py` 中 `triangle_area_histogram_2d`，统计自旋平面投影分布 |
| 9 | `672_lights_out` | 邻接矩阵构造（格点近邻连接） | `spin_lattice.py` 中 `PyrochloreLattice._build_bonds`，构造三维晶格邻接关系 |
| 10 | `997_r8ss` | 对称 skyline 稀疏矩阵存储与矩阵-向量乘法 | `utils.py` 中 `skyline_mv` / `build_skyline_from_tridiagonal`，压缩存储大规模交换矩阵 |
| 11 | `1286_trapezoidal` | 隐式梯形法 + Newton 迭代求解 ODE | `spin_dynamics.py` 中 `trapezoidal_integrate_llg`，隐式积分 LLG 方程 |
| 12 | `205_components` | 2D 连通分量标记（四邻域 BFS） | `spin_lattice.py` / `domain_analysis.py` 中 `connected_components_2d_spin_map`，识别磁畴团簇 |
| 13 | `384_fem1d_adaptive` | 自适应有限元方法（1D 边值问题） | `adaptive_fem_solver.py`，求解 Ginzburg-Landau 型序参量方程 |
| 14 | `960_quaternions` | 四元数与 3D 旋转矩阵互转 | `spin_quaternion.py`，自旋矢量的无奇异点球面表示与旋转插值 |
| 15 | `915_prime_plot` | 素数检测与周期性分布 | `spin_lattice.py` 中 `PrimeFrustratedLattice`，利用素数序列调制产生非均匀阻挫 |

---

## 核心数学物理模型与公式

### 1. 经典 Heisenberg 哈密顿量

对于 N 个经典自旋 $\{\mathbf{S}_i\}$（$|\mathbf{S}_i| = S$），交换相互作用能量为

$$
\mathcal{H} = \frac{1}{2} \sum_{i,j} J_{ij} \, \mathbf{S}_i \cdot \mathbf{S}_j
- \sum_i \mathbf{H}_{\text{ext}} \cdot \mathbf{S}_i
- K_{\text{anis}} \sum_i (\mathbf{S}_i \cdot \hat{\mathbf{n}})^2
$$

其中：
- $J_{ij}$ 为交换耦合矩阵（反铁磁时 $J_{ij} > 0$ 倾向于反平行排列）
- $\mathbf{H}_{\text{ext}}$ 为外磁场
- $K_{\text{anis}}$ 为单轴各向异性能量密度
- $\hat{\mathbf{n}}$ 为各向异性轴

### 2. Landau-Lifshitz-Gilbert (LLG) 方程

自旋在有效场中的阻尼进动满足

$$
\frac{d\mathbf{S}_i}{dt}
= -\gamma \, \mathbf{S}_i \times \mathbf{H}_i^{\text{eff}}
+ \alpha \, \mathbf{S}_i \times \bigl( \mathbf{S}_i \times \mathbf{H}_i^{\text{eff}} \bigr)
$$

有效场由能量泛函的变分导数给出：

$$
\mathbf{H}_i^{\text{eff}} = -\frac{\delta \mathcal{H}}{\delta \mathbf{S}_i}
= \sum_j J_{ij} \mathbf{S}_j + \mathbf{H}_{\text{ext}}
+ 2 K_{\text{anis}} (\mathbf{S}_i \cdot \hat{\mathbf{n}}) \hat{\mathbf{n}}
$$

利用向量三重积恒等式 $\mathbf{S} \times (\mathbf{S} \times \mathbf{H}) = \mathbf{S}(\mathbf{S}\cdot\mathbf{H}) - \mathbf{H}$，阻尼项可改写为

$$
\mathbf{S}_i \times (\mathbf{S}_i \times \mathbf{H}_i)
= (\mathbf{S}_i \cdot \mathbf{H}_i) \mathbf{S}_i - \mathbf{H}_i
$$

数值积分时，每一步需将 $\mathbf{S}_i$ 重新投影到单位球面，保证 $|\mathbf{S}_i| = 1$ 的约束。

### 3. 一维自旋波色散（Holstein-Primakoff 近似）

对于铁磁链（$J < 0$ 或重新定义），线性化后的自旋波频率为

$$
\omega(k) = 2 J S \bigl[ 1 - \cos(ka) \bigr]
$$

在长波极限 $ka \ll 1$ 下，

$$
\omega(k) \approx J S a^2 k^2 = D k^2
$$

其中 $D = J S a^2$ 为自旋波刚度。

### 4. Fisher-KPP 磁畴壁行波解

一维易轴铁磁体中的磁化强度分布可由 Fisher-KPP 方程描述：

$$
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + r u (1 - u)
$$

对于特定波速 $c = \frac{5}{\sqrt{6}} \sqrt{D r}$，存在解析行波解

$$
u(x,t) = \frac{1}{\bigl[ 1 + a \, e^{k(x - ct)} \bigr]^2}
$$

磁化强度纵向分量为 $M_z(x,t) = M_s \bigl[ 2\nu(x,t) - 1 \bigr]$，描述以速度 $c$ 传播的 $180^\circ$ 畴壁。

### 5. 离散 Laplacian 交换算子

一维有限差分离散化给出三对角交换矩阵（Dirichlet 边界）：

$$
L = \frac{1}{h^2}
\begin{pmatrix}
2 & -1 & 0 & \cdots & 0 \\
-1 & 2 & -1 & \cdots & 0 \\
0 & -1 & 2 & \ddots & \vdots \\
\vdots & \vdots & \ddots & \ddots & -1 \\
0 & 0 & \cdots & -1 & 2
\end{pmatrix}
$$

二维通过 Kronecker 和构造：$L_{2D} = I_y \otimes L_x + L_y \otimes I_x$。

### 6. Ginzburg-Landau 序参量方程

平均场近似下，序参量 $m(x)$ 满足

$$
-\frac{d}{dx}\Bigl( A(x) \frac{dm}{dx} \Bigr) + B(x) m(x) = F(x)
$$

对应能量泛函

$$
I[m] = \int_0^1 \Bigl[ \frac{1}{2} A(x) (m')^2 + \frac{1}{2} B(x) m^2 - F(x) m \Bigr] dx
$$

使用 P1（分段线性）有限元离散，单元 $[x_L, x_R]$ 上的基函数为帽子函数

$$
\phi_L(x) = \frac{x_R - x}{h}, \qquad
\phi_R(x) = \frac{x - x_L}{h}
$$

单元刚度矩阵与载荷向量通过两点 Gauss 积分精确组装。

### 7. 四元数旋转与 LLG 切平面投影

单位四元数 $\mathbf{q} = [q_0, q_1, q_2, q_3]$ 满足 $|\mathbf{q}| = 1$。旋转矩阵为

$$
R(\mathbf{q}) =
\begin{pmatrix}
1-2(q_2^2+q_3^2) & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\
2(q_1 q_2 + q_0 q_3) & 1-2(q_1^2+q_3^2) & 2(q_2 q_3 - q_0 q_1) \\
2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & 1-2(q_1^2+q_2^2)
\end{pmatrix}
$$

自旋矢量的旋转：$\mathbf{S}' = R(\mathbf{q}) \mathbf{S}$。

在数值积分中，为严格保持约束 $|\mathbf{S}_i| = 1$，将 LLG 右端项投影到切平面：

$$
\frac{d\mathbf{S}_i}{dt} \Big|_{\text{tangent}}
= \frac{d\mathbf{S}_i}{dt} - \mathbf{S}_i \Bigl( \mathbf{S}_i \cdot \frac{d\mathbf{S}_i}{dt} \Bigr)
$$

### 8. 磁畴构型熵

连通分量标记后，第 $\alpha$ 个磁畴的体积分数为 $p_\alpha = V_\alpha / V_{\text{total}}$，构型熵为

$$
S_{\text{config}} = -\sum_\alpha p_\alpha \ln p_\alpha
$$

### 9. 模拟退火 Metropolis 准则

在温度 $T$ 下，以概率

$$
P(\text{accept}) = \min\Bigl\{ 1, \exp\Bigl(-\frac{\Delta E}{k_B T}\Bigr) \Bigr\}
$$

接受能量上升的构型变化，实现基态的全局搜索。

### 10. 谱隙与关联长度

交换矩阵 $J$ 的最小两个特征值之差 $\Delta = \lambda_2 - \lambda_1$ 称为谱隙。在平均场近似下，关联长度

$$
\xi \sim \frac{a}{\sqrt{\Delta / J}}
$$

当 $\Delta \to 0$ 时，$\xi$ 发散，标志相变或 Goldstone 软模的出现。

---

## 文件结构

```
014_synth_project/
├── main.py                  # 统一入口，零参数运行
├── spin_lattice.py          # 晶格构造（Pyrochlore + Prime-Frustrated + 连通分量）
├── exchange_matrix.py       # 交换矩阵与离散 Laplacian（1D/2D/Skyline）
├── spin_quaternion.py       # 四元数自旋表示与 3D 旋转
├── energy_landscape.py      # Brent 线搜索 + 模拟退火 + 贪心松弛
├── eigen_analysis.py        # 幂法 + 逆迭代 + 自旋波色散
├── spin_dynamics.py         # LLG 显式/隐式积分 + Brusselator 泵 + Fisher-KPP 行波
├── domain_analysis.py       # 磁畴统计 + 直方图 + 径向关联函数
├── adaptive_fem_solver.py   # 自适应 P1 有限元求解 Ginzburg-Landau 方程
├── utils.py                 # Skyline 存储 + 数值鲁棒性工具 + 直方图统计
└── README_博士级合成说明.md  # 本说明文档
```

---

## 运行方式

在项目根目录下直接执行：

```bash
python3 main.py
```

无需任何命令行参数。程序将自动完成以下 10 个计算步骤并输出关键物理量：

1. **晶格构造**：生成 108 格点的 3D 烧绿石晶格与 144 格点的 2D 素数调制阻挫晶格
2. **交换矩阵**：建立含键无序的耦合矩阵，验证 skyline 稀疏格式的数值精度
3. **自旋初始化**：利用 Marsaglia 方法生成均匀分布的单位自旋矢量
4. **能量优化**：模拟退火（798 步记录）+ 贪心松弛（8 轮），Brent 线搜索精化单自旋
5. **特征值分析**：全谱对角化求谱隙与关联长度，幂法求主导模式，逆迭代识别软模
6. **LLG 动力学**：显式 Euler（400 步）与隐式梯形法（100 步）对比积分
7. **非线性自旋泵**：Brusselator 型 ODE 积分 2000 步
8. **磁畴壁行波**：Fisher-KPP 精确解在 200 个空间点上的求值
9. **磁畴统计**：连通分量标记提取 11 个磁畴，计算构型熵、取向直方图、径向关联
10. **自适应 FEM**：从 9 个节点开始，经 5 轮自适应加密至 29 个节点，求解序参量分布

---

## 数值鲁棒性设计

- **除零保护**：`safe_divide`、`safe_sqrt` 在分母/被开方数接近机器精度时提供 fallback
- **自旋归一化**：每步积分后调用 `clip_spin_norm`，退化时自动对齐 $z$ 轴
- **切平面投影**：LLG 右端项显式投影，消除数值漂移导致的 $|S| \neq 1$
- **矩阵正则化**：逆迭代中若 $M - \sigma I$ 奇异，自动加 $\varepsilon I$ 扰动
- **Thomas 算法稳定性**：三对角求解中对角元进行微小正则化，避免除零
- **边界条件完备性**：Laplacian 支持 Dirichlet / Neumann / Periodic 三种边界
- **自适应终止**：FEM 误差低于阈值或节点数达到上限时自动停止加密

---

## 科学意义

本项目实现的计算框架可用于：
- **自旋玻璃基态搜索**：通过模拟退火 + 局部松弛逼近 NP-hard 的基态问题
- **自旋动力学模拟**：LLG 方程积分揭示阻挫系统中的非平衡弛豫路径
- **磁畴演化预测**：Fisher-KPP 行波解与连通分量分析结合，描述畴壁运动与钉扎
- **相变前驱识别**：通过谱隙与软模分析预判阻挫磁体的量子/热相变
- **微磁学有限元**：自适应加密精确捕捉畴壁过渡区，避免均匀网格的过度计算

所有代码均严格基于提供的 15 个种子项目的核心算法思想，经数学物理重构后融入凝聚态物理前沿问题，具备博士级科学计算复杂度。
