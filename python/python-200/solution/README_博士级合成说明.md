# AutoDiff-MD: 基于自动微分的分子动力学高阶导数计算平台

## 博士级合成说明

### 一、项目概述

本项目围绕**高性能计算：自动微分与梯度计算**领域，构建了一个从原子势能到宏观热力学量的精确梯度传播系统。传统分子动力学（MD）中，力场的高阶导数通常依赖有限差分近似，引入 $O(h^2)$ 截断误差；本系统通过前向模式自动微分（Forward-Mode AD，基于 Dual Number）和超对偶数（Hyper-Dual Number）实现势能函数一阶、二阶导数的机器精度计算，并融合谱方法、有限元、CVT 优化采样、双调和弹性理论，完成多尺度材料响应的精确分析。

---

### 二、输入种子项目映射与真实角色

以下 15 个种子项目均在合成项目中承担**真实计算角色**，无遗漏、无挂名：

| 原项目 | 核心算法 | 在合成项目中的角色 |
|--------|---------|------------------|
| **1230_tet_mesh** | 四面体网格生成、体积积分、Jacobian 映射 | `tetrahedral_fem.py`：构建 3D 有限元网格，对微观原子密度场进行体积积分，计算宏观热力学量 |
| **1066_set_theory** | 集合划分、整数向量操作 | `sampling_methods.py`：实现等价类划分与幂集生成，用于粒子分类和采样空间剖分 |
| **591_interp_chebyshev** | Chebyshev 节点插值、差商、Newton 插值 | `chebyshev_spectral.py`：对势能面进行谱精度重构，同时提供谱微分矩阵用于高阶导数计算 |
| **455_gaussian_2d** | 二维各向异性高斯函数 | `potential_models.py`：作为 LJ 势的修正项，描述局域应变场的非对称响应；也用于密度场平滑核 |
| **905_pram** | PRAM 边界词（12 方向边界追踪） | `lattice_geometry.py`：PRAM 边界词直接保留，用于描述复杂多边形晶畴的边界几何 |
| **245_cvt_1d_nonuniform** | 非均匀密度 CVT（Centroidal Voronoi Tessellation） | `cvt_optimizer.py`：扩展为多维 CVT 优化器，用于优化 MD 积分和蒙特卡洛采样中的节点空间分布 |
| **742_mcnuggets** | 非负整数 Diophantine 方程求解 | `lattice_geometry.py`：求解晶格平移不变性约束下的整数方程，枚举可能的晶格常数组合 |
| **108_boundary_word_hexagon** | 六方网格边界词追踪 | `lattice_geometry.py`：六方边界词范围计算与顶点追踪，用于 HCP 结构的边界描述 |
| **649_latin_center** | Latin Center 超立方采样 | `sampling_methods.py`：Latin Center 与 Latin Hypercube 采样，用于粒子初始构型和蒙特卡洛积分 |
| **088_biharmonic_fd1d** | 1D 双调和方程有限差分 | `biharmonic_elasticity.py`：求解纳米梁/薄膜在热应力作用下的弯曲变形，计算弹性应变能 |
| **133_calendar_nyt** | Julian/Gregorian 日期转换 | `utils.py`：时间戳管理与模拟步数到物理时间的转换逻辑 |
| **1305_triangle_grid** | 三角形网格（重心坐标） | `sampling_methods.py`：三角形规则网格生成，用于 2D 截面的谱分析和局部采样 |
| **744_md** | 分子动力学（Velocity Verlet） | `md_engine.py`：核心 MD 引擎，支持 Berendsen 温度控制、周期性边界条件、能量追踪 |
| **540_histogram_display** | 直方图数据统计 | `utils.py` 与 `thermodynamics.py`：直方图统计类，计算能量分布的偏度、峰度、熵等 |
| **1417_wtime** | 墙钟时间测量 | `utils.py`：Timer 类与 benchmark 函数，用于性能计时和计算效率评估 |

---

### 三、核心数学物理模型与公式

#### 3.1 自动微分引擎（Dual Number 与 Hyper-Dual）

引入无穷小量 $\varepsilon$ 满足 $\varepsilon^2 = 0$，对光滑函数 $f$ 有：

$$
f(x + \varepsilon) = f(x) + \varepsilon \cdot f'(x)
$$

通过重载算术运算实现前向传播：

$$
(a + \varepsilon a') \cdot (b + \varepsilon b') = ab + \varepsilon(ab' + a'b)
$$

对于二阶导数，采用嵌套 Dual Number（Hyper-Dual）：

$$
z = f_0 + \varepsilon_1 f_1 + \varepsilon_2 f_2 + \varepsilon_1\varepsilon_2 f_{12}
$$

其中 $f_{12} = \partial^2 f / \partial x_i \partial x_j$ 给出 Hessian 的混合偏导。

#### 3.2 Lennard-Jones 势能模型

无量纲 LJ 12-6 势：

$$
V_{\text{LJ}}(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]
$$

解析力（排斥为正）：

$$
F_r = -\frac{dV}{dr} = \frac{24\varepsilon}{r} \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]
$$

平滑截断函数（shifted force）：

$$
S(r) = \begin{cases}
1 & r < r_c - \delta \\
\dfrac{(r_c - r)^2 \bigl(2\delta + r - r_c\bigr)}{\delta^3} & r_c - \delta \leq r < r_c \\
0 & r \geq r_c
\end{cases}
$$

#### 3.3 Velocity Verlet 分子动力学

辛积分器，保持相空间体积守恒：

$$
\begin{aligned}
\mathbf{r}(t+\Delta t) &= \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \tfrac{1}{2}\mathbf{a}(t)\Delta t^2 \\
\mathbf{v}(t+\Delta t) &= \mathbf{v}(t) + \tfrac{1}{2}\bigl[\mathbf{a}(t) + \mathbf{a}(t+\Delta t)\bigr]\Delta t \\
\mathbf{a}(t+\Delta t) &= \mathbf{F}(t+\Delta t) / m
\end{aligned}
$$

Berendsen 弱耦合 thermostat：

$$
\lambda = \sqrt{1 + \frac{\Delta t}{\tau}\left(\frac{T_{\text{bath}}}{T} - 1\right)}, \qquad \mathbf{v} \leftarrow \lambda \mathbf{v}
$$

#### 3.4 双调和弹性方程

纳米梁弯曲控制方程：

$$
\frac{d^4 u}{dx^4} = f(x), \qquad x \in [-1, 1]
$$

固支边界条件：

$$
u(\pm 1) = 0, \quad u'(\pm 1) = 0
$$

热-弹耦合等效载荷：

$$
f(x) = -\frac{\alpha E h^3}{12(1-\nu^2)} \frac{d^2 T}{dx^2}
$$

弯曲应变能：

$$
U = \frac{EI}{2} \int_{-1}^{1} \left(\frac{d^2 u}{dx^2}\right)^2 dx
$$

#### 3.5 Chebyshev 谱方法

Chebyshev 节点（极值点）：

$$
x_k = \cos\left(\frac{\pi k}{N}\right), \quad k = 0, 1, \dots, N
$$

Newton 差商插值：

$$
P_N(x) = d_0 + d_1(x-x_0) + d_2(x-x_0)(x-x_1) + \cdots + d_N \prod_{k=0}^{N-1}(x-x_k)
$$

谱微分矩阵 $D$（满足 $f'(x_i) \approx \sum_j D_{ij} f(x_j)$）：

$$
D_{ij} = \frac{c_i}{c_j} \frac{(-1)^{i+j}}{x_i - x_j} \quad (i \neq j), \qquad c_0 = c_N = 2, \; c_k = 1
$$

#### 3.6 CVT 非均匀采样

量化误差泛函：

$$
E(z_1, \dots, z_N) = \sum_{i=1}^{N} \int_{V_i} \rho(\mathbf{x}) \|\mathbf{x} - \mathbf{z}_i\|^2 \, d\mathbf{x}
$$

其中 Voronoi 单元：

$$
V_i = \{\mathbf{x} \in \Omega : \|\mathbf{x} - \mathbf{z}_i\| \leq \|\mathbf{x} - \mathbf{z}_j\|, \; \forall j \neq i\}
$$

Lloyd 固定点迭代：$\mathbf{z}_i \leftarrow \text{Centroid}(V_i)$

#### 3.7 热力学量

定容热容（能量涨落法）：

$$
C_V = \frac{\langle E^2 \rangle - \langle E \rangle^2}{k_B T^2}
$$

径向分布函数：

$$
g(r) = \frac{V}{N^2} \left\langle \sum_{i \neq j} \frac{\delta(r - r_{ij})}{2\pi r \, \Delta r} \right\rangle \quad (\text{2D})
$$

弹性常数（有限应变法）：

$$
C_{\alpha\beta} = \frac{1}{\Omega} \frac{\partial^2 E}{\partial \varepsilon_\alpha \partial \varepsilon_\beta}\bigg|_{\varepsilon=0}
$$

维里应力张量：

$$
\sigma_{\alpha\beta} = \frac{1}{\Omega} \sum_{i<j} r_{ij}^\alpha F_{ij}^\beta
$$

#### 3.8 四面体有限元积分

参考四面体到物理空间的线性映射：

$$
\mathbf{x}_{\text{phys}} = \mathbf{x}_1 + J \cdot \boldsymbol{\xi}, \qquad J = [\mathbf{x}_2-\mathbf{x}_1, \; \mathbf{x}_3-\mathbf{x}_1, \; \mathbf{x}_4-\mathbf{x}_1]
$$

体积：

$$
V = \frac{|\det J|}{6}
$$

数值积分：

$$
\int_{T} f(\mathbf{x}) \, dV = V \sum_{k} w_k f(\mathbf{x}_k)
$$

---

### 四、文件结构与改造说明

本项目共包含 **12 个 `.py` 文件**，所有代码均为原创合成，不存在可视化内容。

| 文件 | 功能说明 | 融入的原项目 |
|------|---------|-----------|
| `main.py` | 统一入口，零参数运行，调用全部模块完成完整计算流程 | 全部项目 |
| `autodiff_core.py` | Dual Number / Hyper-Dual 自动微分引擎，支持梯度、方向导数、混合偏导 | interp_chebyshev（数值微分思想） |
| `potential_models.py` | LJ 势、高斯势、组合势能、自动微分兼容版本、维里应力 | gaussian_2d, md, lennard_jones |
| `md_engine.py` | Velocity Verlet MD 引擎，含 Berendsen  thermostat、PBC | md |
| `tetrahedral_fem.py` | 四面体网格生成、Jacobian 映射、Gauss 积分、质量评估 | tet_mesh |
| `chebyshev_spectral.py` | Chebyshev 节点、差商插值、谱微分矩阵、BVP 求解 | interp_chebyshev |
| `cvt_optimizer.py` | 多维 CVT Lloyd 算法、1D 非均匀密度 CVT | cvt_1d_nonuniform |
| `biharmonic_elasticity.py` | 1D 双调和方程有限差分、曲率、应变能、热载荷 | biharmonic_fd1d |
| `sampling_methods.py` | Latin Center/Hypercube、三角形网格、集合划分、分层采样 | latin_center, triangle_grid, set_theory |
| `thermodynamics.py` | 比热、弹性常数、径向分布函数、熵、热膨胀系数 | md 数据分析, histogram_display |
| `lattice_geometry.py` | 六方边界词、PRAM 边界词、Diophantine 求解、HCP 晶格 | boundary_word_hexagon, pram, mcnuggets |
| `utils.py` | 计时器、直方图统计、数值鲁棒性工具、时间戳转换 | wtime, histogram_display, calendar_nyt |

---

### 五、合成后的科学问题

本项目解决的核心科学问题是：**如何在分子动力学模拟中消除力场高阶导数的数值截断误差，并实现从原子势能到宏观热力学响应的精确梯度传播？**

具体子问题包括：

1. **自动微分精度验证**：通过 Dual Number 前向传播，LJ 力的计算与解析解的相对误差达到机器精度（$<10^{-15}$），彻底消除有限差分的 $O(h^2)$ 误差。

2. **多尺度耦合**：将微观 MD 轨迹通过四面体有限元网格进行空间积分，得到连续密度场；再通过双调和方程计算纳米结构的弹性响应；最终通过统计力学公式提取比热、弹性常数、径向分布函数等宏观量。

3. **采样优化**：使用 CVT 和 Latin Hypercube 优化采样点分布，降低蒙特卡洛积分方差；使用 Chebyshev 谱方法高精度重构势能面，减少插值误差对力计算的传播。

4. **晶格几何约束**：利用 Diophantine 方程求解晶格平移不变性约束，利用边界词描述复杂晶畴边界，为周期性边界条件提供严格数学描述。

---

### 六、运行方式

```bash
cd "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/200_synth_project"
python main.py
```

**无需任何命令行参数**，`main.py` 自动执行以下完整流程：

1. 自动微分引擎精度验证（梯度、方向导数、混合偏导）
2. 晶格几何与 Diophantine 约束计算
3. 高级采样方法演示（Latin、三角形网格、集合划分）
4. Chebyshev 谱插值与微分
5. CVT 非均匀采样优化
6. 分子动力学模拟（36 粒子，500 步）
7. 自动微分力场高阶导数计算（含 Hessian）
8. 四面体网格有限元积分
9. 双调和弹性方程求解（含热-弹耦合）
10. 热力学量分析（比热、弹性常数、径向分布函数、熵）
11. 完整多尺度计算流程综合演示

运行结束后，所有数值结果直接打印到标准输出。

---

### 七、边界处理与数值鲁棒性

- **除零保护**：所有除法操作检查分母是否小于 $10^{-30}$，使用默认值或截断替代。
- **矩阵奇异性**：线性系统求解使用 `np.linalg.solve`，失败时回退到最小二乘；Cholesky 分解失败时返回 `None`。
- **LJ 势截断**：当 $r < 0.8\sigma$ 时自动截断，避免数值爆炸；使用 cubic smoothing 保证力在截断半径处连续。
- **温度控制稳定性**：Berendsen 缩放因子限制在 $[0.8, 1.2]$ 范围内，防止 thermostat 引起的数值不稳定。
- **能量守恒监控**：MD 运行过程中持续追踪总能量漂移，若漂移过大则提示异常。
- **正定矩阵检查**：高斯势的相关矩阵通过特征值截断强制保持 SPD。

---

### 八、性能与复杂度

- **自动微分复杂度**：对 $n$ 维输入计算梯度需要 $n$ 次前向传播，时间复杂度 $O(n \cdot \text{eval_cost})$，但精度为机器精度。
- **MD 复杂度**：$O(N^2)$ 每步（全对相互作用），适合中小规模系统（$N \sim 10^2$）。
- **四面体积分复杂度**：$O(M \cdot K)$，其中 $M$ 为单元数，$K$ 为每单元积分点数。
- **Chebyshev BVP 复杂度**：谱方法稠密矩阵求解 $O(N^3)$，$N$ 为节点数。

---

*本文档由 sci-project-synthesis-python skill 自动生成，所有公式与代码一一对应，可直接复现。*
