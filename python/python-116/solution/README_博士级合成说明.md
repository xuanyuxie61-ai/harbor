# 脂质双分子层凝胶-液晶相变：博士级综合计算项目

## 一、项目概述

本项目围绕**分子动力学中的脂质双分子层（Lipid Bilayer）凝胶-液晶相变**这一前沿博士级科学问题，将 15 个种子科研代码项目的核心算法融合为一个完整的 Python 计算框架。系统模拟粗粒化脂质双层膜的相行为，涵盖从分子尺度相互作用到宏观相图分析的完整链条。

---

## 二、科学问题背景

### 2.1 物理化学模型

脂质双分子层是细胞膜的基本骨架，其**主相变（Main Transition）**指凝胶相（Gel, L_β'）与液晶相（Liquid-Crystalline, L_α）之间的可逆转变。相变由链熔化和取向无序化共同驱动，序参数 $S_2$ 表征系统的取向有序程度：

$$
S_2 = \left\langle P_2(\cos\theta) \right\rangle = \left\langle \frac{3\cos^2\theta - 1}{2} \right\rangle
$$

其中 $\theta$ 为脂质链长轴与膜法线的夹角。$S_2 \approx 1$ 对应完全有序的凝胶相，$S_2 \approx 0$ 对应完全无序的液晶相。

### 2.2 系统哈密顿量

采用粗粒化 Maier-Saupe 型模型：

$$
H = \sum_i H_{\text{orient}}(i) + \sum_{\langle ij \rangle} H_{\text{nn}}(i,j) + H_{\text{compress}} + H_{\text{tail}}
$$

各分项为：

- **取向势**：$H_{\text{orient}} = -J \cdot P_2(\cos\theta_i) \cdot S_2^{\text{local}}(i)$
- **最近邻耦合**：$H_{\text{nn}} = -\varepsilon (\mathbf{n}_i \cdot \mathbf{n}_j)^2$
- **面积压缩**：$H_{\text{compress}} = \frac{\kappa_A}{2} \frac{(A_i - A_0)^2}{A_0}$

### 2.3 平均场相变理论

自洽方程（Maier-Saupe 平均场）：

$$
S = \frac{I_1(\beta J S)}{I_0(\beta J S)}, \quad \beta = \frac{1}{k_B T}
$$

临界温度估计：

$$
T_c = \frac{J}{5 k_B} \approx 0.22 \frac{J}{k_B}
$$

Landau 自由能展开：

$$
F(S) = a\tau S^2 + b S^4 + c S^6, \quad \tau = \frac{T - T_c}{T_c}
$$

---

## 三、种子项目映射与合成方法

| 编号 | 种子项目 | 核心算法 | 合成后角色 | 所在文件 |
|:---:|:---|:---|:---|:---|
| 1 | 287_dijkstra | Dijkstra 最短路径 | 水分子跨膜渗透的最低自由能路径（MFEP）搜索 | `permeation_analysis.py` |
| 2 | 607_jacobi_polynomial | Jacobi 多项式递推 | 取向分布函数（ODF）的广义谱展开与序参数提取 | `order_parameters.py` |
| 3 | 104_boundary_locus | ODE 方法稳定区域分析 | MD 积分器（Verlet）在复平面上的绝对稳定区域判定 | `integrator.py` |
| 4 | 202_combo | 回溯搜索、排列/子集/划分枚举 | 脂质离散构象空间的组合采样与 Gray 码遍历 | `combinatorial_sampler.py` |
| 5 | 512_heated_plate | Jacobi 迭代稳态热方程 | 膜平面内非均匀温度场的稳态求解 | `bilayer_system.py` |
| 6 | 492_gridlines | 极坐标/矩形/三角网格生成 | 膜表面离散化的多类型计算网格 | `grid_topology.py` |
| 7 | 078_bernstein_polynomial | Bernstein 基函数 | 跨膜质量密度轮廓的平滑谱表示 | `density_profile.py` |
| 8 | 1016_rcm | Reverse Cuthill-McKee 重排序 | 脂质邻接图稀疏矩阵的带宽优化 | `sparse_matrix_ops.py` |
| 9 | 322_duffing_ode | Duffing 非线性振荡器 | 膜厚度集体涨落的受驱阻尼非线性动力学 | `phase_diagram.py` |
| 10 | 154_chain_letter_tree | 距离矩阵与层次聚类 | 脂质局域构象的聚类与凝胶/液晶畴识别 | `clustering_analysis.py` |
| 11 | 1055_sandia_sgmgg | 稀疏网格组合系数 | 多维集体变量空间自由能的稀疏网格数值积分 | `free_energy.py` |
| 12 | 905_pram | PRAM 多连方边界词 | 膜局部畴（domain）的六方格边界描述 | `grid_topology.py` |
| 13 | 107_boundary_word_equilateral | 等边多边形边界追踪 | 膜缺陷与畴边界的六方向步进追踪 | `grid_topology.py` |
| 14 | 844_pagerank | PageRank 幂法特征向量 | Markov 态模型（MSM）稳态分布与隐含时间尺度 | `sparse_matrix_ops.py` |
| 15 | 801_newton_maehly | Newton-Maehly 多项式求根 | 自洽场 Landau 展开稳定点的同步多根求解 | `phase_diagram.py` |

---

## 四、核心数学物理公式汇总

### 4.1 分子动力学积分

**速度 Verlet 算法**：

$$
\begin{aligned}
\mathbf{r}(t+\Delta t) &= \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \frac{1}{2}\mathbf{a}(t)\Delta t^2 \\
\mathbf{v}\left(t+\frac{\Delta t}{2}\right) &= \mathbf{v}(t) + \frac{1}{2}\mathbf{a}(t)\Delta t \\
\mathbf{a}(t+\Delta t) &= \frac{\mathbf{F}(t+\Delta t)}{m} \\
\mathbf{v}(t+\Delta t) &= \mathbf{v}\left(t+\frac{\Delta t}{2}\right) + \frac{1}{2}\mathbf{a}(t+\Delta t)\Delta t
\end{aligned}
$$

**Langevin 热浴**：

$$
m\frac{d\mathbf{v}}{dt} = \mathbf{F} - \gamma \mathbf{v} + \sqrt{\frac{2\gamma k_B T}{\Delta t}} \, \boldsymbol{\xi}(t), \quad \langle \xi_i(t)\xi_j(t')\rangle = \delta_{ij}\delta(t-t')
$$

### 4.2 Jacobi 多项式递推

$$
\begin{aligned}
P_0^{(\alpha,\beta)}(x) &= 1 \\
P_1^{(\alpha,\beta)}(x) &= \frac{(\alpha+\beta+2)x + (\alpha-\beta)}{2} \\
a_k P_{k+1} &= (b_k + c_k x)P_k - d_k P_{k-1}
\end{aligned}
$$

其中系数：

$$
\begin{aligned}
a_k &= 2(k+1)(k+\alpha+\beta+1)(2k+\alpha+\beta) \\
b_k &= (2k+\alpha+\beta+1)(\alpha^2-\beta^2) \\
c_k &= (2k+\alpha+\beta+1)(2k+\alpha+\beta)(2k+\alpha+\beta+2) \\
d_k &= 2(k+\alpha)(k+\beta)(2k+\alpha+\beta+2)
\end{aligned}
$$

归一化常数：

$$
h_n = \int_{-1}^{1}(1-x)^\alpha(1+x)^\beta [P_n^{(\alpha,\beta)}(x)]^2 dx = \frac{2^{\alpha+\beta+1}\Gamma(n+\alpha+1)\Gamma(n+\beta+1)}{(2n+\alpha+\beta+1)n!\Gamma(n+\alpha+\beta+1)}
$$

### 4.3 Bernstein 多项式基

$$
B_{n,k}(u) = \binom{n}{k} u^k (1-u)^{n-k}, \quad u \in [0,1]
$$

密度轮廓重构：

$$
\tilde{\rho}(u) = \sum_{k=0}^{n} b_k B_{n,k}(u)
$$

### 4.4 Helfrich 弹性理论

弯曲刚度估计：

$$
K_C = \frac{K_A d^2}{24}
$$

其中 $d$ 为单层厚度，$K_A$ 为面积压缩模量。

### 4.5 Duffing 型膜厚度涨落

$$
\ddot{d} + \delta \dot{d} + \alpha d + \beta d^3 = \gamma \cos(\omega t) + \xi(t)
$$

状态空间形式：

$$
\frac{d}{dt}\begin{pmatrix} d \\ v \end{pmatrix} = \begin{pmatrix} v \\ -\delta v - \alpha d - \beta d^3 + \gamma\cos(\omega t) + \xi(t) \end{pmatrix}
$$

### 4.6 Markov 态模型（MSM）

转移矩阵 $\mathbf{P}$ 的行随机条件：$\sum_j P_{ij} = 1$。

稳态分布满足：$\boldsymbol{\pi}^T = \boldsymbol{\pi}^T \mathbf{P}$。

PageRank 修正：$\mathbf{r} = \frac{1-d}{N}\mathbf{1} + d \mathbf{P}^T \mathbf{r}$。

隐含时间尺度：$\tau_k = -\frac{\Delta t}{\ln \lambda_k}$，$\lambda_k$ 为 $|\lambda_k|<1$ 的特征值。

### 4.7 稀疏网格积分（Smolyak）

配分函数近似：

$$
Z = \int_{[-1,1]^D} e^{-\beta E(\mathbf{x})} d\mathbf{x} \approx \sum_{i} w_i e^{-\beta E(\mathbf{x}_i)}
$$

组合系数（种子项目 1055_sandia_sgmgg）：

$$
c(\mathbf{i}) = \sum_{\mathbf{j} \in \mathcal{N}(\mathbf{i})} (-1)^{|\mathbf{j}-\mathbf{i}|_1}
$$

### 4.8 Newton-Maehly 同步求根

对于多项式 $P(z)$，第 $i$ 个根的迭代修正：

$$
z_i^{\text{new}} = z_i - \frac{P(z_i)}{P'(z_i) - P(z_i)\sum_{j\neq i}\frac{1}{z_i-z_j}}
$$

### 4.9 Debye-Waller 因子

$$
B = \frac{8\pi^2}{3}\langle u^2\rangle, \quad \langle u^2\rangle = \frac{k_B T}{I\omega_0^2}(1-S_2)
$$

---

## 五、文件结构

```
116_synth_project/
├── main.py                      # 统一入口（零参数运行）
├── bilayer_system.py            # 粗粒化双层系统 + 热场平衡
├── integrator.py                # 速度 Verlet + Langevin + 稳定性分析
├── order_parameters.py          # Jacobi 谱分析 + 取向序
├── density_profile.py           # Bernstein 密度轮廓 + Helfrich 弹性
├── grid_topology.py             # 多类型网格 + 边界追踪
├── sparse_matrix_ops.py         # RCM 重排序 + MSM/PageRank
├── combinatorial_sampler.py     # 回溯搜索 + Gray 码 + 组合枚举
├── free_energy.py               # 稀疏网格自由能积分
├── phase_diagram.py             # Newton-Maehly + Duffing 动力学 + 相图
├── permeation_analysis.py       # Dijkstra MFEP + 渗透系数
└── README_博士级合成说明.md     # 本文档
```

---

## 六、运行方式

```bash
cd 116_synth_project
python main.py
```

程序无需任何命令行参数，所有物理参数（温度、耦合常数、网格尺寸等）均以内置默认值提供。运行后将依次输出：

1. 系统初始化参数
2. 稳态温度场收敛信息
3. MD 积分器稳定性检验结果
4. 平衡化 MD 轨迹的能量与序参数
5. Jacobi 谱分析结果
6. Bernstein 密度轮廓与膜厚度
7. 网格生成与边界追踪统计
8. 稀疏矩阵 RCM 重排序与 MSM 分析
9. 组合构象采样与 Gray 码遍历
10. 稀疏网格自由能积分
11. 相图、Newton-Maehly 根与 Duffing 动力学
12. Dijkstra 最低自由能渗透路径
13. 层次聚类与畴大小分布
14. 综合结果汇总

---

## 七、边界处理与数值鲁棒性

- **周期边界条件**：所有格点邻域搜索均采用模运算实现周期边界。
- **角度范围保护**：取向角 $\theta$ 和 $\phi$ 通过 `np.mod` 限制在 $[0, 2\pi)$。
- **面积非负约束**：`system.area` 被 `np.clip` 限制在 $[0.1A_0, 5A_0]$。
- **分母保护**：Jacobi 递推中检查 $a_k$ 接近零的情况；Newton-Maehly 中检查 `denom` 接近零。
- **数值溢出防护**：Bessel 函数比值使用缩放版本 $I_1(x)/I_0(x) = i_{1e}(x)/i_{0e}(x)$；大参数使用渐近展开。
- **矩阵正则化**：最小二乘拟合添加 $10^{-10}I$ 正则项；PageRank 迭代检查收敛。
- **空路径保护**：Dijkstra 返回空路径时，渗透系数直接置零。

---

## 八、科学前沿性说明

本项目所涉问题属于**生物物理与统计力学交叉领域**的前沿方向：

- **相变临界现象**：脂质双层主相变是一类弱一级相变，其临界涨落、畴形成与线张力是当前膜生物物理的研究热点。
- **粗粒化分子动力学**：通过将原子细节粗粒化为有效粒子，可在介观尺度上捕捉相变行为，是连接全原子 MD 与连续介质理论的桥梁。
- **Markov 态模型（MSM）**：从短轨迹构建长尺度动力学，是当前分析生物大分子构象转变的主流方法。
- **稀疏网格高维积分**：解决了集体变量空间的"维度灾难"，是计算自由能面的前沿数值技术。
- **非线性动力学与混沌**：Duffing 型膜厚度涨落可 exhibit 混沌行为，与膜超声穿孔（sonoporation）等生物医学应用直接相关。
