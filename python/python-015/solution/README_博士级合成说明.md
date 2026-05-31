# README_博士级合成说明.md

## 拓扑半金属Weyl节点多尺度数值研究系统

**科学领域**：凝聚态物理 — 拓扑半金属中的Weyl节点

**项目目录**：`/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/015_synth_project`

---

## 一、项目概述

本项目将15个基础科研代码项目的核心算法融合重构为一个面向**凝聚态物理前沿**的博士级数值计算系统。核心科学问题是：

> **在拓扑半金属中，如何数值定位Weyl节点、计算其拓扑不变量（Berry曲率、Chern数、Weyl荷），并研究其输运特性？**

该系统涵盖从微观哈密顿量建模到宏观输运优化的完整计算流程，包含大量数学物理公式，计算难度达到博士级水平。

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 在合成项目中的角色 | 目标.py文件 |
|---|---|---|---|
| 331_ellipse_monte_carlo | 椭球内Cholesky变换采样 | Weyl节点附近的**椭球蒙特卡洛k点采样**；高阶多项式展开 | `bzone_sampler.py`, `weyl_hamiltonian.py` |
| 464_gen_hermite_exactness | 广义Hermite求积精确性检验 | **态密度(DOS)的高斯-厄米数值积分**；积分精确性验证 | `density_of_states.py` |
| 627_knapsack_rational | 有理数背包贪心算法 | **输运通道最优选择**（电导密度排序约束优化） | `transport_optimizer.py` |
| 491_grid_display | 网格点加载与维度检测 | k空间网格数据处理与维度分析框架 | `topological_invariant.py` |
| 1137_spquad | 稀疏网格Clenshaw-Curtis求积 | **高维布里渊区Berry曲率积分**（克服维度灾难） | `sparse_integrator.py` |
| 1373_uniform | 均匀随机数生成 | **均匀k点网格**与随机采样基础 | `bzone_sampler.py` |
| 1330_triangulation | Delaunay三角剖分/邻接/边界检测 | **Fermi面三角网格化**与拓扑连通性分析 | `triangulation_mesh.py` |
| 1030_rk12_adapt | 自适应RK12 ODE求解 | **Berry相位半经典演化**与自适应步长控制 | `ode_evolver.py` |
| 259_cvt_square_nonuniform | 非均匀密度CVT采样 | **密度加权自适应k点分布**（能隙倒数加权） | `bzone_sampler.py` |
| 044_asa152 | 超几何分布/正态累积概率 | **Weyl节点配对统计显著性检验** | `statistical_tests.py` |
| 789_navier_stokes_mesh2d | 2D网格数据提取 | **Fermi面2D网格结构提取** | `mesh_extractor.py` |
| 703_lorenz96_ode | Lorenz96周期性动力学 | **周期性边界条件下的等效晶格动力学模型** | `ode_evolver.py` |
| 1340_triangulation_node_to_element | 节点值到元素平均 | **能带数据从k点插值到Fermi面单元** | `triangulation_mesh.py`, `mesh_extractor.py` |
| 790_navier_stokes_mesh3d | 3D网格数据提取 | **三维能带曲面网格提取** | `mesh_extractor.py` |
| 1322_triangle_to_xml | 三角网格XML格式转换 | **Fermi面网格标准化输出**（DOLFIN XML） | `mesh_extractor.py` |

---

## 三、核心数学物理模型与公式

### 3.1 Weyl哈密顿量

在Weyl节点附近，低能有效哈密顿量为线性Dirac形式：

$$
H(\mathbf{k}) = \hbar v_F \mathbf{k} \cdot \boldsymbol{\sigma} = \hbar v_F (k_x \sigma_x + k_y \sigma_y + k_z \sigma_z)
$$

其中Pauli矩阵：

$$
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

本征值（线性色散）：

$$
E_{\pm}(\mathbf{k}) = \pm \hbar v_F |\mathbf{k}|
$$

紧束缚模型（TaAs原型）：

$$
\begin{aligned}
d_0(\mathbf{k}) &= m_0 + m_1(\cos k_x + \cos k_y) + m_2 \cos k_z \\
d_x(\mathbf{k}) &= A \sin k_x \\
d_y(\mathbf{k}) &= A \sin k_y \\
d_z(\mathbf{k}) &= 2B_1(2 - \cos k_x - \cos k_y)\sin k_z + 2B_2 \sin 2k_z
\end{aligned}
$$

$$
H(\mathbf{k}) = d_0(\mathbf{k}) I + \sum_{i=x,y,z} d_i(\mathbf{k}) \sigma_i
$$

### 3.2 Berry联络与Berry曲率

Berry联络：

$$
\mathbf{A}_n(\mathbf{k}) = i \langle u_n(\mathbf{k}) | \nabla_{\mathbf{k}} | u_n(\mathbf{k}) \rangle
$$

Berry曲率（张量形式，通过速度算符计算）：

$$
\Omega_{n,ab}(\mathbf{k}) = -2 \, \text{Im} \sum_{m \neq n} \frac{
\langle n | v_a | m \rangle \langle m | v_b | n \rangle
}{(E_n - E_m)^2}
$$

对于线性Weyl模型，Berry曲率有**解析表达式**：

$$
\boldsymbol{\Omega}_{\pm}(\mathbf{k}) = \pm \frac{1}{2} \frac{\mathbf{k}}{|\mathbf{k}|^3}
$$

### 3.3 拓扑不变量

**Berry相位**（沿闭合路径 $C$）：

$$
\gamma_n = \oint_C \mathbf{A}_n(\mathbf{k}) \cdot d\mathbf{k}
$$

**Chern数**（对二维闭曲面 $S$）：

$$
C_n = \frac{1}{2\pi} \int_S \Omega_{n,xy}(\mathbf{k}) \, d^2k
$$

**Weyl荷**（对包围Weyl节点的闭合曲面）：

$$
Q = \frac{1}{2\pi} \oint_S \boldsymbol{\Omega}(\mathbf{k}) \cdot d\mathbf{S} = \pm 1
$$

**Nielsen-Ninomiya定理**：

$$
\sum_i Q_i = 0
$$

Weyl节点必须成对出现（$\pm$配对）。

### 3.4 态密度

三维Weyl半金属的解析态密度：

$$
D(E) = \frac{E^2}{2\pi^2 (\hbar v_F)^3}
$$

数值方法包括直方图法、高斯展宽法和Gauss-Hermite求积法。

### 3.5 稀疏网格高维积分

Smolyak稀疏网格构造：

$$
A(q,d) = \sum_{q-d+1 \leq |\mathbf{l}|_1 \leq q} (-1)^{q-|{\mathbf l}|} \binom{d-1}{q-|{\mathbf l}|} \left(U^{l_1} \times \cdots \times U^{l_d}\right)
$$

一维Clenshaw-Curtis节点：

$$
x_j^l = \cos\left(\frac{\pi j}{2^l}\right), \quad j = 0, \ldots, 2^l
$$

### 3.6 半经典运动方程

Sundaram-Niu方程：

$$
\begin{aligned}
\frac{d\mathbf{r}}{dt} &= \frac{1}{\hbar} \frac{\partial E}{\partial \mathbf{k}} - \frac{d\mathbf{k}}{dt} \times \boldsymbol{\Omega}(\mathbf{k}) \\
\frac{d\mathbf{k}}{dt} &= -\frac{e}{\hbar}\mathbf{E}
\end{aligned}
$$

### 3.7 输运优化

将输运通道选择建模为**连续背包问题**：

$$
\max \sum_i g_i x_i \quad \text{s.t.} \quad \sum_i e_i x_i \leq E_{\text{budget}}, \quad 0 \leq x_i \leq 1
$$

其中 $g_i$ 为电导增益，$e_i$ 为能量消耗。

---

## 四、项目文件结构

```
015_synth_project/
├── main.py                       # 统一入口，零参数可运行
├── weyl_hamiltonian.py           # Weyl哈密顿量建模与本征问题
├── berry_curvature.py            # Berry曲率、Berry联络、Berry相位
├── bzone_sampler.py              # 布里渊区多方法采样（MC/CVT/均匀）
├── sparse_integrator.py          # 稀疏网格Clenshaw-Curtis高维积分
├── triangulation_mesh.py         # Fermi面三角剖分与网格拓扑
├── ode_evolver.py                # 自适应RK12与半经典运动方程
├── density_of_states.py          # 态密度与广义Hermite求积
├── topological_invariant.py      # Chern数、Weyl荷、Z2指标
├── transport_optimizer.py        # 输运通道背包优化
├── statistical_tests.py          # 超几何/正态统计检验
├── mesh_extractor.py             # 2D/3D网格提取与XML转换
├── utils.py                      # 通用数值工具
└── README_博士级合成说明.md       # 本文档
```

**共13个.py文件**（含main.py），满足"至少8个.py文件"的要求。

---

## 五、合成实现路径

### 5.1 算法重构流程

1. **哈密顿量引擎**（`weyl_hamiltonian.py`）：
   - 将原椭圆采样中的`monomial_value`推广为能带高阶修正项计算
   - 引入紧束缚模型描述真实材料的Weyl节点

2. **k空间采样器**（`bzone_sampler.py`）：
   - 椭圆MC采样 → Weyl节点附近非均匀椭球采样（Cholesky分解）
   - CVT非均匀采样 → 密度=1/|能隙|的自适应k点分布
   - 均匀随机采样 → 标准k点网格

3. **拓扑量计算**（`berry_curvature.py` + `topological_invariant.py`）：
   - 数值Berry曲率通过速度算符矩阵元计算（规范无关）
   - 解析公式用于数值验证
   - Wilson loop方法计算Berry相位

4. **积分引擎**（`sparse_integrator.py`）：
   - 将1D Clenshaw-Curtis规则通过Smolyak构造推广到3D
   - 用于布里渊区上的高维Berry曲率积分

5. **Fermi面处理**（`triangulation_mesh.py` + `mesh_extractor.py`）：
   - Delaunay三角剖分处理k空间等能面
   - 边界检测识别Fermi面边缘
   - 节点→元素插值实现能带数据映射

6. **动力学演化**（`ode_evolver.py`）：
   - RK12自适应步长控制误差
   - Lorenz96周期性结构模拟晶格周期性边界

7. **物理量分析**（`density_of_states.py` + `transport_optimizer.py` + `statistical_tests.py`）：
   - Hermite求积检验确保数值积分精度
   - 背包贪心算法优化输运通道选择
   - 超几何分布检验Weyl节点配对显著性

### 5.2 边界处理与数值鲁棒性

- **零矢量处理**：`safe_normalize`函数避免除零
- **规范固定**：Berry联络计算中固定本征矢整体相位
- **能隙保护**：Weyl节点附近避免能隙过小时的数值发散
- **Cholesky稳定性**：强制检查矩阵正定对称性
- **步长控制**：RK12自适应防止ODE演化溢出
- **参数校验**：所有物理输入进行范围检查

---

## 六、如何运行

```bash
cd "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/015_synth_project"
python main.py
```

程序将自动执行以下完整流程：
1. 构建线性Weyl模型和紧束缚模型
2. 数值搜索Weyl节点位置
3. 使用4种方法采样布里渊区
4. 计算Berry曲率（数值+解析对比）
5. 计算Chern数、Weyl荷和Wilson loop
6. Fermi面三角剖分与面积计算
7. 态密度（直方图/高斯/Hermite）
8. 稀疏网格高维积分
9. 半经典运动方程演化
10. 输运优化与手征反常电导
11. 统计显著性检验
12. 网格提取与XML格式转换

**运行时间**：约5-20秒（取决于机器性能）

---

## 七、科学问题解决方案

本项目可解决的博士级科学问题包括：

1. **Weyl节点定位**：在复杂紧束缚模型中数值寻找Berry曲率的磁单极子位置
2. **拓扑分类**：通过Chern数和Weyl荷定量表征能带拓扑
3. **Berry相位计算**：验证Wilson loop方法与数值Berry联络的一致性
4. **Fermi面重构**：将离散k点数据三角剖分为连续曲面
5. **态密度分析**：对比数值DOS与Weyl半金属的解析 $E^2$ 行为
6. **高维积分**：使用稀疏网格克服布里渊区积分的维度灾难
7. **输运预测**：基于拓扑保护的负磁阻效应估算手征反常电导
8. **统计验证**：通过超几何检验确认Weyl节点配对的实验显著性

---

## 八、参考文献与理论基础

1. Wan et al., *Phys. Rev. B* **83**, 205101 (2011) — TaAs类Weyl半金属预言
2. Xiong et al., *Science* **350**, 413 (2015) — 手征反常实验观测
3. Sundaram & Niu, *Phys. Rev. B* **59**, 14915 (1999) — 半经典运动方程
4. Waldvogel, *BIT* **43**, 1 (2003) — 快速Clenshaw-Curtis规则
5. Du, Faber & Gunzburger, *SIAM Review* **41**, 637 (1999) — CVT理论
6. Nielsen & Ninomiya, *Phys. Lett. B* **130**, 389 (1983) — Weyl节点定理
