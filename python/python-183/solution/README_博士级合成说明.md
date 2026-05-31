# README_博士级合成说明.md

## 项目概述

**项目名称**：基于时空结构方程模型与间断 Galerkin 方法的高维因果推断网络分析系统

**科学领域**：数据科学 — 因果推断与结构方程 (Causal Inference & Structural Equation Modeling)

**项目编号**：PROJECT_183

**合成语言**：Python 3

---

## 一、前沿科学问题定义

### 1.1 问题背景

在当代数据科学的前沿研究中，**因果推断 (Causal Inference)** 已从传统的统计关联分析跃升为揭示系统内部机制的核心方法论。结构方程模型 (SEM) 作为因果推断的数学基石，其形式为：

$$
\mathbf{Y} = B\mathbf{Y} + \Gamma\mathbf{X} + \boldsymbol{\epsilon}
$$

其中 $B$ 刻画内生变量间的因果网络，$\Gamma$ 为外生变量系数，$\boldsymbol{\epsilon}$ 为结构误差。然而，现有方法面临三大瓶颈：

1. **高维稀疏性**：当变量维度 $p \gg n$ 时，传统协方差估计失效；
2. **时空动态性**：因果效应并非瞬时完成，而是在时空域中扩散演化；
3. **非欧几何约束**：气候、神经影像等数据天然定义在球面或流形上。

### 1.2 本项目解决的核心科学问题

> **如何在非欧时空域中，基于稀疏观测数据，构建具有高数值鲁棒性的结构因果模型，并量化外生干预的时空传播效应？**

本项目融合 15 个种子项目的核心算法，构建了一个从"数据输入 → 因果骨架发现 → 时空传播模拟 → 干预效应评估"的完整博士级计算 pipeline。

---

## 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 在合成项目中的角色 |
|------|--------|----------|-------------------|
| 1 | 510_hb_to_st | Harwell-Boeing 稀疏矩阵 CSR 格式转换 | **sparse_sem_matrix.py**：将稠密精度矩阵压缩为 CSR 格式，支持大规模因果网络存储 |
| 2 | 275_dg1d_poisson | 1D 泊松方程间断 Galerkin 有限元 | **dg_causal_solver.py**：因果效应在时空域中的扩散传播 PDE 求解器 |
| 3 | 1200_tennis_matrix | 吸收态马尔可夫链转移矩阵 | **markov_causal_chain.py**：离散状态因果网络的转移分析与 do-干预效应计算 |
| 4 | 1425_xyzf_display | 3D 点云与面片数据读取 | **geometry_utils.py**：三维因果曲面的节点/面片数据结构与法向量计算（已删除可视化） |
| 5 | 1263_toeplitz_inverse | Fiedler Toeplitz 矩阵快速求逆 | **toeplitz_time_inverse.py**：时间序列自协方差矩阵快速求逆与 Yule-Walker 方程求解 |
| 6 | 1122_sphere_llq_grid | 球面经纬度网格生成 | **spherical_causal_field.py**：球面因果场的离散化与 Laplace-Beltrami 谱分析 |
| 7 | 066_ball_distance | 单位球内随机点距离统计 | **causal_ode_dynamics.py**：高维因果参数空间中的蒙特卡洛距离估计 |
| 8 | 937_pyramid_witherden_rule | 金字塔区域高斯求积 | **pyramid_integrator.py**：因果参数空间的后验期望数值积分 |
| 9 | 425_ffmatlib | 有限元三角网格插值 | **causal_mesh_interpolator.py**：因果场在三角网格上的 P1 插值与区域积分 |
| 10 | 844_pagerank | Google 矩阵幂迭代 | **pagerank_causal_rank.py**：因果网络节点重要性排序 (CausalRank) 与混淆变量识别 |
| 11 | 033_asa076 | Owen T 函数高斯求积 | **gaussian_causal_test.py**：双变量正态分布下的条件独立性精确检验 |
| 12 | 135_calpak | 日期时间差计算 | **time_series_utils.py**：时间序列对齐、互相关与 Granger 因果检验 |
| 13 | 1296_tri_surface_to_stla | 三角曲面到 STL 转换 | **geometry_utils.py**：三维因果网格的法向量计算与 STL 格式输出（已删除可视化） |
| 14 | 1036_rk4 | 四阶 Runge-Kutta ODE 求解 | **causal_ode_dynamics.py**：动态因果模型 (DCM) 的干预扩散模拟 |
| 15 | 109_boundary_word_right | 多边形点包含判定（射线投射） | **causal_mesh_interpolator.py**：因果影响区域的空间边界判定 |

---

## 三、新增数学物理模型与核心公式

### 3.1 稀疏精度矩阵估计（Graphical Lasso）

样本协方差矩阵：
$$ \hat{\Sigma} = \frac{1}{n}\sum_{k=1}^{n}(\mathbf{x}^{(k)}-\bar{\mathbf{x}})(\mathbf{x}^{(k)}-\bar{\mathbf{x}})^T $$

带 $L_1$ 惩罚的极大似然目标泛函：
$$ \min_{\Theta \succ 0} \; -\log\det\Theta + \text{tr}(\hat{\Sigma}\Theta) + \lambda\|\Theta\|_1 $$

近端梯度更新（ISTA）：
$$ \Theta_{t+1} = \text{ST}_{\eta\lambda}\left(\Theta_t - \eta(\Theta_t^{-1} - \hat{\Sigma})\right) $$

### 3.2 因果效应时空传播 PDE

外生干预 $do(X_j=x)$ 的因果势函数 $u(x,t)$ 满足带有狄拉克源项的扩散方程：

$$ \frac{\partial u}{\partial t} - \frac{\partial}{\partial x}\left(K(x)\frac{\partial u}{\partial x}\right) = f(x,t) + \sum_{j}\beta_j\,\delta(x-x_j)\,Y_j $$

采用**局部间断 Galerkin (LDG)** 方法离散，弱形式为：
$$ \int_{I_i} u_t v\,dx + \hat{u}v|_{\partial I_i} - \int_{I_i} u v_x\,dx = \int_{I_i} q v\,dx $$

数值通量（SIPG）：
$$ \hat{u} = \{u\} - \frac{1}{2}\llbracket u \rrbracket, \qquad \hat{q} = \{q\} + \frac{\sigma}{h}\llbracket u \rrbracket $$

### 3.3 因果网络马尔可夫链

转移矩阵标准形：
$$ P = \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix} $$

基本矩阵与吸收概率：
$$ N = (I - Q)^{-1}, \qquad B = NR $$

干预 $do(X=x)$ 的因果效应：
$$ \text{CE} = B^{(do)} - B $$

### 3.4 CausalRank（因果 PageRank）

Google 矩阵：
$$ G = \alpha S + \frac{1-\alpha}{n}\mathbf{1}\mathbf{1}^T $$

幂迭代：
$$ \pi_{k+1} = G \pi_k $$

混淆变量识别分数：
$$ \text{score}_i = \pi_i \cdot \log\left(1 + \frac{\text{outdeg}_i}{\text{indeg}_i}\right) $$

### 3.5 Toeplitz 时间矩阵快速求逆

利用交换矩阵 $J$ 将 Toeplitz 矩阵 $T$ 转化为 Hankel 矩阵 $H=JT$，通过 Fiedler 算法：
$$ T^{-1} = K \cdot J, \qquad K = M_1 M_2 - M_3 M_4 $$

时滞因果强度指标：
$$ C(h) = \sum_{i,j:|i-j|=h} |(T^{-1})_{ij}| $$

### 3.6 球面因果场与 Laplace-Beltrami 算子

球面扩散方程：
$$ \frac{\partial u}{\partial t} = D \Delta_{S^2} u + f(\theta,\phi,t) $$

球面调和展开：
$$ u(\theta,\phi) = \sum_{l=0}^{L_{\max}}\sum_{m=-l}^{l} a_l^m Y_l^m(\theta,\phi) $$

特征值：
$$ \Delta_{S^2} Y_l^m = -l(l+1) Y_l^m $$

### 3.7 Owen T 函数与条件独立性检验

Owen T 函数（双变量正态积分）：
$$ T(h, a) = \frac{1}{2\pi}\int_{0}^{a}\frac{\exp\left(-\frac{h^2}{2}(1+x^2)\right)}{1+x^2}\,dx $$

偏相关系数：
$$ r_{ij|\cdot} = -\frac{\Theta_{ij}}{\sqrt{\Theta_{ii}\Theta_{jj}}} $$

### 3.8 动态因果模型（DCM）

$$ \frac{d\mathbf{y}}{dt} = A\mathbf{y} + B\mathbf{u} + C(\mathbf{y}\odot\mathbf{u}) $$

四阶 Runge-Kutta 时间推进：
$$ \mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4) $$

---

## 四、项目文件结构

```
183_synth_project/
├── main.py                           # 统一入口，零参数运行
├── sparse_sem_matrix.py              # 稀疏精度矩阵与 CSR 格式
├── dg_causal_solver.py               # DG 因果扩散方程求解器
├── markov_causal_chain.py            # 因果马尔可夫链与 do-干预
├── pagerank_causal_rank.py           # CausalRank 与混淆变量识别
├── toeplitz_time_inverse.py          # Toeplitz 快速求逆与时滞分析
├── gaussian_causal_test.py           # Owen T 函数与偏相关检验
├── causal_ode_dynamics.py            # RK4 动态因果模型与 MC 距离
├── spherical_causal_field.py         # 球面因果场与调和展开
├── pyramid_integrator.py             # Witherden 高维数值积分
├── causal_mesh_interpolator.py       # 三角网格插值与多边形判定
├── geometry_utils.py                 # 三维几何处理与 STL 转换
├── time_series_utils.py              # 时间序列对齐与 Granger 检验
└── README_博士级合成说明.md          # 本文档
```

---

## 五、修改与合成方法

### 5.1 代码改造路径

1. **语言迁移**：所有 15 个种子项目原为 MATLAB 脚本，已全面改写为 Python 3，保留核心数值算法不变。
2. **科学问题重构**：将原本独立的数值算法（矩阵求逆、ODE 求解、网格生成等）统一置于"因果推断"的框架下，赋予每个算法明确的因果科学语义。
3. **可视化剥离**：原项目中所有 `figure()`, `plot()`, `patch()`, `print('-dpng')` 等可视化代码已全部删除，仅保留纯数值计算功能。
4. **边界处理增强**：
   - Graphical Lasso 中加入谱截断保证正定性；
   - DG 求解器中引入正则化隐式 Euler；
   - Owen T 函数中对 $h\approx 0$ 和 $|h|\to\infty$ 做截断处理；
   - 所有矩阵求逆前检查条件数，必要时加入抖动 (jitter)。

### 5.2 工程复杂性设计

- **模块解耦**：12 个功能模块各自独立，通过 `main.py` 统一 orchestration；
- **类型注解**：全面使用 Python typing 提示，增强代码可读性与可维护性；
- **鲁棒性检查**：所有公共函数均包含输入维度、数值范围、奇异性的边界判定；
- **零参数运行**：`main.py` 无需任何命令行参数，内部自动生成合成数据并执行完整 pipeline。

---

## 六、运行方式

```bash
cd Synthesis-project-python/183_synth_project
python main.py
```

程序将依次执行 12 个科学计算模块，输出因果骨架、DG 扩散能量、马尔可夫吸收概率、CausalRank、AR 系数、Owen T 值、ODE 终态、球面调和系数、高维积分、网格插值、三维几何、时间序列 Granger 检验等结果。

---

## 七、质量检查清单

- [x] 原目录未被修改
- [x] 合成后的项目为 Python 语言
- [x] 新目录完整包含合成后的项目（13 个 `.py` 文件 + 1 个文档）
- [x] 只有一个博士级数学/物理科学计算问题已落地为可执行代码
- [x] 用户提供的每一个输入项目都已真实融入合成项目，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 无可视化代码残留
