# 非厄米物理与例外点：博士级科研计算合成项目

## 1. 项目概述

本项目围绕**凝聚态物理：非厄米物理与例外点（Exceptional Points, EPs）**展开，将 15 个种子项目的核心算法融合为一个前沿博士级自然科学计算框架。项目使用 **Python** 实现，包含 **16 个 `.py` 文件**与一个统一入口 `main.py`，零参数即可运行。

---

## 2. 合成后的科学问题

### 2.1 核心科学问题

非厄米哈密顿量 $H \neq H^\dagger$ 在开放量子系统、光子晶体、声学超材料等前沿领域中具有核心地位。与传统厄米系统不同，非厄米系统支持**例外点（Exceptional Points）**——参数空间中本征值与本征矢量同时简并的奇点。在 EP 附近，系统表现出极端的灵敏度、单向传输、拓扑能量交换等独特现象。

本合成项目构建了一个完整的计算框架，用于研究以下博士级科学问题：

1. **非厄米哈密顿量的系统构造**：包括 PT 对称模型、非厄米 SSH 模型、非厄米 Hofstadter 模型。
2. **例外点的定位与阶数判定**：利用 Laguerre 立方收敛根求法在复参数平面中寻找 EP，并判定其阶数。
3. **双正交拓扑不变量**：计算非厄米系统的 Berry 联络、Berry 曲率、Zak 相位、Chern 数以及复能量 winding number。
4. **三维布里渊区数值积分**：使用四面体 NCO（Newton-Cotes Open）高阶求积公式计算 BZ 上的拓扑不变量与平均能量。
5. **非厄米动力学与开放量子演化**：求解非厄米薛定谔方程与 Lindblad 主方程，模拟 PT 对称破缺前后的时间演化。
6. **随机矩阵能级统计**：利用不完全 Beta 函数分析非厄米 Ginibre 系综的能级间距分布，验证 Wigner-Dyson 到 Poisson 的过渡。
7. **有限元空间离散化**：对含复势的连续非厄米薛定谔算子进行三角剖分与刚度/质量矩阵组装。
8. **并行参数扫描与蒙特卡洛搜索**：在多核 CPU 上并行扫描参数空间，使用 MCMC 与自适应局域搜索快速定位 EP 流形。
9. **传递矩阵与李雅普诺夫指数**：研究一维非厄米系统的局域化长度与非厄米皮肤效应。
10. **Vandermonde 谱插值**：对能带进行稳定的重心 Lagrange 插值，重构特征多项式。

---

## 3. 种子项目映射关系（15 → 16）

| 编号 | 原种子项目 | 核心算法/思想 | 合成后文件 | 承担的科学角色 |
|:---|:---|:---|:---|:---|
| 1 | `431_filum` | 文件/字符串 I/O、字符串转浮点向量 | `config_parser.py` | 读取非厄米模拟参数配置文件，解析复矩阵文本输入 |
| 2 | `1253_tetrahedron_nco_rule` | 四面体对称数值积分（NCO 规则） | `brillouin_integrator.py` | 3D 布里渊区高阶四面体求积，用于拓扑不变量积分 |
| 3 | `1430_zero_laguerre` | Laguerre 多项式根求法 | `exceptional_point_solver.py` | 在复 k 平面寻找例外点（判别式零点），立方收敛 |
| 4 | `514_hello_parfor` | 并行 for 循环 | `parallel_sweep.py` | 多进程并行参数空间扫描，加速 EP 流形映射 |
| 5 | `1377_usa_box_plot` | 矩形区域填充 | `parameter_box.py` | 参数空间超矩形包围盒生成与自适应八叉树细分 |
| 6 | `1052_sammon_data` | 多维数据生成（圆、螺旋、单形） | `manifold_generator.py` | 生成环绕 EP 的绝热循环路径与参数单形采样 |
| 7 | `474_gmsh_io` | GMSH 网格文件读写 | `mesh_discretization.py` | 非厄米有限元问题的网格解析、刚度/质量矩阵组装 |
| 8 | `696_locker_simulation` | 概率搜索策略 | `monte_carlo_sampler.py` | 蒙特卡洛、重要性抽样、MCMC 搜索 EP 候选点 |
| 9 | `1094_snakes_matrix` | 马尔可夫转移矩阵 | `transfer_matrix.py` | 非厄米传递矩阵、李雅普诺夫指数、非厄米马尔可夫链 |
| 10 | `1004_r8vm` | Vandermonde 矩阵求解 | `vandermonde_solver.py` | 能带谱插值、Björck-Pereyra 算法、重心 Lagrange 插值 |
| 11 | `920_profile_data` | 1D 空间轮廓数据 | `potential_profile.py` | 复 Pöschl-Teller、Kronig-Penney、双阱势场剖面构造 |
| 12 | `675_lindberg_ode` | 刚性 ODE 系统 | `nonherm_dynamics.py` | 自适应 RKF45 求解非厄米薛定谔方程与 Lindblad 方程 |
| 13 | `031_asa063` | 不完全 Beta 函数 | `random_matrix_stats.py` | 能级间距统计检验、Wigner-Poisson 混合分布拟合 |
| 14 | `1086_sir_ode` | 房室模型 ODE（SIR） | `nonherm_dynamics.py` | Lindblad 主方程结构（增益/损耗/耦合 compartments） |
| 15 | `1352_triangulation_svg` | 三角剖分 | `triangulation.py` | 2D Delaunay 三角剖分（Bowyer-Watson）用于有限元离散 |

此外，`hamiltonian_builder.py` 与 `biorthogonal_topology.py` 是本项目原创的核心物理模块，负责哈密顿量构造与双正交拓扑计算；`main.py` 为统一入口。

---

## 4. 核心数学物理公式

### 4.1 非厄米哈密顿量

PT 对称一维双带模型：
$$H(k) = (m + t\cos k)\,\sigma_z + t\sin k\,\sigma_y + i\gamma\,\sigma_x$$

非厄米 SSH 模型：
$$H(k) = (t_1 + t_2\cos k)\,\sigma_x + t_2\sin k\,\sigma_y + i\gamma\,\sigma_z$$

### 4.2 例外点条件

对于 $2\times 2$ 哈密顿量，特征多项式为
$$p(E) = E^2 - \mathrm{Tr}(H)E + \det H = 0$$

例外点要求判别式同时为零：
$$\Delta = \big[\mathrm{Tr}(H)\big]^2 - 4\det H = 0$$

### 4.3 Laguerre 根求法

迭代格式（立方收敛）：
$$z_{n+1} = z_n - \frac{m}{G \pm \sqrt{(m-1)(mH - G^2)}} \frac{f(z_n)}{f'(z_n)}$$

其中 $G = f'/f$，$H = G^2 - f''/f$，$m$ 为多项式次数。

### 4.4 双正交 Berry 联络与曲率

右/左本征矢量满足 $H|\psi_n^R\rangle = E_n|\psi_n^R\rangle$，$\langle\psi_n^L|H = E_n\langle\psi_n^L|$，且双正交归一化 $\langle\psi_n^L|\psi_m^R\rangle = \delta_{nm}$。

Berry 联络：
$$\mathcal{A}_n(\mathbf{k}) = i\langle\psi_n^L(\mathbf{k})|\nabla_{\mathbf{k}}|\psi_n^R(\mathbf{k})\rangle$$

Berry 曲率：
$$\Omega_n(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathcal{A}_n(\mathbf{k})$$

Zak 相位：
$$\gamma_{\mathrm{Zak}} = \int_{-\pi/a}^{\pi/a} \mathcal{A}(k)\,dk$$

Chern 数：
$$C_n = \frac{1}{2\pi} \iint_{\mathrm{BZ}} \Omega_n(\mathbf{k})\,dk_x\,dk_y$$

复能量绕数：
$$W = \frac{1}{2\pi i} \oint_C \frac{dE}{E}$$

### 4.5 四面体 NCO 求积

对参考四面体上的积分采用对称的 Newton-Cotes Open 规则：
$$\int_{\text{tet}} f(\mathbf{r})\,d^3r \approx V \sum_{q} w_q\,f(\mathbf{r}_q)$$

其中权重 $w_q$ 与节点 $\mathbf{r}_q$ 由 Silvester (1970) 的对称公式确定，对 5 次多项式精确。

### 4.6 非厄米薛定谔方程

$$i\partial_t |\psi(t)\rangle = H_{\text{eff}}|\psi(t)\rangle, \quad H_{\text{eff}} = H - i\Gamma$$

Lindblad 主方程：
$$\frac{d\rho}{dt} = -i[H,\rho] + \sum_j \Big( L_j\rho L_j^\dagger - \frac{1}{2}\{L_j^\dagger L_j, \rho\} \Big)$$

### 4.7 不完全 Beta 函数

$$I_x(p,q) = \frac{1}{B(p,q)} \int_0^x t^{p-1}(1-t)^{q-1}\,dt$$

使用 Soper 递降公式进行级数展开计算。

### 4.8 传递矩阵与李雅普诺夫指数

SSH 模型在能量 $E$ 处的传递矩阵：
$$T(E) = \begin{pmatrix} (E-i\gamma)/t_2 & -t_1/t_2 \\ 1 & 0 \end{pmatrix}$$

李雅普诺夫指数（逆局域化长度）：
$$\lambda = \lim_{N\to\infty} \frac{1}{N} \sum_{n=1}^{N} \ln \sigma_{\max}(T_n)$$

### 4.9 有限元弱形式

对非厄米 Helmholtz 方程 $[\nabla^2 + k^2 n^2(\mathbf{r})]\psi = E\psi$，离散后得到广义本征值问题：
$$(K + M_V + iM_W)\boldsymbol{\psi} = E\,M_{\text{mass}}\boldsymbol{\psi}$$

其中 $K$ 为刚度矩阵（离散 Laplacian），$M_V$、$M_W$ 分别为实部与虚部势场的质量矩阵。

---

## 5. 文件结构

```
019_synth_project/
├── main.py                      # 统一入口，零参数运行
├── hamiltonian_builder.py       # 非厄米哈密顿量构造
├── exceptional_point_solver.py  # Laguerre 方法求 EP
├── biorthogonal_topology.py     # Berry 相位 / Chern 数 / Zak 相位
├── brillouin_integrator.py      # 四面体 BZ 积分
├── nonherm_dynamics.py          # RKF45 非厄米动力学 / Lindblad
├── random_matrix_stats.py       # Ginibre 系综 / 不完全 Beta
├── mesh_discretization.py       # GMSH 读取 / FE 矩阵组装
├── parallel_sweep.py            # 多进程并行参数扫描
├── transfer_matrix.py           # 传递矩阵 / 李雅普诺夫指数
├── vandermonde_solver.py        # Vandermonde 求解 / 谱插值
├── potential_profile.py         # 复势场剖面
├── manifold_generator.py        # 参数流形生成
├── monte_carlo_sampler.py       # MCMC / 蒙特卡洛 EP 搜索
├── parameter_box.py             # 参数空间包围盒与自适应细分
├── triangulation.py             # Delaunay 三角剖分
└── README_博士级合成说明.md     # 本文档
```

---

## 6. 运行方式

```bash
cd Synthesis-project-python/019_synth_project
python main.py
```

无需安装额外依赖（仅需 NumPy）。程序将依次执行全部 15 个模块的演示计算，输出各阶段结果到终端。

---

## 7. 边界处理与数值鲁棒性

- **Hamiltonian 构造**：所有参数经过合法性检查（如 $t_2 \neq 0$、$q > 0$）。
- **Laguerre 根求法**：对分母趋于零、迭代次数超限等情形返回错误码，避免死循环。
- **双正交归一化**：检测零重叠情形并抛出异常，防止数值爆炸。
- **自适应 RKF45**：步长自动缩放，对刚性系统（如 Lindberg 型 ODE）采用严格误差控制；范数崩溃时自动减小步长。
- **不完全 Beta**：对 $x \notin [0,1]$ 或 $p,q \le 0$ 返回错误标志。
- **有限元矩阵**：刚度矩阵采用 cotangent 公式，对退化三角形自动返回零面积。
- **并行扫描**：使用 `multiprocessing.Pool`，构建器函数定义在模块顶层以确保可 pickle。
- **Vandermonde 求解**：Björck-Pereyra 算法自动检测重复节点并返回奇异标志。

---

## 8. 科学创新点与博士级难度

1. **多模型融合**：同时处理 PT 对称、SSH、Hofstadter 三种非厄米模型，涵盖连续与离散、1D/2D/3D。
2. **复平面 EP 搜索**：在复动量平面使用 Laguerre 方法寻找高次多项式的重根，需处理复变函数与数值稳定性。
3. **双正交拓扑**：非厄米系统的 Berry 相位不再实数量化，本项目完整实现了复 Berry 联络、曲率与 Chern 数计算。
4. **高维数值积分**：将四面体 NCO 规则应用于三维布里渊区，涉及参考-物理坐标映射与体积计算。
5. **开放量子动力学**：同时求解非厄米薛定谔方程与 Lindblad 主方程，涵盖纯态与混合态演化。
6. **跨尺度分析**：从随机矩阵统计（微观普适性）到有限元空间离散化（宏观连续模型），形成完整研究链条。
7. **非厄米皮肤效应**：通过传递矩阵与李雅普诺夫指数定量刻画边界态聚集。

---

## 9. 结论

本项目将 15 个独立的科研代码种子项目有机融合为一个面向**非厄米物理与例外点**前沿领域的博士级计算框架。每个种子项目都在合成系统中承担了真实、不可替代的科学角色，代码具备严格的边界处理与数值鲁棒性，可直接用于非厄米拓扑、开放量子系统、光子晶体等前沿课题研究。
