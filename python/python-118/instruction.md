# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 项目描述：合金固液界面动力学分子模拟与分析

## 1. 项目简介
本项目模拟二元合金（典型如 Ni‑Cu）的固液界面结构与动力学行为。采用嵌入原子法（EAM）势能、速度 Verlet 时间积分器，结合 Nose‑Hoover 链恒温器在 NVT 系综下进行分子动力学模拟。通过对轨迹进行 Steinhardt 序参量分析、界面定位、毛细波谱、溶质扩散、径向分布函数谱分解以及高维参数空间的稀疏网格采样，提取界面的物理特征。最终由 `main.py` 编排全部计算流程（零点运行），其余模块负责具体的物理建模和数值算法。

## 2. 文件清单与职责

| 文件 | 职责 |
|------|------|
| `utils_numeric.py` | 底层数值工具：伪随机数生成（LCG）、Maxwell‑Boltzmann 速度采样、Bessel 零点计算、广义 Laguerre 多项式、Gegenbauer 多项式、安全开方、边界裁剪等。几乎所有其它模块都依赖于它。 |
| `lattice_builder.py` | 晶体与液相的初始结构构建器。提供 FCC/BCC 晶格生成、液相随机投放、固液双相界面构型、近邻列表构建以及热位移。 |
| `eam_potential.py` | EAM 势能模型。实现二元合金的 Morse 对势、电子密度函数、嵌入能的解析表达式，并提供统一的 `compute_forces_and_energies` 接口以供积分器调用。包含生成 Ni‑Cu 参数集的工厂函数。 |
| `velocity_verlet.py` | NVT 系综的速度 Verlet 积分器，耦合 Nose‑Hoover 链恒温器。核心方法 `step` 接受当前位置、速度、质量、物种、势能对象和盒子，返回更新后的位置、速度、能量、温度等。 |
| `structure_factor.py` | 局部结构分析器，实现球谐展开的 Steinhardt 键序参量（核心阶数 l=6）以区分固／液相，并提供近邻列表构建和径向分布函数（RDF）计算。 |
| `interface_dynamics.py` | 界面追踪与动力学分析。基于序参量权重定位界面高度场，计算界面粗糙度、毛细波涨落谱和团簇连通性；同时包含毛细波理论模型，用于从谱拟合界面自由能。 |
| `solute_diffusion.py` | 溶质浓度场的欧拉网格分析。利用 Gaussian 涂抹将原子分布映射到规则网格，计算浓度、梯度、Laplacian、有效分凝系数，并通过均方位移（MSD）拟合扩散系数；配备三维插值器用于任意点浓度查询。 |
| `sparse_quadrature.py` | 稀疏网格求积与参数空间采样。基于 Smolyak 构造生成多维 Gauss‑Legendre 节点与权重，提供积分和期望值计算；`AlloyPhaseSpaceSampler` 封装了参数空间采样与响应函数求值。 |
| `spectral_decomposition.py` | 谱分解与正交多项式展开。使用广义 Laguerre 多项式对 RDF 进行展开并计算结构指数；用 Gegenbauer 多项式展开角分布；同时提供谱的 Shannon 熵和参与比以量化结构无序度。 |

说明：`main.py` 在运行时依次调用上述模块。也可能涉及少量辅助参数文件（如 `config.py`），实现者可根据需要自行设定物理常数，不作为必须复现的部分。

## 3. 模块功能详解与接口关系

### 3.1 `utils_numeric.py` — 数值基础设施
- **`RandomState` 类**：基于线性同余生成器（LCG）的伪随机状态。提供 `uniform_ab(n, a, b)` 在 [a, b] 生成均匀分布，以及 `maxwell_boltzmann(n, T, m)` 生成符合 Maxwell‑Boltzmann 分布的速度向量（3 维分量）。其他模块通过该生成器获得随机数。
- **`bessel_zero_newton(n, k, kind=1)`**：用 Halley‑Newton 法计算第 k 个 Bessel J_n 或 Y_n 的正零点，用于谱分解中的模态。
- **`laguerre_polynomial_alpha(x, n, alpha)`**：通过三项递推求广义 Laguerre 多项式 L_n^{(alpha)}(x) 的值。
- **`gegenbauer_polynomial(x, n, lambda_)`**：通过递推求 Gegenbauer 多项式 C_n^{(lambda)}(x) 的值。
- **`check_bounds(x, lower, upper)`**：输出裁剪到区间，并给出警告。
- **`safe_sqrt(x, eps)`**：对非负值开方，避免负数 NaN。
- 其它辅助：`relative_convergence_check` 等。

### 3.2 `lattice_builder.py` — 初始构型生成
- **`LatticeBuilder` 类**：
  - 构造参数：晶格类型（fcc/bcc）、晶格常数 a0、可选随机种子。
  - **`build_fcc_block(nx, ny, nz)`** / **`build_bcc_block`**：生成指定晶胞数、以基元位置叠加的完美晶体坐标。
  - **`build_liquid_block(n_atoms, box)`**：在盒子内随机投放原子，并通过拒绝采样保证最小距离，模拟液相无序结构。
  - **`build_solid_liquid_interface(...)`**：组合固体块与液体块，生成固液界面构型。返回原子位置、物种索引（0=A，1=B）、盒子大小和固相布尔掩码 `is_solid`。液相原子位于固相上方，溶质随机分配到两种相中。
  - **`build_neighbor_list(positions, box, r_cut)`**：对给定原子坐标构建近邻表（返回三元组：邻居索引、距离、距离向量），供后续结构分析和力计算使用。
  - **`apply_thermal_displacement(positions, T, masses_amu, species_idx)`**：基于简谐振动模型给原子位置施加随机热位移。

### 3.3 `eam_potential.py` — EAM 势能
- **`EAMPotential` 类**：
  - 构造函数接收若干物理参数以定义二元合金 EAM：Morse 对势强度的矩阵 `D_mat`、指数系数 `alpha_mat`、平衡距离 `r0_mat`；嵌入函数参数 `A_emb`, `B_emb`；电子密度参数 `f_rho`, `beta_rho`；截断距离 `r_cut` 及原子质量。
  - 关键方法：
    * `pair_potential(r, si, sj)`：给定原子间距和两物种，计算带截断平滑的 Morse 对势。
    * `pair_force_magnitude(r, si, sj)`：对势的径向导数的负值。
    * `electron_density(r, sj)`：原子 j 在距离 r 处贡献的电子密度。
    * `embedding_energy(rho_bar, si)` 和 `embedding_derivative(rho_bar, si)`：嵌入能及其导数。
    * **`compute_forces_and_energies(positions, species_idx, box)`**：主接口。计算每个原子的电子密度、嵌入能、对势，并累加总能量、原子受力和维里。力计算包括对势部分和嵌入能对密度的梯度贡献，使用最小镜像约定。
- **`eam_parameterized_alloy(species_pair)`**：返回预设 Ni‑Cu 参数的 `EAMPotential` 实例。

### 3.4 `velocity_verlet.py` — NVT 分子动力学积分器
- **`VelocityVerletNVT` 类**：
  - 初始化参数：时间步长 `dt`、目标温度 `T_target`、Nose‑Hoover 链长度、热浴质量因子。
  - 内部维护热浴变量 `v_xi`、`Q` 等，按体系自由度自动初始化。
  - **`step(positions,
