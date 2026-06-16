# 声子谱与热输运计算：高阶有限差分与稳定性分析

## 博士级合成项目说明

### 一、科学问题定义

本项目围绕**计算凝聚态物理**的前沿问题展开：**FCC 铜 (Cu) 晶体的声子色散关系与晶格热导率计算**。

核心科学挑战：
1. **高阶有限差分精度**对声子色散关系的色散误差控制
2. **von Neumann 稳定性分析**确定时间积分的 CFL 条件
3. **非谐声子散射**对热导率的温度依赖性
4. **声子动力学状态切换**（准谐 -> 非谐）的 SLDS 跟踪

### 二、原项目到科学问题的映射

| # | 种子项目 | 核心算法 | 映射到声子计算 |
|---|---------|---------|---------------|
| 1 | 1239_DurationModulatedDynamics | SLDS + Kalman滤波 + DSUP比率 | 声子动力学状态切换跟踪 (slds_phonon.py) |
| 2 | 341_eternity_tile | 非周期密铺邻接矩阵 adj(N,3) | Wigner-Seitz 原胞三角化与邻接编码 (lattice_geometry.py) |
| 3 | 195_coin_simulation | Bernoulli试验 + 运行平均/和 + streak | 声子占据数MC采样 + 统计收敛监测 (monte_carlo.py) |
| 4 | 995_r8sm | Sherman-Morrison秩1修正 + LU分解 | 缺陷声子Green函数 + 特征值修正 (sherman_morrison.py) |
| 5 | 170_chinese_remainder_theorem | CRT重构 + Bezout系数 | Brillouin区k点CRT索引映射 (bz_sampling.py) |
| 6 | 1149_BanerjeeLab | Euler自适应积分 + 稳态搜索 + 拟合 | 热输运ODE + 散射率温度依赖 (thermal_transport.py) |
| 7 | 1242_ce335805_PhotonDosReference | 光子DOS + 根搜索 + 频率积分 | 声子DOS计算 + van Hove奇点检测 (dos_calc.py) |
| 8 | 004_alpert_rule | Gauss-梯形混合求积 + 奇异积分 | DOS奇异积分 + 差分精度验证 (fd_stencil.py) |
| 9 | 758_mesh2d_to_medit | 网格格式转换 + 拓扑连接表 | 晶格网格拓扑与WS原胞网格化 (lattice_geometry.py) |
| 10 | 368_fd2d_poisson | 2D Poisson 5点差分模板 | 高阶Laplacian矩阵组装 (fd_stencil.py) |
| 11 | 746_md_parfor | Velocity-Verlet + 对势 + 能量守恒 | 晶格动力学MD模拟 (time_integration.py) |
| 12 | 067_ball_grid | 球内网格 + 八分面对称 | BZ球内k点采样 (bz_sampling.py) |
| 13 | 1250_fjarri_qsim | 截断Wigner + Bogoliubov变换 | Bogoliubov准粒子色散 + Wigner采样 (eigen_solver.py, monte_carlo.py) |
| 14 | 818_normal_ode | dy/dt = -t*y 精确解 | 时间积分器精度验证 (time_integration.py) |
| 15 | 218_coordinate_search | 模板搜索 + 步长收缩 | 力常数直接搜索拟合 (optimize_phonon.py) |

### 三、新增数学物理模型与核心公式

#### 3.1 晶格动力学基本方程

**运动方程**（Newton第二定律 + 简谐近似）:
$$m_i \ddot{u}_{i,\alpha} = -\sum_{j,\beta} \Phi_{i\alpha,j\beta} u_{j,\beta}$$

**动力学矩阵**（Fourier变换 + 质量归一化）:
$$D_{\alpha\beta}(\mathbf{q}) = \frac{1}{\sqrt{m_\alpha m_\beta}} \sum_l \Phi_{\alpha\beta}(l) e^{i\mathbf{q}\cdot\mathbf{R}_l}$$

**声子本征方程**:
$$D(\mathbf{q}) \mathbf{e}_\lambda(\mathbf{q}) = \omega_\lambda^2(\mathbf{q}) \mathbf{e}_\lambda(\mathbf{q})$$

#### 3.2 高阶有限差分模板

**2阶** (3点): $f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$

**4阶** (5点): $f''(x) \approx \frac{-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)}{12h^2}$

**6阶** (7点): 系数 $[1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]/h^2$

**8阶** (9点): 系数 $[-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]/h^2$

#### 3.3 稳定性分析 (von Neumann)

**放大因子**: $g = 1 - \frac{dt^2 \lambda}{2} \pm \sqrt{\left(\frac{dt^2 \lambda}{2}\right)^2 - dt^2 \lambda}$

**CFL条件**: $dt \leq \frac{2}{\omega_{\max}}$

#### 3.4 热输运 — Boltzmann输运方程

**晶格热导率**:
$$\kappa_{ab} = \frac{1}{V} \sum_\lambda C_\lambda v_{\lambda,a} v_{\lambda,b} \tau_\lambda$$

**模式热容**: $C_\lambda = k_B x^2 \frac{e^x}{(e^x - 1)^2}$, $x = \hbar\omega/k_BT$

**散射率** (Matthiessen规则):
- Umklapp: $\tau_U^{-1} = A_U \omega^2 T \exp(-\Theta_D/3T)$
- Normal: $\tau_N^{-1} = A_N \omega T^3$
- 边界: $\tau_B^{-1} = v_s/L$
- 同位素: $\tau_{iso}^{-1} = B_{iso} \omega^2$

**Callaway模型**: $\kappa = \frac{k_B}{2\pi^2 v_s} \left(\frac{k_BT}{\hbar}\right)^3 \int_0^{\Theta_D/T} \tau_c \frac{x^4 e^x}{(e^x - 1)^2} dx$

#### 3.5 Bose-Einstein统计与Wigner采样

**占据数**: $n_{BE} = \frac{1}{e^{\hbar\omega/k_BT} - 1}$

**Wigner分布**: $\sigma_q^2 = \frac{\hbar}{2m\omega} \coth\left(\frac{\hbar\omega}{2k_BT}\right)$

#### 3.6 Sherman-Morrison公式

$$(A - uv^T)^{-1} = A^{-1} + \frac{A^{-1}uv^TA^{-1}}{1 - v^TA^{-1}u}$$

#### 3.7 中国剩余定理 (CRT)

给定互素模数 $m_1, ..., m_k$，余数 $r_1, ..., r_k$:
$$f = \sum_i s_i t_i r_i \pmod{M}, \quad s_i = M/m_i, \quad s_i t_i \equiv 1 \pmod{m_i}$$

#### 3.8 SLDS声子动力学

$$z_t \sim \text{Cat}(\text{softmax}(Rx_{t-1} + r))$$
$$x_t = A_{z_t} x_{t-1} + b_{z_t} + \text{noise}$$
$$\text{DSUP} = \frac{\|(A-I)x + b\|}{\|(A-I)x + b\| + \|K \cdot \text{innovation}\|}$$

### 四、文件结构与修改说明

```
273_synth_project_Advanced/
├── main.py                  # 统一入口 (零参数运行)
├── lattice_geometry.py      # 晶格几何 (融合 ball_grid + eternity_tile + mesh2d_to_medit)
├── interatomic_potential.py # 原子间势与力常数 (融合 md_parfor + fd2d_poisson)
├── fd_stencil.py            # 高阶有限差分 (融合 fd2d_poisson + alpert_rule)
├── sherman_morrison.py      # 秩1修正求解 (融合 r8sm)
├── dynamical_matrix.py      # 动力学矩阵 (融合 r8sm + fd2d_poisson + md_parfor)
├── eigen_solver.py          # 本征值求解 (融合 r8sm + qsim_letter)
├── time_integration.py      # 时间积分 (融合 md_parfor + normal_ode + BanerjeeLab)
├── stability_analysis.py    # 稳定性分析 (融合 fd2d_poisson + alpert_rule + normal_ode)
├── thermal_transport.py     # 热输运BTE (融合 BanerjeeLab + PhotonDosReference)
├── dos_calc.py              # 声子DOS (融合 PhotonDosReference + alpert_rule + coin_simulation)
├── bz_sampling.py           # BZ采样 (融合 chinese_remainder_theorem + ball_grid + eternity_tile)
├── slds_phonon.py           # SLDS状态跟踪 (融合 DurationModulatedDynamics + normal_ode)
├── monte_carlo.py           # MC统计采样 (融合 coin_simulation + qsim_letter)
├── optimize_phonon.py       # 力常数优化 (融合 coordinate_search + BanerjeeLab)
└── README_博士级合成说明.md  # 本文档
```

共 **14 个 Python 模块** + 1 个 README。

### 五、运行方式

```bash
cd 273_synth_project_Advanced
python main.py
```

零参数运行，输出包括：
- 晶格几何与近邻壳层信息
- 有限差分模板精度验证
- Brillouin区采样与CRT索引
- 动力学矩阵与声子频率
- von Neumann稳定性分析结果
- MD模拟与能量守恒
- 声子DOS与van Hove奇点
- 热导率温度依赖
- Monte Carlo统计采样
- SLDS声子状态跟踪
- 力常数优化拟合

### 六、科学意义

1. **高阶差分精度控制**：6阶/8阶模板将色散误差从O(h²)降至O(h⁶)/O(h⁸)
2. **稳定性保障**：CFL条件确保辛积分器长期能量守恒
3. **多尺度热输运**：从声子色散到宏观热导率的完整计算链
4. **非谐效应跟踪**：SLDS方法捕捉准谐-非谐动力学转变
5. **统计可靠性**：Monte Carlo采样 + 运行平均收敛保证结果可复现

### 七、边界处理与数值鲁棒性

- 所有除法操作添加零保护 (`max(x, 1e-15)`)
- 指数函数参数截断 (`min(x, 500)`) 防止溢出
- Sherman-Morrison分母非零检查
- 特征值非负保护 (虚频处理)
- 自适应时间步的上下界限制
- 矩阵条件数监控
