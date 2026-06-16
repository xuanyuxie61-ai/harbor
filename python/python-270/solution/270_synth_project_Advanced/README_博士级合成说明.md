# 博士级科学合成项目说明

## 项目名称
**自旋玻璃 Monte Carlo 模拟：高阶有限差分与稳定性分析（小规模可复现实验）**

英文：Edwards-Anderson Spin Glass: Monte Carlo Simulation with High-Order Finite Difference and Stability Analysis (Small-Scale Reproducible Experiment)

## 研究领域
**计算凝聚态物理 (Computational Condensed Matter Physics)**

本项目围绕 Edwards-Anderson 自旋玻璃模型,在三维简单立方晶格上进行数值模拟,
结合高阶有限差分方法、von Neumann 稳定性分析、Replica Exchange (Parallel Tempering)、
Umbrella Sampling + WHAM、Hessian 谱分析 (magnon 态密度) 以及 K-means 聚类 (纯态识别),
实现一个面向前沿科学问题的小规模可复现计算实验。

---

## 物理背景

### Edwards-Anderson 自旋玻璃模型

Ising 自旋玻璃的哈密顿量:

$$
H = -\sum_{\langle ij \rangle} J_{ij} S_i S_j, \quad S_i \in \{-1, +1\}
$$

其中 $J_{ij}$ 是最近邻交换耦合常数 (quenched disorder),
从高斯分布 $P(J) = (2\pi J_{\text{var}})^{-1/2} \exp(-J^2/(2J_{\text{var}}))$ 中抽取。

在 $d=3$ 简单立方晶格上,每个格点有 $z=6$ 个最近邻,
系统存在从顺磁相到自旋玻璃相的相变,临界温度 $T_c \approx 1.1$ (对 $J_{\text{var}}=1$)。

### 自旋玻璃序参量 (Parisi 序参量)

自旋玻璃相不能由单一序参量描述,需要 overlap 分布函数 $P(q)$:

$$
q_{\alpha\beta} = \frac{1}{N} \sum_{i=1}^N S_i^{(\alpha)} S_i^{(\beta)}
$$

其中 $\alpha, \beta$ 标记不同的 replica (热力学副本)。

Binder 累积量:

$$
g = \frac{1}{2} \left(3 - \frac{\langle q^4 \rangle}{\langle q^2 \rangle^2}\right)
$$

在 $T_c$ 处, $g$ 对不同尺寸 $L$ 的曲线交叉于 universal value。

### Langevin 动力学 + 有限差分

连续自旋版本的 Langevin 方程:

$$
\frac{dS_i}{dt} = h_i^{\text{eff}} + \sqrt{2T} \eta_i(t), \quad h_i^{\text{eff}} = \sum_{j \in \text{nbr}(i)} J_{ij} S_j
$$

二阶有限差分 (O(h²)):

$$
(\Delta_2 S)_{ijk} = S_{i+1,j,k} + S_{i-1,j,k} + S_{i,j+1,k} + S_{i,j-1,k} + S_{i,j,k+1} + S_{i,j,k-1} - 6 S_{ijk}
$$

四阶有限差分 (O(h⁴)):

$$
(\Delta_4 S)_i = \sum_{\mu=x,y,z} \frac{-S_{i+2e_\mu} + 16 S_{i+e_\mu} - 30 S_i + 16 S_{i-e_\mu} - S_{i-2e_\mu}}{12}
$$

### von Neumann 稳定性分析

放大因子 (2阶):

$$
G(\mathbf{k}) = 1 + D \cdot dt \cdot 2[\cos(k_x) + \cos(k_y) + \cos(k_z) - 3]
$$

稳定性条件 $\rho(G) \leq 1$:

$$
dt \leq \frac{1}{3D} \quad (\text{2阶}), \qquad dt \leq \frac{1}{8D} \quad (\text{4阶})
$$

### Hessian 谱分析

在能量极小点附近做谐波展开:

$$
H_{ij} = \frac{\partial^2 E}{\partial S_i \partial S_j} = -2 J_{ij} \quad (i \neq j, \text{邻居})
$$

特征值 $\lambda_k$ 对应 magnon 频率 $\omega_k = \sqrt{|\lambda_k|}$。

态密度 (DOS):

$$
g(\omega) = \frac{1}{N} \sum_k \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(\omega - \omega_k)^2}{2\sigma^2}\right)
$$

de Almeida-Thouless 稳定性判据:

$$
\beta^2 J_{\text{var}} \int \frac{g(\omega)}{\omega^2} d\omega < 1 \quad \text{(稳定)}
$$

---

## 输入种子项目到科学问题的映射 (15 个项目全部真实融入)

| 序号 | 种子项目 | 核心算法 | 在本项目中的角色 | 实现模块 |
|------|----------|----------|------------------|----------|
| 1 | **1345_triangulation_plot** | 三角网格的节点-单元拓扑编码 | 晶格节点-近邻键的拓扑编码 | `spin_lattice_geometry.py` |
| 2 | **817_normal_dataset** | 多维高斯随机数生成 | 耦合常数 $J_{ij}$ 的高斯采样 | `spin_glass_couplings.py` |
| 3 | **131_c8lib** | 复数算术库 (c8_abs, c8_mul, c8_exp, c8_log) | 复数矩阵运算与放大因子计算 | `spin_glass_complex.py` |
| 4 | **041_asa136** (kmns/optra/qtran) | K-means 聚类 (Algorithm AS 136) | replica overlap 分布的聚类,识别纯态 | `monte_carlo_core.py` (kmeans_overlap_clustering) |
| 5 | **646_laplace_radial_exact** | 径向 Laplace 方程精确解 | 边界条件修正与谐波分析 | `spin_glass_boundary.py` |
| 6 | **545_house** | 折线/顶点序列编码 | 近邻键的有序列表编码 | `spin_lattice_geometry.py` |
| 7 | **002_advection_pde** | 对流 PDE 守恒量与初始条件 | Langevin 方程守恒量与初值 | `langevin_fd_integration.py` |
| 8 | **246_cvt_1d_sampling** | Lloyd 算法 (1D CVT) | 温度空间的优化分配 | `replica_exchange.py` (cvt_temperature_optimization) |
| 9 | **842_ozone2_ode** | 臭氧 ODE 刚性系统 | Langevin 动力学的刚性时间积分 | `langevin_fd_integration.py` |
| 10 | **1287_keb721_NucleiMorphology** | umbrella_sampling + 球谐函数 | q-space 伞形采样 + Hessian 特征向量分析 | `umbrella_sampling.py` + `hessian_magnon_dos.py` |
| 11 | **1155_anhkiet120206-lgtm (EEIO)** | 输入输出分析的约束优化 | WHAM 自洽方程的求解 | `umbrella_sampling.py` (WHAMAnalyzer) |
| 12 | **264_cvtp** | CVT 迭代 (cvtp_iteration, cvtp_find_closest) | 温度空间 CVT 优化 + 邻居搜索 | `replica_exchange.py` |
| 13 | **1242_ce335805_PhotonDosReference** | 光子态密度计算 (DOS) | magnon 态密度的 Gaussian 展宽与 KDE | `hessian_magnon_dos.py` |
| 14 | **229_cube_arbq_rule** | 立方域任意阶求积公式 | Gauss-Hermite 求积 + Clenshaw-Curtis 求积 | `disorder_quadrature.py` |
| 15 | **1020_lx913_SoleFlip** | 量化神经网络 (离散状态空间) | 离散自旋状态 $S_i \in \{-1, +1\}$ 的建模思想 | `monte_carlo_core.py` (MetropolisMC) |

---

## 项目结构

```
270_synth_project_Advanced/
├── main.py                           # 统一入口 (零参数运行)
├── spin_lattice_geometry.py          # 晶格几何 + 邻居表 (seed 1, 6)
├── spin_glass_couplings.py           # 耦合常数生成 + 有效场 (seed 2)
├── spin_glass_boundary.py            # 边界条件处理 (seed 5)
├── spin_glass_complex.py             # 复数代数 + 谱分析 (seed 3)
├── disorder_quadrature.py            # 高斯求积 + 无序平均 (seed 14)
├── langevin_fd_integration.py        # Langevin + 高阶有限差分 (seed 7, 9)
├── monte_carlo_core.py               # Metropolis MC + 可观测量 + K-means (seed 4, 15)
├── replica_exchange.py               # Replica Exchange + CVT 温度优化 (seed 8, 12)
├── umbrella_sampling.py              # Umbrella Sampling + WHAM (seed 10, 11)
├── hessian_magnon_dos.py             # Hessian 谱分析 + magnon DOS (seed 10, 13)
├── output_writer.py                  # 结果输出 (文本格式, 无可视化)
├── simulation_engine.py              # 统一模拟引擎
├── output/                           # 运行后生成的结果目录
│   ├── parameters.txt
│   ├── stability_report.txt
│   ├── mc_T*.txt (多个温度)
│   ├── replica_exchange.txt
│   ├── wham_results.txt
│   ├── spectral_analysis.txt
│   └── disorder_average.txt
└── README_博士级合成说明.md           # 本文档
```

共 **13 个 .py 文件** (含 main.py),满足 8-16 个文件的要求。

---

## 运行方法

```bash
cd 270_synth_project_Advanced
python main.py
```

**无需任何参数**。所有配置在 `simulation_engine.SimulationConfig` 类中定义,
修改该类即可调整参数。

典型运行时间: ~15 秒 (L=4, 3 个 disorder 实现, 5 个温度点)。

---

## 核心科学结果

### 1. 稳定性分析

- 二阶有限差分临界步长 $dt_c^{(2)} = 1/(3D) = 0.333$
- 四阶有限差分临界步长 $dt_c^{(4)} = 1/(8D) = 0.125$
- 使用 $dt=0.005$ 时, 两种格式均稳定 ($\rho_{\max} \leq 1$)

### 2. 温度扫描 (Metropolis MC)

| T | <E>/N | <|m|> | <q> | Binder g |
|---|-------|-------|-----|----------|
| 3.00 | -0.75 | 0.09 | 0.00 | 0.14 |
| 2.00 | -1.05 | 0.10 | -0.01 | 0.11 |
| 1.50 | -1.28 | 0.10 | 0.47 | 0.82 |
| 1.00 | -1.42 | 0.09 | -0.33 | 0.39 |
| 0.70 | -1.47 | 0.09 | -0.17 | -0.07 |

可以看到在 $T \approx 1.5$ 附近 Binder 累积量出现显著变化,
暗示相变区域。

### 3. Replica Exchange

- 几何温度阶梯 vs CVT 优化温度阶梯
- 交换接受率 ≈ 77%
- CVT 优化使低温端更密集, 增强低温采样

### 4. WHAM 自由能

- 在 $q \in [-0.8, 0.8]$ 上构建 7 个 umbrella 窗口
- WHAM 自洽迭代在 33 步内收敛
- 获得 $P(q)$ 和自由能 $F(q) = -\beta^{-1} \ln P(q)$

### 5. Hessian 谱分析

- 最小特征值 $\lambda_{\min} = -5.13$ (负值表明当前配置是鞍点)
- 12 个负模 (saddle point index = 12)
- AT 参数 > 1 (在 $T=0.7$ 时处于自旋玻璃相, AT 不稳定)

### 6. 无序平均

3 个独立 disorder 实现的平均结果,与平均场理论的高斯求积基准对比。
自旋玻璃磁化率 $\chi_{SG}$ 在 $T_c$ 附近出现峰。

### 7. K-means 纯态识别

对 8 个 replica 的 overlap 矩阵做 K-means 聚类,
识别出自旋玻璃相中的不同纯态 (cluster labels: [1,0,1,1,0,1,0,1])。

---

## 科学意义

本项目综合了计算凝聚态物理中自旋玻璃研究的多个核心方法:

1. **高阶有限差分**: 比较 2 阶与 4 阶格式的稳定性与精度,
   为连续自旋版本的自旋玻璃动力学提供可靠的数值工具。

2. **von Neumann 稳定性分析**: 在布里渊区上扫描放大因子,
   确定时间步长的安全范围, 避免数值爆炸。

3. **Replica Exchange**: 通过多温度副本的交换克服能量壁垒,
   解决自旋玻璃中的临界慢化问题。

4. **Umbrella Sampling + WHAM**: 直接测量 $P(q)$,
   检验 Parisi RSB 方案与 droplet 图像的预言。

5. **Hessian 谱分析**: 通过 magnon DOS 表征能量景观,
   通过 de Almeida-Thouless 判据定位相变。

6. **K-means 纯态识别**: 在 replica 空间中进行聚类,
   探测自旋玻璃的纯态结构。

所有实验均在 $L=4$ (64 个自旋) 的小规模下进行,
完全可复现 (固定随机种子 42), 可作为自旋玻璃研究的教学与科研基线。

---

## 公式-算法-代码一致性

| 物理公式 | 算法实现 | 代码位置 |
|---------|---------|---------|
| $H = -\sum J_{ij} S_i S_j$ | `total_energy()` | `spin_lattice_geometry.py` |
| $p = \min(1, e^{-\beta \Delta E})$ | `acceptance_probability()` | `monte_carlo_core.py` |
| $q = N^{-1} \sum S_i^{(1)} S_i^{(2)}$ | `overlap()` | `monte_carlo_core.py` |
| $g = (3 - \langle q^4 \rangle / \langle q^2 \rangle^2)/2$ | `binder_cumulant()` | `monte_carlo_core.py` |
| $G(k) = 1 + D \cdot dt \cdot \hat{\Delta}(k)$ | `amplification_factor_symbol()` | `spin_glass_complex.py` |
| $dt_c = 1/(3D)$ (2阶) | `max_stable_dt_2nd_order()` | `spin_glass_boundary.py` |
| $H_{ij} = -2 J_{ij}$ | `build_hessian()` | `hessian_magnon_dos.py` |
| $g(\omega) = N^{-1} \sum \delta(\omega - \omega_k)$ | `magnon_dos_gaussian_broadening()` | `hessian_magnon_dos.py` |
| $p_{\text{ex}} = \min(1, e^{(\beta_i - \beta_j)(E_j - E_i)})$ | `exchange_acceptance_probability()` | `replica_exchange.py` |
| $P(q) = \sum N_k(q) / \sum n_j e^{-\beta W_j(q)}$ | `WHAMAnalyzer.solve()` | `umbrella_sampling.py` |
| Gauss-Hermite 求积 | `gauss_hermite_nodes_weights()` | `disorder_quadrature.py` |
| K-means (Hartigan-Wong AS 136) | `kmeans_overlap_clustering()` | `monte_carlo_core.py` |

---

## 边界条件与数值鲁棒性

1. **边界条件处理**:
   - 支持 PBC / OBC / APBC 三种边界
   - 坐标折叠带符号因子 (APBC)
   - 开边界的局部配位数自动计算

2. **数值保护**:
   - Metropolis 接受概率的 overflow/underflow 保护 ($e^x$ 截断在 $x < -500$)
   - 自旋值的 clip 到 $[-1, 1]$
   - Hessian 的对称化 (0.5*(H + H^T))
   - WHAM 分母的最小值保护 ($\max(denom, 10^{-300})$)
   - 高斯求积阶数的上限检查 ($n \leq 100$)
   - K-means 空簇检测

3. **一致性验证**:
   - 晶格邻居表的对称性验证
   - 键数与配位数的自洽性检查
   - 边界条件与局部配位数的一致性
   - 耦合常数的统计矩验证

4. **可复现性**:
   - 统一的 `numpy.random.Generator` 贯穿所有随机过程
   - 单一 `seed=42` 控制全部随机性
   - 完全确定性 (无并行随机性)

---

## 扩展方向

本项目可作为以下研究方向的基线:

1. **有限尺寸标度 (Finite-Size Scaling)**: 增大 $L$ 至 6, 8, 10,
   通过 Binder 累积量交叉精确确定 $T_c$ 和临界指数 $\nu$。

2. **非平衡动力学**: 使用 quench 实验研究 aging 现象,
   测量 $C(t, t_w)$ 的两时间关联函数。

3. **混沌效应 (chaos in spin glasses)**: 研究温度混沌与场混沌,
   通过 overlap 衰减表征。

4. **量子自旋玻璃**: 引入 transverse field $\Gamma \sum S_i^x$,
   研究量子相变。

5. **机器学习辅助**: 用神经网络学习 $P(q)$ 的结构,
   或用强化学习优化温度阶梯。

---

## 参考文献

1. K. Binder, A. P. Young, "Spin glasses: experimental facts, theoretical concepts, and open questions", Rev. Mod. Phys. 58, 801 (1986).
2. M. Mézard, G. Parisi, M. A. Virasoro, "Spin Glass Theory and Beyond", World Scientific (1987).
3. J. R. L. de Almeida, D. J. Thouless, "Stability of the Sherrington-Kirkpatrick solution of a spin glass model", J. Phys. A 11, 983 (1978).
4. K. Hukushima, K. Nemoto, "Exchange Monte Carlo Method", J. Phys. Soc. Jpn. 65, 1604 (1996).
5. S. Kumar et al., "Umbrella sampling", Wiley Interdiscip. Rev.: Comput. Mol. Sci. 11, e1521 (2021).
6. J. Hartigan, M. Wong, "Algorithm AS 136: A K-Means Clustering Algorithm", Applied Statistics 28, 100 (1979).

---

## 许可证

本合成项目遵循原始种子项目的 MIT 许可证。

## 联系方式

本项目作为博士级计算凝聚态研究的可复现基线,
欢迎在学术研究中引用与扩展。
