# 格点 QCD：夸克传播子与规范场采样 —— 高阶有限差分与稳定性分析

## 博士级科研合成项目说明文档

### 项目定位

本项目围绕**格点量子色动力学（Lattice QCD）** 的核心前沿问题展开：
在欧几里得时空 4 维超立方格子（4⁴ = 256 个格点）上，以 **SU(2)** 规范群为对象，研究：

1. **夸克传播子（Quark Propagator）** 的数值求解：Wilson-Dirac 算子的逆
2. **规范场组态的蒙特卡罗采样**：Metropolis、Langevin、仿生 Levy 飞行等多种算法
3. **高阶协变有限差分**：2阶、4阶、6阶精度的协变导数与协变 Laplace 算子
4. **谱分析与稳定性**：Arnoldi 迭代求本征值、Banks-Casher 关系、谱密度
5. **拓扑荷与分形维数**：Clover 场强张量、拓扑磁化率、盒计数维数
6. **规范轨道分析**：Landau 规范固定、置换距离、Ulam 距离
7. **码字优化规范固定**：将规范固定转化为二进制码的重叠最小化问题
8. **多任务耦合本征值估计**：借鉴 TC-MTLR 的多任务学习框架
9. **连续极限外推与 Newton 插值**：格距 a → 0 的外推

---

### 种子项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 在本项目中的角色 |
|------|---------|---------|----------------|
| 1 | `1007_snel-repo_spinal-population-dynamics-paper` | 脊线轨迹演化 | `gauge_sampler.py` 的 SpinalTrajectorySampler：沿配置空间的脊线轨迹做 Langevin 采样 |
| 2 | `378_fem_to_gmsh` | FEM 网格生成 | `lattice_geometry.py`：构建 4D 超立方格子、plaquette 枚举、格点索引 |
| 3 | `866_permutation_distance` | Ulam 距离、Kendall-tau | `gauge_orbit.py`：基于置换的规范场组态距离度量 |
| 4 | `192_closest_point_brute` | 暴力最近点搜索 | `gauge_orbit.py`：规范轨道上的最近点搜索（暴力搜索最优规范变换） |
| 5 | `1348_triangulation_quality` | 三角形质量度量 | `plaquette_topology.py`：plaquette 质量分析（类比 FEM 单元质量） |
| 6 | `446_fractal_coastline` | 分形维数（盒计数） | `plaquette_topology.py`：拓扑荷支撑集的分形维数 |
| 7 | `927_pwl_interp_2d` | 分段线性插值 | `interpolation_utils.py`：多维多重线性插值、规范协变插值 |
| 8 | `1081_FranciscoHS_toy-model-cis-code` | 二进制码构造、重叠约化 | `codeword_gauge_fix.py`：码字规范固定（边交换降重叠） |
| 9 | `1089_HarrisonFah_TC-MTLR` | 多任务学习、时间卷积 | `coupled_eigenvalue.py`：耦合本征值估计、时间谱预测 |
| 10 | `003_allen_cahn_pde` | 高阶有限差分模板 | `gauge_field.py`：2/4/6 阶协变有限差分算子 |
| 11 | `130_bvp_shooting` | 打靶法解 BVP | `quark_propagator.py`：迭代法解 Wilson-Dirac 方程（类打靶思路） |
| 12 | `543_histogramize` | 直方图分箱 | `spectral_analysis.py`：谱密度直方图估计 |
| 13 | `1057_Chandan118_Bio-Inspired-Navigation` | 仿生导航 Levy 飞行 | `gauge_sampler.py`：BioInspiredSampler 的 Levy 飞行探索 |
| 14 | `1159_KadelkaLab_nondegenerate-canalization` | 布尔函数枚举、管道化计数 | `canalization_topology.py`：B* 计数、拓扑扇区枚举 |
| 15 | `800_newton_interp_1d` | Newton 多项式插值 | `spectral_analysis.py` 和 `interpolation_utils.py`：Newton 插值求谱函数与连续极限 |

---

### 核心数学物理公式

#### 1. Wilson 规范作用量
$$S_G = \beta \sum_{x,\, \mu<\nu} \left[ 1 - \frac{1}{2} \mathrm{Re}\,\mathrm{Tr}\, U_{\mu\nu}(x) \right]$$

其中 $\beta = 4/g^2$（SU(2)），$U_{\mu\nu}(x) = U_\mu(x) U_\nu(x+\hat\mu) U_\mu^\dagger(x+\hat\nu) U_\nu^\dagger(x)$ 为 plaquette。

#### 2. Wilson-Dirac 算子
$$(D_W \psi)(x) = \psi(x) - \kappa \sum_{\mu=0}^{3} \left[ (1-\gamma_\mu) U_\mu(x) \psi(x+\hat\mu) + (1+\gamma_\mu) U_\mu^\dagger(x-\hat\mu) \psi(x-\hat\mu) \right]$$

跳跃参数 $\kappa = 1/(2(m_0 a + 4))$，临界值 $\kappa_c = 1/8$。

#### 3. 高阶协变有限差分

**2 阶**：$\nabla_\mu^{(2)} \psi(x) = [U_\mu(x)\psi(x+\hat\mu) - \psi(x)]/a$

**4 阶**：$\nabla_\mu^{(4)} \psi(x) = [-U_\mu(x)U_\mu(x+\hat\mu)\psi(x+2\hat\mu) + 8 U_\mu(x)\psi(x+\hat\mu) - 3\psi(x)]/(6a)$

**6 阶**：$\nabla_\mu^{(6)} \psi(x) = [U\cdots U\psi(x+3\hat\mu) - 9 U\cdots U\psi(x+2\hat\mu) + 45 U\psi(x+\hat\mu) - 20\psi(x)]/(60a)$

#### 4. 协变 Laplace 算子（4 阶）
$$\Delta^{(4)} \psi(x) = \frac{1}{12 a^2} \sum_\mu \left[ -(U_1 U_2 \psi(x{+}2\hat\mu) + \text{h.c.}) + 16 (U_1 \psi(x{+}\hat\mu) + \text{h.c.}) - 30\psi(x) \right]$$

#### 5. Banks-Casher 关系
$$\langle \bar\psi \psi \rangle = -\frac{\pi\, \rho(0)}{V}$$
其中 $\rho(0)$ 为 Dirac 算子谱密度在零点的值，$V$ 为格子体积。

#### 6. 拓扑荷（Clover 定义）
$$Q = \frac{1}{32\pi^2} \sum_x \varepsilon_{\mu\nu\rho\sigma}\, \mathrm{Tr}\, F_{\mu\nu}(x) F_{\rho\sigma}(x)$$

#### 7. 拓扑磁化率（Witten-Veneziano）
$$\chi_t = \frac{\langle Q^2 \rangle}{V}, \qquad m_{\eta'}^2 = \frac{2 N_f \chi_t}{f_\pi^2}$$

#### 8. 分形维数（盒计数）
$$D_{\mathrm{box}} = -\lim_{\varepsilon \to 0} \frac{\log N(\varepsilon)}{\log \varepsilon}$$

#### 9. 布尔函数计数
$$B^*(n) = 2^{2^n} - 2((-1)^n - n) + \sum_{k=1}^{n} (-1)^k \binom{n}{k} 2^{k+1} 2^{2^{n-k}}$$

#### 10. Landau 规范泛函
$$F_U[\Omega] = \sum_{x,\mu} \mathrm{Re}\,\mathrm{Tr}\, \Omega(x) U_\mu(x) \Omega^\dagger(x+\hat\mu)$$

#### 11. Ulam 距离
$$d_{\mathrm{Ulam}}(\pi, \sigma) = N - \mathrm{LIS}(\pi^{-1} \sigma)$$
其中 $\mathrm{LIS}$ 为最长递增子序列长度。

---

### 项目文件结构

```
237_synth_project_Advanced/
├── main.py                     # 统一入口，零参数运行
├── constants.py                # 物理常数、SU(2) 代数、Dirac γ 矩阵
├── lattice_geometry.py         # 格子几何、plaquette 枚举（来自 FEM 网格）
├── gauge_field.py              # SU(2) 规范场、高阶有限差分（来自 Allen-Cahn）
├── wilson_dirac.py             # Wilson-Dirac 算子、条件数估计
├── quark_propagator.py         # 夸克传播子求解器（Jacobi/CGNE/BiCGSTAB）
├── gauge_sampler.py            # 蒙特卡罗采样（脊线轨迹、Levy 飞行）
├── spectral_analysis.py        # 谱分析（Arnoldi、直方图、Banks-Casher）
├── gauge_orbit.py              # 规范轨道（暴力搜索、Landau 规范、置换距离）
├── plaquette_topology.py       # Plaquette 质量、拓扑荷、分形维数
├── canalization_topology.py    # 拓扑扇区枚举（布尔管道化）
├── interpolation_utils.py      # 多重线性插值、规范协变插值、Newton 插值
├── codeword_gauge_fix.py       # 码字规范固定（来自 CIS code）
├── coupled_eigenvalue.py       # 耦合本征值估计（来自 TC-MTLR）
└── README_博士级合成说明.md     # 本说明文档
```

共计 **15 个 Python 源文件 + 1 个中文说明文档**。

---

### 运行方法

本项目零参数运行：

```bash
cd /mnt/data/zpy/sci-swe/source code/Synthesis-project-python/237_synth_project/237_synth_project_Advanced
python main.py
```

无需安装额外依赖，仅使用 `numpy`（Python 标准科学计算库）。

运行流程：
1. 初始化 4⁴ 格子与冷启动规范场
2. 三种算法热化（Metropolis / Langevin / Levy 飞行）
3. Plaquette 质量分析与拓扑荷计算
4. 2/4/6 阶协变有限差分测试
5. Wilson-Dirac 算子构造与夸克传播子求解
6. Arnoldi 谱分析
7. 规范轨道分析与 Landau 规范固定
8. 码字优化规范固定
9. 耦合本征值估计
10. 拓扑扇区枚举与 Newton 插值外推

---

### 解决的科学问题

本项目实现了格点 QCD 中以下关键科学计算任务：

1. **夸克传播子的数值求解**：通过 CGNE、BiCGSTAB、Jacobi 三种迭代法求解 Wilson-Dirac 方程，并对比收敛性与稳定性。

2. **规范场组态的高效采样**：融合脊线轨迹 Langevin 动力学、仿生 Levy 飞行探索、经典 Metropolis 算法，实现 SU(2) 规范场的蒙特卡罗采样。

3. **高阶有限差分的精度验证**：系统实现 2/4/6 阶协变有限差分模板，比较离散化误差随阶数的收敛行为。

4. **拓扑结构的识别与量化**：通过 Clover 场强张量定义计算拓扑荷，分析拓扑荷支撑集的分形维数，估计拓扑磁化率。

5. **规范不变量的计算**：计算 plaquette 期望值、谱密度、Banks-Casher 手征凝聚估计。

6. **规范固定的组合优化视角**：将 Landau 规范固定与码字重叠最小化问题结合，提供新的数值思路。

7. **连续极限外推**：使用 Newton 多项式插值与最小二乘拟合，从多个格距的观测值外推到 a = 0。

---

### 工程鲁棒性与边界处理

- **参数边界检查**：所有构造函数均检查 kappa < kappa_c、beta > 0、m0 ≥ 0、Ns ≥ 2 等物理约束
- **周期性边界条件**：格子的空间和时间方向均实现周期性边界条件
- **SU(2) 重投影**：所有 SU(2) 矩阵操作后通过四元数归一化重投影，防止浮点漂移
- **数值稳定性**：Arnoldi 迭代中检测 breakdown、CGNE 中检测分母为零、BiCGSTAB 中检测 rho/omega 消失
- **迭代容错**：所有迭代求解器均有最大迭代数限制与收敛判据
- **随机种子固定**：使用 `np.random.default_rng(237)` 保证实验可复现

---

### 物理参数设置

| 参数 | 取值 | 物理含义 |
|------|-----|---------|
| `Ns = Nt = 4` | 格子尺寸 | 4⁴ = 256 个格点 |
| `a = 0.1 fm` | 格距 | 物理尺度 ~ 0.4 fm |
| `beta = 2.5` | 规范耦合 | g² = 1.6 |
| `m0 = 0.1` | 裸夸克质量 | 格点单位 |
| `kappa = 0.12195` | 跳跃参数 | 接近临界值 1/8 |

该参数设置处于物理上感兴趣的区域（kappa/kappa_c ≈ 0.976），使得 Wilson-Dirac 算子接近手征极限但仍有良好条件数。

---

### 合成方法学总结

本项目的核心合成思路是：

**从数值方法到物理问题的深度耦合**：不是将通用数值方法（FEM、插值、优化）简单"换皮"到物理问题，而是让每个算法在物理语境中获得新的意义：
- FEM 网格生成 → 格子几何与 plaquette 枚举
- 分段线性插值 → 规范协变插值（必须沿 Wilson 线平移）
- 布尔管道化 → 拓扑扇区计数
- 二进制码构造 → 规范固定的组合优化表述
- 多任务学习 → 耦合谱估计（利用 γ5-厄米性约束）

**跨学科融合**：本项目融合了粒子物理（格点 QCD）、数值分析（有限差分、迭代法）、组合优化（码字设计）、随机过程（Langevin 动力学、Levy 飞行）、布尔函数理论（管道化计数）等多个学科，体现了现代科学计算的交叉特征。

**可复现的小规模实验**：所有计算在 4⁴ 小规模格子上完成，单台机器秒级完成，便于教学与算法验证。
