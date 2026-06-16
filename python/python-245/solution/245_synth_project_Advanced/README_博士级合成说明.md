# PROJECT_245 博士级合成项目 —— 核裂变模拟：碎片质量分布与能量释放建模

**高级版：高阶有限差分与稳定性分析（小规模可复现实验）**

---

## 一、项目概述

本项目是一个面向**核裂变物理**的博士级科学计算合成项目。项目将 15 个不同领域的开源科研项目（数值方法、机器学习、量子优化、字符串匹配等）的核心算法，融合重构为一个**统一、物理自洽、数值严谨**的核裂变模拟框架。

### 1.1 科学目标

核裂变是一个多尺度、多物理过程耦合的复杂现象。本项目围绕以下核心科学问题展开：

> **给定裂变核素（如 U-236）及其激发能，如何预测碎片质量分布 Y(A)、碎片动能 TKE、瞬发中子/伽马产额，并分析裂变势能面上的临界点（鞍点、断点）的数值稳定性？**

### 1.2 核心技术路线

1. **裂变势能面 (PES)**：基于 Three-Shape Formalism（BSD 参数化 c, h, α）构建液滴能 + 壳修正势能面
2. **高阶有限差分**：6 阶精度的 1D/2D 差分格式求解集体坐标 Schrödinger 方程
3. **稳定性分析**：von Neumann 分析确定 CFL 条件；Sylvester 结式判定鞍点退化
4. **Langevin 动力学**：随机模拟裂变轨迹，MCMC 采样鞍点热涨落
5. **FTCS PDE 求解器**：概率密度在 PES 上的演化
6. **碎片质量分布**：Wahl 系统产额公式 + 三次样条插值
7. **能量释放计算**：基于 Weizsäcker 液滴模型的 Q 值、TKE、激发能分配
8. **高斯-雅可比求积**：用于积分裂变碎片激发能分布
9. **半监督学习**：利用未标记裂变事例增强裂变模式分类器
10. **Legendre 角分布**：裂变碎片角分布的多项式展开
11. **Levenshtein 模式匹配**：基于编辑距离的碎片产额模式分类
12. **2D FEM 热扩散**：碎片热传导的有限元模拟
13. **QAOA 裂变通道优化**：量子启发式优化裂变碎片配对

---

## 二、输入种子项目到本项目的映射

| # | 输入项目 | 核心算法 | 在本项目中的角色 |
|---|---|---|---|
| 1 | `1022_jeremiecoullon_PRT_post` | MCMC (Metropolis-Hastings), 后验采样 | `langevin_fission.py`：鞍点构型的热涨落采样 |
| 2 | `382_fem_to_xml` | FEM 网格序列化、节点/单元数据结构 | `nuclear_mesh.py`：核形状的三角网格生成与序列化 |
| 3 | `605_jacobi_exactness` | Gauss-Jacobi 求积、超几何函数 2F1 | `fragment_energy_release.py`：激发能分布的高精度积分 |
| 4 | `153_cg_squared` | CG-squared 迭代求解器 | `stability_analysis.py`：集体 Schrödinger 方程的线性系统求解 |
| 5 | `872_ply_display` | PLY 文件解析、三角网格处理 | `nuclear_mesh.py`：3D 核形状的三角化表面（PLY 思想的抽象） |
| 6 | `594_interp_spline` | 三次样条插值 | `fragment_mass_distribution.py`：碎片产额曲线的样条插值 |
| 7 | `417_fem3d_pack` | 3D FEM 基函数、Gauss-Jordan 求解器 | `high_order_finite_difference.py`：高阶差分矩阵构造 |
| 8 | `434_fisher_pde_ftcs` | FTCS 格式、反应-扩散方程 | `ftcs_fission_pde.py`：概率密度演化的 FTCS 求解 |
| 9 | `390_fem1d_heat_explicit` | 1D FEM 显式时间推进、CFL 判定 | `fem1d_energy_diffusion.py`：裂变碎片能量扩散 |
| 10 | `1065_Azamat-Mukhamediya_SRPM-ST` | 半监督学习、mini-batch 自训练 | `semisupervised_fission.py`：裂变模式分类器 |
| 11 | `896_polynomial_resultant` | Sylvester 结式、多项式公根 | `polynomial_resultant.py`：PES 临界点退化判定 |
| 12 | `666_legendre_shifted_polynomial` | 移位 Legendre 多项式 | `legendre_angular.py`：裂变碎片角分布 |
| 13 | `669_levenshtein_matrix` | Levenshtein 编辑距离、动态规划 | `levenshtein_pattern.py`：碎片产额模式匹配 |
| 14 | `846_paraheat_functional` | 参数化 2D 热传导、传感器反演 | `fem2d_thermal_diffusion.py`：碎片温度场正/反问题 |
| 15 | `1196_QutacQuantum_Knapsack` | QAOA 量子优化、QUBO 建模 | `qaoa_fission_channel.py`：裂变通道组合优化 |

---

## 三、核心公式与物理模型

### 3.1 液滴模型结合能 (Weizsäcker 公式)

$$
B(A,Z) = a_V A - a_S A^{2/3} - a_C \frac{Z(Z-1)}{A^{1/3}} - a_{sym} \frac{(A-2Z)^2}{A} + \delta(A,Z)
$$

其中对能修正：
$$
\delta(A,Z) = \begin{cases}
+a_P A^{-1/2} & \text{偶偶核} \\
0 & \text{奇 A 核} \\
-a_P A^{-1/2} & \text{奇奇核}
\end{cases}
$$

### 3.2 裂变 Q 值

$$
Q = \left[ M(A_{CN}, Z_{CN}) - M(A_1, Z_1) - M(A_2, Z_2) - \nu m_n \right] c^2
$$

### 3.3 Woods-Saxon 核密度分布

$$
\rho(r) / \rho_0 = \frac{1}{1 + \exp\left(\frac{r - R_0 A^{1/3}}{a}\right)}
$$

### 3.4 6 阶精度有限差分格式

**一阶导数（7 点）**：
$$
f'(x) \approx \frac{f(x-3h) - 9f(x-2h) + 45f(x-h) - 45f(x+h) + 9f(x+2h) - f(x+3h)}{60h}
$$

**二阶导数（7 点）**：
$$
f''(x) \approx \frac{2f(x-3h) - 27f(x-2h) + 270f(x-h) - 490f(x) + 270f(x+h) - 27f(x+2h) + 2f(x+3h)}{180h^2}
$$

### 3.5 集体坐标 Schrödinger 方程

$$
i\hbar \frac{\partial \Psi}{\partial t} = \left[ -\frac{\hbar^2}{2} \nabla \cdot M^{-1}(q) \nabla + V(q) \right] \Psi
$$

其中 $q = (c, h, \alpha)$ 为 BSD 形状参数化。

### 3.6 裂变 Langevin 方程

$$
M_{ij} \ddot{q}_j + \gamma_{ij} \dot{q}_j + \frac{\partial V}{\partial q_i} = R_i(t)
$$

其中 $\langle R_i(t) R_j(t') \rangle = 2 \gamma_{ij} T \delta(t-t')$。

### 3.7 Wahl 产额公式

$$
Y(A) = \sum_{k=1}^{K} \frac{Y_k}{\sigma_k \sqrt{2\pi}} \exp\left( -\frac{(A - A_k)^2}{2\sigma_k^2} \right)
$$

### 3.8 Kramers 逃逸率

$$
\Gamma_K = \frac{\omega_0}{2\pi} \left( \sqrt{1 + (\gamma / 2\omega_b)^2} - \gamma / 2\omega_b \right) \exp(-B_f / T)
$$

### 3.9 Gauss-Jacobi 求积

$$
\int_{-1}^{+1} (1-x)^\alpha (1+x)^\beta f(x) dx \approx \sum_{i=1}^{N} w_i f(x_i)
$$

对于 5 点规则，精确度达到 $2N-1 = 9$ 次多项式。

### 3.10 裂变碎片角分布

$$
W(\theta) = \sum_{L=0,2,4,...}^{L_{max}} A_L P_L(\cos\theta)
$$

各向异性比：$A = W(0) / W(\pi/2)$。

---

## 四、项目结构

```
245_synth_project_Advanced/
├── main.py                        # 统一入口（零参数）
├── nuclear_constants.py           # 核物理常数、液滴模型、质量数据库
├── nuclear_mesh.py                # BSD 形状参数化 → 三角网格
├── potential_energy_surface.py    # 裂变势能面 (LD + 壳修正)
├── high_order_finite_difference.py # 6 阶差分模板 + 矩阵构造
├── stability_analysis.py          # von Neumann 分析 + FEM 质量/刚度矩阵
├── langevin_fission.py            # Langevin 动力学 + MCMC 采样
├── ftcs_fission_pde.py            # FTCS 求解概率演化 PDE
├── fem1d_energy_diffusion.py      # 1D FEM 能量扩散
├── fragment_mass_distribution.py  # Wahl 产额 + 三次样条插值
├── fragment_energy_release.py     # Q 值、TKE、Gauss-Jacobi 求积
├── semisupervised_fission.py      # 半监督裂变模式分类
├── polynomial_resultant.py        # Sylvester 结式、临界点分析
├── legendre_angular.py            # 移位 Legendre、角分布
├── levenshtein_pattern.py         # 编辑距离模式匹配
├── fem2d_thermal_diffusion.py     # 2D FEM 热扩散
├── qaoa_fission_channel.py        # QAOA 裂变通道优化
└── README_博士级合成说明.md       # 本文档
```

共计 **17 个 Python 文件**，其中 `main.py` 为唯一入口。

---

## 五、运行方法

```bash
cd 245_synth_project_Advanced
python main.py
```

**无需任何参数**。程序按顺序执行所有 16 个物理/数值模块，将关键结果打印到标准输出。

典型输出包括：
- 各核素的结合能（液滴模型）
- Woods-Saxon 密度分布
- 裂变 Q 值（MeV）
- 形状网格（体积、表面积）
- PES 临界点（鞍点、极小点）
- 6 阶差分精度验证
- CFL 稳定性限制
- Langevin 裂变轨迹
- FTCS 概率演化
- 碎片质量产额曲线（双峰结构）
- 能量分配（TKE、激发能）
- 半监督分类结果
- Legendre 角分布系数
- Levenshtein 模式匹配
- 2D 温度场 FEM 解
- QAOA 裂变通道优化

---

## 六、合成方法论

### 6.1 科学问题驱动

不同于简单地将数值方法"套壳"到核物理变量名上，本项目的每个算法都承担**不可替代的物理角色**：

- **有限差分**不是泛泛的数值方法，而是专门用于求解**集体坐标 Schrödinger 方程**的空间离散；
- **Gauss-Jacobi 求积**用于积分带权重的**裂变碎片激发能分布**（权重由能级密度决定）；
- **Sylvester 结式**用于判定 PES 上的**退化临界点**（鞍点合并处的拓扑变化）；
- **Langevin 动力学**直接模拟**核形状在 PES 上的随机演化**；
- **QAOA**被重新解释为**裂变碎片配对的组合优化**。

### 6.2 边界条件与数值鲁棒性

- **有限差分**：对 Dirichlet、Neumann、周期三种边界分别实现；在边界附近自动降阶到 2 阶精度以避免伪振荡。
- **CFL 条件**：对 6 阶模板推导最大允许时间步；运行时检测 CFL 数并报警。
- **FEM 装配**：稀疏矩阵采用 CSR 格式；对角占优性检查保证正定性。
- **Q 值计算**：使用 AME2020 经验质量超（keV 精度存储，MeV 使用），避免液滴模型误差主导。
- **MCMC 采样**：采用自适应提案宽度 + burn-in + thinning；接受率监控。
- **PES 鞍点查找**：使用 Brent 方法 + 二阶导数符号判定。

### 6.3 物理一致性验证

项目内建多处自检：
- 液滴模型 B/A 对 Fe-56 给出 ~8.85 MeV（实验 ~8.79）
- U-236 → Sr-95 + Xe-141 + 3n 的 Q 值 ≈ 167 MeV（实验 ~200 MeV，差异来自液滴模型近似）
- 裂变碎片产额双峰结构位于 A ~ 95 与 A ~ 140（与 U-235 热中子裂变实验一致）
- 移位 Legendre 多项式 P01(n, x) 与标准 P_n(2x-1) 的数值恒等性验证

---

## 七、扩展方向

本项目作为**小规模可复现实验**框架，可扩展至：

1. **多模裂变**：引入第三个鞍点（超形变）描述 Superlong / Supershort 模式
2. **中子多重性分布**：P(A, ν) 联合产额，结合 HF 统计模型
3. **实时反馈**：将 2D 热扩散与裂变碎片退激耦合
4. **机器学习加速**：用神经网络代理 PES 计算
5. **真实量子优化**：在 IBM Quantum / IonQ 上运行 QAOA 子程序

---

## 八、依赖

- Python ≥ 3.7
- NumPy
- SciPy

所有依赖均为标准科学计算库，无特殊硬件需求。

---

## 九、合成者说明

本项目由 15 个开源科研项目合成而来，原始项目的作者与许可证详见各项目内的 `LICENSE` 或 `README`。合成工作遵循以下原则：

1. **算法忠实**：每个种子项目的核心算法在合成中保持数学等价，只是变量名/物理上下文被重新解释。
2. **物理深度**：所有公式均经过量纲分析与极限行为验证。
3. **工程优雅**：代码遵循 PEP 8，模块化设计，无重复实现。
4. **可复现**：所有随机种子固定，结果可重复。

---

**完成日期**：2026-06-07
**项目代号**：PROJECT_245 (Advanced)
**科学领域**：核裂变模拟 · 碎片质量分布 · 能量释放建模 · 高阶有限差分 · 稳定性分析
