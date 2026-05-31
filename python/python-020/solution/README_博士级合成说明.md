# 分数量子霍尔效应 Laughlin 态的多体数值模拟平台

## 项目概述

本项目围绕**凝聚态物理：量子霍尔效应分数量子态**展开，将15个种子科研代码项目的核心算法融合重构为一个面向前沿科学问题的博士级Python计算平台。项目实现了从单粒子Landau能级到多体Laughlin波函数、自洽Hartree-Fock场、边缘态BVP求解、密度演化动力学、关联函数分析、高维数值积分以及拓扑不变量（Berry曲率与Chern数）计算的完整数值模拟链路。

---

## 一、原项目到科学问题的映射

| 原项目 | 核心算法 | 在合成项目中的角色 |
|--------|---------|------------------|
| 1221_test_nonlin | 非线性方程组Newton迭代（Chandrasekhar H函数） | **hartree_fock_solver.py**：自洽Hartree-Fock迭代中的Newton修正与Jacobian处理 |
| 399_fem1d_spectral_numeric | 一维谱有限元方法（多项式基、L2/H1误差） | **landau_levels.py**：Landau轨道波函数的谱展开基函数与刚度矩阵构建 |
| 561_hypercube_surface_distance | 超立方体表面距离统计 | **correlation_functions.py**：高维空间中的电子构型距离统计与几何分析 |
| 552_hyperball_distance | 超球体内距离统计（蒙特卡洛） | **laughlin_wavefunction.py**： Laughlin态的配对关联函数g^(2)(r)与结构因子S(q) |
| 130_bvp_shooting | 边值问题打靶法（Secant迭代+ODE积分） | **edge_state_solver.py**：边缘态径向方程的BVP求解 |
| 388_fem1d_display | 一维有限元Lagrange基函数 | **landau_levels.py**：局部Lagrange插值基函数与有限元函数求值 |
| 101_blowup_ode | ODE爆炸解的稳定性处理 | **edge_state_solver.py**：边缘态密度在临界点附近的非线性增长稳定化 |
| 1219_test_nearest | 最近邻搜索算法 | **correlation_functions.py**：电子构型的最近邻分析与Wigner晶体序识别 |
| 343_euler | 欧拉法求解ODE | **edge_state_solver.py**：BVP转化为IVP后的ODE时间推进 |
| 005_analemma | 轨道角度参数化与周期性演化 | **topological_invariants.py**：Berry相位绝热参数演化与磁通量子化 |
| 366_fd1d_wave | 一维波动方程有限差分法（CTCS格式） | **density_evolution.py**：密度扰动传播的有限差分模拟 |
| 1406_wedge_exactness | 楔形体多项式精确积分与组合枚举 | **quadrature_integrals.py**：高维数值积分的精确性验证与组合枚举 |
| 498_hammersley | Hammersley低差异序列（准蒙特卡洛） | **monte_carlo_sampler.py**：Laughlin态的准随机采样与变分Monte Carlo |
| 151_cg_ne | 共轭梯度法求解正规方程 | **hartree_fock_solver.py**：Hartree-Fock矩阵方程的CGNE求解器 |
| 433_fisher_exact | KPP-Fisher方程精确解（反应扩散） | **density_evolution.py**：密度演化的Fisher-KPP反应扩散模型 |

---

## 二、新增数学物理模型与核心公式

### 2.1 Landau能级与单粒子哈密顿量

二维电子气在垂直磁场 $B$ 中的哈密顿量（对称规范 $\mathbf{A} = \frac{B}{2}(-y, x, 0)$）：

$$
H = \frac{1}{2m^*} \left[ \left(p_x + \frac{eBy}{2}\right)^2 + \left(p_y - \frac{eBx}{2}\right)^2 \right]
$$

Landau能级能量：

$$
E_n = \hbar \omega_c \left(n + \frac{1}{2}\right), \quad \omega_c = \frac{eB}{m^*}
$$

单粒子轨道波函数（复坐标 $z = x + iy$）：

$$
\psi_{n,m}(z) = N_{n,m} \cdot \left(\frac{z}{\sqrt{2}l_B}\right)^m \cdot L_n^m\left(\frac{|z|^2}{2l_B^2}\right) \cdot \exp\left(-\frac{|z|^2}{4l_B^2}\right)
$$

其中磁长度 $l_B = \sqrt{\hbar / (eB)}$，$L_n^m$ 为连带Laguerre多项式，归一化常数：

$$
N_{n,m} = \frac{(-1)^n}{\sqrt{2\pi \cdot 2^m \cdot l_B^2}} \sqrt{\frac{n!}{(n+m)!}}
$$

### 2.2 Laughlin多体波函数

填充因子 $\nu = 1/m$（$m$ 为奇整数）的Laughlin波函数：

$$
\Psi_m(z_1, \ldots, z_N) = \prod_{i<j} (z_i - z_j)^m \cdot \exp\left(-\sum_{k=1}^N \frac{|z_k|^2}{4l_B^2}\right)
$$

**关键性质**：
- 当 $z_i \to z_j$ 时，$\Psi_m \sim (z_i - z_j)^m \to 0$，体现强关联排斥
- 交换对称性：$\Psi_m(\ldots, z_i, \ldots, z_j, \ldots) = (-1)^m \Psi_m(\ldots, z_j, \ldots, z_i, \ldots)$，$m$ 为奇数保证费米子反对称性

准空穴激发（在位置 $z_0$）：

$$
\Psi_m^{qh}(z_0) = \prod_{j=1}^N (z_j - z_0) \cdot \Psi_m(z_1, \ldots, z_N)
$$

准空穴携带**分数电荷** $e^* = e/m$。

### 2.3 配对关联函数与结构因子

配对关联函数：

$$
g^{(2)}(r) = \frac{V^2}{N(N-1)} \left\langle \sum_{i \neq j} \delta(\mathbf{r} - \mathbf{r}_i + \mathbf{r}_j) \right\rangle
$$

小距离极限行为：$g^{(2)}(r) \sim (r/l_B)^{2m}$（$r \to 0$）。

静态结构因子：

$$
S(\mathbf{q}) = \frac{1}{N} \left\langle \left| \sum_{j=1}^N e^{-i\mathbf{q} \cdot \mathbf{r}_j} \right|^2 \right\rangle
$$

### 2.4 Hartree-Fock自洽场方程

有效哈密顿量：

$$
H_{HF} = H_0 + V_H + V_F
$$

Hartree（直接）势：

$$
V_H(\mathbf{r}) = \int d^2r' \, V_C(\mathbf{r} - \mathbf{r}') \rho(\mathbf{r}')
$$

Fock（交换）势（积分算符）：

$$
V_F \phi_i(\mathbf{r}) = -\sum_{j \in occ} \int d^2r' \, V_C(\mathbf{r} - \mathbf{r}') \phi_j^*(\mathbf{r}') \phi_i(\mathbf{r}') \phi_j(\mathbf{r})
$$

自洽迭代：

$$
H_{HF}^{(k)} |\phi_i^{(k+1)}\rangle = \varepsilon_i^{(k+1)} |\phi_i^{(k+1)}\rangle, \quad \rho^{(k+1)} = \sum_{i \in occ} |\phi_i^{(k+1)}\rangle\langle\phi_i^{(k+1)}|
$$

### 2.5 边缘态与手性Luttinger液体

有限几何中Landau能级在边界处弯曲。手性Luttinger液体的低能有效哈密顿量：

$$
H_{edge} = \hbar v_F \int dx \, \psi^\dagger(x) (-i\partial_x) \psi(x)
$$

色散关系：

$$
\varepsilon(k) = \frac{\hbar v_F k}{g}
$$

其中 $g$ 为Luttinger参数（$g=1$ 对应自由费米子，$g<1$ 对应相互作用系统）。

边缘态径向方程：

$$
u''(r) + \frac{1}{r}\nu'(r) + \left[k^2 - \left(\frac{m^*\omega_c r}{2\hbar}\right)^2 - \frac{m^2}{r^2}\right]\nu(r) = 0
$$

采用**打靶法**（Shooting Method）求解：猜测初始斜率 $\alpha = u'(0)$，用ODE积分器推进到 $r=R$，用割线法修正使 $u(R) \to 0$。

### 2.6 Fisher-KPP反应扩散方程

密度演化的平均场近似：

$$
\frac{\partial n}{\partial t} = D \nabla^2 n + r n \left(1 - \frac{n}{K}\right)
$$

行波精确解：

$$
u(t,x) = \frac{1}{[1 + a \exp(k(x - ct))]^2}
$$

其中波速 $c = 5/\sqrt{6}$，$k = -1/\sqrt{2}$。

### 2.7 Lindblad主方程

开放量子系统的密度矩阵演化：

$$
\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \sum_k \left[ L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right]
$$

### 2.8 Berry相位与Chern数

Berry联络：

$$
A_\mu(\lambda) = i \langle u_n(\lambda) | \partial_{\lambda_\mu} | u_n(\lambda) \rangle
$$

Berry曲率：

$$
\Omega(k) = \partial_{k_x} A_y(k) - \partial_{k_y} A_x(k)
$$

Chern数：

$$
C = \frac{1}{2\pi} \int_{T^2} d^2k \, \Omega(k)
$$

TKNN霍尔电导：

$$
\sigma_{xy} = \frac{e^2}{h} \cdot C
$$

### 2.9 Hammersley低差异序列

$d$维Hammersley点：

$$
x_{i,0} = \frac{i}{N}, \quad x_{i,j} = \Phi_{p_j}(i) \quad (j = 1, \ldots, d-1)
$$

其中 $\Phi_p(n)$ 为基$p$下的radical inverse：

$$
\Phi_p(n) = \sum_{k=0}^{\infty} a_k(n) p^{-k-1}, \quad n = \sum_{k=0}^{\infty} a_k(n) p^k
$$

---

## 三、文件结构与修改说明

项目包含 **11个Python文件** 和 **1个中文说明文档**：

| 文件 | 功能 | 融合的原项目 |
|------|------|------------|
| `main.py` | 统一入口，零参数运行，串联9大模块 | 全部 |
| `utils.py` | 物理常数、数值稳定性工具、Fermi-Dirac分布、Gram-Schmidt正交化 | — |
| `landau_levels.py` | Landau能级波函数、谱FEM刚度矩阵、Lagrange基函数 | 399, 388 |
| `laughlin_wavefunction.py` | Laughlin波函数、准空穴/准电子、配对关联函数、结构因子 | 552, 561 |
| `monte_carlo_sampler.py` | Hammersley序列、圆盘映射、变分Monte Carlo | 498 |
| `hartree_fock_solver.py` | CGNE共轭梯度、Newton迭代、自洽HF求解 | 1221, 151 |
| `edge_state_solver.py` | 欧拉法ODE积分、打靶法BVP、手性Luttinger色散 | 130, 343, 101 |
| `density_evolution.py` | 波动方程FD、Fisher-KPP、Lindblad密度矩阵演化 | 366, 433, 101 |
| `correlation_functions.py` | 最近邻搜索、超球/超立方体距离统计、两点关联、密度关联 | 1219, 552, 561 |
| `quadrature_integrals.py` | 组合枚举、楔形体精确积分、多维Gauss-Legendre、库仑积分 | 1406 |
| `topological_invariants.py` | Berry曲率、Chern数、TKNN电导、磁通量子化、绝热参数演化 | 005 |
| `README_博士级合成说明.md` | 中文说明文档 | — |

---

## 四、合成后的项目能够解决的科学问题

1. **Landau能级量子化与态密度计算**：在任意磁场强度和有效质量下，计算二维电子气的Landau能级、单粒子波函数、简并度和态密度，并分析无序展宽效应。

2. **Laughlin分数量子态的多体波函数**：构建并评估Laughlin波函数（$m=3,5,7,\ldots$），计算准空穴和准电子激发，验证分数电荷 $e^* = e/m$。

3. **强关联电子的配对关联与结构因子**：计算两点关联函数 $g^{(2)}(r)$ 和静态结构因子 $S(q)$，分析电子间的强关联排斥特征。

4. **准蒙特卡洛采样与变分能量**：利用Hammersley低差异序列进行高效采样，估计多体期望值，降低统计误差。

5. **自洽Hartree-Fock平均场**：在离散格点上构建并求解HF方程，计算单粒子能级、电子密度分布和有效势。

6. **量子霍尔边缘态的BVP求解**：使用打靶法求解边缘态径向方程，分析手性Luttinger液体的色散关系。

7. **密度演化动力学**：求解波动方程、Fisher-KPP反应扩散方程和Lindblad主方程，模拟密度扰动和开放量子系统的时间演化。

8. **拓扑不变量与量子化电导**：计算Berry曲率和Chern数，推导TKNN量子化电导，分析磁通量子化引起的Berry相位和分数电荷。

---

## 五、运行方式

```bash
cd Synthesis-project-python/020_synth_project
python3 main.py
```

程序无需任何输入参数，自动执行所有模块并输出结果摘要。

---

## 六、边界处理与数值鲁棒性

- **安全指数/对数函数**：`safe_exp` 和 `safe_log` 防止数值溢出和下溢
- **Gram-Schmidt正交化**：处理线性相关基函数的退化情况
- **矩阵条件数检查**：在Newton迭代和HF矩阵对角化前检测奇异性
- **ODE稳定性处理**：波动方程自动满足CFL条件，Fisher-KPP自动调整时间步
- **密度矩阵半正定性**：Lindblad演化后通过特征值截断保证物理合理性
- **Laughlin波函数对数计算**：大$N$时直接计算对数波函数防止溢出

---

## 七、科学公式注入说明

本项目在代码和文档中系统注入了以下高难科学公式：
- Landau哈密顿量与能级公式
- 连带Laguerre多项式波函数
- Laughlin波函数及其Jastrow因子
- 准空穴分数电荷公式
- 配对关联函数与结构因子
- Hartree-Fock自洽场方程
- 库仑相互作用（含短程截断）
- 手性Luttinger液体色散
- 边缘态径向方程
- Fisher-KPP反应扩散方程及行波精确解
- Lindblad主方程
- Berry联络、Berry曲率、Chern数、TKNN公式
- Hammersley序列的radical inverse
- 楔形体上的多项式精确积分公式
