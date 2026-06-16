# Project 260: 暗能量状态方程约束 -- 高阶有限差分与稳定性分析

## 项目概述

本项目融合 **15 个种子项目**的核心算法，围绕 **计算宇宙学中的暗能量状态方程约束** 这一前沿博士级科学问题，构建了一套完整的高阶有限差分求解与稳定性分析框架。

**科学目标**: 通过 CPL 参数化暗能量模型 $w(a) = w_0 + w_a(1-a)$，利用高阶有限差分方法和严格的 von Neumann 稳定性分析，约束暗能量状态方程参数 $(w_0, w_a)$，并评估数值方法的收敛性、稳定性和计算效率。

---

## 核心物理与数学模型

### 1. 暗能量状态方程 (CPL 参数化)

$$w(a) = w_0 + w_a (1 - a)$$

暗能量密度演化:

$$\frac{\rho_{DE}(a)}{\rho_{DE}(1)} = a^{-3(1+w_0+w_a)} \exp(-3w_a(1-a))$$

### 2. FLRW 背景宇宙学

Friedmann 方程:

$$E^2(a) = \frac{H^2(a)}{H_0^2} = \Omega_m a^{-3} + \Omega_r a^{-4} + \Omega_{DE} f(a) + \Omega_k a^{-2}$$

减速度参数:

$$q(a) = \frac{1}{2} \sum_i \Omega_i(a)(1+3w_i(a))$$

### 3. 线性增长方程

$$D''(u) + P(u)D'(u) + Q(u)D(u) = 0$$

其中 $u = \ln a$, $P(u) = 2 + H'/H$, $Q(u) = -\frac{3}{2}\Omega_m(a)$.

### 4. 高阶有限差分

**4阶中心差分**:

$$f'(x) \approx \frac{f_{i-2} - 8f_{i-1} + 8f_{i+1} - f_{i+2}}{12h}$$

**紧致 (Padé) 4阶格式**:

$$\frac{1}{6}f'_{i-1} + \frac{2}{3}f'_i + \frac{1}{6}f'_{i+1} = \frac{f_{i+1} - f_{i-1}}{2h}$$

### 5. von Neumann 稳定性分析

模型方程 $u_t = \lambda u$，放大因子 $G(z) = G(\lambda \Delta t)$:

- **显式 Euler**: $G(z) = 1 + z$
- **RK4**: $G(z) = 1 + z + \frac{z^2}{2} + \frac{z^3}{6} + \frac{z^4}{24}$
- **BDF2**: $G(z) = \frac{1}{3 - 4z + z^2}$

稳定条件: $|G(z)| \leq 1$.

### 6. BDF 多步法

**BDF2**:

$$\frac{3}{2}y_{n+1} - 2y_n + \frac{1}{2}y_{n-1} = \Delta t \, f(t_{n+1}, y_{n+1})$$

### 7. Fisher 矩阵预测

$$F_{ab} = \sum_i \frac{1}{\sigma_i^2} \frac{\partial M_i}{\partial p_a} \frac{\partial M_i}{\partial p_b}$$

DETF Figure of Merit:

$$\text{FoM} = \frac{1}{\sqrt{\det(\text{Cov}(w_0, w_a))}}$$

### 8. Sobol 全局灵敏度

$$S_i = \frac{V[\mathbb{E}(Y|X_i)]}{V(Y)}, \quad S_{Ti} = \frac{\mathbb{E}[V(Y|X_{\sim i})]}{V(Y)}$$

---

## 15 个种子项目融合映射

| # | 种子项目 | 核心算法 | 在本项目中的角色 | 对应模块 |
|---|---------|---------|----------------|---------|
| 1 | 1312_triangle_monte_carlo | 三角形重心坐标采样 + MC 积分 | 天区立体角积分、巡天边界几何 | monte_carlo_geom.py |
| 2 | 238_cvt | CVT Lloyd 算法 | 巡天望远镜场分布优化 | cvt_mesh.py |
| 3 | 1188_Gamma_Oscillations | Welch PSD + Granger 因果 + PAC | BAO 信号提取与功率谱分析 | fisher_forecast.py (dsc_filter) |
| 4 | 1081_toy-model-cis-code | K-sweep + 特征重要性 | (w0,wa) 参数网格扫描 | fisher_forecast.py (ParameterSweep) |
| 5 | 1374_unstable_ode | 不稳定 ODE 测试 | 增长方程刚性验证 | stability_analysis.py |
| 6 | 295_disk_monte_carlo | 圆盘精确积分 (Gamma 函数) | 天球区域积分 | monte_carlo_geom.py |
| 7 | 378_fem_to_gmsh | 网格节点映射 + 带宽分析 | 有限差分网格带宽计算 | fd_operators.py |
| 8 | 619_kepler_perturbed_ode | 摄动 Hamilton 系统 | 守恒量验证 (H, L) | stability_analysis.py |
| 9 | 1001_Eddien826_SA | CFRAM 分解 + 区域平均 | 背景演化多红移诊断 | cosmo_constants.py |
| 10 | 971_r8bto | 块 Toeplitz 矩阵 + Levinson 求解 | 空间离散矩阵高效存储 | fd_operators.py (BlockToeplitzMatrix) |
| 11 | 179_circle_integrals | 圆周 Gamma 函数精确积分 | 球面几何积分 | monte_carlo_geom.py |
| 12 | 1396_voronoi_mountains | Voronoi 距离场 | 巡天覆盖分析 | cvt_mesh.py |
| 13 | 1236_OneFlipBackdoor | 位翻转扰动检测 | MCMC 链异常跳变检测 | fisher_forecast.py |
| 14 | 064_backward_euler_fixed | 隐式 Euler + Picard 迭代 | BDF1 增长因子求解 | growth_solver.py |
| 15 | 395_fem1d_pack | Lagrange 基函数 + Gauss-Legendre | FD 算子构造、数值积分 | fd_operators.py |

---

## 文件结构

```
260_synth_project_Advanced/
├── main.py                    # 统一入口 (零参数运行)
├── cosmo_constants.py         # 宇宙学常数、CPL EoS、背景量
├── fd_operators.py            # 高阶 FD 算子、紧致格式、块 Toeplitz
├── stability_analysis.py      # von Neumann 稳定性、CFL、摄动 Kepler
├── growth_solver.py           # BDF/IMEX 增长因子求解器
├── monte_carlo_geom.py        # 三角形/圆盘/圆周 MC 积分
├── cvt_mesh.py                # CVT 网格生成、Voronoi 距离场
├── adaptive_mesh.py           # 自适应网格、Richardson 外推
├── fisher_forecast.py         # Fisher 矩阵、Gibbs 采样、K-sweep
├── sensitivity_analysis.py    # Sobol 灵敏度、OAT、弹性
└── README_博士级合成说明.md     # 本文档
```

---

## 运行方式

```bash
cd 260_synth_project_Advanced
python main.py
```

**无需任何参数**，程序自动完成从背景宇宙学计算到参数约束的完整流程。

---

## 科学输出

### 1. 背景诊断
- Hubble 参数 $H(z)$、共动距离 $\chi(z)$、光度距离 $d_L(z)$
- 宇宙年龄 $t(z)$、减速度参数 $q(a)$
- Phantom 穿越检测

### 2. 数值方法精度
- 各阶 FD 算子的截断误差
- 紧致格式修正波数与色散关系
- Gauss-Legendre 积分精度

### 3. 稳定性分析
- 各时间推进格式的放大因子
- CFL 条件数值
- 增长方程特征值与刚性比
- 不稳定 ODE 与摄动 Kepler 守恒验证

### 4. 增长因子求解
- BDF1、BDF2、IMEX 三种方法
- FD 配置法交叉验证
- 物质主导极限检验
- CPL 模型结果

### 5. 巡天几何与体积
- CVT 优化的望远镜场分布
- Voronoi 距离场
- 有效巡天体积 $V_{\text{eff}}$

### 6. 参数约束
- Fisher 矩阵预测 $(\sigma_{w_0}, \sigma_{w_a}, \text{FoM})$
- Gibbs 采样后验分布
- 参数扫描特征重要性
- Sobol 全局灵敏度指数

---

## 博士级难度体现

1. **高阶数值方法**: 2/4/6/8 阶有限差分 + 紧致 Padé 格式 + Fornberg 算法
2. **严格稳定性理论**: von Neumann 放大因子 + 修正波数色散 + CFL 推导
3. **多方法交叉验证**: BDF1/BDF2/IMEX/FD 配置法四种独立求解
4. **自适应网格**: Richardson 外推 + 等分布原理 + 梯度加密
5. **全局灵敏度**: Sobol (Saltelli 方案) + OAT + 弹性分析
6. **前沿宇宙学**: CPL EoS + Fisher forecasting + Gibbs sampling + BAO 信号

---

## 参考文献

1. Planck 2018 results. VI. Cosmological parameters. A&A, 641, A6
2. Chevallier & Polarski (2001). Int.J.Mod.Phys.D, 10, 213
3. Linder (2003). PRL, 90, 091301 (CPL 参数化)
4. Lele (1992). JCP, 103, 16 (紧致差分格式)
5. Fornberg (1988). Math.Comp., 51, 699 (任意点 FD 系数)
6. Wang et al. (2010). MNRAS, 405, 2231 (BAO 信号提取)
7. Saltelli et al. (2010). Comput.Phys.Commun., 181, 259 (Sobol 指数)
8. Albrecht et al. (2006). PSTT, 0609018 (DETF FoM)

---

## 合成说明

本项目严格围绕 **"计算宇宙学: 暗能量状态方程约束: 高阶有限差分与稳定性分析"** 展开，所有算法、变量命名、代码架构均与该领域深度耦合。与通用数值方法不同，本项目的有限差分算子直接作用于增长方程 $\ln a$ 空间，紧致格式针对宇宙学阻尼项优化，稳定性分析考虑了增长方程的刚性特征。

**独特性**: 不同于已有的谱配置法 (Chebyshev collocation) 项目，本项目采用高阶有限差分 + von Neumann 稳定性分析为核心，结合 BDF 多步隐式方法和自适应网格，形成了一套独特的计算宇宙学数值框架。
