# 中子星物态方程与致密物质多物理场数值模拟

## 项目概述

本项目将 **15 个种子科研代码项目** 的核心算法融合重构为一个面向天体物理前沿问题的博士级科学计算平台，严格围绕 **"天体物理：中子星物态方程与致密物质"** 领域展开。

项目解决的核心科学问题是：

> **中子星内部极端致密环境下的多物理场耦合问题**，包括核物质状态方程（EOS）的理论建模、广义相对论流体静力学平衡（TOV方程）的数值求解、地壳核pasta相的多区域热力学积分、晶格弹性结构分析、中微子扩散-对流-反应输运，以及全链条数值算法的稳定性验证与积分精度评估。

---

## 原项目到科学问题的映射

| 原项目 | 核心算法 | 合成后角色 |
|--------|---------|-----------|
| **1302_triangle_exactness** | 三角形区域单值积分精确度检验 | 核pasta相（gnocchi相）Wigner-Seitz原胞的自由能密度积分 |
| **957_quadrilateral_witherden_rule** | 四边形Witherden高斯积分规则 | 板状核（lasagna相）的周期性四边形单元自由能计算 |
| **1409_wedge_integrals** | 三维楔形区域单值积分 | 柱形核（spaghetti相）的三维楔形单元能量积分 |
| **984_r8lt** | 下三角矩阵行列式与求逆 | TOV方程线性化稳定性分析的Jacobian结构分析 |
| **912_prime_fermat** | 费马素性检验 | 数值模拟伪随机种子生成（Monte Carlo采样初始化） |
| **1045_rot13** | 字符串ROT13编码 | 结果摘要轻度混淆输出与状态验证 |
| **832_ode_sweep_parfor** | ODE参数网格扫描 | TOV方程质量-半径关系参数扫描 |
| **885_polygon_grid** | 多边形网格点生成 | 地壳晶格Wigner-Seitz原胞的六边形网格离散化 |
| **1230_tet_mesh** | 四面体网格排序与Gauss-Jordan消元 | 晶格弹性平衡方程有限元刚度矩阵求解 |
| **661_legendre_polynomial** | 连带Legendre多项式递推计算 | 核子-核子相互作用势的角向分波展开 |
| **1368_tumor_pde** | 反应-扩散-对流PDE系数结构 | 中微子在致密物质中的扩散-对流-弱反应输运方程 |
| **1216_test_matrix** | Hilbert矩阵等病态测试矩阵 | 物态方程参数反演中的数值稳定性评估 |
| **513_hello** | 系统初始化问候 | 物理计算系统初始化与诊断输出 |
| **344_exactness** | Hermite/Chebyshev/Laguerre积分精确度检验 | 高斯积分规则在费米-狄拉克分布积分中的精度验证 |
| **1414_whale** | 拼图瓦片计数 | 地壳晶格多边形单元数与填充因子计算 |

---

## 新增数学物理模型与核心公式

### 1. Skyrme有效相互作用与状态方程

能量密度泛函（非对称核物质）：

$$
\varepsilon(\rho_b, \delta) = \varepsilon_{\text{kin}} + \varepsilon_{\text{pot}}
$$

其中动能项：

$$
\varepsilon_{\text{kin}} = \frac{3}{5}\frac{\hbar^2}{2m}k_F^2 \cdot \frac{1}{2}\left[(1+\delta)^{5/3} + (1-\delta)^{5/3}\right]
$$

势能项（Skyrme中心力）：

$$
\varepsilon_{\text{pot}} = \frac{3}{8}t_0\rho_b^2\left[\left(1+\frac{x_0}{2}\right) - \left(x_0+\frac{1}{2}\right)\delta^2\right] + \frac{1}{16}t_3\rho_b^{\alpha+2}\left[\left(1+\frac{x_3}{2}\right) - \left(x_3+\frac{1}{2}\right)\delta^2\right]
$$

压强由热力学关系给出：

$$
P = \rho_b^2 \frac{\partial(\varepsilon/\rho_b)}{\partial\rho_b}
$$

声速平方（因果性约束 $c_s^2 \leq c^2$）：

$$
c_s^2 = \frac{\partial P}{\partial\varepsilon}
$$

### 2. 连带Legendre多项式展开

核子-核子相互作用势的角向分波展开：

$$
V(\cos\theta) = \sum_{l=0}^{L_{\max}} c_l P_l(\cos\theta)
$$

连带Legendre多项式 $P_n^m(x)$ 满足微分方程：

$$
(1-x^2)y'' - 2xy' + \left[n(n+1) - \frac{m^2}{1-x^2}\right]y = 0
$$

递推关系：

$$
\begin{aligned}
P_m^m(x) &= -(2m-1)!!(1-x^2)^{m/2} \\
P_{m+1}^m(x) &= x(2m+1)P_m^m(x) \\
(n-m)P_n^m(x) &= x(2n-1)P_{n-1}^m(x) - (n+m-1)P_{n-2}^m(x)
\end{aligned}
$$

### 3. Tolman-Oppenheimer-Volkoff (TOV) 方程

中子星静力学结构方程（几何单位 $G = c = 1$）：

$$
\frac{dP}{dr} = -\frac{(\varepsilon+P)(m+4\pi r^3 P)}{r(r-2m)}
$$

$$
\frac{dm}{dr} = 4\pi r^2 \varepsilon
$$

边界条件：
- $r = 0$: $m(0) = 0$, $P(0) = P_c$
- $r = R$: $P(R) = 0$, $M = m(R)$

### 4. 潮汐形变参数

二阶Love数近似与潮汐形变度：

$$
\Lambda = \frac{2}{3}\frac{k_2}{C^5}
$$

其中致密性参数 $C = GM/(Rc^2)$，Love数近似：

$$
k_2 \approx \frac{8}{5}(1-2C)^2 C^5 \frac{1+1.75C-2.8C^2}{1-2C+4C^2}
$$

### 5. 核pasta相多区域积分

**三角形区域（gnocchi相）**：

$$
\int_{\Delta} x^m y^n \,dx\,dy = \frac{m!\,n!}{(m+n+2)!}
$$

**四边形区域（lasagna相）**：使用Witherden-Vincent对称积分规则，$n$点规则对 $2n-1$ 次多项式精确。

**楔形区域（spaghetti相）**：

$$
\int_{\text{wedge}} x^{e_1} y^{e_2} z^{e_3} \,dx\,dy\,dz = \frac{2}{e_3+1} \cdot \frac{e_2!}{(e_1+e_2+2)!} \quad (e_3 \text{ even})
$$

### 6. 中微子扩散-对流-反应方程

电子丰度与温度耦合方程组：

$$
\frac{\partial Y_e}{\partial t} = D\nabla^2 Y_e + \mathbf{v}\cdot\nabla Y_e + S(Y_e, T)
$$

$$
\frac{\partial T}{\partial t} = \frac{K_{\text{th}}}{c_v}\nabla^2 T + Q_\nu(T)
$$

弱反应源项：

$$
S(Y_e, T) = -\lambda\left(Y_e - Y_{\text{eq}}(T)\right)
$$

中微子冷却率（URCA过程）：

$$
Q_\nu \propto -T^6
$$

### 7. 晶格弹性平衡方程

有限元刚度矩阵组装：

$$
K_{ij} = \sum_e \int_{\Omega_e} B^T D B \,d\Omega
$$

其中弹性矩阵（平面应力，$E=1$, $\nu=0.3$）：

$$
D = \frac{E}{1-\nu^2}\begin{pmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{pmatrix}
$$

### 8. 高斯积分规则精确度验证

**Gauss-Hermite**：$n$点规则对 $2n-1$ 次多项式精确

$$
\int_{-\infty}^{\infty} x^p e^{-x^2}\,dx = \begin{cases} 0 & p\text{ odd} \\ (p-1)!!\sqrt{\pi}/2^{p/2} & p\text{ even} \end{cases}
$$

**Gauss-Laguerre**：

$$
\int_0^\infty x^p e^{-x}\,dx = p!
$$

### 9. 费马素性检验

基于费马小定理：若 $p$ 为素数且 $\gcd(a,p)=1$，则

$$
a^{p-1} \equiv 1 \pmod{p}
$$

---

## 项目文件结构

```
006_synth_project/
├── main.py                  # 统一入口，零参数可运行
├── utils_physics.py         # 物理常数、费马素性检验、ROT13、工具函数
├── eos_legendre.py          # Skyrme EOS、Legendre多项式、物态方程
├── tov_solver.py            # TOV方程RK4积分器、质量-半径关系扫描
├── crust_integrals.py       # 三角形/四边形/楔形积分、核pasta自由能
├── lattice_grid.py          # 多边形网格、六边形晶格、Gauss-Jordan求解
├── neutrino_diffusion.py    # 中微子扩散-对流-反应方程有限差分解
├── matrix_stability.py      # 下三角矩阵、Hilbert矩阵、稳定性分析
└── quadrature_verify.py     # Hermite/Laguerre/Chebyshev积分精度验证
```

---

## 运行方式

```bash
cd /mnt/data/zpy/sci-swe/source code/Synthesis-project-python/006_synth_project
python main.py
```

程序将依次执行8个阶段的科学计算：

1. **物态方程与Legendre展开**：计算不同密度下的压强、能量密度、声速、化学势
2. **TOV方程与质量-半径关系**：参数扫描中心压强，输出中子星半径与质量
3. **地壳核pasta相积分**：验证三角形/四边形/楔形积分规则，估算各相自由能
4. **晶格网格与弹性分析**：生成六边形晶格，求解弹性平衡方程
5. **中微子扩散与冷却**：求解1D中微子输运方程，计算光度与弛豫时标
6. **矩阵稳定性分析**：下三角矩阵运算、Hilbert矩阵病态测试、特征值稳定性
7. **积分精度验证**：Gauss-Hermite/Laguerre节点生成与精度检验
8. **合成总结**：素数种子生成与编码输出

---

## 工程特性

- **边界处理**：所有数值运算包含参数合法性检查（密度非负、不对称度 $|\delta| \leq 1$、压强非负等）
- **数值鲁棒性**：safe_sqrt、safe_divide 等保护函数避免 nan/inf；TOV积分自适应步长控制；Schwarzschild半径检测
- **因果性约束**：声速平方上限截断为 $c_s^2 \leq c^2$
- **物理约束**：电子丰度 $Y_e \in [0,1]$，温度截断在合理范围
- **病态问题处理**：Gauss-Jordan选主元消元；Hilbert矩阵稳定性测试评估数值误差增长

---

## 合成后的项目能够解决什么科学问题

1. **中子星结构计算**：给定核物质物态方程，计算中子星的质量-半径关系、最大质量、潮汐形变参数
2. **核pasta相热力学**：定量评估地壳中不同核几何相（gnocchi/spaghetti/lasagna）的自由能与稳定性
3. **中微子冷却机制**：模拟中子星诞生后中微子扩散驱动的脱轻子化与热冷却过程
4. **数值算法验证**：通过Hilbert矩阵、积分精确度检验等手段，定量评估科学计算中的数值误差与稳定性
5. **多尺度耦合**：从核尺度（Skyrme相互作用）到中子星宏观尺度（TOV方程）的多尺度建模
