# 聚合物玻璃化转变分子动力学模拟 —— 博士级合成说明

## 一、项目概述

本项目基于 **15 个科研代码种子项目**，在 **分子动力学：聚合物玻璃化转变** 这一前沿科学领域内，合成了一套完整的粗粒化分子动力学（Coarse-Grained MD）模拟系统。项目使用 Python 语言实现，包含 10 个 `.py` 文件，统一入口为 `main.py`，零参数可直接运行。

### 核心科学问题

**聚合物熔体在温度淬火过程中的玻璃化转变（Glass Transition）行为研究。**

具体研究内容包括：
1. 粗粒化 bead-spring 聚合物链的构象演化与动力学
2. 温度-比容（v-T）关系的测定与玻璃化转变温度 $T_g$ 的确定
3. 自由体积（Free Volume）的 Voronoi  tessellation 分析
4. 热扩散系数与脆性指数（Fragility Index）的关联
5. VFT（Vogel-Fulcher-Tammann）粘弹性行为的数值验证

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法/思想 | 合成后的角色与功能 |
|--------|--------------|-------------------|
| **1208_test_int_2d** | 2D Legendre-Gauss 数值积分 | `numeric_utils.py` 中的 `integrate_2d_gauss()`，用于径向分布函数 $g(r)$ 的积分计算 |
| **312_dosage_ode** | ODE 参数管理与时间演化协议 | `thermostat.py` 中的 `TemperatureProtocol`，管理淬火温度-时间曲线 |
| **330_ellipse_grid** | 椭圆内网格点生成 | `polymer_chain.py` 中的 `generate_ellipse_cross_section()`，用于链截面单体分布建模 |
| **146_ccvt_reflect** | 反射边界 CVT 迭代 | `cvt_sampler.py` 中 Voronoi 生成器的反射边界约束处理 |
| **360_fd1d_heat_explicit** | 1D 显式热方程有限差分 | `heat_diffusion.py` 中的 `HeatDiffusion1D`，模拟薄膜沿厚度方向的温度梯度 |
| **404_fem2d_heat_rectangle** | 2D 有限元热传导（三角形网格、二次基函数、后向 Euler） | `heat_diffusion.py` 中的 `HeatDiffusion2DFEM`，求解基底-薄膜界面的 2D 热传导 |
| **259_cvt_square_nonuniform** | 非均匀密度 CVT | `cvt_sampler.py` 中的 `_free_volume_density()`，基于自由体积密度加权优化 Voronoi 区域 |
| **975_r8ccs** | 压缩列存储（CCS）稀疏矩阵 | `sparse_solver.py` 中的 `SparseCCS` 类及 `conjugate_gradient()`，用于大规模邻居矩阵运算 |
| **809_nonlin_regula** | Regula Falsi 非线性求根 | `glass_transition.py` 中的 `regula_falsi()`，精确求解比容-温度曲线的交点以确定 $T_g$ |
| **1008_random_walk_1d_simulation** | 1D 随机游走模拟 | `polymer_chain.py` 中扩展为 3D 自回避随机游走（SARW），初始化聚合物链构象 |
| **877_poisson_2d** | 2D 泊松方程 Jacobi 迭代 | `heat_diffusion.py` 中的 `solve_steady_jacobi()`，求解稳态热平衡方程 |
| **910_prime** | 素数计数与生成 | `numeric_utils.py` 中的 `generate_primes()`，用于高维随机数种子管理 |
| **252_cvt_box** | 盒子约束 CVT 投影 | `cvt_sampler.py` 中的盒子投影约束，确保生成器在模拟盒子内 |
| **1033_rk23** | RK23（Runge-Kutta 2/3 阶）ODE 求解器 | `integrator.py` 中的 `rk23_step()`，用于 Nose-Hoover 热浴扩展变量的耦合演化 |
| **242_cvt_4_movie** | CVT 迭代与密度采样 | `cvt_sampler.py` 中的 CVT 迭代循环与蒙特卡洛采样估计 Voronoi 体积 |

---

## 三、新增数学物理模型与核心公式

### 3.1 粗粒化分子动力学力场

系统势能由三部分组成：

$$
U_{\text{total}} = U_{\text{LJ}} + U_{\text{FENE}} + U_{\text{angle}}
$$

#### (1) Lennard-Jones 非键势

$$
U_{\text{LJ}}(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right] - U_{\text{shift}}, \quad r \leq r_c
$$

其中截断偏移量 $U_{\text{shift}} = U_{\text{LJ}}(r_c)$ 保证势能在截断处连续。

力的大小：

$$
F(r) = \frac{24\varepsilon}{\sigma} \left[ 2\left(\frac{\sigma}{r}\right)^{13} - \left(\frac{\sigma}{r}\right)^{7} \right]
$$

#### (2) FENE 键合势（Finite Extensible Nonlinear Elastic）

$$
U_{\text{FENE}}(r) = -\frac{1}{2} k R_0^2 \ln\left[ 1 - \left(\frac{r}{R_0}\right)^2 \right], \quad r < R_0
$$

力的大小：

$$
F_{\text{FENE}}(r) = -\frac{k r}{1 - (r/R_0)^2}
$$

#### (3) 弯曲角势

$$
U_{\text{angle}}(\theta) = k_\theta (\theta - \theta_0)^2
$$

其中 $\theta$ 为三个连续单体的键角，$\theta_0 = \pi$ 对应伸直链构象。

### 3.2 Velocity Verlet 积分器

辛积分算法，保持时间可逆性：

$$
\begin{aligned}
\mathbf{v}(t + \Delta t/2) &= \mathbf{v}(t) + \frac{\Delta t}{2m} \mathbf{F}(t) \\
\mathbf{r}(t + \Delta t) &= \mathbf{r}(t) + \Delta t \, \mathbf{v}(t + \Delta t/2) \\
\mathbf{F}(t + \Delta t) &= -\nabla U(\mathbf{r}(t + \Delta t)) \\
\mathbf{v}(t + \Delta t) &= \mathbf{v}(t + \Delta t/2) + \frac{\Delta t}{2m} \mathbf{F}(t + \Delta t)
\end{aligned}
$$

### 3.3 Nose-Hoover 恒温扩展系统

扩展哈密顿量：

$$
H_{\text{NH}} = \sum_i \frac{\mathbf{p}_i^2}{2m_i s^2} + U(\mathbf{r}) + \frac{p_s^2}{2Q} + g k_B T \ln s
$$

热浴变量演化方程（使用 RK23 求解）：

$$
\frac{d\xi}{dt} = \frac{2E_k - g k_B T}{Q}, \quad \xi = \frac{p_s}{Q}
$$

### 3.4 CVT 自由体积分析

Centroidal Voronoi Tessellation 能量泛函：

$$
E(\{\mathbf{z}_i\}) = \sum_i \int_{\Omega_i} \rho(\mathbf{x}) \|\mathbf{x} - \mathbf{z}_i\|^2 \, d\mathbf{x}
$$

最优性条件（密度加权质心）：

$$
\mathbf{z}_i = \frac{\int_{\Omega_i} \rho(\mathbf{x}) \mathbf{x} \, d\mathbf{x}}{\int_{\Omega_i} \rho(\mathbf{x}) \, d\mathbf{x}}
$$

自由体积密度权重（高斯包络叠加）：

$$
\rho(\mathbf{x}) = \exp\left( -\sum_j \exp\left(-\frac{\|\mathbf{x} - \mathbf{r}_j\|^2}{2\sigma^2}\right) \right)
$$

### 3.5 热传导方程

#### 1D 显式有限差分（融合 360_fd1d_heat_explicit）

$$
\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2} + Q(x,t)
$$

显式离散格式（CFL 稳定性条件 $\alpha \Delta t / \Delta x^2 < 0.5$）：

$$
T_i^{n+1} = T_i^n + \text{CFL} \cdot (T_{i-1}^n - 2T_i^n + T_{i+1}^n) + \Delta t \cdot Q_i
$$

#### 2D 稳态热传导（融合 877_poisson_2d）

Jacobi 迭代求解 $\nabla^2 T = -Q/\alpha$：

$$
T_{i,j}^{\text{new}} = \frac{1}{4}\left( T_{i-1,j} + T_{i+1,j} + T_{i,j-1} + T_{i,j+1} + \frac{\Delta x^2 Q_{i,j}}{\alpha} \right)
$$

### 3.6 玻璃化转变分析

#### 比容-温度关系

橡胶态（$T > T_g$）：

$$
v(T) = v_g + \alpha_{\text{rubber}} (T - T_g)
$$

玻璃态（$T \leq T_g$）：

$$
v(T) = v_g + \alpha_{\text{glass}} (T - T_g)
$$

其中 $\alpha_{\text{rubber}} > \alpha_{\text{glass}}$ 为热膨胀系数。

#### VFT 方程（Vogel-Fulcher-Tammann）

描述过冷液体的弛豫时间-温度关系：

$$
\tau(T) = A \exp\left( \frac{B}{T - T_0} \right)
$$

其中 $T_0$ 为 Vogel 温度（理想玻璃化转变温度），$B$ 为 VFT 温度参数。

#### 脆性指数（Fragility Index）

$$
m = \left. \frac{d(\log_{10} \tau)}{d(T_g/T)} \right|_{T=T_g} = \frac{B \cdot T_g}{\ln 10 \cdot (T_g - T_0)^2}
$$

#### Regula Falsi 求根法（融合 809_nonlin_regula）

用于精确确定比容-温度曲线两切线的交点：

$$
c = \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}
$$

迭代直至 $|b - a| < \text{tol}$。

### 3.7 稀疏矩阵与共轭梯度法

#### CCS（Compressed Column Storage）格式

稀疏矩阵 $A \in \mathbb{R}^{m \times n}$ 存储为三元组：
- `values`: 非零元素值（长度 $nnz$）
- `row_indices`: 非零元素的行索引（长度 $nnz$）
- `col_pointers`: 每列的起始指针（长度 $n+1$）

#### 共轭梯度法（CG）

求解对称正定线性系统 $A\mathbf{x} = \mathbf{b}$：

$$
\begin{aligned}
\alpha_k &= \frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{p}_k^T A \mathbf{p}_k} \\
\mathbf{x}_{k+1} &= \mathbf{x}_k + \alpha_k \mathbf{p}_k \\
\mathbf{r}_{k+1} &= \mathbf{r}_k - \alpha_k A \mathbf{p}_k \\
\beta_k &= \frac{\mathbf{r}_{k+1}^T \mathbf{r}_{k+1}}{\mathbf{r}_k^T \mathbf{r}_k} \\
\mathbf{p}_{k+1} &= \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k
\end{aligned}
$$

---

## 四、项目文件结构

```
119_synth_project/
├── main.py                          # 统一入口，零参数运行
├── polymer_chain.py                 # 聚合物链构建（SARW + 椭圆网格）
├── force_field.py                   # 力场定义（LJ + FENE + Angle）
├── integrator.py                    # MD 积分器（Velocity Verlet + RK23）
├── thermostat.py                    # 热浴与温度协议
├── cvt_sampler.py                   # CVT 自由体积分析
├── heat_diffusion.py                # 热传导求解（1D FD + 2D FEM + Jacobi）
├── sparse_solver.py                 # 稀疏矩阵与 CG 求解器
├── glass_transition.py              # 玻璃化转变分析（Regula Falsi + VFT）
├── numeric_utils.py                 # 数值工具（2D 积分 + 素数 + PBC）
└── README_博士级合成说明.md         # 本文档
```

---

## 五、运行方式

```bash
cd 119_synth_project
python main.py
```

程序将自动执行以下流程：
1. 初始化聚合物系统（4 条链，每条 20 个单体）
2. 高温平衡化（$T = 2.0$，100 步）
3. 温度淬火模拟（8 个温度点，每个 80 步生产运行）
4. 物理量采集：比容、回转半径、自由体积分数、扩散系数
5. 玻璃化转变分析：切线法与 Regula Falsi 确定 $T_g$，VFT 拟合
6. 热扩散分析：1D 稳态温度梯度与 2D FEM 温度场
7. 稀疏矩阵求解演示：CG 法求解 Poisson-like 系统

---

## 六、关键数值方法与边界处理

### 6.1 边界条件

- **周期性边界条件（PBC）**：所有 MD 力计算使用最小像约定（Minimum Image Convention）
- **Dirichlet 边界条件**：热传导方程的上下边界固定温度
- **反射边界**：CVT 生成器超出盒子时执行反射回盒

### 6.2 数值鲁棒性措施

1. **LJ 势能截断偏移**：保证 $U(r_c) = 0$，避免能量跳跃
2. **FENE 势饱和保护**：$r \geq R_0$ 时返回大数而非发散
3. **速度上限截断**：防止数值不稳定导致的粒子超速
4. **CFL 条件检查**：显式热扩散自动调整时间步长
5. **Jacobi 迭代收敛监控**：残差低于容差时自动终止
6. **除零保护**：`safe_divide` 函数避免所有除零错误
7. **nan/inf 检测**：力计算异常时自动回退到上一时间步的力

### 6.3 精度与验证

- Velocity Verlet 积分器：二阶精度 $O(\Delta t^2)$
- RK23 热浴耦合：局部截断误差 $O(\Delta t^3)$，自适应步长建议
- 2D Legendre-Gauss 积分：$n$ 点公式精确积分 $2n-1$ 次多项式
- CG 求解器：残差可达机器精度（$\sim 10^{-16}$）

---

## 七、科学问题的物理意义

### 7.1 聚合物玻璃化转变的本质

玻璃化转变是聚合物从橡胶态（高弹态）向玻璃态转变的二级相变（或严格来说是一种动力学转变）。在 $T_g$ 附近：

- **橡胶态**：链段运动自由，体系具有高的热膨胀系数和高的扩散系数
- **玻璃态**：链段运动冻结，热膨胀系数显著降低，扩散系数趋近于零

### 7.2 自由体积理论

Fox-Flory 自由体积理论认为，$T_g$ 对应于自由体积分数降至临界值 $f_{v,c} \approx 0.025$ 的温度。本项目的 CVT 分析提供了自由体积空间分布的数值表征手段。

### 7.3 VFT 行为与脆性

强液体（Strong Liquids）遵循 Arrhenius 行为（$B \to \infty$），脆性指数 $m \approx 16$；
弱液体（Fragile Liquids）显著偏离 Arrhenius，$m$ 可达 100-200。

本项目通过数值拟合 VFT 参数，可定量评估聚合物液体的脆性特征。

---

## 八、总结

本项目成功将 15 个独立的科研代码项目融合为一个面向**聚合物玻璃化转变**的博士级分子动力学模拟系统。所有原始项目的核心算法都在合成系统中承担了真实角色：

- 随机游走与椭圆网格用于链构象初始化
- CVT 系列算法（反射边界、盒子投影、非均匀密度）用于自由体积分析
- 热方程系列（1D 显式 FD、2D FEM、Jacobi 迭代）用于热传导与稳态分析
- 稀疏矩阵与 CG 法用于大规模线性代数运算
- Regula Falsi 与 VFT 拟合用于 $T_g$ 的精确确定

代码具备完整的边界处理、数值鲁棒性和大量科学物理公式，可直接零参数运行并输出可解释的物理结果。
