# 分子动力学：表面催化反应机理 — 博士级合成项目说明

## 项目概述

本项目围绕 **Pt(111) 表面 CO 氧化催化反应** 这一前沿表面科学问题，融合 15 个种子科研代码项目的核心算法，构建了一个多尺度、多物理场耦合的博士级分子动力学计算框架。

### 科学问题定位

表面催化反应是能源转化、环境净化和化工合成的核心过程。CO 在 Pt(111) 表面的氧化反应 (Langmuir-Hinshelwood 机理) 是表面催化研究的原型体系：

```
CO(g) + *   ⇌ CO*
O₂(g) + 2* ⇌ 2O*
CO* + O*    → CO₂(g) + 2*
```

本项目从**电子结构**→**势能面**→**分子动力学**→**反应动力学**→**宏观反应-扩散**五个尺度递进，系统模拟表面催化反应的全流程。

---

## 原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 在合成项目中的角色 |
|------|--------|----------|-------------------|
| 1 | 1088_slap_io | SLAP Triad 稀疏矩阵 I/O | 紧束缚 Hamiltonian 矩阵的稀疏存储与读写 |
| 2 | 711_mandelbrot_area | 蒙特卡洛面积估计 | 三维空间吸附截面的蒙特卡洛积分与收敛性分析 |
| 3 | 974_r8cbb | 压缩边界带状矩阵分解 | 紧束缚电子结构计算中的特殊稀疏矩阵求解 |
| 4 | 358_fd1d_bvp | 一维有限差分边值问题 | 表面电双层 Poisson 方程求解 |
| 5 | 148_cellular_automaton | 规则 30 元胞自动机 | 表面吸附位点占据态的离散演化 |
| 6 | 1381_vandermonde | Vandermonde 系统求解 | 势能面的高阶多项式插值系数求解 |
| 7 | 929_pwl_product_integral | 分段线性乘积积分 | 反应速率热平均的精确数值积分 |
| 8 | 819_normal01_multivariate_distance | 多元正态距离统计 | Langevin 动力学中的随机力采样 |
| 9 | 1191_svd_snowfall | SVD 主成分分析 | MD 轨迹反应坐标提取与自由能面构建 |
| 10 | 249_cvt_3d_lumping | 3D CVT Lloyd 算法 | 吸附原子空间分布的最优采样 |
| 11 | 1063_sde | 随机微分方程 EM 方法 | Langevin 动力学的 BAOAB 分裂积分 |
| 12 | 778_monopoly_matrix | 马尔可夫转移矩阵 | 表面反应网络的稳态分析与主方程 |
| 13 | 944_quad_serial | 复合求积公式 | 速率常数的 Gauss-Legendre 数值积分 |
| 14 | 1175_subpak | 多维网格生成 | 表面空间离散化与有限差分网格 |
| 15 | 283_diffusion_pde | 扩散 PDE 时间演化 | 表面物种浓度反应-扩散方程求解 |

---

## 新增数学物理模型与核心公式

### 1. 表面晶格结构

Pt(111) FCC 表面最近邻原子间距：

$$d_{nn} = \frac{a}{\sqrt{2}}$$

层间堆垛 (ABC 序列) 层间距：

$$d_z = \frac{a}{\sqrt{3}}$$

### 2. 势能面 (PES)

**Morse 势能** (吸附键)：

$$V_M(r) = D_e \left[1 - e^{-a(r - r_e)}\right]^2 - D_e$$

**Lennard-Jones 12-6 势能** (弱相互作用)：

$$V_{LJ}(r) = 4\varepsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right]$$

**多项式插值 PES** (Vandermonde 系统)：

$$V(x,y,z) = \sum_{i+j+k \leq N} c_{ijk} \cdot (x-x_0)^i (y-y_0)^j (z-z_0)^k$$

系数 $\mathbf{c}$ 通过求解 Vandermonde 线性系统 $A\mathbf{c} = \mathbf{E}$ 获得。

### 3. 紧束缚电子结构

**Slater-Koster 双中心近似**：

$$H_{ij} = \begin{cases}
\varepsilon_i & i = j \\
V_{ss\sigma} \cdot e^{-(r_{ij}-r_0)/d_0} & i \neq j, \, r_{ij} \leq r_c
\end{cases}$$

**Fermi-Dirac 占据**：

$$f_i = \frac{1}{\exp\left(\frac{\varepsilon_i - \varepsilon_F}{k_B T}\right) + 1}$$

**态密度 (Gaussian 展宽)**：

$$\rho(E) = \sum_i \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(E-\varepsilon_i)^2}{2\sigma^2}\right)$$

### 4. Langevin 随机分子动力学

**Langevin 方程**：

$$m\frac{d^2\mathbf{r}}{dt^2} = -\nabla V(\mathbf{r}) - \gamma m \frac{d\mathbf{r}}{dt} + \sqrt{2\gamma m k_B T} \, \boldsymbol{\eta}(t)$$

其中白噪声满足涨落-耗散定理：

$$\langle \eta_\alpha(t) \eta_\beta(t') \rangle = \delta_{\alpha\beta} \delta(t-t')$$

**BAOAB 分裂积分方案** (二阶精度辛积分器)：

$$\begin{aligned}
\text{B}: & \quad \mathbf{v} \leftarrow \mathbf{v} - \frac{\nabla V}{m} \frac{\Delta t}{2} \\
\text{A}: & \quad \mathbf{r} \leftarrow \mathbf{r} + \mathbf{v} \frac{\Delta t}{2} \\
\text{O}: & \quad \mathbf{v} \leftarrow c_1 \mathbf{v} + c_2 \sqrt{\frac{k_B T}{m}} \boldsymbol{\xi} \\
\text{A}: & \quad \mathbf{r} \leftarrow \mathbf{r} + \mathbf{v} \frac{\Delta t}{2} \\
\text{B}: & \quad \mathbf{v} \leftarrow \mathbf{v} - \frac{\nabla V}{m} \frac{\Delta t}{2}
\end{aligned}$$

其中 $c_1 = e^{-\gamma \Delta t}$, $c_2 = \sqrt{1-c_1^2}$。

**Einstein 扩散关系**：

$$D = \lim_{t \to \infty} \frac{\langle |\mathbf{r}(t) - \mathbf{r}(0)|^2 \rangle}{2 d \, t}$$

### 5. Langmuir-Hinshelwood 反应动力学

**覆盖度演化方程** (mean-field)：

$$\frac{d\theta_{CO}}{dt} = k_{ads}^{CO} P_{CO} (1-\theta_{CO}-\theta_O) - k_{des}^{CO} \theta_{CO} - k_{rxn} \theta_{CO} \theta_O$$

$$\frac{d\theta_O}{dt} = 2k_{ads}^{O_2} P_{O_2} (1-\theta_{CO}-\theta_O)^2 - k_{des}^O \theta_O - k_{rxn} \theta_{CO} \theta_O$$

**Arrhenius 速率常数**：

$$k = A \exp\left(-\frac{E_a}{k_B T}\right)$$

### 6. 反应-扩散方程

**一维反应-扩散方程**：

$$\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2} + R(c)$$

**稳态有限差分离散化** (非均匀网格)：

$$-D \frac{c_{i-1} - 2c_i + c_{i+1}}{\Delta x_L \Delta x_R} + R(c_i) = 0$$

### 7. 马尔可夫主方程

**主方程**：

$$\frac{dP_i}{dt} = \sum_j (W_{ji} P_j - W_{ij} P_i)$$

**稳态条件**：

$$\mathbf{W}^T \mathbf{P}_{ss} = 0, \quad \sum_i P_{ss,i} = 1$$

**熵产生率** (Schnakenberg)：

$$\sigma = \frac{1}{2} \sum_{i,j} (W_{ij} P_i - W_{ji} P_j) \ln\frac{W_{ij} P_i}{W_{ji} P_j}$$

### 8. SVD 反应坐标分析

**轨迹 SVD 分解**：

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

**自由能面**：

$$F(q) = -k_B T \ln P(q)$$

**集体性指数**：

$$\kappa = \frac{1}{N} \exp\left(-\sum_i p_i \ln p_i\right), \quad p_i = v_{1,i}^2$$

### 9. 蒙特卡洛与数值积分

**蒙特卡洛积分**：

$$I \approx V \cdot \frac{1}{N} \sum_{i=1}^N f(\mathbf{x}_i), \quad \sigma_I = V \sqrt{\frac{\text{Var}(f)}{N}}$$

**复合 Simpson 公式**：

$$\int_a^b f(x) dx \approx \frac{h}{3} \left[f_0 + 4\sum_{odd} f_i + 2\sum_{even} f_i + f_n\right]$$

**分段线性乘积积分** (解析精确)：

$$\int_{x_L}^{x_R} f(x)g(x) dx = \int_{x_L}^{x_R} (\alpha_f + \beta_f x)(\alpha_g + \beta_g x) dx$$

### 10. Centroidal Voronoi Tessellation

**Lloyd 算法能量泛函**：

$$E = \sum_j \rho(\mathbf{s}_j) \|\mathbf{s}_j - \mathbf{g}_{k(j)}\|^2$$

其中 $k(j) = \arg\min_i \|\mathbf{s}_j - \mathbf{g}_i\|$。

---

## 项目文件结构

```
120_synth_project/
├── main.py                          # 统一入口，零参数运行
├── utils.py                         # 物理常数、数值工具、网格生成
├── catalyst_surface.py              # Pt(111) 表面晶格、CA、CVT
├── potential_surface.py             # PES 构建、Vandermonde 插值、过渡态搜索
├── tight_binding.py                 # 紧束缚电子结构、稀疏矩阵、Poisson 方程
├── langevin_integrator.py           # Langevin MD (BAOAB)、Gillespie 反应动力学
├── reaction_diffusion.py            # 反应-扩散 PDE、LH 动力学
├── monte_carlo.py                   # MC 采样、复合求积、分段线性积分
├── markov_kinetics.py               # 马尔可夫链主方程、稳态分析
├── svd_reaction_coords.py           # SVD 反应坐标、自由能面、PCA
└── README_博士级合成说明.md          # 本文档
```

共 **10 个 .py 文件** + **1 个 README** + **1 个 main.py** = 12 个文件。

---

## 如何运行

```bash
cd /mnt/data/zpy/sci-swe/source\ code/Synthesis-project-python/120_synth_project
python main.py
```

程序将依次执行 8 个模块的计算演示，输出各物理量的计算结果。

---

## 合成方法说明

### 代码改造路径

1. **1175_subpak/grid1** → `utils.py:grid_uniform_1d()` / `grid_uniform_nd()`：
   保留多维均匀网格生成核心算法，注入物理单位转换和边界检查

2. **148_cellular_automaton** → `catalyst_surface.py:update_occupancy_ca()`：
   将元胞自动机规则 30 应用于表面吸附位点占据态的离散时间演化

3. **249_cvt_3d_lumping** → `catalyst_surface.py:cvt_optimize_sites()`：
   保留 Lloyd 迭代和密度加权质心更新，将应用域从通用 3D 空间改为催化剂表面附近

4. **1381_vandermonde/dvand** → `potential_surface.py:fit()`：
   将 Bjorck-Pereyra 求解 Vandermonde 系统扩展为三维 PES 多项式拟合，使用 Tikhonov 正则化

5. **974_r8cbb** → `tight_binding.py:apply_border_banded_factorization()`：
   将压缩边界带状矩阵分解算法应用于紧束缚 Hamiltonian 的结构化求解

6. **1088_slap_io** → `tight_binding.py:write_slap_format()` / `read_slap_format()`：
   将 SLAP Triad 稀疏矩阵 I/O 用于 Hamiltonian 矩阵的存储和读取

7. **358_fd1d_bvp** → `tight_binding.py:solve_poisson_fd1d()`：
   将非均匀网格有限差分用于表面电双层 Poisson 方程

8. **1063_sde/emstrong** → `langevin_integrator.py:step()`：
   将 Euler-Maruyama 思想升级为 BAOAB 二阶辛积分器，用于 Langevin 方程

9. **819_normal01_multivariate_distance** → `langevin_integrator.py:step()`：
   多元正态随机采样用于 Ornstein-Uhlenbeck 速度更新

10. **283_diffusion_pde** → `reaction_diffusion.py:solve_time_dependent()`：
    将扩散 PDE 扩展为含非线性反应源项的反应-扩散方程

11. **929_pwl_product_integral** → `monte_carlo.py:PiecewiseLinearProductIntegral.integrate()`：
    保留分段线性乘积的解析积分算法，应用于反应速率热平均

12. **944_quad_serial** → `monte_carlo.py:QuadratureIntegrator`：
    将复合梯形求积扩展为梯形、Simpson、Gauss-Legendre 三种方法

13. **711_mandelbrot_area** → `monte_carlo.py:MonteCarloSampler`：
    保留蒙特卡洛采样和收敛性测试框架，应用于高维吸附截面积分

14. **778_monopoly_matrix** → `markov_kinetics.py:SurfaceReactionNetwork`：
    将马尔可夫转移矩阵思想从游戏状态迁移到表面催化反应状态网络

15. **1191_svd_snowfall** → `svd_reaction_coords.py:ReactionCoordinateAnalyzer`：
    将 SVD 主成分分析从气象数据迁移到 MD 轨迹反应坐标提取

---

## 数值鲁棒性与边界处理

- **除零保护**：`safe_divide()` 函数处理所有可能出现除零的位置
- **浓度截断**：反应-扩散求解中对浓度进行 `[0, 1]` 物理约束
- **概率归一化**：马尔可夫主方程积分中每步重新归一化概率
- **Hessian 正则化**：过渡态搜索中 Hessian 奇异时使用伪逆
- **网格稳定性**：反应-扩散显式 Euler 自动调整时间步以满足 CFL 条件
- **能量截断**：CVT 密度函数上限截断避免数值发散
- **Fermi-Dirac 稳定性**：低温下自动退化为 T=0 阶跃占据
