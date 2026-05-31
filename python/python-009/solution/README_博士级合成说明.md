# 系外行星大气光谱反演系统 — 博士级合成说明

## 一、项目概述

本项目围绕**天体物理：系外行星大气光谱反演**这一前沿科学问题，将15个原始科研代码项目的核心算法融合为一个完整的Python科研计算系统。系统实现了从模拟观测数据生成、正向辐射传输建模、到贝叶斯参数反演与不确定性量化的全流程，计算复杂度达到博士级前沿水平。

---

## 二、原项目到科学问题的映射

| 原始项目 | 核心算法 | 在合成项目中的角色 |
|---------|---------|------------------|
| **444_football_dynamic** | 动态规划组合计数 | 启发化学平衡路径的组合优化与多物种丰度的递推计算框架 |
| **780_mortality** | 统计分布（CDF/PDF）、期望计算 | 大气参数不确定性建模：化学丰度的对数正态分布采样、统计推断 |
| **384_fem1d_adaptive** | 自适应有限元方法（FEM）、误差估计、网格加密 | 一维辐射传输方程的Galerkin有限元离散化与自适应加密求解 |
| **308_distmesh** | 基于距离函数的Delaunay网格生成 | 行星大气二维球壳截面的结构化网格生成与质量评估 |
| **1320_triangle_to_fem** | 网格格式转换（TRIANGLE→FEM） | 大气网格数据在不同数值格式间的转换与清洗 |
| **116_box_plot** | 数据分箱与统计离散化 | 大气压强分层的对数等间距离散化、温度剖面分箱分析 |
| **611_joukowsky_transform** | Joukowsky复变映射 | Voigt线型函数的Faddeeva复误差函数计算与非对称线型构造 |
| **577_image_diffuse** | 扩散方程数值平滑 | 辐射传输有限元解的数值扩散稳定化处理与边界平滑 |
| **1081_simplex_monte_carlo** | 单纯形上的Monte Carlo采样与积分 | 化学丰度单纯形约束下的Dirichlet分布采样、参数空间探索 |
| **097_bisection_rc** | 反向通信二分法求根 | 非线性反演中光学深度与有效半径的精确求根计算 |
| **1116_sphere_exactness** | 球面积分规则精确性测试 | 辐射传输角度离散化的Gauss-Legendre球面积分、Henyey-Greenstein相函数 |
| **1197_tec_io** | TEC科学数据文件解析 | 光谱数据与网格数据的结构化读写（ASCII/JSON格式） |
| **870_pink_noise** | 1/f噪声生成（多尺度随机游走） | MCMC采样器中的长程相关噪声注入，增强参数空间探索效率 |
| **120_broyden** | Broyden拟牛顿法 | 非线性大气参数反演的低秩Jacobian更新优化 |
| **978_r8crs** | 稀疏矩阵CRS存储与运算 | 辐射传输有限元离散化后的大型稀疏线性系统求解（GMRES） |

---

## 三、新增数学物理模型与核心公式

### 3.1 行星大气物理模型

#### 流体静力学平衡
行星大气满足流体静力学平衡方程：

$$
\frac{dP}{dz} = -\rho(z) g(z)
$$

其中重力加速度随高度变化：

$$
g(z) = \frac{GM_p}{(R_p + z)^2}
$$

#### 理想气体状态方程

$$
P = \frac{\rho k_B T}{\mu m_u}
$$

其中 $\mu$ 为平均分子量，$m_u$ 为原子质量单位。

#### 大气标高

$$
H = \frac{k_B T}{\mu m_u g(z)}
$$

#### 行星平衡温度
假设零反照率与全球能量再分配：

$$
T_{\text{eq}} = T_* \sqrt{\frac{R_*}{2a}} \cdot f^{1/4}
$$

其中 $f$ 为能量再分配因子（$f=1/4$ 对应全球平均）。

### 3.2 Guillot (2010) 温度-压强剖面模型

用于强烈辐射加热的行星大气（热木星）：

$$
T^4 = \frac{3T_{\text{int}}^4}{4}\left(\frac{2}{3} + \tau\right) + \frac{3T_{\text{irr}}^4}{4} f \left[\frac{2}{3} + \frac{1}{\gamma\sqrt{3}} + \left(\frac{\gamma}{\sqrt{3}} - \frac{1}{\gamma\sqrt{3}}\right) e^{-\gamma\tau\sqrt{3}}\right]
$$

其中光学厚度 $\tau = P \kappa_{\text{ir}} / g$，$\gamma$ 为可见光与红外不透明度之比。

### 3.3 辐射传输方程

平面平行近似下的辐射传输方程：

$$
\mu \frac{dI(\tau, \mu)}{d\tau} = I(\tau, \mu) - S(\tau, \mu)
$$

其中源函数 $S$ 包含热辐射与散射贡献：

$$
S = (1-\omega)B + \frac{\omega}{4\pi}\int_{4\pi} P(\mu, \mu') I(\mu') d\Omega'
$$

#### 透射深度计算（等效高度法）

$$
\delta(\lambda) = \left[\frac{R_{\text{eff}}(\lambda)}{R_*}\right]^2 - \left[\frac{R_p}{R_*}\right]^2 \quad [\text{ppm}]
$$

有效半径由光学深度条件确定：$\tau_{\text{vert}}(\lambda, z_{\text{eff}}) \approx 1$。

### 3.4 分子吸收截面与线型

#### Voigt 线型函数
高斯与洛伦兹线型的卷积：

$$
V(x; \sigma, \gamma) = \frac{\text{Re}[w(z)]}{\sigma\sqrt{2\pi}}, \quad z = \frac{x + i\gamma}{\sigma\sqrt{2}}
$$

其中 $w(z)$ 为Faddeeva函数：$w(z) = e^{-z^2}\text{erfc}(-iz)$。

#### 温度依赖线强

$$
S(T) = S(T_0) \left(\frac{T_0}{T}\right)^{3/2} \exp\left[-c_2 E_{\text{low}}\left(\frac{1}{T} - \frac{1}{T_0}\right)\right] \frac{1 - e^{-c_2 \tilde{\nu}_0/T}}{1 - e^{-c_2 \tilde{\nu}_0/T_0}}
$$

其中 $c_2 = hc/k_B \approx 1.4387770\ \text{cm}\cdot\text{K}$。

### 3.5 Henyey-Greenstein 相函数

描述散射角分布：

$$
P_{\text{HG}}(\cos\Theta) = \frac{1}{4\pi} \frac{1-g^2}{(1+g^2-2g\cos\Theta)^{3/2}}
$$

归一化条件：$\int_{4\pi} P_{\text{HG}} d\Omega = 1$。

### 3.6 Delta-Eddington 近似

求解辐射传输的简化解析近似：

$$
\tau^* = (1-\omega f)\tau, \quad \omega^* = \frac{(1-f)\omega}{1-\omega f}, \quad g^* = \frac{g-f}{1-f}
$$

其中 $f = g^3$。反射率：

$$
R = \frac{(r_\infty - r_0)(r_\infty + r_0)(1-e^{-2k\tau^*})}{(r_\infty+r_0)^2 - (r_\infty-r_0)^2 e^{-2k\tau^*}}
$$

$k = \sqrt{3(1-\omega^*)(1-\omega^* g^*)}$。

### 3.7 反演优化框架

#### 最小二乘目标函数

$$
\chi^2(\boldsymbol{\theta}) = \sum_{i=1}^{N_\lambda} \frac{(F_{\text{obs}}(\lambda_i) - F_{\text{model}}(\lambda_i; \boldsymbol{\theta}))^2}{\sigma_i^2}
$$

#### Levenberg-Marquardt 迭代

$$
(\mathbf{J}^T\mathbf{J} + \lambda \cdot \text{diag}(\mathbf{J}^T\mathbf{J})) \Delta\boldsymbol{\theta} = -\mathbf{J}^T\mathbf{r}
$$

#### Tikhonov 正则化

$$
\min_{\boldsymbol{\theta}} \left[ \|\mathbf{F}_{\text{obs}} - \mathbf{F}_{\text{model}}(\boldsymbol{\theta})\|^2 + \alpha \|\mathbf{L}\boldsymbol{\theta}\|^2 \right]
$$

其中 $\mathbf{L}$ 为差分正则化矩阵。

### 3.8 贝叶斯后验分布

$$
P(\boldsymbol{\theta}|\mathbf{D}) \propto P(\mathbf{D}|\boldsymbol{\theta}) P(\boldsymbol{\theta}) = \exp\left(-\frac{1}{2}\chi^2(\boldsymbol{\theta})\right) \cdot P_{\text{prior}}(\boldsymbol{\theta})
$$

Metropolis-Hastings 接受率：

$$
\alpha = \min\left(1, \frac{P(\boldsymbol{\theta}'|\mathbf{D})}{P(\boldsymbol{\theta}|\mathbf{D})} \cdot \frac{q(\boldsymbol{\theta}|\boldsymbol{\theta}')}{q(\boldsymbol{\theta}'|\boldsymbol{\theta})}\right)
$$

### 3.9 稀疏线性系统求解

辐射传输有限元离散化产生稀疏线性系统 $\mathbf{A}\mathbf{x} = \mathbf{b}$，使用GMRES迭代求解：

在Krylov子空间 $\mathcal{K}_k(\mathbf{A}, \mathbf{r}_0)$ 中寻找使残差最小的近似解，其中Arnoldi过程满足：

$$
\mathbf{A}\mathbf{V}_k = \mathbf{V}_{k+1}\mathbf{H}_k
$$

### 3.10 球面积分与角度离散化

Gauss-Legendre × 均匀方位角积分公式：

$$
\int_{4\pi} f(\Omega) d\Omega = \int_0^{2\pi}\int_{-1}^{1} f(\mu, \phi) d\mu d\phi \approx \sum_{k=1}^{N_\mu}\sum_{l=1}^{N_\phi} w_k \cdot \frac{2\pi}{N_\phi} \cdot f(\mu_k, \phi_l)
$$

---

## 四、文件架构与实现路径

### 4.1 项目文件列表

| 文件 | 功能 | 对应原始项目 |
|-----|------|-----------|
| `main.py` | 统一入口，执行完整反演流程 | — |
| `atmospheric_model.py` | 大气物理模型（T-P剖面、化学平衡、云层） | 780_mortality, 116_box_plot |
| `radiative_transfer.py` | 1D辐射传输有限元求解与自适应加密 | 384_fem1d_adaptive, 577_image_diffuse |
| `mesh_generator.py` | 大气网格生成与质量评估 | 308_distmesh, 1320_triangle_to_fem |
| `spectral_synthesis.py` | 分子截面、Voigt线型、瑞利散射 | 611_joukowsky_transform |
| `monte_carlo_sampler.py` | MCMC、嵌套采样、单纯形采样、粉红噪声 | 1081_simplex_monte_carlo, 870_pink_noise |
| `inversion_solver.py` | 非线性优化（LM、Broyden、二分法、Tikhonov） | 120_broyden, 097_bisection_rc |
| `sparse_linear_algebra.py` | 稀疏矩阵CRS格式、GMRES、ILU预条件 | 978_r8crs |
| `sphere_quadrature.py` | 球面积分、HG相函数、Delta-Eddington | 1116_sphere_exactness |
| `data_io.py` | 数据读写、格式转换、元数据管理 | 1197_tec_io, 1320_triangle_to_fem |

### 4.2 运行流程

```
main.py
  ├── Step 1: generate_synthetic_observation()
  │     ├── 构建行星大气物理模型 (atmospheric_model.py)
  │     ├── 生成大气网格 (mesh_generator.py)
  │     ├── 计算分子吸收截面 (spectral_synthesis.py)
  │     ├── 求解辐射传输方程 (radiative_transfer.py)
  │     ├── 球面积分与角度离散 (sphere_quadrature.py)
  │     └── 添加观测噪声 → observed_spectrum.dat
  ├── Step 2: inversion_least_squares()
  │     ├── Levenberg-Marquardt优化 (inversion_solver.py)
  │     └── 输出优化参数
  ├── Step 3: bayesian_mcmc_analysis()
  │     ├── Metropolis-Hastings采样 (monte_carlo_sampler.py)
  │     └── 后验统计 → mcmc_samples.dat
  ├── Step 4: numerical_diagnostics()
  │     ├── 网格质量测试 (mesh_generator.py)
  │     ├── 球面积分精确性测试 (sphere_quadrature.py)
  │     ├── 稀疏线性求解器测试 (sparse_linear_algebra.py)
  │     └── 单纯形采样测试 (monte_carlo_sampler.py)
  └── Step 5: 保存结果文件
```

---

## 五、科学问题解决能力

本合成项目能够解决以下前沿科学问题：

1. **系外行星大气成分识别**：通过透射光谱中的分子吸收特征（H₂O、CH₄、CO、CO₂、Na等），定量反演大气化学丰度。

2. **温度-压强剖面重构**：利用不同波长的光谱探针穿透不同大气深度，结合辐射传输模型反演温度随压强的分布。

3. **云层特性约束**：通过光谱连续区与分子带区的对比分析，约束云层的光学厚度和顶部压强。

4. **金属丰度与C/O比测量**：通过多种含碳/含氧分子的相对强度，推断行星形成历史与迁移路径。

5. **贝叶斯不确定性量化**：利用MCMC采样全面评估参数反演的不确定性，避免局部最优陷阱。

---

## 六、运行方式

### 环境要求
- Python 3.8+
- NumPy
- SciPy

### 安装依赖
```bash
pip install numpy scipy
```

### 运行项目
```bash
cd 009_synth_project
python main.py
```

无需任何命令行参数，程序将自动：
1. 生成模拟观测透射光谱
2. 执行最小二乘参数反演
3. 进行贝叶斯MCMC不确定性分析
4. 输出数值诊断报告
5. 保存所有结果文件

### 输出文件
- `observed_spectrum.dat` — 模拟观测光谱（波长、流量、误差）
- `retrieved_spectrum.dat` — 反演模型光谱
- `temperature_profile_comparison.dat` — 真实与反演温度剖面对比
- `mcmc_samples.dat` — MCMC后验样本
- `retrieval_metadata.json` — 反演元数据与诊断结果

---

## 七、数值鲁棒性与边界处理

1. **压强/温度边界**：所有大气参数严格限制在物理合理范围（T > 50 K, P > 0, VMR > 0）
2. **除零保护**：标高、重力、Voigt线型等计算中均设置最小阈值（1e-15 ~ 1e-30）
3. **数值截断**：反三角函数输入截断至[-1, 1]，对数输入截断至正数域
4. **稀疏矩阵稳定性**：GMRES求解设置最大迭代次数与残差容差，退化时自动回退
5. **MCMC边界反射**：参数提议采样后自动裁剪至先验边界，避免非法参数空间
6. **有限元扩散平滑**：辐射传输数值解后处理加入可控扩散项，抑制数值振荡

---

## 八、学术参考

1. Guillot, T. (2010). *On the radiative equilibrium of irradiated planetary atmospheres*. A&A, 520, A27.
2. Madhusudhan, N. (2018). *Atmospheric Retrieval of Exoplanets*. arXiv:1805.10691.
3. Skilling, J. (2006). *Nested Sampling for General Bayesian Computation*. Bayesian Analysis, 1(4), 833-860.
4. Line, M. R., et al. (2013). *A Systematic Retrieval Analysis of Secondary Eclipse Spectra. I. A Comparison of Atmospheric Retrieval Techniques*. ApJ, 775(2), 137.
