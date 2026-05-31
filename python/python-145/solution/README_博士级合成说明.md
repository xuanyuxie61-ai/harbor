# 博士级合成说明：多因子 HJM 利率期限结构模型

## 1. 项目概述

本项目将 **15 个科研代码种子项目** 的核心算法融合为一个面向**金融工程：利率期限结构模型**的博士级科学计算系统。

核心科学问题：
> **在多因子随机扰动环境下，使用前向利率随机偏微分方程（HJM 框架）模拟利率期限结构的非线性演化，并通过有限元空间离散、多项式混沌展开进行不确定性量化。**

## 2. 原项目到科学问题的映射

| 原项目 | 核心算法 | 合成后角色 |
|--------|---------|-----------|
| 703_lorenz96_ode | Lorenz-96 混沌 ODE | `stochastic_dynamics.py`：高维市场微观噪声驱动因子 |
| 757_mesh2d | 2D 非结构化三角网格 | `fem_maturity_grid.py`：期限-时间域 FEM 空间剖分 |
| 322_duffing_ode | Duffing 非线性振子 | `stochastic_dynamics.py`：利率周期性波动与制度切换 |
| 524_hermite_product_polynomial | Hermite 乘积多项式 | `polynomial_chaos_uq.py`：Wiener 混沌展开基函数 |
| 698_log_normal | 对数正态分布 | `special_functions.py`：利率 positivity 约束与债券定价 |
| 1153_st_io | 稀疏三元组矩阵 I/O | `sparse_linear_algebra.py`：大规模 FEM 稀疏系统存储与读取 |
| 838_oregonator_ode | Oregonator 化学反应 | `stochastic_dynamics.py`：流动性冲击传播模型 |
| 1072_shepard_interp_2d | 2D Shepard 插值 | `yield_curve_calibration.py`：不规则市场报价插值 |
| 1033_rk23 | RK2/RK3 时间积分 | `time_stepping.py`：ODE/PDE 时间推进与误差估计 |
| 200_collocation | Horner 多项式求值 | `yield_curve_calibration.py`：收益率曲线快速多项式求值 |
| 382_fem_to_xml | FEM 网格序列化 | `fem_maturity_grid.py` 与 `sparse_linear_algebra.py`：数据持久化 |
| 547_human_data | 边界轮廓提取 | `yield_curve_calibration.py`：收益率曲线形状特征点提取 |
| 644_lambert_w | Lambert W 函数 | `special_functions.py`：闭式解析解辅助计算 |
| 408_fem2d_poisson_rectangle | 2D FEM Poisson 求解 | `fem_maturity_grid.py`：静态收益率曲线椭圆型求解器 |
| 405_fem2d_heat_sparse | 2D FEM 热方程（稀疏） | `fem_maturity_grid.py`：期限结构抛物型 PDE 时间推进 |

## 3. 核心数学物理模型与公式

### 3.1 HJM 无套利框架

Heath-Jarrow-Morton (1992) 一般框架：

$$
df(t,T) = \alpha(t,T)\,dt + \sum_{i=1}^{d} \sigma_i(t,T)\,dW_i(t)
$$

无套利漂移限制：

$$
\alpha(t,T) = \sum_{i=1}^{d} \sigma_i(t,T) \int_t^T \sigma_i(t,u)\,du
$$

### 3.2 Musiela 参数化

令 $s = T - t$（剩余期限），定义 $r_t(s) = f(t, t+s)$，则：

$$
\frac{\partial r_t(s)}{\partial t} = -\frac{\partial r_t(s)}{\partial s} + \nu \frac{\partial^2 r_t(s)}{\partial s^2} + \mu(s) \frac{\partial r_t(s)}{\partial s} + \alpha(t,s) + F(t,s)
$$

边界条件：
- $r_t(0) = r_0(t)$（短期利率 Dirichlet 边界）
- $r_t(s_{\max}) = r_\infty$（长期利率渐近值）
- $r_0(s) = r_{\text{init}}(s)$（初始期限结构）

### 3.3 多因子波动率结构

$$
\sigma_1(t,s) = \sigma_0 e^{-\kappa_1 s} \quad \text{（水平因子）}
$$
$$
\sigma_2(t,s) = \sigma_0 s e^{-\kappa_2 s} \quad \text{（斜率因子）}
$$
$$
\sigma_3(t,s) = \sigma_{\text{chaos}}(t) \cdot e^{-\kappa_3 s} \quad \text{（混沌驱动因子）}
$$

其中 $\sigma_{\text{chaos}}(t)$ 由 Lorenz-96、Duffing 与 Oregonator 系统的状态通过非线性耦合矩阵投影得到：

$$
\sigma_{\text{chaos}}(t) = \mathbf{C} \cdot \begin{bmatrix} \|\mathbf{y}_{\text{L96}}(t)\|_2 / \sqrt{N} \\ y_{\text{Duff}}^{(1)}(t) \\ y_{\text{Oreg}}^{(1)}(t) \end{bmatrix}
$$

### 3.4 Lorenz-96 混沌系统

$$
\frac{dy_i}{dt} = (y_{i+1} - y_{i-2}) y_{i-1} - y_i + F, \quad i = 1, \ldots, N
$$

指标循环 wrapping，$F$ 为外部强迫（对应央行政策锚定）。

### 3.5 Duffing 振子

$$
x'' + \delta x' + \alpha x + \beta x^3 = \gamma \cos(\omega t)
$$

一阶系统形式：

$$
\begin{cases}
y_1' = y_2 \\
y_2' = -\delta y_2 - \alpha y_1 - \beta y_1^3 + \gamma \cos(\omega t)
\end{cases}
$$

### 3.6 Oregonator 化学反应系统

无量纲化方程（Field-Körös-Noyes 机理）：

$$
\begin{cases}
\displaystyle\frac{du}{dt} = \frac{qv - uv + u(1-u)}{\eta_1} \\[6pt]
\displaystyle\frac{dv}{dt} = \frac{-qv - uv + fw}{\eta_2} \\[6pt]
\displaystyle\frac{dw}{dt} = u - w
\end{cases}
$$

其中：
- $\eta_1 = \dfrac{k_c b}{k_5 a}$
- $\eta_2 = \dfrac{2 k_c k_4 b}{k_2 k_5 a}$
- $q = \dfrac{2 k_3 k_4}{k_2 k_5}$

### 3.7 债券定价与零息收益率

零息债券价格：

$$
P(t,T) = \exp\left(-\int_t^T f(t,s)\,ds\right)
$$

零息收益率：

$$
y(t,T) = -\frac{\ln P(t,T)}{T-t} = \frac{1}{T-t}\int_t^T f(t,s)\,ds
$$

### 3.8 多项式混沌展开（Polynomial Chaos）

前向利率的混沌展开：

$$
r_t(s; \boldsymbol{\xi}) = \sum_{|\boldsymbol{\alpha}| \leq p} r_{t,\boldsymbol{\alpha}}(s) \cdot \text{He}_{\boldsymbol{\alpha}}(\boldsymbol{\xi})
$$

其中 $\boldsymbol{\xi} \sim \mathcal{N}(0, \mathbf{I}_d)$，$\text{He}_{\boldsymbol{\alpha}}$ 为概率论 Hermite 乘积多项式：

$$
\text{He}_{\boldsymbol{\alpha}}(\boldsymbol{\xi}) = \prod_{j=1}^{d} \text{He}_{\alpha_j}(\xi_j)
$$

概率论 Hermite 多项式递推：

$$
\text{He}_{n+1}(x) = x \cdot \text{He}_n(x) - n \cdot \text{He}_{n-1}(x)
$$

Sobol 主效应敏感性指标：

$$
S_i = \frac{1}{\text{Var}(Y)} \sum_{\boldsymbol{\alpha}: \alpha_i > 0, \alpha_{j \neq i} = 0} \boldsymbol{\alpha}! \cdot c_{\boldsymbol{\alpha}}^2
$$

### 3.9 后向 Euler 时间离散

$$
(\mathbf{M} + \Delta t \, \mathbf{A}) \mathbf{u}^{n+1} = \mathbf{M} \mathbf{u}^n + \Delta t \, \mathbf{f}^{n+1}
$$

其中 $\mathbf{M}$ 为质量矩阵，$\mathbf{A}$ 为刚度矩阵。

### 3.10 RK3 时间推进

$$
\begin{aligned}
\mathbf{k}_1 &= \Delta t \, \mathbf{f}(t_n, \mathbf{u}_n) \\
\mathbf{k}_2 &= \Delta t \, \mathbf{f}(t_n + \Delta t, \mathbf{u}_n + \mathbf{k}_1) \\
\mathbf{k}_3 &= \Delta t \, \mathbf{f}(t_n + \tfrac{\Delta t}{2}, \mathbf{u}_n + \tfrac{\mathbf{k}_1 + \mathbf{k}_2}{4}) \\
\mathbf{u}_{n+1} &= \mathbf{u}_n + \frac{\mathbf{k}_1 + \mathbf{k}_2 + 4\mathbf{k}_3}{6}
\end{aligned}
$$

### 3.11 有限元二次基函数（T6 元）

面积坐标 $(L_1, L_2, L_3)$ 下的二次基：

$$
\begin{aligned}
N_1 &= L_1(2L_1 - 1), & N_2 &= L_2(2L_2 - 1), & N_3 &= L_3(2L_3 - 1) \\
N_4 &= 4L_1 L_2, & N_5 &= 4L_2 L_3, & N_6 &= 4L_3 L_1
\end{aligned}
$$

### 3.12 对数正态分布

PDF：

$$
p(x; \mu, \sigma) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0
$$

逆 CDF：

$$
F^{-1}(p) = \exp\left(\mu + \sigma \cdot \Phi^{-1}(p)\right)
$$

其中 $\Phi^{-1}$ 采用 Wichura 算法 AS 241（精度达 $10^{-16}$）。

### 3.13 Lambert W 函数

定义：$W(z) e^{W(z)} = z$

在利率模型中用于某些隐式方程的闭式解：

$$
r = a + b e^{-cr} \quad \Longrightarrow \quad r = \frac{1}{c} W\left(\frac{bc}{e^{ac}}\right) + a
$$

本项目采用 WAPR 算法（Barry et al., ACM TOMS 1995）进行高精度近似。

### 3.14 Shepard 插值

$$
w_j = \frac{\|\mathbf{x} - \mathbf{x}_j\|^{-p}}{\sum_k \|\mathbf{x} - \mathbf{x}_k\|^{-p}}, \quad f(\mathbf{x}) = \sum_j w_j z_j
$$

### 3.15 Horner 多项式求值

$$
p(x) = c_0 + x(c_1 + x(c_2 + \cdots + x(c_m))\cdots)
$$

计算复杂度 $O(m)$，避免显式幂运算。

## 4. 项目文件结构

```
145_synth_project/
├── main.py                        # 统一入口，零参数可运行
├── special_functions.py           # 对数正态、正态逆 CDF、Lambert W
├── polynomial_chaos_uq.py         # Hermite 多项式、混沌展开、Sobol 分析
├── stochastic_dynamics.py         # Lorenz-96、Duffing、Oregonator、多因子耦合
├── time_stepping.py               # RK2/RK3/RK23、后向 Euler、自适应步长
├── fem_maturity_grid.py           # 2D 网格生成、T6 基函数、FEM 矩阵组装
├── yield_curve_calibration.py     # Shepard 插值、Horner 求值、曲线拟合与特征提取
├── sparse_linear_algebra.py       # 稀疏三元组 I/O、带宽估计、稀疏求解器
├── term_structure_pde.py          # HJM 漂移、前向利率 PDE、债券定价
├── hjm_model.py                   # HJM 多因子模型类、路径模拟
└── README_博士级合成说明.md        # 中文说明文档
```

## 5. 修改说明

### 5.1 新增文件与改造方法

1. **special_functions.py**
   - 将 `698_log_normal` 的对数正态 PDF/CDF/逆 CDF/采样移植为 Python
   - 将 `698_log_normal` 的 Wichura 正态逆 CDF（AS 241）完整移植
   - 将 `644_lambert_w` 的 WAPR 算法完整移植为向量化 NumPy 实现
   - 加入边界检查（CDF ∈ [0,1]、σ > 0、Lambert W 分支定义域）

2. **polynomial_chaos_uq.py**
   - 将 `524_hermite_product_polynomial` 的概率论 Hermite 多项式系数与求值移植
   - 新增多维乘积多项式求值 `hermite_product_polynomial_value`
   - 新增多项式混沌展开 `polynomial_chaos_expand`
   - 新增多维指标生成 `generate_multi_indices`
   - 新增 Sobol 敏感性分析 `sobol_sensitivity`

3. **stochastic_dynamics.py**
   - 将 `703_lorenz96_ode` 的循环导数与参数系统完整移植
   - 将 `322_duffing_ode` 的非线性受迫振子移植
   - 将 `838_oregonator_ode` 的化学反应系统（含全部无量纲参数推导）移植
   - 新增 `multi_factor_coupling`：将三个系统的状态投影到 HJM 波动率空间

4. **time_stepping.py**
   - 将 `1033_rk23` 的 RK2、RK3 与嵌入式 RK23（带误差估计）完整移植
   - 将 `405_fem2d_heat_sparse` 的后向 Euler 格式移植并泛化为通用 PDE 步进器
   - 新增自适应步长 RK23（基于局部误差范数的步长控制策略）

5. **fem_maturity_grid.py**
   - 将 `757_mesh2d` 的矩形网格生成与三角形面积计算移植
   - 将 `408_fem2d_poisson_rectangle` 的 T6 二次基函数、刚度/质量矩阵组装、Dirichlet 边界处理完整移植
   - 将 `405_fem2d_heat_sparse` 的后向 Euler 热方程求解器移植
   - 新增参考单元到物理单元的坐标映射 `reference_to_physical_t3`
   - 新增三点 Gauss 积分规则

6. **yield_curve_calibration.py**
   - 将 `1072_shepard_interp_2d` 的逆距离加权插值完整移植
   - 将 `200_collocation` 的 Horner 多项式求值移植
   - 基于 `547_human_data` 的边界轮廓思想，新增收益率曲线特征提取（峰值、谷值、拐点）
   - 新增最小二乘多项式拟合（含 Vandermonde 条件数分析）
   - 新增综合校准函数 `calibrate_yield_curve`

7. **sparse_linear_algebra.py**
   - 将 `1153_st_io` 的稀疏三元组读写格式完整移植
   - 新增 COO ↔ ST 格式转换
   - 新增稀疏矩阵带宽估计（用于 FEM 性能分析）
   - 新增稀疏直接求解器封装（含 LU 分解选项与残差检验）

8. **term_structure_pde.py**
   - 实现 HJM 无套利漂移项的 10 点 Gauss-Legendre 高效数值积分
   - 实现前向利率输运-扩散 PDE 的有限差分离散
   - 实现债券价格与零息收益率的数值积分计算
   - 实现完整的隐式后向 Euler PDE 求解流程

9. **hjm_model.py**
   - 设计 `HJMMultiFactorModel` 类，封装全部多因子参数
   - 实现波动率结构计算（含混沌耦合投影）
   - 实现随机动力学系统的联合时间推进（含 Oregonator 子步细分与 NaN 回退）
   - 实现完整的前向利率曲线路径模拟

10. **main.py**
    - 零参数入口，按 10 个阶段顺序执行完整计算流程
    - 阶段 1：特殊函数验证
    - 阶段 2：市场收益率曲线校准
    - 阶段 3：FEM 空间网格生成与验证
    - 阶段 4：稀疏矩阵 I/O 测试
    - 阶段 5：随机动力学验证
    - 阶段 6：HJM 模型初始化
    - 阶段 7：多因子期限结构模拟
    - 阶段 8：债券定价与收益率计算
    - 阶段 9：多项式混沌不确定性量化
    - 阶段 10：结果汇总与稀疏矩阵存储

### 5.2 数值鲁棒性措施

- **非负性约束**：前向利率 `np.clip(f, 0.0, None)`，债券价格 `np.clip(price, 0.0, 1.0)`
- **NaN/Inf 回退**：Oregonator 子步积分中检测到 NaN/Inf 时重置为安全状态
- **边界检查**：对数正态 σ > 0，CDF ∈ [0,1]，Lambert W 定义域检查
- **矩阵奇异性防护**：稀疏求解后计算残差，条件数监控
- **Vandermonde 稳定性**：收益率曲线拟合前进行期限归一化
- **自适应步长**：Oregonator 刚性系统使用子步细分（dt_sub ≤ 0.01）

### 5.3 删除的可视化内容

- 所有 `matplotlib`、`plot`、`imshow` 相关代码均已删除
- 所有图形文件输出已替换为数值结果输出
- 仅保留文本/数值形式的计算结果

## 6. 运行方法

```bash
cd Synthesis-project-python/145_synth_project
python main.py
```

程序无需任何命令行参数，运行后自动执行从参数初始化、市场数据校准、FEM 网格生成、随机动力学验证、HJM 多因子模拟、债券定价、不确定性量化到结果汇总的完整流程。

## 7. 依赖环境

- Python ≥ 3.9
- NumPy
- SciPy（sparse、linalg）

无其他第三方依赖。

## 8. 科学问题总结

本项目解决的核心科学问题是：

> **在多因子随机扰动（混沌、周期振荡、化学反应型冲击）耦合驱动的环境下，如何数值求解 HJM 框架下的前向利率随机偏微分方程，并量化模型输出的不确定性？**

通过将 15 个独立科研项目的核心算法有机融合，本项目构建了一个具备以下特征的博士级计算平台：
- **多尺度耦合**：宏观期限结构 PDE + 中观随机动力学 + 微观多项式混沌展开
- **多物理场驱动**：流体力学型输运扩散 + 化学反应型冲击 + 非线性振子周期扰动
- **高维不确定性量化**：Hermite 混沌展开 + Sobol 敏感性分析
- **大规模稀疏计算**：FEM 二次基函数 + 稀疏矩阵直接求解
