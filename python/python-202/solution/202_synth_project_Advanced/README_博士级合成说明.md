# 不确定性量化：随机配置方法 — 博士级科研代码合成项目

## PROJECT 202: Stochastic Collocation Methods for Uncertainty Quantification

---

## 一、项目概述

本项目是一个面向**前沿科学计算**的博士级代码合成项目，核心围绕**不确定性量化 (Uncertainty Quantification, UQ)** 领域的**随机配置方法 (Stochastic Collocation Methods)** 展开。

### 科学问题

给定参数化随机椭圆偏微分方程：

$$-\frac{d}{dx}\left[\kappa(x,\omega) \frac{du}{dx}\right] = f(x), \quad x \in (0, L)$$
$$u(0) = u(L) = 0$$

其中扩散系数 $\kappa(x,\omega)$ 是**随机场**（通过 Karhunen-Loève 展开参数化），目标是计算解 $u(x,\omega)$ 的统计矩（均值、方差、偏度、峰度）、灵敏度指数（Sobol indices）和概率分布特征。

### 方法论

本项目实现了完整的随机配置法 UQ 框架：

1. **KL 展开**：将无限维随机场降维为有限个随机变量
2. **Smolyak 稀疏网格**：在高维随机空间高效构造配置点
3. **配置点求解**：在每个配置点求解确定性 PDE
4. **统计矩计算**：加权组合计算均值、方差、高阶矩
5. **PC 灵敏度分析**：多项式混沌展开计算 Sobol 指数
6. **自适应细化**：基于后验误差估计的 CVT 最优配置

---

## 二、核心数学模型与公式

### 2.1 Karhunen-Loève 展开

随机场的 KL 展开：

$$\kappa(x,\omega) = \mu_\kappa \exp\left(\sum_{k=1}^{K} \sqrt{\lambda_k} \, \varphi_k(x) \, \xi_k(\omega)\right)$$

其中 $\lambda_k, \varphi_k(x)$ 是协方差核 $C(x,x')$ 的特征值和特征函数：

$$\int_\Omega C(x,x') \varphi_k(x') dx' = \lambda_k \varphi_k(x)$$

对于平方指数核：$C(x,x') = \sigma^2 \exp\left(-\frac{|x-x'|^2}{2l_c^2}\right)$

对数变换保证 $\kappa > 0$（正定性）。

### 2.2 Smolyak 稀疏网格

Smolyak 算子避免维数灾难：

$$A(q,D) = \sum_{q \leq |\mathbf{i}|_1 \leq q+D-1} (-1)^{q+D-|\mathbf{i}|_1} \binom{D-1}{q+D-1-|\mathbf{i}|_1} \left(Q^{i_1} \otimes \cdots \otimes Q^{i_D}\right)$$

**对比**：
- 全张量积：$N_{full} = n^D$ 个配置点（指数增长）
- Smolyak 稀疏网格：$N_{sg} \sim N \cdot (\log N)^{D-1}$（近线性增长）

### 2.3 有限差分离散化

随机椭圆方程的有限差分格式：

$$-\frac{\kappa_{i+1/2}(u_{i+1} - u_i) - \kappa_{i-1/2}(u_i - u_{i-1})}{h^2} = f_i$$

界面扩散系数使用**调和平均**处理系数跳跃：

$$\kappa_{i+1/2} = \frac{2\kappa_i \kappa_{i+1}}{\kappa_i + \kappa_{i+1}}$$

### 2.4 Sobol 灵敏度指数

基于多项式混沌展开的方差分解：

$$\text{Var}[u] = \sum_{\alpha \neq 0} c_\alpha^2 \|\Psi_\alpha\|^2$$

- 一阶 Sobol 指数：$S_d = \frac{\sum_{\alpha: \alpha_d > 0, \alpha_{j \neq d} = 0} c_\alpha^2 \|\Psi_\alpha\|^2}{\text{Var}[u]}$
- 总效应 Sobol 指数：$S_{T,d} = 1 - \frac{\sum_{\alpha: \alpha_d = 0} c_\alpha^2 \|\Psi_\alpha\|^2}{\text{Var}[u]}$

### 2.5 多项式混沌基转换

不同 PC 基之间的转换矩阵：

$$T_{mn} = \frac{\langle P_m^{(target)}, P_n^{(source)} \rangle_\rho}{\|P_m^{(target)}\|^2}$$

多变量转换通过 Kronecker 积分解：$T_{\alpha\beta} = \prod_{d=1}^D T^{(d)}_{\alpha_d, \beta_d}$

### 2.6 自适应 CVT 细化

Lloyd 算法迭代求最优采样点：

$$c_i = \frac{\int_{V_i} \xi \, \rho(\xi) d\xi}{\int_{V_i} \rho(\xi) d\xi}$$

其中密度函数 $\rho(\xi) \propto |\Delta u(\xi)|^\alpha$ 由后验误差指示器驱动。

### 2.7 Lindberg 刚性 ODE

随机刚性化学动力学系统：

$$\frac{du_1}{dt} = -k_1 u_1 + k_2 u_2 u_3$$
$$\frac{du_2}{dt} = k_1 u_1 - k_2 u_2 u_3 - k_3 u_2^2$$
$$\frac{du_3}{dt} = k_3 u_2^2$$

刚性比 $S = \max|\text{Re}(\lambda)|/\min|\text{Re}(\lambda)| \sim 10^4 - 10^8$

隐式 Euler + Newton 迭代求解。

### 2.8 Fokker-Planck 方程

概率密度演化方程：

$$\frac{\partial p}{\partial t} + c \frac{\partial p}{\partial x} = D \frac{\partial^2 p}{\partial x^2}$$

Crank-Nicolson 时间推进，二阶精度无条件稳定。

---

## 三、项目结构

```
202_synth_project_Advanced/
├── main.py                      # 统一入口（零参数可运行）
├── polynomial_utils.py          # 正交多项式与 Gauss 求积
├── sparse_grid.py               # Smolyak 稀疏网格构造
├── stochastic_field.py          # KL 展开与随机场采样
├── elliptic_solver.py           # 随机椭圆方程有限差分求解
├── advection_solver.py          # 随机对流扩散方程求解
├── moment_computation.py        # 统计矩计算
├── basis_conversion.py          # 多项式混沌基转换
├── sensitivity_analysis.py      # Sobol 全局灵敏度分析
├── adaptive_refinement.py       # 自适应 CVT 配置细化
├── particle_transport.py        # 随机粒子输运与 Fokker-Planck
├── stiff_stochastic_ode.py      # 随机刚性 ODE (Lindberg)
├── response_surface.py          # 自然三次样条响应面重构
├── domain_mapper.py             # 随机域映射
├── sparse_operators.py          # 稀疏矩阵与 Block Toeplitz
├── classification_surrogate.py  # 解行为分类代理模型
├── calendar_converter.py        # 多尺度耦合与周期边界
└── README_博士级合成说明.md      # 本文档
```

共计 **17 个 .py 文件** + **1 个 README 文档**。

---

## 四、种子项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 映射到本项目 | 对应模块 |
|:---:|:---|:---|:---|:---|
| 1 | DeepHistoPathology | CNN 分类 + 类别平衡 | 解行为区域分类 | `classification_surrogate.py` |
| 2 | circle_integrals | Gamma 函数积分 | 正交多项式求积规则 | `polynomial_utils.py` |
| 3 | ToponymGeoreferencing | 空间坐标变换 | 随机域映射 | `domain_mapper.py` |
| 4 | matrix_assemble_parfor | 并行矩阵组装 | 稀疏网格张量积组装 | `sparse_grid.py` |
| 5 | calpak | 日历系统间转换 | 多项式基间转换 | `calendar_converter.py` |
| 6 | md | 速度 Verlet 积分 | 粒子拉格朗日输运 | `particle_transport.py` |
| 7 | fd1d_poisson | 三对角有限差分 | 随机椭圆方程求解 | `elliptic_solver.py` |
| 8 | r8bto | Block Toeplitz 求解 | 协方差算子求解 | `sparse_operators.py` |
| 9 | lindberg_ode | 刚性 ODE 系统 | 随机刚性化学动力学 | `stiff_stochastic_ode.py` |
| 10 | cvt_1d_sampling | Lloyd CVT 算法 | 自适应配置点优化 | `adaptive_refinement.py` |
| 11 | fd1d_advection_ftcs | FTCS 对流差分 | 随机对流扩散求解 | `advection_solver.py` |
| 12 | AccMLBio-esvlsss | 半监督 VAE | 弱监督标签传播 | `classification_surrogate.py` |
| 13 | r8sr | 稀疏矩阵存储 | CSR + 对角分离格式 | `sparse_operators.py` |
| 14 | BoolQ YesNoQA | 弱监督分类 | 多项式分类边界 | `classification_surrogate.py` |
| 15 | interp_ncs | 自然三次样条 | 响应面重构 | `response_surface.py` |

---

## 五、运行方法

```bash
cd /path/to/202_synth_project_Advanced
python main.py
```

**零参数**即可运行。程序将依次执行 15 个演示模块：

1. Smolyak 稀疏网格构造与验证
2. Karhunen-Loève 随机场展开
3. 随机椭圆方程求解
4. 统计矩计算（均值、方差、偏度、峰度）
5. 多项式混沌基转换
6. Sobol 全局灵敏度分析
7. 随机对流扩散方程
8. 随机刚性 ODE (Lindberg)
9. 随机粒子输运与 Fokker-Planck
10. 自然三次样条响应面重构
11. 随机域映射
12. 稀疏算子与协方差求解
13. 解行为分类与自适应细化
14. 多尺度耦合与周期边界
15. 完整 UQ 工作流集成（端到端）

**依赖**：仅需 `numpy` 和 `scipy`。

---

## 六、技术特色

### 6.1 博士级科学计算难度

- **Fredholm 积分方程**的特征值问题（KL 展开）
- **Smolyak 组合公式**的严格实现（含二项式系数和符号交替）
- **Golub-Welsch 算法**计算 Gauss 求积规则（Jacobi 矩阵特征值）
- **Newton 迭代**求解隐式刚性 ODE（含 Jacobian 解析计算）
- **Crank-Nicolson** 方法求解 Fokker-Planck 方程

### 6.2 边界处理与数值鲁棒性

- 扩散系数正定性保证（对数变换 + 下界截断）
- 调和平均处理系数跳跃（保证通量连续性）
- 三对角系统的 banded solver（Thomas 算法）
- CFL 条件自动检查和自适应时间步
- 特征值非负性保证
- 矩阵条件数监控

### 6.3 高工程复杂度

- 17 个 Python 模块，清晰的职责分离
- 完整的类型注解和文档字符串
- 丰富的公式注释（数学推导直接对应代码实现）
- 统一的误差估计和收敛验证框架
- 支持多种协方差核（SE、Exp、Matérn 3/2、Matérn 5/2）
- 支持多种求积规则（Gauss-Legendre、Gauss-Hermite、Gauss-Laguerre、Clenshaw-Curtis）

### 6.4 唯一性

本项目严格围绕**随机配置法 UQ**展开，具有以下独特方法论：

- Smolyak 稀疏网格 + 自适应细化的配置策略
- KL 展开 + PC 投影的随机参数化
- 多物理场耦合（椭圆 + 对流扩散 + ODE + 粒子输运）
- 基于 CVT 的后验误差驱动细化
- 多项式基转换的跨分布 UQ

---

## 七、修改文件清单

| 文件 | 功能 | 涉及种子项目 |
|:---|:---|:---|
| `polynomial_utils.py` | 正交多项式族 + Gauss 求积 + Golub-Welsch | circle_integrals |
| `sparse_grid.py` | Smolyak 稀疏网格 + 多指标枚举 + 各向异性 | matrix_assemble_parfor |
| `stochastic_field.py` | KL 展开 + 协方差矩阵 + MC/LHS 采样 | (新建) |
| `elliptic_solver.py` | 有限差分 + 调和平均 + 能量范数 | fd1d_poisson |
| `advection_solver.py` | 迎风差分 + CFL 自适应 + Péclet 分析 | fd1d_advection_ftcs |
| `moment_computation.py` | 统计矩 + 置信区间 + 超越概率 | (新建) |
| `basis_conversion.py` | PC 基转换 + 多指标集 + 数值投影 | calpak |
| `sensitivity_analysis.py` | Sobol 指数 + 有效维度 + 交互指标 | (新建) |
| `adaptive_refinement.py` | Dörfler 标记 + Lloyd CVT + 收敛估计 | cvt_1d_sampling |
| `particle_transport.py` | 随机轨迹 + Fokker-Planck + Taylor 分散 | md |
| `stiff_stochastic_ode.py` | 隐式 Euler + Newton + Lindberg 问题 | lindberg_ode |
| `response_surface.py` | 自然三次样条 + 三弯矩法 + 统计矩 | interp_ncs |
| `domain_mapper.py` | 仿射/多项式映射 + Jacobian + PDE 变换 | ToponymGeoreferencing |
| `sparse_operators.py` | CSR + 对角分离 + Block Toeplitz + 协方差采样 | r8sr, r8bto |
| `classification_surrogate.py` | 多项式分类 + 弱监督标签传播 | DeepHistoPathology, AccMLBio, BoolQ |
| `calendar_converter.py` | 多尺度耦合 + Fourier 分析 + 周期边界 | calpak |

---

## 八、解决的科学问题

本项目能够解决以下前沿科学计算问题：

1. **随机介质中的不确定性传播**：当材料属性（扩散系数、弹性模量等）具有空间随机性时，定量评估结构响应的不确定性
2. **高维参数空间的维度灾难缓解**：通过 Smolyak 稀疏网格，将配置点数从 $O(n^D)$ 降至 $O(n \log^{D-1} n)$
3. **关键输入参数识别**：通过 Sobol 灵敏度分析，识别对输出不确定性贡献最大的随机参数
4. **多尺度随机系统的高效模拟**：处理快慢时间尺度分离的随机动力学系统
5. **自适应计算资源分配**：基于后验误差估计，将计算资源集中在解变化剧烈的参数区域
6. **随机刚性化学反应动力学**：量化反应速率不确定性对化学系统演化的影响

---

## 九、数值验证

程序运行时自动进行以下验证：

- **稀疏网格精度**：$\int 1 \, d\xi = 2^D$，$\int \xi_d^2 \, d\xi = \frac{2^D}{3}$
- **质量守恒**：Lindberg 问题 $u_1 + u_2 + u_3 = 1$
- **离散残差**：椭圆方程 $||r|| < 10^{-12}$
- **Fokker-Planck 归一化**：$\int p \, dx = 1$
- **Fourier 重构精度**：$||u - u_{recon}|| < 10^{-15}$
- **收敛研究**：不同稀疏网格 level 的误差衰减
