# 博士级合成项目说明 — PROJECT 220

## 分布式 ADMM 多物理场 PDE 约束优化框架

### 科学领域
**数学优化：分布式优化与 ADMM 方法**

---

## 一、项目概述

本项目将 15 个科研种子项目的核心算法融合为一个面向**前沿科学计算**的博士级框架，聚焦于**分布式 ADMM (Alternating Direction Method of Multipliers)** 方法在多物理场 PDE 约束优化中的应用。

### 核心科学问题

考虑一般形式的分布式 PDE 约束优化问题：

$$\min_{\{u_i, \theta_i\}_{i=1}^{N}} \sum_{i=1}^{N} \left[ \frac{1}{2}\|u_i - u_i^{\text{obs}}\|_{Q_i}^2 + \frac{\alpha}{2}\|\theta_i\|^2 \right]$$

约束条件：
- PDE 约束：$F_i(u_i, \theta_i) = 0$ on $\Omega_i$（每个子域上的反应扩散/ODE 系统）
- 接口耦合：$u_i = u_j$ on $\Gamma_{ij}$（子域间一致性）
- 边界条件：$u_i|_{\partial\Omega_i \setminus \Gamma} = g_i$

### 算法框架

采用 Consensus ADMM 分解为分布式子问题：

$$x_i^{k+1} = \arg\min_{x_i} \left[ f_i(x_i) + \frac{\rho}{2}\|x_i - z^k + u_i^k\|^2 \right]$$

$$z^{k+1} = \frac{1}{N} \sum_{i=1}^{N} x_i^{k+1}$$

$$u_i^{k+1} = u_i^k + x_i^{k+1} - z^{k+1}$$

---

## 二、输入项目到科学问题的映射

| # | 种子项目 | 原功能 | 在合成项目中的角色 | 映射目标文件 |
|---|---------|--------|------------------|-------------|
| 1 | 513_hello | Hello World (初始化) | 项目配置与初始化框架 | `config.py` |
| 2 | 1343_triangulation_order6_contour | 6节点三角形可视化 | 6节点三角形网格拓扑操作 | `mesh.py` |
| 3 | 396_fem1d_pmethod | p-version 1D FEM | p-version 高阶有限元基函数 | `fem_operators.py` |
| 4 | 874_ply_to_tri_surface | PLY格式三角面片转换 | 扇形三角化与表面网格操作 | `mesh.py` |
| 5 | 1255_hyper-mpc_hypermpc_code | 超参数预测MPC | B样条自适应罚参数调度 | `adaptive_penalty.py` |
| 6 | 1259_EmperorAiphaton_MetricDistortion | LP度量失真 | 子域间一致性度量失真诊断 | `convergence.py`, `inverse_problem.py` |
| 7 | 896_polynomial_resultant | Sylvester结式 | 多项式基与 resultant 理论 | `quadrature.py` (基函数构造) |
| 8 | 931_pyramid_felippa_rule | 金字塔Felippa积分 | 高阶数值积分规则 | `quadrature.py` |
| 9 | 452_gauss_seidel_poisson_1d | Gauss-Seidel Poisson求解 | 迭代线性求解器 | `linear_solvers.py` |
| 10 | 377_fem_neumann | Neumann FEM 反应扩散 | Neumann边界FEM组装 | `fem_operators.py` |
| 11 | 060_axon_ode | Hodgkin-Huxley轴突模型 | 神经动力学子问题 | `hh_ode.py` |
| 12 | 842_ozone2_ode | 大气臭氧化学ODE | 化学反应动力学子问题 | `ozone_ode.py` |
| 13 | 127_burgers_time_viscous | 粘性Burgers方程 | 流体力学子问题 | `burgers.py` |
| 14 | 487_gray_scott_pde | Gray-Scott反应扩散 | 图案形成子问题 | `gray_scott.py` |
| 15 | 797_nelder_mead | Nelder-Mead优化 | 无导数局部子问题求解 | `nelder_mead.py` |

---

## 三、核心数学公式

### 3.1 ADMM 增广 Lagrangian

$$\mathcal{L}_\rho(x, z, \lambda) = \sum_{i=1}^{N} f_i(x_i) + \frac{\rho}{2} \sum_{i=1}^{N} \|x_i - z + u_i\|^2$$

其中 $u_i = \lambda_i / \rho$ 为 scaled dual variable。

### 3.2 收敛条件 (Boyd et al. 2011)

原始残差: $\|r^k\|_2 \leq \varepsilon^{\text{pri}}$

对偶残差: $\|s^k\|_2 \leq \varepsilon^{\text{dual}}$

其中：
$$\varepsilon^{\text{pri}} = \sqrt{nN} \varepsilon^{\text{abs}} + \varepsilon^{\text{rel}} \max\{\|Ax^k\|, \|Bz^k\|, \|c\|\}$$
$$\varepsilon^{\text{dual}} = \sqrt{n} \varepsilon^{\text{abs}} + \varepsilon^{\text{rel}} \|\rho A^T \lambda^k\|$$

### 3.3 自适应罚参数 (Residual Balancing)

$$\rho^{k+1} = \begin{cases} \tau \cdot \rho^k & \text{if } \|r^k\| > \mu \|s^k\| \\ \rho^k / \tau & \text{if } \|s^k\| > \mu \|r^k\| \\ \rho^k & \text{otherwise} \end{cases}$$

### 3.4 Gray-Scott 反应扩散

$$\frac{\partial U}{\partial t} = D_u \nabla^2 U - UV^2 + \gamma(1 - U)$$
$$\frac{\partial V}{\partial t} = D_v \nabla^2 V + UV^2 - (\gamma + \kappa)V$$

9 点 Laplacian:
$$\nabla^2 u_{i,j} \approx \frac{1}{6h^2} \begin{pmatrix} 1 & 4 & 1 \\ 4 & -20 & 4 \\ 1 & 4 & 1 \end{pmatrix} * u$$

### 3.5 Hodgkin-Huxley 模型

$$C_m \frac{dV}{dt} = I_{\text{ext}} - g_{\text{Na}} m^3 h (V - E_{\text{Na}}) - g_K n^4 (V - E_K) - g_L (V - E_L)$$

门控动力学:
$$\frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V) x, \quad x \in \{n, m, h\}$$

### 3.6 Burgers 方程

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

Hopf-Cole 变换: $u = -2\nu \frac{\partial}{\partial x} \ln \varphi$, 其中 $\varphi_t = \nu \varphi_{xx}$

### 3.7 p-version 有限元

基函数: $\phi_i(x) = (1 - x^2) q_i(x)$

三项递推: $q_i(x) = (x - \alpha_i) q_{i-1}(x) - \beta_i q_{i-2}(x)$

能量内积: $\langle u, v \rangle_E = \int_{-1}^{1} [P u'v' + Q uv] dx$

### 3.8 L-BFGS 两段递推

$$H_k g_k = \text{TwoLoopRecursion}(g_k, \{s_i, y_i\}_{i=k-m}^{k-1})$$

---

## 四、项目文件结构

```
220_synth_project_Advanced/
├── main.py                  # 统一入口 (零参数运行)
├── config.py                # 全局配置与物理参数
├── mesh.py                  # 6节点三角形网格生成
├── domain_decomp.py         # 区域分解与接口管理
├── quadrature.py            # 高阶积分规则 (Gauss-Legendre, Felippa)
├── fem_operators.py         # p-version FEM 与 Neumann FEM
├── linear_solvers.py        # Gauss-Seidel, SOR, CG, PCG
├── gray_scott.py            # Gray-Scott 反应扩散
├── hh_ode.py                # Hodgkin-Huxley 神经动力学
├── ozone_ode.py             # 大气臭氧化学
├── burgers.py               # 粘性 Burgers 方程
├── nelder_mead.py           # Nelder-Mead 无导数优化
├── inverse_problem.py       # PDE 约束反问题 (L-BFGS)
├── admm_core.py             # 分布式 Consensus ADMM
├── adaptive_penalty.py      # 自适应罚参数策略
├── convergence.py           # 收敛性分析与诊断
└── README_博士级合成说明.md   # 本文档
```

**共计 16 个 .py 文件 + 1 个 README**

---

## 五、运行方法

```bash
cd 220_synth_project_Advanced
python main.py
```

零参数运行，自动执行 6 个实验并输出结果。

---

## 六、实验说明

### 实验 1: 分布式 Poisson 反问题
- 生成结构化 6 节点三角形网格
- 递归坐标二分区域分解 (4 子域)
- Gauss-Seidel 求解 1D Poisson 基准
- FEM 矩阵组装 (Neumann BC)
- Consensus ADMM 分布式优化

### 实验 2: Gray-Scott 反应扩散
- Turing 模式形成模拟
- 9 点 Laplacian 模板验证
- 质量守恒监测

### 实验 3: Hodgkin-Huxley 神经动力学
- 动作电位模拟 (RK4)
- 稳态门控变量分析
- I-V 曲线计算

### 实验 4: 臭氧化学 + Burgers 方程
- 24 小时大气化学模拟
- 光稳态分析
- 粘性 Burgers 激波衰减
- 能量耗散验证

### 实验 5: 多物理场 ADMM 数据融合
- Nelder-Mead 求解器验证
- 数值积分精度验证
- 4 物理场分布式 ADMM 融合
- 收敛诊断与度量失真分析

### 实验 6: p-version FEM 与高阶积分
- 基函数正交性验证
- 刚度矩阵组装
- Gauss-Legendre 精度测试
- 三角形/金字塔积分

---

## 七、创新点与独特性

1. **分布式 ADMM + 多物理场**: 首次将 ADMM 区域分解框架应用于包含反应扩散、神经动力学、大气化学、流体力学的耦合系统
2. **自适应罚参数**: 融合残差均衡、谱估计和 B 样条参数化三种策略
3. **度量失真诊断**: 借鉴计算社会选择的度量失真概念监测子域一致性
4. **p-version FEM**: 高阶基函数自动满足 Dirichlet 边界条件
5. **无导数接口求解**: 对非光滑接口问题使用 Nelder-Mead 替代梯度法

---

## 八、数值鲁棒性

- 所有除法操作均有 `EPS_NUM = 1e-14` 保护
- 物理变量 (浓度、门控变量) 有上下界截断
- CFL 条件自动计算并限制时间步长
- 矩阵求解前检查对角元素非零
- 自适应罚参数有上下界限制

---

## 九、依赖

仅需标准科学计算库:
- `numpy` (核心数值计算)

无需其他第三方依赖，无需可视化库。
