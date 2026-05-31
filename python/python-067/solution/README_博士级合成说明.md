# 裂隙介质渗流与示踪试验数值模拟系统

## 博士级合成说明文档

---

## 一、项目概述

本项目围绕**水文地质：裂隙介质渗流与示踪试验**这一前沿科学领域，融合 15 个科研代码项目的核心算法，构建了一个面向博士级难度的综合科学计算系统。项目实现了从裂隙网络生成、水力场求解、示踪剂迁移模拟到参数反演与不确定性量化的完整科研计算流程。

---

## 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 合成后角色 | 科学映射 |
|:---:|:---|:---|:---|:---|
| 1 | 673_lights_out_game | 网格邻域状态切换 | `fracture_network.py` | 裂隙连通性的邻域耦合力学模型 |
| 2 | 578_image_double | 图像上采样插值 | `fracture_network.py` | 裂隙开度场的网格映射与细化 |
| 3 | 463_gegenbauer_rule | 高斯-盖根堡尔求积 | `flow_integrator.py` | 裂隙段抛物线速度剖面的精确积分 |
| 4 | 1061_schroedinger_nonlinear_pde | 守恒量监测 | `transport_solver.py` | 示踪剂质量守恒的严格监测 |
| 5 | 809_nonlin_regula | Regula Falsi 求根 | `inverse_model.py` | 裂隙开度/渗透率的非线性反演 |
| 6 | 082_beta_nc | 非中心 Beta 分布 | `uncertainty_quant.py` | 渗透率置信区间的概率建模 |
| 7 | 451_gauss_seidel | Gauss-Seidel 迭代 | `hydraulic_solver.py` | 裂隙网络压力场的迭代求解 |
| 8 | 227_cross_chaos | IFS 迭代函数系统 | `fracture_network.py` | 分形裂隙网络的几何生成 |
| 9 | 821_obj_display | OBJ 三维数据解析 | `geometry_parser.py` | 裂隙表面几何参数的提取 |
| 10 | 1378_usa_cvt_geo | CVT 质心 Voronoi | `mesh_generator.py` | 裂隙面内计算网格的优化采样 |
| 11 | 053_asa266 | Dirichlet MLE / Gamma | `inverse_model.py` | 裂隙方向比例的统计推断 |
| 12 | 1082_sinc | Sinc 函数插值 | `sinc_interpolator.py` | 浓度场的高精度谱插值 |
| 13 | 1338_triangulation_l2q | 线性到二次元升级 | `mesh_generator.py` | 二次三角元网格的自动生成 |
| 14 | 965_r83s | 三对角 CG/GS 求解 | `hydraulic_solver.py` | 大规模稀疏线性系统的快速求解 |
| 15 | 763_middle_square | 中平方伪随机数 | `random_generator.py` | 可复现的蒙特卡洛采样序列 |

---

## 三、核心数学物理模型与公式

### 3.1 裂隙网络水力学模型

#### 立方定律（Cubic Law）
裂隙中的层流流量服从修正的立方定律：

$$
Q = \frac{\rho g b^3}{12 \mu} W \frac{\Delta h}{L}
$$

其中 $b$ 为裂隙开度，$W$ 为裂隙宽度，$\mu$ 为动力粘度。等效渗透率为：

$$
K_{eq} = \frac{\rho g b^2}{12 \mu}
$$

#### 等效渗透率（Snow, 1969）
对于多组裂隙系统：

$$
k_{eq} = \frac{1}{12} \frac{\sum_i b_i^3 \cos^2 \theta_i}{A}
$$

#### 迂曲度
裂隙网络的迂曲度定义为实际路径长度与欧氏距离之比：

$$
\tau = \frac{L_{actual}}{L_{euclidean}} \geq 1
$$

### 3.2 稳态渗流控制方程

裂隙网络中的稳态水流满足质量守恒方程：

$$
\nabla \cdot (T \nabla h) = -q_s
$$

其中 $T$ 为水力传导系数 $[m^2/s]$，$h$ 为水头 $[m]$，$q_s$ 为源汇项。

离散化为五点差分格式：

$$
\frac{T_{i+1/2,j}(h_{i+1,j} - h_{i,j})}{\Delta x^2} - \frac{T_{i-1/2,j}(h_{i,j} - h_{i-1,j})}{\Delta x^2} + \frac{T_{i,j+1/2}(h_{i,j+1} - h_{i,j})}{\Delta y^2} - \frac{T_{i,j-1/2}(h_{i,j} - h_{i,j-1})}{\Delta y^2} = -q_{s,i,j}
$$

界面传导系数采用调和平均：

$$
T_{i+1/2,j} = \frac{2 T_{i,j} T_{i+1,j}}{T_{i,j} + T_{i+1,j}}
$$

### 3.3 对流-弥散方程（ADE）

示踪剂在裂隙介质中的迁移满足二维 ADE：

$$
R \frac{\partial C}{\partial t} + \mathbf{v} \cdot \nabla C = \nabla \cdot (\mathbf{D} \cdot \nabla C) - \lambda C + S
$$

其中：
- $C(x, y, t)$：示踪剂浓度 $[kg/m^3]$
- $R$：滞留因子 $[-]$
- $\mathbf{v} = (v_x, v_y)$：达西流速 $[m/s]$
- $\mathbf{D}$：水动力弥散张量 $[m^2/s]$
- $\lambda$：一阶衰减速率 $[1/s]$
- $S$：源汇项 $[kg/(m^3 \cdot s)]$

弥散系数：

$$
D_L = D_m + \alpha_L |\mathbf{v}|, \quad D_T = D_m + \alpha_T |\mathbf{v}|
$$

#### Ogata-Banks 解析解（一维脉冲注入）

$$
\frac{C(x,t)}{C_0} = \frac{1}{2} \left[ \text{erfc}\left(\frac{x - vt}{2\sqrt{D_L t}}\right) + \exp\left(\frac{vx}{D_L}\right) \text{erfc}\left(\frac{x + vt}{2\sqrt{D_L t}}\right) \right]
$$

### 3.4 质量守恒监测

基于非线性薛定谔方程守恒量监测思想，严格验证示踪剂质量守恒：

$$
M(t) = \int_\Omega C(\mathbf{x}, t) \, d\Omega
$$

对于保守示踪剂：

$$
M(t) = M(0) = \text{const}
$$

对于衰变示踪剂：

$$
M(t) = M(0) \exp(-\lambda t)
$$

质量守恒相对误差：

$$
\varepsilon_{mass} = \frac{|M_{numerical} - M_{theoretical}|}{|M_{theoretical}|}
$$

### 3.5 高斯-盖根堡尔数值积分

用于裂隙段内速度剖面的精确积分：

$$
\int_a^b [(x-a)(b-x)]^\alpha f(x) \, dx \approx \sum_{i=1}^n w_i f(x_i)
$$

节点和权重由 Jacobi 矩阵的特征值问题确定：

$$
J_{ij} = \begin{cases} a_i & i = j \\ b_i & j = i+1 \\ b_{i-1} & j = i-1 \\ 0 & \text{otherwise} \end{cases}
$$

Gegenbauer 递推系数：

$$
b_n^2 = \frac{n(n + 2\alpha)}{4(n + \alpha)^2 - 1}
$$

### 3.6 Sinc 谱插值

Whittaker-Shannon 采样定理：

$$
f(x) \approx \sum_{k=-\infty}^{\infty} f(kh) \, \text{sinc}\left(\frac{x - kh}{h}\right)
$$

归一化 sinc 函数：

$$
\text{sinc}(x) = \begin{cases} \dfrac{\sin(\pi x)}{\pi x} & x \neq 0 \\ 1 & x = 0 \end{cases}
$$

导数：

$$
\text{sinc}'(x) = \frac{\pi x \cos(\pi x) - \sin(\pi x)}{(\pi x)^2}
$$

### 3.7 参数反演模型

#### 裂隙开度反演（穿透时间法）

从示踪剂穿透时间反演裂隙开度：

$$
t_b = \frac{L \cdot n_e}{K_{eq} \cdot i} = \frac{12 \mu L n_e}{\rho g b^2 i}
$$

反演目标：求解 $f(b) = t_{sim}(b) - t_{obs} = 0$

使用 Regula Falsi 迭代：

$$
b_{new} = \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}
$$

#### 弥散度反演（最小二乘法）

最小化目标函数：

$$
J(D) = \sum_{i} \left[ C_{obs}(t_i) - C_{sim}(t_i; D) \right]^2
$$

#### Dirichlet 分布最大似然估计

裂隙方向比例的 Dirichlet PDF：

$$
p(\mathbf{x} | \boldsymbol{\alpha}) = \frac{\Gamma(\sum_k \alpha_k)}{\prod_k \Gamma(\alpha_k)} \prod_k x_k^{\alpha_k - 1}
$$

对数似然：

$$
\ell(\boldsymbol{\alpha}) = \ln \Gamma\left(\sum_k \alpha_k\right) - \sum_k \ln \Gamma(\alpha_k) + \sum_k (\alpha_k - 1) \overline{\ln x_k}
$$

Newton-Raphson 更新：

$$
\boldsymbol{\alpha}^{(t+1)} = \boldsymbol{\alpha}^{(t)} - \mathbf{H}^{-1} \nabla \ell
$$

### 3.8 不确定性量化

#### 非中心 Beta 分布 CDF

$$
F(x; a, b, \lambda) = \sum_{j=0}^{\infty} p_j I_x(a+j, b), \quad p_j = e^{-\lambda/2} \frac{(\lambda/2)^j}{j!}
$$

其中 $I_x$ 为不完全 Beta 函数比。

#### 对数正态渗透率置信区间

假设 $\ln K \sim \mathcal{N}(\mu, \sigma^2)$：

$$
\mu_{\ln} = \ln K_{est} - \frac{\sigma_{\ln}^2}{2}, \quad \sigma_{\ln}^2 = \ln\left(1 + \frac{\sigma_K^2}{K_{est}^2}\right)
$$

$95\%$ 置信区间：

$$
K_{low} = \exp(\mu_{\ln} + z_{0.025} \sigma_{\ln}), \quad K_{high} = \exp(\mu_{\ln} + z_{0.975} \sigma_{\ln})
$$

#### 一阶可靠性方法（FORM）

失效概率：

$$
P_f \approx \Phi(-\beta)
$$

其中 $\beta = \|\mathbf{u}^*\|$ 为可靠性指标，$\mathbf{u}^*$ 为标准正态空间中的设计点。

### 3.9 双孔隙介质模型

简化的单速率质量交换模型：

$$
\phi_m \frac{\partial C_m}{\partial t} + \phi_{im} \frac{\partial C_{im}}{\partial t} + v \phi_m \frac{\partial C_m}{\partial x} = 0
$$

$$
\phi_{im} \frac{\partial C_{im}}{\partial t} = \alpha (C_m - C_{im})
$$

### 3.10 突破曲线统计矩

零阶矩（总质量）：

$$
M_0 = \int_0^{\infty} C(t) \, dt
$$

平均突破时间：

$$
\bar{t} = \frac{M_1}{M_0} = \frac{\int_0^{\infty} t C(t) \, dt}{\int_0^{\infty} C(t) \, dt}
$$

时间方差：

$$
\sigma_t^2 = \frac{M_2}{M_0} - \bar{t}^2
$$

纵向弥散度估算：

$$
\alpha_L \approx \frac{\sigma_t^2 v^3}{2L}
$$

### 3.11 无量纲数

Peclet 数：

$$
Pe = \frac{vL}{D}
$$

Reynolds 数（裂隙流）：

$$
Re = \frac{\rho v b}{\mu}
$$

---

## 四、项目文件结构

```
067_synth_project/
├── main.py                      # 统一入口，零参数可运行
├── random_generator.py          # 中平方伪随机数生成器
├── fracture_network.py          # 分形裂隙网络生成与拓扑分析
├── geometry_parser.py           # 三维裂隙几何 OBJ 解析
├── mesh_generator.py            # CVT 网格生成与二次元升级
├── sinc_interpolator.py         # Sinc 谱插值与导数计算
├── hydraulic_solver.py          # 水力压力场求解（GS/CG）
├── flow_integrator.py           # 盖根堡尔数值积分与通量分析
├── transport_solver.py          # 示踪剂 ADE 求解与质量守恒监测
├── inverse_model.py             # 渗透率反演与 Dirichlet MLE
├── uncertainty_quant.py         # 不确定性量化与可靠性分析
└── README_博士级合成说明.md      # 本文档
```

---

## 五、合成后的项目能够解决什么科学问题

1. **裂隙网络几何建模**：使用 IFS 分形算法生成具有自相似特征的裂隙网络，并通过网格状态切换模型模拟应力-裂隙耦合效应。

2. **裂隙介质渗流模拟**：基于立方定律和达西定律，求解裂隙网络中的稳态水头分布和流速场，支持 Gauss-Seidel 和共轭梯度两种高效求解器。

3. **示踪试验数值模拟**：求解二维对流-弥散方程，模拟示踪剂在裂隙介质中的迁移过程，严格监测质量守恒，计算突破曲线。

4. **参数反演**：从示踪剂穿透时间反演裂隙开度和等效渗透率；从突破曲线反演纵向弥散度；使用 Dirichlet MLE 估计裂隙方向比例。

5. **不确定性量化**：使用非中心 Beta 分布、Gamma 分布建模参数不确定性；通过蒙特卡洛方法传播不确定性；使用一阶可靠性方法评估失效概率。

6. **计算网格生成**：使用 CVT 优化和 Delaunay 三角剖分生成高质量计算网格，支持线性到二次元的自动升级。

---

## 六、如何运行

### 环境要求
- Python 3.8+
- NumPy
- SciPy

### 安装依赖
```bash
pip install numpy scipy
```

### 运行模拟
```bash
cd 067_synth_project
python main.py
```

程序将自动执行以下 9 个步骤：
1. 分形裂隙网络生成
2. 三维裂隙几何解析
3. CVT 计算网格生成
4. Sinc 谱插值验证
5. 水力压力场求解
6. 高斯-盖根堡尔流量数值积分
7. 示踪剂对流-弥散迁移模拟
8. 渗透率参数反演
9. 不确定性量化

无需输入任何参数，运行完成后会在终端输出所有计算结果。

---

## 七、数值鲁棒性与边界处理

本项目在以下方面实现了严格的数值鲁棒性：

1. **零值/负值检查**：所有物理参数（开度、渗透率、粘度等）均设有正数约束，违反时抛出 ValueError。

2. **最小开度约束**：裂隙开度下限设为 $10^{-6}$ m，避免立方定律中的奇异性。

3. **传导系数下限**：水力传导系数 clip 到 $10^{-12}$ m²/s，防止线性系统奇异。

4. **CFL 条件监测**：显式 ADE 求解前自动检查 CFL 数和弥散数，超限则自适应调整时间步长。

5. **迭代收敛控制**：Gauss-Seidel 和 CG 均设有最大迭代次数和残差容差，防止无限循环。

6. **质量守恒验证**：示踪剂迁移求解后自动计算质量守恒相对误差，确保数值解的物理合理性。

7. **边界条件一致性**：Dirichlet 和 Neumann 边界条件在离散格式中严格保持相容性。

8. **概率分布参数约束**：Dirichlet 参数、Gamma 参数等均进行正数检查；Beta 分布的自变量限制在 $[0, 1]$ 区间内。

---

## 八、科学计算复杂度说明

本项目涉及的计算复杂度包括：

- **线性代数**：稀疏矩阵的 Gauss-Seidel 迭代（$O(N_{iter} \cdot N_{grid})$）与共轭梯度法（$O(N_{grid}^{1.5})$ 对于二维问题）
- **数值积分**：高斯-盖根堡尔求积的 Jacobi 矩阵特征值问题（$O(n^3)$，$n$ 为求积阶数）
- **优化问题**：CVT Lloyd 松弛（$O(N_{iter} \cdot N_{samples} \cdot N_{generators})$）
- **参数估计**：Dirichlet MLE 的 Newton-Raphson 迭代（$O(N_{iter} \cdot k^3)$，$k$ 为维度）
- **蒙特卡洛**：不确定性传播的随机采样（$O(N_{samples} \cdot C_{forward})$）
- **谱方法**：Sinc 插值的无限求和截断（$O(N_{query} \cdot N_{trunc})$）

---

*文档生成日期：2026-05-04*
