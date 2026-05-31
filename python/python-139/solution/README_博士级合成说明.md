# 博士级合成项目说明：膜分离过程多尺度传质模型

## 1. 项目概述

本项目围绕**化学工程：膜分离过程传质模型**展开，将 15 个原始科研代码项目的核心算法融合为一个面向前沿科学问题的博士级 Python 计算项目。项目模拟了中空纤维膜模块中 CO₂/CH₄ 气体分离的多尺度传质过程，涵盖分子尺度孔道扩散、膜层有限元（FEM）离散、表面反应动力学、非线性渗透率方程求解、多级膜联网络优化及周期性维护调度。

---

## 2. 科学问题与核心公式

### 2.1 传质控制方程

在膜活性层内，组分 \(i\) 的浓度分布满足扩散-反应方程：

$$
\frac{\partial c_i}{\partial t} = D_i \frac{\partial^2 c_i}{\partial x^2} - k \, c_i
$$

稳态形式为：

$$
-D_i \frac{d^2 c_i}{dx^2} + k \, c_i = 0
$$

边界条件采用 Dirichlet 型：

$$
c(0) = c_{\text{feed}}, \quad c(L) = c_{\text{perm}}
$$

### 2.2 渗透率与溶解-扩散模型

组分渗透率为扩散系数与溶解度系数的乘积：

$$
P_i = D_i \, S_i \quad [\text{mol}\cdot\text{m}/(\text{m}^2\cdot\text{s}\cdot\text{Pa})]
$$

跨膜摩尔通量由溶解-扩散模型给出：

$$
J_i = \frac{P_i}{L} \bigl(p_{\text{feed},i} - p_{\text{perm},i}\bigr)
$$

### 2.3 无量纲数

**Damköhler 数**（反应与扩散速率之比）：

$$
\text{Da} = \frac{k L^2}{D_i}
$$

**Thiele 模数**：

$$
\phi = L \sqrt{\frac{k}{D_i}}
$$

### 2.4 表面反应动力学（Langmuir-Hinshelwood）

膜表面吸附覆盖度：

$$
\theta_i = \frac{K_i c_i}{1 + \sum_j K_j c_j}
$$

表面反应速率：

$$
r_i = k \, \theta_i
$$

### 2.5 Knudsen 扩散与有效扩散系数

孔道内 Knudsen 扩散系数：

$$
D_{K,i} = \frac{2}{3} r_{\text{pore}} \sqrt{\frac{8 R T}{\pi M_i}}
$$

多孔支撑层有效扩散系数：

$$
D_{\text{eff},i} = \frac{\varepsilon}{\tau} D_{K,i}
$$

### 2.6 温度依赖溶解度（van't Hoff）

$$
S(T) = S_0 \exp\!\left[-\frac{\Delta H_{\text{sorp}}}{R}\left(\frac{1}{T} - \frac{1}{T_0}\right)\right]
$$

### 2.7 非线性渗透率代数系统

对于理想混合渗透侧，渗透分压满足：

$$
\frac{p_{\text{perm,CO}_2}}{p_{\text{perm,CH}_4}} = \frac{P_{\text{CO}_2}}{P_{\text{CH}_4}} \cdot \frac{p_{\text{feed,CO}_2} - p_{\text{perm,CO}_2}}{p_{\text{feed,CH}_4} - p_{\text{perm,CH}_4}}
$$

结合总压约束 \(p_{\text{perm,CO}_2} + p_{\text{perm,CH}_4} = p_{\text{perm}}\)，构成标量非线性方程，采用二分法 + Broyden 方法求解。

### 2.8 多级膜联 PageRank 流分布

将膜级联视为有向图，邻接矩阵 \(A\) 转化为 Google 矩阵：

$$
G = (1-d) S + \frac{d}{n} \mathbf{1}\mathbf{1}^T
$$

其中 \(S\) 为列随机转移矩阵，\(d=0.15\) 为阻尼因子。PageRank 向量 \(x\) 通过幂法迭代求得：

$$
x^{(k+1)} = G \, x^{(k)}
$$

### 2.9 准周期振荡 forcing ODE

模拟进料浓度周期性扰动：

$$
\frac{d^4 u}{dt^4} + (\pi^2 + 1) \frac{d^2 u}{dt^2} + \pi^2 u = 0
$$

守恒量：

$$
I = \ddot{u}^2 + (\pi^2+1) \dot{u}^2 + \pi^2 u^2 + 2 u \ddot{u}
$$

### 2.10 类 Kepler 分子轨迹

将孔道内分子运动建模为中心力场中的保守系统：

$$
\dot{q}_1 = p_1, \quad \dot{q}_2 = p_2, \quad \dot{p}_1 = -\mu \frac{q_1}{r^3}, \quad \dot{p}_2 = -\mu \frac{q_2}{r^3}
$$

Hamiltonian：

$$
H = \frac{1}{2}(p_1^2 + p_2^2) - \frac{\mu}{\sqrt{q_1^2 + q_2^2}}
$$

---

## 3. 原始项目到科学问题的映射

| 原始项目 | 核心算法 | 在本项目中的角色 |
|---|---|---|
| **844_pagerank** | 幂法求主特征向量、Google 矩阵 | 多级膜联流分布排序（`cascade_network.py`） |
| **457_ge_to_ccs** | 稠密矩阵转 CCS 稀疏格式 | FEM 刚度矩阵稀疏化（`sparse_matrix.py`） |
| **1423_xyz_display** | 3D 点坐标读取与处理 | 孔道网络 3D 坐标生成（`utils.py`） |
| **894_polynomial_conversion** | Bernstein/Legendre/Chebyshev 基转换 | 浓度分布谱方法插值（`polynomial_spectral.py`） |
| **1017_reaction_ode** | A+B→C 反应 ODE | 膜表面 Langmuir-Hinshelwood 反应动力学（`mass_transfer_ode.py`） |
| **1051_runge** | Runge 函数导数与幂级数 | 自适应网格监控函数与插值测试（`mass_transfer_ode.py`, `pore_dynamics.py`） |
| **959_quasiperiodic_ode** | 四阶准周期 ODE | 进料浓度周期性扰动模型（`mass_transfer_ode.py`, `time_integrator.py`） |
| **890_polygon_triangulate** | 耳切法多边形三角化 | 中空纤维膜截面剖分与面积通量积分（`membrane_geometry.py`） |
| **398_fem1d_sample** | 1D FEM 节点搜索与线性插值 | 膜厚度方向扩散-反应 FEM 求解（`membrane_fem.py`） |
| **311_doomsday** | 模运算与日历算法 | 膜模块循环再生调度（`utils.py`, `cascade_network.py`） |
| **618_kepler_ode** | Kepler 二体守恒系统 | 孔道内分子保守轨迹模拟（`mass_transfer_ode.py`, `time_integrator.py`） |
| **976_r8ci** | 循环矩阵-向量乘法 | 周期边界条件 FEM 刚度矩阵构造（`sparse_matrix.py`） |
| **120_broyden** | Broyden 拟牛顿法 | 非线性渗透率方程求解（`nonlinear_solver.py`） |
| **064_backward_euler_fixed** | 定点迭代后向 Euler | 瞬态扩散-反应时间积分（`time_integrator.py`） |
| **1178_subset_sum** | 动态规划子集和 | 最优膜模块产能组合（`cascade_network.py`） |

---

## 4. 文件结构

```
139_synth_project/
├── main.py                     # 统一入口，零参数可运行
├── parameters.py               # 物理参数、热力学常数、验证与无量纲数
├── utils.py                    # 数值安全函数、模运算、插值、孔坐标生成
├── sparse_matrix.py            # 稀疏矩阵格式转换、循环矩阵、FEM 刚度矩阵
├── polynomial_spectral.py      # 正交多项式基转换与谱插值
├── membrane_geometry.py        # 耳切法三角化、面积通量积分
├── membrane_fem.py             # 1D FEM 稳态/瞬态扩散-反应求解
├── mass_transfer_ode.py        # 反应 ODE、Kepler 轨迹、准周期 forcing、Runge 测试
├── time_integrator.py          # Backward Euler、RK4、自适应 RK45、守恒量监测
├── nonlinear_solver.py         # Broyden 方法、牛顿法、渗透率非线性系统
├── pore_dynamics.py            # Knudsen 扩散、孔道网络 Monte-Carlo、自适应网格
└── cascade_network.py          # PageRank 流分布、子集和优化、级联物料衡算
```

---

## 5. 运行方式

在项目根目录下直接执行：

```bash
python main.py
```

程序将依次执行以下 12 个科研计算模块并输出结果：

1. 参数初始化与验证
2. 无量纲数分析（Damköhler、Thiele）
3. 稀疏矩阵与循环矩阵运算
4. 正交多项式谱方法
5. 膜几何剖分与面积通量积分
6. 稳态 FEM 扩散-反应求解
7. 瞬态 FEM 扩散-反应求解（Backward Euler）
8. 耦合质量传递 ODE 积分（反应、准周期、类 Kepler）
9. 非线性渗透率方程求解（Broyden 方法）
10. 孔道网络传质模型
11. 多级膜联优化与网络流分析
12. Runge 函数自适应网格生成

---

## 6. 关键边界处理与数值鲁棒性

- **参数验证**：`validate_parameters` 检查所有物理量的正定性、区间约束及压力大小关系。
- **安全平方根 / 除法**：`safe_sqrt`、`safe_divide` 防止浮点下溢与除零。
- **Broyden 重启机制**：迭代维度耗尽后自动重置搜索方向，避免 stagnation。
- **自适应步长 RK45**：局部误差估计与步长减半/倍增策略，防止积分步长下溢。
- **多边形三角化鲁棒性**：角度容差 5.7e-05°、共线性检测与面积正定性校验。
- **非线性方程标量二分 fallback**：Broyden 系统若出现 bracket 失败，自动回退到解析近似。

---

## 7. 合成难度说明

本项目将 15 个独立科研代码项目的算法思想深度整合到化学工程膜分离领域，实现了：

- **多尺度耦合**：从分子尺度（Kepler 轨迹、Knudsen 扩散）到介观尺度（FEM、谱方法）再到宏观尺度（级联网络、PageRank 优化）。
- **高阶数值方法**：自适应 RK45、Broyden 拟牛顿、稀疏矩阵直接求解、正交多项式谱插值。
- **大量科学公式**：溶解-扩散模型、Langmuir-Hinshelwood 动力学、van't Hoff 方程、Damköhler/Thiele 无量纲分析、守恒 Hamiltonian 系统、Google 矩阵特征向量等。
- **工程鲁棒性**：边界校验、数值安全包装、重启策略、自动 fallback。

---

## 8. 版本与依赖

- Python >= 3.9
- NumPy >= 1.21
- SciPy >= 1.7

---

*本项目为 PROJECT_139 的 Python 博士级科研代码合成结果，遵循 sci-project-synthesis-python 工作流规范。*
