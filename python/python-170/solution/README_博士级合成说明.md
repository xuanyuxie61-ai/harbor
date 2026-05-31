# README：博士级合成说明

## 项目概述

**项目名称**：集群机器人涌现行为的多尺度建模与仿真  
**科学领域**：机器人学 — 集群机器人涌现行为（Swarm Robotics Emergence）  
**合成语言**：Python 3  
**合成目录**：`Synthesis-project-python/170_synth_project`

本项目将 15 个独立的科研代码项目的核心算法融合为一个面向前沿科学问题的博士级计算框架，用于研究**大规模集群机器人在复杂环境中的自组织涌现行为**。项目涵盖从微观个体动力学（混沌ODE、随机控制）到宏观连续介质建模（对流-扩散PDE、谱方法）的多尺度耦合，具备完整的数值鲁棒性处理和边界条件判定。

---

## 一、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 在合成项目中的科学角色 |
|------|--------|----------|------------------------|
| 1 | `1238_tet_mesh_refine` | 四面体网格8倍细化 | `spatial_mesh.py`：三维工作空间的自适应离散化，支持碰撞检测与环境场插值 |
| 2 | `422_feynman_kac_1d` | Feynman-Kac蒙特卡洛路径积分 | `stochastic_control.py`：机器人在随机势场中的碰撞规避概率势计算 |
| 3 | `123_burgers_pde_etdrk4` | Burgers方程ETDRK4谱方法 | `density_field.py`：集群宏观密度场的对流-扩散-超粘性演化 |
| 4 | `075_bdf3` | BDF3隐式ODE求解器 | `swarm_dynamics.py`：Arneodo混沌子系统的高精度刚性积分 |
| 5 | `380_fem_to_tec` | FEM有限元数据读写 | `environment_field.py`：环境标量场（温度/化学浓度）的有限元插值与梯度计算 |
| 6 | `581_image_noise` | 图像噪声注入 | `sensor_noise.py`：机器人传感器读数的盐椒噪声、均匀噪声与高斯白噪声复合模型 |
| 7 | `1259_theta_method` | Theta方法（Crank-Nicolson） | `swarm_dynamics.py`：耦合PDE-ODE系统的隐式-显式混合时间积分 |
| 8 | `159_chebyshev` | Chebyshev插值与Clenshaw递推 | `spectral_approx.py`：通信时延核的Chebyshev谱参数化与快速求值 |
| 9 | `077_bernstein_approximation` | Bernstein多项式逼近 | `spectral_approx.py`：局部控制策略的Bernstein基展开，保证正性与单位分解 |
| 10 | `019_arneodo_ode` | Arneodo混沌ODE系统 | `swarm_dynamics.py`：机器人内部神经元状态的混沌动力学模型 |
| 11 | `178_circle_distance` | 圆上随机点距离统计 | `distance_statistics.py`：圆形巡逻区域中机器人间距分布的涌现度量 |
| 12 | `747_medit_mesh_io` | MEDIT网格文件I/O | `spatial_mesh.py`：复杂地形网格的读取与解析 |
| 13 | `819_normal01_multivariate_distance` | 多元正态距离统计 | `distance_statistics.py`：高维状态空间嵌入的距离统计分析 |
| 14 | `1111_sparse_parfor` | 稀疏矩阵分块并行组装 | `interaction_matrix.py`：大规模集群通信拓扑的稀疏Laplacian矩阵构建 |
| 15 | `253_cvt_circle_nonuniform` | 非均匀圆域CVT计算 | `coverage_optimization.py`：集群最优覆盖部署的Lloyd迭代算法 |

---

## 二、新增数学物理模型与核心公式

### 2.1 个体机器人混合动力学

每个机器人 $i$ 的状态向量为

$$
\mathbf{z}_i = [\mathbf{p}_i^\top, \mathbf{v}_i^\top, \mathbf{s}_i^\top]^\top \in \mathbb{R}^9
$$

其中 $\mathbf{p}_i \in \mathbb{R}^3$ 为位置，$\mathbf{v}_i \in \mathbb{R}^3$ 为速度，$\mathbf{s}_i \in \mathbb{R}^3$ 为内部混沌状态。

**机械子系统**：

$$
\frac{d\mathbf{p}_i}{dt} = \mathbf{v}_i
$$

$$
\frac{d\mathbf{v}_i}{dt} = \mathbf{u}_i + \mathbf{F}_{\text{env}}(\mathbf{p}_i) + \sum_{j \neq i} \mathbf{F}_{\text{rep}}(\mathbf{p}_i, \mathbf{p}_j) - \gamma \mathbf{v}_i
$$

控制输入采用**Bernstein参数化共识控制**：

$$
\mathbf{u}_i = k_p (\bar{c} - \bar{p}_i) \mathbf{1} - k_v \mathbf{v}_i + k_e \frac{\nabla \phi(\mathbf{p}_i)}{\|\nabla \phi(\mathbf{p}_i)\| + \varepsilon}
$$

其中 $\bar{c}$ 为噪声污染后的传感器共识目标，$\phi$ 为环境势场。

**内部混沌子系统（Arneodo吸引子）**：

$$
\begin{aligned}
\dot{s}_1 &= s_2 \\
\dot{s}_2 &= s_3 \\
\dot{s}_3 &= -\alpha s_1 - \beta s_2 - s_3 + \delta s_1^3
\end{aligned}
$$

参数取 $\alpha = -5.5, \beta = 3.5, \delta = -1.0$，系统具有**奇异吸引子**与**拓扑混沌**。

**Lennard-Jones型排斥力**：

$$
\mathbf{F}_{\text{rep}}(\mathbf{p}_i, \mathbf{p}_j) = \sigma \left( \frac{1}{r^2 + \varepsilon} - \frac{1}{R_{\text{rep}}^2} \right)_+ \frac{\mathbf{p}_i - \mathbf{p}_j}{r}
$$

其中 $r = \|\mathbf{p}_i - \mathbf{p}_j\|$，$(\cdot)_+$ 为正部算子。

### 2.2 宏观密度连续介质方程

从介观视角，集群密度 $\rho(\mathbf{x}, t)$ 满足**带超粘性的对流-扩散方程**：

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = \nu \Delta \rho - D_4 \Delta^2 \rho + S(\mathbf{x}, t)
$$

在一维周期原型域 $[-\pi, \pi]$ 上，使用**Fourier-Galerkin谱离散化**与**指数时间差分Runge-Kutta 4阶方法（ETDRK4）**求解。ETDRK4将刚性线性部分通过矩阵指数精确积分，非线性对流项通过围道积分计算 $\phi$ 函数：

$$
\mathbf{v}_{n+1} = E \mathbf{v}_n + N_n \phi_1 + 2(N_a + N_b)\phi_2 + N_c \phi_3
$$

其中 $E = e^{\Delta t L}$，$L$ 为Fourier空间中的扩散算子特征值，中间阶段通过复平面上64点围道积分稳定求值。

### 2.3 Feynman-Kac碰撞规避势

机器人在势场 $V(\mathbf{x})$ 中的安全概率由**Feynman-Kac公式**给出：

$$
u(\mathbf{x}) = \mathbb{E}\left[ \exp\left( -\int_0^{\tau} V(\mathbf{X}_s) \, ds \right) \right]
$$

其中 $\mathbf{X}_s$ 为从 $\mathbf{x}$ 出发的布朗运动，$\tau$ 为首次退出工作域的时刻。蒙特卡洛估计器采用二阶弱精确格式：

$$
\mathbf{X}_{k+1} = \mathbf{X}_k + \sqrt{h} \, \mathbf{Z}_k, \quad \mathbf{Z}_k \sim \mathcal{N}(0, I)
$$

$$
Y_{k+1} = Y_k - \frac{h}{2} \left[ V(\mathbf{X}_{k+1}) Y_e + V(\mathbf{X}_k) Y_k \right], \quad Y_e = (1 - h V(\mathbf{X}_k)) Y_k
$$

### 2.4 图Laplacian与共识动力学

机器人通信拓扑由几何近邻图定义：

$$
W_{ij} = w(\|\mathbf{p}_i - \mathbf{p}_j\|) \cdot \mathbb{1}_{\{\|\mathbf{p}_i - \mathbf{p}_j\| \le R_s\}}
$$

权重函数取 $w(d) = \max(0, 1 - d/R_s)^2$。图Laplacian为 $\mathbf{L} = \mathbf{D} - \mathbf{W}$，其**Fiedler值**（第二小特征值）$\lambda_2$ 决定线性共识协议的指数收敛率：

$$
\|\mathbf{x}(t) - \bar{x}\mathbf{1}\| \le C e^{-\lambda_2 t}
$$

### 2.5 CVT最优覆盖

集群的最优空间部署最小化**CVT能量泛函**：

$$
E(\{\mathbf{p}_i\}) = \sum_{i=1}^{N} \int_{V_i} \rho(\mathbf{x}) \|\mathbf{x} - \mathbf{p}_i\|^2 \, d\mathbf{x}
$$

其中 $V_i$ 为第 $i$ 个机器人的Voronoi cell，$\rho(\mathbf{x})$ 为任务优先级密度。通过**Lloyd算法**迭代将生成器移至质心：

$$
\mathbf{p}_i^{(k+1)} = \frac{\int_{V_i} \mathbf{x} \rho(\mathbf{x}) \, d\mathbf{x}}{\int_{V_i} \rho(\mathbf{x}) \, d\mathbf{x}}
$$

### 2.6 涌现度量：KL散度

将机器人对间距的经验直方图 $P_{\text{emp}}$ 与均匀随机分布的理论PDF $Q_{\text{unif}}$ 比较：

$$
D_{\text{KL}}(P \| Q) = \sum_k P_k \log \frac{P_k}{Q_k}
$$

对于圆上均匀随机点，弦长的理论PDF为：

$$
f(d) = \frac{1}{\pi} \frac{1}{\sqrt{1 - 0.25 d^2/R^2}}, \quad 0 \le d \le 2R
$$

**高KL散度意味着机器人间距分布显著偏离随机均匀分布，即空间结构已涌现。**

### 2.7 传感器噪声模型

复合传感器噪声包含三个独立机制：

1. **盐椒噪声**（突发故障）：概率 $p_{sp}$ 下读数置为极值 $0$ 或 $1$
2. **均匀噪声**（量化误差）：概率 $p_u$ 下读数替换为 $\mathcal{U}(0,1)$
3. **高斯白噪声**（热噪声）：$\eta \sim \mathcal{N}(0, \sigma^2)$

---

## 三、文件结构与修改说明

合成项目共包含 **11 个 `.py` 文件** 与 1 份说明文档：

| 文件 | 功能 | 对应的原始项目 |
|------|------|----------------|
| `main.py` | 统一入口，零参数运行完整流程 | — |
| `spatial_mesh.py` | 四面体网格生成、细化、点定位 | 1238, 747 |
| `environment_field.py` | 环境标量场的有限元插值与梯度 | 380 |
| `sensor_noise.py` | 复合传感器噪声模型 | 581 |
| `coverage_optimization.py` | CVT Lloyd算法与覆盖度量 | 253 |
| `interaction_matrix.py` | 稀疏Laplacian组装与Fiedler值 | 1111 |
| `stochastic_control.py` | Feynman-Kac路径积分势 | 422 |
| `swarm_dynamics.py` | 机器人混合动力学、BDF3、Theta、RK4 | 019, 075, 1259 |
| `density_field.py` | 宏观密度ETDRK4谱求解 | 123 |
| `distance_statistics.py` | 距离统计、KL散度、涌现指数 | 178, 819 |
| `spectral_approx.py` | Chebyshev与Bernstein谱工具 | 159, 077 |

**修改策略**：
- 所有MATLAB代码均重写为符合PEP8规范的Python 3代码；
- 删除了原始项目中所有可视化、图形输出、文件交互式输入；
- 增加了边界条件检查（如 mesh 点定位的 tol 容差、Bernstein 的 a≠b 判定、Chebyshev 的区间裁剪等）；
- 将原用于纯数学演示的代码（如BDF3求解简单ODE）升级为高维耦合系统的子模块；
- 引入数值鲁棒性处理（如 `np.clip`、奇异矩阵回退、`fsolve` 的 `maxfev` 限制等）。

---

## 四、科学问题解决能力

合成后的项目能够系统研究以下前沿科学问题：

1. **涌现行为的定量刻画**：通过图Laplacian谱（Fiedler值）与KL散度，从图论和信息论两个角度量化集群从无序到有序的自组织相变。
2. **多尺度耦合机制**：将微观个体混沌（Arneodo）、中观随机决策（Feynman-Kac）与宏观流体动力学（密度场）统一在同一框架下，揭示微观规则如何导致宏观模式。
3. **环境适应性覆盖**：CVT算法使集群在非均匀优先级密度下实现能量最优部署，适用于灾难搜救、环境监测等场景。
4. **通信拓扑与共识收敛**：稀疏Laplacian的谱分析直接给出通信约束下的信息融合速度下界。
5. **噪声鲁棒性评估**：复合噪声模型允许定量研究传感器失效、量化误差和热噪声对群体共识与覆盖性能的影响。

---

## 五、运行方式

### 环境要求
- Python 3.8+
- NumPy
- SciPy

### 安装依赖
```bash
pip install numpy scipy
```

### 运行
```bash
cd Synthesis-project-python/170_synth_project
python main.py
```

程序将自动执行以下流程并输出定量结果：
1. 网格生成与细化
2. 环境场构建
3. 机器人初始化与CVT优化
4. 传感器噪声模拟
5. 稀疏交互图构建
6. Feynman-Kac势计算
7. 多方法动力学积分（RK4主流程 + BDF3/Theta验证）
8. 宏观密度场演化
9. 涌现指标统计
10. 谱工具验证

**无需输入任何参数。**

---

## 六、质量检查清单

- [x] 原目录未被修改
- [x] 合成项目为Python语言
- [x] 新目录完整包含合成项目（11个.py文件 + 1个.md文档）
- [x] 单一博士级科学问题已落地为可执行代码
- [x] 全部15个输入项目均已真实融入，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数且无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 已删除所有可视化相关内容
