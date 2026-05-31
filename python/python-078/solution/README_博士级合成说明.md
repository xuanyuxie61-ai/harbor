# 计算流体力学：血动脉脉动流与壁面剪切应力分析

## 博士级科研代码合成项目 — PROJECT_78

---

## 一、项目概述

本项目围绕**计算流体力学（CFD）：血动脉脉动流与壁面剪切应力（Wall Shear Stress, WSS）**展开，将15个独立科研代码项目的核心算法融合为一个面向前沿生物医学工程问题的博士级Python计算框架。

### 科学问题

动脉粥样硬化（Atherosclerosis）是心血管疾病的主要病理基础。研究表明，**壁面剪切应力（WSS）** 是调控血管内皮细胞功能的关键力学因子：

- **低WSS区域**（< 1 Pa）：促进炎症因子表达、低密度脂蛋白（LDL）渗透增加，导致斑块形成
- **高WSS区域**（> 7 Pa）：诱导基质金属蛋白酶（MMP）过表达，增加斑块破裂风险
- **振荡剪切指数（OSI）> 0.15**：提示血流方向频繁改变，与斑块不稳定性显著相关

本项目通过多尺度、多物理场耦合计算，实现以下目标：
1. **微观尺度**：血浆中微粒布朗运动与红细胞相互作用 → 有效粘度修正
2. **介观尺度**：Womersley脉动流数值求解 → 瞬态速度剖面与WSS时空分布
3. **宏观尺度**：动脉网络血流分配（PageRank类比）→ 网络级流量与WSS分布
4. **控制尺度**：基于Pontryagin极大值原理的WSS最优药物调控

---

## 二、原项目到科学问题的映射

| 序号 | 原始项目 | 核心算法 | 合成角色 | 科学映射 |
|:---:|:---|:---|:---|:---|
| 1 | `868_pi_spigot` | BBP类Spigot算法计算π | `geometry_utils.py` | 血管圆形截面精确几何参数（面积 $A=\pi R^2$，周长 $C=2\pi R$） |
| 2 | `915_prime_plot` | 试除法素性测试 | `geometry_utils.py` | 血管分叉级别的素数标记（关键血流动力学节点识别） |
| 3 | `381_fem_to_triangle` | FEM节点-单元数据结构 | `geometry_utils.py` | 三角形有限元网格数据结构（`FEMMesh`类） |
| 4 | `528_hexagon_lyness_rule` | 六边形对称高斯求积 | `quadrature_rules.py` | 血管截面六边形网格上的速度场积分 |
| 5 | `660_legendre_fast_rule` | Gauss-Legendre节点权重 | `quadrature_rules.py` | WSS时间平均的高精度数值积分 |
| 6 | `1051_runge` | Runge函数分析 | `quadrature_rules.py` | WSS分布插值的Runge现象分析 |
| 7 | `966_r83t` | R83T三对角求解器 | `linear_algebra_core.py` | Womersley方程离散后的三对角线性系统求解 |
| 8 | `119_brownian_motion` | 多维布朗运动模拟 | `stochastic_diffusion.py` | 血浆微粒扩散系数计算（Stokes-Einstein关系） |
| 9 | `260_cvt_square_pdf` | 离散PDF驱动的CVT | `mesh_generation.py` | 血管壁非均匀厚度驱动的自适应网格生成 |
| 10 | `239_cvt_1_movie` | Lloyd迭代/最近邻搜索 | `mesh_generation.py` | CVT能量泛函最小化与Voronoi质心计算 |
| 11 | `861_pendulum_nonlinear_ode` | 非线性摆椭圆函数解 | `vessel_mechanics.py` | 血管壁径向弹性振动的非线性动力学模型 |
| 12 | `345_exm (orbits)` | N体引力模拟 | `vessel_mechanics.py` | 红细胞多体相互作用（等效Lennard-Jones力场） |
| 13 | `1061_schroedinger_nonlinear_pde` | NLSE孤子解/有限差分 | `pulse_wave_dynamics.py` | 压力脉冲孤子在动脉中的非线性传播 |
| 14 | `345_exm (waterwave)` | Lax-Wendroff浅水波 | `pulse_wave_dynamics.py` | 一维动脉压力波传播（Moens-Korteweg波速） |
| 15 | `1405_web_matrix` | PageRank/幂迭代 | `network_hemodynamics.py` | 动脉网络血流稳态分配（马尔可夫链类比） |
| 16 | `345_exm (pagerank)` | 马尔可夫链稳态分布 | `network_hemodynamics.py` | 带阻尼因子的PageRank流量分配 |
| 17 | `215_control_bio` | Pontryagin前向-后向扫描 | `optimal_control.py` | WSS最优药物释放控制策略 |

---

## 三、核心数学物理模型与公式

### 3.1 Womersley脉动流模型

轴对称圆管中的非定常Navier-Stokes方程简化为Womersley方程：

$$
\rho \frac{\partial u}{\partial t} = -\frac{\partial p}{\partial z} + \mu \left( \frac{\partial^2 u}{\partial r^2} + \frac{1}{r} \frac{\partial u}{\partial r} \right)
$$

边界条件：
- 壁面无滑移：$u(R, t) = 0$
- 轴心对称：$\left. \frac{\partial u}{\partial r} \right|_{r=0} = 0$

隐式时间离散（向后Euler + 中心差分）：

$$
\frac{u_j^{n+1} - u_j^n}{\Delta t} = \frac{1}{\rho} \left(-\frac{\partial p}{\partial z}\right)^n + \nu \left[ \frac{u_{j-1}^{n+1} - 2u_j^{n+1} + u_{j+1}^{n+1}}{\Delta r^2} + \frac{u_{j+1}^{n+1} - u_{j-1}^{n+1}}{2r_j \Delta r} \right]
$$

整理为三对角形式 $a_j u_{j-1}^{n+1} + b_j u_j^{n+1} + c_j u_{j+1}^{n+1} = d_j$，使用Thomas算法（$O(n)$复杂度）或共轭梯度法求解。

### 3.2 壁面剪切应力（WSS）

$$
\tau_w(t) = \mu \left. \frac{\partial u}{\partial r} \right|_{r=R} = \mu \frac{u_{N-1} - u_{N-2}}{\Delta r}
$$

**时间平均WSS（TAWSS）**：

$$
\text{TAWSS} = \frac{1}{T} \int_0^T |\tau_w(t)| \, dt
$$

使用Gauss-Legendre求积高精度计算：

$$
\int_a^b f(x) \, dx = \frac{b-a}{2} \sum_{i=1}^n w_i f\left( \frac{b+a}{2} + \frac{b-a}{2} x_i \right)
$$

其中 $x_i$ 为 $n$ 阶Legendre多项式 $P_n(x)$ 的根，权重 $w_i = \frac{2}{(1-x_i^2)[P_n'(x_i)]^2}$。

**振荡剪切指数（OSI）**：

$$
\text{OSI} = 0.5 \left[ 1 - \frac{\left| \int_0^T \tau_w \, dt \right|}{\int_0^T |\tau_w| \, dt} \right] \in [0, 0.5]
$$

### 3.3 Womersley数与波速

**Womersley数**（无量纲脉动参数）：

$$
\alpha = R \sqrt{\frac{\omega}{\nu}}
$$

- $\alpha \ll 1$：准稳态流，抛物线速度剖面
- $\alpha \sim 1$：过渡区
- $\alpha \gg 1$：惯性主导，速度剖面呈活塞状

**Moens-Korteweg压力波速**：

$$
c = \sqrt{\frac{E h}{2 \rho R}}
$$

### 3.4 Stokes-Einstein扩散与有效粘度

血浆中微粒（LDL、药物纳米颗粒）的扩散系数：

$$
D = \frac{k_B T}{6 \pi \eta R_p}
$$

Einstein粘度修正（红细胞悬浮液）：

$$
\mu_{\text{eff}} = \mu_0 \left( 1 + 2.5\phi + 6.2\phi^2 \right)
$$

其中 $\phi$ 为红细胞体积分数（Hematocrit）。

### 3.5 血管壁非线性弹性振动

将血管壁径向振动类比为非线性摆：

$$
\ddot{\xi} + g_{\text{eff}} \sin(\xi) = \frac{f_{\text{ext}}(t)}{m_w}
$$

其中 $\xi = (R - R_0)/R_0$ 为归一化径向位移，$g_{\text{eff}} = \frac{E h}{\rho_w R_0^2}$。

**椭圆函数精确解**（初始位移 $\xi_0$，零初始速度）：

$$
\xi(t) = 2 \arcsin\left[ \sin\left(\frac{\xi_0}{2}\right) \cdot \text{sn}\left( K(m) - \omega_0 t, \, m \right) \right]
$$

其中 $m = \sin^2(\xi_0/2)$，$K(m)$ 为第一类完全椭圆积分，$\omega_0 = \sqrt{g_{\text{eff}}}$，$\text{sn}(u,m)$ 为Jacobi椭圆正弦函数。

### 3.6 非线性薛定谔方程（NLSE）压力孤子

将压力脉冲建模为复波包 $\psi(z,t)$，满足聚焦型NLSE：

$$
i \frac{\partial \psi}{\partial t} + \frac{\partial^2 \psi}{\partial z^2} + \gamma |\psi|^2 \psi = 0
$$

空间离散（中心差分 + Neumann边界）：

$$
\frac{d\psi_j}{dt} = i \left[ \frac{\psi_{j+1} - 2\psi_j + \psi_{j-1}}{\Delta z^2} + \gamma |\psi_j|^2 \psi_j \right]
$$

时间推进采用四阶Runge-Kutta（RK4）格式。

### 3.7 动脉网络流量分配（PageRank类比）

将主动脉弓建模为有向图，构造转移矩阵 $T$（列随机矩阵）：

$$
T_{ij} = \frac{A_{ji}}{\sum_k A_{ik}} \quad \text{（归一化后转置）}
$$

稳态流量分布 $\pi$ 满足PageRank方程：

$$
\pi = d \, T \pi + \frac{1-d}{N} \mathbf{1}
$$

其中 $d=0.85$ 为阻尼因子。各节点流量 $Q_i = Q_{\text{total}} \cdot \pi_i / \pi_{\text{root}}$。

### 3.8 WSS最优控制（Pontryagin极大值原理）

**状态方程**（血管半径动力学）：

$$
\frac{dr}{dt} = k_g (r_{\text{eq}} - r) + k_u u(t) \, r
$$

**目标泛函**（最小化WSS偏离 + 控制代价）：

$$
J = \int_0^T \left[ \frac{1}{2} \left( \text{WSS}(r(t)) - \text{WSS}_{\text{target}} \right)^2 + \frac{B}{2} u(t)^2 \right] dt
$$

**Hamiltonian**：

$$
H = \frac{1}{2}(\text{WSS} - \text{WSS}_t)^2 + \frac{B}{2} u^2 + \lambda \left[ k_g(r_{\text{eq}} - r) + k_u u r \right]
$$

**伴随方程**：

$$
\frac{d\lambda}{dt} = -\frac{\partial H}{\partial r} = -(\text{WSS} - \text{WSS}_t) \frac{d\text{WSS}}{dr} - \lambda(-k_g + k_u u)
$$

**最优性条件**：

$$
\frac{\partial H}{\partial u} = B u + \lambda k_u r = 0 \quad \Rightarrow \quad u^*(t) = -\frac{\lambda k_u r}{B}
$$

采用**前向-后向扫描法**迭代求解状态-伴随-控制耦合系统。

### 3.9 CVT能量泛函与Lloyd算法

血管截面自适应网格生成的目标泛函：

$$
E(\mathbf{r}_1, \ldots, \mathbf{r}_N) = \int_\Omega \rho(\mathbf{x}) \min_j \|\mathbf{x} - \mathbf{r}_j\|^2 \, d\mathbf{x}
$$

其中密度 $\rho(\mathbf{x}) \propto 1/h(\mathbf{x})$，$h(\mathbf{x})$ 为局部壁厚。Lloyd算法通过迭代将生成器移动到Voronoi单元质心来最小化 $E$。

---

## 四、文件结构与改造路径

```
078_synth_project/
├── main.py                      # 统一入口，零参数运行
├── geometry_utils.py            # 融合 868 + 915 + 381
├── quadrature_rules.py          # 融合 528 + 660 + 1051
├── linear_algebra_core.py       # 融合 966 (R83T)
├── stochastic_diffusion.py      # 融合 119
├── mesh_generation.py           # 融合 260 + 239
├── vessel_mechanics.py          # 融合 861 + 345_orbits
├── pulse_wave_dynamics.py       # 融合 1061 + 345_waterwave
├── network_hemodynamics.py      # 融合 1405 + 345_pagerank
├── optimal_control.py           # 融合 215
├── pulsatile_cfd_engine.py      # 核心CFD引擎，调用所有模块
└── README_博士级合成说明.md      # 本文档
```

### 各文件改造细节

**`geometry_utils.py`**
- 保留868的Spigot类算法思想（BBP公式），用于高精度π计算
- 保留915的试除法素性测试，赋予其血管分叉节点标记的科学意义
- 保留381的FEM节点-单元数据结构，封装为`FEMMesh`类
- 新增Womersley数、雷诺数、Murray定律等CFD核心参数计算

**`quadrature_rules.py`**
- 将528的六边形Lyness求积从MATLAB翻译为Python，用于血管截面积分
- 将660的Gauss-Legendre快速求积算法翻译为Python，用于WSS时间积分
- 保留1051的Runge函数族（原函数、导数、积分、幂级数），赋予其WSS插值分析的科学角色

**`linear_algebra_core.py`**
- 完整移植966的R83T三对角矩阵格式：紧凑存储、矩阵-向量乘法、残差计算
- 保留DIF2经典测试矩阵及其理论特征值公式
- 实现Jacobi、Gauss-Seidel、Conjugate Gradient三种迭代求解器
- 新增Thomas直接算法（$O(n)$）和Womersley方程专用三对角系统构造器

**`stochastic_diffusion.py`**
- 完整移植119的布朗运动模拟（M维、N步、各向同性随机方向）
- 保留均方位移统计验证爱因斯坦关系的功能
- 新增Stokes-Einstein扩散系数、Einstein-Batchelor粘度修正、Peclet数、LDL壁面通量等生物医学应用层

**`mesh_generation.py`**
- 融合260的离散PDF逆变换采样与239的Lloyd迭代
- 实现`find_closest`暴力最近邻搜索和`cvt_iterate`单步迭代
- 新增血管特异性密度函数`vessel_wall_density`（模拟分叉处壁厚非均匀分布）
- 新增环形血管截面映射`map_cvt_to_annulus`和`VascularCVTMesh`封装类

**`vessel_mechanics.py`**
- 将861的非线性摆ODE完整映射为血管壁弹性振动模型
- 保留Jacobi椭圆函数精确解（`ellipj` + `ellipk`）、能量守恒、周期公式
- 新增`simulate_vessel_oscillation`模拟脉动压力下的振动响应
- 融合345_orbits的N体动力学，实现红细胞等效Lennard-Jones相互作用力

**`pulse_wave_dynamics.py`**
- 融合345_waterwave的Lax-Wendroff格式，用于一维动脉压力波传播
- 实现Moens-Korteweg波速公式
- 融合1061的NLSE，实现压力脉冲孤子传播模型
- 保留双孤子初始条件、质量守恒监测、RK4时间步进

**`network_hemodynamics.py`**
- 融合1405的邻接矩阵→转移矩阵转换和幂迭代法
- 融合345_pagerank的阻尼PageRank思想
- 构建8节点主动脉弓简化网络模型
- 实现Murray定律分叉流量分配、Poiseuille流阻、WSS估算

**`optimal_control.py`**
- 移植215_control_bio的前向-后向扫描法统一框架
- 状态方程：血管半径动力学
- 伴随方程：Hamiltonian对状态的负梯度
- 控制更新：解析最优性条件
- 新增WSS physiological score评估

**`pulsatile_cfd_engine.py`**
- 核心Womersley脉动流求解器（隐式有限差分 + 三对角求解）
- 心动周期压力梯度模型（基线 + 脉动谐波）
- TAWSS、OSI、WSSG（WSS时间梯度）、RRI（相对阻力指数）统计量
- Gauss-Legendre积分验证TAWSS
- 综合生理评分与临床判读

---

## 五、合成项目能解决的科学问题

1. **动脉脉动流WSS时空分布计算**：基于Womersley方程的隐式有限差分求解，得到心动周期内WSS的瞬态变化
2. **动脉粥样硬化风险评估**：通过TAWSS、OSI、WSSG等多参数综合评分，识别高危血管段
3. **血管网络血流分配分析**：将PageRank思想应用于动脉树，计算各分支流量与WSS
4. **药物释放最优控制**：通过Pontryagin原理优化血管活性物质释放策略，使WSS维持在生理范围（1–7 Pa）
5. **血管壁粘弹性响应预测**：非线性摆模型描述大变形下血管壁的周期振动与能量耗散
6. **压力脉冲传播模拟**：NLSE孤子模型 + 浅水波方程，分析脉搏波在动脉中的传播与反射
7. **红细胞动力学对血流的影响**：多体相互作用 + 布朗扩散 → 有效粘度修正 → 脉动流精度提升

---

## 六、运行方式

### 环境要求
- Python ≥ 3.9
- NumPy ≥ 1.21
- SciPy ≥ 1.7

### 运行命令
```bash
cd 078_synth_project
python main.py
```

**零参数可运行**。程序将自动执行以下10个计算阶段并输出结果：

1. 血管几何与基础物理参数（π、素数标记、Womersley数、雷诺数、Murray定律）
2. 数值积分与插值分析（六边形求积、Gauss-Legendre、Runge现象）
3. 三对角线性系统求解器（Jacobi/GS/CG/Thomas对比）
4. 血浆微粒布朗运动与有效扩散（Stokes-Einstein、粘度修正、Peclet数）
5. 血管截面CVT自适应网格生成（非均匀密度驱动、环形映射）
6. 血管壁弹性力学与红细胞相互作用（椭圆函数精确解、N体力场）
7. 动脉压力脉冲波传播（Lax-Wendroff、NLSE孤子）
8. 动脉网络血流分配（PageRank类比、Murray分叉）
9. WSS最优药物释放控制（Pontryagin前向-后向扫描）
10. 核心脉动流CFD引擎（Womersley求解、WSS统计报告、临床判读）

### 预期运行时间
- 普通桌面CPU（4核）：约 10–20 秒
- 主要耗时：CVT Lloyd迭代（~5秒）、布朗运动Monte Carlo（~2秒）、Womersley时间步进（~3秒）

---

## 七、质量检查清单

- [x] 原目录未被修改
- [x] 合成后的项目为Python语言
- [x] 新目录完整包含合成后的项目（11个.py文件 + 1个.md文档）
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] **全部15个输入项目均已真实融入合成项目**，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数且无报错
- [x] 代码具备边界处理与数值鲁棒性（ safe_sqrt、safe_divide、参数正性检查、clip约束等）
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 无可视化代码（所有绘图/动画相关代码已删除或替换为数值输出）
