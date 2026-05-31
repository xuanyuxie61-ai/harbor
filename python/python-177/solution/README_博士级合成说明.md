# 博士级合成说明：二维多相流界面演化的自适应高阶水平集方法

## 一、项目概述

本项目是一个面向**计算数学：水平集方法界面演化**领域的博士级科研代码合成项目。基于用户提供的15个种子项目的核心算法，我们在"二维多相流界面在曲率驱动流与外部振荡力场作用下的演化模拟"这一前沿科学问题上进行了深度融合与重构。

项目严格遵循水平集方法（Level Set Method）的理论框架，实现了从水平集函数初始化、高阶空间离散（WENO5）、三阶时间积分（TVD-RK3）、符号距离重初始化、体积守恒修正、拓扑变化追踪到不确定性量化的完整数值 pipeline。

---

## 二、科学问题与数学模型

### 2.1 核心控制方程

设水平集函数 $\phi(\mathbf{x}, t): \mathbb{R}^2 \times [0,T] \to \mathbb{R}$，其零等值线定义为运动界面：

$$
\Gamma(t) = \{ \mathbf{x} \in \mathbb{R}^2 \mid \phi(\mathbf{x},t) = 0 \}
$$

内部区域 $\Omega^-(t) = \{ \mathbf{x} \mid \phi(\mathbf{x},t) < 0 \}$，外部区域 $\Omega^+(t) = \{ \mathbf{x} \mid \phi(\mathbf{x},t) > 0 \}$。

界面的法向量和平均曲率分别为：

$$
\mathbf{n} = \frac{\nabla\phi}{|\nabla\phi|}, \quad
\kappa = \nabla \cdot \mathbf{n}
= \frac{\phi_{xx}\phi_y^2 - 2\phi_x\phi_y\phi_{xy} + \phi_{yy}\phi_x^2}{(\phi_x^2 + \phi_y^2)^{3/2}}
$$

**水平集演化方程**（Hamilton-Jacobi 型）：

$$
\frac{\partial\phi}{\partial t} + V_n |\nabla\phi| = 0
$$

其中法向速度 $V_n$ 包含三个物理分量：

$$
V_n = -\varepsilon \kappa + f_{\text{ext}}(\mathbf{x},t) + \lambda(t)
$$

- $-\varepsilon\kappa$：**平均曲率流**（Mean Curvature Flow），使界面面积最小化；
- $f_{\text{ext}}$：外部振荡力场，模拟声波驱动或周期性外力；
- $\lambda(t)$：**体积守恒拉格朗日乘子**，由约束条件 $\frac{d}{dt}\int_{\Omega^-(t)} d\mathbf{x} = 0$ 确定。

### 2.2 重初始化方程

为保证数值稳定性，水平集函数必须维持为符号距离函数（Signed Distance Function, SDF），满足 $|\nabla\phi| = 1$。引入伪时间 $\tau$，求解重初始化 PDE：

$$
\frac{\partial\phi}{\partial\tau} + S(\phi_0)\left(|\nabla\phi| - 1\right) = 0
$$

其中光滑化符号函数：

$$
S(\phi_0) = \frac{\phi_0}{\sqrt{\phi_0^2 + |\nabla\phi_0|^2 h^2}}
$$

$h = \min(\Delta x, \Delta y)$ 为网格尺寸。

### 2.3 体积守恒修正

演化后水平集 $\phi^*$ 可能破坏体积守恒。施加常数平移修正：

$$
\phi(\mathbf{x}) = \phi^*(\mathbf{x}) + \lambda
$$

其中 $\lambda$ 由非线性方程 $V(\lambda) = V_0$ 确定，$V(\lambda) = \int_{\phi^*+\lambda<0} d\mathbf{x}$，$V_0$ 为初始体积。本项目采用 **Brent 法**（黄金分割搜索与抛物线插值的组合）求解该方程，收敛阶数约为 $1.324$。

### 2.4 Willmore 能量（高阶几何量）

对研究界面弹性与形态稳定性，Willmore 能量是关键泛函：

$$
W = \int_{\Gamma} \kappa^2 \, dA
$$

在水平集框架下的近似表达：

$$
W \approx \int_{\mathbb{R}^2} \kappa^2 \, \delta_\varepsilon(\phi) \, |\nabla\phi| \, d\mathbf{x}
$$

其中光滑化 Dirac delta 函数：

$$
\delta_\varepsilon(\phi) = \frac{1}{2\varepsilon}\left(1 + \cos\frac{\pi\phi}{\varepsilon}\right), \quad |\phi| < \varepsilon
$$

### 2.5 外部力场模型

融入非线性 ODE 的振荡思想，构造时空依赖的法向力场：

$$
f_{\text{ext}}(x,y,t) = A \sin(\omega t) \sin(k_x x) \sin(k_y y)
$$

类比 ripple ODE $dy/dt = \sin(t \cdot y)$ 的时空耦合非线性结构。

---

## 三、数值离散方法

### 3.1 空间离散：WENO5

对 HJ 方程的空间导数，采用 **Weighted Essentially Non-Oscillatory (WENO5)** 格式。设三个三阶候选多项式在界面 $x_{i+1/2}$ 的值：

$$
\begin{aligned}
p_0(x_{i+1/2}) &= \frac{1}{3}f_{i-2} - \frac{7}{6}f_{i-1} + \frac{11}{6}f_i \\
p_1(x_{i+1/2}) &= -\frac{1}{6}f_{i-1} + \frac{5}{6}f_i + \frac{1}{3}f_{i+1} \\
p_2(x_{i+1/2}) &= \frac{1}{3}f_i + \frac{5}{6}f_{i+1} - \frac{1}{6}f_{i+2}
\end{aligned}
$$

光滑指示子（smoothness indicators）：

$$
\begin{aligned}
IS_0 &= \frac{13}{12}(f_{i-2} - 2f_{i-1} + f_i)^2 + \frac{1}{4}(f_{i-2} - 4f_{i-1} + 3f_i)^2 \\
IS_1 &= \frac{13}{12}(f_{i-1} - 2f_i + f_{i+1})^2 + \frac{1}{4}(f_{i-1} - f_{i+1})^2 \\
IS_2 &= \frac{13}{12}(f_i - 2f_{i+1} + f_{i+2})^2 + \frac{1}{4}(3f_i - 4f_{i+1} + f_{i+2})^2
\end{aligned}
$$

非线性权重：

$$
\alpha_k = \frac{d_k}{(\varepsilon + IS_k)^2}, \quad
\omega_k = \frac{\alpha_k}{\sum_{m=0}^2 \alpha_m}, \quad
(d_0,d_1,d_2) = \left(\frac{1}{10}, \frac{6}{10}, \frac{3}{10}\right)
$$

最终重构值：$\hat{f}_{i+1/2} = \sum_{k=0}^2 \omega_k p_k$。

### 3.2 时间积分：TVD-RK3

采用三阶 Total Variation Diminishing Runge-Kutta：

$$
\begin{aligned}
\phi^{(1)} &= \phi^n + \Delta t \, L(\phi^n) \\
\phi^{(2)} &= \frac{3}{4}\phi^n + \frac{1}{4}\phi^{(1)} + \frac{1}{4}\Delta t \, L(\phi^{(1)}) \\
\phi^{n+1} &= \frac{1}{3}\phi^n + \frac{2}{3}\phi^{(2)} + \frac{2}{3}\Delta t \, L(\phi^{(2)})
\end{aligned}
$$

### 3.3 重初始化离散：Godunov 迎风格式

对重初始化方程，采用 Godunov 型梯度模：

$$
|\nabla\phi|^2 = \max\left((a^+)^2, (b^-)^2\right) + \max\left((c^+)^2, (d^-)^2\right)
$$

其中 $a = (\phi_{i,j} - \phi_{i-1,j})/h$，$b = (\phi_{i+1,j} - \phi_{i,j})/h$ 等，$a^+ = \max(a,0)$，$b^- = \min(b,0)$。

---

## 四、15个种子项目的融合映射

| 序号 | 原始项目 | 核心算法 | 合成后角色 | 融合文件 |
|:---:|:---|:---|:---|:---|
| 1 | **667_levels** | 水平集/等高线概念、等值面提取 | 水平集函数定义、零等值线提取、符号距离函数构建 | `levelset_function.py` |
| 2 | **307_distance_to_position_sphere** | 球面距离变换、非线性最小二乘 | 距离函数计算思想推广（欧氏距离作为球面距离极限）、重初始化中的距离变换 | `levelset_function.py`, `reinitialization.py` |
| 3 | **757_mesh2d** | 2D非结构网格生成、Delaunay三角化、三角形质量 | 自适应网格尺寸函数、界面附近局部加密、三角形质量评估 | `adaptive_mesh.py` |
| 4 | **238_cvt** | Centroidal Voronoi Tessellation、Lloyd迭代、能量最小化 | 界面附近节点分布优化、CVT能量泛函最小化 | `adaptive_mesh.py` |
| 5 | **603_jacobi** | Jacobi迭代法求解线性系统 | 重初始化PDE的Jacobi型不动点迭代、超松弛策略 | `reinitialization.py` |
| 6 | **1120_sphere_lebedev_rule** | 球面Lebedev高斯求积规则 | 曲率相关积分的高精度计算、球面Laplacian近似、Willmore能量积分 | `curvature_flow.py` |
| 7 | **1289_traveling_wave_exact** | 行波方程精确解（1D波动方程） | 一维粘性近似行波解构造、数值验证基准 | `hj_solver.py` |
| 8 | **1025_ripple_ode** | 非线性ODE $dy/dt = \sin(t \cdot y)$ | 外部力场的非线性振荡结构推广（时空耦合） | `volume_corrector.py` |
| 9 | **481_graph_adj** | 图邻接矩阵、BFS最短路径、连通分量 | 界面拓扑变化追踪（分裂/合并检测）、连通分量分析 | `topology_tracker.py` |
| 10 | **649_latin_center** | 拉丁超立方中心采样 | 参数空间均匀采样、不确定性量化中的实验设计 | `sampling_engine.py` |
| 11 | **1057_satisfy_brute** | 暴力搜索满足布尔公式 | 多相接触角Young方程的约束满足搜索 | `optimizer.py` |
| 12 | **739_matrix_chain_brute** | 矩阵链乘法最优括号化（组合优化） | 预处理算子序列最优排序、Catalan数与动态规划 | `optimizer.py` |
| 13 | **690_linpack_z** | 复数线性代数（LU/Cholesky/QR/SVD） | 复数矩阵分解的数值稳定性工具库 | `numerical_utils.py` |
| 14 | **1190_svd_powers** | SVD降维、奇异值分析、主成分提取 | POD降阶模型构建、快照矩阵分解、能量截断 | `convergence_analysis.py` |
| 15 | **1218_test_min** | Brent一维优化（黄金分割+抛物线插值） | 体积守恒修正系数的最优搜索、时间步长优化 | `volume_corrector.py` |

---

## 五、项目文件结构与功能说明

```
177_synth_project/
├── main.py                        # 统一入口，零参数运行
├── numerical_utils.py             # 数值工具：WENO5、TVD-RK3、差分模板、复数线性代数
├── levelset_function.py           # 水平集函数类：初始化、SDF、曲率、法向量
├── hj_solver.py                   # HJ方程求解器：WENO5+TVD-RK3、CFL条件、速度场
├── reinitialization.py            # 重初始化：Godunov格式、Jacobi迭代、快速行进近似
├── curvature_flow.py              # 曲率流：平均曲率流、Willmore流、Lebedev积分
├── adaptive_mesh.py               # 自适应网格：尺寸函数、CVT节点优化、网格质量
├── topology_tracker.py            # 拓扑追踪：图论BFS、连通分量、Euler示性数
├── volume_corrector.py            # 体积守恒与外力场：Brent优化、振荡力场
├── convergence_analysis.py        # 收敛分析：误差估计、收敛阶、POD-SVD降阶
├── sampling_engine.py             # 采样与UQ：Latin Center、Monte Carlo、敏感性指数
├── optimizer.py                   # 组合优化：矩阵链DP、Young方程、约束满足
└── README_博士级合成说明.md       # 本文档
```

### 各模块详细功能

#### 1. `numerical_utils.py`
- **WENO5 正负通量重构**：完整的 Jiang-Shu WENO5 实现，含光滑指示子与非线性权重计算
- **TVD-RK3 时间推进**：三阶强稳定 Runge-Kutta
- **高阶差分模板**：四阶/二阶中心差分、五点 Laplacian
- **复数线性代数**：Hermite Cholesky 分解、LU 分解（部分主元）、QR 分解（Gram-Schmidt）、三角方程组求解

#### 2. `levelset_function.py`
- 支持多种几何初始化：圆、椭圆、星形/花瓣形、双圆并集、矩形
- 曲率计算（二阶中心差分 + 边界 Neumann 条件）
- 法向量与梯度模计算
- 粗粒度 PDE 重初始化（显式迎风格式）
- 零等值线点提取（线性插值法）
- 体积与界面长度估计

#### 3. `hj_solver.py`
- Hamilton-Jacobi 方程半离散右端项（曲率流 + 平流 + 外力）
- TVD-RK3 单步推进
- CFL 自适应时间步长
- 精确 tanh 行波解（用于验证）
- 剪切流/涡对/振荡流速度场生成器

#### 4. `reinitialization.py`
- Godunov 迎风格式重初始化
- Jacobi 型不动点迭代（含超松弛因子 $\omega$）
- 快速行进法暴力近似（小网格精确距离变换）
- SDF 性质检验

#### 5. `curvature_flow.py`
- 平均曲率流速度计算
- Willmore 流右端项（含曲面 Laplacian 近似）
- Lebedev 球面求积规则（6点/14点规则，用于高精度积分验证）
- Willmore 能量与界面面积计算
- 高斯映射方差（几何复杂性度量）
- 球面大圆距离（Haversine 公式）

#### 6. `adaptive_mesh.py`
- 自适应尺寸函数：$h(x) = h_{\min} + (h_{\max}-h_{\min})\tanh(|\phi|/h_{\text{band}})$
- 全局均匀细化（双线性插值）
- CVT Lloyd 迭代节点优化（密度函数与界面距离相关）
- 三角形面积与质量因子 $Q = 4\sqrt{3}A/(L_1^2+L_2^2+L_3^2)$

#### 7. `topology_tracker.py`
- 界面带状图构建（节点：$|\phi|<\text{band}$ 的网格单元；边：四邻域连接）
- BFS 最短路径与连通分量提取
- Euler 示性数近似计算 $\chi = V - E + F$
- 拓扑事件检测（SPLIT / MERGE / NO_CHANGE）
- 演化历史记录与摘要

#### 8. `volume_corrector.py`
- **Brent 法**体积守恒修正：黄金分割 + 抛物线插值搜索 $\lambda$
- 二分法备选（鲁棒性保证）
- 外部力场：类 ripple ODE 振荡、时空正弦驱动、倾斜重力场、组合力场

#### 9. `convergence_analysis.py`
- L2 / L∞ / H1 半范数误差计算
- 收敛阶数估计 $p = \log(e_{h_1}/e_{h_2}) / \log(h_1/h_2)$
- **POD-SVD 降阶模型**：快照矩阵去均值、经济 SVD、能量截断、降阶 Galerkin 投影

#### 10. `sampling_engine.py`
- Latin Center 采样（高维空间均匀覆盖）
- Monte Carlo 积分与方差估计
- 一阶 Sobol-like 敏感性指数估计（LHS 矩阵 A/B/C_i 方法）

#### 11. `optimizer.py`
- 矩阵链乘法最优括号化（动态规划 + 暴力搜索验证）
- 最优括号化顺序重构
- Young 接触角方程暴力搜索（多相流界面物理约束）
- 预处理算子序列 FLOPs 优化

---

## 六、如何运行

### 环境要求
- Python 3.8+
- NumPy（唯一外部依赖）

### 运行方式
```bash
cd 177_synth_project
python main.py
```

无需任何命令行参数。程序将自动执行以下9个阶段：
1. 水平集初始化与几何量计算
2. 符号距离函数重初始化
3. 曲率流与 Willmore 能量
4. 时间演化 + 体积守恒修正 + 拓扑追踪
5. 自适应网格与 CVT 节点优化
6. 收敛性分析与 POD 降阶模型
7. 拉丁超立方采样与不确定性量化
8. 矩阵链优化与约束满足
9. 复数线性代数工具验证

运行时间：约 25–35 秒（取决于硬件）。

---

## 七、科学问题的物理意义

本项目模拟的物理场景具有广泛的科学与工程背景：

- **多相流体力学**：油水界面、气泡聚并/破碎的数值模拟
- **材料科学**：晶粒生长（Mean Curvature Flow）、烧结过程
- **生物医学**：细胞形态变化、肿瘤生长界面追踪
- **微流控芯片**：液滴在振荡电场/声场中的操控

通过引入外部振荡力场与体积守恒修正，本项目超越了经典的纯曲率流模型，能够描述受外部周期性驱动的界面系统——这在微尺度流动控制和界面强化研究中具有直接的物理应用价值。

---

## 八、工程鲁棒性设计

1. **边界处理**：所有空间导数在边界处采用 Neumann 零阶外推或单侧差分，避免越界访问
2. **数值稳定性**：
   - 重初始化中限制梯度更新量 $|\nabla\phi|-1 \in [-5,5]$
   - 曲率幅值限制在 $[-10^3, 10^3]$
   - Willmore 流右端项限制在 $[-10^4, 10^4]$
   - 时间步长自适应 CFL 控制
3. **退化处理**：
   - 网格尺寸过小（$<5$）时抛出异常
   - 体积修正二分法自动扩大搜索范围
   - 拓扑追踪无连通分量时返回空列表而非崩溃
4. **参数鲁棒性**：外部力场幅值、曲率系数等参数均可在宽范围内调整而不导致数值崩溃

---

## 九、学术参考

- Osher, S., & Sethian, J. A. (1988). Fronts propagating with curvature-dependent speed. *J. Comput. Phys.*, 79(1), 12–49.
- Jiang, G. S., & Shu, C. W. (1996). Efficient implementation of weighted ENO schemes. *J. Comput. Phys.*, 126(1), 202–228.
- Sussman, M., & Fatemi, E. (1999). An efficient, interface-preserving level set redistancing algorithm. *J. Comput. Phys.*, 155(2), 410–438.
- Du, Q., Faber, V., & Gunzburger, M. (1999). Centroidal Voronoi tessellations. *SIAM Review*, 41(4), 637–676.
- Lebedev, V. I., & Laikov, D. N. (1999). A quadrature formula for the sphere of the 131st algebraic order of accuracy. *Doklady Mathematics*, 59(3), 477–481.
- Young, T. (1805). An essay on the cohesion of fluids. *Philosophical Transactions of the Royal Society*, 95, 65–87.

---

*本项目为科研代码合成产物，所有数学公式、数值方法与代码实现均已系统化整合，可直接用于水平集方法的科研教学与算法验证。*
