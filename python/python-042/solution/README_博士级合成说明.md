# 地幔对流与板块运动数值模拟系统 — 博士级合成说明

## 1. 项目概述

本项目围绕**地球物理：地幔对流与板块运动数值模拟**这一前沿科学领域，将15个科研代码项目的核心算法融合重构为一个完整的、可运行的博士级科学计算系统。

项目采用 Python 语言实现，包含9个 `.py` 文件（含 `main.py` 统一入口），零参数即可运行，输出地幔温度场演化、斯托克斯流场、化学交换动力学、无量纲诊断参数等完整结果。

---

## 2. 原项目到科学问题的映射

| 原项目编号 | 原项目核心算法 | 在合成项目中的科学角色 |
|:---|:---|:---|
| 868_pi_spigot | π 的 spigot 算法 | `spherical_geometry.py`：高精度球面几何常数计算，支撑地球表面积、地幔体积等公式 |
| 1423_xyz_display | 3D 坐标显示 | `spherical_geometry.py`：球坐标 ↔ 直角坐标变换，用于地幔网格节点三维定位（删除可视化） |
| 480_gram_schmidt | Gram-Schmidt 正交化 | `spectral_basis.py`：构造斯托克斯求解器的谱基函数正交基，保证 Galerkin 投影数值稳定性 |
| 1357_trig_interp_basis | 三角插值基函数 | `spectral_basis.py`：角向周期边界条件的谱展开基，用于温度场与流函数的方位角离散 |
| 635_lagrange_interp_1d | 一维 Lagrange 插值 | `spectral_basis.py`：径向地幔物性（粘度、密度）的层间插值 |
| 1208_test_int_2d / legendre_dr_compute | Gauss-Legendre 求积 | `quadrature_engine.py`：二维矩形/三角形单元上的高斯数值积分，用于 Galerkin 刚度矩阵组装 |
| 467_gen_laguerre_rule | 广义 Gauss-Laguerre 求积 | `quadrature_engine.py`：热边界层在半无限径向域上的指数加权积分 |
| 559_hypercube_integrals | 超立方体蒙特卡洛采样 | `quadrature_engine.py`：地幔参数空间（瑞利数、热膨胀系数、粘度）的不确定性量化 |
| 890_polygon_triangulate | 多边形耳切法三角剖分 | `mesh_generator.py`：复杂俯冲带几何截面的非结构化网格生成 |
| 1394_voronoi_city | Voronoi 图构造 | `mesh_generator.py`：地表速度场 Voronoi 划分，模拟板块边界动力学 |
| 1372_unicycle | 单循环置换索引 | `mesh_generator.py`：环形截面边界节点的循环重编号，保证周期性边界条件拓扑一致性 |
| 488_grazing_ode | 捕食-食草耦合 ODE | `thermal_solver.py`：上/下地幔化学储库交换模型（不相容元素富集/亏损动力学） |
| 1428_zero_chandrupatla | Chandrupatla 根查找 | `diagnostics.py`：求解临界瑞利数 $Ra_c$，判定对流起始条件 |
| 691_lissajous | Lissajous 曲线 | `diagnostics.py`：潮汐周期强迫参数化，调制 CMB 热流边界条件 |
| 539_histogram_discrete | 离散直方图 | `diagnostics.py`：温度场统计分布、熵与混合程度分析 |

---

## 3. 新增数学物理模型与核心公式

### 3.1 球面几何与坐标系

- 地表面积：
  $$A = 4\pi R_{\text{surf}}^2$$
- 地幔球壳体积：
  $$V = \frac{4}{3}\pi \left(R_{\text{surf}}^3 - R_{\text{cmb}}^3\right)$$
- 球坐标 ↔ 直角坐标：
  $$x = r\sin\theta\cos\phi, \quad y = r\sin\theta\sin\phi, \quad z = r\cos\theta$$

### 3.2 温度依赖型粘度（Arrhenius 定律）

$$\eta(T) = \eta_0 \exp\!\left[\frac{E^*}{R_{\text{gas}}}\left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right)\right]$$

其中 $E^*$ 为活化能，$R_{\text{gas}}$ 为气体常数。代码中实现了物理边界截断：
$\eta \in [\eta_0/100, 100\eta_0]$。

### 3.3 Boussinesq 近似密度

$$\rho(T) = \rho_0 \bigl[1 - \alpha (T - T_{\text{ref}})\bigr]$$

浮力：$\Delta\rho = \rho_0 \alpha (T - T_{\text{ref}})$，热物质上浮、冷物质下沉。

### 3.4 瑞利数与临界对流

$$Ra = \frac{\rho_0 g \alpha \Delta T D^3}{\eta \kappa}$$

其中 $D = R_{\text{surf}} - R_{\text{cmb}}$ 为地幔厚度。临界瑞利数 $Ra_c \approx 657.5$（刚性边界、等粘度），本项目通过 Chandrupatla 混合二次/二分算法数值求解 $Nu(Ra) - 1.05 = 0$ 得到 $Ra_c$。

### 3.5 斯托克斯方程（无穷普朗特数极限）

对于地幔蠕变流，动量平衡为：
$$\nabla\cdot\boldsymbol{\sigma} + \rho g \hat{\boldsymbol{r}} = 0, \quad \nabla\cdot\boldsymbol{u} = 0$$

其中应力张量：
$$\boldsymbol{\sigma} = -p\boldsymbol{I} + \eta\bigl(\nabla\boldsymbol{u} + \nabla\boldsymbol{u}^T\bigr)$$

本项目采用流函数-涡量形式：
$$\nabla^4 \psi \approx -Ra_{\text{eff}} \, (T - \bar{T})$$

并通过 Jacobi 迭代求解泊松方程，导出速度分量：
$$u_r = \frac{1}{r}\frac{\partial\psi}{\partial\theta}, \quad u_\theta = -\frac{\partial\psi}{\partial r}$$

### 3.6 热对流-扩散方程

极坐标下的能量守恒：
$$\frac{\partial T}{\partial t} + u_r\frac{\partial T}{\partial r} + \frac{u_\theta}{r}\frac{\partial T}{\partial\theta} = \kappa\left[\frac{1}{r}\frac{\partial}{\partial r}\!\left(r\frac{\partial T}{\partial r}\right) + \frac{1}{r^2}\frac{\partial^2 T}{\partial\theta^2}\right] + \frac{H}{\rho_0 C_p}$$

时间离散采用算子分裂：显式平流 + 隐式扩散（简化为带 CFL 约束的前向欧拉）。

### 3.7 上/下地幔化学交换（Grazing ODE 类比）

将原 grazing 捕食-食草模型映射为地幔化学储库交换：

$$\frac{dC_{\text{um}}}{dt} = r_1 C_{\text{um}}\!\left(1 - \frac{C_{\text{um}}}{K}\right) - c_1 C_{\text{lm}}\bigl(1 - e^{-d_1 C_{\text{um}}}\bigr)$$

$$\frac{dC_{\text{lm}}}{dt} = -a C_{\text{lm}} + c_2 C_{\text{lm}}\bigl(1 - e^{-d_2 C_{\text{um}}}\bigr)$$

其中 $C_{\text{um}}$、$C_{\text{lm}}$ 分别为上、下地幔不相容元素浓度。采用 4 阶 Runge-Kutta 积分。

### 3.8 谱展开与 Galerkin 投影

- 径向基：通过 Gram-Schmidt 正交化构造 Legendre 型多项式基 $R_k(r)$。
- 角向基：三角基函数（Dirichlet 核族）：
  $$B_k(x) = \frac{\sin(k\pi x/2)}{k\sin(\pi x/2)} \quad (k\text{ 为奇数})$$
  $$B_k(x) = \frac{\sin(k\pi x/2)}{k\tan(\pi x/2)} \quad (k\text{ 为偶数})$$
- 模态展开：
  $$\psi(r,\theta) = \sum_{k=1}^{N_r}\sum_{l=1}^{N_\theta} c_{kl} \, R_k(r) \, \Theta_l(\theta)$$

### 3.9 Voronoi 板块划分

对于地表生成元 $\{\boldsymbol{p}_i\}$，Voronoi 单元定义为：
$$V_i = \left\{\boldsymbol{x} \in \mathbb{R}^2 : \|\boldsymbol{x} - \boldsymbol{p}_i\| \le \|\boldsymbol{x} - \boldsymbol{p}_j\|, \; \forall j \neq i\right\}$$

通过网格化距离判据近似计算各板块面积。

### 3.10 离散直方图与熵

对温度场样本构造分段线性 PDF $p(x)$，满足归一化：
$$\int_{T_{\min}}^{T_{\max}} p(x)\,dx = 1$$

微分熵：
$$S = -\int p(x)\ln p(x)\,dx$$

用于量化地幔热混合程度。

---

## 4. 项目文件结构与运行方式

### 4.1 文件清单

| 文件名 | 功能说明 |
|:---|:---|
| `main.py` | **统一入口**，零参数运行，依次调用所有模块并输出完整结果 |
| `spherical_geometry.py` | 球面几何、高精度 π、坐标变换 |
| `mesh_generator.py` | 耳切法三角剖分、Voronoi 板块划分、Unicycle 循环索引、环形网格生成 |
| `spectral_basis.py` | Gram-Schmidt 正交化、三角基函数、Lagrange 插值、谱展开构造 |
| `quadrature_engine.py` | Gauss-Legendre/Laguerre 求积、2D 积分、超立方体蒙特卡洛采样 |
| `mantle_physics.py` | 地幔物理模型：Arrhenius 粘度、Boussinesq 密度、瑞利数/努塞尔数/普朗特数、斯托克斯与热物理方程 |
| `stokes_solver.py` | 流函数-涡量斯托克斯求解器、Galerkin 投影、速度场推导 |
| `thermal_solver.py` | 热对流-扩散时间步进、初始温度场、表面热流、Grazing ODE 化学交换 |
| `diagnostics.py` | 温度直方图/熵、Chandrupatla 根查找、Lissajous 周期强迫、参数不确定性蒙特卡洛传播 |

### 4.2 运行方式

```bash
cd Synthesis-project-python/042_synth_project
python main.py
```

无需任何输入参数。程序将依次执行：
1. 球面几何计算与坐标变换验证
2. 网格生成与 Voronoi 板块划分
3. 谱基构造与插值精度测试
4. 高斯/蒙特卡洛积分验证
5. 地幔物理参数与无量纲数输出
6. 斯托克斯流场求解与温度场时间演化
7. 化学交换 ODE 积分
8. 诊断分析：统计、根查找、周期强迫、不确定性传播
9. 综合模拟结果汇总

---

## 5. 代码边界处理与数值鲁棒性

- **温度截断**：所有温度场通过 `np.clip(T, T_surf, T_cmb)` 限制在物理范围内。
- **粘度截断**：Arrhenius 粘度计算限制在 $[\eta_0/100, 100\eta_0]$，避免数值溢出。
- **零除保护**：坐标变换中 `r_safe = max(r, 1e-15)`；所有分母均检查 `abs(denom) > eps`。
- **CFL 稳定性**：时间步进自动根据 `u_max` 和 `kappa` 调整 `dt`，保证显式格式稳定。
- **根查找边界**：Chandrupatla 算法要求区间端点异号，若不满足则返回最接近的边界值。
- **多边形鲁棒性**：耳切法检查顶点共线、角度下限（$>5.7\times10^{-5}$°）、正面积，防止退化输入导致崩溃。
- **Voronoi 鲁棒性**：生成元坐标通过网格采样近似，避免解析几何中的无穷远边处理。

---

## 6. 科学问题说明

本项目合成的核心科学问题是：**在地球地幔球壳二维截面中，温度依赖粘度的热化学对流如何驱动板块运动？**

具体包含以下子问题：
1. **流体力学**：通过斯托克斯方程求解无穷普朗特数下的蠕变流场，获得地幔内部速度分布。
2. **热力学**：求解平流-扩散方程，追踪温度场从纯传导态到对流态的演化。
3. **化学地球动力学**：通过 Grazing ODE 类比，模拟上/下地幔间不相容元素的交换与分异。
4. **板块构造**：利用 Voronoi 划分将地表速度场离散为“板块”，量化各板块面积。
5. **临界现象**：通过 Chandrupatla 根查找确定对流起始的临界瑞利数。
6. **参数敏感性**：利用超立方体蒙特卡洛采样评估地幔参数不确定性对热对流的影响。

该问题涉及流体力学、热力学、化学动力学、几何拓扑与数值分析的多学科交叉，计算难度达到博士级前沿水平。

---

## 7. 合成改造总结

- **未修改原目录**：所有原始15个项目目录保持完整，仅读取其算法逻辑。
- **新建合成目录**：所有代码均写入 `Synthesis-project-python/042_synth_project`。
- **删除可视化**：原项目中所有 `plot`、`figure`、`scatter3` 等可视化代码已全部删除，仅保留科学计算核心。
- **语言转换**：原项目多为 MATLAB，已全部改写为 Python 3。
- **公式-算法-代码一致性**：文档中的每个公式均在对应 `.py` 文件中有直接实现，确保可追溯。
- **已验证运行**：`main.py` 已实际运行通过，零报错。
