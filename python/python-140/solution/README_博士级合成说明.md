# 生物质热解反应器多物理场耦合模拟系统 — 博士级合成说明

## 1. 项目概述

本项目围绕**化学工程：生物质热解反应器模拟**展开，构建了一个多尺度、多物理场耦合的博士级数值模拟平台。系统集成了复杂多组分热解反应动力学、非稳态传热传质、催化剂颗粒三维离散化、颗粒运动学分子动力学模拟、高精度数值积分、温度相关物性插值、自适应非结构化网格生成以及有限元数据组装与输出等八大核心模块。

所有代码以 **Python** 实现，统一入口为 `main.py`，零参数可直接运行。

---

## 2. 科学问题与核心公式

### 2.1 反应动力学：分布式活化能模型（DAEM）与三组分并行反应

生物质热解采用 Broido-Shafizadeh 三组分并行反应网络：

$$
\begin{aligned}
\text{纤维素 (B)} &\xrightarrow{k_1} \text{活性纤维素 (A)} \\
\text{半纤维素 (H)} &\xrightarrow{k_4} \text{挥发分 (V)} + \text{焦炭 (C)} \\
\text{木质素 (L)} &\xrightarrow{k_5} \text{挥发分 (V)} + \text{焦炭 (C)} \\
\text{活性纤维素 (A)} &\xrightarrow{k_2} \text{挥发分 (V)} + \text{焦炭 (C)} \\
\text{活性纤维素 (A)} &\xrightarrow{k_3} \text{焦油 + 气体 (TG)}
\end{aligned}
$$

各反应速率遵循 **Arrhenius 定律**：

$$
k_i = A_i \exp\left(-\frac{E_{a,i}}{R T}\right), \quad i = 1, 2, 3, 4, 5
$$

其中 $A_i$ 为指前因子 $[\text{s}^{-1}]$，$E_{a,i}$ 为活化能 $[\text{J}\cdot\text{mol}^{-1}]$，$R = 8.314 \, \text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}$，$T$ 为温度 $[\text{K}]$。

质量守恒方程（ODE 系统）：

$$
\frac{d\mathbf{y}}{dt} = \mathbf{f}(T, \mathbf{y}), \quad \mathbf{y} = [y_B, y_H, y_L, y_A, y_V, y_C, y_{TG}]^\top
$$

### 2.2 传热模型：一维热传导-对流-反应热源耦合

反应器内非稳态能量守恒方程：

$$
\rho C_p \frac{\partial T}{\partial t} = k_{\text{eff}} \frac{\partial^2 T}{\partial x^2} - \rho C_p u \frac{\partial T}{\partial x} + Q_{\text{rxn}}(x, t)
$$

离散化采用**隐式欧拉 + 迎风格式**（保证高 Péclet 数下的数值稳定性）：

$$
\frac{T_i^{n+1} - T_i^n}{\Delta t} = \alpha \frac{T_{i+1}^{n+1} - 2T_i^{n+1} + T_{i-1}^{n+1}}{\Delta x^2} - u \frac{T_i^{n+1} - T_{i-1}^{n+1}}{\Delta x} + \frac{Q_i}{\rho C_p}
$$

其中 $\alpha = k_{\text{eff}} / (\rho C_p)$ 为热扩散系数。

整理为紧凑带状矩阵（Compact Banded Matrix, R8CB 格式）线性系统：

$$
\mathbf{A} \mathbf{T}^{n+1} = \mathbf{b}
$$

带状存储映射（0-based）：
- $a[0, j] = A_{j-1, j}$（上对角元）
- $a[1, j] = A_{j, j}$（对角元）
- $a[2, j] = A_{j+1, j}$（下对角元）

### 2.3 反应热源项

热解反应吸热效应：

$$
Q_{\text{rxn}} = \rho \sum_i k_i(T) \, y_i \, \Delta H_{\text{rxn}}
$$

其中 $\Delta H_{\text{rxn}} > 0$ 表示吸热反应。

### 2.4 分布式活化能模型（DAEM）积分

对于活化能呈 Gaussian 分布的反应：

$$
k_{\text{eff}}(T) = \frac{A}{\sqrt{\pi}} \int_{-\infty}^{+\infty} e^{-x^2} \exp\left(-\frac{E_0 + \sqrt{2}\sigma_E x}{RT}\right) dx
$$

采用 **广义 Gauss-Hermite 求积**计算：

$$
k_{\text{eff}}(T) \approx \frac{A}{\sqrt{\pi}} \sum_{i=1}^{n} w_i \exp\left(-\frac{E_0 + \sqrt{2}\sigma_E x_i}{RT}\right)
$$

### 2.5 颗粒运动学：Velocity Verlet 分子动力学

颗粒运动遵循 Newton 第二定律：

$$
m \frac{d^2 \mathbf{r}}{dt^2} = \mathbf{F}, \quad \mathbf{F} = -\nabla V(r)
$$

采用 **Velocity Verlet 算法**：

$$
\begin{aligned}
\mathbf{r}(t+\Delta t) &= \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \frac{1}{2}\mathbf{a}(t)\Delta t^2 \\
\mathbf{v}(t+\Delta t) &= \mathbf{v}(t) + \frac{1}{2}\Delta t \left[\mathbf{a}(t) + \mathbf{a}(t+\Delta t)\right] \\
\mathbf{a}(t+\Delta t) &= \frac{\mathbf{F}(t+\Delta t)}{m}
\end{aligned}
$$

势能为修正 $\sin^2$ 势：

$$
V(r) = \sin^2\left(\min\left(r, \frac{\pi}{2}\right)\right)
$$

### 2.6 材料物性插值：重心拉格朗日插值

温度相关物性采用 **Chebyshev 第一类节点**上的重心拉格朗日插值：

$$
p(x) = \frac{\sum_{j=0}^{n-1} \frac{w_j}{x - x_j} y_j}{\sum_{j=0}^{n-1} \frac{w_j}{x - x_j}}
$$

Chebyshev 节点：

$$
x_j = \cos\left(\frac{2j+1}{2n}\pi\right), \quad j = 0, 1, \dots, n-1
$$

重心权重：

$$
w_j = (-1)^j \sin\left(\frac{2j+1}{2n}\pi\right)
$$

---

## 3. 原项目到科学问题的映射

| 序号 | 原项目 | 核心算法/结构 | 合成后角色 |
|------|--------|---------------|------------|
| 1 | `829_ode_midpoint_system` | 隐式中点法求解刚性 ODE 系统 | 热解动力学刚性 ODE 的隐式中点法求解器 |
| 2 | `973_r8cb` | 紧凑带状矩阵 LU 分解与求解 | 传热模型隐式离散化产生的大型带状线性系统直接求解 |
| 3 | `464_gen_hermite_exactness` | 广义 Gauss-Hermite 求积 | DAEM 活化能分布积分的高精度计算 |
| 4 | `072_barycentric_interp_1d` | 重心 Chebyshev 插值 | 温度相关导热系数、比热容、密度的物性插值 |
| 5 | `1147_square_integrals` | 单位正方形 Monte Carlo 积分 | 反应器截面上热释放分布的统计平均 |
| 6 | `1036_rk4` | 经典四阶 Runge-Kutta | 热解反应动力学的显式高精度时间推进 |
| 7 | `1351_triangulation_refine_local` | 局部三角剖分细化 | 反应器网格自适应局部细化（质量最差单元 1:4 细分） |
| 8 | `906_pram_view` | 几何变换与旋转配置 | 环面反应器几何变换、旋转矩阵、反射操作 |
| 9 | `317_doughnut_ode` | 环面非线性 ODE | 环面反应器内流体非线性流动路径模拟 |
| 10 | `308_distmesh` | 基于距离函数的网格生成 | 圆柱形反应器的 DistMesh 非结构化网格生成 |
| 11 | `351_fd_to_tec` | 有限差分数据格式转换 | FEM 节点/单元/数值文件的格式化输出 |
| 12 | `745_md_fast` | 快速分子动力学 | 生物质颗粒在流场中的 Velocity Verlet 运动模拟 |
| 13 | `067_ball_grid` | 球体内规则网格点生成 | 催化剂颗粒内部三维离散化网格 |
| 14 | `824_octopus` | 环境检测 | Python 运行环境检测与数值稳定性工具 |
| 15 | `1198_tec_to_fem` | TECPLOT 到 FEM 转换 | FEM 数据组装、质量矩阵与刚度矩阵计算、TECPLOT ASCII 输出 |

---

## 4. 文件结构与功能说明

| 文件名 | 功能描述 | 对应原项目 |
|--------|----------|------------|
| `main.py` | 统一入口，零参数运行，串联所有模块 | — |
| `utils.py` | 安全指数、安全除法、边界检查、雅可比矩阵、条件数估计 | `824_octopus` |
| `geometry_utils.py` | 有向距离函数（SDF）、环面/圆柱/矩形几何、旋转反射矩阵、外心计算 | `906_pram_view`, `317_doughnut_ode` |
| `reactor_mesh.py` | DistMesh 2D 网格生成、球体网格、局部三角细化、网格质量评估 | `308_distmesh`, `067_ball_grid`, `1351_triangulation_refine_local` |
| `pyrolysis_kinetics.py` | Arrhenius 反应网络、RK4/隐式中点法 ODE 求解、环面流动 ODE | `1036_rk4`, `829_ode_midpoint_system`, `317_doughnut_ode` |
| `thermal_model.py` | 一维传热 PDE 隐式离散化、R8CB 带状矩阵 LU 直接求解器 | `973_r8cb` |
| `quadrature_integrator.py` | Gauss-Hermite 求积、DAEM 积分、Monte Carlo 积分、求积误差分析 | `464_gen_hermite_exactness`, `1147_square_integrals` |
| `property_interpolator.py` | Chebyshev 重心拉格朗日插值、导热系数/比热容/密度插值器 | `072_barycentric_interp_1d` |
| `particle_dynamics.py` | Velocity Verlet 分子动力学、$\sin^2$ 势能、力与能量计算 | `745_md_fast` |
| `fem_assembler.py` | FEM 数据组装、节点/单元/数值文件 I/O、质量矩阵、刚度矩阵、TECPLOT ASCII | `351_fd_to_tec`, `1198_tec_to_fem` |

---

## 5. 关键数值方法与工程复杂性

### 5.1 刚性 ODE 的多方法求解
热解动力学具有强烈的刚性特征（反应速率常数跨越多个数量级）。系统同时提供显式 RK4 与隐式中点法，允许用户根据稳定性需求选择方法。隐式中点法采用不动点迭代求解非线性方程，适用于高温快速反应阶段。

### 5.2 高 Péclet 数对流的迎风格式
反应器内气流速度 $u = 0.05 \, \text{m/s}$，网格尺度 $\Delta x \approx 0.025 \, \text{m}$，热扩散系数 $\alpha \approx 5 \times 10^{-7} \, \text{m}^2/\text{s}$，Péclet 数：

$$
\text{Pe} = \frac{u \Delta x}{\alpha} \approx 2560 \gg 2
$$

中心差分在此条件下不稳定。系统采用迎风格式，确保矩阵严格对角占优，数值解稳定无振荡。

### 5.3 带状矩阵直接求解（R8CB）
传热隐式离散化产生三对角系统。系统实现了完整的紧凑带状矩阵 LU 分解（无选主元）与前代/回代求解，映射自 LINPACK/LAPACK 的 R8CB 格式，适用于无需填充（fill-in）保证可逆的物理问题。

### 5.4 边界处理与数值鲁棒性
- 温度场：Dirichlet 边界，自动截断至 $[250, 1500] \, \text{K}$ 物理合理区间
- 质量分数：非负截断与自动归一化，保证 $\sum y_i = 1$
- 反应速率：安全指数函数 `safe_exp`，防止高温下 $e^{-E_a/RT}$ 上溢/下溢
- 矩阵奇异检测：零主元异常抛出，条件数估算用于数值稳定性诊断

---

## 6. 如何运行

```bash
cd Synthesis-project-python/140_synth_project
python main.py
```

无需任何命令行参数。程序将依次执行：
1. 反应器网格生成与局部细化
2. 热解反应动力学求解（RK4 + 隐式中点法）
3. 一维传热模型时间推进
4. 高精度数值积分与误差分析
5. 温度相关物性插值
6. 颗粒运动学模拟
7. FEM 数据组装与 TECPLOT 输出
8. 结果汇总

输出文件（运行后生成）：
- `reactor_result_nodes.txt` — 节点坐标
- `reactor_result_elements.txt` — 单元连接
- `reactor_result_values.txt` — 节点数值
- `reactor_result.dat` — TECPLOT ASCII 格式

---

## 7. 合成后的项目解决什么科学问题

本项目解决了**生物质热解反应器设计中的多物理场耦合预测问题**，具体包括：

1. **反应器尺度**：预测不同升温速率下生物质三组分（纤维素、半纤维素、木质素）的转化率、产物分布（挥发分、焦炭、焦油+气体）。
2. **传热尺度**：模拟反应器轴向温度场演变，考虑吸热反应对温度分布的反馈效应。
3. **颗粒尺度**：通过分子动力学模拟颗粒在流场中的运动与碰撞，评估混合均匀性。
4. **催化剂尺度**：生成催化剂颗粒内部三维网格，为后续孔道传质模拟提供几何基础。
5. **数值验证**：通过 Gauss-Hermite 求积精确度分析与 Monte Carlo 误差估计，量化数值不确定性。

---

## 8. 修改记录

- **新建目录**：`Synthesis-project-python/140_synth_project/`，未修改任何原种子项目文件夹。
- **语言迁移**：所有原 MATLAB 代码迁移/重译为 Python 3。
- **删除可视化**：移除了所有 `matplotlib` / `plot` / `figure` 相关代码。
- **科学增强**：注入了 Arrhenius 反应网络、DAEM 积分、迎风格式传热、Velocity Verlet 分子动力学等大量科学公式。
- **工程鲁棒性**：增加了安全指数、安全除法、边界检查、矩阵条件数估计、质量守恒归一化等边界处理代码。
- **运行验证**：`main.py` 已实际运行通过，零参数无报错。
