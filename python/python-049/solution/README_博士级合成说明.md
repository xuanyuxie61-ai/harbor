# 博士级合成项目说明：海底地震引发海啸的非线性浅水波方程数值模拟

## 项目概述

本项目是一个面向**地球物理前沿科学问题**——**海啸生成与传播数值模拟**——的博士级计算系统。项目融合了 **15 个种子科研代码项目**的核心算法，构建了一个具备不确定性量化、自适应网格覆盖、高精度能量守恒监测的完整数值模拟流程。

项目围绕以下核心科学问题展开：

> **如何利用高精度数值方法，模拟海底地震断层破裂引发的海啸生成、传播与能量演化过程，并量化断层参数不确定性对海啸预报的影响？**

---

## 一、原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 在合成项目中的角色 | 对应文件 |
|:---:|---|---|---|---|
| 1 | `064_backward_euler_fixed` | 固定点迭代后向 Euler ODE 求解器 | 浅水波方程时间积分的核心算法：将后向 Euler 固定点迭代应用于非线性浅水波方程的隐式时间步进 | `tsunami_pde_solver.py` |
| 2 | `1372_unicycle` | 随机单圈排列生成 | 蒙特卡洛不确定性量化中的随机断层参数排列与随机相位生成 | `bathymetry_sampling.py` |
| 3 | `1283_tough_ode` | 复杂刚性四变量 ODE 系统 | 地震断层破裂动力学模型：将 tough_ode 的非线性结构改造为速率-状态摩擦定律下的断层滑动演化 | `fault_dynamics.py` |
| 4 | `042_asa144` (rcont) | 给定行列边界的随机二维表生成 | 随机海底地形采样：保持边缘分布约束下的随机地形矩阵生成 | `bathymetry_sampling.py` |
| 5 | `857_pendulum_comparison_ode` | 单摆非线性 ODE 与能量守恒监测 | 断层滑动能量守恒检验：借鉴单摆能量守恒思想监测数值计算的物理一致性 | `fault_dynamics.py` |
| 6 | `1196_task_division` | 任务在处理器间的均衡分配 | 多子域并行计算的任务调度：将模拟域划分为条带子域并负载均衡分配 | `parallel_scheduler.py` |
| 7 | `1356_trig_interp` | 三角基函数插值（周期性边界） | 海啸波场周期性边界处理：使用三角 Cardinal 基函数保证经度方向边界连续性 | `mesh_interpolation.py` |
| 8 | `1147_square_integrals` | 正方形区域单核积分 | 海啸能量计算中的正方形子域积分基准检验 | `energy_quadrature.py` |
| 9 | `536_hilbert_curve_3d` | 3D Hilbert 曲线空间填充 | 计算网格点的 Hilbert 空间排序优化：提升有限差分计算的缓存局部性 | `hilbert_mesh_ordering.py` |
| 10 | `956_quadrilateral_surface_display` | 四边形网格双线性插值 | 不同分辨率网格间的波场与地形映射插值 | `mesh_interpolation.py` |
| 11 | `1125_sphere_positive_distance` | 球面上随机点距离统计 | 地球曲率效应校正：震中与网格点的大圆距离计算 | `spherical_geodesics.py` |
| 12 | `739_matrix_chain_brute` | 矩阵链乘法最优括号化穷举 | 多步状态转移矩阵的最优累积计算序列优化 | `matrix_chain_optimizer.py` |
| 13 | `1318_triangle_symq_rule_original` | 三角形对称求积规则 | 海啸能量高精度积分：将网格单元剖分为三角形进行高精度数值积分 | `energy_quadrature.py` |
| 14 | `519_hermite_exactness` | Hermite 求积规则精度检验 | 高斯型初始海面形变的积分精度检验与数值求积基准 | `energy_quadrature.py` |
| 15 | `1389_variomino` (variomino_matrix) | 变体多米诺平铺矩阵系统 | 自适应多分辨率网格覆盖：将不同尺寸"瓦片"自适应覆盖计算域 | `variomino_adaptive_cover.py` |

---

## 二、新增数学物理模型与核心公式

### 2.1 Okada 弹性半空间位错模型（初始条件）

海底地震断层滑动导致的海面初始位移由 **Okada (1985) 弹性半空间位错模型**计算：

$$u_i(\mathbf{x}) = \frac{1}{2\pi} \iint_\Sigma \Delta u_j \left[ \frac{\partial}{\partial \xi_j}\left(\frac{1}{r}\right) - \frac{1}{1-\nu} \frac{\partial}{\partial x_i} \frac{\partial J}{\partial \xi_j} \right] d\Sigma$$

其中 $u_i$ 为位移分量，$\Delta u_j$ 为断层滑动矢量，$\nu$ 为泊松比，$r$ 为源点到场点的距离。

对于纯走滑和倾滑分量，垂直位移可分解为：

$$u_z = \frac{U}{2\pi} \left[ u_z^{SS} \cos\lambda + u_z^{DS} \sin\lambda \right]$$

其中 $U$ 为滑动量，$\lambda$ 为滑动角（rake），$SS$ 和 $DS$ 分别为单位走滑和单位倾滑的位移核函数。

### 2.2 非线性浅水波方程（Saint-Venant 方程组）

海啸在海洋中的传播由**非线性浅水波方程**描述：

**连续性方程：**
$$\frac{\partial \eta}{\partial t} + \nabla \cdot \left[(h + \eta) \mathbf{u}\right] = 0$$

**动量方程（x 方向）：**
$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + g\frac{\partial \eta}{\partial x} + \frac{\tau_{bx}}{\rho(h+\eta)} = 0$$

**动量方程（y 方向）：**
$$\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + g\frac{\partial \eta}{\partial y} + \frac{\tau_{by}}{\rho(h+\eta)} = 0$$

其中：
- $\eta(x,y,t)$：海面高度异常（m）
- $h(x,y)$：静水深（m）
- $\mathbf{u} = (u, v)$：水平流速（m/s）
- $g = 9.81$：重力加速度（m/s²）
- $\tau_b$：底摩擦应力

### 2.3 底摩擦模型（二次摩擦定律）

$$\tau_{bx} = \rho C_d |\mathbf{u}| u, \quad \tau_{by} = \rho C_d |\mathbf{u}| v$$

其中 $C_d$ 为底摩擦系数，$|\mathbf{u}| = \sqrt{u^2 + v^2}$。

### 2.4 速率-状态摩擦定律（断层动力学）

断层滑动遵循 **Dieterich-Ruina 速率-状态摩擦定律**：

$$\mu(V, \theta) = \mu_0 + a \ln\frac{V}{V_0} + b \ln\frac{\theta V_0}{D_c}$$

**状态变量演化（老化定律）：**
$$\frac{d\theta}{dt} = 1 - \frac{V\theta}{D_c}$$

**应力演化：**
$$\frac{d\tau}{dt} = k(V_{pl} - V)$$

其中 $a < b$ 对应速度弱化（地震不稳定滑动）。

### 2.5 球面测地距离（Haversine 公式）

考虑地球曲率的海啸传播距离：

$$a = \sin^2\frac{\Delta\phi}{2} + \cos\phi_1 \cos\phi_2 \sin^2\frac{\Delta\lambda}{2}$$

$$c = 2 \cdot \text{atan2}(\sqrt{a}, \sqrt{1-a})$$

$$d = R_{\text{Earth}} \cdot c$$

### 2.6 海啸能量

**势能密度：**
$$E_p = \frac{1}{2} \rho g \eta^2$$

**动能密度：**
$$E_k = \frac{1}{2} \rho (h+\eta)(u^2 + v^2)$$

**总能量：**
$$E_{\text{total}} = \iint_\Omega \left[ E_p + E_k \right] dx\,dy$$

### 2.7 格林定律（波幅衰减）

海啸从深海传播到浅海时，波幅遵循格林定律衰减：

$$\eta \propto (gh)^{-1/4}$$

### 2.8 深水波速

$$c = \sqrt{g \cdot h}$$

### 2.9 von Kármán 地形功率谱

随机海底地形服从 von Kármán 谱：

$$P(k) = C \cdot (k^2 + k_0^2)^{-H-1}$$

其中 $H$ 为 Hurst 指数（~0.7–0.9）。

### 2.10 数值离散格式

**空间离散**：交错网格（Arakawa C-grid）
- $\eta$ 定义在整数网格点 $(i, j)$
- $u$ 定义在 $(i+1/2, j)$
- $v$ 定义在 $(i, j+1/2)$

**时间离散**：线性化后向 Euler 分裂格式

1. **动量预测（显式）**：
   $$\mathbf{u}^* = \mathbf{u}^n + \Delta t \cdot \left[ -\mathbf{u}^n \cdot \nabla \mathbf{u}^n - \frac{\boldsymbol{\tau}_b^n}{\rho H^n} \right]$$

2. **压强修正（隐式）**：
   $$\mathbf{u}^{n+1} = \mathbf{u}^* - g\Delta t \cdot \nabla \eta^{n+1}$$

3. **连续性方程（代入后得到椭圆方程）**：
   $$\eta^{n+1} - g\Delta t^2 \nabla \cdot (H \nabla \eta^{n+1}) = \eta^n - \Delta t \nabla \cdot (H \mathbf{u}^*)$$

椭圆方程使用**固定点迭代**求解（来源于 `backward_euler_fixed`）：
$$\eta^{(k+1)} = \text{RHS} + g\Delta t^2 \nabla \cdot (H \nabla \eta^{(k)})$$

---

## 三、项目文件结构

```
049_synth_project/
├── main.py                          # 统一入口，零参数可运行
├── fault_dynamics.py                # 断层破裂动力学（tough_ode + pendulum）
├── okada_dislocation.py             # Okada 弹性位错模型
├── tsunami_pde_solver.py            # 非线性浅水波方程求解器（backward_euler）
├── bathymetry_sampling.py           # 随机海底地形生成（rcont + unicycle_random）
├── spherical_geodesics.py           # 球面测地距离（sphere_positive_distance）
├── mesh_interpolation.py            # 网格插值系统（quadrilateral + trig_interp）
├── hilbert_mesh_ordering.py         # Hilbert 曲线空间排序（hilbert_curve_3d）
├── energy_quadrature.py             # 能量高精度积分（square + triangle + hermite）
├── variomino_adaptive_cover.py      # 自适应多分辨率网格覆盖（variomino_matrix）
├── matrix_chain_optimizer.py        # 矩阵链最优计算序列（matrix_chain_brute）
├── parallel_scheduler.py            # 并行任务调度（task_division）
└── README_博士级合成说明.md          # 本文档
```

共 **12 个 `.py` 文件**，超过要求的 8 个。

---

## 四、合成后的项目能够解决什么科学问题

1. **海啸生成机制**：基于 Okada 弹性位错模型，从地震断层参数（走向、倾角、滑动角、滑动量）计算初始海面位移场。

2. **海啸传播模拟**：基于非线性浅水波方程，模拟海啸在复杂海底地形上的传播过程，包括波速变化、波幅衰减和能量演化。

3. **地球曲率效应**：通过 Haversine 公式校正长距离传播的球面几何效应，适用于跨洋海啸模拟。

4. **能量守恒检验**：利用三角形对称求积和 Hermite 求积精度验证，严格监测数值方法的总能量守恒性。

5. **不确定性量化**：通过蒙特卡洛随机采样断层参数，评估海啸最大波幅的不确定性范围和置信区间。

6. **自适应计算优化**：基于 Hilbert 曲线排序和多分辨率网格覆盖，优化大规模并行计算的内存局部性和计算效率。

---

## 五、合成后的项目如何运行

### 环境要求
- Python 3.8+
- NumPy
- SciPy（仅用于 Okada 模型中的高斯平滑）

### 运行方式

```bash
cd Synthesis-project-python/049_synth_project
python main.py
```

**零参数**，所有物理参数内置，自动完成从断层破裂 → 初始位移 → 数值求解 → 能量检验 → 不确定性量化的完整流程。

### 运行输出示例

程序将依次执行以下步骤并输出结果：
1. 断层破裂动力学模拟（30 秒滑动演化）
2. Okada 弹性位错模型计算（81×81 网格）
3. 随机海底地形生成
4. 球面测地距离校正
5. 网格插值与周期边界处理
6. Hilbert 曲线空间排序优化
7. 多子域并行任务调度
8. 自适应多分辨率网格覆盖
9. 矩阵链最优计算序列
10. 非线性浅水波方程数值求解（60 分钟模拟时长）
11. 能量积分与守恒检验（Hermite 求积精度验证）
12. 海啸传播特征分析（波速、格林定律）
13. 蒙特卡洛不确定性量化（20 次模拟）

---

## 六、质量检查清单

- [x] 原目录未被修改
- [x] 合成后的项目为 Python 语言
- [x] 新目录完整包含合成后的项目（12 个 `.py` 文件 + 1 个文档）
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] **15 个输入项目均已真实融入合成项目**，无遗漏、无挂名
- [x] **`main.py` 已实际运行通过**，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性（NaN/Inf 检查、物理量限制、sponge 边界）
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 无可视化代码
