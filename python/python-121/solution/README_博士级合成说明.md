# README_博士级合成说明.md

## 项目概述

**项目名称**: 心脏电生理与心律失常多尺度数值模拟系统

**科学领域**: 生物医学 — 心脏电生理与心律失常模拟

**合成语言**: Python

---

## 1. 原项目到科学问题的映射

本项目基于15个原始科研代码项目，将其核心算法融合重构为一个面向**心脏电生理与心律失常模拟**的博士级科学计算系统。每个输入项目都在合成项目中承担了真实角色，无遗漏、无挂名。

### 原项目映射表

| 原项目 | 核心算法 | 在合成项目中的角色 |
|--------|----------|-------------------|
| 202_combo | 组合数学（排列、子集、斯特林数、贝尔数） | `utils.py`：离子通道门控状态枚举、格雷码生成 |
| 928_pwl_interp_2d_scattered | 散乱数据2D分段线性插值 | `mesh_generator.py`：心脏组织电势散乱插值（Shepard方法） |
| 941_quad_monte_carlo | 蒙特卡洛数值积分 | `numerical_integration.py`：电生理参数统计量蒙特卡洛估计 |
| 369_fd2d_predator_prey | 2D反应扩散方程有限差分 | `tissue_reaction_diffusion.py`：Monodomain模型组织电传播求解 |
| 1152_squircle_ode | Squircle ODE守恒系统 | `ion_channel_dynamics.py`：数值守恒性测试与稳定性验证 |
| 803_niederreiter2 | Niederreiter基2低差异序列 | `stochastic_sampler.py`：电导率参数准随机采样 |
| 106_boundary_word_drafter | 多边形边界词与点在多边形内测试 | `mesh_generator.py`：心脏几何边界定义与内部点裁剪 |
| 201_colored_noise | 1/f^α有色噪声生成 | `ion_channel_dynamics.py`：离子通道随机开放噪声 |
| 713_maple_area | Hammersley准蒙特卡洛面积估计 | `stochastic_sampler.py`：心脏组织区域面积估计 |
| 338_errors | 数值误差分析 | `utils.py`：灾难性抵消测试、收敛率估计、条件数分析 |
| 902_power_method | 幂法求主特征值 | `linear_algebra_core.py`：电传播稳定性特征值分析 |
| 239_cvt_1_movie | Centroidal Voronoi Tessellation | `mesh_generator.py`：心脏组织非结构化CVT网格生成 |
| 114_box_flow | Navier-Stokes有限元（质量/刚度矩阵、时间步进） | `tissue_reaction_diffusion.py`：有限差分矩阵组装、ADI/Crank-Nicolson时间步进 |
| 957_quadrilateral_witherden_rule | 四边形高斯求积规则 | `numerical_integration.py`：高阶求积用于有限元积分验证 |
| 988_r8pbu | 对称正定带状矩阵共轭梯度法 | `linear_algebra_core.py`：泊松方程CG求解、大型稀疏系统求解 |

---

## 2. 新增数学物理模型与核心公式

### 2.1 Monodomain模型（组织电传播）

心脏组织的电传播由**单域模型（Monodomain Model）**描述：

$$
\frac{\partial V_m}{\partial t} = \nabla \cdot (D \nabla V_m) - \frac{I_{ion}}{C_m}
$$

其中：
- $V_m$：跨膜电位（mV）
- $D$：有效扩散张量（cm²/ms）
- $I_{ion}$：离子电流密度（μA/cm²）
- $C_m = 1\,\mu\text{F/cm}^2$：膜电容

### 2.2 各向异性扩散张量

心肌具有纤维结构，扩散是**各向异性**的：

$$
\mathbf{D} = D_f \, \mathbf{e}_f \otimes \mathbf{e}_f + D_t \, \mathbf{e}_t \otimes \mathbf{e}_t
$$

其中 $\mathbf{e}_f = (\cos\theta, \sin\theta)$ 为纤维方向，$\mathbf{e}_t = (-\sin\theta, \cos\theta)$ 为横纤维方向，$D_f$ 和 $D_t$ 分别为两个方向的扩散系数。

分量形式：

$$
D_{xx} = D_f \cos^2\theta + D_t \sin^2\theta \\
D_{xy} = (D_f - D_t) \sin\theta \cos\theta \\
D_{yy} = D_f \sin^2\theta + D_t \cos^2\theta
$$

### 2.3 Aliev-Panfilov简化模型

用于组织层面的**反应项**：

$$
\frac{\partial u}{\partial t} = D \nabla^2 u - k u (u - a)(u - 1) - u v
$$

$$
\frac{\partial v}{\partial t} = \varepsilon(u) \left[ -v - k u (u - a - 1) \right]
$$

其中：
- $u \in [0,1]$：无量纲化膜电位
- $v$：恢复变量
- $a \approx 0.1$：兴奋阈值
- $k \approx 8$：非线性强度
- $\varepsilon(u) = \varepsilon_0 + \frac{\mu_1 v}{u + \mu_2}$：恢复速率

### 2.4 Hodgkin-Huxley类型离子通道门控

门控变量 $x$ 满足：

$$
\frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V) x
$$

稳态值与时间常数：

$$
x_\infty = \frac{\alpha_x}{\alpha_x + \beta_x}, \quad \tau_x = \frac{1}{\alpha_x + \beta_x}
$$

**Na⁺激活门（m门）**：

$$
\alpha_m = \frac{0.32(V + 47.13)}{1 - e^{-0.1(V + 47.13)}}, \quad \beta_m = 0.08 e^{-V/11}
$$

**Na⁺失活门（h门）**：

$$
\alpha_h = 0.135 e^{-(V + 80)/6.8}, \quad \beta_h = \frac{3.56}{1 + e^{-0.1(V + 40)}} + 0.0075
$$

### 2.5 离子电流

总离子电流：

$$
I_{ion} = I_{Na} + I_{Ca} + I_K + I_{K1} + I_{Kp} + I_b
$$

**Na⁺电流**：

$$
I_{Na} = G_{Na} \, m^3 h j \, (V - E_{Na})
$$

**Ca²⁺L型电流**：

$$
I_{Ca} = G_{Ca} \, d f \, (V - E_{Ca})
$$

**K⁺延迟整流电流**：

$$
I_K = G_K \, x \, x_i \, (V - E_K)
$$

### 2.6 1/f^α有色噪声（离子通道随机性）

离子通道的随机开放由Langevin方程描述：

$$
d\xi(t) = -\gamma \xi(t) dt + \sqrt{2D} \, dW(t)
$$

功率谱密度 $S(f) \propto 1/f^\alpha$，通过Kasdin方法生成：

$$
h_0 = 1, \quad h_k = h_{k-1} \cdot \frac{\alpha/2 + k - 2}{k - 1}
$$

$$
X = \text{IFFT}\left[ \text{FFT}(h) \cdot \text{FFT}(w) \right]
$$

### 2.7 有限差分离散化

**各向同性五点差分格式**：

$$
\nabla^2 u \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{\Delta x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{\Delta y^2}
$$

**各向异性拉普拉斯**（通量守恒形式）：

$$
\nabla \cdot (D \nabla u)_{i,j} \approx \frac{D_{xx}^{i+1/2}(u_{i+1,j} - u_{i,j}) - D_{xx}^{i-1/2}(u_{i,j} - u_{i-1,j})}{\Delta x^2} + \cdots
$$

### 2.8 时间步进方法

**前向欧拉**（显式）：

$$
u^{n+1} = u^n + \Delta t \left[ D \nabla^2 u^n + f(u^n, v^n) \right]
$$

CFL稳定性条件：$\Delta t \leq \frac{\Delta x^2}{4D}$

**Crank-Nicolson**（隐式）：

$$
\left(I - \frac{\Delta t}{2} D \nabla^2\right) u^{n+1} = \left(I + \frac{\Delta t}{2} D \nabla^2\right) u^n + \Delta t \, f(u^n, v^n)
$$

无条件稳定，通过**共轭梯度法**求解。

**ADI（交替方向隐式）**：

$$
\left(I - \frac{\Delta t}{2} D \frac{\partial^2}{\partial x^2}\right) u^* = \left(I + \frac{\Delta t}{2} D \frac{\partial^2}{\partial y^2}\right) u^n
$$

$$
\left(I - \frac{\Delta t}{2} D \frac{\partial^2}{\partial y^2}\right) u^{n+1} = \left(I + \frac{\Delta t}{2} D \frac{\partial^2}{\partial x^2}\right) u^*
$$

### 2.9 共轭梯度法

求解 $Ax = b$，其中 $A$ 对称正定：

$$
\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}, \quad x_{k+1} = x_k + \alpha_k p_k
$$

$$
r_{k+1} = r_k - \alpha_k A p_k, \quad \beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}
$$

$$
p_{k+1} = r_{k+1} + \beta_k p_k
$$

收敛速率：$\|e_k\|_A / \|e_0\|_A \leq 2 \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^k$

### 2.10 Niederreiter低差异序列

$s$维Niederreiter序列的差异度：

$$
D_N = O\left( \frac{(\log N)^s}{N} \right)
$$

优于纯蒙特卡洛的 $O(1/\sqrt{N})$。

### 2.11 心律失常分析指标

**传导速度**：$v = \frac{\Delta x}{\Delta t_{wavefront}}$

**动作电位时程（APD）**：

$$
APD = t_{repolarization} - t_{depolarization}
$$

**波长**：$\lambda = v \times APD$

**折返稳定性判据**：组织尺寸 $L > \lambda$ 时折返可能稳定存在

---

## 3. 文件结构与修改说明

### 文件清单（共9个.py文件 + 1个文档）

| 文件 | 说明 |
|------|------|
| `main.py` | 统一入口，零参数运行完整模拟流程 |
| `ion_channel_dynamics.py` | 离子通道动力学、有色噪声、Aliev-Panfilov模型、单细胞AP模拟 |
| `tissue_reaction_diffusion.py` | 反应扩散方程求解（前向欧拉、Crank-Nicolson、ADI） |
| `mesh_generator.py` | CVT网格生成、多边形边界处理、散乱数据插值 |
| `numerical_integration.py` | Witherden高斯求积、蒙特卡洛积分、矩计算 |
| `stochastic_sampler.py` | Niederreiter序列、Hammersley序列、QMC面积估计、电导率采样 |
| `linear_algebra_core.py` | 带状矩阵CG求解、幂法特征值分析、泊松方程求解 |
| `electrophysiology_simulator.py` | 集成模拟器、心律失常检测、APD/波长/风险指数计算 |
| `utils.py` | 组合数学工具、数值误差分析、收敛率估计 |
| `README_博士级合成说明.md` | 本文档 |

### 修改说明

- **所有原项目代码均经过Python化重构**，从MATLAB语法转换为Python/NumPy实现
- **删除了所有可视化相关代码**（plot、figure、imshow等）
- **保留了所有核心数值算法**，并注入了心脏电生理科学背景
- **新增了大量边界处理和数值鲁棒性代码**：
  - 门控变量截断到 $[0,1]$ 区间
  - 膜电位限制在 $[-100, 60]$ mV
  - CG求解器的零除保护
  - 三对角求解器的奇异矩阵处理

---

## 4. 合成后的项目能解决什么科学问题

1. **心肌动作电位生成与传播模拟**：从单细胞离子通道到组织层面的电信号传播
2. **心律失常机制分析**：
   - 传导速度异常检测
   - 折返活动（reentrant activity）识别
   - 螺旋波（spiral wave）相位奇点定位
3. **电生理参数敏感性研究**：恢复变量速率 $\varepsilon$、扩散系数 $D$ 对波形的影响
4. **数值方法验证**：
   - 不同时间步进方案（显式/隐式/ADI）的精度与效率比较
   - 守恒量监测（Squircle ODE）
   - 收敛速率分析
5. **随机电生理效应**：离子通道噪声对动作电位时程离散度的影响
6. **组织几何效应**：不同纤维结构（平行/旋转/放射）对传导速度的影响

---

## 5. 如何运行

```bash
cd "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/121_synth_project"
python main.py
```

程序将自动执行以下9个阶段：
1. 组合数学与误差分析验证
2. 数值积分验证（高斯求积 + 蒙特卡洛）
3. 准随机采样与参数探索
4. 心脏网格生成
5. 线性代数求解器测试（CG + 幂法）
6. 单细胞离子通道动力学验证
7. 组织层面反应扩散模拟（核心）
8. 散乱数据插值验证
9. 参数敏感性研究

运行结束后将输出完整的心律失常分析摘要。

---

## 6. 科学复杂度说明

本项目的科学复杂度达到博士级标准：

- **多尺度建模**：从离子通道门控（毫秒级）到组织传播（厘米级）的跨尺度耦合
- **高阶数值方法**：ADI隐式格式、Crank-Nicolson、共轭梯度法、Thomas算法
- **各向异性扩散**：基于真实心肌纤维结构的方向依赖扩散张量
- **随机微分方程**：1/f^α有色噪声模拟离子通道随机动力学
- **非结构化网格**：CVT生成适应心脏几何的Voronoi网格
- **低差异序列**：Niederreiter/Hammersley准蒙特卡洛用于高效参数采样
- **稳定性分析**：幂法特征值分析用于判断电传播稳定性
- **非线性动力学**：Aliev-Panfilov双变量反应扩散系统的非线性行为分析
