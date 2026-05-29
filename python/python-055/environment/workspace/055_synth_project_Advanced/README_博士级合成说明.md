# 海底地形声纳反演系统 — 博士级合成说明

## 一、项目概述

本项目围绕**海洋科学：海底地形声纳反演**这一前沿领域，基于用户提供的 15 个种子科研代码项目，融合构建了一个博士级海洋地球物理计算系统。系统模拟多波束声纳（Multibeam Sonar）在分层海洋中的完整探测-处理-反演流程，涵盖声线追踪、信号处理、混响建模、地形插值、谱元声学正演、不确定性量化与最优采样规划等核心环节。

---

## 二、科学问题定义

### 2.1 问题背景

海底地形（Bathymetry）是海洋学、地球物理学与海洋工程的核心基础数据。多波束声纳通过向海底发射宽扇形声学脉冲并接收回波，利用双程传播时间（Two-Way Travel Time, TTW）反演水深。然而，真实海洋环境存在以下挑战：

1. **声速剖面分层效应**：海水声速 $c(z)$ 随深度变化，导致声线弯曲（折射），简单的垂直假设 $z = c \cdot t_{tw}/2$ 产生显著误差；
2. **混响干扰**：海底粗糙散射、水体体积散射与海面多次反射构成强混响背景，降低回波检测信噪比；
3. **测量误差传播**：声速、角度与时间测量误差通过非线性反演模型耦合放大；
4. **采样资源约束**：船舶续航与作业时间有限，需在覆盖效率与分辨率之间取得最优平衡。

### 2.2 核心科学问题

> **问题**：在分层海洋声速剖面与复杂海底地形条件下，如何构建一套从声波传播物理、信号处理到统计反演的完整计算框架，实现高精度、高鲁棒性的海底数字高程模型（DEM）重建，并量化其不确定性？

---

## 三、原项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 合成后角色 | 科学映射说明 |
|:---:|:---|:---|:---|:---|
| 1 | `1307_triangle_integrals` | 三角形单项式解析积分、Hammer 数值积分 | `seafloor_geometry.py` | 海底三角网面片上的几何量计算（面积、形心、法向量、点到面片距离），单项式积分用于海底反射系数面积分 |
| 2 | `293_disk_grid` | 圆盘内规则网格生成 | `adaptive_beam_sampler.py` | 单波束 footprint 的圆盘形采样点离散化，用于计算 footprint 内的平均深度与回波强度 |
| 3 | `1393_vin` | 加权校验和算法 | `data_validator.py` | 声纳数据包完整性校验与物理边界一致性检测（深度、声速、角度范围） |
| 4 | `264_cvtp` | 周期边界 CVT（Lloyd 迭代） | `adaptive_beam_sampler.py` | 测区内的最优波束 footprint 中心分布，最小化覆盖不均匀度 |
| 5 | `1432_zero_rc` | Brent 反通信求根 | `acoustic_ray_tracer.py` | 声线追踪中射线与海底地形函数的精确交点计算 |
| 6 | `805_nintlib` | 多维蒙特卡洛积分 | `uncertainty_quantifier.py` | 输入误差分布下深度反演输出的统计量估计 |
| 7 | `1426_xyzl_display` | 原始为可视化 | — | **已删除可视化内容**，其数据结构思想用于数据包载荷组织 |
| 8 | `707_mackey_glass_dde` | Mackey-Glass 延迟微分方程 | `reverberation_model.py` | 海洋混响的多途非线性衰减动力学建模 |
| 9 | `021_asa_geometry_2011` | 线参数化、三角形距离、含点判定 | `acoustic_ray_tracer.py` + `seafloor_geometry.py` | 射线-面片相交检测、点到三角形最短距离、几何边界判定 |
| 10 | `626_knapsack_random` | 随机子集生成 | `adaptive_beam_sampler.py` | 在时间与能耗约束下的最优波束子集选择（0-1 背包问题） |
| 11 | `1268_toms243` | 复数对数稳定算法 | `signal_processor.py` | 声纳信号复包络频谱的对数幅度计算，避免溢出/下溢 |
| 12 | `1071_shepard_interp_1d` | 反距离权重（Shepard）插值 | `bathymetry_interpolator.py` + `signal_processor.py` | 稀疏测深点→连续海底 DEM 的二维空间插值；频谱包络精化插值 |
| 13 | `1361_truncated_normal_rule` | 截断正态分布矩与求积 | `uncertainty_quantifier.py` | 有界物理量测量误差的截断正态采样与矩计算 |
| 14 | `399_fem1d_spectral_numeric` | 一维谱有限元方法 | `spectral_acoustic_solver.py` | 分层海洋中一维 Helmholtz 方程的谱元求解，正演声压场 |
| 15 | `567_hypersphere_positive_distance` | 正超球面随机距离统计 | `beam_statistics.py` | 多波束三维指向向量分布的均匀性评估与盲区风险量化 |

---

## 四、新增数学物理模型与核心公式

### 4.1 海洋声速剖面模型（Munk 变体）

$$c(z) = c_0 + g \cdot z + \Delta c \cdot \exp\!\left(-\frac{(z-z_1)^2}{2\sigma^2}\right)$$

其中 $c_0 = 1500\,\text{m/s}$，$g = 0.015\,\text{s}^{-1}$，$\Delta c = 30\,\text{m/s}$，$z_1 = 1000\,\text{m}$，$\sigma = 500\,\text{m}$。

### 4.2 Snell 定律与射线方程

射线参数守恒：
$$p = \frac{\sin\theta(z)}{c(z)} = \text{const}$$

时间参数化的射线 ODE 组（RK4 数值求解）：
$$\frac{dx}{dt} = c(z)\cos\theta, \quad \frac{dz}{dt} = c(z)\sin\theta, \quad \frac{d\theta}{dt} = -\frac{dc}{dz}\sin\theta$$

### 4.3 Helmholtz 方程（一维谱 FEM）

$$\frac{d^2p}{dz^2} + k^2(z)\,p = f(z), \qquad k(z) = \frac{\omega}{c(z)} = \frac{2\pi f}{c(z)}$$

弱形式：
$$-\int_0^H \frac{dp}{dz}\frac{dv}{dz}\,dz + \int_0^H k^2(z)\,p\,v\,dz = \int_0^H f(z)\,v\,dz$$

基函数：$\phi_i(z) = (z/H)^i \cdot (z/H - 1)$，满足齐次 Dirichlet 边界 $\phi_i(0)=\phi_i(H)=0$。

### 4.4 Mackey-Glass 混响模型

$$\frac{dx}{dt} = \frac{\beta \cdot x(t-\tau)}{1 + x(t-\tau)^n} - \gamma \cdot x(t)$$

参数：$\beta$（反馈增益），$\gamma$（衰减），$\tau$（多途延迟），$n$（非线性指数）。

### 4.5 Shepard 插值（反距离权重）

$$w_i(\mathbf{x}) = \frac{\|\mathbf{x}-\mathbf{x}_i\|^{-p}}{\sum_j \|\mathbf{x}-\mathbf{x}_j\|^{-p}}, \qquad z(\mathbf{x}) = \sum_i w_i(\mathbf{x}) \, z_i$$

### 4.6 误差传播公式

深度反演模型（简化几何）：
$$z = \frac{c \cdot t_{tw} \cdot \cos\theta}{2}$$

Gauss 误差传播：
$$\sigma_z^2 \approx \left(\frac{\partial z}{\partial c}\right)^2\sigma_c^2 + \left(\frac{\partial z}{\partial \theta}\right)^2\sigma_\theta^2 + \left(\frac{\partial z}{\partial t}\right)^2\sigma_t^2$$

### 4.7 截断正态分布矩递推

$$I_r = (r-1)I_{r-2} - \frac{\beta^{r-1}\phi(\beta) - \alpha^{r-1}\phi(\alpha)}{\Phi(\beta)-\Phi(\alpha)}$$

其中 $\alpha = (a-\mu)/\sigma$，$\beta = (b-\mu)/\sigma$。

### 4.8 CVT 最优采样目标泛函

$$F(\{z_i\}) = \sum_{i} \int_{V_i} \rho(\mathbf{x}) \|\mathbf{x} - z_i\|^2 \,dA$$

Lloyd 迭代：$z_i^{(k+1)} = \frac{1}{|V_i|}\int_{V_i} \mathbf{x}\,dA$。

---

## 五、项目文件结构

```
055_synth_project/
├── main.py                          # 统一入口，零参数运行
├── data_validator.py                # 数据校验（1393_vin）
├── beam_statistics.py               # 波束统计（567_hypersphere_positive_distance）
├── seafloor_geometry.py             # 三角网几何（1307_triangle_integrals + 021_asa_geometry_2011）
├── acoustic_ray_tracer.py           # 声线追踪（1432_zero_rc + 021_asa_geometry_2011）
├── reverberation_model.py           # 混响建模（707_mackey_glass_dde）
├── signal_processor.py              # 信号处理（1268_toms243 + 1071_shepard_interp_1d）
├── bathymetry_interpolator.py       # 地形插值（1071_shepard_interp_1d 扩展）
├── adaptive_beam_sampler.py         # 自适应采样（264_cvtp + 293_disk_grid + 626_knapsack_random）
├── uncertainty_quantifier.py        # 不确定性量化（805_nintlib + 1361_truncated_normal_rule）
├── spectral_acoustic_solver.py      # 谱 FEM 声学正演（399_fem1d_spectral_numeric）
└── README_博士级合成说明.md          # 本文档
```

---

## 六、边界处理与数值鲁棒性

1. **深度非负保护**：声线追踪中若 $z < 0$，强制 $z = 0$ 并取海面反射 $\theta \to -\theta$；
2. **Brent 求根退化回退**：若区间不变号，回退到线性插值中点；
3. **矩阵奇异处理**：谱 FEM 矩阵若条件数过大或奇异，自动切换最小二乘求解；
4. **插值零距离保护**：Shepard 插值中若插值点与数据点重合，直接返回该点值；
5. **截断采样边界**：拒绝采样生成截断正态样本时，确保样本严格落在 $[a,b]$ 内；
6. **混响非负约束**：Mackey-Glass DDE 求解中若 $x_{new} < 0$，强制置零；
7. **声速范围校验**：数据验证模块确保声速在 $[1400, 1600]\,\text{m/s}$ 物理范围内。

---

## 七、如何运行

```bash
cd Synthesis-project-python/055_synth_project
python main.py
```

程序将自动执行以下流程：
1. 生成 30 个模拟声纳数据包并校验；
2. 分析 21 波束扇形覆盖的统计均匀性；
3. 用 CVT + 背包优化规划 48 候选波束中的最优子集；
4. 追踪 5 条不同掠射角的声线至起伏海底；
5. 处理含混响的 LFM 回波信号并检测峰值时间；
6. 生成 4 模式复合混响包络；
7. 对 80 个稀疏测深点进行 Shepard 插值并估计梯度/曲率；
8. 构建海底三角网并计算几何量；
9. 用谱 FEM 求解 Helmholtz 方程并估计 L²/H¹ 误差；
10. 蒙特卡洛误差传播与截断正态矩计算；
11. 三维蒙特卡洛积分验证。

---

## 八、合成方法总结

本项目严格遵循"每一个输入项目都必须在合成项目中承担真实角色"的原则：
- **数值积分类**（triangle_integrals, nintlib, truncated_normal_rule）转化为海底面片积分、MC 不确定性量化和截断正态矩计算；
- **几何与网格类**（disk_grid, cvtp, asa_geometry）转化为波束 footprint 离散化、最优采样规划和射线-地形相交检测；
- **信号与方程类**（toms243, shepard_interp, fem1d_spectral, mackey_glass）转化为声纳信号复包络处理、地形插值、声学 Helmholtz 正演和混响 DDE 建模；
- **优化与统计类**（knapsack_random, hypersphere_positive_distance, zero_rc, vin）转化为波束子集背包优化、覆盖均匀度统计、射线求根和数据包校验。

全部代码已删除可视化相关内容，以纯数值计算与文本输出形式呈现。
