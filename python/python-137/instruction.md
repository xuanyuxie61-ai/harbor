# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 项目描述：结晶过程成核与生长动力学仿真系统

## 1. 项目背景与总体目标
本项目是一个面向化学工程结晶过程的博士级研究代码，围绕“程序冷却 + 混沌混合”条件下的**晶体尺寸分布 (CSD) 演化**展开。核心任务是通过**人口平衡方程 (PBE)** 的求解、贝叶斯参数反演（DREAM MCMC）和高维不确定性量化（Smolyak 稀疏网格）的耦合计算框架，分析冷却策略与混合状态对成核及生长动力学的影响，实现工艺稳健性评估与参数推断。

系统被拆分为多个职责单一的 Python 模块，`main.py` 作为统一的入口脚本串联所有功能，并运行自包含的演示计算。

## 2. 文件结构与模块职责

### `main.py`
- **职责**：项目主入口，演示整个计算流程。
- **内容**：依次调用各模块进行特殊函数验证、冷却曲线对比、混沌混合分析、成核/生长模型比较、人口平衡方程求解、CSD 统计分析、单纯形积分、稀疏网格 UQ 和 MCMC 参数推断，并将结果通过 `data_io` 保存。脚本直接运行，不依赖额外配置文件。

### `cooling_profile.py`
- **职责**：定义多种程序冷却曲线，并提供溶解度与过饱和度计算。
- **核心函数**：
  - `linear_cooling`: 线性降温曲线 `T(t) = T0 + (Tf - T0)·t/t_total`。
  - `natural_cooling`: 牛顿冷却定律指数降温。
  - `sawtooth_cooling`: 锯齿波温控，在基础温度上叠加周期性波动，用于诱导受控成核。
  - `optimal_cooling_polynomial`: 多项式最优冷却 `T(t) = T0 + (Tf - T0)·(t/t_total)^p`，基于矩方法推导的恒定过饱和度策略。
  - `solubility_vanthoff` / `supersaturation`: 基于 van’t Hoff 方程计算饱和浓度和过饱和度。
- **关键依赖**：仅依赖 NumPy。
- **算法要点**：锯齿波通过模运算生成归一化波形；最优冷却指数 p 取决于结晶动力学参数。

### `chaotic_mixing.py`
- **职责**：基于 Chen 混沌吸引子模拟搅拌结晶器中的局部过饱和度涨落。
- **核心函数**：
  - `chen_attractor_rhs`: Chen 混沌系统的右端项 `(dx/dt, dy/dt, dz/dt)`。
  - `generate_chaotic_mixing_trajectory`: 使用 `scipy.integrate.solve_ivp` 生成吸引子轨迹。
  - `map_chen_to_supersaturation_fluctuation`: 将混沌状态 (x, y, z) 映射为过饱和度波动 Δσ，模型简化为线性组合 `Δσ ≈ scale_c·y - scale_T·x`。
  - `mixing_enhanced_nucleation_rate`: 计算混沌条件下的时间平均成核率，利用经典成核理论中的指数关系并通过对局部过饱和度的时均处理体现“爆发成核”效应。
- **关键依赖**：NumPy, SciPy (solve_ivp)。

### `nucleation_model.py`
- **职责**：实现成核动力学模型，包括经典成核、二级成核、随机事件模拟及尺寸依赖生长的解析解。
- **核心函数**：
  - `lcg_park_miller`: Park-Miller 线性同余生成器，用于随机模拟。
  - `classical_nucleation_rate`: 基于经典成核理论 (CNT)，计算 `B = A·exp(-ΔG*/(k_B T))`，其中 ΔG* 是表面能和过饱和度的函数。
  - `secondary_nucleation_rate`: 二级成核，依赖于悬浮密度和过饱和度。
  - `total_nucleation_rate`: 两种成核机制的线性叠加。
  - `critical_nucleus_radius`: 计算临界核半径。
  - `stochastic_nucleation_events`: 利用伪随机数模拟离散成核事件（泊松过程或高斯近似）。
  - `analytical_size_dependent_growth_law`: 使用 Lambert W 函数求解特定生长律（如 `G ∝ L/(1+αL)`）下的特征线方程。
- **关键依赖**：`special_functions.lambert_w`，NumPy。

### `growth_kinetics.py`
- **职责**：提供多种晶体生长速率模型。
- **核心函数**：
  - `power_law_growth`: 温度活化的幂律生长 `G = k_g0·exp(-E_g/(RT))·σ^g`。
  - `size_dependent_growth`: 包含尺寸修正因子 `(1+αL)^β`。
  - `two_step_growth`: 两步模型（扩散-表面集成联合控制），采用阻力串联形式 `1/G = 1/k_d + 1/(k_r·σ^{g_r})`。
  - `bcf_spiral_growth`: BCF 螺旋位错模型 `G = A·σ²·tanh(B/σ)`，体现低/高过饱和度下的不同幂律行为。
  - `growth_rate_dispersion`: 基于对数正态分布模拟生长速率的统计分散。
- **关键依赖**：NumPy。

### `population_balance.py`
- **职责**：基于矩方法求解人口平衡方程 (PBE)，计算尺寸分布的时间演化。
- **核心类/函数**：
  - `newton_cotes_open_weights` / `quadrature_integrate`: 一维 Newton–Cotes Open 求积规则，用于矩方程中的数值积分。
  - `PopulationBalanceSolver` 类：
    - 构造时接收物性参数（密度、形状因子、动力学参数等）。
    - `_rhs_moments`: 矩方程右端项，耦合成核、尺寸依赖生长和质量平衡，内部调用 `cooling_profile` 和 `nucleation_model` 的函数。
    - `solve`: 包装 `scipy.integrate.solve_ivp`，初始条件可选默认高斯分布。
    - `get_moments_at_time`: 从解中提取指定时刻的矩量。
    - `get_csd_parameters`: 假设 CSD 服从对数正态分布，从低阶矩反算分布参数（均值、CV 等）。
- **关键依赖**：SciPy, NumPy，`growth_kinetics`, `nucleation_model`, `cooling_profile`。

### `csd_analysis.py`
- **职责**：晶体尺寸分布 (CSD) 的分析工具，包括离散化、衍射反演和统计矩计算。
- **核心函数**：
  - `kmeans_1d`: 一维 K-Means 聚类，用于将数据点分配到指定数量的类别。
  - `discretize_csd_kmeans`: 将连续 CSD 离散化为有限尺寸类，通过加权采样和聚类确定代表尺寸和粒数密度。
  - `diffraction_inversion_feret`: 基于 Fraunhofer 衍射的粒度反演。利用 `special_functions.fraunhofer_diffraction_particle_size` 构造线性方程组，并用非负最小二乘法 (NNLS) 求解粒度分布。
  - `csd_statistical_moments`: 计算分布的均值、标准差、偏度和峰度。
- **关键依赖**：NumPy, SciPy (nnls), `special_functions.fraunhofer_diffraction_particle_size`。

### `special_functions.py`
- **职责**：实现数值工作所需的特殊函数。
- **核心函数**：
  - `lambert_w`: Lambert W 函数，基于分段初值近似和 Halley 迭代精化，支持主分支和下分支。
  - `fresnel_integrals`: Fresnel 积分 C(x) 和 S(x)，采用分段策略：小参数用幂级数，中等参数用反向递推辅助函数，大参数用渐近展开。
  - `fraunhofer_diffraction_particle_size`: 球形颗粒 Fraunhofer 衍射光强分布 `[2·J_1(x
