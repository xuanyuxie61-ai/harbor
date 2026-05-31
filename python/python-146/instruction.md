# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：脉冲神经网络多物理场耦合仿真系统 (python-146)

本项目通过多个物理与数学模型的融合，构建一个用于模拟、编码、传播和解码神经信号的仿真系统。项目包含8个可被替换的功能模块，以及一个整合入口文件 `main.py`。各模块分别覆盖神经元动力学、轴突传播、突触编码、脉冲模式分析、信号重建、皮层空间编码、突触权重随机演化和脑流场体积效应。后续将保留 `main.py`，删除其余 `.py` 文件，要求依据本描述及 `main.py` 中的导入与调用接口，独立完成各缺失模块的编码。

## 整体结构

`main.py` 是运行入口。它导入以下8个模块并依次执行它们的演示程序，最终给出一个端到端的综合评估。你的任务是实现这8个模块，确保 `main.py` 能够无错误地运行。

模块文件列表：
- `spike_neuron.py`
- `axon_propagation.py`
- `synaptic_encoding.py`
- `spike_pattern.py`
- `signal_reconstruction.py`
- `cortical_grid.py`
- `stochastic_weights.py`
- `brain_field.py`

## 模块详细说明

### 1. spike_neuron.py — Hodgkin‑Huxley 神经元与群体仿真

**目的**：实现单神经元及兴奋‑抑制神经元群体的动力学模拟。

**核心类与方法**：

- **HHNeuron**  
  采用 Hodgkin‑Huxley 模型描述膜电位 `V` 和三个门控变量 `m, h, n` 的演化。  
  - `__init__(self, dt)`：设定时间步长并进行参数和状态初始化。  
  - `step(self, t, I_syn, I_ext)`：使用经典四阶 Runge‑Kutta (RK4) 方法推动一个时间步，检测并记录脉冲发放。  
  - 内部包含电压依赖的速率函数（alpha/beta for m, h, n）和膜电位导数计算。  
  - 包含不应期处理、发放阈值检测和重置机制。

- **NeuronPopulation**  
  构建脉冲耦合网络，包含指定数量的兴奋性和抑制性神经元。  
  - `__init__(N_exc, N_inh, dt, p_conn)`：创建神经元列表，生成随机连接权重矩阵（兴奋为正，抑制为负）。  
  - `simulate(T_total, I_ext_per_neuron)`：运行群体仿真，计算突触电流，记录所有神经元的膜电位轨迹和脉冲栅格。

**接口要求**：  
- `HHNeuron` 可被独立调用，`step` 返回布尔值表示是否发放。  
- `NeuronPopulation` 返回电压轨迹矩阵、脉冲栅格矩阵和脉冲记录列表。  
两个模块级 `demo_*` 函数分别返回单神经元仿真结果和群体仿真结果，这些结果被 `main.py` 调用（如 `demo_single_neuron` 返回 `V_trace, spikes`，`demo_population` 返回 `voltage_trace, spike_raster, pop_spikes`）。

### 2. axon_propagation.py — 轴突非线性信号传播与 MHD 耦合

**目的**：使用间断 Galerkin (DG) 方法模拟动作电位在一维轴突上的空间传播，并结合磁流体动力学 (MHD) 效应描述离子流与电磁场的耦合。

**核心类与方法**：

- **JacobiPolynomial**  
  提供指定参数的 Jacobi 多项式计算、Gauss‑Lobatto 节点生成等工具，用于 DG 方法的谱基函数构建。  
  - `evaluate(x, alpha, beta, N)`：计算归一化 Jacobi 多项式。  
  - `gauss_lobatto_nodes(N)`：返回 N+1 个区间内的节点。

- **DG1DNeuralCable**  
  一维神经电缆的 DG 离散求解器，包含网格生成、Vandermonde 矩阵、质量矩阵和求导矩阵的构建。  
  - `__init__(xL, xR, K, Np, dt, epsilon)`：初始化空间区间、单元数、节点数、时间步长和数值粘性系数。  
  - `rhs(u, I_ion)`：计算 DG 右端项（扩散、对流、源项及界面通量）。  
  - `step_rk4(u, I_ion)`：用四阶 Runge‑Kutta 方法推进一个时间步。  
  - `simulate(u0, T_final, I_ion_func)`：运行全时程仿真，返回最终膜电位和中间状态历史。

- **MHDNeuralCoupling**  
  处理离子流、磁场及洛伦兹力调制，计算 MHD 耦合下电导率的修正因子。  
  - `ionic_current_density`、`magnetic_field_from_current`、`lorentz_force_modulation` 三个静态方法分别估算离子流密度、磁场和调制因子。  
  - `compute_effective_conductivity(V, y_coord)`：综合上述步骤，返回有效电导率修正因子。

**接口要求**：  
- `demo_axon_propagation` 返回 `u_final, cable.x`。  
- `demo_mhd_coupling` 返回 `x, V, correction`（修正因子数组）。

### 3. synaptic_encoding.py — 突触编码与最优资源分配

**目的**：基于 alpha 突触核函数和有理背包问题优化，在能量约束下最大化信息编码量；提供离散卷积工具。

**核心类与方法**：

- **AlphaSynapse**  
  描述 alpha 函数型突触后电流核函数。  
  - `__init__(tau_s)`：设定时间常数。  
  - `kernel(t)`：计算核函数值。  
  - `convolve_spikes(spike_times, weights, t_grid)`：将脉冲序列与核函数卷积，生成突触后信号。

- **polynomial_multiply_convolution(a, b)**  
  实现两个一维数组的多项式乘法（即离散卷积），去除尾部零并返回乘积系数。

- **rational_knapsack_encoding(profits, weights, budget)**  
  按物品价值密度进行贪心分配，解决连续凸松弛的背包问题，返回分配比例、实际消耗总量和总收益。

- **optimal_synaptic_weights(spike_times, signal_target, t_grid, tau_s, E_budget, sigma_noise)**  
  综合上述工具：构建基函数矩阵，计算每个脉冲的信息增益（利润），利用背包算法选择激活脉冲，最后通过正则化最小二乘计算权重并施加 L1 范数约束，返回最优权重、编码信号和近似互信息。

**接口要求**：  
- 模块级 `demo_encoding` 返回 `weights, encoded, mi, spike_times`，其中 `mi` 为互信息，`encoded` 为编码信号。

### 4. spike_pattern.py — 脉冲模式组合分析与聚类

**目的**：对二进制脉冲模式进行组合计数、去重和聚类，利用连通模式类比和一维多格拼板（polyomino）概念。

**核心类与方法**：

- **SpikePatternAnalyzer**  
  - `__init__(n_bins, pattern_binwidth)`：设定时间 bins 数和 bin 宽度。  
  - `encode_spike_train(spike_times, T_window)`：将脉冲时刻转化为二进制模式向量。  
  - `pattern_entropy(patterns)`：计算经验香农熵。  
  - `pattern_capacity()`：基于一维连通模式公式计算容量。

- **连通模式函数**  
  - `connected_spike_patterns_1d(n_bins)`：返回一维连通模式的总数（公式计算）。  
  - `connected_spike_patterns_2d(nx, ny, max_order)`：返回二维感受野下不同阶数的连通模式计数（固定多格拼板数据）。  
  - `polyomino_enumerate_fixed(order)`：返回指定阶数的固定 polyomino 数量（内置数据表）。

- **模式去重与聚类**  
  - `r8col_sorted_tol_unique(patterns, tol)`：对列向量
