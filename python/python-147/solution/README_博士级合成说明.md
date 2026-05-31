# README_博士级合成说明.md

## 项目概述

**项目名称**：深度学习物理信息神经网络 (PINN) 求解 Kuramoto-Sivashinsky 混沌偏微分方程

**科学领域**：神经计算 —— 深度学习物理信息神经网络 PINN

**合成目标**：基于 15 个种子科研代码项目的核心算法，在"神经计算：深度学习物理信息神经网络 PINN"领域内，融合构建一个前沿博士级自然科学计算问题。本项目使用纯 NumPy 实现了一个完整的 Physics-Informed Neural Network (PINN) 框架，用于求解具有时空混沌特性的 Kuramoto-Sivashinsky (KS) 方程，并附带高阶数值积分、谱方法、RBF 基线对比、自适应采样、混沌初始条件生成等高级模块。

---

## 核心科学问题

### Kuramoto-Sivashinsky 方程

KS 方程是描述等离子体物理、火焰前沿传播和两相流的一维空间周期域非线性偏微分方程：

```
u_t + u * u_x + u_xx + u_xxxx = 0,   x ∈ [0, 32π],  t > 0
```

边界条件为周期性边界条件 `u(t, 0) = u(t, 32π)`，初始条件为 `u(0, x) = cos(x/16) * (1 + sin(x/16))`。

该方程的解具有**时空混沌**特性，是检验 PINN 方法在非线性动力系统上表现的经典基准问题。

### PINN 损失泛函

PINN 通过最小化以下复合损失函数来训练神经网络 `u_θ(t, x)`：

```
L_total(θ) = λ_pde * L_pde(θ) + λ_ic * L_ic(θ) + λ_bc * L_bc(θ)
```

其中：

- **PDE 残差损失**：
  ```
  L_pde = (1/N_f) Σ_i | r(t_i, x_i) |²
  r(t,x) = ∂_t u_θ + u_θ * ∂_x u_θ + ∂_xx u_θ + ∂_xxxx u_θ
  ```

- **初值条件损失**：
  ```
  L_ic = (1/N_ic) Σ_j | u_θ(0, x_j) - u_0(x_j) |²
  ```

- **周期边界条件损失**：
  ```
  L_bc = (1/N_bc) Σ_k | u_θ(t_k, 0) - u_θ(t_k, L) |²
  ```

### 谱方法参考解

使用指数时间差分 Runge-Kutta 4 阶方法 (ETDRK4) 在傅里叶空间求解 KS 方程作为"真值"参考：

线性算子：`L = k² - k⁴`

非线性项：`N(v) = -0.5 * i * k * fft( real(ifft(v))² )`

ETDRK4 系数通过围道积分（Kassam-Trefethen 根式法）预计算：

```
Q  = dt * Re{ mean( (exp(LR/2) - 1) / LR ) }
f1 = dt * Re{ mean( (-4 - LR + exp(LR)*(4 - 3LR + LR²)) / LR³ ) }
f2 = dt * Re{ mean( (2 + LR + exp(LR)*(-2 + LR)) / LR³ ) }
f3 = dt * Re{ mean( (-4 - 3LR - LR² + exp(LR)*(4 - LR)) / LR³ ) }
```

---

## 15 个种子项目映射关系

| 序号 | 种子项目 | 核心算法 | 在合成项目中的角色 |
|------|----------|----------|-------------------|
| 1 | 1152_squircle_ode | 广义三角函数 ODE (超椭圆运动) | `chaos_utils.py` 中 Squircle 轨迹生成，作为周期性激活基函数和初始条件参数化 |
| 2 | 453_gauss_seidel_stochastic | 随机 Gauss-Seidel 迭代 | `stochastic_optimizer.py` 中随机坐标下降优化器，类比随机梯度下降 |
| 3 | 597_iplot | 文本格式化与字符串处理 | `data_io.py` 中科学数据格式化输出与日志记录 |
| 4 | 1013_rbf_interp_1d | 径向基函数插值 | `rbf_kernel.py` 中 RBF 核函数层、多核类型 (MQ/IMQ/TPS/高斯) 及权重求解 |
| 5 | 1143_square_exactness | 2D Gauss-Legendre 求积、Padua 点 | `quadrature_rules.py` 中高维数值积分、非张量积求积节点 |
| 6 | 629_kronrod_rule | Gauss-Kronrod 自适应求积 | `quadrature_rules.py` 中 (7,15) Kronrod 节点权重表及误差估计 |
| 7 | 1197_tec_io | 科学数据文件 I/O 解析 | `data_io.py` 中矩阵转置打印、检查点存取、变量行解析 |
| 8 | 189_clock_solitaire_simulation | 随机状态转移模拟 | 启发 `adaptive_sampler.py` 中基于残差的随机拒绝采样 |
| 9 | 1219_test_nearest | 最近邻搜索 | `domain_mesh.py` 中暴力法最近邻搜索，用于自适应配点加密 |
| 10 | 1213_test_interp_fun | 测试插值函数族 | `convergence_test.py` 中 Manufactured Solution 精确解构造 |
| 11 | 227_cross_chaos | 迭代函数系统 (IFS) 生成分形 | `chaos_utils.py` 中 Cross 混沌吸引子点生成，用于非平凡初始条件 |
| 12 | 1331_triangulation_boundary | 三角剖分边界边提取 | `domain_mesh.py` 中边界边检测与路径闭合算法 |
| 13 | 275_dg1d_poisson | 间断 Galerkin 1D Poisson 求解器 | `convergence_test.py` 中 DG 参考解对比框架与误差指标 |
| 14 | 630_kursiv_pde_etdrk4 | Kuramoto-Sivashinsky ETDRK4 求解器 | `ks_pde_solver.py` 核心参考解生成；`spectral_ops.py` 谱微分与 ETD 系数 |
| 15 | 148_cellular_automaton | Rule-30 元胞自动机 | `chaos_utils.py` 中二元空间模式生成，作为非光滑初值条件 |

---

## 文件结构与功能

本项目共包含 **13 个 Python 文件**：

| 文件名 | 功能说明 |
|--------|----------|
| `main.py` | 统一入口，零参数运行，执行完整 PINN 科研流程 |
| `ks_pde_solver.py` | ETDRK4 谱方法参考解生成，KS 方程残差计算 |
| `pinn_network.py` | 全连接神经网络，含 Gaussian RBF / Squircle / tanh 激活，有限差分导数计算 |
| `physics_loss.py` | PDE 残差、初值损失、边界损失、总损失及数值梯度计算 |
| `stochastic_optimizer.py` | SGD 动量优化器、随机坐标下降 (Gauss-Seidel 风格)、余弦退火学习率调度 |
| `domain_mesh.py` | 时空配点网格生成、周期边界点、三角剖分边界提取、最近邻搜索、空间聚类 |
| `rbf_kernel.py` | 四种 RBF 核函数、RBF 插值权重求解、RBF 核网络层 |
| `chaos_utils.py` | Squircle ODE 积分、Cross IFS 分形生成、Rule-30 元胞自动机、混沌初值生成器 |
| `quadrature_rules.py` | 1D/2D Gauss-Legendre 求积、Gauss-Kronrod (7,15) 自适应求积、Padua 点集 |
| `convergence_test.py` | Manufactured Solution (3 组)、误差指标计算、DG 参考对比 |
| `data_io.py` | 科学数据格式化打印、PINN 检查点存取、指标日志记录 |
| `spectral_ops.py` | FFT 谱微分、ETDRK4 系数预计算、2/3 去混叠规则、能量谱与 Kolmogorov 尺度 |
| `adaptive_sampler.py` | 残差自适应配点加密、多层网格细化、基于拒绝采样的自适应采样 |

---

## 关键科学公式与数值方法

### 1. 神经网络前向传播

对于 L 层全连接网络，输入 `z = [t, x]`：

```
h^(0) = z
h^(l) = σ( W^(l) h^(l-1) + b^(l) ),   l = 1, ..., L-1
u_θ   = W^(L) h^(L-1) + b^(L)
```

激活函数：
- Gaussian RBF: `σ(z) = exp(-0.5 z² / r₀²)`
- Squircle: `σ(z) = tanh(z) · √|z| / (1 + √|z|)`
- tanh: `σ(z) = tanh(z)`

### 2. 有限差分导数（纯 NumPy 实现）

由于本项目不依赖 PyTorch/TensorFlow 等自动微分框架，所有导数通过高精度有限差分计算：

- 一阶导数（中心差分）：`∂u/∂z ≈ (u(z+h) - u(z-h)) / (2h)`
- 二阶导数：`∂²u/∂z² ≈ (u(z+h) - 2u(z) + u(z-h)) / h²`
- 四阶导数（五点 stencil）：`∂⁴u/∂z⁴ ≈ (u_{-2} - 4u_{-1} + 6u_0 - 4u_{+1} + u_{+2}) / h⁴`

### 3. Gauss-Legendre 张量积求积

对于矩形域 `[ax,bx] × [ay,by]` 上的积分：

```
∫∫ f(x,y) dx dy ≈ Σ_i Σ_j w_i^x w_j^y f(x_i, y_j)
```

节点通过仿射变换从 `[-1,1]` 映射到目标区间：
```
x = (b-a)/2 · x_ref + (a+b)/2
w = (b-a)/2 · w_ref
```

### 4. Gauss-Kronrod 自适应求积

(7,15) Kronrod 规则在 `[-1,1]` 上的节点 `x_k` 和权重 `w_k` 满足：

```
∫_{-1}^{1} f(x) dx ≈ Σ_{k=1}^{15} w_k f(x_k)
```

嵌入的 7 点 Gauss 规则提供误差估计：`|I_Kronrod - I_Gauss|`。

### 5. 能量谱与 Kolmogorov 尺度

能量谱：`E(k) = 0.5 |û_k|²`

Kolmogorov 耗散尺度：
```
η = (ν³ / ε)^{1/4}
```
其中 `ε = mean(u_xx²)` 为耗散率，对于 KS 方程取 `ν = 1`。

### 6. 随机坐标下降优化

受 Gauss-Seidel 迭代启发，每次迭代随机选取参数块 `B ⊂ {1,...,P}` 并更新：

```
θ_B ← θ_B - η ∇_B L(θ)
```

这与线性系统中随机选取方程更新的随机 Gauss-Seidel 方法形成直接类比。

---

## 运行方式

```bash
cd Synthesis-project-python/147_synth_project
python3 main.py
```

脚本零参数运行，自动执行以下六个阶段：

1. **Manufactured Solution 收敛性验证**：在已知精确解上训练 PINN，验证网络架构与梯度计算正确性
2. **KS 方程 PINN 训练**：在真实 KS 方程配点上训练，对比 ETDRK4 参考解
3. **RBF 基线对比**：使用高斯 RBF 插值作为经典数值方法的性能基准
4. **高斯积分与三角剖分诊断**：验证 Gauss-Legendre、Kronrod 积分精度及边界提取算法
5. **混沌动力学初始条件生成**：展示 Squircle、Cross IFS、元胞自动机三种非平凡初值
6. **最近邻搜索与自适应采样**：验证空间搜索与聚类算法

---

## 边界处理与数值鲁棒性

1. **参数维度校验**：所有模块在入口处检查输入维度、形状与取值范围
2. **除零保护**：RBF 核、高斯激活、Kronrod 权重中均加入 `1e-12` 级正则化
3. **FFT 偶数点要求**：KS 求解器强制要求 `nx` 为偶数
4. **有限差分稳定性**：四阶导数使用适度步长 `h = 2e-3` 避免灾难性抵消
5. **Padua 点边界检查**：仅提供预计算的 0~5 级点集，超出范围抛出明确异常
6. **三角形边界路径闭合**：`boundary_edge_to_path` 检测不闭合路径并抛出异常

---

## 科学指标输出示例

运行完成后，终端输出包含以下指标：

- Manufactured Solution L2 / L∞ 误差
- KS PINN 与 ETDRK4 参考解的 L2 / L∞ 误差
- 能量谱平均偏差
- PINN / 参考 Kolmogorov 尺度
- RBF 基线插值误差与条件数
- Gauss-Legendre / Kronrod 积分精度
- 混沌初值能量统计

---

## 注意事项

- 本项目仅依赖 **NumPy** 和标准库，无需 PyTorch/TensorFlow/JAX
- 所有可视化代码已删除，仅保留数值结果与文本输出
- 原始种子项目文件夹未被修改，所有合成代码存放于新建目录 `147_synth_project`
