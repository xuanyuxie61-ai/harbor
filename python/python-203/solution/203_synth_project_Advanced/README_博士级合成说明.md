# PROJECT 203: 准蒙特卡洛不确定性量化 — 耦合随机地震-大气化学动力学系统

## 一、科学问题描述

本项目解决**不确定性量化 (Uncertainty Quantification, UQ)** 领域的前沿博士级科学计算问题：

### 核心科学问题

**在异质随机介质中，地震波传播与大气光化学过程的耦合系统如何进行高效不确定性量化？**

给定随机速度场 $c(x, \omega)$（由 Karhunen-Loève 展开参数化），求解耦合的随机偏微分方程-常微分方程系统：

**弹性波方程（随机PDE）：**
$$\rho(x) \frac{\partial^2 u}{\partial t^2} = \frac{\partial}{\partial x}\left[\rho(x) c^2(x, \omega) \frac{\partial u}{\partial x}\right] + f(x,t)$$

**大气臭氧化学动力学（随机ODE）：**
$$\frac{dy_1}{dt} = k_1(t) y_3 - k_2 y_1, \quad \frac{dy_2}{dt} = k_1(t) y_3 - k_3 y_2 y_4 + \sigma_2$$
$$\frac{dy_3}{dt} = -k_1(t) y_3 + k_3 y_2 y_4, \quad \frac{dy_4}{dt} = k_2 y_1 - k_3 y_2 y_4$$

### 数学方法

1. **广义多项式混沌展开 (gPC)**：使用 Gegenbauer 多项式 $C_n^{(\alpha)}(\xi)$ 作为正交基：
   $$u(x, t, \xi) = \sum_{k=0}^{P} u_k(x, t) \Phi_k(\xi)$$

2. **随机 Galerkin 投影**：将随机 PDE 投影到 PC 基函数空间，得到确定性的耦合方程组：
   $$\sum_k \mathbb{E}[\kappa(x,\xi) \Phi_j \Phi_k] \frac{du_k}{dx} = \mathbb{E}[f \Phi_j]$$

3. **准蒙特卡洛采样**：使用 Halton 序列（低偏差序列）替代随机采样，利用 Koksma-Hlawka 不等式保证：
   $$|QMC - I| \leq V(g) \cdot D_N^*$$

4. **CVT 自适应采样**：Centroidal Voronoi Tessellation 迭代优化随机空间中的配点分布。

---

## 二、核心数学公式

### 2.1 Gegenbauer 多项式三项递推

$$C_0^{(\alpha)}(x) = 1, \quad C_1^{(\alpha)}(x) = 2\alpha x$$
$$n C_n^{(\alpha)}(x) = 2(n+\alpha-1) x C_{n-1}^{(\alpha)}(x) - (n+2\alpha-2) C_{n-2}^{(\alpha)}(x)$$

### 2.2 Gauss-Gegenbauer 求积公式

$$\int_{-1}^{1} f(x) (1-x^2)^{\alpha-1/2} dx \approx \sum_{i=1}^{n} w_i f(x_i)$$

节点 $x_i$ 为 $C_n^{(\alpha)}(x)$ 的零点，权重由 Christoffel-Darboux 公式计算。

### 2.3 Karhunen-Loève 展开

$$\kappa(x, \omega) = \kappa_0(x) \exp\left(\sum_{m=1}^{M} \sqrt{\lambda_m} \phi_m(x) \xi_m\right)$$

其中 $\lambda_m$, $\phi_m(x)$ 为指数协方差核的特征值和特征函数。

### 2.4 DG 内罚格式（SIPG/NIPG/IIPG）

通过参数 $ss \in \{-1, 0, 1\}$ 统一三种 DG 格式：
- $ss = -1$: SIPG（对称内罚）
- $ss = 0$: IIPG（不完全内罚）
- $ss = 1$: NIPG（非对称内罚）

内罚参数：$\sigma = C \cdot p^2 / h$，其中 $p$ 为多项式阶数，$h$ 为网格尺寸。

### 2.5 B1G3 隐式多步法

三步隐式格式，系数由 $\sqrt{3}$ 导出：
$$A_3 = 0.5 + 1/\sqrt{3}, \quad A_2 = -2/\sqrt{3}, \quad A_1 = -0.5 + 1/\sqrt{3}, \quad B = 1/\sqrt{3}$$

### 2.6 Halton 序列

$s$ 维 Halton 序列使用 $s$ 个不同素数作为基：
$$x_i = (\text{vdc}_{b_1}(i), \text{vdc}_{b_2}(i), \ldots, \text{vdc}_{b_s}(i))$$

星偏差：$D_N^* = O(N^{-1} (\log N)^s)$，优于 MC 的 $O(N^{-1/2})$。

### 2.7 CVT 量化误差

$$E = \sum_{i=1}^{n} \int_{V_i} \rho(x) \|x - g_i\|^2 dx$$

Lloyd 迭代使每个发生器点移至其 Voronoi 单元的质心。

---

## 三、种子项目到科学问题的映射

| 序号 | 种子项目 | 原始功能 | 合成项目中的角色 |
|------|---------|---------|----------------|
| 1 | 275_dg1d_poisson | 1D DG 求解 Poisson 方程 | DG 空间离散化框架 → 随机波方程的空间离散 |
| 2 | 100_blood_pressure_ode | 血压 ODE（周期跳跃） | 周期性跳跃 ODE 模型 → 验证 UQ 对周期系统的有效性 |
| 3 | 1094_QuakeMigrate | 地震波形迁移叠加检测 | 波形迁移定位 → 随机震源定位与检测 |
| 4 | 462_gegenbauer_polynomial | Gegenbauer 正交多项式 | PC 展开基函数 → 广义多项式混沌的核心数学工具 |
| 5 | 1344_triangulation_orient | 三角形网格定向 | 网格预处理 → 随机域分解的网格质量保证 |
| 6 | 1053_DiminishedRhythms | iEEG 信号周期分析 | 信号处理与 AUC → UQ 输出的统计分析 |
| 7 | 264_cvtp | 周期性 CVT 迭代 | CVT 自适应采样 → 随机空间最优配点生成 |
| 8 | 824_octopus | Octave 环境检测 | 运行时环境检测 → 跨平台鲁棒性保障 |
| 9 | 842_ozone2_ode | 大气臭氧化学 ODE | 化学反应动力学 → 耦合随机 ODE 子系统 |
| 10 | 804_nint_exactness_mixed | 多维求积精确度测试 | 求积规则验证 → 确保 PC 积分的数值精度 |
| 11 | 379_fem_to_medit | FEM→MEDIT 格式转换 | 网格格式转换 → 计算网格的标准化输出 |
| 12 | 235_cube_monte_carlo | 单位立方体 MC 积分 | MC 参考积分 → QMC 方法的精度基准 |
| 13 | 061_b1g3 | B1G3 隐式多步 ODE 求解器 | 时间积分 → 刚性随机 ODE 的稳定求解 |
| 14 | 883_polygon_average | 多边形平均迭代 | 信号平滑 → UQ 输出的特征提取 |
| 15 | 1420_xy_io | XY 点数据 I/O 库 | 数据读写 → 科学数据交换与序列化 |

---

## 四、代码文件结构

```
203_synth_project_Advanced/
├── main.py                    # 统一入口，零参数运行
├── quasi_mc.py                # 准蒙特卡洛采样模块
├── polynomial_chaos.py        # Gegenbauer 多项式混沌展开
├── stochastic_galerkin.py     # 随机 Galerkin 投影与求解
├── cvt_sampler.py             # CVT 自适应随机采样
├── seismic_wave.py            # DG 地震波传播求解器
├── ozone_kinetics.py          # 大气臭氧化学动力学 ODE
├── ode_integrator.py          # B1G3 隐式多步 ODE 积分器
├── waveform_migration.py      # 地震波形迁移与检测
├── signal_cycles.py           # 信号处理与统计分析
├── mesh_utils.py              # 网格处理工具
├── quadrature_exactness.py    # 求积精确度测试
├── io_utils.py                # 科学数据 I/O 工具
└── README_博士级合成说明.md    # 本文档
```

---

## 五、计算流程（10个阶段）

### Phase 1: 网格生成与求积设置
- 生成结构化三角形网格（`mesh_utils.generate_stochastic_mesh`）
- 检查并修正三角形方向（`mesh_utils.orient_triangles`）
- 转换为 MEDIT 格式（`mesh_utils.fem_to_medit`）
- 验证 Gauss-Gegenbauer 求积精确度（`quadrature_exactness.test_gegenbauer_exactness`）

### Phase 2: 准蒙特卡洛收敛验证
- 生成 Halton 低偏差序列（`quasi_mc.halton`）
- 生成 Owen 随机化 Halton 序列（`quasi_mc.scrambled_halton`）
- 计算星偏差（`quasi_mc.star_discrepancy`）
- 验证 QMC 收敛速度优于 MC（`quasi_mc.qmc_convergence_test`）

### Phase 3: 多项式混沌基构造
- Gegenbauer 三项递推求值（`pc.gegenbauer_value`）
- Gauss-Gegenbauer 求积节点/权重（`pc.gauss_gegenbauer`）
- 三重积张量计算（`pc.gegenbauer_triple_product`）
- 多维多指标集生成（`pc.multi_index_set`）
- Jacobi 矩阵特征值（`pc.gegenbauer_jacobi_matrix`）

### Phase 4: 随机 Galerkin 系统
- Karhunen-Loève 展开（`sg.karhunen_loeve_1d`）
- DG 空间离散 + PC 随机离散耦合求解（`sg.stochastic_galerkin_diffusion_1d`）
- 统计量计算：均值、方差、置信区间（`sg.compute_pc_statistics`）

### Phase 5: 地震波传播
- DG 波方程求解器（`seismic_wave.DGWaveSolver1D`）
- 随机速度场生成（`seismic_wave.generate_random_velocity_field`）
- Newmark-beta 时间积分
- 能量守恒监控（`seismic_wave.compute_wave_energy`）

### Phase 6: 臭氧化学耦合
- 4 物种化学动力学 ODE（`ozone_kinetics.OzoneChemistry`）
- 含时光解速率 $k_1(t)$（`ozone_kinetics.photolysis_rate`）
- 守恒量监控（`ozone_kinetics.conserved_quantity`）
- 参数扰动 UQ 分析（`ozone_kinetics.run_uncertain`）
- 血压 ODE 验证（`ode_integrator.blood_pressure_ode_model`）

### Phase 7: 波形迁移检测
- 走时查找表构建（`waveform_migration.TraveltimeLUT`）
- 合成波形生成（`waveform_migration.simulate_waveforms`）
- 迁移叠加定位（`waveform_migration.migrate_and_stack`）
- STA/LTA 事件检测（`waveform_migration.compute_stalta`）

### Phase 8: 信号处理与统计
- 缺失数据插补（`signal_cycles.compute_impute_missing`）
- Butterworth 带通滤波与周期提取（`signal_cycles.extract_cycles`）
- AUC 可分性度量（`signal_cycles.compute_auc`）
- 线性混合效应模型（`signal_cycles.fit_mixed_effects`）
- 多边形平均平滑（`signal_cycles.polygon_average_iteration`）

### Phase 9: CVT 自适应采样
- CVT Lloyd 迭代（`cvt_sampler.cvt_run`）
- 偏差对比研究（`cvt_sampler.cvt_discrepancy_study`）

### Phase 10: B1G3 积分与 I/O
- B1G3 隐式多步 ODE 积分（`ode_integrator.integrate_ode`）
- 指数衰减测试 + Van der Pol 刚性振荡测试
- XY/XYF/XYL 数据读写（`io_utils.xy_read/write`）
- UQ 结果序列化（`io_utils.save_uq_results`）

---

## 六、运行方法

本项目为 **零参数可运行** 的统一入口：

```bash
python main.py
```

无需安装任何第三方科学计算库（仅依赖 `numpy` 和 `scipy`）。所有 10 个阶段将自动顺序执行，并在终端输出详细的计算结果。

### 依赖

- Python 3.x
- numpy >= 1.18
- scipy >= 1.4

---

## 七、合成方法论

### 7.1 从通用数值方法到 UQ 专用框架

原种子项目中的数值方法（DG、ODE 求解器、正交多项式等）被深度改造为不确定性量化专用框架：

- **DG 方法**：从确定性 Poisson 方程改造为随机波方程的空间离散，引入 SIPG/NIPG/IIPG 通量耦合
- **Gegenbauer 多项式**：从纯数学工具改造为多项式混沌展开的核心基函数，支撑随机 Galerkin 投影
- **ODE 求解器**：从标准时间积分改造为刚性随机 ODE 的稳定求解，支持 B1G3 隐式多步法

### 7.2 领域独特性

本项目的算法结构与"不确定性量化"领域深度耦合：

1. **三重积张量** $\mathbb{E}[\Phi_i \Phi_j \Phi_k]$ 是随机 Galerkin 方法的核心计算
2. **KL 展开 + gPC** 构成随机场参数化的标准框架
3. **Halton 序列的星偏差** 是 QMC 理论的核心度量
4. **CVT 量化误差** 直接关联最优随机配点设计

### 7.3 边界处理与鲁棒性

- 非负性约束：浓度场 $y_i \geq 0$，速度场 $c(x) > 0$
- 奇异性处理：雅可比矩阵退化时使用 `lstsq` 替代 `solve`
- 缺失数据：NaN 插补支持单点、短间隙、长间隙三种策略
- 数值溢出保护：对数计算前的正性检查，指数函数的范围限制

---

## 八、验证结果摘要

运行 `python main.py` 后的关键验证指标：

| 验证项目 | 指标 | 结果 |
|---------|------|------|
| 求积精确度 | 15 阶最大相对误差 | $3.43 \times 10^{-15}$ |
| 正交性验证 | Gram 矩阵非对角元最大值 | $5.00 \times 10^{-16}$ |
| 星偏差 | Halton vs Random (dim=4) | 0.038 vs 0.100 |
| 守恒律 | 奇氧总量相对变化 | $4.72 \times 10^{-11}$ |
| 血压 ODE | 最大误差 vs 精确解 | $1.71 \times 10^{-13}$ mmHg |
| 多边形平滑 | 粗糙度降低倍数 | 77.22x |
| XY I/O 往返误差 | 最大偏差 | $5.55 \times 10^{-17}$ |
