# CO2捕集胺吸收动力学综合模拟平台 — 博士级合成说明

## 一、项目概述

本项目围绕**化学工程：CO2捕集胺吸收动力学**这一前沿博士级科学领域，将15个种子项目的核心算法融合重构为一个统一的Python科研计算框架。项目模拟了CO2在含水胺溶液（MEA、MDEA、PZ）中的吸收、传质、反应动力学、降解路径分析、工艺优化与再生全过程，涵盖从分子尺度反应机理到设备尺度工艺优化的多尺度建模。

## 二、原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 在合成项目中的角色 |
|:---:|---|---|---|
| 1 | `076_bellman_ford` | Bellman-Ford最短路径 | 胺降解反应网络中最低能量路径优化 (`reaction_network_optimizer.py`) |
| 2 | `167_chebyshev2_rule` | Gauss-Chebyshev第二类求积 | 反应速率剖面高精度积分、谱方法求积 (`spectral_methods.py`) |
| 3 | `1287_trapezoidal_explicit` | 显式梯形法(Heun) | 非刚性ODE系统求解 (`ode_integrators.py`) |
| 4 | `090_biochemical_linear_ode` | 生化线性ODE守恒矩阵 | 化学计量矩阵与守恒关系分析 (`conservation_matrix.py`) |
| 5 | `1283_tough_ode` | 刚性ODE | 自动 stiff/non-stiff 检测与BDF切换 (`ode_integrators.py`) |
| 6 | `086_biharmonic_cheby1d` | Chebyshev谱微分 | 液膜扩散-反应边值问题谱求解 (`spectral_methods.py`) |
| 7 | `213_contour_gradient_3d` | 3D梯度计算 | 操作条件优化空间梯度下降 (`process_optimizer.py`) |
| 8 | `202_combo` | 组合数学(子集/排列/划分) | 降解产物组合枚举与Bell数分析 (`degradation_pathways.py`) |
| 9 | `623_knapsack_brute` | 0/1背包暴力枚举 | 吸收剂添加剂配方成本约束优化 (`degradation_pathways.py`) |
| 10 | `382_fem_to_xml` | FEM网格转换 | 填料塔几何结构化网格生成 (`packing_simulation.py`) |
| 11 | `680_line_grid` | 线段网格生成 | 液膜与填料塔轴向离散化 (`mass_transfer.py`) |
| 12 | `536_hilbert_curve_3d` | 3D Hilbert空间填充曲线 | 填料孔隙结构有序遍历采样 (`packing_simulation.py`) |
| 13 | `1383_vandermonde_approx_2d` | 2D Vandermonde多项式拟合 | 气液平衡VLE数据曲面拟合 (`spectral_methods.py`, `equilibrium_models.py`) |
| 14 | `075_bdf3` | BDF3隐式多步法 | 刚性反应动力学与传质耦合方程求解 (`ode_integrators.py`) |
| 15 | `909_predator_prey_ode_period` | 周期轨道检测 | 吸收-再生循环动态周期分析 (`regenerator.py`) |

## 三、核心科学公式与数学模型

### 3.1 CO2-胺反应动力学（两性离子机理）

**Zwitterion机理**（Caplow, 1968; Danckwerts, 1979）：

$$\text{CO}_2 + \text{R}_1\text{R}_2\text{NH} \underset{k_{-1}}{\stackrel{k_2}{\rightleftharpoons}} \text{R}_1\text{R}_2\text{NH}^+\text{COO}^-$$

$$\text{R}_1\text{R}_2\text{NH}^+\text{COO}^- + \text{B} \xrightarrow{k_B} \text{R}_1\text{R}_2\text{NCOO}^- + \text{BH}^+$$

总反应速率：

$$r = \frac{k_2[\text{CO}_2][\text{Amine}]}{1 + k_{-1}/\sum k_B[B]} \approx k_2[\text{CO}_2][\text{Amine}] \quad (\text{对于伯胺})$$

Arrhenius温度依赖：

$$k_2(T) = A \exp\left(-\frac{E_a}{RT}\right)$$

### 3.2 双膜传质理论

气相：

$$N_A = k_G(p_{A,G} - p_{A,i})$$

液相（含化学反应增强）：

$$N_A = k_L E(c_{A,i} - c_{A,L})$$

**Hatta数**（判断反应 regime）：

$$\text{Ha} = \frac{\sqrt{k_2 D_A c_B}}{k_L}$$

**增强因子**（van Krevelen-Hoftijzer）：

$$E = \text{Ha} \cdot \sqrt{\frac{E_i - E}{E_i - 1}}, \quad E_i = 1 + \frac{D_B c_B}{b D_A c_{A,i}}$$

### 3.3 气液平衡（Kent-Eisenberg模型）

对于反应 $\text{CO}_2 + 2\text{RNH}_2 \rightleftharpoons \text{RNHCOO}^- + \text{RNH}_3^+$：

$$K_{\text{eq}} = \frac{[\text{RNHCOO}^-][\text{RNH}_3^+]}{[\text{CO}_2][\text{RNH}_2]^2}$$

平衡加载度：

$$\alpha_{\text{eq}} = \frac{\sqrt{K_{\text{eq}} P_{\text{CO}_2}/H_{\text{CO}_2}}}{1 + 2\sqrt{K_{\text{eq}} P_{\text{CO}_2}/H_{\text{CO}_2}}}$$

### 3.4 Chebyshev谱方法

液膜内扩散-反应方程：

$$D \frac{d^2c}{dz^2} - kc = 0, \quad c(0)=c_i, \; c(\delta)=c_b$$

Chebyshev微分矩阵 $D_{ij}$（Trefethen谱方法）：

$$D_{ij} = \frac{c_i}{c_j} \frac{(-1)^{i+j}}{x_i - x_j}, \quad i \neq j$$

$$D_{ii} = -\frac{x_i}{2(1-x_i^2)}, \quad i \neq 0,N$$

### 3.5 填料塔轴向模型（HTU-NTU）

气相轴向质量平衡：

$$\frac{d(G y_A)}{dz} = -N_A \cdot a$$

能量平衡：

$$\Delta T = \frac{-\Delta N \cdot \Delta H_{\text{rxn}}}{L_{\text{flow}} C_p}$$

### 3.6 化学计量守恒分析

反应网络满足：

$$\mathbf{S} \cdot \mathbf{v} = \mathbf{0}$$

守恒关系（左零空间）：

$$\mathbf{L} \cdot \mathbf{S} = \mathbf{0}$$

元素守恒验证：

$$\mathbf{E} \cdot \mathbf{S} = \mathbf{0}$$

其中 $\mathbf{E}$ 为元素组成矩阵。

### 3.7 再生器（汽提塔）模型

再沸器热负荷：

$$Q_{\text{reb}} = \frac{Q_{\text{sensible}} + Q_{\text{reaction}} + Q_{\text{steam}}}{\alpha_{\text{rich}} - \alpha_{\text{lean}}}$$

热集成分析（夹点分析）：

$$Q_{\text{recoverable}} = \min(Q_{\text{reb}}, Q_{\text{cond}})$$

### 3.8 工艺优化模型

目标函数（最小化CO2规避成本）：

$$\min_{T,P,c,L/G} \frac{C_{\text{operating}} + C_{\text{capital}}}{\dot{m}_{\text{CO}_2}}$$

吸收效率关联：

$$\eta = \exp[-0.02(T-T_{\text{ref}})] \cdot \left(\frac{P}{P_{\text{ref}}}\right)^{0.3} \tanh\left(\frac{c}{3000}\right) \tanh\left(\frac{L/G}{3}\right)$$

## 四、文件结构

```
135_synth_project/
├── main.py                          # 统一入口，零参数运行
├── utils.py                         # 物理常数、数值鲁棒性工具
├── reaction_kinetics.py             # 胺反应动力学与Arrhenius模型
├── ode_integrators.py               # 显式梯形/BDF3/Gear/周期检测求解器
├── spectral_methods.py              # Chebyshev谱方法与2D多项式拟合
├── mass_transfer.py                 # 双膜传质与填料塔轴向模型
├── packing_simulation.py            # 填料几何/FEM网格/Hilbert曲线
├── equilibrium_models.py            # VLE热力学/e-UNIQUAC/数据拟合
├── reaction_network_optimizer.py    # Bellman-Ford降解路径优化
├── degradation_pathways.py          # 组合枚举/背包优化
├── process_optimizer.py             # 梯度优化/灵敏度/不确定性分析
├── conservation_matrix.py           # 化学计量矩阵与守恒关系
└── regenerator.py                   # 汽提塔/循环动态/热集成
```

## 五、运行方式

```bash
cd Synthesis-project-python/135_synth_project
python main.py
```

程序无需任何输入参数，自动执行全部11个模块的计算并输出结果。

## 六、边界处理与数值鲁棒性

- **浓度裁剪**：所有浓度计算通过 `clip_concentration()` 保证非负
- **安全对数/除法**：`safe_log()` 和 `safe_divide()` 防止 `log(0)` 和除零
- **参数验证**：`validate_positive()` 对所有物理参数进行合法性检查
- ** stiff 检测**：自动特征值分析选择显式或隐式积分器
- **BDF3回退**：fsolve失败时自动回退到BDF1
- **多项式拟合条件数监控**：Vandermonde矩阵条件数检查数值稳定性

## 七、合成后的科学问题解决能力

本项目可解决以下前沿科学问题：

1. **CO2-胺反应机理与动力学参数估计**：基于Arrhenius和两性离子机理
2. **气液传质增强因子预测**：Hatta数分析与双膜模型
3. **填料塔吸收性能模拟**：轴向浓度/温度分布与去除效率
4. **胺降解路径预测与寿命评估**：Bellman-Ford最小能量路径
5. **溶剂配方优化**：背包问题约束下的添加剂选择
6. **全流程操作优化**：多变量梯度下降与灵敏度分析
7. **再生能耗评估**：再沸器热负荷与热集成潜力
8. **循环动态稳定性**：周期性吸收-再生过程分析

## 八、注意事项

- 所有可视化相关内容已删除
- 原种子项目目录未被修改
- 代码为纯Python，依赖numpy和scipy
- 合成项目已运行验证无报错
