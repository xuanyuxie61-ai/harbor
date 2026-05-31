# 森林冠层光合与碳通量耦合模拟系统 —— 博士级合成说明

## 一、项目概述

本项目围绕**生态建模：森林冠层光合与碳通量**这一前沿科学领域，将 15 个种子项目的核心算法融合重构为一个面向三维异质性森林生态系统的博士级耦合计算框架。系统实现了从冠层几何结构构建、辐射传输、微气候有限元求解、Farquhar 光合生化模型、CVT 叶片空间分布优化、土壤-冠层高阶碳通量求积、CO₂ 反应扩散、不确定性量化、碳分配优化到数据同化的完整科学计算流程。

---

## 二、原项目到科学问题的映射

| 种子项目 | 核心算法 | 在合成项目中的角色 |
|---------|---------|------------------|
| 885_polygon_grid | 多边形网格生成 | **canopy_geometry.py**：用多边形重心细分法生成树冠二维截面网格与三维 LAI 场 |
| 574_image_contrast | 对比度增强 | **radiation_transfer.py**：将光斑-阴影异质性建模为冠层内辐射场的对比度增强 |
| 1081_simplex_monte_carlo | 单纯形采样与积分 | **leaf_angle_sampling.py**：在叶片倾角-方位角分布的单纯形上进行蒙特卡洛采样，计算 Ross G 函数 |
| 529_hexagon_monte_carlo | 六边形区域蒙特卡洛 | **boundary_flux.py**：在六边形边界上估算冠层侧边界碳通量 |
| 279_diff_center | 中心差分求导 | **photosynthesis_model.py**：中心差分计算光合速率对温度的敏感性 dAₙ/dT |
| 403_fem2d_heat | 2D 热方程有限元 | **microclimate_fem.py**：用 T3/T6 三角形元求解冠层微气候温度场 |
| 249_cvt_3d_lumping | 3D CVT Lloyd 算法 | **cvt_leaf_distribution.py**：三维 Lloyd 迭代优化叶片簇空间分布，最大化光截获 |
| 242_cvt_4_movie | 2D CVT 迭代 | **cvt_leaf_distribution.py**：2D CVT 密度驱动思想嵌入 3D 优化 |
| 410_fem2d_predator_prey_fast | 捕食者- prey 反应扩散 FEM | **co2_diffusion.py**：将 CO₂ 扩散-吸收类比为反应扩散系统，有限差分求解 |
| 957_quadrilateral_witherden_rule | 四边形高阶求积 | **soil_carbon_flux.py**：在土壤-冠层界面使用 Witherden 型高阶求积计算 NEE |
| 1107_sparse_grid_laguerre | Smolyak 稀疏网格 | **uncertainty_quantification.py**：用 Laguerre 稀疏网格量化 Vcmax/Jmax 参数不确定性 |
| 1211_test_interp_1d | 1D 插值测试 | **environment_response.py**：对环境因子-光合响应曲线进行分段线性插值 |
| 581_image_noise | 图像噪声 | **data_assimilation.py**：为涡度相关观测通量添加高斯与盐椒噪声 |
| 157_change_greedy | 贪心找零 | **carbon_allocation.py**：贪心策略模拟光合碳在叶、干、根、储存间的分配 |
| 224_cplex_solution_read | CPLEX 解解析 | **optimization_parser.py**：解析碳分配线性规划的最优解与影子价格 |

---

## 三、核心科学公式

### 3.1 冠层几何与 LAI 分布

冠层 LAI 垂直剖面服从归一化 Beta 分布：

$$\text{LAI}(z) = \text{LAI}_{\max} \cdot \frac{(z/h)^{\alpha-1}(1-z/h)^{\beta-1}}{B(\alpha,\beta)}$$

### 3.2 辐射传输（Beer-Lambert + 对比度修正）

$$I(z) = I_0 \exp(-k_{\text{ext}} \cdot L(z))$$

光环境异质性增强：

$$I_{\text{enh}} = s \cdot I + (1-s) \cdot I_{\text{avg}}$$

### 3.3 叶片角度分布与 G 函数

$$G(\theta_s) = \int_0^{2\pi} \int_0^{\pi/2} |\cos\xi| \cdot f(\theta_l,\phi_l) \sin\theta_l \, d\theta_l d\phi_l$$

其中 $\cos\xi = \cos\theta_l\cos\theta_s + \sin\theta_l\sin\theta_s\cos(\phi_l-\phi_s)$。

### 3.4 Farquhar-von Caemmerer-Berry 光合模型

$$A_n = \min(W_c, W_j) - R_d$$

$$W_c = \frac{V_{c\max}(C_i - \Gamma^*)}{C_i + K_c(1 + O_i/K_o)}$$

$$W_j = \frac{J(C_i - \Gamma^*)}{4C_i + 8\Gamma^*}$$

温度响应（Arrhenius + 高温失活）：

$$V_{c\max}(T) = V_{c\max,25} \exp\left(\frac{E_a(T-298.15)}{298.15RT}\right) \frac{1 + \exp\left(\frac{298.15\Delta S - H_d}{298.15R}\right)}{1 + \exp\left(\frac{T\Delta S - H_d}{RT}\right)}$$

### 3.5 微气候有限元

$$\rho C_p \frac{\partial T}{\partial t} = \kappa \nabla^2 T + Q_{\text{rad}} - Q_{\text{latent}}$$

Tetens 公式：

$$e_s(T) = 0.6108 \exp\left(\frac{17.27T}{T+237.3}\right)$$

向后 Euler 时间离散 + T3/T6 有限元空间离散。

### 3.6 CVT 叶片分布优化

能量泛函：

$$\mathcal{F}(P) = \sum_{i=1}^{n} \int_{V_i} \rho(x) \|x - p_i\|^2 dx$$

Lloyd 迭代：

$$p_i^{\text{new}} = \frac{\int_{V_i} \rho(x)x \, dx}{\int_{V_i} \rho(x) dx}$$

### 3.7 土壤-冠层碳通量（Witherden 求积）

$$F_c = \int_{\Omega} R_d(x,y) \cdot \text{LAI}(x,y) \, d\Omega \approx \sum_{q} w_q |J_q| R_d(x_q,y_q) \text{LAI}(x_q,y_q)$$

Lloyd-Taylor 土壤呼吸：

$$R_s = R_{10} \exp\left(E_0 \left(\frac{1}{T_{\text{ref}} - T_0} - \frac{1}{T_{\text{soil}} - T_0}\right)\right)$$

### 3.8 CO₂ 反应扩散（类比 predator-prey）

$$\frac{\partial C}{\partial t} = D\nabla^2 C + R_{\text{soil}} - \frac{V_{\max}C}{K_m + C} \cdot \text{LAI}(x,y)$$

### 3.9 Smolyak 稀疏网格不确定性量化

$$Q_L^{(d)} f = \sum_{|l|_1 \leq L} (-1)^{L-|l|_1} \binom{d-1}{L-|l|_1} (Q_{l_1} \times \cdots \times Q_{l_d}) f$$

### 3.10 碳分配线性规划

$$\max Z = c_{\text{leaf}} x_{\text{leaf}} + c_{\text{stem}} x_{\text{stem}} + c_{\text{root}} x_{\text{root}} + c_{\text{storage}} x_{\text{storage}}$$

$$\text{s.t.} \quad \sum x_i \leq C_{\text{total}}, \quad x_i \geq R_{\text{maint},i}$$

---

## 四、代码文件说明

| 文件名 | 功能 | 对应种子项目 |
|--------|------|-------------|
| `main.py` | 统一入口，零参数运行 | — |
| `utils.py` | 数值工具、带状矩阵求解、T6 基函数 | 403, 410 |
| `canopy_geometry.py` | 冠层多边形网格与 LAI 场 | 885 |
| `radiation_transfer.py` | Beer-Lambert 辐射传输与对比度 | 574 |
| `leaf_angle_sampling.py` | 单纯形 MC 采样与 G 函数 | 1081 |
| `microclimate_fem.py` | 冠层温度场 FEM（T3/T6） | 403 |
| `photosynthesis_model.py` | FvCB 模型与中心差分 | 279 |
| `cvt_leaf_distribution.py` | 3D CVT Lloyd 叶片优化 | 249, 242 |
| `soil_carbon_flux.py` | Witherden 求积与 NEE | 957 |
| `uncertainty_quantification.py` | Smolyak 稀疏网格 UQ | 1107 |
| `co2_diffusion.py` | CO₂ 反应扩散有限差分 | 410 |
| `carbon_allocation.py` | 贪心碳分配 | 157 |
| `environment_response.py` | 环境响应 1D 插值 | 1211 |
| `data_assimilation.py` | 噪声模型与卡尔曼滤波 | 581 |
| `boundary_flux.py` | 六边形 MC 边界通量 | 529 |
| `optimization_parser.py` | LP 解解析与影子价格 | 224 |

---

## 五、运行方式

```bash
cd Synthesis-project-python/069_synth_project
python main.py
```

程序将自动执行 14 个模块的完整计算流程，输出各模块的中间结果与最终的碳通量汇总报告。无需任何输入参数。

---

## 六、科学问题的深度与前沿性

1. **多尺度耦合**：从单叶生化反应（Farquhar）→ 冠层微气候（FEM）→ 生态系统通量（NEE），跨越 3 个数量级时空尺度。
2. **异质性刻画**：辐射传输的对比度修正、CVT 叶片空间优化、CO₂ 反应扩散，共同构建了非均匀冠层的精细描述。
3. **不确定性量化**：稀疏网格方法将参数不确定性系统传播到碳通量预测，是 IPCC 级别地球系统模型的核心需求。
4. **数据同化**：卡尔曼滤波与集合方法为涡度相关观测与模型融合提供了方法论基础。
5. **优化理论**：碳分配的线性规划与影子价格分析，连接了生态生理学与运筹学。

---

## 七、边界处理与数值鲁棒性

- **安全除法**：`safe_divide` 防止除以零。
- **温度截断**：FEM 求解后温度 clip 到 [-20, 60] °C，防止 Tetens 公式溢出。
- **CO₂ 浓度边界**：clip 到 [380, 2000] umol/mol，符合大气物理约束。
- **矩阵病态处理**：`solve_banded` 替代手动 LU，避免奇异矩阵崩溃。
- **异常回退**：FEM 模块异常时自动切换到解析温度近似，保证 main.py 不中断。
- **非负约束**：所有通量、LAI、生物量均强制非负。
