# README_博士级合成说明.md

## 项目概述

**项目名称**: 人工耳蜗电刺激电场分布数值模拟与神经响应分析系统  
**科学领域**: 生物医学工程 —— 人工耳蜗（Cochlear Implant）电刺激电场分布  
**编程语言**: Python 3  
**合成目录**: `127_synth_project/`

本项目基于 15 个原始科研代码项目的核心算法，融合构建了一个面向前沿生物医学计算问题的博士级科研代码系统。项目围绕**人工耳蜗电极阵列在耳蜗鼓阶内产生的三维电场分布及其诱发的螺旋神经节神经元（SGN）响应**展开，涵盖了从几何建模、偏微分方程数值求解、特殊函数解析展开、神经元膜动力学、反应-扩散斑图分析、高精度有限元积分、患者参数降维到统计变异性的完整计算链路。

---

## 1. 原项目到科学问题的映射

| 序号 | 原始项目 | 核心算法/思想 | 在合成项目中的角色 |
|:---:|:---|:---|:---|
| 1 | `150_cg_lab_triangles` | 点到线有符号距离计算 | `cochlea_geometry.py` 中计算电极-蜗轴距离，评估电极植入位置 |
| 2 | `1068_shallow_water_1d` | Lax-Wendroff 守恒律格式 | `current_continuity.py` 中求解一维电流连续性方程（电荷守恒） |
| 3 | `1210_test_interp` | 多维插值测试数据 | `cochlea_geometry.py` 中三次样条插值重建患者个性化耳蜗轮廓 |
| 4 | `141_cavity_flow_display` | 矢量场数据 thinning 与网格处理 | `cochlea_geometry.py` 中 SGN 神经元图拓扑的网格索引逻辑 |
| 5 | `487_gray_scott_pde` | 9点 Laplacian + 反应-扩散方程 | `laplacian_operator.py` (9点 stencil) + `reaction_diffusion.py` (神经激活斑图) |
| 6 | `619_kepler_perturbed_ode` | 受摄动 ODE 数值积分 | `neural_membrane.py` 中详细 HH 模型的 RK 积分与事件检测 |
| 7 | `1186_svd_faces` | SVD 降维与主成分提取 | `svd_analysis.py` 中患者电场数据降维与电极配置优化 |
| 8 | `462_gegenbauer_polynomial` | Gegenbauer 正交多项式递推 | `potential_field.py` 中球坐标电势的 Gegenbauer 级数展开 |
| 9 | `1082_sinc` | Sinc 函数与带限插值 | `potential_field.py` 中 Whittaker-Shannon sinc 插值验证 |
| 10 | `1325_triangle_witherden_rule` | 高精度对称三角形求积 | `quadrature_integration.py` 中 FEM 刚度矩阵与能量积分的精确计算 |
| 11 | `490_grf_io` | 图结构节点-边 I/O | `cochlea_geometry.py` 中螺旋神经节神经元的图拓扑构建 |
| 12 | `282_differ` | Vandermonde 差分模板矩阵 | `laplacian_operator.py` 中高阶有限差分系数计算 |
| 13 | `055_asa310` | 非中心 Beta CDF 级数算法 | `statistics_patient.py` 中患者神经存活率的统计建模 |
| 14 | `416_fem2d_scalar_display_gpl` | 2D FEM 三角网格求解 | `fem_solver.py` 中 Poisson 方程 Galerkin FEM 求解器 |
| 15 | `100_blood_pressure_ode` | 参数化 ODE 模型（血流动力学） | `neural_membrane.py` 中 FitzHugh-Nagumo 简化膜模型的参数化设计 |

---

## 2. 新增数学物理模型与核心公式

### 2.1 耳蜗几何模型（对数螺旋）

耳蜗蜗轴中心线的参数方程：

$$
r(\theta) = r_0 \exp(-b \theta), \quad \theta \in [0, \theta_{\max}]
$$

$$
x(\theta) = r(\theta) \cos\theta, \quad y(\theta) = r(\theta) \sin\theta
$$

其中 $r_0 = 3.5\,\text{mm}$ 为基底半径，$b = 0.15$ 为螺旋紧缩系数，$\theta_{\max} \approx 4.5\pi$（约 2.25 圈）。

### 2.2 电场泊松方程（准静态近似）

在耳蜗组织（电导率 $\sigma$）中，电极电流 $I_e$ 产生的电势 $V$ 满足：

$$
\nabla \cdot (\sigma \nabla V) = -I_e \delta(\mathbf{r} - \mathbf{r}_e)
$$

**弱形式（Galerkin FEM）**：

对任意测试函数 $w \in H_0^1(\Omega)$：

$$
\int_\Omega \sigma \nabla V \cdot \nabla w \, d\Omega = I_e w(\mathbf{r}_e)
$$

离散后得到线性系统：

$$
\mathbf{K} \mathbf{V} = \mathbf{F}
$$

其中刚度矩阵 $K_{ij} = \sum_{T} \sigma_T A_T (\mathbf{b}_i \cdot \mathbf{b}_j + \mathbf{c}_i \cdot \mathbf{c}_j)$，$\mathbf{b}, \mathbf{c}$ 为线性形函数梯度。

### 2.3 激活函数（Activation Function）

神经刺激理论中，激活函数正比于细胞外电势沿神经纤维方向的二阶空间导数：

$$
\text{AF}(x) = \frac{\partial^2 V}{\partial x^2}
$$

本项目使用节点邻域最小二乘二次曲面拟合估计此二阶导。

### 2.4 Gegenbauer 多项式电势展开

在球坐标近似下，电势可用 Gegenbauer 级数展开：

$$
V(r, \theta) = \frac{I}{4\pi\sigma R} \sum_{n=0}^{\infty} C_n^{(\alpha)}(\cos\theta) \left(\frac{r}{R}\right)^n \frac{1}{n+1}
$$

递推关系：

$$
C_0^{(\alpha)}(x) = 1, \quad C_1^{(\alpha)}(x) = 2\alpha x
$$

$$
C_n^{(\alpha)}(x) = \frac{(2n-2+2\alpha) x C_{n-1}^{(\alpha)}(x) + (-n+2-2\alpha) C_{n-2}^{(\alpha)}(x)}{n}
$$

### 2.5 FitzHugh-Nagumo 神经元膜模型

简化 SGN 膜动力学：

$$
\frac{dV_m}{dt} = \frac{1}{\tau_m}\left(V_m - \frac{V_m^3}{3} - W + I_{\text{stim}}(t)\right)
$$

$$
\frac{dW}{dt} = \varepsilon (V_m + a - bW)
$$

### 2.6 Hodgkin-Huxley 详细模型

$$
C_m \frac{dV_m}{dt} = -g_{\text{Na}} m^3 h (V_m - E_{\text{Na}}) - g_{\text{K}} n^4 (V_m - E_{\text{K}}) - g_L (V_m - E_L) + I_{\text{stim}}
$$

门控变量：

$$
\frac{dx}{dt} = \phi \left[\alpha_x(V)(1-x) - \beta_x(V)x\right], \quad x \in \{m, h, n\}
$$

温度修正因子 $\phi = Q_{10}^{(T - T_{\text{ref}})/10}$，$Q_{10} = 6.3$。

### 2.7 神经激活反应-扩散方程（Gray-Scott 变体）

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u - uv^2 + \gamma(1-u) + I_e(x,t)
$$

$$
\frac{\partial v}{\partial t} = D_v \nabla^2 v + uv^2 - (\gamma + \kappa)v
$$

其中 $u$ 为神经元激活密度，$v$ 为抑制因子，$I_e$ 为外部电刺激。

### 2.8 非中心 Beta 患者变异性模型

神经存活率 $X \sim \text{Beta}(\alpha, \beta, \lambda)$ 的 CDF：

$$
F(x; \alpha, \beta, \lambda) = \sum_{j=0}^{\infty} e^{-\lambda/2} \frac{(\lambda/2)^j}{j!} I_x(\alpha+j, \beta)
$$

其中 $I_x(a,b)$ 为正则化不完全 Beta 函数。

### 2.9 高精度三角形求积（Witherden-Vincent）

参考三角形 $\{(0,0), (1,0), (0,1)\}$ 上的求积：

$$
\int_T f(x,y)\, dA \approx A_T \sum_{i=1}^{n} w_i f(x_i, y_i)
$$

规则精度可达 $p=20$，用于 FEM 刚度矩阵精确组装。

---

## 3. 修改与合成路径

### 3.1 文件结构

```
127_synth_project/
├── main.py                     # 统一入口，零参数运行
├── cochlea_geometry.py         # 耳蜗几何 + 插值 + 距离 + 图拓扑
├── electrode_array.py          # 电极阵列配置与刺激模式
├── fem_solver.py               # 2D Galerkin FEM Poisson 求解器
├── laplacian_operator.py       # 5点/9点/高阶/各向异性 Laplacian
├── potential_field.py          # Gegenbauer + sinc + 解析电势
├── neural_membrane.py          # FHN + HH 膜动力学模型
├── reaction_diffusion.py       # 神经激活反应-扩散斑图
├── quadrature_integration.py   # Witherden 三角形高精度求积
├── svd_analysis.py             # 患者 SVD 降维与电极优化
├── statistics_patient.py       # 非中心 Beta 统计变异性
├── current_continuity.py       # Lax-Wendroff 电流连续性
└── utils.py                    # 数值稳定性与工具函数
```

### 3.2 关键工程改造

1. **MATLAB → Python 迁移**：所有原始 MATLAB/Octave 代码迁移为 Python，使用 NumPy/SciPy 替代矩阵运算。
2. **删除可视化**：移除所有 `figure`, `plot`, `image`, `quiver`, `print('-dpng')` 等可视化代码，仅保留数值计算与文本输出。
3. **零参数运行**：`main.py` 内置全部默认参数，无需命令行输入。
4. **边界处理**：FEM 边界施加 Dirichlet/Neumann 条件；Laplacian 算子边界使用镜像/零梯度；ODE 积分使用事件检测捕捉动作电位发放。
5. **数值鲁棒性**：所有除法操作使用 `safe_divide`；数组操作前检查形状与有限性；电导率、存活率等物理量截断到合理区间。
6. **工程复杂度**：
   - 多物理场耦合（电场 + 神经膜 + 反应扩散）
   - 多尺度数值方法（FEM + FDM + ODE + 统计采样）
   - 患者个性化参数 pipeline（几何插值 → 电场计算 → SVD 降维 → 统计评估）

---

## 4. 合成项目解决的科学问题

本项目可解决以下前沿科学计算问题：

1. **电极-神经界面电场定量预测**：通过 FEM 求解患者个性化耳蜗几何中的 Poisson 方程，预测不同电极配置（单极/双极/三极）下的三维电场分布。

2. **神经激活阈值判定**：结合激活函数 AF = ∂²V/∂x² 与 HH 膜模型，计算特定刺激参数下的神经元发放概率与时空模式。

3. **电流聚焦与串扰评估**：利用反应-扩散模型分析相邻电极间的电流扩散与神经激活斑图重叠，指导电极间距优化。

4. **患者预后统计推断**：基于非中心 Beta 分布建模神经存活率的个体差异，量化不同刺激策略的临床成功率。

5. **参数降维与实时优化**：通过 SVD 将高维患者-电极响应矩阵降维，提取关键模式，为术中实时电极配置优化提供低维决策空间。

---

## 5. 运行方式

### 环境要求
- Python >= 3.8
- NumPy, SciPy

### 运行命令
```bash
cd 127_synth_project
python main.py
```

程序将自动执行以下 10 个计算步骤并输出结果：
1. 耳蜗几何建模与个性化插值
2. 电极阵列配置与刺激模式设置
3. FEM 电场求解与激活函数计算
4. 高阶 Laplacian 与解析电势验证
5. 神经元膜电位响应（FHN + HH）
6. 神经激活反应-扩散斑图演化
7. 高精度三角形数值积分验证
8. SVD 患者参数降维分析
9. 患者变异性统计建模
10. 一维电流连续性方程求解

### 预期输出
所有步骤完成后，终端将显示各模块的数值结果与 `[PASS]` 验证信息，确认物理一致性与数值稳定性。

---

## 6. 质量检查清单

- [x] 原目录未被修改
- [x] 合成后项目为 Python 语言
- [x] 新目录完整包含合成后的项目（13 个 .py 文件）
- [x] 单一博士级科学计算问题已落地为可执行代码
- [x] **全部 15 个输入项目已真实融入合成项目**，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数且无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 无可视化代码残留
