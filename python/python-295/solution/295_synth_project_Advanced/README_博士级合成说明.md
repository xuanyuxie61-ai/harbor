# 惯性约束聚变内爆对称性模拟 — 高阶有限差分与稳定性分析

## 博士级合成项目说明

### 项目概述

本项目围绕 **计算等离子体物理** 前沿领域, 具体研究 **惯性约束聚变 (ICF) 内爆对称性模拟** 中的 **高阶有限差分方法与数值稳定性分析**。项目成功融合了 15 个种子项目的核心算法, 构建了完整的 ICF 内爆模拟研究框架, 包含物理建模、数值方法、稳定性分析、机器学习代理与形式化验证等模块。

**科学问题**: 在国家点火装置 (NIF) 类 ICF 实验中, 内爆对称性偏离 (P₂/P₄ 模式) 是限制聚变产额的关键因素。本项目通过高阶有限差分格式 (WENO5, 紧致差分, 4~8 阶中心差分) 精确捕捉内爆激波与界面不稳定性, 结合 von Neumann 稳定性分析、矩阵谱分析、辛积分哈密顿守恒检验, 系统评估数值方法的可靠性。

---

### 种子项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 在本项目中的角色 |
|------|---------|---------|----------------|
| 1 | [1002] LLZO 分子动力学 | 弹性张量计算, BFGS 松弛, Arrhenius 拟合 | **靶丸材料力学性质**: DT 冰层/烧蚀层弹性常数、体弹模量、剪切模量、Born 稳定性判据 |
| 2 | [1215] 风险决策解码 | SVM 分类, 交叉验证, 数据管道 | **ML 代理模型**: 基于 SVM 的 ICF 不稳定性快速分类器, 5-fold 交叉验证 |
| 3 | [756] mesh_vtoe | 顶点-单元邻接关系 | **网格拓扑**: 构建 vertex-to-element 索引, 用于梯度重构与通量计算 |
| 4 | [1310] triangle_io | TRIANGLE 格式网格 I/O | **网格 I/O**: 三角形节点/单元文件读写, 边界标记管理 |
| 5 | [1324] wandzura_rule | Wandzura 对称三角形求积 | **高斯求积**: 三角形上 1~7 阶精度的对称求积规则, 用于源项积分 |
| 6 | [910] prime | 素数筛 | **低差异序列**: Eratosthenes 素数筛, Halton 序列, 准随机扰动生成 |
| 7 | [1331] triangulation_boundary | 边界边检测 | **边界识别**: 通过边统计识别三角形网格的边界, 施加 Dirichlet 条件 |
| 8 | [972] r8but | 带状上三角矩阵运算 | **矩阵求解**: 带状上三角矩阵的 mv、det、回代求解 |
| 9 | [1297] FormalCellular | 形式化验证, CDF 分析 | **配置空间验证**: 参数空间覆盖率分析, 经验 CDF, 鲁棒性评估 |
| 10 | [1413] welzl/icosahedron | Welzl 包围球, 正二十面体 | **球面网格**: 球谐展开用的正二十面体细分网格, 内爆包络包围球 |
| 11 | [919] product_rule | 多维乘积求积 | **张量积求积**: 一维 Gauss-Legendre → 2D/3D 乘积求积, 球面积分 |
| 12 | [405] fem2d_heat_sparse | 稀疏 FEM, 后向 Euler | **隐式求解**: 稀疏矩阵组装 (Laplace, 质量矩阵), 后向 Euler 热传导 |
| 13 | [619] kepler_perturbed_ode | 哈密顿守恒, 辛积分 | **结构保持积分**: Störmer-Verlet 辛积分, 受扰开普勒哈密顿量守恒检验 |
| 14 | [042] asa144 | 随机列联表 (Boyett) | **扰动分布**: 给定边际约束的随机靶丸表面扰动生成 |
| 15 | [982] r8ge_np | 非主元 LU 分解 | **稠密求解**: 一般稠密矩阵 LU 分解、求解、行列式计算 |

---

### 项目文件结构

```
295_synth_project_Advanced/
├── main.py                    # 统一入口 (零参数运行)
├── icf_physics.py             # ICF 物理: 状态方程, 激波, 输运, 激光耦合
├── mesh_generator.py          # 网格: vtoe, 边界, 球面, 包围球
├── high_order_fd.py           # 高阶有限差分: 中心/WENO5/紧致/柱坐标
├── stability_analysis.py      # 稳定性: von Neumann, CFL, 矩阵谱
├── quadrature_rules.py        # 求积: Wandzura, 乘积, 球面积分
├── time_integrator.py         # 时间积分: FE, SSP-RK, RK4, 后向Euler, 辛
├── linear_solver.py           # 线性代数: 带状矩阵, LU, 稀疏FEM
├── symmetry_decomposition.py  # 对称性: Legendre分解, RT增长
├── perturbation_generator.py  # 扰动: 素数序列, 随机列联表, 功率谱
├── ml_surrogate.py            # ML代理: SVM, 交叉验证, 弹性张量
├── verification.py            # 验证: MMS, 守恒律, 哈密顿, 配置空间
└── README_博士级合成说明.md    # 本文档
```

共 12 个 Python 模块 + 1 个 README, 合计 **13 个文件**。

---

### 核心数学物理模型与公式

#### 1. 控制方程 (轴对称可压缩流体)

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0
$$

$$
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) = -\nabla p + \nabla \cdot \boldsymbol{\tau} + \rho \mathbf{g}
$$

$$
\frac{\partial (\rho E)}{\partial t} + \nabla \cdot [(\rho E + p)\mathbf{u}] = \nabla \cdot (\kappa \nabla T) + \nabla \cdot (\boldsymbol{\tau} \cdot \mathbf{u}) + Q_{\text{nuc}} - Q_{\text{rad}}
$$

#### 2. 理想气体状态方程

$$
p = (\gamma - 1) \rho e_{\text{int}}, \quad T = \frac{p M}{\rho R_{\text{gas}}}, \quad c_s = \sqrt{\frac{\gamma p}{\rho}}
$$

#### 3. Rankine-Hugoniot 激波跳跃

$$
\frac{\rho_2}{\rho_1} = \frac{(\gamma+1)M_s^2}{(\gamma-1)M_s^2 + 2}, \quad \frac{p_2}{p_1} = \frac{2\gamma M_s^2}{\gamma+1} - \frac{\gamma-1}{\gamma+1}
$$

#### 4. Braginskii 离子粘性

$$
\eta_i = 0.96 \, n_i k_B T \tau_i, \quad \tau_i = \frac{3\sqrt{m_i}(k_B T)^{3/2}}{4\sqrt{\pi} n_i Z^4 e^4 \ln\Lambda}
$$

#### 5. Spitzer-Härm 电子热导率

$$
\kappa_{\text{SH}} \approx \frac{1.84 \times 10^{-5} T_e^{5/2}}{Z \ln\Lambda} \quad [\text{erg/(s·cm·K)}]
$$

#### 6. 高阶有限差分 (WENO5 Jiang-Shu)

三个候选模板上的导数近似:
$$
f'^{(0)} = \frac{2f_i + 5f_{i+1} - f_{i+2}}{6}, \quad f'^{(1)} = \frac{-f_{i-1} + 5f_i + 2f_{i+1}}{6}, \quad f'^{(2)} = \frac{2f_{i-2} - 7f_{i-1} + 11f_i}{6}
$$

光滑性指示子:
$$
\beta_0 = \frac{13}{12}(f_i - 2f_{i+1} + f_{i+2})^2 + \frac{1}{4}(3f_i - 4f_{i+1} + f_{i+2})^2
$$

非线性权重:
$$
\alpha_k = \frac{d_k}{(\varepsilon + \beta_k)^2}, \quad \omega_k = \frac{\alpha_k}{\alpha_0 + \alpha_1 + \alpha_2}
$$

#### 7. Von Neumann 放大因子 (Lax-Wendroff)

$$
g(\theta) = 1 - i\nu\sin\theta + \nu^2(\cos\theta - 1), \quad \nu = \frac{c\Delta t}{\Delta x}
$$

稳定条件: $|\nu| \leq 1$

#### 8. CFL 条件

$$
\Delta t \leq C_{\text{CFL}} \frac{\min(\Delta x, \Delta y)}{\max(c_s) + \max(|u|) + \frac{\max(\nu)}{\min(\Delta x, \Delta y)}}
$$

#### 9. Rayleigh-Taylor 增长率

$$
\gamma_{\text{RT}} = \sqrt{A k g_{\text{eff}} - \frac{l(l+1)\sigma}{\rho_h R^3}}, \quad A = \frac{\rho_h - \rho_l}{\rho_h + \rho_l}
$$

#### 10. Legendre 球谐分解 (对称性度量)

$$
R(\theta) = R_0 \left[1 + \sum_{l=0}^{\infty} a_l P_l(\cos\theta)\right]
$$

$$
a_l = \frac{2l+1}{2} \int_0^\pi R(\theta) P_l(\cos\theta) \sin\theta \, d\theta
$$

#### 11. Störmer-Verlet 辛积分

$$
\mathbf{p}^{n+1/2} = \mathbf{p}^n - \frac{\Delta t}{2} \nabla V(\mathbf{q}^n)
$$
$$
\mathbf{q}^{n+1} = \mathbf{q}^n + \Delta t \frac{\mathbf{p}^{n+1/2}}{m}
$$
$$
\mathbf{p}^{n+1} = \mathbf{p}^{n+1/2} - \frac{\Delta t}{2} \nabla V(\mathbf{q}^{n+1})
$$

#### 12. 受扰开普勒哈密顿量

$$
H = \frac{1}{2}(p_1^2 + p_2^2) - \frac{1}{r} - \frac{\delta}{2r^3}, \quad r = \sqrt{q_1^2 + q_2^2}
$$

---

### 运行方式

```bash
cd 295_synth_project_Advanced
python main.py
```

**零参数运行**: 无需任何命令行参数, 直接执行 `python main.py` 即可完成全部 10 个阶段的计算。

---

### 输出说明

程序运行将依次输出 10 个阶段的计算结果:

1. **阶段 1**: ICF 物理参数 (状态方程、激波关系、输运系数、弹性张量)
2. **阶段 2**: 网格生成 (矩形网格、球面网格、包围球、I/O 测试)
3. **阶段 3**: 高阶有限差分 (收敛阶、WENO5、紧致差分、柱坐标算子)
4. **阶段 4**: 稳定性分析 (von Neumann、CFL、矩阵谱、综合评估)
5. **阶段 5**: 求积规则 (Wandzura、乘积规则、球面积分)
6. **阶段 6**: 时间积分 (ODE 比较、辛积分、后向 Euler)
7. **阶段 7**: 线性代数 (带状矩阵、LU 分解、稀疏 FEM)
8. **阶段 8**: 对称性与扰动 (素数筛、列联表、Legendre 分解、RT 增长)
9. **阶段 9**: ML 代理 (SVM 分类、交叉验证、预测)
10. **阶段 10**: 验证 (MMS、哈密顿守恒、配置空间覆盖)

---

### 工程鲁棒性设计

1. **边界处理**: 所有有限差分算子提供单侧、周期、外推三种边界模式
2. **数值保护**: 密度、压强、温度强制非负 (`np.maximum(x, eps)`)
3. **奇点处理**: 除以零保护 (`1/(x+eps)`)，对数域操作 (`np.log(max(x, eps))`)
4. **病态矩阵**: 使用 `lstsq` 替代 `solve`，LU 分解检测零主元
5. **自适应时间步**: PID 控制器，基于误差估计动态调整 Δt
6. **输入验证**: 所有物理参数范围检查 (`ValueError` on invalid)
7. **Welzl 算法退化处理**: 三点共线/共面情况特殊处理

---

### 科学计算特色

- **多尺度物理**: 从离子碰撞时间 (ps) 到内爆时间 (ns) 的跨尺度模拟
- **多方法融合**: 有限差分 + 有限元 + 谱方法 + 机器学习
- **可复现性**: 固定随机种子，确定性算法，小规模网格 (~1000 节点)
- **验证驱动**: MMS 收敛性验证、守恒律检验、哈密顿量漂移监测
- **前沿性**: 涵盖 ICF 研究的最新进展 (对称性控制、RT 抑制、ML 加速)

---

### 依赖

- Python 3.8+
- NumPy
- SciPy
- scikit-learn (可选, 用于对比)

```bash
pip install numpy scipy
```

---

### 运行时间

典型运行时间: **~2 秒** (在普通笔记本上)，小规模网格 (N ≤ 200)，适合教学演示与方法验证。
