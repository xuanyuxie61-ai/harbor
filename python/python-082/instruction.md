# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 复合材料损伤多尺度模拟平台 (python-082)

## 1. 项目整体目标
本项目实现了一个纤维增强复合材料层合板在循环载荷下的渐进损伤演化与失效预测的多尺度计算框架。它整合了微观代表性体积单元（RVE）几何、细观力学均匀化、连续损伤力学（CDM）疲劳演化、经典层合板理论（CLT）、显式/隐式求解器、波传播数值方法、特征值稳定分析以及铺层优化等功能。主入口为 `main.py`，其余 `.py` 文件提供被 `main.py` 调用的各类专用模块。

## 2. 文件清单与职责概述

| 文件名 | 职责 |
|--------|------|
| `main.py` | 平台主流程：串联 RVE 生成、材料均匀化、层合板刚度、损伤演化、波传播、屈曲、优化等，输出结果汇总 |
| `rve_geometry.py` | 代表性体积单元几何：多边形纤维定义、纤维体积分数计算、背景网格生成、二维 Vandermonde 插值 |
| `material_properties.py` | 细观力学均匀化：Halpin‑Tsai 与混合律计算等效弹性常数；`CompositeMaterial` 类与创建函数 |
| `material_model.py` | 连续损伤本构：Hashin 失效准则、指数损伤演化、退化刚度矩阵、层合板 ABD 矩阵 |
| `mesh_geometry.py` | 一维非均匀网格生成与铺层定义：正弦加密映射、层合板聚合、`CompositePly` 与 `Mesh1D` 类 |
| `composite_mesh.py` | 复合材料层合板三维四边形网格生成：节点、单元、铺层标识、纤维角度、面积/法向计算、旋转矩阵 |
| `damage_evolution.py` | 载荷循环下的 van der Pol 型快‑慢损伤 ODE、Paris 定律裂纹扩展、Miner 累积损伤规则 |
| `damage_ode.py` | 相场损伤模型、快‑慢 stiff ODE 系统、疲劳损伤累积模型及 Van der Pol 型周期估算 |
| `damage_mechanics.py` | CDM 疲劳损伤演化 ODE（纤维/基体/剪切/界面）、Hashin 准则驱动、损伤耗散能与失效周期估计 |
| `stiffness_assembly.py` | 层合板 ABD 矩阵计算、退化刚度更新、带状上三角求解器、残差与条件数评估 |
| `sparse_assembler.py` | 稀疏矩阵 COO 格式组装、全局刚度装配、Dirichlet 处理、带状存储与求解、条件数估计 |
| `nonlinear_solver.py` | Newton‑Raphson 非线性求解器、弧长法控制、线性搜索、带状系统求解与 LINPACK 风格残差基准 |
| `dg_solver.py` | 一维间断伽辽金（DG）谱元法应力波求解器：GLL 节点、质量/刚度矩阵、迎风通量、LSERK45 时间推进 |
| `spectral_element.py` | 一维 DG 谱元框架（Jacobi 多项式、微分矩阵、RK 时间积分），针对复合材料杆中弹性波传播 |
| `eigen_analysis.py` | 结构模态分析、测试矩阵生成、稳定性区域边界估计、频率漂移与失稳指标 |
| `eigen_buckling.py` | 层合板屈曲载荷与自由振动频率求解：有限差分离散、广义特征值问题、正交矩阵生成 |
| `eigenvalue_analysis.py` | 广义特征值屈曲分析、模态质量归一化、损伤敏感性评估 |
| `quadrature_rules.py` | 高斯‑勒让德、高斯‑赫尔米特求积节点与权重、Vandermonde 求积、随机强度概率积分 |
| `quadrature_integrals.py` | J 积分、VCCT 能量释放率、概率化强度期望计算、Vandermonde 求积权重、Hermite 单项式精确积分 |
| `stability_analysis.py` | 时间积分绝对稳定性分析：R(z) 函数、稳定性区域网格、A‑稳定性检验、损伤 ODE 的 CFL 诊断 |
| `stability_solver.py` | 损伤演化 ODE Jacobian 特征值分析、最大稳定步长推荐、稳定性区域判断 |
| `optimization.py` | 全局优化（Brent 风格）与动态规划求解层合板最优失效序列 |
| `optimization_design.py` | 层合板铺层角度全局优化（单角度、离散动态规划），目标函数包含屈曲与制造约束 |
| `wave_pde.py` (未提供源码，但被 main 调用) | 一维应力波传播 PDE 求解器，包含衰减系数与界面反射/透射系数 |

## 3. 模块核心内容与边界

### 3.1 几何与 RVE 模块
- **rve_geometry.py**  
  类 `PolygonalFiber` 表示正多边形纤维截面，存储中心、半径、边数、节点和面片。`RVEGeometry` 管理矩形区域内的纤维集合，计算纤维体积分数，判断点是否在纤维内，并提供二维多项式插值方法 `vandermonde_interp_2d_field`。工厂函数 `generate_hexagonal_fiber_rve` 生成六边形排列的纤维 RVE。

- **mesh_geometry.py**  
  类 `CompositePly` 定义单层属性：厚度、角度、弹性常数，并预计算平面应力偏轴刚度 `Qbar`。`CompositeLaminate` 聚合多层铺层，计算等效厚度和轴向刚度。`Mesh1D` 利用正弦映射生成一维非均匀网格，支持中部加密，并提供定位、Jacobi 等功能。

- **composite_mesh.py**  
  类 `CompositeMesh` 管理三维四边形网格，存储节点、单元、铺层 ID、纤维角度。类方法 `generate_laminate` 按层生成规则网格。提供单元形心、面积（四边形三角剖分）、法向、总表面积、旋转矩阵（绕 z 轴）和 Reuter 矩阵计算。

### 3.2 材料均匀化与连续损伤
- **material_properties.py**  
  类 `CompositeMaterial` 接受纤维/基体弹性常数和纤维体积分数，内部计算纤维和基体的剪切模量，采用混合律和 Halpin‑Tsai 方法计算 `E1, E2, G12, nu12`，并得到平面应力柔度矩阵 `S` 和刚度矩阵 `Q`。还提供偏轴刚度转换和退化刚度计算。工厂函数 `create_carbon_epoxy` 使用典型 T300/环氧值。

- **material_model.py**  
  类 `CompositeMaterial`（冗余但侧重不同）包含完整的 Hashin 失效指标评估、指数软化损伤演化（需要等效应变比和特征长度）、热力学驱动力计算、以及 `LaminateProperties` 类计算标准 ABD 矩阵和等效工程常数。退化刚度通过损伤张量 `M` 作用在初始刚度上得到。

### 3.3 损伤演化 ODE 系统
- **damage_evolution.py**  
  类 `CyclicDamageModel` 封装快‑慢损伤 ODE（立方非线性 + 谐波驱动），提供 RK4 积分器、Paris 裂纹扩展积分、van der Pol 渐近周期估计、磁滞回线能量计算。模块级函数 `cumulative_damage_miner` 实现 Miner 线性累积。

- **damage_ode.py**  
  三类模型：`PhaseFieldDamageModel`（相场损伤演化率由驱动力和临界值决定）；`FastSlowDamageODE`（三维立方快方程与阈值慢变量）；`FatigueDamageModel`（Basquin 型损伤累积，Goodman 修正，积分得到循环‑损伤关系和渐近时间尺度）。

- **damage_mechanics.py**  
  `DamageParameters` 容器定义 Hashin 强度和疲劳参数。`DamageState` 是四维损伤状态向量。模块函数 `hashin_failure_criteria` 计算纤维拉伸/压缩、基体拉伸/
