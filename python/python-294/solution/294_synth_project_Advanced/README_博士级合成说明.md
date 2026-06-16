# 博士级合成说明：激光等离子体相互作用高阶有限差分仿真

## 项目概述

本项目将 15 个不同领域的科研代码项目融合为一个面向**计算等离子体物理**前沿问题的博士级计算仿真系统。核心科学问题是：

**强激光脉冲与非均匀等离子体靶的一维相互作用——高阶有限差分格式的设计、稳定性分析与可复现数值实验。**

## 科学问题描述

### 物理模型

激光等离子体相互作用 (Laser-Plasma Interaction, LPI) 是惯性约束聚变 (ICF)、激光粒子加速、实验室天体物理等领域的核心物理过程。本项目使用最基本的 **1D1V Vlasov-Maxwell 方程组**描述这一过程：

**Vlasov 方程** (电子相空间分布函数演化):
$$\frac{\partial f}{\partial t} + v_x \frac{\partial f}{\partial x} - \frac{e}{m_e}\left(E_x + v_y B_z\right)\frac{\partial f}{\partial v_x} = C[f]$$

**Maxwell 方程组**:
$$\frac{\partial E_y}{\partial t} = c^2 \frac{\partial B_z}{\partial x} - \frac{J_y}{\varepsilon_0}$$
$$\frac{\partial B_z}{\partial t} = -\frac{\partial E_y}{\partial x}$$

**电流密度**:
$$J(x,t) = -e \int v \, f(x,v,t) \, dv$$

**冷等离子体色散关系**:
$$\omega^2 = \omega_p^2 + c^2 k^2$$

**Bohm-Gross 色散关系** (Langmuir 波):
$$\omega^2 = \omega_p^2 + 3k^2 v_{th}^2$$

### 数值挑战

1. **高阶精度需求**: 标准 2 阶差分的数值色散误差 O((kh)²) 无法正确模拟波的传播，需要 4-8 阶格式
2. **多尺度耦合**: 电子德拜长度 λ_D << 激光波长 λ_L << 系统尺寸 L
3. **非线性效应**: 相对论有质动力、参量不稳定性 (SRS, TPI)
4. **长时间稳定性**: 需要满足 CFL 条件，避免数值不稳定性增长

## 原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 在本项目中的角色 |
|------|--------|----------|-----------------|
| 1 | 801_newton_maehly | Newton-Maehly 多项式求根 | 同时求解等离子体色散多项式的所有根，确定允许传播的电磁模式 |
| 2 | 480_gram_schmidt | 经典/修正 Gram-Schmidt 正交化 | 对电磁场快照进行模式分解，提取正交空间模式和时间系数 |
| 3 | 1242_PhotonDosReference | 光子态密度计算 | 计算等离子体电磁模式 (横波+纵波) 的态密度 D(ω) |
| 4 | 538_histogram_data_2d | 2D 直方图逆 CDF 采样 | 从 2D 相空间分布 (x,v) 中采样初始化粒子 |
| 5 | 786_nas | NAS 核心算子基准 | 验证核心数值算子 (FFT, 矩阵运算) 的正确性 |
| 6 | 698_log_normal | Log-normal 分布函数集 | 模拟超热电子的能量分布 (激光尾波场加速产生) |
| 7 | 283_diffusion_pde | 扩散 PDE 方法线求解 | 实现 Fokker-Planck 碰撞算子的速度空间扩散 |
| 8 | 272_dg1d_burgers | DG 方法求解 Burgers 方程 | 将 DG 数值通量方法迁移到 Vlasov 方程的速度空间对流 |
| 9 | 1216_borexino | 贝叶斯 MCMC 参数估计 | 从仿真诊断数据反推等离子体参数 (密度、温度、梯度长度) |
| 10 | 1393_vin | VIN 校验和算法 | 验证仿真参数向量的完整性，防止配置被意外修改 |
| 11 | 995_r8sm | Sherman-Morrison 公式 | 高效处理时变介电常数导致的系统矩阵低秩更新 |
| 12 | 1080_safe_CCMPC | 抽象基类场景管理 | 构建场景化的边界条件管理架构 |
| 13 | 150_cg_lab_triangles | 有符号距离计算 | 计算等离子体边界的几何特性 (法线、距离) |
| 14 | 057_atbash | Atbash 替换密码 | 参数标签的对称编码/解码 |
| 15 | 358_fd1d_bvp | 有限差分 BVP 求解 | 求解等离子体平衡态密度剖面边值问题 |

## 新增数学物理模型与核心公式

### 1. 高阶有限差分算子

**2p 阶中心差分一阶导数**:
$$(D_h f)_i = \frac{1}{h}\sum_{j=1}^{p} a_j (f_{i+j} - f_{i-j}) + O(h^{2p})$$

修正波数:
$$k_{eff} h = \sum_{j=1}^{p} a_j \sin(jkh)$$

### 2. 冯·诺伊曼稳定性分析

放大因子 G 的稳定性条件:
$$|G(kh)| \leq 1 + C \cdot dt \quad \forall k$$

**Leapfrog**: $G = i\nu(k_{eff}h) \pm \sqrt{1 - \nu^2(k_{eff}h)^2}$

**RK4**: $G = 1 + z + z^2/2 + z^3/6 + z^4/24$, $z = -i\nu \cdot k_{eff}h$

### 3. Newton-Maehly 求根

$$z_i^{(n+1)} = z_i^{(n)} - \frac{p(z_i)}{p'(z_i) - p(z_i) \cdot S_i}$$
$$S_i = \sum_{j \neq i} \frac{1}{z_i - z_j}$$

### 4. Fokker-Planck 碰撞算子

$$C[f] = \nu_{ei} \frac{\partial}{\partial v}\left[v f + \frac{v_{th}^2}{2}\frac{\partial f}{\partial v}\right]$$

### 5. Coulomb 对数

$$\ln\Lambda = \ln\left(\frac{\lambda_D}{b_{min}}\right), \quad b_{min} = \max\left(\frac{e^2}{4\pi\varepsilon_0 k_BT_e}, \frac{\hbar}{m_e v_{th}}\right)$$

### 6. 贝叶斯后验

$$p(\theta|D) \propto \exp\left(-\frac{1}{2}\sum_i \frac{(D_{model,i}(\theta) - D_{obs,i})^2}{\sigma_i^2}\right) \cdot p(\theta)$$

## 项目文件结构

```
294_synth_project_Advanced/
├── main.py                      # 统一入口 (零参数运行)
├── config.py                    # 物理常数与仿真参数
├── numerical_core.py            # VIN校验和 + Atbash编码 + 数值验证
├── fd_operators.py              # 高阶有限差分算子 (2/4/6/8阶)
├── stability.py                 # Von Neumann稳定性分析
├── dispersion_solver.py         # Newton-Maehly色散关系求解
├── mode_decomposition.py        # Gram-Schmidt模式分解
├── dos_plasma.py                # 等离子体态密度计算
├── distribution_sampler.py      # 粒子分布采样 (log-normal + 2D直方图)
├── collision_operator.py        # Fokker-Planck碰撞算子
├── vlasov_maxwell.py            # Vlasov-Maxwell DG求解器
├── boundary_conditions.py       # 边界条件处理
├── boundary_geometry.py         # 边界几何计算
├── matrix_update.py             # Sherman-Morrison矩阵更新
├── diagnostics_bayesian.py      # 贝叶斯MCMC参数诊断
└── README_博士级合成说明.md      # 本文档
```

**文件总数**: 15 个 Python 源文件 + 1 个 README = 16 个文件

## 仿真流程

程序运行 13 个阶段:

1. **物理参数设置与校验** - 初始化常数，VIN 校验和验证
2. **高阶有限差分算子验证** - 测试 2/4/6/8 阶精度
3. **Von Neumann 稳定性分析** - CFL 条件、放大因子计算
4. **色散关系求解** - Newton-Maehly 求根，双束不稳定性分析
5. **模式分解** - CGS vs MGS 比较，电磁模式提取
6. **分布函数初始化** - Maxwell + log-normal 超热尾部
7. **态密度计算** - 电磁模式 + Langmuir 模式 DOS
8. **边界几何分析** - 临界面、梯度尺度长度、有符号距离
9. **碰撞算子** - Coulomb 对数、Spitzer 电阻率、弛豫仿真
10. **矩阵更新** - Sherman-Morrison 处理介电常数变化
11. **Vlasov-Maxwell 仿真** - DG 时间推进 (简化版)
12. **贝叶斯诊断** - MCMC 参数估计
13. **总结** - 结果汇总与验证

## 运行方式

```bash
cd 294_synth_project_Advanced
python main.py
```

无需任何参数，程序自动完成所有计算阶段并输出结果。

## 合成后解决的科学问题

本项目通过融合 15 个不同领域的算法，解决了以下科学计算问题：

1. **高阶格式精度评估**: 比较 2-8 阶差分格式在等离子体波传播中的色散误差
2. **稳定性边界确定**: 通过 Von Neumann 分析确定各格式组合的 CFL 限制
3. **模式识别**: 通过色散多项式求根和 Gram-Schmidt 分解识别等离子体中允许的模式
4. **能量耦合分析**: 通过态密度计算分析激光能量耦合到不同等离子体模式的效率
5. **碰撞效应量化**: 通过 Fokker-Planck 算子评估碰撞对分布函数演化的影响
6. **参数反演**: 通过贝叶斯 MCMC 从观测数据推断等离子体内部参数

## 关键技术特点

1. **博士级难度**: 融合等离子体物理、数值分析、统计学等多学科前沿知识
2. **公式密集**: 每个模块包含完整的物理推导和数值格式说明
3. **工程鲁棒性**: 所有模块包含边界检查、NaN/Inf 检测、物理量合理性验证
4. **可复现性**: 固定随机种子，参数校验和追踪，完整计算日志
5. **零参数运行**: main.py 统一入口，无需任何配置即可运行

## 依赖

- Python >= 3.7
- NumPy >= 1.18
- SciPy >= 1.4 (用于 brentq, erf, erfinv)

## 参考文献

1. Birdsall, C.K. & Langdon, A.B. *Plasma Physics via Computer Simulation* (IOP, 2004)
2. Hesthaven, J.S. & Warburton, T. *Nodal Discontinuous Galerkin Methods* (Springer, 2008)
3. Tikhonchuk, V. et al. *Physics of Laser-Plasma Interactions* (Cambridge, 2016)
4. Goodman, J. & Weare, J. "Ensemble MCMC sampling" *Comm. App. Math. Comp. Sci.* 5, 65 (2010)
