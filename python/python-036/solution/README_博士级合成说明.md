# 中微子振荡与质量 Hierarchy 综合分析平台

## 项目概述

本项目是一个面向**粒子物理前沿问题**的博士级 Python 科学计算平台, 核心科学问题为:

> **中微子振荡与质量 Hierarchy (Normal vs Inverted) 的数值判定**

中微子振荡是粒子物理学中最深刻的发现之一, 它揭示了中微子具有非零质量, 且三种味本征态 (ν_e, ν_μ, ν_τ) 与三种质量本征态 (ν_1, ν_2, ν_3) 之间存在非平凡的幺正混合 (PMNS 矩阵)。本项目基于 15 个科研代码项目的核心算法, 融合构造了一个完整的中微子振荡计算框架。

---

## 科学背景与核心公式

### 1. PMNS 混合矩阵

Pontecorvo–Maki–Nakagawa–Sakata (PMNS) 矩阵描述味基与质量基之间的幺正变换:

```
|ν_α⟩ = Σ_i U_{αi} |ν_i⟩
```

标准参数化 (PDG 约定):

```
U = R₂₃(θ₂₃) · R₁₃(θ₁₃, δ_CP) · R₁₂(θ₁₂)
```

其中 R_{ij}(θ) 为实空间旋转矩阵, δ_CP 为 CP 破坏相位。

### 2. 真空中微子哈密顿量

在超相对论极限下 (E ≫ m_i), 中微子传播由有效薛定谔方程描述:

```
i d|ν_α⟩/dt = H_vac |ν_α⟩
```

其中:

```
H_vac = (1 / 2E) · U · diag(0, Δm²₂₁, Δm²₃₁) · U†
```

### 3. 物质中的 MSW 效应

电子中微子通过与电子的带电流 (CC) forward scattering 获得额外相位:

```
H_mat = H_vac + V_CC · diag(1, 0, 0)
```

其中 V_CC = √2 G_F N_e 为物质势。

**MSW 共振条件** (双味近似):

```
2√2 G_F N_e^res E = Δm² cos(2θ)
```

### 4. 振荡概率

```
P(ν_α → ν_β; E, L) = |⟨ν_β| exp(-i H L)|ν_α⟩|²
```

### 5. Jarlskog 不变量

CP 破坏的度量:

```
J_CP = Im[ U_{e1} U_{μ2} U*_{e2} U*_{μ1} ]
     = sin(θ₁₂) sin(θ₂₃) sin(θ₁₃) cos(θ₁₂) cos(θ₂₃) cos²(θ₁₃) sin(δ_CP)
```

### 6. 质量 Hierarchy

**Normal Hierarchy (NH):**
- m₁ < m₂ < m₃
- Δm²₂₁ = m²₂ - m²₁ > 0
- Δm²₃₁ = m²₃ - m²₁ > 0

**Inverted Hierarchy (IH):**
- m₃ < m₁ < m₂
- Δm²₂₁ = m²₂ - m²₁ > 0
- Δm²₃₁ = m²₃ - m²₁ < 0

---

## 输入种子项目映射

| 原始项目 | 核心算法 | 本项目中的角色 |
|---------|---------|--------------|
| 403_fem2d_heat | 2D FEM, T6 基函数, 带状矩阵 | `matter_profile_fem2d.py`: 地球截面二维密度分布 FEM 求解 |
| 235_cube_monte_carlo | 3D 蒙特卡洛采样 | `monte_carlo_oscillation.py`: 能量-角度-基线三维参数空间采样 |
| 391_fem1d_heat_implicit | 1D FEM, 向后 Euler, 质量矩阵 | `matter_profile_fem1d.py`: 地球径向密度剖面 FEM 求解 |
| 172_chladni_figures | 广义本征值问题, 稀疏矩阵 | `neutrino_hamiltonian.py`: 中微子哈密顿量本征值分析 |
| 704_luhn | 校验和算法 | `validation_utils.py`: PMNS 矩阵幺正性校验与概率守恒验证 |
| 1149_square_monte_carlo | 2D 蒙特卡洛采样 | `monte_carlo_oscillation.py`: CP 相位-混合角二维参数采样 |
| 1269_toms291 | Gamma 函数对数 | `neutrino_hamiltonian.py`: Fermi-Dirac 分布计算 (中微子产生) |
| 428_file_increment | 文件索引偏移 | `data_io.py`: 网格索引 0-based/1-based 转换与数据持久化 |
| 946_quad2d | 2D 矩形中点积分 | `numerical_integration.py`: (E, L) 平面上振荡概率二维积分 |
| 343_euler | Euler ODE 求解 | `neutrino_ode_solver.py`: 中微子味演化 Euler 方法 |
| 972_r8but | 带状上三角矩阵求解 | `neutrino_ode_solver.py`: 大规模离散系统的 R8BUT 求解器 |
| 1236_tet_mesh_quality | 四面体网格质量度量 | `mesh_utils.py`: 地球三维网格质量评估 |
| 845_pagerank2 | 稀疏矩阵幂迭代 | `sparse_iterative.py`: 主导振荡模式迭代求解 |
| 749_medit_to_ice | MESH 文件解析 | `mesh_utils.py`: 网格数据解析与结构化读取 |
| 183_circle_rule | 圆上等距积分 | `numerical_integration.py`: CP 破坏相位周期性积分 |

---

## 项目结构

```
036_synth_project/
├── main.py                          # 统一入口, 零参数运行
├── constants.py                     # 物理常数, PMNS 参数, PREM 密度模型
├── pmns_matrix.py                   # PMNS 矩阵构造与验证
├── neutrino_hamiltonian.py          # 哈密顿量, 本征值, MSW 共振
├── matter_profile_fem1d.py          # 1D FEM 地球密度剖面
├── matter_profile_fem2d.py          # 2D FEM 地球截面密度
├── monte_carlo_oscillation.py       # MC 参数不确定性与概率积分
├── numerical_integration.py         # 2D/圆上/自适应数值积分
├── neutrino_ode_solver.py           # Euler/RK4/矩阵指数/R8BUT ODE 求解
├── mesh_utils.py                    # 四面体网格质量与解析
├── sparse_iterative.py              # 幂迭代与 PageRank 风格分析
├── validation_utils.py              # 幺正性/概率守恒/厄米性验证
├── data_io.py                       # 数据文件读写与索引转换
└── README_博士级合成说明.md         # 本文档
```

---

## 运行方法

```bash
cd Synthesis-project-python/036_synth_project
python main.py
```

程序将自动执行所有计算任务并输出结果, 无需任何参数。

---

## 核心计算流程

1. **PMNS 矩阵构造**: 使用标准参数化构建 3×3 复数幺正矩阵
2. **哈密顿量求解**: 真空与物质中的有效哈密顿量本征值分解
3. **有限元密度模型**: 1D/2D FEM 求解 PREM 地球密度剖面
4. **蒙特卡洛分析**: 对 PMNS 参数不确定度进行 5000+ 次采样
5. **数值积分**: 在 (E, L) 参数空间上二维积分振荡概率
6. **ODE 演化**: Euler/RK4/矩阵指数三种方法求解味演化方程
7. **网格评估**: 四面体质量度量确保数值网格非退化
8. **迭代求解**: 幂迭代找到主导振荡模式
9. **Hierarchy 判定**: 基于 Δm²₃₁ 符号的统计显著性分析
10. **验证**: 幺正性、概率守恒、厄米性全面校验

---

## 数值鲁棒性

- 所有矩阵操作包含奇异值检查与最小二乘回退
- 混合角严格限制在 (0, π/2) 物理范围内
- 概率计算自动裁剪到 [0, 1] 区间
- 物质势包含负半径/超半径的边界处理
- 有限元矩阵组装包含退化单元检测
- ODE 求解器支持自适应步长 (Euler/RK4)

---

## 科学意义

本项目实现了一个**完整的中微子振荡数值实验平台**, 可用于:
- 评估下一代中微子实验 (DUNE, Hyper-Kamiokande) 的 sensitivity
- 量化质量 hierarchy 判别的统计显著性
- 模拟地球物质效应对长基线实验的影响
- 研究 CP 破坏相位的测量精度
