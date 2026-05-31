# 魔角扭转双层石墨烯能带工程与量子输运性质计算平台

## 项目概述

本项目围绕**凝聚态物理：二维材料异质结能带工程**这一前沿领域，将15个基础科研代码项目的核心算法融合为一个面向博士级科学计算的综合平台。研究对象为**魔角扭转双层石墨烯（Magic-Angle Twisted Bilayer Graphene, MATBG）**，该系统在约1.05°的魔角附近展现出超导电性、关联绝缘体态、非平庸拓扑等新奇量子现象，是当前凝聚态物理研究的最前沿课题之一。

## 科学问题与核心模型

### 物理背景

扭转双层石墨烯由两层石墨烯以微小转角θ堆叠而成。当θ≈1.05°时，层间耦合在莫尔超晶格尺度上产生近乎平展的能带，电子关联效应被极度增强。系统的低能有效哈密顿量可写为

```
Ĥ = Σ_n ε_n c_n^† c_n + Σ_{⟨n,m⟩} t_{nm} c_n^† c_m + Σ_{n∈A,m∈B} w(r_n−r_m) c_n^† c_m + h.c.
```

其中：
- **层内跃迁**采用Slater-Koster型：t(r,φ) = t₀ exp[−(r−a_CC)/δ₀] · [cos²φ + α sin²φ]
- **层间耦合**采用Bistritzer-MacDonald高斯模型：w(r) = w₀ exp(−r²/2ξ²)
- **莫尔周期**：L_M = a / (2 sin(θ/2))
- **莫尔倒格矢**：|q| = (8π/3a) sin(θ/2)

### 注入的科学公式

本项目系统注入了大量凝聚态物理、量子力学和数值分析公式：

1. **紧束缚哈密顿量**：包含最近邻、次近邻层内跃迁及位置依赖的层间耦合
2. **布洛赫定理**：ψ_nk(r) = e^{ik·r} u_nk(r)
3. **泊松方程**：−∇²V_ℓ = ρ_ℓ/ε，用于自洽层间电势计算
4. **态密度与高阶Van Hove奇异点**：D(E) = Σ_{n,k} δ(E−E_n(k))， saddle点附近对数发散
5. **半经典运动方程**：ħ dk/dt = −e(E + v×B)，dr/dt = (1/ħ)∇_k E_n(k)
6. **Berry曲率与Chern数**：Ω_n(k) = ∇_k × ⟨u_nk|i∇_k|u_nk⟩，C_n = (1/2π)∫_BZ Ω_n d²k
7. **Z₂拓扑不变量（Fu-Kane公式）**：(−1)^ν = Π_{K_i} sgn[Pf(⟨u_m|Θ|u_n⟩)]
8. **Wilson loop（Wilson圈）**：W = Π_j ⟨u_{k_j}|u_{k_{j+1}}⟩，Berry相位φ = −Im[ln W]
9. **CVT能量泛函**：E = Σ_j ∫_{V_j} ρ(k)|k−g_j|² d²k
10. **最小二乘拟合**：χ²({t}) = Σ_i w_i [E_ref(k_i) − E_TB(k_i;{t})]²

## 原项目到科学问题的映射

| 原项目 | 核心算法 | 在合成项目中的角色 |
|--------|---------|------------------|
| 015_approx_leastsquares | 切比雪夫节点最小二乘拟合、拉格朗日基函数 | `fitting.py`：从DFT参考数据拟合紧束缚参数t₀、w₀等 |
| 1044_roots_rc | 反向通信多步割线法求根 | `root_finder.py`：搜索能带简并点（Dirac点），求解自洽方程 |
| 877_poisson_2d | 二维泊松方程五点差分+Jacobi迭代 | `poisson_solver.py`：自洽计算层间静电势分布 |
| 101_blowup_ode | ODE爆破现象分析（y'=y²） | `dos_utils.py`：分析Van Hove奇异点的发散行为（D_peak ∝ ln(1/σ)） |
| 1338_triangulation_l2q | 线性三角剖分→二次升阶 | `mesh_utils.py`：莫尔超晶格网格的P1→P2单元升级 |
| 1037_rk45 | Runge-Kutta 4/5阶自适应ODE积分 | `semiclassical_dynamics.py`：布洛赫电子半经典运动方程积分 |
| 065_ball_and_stick_display | Lax-Wendroff双曲型PDE格式 | `semiclassical_dynamics.py`：Lax-Wendroff预测-校正步作为快速二阶积分器 |
| 243_cvt_1d_lloyd | Lloyd重心Voronoi镶嵌 | `kpoint_sampling.py`：k空间自适应最优采样点生成 |
| 620_kmeans | K-Means/H-Means聚类 | `clustering.py`：莫尔超晶格AA/AB/BA堆垛区域聚类分类 |
| 1355_tridiagonal_solver | Thomas算法（TDMA） | `tridiagonal.py`：一维层链电势求解、三对角系统快速求解 |
| 1430_zero_laguerre | Laguerre多项式求根迭代 | `root_finder.py`：Laguerre-inspired梯度方向优化用于能带交叉搜索 |
| 955_quadrilateral_mesh_rcm | 反向Cuthill-McKee重排序 | `mesh_utils.py`：紧束缚稀疏哈密顿量带宽缩减 |
| 561_hypercube_surface_distance | 高维蒙特卡洛采样 | `kpoint_sampling.py`：Voronoi单元重心计算的蒙特卡洛积分 |
| 1310_triangle_io | TRIANGLE网格文件I/O | `mesh_utils.py`：网格数据的.node/.ele格式读写 |
| 672_lights_out | 模2线性代数（F_2矩阵运算） | `topology.py`：Z₂拓扑数的奇偶性计算、Lights-Out矩阵秩分析 |

## 合成后的代码结构

```
016_synth_project/
├── main.py                          # 统一入口，零参数运行
├── tight_binding.py                 # 紧束缚哈密顿量构建
├── band_solver.py                   # 能带对角化、费米面、Dirac点搜索
├── poisson_solver.py                # 2D泊松方程自洽求解（SCF）
├── kpoint_sampling.py               # CVT自适应k点采样
├── root_finder.py                   # 能带简并点/自洽方程求根
├── tridiagonal.py                   # Thomas算法、一维层链电势
├── mesh_utils.py                    # 三角网格升阶、RCM重排序、网格I/O
├── semiclassical_dynamics.py        # 半经典动力学（RK45 + Lax-Wendroff）
├── dos_utils.py                     # 态密度、Van Hove奇异点、Fermi速度
├── clustering.py                    # K-Means/H-Means堆垛区域聚类
├── topology.py                      # Berry曲率、Chern数、Z₂、Wilson loop
└── fitting.py                       # 最小二乘参数拟合、交叉验证
```

## 运行方式

```bash
cd Synthesis-project-python/016_synth_project
python main.py
```

程序无需任何参数，内部已配置魔角θ=1.05°、3×3超胞等默认参数。运行后将依次输出：

1. 紧束缚哈密顿量维度与原子信息
2. 自洽泊松求解收敛历史
3. Γ→M→K→Γ高对称路径能带结构及费米能级
4. 态密度与Van Hove奇异点位置
5. 半经典电子动力学轨迹
6. 堆垛区域聚类统计
7. 拓扑不变量（Wilson loop Berry相位）
8. 稀疏矩阵RCM带宽缩减效果
9. CVT自适应k点分布
10. 紧束缚参数最小二乘拟合结果
11. Thomas算法求解验证
12. 能带交叉点数值搜索

## 工程鲁棒性设计

- **边界处理**：所有物理函数均含输入合法性检查（角度范围、网格尺寸、正定性等）
- **数值稳定性**：
  - 哈密顿量厄米性强制对称化：H = (H + H†)/2
  - 三对角求解中零主元检测与退化格点向量检测
  - Lagrange基函数中重复节点检测
  - SOR松弛因子安全范围限制
- **容错机制**：
  - SCF循环失败时自动回退到非自洽哈密顿量
  - Wilson loop/Chern数计算失败时优雅降级
  - 三角剖分不可用时跳过网格操作
- **精度控制**：Jacobi迭代Frobenius范数残差监控、RK45局部截断误差自适应步长

## 科学计算复杂度

本项目达到博士级计算复杂度，体现在：

- **多物理场耦合**：紧束缚模型 + 静电学自洽 + 半经典输运 + 拓扑量子数
- **高维数值方法**：
  - 自适应RK45积分（6级Butcher表）
  - CVT最优采样（Lloyd迭代能量泛函最小化）
  - 三角剖分升阶与RCM稀疏矩阵优化
- **前沿物理模型**：魔角石墨烯紧束缚、Berry相位、Z₂拓扑分类
- **大规模公式-算法-代码一致性**：文档与代码中物理公式严格对应

## 作者与许可

本项目为科研代码合成教学/研究用途，基于用户提供的15个种子项目算法融合构建。所有原始代码逻辑已按照MIT风格学术惯例进行重构和扩展。
