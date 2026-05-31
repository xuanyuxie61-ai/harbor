# 圆柱壳屈曲与后屈曲路径跟踪 — 博士级科研代码合成说明

## 一、项目概述

本项目围绕**结构力学：壳体屈曲后屈曲路径跟踪**这一前沿科学问题，将15个原始科研代码项目的核心算法融合重构为一个完整的Python科研计算程序。项目实现了基于Donnell-Mushtari-Vlasov（DMV）壳理论的圆柱壳有限元模型，集成线性屈曲特征值分析、Newton-Raphson非线性求解、Riks-Wempner弧长法后屈曲路径跟踪、Koiter初始缺陷敏感性分析以及基于Chirikov共振重叠判据的稳定性检测。

## 二、科学问题与核心公式

### 2.1 控制方程

薄壁圆柱壳在轴向压力作用下的非线性平衡方程为：

```
R(u, λ) = F_int(u) - λ F_ext = 0
```

其中 `u` 为位移场，`λ` 为载荷因子，`F_ext` 为参考外载荷。

### 2.2 Donnell-Mushtari-Vlasov 壳理论

中曲面位移场 `u = [u, v, w]ᵀ`，薄膜应变：

```
ε_x  = ∂u/∂x
ε_θ  = (1/R)(∂v/∂θ + w)
γ_xθ = ∂v/∂x + (1/R)∂u/∂θ
```

弯曲应变（曲率变化）：

```
κ_x  = -∂²w/∂x²
κ_θ  = -(1/R²)(∂²w/∂θ² - ∂v/∂θ)
κ_xθ = -(1/R)(∂²w/∂x∂θ - (3/4)∂v/∂x + (1/4R)∂u/∂θ)
```

本构关系（线弹性各向同性）：

```
N = C_m · ε,   M = C_b · κ
```

薄膜刚度矩阵：

```
C_m = (Et/(1-ν²)) * [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]
```

弯曲刚度矩阵：

```
C_b = (Et³/(12(1-ν²))) * [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]
```

### 2.3 经典线性屈曲载荷

对于简支边界条件的圆柱壳，经典轴向屈曲临界载荷为：

```
N_x,cr = E·t² / (R·√(3(1-ν²)))
```

当 ν = 0.33 时，N_x,cr ≈ 0.605·E·t²/R。

### 2.4 弧长法约束方程

球面弧长约束（Riks-Wempner）：

```
g(u, λ) = (u - u₀)ᵀ(u - u₀) + ψ²(λ - λ₀)² - Δs² = 0
```

其中 ψ 为载荷-位移缩放因子，Δs 为弧长增量。

扩展Newton-Raphson迭代：

```
[ K_T    -F_ext ] [ Δu ]   = [ -R(u, λ) ]
[ 2Δuᵀ   2ψ²Δλ ] [ Δλ ]   = [ -g(Δu, Δλ) ]
```

### 2.5 Koiter 初始后屈曲理论

非对称屈曲模式下，缺陷敏感性公式：

```
λ_s / λ_c = 1 - a·(δ/t)
```

其中 a ≈ 1.5·√(3(1-ν²)) ≈ 2.4（当 ν = 0.33）。

### 2.6 Chirikov 共振重叠判据（启发式）

将后屈曲路径离散映射视为参数驱动非线性振子，相邻环向模态 n 与 n+Δn 的共振重叠条件：

```
ε > ε_crit = (Δn / (2n))²
```

其中 ε = ||w_max|| / t 为非线性强度参数。

### 2.7 Bessel 函数零点与屈曲模态验证

圆柱壳环向谐波对应 Bessel 方程特征条件：

```
J_n(kR) = 0   (固支边界)
J_n'(kR) = 0  (自由边界)
```

临界波数 k_cr 与屈曲载荷关系：

```
N_x,cr = D·k_cr² + E·t/(R²·k_cr²)
```

## 三、原项目到科学问题的映射

| 原项目 | 核心算法 | 合成后角色 |
|--------|---------|-----------|
| 908_predator_prey_ode | ODE守恒量检测与动力学参数 | 伪时间步进动态松弛求解器中的能量守恒检测与演化策略 |
| 081_besselzero | Bessel函数零点Halley迭代 | 圆柱壳线性屈曲环向波数解析验证与特征值交叉检验 |
| 199_collatz_recursive | 递归路径追踪 | 弧长法路径跟踪中的自适应步长递归细分策略 |
| 469_geompack | Delaunay三角化与网格质量α测度 | 壳体中曲面三角网格生成与单元质量评估 |
| 069_ball_monte_carlo | 球体内蒙特卡洛采样 | 几何缺陷Fourier系数空间的蒙特卡洛随机采样 |
| 771_mm_to_msm | Matrix Market格式稀疏矩阵读取 | 稀疏刚度矩阵的Matrix Market格式I/O接口 |
| 314_double_c_data | 双"C"型嵌套数据生成 | 双"C"型反向螺旋模态缺陷场构造 |
| 1157_st_to_hb | ST到Harwell-Boeing格式转换 | 稀疏矩阵格式转换（ST→HB） |
| 107_boundary_word_equilateral | 边界节点识别与排序 | 壳体边界条件描述与周期性边界节点排序 |
| 171_chirikov_iteration | 标准映射迭代与混沌检测 | 后屈曲路径稳定性分析中的KAM环面破坏与Lyapunov指数计算 |
| 236_cube_surface_distance | 表面测地距离统计 | 圆柱壳中曲面测地距离计算与局部屈曲波传播分析 |
| 417_fem3d_pack | 3D有限元基函数与坐标变换 | 壳体有限元离散化中的形函数导数与雅可比矩阵计算 |
| 1239_tet_mesh_tet_neighbors | 四面体网格邻居关系 | 三角网格单元邻居关系构建（用于局部刚度组装） |
| 1286_trapezoidal | 隐式梯形积分与残差构造 | Newton-Raphson迭代中的隐式残差构造与收敛判据设计 |
| 1158_st_to_mm | ST到Matrix Market格式转换 | 稀疏矩阵格式转换（ST→MM）与带宽分析 |

## 四、文件结构

```
086_synth_project/
├── main.py                     # 统一入口，零参数运行
├── shell_geometry.py           # 圆柱壳几何、曲率、测地距离
├── mesh_triangulation.py       # 三角网格生成、Delaunay优化、邻居关系
├── shell_fem_element.py        # DMV壳理论有限元离散化与刚度矩阵组装
├── sparse_matrix_io.py         # 稀疏矩阵格式转换（MM/ST/HB）与带宽缩减
├── linear_buckling.py          # 线性屈曲特征值分析、Bessel零点验证
├── nonlinear_solver.py         # Newton-Raphson迭代、伪时间动态松弛
├── arc_length_tracker.py       # 弧长法后屈曲路径跟踪、分岔检测
├── defect_generator.py         # 几何缺陷生成（单模态/双C/蒙特卡洛）
├── stability_analysis.py       # 稳定性分析、Lyapunov指数、Koiter分类
└── README_博士级合成说明.md     # 本文档
```

## 五、运行方式

```bash
python main.py
```

程序零参数运行，自动完成以下完整流程：
1. 定义圆柱壳几何与铝合金材料参数
2. 生成中曲面三角网格并评估质量
3. 组装线性刚度矩阵与几何刚度矩阵
4. 计算经典线性屈曲载荷与最优屈曲模态
5. 执行线性静力分析验证
6. 使用Newton-Raphson迭代求解非线性平衡
7. 使用弧长法跟踪后屈曲平衡路径（15步）
8. 分析路径稳定性（Lyapunov指数、Koiter分类）
9. 生成多种几何缺陷并评估Koiter敏感性
10. 执行稀疏矩阵格式转换与轮廓缩减
11. 使用伪时间动态松弛求解验证

## 六、关键技术难点与解决方案

### 6.1 数值稳定性
- 壳体极薄（t=1mm，R=250mm），位移量级极小（10⁻⁵ m），导致弧长约束中载荷项与位移项严重失衡。
- **解决方案**：在弧长法缩放因子 ψ 中引入下限保护 `ψ_min = 1/f_norm`，确保扩展Newton系统的分母不会过小。

### 6.2 切线刚度矩阵奇异
- 接近屈曲点时，切线刚度矩阵条件数急剧恶化，稀疏直接求解器可能失败。
- **解决方案**：对刚度矩阵添加 `1e-8` 量级的小量正则化，并采用 try-except  fallback 机制。

### 6.3 伪时间步进发散
- 显式伪动力松弛在薄壳问题中容易因高频模态激发而发散。
- **解决方案**：引入质量比例阻尼、NaN/inf检测与早期退出机制，将时间步长限制在稳定域内。

### 6.4 边界条件处理
- 圆柱壳轴向压缩需要精确施加"底部固支、顶部允许轴向位移"的混合边界条件，防止刚体运动。
- **解决方案**：底部节点固定全部3个自由度，顶部节点固定径向/环向（x,y）自由度，并额外固定底部某一节点的轴向自由度以消除绕轴转动。

## 七、科学意义

本项目代码可用于：
- 航空航天薄壁结构（燃料箱、火箭箭体）的屈曲安全评估
- 海洋工程圆柱壳体（钻井平台支撑、水下航行器）的稳定性分析
- 基于Koiter理论的初始缺陷数据库标定
- 非线性结构力学教学中的弧长法与分岔分析演示

## 八、参考文献

1. Donnell, L.H. (1933). Stability of Thin-Walled Tubes under Torsion. *NACA Report No. 479*.
2. Koiter, W.T. (1945). On the Stability of Elastic Equilibrium. *PhD Thesis, Delft University*.
3. Riks, E. (1979). An incremental approach to the solution of snapping and buckling problems. *International Journal of Solids and Structures*, 15(7), 529-551.
4. Chirikov, B.V. (1979). A universal instability of many-dimensional oscillator systems. *Physics Reports*, 52(5), 263-379.
5. Zienkiewicz, O.C. & Taylor, R.L. (2000). *The Finite Element Method*, 5th Ed., Butterworth-Heinemann.
