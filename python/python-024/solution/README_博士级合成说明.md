# 太阳耀斑磁重联数值模拟平台 —— 博士级合成说明

## 一、项目概述

本项目围绕**等离子体物理：磁重联与太阳耀斑**展开，基于 15 个种子科研项目的核心算法，构建了一个面向前沿科学问题的博士级数值计算平台。磁重联（Magnetic Reconnection）是太阳耀斑、日冕物质抛射等剧烈爆发现象的核心能量释放机制，其研究涉及磁流体动力学（MHD）、非线性偏微分方程、稳定性分析、粒子加速等多个高难度领域。

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 在合成项目中的角色 |
|---|---|---|
| 956_quadrilateral_surface_display | 四边形网格双线性插值 | Harris 电流片平衡态在四边形有限元网格上的物理场插值（`harris_equilibrium.py`） |
| 210_continuation | 伪弧长延拓法（Newton 迭代、切向量、步进控制） | 追踪磁重联平衡态随反常电阻率参数变化的解分支（`continuation_solver.py`） |
| 1356_trig_interp | 三角基函数与三角插值 | 日冕环角向周期性边界的扰动重构（`periodic_interpolation.py`） |
| 1406_wedge_exactness | 楔形区域多项式精确积分 | 电流片尖角（Separator 附近）的磁通量与焦耳加热精确计算（`wedge_flux.py`） |
| 377_fem_neumann | 1D 有限元帽子函数、质量/刚度矩阵、Neumann 边界 | 反常电阻率反应-扩散方程的有限元离散（`fem_assembler.py`） |
| 1156_st_to_ge | 稀疏矩阵 ST 格式到稠密 GE 格式的累加组装 | 全局刚度矩阵和质量矩阵的组装（`fem_assembler.py`） |
| 960_quaternions | 四元数乘法、旋转矩阵转换、轴角表示 | 磁场线在重联过程中的三维旋转与拓扑演化（`field_rotation.py`） |
| 906_pram_view | 几何变换（旋转、反射、平移） | Harris 电流片反射对称性与 X-line 中心对称性分析（`field_rotation.py`） |
| 181_circle_monte_carlo | 圆周上的蒙特卡洛随机采样 | 高能粒子在速度空间中的圆环分布采样与随机加速（`particle_acceleration.py`） |
| 503_hand_mesh2d | 2D 网格拓扑结构生成 | 四边形结构化网格的节点-单元拓扑构造（`harris_equilibrium.py`） |
| 214_contour_sequence4 | 序列数据重排与网格化 reshape | 极坐标场到笛卡尔网格的数据重排（`periodic_interpolation.py`） |
| 1047_row_echelon_integer | 整数矩阵行简化阶梯形（IRREF） | 离散散度-自由约束的精确整数验证（`mhd_stability.py`） |
| 020_artery_pde | 双曲型波动方程（阻尼项、强迫项） | 反常电阻率在 Alfven 波驱动下的阻尼振荡响应（`resistivity_evolution.py`） |
| 505_hankel_inverse | Hankel 矩阵求逆（Fiedler 算法） | MHD 线性化算子在径向坐标变换中的快速求逆（`mhd_stability.py`） |
| 857_pendulum_comparison_ode | 非线性摆 ODE（保守系统动力学） | 粒子在 X-point 磁场梯度中的非线性轨道与混沌分析（`particle_acceleration.py`） |

## 三、新增数学物理模型与核心公式

### 3.1 Harris 电流片平衡态

磁场分布（一维电流片）：
```
B_x(y) = B_0 * tanh(y / lambda)
B_y = 0
B_z = B_g  （引导场）
```

压强分布（由力学平衡 nabla p = J x B 导出）：
```
p(y) = p_0 + (B_0^2 / (2*mu_0)) * sech^2(y / lambda)
```

电流密度（安培定律 J = (1/mu_0) nabla x B）：
```
J_z(y) = (B_0 / (mu_0 * lambda)) * sech^2(y / lambda)
```

等离子体 beta（衡量热压与磁压之比）：
```
beta = 2 * mu_0 * p / |B|^2
```

阿尔芬速度：
```
v_A = |B| / sqrt(mu_0 * rho)
```

### 3.2 伪弧长延拓法

追踪非线性方程组 F(U, eta) = 0 的解分支，引入弧长参数 s：

```
G(U, eta) = | F(U, eta)            | = 0
             | N(U, eta, s)         |

N = (U - U_0)^T * dU/ds + (eta - eta_0) * d eta/ds - Delta s = 0
```

Newton 迭代增广系统：
```
[ F_U   F_eta ] [ delta U   ]   [ -F(U, eta) ]
[ dU_0  d eta_0] [ delta eta] = [ -N        ]
```

### 3.3 MHD 线性稳定性（撕裂模）

线性化不可压缩 MHD 方程：
```
rho_0 * partial v_1/partial t = -nabla p_1 + J_0 x B_1 + J_1 x B_0 + nu nabla^2 v_1
partial B_1/partial t = nabla x (v_1 x B_0) + eta nabla^2 B_1
nabla . v_1 = 0,   nabla . B_1 = 0
```

简化 tearing mode 特征值问题：
```
gamma * L * psi = - (k^2 B_0'^2 / B_0) * psi + eta * L^2 * psi
```

稳定性判据：Re(gamma) > 0 时不稳定（撕裂模增长）。

### 3.4 反常电阻率反应-扩散方程

```
partial eta/partial t = D_eta nabla^2 eta + R(eta, J)

R(eta, J) = alpha * max(0, |J| - J_c) * (eta_max - eta) / eta_max - beta * (eta - eta_cl)
```

当电流密度 |J| 超过临界值 J_c 时，微观不稳定性激发，电阻率向 eta_max 增长。

### 3.5 高能粒子 Lorentz 方程与蒙特卡洛加速

相对论动量方程：
```
dp/dt = q (E + v x B)
p = gamma_r * m * v
gamma_r = 1 / sqrt(1 - v^2/c^2)
```

速度空间圆环采样（对应圆周蒙特卡洛）：
```
theta ~ U[0, 2*pi)
v_perp = v_perp_mag * (cos(theta), sin(theta))
```

### 3.6 四元数磁场旋转

单位四元数表示旋转：
```
q = (w, x, y, z) = cos(theta/2) + sin(theta/2) * u
```

旋转向量：
```
v' = q * v * q^{-1}
```

球面线性插值（SLERP）：
```
q(t) = [sin((1-t)*theta_0) / sin(theta_0)] * q_1 + [sin(t*theta_0) / sin(theta_0)] * q_2
```

### 3.7 三角插值（周期性边界）

等距节点 phi_j = 2*pi * j / N 上的三角基函数：
```
N 为奇数: C_j(phi) = sin(N/2 * (phi-phi_j)) / [N * sin((phi-phi_j)/2)]
N 为偶数: C_j(phi) = sin(N/2 * (phi-phi_j)) / [N * tan((phi-phi_j)/2)]
```

### 3.8 楔形区域精确积分

单位楔形 W = {0<=x, 0<=y, x+y<=1, -1<=z<=1} 上的单项式积分：
```
I(e1, e2, e3) = integral_W x^{e1} y^{e2} z^{e3} dV
```

解析递推公式：
```
k = e1
for i = 1 to e2:
    k = k + 1
    value = value * i / k
k = k + 1; value = value / k
k = k + 1; value = value / k
if e3 为奇数: value = 0
else: value = value * 2 / (e3 + 1)
```

## 四、项目文件结构

```
024_synth_project/
├── main.py                      # 统一入口，零参数运行完整模拟
├── harris_equilibrium.py        # Harris 电流片平衡态与四边形网格插值
├── continuation_solver.py       # 伪弧长延拓法解分支追踪
├── fem_assembler.py             # 有限元稀疏矩阵组装（ST->GE，Neumann 边界）
├── mhd_stability.py             # MHD 撕裂模稳定性、Hankel 逆、整数 RREF
├── resistivity_evolution.py     # 反常电阻率反应-扩散方程与阻尼波动模型
├── particle_acceleration.py     # 蒙特卡洛粒子加速与非线性轨道追踪
├── field_rotation.py            # 四元数磁场旋转与几何对称性变换
├── periodic_interpolation.py    # 三角插值与极坐标-笛卡尔数据重排
├── wedge_flux.py                # 楔形区域精确积分与求积规则验证
└── README_博士级合成说明.md      # 本文档
```

## 五、修改与合成方法

1. **科学问题重构**：将原项目中零散的数学算法（插值、矩阵求逆、ODE 积分、有限元等）统一映射到“太阳耀斑磁重联”这一前沿等离子体物理问题中，每个原项目的算法都在合成项目中承担真实计算角色。
2. **高难公式注入**：在代码中系统引入了 Harris 电流片模型、MHD 线性化方程、撕裂模特征值问题、反常电阻率阈值模型、Lorentz 粒子动力学、四元数旋转、三角插值基函数、楔形精确积分公式等大量博士级物理数学内容。
3. **复杂度升级**：
   - 数值方法涵盖有限元、有限差分（Crank-Nicolson）、RK4、伪弧长延拓、蒙特卡洛、特征值分析、Lyapunov 指数计算。
   - 工程上实现了参数边界检查、除零保护、CFL 条件提示、物理截断（如电阻率限制在 [eta_cl, eta_max]、速度限制在 0.99c）。
4. **删除可视化**：所有原项目中与图形显示、图像输出、动画相关的内容已全部删除，仅保留数值计算与数据分析。
5. **语言迁移**：所有原 MATLAB 项目已完整迁移为 Python 3，利用 NumPy/SciPy 进行科学计算。

## 六、如何运行

```bash
cd 024_synth_project
python main.py
```

无需任何命令行参数。程序将依次执行 9 个模块的数值演示，并在终端输出各关键物理量的计算结果。

## 七、合成后的项目能够解决的科学问题

1. **磁重联平衡态构造**：基于 Harris 模型生成太阳耀斑电流片的解析平衡态，评估等离子体 beta、阿尔芬速度、磁剪切等关键物理量。
2. **重联参数分支追踪**：使用延拓法追踪 MHD 平衡态随电阻率参数的演化，探测 fold 分歧点（对应重联的触发阈值）。
3. **撕裂模稳定性判断**：通过线性化 MHD 算子的特征值分析，判定电流片对 tearing mode 的稳定性，为重联是否自发发生提供理论依据。
4. **反常电阻率演化**：模拟微观不稳定性激发的有效电阻率增长过程，揭示从经典 Spitzer 电阻率到反常电阻率的过渡机制。
5. **高能粒子加速**：蒙特卡洛模拟电子/质子在重联电场中的随机加速过程，估算能量谱和最大能量。
6. **磁场拓扑演化**：使用四元数描述磁场线在重联过程中的旋转与重连，分析对称性约束。
7. **周期性扰动重构**：利用三角插值在任意角向位置重构日冕环中的 MHD 扰动模式。
8. **尖角区域精确积分**：在电流片 Separator 线附近的楔形区域进行精确的磁通量和能量沉积计算，验证数值求积规则的精度。
