# README_博士级合成说明.md

## 项目概述

本项目将 **15 个原始科研代码项目** 融合重构为一个面向 **光学工程：超连续谱产生与非线性** 前沿领域的博士级 Python 科研计算系统。

核心科学问题：
> **基于广义非线性薛定谔方程（GNLSE）的光子晶体光纤超连续谱产生全波形数值仿真与自适应优化**

---

## 一、原项目到科学问题的映射

| 原项目 | 核心算法 | 合成后角色 | 所在文件 |
|---|---|---|---|
| 184_circle_segment | 圆段几何解析（面积、质心、高度） | PCF 空气孔圆截面几何参数计算、填充率与模场面积估算 | `pcf_geometry.py` |
| 1305_triangle_grid | 三角形区域规则网格生成 | 光纤横截面三角形网格离散化，用于横向模式采样 | `pcf_geometry.py` |
| 1292_tri_surface_display | 3D 三角网格数据读写与处理 | 三角网格节点/元素文件 I/O，PCF 表面网格管理 | `mesh_utils.py` |
| 1168_stla_to_tri_surface_fast | ASCII STL 快速解析与转换 | 光纤端面几何模型的 STL 导入与三角表面重建 | `mesh_utils.py` |
| 1239_tet_mesh_tet_neighbors | 四面体网格邻接关系计算 | 3D 四面体拓扑邻接表，用于光子晶体光纤三维结构分析 | `mesh_utils.py` |
| 159_chebyshev | Chebyshev 多项式插值系数 | 色散曲线 beta(omega) 的全局光滑 Chebyshev 谱展开 | `dispersion_calculus.py` |
| 1388_vanloan (CubicSpline) | 三次样条插值 | 色散关系局部精细结构的高阶样条逼近 | `dispersion_calculus.py` |
| 658_lebesgue | Lebesgue 常数估计 | 插值稳定性监控，对比 Chebyshev 与等距节点的 Runge 抑制能力 | `dispersion_calculus.py` |
| 238_cvt | Centroidal Voronoi Tessellation 迭代优化 | 自适应时间/频率采样点的 Lloyd 算法优化 | `adaptive_grid.py` |
| 585_image_sample | 图像采样与边界坐标提取 | 光谱功率密度的边界检测与有效带宽提取 | `adaptive_grid.py` |
| 1135_spiral_pde_movie | 反应扩散 PDE 时间步进 | GNLSE 分步傅里叶法（SSFM）的频域-时域交替步进框架 | `gnlse_propagator.py` |
| 016_arclength | 参数曲线弧长计算 | 脉冲包络弧长参数化，用于自适应步长控制与数值稳定性 | `gnlse_propagator.py` |
| 1152_squircle_ode | 非线性 ODE 右端构造 | Raman 响应辅助 ODE 系统，将卷积积分降维为局部微分方程 | `nonlinear_response.py` |
| 448_fresnel | Fresnel 积分 C(x), S(x) | 光纤输出端近场衍射的 Fresnel-Kirchhoff 积分计算 | `fresnel_output.py` |
| 345_exm/orbits | N 体引力轨道动力学 | 多模光纤中模式耦合的 Hamilton 轨道动力学类比 | `multimode_coupling.py` |

---

## 二、核心科学公式与物理模型

### 2.1 广义非线性薛定谔方程（GNLSE）

光子晶体光纤中脉冲传播由 GNLSE 描述：

```
dA/dz = -alpha/2 * A + D_hat(A) + i*gamma * N(A)
```

其中：
- **色散算子**（频域实现）：
  ```
  D_hat(A) <-> D(omega) = -alpha/2 + sum_{m=2}^{M} i^{m+1} * beta_m / m! * omega^m
  ```
- **非线性算子**：
  ```
  N(A) = (1 + i/omega0 * d/dT) * [ A(z,T) * integral_{-inf}^{+inf} R(T') |A(z,T-T')|^2 dT' ]
  ```

### 2.2 Raman 响应函数

Blow-Wood 单振子模型：

```
h_R(T) = A * exp(-T/tau2) * sin(T/tau1),   T >= 0
A = (tau1^2 + tau2^2) / (tau1 * tau2^2)
```

完整非线性响应：
```
R(T) = (1 - f_R) * delta(T) + f_R * h_R(T)
```

其中 `f_R ~ 0.18`，`tau1 = 12.2 fs`，`tau2 = 32.0 fs`。

### 2.3 分步傅里叶法（SSFM）

对称 SSFM 将线性色散步与非线性步分离：

```
A(z+dz/2, omega) = A(z, omega) * exp( D(omega) * dz/2 )          [半线性步]
A(z+dz, T)       = exp( i*gamma*N*dz ) * A(z+dz/2, T)            [非线性步，RK4]
A(z+dz, omega)   = A(z+dz, omega) * exp( D(omega) * dz/2 )       [半线性步]
```

### 2.4 Fresnel 衍射积分

一维 Fresnel-Kirchhoff 衍射：

```
E(x,z) = exp(i*k*z) / sqrt(i*lambda*z) * integral E0(x') * exp(i*k*(x-x')^2/(2*z)) dx'
```

Fresnel 数：`N_F = a^2 / (lambda * z)`，当 `N_F >> 1` 时为近场区。

### 2.5 孤子参数

孤子阶数：
```
N = sqrt( gamma * P0 * T0^2 / |beta2| )
```

色散长度：`L_D = T0^2 / |beta2|`  
非线性长度：`L_NL = 1 / (gamma * P0)`

### 2.6 Chebyshev 插值与 Lebesgue 稳定性

Chebyshev 节点：
```
x_k = (a+b)/2 + (b-a)/2 * cos( (2k-1)*pi / (2n) )
```

Lebesgue 函数：
```
L(x) = sum_{j=1}^{n} |l_j(x)|
```

Chebyshev 节点的 Lebesgue 常数以 `O(log n)` 增长，远优于等距节点的指数增长。

### 2.7 CVT 自适应采样

能量泛函：
```
E(z_1,...,z_n) = sum_i integral_{V_i} rho(x) ||x - z_i||^2 dx
```

Lloyd 迭代：交替执行 Voronoi 剖分 + 质心更新。

---

## 三、项目文件结构

```
109_synth_project/
├── main.py                     # 统一入口（零参数运行）
├── pcf_geometry.py             # PCF 几何 + 圆段/三角网格
├── mesh_utils.py               # 网格拓扑 + STL + 四面体邻接
├── dispersion_calculus.py      # Chebyshev + 样条 + Lebesgue + Sellmeier
├── adaptive_grid.py            # CVT 优化 + 光谱边界检测
├── nonlinear_response.py       # Raman 响应 + 辅助 ODE
├── gnlse_propagator.py         # SSFM 核心求解器 + 弧长自适应步长
├── fresnel_output.py           # Fresnel 积分 + 衍射场计算
├── multimode_coupling.py       # 多模耦合轨道动力学
├── spectrum_analysis.py        # 光谱特征提取（带宽/平坦度/孤子阶数）
└── README_博士级合成说明.md     # 本文档
```

---

## 四、运行方式

```bash
cd "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/109_synth_project"
python main.py
```

程序将依次执行 8 个科学计算任务，输出超连续谱仿真结果与关键物理参数。

---

## 五、关键设计决策

1. **零参数运行**：所有物理参数（光纤几何、脉冲参数、传播距离）在 `main.py` 中硬编码为典型实验值，无需用户输入。
2. **删除可视化**：所有绘图代码已移除，仅保留数值计算与文本输出。
3. **边界处理**：每个模块包含严格的输入校验（半径>0、波长>0、数组维度匹配等），异常时抛出 `ValueError` 或 `RuntimeError`。
4. **数值鲁棒性**：
   - Fresnel 积分采用分段算法（级数/递推/渐近），覆盖全定义域。
   - SSFM 使用对称步进 + RK4 非线性子步，保证二阶精度。
   - 自适应步长基于弧长变化率，防止脉冲陡峭化时的数值发散。
5. **工程复杂度**：项目融合了 PDE 求解、谱方法、几何计算、ODE 系统、优化算法、衍射物理、多体动力学等多个博士级数值方法。

---

## 六、科学问题说明

合成后的项目解决的核心科学问题是：

> **如何在光子晶体光纤中，通过精确控制色散、非线性与脉冲参数，实现高相干、宽带宽的超连续谱产生？**

具体包括：
- PCF 几何结构对非线性系数与色散特性的调控；
- 高阶色散与延迟 Raman 响应对光谱展宽的动力学影响；
- 自适应采样技术对计算效率与精度的优化；
- 输出端衍射效应对光谱测量的修正；
- 多模耦合对能量分配与相干度的影响。

---

## 七、依赖说明

本项目仅依赖标准库 `numpy` 和 `scipy`。运行前请确保已安装：

```bash
pip install numpy scipy
```
