# PROJECT 228: 计算高能物理 — 量能器 Shower Profile 快速模拟

## 博士级合成说明文档

---

## 一、科学问题描述

本项目聚焦于**计算高能物理**中的核心数值问题：**电磁量能器中电磁簇射 (electromagnetic shower) 纵向与横向能量沉积剖面的快速精确模拟**。

### 物理背景

当高能电子或光子进入致密材料（如 CMS 实验使用的 PbWO₄ 晶体），会通过以下级联过程产生电磁簇射：

1. **Bremsstrahlung**: 高能电子在核库仑场中辐射光子 (e → e + γ)
2. **对产生**: 高能光子在核场中转化为正负电子对 (γ → e⁺ + e⁻)
3. **电离损失**: 低能电子通过电离原子损失能量

这一过程的数学描述由 **Rossi-Greisen 级联方程**给出：

$$\frac{\partial \phi_e(t, k)}{\partial t} = -(\sigma_b + \sigma_{ion}) \phi_e(t, k) + 2\sigma_p \int_k^\infty \frac{\phi_\gamma(t, k')}{k'} dk'$$

$$\frac{\partial \phi_\gamma(t, k)}{\partial t} = -\sigma_p \phi_\gamma(t, k) + \sigma_b \int_k^\infty \frac{\phi_e(t, k')}{k'} dk'$$

### 数值挑战

- **多尺度问题**: 辐射长度 X₀ ~ 0.89 cm (PbWO₄)，但簇射纵向发展达 25-30 X₀
- **刚性方程**: 高能截面差异大（σ_b ~ 1/X₀ vs σ_ion ~ Ec/(k·X₀)）
- **积分-微分方程**: 右端项包含从 k 到 ∞ 的积分
- **统计涨落**: 蒙特卡罗模拟需要大量事件才能获得平滑剖面

---

## 二、15 个种子项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 在合成项目中的角色 |
|------|----------|----------|-------------------|
| 1 | 757_mesh2d | 2D Delaunay 网格生成 + Laplacian 光滑 | **量能器横向网格生成器**：构建极坐标三角网格描述 shower 横向扩展 |
| 2 | 1124_rspence821505_Variational | 4D-Var 数据同化 + L-BFGS 优化 | **剖面融合引擎**：将解析模型与 MC 结果通过变分方法最优融合 |
| 3 | 1410_wedge_monte_carlo | 蒙特卡罗采样 + 指数分布自由程 | **电磁簇射 MC 模拟**：采样作用长度、能量分配、散射角 |
| 4 | 680_line_grid | 1D 非均匀网格 (5种中心方式) | **纵向自适应网格**：几何拉伸 + shower max 附近加密 |
| 5 | 318_dragon_chaos | 迭代函数系统 (IFS) 混沌吸引子 | **混沌特性分析**：Lyapunov 指数、分形维数、庞加莱回归 |
| 6 | 678_line_fekete_rule | Fekete 点 (Vandermonde 最大化) | **最优横向采样**：Bessel 函数零点分布的 Fekete 采样点 |
| 7 | 086_biharmonic_cheby1d | Chebyshev 谱微分矩阵 | **谱方法求解器**：配置法求解级联方程的刚性部分 |
| 8 | 470_gl_fast_rule | 无迭代 Gauss-Legendre 节点计算 | **高精度求积**：Newton 迭代 + Chebyshev 初值的 GL 求积器 |
| 9 | 687_linpack_bench | LU 分解 + 性能基准 | **性能评估**：矩阵求解效率、精度验证、条件数估计 |
| 10 | 1319_triangle_symq_to_ref | 三角形对称求积 (Xiao-Gimbutas) | **2D 能量沉积积分**：在非结构网格上计算沉积能量 |
| 11 | 437_flame_ode | 火焰 ODE + RK4 + Lambert W | **级联方程时间推进**：RK4 + 自适应步长控制 |
| 12 | 1102_ryan597_RepresentationLearningWaves | 波表示 + U-Net 特征提取 | **修正波数分析**：有限差分格式的色散/耗散特性 |
| 13 | 923_pwc_plot_1d | 分段常数/线性函数表示 | **通量分段重建**：PCHIP 插值 + TVD 斜率限制器 |
| 14 | 1372_unicycle | 循环排列 (Nijenhuis-Wilf) | **相位空间采样**：排列组合方法用于角向均匀采样 |
| 15 | 176_circle_arc_grid | 曲线弧等距网格 | **探测器曲面网格**：沿量能器曲面的弧长参数化 |

---

## 三、核心数学物理模型

### 3.1 Rossi-Greisen 级联方程 (近似 B)

电子-光子通量 φ_e(t,k) 和 φ_γ(t,k) 满足：

$$\frac{\partial \phi_e}{\partial t} = -(\sigma_b(k) + \sigma_{ion}(k)) \phi_e + 2\sigma_p \int_k^\infty \frac{\phi_\gamma(k')}{k'} dk'$$

**截面公式**：

- Bremsstrahlung: $\sigma_b(k) = \frac{1}{X_0}\left(1 - \frac{E_c}{3k}\right)$ for $k \gg E_c$
- 对产生: $\sigma_p(k) = \frac{7}{9X_0}\left(1 - \frac{E_c}{3k}\right)$ for $k > 2m_e c^2$
- 电离: $\sigma_{ion}(k) = \frac{E_c}{X_0 \cdot k}$ (低能极限)

### 3.2 纵向剖面参数化 (Gamma 分布)

$$\Gamma(t) = E_0 \cdot \frac{b \cdot (bt)^{a-1} \cdot e^{-bt}}{\Gamma(a)}$$

其中 $a = \ln(E_0/E_c)/\ln(2)$, $b = 0.5$

**极大值位置 (Rossi 公式)**:
$$t_{max} = \frac{\ln(E_0/E_c)}{\ln 2} - C$$
电子入射 C ≈ 1.0, 光子入射 C ≈ 0.5

### 3.3 横向剖面 (Molière 参数化)

$$f(r) = \frac{1}{R_M^2}\left[\frac{C_1}{s_1}e^{-r/(s_1 R_M)} + \frac{C_2}{s_2}e^{-r/(s_2 R_M)}\right]$$

$C_1 = 0.2219, s_1 = 0.8394, C_2 = 0.7781, s_2 = 2.3286$

**Molière 半径**: $R_M = X_0 \times 21.2\,\text{MeV} / E_c$

### 3.4 NKG (Nishimura-Kamata-Greisen) 函数

$$f(r, s) = \frac{C(s)}{R_M^2}\left(\frac{r}{R_M}\right)^{s-2}\left(1 + \frac{r}{R_M}\right)^{s-4.5}$$

年龄参数: $s(t) = \frac{3t}{t + 2t_{max}}$

### 3.5 高阶有限差分格式

**4 阶中心差分 (一阶导数)**:
$$D^{(4)} f_j = \frac{-f_{j+2} + 8f_{j+1} - 8f_{j-1} + f_{j-2}}{12h}$$

**修正波数**:
$$\hat{k}h = \frac{8\sin(kh) - \sin(2kh)}{6}$$

**紧致 Pade 格式 (六阶)**:
$$\frac{1}{4}f'_{j-1} + f'_j + \frac{1}{4}f'_{j+1} = \frac{3}{2}\frac{f_{j+1} - f_{j-1}}{2h}$$

### 3.6 von Neumann 稳定性分析

对于半离散格式 $\dot{u} = Lu$，放大因子 $g(z)$ 需满足 $|g| \leq 1$：

- **显式 Euler**: $g = 1 + z$, 稳定域: $\text{Re}(z) \leq 0, |1+z| \leq 1$
- **RK4**: $g = 1 + z + z^2/2 + z^3/6 + z^4/24$, 稳定域扩大 ~2.8 倍

**CFL 条件** (扩散): $\Delta t \leq \frac{h^2}{2D}$ (Euler), $\Delta t \leq \frac{2.785 h^2}{2D}$ (RK4)

### 3.7 Chebyshev 谱微分矩阵

节点: $x_j = \cos(\pi j / N), \quad j = 0, 1, \ldots, N$

$$D_{ij} = \frac{c_i}{c_j}\frac{(-1)^{i+j}}{x_i - x_j} \quad (i \neq j)$$

$$D_{ii} = -\frac{x_i}{2(1-x_i^2)} \quad (0 < i < N), \quad D_{00} = \frac{2N^2+1}{6}$$

### 3.8 4D-Var 代价函数

$$J(\mathbf{x}_0) = \frac{1}{2}(\mathbf{x}_0 - \mathbf{x}_b)^T B^{-1}(\mathbf{x}_0 - \mathbf{x}_b) + \frac{1}{2}\sum_k (\mathcal{H}_k \mathcal{M}_{0\to k}(\mathbf{x}_0) - \mathbf{y}_k)^T R_k^{-1}(\cdots)$$

**L-BFGS 两步递推**:
$$q \leftarrow \nabla J; \quad \alpha_i = \rho_i s_i^T q; \quad q \leftarrow q - \alpha_i y_i$$
$$r = \gamma q; \quad r \leftarrow r + s_i(\alpha_i - \beta_i)$$

### 3.9 Lyapunov 指数与混沌特性

级联矩阵特征值:
$$\lambda_{1,2} = \frac{\text{tr}(M)}{2} \pm \sqrt{\frac{\text{tr}(M)^2}{4} - \det(M)}$$

KS 熵 (Pesin 公式): $h_{KS} = \sum_{\lambda_i > 0} \lambda_i$

### 3.10 Highland 散射公式

$$\theta_0 = \frac{13.6\,\text{MeV}}{\beta p c} z \sqrt{\frac{x}{X_0}}\left[1 + 0.038\ln\left(\frac{x}{X_0}\right)\right]$$

---

## 四、项目文件结构

```
228_synth_project_Advanced/
├── main.py                      # 统一入口 (零参数运行)
├── material_properties.py       # 材料物理属性库 (PDG 2024)
├── calorimeter_grid.py          # 量能器网格生成 (纵向+横向+2D)
├── finite_diff_schemes.py       # 高阶有限差分 + 稳定性分析
├── cascade_equation_solver.py   # Rossi-Greisen 级联方程求解
├── transverse_profile.py        # 横向剖面 (Molière/NKG/Fekete)
├── monte_carlo_shower.py        # 蒙特卡罗簇射模拟
├── quadrature.py                # 高精度求积 (GL/Chebyshev/三角形)
├── chebyshev_spectral.py        # Chebyshev 谱方法求解器
├── variational_assimilation.py  # 4D-Var 变分数据同化
├── chaos_analysis.py            # 混沌特性分析
├── piecewise_flux.py            # 分段通量重建 (PCHIP/TVD)
├── linpack_benchmark.py         # LINPACK 性能基准
└── README_博士级合成说明.md      # 本文档
```

---

## 五、运行方法

### 零参数运行

```bash
cd 228_synth_project_Advanced
python main.py
```

无需任何输入参数。程序将自动执行以下完整流程：

1. 初始化材料参数 (PbWO₄)
2. 生成多维计算网格
3. 构造高阶有限差分格式并分析稳定性
4. 求解 Rossi-Greisen 级联方程
5. 计算横向剖面
6. 运行蒙特卡罗模拟
7. 高精度求积计算能量沉积
8. Chebyshev 谱方法验证
9. 4D-Var 数据同化融合
10. 混沌特性分析
11. 分段通量重建
12. LINPACK 性能基准
13. 综合分析与验证

---

## 六、合成后的科学问题

本项目解决的核心科学问题：

> **如何使用高阶数值方法在保持物理精度的前提下，实现电磁量能器中簇射剖面的快速模拟，并通过数据同化将确定性模型与随机模拟最优融合？**

### 创新点

1. **多方法交叉验证**: 有限差分 + 谱方法 + MC 三条独立路径互相验证
2. **自适应网格**: 纵向网格在 shower max 附近自动加密
3. **稳定性保障**: von Neumann 分析确保每个时间步的数值稳定性
4. **数据同化融合**: 4D-Var 将解析先验与 MC 观测最优结合
5. **混沌诊断**: Lyapunov 指数和分形维数量化簇射过程的确定性/随机性边界

---

## 七、预期输出

程序运行后将输出：

- 材料物理属性 (X₀, Ec, R_M, λ_I)
- 网格质量统计
- 有限差分格式精度与稳定性信息
- 级联方程解 (shower max, 包含深度, 能量守恒)
- 横向剖面参数 (Molière/NKG 函数值, 横向矩)
- MC 模拟结果 (沉积能量, 泄漏, shower max)
- 求积精度 (GL, Chebyshev, 三角形, 自适应 Simpson)
- 谱方法解
- 同化收敛信息
- 混沌特性指标
- 分段通量重建结果
- LINPACK 性能数据
- 综合验证结果

---

## 八、物理参数参考值

### PbWO₄ (CMS ECAL)

| 参数 | 值 | 单位 |
|------|-----|------|
| X₀ | 0.89 | cm |
| Ec | 7.97 | MeV |
| R_M | 2.37 | cm |
| λ_I | 20.6 | cm |
| ρ | 8.28 | g/cm³ |

### 10 GeV 电子在 PbWO₄ 中的典型值

| 观测量 | 值 | 单位 |
|--------|-----|------|
| t_max | ~7.6 | X₀ |
| t_max (物理) | ~6.8 | cm |
| 95% 包含深度 | ~20 | X₀ |
| 最大粒子数 | ~1000 | - |

---

## 九、依赖

仅使用 Python 标准库：
- `math` — 数学函数
- `dataclasses` — 数据结构
- `typing` — 类型注解
- `time` — 性能计时
- `random` — 随机数生成

无需安装第三方包。

---

## 十、版本信息

- 项目编号: PROJECT_228
- 合成日期: 2026-06-07
- 语言: Python 3
- 科学领域: 计算高能物理
- 难度等级: 博士级
