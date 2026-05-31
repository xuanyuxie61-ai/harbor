# README_博士级合成说明.md

## 项目概述

**项目名称**：多尺度聚合反应动力学与分子量分布耦合模拟系统  
**科学领域**：化学工程 —— 聚合反应动力学与分子量分布 (Polymerization Kinetics & Molecular Weight Distribution)  
**合成目录**：`/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/133_synth_project`

本项目基于 15 个种子科研代码项目的核心算法，融合重构为一个面向**前沿博士级**化学工程计算问题的 Python 科研代码系统。项目围绕**自由基乳液聚合反应器中链增长动力学与分子量分布预测**这一核心科学问题，整合了ODE动力学、偏微分方程数值求解、蒙特卡洛采样、不确定性量化、非线性稀疏矩阵求解、最优空间离散化等多个高难计算模块。

---

## 一、种子项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 合成后模块 | 科学角色 |
|------|----------|----------|------------|----------|
| 1 | `090_biochemical_linear_ode` | 生化线性ODE、守恒量、参数管理、精确解 | `polymerization_kinetics.py` | 自由基聚合动力学矩方程的ODE系统构建与参数管理 |
| 2 | `1037_rk45` | Runge-Kutta 4/5阶嵌入式方法 | `polymerization_kinetics.py` | 聚合动力学ODE的时间积分与自适应步长控制 |
| 3 | `273_dg1d_heat` | 一维Discontinuous Galerkin热方程 | `reaction_diffusion_dg.py` | 反应器轴向反应-扩散-对流方程的DG空间离散 |
| 4 | `359_fd1d_display` | 一维有限差分数据展示 | `reaction_diffusion_dg.py` | DG结果的后处理与空间网格映射思想 |
| 5 | `1360_truncated_normal` | 截断正态分布采样/PDF/CDF | `molecular_weight_distribution.py` | 链长分布的截断对数正态建模与随机采样 |
| 6 | `886_polygon_integrals` | 多边形域矩计算 | `molecular_weight_distribution.py` | 反应器截面几何矩用于局部浓度梯度估计 |
| 7 | `331_ellipse_monte_carlo` | 椭圆内均匀采样(Cholesky) | `monte_carlo_chain_sampler.py` | 聚合物链构象空间的受限椭球采样 |
| 8 | `298_disk_triangle_picking` | 圆盘内随机三角形面积 | `monte_carlo_chain_sampler.py` | 搅拌混合效率的蒙特卡洛统计估计 |
| 9 | `1105_sparse_grid_hermite` | Smolyak稀疏网格Gauss-Hermite求积 | `uncertainty_quantification.py` | 动力学参数不确定性的高效随机配点传播 |
| 10 | `1384_vandermonde_interp_1d` | 一维Vandermonde多项式插值 | `vandermonde_reconstruction.py` | 离散MWD数据点的连续多项式重构 |
| 11 | `871_plasma_matrix` | 等离子体问题稀疏Jacobian组装 | `nonlinear_solver.py` | 凝胶效应非线性反应-扩散的稀疏矩阵组装 |
| 12 | `034_asa082` | 正交矩阵行列式(detq) | `nonlinear_solver.py` | Newton迭代中Jacobian QR正交性数值检验 |
| 13 | `247_cvt_2d_lumping` | 二维Lloyd CVT算法 | `cvt_reactor_discretization.py` | 反应器空间的最优量化节点配置 |
| 14 | `930_pyramid_exactness` | 金字塔求积规则精确性检验 | `quadrature_validation.py` | Gauss-Hermite求积规则的精度验证框架 |
| 15 | `776_monomial_symmetrize` | 单项式系数对称化 | `quadrature_validation.py` | 高维矩计算中对称等价类的归并处理 |

**每一个输入项目均已真实融入合成项目，无遗漏、无挂名。**

---

## 二、新增数学物理模型与核心公式

### 2.1 自由基聚合动力学矩方程

活性链矩（准稳态近似）：

$$
\frac{d\lambda_0}{dt} = 2f k_d I - k_t \lambda_0^2 \approx 0
$$

$$
\frac{d\lambda_1}{dt} = k_i M R + k_p M \lambda_0 - k_t \lambda_0 \lambda_1 - k_{tr} S \lambda_1 + k_{tr} S \lambda_0 \approx 0
$$

$$
\frac{d\lambda_2}{dt} = k_i M R + k_p M (2\lambda_1 + \lambda_0) - k_t \lambda_0 \lambda_2 - k_{tr} S \lambda_2 + k_{tr} S \lambda_0 \approx 0
$$

死链矩演化：

$$
\frac{d\mu_0}{dt} = \left(\frac{k_{tc}}{2} + k_{td}\right) \lambda_0^2
$$

$$
\frac{d\mu_1}{dt} = k_t \lambda_0 \lambda_1, \quad
\frac{d\mu_2}{dt} = k_t \lambda_0 \lambda_2 + k_{tc} \lambda_1^2
$$

工程指标：

$$
\bar{X}_n = \frac{\lambda_1 + \mu_1}{\lambda_0 + \mu_0}, \quad
\bar{X}_w = \frac{\lambda_2 + \mu_2}{\lambda_1 + \mu_1}, \quad
PDI = \frac{\bar{X}_w}{\bar{X}_n}
$$

### 2.2 Flory-Schulz 分布

$$P(n) = (1-p) p^{n-1}, \quad p = \frac{k_p [M]}{k_p [M] + k_t [P^\bullet] + k_{tr}[S]}$$

### 2.3 截断对数正态分布

$$f(M) = \frac{1}{M \sigma \sqrt{2\pi}} \frac{\exp\left(-\frac{(\ln M - \mu)^2}{2\sigma^2}\right)}{\Phi\left(\frac{\ln b - \mu}{\sigma}\right) - \Phi\left(\frac{\ln a - \mu}{\sigma}\right)}$$

### 2.4 DG 弱形式

单元 $D^k$ 上的弱形式：

$$\int_{D^k} \frac{\partial u_h}{\partial t} \phi_j \, dx - \int_{D^k} v u_h \frac{\partial \phi_j}{\partial x} \, dx + \int_{D^k} D \frac{\partial u_h}{\partial x} \frac{\partial \phi_j}{\partial x} \, dx + [\hat{F} \phi_j]_{\partial D^k} = \int_{D^k} R(u_h) \phi_j \, dx$$

数值通量（LDG）：

$$\hat{u} = \{u\} + \frac{1}{2} n_x (u^- - u^+), \quad
\hat{q} = \{q\} - \frac{1}{2} n_x (q^- - q^+)$$

### 2.5 凝胶效应扩散模型 (Trommsdorff Effect)

$$D(c) = D_0 (1-c)^\beta \quad (c < c_{crit})$$

$$D(c) = D_0 (1-c_{crit})^\beta \exp[-10(c-c_{crit})] \quad (c \geq c_{crit})$$

### 2.6 Smolyak 稀疏网格

$$A(q,d) = \sum_{q-d+1 \leq |\mathbf{i}| \leq q} (-1)^{q-|\mathbf{i}|} \binom{d-1}{q-|\mathbf{i}|} \left(Q^{i_1} \otimes \cdots \otimes Q^{i_d}\right)$$

### 2.7 CVT Lloyd 算法

$$g_i^{(new)} = \frac{\int_{V_i} \mathbf{x} \rho(\mathbf{x}) \, d\mathbf{x}}{\int_{V_i} \rho(\mathbf{x}) \, d\mathbf{x}}$$

其中 $V_i$ 为生成元 $g_i$ 的 Voronoi 单元。

### 2.8 多边形矩 (Steger 公式)

$$\nu_{pq} = \frac{1}{(p+q+2)(p+q+1)\binom{p+q}{p}} \sum_{i=1}^{n} (x_{i-1}y_i - x_i y_{i-1}) \sum_{k=0}^{p}\sum_{l=0}^{q} \binom{k+l}{l}\binom{p+q-k-l}{q-l} x_i^k x_{i-1}^{p-k} y_i^l y_{i-1}^{q-l}$$

---

## 三、项目文件结构

```
133_synth_project/
├── main.py                           # 统一入口，零参数运行
├── polymerization_kinetics.py        # 聚合动力学ODE + RK45
├── reaction_diffusion_dg.py          # 一维DG反应-扩散求解器
├── molecular_weight_distribution.py  # MWD矩方法 + 截断分布
├── monte_carlo_chain_sampler.py      # 链构象MC采样 + 混合效率
├── uncertainty_quantification.py     # 稀疏网格UQ + Sobol敏感度
├── vandermonde_reconstruction.py     # Vandermonde插值重构MWD
├── nonlinear_solver.py               # Newton-Krylov + 凝胶效应
├── cvt_reactor_discretization.py     # CVT Lloyd最优离散化
├── quadrature_validation.py          # 求积规则精确性验证
└── README_博士级合成说明.md          # 本说明文档
```

共 **10 个 .py 文件** + **1 个 .md 文档**。

---

## 四、如何运行

```bash
cd "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/133_synth_project"
python main.py
```

程序无需任何命令行参数，运行后将依次执行 9 大模块的完整计算流程，并输出关键科学指标（转化率、聚合度、PDI、混合效率、不确定性统计量、收敛阶等）。

**依赖**：`numpy`, `scipy`（标准科学计算库）

---

## 五、合成方法说明

### 5.1 代码语言转换
所有种子项目原为 MATLAB/Octave 代码，已完全转换为 Python 3，采用 NumPy/SciPy 实现向量化运算与稀疏矩阵操作。

### 5.2 可视化删除
原种子项目中所有 `plot`, `figure`, `surf`, `print('-dpng', ...)` 等可视化代码已彻底删除，仅保留数值计算与结果输出。

### 5.3 边界处理与数值鲁棒性
- ODE 求解：浓度非负截断 (`np.maximum(y, 1.0e-15)`)，自适应步长上下限保护
- DG 求解：CFL 条件自动计算，非负性保护，RHS 截断 (`np.clip`)
- Newton 求解：线搜索阻尼、残差范数监控、Jacobian 正交性检验
- 截断正态：sigma 下限保护 (`1.0e-12`)，CDF 反函数概率截断
- Vandermonde：条件数过大时自动降阶或切换最小二乘

### 5.4 工程复杂度提升
- 引入 Arrhenius 温度修正、凝胶效应非线性扩散、自催化反应源项
- 实现高阶矩方法（0-4 阶）、流场展宽修正、多尺度耦合
- 集成稀疏网格随机配点、Sobol 敏感度分析、QR 正交性检验

---

## 六、科学问题解决能力

本合成项目能够解决以下前沿科学计算问题：

1. **聚合反应器设计优化**：预测不同操作条件下（温度、引发剂浓度、停留时间）的分子量分布演化
2. **凝胶效应预警**：通过二维非线性反应-扩散求解器模拟局部扩散控制终止导致的反应失控
3. **质量控制**：基于不确定性量化评估动力学参数测定误差对 PDI 预测的影响，确定关键敏感参数
4. **反应器放大**：利用 CVT 最优离散化方法为工业规模反应器提供空间节点配置方案
5. **混合效率评估**：通过蒙特卡洛随机三角形面积估计搅拌反应器的微观混合效率

---

*合成完成日期：2026-05-06*
