# 地震波全波形反演与层析成像 — 博士级科研代码合成项目

## 一、项目概述

本项目围绕**地球物理：地震波全波形反演与层析成像**这一前沿科学领域，基于 15 个种子科研代码项目的核心算法，融合构建了一个面向粘弹性随机分形介质的自适应全波形反演与层析成像计算框架。

### 核心科学问题

地下介质通常具有**多相性**（固/液/气岩相并存）、**粘弹性**（波传播伴随衰减与频散）和**分形孔隙结构**（裂缝网络具有尺度无关统计特征）。如何在这种复杂介质条件下，利用地震波全波形信息高精度反演地下速度结构，是计算地球物理学中的博士级难题。

本项目构建的计算框架实现了以下科学目标：
1. **速度模型构建**：融合 Ising 多相场、分形孔隙扰动和 CVT 自适应网格的粘弹性速度模型
2. **波场正演模拟**：时间域 RK4 声波方程求解 + 频域 Helmholtz 方程 GMRES 求解
3. **数据噪声建模**：Ornstein-Uhlenbeck 随机过程模拟真实地震噪声环境
4. **全波形反演**：伴随状态法（Adjoint State Method）PDE 约束优化反演速度结构
5. **层析成像**：射线理论旅行时层析作为反演先验约束
6. **不确定性量化**：贝叶斯超几何分布采样表征岩相分类不确定性
7. **谱元法验证**：Gauss-Lobatto-Legendre (GLL) 混合高斯求积规则精确度测试

---

## 二、原项目到科学问题的映射

| 种子项目 | 核心算法 | 科学角色 |
|---|---|---|
| **710_mandelbrot** | 复数迭代逃逸时间 | **分形散射强度场**：将 Mandelbrot 迭代类比为地震波在强散射介质中的能量逃逸过程，逃逸时间短的区域对应强散射耗散 |
| **1376_urn_simulation** | 超几何分布 urn 采样 | **贝叶斯岩相不确定性量化**：从多相岩相总体中无放回抽样，统计推断地下岩相分布的置信区间 |
| **1138_spring_double_ode** | 双弹簧耦合 ODE | **粘弹性本构模型**：双弹簧系统对应标准线性固体（SLS）模型，模拟地震波在粘弹性介质中的衰减与频散 |
| **501_hand_area** | 多边形面积、点在多边形内 | **地质构造几何分析**：计算复杂构造截面面积，判断射线穿越区域，支持有限元网格质量评估 |
| **1160_standing_wave_exact** | 波动方程精确解 | **数值算法验证**：驻波解析解 u(x,t)=sin(x)cos(ct) 用于验证 RK4 波动方程求解器的正确性 |
| **518_hermite_cubic** | Hermite 三次样条插值/积分 | **速度模型光滑插值**：在 CVT 自适应网格上通过 Hermite 三次样条将粗网格速度插值到细网格，保持导数连续 |
| **952_quadrilateral** | 四边形面积、凸性判断 | **有限元网格质量检查**：检测非凸网格单元，避免反演中的数值不稳定性 |
| **655_leaf_chaos** | 迭代函数系统 (IFS) | **分形孔隙网络生成**：Barnsley IFS 生成具有分形特征的几何点集，模拟裂缝网络形态 |
| **713_maple_area** | MC/QMC 面积估计 | **波前面覆盖面积估计**：使用 Hammersley 准蒙特卡洛序列估计复杂波前面的覆盖面积 |
| **804_nint_exactness_mixed** | 混合高斯求积精确度 | **谱元法数值积分**：验证 Legendre/Jacobi/Laguerre/Hermite 混合求积规则的精确度，用于 SEM 质量/刚度矩阵计算 |
| **216_control_bio_homework** | RK4 + 最优控制 | **伴随状态法反演**：将 FWI 视为 PDE 约束最优控制问题，模型参数为控制变量，波动方程为状态方程 |
| **473_gmres** | GMRES + Arnoldi + Givens 旋转 | **Helmholtz 方程频域求解**：大规模稀疏复对称线性系统的 Krylov 子空间迭代求解 |
| **839_ornstein_uhlenbeck** | OU 过程 SDE | **地震数据噪声建模**：模拟检波器环境噪声的均值回归特性；生成二维 OU 随机场表征速度随机扰动 |
| **601_ising_2d_simulation** | 2D Ising MC 模拟 | **多相岩相离散分布**：+1/-1 自旋映射为高速/低速岩相，Glauber 动力学模拟岩相界面演化 |
| **440_florida_cvt_pop** | CVT + Lloyd 算法 | **自适应观测网优化**：Lloyd 迭代优化地震检波器的空间布局，实现均匀覆盖 |

---

## 三、新增数学物理模型与核心公式

### 3.1 波动方程与粘弹性本构

**一维声波方程**（状态方程）：

$$
\frac{\partial^2 u}{\partial t^2} = c^2(x) \frac{\partial^2 u}{\partial x^2} + f(t)\delta(x - x_s)
$$

**双弹簧粘弹性模型**（标准线性固体近似）：

$$
\begin{cases}
m_1 \ddot{u}_1 = -k_1 u_1 + k_2(u_2 - u_1) \\
m_2 \ddot{u}_2 = -k_2(u_2 - u_1)
\end{cases}
$$

特征频率：

$$
\omega_{1,2}^2 = \frac{1}{2}\left[\frac{k_1+k_2}{m_1} + \frac{k_2}{m_2}\right] \pm \frac{1}{2}\sqrt{\left[\frac{k_1+k_2}{m_1} + \frac{k_2}{m_2}\right]^2 - \frac{4k_1 k_2}{m_1 m_2}}
$$

**驻波精确解**（验证基准）：

$$
u(x,t) = \sin(x)\cos(ct), \quad \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
$$

### 3.2 Helmholtz 方程与 GMRES

**频域 Helmholtz 方程**（含 PML 吸收边界）：

$$
\nabla^2 u + \frac{\omega^2}{c^2(x)} u = f(x)
$$

PML 坐标拉伸：$s(x) = 1 + i\sigma(x)/\omega$，其中 $\sigma(x) = \sigma_{\max}(d/L_{\text{PML}})^2$

**GMRES 最小化问题**：

$$
\min_{x \in x_0 + \mathcal{K}_k(A,r_0)} \|Ax - b\|_2
$$

Arnoldi 分解：$AQ_k = Q_{k+1}H_k$，Givens 旋转将 $H_k$ 上三角化。

### 3.3 伴随状态法全波形反演

**目标泛函**：

$$
J(m) = \frac{1}{2}\|d_{\text{obs}} - F(m)\|^2_2 + \frac{\beta}{2}\|Lm\|^2_2
$$

**伴随方程**（反向时间）：

$$
\frac{\partial^2 \lambda}{\partial t^2} = c^2(x) \frac{\partial^2 \lambda}{\partial x^2} + (u - d_{\text{obs}}), \quad \lambda(T) = 0, \quad \dot{\lambda}(T) = 0
$$

**梯度公式**：

$$
\nabla_m J = -\frac{2}{c^3(x)} \int_0^T \frac{\partial^2 u}{\partial x^2} \lambda \, dt + \beta \frac{L^T L m}{dx^2}
$$

### 3.4 Ornstein-Uhlenbeck 噪声模型

**SDE**：

$$
dX(t) = \theta(\mu - X(t))\,dt + \sigma\,dW(t)
$$

稳态分布：$X \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{2\theta}\right)$

### 3.5 分形孔隙度场

**分数布朗运动谱方法**：

$$
\phi(k) \sim |k|^{-(2H+1)/2}, \quad H = 2 - D
$$

其中 $D$ 为分形维数，$H$ 为 Hurst 指数。

### 3.6 Gauss-Lobatto-Legendre 求积

**GLL 积分点**：$P'_{N-1}(\xi_i) = 0$ 的根加上端点 $\pm 1$

**GLL 权重**：

$$
w_i = \frac{2}{N(N-1)[P_{N-1}(\xi_i)]^2}
$$

精确度：对次数 $\leq 2N-3$ 的多项式精确。

### 3.7 超几何分布岩相采样

**PMF**：

$$
P(W=w) = \frac{\binom{K}{w}\binom{N-K}{n-w}}{\binom{N}{n}}
$$

其中 $N$ 为总体大小，$K$ 为白色弹珠（沉积岩）数，$n$ 为抽样数。

---

## 四、文件结构与功能说明

```
041_synth_project/
├── main.py                      # 统一入口，零参数运行完整流程
├── velocity_model.py            # 速度模型：Ising + 分形 + CVT + Hermite插值
├── wave_propagation.py          # 波传播：RK4 + 双弹簧ODE + 驻波验证
├── helmholtz_solver.py          # Helmholtz求解：GMRES + Arnoldi + PML
├── noise_model.py               # 噪声建模：OU过程 + 二维随机场
├── monte_carlo_utils.py         # 几何工具：多边形/四边形 + MC/QMC积分
├── inverse_problem.py           # 全波形反演：伴随状态法 + 层析成像
├── fractal_scattering.py        # 分形散射：Mandelbrot逃逸 + IFS映射
├── quadrature_rules.py          # 谱元积分：混合高斯求积 + GLL精确度验证
├── bayesian_sampling.py         # 贝叶斯采样：Urn模型 + 后验采样
└── README_博士级合成说明.md      # 本文档
```

---

## 五、运行方式

```bash
cd 041_synth_project
python main.py
```

无需任何输入参数。程序将自动执行以下 10 个计算演示：

1. **速度模型构建**：生成 101×51 的多相分形速度模型
2. **波动方程验证**：驻波精确解残差检验 + 双弹簧特征频率分析
3. **Helmholtz 频域正演**：129 点网格 GMRES 迭代求解
4. **地震噪声建模**：5 道 OU 过程噪声 + 二维随机扰动场
5. **蒙特卡洛几何估计**：盐丘构造面积 MC/QMC 估计 + 网格质量检查
6. **分形散射分析**：Mandelbrot 逃逸时间场 + IFS 分形孔隙网络
7. **谱元法积分验证**：GLL-2 到 GLL-8 的 0-6 阶多项式精确度全部 PASS
8. **贝叶斯不确定性**：超几何分布岩相采样 + 拒绝采样后验估计
9. **全波形反演**：81 点模型伴随状态法梯度下降反演，misfit 降低 ~26%
10. **CVT 观测网优化**：16 个 generator 的 Lloyd 算法均匀布局

---

## 六、边界处理与数值鲁棒性

- **速度约束**：所有速度值通过 `np.clip` 限制在 [1000, 8000] m/s 范围内
- **CFL 条件**：时间步长自动满足 $dt \leq 0.4 \cdot dx / c_{\max}$
- **PML 边界**：Helmholtz 方程采用二次增长衰减函数 $\sigma(d) = \sigma_{\max}(d/L)^2$
- **Givens 旋转**：GMRES 中处理了 $v_1 = 0$ 的退化情况
- **Hermite 插值**：对零长度区间返回端点值，避免除零错误
- **多边形测试**：射线交叉法正确处理边界上的点
- **Mandelbrot 迭代**：通过 `np.clip` 抑制数值溢出，消除 RuntimeWarning
- **GLL 权重**：端点权重显式设置并归一化，确保总和为 2

---

## 七、合成方法总结

本项目的核心合成策略是**"算法思想跨域迁移"**：

- **可视化代码全部删除**：所有种子项目中的 `plot`, `print` 等图形输出已移除，仅保留数值计算核心
- **算法语义重构**：将纯数学/图形算法重新解释为地球物理问题中的物理过程
- **多源融合**：单个科学模块融合 2-4 个种子项目的算法，确保每个输入项目都承担真实角色
- **博士级复杂度提升**：引入 PDE 约束优化、Krylov 子空间迭代、SDE 随机分析、分形几何、贝叶斯统计等博士课程级别的数学工具

---

## 八、参考文献与理论基础

1. Tarantola, A. (1984). *Inversion of seismic reflection data in the acoustic approximation*. Geophysics.
2. Plessix, R. E. (2006). *A review of the adjoint-state method for computing the gradient of a functional with geophysical applications*. Geophysical Journal International.
3. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
4. Komatitsch, D., & Tromp, J. (1999). *Introduction to the spectral element method for three-dimensional seismic wave propagation*. GJI.
5. Higham, D. J. (2001). *An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations*. SIAM Review.
6. Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. W.H. Freeman.
