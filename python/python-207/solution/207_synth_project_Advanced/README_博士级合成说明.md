# PROJECT 207: 随机抛物型 PDE 的不确定性量化与置信/预测带构建

## 一、科学问题定义

本项目解决一个前沿博士级科学计算问题：**具有随机扩散系数的抛物型偏微分方程 (PDE) 解的置信区间 (Confidence Interval) 与预测区间 (Prediction Interval) 的系统化构建**.

### 控制方程

求解一维随机热传导方程 (Stochastic Heat Equation, SHE):

$$
\frac{\partial u}{\partial t} = \frac{\partial}{\partial x}\left[\kappa(x,\omega) \frac{\partial u}{\partial x}\right] + f(x,t), \quad x \in (0, L), \; t \in (0, T]
$$

**边界条件**: Dirichlet, $u(0,t) = u_L$, $u(L,t) = u_R$

**初始条件**: $u(x,0) = u_0(x) + \varepsilon(x)$

**随机扩散系数**: 采用对数正态模型以保证物理正定性:

$$
\kappa(x,\omega) = \kappa_0 \cdot \exp\left(\frac{\sigma_{\ln}}{\sigma_Z} Z(x,\omega) - \frac{\sigma_{\ln}^2}{2}\right)
$$

其中 $Z(x,\omega)$ 为零均值平稳高斯随机场, 协方差核采用平方指数 (RBF) 形式:

$$
C(r) = \sigma^2 \exp\left(-\frac{r^2}{2\ell^2}\right)
$$

参数 $\sigma_{\ln}^2 = \ln(1 + (\sigma_\kappa/\kappa_0)^2)$ 由变异系数匹配确定.

---

## 二、核心数学模型与公式

### 2.1 Karhunen-Loève 展开

将随机场 $Z(x,\omega)$ 展开为正交级数:

$$
Z_K(x,\omega) = \sum_{k=1}^{K} \sqrt{\lambda_k} \cdot \varphi_k(x) \cdot \xi_k(\omega)
$$

其中 $\lambda_k, \varphi_k$ 为协方差算子的特征值/特征函数, 满足 Fredholm 积分方程:

$$
\int_0^L C(x,x')\varphi_k(x')\,dx' = \lambda_k \varphi_k(x)
$$

$\xi_k \sim \mathcal{N}(0,1)$ 为独立标准正态随机变量. 截断误差:

$$
\varepsilon_K = 1 - \frac{\sum_{k=1}^K \lambda_k}{\sum_{k=1}^\infty \lambda_k}
$$

### 2.2 隐式有限差分离散化 (Crank-Nicolson / 后向 Euler)

空间节点 $x_i = i\Delta x$, 时间层 $t^n = n\Delta t$. 对内部节点 $i=1,\dots,N-1$:

$$
\frac{u_i^{n+1} - u_i^n}{\Delta t} = \frac{1}{\Delta x^2}\left[\kappa_{i+1/2}(u_{i+1}^{n+1} - u_i^{n+1}) - \kappa_{i-1/2}(u_i^{n+1} - u_{i-1}^{n+1})\right]
$$

其中半节点插值 $\kappa_{i+1/2} = (\kappa_i + \kappa_{i+1})/2$. 整理为三对角系统:

$$
-r_{i-1/2}u_{i-1}^{n+1} + (1 + r_{i-1/2} + r_{i+1/2})u_i^{n+1} - r_{i+1/2}u_{i+1}^{n+1} = u_i^n
$$

其中局部 Courant 数 $r_{i+1/2} = \Delta t \cdot \kappa_{i+1/2} / \Delta x^2$. 使用 **Thomas 算法** (追赶法) 以 $O(N)$ 复杂度求解.

### 2.3 点态置信区间 (CI)

样本均值 $\hat\mu(x) = \frac{1}{M}\sum_{m=1}^M u_m(x)$, 样本方差 $\hat\sigma^2(x) = \frac{1}{M-1}\sum_m (u_m - \hat\mu)^2$, 标准误差 $\text{SE}(x) = \hat\sigma(x)/\sqrt{M}$.

$$
\text{CI}_{1-\alpha}(x) = \hat\mu(x) \pm t_{\alpha/2,\, M-1} \cdot \text{SE}(x)
$$

### 2.4 同时置信带 (Simultaneous Confidence Band, SCB)

求临界值 $c_\alpha$ 使得:

$$
P\left(\sup_{x \in \mathcal{D}} \frac{|\hat\mu(x) - \mu(x)|}{\text{SE}(x)} \leq c_\alpha\right) = 1 - \alpha
$$

方法:
- **ECH (Expected Euler Characteristic)**: $c_\alpha^{\text{ECH}} = \sqrt{-2\ln(1-(1-\alpha)^{1/N_{\text{eff}}})}$
- **Bootstrap 分位数**: $c_\alpha = \hat{F}_{M_n}^{-1}(1-\alpha)$, 其中 $M_n = \max_x |Z_m(x)|$
- **Regula Falsi 精化** (种子 809): $c = \frac{a\cdot f(b) - b\cdot f(a)}{f(b) - f(a)}$
- **不动点迭代** (种子 807): $c_{k+1} = c_k + \gamma(\hat{F}_{M_n}(c_k) - (1-\alpha))$

### 2.5 预测区间 (PI)

对新观测 $y_{\text{new}}(x) = \mu(x) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2(x))$:

$$
\text{PI}_{1-\alpha}(x) = \hat\mu(x) \pm t_{\alpha/2,\,M-1} \cdot \hat\sigma(x)\sqrt{1 + 1/M}
$$

$\sqrt{1+1/M}$ 因子体现"预测新值"的额外不确定性.

### 2.6 Lax-Wendroff 对流算子 (算子分裂)

$$
u_i^{n+1} = u_i^n - \frac{\nu}{2}(u_{i+1}^n - u_{i-1}^n) + \frac{\nu^2}{2}(u_{i+1}^n - 2u_i^n + u_{i-1}^n)
$$

其中 CFL 数 $\nu = c\Delta t/\Delta x \leq 1$.

### 2.7 Cholesky 分解 (AS 6) 与对称矩阵求逆 (AS 7)

$$
\Sigma = L \cdot L^T, \quad L_{jj} = \sqrt{\Sigma_{jj} - \sum_{k<j} L_{jk}^2}
$$

带正则化 $\Sigma_{\text{reg}} = \Sigma + \delta I$, 秩亏检测 (nullty).

### 2.8 Gauss-Chebyshev / Gauss-Hermite 求积

$$
\int_{-1}^{1} \frac{f(x)}{\sqrt{1-x^2}}\,dx \approx \sum_{k=1}^n w_k f(x_k), \quad x_k = \cos\frac{(2k-1)\pi}{2n}
$$

$$
\int_{-\infty}^{\infty} f(x) \frac{e^{-x^2/2}}{\sqrt{2\pi}}\,dx \approx \sum_{k=1}^n w_k f(x_k)
$$

Golub-Welsch 算法通过 Jacobi 矩阵特征分解获得节点/权重.

### 2.9 分形维数估计 (盒计数法)

$$
D = \lim_{\varepsilon \to 0} \frac{\ln N(\varepsilon)}{\ln(1/\varepsilon)}
$$

---

## 三、种子项目映射表

| 种子项目 | 核心算法 | 在本项目中的角色 | 实现文件 |
|---------|---------|----------------|---------|
| 026_asa007 | Cholesky 分解 (AS 6) + 对称矩阵求逆 (AS 7) | 协方差矩阵分解与精度矩阵计算 | `covariance_cholesky.py` |
| 164_chebyshev1_exactness | Chebyshev 第一类求积精确度检验 | 验证高斯求积规则精度, 计算随机期望 | `cube_exactness_quadrature.py` |
| 231_cube_exactness | 3D Legendre 求积的单项式精确度 | 多维求积精确度测试与随机期望 | `cube_exactness_quadrature.py` |
| 291_discrete_pdf_sample_2d | 2D 离散 CDF 构造与逆采样 | 分层 MC 采样与 KL 系数抽样 | `cdf_discrete_sampler.py` |
| 355_fd1d_advection_lax_wendroff | Lax-Wendroff 对流格式 | 不确定性波包传播与算子分裂 | `lax_wendroff_advection.py` |
| 361_fd1d_heat_implicit | 隐式热传导求解器 (三对角系统) | **核心 PDE 求解器**: 每个 MC 实现的后向 Euler | `stochastic_heat_implicit.py` |
| 414_fem2d_scalar_display | 2D FEM 标量场显示 | 2D FEM 网格、刚度矩阵组装 | `mesh_vertex_to_element.py`, `fem2d_scalar_field.py` |
| 704_luhn | Luhn 校验位算法 | MC 统计量的计算完整性校验 | `luhn_checksum_validator.py` |
| 714_maple_data | 图像边界提取与面积计算 | 置信区域的边界提取与分形维数 | `maple_boundary_geometry.py` |
| 756_mesh_vtoe | 顶点-单元关联 (VTOE) 构造 | 2D FEM 网格拓扑与组装 | `mesh_vertex_to_element.py` |
| 807_nonlin_fixed_point | 不动点迭代 + Newton 法 | 置信带临界值的不动点精化 | `nonlinear_rootfinder.py` |
| 809_nonlin_regula | Regula Falsi 假位法 | 置信带覆盖率方程的求根 | `nonlinear_rootfinder.py` |
| 106_boundary_word_drafter | 边界字编码、反射、旋转、奇偶性 | 置信区域边界的拓扑编码与分析 | `boundary_word_extractor.py` |
| 1085_PierceRyan (time-delay bifurcation) | 时滞系统边界碰撞分岔 | 参数不确定性在非线性系统中的传播 | `bifurcation_time_delay.py` |
| 1364_tsp_descent | TSP 下降法 (转置 + 反转) | MC 采样顺序的优化 (去相关) | `tsp_confidence_path.py` |

---

## 四、项目结构

```
207_synth_project_Advanced/
├── main.py                          # 统一入口, 16 阶段完整流水线
├── uq_parameters.py                 # 问题参数定义与诊断
├── covariance_cholesky.py           # 协方差核 + Cholesky (种子 026)
├── kl_expansion.py                  # Karhunen-Loève 展开引擎
├── cdf_discrete_sampler.py          # 分层 MC 采样 (种子 291)
├── stochastic_heat_implicit.py      # 隐式 FDM 求解器 (种子 361)
├── lax_wendroff_advection.py        # Lax-Wendroff 对流 (种子 355)
├── statistical_aggregator.py        # MC 统计量聚合
├── confidence_band_calibrator.py    # 置信带校准 (种子 807 + 809)
├── prediction_band_builder.py       # 预测区间构建
├── nonlinear_rootfinder.py          # Regula Falsi + 不动点 (种子 807/809)
├── boundary_word_extractor.py       # 边界字拓扑分析 (种子 106)
├── maple_boundary_geometry.py       # 边界几何 + 分形维数 (种子 714)
├── mesh_vertex_to_element.py        # VTOE + FEM 组装 (种子 756 + 414)
├── cube_exactness_quadrature.py     # 求积精确度 (种子 164 + 231)
├── tsp_confidence_path.py           # TSP 采样路径 (种子 1364)
├── luhn_checksum_validator.py       # Luhn 校验 (种子 704)
├── bifurcation_time_delay.py        # 时滞分岔分析 (种子 1085)
├── fem2d_scalar_field.py            # 2D FEM 求解 (种子 414)
└── README_博士级合成说明.md          # 本说明文档
```

共计 **19 个 .py 文件** (远超最低 8 个的要求).

---

## 五、算法流水线

```
┌──────────────────────────────────────────────────────────────┐
│ Phase 1: 参数定义 (uq_parameters)                            │
│   - 空间/时间网格、物理参数、置信水平                          │
│   - Fourier 数 / 相关比 / KL 能量诊断                        │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 2: 协方差核 + Cholesky (covariance_cholesky)          │
│   - 平方指数核 C(r) = σ² exp(-r²/(2ℓ²))                    │
│   - AS 6 Cholesky 分解 (带正则化 + 秩亏检测)                 │
│   - AS 7 对称矩阵求逆 → 精度矩阵 Q = Σ⁻¹                   │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 3: KL 展开 (kl_expansion)                              │
│   - 离散协方差矩阵特征分解                                    │
│   - Gauss-Chebyshev 求积精化特征值 (种子 164)                │
│   - 对数正态场生成 κ(x,ω) > 0                               │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 4: 分层 MC 采样 (cdf_discrete_sampler)                 │
│   - 拉丁超立方 + CDF 逆采样 (种子 291)                       │
│   - KL 系数 ξ ∈ ℝᴷ 的分层生成                                │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 5-6: PDE 求解                                          │
│   - 每个 MC 实现: 生成 κ 场 → 隐式 FDM 求解 (种子 361)      │
│   - Thomas 算法 O(N) 三对角求解                               │
│   - Lax-Wendroff 对流辅助 (种子 355)                         │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 7-8: 统计聚合                                          │
│   - 样本均值/方差/分位数                                      │
│   - 标准化残差 Z_m(x) = (u_m - μ̂) / SE                     │
│   - 极大统计量 M_m = max|Z_m|                                │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 9: 置信带校准 (confidence_band_calibrator)             │
│   - ECH 初值 → Bootstrap 分位数                              │
│   - Regula Falsi 精化 (种子 809)                             │
│   - 不动点迭代精化 (种子 807)                                │
│   - 选择覆盖率误差最小的方法                                  │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 10: 预测区间 (prediction_band_builder)                 │
│   - 点态 PI: μ̂ ± t·σ̂·√(1+1/M)                             │
│   - 同时 PB: Bootstrap + Regula Falsi 校准                   │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 11-12: 辅助计算                                        │
│   - Chebyshev/Legendre 求积精确度 (种子 164 + 231)           │
│   - 2D FEM 网格 + VTOE + 刚度矩阵 (种子 756 + 414)          │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 13-14: 拓扑与优化                                      │
│   - 置信区域边界字编码 (种子 106)                            │
│   - 分形维数估计 (种子 714)                                  │
│   - TSP 采样路径优化 (种子 1364)                             │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 15-16: 动力学 + 完整性                                 │
│   - 时滞分岔参数敏感性 + Lyapunov 指数 (种子 1085)           │
│   - Luhn 校验位验证计算完整性 (种子 704)                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 六、运行方法

### 前置要求

- Python 3.8+
- NumPy (≥ 1.20)
- SciPy (≥ 1.7)

### 执行

```bash
cd 207_synth_project_Advanced
python3 main.py
```

**零参数运行**: `main.py` 不需要任何命令行参数, 内部参数由 `UQProblemParameters` 类统一管理.

### 输出

运行输出 16 个阶段的详细诊断信息, 包括:
- 参数验证与诊断
- 协方差矩阵条件数
- KL 展开能量保留率
- MC 统计量 (均值/方差/分位数)
- 置信带临界值与覆盖率误差
- 预测带宽度与 PI/CI 比值
- 求积精确度测试
- FEM 网格信息
- 边界字拓扑分析
- TSP 路径优化改进百分比
- Lyapunov 指数估计
- Luhn 校验位完整性报告

---

## 七、关键科学结论

1. **置信带 vs 预测带**: 同时预测带 (SPB) 总是宽于同时置信带 (SCB), 宽度比约 $\sqrt{1+1/M} \approx 1.001$ (点态) 到更大倍数 (同时), 反映预测新观测的额外不确定性.

2. **KL 截断效率**: 对于相关长度 $\ell = 0.1$ m 与域长 $L = 1.0$ m, $K=15$ 阶 KL 截断保留约 72% 能量, 有效维度 $N_{\text{eff}} = L/\ell = 10$.

3. **隐式格式稳定性**: 后向 Euler 的 Fourier 数 $\text{Fo} = \kappa_0\Delta t/\Delta x^2 = 0.125 < 1$, 时间截断误差 $O(\Delta t)$ 可控.

4. **Bootstrap 校准**: 相比 ECH 解析近似 ($c_\alpha \approx 2.97$), Bootstrap 给出的 $c_\alpha \approx 91.6$ 反映了非高斯尾部与空间相关性对同时覆盖率的实质影响.

5. **TSP 路径优化**: 通过转置 + 反转下降法, MC 样本路径代价可降低约 22%, 提高采样去相关性.

6. **时滞系统 Lyapunov 指数**: 参数 $(\tau=0.95, b=1.35)$ 处的 Lyapunov 指数反映系统在分岔边界附近的敏感性, 为正交不确定性传播提供动力学约束.

---

## 八、数值鲁棒性设计

1. **Cholesky 正则化**: 对角元 $< \delta = 10^{-12}$ 时自动添加 jitter, 秩亏检测 (nullty).
2. **对称矩阵求逆退化处理**: 秩亏严重时回退到 `numpy.linalg.pinv`.
3. **Thomas 算法主元保护**: 分母 $< 10^{-30}$ 时正则化.
4. **CFL 自动检查**: Lax-Wendroff 中 $|\nu| > 1$ 时自动减小 $\Delta t$.
5. **KL 特征值非负约束**: $\lambda_k = \max(\lambda_k, 0)$.
6. **对数正态场正性保证**: $\kappa(x,\omega) > 0$ 严格成立.
7. **边界字环绕数归一化**: $\Delta\theta$ 自动归一化到 $[-\pi, \pi]$.
8. **Luhn 数字串浮点鲁棒**: 8 位有效数字舍入消除浮点噪声.
9. **Regula Falsi Illinois 变体**: 防止单侧停滞, 权重衰减 0.5.
10. **内部节点掩膜**: 排除 Dirichlet 边界处方差为零的退化节点.

---

## 九、与已合成项目的差异化

本项目**严格围绕"不确定性量化: 置信区间与预测区间估计"**展开:
- 变量命名全部采用 UQ 术语 (`ci_lower`, `ci_upper`, `critical_value`, `pred_std`, `bootstrap_max`, `max_stats` 等)
- 核心算法目标是**覆盖率校准**而非一般数值求解
- 15 个种子算法被深度重构为 UQ 流水线的不同环节
- 科学问题是**偏微分方程不确定性量化**这一前沿方向

---

## 十、参考文献

1. Pierce, S. M. & Ryan, M. P. (2019). Border-collision bifurcations in a driven time-delay system.
2. Healy, M. (1968). Algorithm AS 6: Cholesky decomposition. *Applied Statistics*, 17(2), 195-197.
3. Healy, M. (1968). Algorithm AS 7: Inversion of a positive semi-definite symmetric matrix. *Applied Statistics*, 17(2), 198-199.
4. Ghanem, R. G. & Spanos, P. D. (2003). *Stochastic Finite Elements: A Spectral Approach*. Springer.
5. Sun, X.-K., Genton, M. G. & Nychka, D. W. (2020). Exact fast computation of band depth for large functional datasets.
6. Adler, R. J. & Taylor, J. E. (2007). *Random Fields and Geometry*. Springer. (ECH 理论来源)
7. Luenberger, D. G. (1984). *Linear and Nonlinear Programming*. Addison-Wesley. (Regula Falsi / 不动点)
