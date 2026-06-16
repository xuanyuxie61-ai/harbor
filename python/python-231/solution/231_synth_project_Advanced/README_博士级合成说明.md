# 计算高能物理：PDF 全局拟合与误差传播 —— 高阶有限差分与稳定性分析

## 项目概述

本项目是一个博士级计算高能物理合成项目，围绕**部分子分布函数 (Parton Distribution Functions, PDF)** 的**全局拟合与误差传播**问题展开，核心数值工具为**高阶有限差分方法**及其**稳定性分析**。

项目通过一个小规模、完全可复现的数值实验，演示从 DGLAP 演化方程求解到 χ² 全局拟合、从 Hessian 误差传播到 Monte Carlo 副本分析的完整 PDF 研究流程。

## 科学问题

在强子对撞机物理 (LHC, RHIC) 中，强子内部夸克与胶子的动量分布由 PDF 描述。PDF 无法从第一性原理计算，必须通过对深度非弹性散射 (DIS)、Drell-Yan 过程、喷注产生等实验数据进行全局拟合得到。

**核心挑战**：
1. **DGLAP 演化**：PDF 在不同能标 Q² 之间通过 DGLAP 积分-微分方程演化，其中分裂函数 P(z) 在 z → 1 处具有可积奇异性。
2. **全局拟合**：需要同时拟合多种实验数据 (DIS、DY、jet)，并满足动量求和规则、价夸克数守恒等物理约束。
3. **误差传播**：PDF 不确定度需通过 Hessian 方法或 Monte Carlo 副本方法从数据不确定度传播到任意物理预言。
4. **数值稳定性**：高阶有限差分格式在 PDF 演化中的稳定性需要精细分析。

## 项目结构

```
231_synth_project_Advanced/
├── main.py                      # 统一入口 (零参数运行)
├── phys_consts.py               # 物理常数、QCD 耦合常数、FD 模板
├── pdf_param.py                 # PDF 参数化、物理约束、截面计算
├── splitting_funcs.py           # LO DGLAP 分裂函数与光滑部分
├── dglap_evolution.py           # DGLAP IMEX 演化求解器
├── experimental_data.py         # 合成实验数据生成 (DIS + DY)
├── pdf_fit.py                   # χ² 全局拟合引擎
├── hessian_error.py             # Hessian 矩阵与误差传播
├── root_finding.py              # Chandrupatla 求根
├── pdf_interp.py                # PWL 插值 (非均匀网格)
├── comb_canal.py                # 退化度组合计数
├── fd_stability.py              # 有限差分稳定性分析
├── mc_replicas.py               # Monte Carlo 副本与 k-means 聚类
├── pipeline.py                  # 流水线编排
└── README_博士级合成说明.md
```

## 运行方式

```bash
python main.py
```

无需任何参数，直接运行即可。流水线将执行 8 个阶段：
1. 初始化参数、网格与合成数据
2. DGLAP 演化
3. χ² 全局拟合
4. Hessian 误差分析
5. Monte Carlo 副本分析
6. 有限差分格式稳定性分析
7. 特征向量退化度分析
8. 结果汇总

## 15 个种子项目的融合映射

本项目融合以下 15 个开源科研项目的核心算法：

| # | 种子项目 | 核心算法 | 在 PDF 项目中的映射 |
|---|---------|---------|---------------------|
| 1 | 757_mesh2d | 自适应三角网格生成 (quadtree 加密) | `dglap_evolution.py` 中 x 网格的自适应加密 |
| 2 | 140_caustic | 圆上模运算 z_j → z_{j·m mod n} | `hessian_error.py` 中 Mellin-N 空间的模块化演化 |
| 3 | 621_kmeans_fast | Elkan 加速 k-means (三角不等式) | `mc_replicas.py` 中 MC 副本的 k-means 聚类 |
| 4 | 976_r8ci | 循环矩阵 (行列式/特征值/二阶差分) | `fd_stability.py` 中有限差分格式的循环矩阵稳定性分析 |
| 5 | 108_boundary_word_hexagon | polyhex 边界词合法性 | `pdf_param.py` 中 PDF 约束词的合法性判定 |
| 6 | 1164_flame-ai-2024 | 火焰传播追踪 (SSIM, Jaccard, centroid) | `dglap_evolution.py` 中 Q² 演化扰动传播追踪 |
| 7 | 409_fem2d_poisson | 2D FEM Poisson 求解 | `dglap_evolution.py` 中 FEM 弱形式残差验证 |
| 8 | 1020_lx913_SoleFlip | 单比特翻转后门注入 | `pdf_fit.py` 中单参数翻转敏感度分析 |
| 9 | 1428_zero_chandrupatla | Chandrupatla 混合二次/二分求根 | `root_finding.py` 中 α_s 反求与特征值求根 |
| 10 | 928_pwl_interp_2d_scattered | 2D 散乱点 PWL 插值 | `pdf_interp.py` 中 (x, Q²) 双线性 PDF 插值 |
| 11 | 1148_agrimUT_SLAI | LLM 服务流水线 (Timer/Metrics) | `pipeline.py` 中 8 阶段流水线编排 |
| 12 | 1159_KadelkaLab_nondegenerate-canalization | 布尔函数管化深度枚举 | `comb_canal.py` 中 Hessian 特征向量退化度计数 |
| 13 | 1377_usa_box_plot | 矩阵热图 box_fill | `experimental_data.py` 中 Hessian 相关矩阵文本热图 |
| 14 | 1184_Secure-Data-Reconstruction | H∞ 控制稀疏数据恢复 | `pdf_fit.py` 中稀疏异常检测与鲁棒 χ² |
| 15 | 226_craps_simulation | 蒙特卡洛概率模拟 | `experimental_data.py` 中 MC 概率估计 |

## 核心物理公式

### LO DGLAP 演化方程

$$\frac{\partial q(x,t)}{\partial t} = \frac{\alpha_s(t)}{2\pi} \left[P_{qq} \otimes q + P_{qg} \otimes g\right]$$

$$\frac{\partial g(x,t)}{\partial t} = \frac{\alpha_s(t)}{2\pi} \left[P_{gq} \otimes \Sigma + P_{gg} \otimes g\right]$$

其中 $t = \ln(Q^2/\Lambda^2)$, $\Sigma = \sum_q (q + \bar{q})$ 为单态夸克组合。

### LO 分裂函数

$$P_{qq}(z) = C_F \left[\frac{1+z^2}{(1-z)}\right]_+ , \quad P_{qg}(z) = T_R[z^2 + (1-z)^2]$$

$$P_{gq}(z) = C_F \frac{1+(1-z)^2}{z}, \quad P_{gg}(z) = 2C_A\left[\frac{z}{1-z} + \frac{1-z}{z} + z(1-z)\right]_+$$

### LO 跑动耦合

$$\alpha_s(Q^2) = \frac{4\pi}{\beta_0 \ln(Q^2/\Lambda^2)}, \quad \beta_0 = \frac{33-2n_f}{3}$$

### 四阶中心差分

$$f'(x) \approx \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h} + O(h^4)$$

$$f''(x) \approx \frac{-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)}{12h^2} + O(h^4)$$

### IMEX 时间推进

$$f^{n+1}(v_k) = \frac{f^n(v_k) + \Delta t \cdot R_{smooth}(v_k)}{1 + \Delta t \cdot (\alpha_s/2\pi) \cdot A_{ii}}$$

### χ² 全局拟合

$$\chi^2(p) = \sum_{i=1}^{N_{data}} \left(\frac{d_i - t_i(p)}{\sigma_i}\right)^2 + \lambda_M\left(\int x\Sigma\,dx - 1\right)^2 + \lambda_{V_u}\left(\int u_v\,dx - 2\right)^2 + \lambda_{V_d}\left(\int d_v\,dx - 1\right)^2$$

### Hessian PDF 不确定度

$$(\Delta f(x))^2 = \sum_{k=1}^{N_{eigen}} \left[f_k^+(x) - f_0(x)\right]^2$$

其中 $f_k^\pm(x) = f(x; p_0 \pm \sqrt{\Delta\chi^2} \cdot v_k)$, $v_k$ 为 Hessian 特征向量。

### Chandrupatla 求根

混合逆二次插值与二分法：当二次插值参数 $\Phi$ 小于阈值 $\Phi_t = \frac{1}{2}\sqrt{|f_1/f_2|}$ 时使用插值，否则使用二分。

### von Neumann 稳定性

放大因子：$|g(\theta)| = |1 + \Delta t \lambda_{exp}(\theta)| / |1 - \Delta t \lambda_{imp}(\theta)| \leq 1$

### CFL 条件

$$\Delta t_{max} = C_{safe} \cdot \frac{h^2}{(\alpha_s/2\pi) \max|P|}$$

### Mellin 模块化演化 (映射自 caustic)

$$E(N) = \exp\left[\frac{\alpha_s}{2\pi} P(N) \ln(Q^2/Q_0^2)\right] \cdot e^{2\pi i (Nm \mod N_{max}) / N_{max}}$$

### k-means Elkan 加速

若 $d(x, c_{assigned}) \leq \frac{1}{2} \min_{j \neq assigned} d(c_j, c_k)$，则跳过距离重算。

### 管化深度 (映射自 Kadelka Lab)

$$K = |\{k : \lambda_{(k)} > \text{threshold} \times \lambda_{max}\}|$$

## 关键模块说明

### `phys_consts.py`
定义标准模型常数、QCD 圈系数、Λ_QCD、LO/NLO 跑动耦合、四阶/六阶有限差分模板、数值鲁棒性参数。

### `pdf_param.py`
PDF 在初始尺度 Q₀² 的参数化：
- $xu_v(x) = A_{uv} x^{a_{uv}} (1-x)^{b_{uv}} (1 + c_{uv}\sqrt{x} + d_{uv}x)$
- $xd_v(x) = A_{dv} x^{a_{dv}} (1-x)^{b_{dv}} (1 + c_{dv}x)$
- $xg(x) = A_g x^{a_g} (1-x)^{b_g} (1 + c_g\sqrt{x})$
- $xS(x) = A_s x^{a_s} (1-x)^{b_s}$

包含动量求和规则归一化、价夸克数守恒、约束词合法性判定、LO DIS 与 DY 截面计算、单参数翻转敏感度分析 (映射自 SoleFlip)。

### `splitting_funcs.py`
LO 分裂函数及其光滑部分提取，用于 IMEX 时间分裂。包含 P_qq, P_qg, P_gq, P_gg 的原始形式与光滑部分，以及离散卷积实现。

### `dglap_evolution.py`
核心 IMEX 求解器，包含：
- 自适应 x 网格生成 (log/lin/adapt, 映射自 mesh2d)
- 二阶 IMEX-RK2 时间推进
- FEM 弱形式残差验证 (映射自 fem2d_poisson)
- Q² 演化扰动传播追踪 (映射自 flame-ai)

### `pdf_fit.py`
梯度下降 + Armijo 线搜索的全局拟合引擎，含：
- 有限差分梯度
- 参数约束投影 (映射自 boundary_word_hexagon)
- 单参数翻转敏感度 (映射自 SoleFlip)
- 稀疏异常检测 (映射自 Secure-Data-Reconstruction)
- 鲁棒 Huber χ²

### `hessian_error.py`
Hessian 矩阵有限差分计算、特征值分解、Mellin 模块化演化、焦散包络密度 (映射自 caustic)、Hessian 相关矩阵热图 (映射自 box_fill)。

### `root_finding.py`
Chandrupatla 混合二次/二分求根 (映射自 zero_chandrupatla)，用于：
- α_s(Q²) 反求
- DGLAP 特征值求根
- χ² 抛物线最小值定位

### `pdf_interp.py`
分段线性插值与 Delaunay 区间搜索 (映射自 pwl_interp_2d_scattered)，支持 1D 与 2D (x, Q²) 插值，含误差界估计。

### `comb_canal.py`
退化度组合计数 (映射自 nondegenerate-canalization)：
- 组合数 C(n,k)
- Stirling 第二类数 S(n,r)
- 管化深度、退化子空间维度
- 非退化函数计数

### `fd_stability.py`
循环矩阵有限差分稳定性分析 (映射自 r8ci)：
- 循环矩阵构造与特征值
- von Neumann 放大因子
- CFL 条件检查
- FD2/FD4/FD6 精度-稳定性权衡

### `mc_replicas.py`
Monte Carlo 副本与 k-means 聚类 (映射自 kmeans_fast 与 craps_simulation)：
- MC 副本生成与拟合
- PDF 不确定度估计
- Elkan 加速 k-means
- 蒙特卡洛概率估计

### `pipeline.py`
流水线编排 (映射自 SLAI)：
- StageTimer 阶段计时
- MetricsStore 指标收集
- 8 阶段完整流水线

## 输出示例

流水线运行后输出：
- 拟合参数及其不确定度
- χ²_min 与 χ²/dof
- Hessian 特征值谱
- MC 副本聚类结果
- 有限差分稳定性报告
- 退化度分析报告
- 各阶段计时报告

## 可扩展方向

1. **NLO 分裂函数**：扩展至 NLO P^{(1)}_{ij}(z)，需要处理更高阶奇异性
2. **真实实验数据**：替换合成数据为 HERA、LHC 真实测量
3. **高阶 IMEX**：三阶/四阶 IMEX-Runge-Kutta
4. **Lagrange 乘子法**：严格实施约束优化
5. **神经网络 PDF**：用神经网络替代参数化

## 复现性

所有随机种子固定，数值结果完全可复现。默认配置下运行时间约 5-10 秒。
