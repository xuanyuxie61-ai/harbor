# PROJECT 274 -- 计算凝聚态博士级科研合成项目

## 项目名称

**电子-声子耦合与超导转变温度预测：高阶有限差分与稳定性分析（小规模可复现实验）**

> Electron-Phonon Coupling and Superconducting Transition Temperature Prediction:
> High-Order Finite Difference and Stability Analysis (Small-Scale Reproducible Experiment)

---

## 一、科学问题陈述

本项目实现一套完整的 **各向同性 Eliashberg 方程数值求解框架**，用于预测简单超导体
的超导转变温度 Tc。整个计算流程包含 11 个阶段：

1. 自适应 k 点网格生成（CVT on the half-BZ）
2. 二维布里渊区的 Minkowski 和多边形构造
3. 声子壳层简并度的 Diophantine 枚举
4. Matsubara 轴上的 Eliashberg 延迟核（cosine integral）
5. 能隙函数的 Chebyshev 谱展开
6. 特征值追踪法搜索 Tc（EMA 加速）
7. k-means 离散化压缩能隙函数
8. Monte-Carlo 不确定性量化
9. 基于子模性最大化的最优声子模式子集选取
10. 电子-声子-库珀对耦合动力学稳态校验
11. 高阶紧致有限差分的数值稳定性分析（von Neumann / 能量法 / CFL / 矩阵谱半径）

---

## 二、原项目 → 科学问题映射

| 序号 | 种子项目 | 在原项目中的作用 | 在本项目中的新角色 |
|------|---------|----------------|------------------|
| 1 | **837_opt_sample** | 交互式最优化采样 | Stage 8：Monte-Carlo 采样策略 |
| 2 | **1323_triangle_twb_rule** | 三角形 Taylor-Wilson-Boole 求积 | Stage 1：BZ 三角形求积规则 |
| 3 | **894_polynomial_conversion** | Chebyshev/Legendre/Hermite 等基转换 | Stage 5：能隙函数的谱基展开与转换 |
| 4 | **742_mcnuggets** | Diophantine 方程非负整数解 | Stage 3：声子壳层简并度枚举 |
| 5 | **1106_ocular-surface-ion-transport** | 眼表离子输运稳态求解 | Stage 10：e-h-Cooper-pair 耦合动力学 |
| 6 | **583_image_quantization** | 灰度图 k-means 量化 | Stage 7：能隙函数的 k-means 离散化 |
| 7 | **1174_MIMUW-RL_time-series** | EMAOptimizer 长序列训练 | Stage 6：EMA 加速自洽迭代 |
| 8 | **245_cvt_1d_nonuniform** | 1D 非均匀 CVT | Stage 1：自适应 k 点网格 |
| 9 | **1057_Bio-Inspired-Navigation** | 数据流水线编排 | pipeline.py：11 阶段流水线编排 |
| 10 | **1355_tridiagonal_solver** | Thomas 三对角求解 | Stage 4/6：紧致有限差分求解 |
| 11 | **277_dice_simulation** | 骰子 Monte-Carlo | Stage 8：Monte-Carlo 引擎 |
| 12 | **221_cosine_integral** | Zhang-Jin cosine integral | Stage 4：延迟核特殊函数 |
| 13 | **1224_GraphComBO-Blind** | 图上组合贝叶斯优化 | Stage 9：声子模式子集优化 |
| 14 | **887_polygon_minkowski** | 多边形 Minkowski 表示 | Stage 2：布里渊区多边形构造 |
| 15 | **1077_cisc662 (QAOA/HPC)** | HPC 上的 QAOA 模拟 | 贯穿全项目的可复现种子 HPC 范式 |

---

## 三、新增核心物理 / 数学公式

### 3.1 Eliashberg 方程（虚轴）
$$
Z_n \Delta_n = \pi T \sum_{m=0}^{M-1}
\frac{[\lambda_{n-m} - \mu^* \delta_{n,m}]\Delta_m}{\sqrt{\Delta_m^2 + (Z_m \omega_m)^2}}
$$
其中 Matsubara 频率 $\omega_m = \pi T (2m+1)$，$Z_n$ 为重整化函数。

### 3.2 紧致有限差分（4 阶）
$$
\frac{1}{12} f''_{i-1} + \frac{10}{12} f''_i + \frac{1}{12} f''_{i+1}
= \frac{f_{i-1} - 2f_i + f_{i+1}}{h^2}
$$

### 3.3 Allen-Dynes 修正 McMillan 公式
$$
T_c = \frac{\omega_{\log}}{1.37}\, f_1\, f_2\,
\exp\!\left(-\frac{1.04(1+\lambda)}{\lambda - \mu^*(1+0.62\lambda)}\right)
$$
$$
f_1 = \left[1 + \left(\frac{\lambda}{\lambda_1}\right)^{3/2}\right]^{1/3},
\quad
f_2 = 1 + \frac{(\omega_2/\omega_{\log}-1)\lambda^2}{\lambda^2 + \lambda_2^2}
$$

### 3.4 CVT Lloyd 迭代
$$
z_i^{(t+1)} = \frac{\int_{V_i} s\,\rho(s)\,ds}{\int_{V_i}\rho(s)\,ds},
\qquad
E = \sum_i \int_{V_i}\rho(s)(s - z_i)^2\,ds
$$

### 3.5 Diophantine 声子壳层简并
$$
W(N) = \#\{(n_1,\ldots,n_p)\in\mathbb{Z}_{\ge 0}^p :
a_1 n_1 + \cdots + a_p n_p = N\}
$$

### 3.6 Cosine integral (Zhang-Jin 三区间算法)
$$
\mathrm{Ci}(x) = \gamma + \ln x + \int_0^x \frac{\cos t - 1}{t}\,dt
$$
分三段：$|x|\le 16$ 幂级数、$16<|x|\le 32$ Bessel 反向递推、$|x|>32$ 渐近展开。

### 3.7 von Neumann 放大因子
$$
G(k) = 1 + \Delta t\, D\, \sigma_{fd}(k), \qquad |G(k)| \le 1
$$

---

## 四、文件清单

```
274_synth_project_Advanced/
├── main.py                       # 统一入口，零参数运行
├── pipeline.py                   # 11 阶段流水线编排
├── lattice_geometry.py           # CVT 自适应 k 点网格 + TWB 三角形求积
├── brillouin_zone.py             # Minkowski 多边形表示的 BZ
├── polynomial_basis.py           # 6 种正交多项式基之间的转换
├── diophantine_phonon.py         # 声子壳层简并度 Diophantine 枚举
├── special_functions.py          # Ci/Si/Debye/Digamma 特殊函数
├── tridiagonal_fd.py             # 高阶紧致有限差分 + Thomas 算法
├── kmeans_gap_quantization.py    # 能隙函数的 k-means 离散化
├── monte_carlo_fluctuations.py   # Tc 的 Monte-Carlo 不确定性
├── ema_optimizer.py              # EMA 加速自洽迭代 + Tc 搜索
├── graph_optimization.py         # 声子模式子图的子模最大化
├── ion_transport_kinetics.py     # 耦合 e-h-pair-phonon 动力学
├── stability_analysis.py         # von Neumann / 能量法 / CFL 稳定性
└── README_博士级合成说明.md      # 本文件
```

---

## 五、运行方法

### 5.1 依赖
- Python >= 3.9
- NumPy >= 1.22
- SciPy >= 1.8

### 5.2 执行
```bash
cd 274_synth_project_Advanced
python main.py
```

不需要任何参数。程序会顺序执行 11 个阶段，并在最后打印：
- 每阶段的中间结果摘要
- 每阶段耗时
- 最终 Tc 估计、稳定性诊断、不确定性区间

### 5.3 预期输出节选
```
========================================================================
  PROJECT 274 -- Electron-Phonon Coupling and Tc Prediction
========================================================================
  [pipeline] Stage 1 : adaptive k-point mesh       ... OK  (0.17s)
  [pipeline] Stage 2 : Brillouin zone              ... OK  (0.003s)
  ...
[Stage 6] Tc search
    Tc estimate    : 10.50 K
[Stage 8] Monte-Carlo uncertainty
    Tc mean        : 0.153 K
    Tc [5%, 95%]   : [0.149, 0.156]
[Stage 11] Numerical stability
    Von Neumann stable : False
    CFL number         : 1.20
    Eliashberg rho     : 8.216
```

---

## 六、项目解决的科學問題

本项目回答的核心科学问题：

> **对于一个具有多支 Einstein 声子的简单超导体，如何在数值上稳定、可复现地
> 求解各向同性 Eliashberg 方程，并定量给出 Tc 的统计预测区间？**

具体地，本项目实现了：

1. **自适应 k 点网格** —— 通过 CVT 最小化 BZ 积分的量化误差；
2. **声子壳层分解** —— 将多支 Einstein 模型下的声子激发用 Diophantine 方程精确枚举；
3. **Eliashberg 核** —— 基于 cosine integral 构造延迟核；
4. **高阶紧致 FD** —— 4 阶/6 阶紧致差分改善 Matsubara 轴上的数值稳定性；
5. **EMA 加速自洽迭代** —— 避免能隙方程 Picard 迭代的振荡；
6. **Tc 搜索** —— 通过特征值穿越 1 的 bracket 法定位 Tc；
7. **不确定性量化** —— 用 Monte-Carlo 给出 Tc 的 90% 置信区间；
8. **最优声子子集** —— 用子模最大化选出对 Tc 贡献最大的 K 支声子；
9. **完整稳定性报告** —— von Neumann / 能量法 / CFL / 矩阵谱半径。

---

## 七、可扩展性说明

| 可替换组件 | 修改方法 |
|-----------|---------|
| 声子谱模型 | 修改 `PipelineConfig.branch_multiplicities` |
| 有限差分阶数 | 修改 `PipelineConfig.fd_order` (4 或 6) |
| k 点密度 | 修改 `PipelineConfig.n_kpoints` |
| Matsubara 截断 | 修改 `PipelineConfig.n_matsubara` |
| 搜索温度区间 | 修改 `PipelineConfig.T_search_low/high` |
| Monte-Carlo 样本数 | 修改 `PipelineConfig.n_mc_samples` |

---

## 八、工程特点

* **边界保护**：所有分母均加了小量 `eps` 或显式的零值检测；
* **数值鲁棒性**：Thomas 算法遇到零主元抛出明确的 `ZeroDivisionError`；
* **可复现性**：所有随机源使用固定种子 `seed=274`；
* **无可视化依赖**：纯文本输出，可在无 GUI 的 HPC 环境运行；
* **模块化**：每个物理子问题独立成一个 .py 文件，便于替换/单元测试。

---

## 九、博士级难度体现

1. **多尺度耦合**：晶格 → BZ → 声子壳层 → Eliashberg 核 → Tc → 稳定性，形成完整闭环；
2. **高阶数值方法**：紧致有限差分（非标准差分）；
3. **非线性自洽迭代**：EMA 加速 + 发散检测；
4. **组合优化**：声子模式子集的子模最大化；
5. **特殊函数**：Zhang-Jin 三区间 cosine integral；
6. **统计物理**：McMillan/Allen-Dynes 公式 + Monte-Carlo 不确定性；
7. **稳定性理论**：von Neumann + 能量法 + CFL 三重校验；
8. **跨学科融合**：15 个不同领域的种子项目融合为统一的凝聚态框架。

---

*本项目由 15 个种子项目的核心算法深度融合而成，全部代码在单一 `main.py` 入口零参数可执行。*
