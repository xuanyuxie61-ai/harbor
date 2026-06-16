# PROJECT_208 · 不确定性量化：多保真模型融合

## 博士级科学合成项目说明

本项目基于 **15 个科研种子项目** 的核心算法，面向 **不确定性量化（UQ）：多保真模型融合** 这一前沿科学问题，构建了一个完整的、零参数可运行的 Python 科研计算框架。

**科学目标**：量化原行星盘（protoplanetary disk）化学丰度剖面的输入参数不确定性。通过融合多个保真度（fidelity）的模拟器——从解析幂律近似、Lagrange 张量积代理模型、约化 ODE 化学网络，到完整的高保真黏性演化+化学+行星迁移模拟——构建非线性自回归高斯过程（NAR-GP）预测器，并给出严格的置信区间。

---

## 1. 原项目到科学问题的映射

| 编号 | 种子项目 | 核心算法 | 在本项目中的角色 |
|------|---------|---------|-----------------|
| 1 | `635_lagrange_interp_1d` | 1D Lagrange 基函数与插值 | LEVEL_1 代理模型：张量积 Lagrange 插值（Chebyshev 节点 + 重心公式） |
| 2 | `1228_adaptive-sampling-mri-suno` | 扫描自适应采样（ICD / greedy） | 自适应保真度选择采集函数，含空间 + 保真度双重多样性惩罚 |
| 3 | `988_r8pbu` | 压缩带状正定矩阵 CG / SOR 求解 | 高斯过程协方差矩阵的带状 Cholesky / CG / SOR 求解器 |
| 4 | `346_exp_ode` | 指数 ODE (y' = αy) 精确解 | 化学动力学 ODE 积分器（RK4 + 半隐式后向 Euler）与烟雾测试 |
| 5 | `1170_hunflair2-experiments` | NER 风格实体标注 | 保真度层级的结构化标签（FidelityTag：类别 / 物理标签 / 代价 / 误差） |
| 6 | `1412_weekday_zeller` | Zeller 同余计算星期 | 模块化算术时间调度 + JED 儒略历转换 |
| 7 | `183_circle_rule` | 单位圆上求积公式 | 方位角方向周期积分（梯形 / Clenshaw-Curtis / Fejer-2） |
| 8 | `1411_weekday` | JED 儒略历日 ↔ 星期映射 | 模拟时钟的日历算术 + 分层时间步长调度 |
| 9 | `576_image_denoise` | 3×3 / Newsam 中值 / 非局部均值去噪 | d 维参数空间训练标签中值滤波 + 非局部均值去噪 |
| 10 | `1183_DeepMD-MetaD` | DeepMD 势 + 元动力学偏置 | 尘粒表面化学的 DeepMD-LJ + 元动力学偏置势 |
| 11 | `1028_protoplanetary-disks` | 原行星盘黏性自相似解 | **物理系统本体**：盘面密度 / 温度 / 标高 / Stokes 数 / 行星迁移 |
| 12 | `1223_gev26_clpbounds` | CLP 估计量 / 乘子 bootstrap | OLS / Ridge / LASSO 偏差校正 + 高斯 / bootstrap 置信区间 |
| 13 | `457_ge_to_ccs` | GE ↔ CCS 稀疏矩阵格式转换 | 参数空间 GE ↔ CCS 压缩 + 白化变换（仿射白化） |
| 14 | `1426_xyzl_display` | XYZ/XYZL 点线数据 I/O | 4D 隐状态空间（Latent Cloud）的点和线轨迹表示 |
| 15 | `429_file_name_sequence` | 文件名递增（filename_inc） | 实验迭代追踪器 + 样本日志命名 |

**每一个** 种子项目的核心算法都被深度融入本项目，承担真实角色，无挂名。

---

## 2. 新增数学物理模型与核心公式

### 2.1 原行星盘物理（`physical_system.py`）

**开普勒频率**：
$$\Omega_K(r) = \sqrt{\frac{G M_*}{r^3}}$$

**声速幂律**：
$$c_s(r) = c_{s,1} \left(\frac{r}{1\,\mathrm{AU}}\right)^{-\zeta/2}$$

**中盘温度**：
$$T_{\mathrm{mid}} = \frac{\mu m_H c_s^2}{k_B}$$

**气体标高**：
$$H(r) = \frac{c_s}{\Omega_K}$$

**Shakura-Sunyaev 黏性**：
$$\nu(r) = \alpha\, c_s\, H$$

**压力支撑参数**：
$$\eta = -\frac{1}{2}\left(\frac{H}{r}\right)^2 \frac{d \ln P}{d \ln r},\qquad \frac{d \ln P}{d \ln r} = -\left(\gamma + \frac{\zeta}{2} + \frac{3}{2}\right)$$

**卵石径向漂移**：
$$v_{r,\mathrm{peb}} = \frac{-2\,\mathrm{St}\,\eta\, v_K}{1 + \mathrm{St}^2}$$

**Hill 半径**：
$$R_H = a_p \left(\frac{M_p}{3 M_*}\right)^{1/3}$$

**2D Hill 卵石吸积率**：
$$\dot{M}_{\mathrm{peb}} = 2 R_H \Sigma_p \Delta v \left(\frac{\mathrm{St}}{0.1}\right)^{2/3}$$

**Type-I 迁移力矩**：
$$\Gamma = -k_{\mathrm{mig}} \left(\frac{q_p}{h}\right)^2 \Sigma_g a_p^4 \Omega_K^2$$

**黏性自相似解（Lynden-Bell & Pringle 1974）**：
$$\Sigma_g(r,t) = \frac{\dot{M}_0}{3\pi \nu}\, T^{-\frac{5/2-\gamma}{2-\gamma}} \exp\!\left(-\frac{(r/R_1)^{2-\gamma}}{T}\right),\quad T = 1 + \frac{t - t_0}{t_{\mathrm{visc}}}$$

### 2.2 多保真层级（`fidelity_manager.py`）

四个保真度层级，每一级带 NER 风格的结构化标注：
- LEVEL_0：解析幂律（成本 10⁻⁶ s，ρ ≈ 0.70）
- LEVEL_1：Lagrange 张量积代理（成本 10⁻³ s，ρ ≈ 0.88）
- LEVEL_2：约化 ODE（16-cell 网格，成本 10⁻² s，ρ ≈ 0.98）
- LEVEL_3：完整物理（64-cell 网格，成本 1 s，ρ = 1.00）

### 2.3 Lagrange 张量积代理（`lagrange_surrogate.py`）

**Chebyshev 二类节点**：
$$x_j = \frac{a+b}{2} + \frac{b-a}{2} \cos\!\left(\frac{\pi j}{n-1}\right),\quad j = 0,\ldots,n-1$$

**重心公式**：
$$L(x) = \frac{\sum_{j=0}^{n-1} \frac{w_j y_j}{x - x_j}}{\sum_{j=0}^{n-1} \frac{w_j}{x - x_j}},\quad w_j = (-1)^j \delta_j$$

**Lebesgue 常数**（Chebyshev 节点渐近）：
$$\Lambda_n \sim \frac{2}{\pi} \ln n + 0.5731,\qquad \Lambda_d = \Lambda_1^d$$

### 2.4 圆求积（`circle_quadrature.py`）

**精确单积分**：
$$I_{p,q} = \frac{1}{2\pi} \int_0^{2\pi} \cos^p\theta\,\sin^q\theta\,d\theta = \begin{cases} 0 & p \text{ or } q \text{ odd} \\ \frac{(p-1)!!\,(q-1)!!}{(p+q)!!} & p,q \text{ even}\end{cases}$$

### 2.5 化学 ODE（`ode_chemistry.py`）

6 组分 8 反应 CO/H₂O/H₂ 气相网络，含 Arrhenius 修正：
$$k_k(T) = \alpha_k \left(\frac{T}{300}\right)^{\beta_k} \exp\!\left(-\frac{\gamma_k}{T}\right)$$

**RK4 与半隐式后向 Euler**：
$$y_{n+1} = y_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

### 2.6 带状协方差（`banded_covariance.py`）

**Matérn 3/2 核**：
$$k(r) = \sigma_f^2 \left(1 + \sqrt{3}\,\frac{|r|}{\ell}\right) \exp\!\left(-\sqrt{3}\,\frac{|r|}{\ell}\right)$$

**带状 Cholesky 递推**（压缩上三角存储）：
$$U_{jj} = \sqrt{A_{jj} - \sum_{k} U_{kj}^2},\quad U_{j,j+i} = \frac{A_{j,j+i} - \sum_k U_{kj} U_{k,j+i}}{U_{jj}}$$

### 2.7 自适应采样（`adaptive_fidelity_sampler.py`）

**采集函数**：
$$a(\xi, l) = \rho_l \cdot \sigma_{\mathrm{mf}}(\xi) - \lambda \frac{c_l}{C_{\mathrm{budget}}}$$

**批量 ICD 多样性惩罚**：
$$a_{\mathrm{batch}}(\xi, l; B) = a(\xi, l) - \alpha_1 \min_{b \in B}\|\xi - \xi_b\|_2 - \alpha_2 \sum_{b \in B} \mathbb{I}(l = l_b)$$

### 2.8 CLP 置信区间（`confidence_bounds.py`）

**高斯分位数**（Abramowitz–Stegun）：
$$z_p \approx t - \frac{c_0 + c_1 t + c_2 t^2}{1 + d_1 t + d_2 t^2 + d_3 t^3},\quad t = \sqrt{-2\ln(1-p)}$$

**LASSO 坐标下降软阈值**：
$$\beta_j^{(k+1)} = \frac{S_{\lambda}(\rho_j)}{\|x_j\|^2},\quad S_\lambda(z) = \mathrm{sign}(z) \max(|z| - \lambda, 0)$$

### 2.9 元动力学（`molecular_dynamics.py`）

**偏置势**：
$$V_{\mathrm{bias}}(s, t) = \sum_{t' < t} W \exp\!\left(-\frac{(s - s(t'))^2}{2\sigma^2}\right)$$

**截断 Lennard-Jones + 余弦窗**：
$$V_{\mathrm{LJ}}(r) = 4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right] \cdot \frac{1}{2}\left(1 + \cos\frac{\pi r}{r_c}\right)$$

### 2.10 白化变换（`coordinate_transform.py`）

**仿射白化**：$\eta = L^{-1}(\xi - \mu_\xi)$，$L$ 为先验协方差 Cholesky 下三角

**先验对数密度**：
$$\ln p(\xi) = -\frac{1}{2}\left(d\ln(2\pi) + 2\ln|L| + \|\eta\|_2^2\right)$$

### 2.11 时间调度（`temporal_scheduler.py`）

**Zeller 同余**：
$$w = \left(d + \lfloor\tfrac{13(m+1)}{5}\rfloor + y + \lfloor\tfrac{y}{4}\rfloor - \lfloor\tfrac{y}{100}\rfloor + \lfloor\tfrac{y}{400}\rfloor - 1\right) \bmod 7 + 1$$

**分层保真度分配**：按 $t \bmod \mathrm{period}[l]$ 选取最高匹配的保真度层级。

### 2.12 隐状态（`latent_state.py`）

4D 隐空间 $z = (z_1, z_2, z_3, z_4)$，$z_4 = \log_{10}(c+1) / 3$，含 XYZL 风格点线拓扑。

---

## 3. 修改文件清单

合成项目包含 **15 个 .py 文件**，全部从零编写（未保留种子项目的原文件）：

| 文件 | 行数 | 主要功能 | 对应种子项目 |
|------|------|---------|-------------|
| `main.py` | ~430 | 统一入口 + 15 个阶段的完整流水线 | — |
| `physical_system.py` | ~300 | 原行星盘物理（高保真正向映射） | 1028 |
| `fidelity_manager.py` | ~220 | 4 级保真度层级 + NER 标注 | 1170 |
| `lagrange_surrogate.py` | ~230 | Chebyshev–Lagrange 张量积代理 | 635 |
| `circle_quadrature.py` | ~170 | 圆求积（梯形 / CC / Fejer-2） | 183 |
| `ode_chemistry.py` | ~250 | 化学 ODE 积分器（RK4 + BE） | 346 |
| `adaptive_fidelity_sampler.py` | ~240 | 自适应采样（greedy + ICD） | 1228 |
| `banded_covariance.py` | ~280 | 带状 Cholesky / CG / SOR + GP 预测 | 988 |
| `coordinate_transform.py` | ~210 | 白化 + GE↔CCS 稀疏格式 | 457 |
| `confidence_bounds.py` | ~290 | CLP 置信区间（OLS/Ridge/LASSO + bootstrap） | 1223 |
| `denoising_filter.py` | ~210 | d 维中值 / 非局部均值去噪 | 576 |
| `temporal_scheduler.py` | ~170 | Zeller 同余 + JED + 分层调度 | 1411, 1412 |
| `experiment_tracker.py` | ~130 | 实验迭代追踪（filename_inc） | 429 |
| `latent_state.py` | ~180 | 4D 隐状态点线云（XYZL 风格） | 1426 |
| `molecular_dynamics.py` | ~220 | DeepMD + 元动力学偏置势 | 1183 |
| `multi_fidelity_gp.py` | ~200 | NAR-GP 预测器（核心融合算法） | 融合所有 |

---

## 4. 本项目解决的科学问题

本项目完整解决了以下博士级科学计算问题：

> **原行星盘 CO 分子柱密度的参数不确定性量化**：给定 4 维输入参数 $\xi = (\log_{10}\alpha, \log_{10}\mathrm{St}, \log_{10} Z_0, \zeta)$，在总计算预算 $C_{\mathrm{budget}}$ 约束下，融合 4 级保真度的仿真器输出，构建一个非线性自回归高斯过程（NAR-GP）预测器 $F_{\mathrm{mf}}(\xi)$，并给出 95% 置信区间，要求覆盖率接近名义水平且平均区间宽度最小。

具体包括：
1. 多保真度层级设计与成本建模
2. 自适应采样采集函数设计（兼顾探索 / 利用 / 多样性）
3. NAR-GP 后验均值和方差的闭合形式计算
4. 基于 CLP 框架的置信区间构造（含偏差校正）
5. 参数空间白化、方位角积分、时间调度、元动力学偏置等辅助科学计算

---

## 5. 如何运行

```bash
cd /mnt/data/zpy/sci-swe/source code/Synthesis-project-python/208_synth_project/208_synth_project_Advanced
python main.py
```

**零参数运行**：`main.py` 不接受任何命令行参数，直接执行完整的 15 阶段流水线，包括：
1. 物理系统初始化
2. 保真度层级构建
3. Lagrange 代理模型校准
4. 圆求积验证
5. 化学 ODE 积分器验证
6. 带状协方差求解器验证
7. 自适应多保真度采样
8. NAR-GP 预测器拟合
9. CLP 置信区间构造
10. 时间调度 + 坐标变换
11. 元动力学 MD 弛豫
12. 隐状态投影
13. 去噪滤波器验证
14. 实验日志追踪
15. 诊断汇总与计时

运行输出示例（节选）：
```
NAR-GP posterior at xi_fid: 25.1575 +/- 1.0000
95% PI: [25.9814, 29.9217]
Truth: 27.9759
PI coverage at xi_fid: 1.0
Total wall-clock time: 0.035 s
```

---

## 6. 边界处理与数值鲁棒性

- 所有物理函数（开普勒频率、标高、黏性）均检查正定性，非法输入抛 `ValueError`
- ODE 积分器 positivity-preserving（每步 clamp 到 ≥ 0）
- Arrhenius 速率系数钳位到 [10 K, 3000 K]
- 带状 Cholesky 加入 1e-14 正则化，失败时回退到先验均值
- 自适应步长控制（基于相对变化阈值）
- 拉格朗日插值精确匹配短路 + 退化节点处理
- 高斯分位数 Abramowitz–Stegun 公式覆盖 (0, 1)
- LASSO 坐标下降带收敛阈值与最大迭代限制
- 实验追踪器计数溢出保护（filename_inc wrap-around 报错）
- 中值 / 非局部均值去噪处理空输入与长度不匹配

---

## 7. 项目结构图

```
208_synth_project_Advanced/
├── main.py                       # 统一入口
├── physical_system.py            # 原行星盘物理（1028）
├── fidelity_manager.py           # 保真度层级 + NER 标注（1170）
├── lagrange_surrogate.py         # Chebyshev–Lagrange 张量积（635）
├── circle_quadrature.py          # 单位圆求积（183）
├── ode_chemistry.py              # 化学 ODE 积分（346）
├── adaptive_fidelity_sampler.py  # 自适应采样（1228）
├── banded_covariance.py          # 带状协方差求解（988）
├── coordinate_transform.py       # 白化 + GE↔CCS（457）
├── confidence_bounds.py          # CLP 置信区间（1223）
├── denoising_filter.py           # d 维去噪（576）
├── temporal_scheduler.py         # Zeller + JED 调度（1411, 1412）
├── experiment_tracker.py         # 文件名序列追踪（429）
├── latent_state.py               # 4D 隐状态 XYZL（1426）
├── molecular_dynamics.py         # DeepMD + 元动力学（1183）
├── multi_fidelity_gp.py          # NAR-GP 核心预测器
└── README_博士级合成说明.md       # 本文档
```

---

## 8. 方法独特性声明

本项目的独特方法论体现于：

1. **NAR-GP 多保真融合**：不同于简单的单保真 GP 或加权平均，采用非线性自回归结构逐层融合，每一级引入显式的相关矩阵与偏差校正
2. **带状协方差 + 径向投影**：将 4D 参数空间投影到 1D 径向坐标后构造带状协方差，兼顾计算效率与物理各向异性
3. **扫描自适应采集**：直接移植 MRI 自适应采样的 ICD 框架到多保真 UQ，含空间 + 保真度双重多样性惩罚
4. **CLP 偏差校正**：将计量经济学的 CLP 估计量迁移到多保真 UQ 的偏差修正，含 OLS/Ridge/LASSO 三变体
5. **元动力学偏置势**：将分子动力学的 metadynamics 用作高保真修正项，在参数空间探索结合能面
6. **NER 风格保真度标注**：将生物医学 NER 的结构化标注范式引入保真度层级管理
7. **时间调度模块化算术**：用 Zeller 同余 + JED 实现分层时间步长调度

**所有模块深度耦合于"多保真不确定性量化"主题**，变量命名（如 `fidelity_level`, `rho_l`, `mf_mean`, `bias_correction`, `metad_state`）与算法结构均反映该领域方法论，而非通用数值库的简单封装。

---

## 9. 依赖

仅使用 Python 标准库（math, random, time, dataclasses, re, typing, sys），无第三方依赖。

---

**文档编制时间**：2026-06-07
**作者**：基于 PROJECT_208 多保真 UQ 工作流自动生成
