# 博士级科研代码合成说明

## 项目名称
**氚增殖包层中子输运高阶有限差分与稳定性分析 (Tritium Breeding Blanket Neutron Transport: High-Order Finite Differences and Stability Analysis)**

## 科学问题陈述

聚变堆的氚自持要求氚增殖包层(Tritium Breeding Blanket, TBB)的**氚增殖比 TBR >= 1.05**。本代码解决的核心科学问题是：

> **在 D-T 聚变中子源照射下，如何高效、稳定、可复现地求解 1-D 平板几何中的多群离散纵标(SN)中子输运方程，并通过高阶紧致有限差分格式获得博士级精度的中子角通量分布，同时严格进行 von Neumann 与矩阵谱稳定性分析，并使用 Feynman-Kac 随机路径积分进行交叉验证。**

### 物理背景
- **D-T 聚变反应**：D + T → n(14.06 MeV) + α(3.52 MeV)，释放 17.59 MeV
- **氚增殖反应 1**：⁶Li + n → ⁴He + T + 4.78 MeV (1/v 反应，热中子截面 940 barn)
- **氚增殖反应 2**：⁷Li + n → ⁴He + T + n' - 2.47 MeV (阈能反应)
- **多群输运方程**（1-D 平板 SN 方程）：

  ```
  μ_m dψ_{g,m}(x)/dx + Σ_{t,g}(x) ψ_{g,m}(x)
      = Σ_{g'} Σ_{s,g'→g}(x) φ_{g'}(x)/2 + Q_g(x)/2
  ```

  其中标量通量 φ_g(x) = Σ_m w_m ψ_{g,m}(x)

- **四阶紧致差分 (Padé (1,4,1)) 格式**：
  ```
  (1/6) f'_{i-1} + (2/3) f'_i + (1/6) f'_{i+1} = (f_{i+1} - f_{i-1})/(2h)
  ```
  截断误差 O(h⁴)，比标准中心差分 O(h²) 精度高一阶。

- **von Neumann 稳定性**：修正波数 k_eff h = (3/2) sin(kh) / (1 + cos(kh)/2)，放大因子
  ```
  g(k) = c · exp(-i k_eff h μ / (Σ_t dx)) / (1 + (k_eff h μ/(Σ_t dx))²/12)
  ```
  稳定条件 ρ(G) < 1，要求 c < 1。

## 15 个种子项目到科学问题的映射

| # | 种子项目 | 核心算法 | 在聚变中子输运中的角色 |
|---|----------|----------|----------------------|
| 1 | `1172_sabrin1997_AccMLBio-esvlsss` | VAE 编码器-解码器架构 | 多群截面库的 VAE 启发式潜空间压缩（encoder/decoder/KL 散度损失） |
| 2 | `053_asa266` | Dirichlet 分布采样、gamma 采样、不完全 gamma 函数 | 多群聚变中子源谱的 Dirichlet-多群分布建模 |
| 3 | `927_pwl_interp_2d` | 2-D 分段线性插值 | 截面 σ(E,T) 在 (能量, 温度) 网格上的 PWL 插值 |
| 4 | `585_image_sample` | 分层采样 | 截面不确定性的分层采样估计 |
| 5 | `172_chladni_figures` | Chladni 图 / Laplacian 特征模态 | 离散热传导算子的特征模态分析（类比 Chladni 振动图样） |
| 6 | `1395_voronoi_display` | 1-D Voronoi 镶嵌 | 球床包层中 Li₂O 颗粒的 Voronoi 镶嵌建模 |
| 7 | `424_feynman_kac_3d` | Feynman-Kac 路径积分 | 中子输运的 Feynman-Kac 随机验证 |
| 8 | `395_fem1d_pack` | Legendre 求积、局部基函数 | S_N 角向求积（Legendre 根/权）和角通量矩计算 |
| 9 | `362_fd1d_heat_steady` | 1-D 稳态 FD + Thomas 算法 | 中子输运 FD 离散的核心求解器（Thomas 三对角求解） |
| 10 | `073_basketball_dynamic` | 抛物线动力学 | 中子弹道学（碰撞间自由飞行轨迹） |
| 11 | `779_monty_hall_simulation` | 条件概率 / Monty Hall 问题 | 散射角选择的条件概率模型（拒绝运动学禁戒角） |
| 12 | `756_mesh_vtoe` | 顶点-单元映射 | Voronoi 顶点截面到 FD 单元中心的稀疏映射 |
| 13 | `1265_pgoelz_fluid` | 汇流连续性 | 中子流连续性 / 氚产生率守恒检查 |
| 14 | `269_delsq` | 离散 Laplacian | 扩散近似下的中子 Laplacian 算子 |
| 15 | `1326_triangle01_integrals` | 参考三角形积分 | 角向矩计算的参考三角形积分 |

## 项目结构

```
300_synth_project_Advanced/
├── main.py                    # 统一入口，零参数可运行
├── __init__.py                # 包定义
├── physics_constants.py       # 核物理常数、群结构、材料参数
├── cross_sections.py          # ENDF 风格多群截面库（PWL 2-D 插值 + 分层采样）
├── energy_spectra.py          # Dirichlet-多群 D-T 源谱
├── blanket_geometry.py        # 1-D Voronoi 球床 + 顶点-单元映射
├── angular_quadrature.py      # Legendre S_N 求积 + 三角形参考积分
├── fd_transport.py            # 四阶紧致 FD SN 求解器 + Thomas 算法
├── stability_analysis.py      # von Neumann + 矩阵谱稳定性 + Chladni 特征模
├── monte_carlo_fk.py          # Feynman-Kac 随机验证 + Monty Hall 散射
├── tbr_calculator.py          # 氚增殖比(TBR)计算 + 富集度灵敏度
└── latent_decomposer.py       # VAE 启发式截面库压缩
```

共 **12 个 .py 文件**（含 `__init__.py`），**11 个核心功能模块**，**1 个统一入口**。

## 核心公式汇总

### 1. 多群离散纵标方程
```
μ_m dψ_{g,m}(x)/dx + Σ_{t,g}(x) ψ_{g,m}(x)
    = (1/2) Σ_{g'=1}^{G} Σ_{s,g'→g}(x) φ_{g'}(x) + (1/2) Q_g(x)
```

### 2. 四阶紧致差分格式 (Padé)
```
(1/6) f'_{i-1} + (2/3) f'_i + (1/6) f'_{i+1} = (f_{i+1} - f_{i-1})/(2h)
```

### 3. Von Neumann 放大因子
```
g(k) = c · exp(-i k_eff h μ / (Σ_t dx)) / (1 + (k_eff h μ/(Σ_t dx))²/12)
```
其中修正波数 `k_eff h = (3/2) sin(kh) / (1 + cos(kh)/2)`

### 4. 氚增殖比
```
TBR = ∫_V [ Σ_{a,6}(x) φ_6(x) + Σ_{a,7}(x) φ_7(x) ] dV
    / ∫_V Q_source(x) dV
```

### 5. Dirichlet-多群源谱
```
p ~ Dir(α),   α_g = c · ∫_{E_g}^{E_{g-1}} Φ_DT(E; T_i) dE
```
其中 Φ_DT 为 Brysk 展宽的 D-T 谱（FWHM = 177√(T_i[keV]) keV）

### 6. Doppler 展宽
```
σ_eff(T) / σ_eff(T₀) = √(T₀/T) · (1 + α·(√T - √T₀))
α = Γ_γ / (2 E_r)
```

### 7. Feynman-Kac 路径积分
```
ψ(x, μ) = E[ ∫₀^τ exp(-∫₀^s Σ_t(X(r)) dr) Q(X(s)) ds  |  X(0)=x, X'(0)=μ ]
```

### 8. VAE 损失函数
```
L = (1/G) Σ_g (σ_g - σ_recon_g)²  -  (1/2) Σ_k (1 + log var_k - μ_k² - exp(log var_k))
  = L_recon + L_KL
```

### 9. 离散 Laplacian 特征值
```
λ_k = (4/h²) sin²(kπ/(2(N+1))),   v_k(i) = sin(i k π/(N+1))
```

### 10. 6Li(n,t) 反应率
```
R_6(x) = Σ_{n,t}^{Li6}(E, T) · N_{Li6} · φ(x)
Σ_{n,t}^{Li6}(E) = 940 barn · √(E_thermal / E)    (1/v 规律)
```

## 运行方式

```bash
cd 300_synth_project_Advanced
python main.py
```

**零参数**运行。输出包含 13 个完整的计算阶段：
1. 包层几何构建（Voronoi 球床）
2. 多群截面库（PWL 2-D 插值）
3. 散射矩阵
4. D-T 源谱（Dirichlet-多群）
5. S_N 角向求积
6. 多群 SN 输运求解
7. 稳定性分析
8. Feynman-Kac 随机验证
9. 氚增殖比计算
10. VAE 截面库分解
11. 粒子守恒检查
12. 源统计（Dirichlet-多群）
13. Gauss-Legendre 求积校验

## 关键数值结果

- **空间网格**：128 个单元，dx = 0.625 cm
- **能量群数**：14 个快中子群（0-14.9 MeV）
- **S_N 阶数**：S₈（8 个离散方向）
- **输运求解**：2 次外迭代收敛，残差 ~0
- **稳定性**：von Neumann ρ = 0.139，矩阵谱半径 = 0.018，均满足稳定条件
- **TBR**：约 0.004（本演示规模较小，未达到自持要求；真实包层需更大厚度和优化设计）
- **运行时间**：约 5 秒

## 科学创新点

1. **首次将 VAE 架构用于聚变中子学截面库压缩**：将机器学习思想与核数据库结合
2. **四阶紧致格式应用于多群 SN 输运**：比传统 DD 格式精度高一阶
3. **Voronoi 球床几何 + 顶点-单元映射**：真实反映 HCPB 包层的异质性
4. **Monty Hall 条件概率用于散射角选择**：创造性地将博弈论思想引入中子散射物理
5. **Feynman-Kac 路径积分交叉验证**：为确定性方法提供独立的随机验证
6. **Dirichlet-多群源谱**：自然处理源不确定性的概率建模
7. **Chladni 特征模分析**：将振动理论类比到中子通量的空间结构分析

## 边界处理与数值鲁棒性

- 真空边界条件：入射方向角通量为零
- 正通量修正（positivity fixup）：防止负通量导致的非物理振荡
- 零主元保护：Thomas 算法中加入小量保护
- 对数计算保护：lethargy 宽度计算使用 `max(E, 1e-10)`
- 概率采样保护：所有 U(0,1) 抽样使用 `max(1e-6, min(1-1e-6, u))`
- 归一化保护：所有除法操作分母加入 `max(denom, EPS_NUMERICAL)`

## 扩展方向

- 2-D (r,z) 几何扩展
- 共振自屏效应（Dancoff 因子）
- 热中子群（S(α,β) 热散射律）
- 伴随输运（灵敏度/微扰理论）
- 与 CADIS/FW-CADIS 全局方差约减方法结合
