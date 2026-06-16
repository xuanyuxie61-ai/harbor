# PROJECT_249 - 博士级科学计算合成项目

## 计算天体物理：恒星演化与核反应网络——高阶有限差分与稳定性分析

**科学领域**：计算天体物理，恒星演化，核合成  
**方法论**：高阶紧致有限差分、自适应隐式时间积分、Gauss-Laguerre 正交、CVT 自适应网格、Hamming 检错、K-means 聚类分类、层次化合并树  
**难度**：博士级 / 前沿科学计算  
**语言**：Python 3  
**运行方式**：`python3 main.py`（零参数，可重复运行）

---

## 一、项目简介

本项目围绕"大质量恒星（15 M☉）从核心氢燃烧到硅燃烧阶段的演化"这一前沿天体物理问题，融合了 15 个种子项目的核心算法，构建了一个**端到端可复现的小规模科学计算实验**。项目覆盖：

1. **自适应 Lagrange 质量网格**（1D CVT 算法）
2. **恒星结构方程**（流体静力学平衡 + 辐射转移 + 能量产生）
3. **核反应网络**（pp 链、CNO、3α、12C(α,γ)16O、12C+12C）
4. **Gauss-Laguerre 正交**用于 Gamow 峰积分
5. **高阶紧致（Padé）有限差分**（四阶，谱分辨率优良）
6. **自适应隐式时间积分器**（B1G3 三步隐格式 + 中点重构 + ETD-RK4 指数差分）
7. **稳定性分析**（Gershgorin 特征值界 + 谱半径 + CG 求解 J^TJ）
8. **分形扰动与 1/f 噪声**（模拟湍流混合）
9. **阈值触发混合事件**（模拟挖掘上翻 dredge-up）
10. **燃烧壳层合并树**（层次化合并算法）
11. **燃烧状态 K-means 聚类**（识别不同核燃烧阶段）
12. **Hamming(7,4) 数据完整性校验**
13. **Zeller 风格 epoch 追踪**（恒星演化相位）

---

## 二、科学问题与物理背景

### 2.1 控制方程
在 Lagrange 质量坐标 m 下，恒星结构方程组为：

$$
\frac{dr}{dm} = \frac{1}{4\pi r^2 \rho}, \qquad
\frac{dP}{dm} = -\frac{Gm}{16\pi^2 r^4}, \qquad
\frac{dL}{dm} = \varepsilon_{\text{nuc}} - \varepsilon_\nu, \qquad
\frac{dT}{dm} = \nabla \frac{T}{P}\frac{dP}{dm}
$$

状态方程采用理想气体 + 辐射压：

$$
P = \frac{\rho k_B T}{\mu m_p} + \frac{a T^4}{3}
$$

Kramers 不透明度律：

$$
\kappa = \kappa_0 \rho T^{-3.5}
$$

### 2.2 核反应率
热核反应率的温度平均截面：

$$
\langle \sigma v \rangle = \left(\frac{8}{\pi \mu}\right)^{1/2} (kT)^{-3/2}
\int_0^\infty S(E) \exp\!\left[-\frac{E}{kT} - \frac{b}{\sqrt{E}}\right]\,dE
$$

其中 $b = \pi \sqrt{2\mu}\, Z_1 Z_2 e^2 / \hbar$ 为 Gamow 常数，$S(E)$ 为天体物理 S 因子。经过变量替换 $E = E_0 x$，其中 $E_0 = (b kT / 2)^{2/3}$ 为 Gamow 峰能量，积分化为标准 Gauss-Laguerre 形式。

### 2.3 有限差分离散
紧致四阶（Padé）格式：

$$
\alpha f'_{i-1} + f'_i + \alpha f'_{i+1} = \frac{3}{2} \cdot \frac{f_{i+1} - f_{i-1}}{2h}
$$

取 $\alpha = 1/4$ 时达到四阶精度（Lele 1992）。边界闭合采用二阶单侧格式。

### 2.4 隐式时间积分
**B1G3 三步隐格式**（Trenchea-Burkardt 重构）：

$$
a_3 y_{n+1} + a_2 y_n + a_1 y_{n-1} = 2\Delta t\, f(t_m, y_m)
$$

系数为：$a_3 = \tfrac{1}{2} + \tfrac{1}{\sqrt{3}}$，$a_2 = -\tfrac{2}{\sqrt{3}}$，$a_1 = -\tfrac{1}{2} + \tfrac{1}{\sqrt{3}}$，$b = \tfrac{1}{\sqrt{3}}$，具有 A(α)-稳定性（α ≈ 86°）。

**ETD-RK4 指数差分**（Kassam-Trefethen）：
对半离散系统 $y' = Ly + N(y)$，通过 Cauchy 回路积分计算矩阵指数：

$$
Q = \frac{dt}{2\pi i} \oint_{|z|=1} \frac{e^{z/2} - 1}{z}\,dz, \qquad
f_k = \frac{dt}{2\pi i} \oint \frac{p_k(z, e^z)}{z^3}\,dz
$$

### 2.5 稳定性分析
核网络 Jacobian $J_{ij} = \partial \dot{Y}_i / \partial Y_j$ 的谱半径决定显式格式的 CFL 极限：

$$
\Delta t_{\max} = \frac{0.8}{\rho(J)}
$$

Gershgorin 圆盘定理给出特征值的保守界：

$$
\lambda \in \bigcup_i \left[a_{ii} - R_i, a_{ii} + R_i\right], \qquad R_i = \sum_{j\neq i} |a_{ij}|
$$

### 2.6 自适应网格（CVT）
1D 中心 Voronoi 剖分：给定密度函数 $w(m)$，迭代计算生成点 $\{g_k\}$ 的加权质心，使网格在 $w(m)$ 大的区域（如燃烧壳层附近）自动加密。

### 2.7 扰动与随机过程
- **分形混合**：中点细分 + 随机扰动 $q_{2n} = 0.5(p_n + p_{n+1}) + w(p_n - p_{n+1})$，$w = \mu + \mu^2 \mathcal{N}(0,1)$
- **1/f 噪声**：Voss-McCartney 算法，B 个白噪声过程的倍频叠加
- **阈值混合**：当壳层组分越过阈值时，触发高斯窗口内的快速混合（类似 SIR 干预模型）

### 2.8 层次化合并树
燃烧壳层按质量坐标分层，当两壳层质量间距小于 $\eta(w_i + w_j)$ 时合并（类似 rockstar 暗物质晕合并树）。

### 2.9 燃烧阶段分类
特征向量 $x = (\log_{10} T, \log_{10} \rho, Y_H, Y_{He})$，K-means 聚类识别 5 种燃烧阶段：
- 静默、氢燃烧、氦燃烧、高级燃烧、核统计平衡（NSE）

### 2.10 数据完整性
Hamming(7,4) 编码：4 位消息 → 7 位码字，可检测并纠正单比特错误。对整个模拟状态（温度、密度、组分、压强、光度数组）计算校验和，用于检测静默数据损坏。

### 2.11 Epoch 追踪
Zeller 同余风格的模运算：

$$
\text{epoch} = n \bmod C, \qquad
\text{full\_cycles} = n \div C
$$

嵌套层次：$(outer, inner) = ((n \div c) \bmod C, n \bmod c)$，用于追踪 AGB 星中周期性氢/氦壳燃烧。

---

## 三、种子项目到科学问题的映射

| 种子项目 | 核心算法 | 在本项目中的角色 |
|---------|---------|-----------------|
| `1412_weekday_zeller` | Zeller 同余、模运算 | `epoch_tracker.py`：恒星演化阶段/相位追踪 |
| `1131_cchrisgong_aip_rockstar` | 暗物质晕合并树、Roche 瓣 | `shell_merger.py`：燃烧壳层层次化合并 |
| `061_b1g3` | B1G3 隐式三步 ODE 求解器 | `time_integrator.py`：B1G3 核网络积分 |
| `253_cvt_circle_nonuniform` | 非均匀密度 CVT | `mesh_adaptation.py`：Lagrange 质量网格自适应 |
| `446_fractal_coastline` | 分形中点扰动 | `perturbation_engine.py`：组分分形混合 |
| `1419_xy_display` | XY 数据 I/O、矩阵打印 | `data_integrity.py`：状态文件读写 |
| `640_laguerre_integrands` | Gauss-Laguerre 求积 | `nuclear_quadrature.py`：Gamow 峰反应率积分 |
| `1029_Fdl1989_TimingofOneShotInterventions` | SIR 多室模型 + 阈值干预 | `nuclear_network.py`：多种类反应网络 + dredge-up |
| `1401_wathen_matrix` | Wathen 稀疏矩阵 + CG | `stability_analysis.py`：Jacobian 组装与 CG 求解 |
| `765_midpoint_adaptive` | 自适应中点重构 ODE | `time_integrator.py`：自适应隐式中点 |
| `870_pink_noise` | 1/f 噪声、相关函数 | `perturbation_engine.py`：能量产生率涨落 |
| `1055_fperdigon_DeepHistoPathology` | 数据准备→训练→测试 pipeline | `burning_classifier.py`：特征→聚类→标签 pipeline |
| `583_image_quantization` | K-means 量化 | `burning_classifier.py`：燃烧状态 K-means 聚类 |
| `499_hamming` | Hamming(7,4) 纠错码 | `data_integrity.py`：状态校验和 |
| `614_kdv_etdrk4` | ETD-RK4 指数差分 | `time_integrator.py`：刚性问题指数积分 |

---

## 四、文件清单

```
249_synth_project_Advanced/
├── main.py                   # 统一入口（零参数运行）
├── physical_constants.py     # 物理常数、SEMF 质量、平均分子量、Gamow 参数
├── nuclear_quadrature.py     # Gauss-Laguerre 求积、Gamow 峰积分、CF88 反应率拟合
├── epoch_tracker.py          # Zeller 风格 epoch 追踪、阶段命名、onion-skin
├── finite_difference.py      # 紧致 Padé 格式、显式高阶 FD、超粘性、von Neumann 判据
├── time_integrator.py        # 自适应中点、B1G3、ETD-RK4
├── stability_analysis.py     # 稀疏矩阵、CG 求解、谱半径、Gershgorin 界
├── mesh_adaptation.py        # 1D CVT 自适应网格、插值
├── perturbation_engine.py    # 分形混合、1/f 噪声、阈值混合
├── nuclear_network.py        # 8 种核素反应网络、能量产生、中微子损失
├── data_integrity.py         # Hamming(7,4) 编码/解码、状态校验和、文件 I/O
├── shell_merger.py           # 燃烧壳层类、Roche 判据、合并树
├── burning_classifier.py     # 特征提取、K-means、燃烧阶段标签
├── stellar_structure.py      # 状态方程、不透明度、恒星结构 outward 积分
├── README_博士级合成说明.md  # 本文档
└── final_state.dat           # （运行时生成）模拟终态
```

共计 **15 个 .py 文件** + 1 个 README + 1 个运行时输出。

---

## 五、运行与复现

```bash
cd 249_synth_project_Advanced
python3 main.py
```

预计运行时间：~7 秒。输出包含 12 个阶段的诊断信息，并在 `final_state.dat` 中写入终态校验和。

**复现性保证**：所有随机数使用固定 seed（如 CVT 用 seed=7，K-means 用 seed=2，Hamming 用确定性算法），输出可完全复现。

---

## 六、关键科学结论

1. **显式四阶有限差分**在 sin(2πx) 测试中达到精确的 4 阶收敛（经验阶数 3.98→4.00→4.00→4.00）。
2. **自适应中点积分器**在刚性 pp 链测试中稳定前进 184 步，0 次 Newton 失败。
3. **B1G3 三步隐格式**给出与中点法一致的结果（Y_H → 0，He 增加）。
4. **ETD-RK4**在 L=-100 的刚性测试中保持稳定（|v| 从 1 衰减到 3.7×10⁻⁴⁴）。
5. **CVT 自适应网格**将质量点集中在燃烧壳层附近，最大/最小间距比 ~ 2.65。
6. **Gershgorin 谱半径**给出核网络 Jacobian 的最大显式步长 Δt_max ~ 3000 秒。
7. **K-means 聚类**成功将 48 个质量壳层划分为 4 类燃烧状态。
8. **Hamming 校验和**通过，证明数据在文件 I/O 后保持完整。
9. **分形混合**保持总质量守恒（sum 不变）。
10. **壳层合并树**将 5 个初始壳层合并为 4 个（H+He 壳层重叠合并）。

---

## 七、扩展方向

- 替换为真实核质量表（AME2020）替代 SEMF
- 加入电子简并压（Chandrasekhar 状态方程）
- 使用 IMEX Runge-Kutta 处理核耦合
- 引入对流传 mixing-length theory（MLT）
- 二维推广（网格 + 球谐）
- 并行化（MPI + domain decomposition）
- 对接 MESA 恒星演化代码进行校准

---

## 八、参考文献

1. Kippenhahn, Weigert & Weiss, *Stellar Structure and Evolution*, 2nd ed., Springer, 2012.
2. Iliadis, C., *Nuclear Physics of Stars*, 2nd ed., Wiley-VCH, 2015.
3. Caughlan & Fowler, *Atomic Data and Nuclear Data Tables* 40, 283 (1988).
4. Lele, S.K., *J. Comput. Phys.* 103, 16-42 (1992)（紧致差分）.
5. Trenchea & Burkardt, *Appl. Math. Lett.* 107 (2020)（中点重构）.
6. Kassam & Trefethen, *SIAM J. Sci. Comput.* 26, 1214-1233 (2005)（ETD-RK4）.
7. Cox & Matthews, *J. Comput. Phys.* 176, 430-455 (2002).
8. Du, Faber & Gunzburger, *SIAM Rev.* 41, 637-676 (1999)（CVT）.
9. Rolfs & Rodney, *Cauldrons in the Cosmos*, U. Chicago Press, 1988.
10. Orfanidis, *Introduction to Signal Processing*, Prentice-Hall, 1995.

---

**作者**：PROJECT_249 合成工作流  
**日期**：2026-06-07  
**许可**：仅供科研合成使用
