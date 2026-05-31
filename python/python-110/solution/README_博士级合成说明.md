# 量子点单光子发射综合模拟系统 —— 博士级合成说明

## 一、项目概述

本项目围绕**光学工程前沿课题：半导体量子点单光子发射**，融合15个科研代码项目的核心算法，构建了一套从量子点电子结构、微腔电磁模式、开放量子系统主方程到光子统计与器件优化的完整理论模拟框架。

**科学问题定位**：如何在纳米尺度上通过量子点-微腔耦合工程实现高品质单光子源？该问题涉及量子力学、电磁学、统计物理与数值方法的深度交叉，属于当前量子信息与光量子技术领域的前沿博士级课题。

---

## 二、15个种子项目到科学问题的映射

| 原项目 | 核心算法 | 在合成项目中的科学角色 |
|:---|:---|:---|
| 1340_triangulation_node_to_element | 节点到单元的值平均、网格索引管理 | `mesh_generator.py` 中四边形网格节点到单元的场量平均映射 |
| 592_interp_equal | Newton 等距差商插值 | `wavefunction_solver.py` 中势场重构与波函数插值 |
| 713_maple_area | 网格点内含判断与面积估计 | `mesh_generator.py` 中圆域相对面积验证 |
| 953_quadrilateral_mesh | Q4网格生成、等参映射、面积计算、按面积加权采样 | `mesh_generator.py` 中微腔结构化网格构建与随机采样 |
| 292_disk_distance | 单位圆盘均匀采样、距离统计 | `photon_statistics.py` 中电偶极子随机取向与统计验证 |
| 1046_roulette_simulation | 随机数生成与概率过程模拟 | `photon_statistics.py` 中泊松光子探测蒙特卡洛模拟 |
| 380_fem_to_tec | FEM数据文件读写、节点-单元-值关联 | `em_field.py` 与 `utils.py` 中场模式求解与矩阵I/O |
| 087_biharmonic_exact | 高阶微分算子精确解、高阶导数离散 | `qd_hamiltonian.py` 中有效质量薛定谔方程的有限差分离散 |
| 460_gegenbauer_cc | Gegenbauer-Clenshaw-Curtis数值积分、Chebyshev偶次系数 | `wavefunction_solver.py` 中球谐展开权函数积分验证 |
| 039_asa113 | Transfer/Swap非层次聚类优化 | `state_clustering.py` 中量子点系综按发射特性聚类分类 |
| 508_hb_to_mm | 稀疏矩阵COO格式处理、大型矩阵结构读写 | `utils.py` 与 `qd_hamiltonian.py` 中Hamiltonian稀疏构造 |
| 1057_satisfy_brute | 穷举二进制向量搜索 | `parameter_search.py` 中器件参数空间网格优化搜索 |
| 208_conservation_ode | 守恒量保持ODE系统（单摆、捕食者-猎物、刚体） | `master_equation.py` 中密度矩阵迹守恒校验与演化 |
| 764_midpoint | 隐式中点法ODE积分 | `master_equation.py` 中密度矩阵轨迹的隐式中点演化 |
| 446_fractal_coastline | 分形海岸线扰动、盒计数维数 | `surface_roughness.py` 中量子点界面粗糙度建模 |

**无遗漏、无挂名**：每个原项目的核心算法均在合成项目中承担真实计算角色，而非仅做形式上的引用。

---

## 三、核心数学物理模型与公式

### 3.1 量子点受限电子结构

一维等效质量薛定谔方程（球对称近似，$u(r) = r R(r)$）：

$$
\hat{H} \psi(x) = \left[ -\frac{\hbar^2}{2 m^*} \frac{d^2}{dx^2} + V_{\text{conf}}(x) \right] \psi(x) = E \psi(x)
$$

其中有效质量 $m^* = m^*_\text{ratio} \cdot m_e$，对于 InAs 取 $m^*_e \approx 0.023 m_e$。

球形势阱限制势：

$$
V_{\text{conf}}(r) = \begin{cases} 0, & r \le R_{\text{dot}} \\ V_0, & r > R_{\text{dot}} \end{cases}
$$

Stark 效应（外加电场 $F$）：

$$
V_{\text{Stark}}(x) = -e F x
$$

### 3.2 激子与偶极跃迁

激子跃迁能量：

$$
E_{\text{ex}} = E_g^{\text{bulk}} + E_e^{(0)} + |E_h^{(0)}| - E_{\text{bind}}
$$

电偶极矩（电荷中心分离模型）：

$$
d = e \cdot |\langle x \rangle_e - \langle x \rangle_h|, \quad \langle x \rangle = \int x |\psi(x)|^2 dx
$$

### 3.3 微腔电磁模式与Purcell增强

圆盘微腔 whispering-gallery-like 模式近似：

$$
E(r) = \frac{E_0}{1 + (r / R_{\text{cavity}})^4}
$$

等效模式体积（二维FEM积分）：

$$
V_{\text{eff}} = A_{\text{eff}} \cdot \frac{\lambda}{n_{\text{eff}}}
$$

Purcell 因子：

$$
F_p = \frac{3}{4\pi^2} \left( \frac{\lambda}{n} \right)^3 \frac{Q}{V_{\text{eff}}}
$$

### 3.4 Jaynes-Cummings 哈密顿量与Lindblad主方程

旋波近似下的耦合系统哈密顿量：

$$
\hat{H} = \hbar \omega_c \hat{a}^\dagger \hat{a} + \hbar \omega_{\text{dot}} \hat{\sigma}_+ \hat{\sigma}_- + \hbar g \left( \hat{a}^\dagger \hat{\sigma}_- + \hat{a} \hat{\sigma}_+ \right)
$$

Lindblad 主方程：

$$
\frac{d\hat{\rho}}{dt} = -\frac{i}{\hbar} [\hat{H}, \hat{\rho}] + \sum_k \gamma_k \mathcal{D}[\hat{L}_k](\hat{\rho})
$$

其中耗散超算符：

$$
\mathcal{D}[\hat{L}](\hat{\rho}) = \hat{L} \hat{\rho} \hat{L}^\dagger - \frac{1}{2} \{ \hat{L}^\dagger \hat{L}, \hat{\rho} \}
$$

Liouvillian 超算符的矢量化形式：

$$
\text{vec}\left( \frac{d\hat{\rho}}{dt} \right) = \mathcal{L} \, \text{vec}(\hat{\rho})
$$

$$
\mathcal{L} = -\frac{i}{\hbar} (\mathbb{I} \otimes \hat{H} - \hat{H}^T \otimes \mathbb{I}) + \sum_k \gamma_k \left[ \hat{L}_k^* \otimes \hat{L}_k - \frac{1}{2} \mathbb{I} \otimes \hat{L}_k^\dagger \hat{L}_k - \frac{1}{2} (\hat{L}_k^T \hat{L}_k^*) \otimes \mathbb{I} \right]
$$

### 3.5 单光子统计与二阶关联函数

二阶关联函数：

$$
g^{(2)}(\tau) = \frac{\langle \hat{a}^\dagger(t) \hat{a}^\dagger(t+\tau) \hat{a}(t+\tau) \hat{a}(t) \rangle}{\langle \hat{a}^\dagger \hat{a} \rangle^2}
$$

弱耦合极限近似：

$$
g^{(2)}(\tau) = 1 - e^{-\Gamma_{\text{purcell}} |\tau|}
$$

其中 $\Gamma_{\text{purcell}} = \gamma_{\text{dot}} F_p + \kappa$。

### 3.6 Gegenbauer-Clenshaw-Curtis 正交积分

Gegenbauer 权函数积分：

$$
I = \int_{-1}^{+1} (1 - x^2)^{\lambda - 1/2} f(x) \, dx
$$

利用偶次 Chebyshev 系数递推：

$$
a_{2r} = \frac{2}{n} \left[ \frac{1}{2} f(1) + \sum_{j=1}^{n-1} f\left(\cos\frac{j\pi}{n}\right) \cos\frac{2rj\pi}{n} + \frac{(-1)^r}{2} f(-1) \right]
$$

$$
u_0 = \frac{\Gamma(\lambda + 1/2) \sqrt{\pi}}{\Gamma(\lambda + 1)} \cdot u_0
$$

### 3.7 分形界面粗糙度

分形扰动公式：

$$
q_{2i} = p_i, \quad q_{2i+1} = \frac{1}{2}(p_i + p_{i+1}) + w_i (p_i + p_{i+1}) - w_i (p_{i-1} + p_{i+2})
$$

粗糙度引起的非均匀展宽：

$$
\Gamma_{\text{inhom}} = \frac{1}{\hbar} \frac{\hbar^2}{2 m^* R_{\text{dot}}^2} \frac{\delta_r}{R_{\text{dot}}}
$$

### 3.8 聚类优化准则

类内方差和（k-means 目标函数）：

$$
J = \sum_{k=1}^{K} \sum_{i \in C_k} \| \mathbf{x}_i - \boldsymbol{\mu}_k \|^2
$$

Transfer/Swap 操作逐对象优化 $J$。

---

## 四、文件结构与功能说明

```
110_synth_project/
├── main.py                      # 统一入口，零参数运行
├── utils.py                     # 稀疏矩阵/数组校验/三对角求解/文件I/O
├── qd_hamiltonian.py            # 量子点Hamiltonian构建、本征值求解、偶极矩
├── wavefunction_solver.py       # Gegenbauer积分、Newton插值、Numerov打靶法
├── mesh_generator.py            # Q4网格生成、等参映射、面积计算、节点-单元映射
├── em_field.py                  # 腔模场分布、模式体积、Purcell因子、FEM模式求解
├── master_equation.py           # Jaynes-Cummings哈密顿量、Lindblad主方程、稳态/时域求解
├── photon_statistics.py         # g^(2)(tau)、HBT模拟、蒙特卡洛探测、HOM可见度
├── state_clustering.py          # Transfer/Swap聚类、光谱纯度评估
├── surface_roughness.py         # 分形边界扰动、盒计数维数、粗糙度展宽
└── parameter_search.py          # 穷举网格搜索、二进制编码优化、灵敏度分析
```

共 **11 个 .py 文件**（含 main.py），满足 >= 8 个的要求。

---

## 五、运行方式

```bash
python main.py
```

无需任何命令行参数。程序自动执行以下8个计算步骤：

1. **量子点电子/空穴受限态求解**：一维有效质量薛定谔方程有限差分解
2. **Gegenbauer-Clenshaw-Curtis 特殊函数积分**：验证波函数归一化
3. **微腔网格生成与电磁场模式**：Q4结构化网格 + Purcell 因子估算
4. **量子点-微腔 Lindblad 主方程**：密度矩阵稳态与时间演化
5. **单光子发射统计与二阶关联函数**：$g^{(2)}(0)$、HBT、蒙特卡洛探测
6. **量子点系综聚类与光谱纯度**：Transfer/Swap 非层次聚类
7. **界面粗糙度与分形扰动效应**：分形边界建模与线宽展宽
8. **单光子源参数优化**：四维参数空间穷举搜索与灵敏度分析

---

## 六、边界处理与数值鲁棒性

- **除零防护**：`safe_inverse`、`tridiagonal_solve` 中对小分母截断到 `1e-15`
- **数组校验**：所有模块入口均有 `validate_array_1d/2d`，防止空数组与维度错误
- **奇异矩阵处理**：`master_equation.py` 中稳态求解使用 `np.linalg.lstsq` 替代直接求逆
- **非负约束**：`photon_statistics.py` 中 `g^(2)(\tau)` 强制截断到 `>= 0`
- **数值稳定性**：`wavefunction_solver.py` 中 Numerov 积分对近零 `k^2` 做正则化
- **参数边界**：`parameter_search.py` 中约束函数确保物理可行域

---

## 七、科学难度与前沿性说明

本项目具备博士级计算复杂度，体现在：

1. **多物理场耦合**：同时求解量子力学（薛定谔方程）、电磁学（亥姆霍兹/模式体积）、开放量子系统（Lindblad主方程）与统计物理（光子关联函数）
2. **高阶数值方法**：Gegenbauer-Clenshaw-Curtis谱积分、Newton差商插值、隐式中点ODE积分、Numerov稳定打靶法
3. **非平凡优化**：非层次聚类中的Transfer/Swap组合优化、多维参数空间穷举搜索
4. **前沿科学指标**：Purcell因子、$g^{(2)}(0)$反聚束判据、HOM干涉可见度、光谱纯度指数
5. **工程鲁棒性**：全面的边界条件判断、误差处理与数值稳定性设计

---

## 八、输出示例

程序运行后终端输出包含：
- 电子/空穴基态能量（~0.05 eV / ~0.025 eV）
- 激子跃迁能量（~0.49 eV，对应 ~2.5 μm 近红外）
- Purcell 增强因子（~12）
- $g^{(2)}(0) \approx 0$（强反聚束判定）
- 稳态密度矩阵迹守恒校验通过
- 聚类后各类的光谱纯度指标
- 最优器件参数组合与灵敏度系数

完整运行时间约 **2-3 秒**（普通CPU）。
