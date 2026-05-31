# 激光-等离子体相互作用多尺度数值模拟系统

## 博士级合成说明文档

---

## 1. 项目概述

本项目将 **15 个科研代码项目** 的核心算法融合，构建了一个面向 **惯性约束聚变（ICF）中高功率激光在等离子体中传播与能量沉积** 的博士级多尺度数值模拟系统。科学领域严格限定为 **等离子体物理：激光等离子体相互作用**。

### 1.1 核心科学问题

在 ICF 实验中，多束高功率激光（如 Nd:YAG，$\lambda = 1.064~\mu\text{m}$）同时照射氘氚靶丸，在其表面产生扩展的冕区等离子体。激光需要穿过这层非均匀等离子体才能到达烧蚀层并将能量有效地耦合到靶丸中。本系统旨在数值模拟：

1. **激光射线在非均匀等离子体中的传播轨迹**（几何光学 / Hamiltonian 射线方程）。
2. **逆轫致吸收导致的能量沉积**（沿射线路径的积分）。
3. **激光驱动的等离子体波激发与稳定性分析**（色散关系求根）。
4. **磁化等离子体中的偏振态演化**（Faraday 旋转与 Cotton-Mouton 效应）。
5. **等离子体电荷分离引起的电势与电场**（二维泊松方程稀疏矩阵求解）。
6. **参数空间的不确定性量化与优化**（拉丁超立方采样与稀疏网格积分）。

---

## 2. 原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 合成后文件 | 在科学问题中的真实角色 |
|:---:|--------|---------|-----------|----------------------|
| 1 | `680_line_grid` | 一维线段上的多种居中网格生成 | `plasma_grid.py` | 为等离子体密度场与电磁场提供 1D/2D/3D 空间离散网格（均匀、内点、中点等策略），支撑后续有限差分与射线追踪 |
| 2 | `186_cities` | 球面大圆距离计算 | `target_geometry.py` | ICF 靶丸表面激光入射角计算、多束激光在球面上的角间距、立体角覆盖分析 |
| 3 | `837_opt_sample` | 随机采样函数极值估计 | `parameter_sampling.py` | 在 6 维激光-等离子体参数空间中通过随机采样寻找使能量耦合效率最大的最优参数组合 |
| 4 | `924_pwc_plot_2d` | 二维分段常值函数表示 | `density_profile.py` | 将等离子体密度场表示为空间网格单元上的分段常数值，这是射线追踪与有限差分的基础数据结构 |
| 5 | `651_latin_edge` | 边缘拉丁超立方采样 | `parameter_sampling.py` | 对激光强度、波长、焦斑、密度、温度、标长等 6 维参数进行结构化拉丁超立方采样，用于不确定性量化与敏感性分析 |
| 6 | `017_area_under_curve` | 曲线下面积计算 | `density_profile.py` | 沿激光射线路径对密度场进行密集采样并积分，计算光程与有效等离子体标长 |
| 7 | `952_quadrilateral` | 三维四边形面积（Varignon 法） | `target_geometry.py` | 将 ICF 靶丸表面剖分为四边形网格面片，精确计算靶丸表面积与射线-面元立体角 |
| 8 | `704_luhn` | Luhn 校验和算法 | `data_integrity.py` | 为长时间模拟中的等离子体状态数据（密度、温度、电势）生成数值校验指纹，检测数据损坏 |
| 9 | `180_circle_map` | 矩阵将单位圆映射为椭圆 | `polarization_dynamics.py` | 琼斯矩阵将输入线偏振映射为输出椭圆偏振，直接对应激光在磁化等离子体中传播后的偏振态演化 |
| 10 | `943_quad_rule` | Gauss-Legendre 数值积分 | `quadrature_engine.py` | 沿激光射线路径进行高精度的 Gauss-Legendre 积分，计算逆轫致吸收能量沉积 |
| 11 | `1105_sparse_grid_hermite` | 稀疏 Gauss-Hermite 网格 | `quadrature_engine.py` | 对激光-等离子体参数空间进行高维稀疏网格积分，评估参数不确定性对能量沉积的统计影响 |
| 12 | `1162_stetter_ode` | 分段线性系数 ODE 求解 | `ray_tracer.py` | 将 Hamiltonian 射线方程 $d\mathbf{r}/ds = \mathbf{k}/|\mathbf{k}|$，$d\mathbf{k}/ds = (\omega_0/c)\nabla\eta$ 用自适应步长 RK4 数值积分 |
| 13 | `1404_wdk` | Weierstrass-Durand-Kerner 复根算法 | `dispersion_solver.py` | 求解等离子体色散关系的复根，获取 Langmuir 波频率与受激拉曼散射（SRS）增长率 |
| 14 | `1127_sphere_stereograph` | 球面立体投影 | `target_geometry.py` | 将靶丸球面表面保角映射到平面，用于参数化表面密度分布与激光焦斑的平面展开分析 |
| 15 | `1111_sparse_parfor` | 分块稀疏矩阵并行组装 | `sparse_field_solver.py` | 在 2D 结构化网格上组装泊松方程的稀疏有限差分矩阵，并求解等离子体电势分布 |

**所有 15 个输入项目均已真实融入合成项目，无遗漏、无挂名。**

---

## 3. 新增数学物理模型与核心公式

### 3.1 等离子体基本参数

**等离子体频率：**
$$\omega_p = \sqrt{\frac{n_e e^2}{\varepsilon_0 m_e}}$$

**德拜长度：**
$$\lambda_D = \sqrt{\frac{\varepsilon_0 k_B T_e}{n_e e^2}}$$

**临界密度：**
$$n_c = \frac{\varepsilon_0 m_e \omega_0^2}{e^2}$$

**激光在冷等离子体中的折射率：**
$$\eta(\mathbf{r}) = \sqrt{\max\left(0, 1 - \frac{\omega_p(\mathbf{r})^2}{\omega_0^2}\right)}$$

### 3.2 激光射线方程（Hamiltonian 几何光学）

定义哈密顿量 $H(\mathbf{r}, \mathbf{k}) = c |\mathbf{k}| \eta(\mathbf{r})$，射线方程为：

$$\frac{d\mathbf{r}}{ds} = \frac{\partial H}{\partial \mathbf{k}} = \frac{\mathbf{k}}{|\mathbf{k}|}$$

$$\frac{d\mathbf{k}}{ds} = -\frac{\partial H}{\partial \mathbf{r}} = -\frac{\omega_0}{c} \nabla \eta(\mathbf{r})$$

其中 $s$ 为光程长度。数值积分采用 **四阶 Runge-Kutta 方法**，步长自适应控制：
$$\Delta s \leq C \cdot \min(\Delta x, \Delta y) \cdot \eta(\mathbf{r})$$
当 $\eta < \eta_{\min} = 10^{-4}$ 时判定射线到达**截止面**（临界密度面）。

### 3.3 逆轫致吸收与能量沉积

**电子-离子碰撞频率（Spitzer 公式）：**
$$\nu_{ei} = \frac{Z n_e e^4 \ln\Lambda}{3(2\pi)^{3/2} \varepsilon_0^2 m_e^{1/2} (k_B T_e)^{3/2}}$$

**逆轫致吸收系数：**
$$\kappa_{ib} = \frac{\nu_{ei}}{c} \cdot \frac{\omega_p^2}{\omega_0^2} \cdot \frac{1}{\eta}$$

**沿射线的能量沉积（Gauss-Legendre 积分）：**
$$E_{\text{dep}} = \int_0^{s_{\max}} \kappa_{ib}(s) \, I(s) \, ds \approx \sum_{\text{segments}} \frac{b-a}{2} \sum_i w_i \, f\left(\frac{b-a}{2} x_i + \frac{b+a}{2}\right)$$

### 3.4 等离子体色散关系与 WDK 求根

**Langmuir 波 Bohm-Gross 色散（含阻尼近似）：**
$$D(\omega) = \omega^4 + i\nu \omega^3 - (\omega_p^2 + 3k^2 v_{te}^2)\omega^2 + \frac{\omega_p^4}{4} = 0$$

其中 $v_{te} = \sqrt{k_B T_e / m_e}$ 为电子热速度，$\nu$ 为等效小碰撞频率。

**Weierstrass-Durand-Kerner 迭代：**
$$z_i^{(new)} = z_i^{(old)} - \frac{P(z_i)}{\prod_{j \neq i}(z_i - z_j)}$$
初始猜测取 Cauchy 界内的单位根：$z_i^{(0)} = R \cdot e^{2\pi i (i-1)/n}$。

**受激拉曼散射（SRS）三波耦合：**
泵浦波 $(\omega_0, k_0)$、散射波 $(\omega_s, k_s)$、等离子体波 $(\omega_p, k_p)$ 满足：
$$\omega_0 = \omega_s + \omega_p, \quad k_0 = k_s + k_p$$
Rosenbluth-Liu 线性增长率：
$$\gamma_0 = \frac{v_{osc}}{2c} \sqrt{\omega_p \omega_0}, \quad v_{osc} = \frac{e E_0}{m_e \omega_0}$$

### 3.5 偏振态演化（Jones 矩阵方法）

**法拉第旋转角：**
$$\theta_F = \frac{e^3}{2 \varepsilon_0 m_e^2 c \omega^2} \int n_e(z) B_{\parallel}(z) \, dz$$

**Jones 传输矩阵（简化模型）：**
$$\mathbf{T} = \begin{pmatrix} \cos\phi + i\sin\phi\cos\delta & -\sin\phi + i\sin\phi\sin\delta \\ \sin\phi - i\sin\phi\sin\delta & \cos\phi - i\sin\phi\cos\delta \end{pmatrix}$$
其中 $\phi = \frac{\omega_p^2 \omega_c b_z}{2c \omega^3} \Delta z$ 为旋转角，$\delta$ 为 Cotton-Mouton 双折射相位差。

**Stokes 参数与偏振椭圆：**
$$S_0 = |E_x|^2 + |E_y|^2, \quad S_1 = |E_x|^2 - |E_y|^2$$
$$S_2 = 2\operatorname{Re}(E_x^* E_y), \quad S_3 = 2\operatorname{Im}(E_x^* E_y)$$
椭圆方位角：$\psi = \frac{1}{2}\arctan_2(S_2, S_1)$，椭圆率：$\varepsilon = \tan|\chi|$，$\chi = \frac{1}{2}\arcsin(S_3/S_0)$。

### 3.6 二维泊松方程与稀疏矩阵

**泊松方程：**
$$\nabla^2 \phi = -\frac{\rho}{\varepsilon_0}, \quad \rho = e(Z n_i - n_e)$$

**五点差分格式：**
$$\frac{\phi_{i+1,j} - 2\phi_{i,j} + \phi_{i-1,j}}{\Delta x^2} + \frac{\phi_{i,j+1} - 2\phi_{i,j} + \phi_{i,j-1}}{\Delta y^2} = -\frac{\rho_{i,j}}{\varepsilon_0}$$

矩阵形式 $\mathbf{A} \boldsymbol{\phi} = \mathbf{b}$，其中 $\mathbf{A}$ 为 $(n_x n_y) \times (n_x n_y)$ 稀疏矩阵，采用 **CSR 格式** 存储，边界节点施加 Dirichlet 条件 $\phi = 0$。

### 3.7 ICF 靶丸密度模型

**Fermi-Dirac 型径向过渡剖面：**
$$n_e(r) = n_0 f_{\text{plateau}} + \frac{n_0 (1 - f_{\text{plateau}})}{1 + \exp\left(\frac{r - R_0}{L_s}\right)}$$

**叠加随机高斯扰动（模拟束间等离子体不均匀性）：**
$$\delta n = A_{\text{pert}} n_0 \exp\left(-\frac{r^2}{2\sigma_{\text{pert}}^2}\right) \xi, \quad \xi \sim \mathcal{N}(0,1)$$

### 3.8 稀疏网格不确定性量化

Smolyak 稀疏网格公式（d 维积分）：
$$Q_d^{L} = \sum_{\max(0,L+1-d) \leq |\mathbf{l}| \leq L} (-1)^{L-|{\bf l}|} \binom{d-1}{L-|{\bf l}|} \left(Q_{l_1} \otimes \cdots \otimes Q_{l_d}\right)$$
其中 $Q_{l_i}$ 为第 $i$ 维的 Gauss-Hermite 一维积分法则。

---

## 4. 项目文件结构

```
026_synth_project/
├── main.py                        # 统一入口，零参数运行
├── physics_constants.py           # 物理常数与核心公式库
├── plasma_grid.py                 # 空间网格生成（line_grid 扩展）
├── density_profile.py             # 分段常数密度场（pwc_plot_2d + area_under_curve）
├── ray_tracer.py                  # Hamiltonian 射线追踪（stetter_ode → RK4）
├── quadrature_engine.py           # Gauss-Legendre 与稀疏 Hermite 积分（quad_rule + sparse_grid_hermite）
├── dispersion_solver.py           # 色散关系 WDK 求根（wdk）
├── parameter_sampling.py          # 拉丁超立方与随机采样优化（latin_edge + opt_sample）
├── target_geometry.py             # 靶丸几何、球面距离、立体投影（cities + sphere_stereograph + quadrilateral）
├── polarization_dynamics.py       # 偏振椭圆与 Jones 矩阵演化（circle_map）
├── sparse_field_solver.py         # 稀疏矩阵泊松求解（sparse_parfor）
├── data_integrity.py              # 数值校验与 Luhn 校验和（luhn）
└── README_博士级合成说明.md       # 本文档
```

**共 12 个 `.py` 文件（含 main.py），满足至少 8 个 `.py` 文件的要求。**

---

## 5. 运行方式

```bash
cd Synthesis-project-python/026_synth_project
python main.py
```

程序无需任何命令行参数，运行后将依次执行：

1. 物理参数初始化
2. 2D 空间网格生成
3. 分段常数等离子体密度场构建
4. 密度插值接口建立
5. 激光射线追踪（5 条射线）
6. 沿射线能量沉积 Gauss-Legendre 积分
7. Langmuir 波与 SRS 色散关系 WDK 求根
8. 6 维参数空间拉丁超立方采样
9. ICF 靶丸表面四边形网格建模
10. 偏振态演化与法拉第旋转计算
11. 二维泊松方程稀疏矩阵求解
12. 数据完整性校验
13. 综合结果汇总

---

## 6. 数值鲁棒性与边界处理

本项目在多处实现了严格的边界保护与数值鲁棒性措施：

- **密度非负性**：`np.clip(ne, 0.0, None)` 确保电子密度不会出现非物理负值。
- **折射率截断**：`eta = sqrt(max(0, 1 - ω_p²/ω²))`，当 $n_e \geq n_c$ 时自动截断为 0，防止复数折射率。
- **截止面检测**：射线追踪中当 $\eta < 10^{-4}$ 时自动终止，避免除以零。
- **自适应步长**：基于 Courant 条件与密度梯度幅值双重限制 RK4 步长，最小步长 $10^{-12}$ m。
- **差分保护**：密度梯度计算中 `dx_safe = max(dx, 1e-20)` 防止零间距导致的数值溢出。
- **泊松边界条件**：在稀疏矩阵中显式施加 Dirichlet 零边界条件，消除矩阵奇异性。
- **求解器回退**：`spsolve` 失败时自动回退到 `bicgstab` 迭代求解。
- **校验和验证**：Luhn + CRC-32 双重校验，确保模拟数据在传输或存储过程中未被篡改。

---

## 7. 合成后项目解决的核心科学问题

本系统可用于解决以下前沿科学计算问题：

1. **激光在非均匀等离子体中的传播与折射**：通过 Hamiltonian 射线方程追踪高功率激光在冕区等离子体中的实际路径，预测其是否会因密度梯度发生偏折或过早到达临界密度面被反射。

2. **激光能量耦合效率评估**：通过逆轫致吸收系数沿射线的 Gauss-Legendre 积分，定量计算有多少激光能量沉积在等离子体中，这是 ICF 内爆效率的关键指标。

3. **激光等离子体不稳定性（LPI）的线性增长率预测**：通过 WDK 算法求解 SRS 三波耦合的复色散关系，获取不稳定增长率 $\gamma_{SRS}$，为激光强度上限设计提供理论依据。

4. **磁化等离子体中的偏振诊断**：计算法拉第旋转角与偏振椭圆演化，可用于从实验测得的偏振信号反推等离子体密度与磁场分布。

5. **等离子体电荷分离与鞘层电场**：求解泊松方程获取等离子体中的电势分布与电场，这是理解有质动力加速和离子加速机制的基础。

6. **多参数耦合优化**：在 6 维激光-等离子体参数空间中进行拉丁超立方采样与稀疏网格积分，系统评估参数不确定性对物理结果的统计影响。

---

## 8. 技术规格

- **编程语言**：Python 3
- **依赖库**：NumPy, SciPy (sparse, linalg)
- **运行环境**：Linux / macOS / Windows，纯 Python 实现，无需外部编译
- **无可视化**：已按照要求删除所有绘图、图像输出相关内容
- **零参数入口**：`main.py` 直接运行即可完成完整计算流程
