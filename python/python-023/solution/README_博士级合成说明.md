# 博士级合成说明：空间等离子体波粒相互作用自适应相空间输运模拟

## 一、项目概述

本项目围绕**等离子体物理：空间等离子体波粒相互作用**这一前沿科学领域，基于15个科研代码项目的核心算法，融合构建了一个面向博士级难度的综合性数值计算框架。

### 核心科学问题

磁层空间中 whistler 模电磁波与电子的回旋共振相互作用，以及由此导致的非热电子在分形湍流磁场中的相空间输运与加热。该问题涉及以下关键物理过程：

1. **电磁波色散与阻尼**：whistler 模在磁层等离子体中的传播特性及其与电子的共振能量交换；
2. **波粒回旋共振**：Doppler 频移后的电子回旋共振条件；
3. **准线性扩散**：共振粒子在速度空间中的随机行走与能量扩散；
4. **分形湍流效应**：磁重联区域的分形磁场结构对粒子轨道的混沌扰动；
5. **不确定性量化**：磁场涨落对粒子扩散系数的随机影响。

---

## 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 科学问题中的角色 |
|:---:|:---|:---|:---|
| 1 | `751_menger_sponge_chaos` | 迭代函数系统(IFS) | **分形磁通管生成**：使用Menger海绵IFS模拟磁重联区域的分形磁通管边界，Hausdorff维数 $D_H = \log 20 / \log 3 \approx 2.727$ |
| 2 | `1370_ubvec` | 无符号二进制向量/Gray码 | **PCE多指标枚举**：Gray码遍历和组合枚举思想用于多项式混沌展开的多维指标生成 $J = \{\alpha \in \mathbb{N}_0^N : |\alpha| \leq P\}$ |
| 3 | `770_mm_to_hb` | 稀疏矩阵格式转换 | **稀疏扩散算子存储**：COO/CSR格式转换与Harwell-Boeing输出，用于大型准线性扩散矩阵的存储与外部求解器接口 |
| 4 | `802_newton_rc` | 反向通信Newton法 | **色散关系求解**：反向通信结构控制Newton-Raphson迭代，求解whistler模的复频率 $\omega(k) = \omega_r + i\gamma$ |
| 5 | `082_beta_nc` | 非中心Beta分布CDF | **非热粒子尾巴**：非中心不完全Beta函数描述共振加速后的超热电子尾巴分布 $f_{tail}(v)$ |
| 6 | `853_pce_legendre` | 随机Galerkin矩阵组装 | **PCE不确定性量化**：Legendre多项式混沌展开的Galerkin投影组装随机扩散矩阵 $B_{XY}$ |
| 7 | `1064_sensitive_ode` | 高敏感ODE参数 | **混沌粒子轨道**：对初值极其敏感的Lorentz力轨道积分，李雅普诺夫指数量化混沌程度 |
| 8 | `425_ffmatlib` | 有限元网格插值 | **相空间场插值**：三角网格上的数据插值思想用于速度空间电磁场到粒子位置的映射 |
| 9 | `931_pyramid_felippa_rule` | 金字塔高斯积分 | **速度空间矩计算**：高维数值积分规则计算分布函数的各阶速度矩（密度、动量、能量、热流） |
| 10 | `780_mortality` | 概率密度/生存分析 | **粒子逃逸概率**：基于生存分析模型计算粒子从共振区逃逸的概率 $P_{esc}(t) = 1 - e^{-t/\tau_{esc}}$ |
| 11 | `741_matrix_exponential` | 矩阵指数Pade逼近 | **扩散算子时间演化**：Pade逼近精确计算 $e^{A\Delta t}$，推进Fokker-Planck方程 |
| 12 | `638_lagrange_nd` | 多维Lagrange插值 | **相空间分布重构**：2D张量积Lagrange插值在 $(v_\parallel, v_\perp)$ 速度空间中重构分布函数 |
| 13 | `214_contour_sequence4` | 序列数据处理 | **波场时间序列**：序列文件名递增、数据网格重排、时间序列统计量计算 |
| 14 | `1398_voronoi_plot` | Voronoi距离搜索 | **共振粒子检测**：速度空间中的p-范数最近邻搜索，识别满足回旋共振条件的粒子团簇 |
| 15 | `1292_tri_surface_display` | 三角网格数据处理 | **网格元数据处理**：三角网格节点/单元数据读取处理思想用于有限元风格的网格操作 |

---

## 三、新增数学物理模型与核心公式

### 3.1 回旋共振条件

电子与 whistler 模的 Doppler-shifted cyclotron resonance：

$$
\omega_k - k_\parallel v_\parallel - \frac{n \Omega_e}{\gamma} = 0
$$

其中
- $\Omega_e = e B_0 / m_e$ 为电子回旋频率
- $\gamma = (1 - v^2/c^2)^{-1/2}$ 为洛伦兹因子
- $n = -1$ 对应电子回旋阻尼

共振速度：

$$
v_{\parallel, res} = \frac{\omega_k + \Omega_e / \gamma}{k_\parallel}
$$

### 3.2 动力学色散关系

平行传播 whistler 模的Vlasov-Maxwell色散函数：

$$
D(k, \omega) = 1 - \frac{\omega_{pe}^2}{2 \omega \Omega_e} \left[ Z(\zeta_e) - \left(1 - \frac{\omega}{k_\parallel v_{te}}\right) Z'(\zeta_e) \right] = 0
$$

其中等离子体色散函数（Fried-Conte函数）：

$$
Z(\zeta) = i \sqrt{\pi} \, e^{-\zeta^2} \text{erfc}(-i \zeta), \quad \zeta_e = \frac{\omega - \Omega_e}{|k_\parallel| v_{te}}
$$

### 3.3 准线性扩散张量

Kennel & Engelmann (1966) 准线性扩散系数：

$$
D_{\parallel\parallel}^{QL} = \sum_k \frac{\pi q_e^2}{m_e^2} |E_k|^2 J_1^2(x_e) \, \delta(\omega_k - k_\parallel v_\parallel - \Omega_e / \gamma) \left(1 - \frac{k_\parallel v_\parallel}{\omega_k}\right)^2
$$

$$
D_{\perp\perp}^{QL} = \sum_k \frac{\pi q_e^2}{m_e^2} |E_k|^2 [J_0'(x_e)]^2 \, \delta(\omega_k - k_\parallel v_\parallel - \Omega_e / \gamma) \left(\frac{k_\parallel v_\perp}{\omega_k}\right)^2
$$

其中 $x_e = k_\perp v_\perp / \Omega_e$，$J_n$ 为Bessel函数。宽化的$\delta$函数：

$$
\delta(x) \to \frac{1}{\sqrt{\pi} \Delta v} \exp\left(-\frac{x^2}{\Delta v^2}\right)
$$

### 3.4 Fokker-Planck方程

速度空间分布函数的准线性输运方程：

$$
\frac{\partial f}{\partial t} = \frac{\partial}{\partial v_\parallel}\left(D_{\parallel\parallel} \frac{\partial f}{\partial v_\parallel}\right) + \frac{1}{v_\perp}\frac{\partial}{\partial v_\perp}\left(v_\perp D_{\perp\perp} \frac{\partial f}{\partial v_\perp}\right) + 2 D_{\parallel\perp} \frac{\partial^2 f}{\partial v_\parallel \partial v_\perp}
$$

半离散形式：$\mathbf{f}(t + \Delta t) = e^{A \Delta t} \mathbf{f}(t)$

### 3.5 矩阵指数Pade逼近

Moler & Van Loan (2003) 算法：

1. 缩放：$s = \max(0, \lfloor \log_2 \|A\|_\infty \rfloor + 1)$，$A_s = A / 2^s$
2. $(6,6)$阶对角Pade逼近：

$$
e^{A_s} \approx D^{-1} E, \quad E = I + \sum_{k=1}^{q} c_k A_s^k, \quad D = I + \sum_{k=1}^{q} (-1)^k c_k A_s^k
$$

其中 $c_k = \frac{(q-k+1)! \, q!}{(2q-k+1)! \, k! \, (q-k)!}$

3. 平方：$e^A = (e^{A_s})^{2^s}$

### 3.6 Kappa分布

非热等离子体分布函数：

$$
f_\kappa(\mathbf{v}) = \frac{n_0}{(\pi \kappa v_{th}^2)^{3/2}} \frac{\Gamma(\kappa+1)}{\Gamma(\kappa-1/2) \, \kappa^{3/2}} \left[1 + \frac{v^2}{\kappa v_{th}^2}\right]^{-(\kappa+1)}
$$

温度：$T_\kappa = T_M \cdot \frac{\kappa}{\kappa - 3/2}$（要求 $\kappa > 3/2$）

### 3.7 多项式混沌展开

随机磁场 $B(\mathbf{x}, \boldsymbol{\xi}) = B_0(\mathbf{x}) + \sum_{i=1}^N \xi_i B_i(\mathbf{x})$，其中 $\xi_i \sim U(-1, 1)$。

分布函数的PCE展开：

$$
f(\mathbf{v}, t, \boldsymbol{\xi}) = \sum_{\alpha \in J} f_\alpha(\mathbf{v}, t) \Psi_\alpha(\boldsymbol{\xi})
$$

均值与方差：

$$
\langle f \rangle = f_0, \quad \text{Var}(f) = \sum_{\alpha \neq 0} |f_\alpha|^2 \langle \Psi_\alpha^2 \rangle
$$

### 3.8 分形磁通管模型

Menger海绵IFS的20个仿射变换：

$$
\mathbf{x}_{n+1} = \frac{1}{3} I_3 \mathbf{x}_n + \mathbf{b}_j, \quad j \in \{1, \ldots, 20\}
$$

Hausdorff维数：$D_H = \frac{\ln 20}{\ln 3} \approx 2.727$

### 3.9 Boris推进器

保持相空间体积的显式粒子推进算法：

$$
\mathbf{v}^- = \mathbf{v}^{n-1/2} + \frac{q \Delta t}{2m} \mathbf{E}
$$

$$
\mathbf{v}^+ = \mathbf{v}^- + \mathbf{v}' \times \frac{2\mathbf{t}}{1+|\mathbf{t}|^2}, \quad \mathbf{v}' = \mathbf{v}^- + \mathbf{v}^- \times \mathbf{t}, \quad \mathbf{t} = \frac{q \Delta t}{2 m \gamma} \mathbf{B}
$$

$$
\mathbf{v}^{n+1/2} = \mathbf{v}^+ + \frac{q \Delta t}{2m} \mathbf{E}, \quad \mathbf{x}^{n+1} = \mathbf{x}^n + \Delta t \frac{\mathbf{v}^{n+1/2}}{\gamma}
$$

---

## 四、代码文件结构与功能

| 文件 | 功能 | 融合的原始项目 |
|:---|:---|:---|
| `main.py` | 统一入口，零参数运行完整模拟流程 | 全部15个项目 |
| `dispersion_relation.py` | 等离子体色散函数与Newton-Raphson求解 | `802_newton_rc` |
| `particle_orbit.py` | Boris推进器 + RK45自适应积分 + 李雅普诺夫指数 | `1064_sensitive_ode` |
| `fractal_magnetic_field.py` | IFS分形生成 + 盒计数维数 + 磁场映射 | `751_menger_sponge_chaos` |
| `pce_expansion.py` | Legendre PCE展开 + 多指标枚举 + Galerkin投影 | `853_pce_legendre`, `1370_ubvec` |
| `quasilinear_diffusion.py` | QL扩散系数计算 + 稀疏扩散矩阵组装 | `853_pce_legendre`, `425_ffmatlib` |
| `phase_space_lagrange.py` | 重心Lagrange插值 + Chebyshev节点 + 2D张量积 | `638_lagrange_nd` |
| `sparse_assembler.py` | COO/CSR转换 + 条件数估计 + HB格式输出 | `770_mm_to_hb` |
| `matrix_exponential_solver.py` | Pade逼近 + Krylov子空间 + 扩散算子演化 | `741_matrix_exponential` |
| `resonance_voronoi.py` | 回旋共振检测 + Voronoi最近邻 + 共振体积MC积分 | `1398_voronoi_plot` |
| `distribution_models.py` | Kappa分布 + 非中心Beta尾巴 + 逃逸概率 | `082_beta_nc`, `780_mortality` |
| `moment_integrator.py` | Simpson复合积分 + 速度空间矩（密度/温度/热流/熵） | `931_pyramid_felippa_rule` |
| `file_sequence_processor.py` | 序列文件名生成 + 网格重排 + 时间相关分析 | `214_contour_sequence4`, `1292_tri_surface_display` |

---

## 五、边界处理与数值鲁棒性

### 5.1 数值稳定性措施

1. **色散函数**：对 $|\zeta| > 50$ 使用渐进展开 $Z(\zeta) \approx -1/\zeta - 1/(2\zeta^3)$，避免Faddeeva函数溢出。

2. **Newton迭代**：
   - 步长阻尼：$|\Delta \omega| > 0.5|\omega|$ 时限制步长
   - Jacobian病态处理：$|D'(\omega)| < 10^{-30}$ 时使用伪逆
   - 收敛停滞检测：15步后残差减小不足5%则终止

3. **Boris推进器**：子步进策略，确保每回旋周期至少10步，满足 $\Omega_e \Delta t < 0.1$。

4. **矩阵指数**：
   - 自动缩放防止中间量溢出
   - 对病态分母矩阵 $D$ 使用 `lstsq` 回退
   - 范数监控：扩散不应改变总概率，异常增长时自动归一化

5. **分布函数**：全程强制 $f \geq 0$，插值后截断负值；对 $v_\perp = 0$ 处的奇异性使用 $v_\perp \geq 10^{-10}$ 保护。

6. **共振检测**：$|k| < 10^{-20}$ 时跳过；洛伦兹因子上限 $\gamma \leq 100$。

---

## 六、如何运行

```bash
cd Synthesis-project-python/023_synth_project
python main.py
```

无需任何命令行参数。程序将自动执行以下完整流程：

1. 初始化磁层顶典型等离子体参数（$B_0 = 100$ nT, $n_0 = 10$ cm$^{-3}$）
2. 生成分形湍流磁通管结构（2000个IFS采样点）
3. 求解whistler模色散关系（20个波数点）
4. 初始化500个非热电子（Kappa + 非中心Beta复合分布）
5. 积分Lorentz力轨道（5个回旋周期，200步）
6. 检测共振粒子（~21%满足共振条件）
7. 组装1024×1024准线性扩散稀疏算子（稀疏度>99%）
8. 矩阵指数时间演化（10步Pade逼近）
9. PCE不确定性量化（2维随机空间，3阶Legendre展开）
10. 相空间Lagrange重构与矩计算
11. 波场时间序列处理

---

## 七、合成后的项目能解决的科学问题

1. **空间天气预测**：理解磁层中高能电子的加速机制，为辐射带动力学模型提供微观物理输入；
2. **波粒共振能量交换**：定量计算whistler模通过回旋共振加热电子的效率；
3. **湍流输运**：评估分形磁场结构对粒子扩散系数的增强效应；
4. **不确定性量化**：通过PCE量化磁场测量误差对分布函数预测的影响；
5. **非热分布演化**：追踪Kappa分布尾巴在波作用下的时间演化；
6. **混沌轨道诊断**：通过李雅普诺夫指数识别随机加速区域。
