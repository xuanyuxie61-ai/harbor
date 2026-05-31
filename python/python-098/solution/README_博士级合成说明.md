# 超表面电磁调控与全息 — 博士级合成说明

## 一、项目概述

本项目围绕**电磁学：超表面电磁调控与全息**前沿领域，将 15 个种子科研代码项目的核心算法融合为一个统一的博士级 Python 计算框架。项目以介电超表面（dielectric metasurface）为物理载体，实现了从非周期布局优化、谱展开相位设计、远场球谐展开、散射算符低秩压缩、波传播模拟、非线性单元响应建模、网格剖分、点云 I/O、制造公差分析到全息相位优化的**全链路闭环计算**。

## 二、原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 合成后角色 | 融入文件 |
|:---:|:---|:---|:---|:---|
| 1 | florida_cvt_pop | 加权密度 CVT（Lloyd 迭代） | 电磁强度加权下非周期超表面单元最优布局 | `cvt_placer.py` |
| 2 | sphere_cvt | 球面 CVT / Delaunay 三角化 | 远场球面采样点优化、全息图案球面离散化 | `cvt_placer.py` |
| 3 | r8poly | Horner 求值、多项式运算 | Legendre/Chebyshev 谱相位展开中的多项式快速求值 | `phase_spectrum.py` |
| 4 | polpak (Legendre) | Legendre 递推、导数公式 | 二维相位剖面的正交多项式基展开 | `phase_spectrum.py` |
| 5 | polpak (spherical_harmonic) | 连带 Legendre、球谐函数 | 远场散射场的球谐函数展开与 Mie 散射系数 | `spherical_expansion.py` |
| 6 | svd_gray | SVD 低秩近似 | 超表面散射算符的低秩压缩与重构误差评估 | `scattering_operator.py` |
| 7 | ode_rk4 | 经典 RK4 ODE 积分 | 梯度有效折射率层中平面波的 RK4 传播 | `wave_propagation.py` |
| 8 | rk23 | RK2/RK3 误差估计 | 同上的自适应步长误差验证与对比 | `wave_propagation.py` |
| 9 | rubber_band_ode | 分段非线性 ODE | 双稳态 meta-atom 的等效分段非线性响应 | `nonlinear_resonator.py` |
| 10 | pendulum_ode_period | 非线性摆、周期椭圆积分 | 大角度失谐 meta-atom 的相位饱和与周期修正 | `nonlinear_resonator.py` |
| 11 | ice_to_medit | NETCDF↔MEDIT 网格转换 | 三维仿真域的四面体网格生成与质量评估 | `mesh_handler.py` |
| 12 | gmsh_to_fem | GMSH↔FEM 节点/单元 I/O | 网格节点坐标与单元连接的读写、维度检测 | `mesh_handler.py` |
| 13 | xyz_io | XYZ 点云 I/O | Meta-atom 三维几何点云的读写与统计 | `geometry_io.py` |
| 14 | craps_simulation | 概率统计/蒙特卡洛 | 纳米加工相位误差的蒙特卡洛良率分析 | `tolerance_analysis.py` |
| 15 | square_integrals | 正方形单重积分 | 方形贴片矩量法基函数的解析矩与数值验证 | `moment_integrals.py` |
| 16 | scip_solution_read | 优化解文件解析 | 二元/多相位全息图配置的读写与约束验证 | `hologram_io.py` |

> **注**：polpak 虽为一个项目，但内部包含 Legendre、spherical_harmonic、Zernike 等多个子模块，本项目提取了其中与电磁球谐展开直接相关的 Legendre 递推与球谐函数。

## 三、新增数学物理模型与核心公式

### 3.1 电磁加权 CVT 布局

超表面孔径上的 meta-atom 密度与目标远场强度成正比：

$$
\rho(x, y) = \frac{|E_{\text{target}}(x, y)|^2}{\iint |E_{\text{target}}|^2 \, dx\, dy}
$$

Lloyd 松弛的加权质心：

$$
\mathbf{C}_i = \frac{\int_{V_i} \mathbf{r} \, \rho(\mathbf{r}) \, dA}{\int_{V_i} \rho(\mathbf{r}) \, dA}
$$

球面 Voronoi 面积元：$dA = \sin\theta \, d\theta \, d\phi$，总球面面积 $4\pi$。

### 3.2 Legendre 谱相位展开

二维相位剖面通过张量积 Legendre 多项式展开：

$$
\phi(x, y) = \sum_{m=0}^{M} \sum_{n=0}^{N} c_{mn} \, P_m\!\left(\frac{2x}{L_x}\right) P_n\!\left(\frac{2y}{L_y}\right)
$$

其中 $P_n(x)$ 满足递推关系：

$$
\begin{aligned}
P_0(x) &= 1, \quad P_1(x) = x \\
P_n(x) &= \frac{(2n-1)x P_{n-1}(x) - (n-1)P_{n-2}(x)}{n}
\end{aligned}
$$

导数递推：

$$
P'_n(x) = \frac{(2n-1)\bigl(P_{n-1}(x) + x P'_{n-1}(x)\bigr) - (n-1)P'_{n-2}(x)}{n}
$$

### 3.3 球谐函数与 Mie 散射

归一化球谐函数：

$$
Y_l^m(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}} \, P_l^m(\cos\theta) \, e^{im\phi}
$$

远场展开：

$$
E(\theta, \phi) = \sum_{l=0}^{L} \sum_{m=-l}^{l} a_{lm} \, Y_l^m(\theta, \phi)
$$

展开系数由正交性得到：

$$
a_{lm} = \int_0^{2\pi} \int_0^{\pi} E(\theta, \phi) \, Y_l^{m*}(\theta, \phi) \, \sin\theta \, d\theta \, d\phi
$$

小参数 ($ka \ll 1$) 下电偶极子 Mie 系数：

$$
a_1 \approx i \frac{2}{3} (ka)^3 \frac{\varepsilon_r - 1}{\varepsilon_r + 2}
$$

### 3.4 SVD 散射算符压缩

超表面局域透射系数：$t_i = A_i e^{i\phi_i}$。散射算符：

$$
S_{mn} = K(\mathbf{r}_m, \mathbf{r}_n) \, t_n
$$

其中 $K$ 为 sinc 型空间带宽限制核。SVD 分解：

$$
S = U \Sigma V^\dagger
$$

秩-$R$ 近似：

$$
S_R = \sum_{k=1}^{R} \sigma_k \, \mathbf{u}_k \mathbf{v}_k^\dagger
$$

相对 Frobenius 误差：

$$
\varepsilon(R) = \frac{\|S - S_R\|_F}{\|S\|_F} = \sqrt{\frac{\sum_{k>R} \sigma_k^2}{\sum_k \sigma_k^2}}
$$

压缩比：$C(R) = R(2N+1)/N^2$。

### 3.5 梯度有效折射率波传播

慢变包络近似下的标量传播方程：

$$
\frac{dE}{dz} = i k_0 n_{\text{eff}}(z) \, E(z)
$$

经典 RK4 格式：

$$
\begin{aligned}
k_1 &= f(z_n, E_n) \\
k_2 &= f\bigl(z_n + \tfrac{h}{2}, E_n + \tfrac{h}{2}k_1\bigr) \\
k_3 &= f\bigl(z_n + \tfrac{h}{2}, E_n + \tfrac{h}{2}k_2\bigr) \\
k_4 &= f(z_n + h, E_n + h k_3) \\
E_{n+1} &= E_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

RK23 的 2/3 阶误差估计：

$$
\begin{aligned}
y^{(2)} &= y_n + \frac{1}{2}(k_1 + k_2) \\
y^{(3)} &= y_n + \frac{1}{6}(k_1 + k_2 + 4k_3) \\
e_{n+1} &= |y^{(3)} - y^{(2)}|
\end{aligned}
$$

角谱传播（倏逝波截断）：

$$
k_z = \begin{cases}
\sqrt{k_0^2 - k_x^2 - k_y^2}, & k_x^2 + k_y^2 \le k_0^2 \\
i \sqrt{k_x^2 + k_y^2 - k_0^2}, & \text{otherwise}
\end{cases}
$$

### 3.6 非线性 Meta-Atom 响应

等效 Duffing 振子（Kerr 非线性）：

$$
\ddot{u} + \gamma \dot{u} + \omega_0^2 u + \beta u^3 = F_0 \cos(\omega t)
$$

非线性相位偏移（饱和模型）：

$$
\Delta\phi(I) = \arctan\!\left(\frac{\omega_0^2 - \omega^2}{\gamma \omega}\right) \cdot \frac{1}{\sqrt{1 + \kappa I / I_{\text{sat}}}}
$$

大角度摆周期（椭圆积分展开）：

$$
T = 4\sqrt{\frac{l}{g}} \, K\!\left(\sin^2\frac{\theta_0}{2}\right), \quad
K(k) = \frac{\pi}{2} \sum_{n=0}^{\infty} \left[\frac{(2n)!}{2^{2n}(n!)^2}\right]^2 k^n
$$

### 3.7 方形贴片矩量法

单位正方形 $[0,1]^2$ 上的单重积分：

$$
\int_0^1 \int_0^1 x^{e_1} y^{e_2} \, dx\, dy = \frac{1}{(e_1+1)(e_2+1)}
$$

对称正方形 $[-1,1]^2$ 上，若任一指数为奇数则积分为 0；否则：

$$
I = \frac{4}{(e_1+1)(e_2+1)}
$$

二维 Gauss-Legendre 求积：

$$
\int_{y_0}^{y_1} \int_{x_0}^{x_1} f(x,y) \, dx\, dy \approx
\frac{(x_1-x_0)(y_1-y_0)}{4} \sum_{i,j} w_i w_j \, f(\xi_i, \eta_j)
$$

### 3.8 Gerchberg-Saxton 全息优化

迭代傅里叶变换算法（IFTA）：

$$
\begin{aligned}
\text{空间域: } & t^{(k)}(x,y) = e^{i\phi^{(k)}(x,y)} \\
\text{频域: } & \tilde{E}^{(k)} = \mathcal{F}\{t^{(k)}\} \\
\text{幅度约束: } & \tilde{E}^{(k+1)} = A_{\text{target}} \cdot e^{i \arg(\tilde{E}^{(k)})} \\
\text{逆变换: } & t^{(k+1)} = \mathcal{F}^{-1}\{\tilde{E}^{(k+1)}\} \\
\text{更新相位: } & \phi^{(k+1)} = \arg(t^{(k+1)})
\end{aligned}
$$

## 四、文件结构与运行方式

```
098_synth_project/
├── main.py                      # 统一入口，零参数运行
├── cvt_placer.py                # CVT 布局优化（种子 1, 2）
├── phase_spectrum.py            # Legendre 谱相位设计（种子 3, 4）
├── spherical_expansion.py       # 球谐展开与 Mie 散射（种子 5）
├── scattering_operator.py       # SVD 散射算符（种子 6）
├── wave_propagation.py          # RK4/RK23 波传播（种子 7, 8）
├── nonlinear_resonator.py       # 非线性振子模型（种子 9, 10）
├── mesh_handler.py              # 网格 I/O 与质量评估（种子 11, 12）
├── geometry_io.py               # XYZ 点云 I/O（种子 13）
├── tolerance_analysis.py        # 蒙特卡洛公差分析（种子 14）
├── moment_integrals.py          # 方形贴片矩量（种子 15）
├── hologram_io.py               # 全息配置 I/O 与 GS 优化（种子 16）
└── README_博士级合成说明.md     # 本文档
```

### 运行命令

```bash
python main.py
```

程序无需任何命令行参数，内部自动完成从布局优化到全息优化的 11 个计算阶段，并输出统计结果。

## 五、修改说明与工程鲁棒性

1. **边界处理**：
   - `phase_spectrum.py` 中 Legendre 自变量通过 `np.clip` 截断到 $[-1,1]$，防止递推溢出。
   - `wave_propagation.py` 中折射率实部强制 $\ge 1$，虚部强制 $\le 0$（无增益）。
   - `nonlinear_resonator.py` 中 Duffing 加速度限制在 $10^6$ 以内，防止数值爆炸。
   - `hologram_io.py` 中相位通过 `np.mod` 归一化到 $[0, 2\pi)$，避免量化歧义。

2. **数值鲁棒性**：
   - 最小二乘拟合引入 Tikhonov 正则化（$\lambda = 10^{-10}$）。
   - SVD 压缩中 Frobenius 范数分母加入 $10^{-15}$ 保护。
   - 球面 CVT 空区域检测：若某生成元无样本，保持原位不动。
   - 高斯-勒让德积分对非有限值返回 0。

3. **零可视化**：所有原种子项目中的 `imshow`、`plot`、`figure` 等可视化代码已全部移除，仅保留 `print` 文本输出与文件 I/O。

4. **零外部依赖**：除 Python 标准库外，仅依赖 `numpy`，无需 `scipy`、`matplotlib`、`netCDF4` 等额外包。

## 六、合成项目解决的前沿科学问题

本项目构建了一个**可运行的介电超表面全息计算框架**，能够解决以下博士级科学问题：

1. **非周期超表面布局设计**：通过电磁加权 CVT 抑制高阶衍射光栅瓣，突破传统周期阵列的衍射极限。
2. **低阶谱相位编码**：利用正交多项式基将高维相位设计问题转化为低维系数优化，降低计算复杂度。
3. **远场全空间全息表征**：通过球谐函数展开在完整 $4\pi$ 球面上描述散射场，适用于三维全息显示。
4. **散射算符压缩与加速**：SVD 低秩近似将超表面-入射场相互作用从 $O(N^2)$ 降至 $O(NR)$，为大规模逆设计提供可能。
5. **非线性光学响应预测**：基于 Duffing/分段振子模型量化 Kerr 非线性下的相位饱和与双稳态行为。
6. **制造公差与良率评估**：蒙特卡洛模拟为电子束光刻、反应离子刻蚀等工艺提供定量容差预算。
7. **全息相位快速优化**：Gerchberg-Saxton 迭代算法实现目标远场幅度到空间相位剖面的直接映射。

## 七、参考文献与算法来源

- Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Trans. Inf. Theory*.
- Du, Q., Faber, V., & Gunzburger, M. (1999). Centroidal Voronoi tessellations. *SIAM Review*.
- Genevieve, J. (2013). Metasurface holograms. *Nature Nanotechnology*.
- Gerchberg, R. W., & Saxton, W. O. (1972). Practical algorithm for phase retrieval. *Optik*.
- Abramowitz, M., & Stegun, I. A. (1964). *Handbook of Mathematical Functions*.
- Mie, G. (1908). Beiträge zur Optik trüber Medien. *Annalen der Physik*.
- Goodman, J. W. (2005). *Introduction to Fourier Optics*.
