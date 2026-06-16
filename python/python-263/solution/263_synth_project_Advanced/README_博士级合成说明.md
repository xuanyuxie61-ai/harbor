# PROJECT 263 — 计算太阳物理：日冕加热与太阳风加速

## 项目概述

本项目是一个面向前沿科学问题的博士级计算太阳物理研究平台，聚焦**日冕加热机制**与**太阳风加速过程**，采用**高阶有限差分方法**并系统开展**数值稳定性分析**。整个项目从小规模可复现实验出发，融合了 15 个种子项目的核心算法，构建了一个从光球到太阳风的多尺度耦合计算框架。

---

## 一、原项目到科学问题的映射

| 编号 | 种子项目 | 核心算法/思想 | 在本项目中的科学角色 |
|------|---------|-------------|-------------------|
| 01 | `088_biharmonic_fd1d` | 一维双调和算子有限差分 | 日冕磁场高阶扩散项 $\nabla^4 B$ (磁重联电阻扩散的高阶修正) |
| 02 | `246_cvt_1d_sampling` | CVT 采样-Lloyd 算法 | 日冕环自适应网格生成 (加热率加权 Voronoi 剖分) |
| 03 | `113_box_distance` | 距离矩阵计算 | 磁场拓扑距离矩阵 (场线连接性, QSL 识别) |
| 04 | `511_heartbeat_ode` | 弛豫振荡 FitzHugh-Nagumo | 纳耀斑 (nanoflare) 能量积累-释放弛豫振荡 |
| 05 | `1004_RehMoritz_vmc_pde` | 变分蒙特卡罗 PDE | 理想 MHD 变分能量原理 $\delta^2 W$ |
| 06 | `1225_Okita0512_VSC_HEOM` | 层级化开放系统 HEOM | 光球-色球-日冕-太阳风四层 HEOM 耦合 |
| 07 | `760_mgmres` | 重启 GMRES 迭代法 | 隐式热传导大型稀疏线性系统求解 |
| 08 | `454_gaussian` | 高斯求积 / Hermite 基 | Parker 加热剖面沿环轴高斯积分 |
| 09 | `375_fem_basis_t6_display` | 6 节点高阶 Lagrange 基 | WKB 修正基函数 (Alfvén 波本征模展开) |
| 10 | `583_image_quantization` | 图像灰度量化 | 加热率场离散化 (识别加热平台与梯度前沿) |
| 11 | `061_b1g3` | 带状矩阵存储/求解 | MHD 算子的带状结构利用 |
| 12 | `350_fd_predator_prey` | 有限差分时步推进 | Parker 太阳风方程 FD 积分 |
| 13 | `984_r8lt` | 下三角矩阵操作 (LU) | 隐式步 ILU 预处理 + 前代求解 |
| 14 | `440_florida_cvt_pop` | 人口加权 CVT | 密度加权网格自适应 (太阳风跨声速加密) |
| 15 | `1083_LinBoNUS_DLCommittor` | Committor 概率 | 跨声速逃逸概率 (Parker 相空间向后方程) |

---

## 二、新增数学物理模型与核心公式

### 2.1 等离子体基础物理

$$
\omega_{pe} = \sqrt{\frac{n_e e^2}{\epsilon_0 m_e}}, \qquad
d_i = \frac{c}{\omega_{pi}}, \qquad
\beta = \frac{2\mu_0 n k_B T}{B^2}
$$

$$
v_A = \frac{B}{\sqrt{\mu_0 \rho}}, \qquad
c_s = \sqrt{\frac{\gamma k_B T}{m_i}}
$$

### 2.2 Parker 太阳风方程

$$
(v^2 - c_s^2)\frac{1}{v}\frac{dv}{dr} = \frac{2c_s^2}{r} - \frac{GM_\odot}{r^2}
$$

跨声速临界点:

$$
r_c = \frac{GM_\odot}{2 c_s^2}, \quad v(r_c) = c_s
$$

### 2.3 纳耀斑弛豫振荡 (FitzHugh-Nagumo 推广)

$$
\frac{dE_{\text{mag}}}{dt} = -\frac{1}{\varepsilon}(E_{\text{mag}}^3 - a E_{\text{mag}} + W) + S_{\text{ext}}
$$

$$
\frac{dW}{dt} = E_{\text{mag}} - \gamma W
$$

能量释放率:

$$
\frac{dQ}{dt} = \frac{1}{\varepsilon}(E^3 - aE + W)
$$

功率律分布: $\frac{dN}{dE} \propto E^{-\alpha}$，Parker 判据 $\alpha > 2$。

### 2.4 变分能量原理 (Bernstein et al. 1958)

$$
\delta^2 W = \frac{1}{2}\int \left[
\frac{|\nabla\times(\boldsymbol{\xi}\times\mathbf{B})|^2}{\mu_0}
+ \gamma p |\nabla\cdot\boldsymbol{\xi}|^2
+ \rho|\mathbf{g}\cdot\boldsymbol{\xi}|^2
\right] dV
$$

Kruskal-Shafranov 安全因子:

$$
q = \frac{r B_z}{R_0 B_\theta} > 1 \quad \text{(kink 稳定)}
$$

Torus 不稳定性判据:

$$
n = -\frac{d\ln B_{\text{ext}}}{d\ln h} > n_{\text{crit}} \sim 1.5
$$

### 2.5 CVT 自适应网格 (Lloyd 算法)

$$
\rho(s) \propto |T''(s)|^{1/3}
$$

迭代更新:

$$
z_i^{(k+1)} = \frac{\int_{V_i} \rho(x) x\, dx}{\int_{V_i} \rho(x)\, dx}
$$

### 2.6 Committor 逃逸概率

向后 Kolmogorov 方程:

$$
\frac{1}{2}\sigma^2 \Delta q + \mathbf{b}\cdot\nabla q = 0
$$

边界: $q=0$ on $v = 0.5 c_s$, $q=1$ on $v = 2 c_s$。

### 2.7 Spitzer 热传导

$$
\kappa_\parallel = \kappa_0 T^{5/2}, \quad \kappa_0 \approx 1.84\times 10^{-5} \text{ SI}
$$

### 2.8 RTV 辐射冷却

$$
Q_{\text{rad}} = \chi n^2 T^\alpha, \quad \chi \sim 10^{-32}, \alpha = -0.5
$$

### 2.9 WKB 修正基

$$
\phi_j(s) = L_j(\xi(s)) \cos\left(\int^s \frac{\omega}{v_A(s')} ds'\right)
$$

### 2.10 von Neumann 稳定性

四阶中心差分 + RK4 放大因子:

$$
G(z) = 1 + z + \frac{z^2}{2} + \frac{z^3}{6} + \frac{z^4}{24}
$$

CFL 条件:

$$
\Delta t \le C_{\text{cfl}} \min\left(\frac{h}{|v|+v_A}, \frac{h^2}{2\kappa}\right)
$$

---

## 三、修改文件与合成路径

| 文件 | 融合来源 | 核心功能 |
|------|---------|---------|
| `main.py` | 统一入口 (11 阶段流程) | 零参数运行完整计算链 |
| `solar_constants.py` | `511_heartbeat_parameters`, `454_gaussian` | SI 物理常量 + 归一化单位制 |
| `coronal_grid.py` | `246_cvt_1d_sampling`, `440_florida_cvt_pop` | CVT 自适应网格 (日冕环 + 太阳风) |
| `fd_operators.py` | `088_biharmonic_fd1d`, `061_b1g3`, `350_fd_predator_prey` | 高阶 FD 算子 + 双调和 + 带状 + RK4 |
| `nanoflare_oscillator.py` | `511_heartbeat_ode` | 纳耀斑弛豫振荡 + 事件检测 + 幂律拟合 |
| `parker_wind.py` | `350_fd_predator_prey`, `1083_DLCommittor` | Parker 跨声速解 + committor 逃逸概率 |
| `coronal_energy.py` | `1004_RehMoritz_vmc_pde` | 变分能量原理 + kink/torus 稳定性 |
| `heom_hierarchy.py` | `1225_Okita0512_VSC_HEOM` | 四层大气 HEOM 耦合演化 |
| `coronal_solver.py` | `760_mgmres`, `984_r8lt` | 重启 GMRES + R8LT 三角求解 + ILU 预处理 |
| `coronal_basis.py` | `375_fem_basis_t6_display` | 6 节点 Lagrange + WKB 修正基 |
| `topology_tools.py` | `113_box_distance`, `583_image_quantization` | 磁场拓扑距离 + 加热率量化 |
| `stability_analysis.py` | 新增 | von Neumann + CFL + 谱半径 + 修正波数分析 |

---

## 四、可解决的科学问题

1. **日冕加热机制诊断**：通过纳耀斑弛豫振荡 + 幂律指数 $\alpha$ 判断 Parker 纳耀斑假说是否成立。
2. **太阳风跨声速转变**：Parker 四类解的数值构造 + committor 概率定量评估逃逸概率。
3. **MHD 本征稳定性**：变分能量原理判定 kink / torus / Rayleigh-Taylor 不稳定性。
4. **自适应网格优化**：CVT 方法在加热率梯度/跨声速点附近的最优节点分配。
5. **隐式热传导**：Spitzer 强各向异性热传导的 IMEX 分裂 + GMRES 高效求解。
6. **多层大气耦合**：光球-色球-日冕-太阳风 Poynting/质量/波通量守恒。
7. **磁场拓扑识别**：准分隔层 (QSL) 通过场线连接距离矩阵定位。
8. **数值稳定性诊断**：von Neumann 放大因子、CFL 数、修正波数色散分析。

---

## 五、运行方法

### 5.1 依赖

- Python >= 3.9
- NumPy >= 1.20
- SciPy >= 1.7

### 5.2 零参数运行

```bash
cd 263_synth_project_Advanced
python main.py
```

### 5.3 输出

程序分 11 个阶段打印诊断信息：

1. 自适应网格生成
2. 初始等离子体状态
3. 纳耀斑振荡
4. Parker 太阳风
5. 变分能量原理
6. HEOM 多层耦合
7. 隐式热传导
8. WKB 模态分解
9. 磁场拓扑
10. 稳定性分析
11. 高阶 FD 算子 + 三角求解

最终输出包含关键物理量：
- 跨声速点位置 $r_c$
- 纳耀斑幂律指数 $\alpha$
- Kruskal-Shafranov 安全因子 $q$
- CFL 稳定性判定
- von Neumann 放大因子上界

---

## 六、工程特色

- **边界处理**：所有差分算子在非均匀网格上正确处理；Dirichlet/Neumann 混合边界。
- **数值鲁棒性**：所有分母加 `1e-30` 小量防除零；物理量（温度、密度、速度）强制非负约束。
- **可复现**：固定随机种子 263；所有模块纯函数化。
- **无可视化**：仅文本诊断输出，符合科学计算工程规范。
- **博士级深度**：融合变分原理、跨声速理论、弛豫振荡、HEOM 层级、WKB 修正基等前沿内容。

---

## 七、科学参考文献

1. Parker, E.N. (1958). "Dynamics of the interplanetary gas and magnetic fields." *ApJ* 128, 664.
2. Rosner, R., Tucker, W.H., Vaiana, G.S. (1978). "Dynamics of the quiescent solar corona." *ApJ* 220, 643.
3. Parker, E.N. (1988). "Nanoflares and the solar X-ray corona." *ApJ* 330, 474.
4. Bernstein, I.B. et al. (1958). "An energy principle for hydromagnetic stability." *Proc. R. Soc. A* 244, 17.
5. Kruskal, M.D., Kulsrud, R.M. (1958). "Equilibrium of a magnetically confined plasma." *Phys. Fluids* 1, 265.
6. Kliem, B., Torok, T. (2006). "Torus instability." *Phys. Rev. Lett.* 96, 255002.
7. Spitzer, L. (1962). *Physics of Fully Ionized Gases*. Interscience.
8. Du, Q., Faber, V., Gunzburger, M. (1999). "Centroidal Voronoi tessellations." *SIAM Rev.* 41, 637.
9. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*. SIAM.
10. Aschwanden, M.J. (2004). *Physics of the Solar Corona*. Springer.
