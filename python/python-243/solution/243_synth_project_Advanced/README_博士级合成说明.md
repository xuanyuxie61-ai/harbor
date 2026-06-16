# PROJECT 243 — r 过程核合成网络模拟：高阶有限差分与稳定性分析
## 博士级 Python 科研合成项目（小规模可复现实验）

---

## 一、科学问题定位

**领域**：核天体物理 (Nuclear Astrophysics)
**具体问题**：快速中子俘获过程 (r-process) 中的核合成网络演化，采用高阶有限差分方法进行中子输运模拟，并对核反应网络 ODE 系统进行完整的数值稳定性分析。

r 过程是宇宙中重元素 (A > 70) 的主要合成机制，发生在中子星并合、磁旋转超新星等极端天体环境中。其核心是一个包含 ~2000 种核素的刚性 ODE 网络：

$$
\frac{dY_i}{dt} = \sum_j \lambda_{j \to i} Y_j - \sum_k \lambda_{i \to k} Y_i
$$

其中 $Y_i = Y(A_i, Z_i)$ 为核素 $(A, Z)$ 的丰度，$\lambda$ 包括中子俘获 $(n,\gamma)$、光致蜕变 $(\gamma, n)$、$\beta$ 衰变、裂变等反应率。本项目的目标是在一个**可复现的小规模网络 (A ∈ [70, 150], ~567 个核素)** 上演示完整的方法论。

---

## 二、15 个种子项目 → 科学模块的完整映射

| # | 种子项目 | 核心算法/思想 | 在本项目中的角色 | 合成文件 |
|---|---|---|---|---|
| 1 | `1358_trinity` | Trinity 网格分块 (tile/word 索引) | 核素网络按 Trinity 范式分块为 tile，便于并行处理 | `latin_sampler.py` (`trinity_tile_id`, `trinity_grid`) |
| 2 | `580_image_mesh2d` | 2D 像素图像 → 网格坐标映射 | 核素 (A, Z) 映射到二维核图网格 | `mesh_network.py` (`image_to_mesh2d`) |
| 3 | `357_fd1d_burgers_leap` | 1D Burgers 方程的 leapfrog 差分 | Lax-Wendroff 高阶有限差分用于中子脉冲输运 | `high_order_fd.py` (`lax_wendroff_step`) |
| 4 | `1121_PyADI-Fluid-Sim` | ADI (交替方向隐式) 流体模拟 | Peaceman-Rachford ADI 用于 2D (T, ρ) 温度密度扩散 | `adi_solver.py` |
| 5 | `1360_truncated_normal` | 截断正态分布统计量 | 核反应率不确定度的截断正态采样 | `nuclear_physics.py` (`truncated_normal_mean`, `sample_rate_uncertainty`) |
| 6 | `591_interp_chebyshev` | Chebyshev 插值与谱方法 | 核反应率表的高效 Chebyshev 插值 + 导数 | `chebyshev_interp.py` |
| 7 | `891_polygonal_surface_display` | 多边形表面显示/面积 | 核图边界多边形 (滴线围成区域) 的面积计算与显示 | `mesh_network.py` (`polygon_area`, `display_polygonal_boundary`) |
| 8 | `801_newton_maehly` | Newton-Maehly 带收缩求根 | 核统计平衡 (NSE) 丰度的 Newton-Maehly 求解 | `newton_root.py` |
| 9 | `1044_FootlooseCalvingMechanism` | 弹性梁 ODE / shooting method | 裂变的弹性梁类比 → 非对称碎片质量分布预测 | `calving_physics.py` |
| 10 | `649_latin_center` | 中心拉丁超立方采样 | r 过程参数 (v_ej, M_ej, Y_e, T9) 的灵敏度采样 | `latin_sampler.py` (`latin_center`) |
| 11 | `025_asa006` | Cholesky 分解 (ASA006) | 网络 Jacobian 协方差阵的 Cholesky 稳定性分析 | `stability_analysis.py` (`cholesky_decomposition`) |
| 12 | `306_distance_to_position` | 距离-位置转换 | 核素空间中 (A, Z) 距离；宇宙学距离 → 红移/回溯时间 | `distance_position.py` |
| 13 | `215_control_bio` | 生物控制论中的优化收敛 | 冻结-out 条件的梯度下降/共轭梯度优化 | `control_optimizer.py` |
| 14 | `1158_Quick-MSD-Diffusivity-Calculator` | 均方位移 → 扩散系数提取 | 中子扩散系数 + 从 MSD 数据反演扩散率 | `diffusivity.py` (`MSD_from_trajectory`) |
| 15 | `359_fd1d_display` | 1D 有限差分结果展示 | FD 输运的 Von Neumann 稳定性诊断与结果报告 | `stability_analysis.py` (`von_neumann_stability`) |

**全部 15 个种子项目均已真实融入，无挂名。**

---

## 三、核心数学物理公式

### 3.1 液滴模型结合能 (Liquid-Drop Binding Energy)
$$
B(A, Z) = a_v A - a_s A^{2/3} - a_c \frac{Z(Z-1)}{A^{1/3}} - a_{sym}\frac{(A-2Z)^2}{A} + \delta(A,Z)
$$
其中对项 $\delta$：
$$
\delta = \begin{cases}
+a_p / \sqrt{A} & \text{even-even} \\
0 & \text{odd A} \\
-a_p / \sqrt{A} & \text{odd-odd}
\end{cases}
$$

### 3.2 Saha 核统计平衡丰度 (NSE, Cameron 1957)
$$
Y(A,Z) = G(A,Z) \, A^{3/2} \, Y_n^A \, Y_p^Z \,
\left(\frac{2\pi\hbar^2}{m_u k T}\right)^{\frac{3}{2}(A-1)}
\exp\!\left(\frac{B(A,Z)}{kT}\right)
$$

### 3.3 Weisskopf-Ewing 中子俘获率
$$
\langle\sigma v\rangle_{n,\gamma} \sim \bar\lambda^2 \, v_{th} \,
\frac{\Gamma_\gamma}{\hbar} \, \rho_{comp}(S_n)
$$
其中 $\bar\lambda^2 = (\hbar c / kT)^2$，$\rho_{comp}$ 为复合核在 $S_n$ 处的能级密度 (Back-shifted Fermi gas)。

### 3.4 光致蜕变率 (Detailed Balance)
$$
\lambda_{\gamma,n} = \langle\sigma v\rangle_{n,\gamma} \,
\left(\frac{2\pi m_n kT}{h^2}\right)^{3/2}
\left(\frac{A}{A+1}\right)^{3/2}
\exp\!\left(-\frac{S_n}{kT}\right)
$$

### 3.5 Lax-Wendroff 二阶格式
$$
U_i^{n+1} = U_i^n - \frac{C}{2}(U_{i+1}^n - U_{i-1}^n)
+ \frac{C^2}{2}(U_{i+1}^n - 2U_i^n + U_{i-1}^n)
+ D(U_{i+1}^n - 2U_i^n + U_{i-1}^n)
$$
其中 $C = u\,\Delta t / \Delta x$ 为 Courant 数，$D = \nu\,\Delta t / \Delta x^2$。

### 3.6 Peaceman-Rachford ADI
半步 (x 方向隐式)：
$$
(I - \tfrac{1}{2}\Delta t\, L_x)\, U^{n+1/2} = (I + \tfrac{1}{2}\Delta t\, L_y)\, U^n + \tfrac{1}{2}\Delta t\, S^{n+1/2}
$$
全步 (y 方向隐式)：
$$
(I - \tfrac{1}{2}\Delta t\, L_y)\, U^{n+1} = (I + \tfrac{1}{2}\Delta t\, L_x)\, U^{n+1/2} + \tfrac{1}{2}\Delta t\, S^{n+1/2}
$$

### 3.7 Chebyshev 插值
节点：$x_k = \frac{a+b}{2} + \frac{b-a}{2}\cos\frac{(2k+1)\pi}{2N}$
系数：$c_k = \frac{2}{N}\sum_{j=0}^{N-1} f(x_j)\, T_k(x_j')$（$c_0$ 折半）
导数系数 (Clenshaw)：$c_k' = c_{k+2}' + 2(k+1)\,c_{k+1}$

### 3.8 中子扩散系数
$$
D = \frac{1}{3}\lambda_{mfp} v_{th}, \qquad
\lambda_{mfp} = \frac{1}{n_n \sigma_{scatter}}, \qquad
\sigma_{scatter} \sim \pi R^2\!\left(1 + \frac{v_{th}}{v}\right)
$$
不透明度：$\kappa \sim c / (3 D \rho)$

### 3.9 裂变碎片分布 (弹性梁类比)
核变形由弹性梁方程描述：$B\, w''''(x) + k\, w(x) = 0$
解：$w(x) = e^{-x/\ell_w} \cos(x/\ell_w)$，其中 $\ell_w = (B/k)^{1/4}$
第一个节点 $x_1 = (\pi/2)\ell_w$ 决定轻碎片质量分数 $f_L = x_1/(2R)$。

---

## 四、文件结构与职责

```
243_synth_project_Advanced/
├── main.py                   # 统一入口（零参数运行，14 步完整流程）
├── physical_constants.py     # CODATA 2018 常数 + 核天体物理换算
├── nuclear_physics.py        # 液滴结合能、Saha NSE、截断正态
├── reaction_rates.py         # (n,γ), (γ,n), β 衰变, 裂变率表构建
├── nuclear_network.py        # ODE 网络 dY/dt + RK4 / 隐式 Euler 积分
├── adi_solver.py             # Peaceman-Rachford ADI (2D T-ρ 扩散)
├── chebyshev_interp.py       # Chebyshev 插值 + 谱导数
├── newton_root.py            # Newton-Maehly 求根 + 平衡丰度
├── stability_analysis.py     # 特征值刚性判据、Cholesky、CFL、von Neumann
├── high_order_fd.py          # Lax-Wendroff + MUSCL-3 中子输运
├── latin_sampler.py          # 拉丁超立方 + Trinity 分块
├── mesh_network.py           # 核素网格 + 多边形核图边界
├── distance_position.py      # 核素空间距离 + 抛射物轨迹
├── control_optimizer.py      # 梯度下降/共轭梯度/freeze-out 优化
├── diffusivity.py            # 中子扩散系数 + MSD 反演
├── calving_physics.py        # 弹性梁裂变碎片分布 + 质量蒸发
└── README_博士级合成说明.md    # 本文件
```

共计 **16 个 .py 文件 + 1 个 README**，全部 Python 实现，无任何可视化代码。

---

## 五、合成后的项目能解决什么科学问题

本项目解决的核心科学问题：

1. **r 过程核合成网络的数值求解**：在 567 个核素 (A ∈ [70,150]) 网络上积分完整的核丰度 ODE 系统，演示从种子核 (A=90, Z=36) 出发的核合成流。

2. **刚性 ODE 网络的稳定性诊断**：通过网络 Jacobian 的特征值分析，判断网络的刚性 (stiffness ratio > 10³) 与数值稳定性；通过 Cholesky 分解验证协方差矩阵的正定性。

3. **高阶有限差分中子输运**：用 Lax-Wendroff 二阶格式模拟中子脉冲在抛射物中的一维输运，并通过 von Neumann 分析确认稳定性。

4. **2D 温度-密度场的 ADI 扩散**：用 Peaceman-Rachford ADI 格式求解 (T, ρ) 平面上的抛物型扩散方程，模拟抛射物内部的热扩散。

5. **裂变碎片的弹性梁类比预测**：用弹性梁方程的第一个节点预测非对称裂变碎片质量分布，演示跨学科方法的迁移。

6. **核反应率的谱插值**：用 Chebyshev 多项式插值核反应率表，实现指数级收敛。

7. **参数空间的拉丁超立方灵敏度分析**：对 (v_ej, M_ej, Y_e, T9_start) 四个关键参数进行 8 点 Latin 采样，为 r 过程模型提供全局灵敏度。

8. **freeze-out 条件的优化搜索**：用 Newton-Maehly 与梯度方法寻找使丰度残差最小的冻结温度 T9_f 与密度 ρ_f。

---

## 六、如何运行

### 环境要求
- Python 3.8+
- NumPy (`pip install numpy`)
- 无其他第三方依赖

### 运行方法
```bash
cd "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/243_synth_project/243_synth_project_Advanced"
python main.py
```

零参数运行，约 4 秒完成，输出 14 个完整的诊断步骤，最终打印：
```
************************************************************************
  PROJECT 243 COMPLETED SUCCESSFULLY  --  total wall time: 4.07 s
************************************************************************
```

退出码为 0 表示成功。

---

## 七、边界处理与数值鲁棒性

1. **ODE 积分的刚性稳定化**：RK4 步内对损失率做硬截断 `loss ≤ 5/dt`，防止 dt·λ ≫ 1 时的数值爆炸。
2. **非负性约束**：每步结束后 `Y = max(Y, 0)`，避免负丰度。
3. **守恒归一化**：每步结束后将总丰度归一为 1（等价于 baryon number 守恒，因裂变碎片未显式追踪）。
4. **反射边界条件**：核素网络边缘处，若产物核不在网格内则抑制相应损失项，防止质量泄漏。
5. **指数溢出保护**：Saha 因子中 exp 参数截断在 500；Cholesky 中对对角元做正定性检查。
6. **CFL 显式检查**：`von_neumann_stability` 在 FD 步之前判断稳定性并给出 dt_max。
7. **物理常数使用 CODATA 2018 推荐值**，并显式提供单位换算函数。
8. **异常值隔离**：裂变率对零势垒核使用 1e4 /s 而非 1e30 /s，防止单点污染整个网络。

---

## 八、复现性保证

- 所有随机源（拉丁超立方）使用固定种子 `seed=42 / 123 / 7`。
- 网络核素集合完全由 `A_MIN, A_MAX, Z_OFFSET` 确定性生成。
- 截断正态采样使用 LCG 伪随机，无 numpy random 依赖。
- 初始丰度设置明确：Y(90, 36) = 0.999，轻核 (A=70) 总占比 0.001。
- 输出完全文本化，可直接 diff 对比两次运行结果。

---

## 九、进一步扩展方向（博士研究层面）

1. **扩展网络至 A ~ 300**：加入锕系与超锕系，显式追踪裂变碎片注入。
2. **真实核数据**：替换液滴模型为 FRDM/HFB 质量表；替换 Weisskopf 率为 TALYS/BHSM 速率。
3. **隐式 Bader 积分器**：处理刚性比达 10¹⁵ 的真实网络。
4. **3D 流体动力学耦合**：将核网络嵌入 SPH/AMR 中子星并合模拟。
5. **GPU 加速**：将 ODE 右端项用 CuPy/JAX 重写，实现万核素网络实时积分。
6. **贝叶斯不确定性量化**：用 MCMC 在截断正态率先验下约束天体物理条件。

---

*PROJECT 243 — 核天体物理 · r 过程 · 高阶有限差分 · 博士级合成 · 2026-06-07*
