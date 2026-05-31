# 滴线核性质的多体关联与衰变动力学 — 博士级合成说明

## 1. 项目概述

本项目围绕**核物理：放射性束流与滴线核性质**，将 15 个输入种子项目的核心算法融合为一个面向前沿科学问题的博士级计算框架。目标核素选取为极端中子富集的滴线候选核 `^28O`（Z=8, N=20），开展从**自洽平均场**、**对关联**、**衰变统计**、**反应截面**到**随机动力学**的全链条计算。

---

## 2. 原项目到科学问题的映射

| 原项目 | 核心算法/思想 | 合成后承担的科学角色 |
|---|---|---|
| `082_beta_nc` | 非中心 Beta 分布 CDF 级数展开 | 衰变分支比贝叶斯不确定性推断 |
| `299_disk01_integrals` | 单位圆盘单项式解析积分 | 角动量耦合权重与反应相空间积分 |
| `1303_triangle_fekete_rule` | 三角形 Fekete 高精度求积 | 核形变参数空间 `(β₂,β₃)` 的概率积分 |
| `981_r8ge` | 一般矩阵共轭梯度法 (CG) | HFB 自洽迭代中的密度矩阵线性求解加速 |
| `1377_usa_box_plot` | 矩形网格离散化 / 区域填充 | 核素图稳定性网格的离散映射概念 |
| `950_quadrature_weights_vandermonde` | Vandermonde 矩阵求积分权重 | 径向薛定谔方程数值求解中的权重生成（保留为 API） |
| `1103_sparse_grid_cc` | Smolyak 稀疏网格 Clenshaw-Curtis | 高维核力参数不确定性量化传播 |
| `291_discrete_pdf_sample_2d` | 离散 CDF 逆变换采样 | 统计衰变链蒙特卡洛模拟 |
| `715_mario` | 像素/网格结构化映射 | 三维核密度分布的网格离散化思想 |
| `399_fem1d_spectral_numeric` | 一维谱有限元求解微分方程 | 径向薛定谔方程的谱-有限差分混合求解器 |
| `856_peaks_movie` | 多峰高斯叠加函数 | 核势多分量（中心+自旋轨道+库仑）叠加模型 |
| `119_brownian_motion_simulation` | 布朗运动随机游走 | 核子在热化平均场中的 Langevin 随机输运 |
| `1352_triangulation_svg` | 三角剖分拓扑结构 | 二十面体球面三角剖分 → 四面体网格生成 |
| `1214_test_interp_nd` | N 维 Genz 测试函数族 | 核质量面的多维径向基插值 |
| `1241_tet_mesh_to_xml` | 四面体网格到 XML 拓扑转换 | 三维核密度四面体网格输出与后处理 |

---

## 3. 新增数学物理模型与核心公式

### 3.1 形变核平均场势
Woods-Saxon 中心势、自旋-轨道耦合与库仑势的叠加：

$$
V(\mathbf{r}) = V_{\text{WS}}(r_{\text{eq}})
+ V_{\text{so}}(r)\,\hat{\mathbf{l}}\!\cdot\!\hat{\mathbf{s}}
+ V_{\text{Coul}}(r)
$$

其中形变半径：

$$
R(\theta) = R_0\Bigl[1 + \beta_2 Y_{20}(\theta)
+ \beta_3 Y_{30}(\theta) + \beta_4 Y_{40}(\theta)\Bigr]
$$

### 3.2 径向薛定谔方程

$$
-\frac{\hbar^2}{2m}\frac{d^2 u}{dr^2}
+ \left[V(r) + \frac{\hbar^2}{2m}\frac{\ell(\ell+1)}{r^2}\right] u
= E\,u
$$

采用**对称三对角有限差分**离散化，数值稳定且可处理中心奇点。

### 3.3 Hartree-Fock-Bogoliubov (BCS 近似)

准粒子哈密顿量：

$$
\mathcal{H} = \begin{pmatrix}
h - \lambda & \Delta \\
\Delta & -(h - \lambda)
\end{pmatrix}
$$

BCS 能隙方程与粒子数约束：

$$
\Delta = G\sum_{k>0} u_k v_k, \qquad
N = 2\sum_{k>0} v_k^2, \qquad
E_k = \sqrt{(\epsilon_k - \lambda)^2 + \Delta^2}
$$

### 3.4 液滴模型与壳修正

结合能的 Bethe-Weizsäcker 分解：

$$
B(Z,A) = a_v A - a_s A^{2/3} - a_c \frac{Z^2}{A^{1/3}}
- a_a \frac{(N-Z)^2}{A} + \delta_{\text{pair}} A^{-1/2}
+ E_{\text{shell}}(Z,N)
$$

### 3.5 β 衰变与不确定性

Q 值与半衰期：

$$
Q_\beta = [M(Z,A) - M(Z+1,A)]c^2, \qquad
T_{1/2} = \frac{D}{f(Z,Q_\beta)\,B_{\text{gt}}}
$$

非中心 Beta 后验分布（Posten 1993 级数）：

$$
F(x;a,b,\lambda) = \sum_{i=0}^{\infty} p_i(\lambda)\,I_x(a+i,b)
$$

### 3.6 随机 Langevin 动力学

过阻尼方程：

$$
\gamma\,\frac{d\mathbf{r}}{dt}
= -\nabla V(\mathbf{r}) + \boldsymbol{\xi}(t),
\qquad
\langle \xi_i(t)\xi_j(t')\rangle
= 2\gamma k_B T\,\delta_{ij}\delta(t-t')
$$

### 3.7 反应相空间

转移反应截面（半经典）：

$$
\sigma_{\text{tr}} = 2\pi\int_{b_{\min}}^{\infty}
b\,P_{\text{tr}}(b)\,db
$$

Coulomb 碎裂（等效光子法）：

$$
\sigma_{\text{CU}} = \int_{\omega_{\min}}^{\omega_{\max}}
n_{\gamma}(\omega)\,\sigma_{\gamma}(\omega)\,d\omega
$$

### 3.8 高维积分工具

- **Fekete 三角形求积**：在形变参数空间实现多项式精确积分。
- **稀疏网格 Clenshaw-Curtis**：Smolyak 构造实现 2–6 维参数不确定性传播。

---

## 4. 文件架构

```
030_synth_project/
├── main.py                     # 统一零参数入口
├── constants.py                # 核物理常数与单位换算
├── nuclear_potential.py        # 形变 Woods-Saxon + SO + Coulomb 势
├── radial_solver.py            # 径向薛定谔方程有限差分求解器
├── hfb_selfconsistent.py       # HFB-BCS 自洽场 + CG 加速
├── quadrature_engine.py        # Fekete 三角形 + 稀疏网格 CC
├── decay_statistics.py         # β 衰变、非中心 Beta、衰变链 MC
├── stochastic_dynamics.py      # Langevin 核子随机输运
├── mass_surface.py             # 核质量面 LDM + RBF 插值
├── density_mesh.py             # 四面体网格密度与 RMS 半径
├── reaction_phasespace.py      # 反应截面与圆盘积分
└── README_博士级合成说明.md    # 本文档
```

共 **11 个 `.py` 文件**（含 `main.py`），满足 ≥8 个的要求。

---

## 5. 运行方式

```bash
cd Synthesis-project-python/030_synth_project
python main.py
```

无需任何命令行参数。程序依次执行 10 个计算阶段，所有结果以文本形式输出到标准输出。

---

## 6. 边界处理与数值鲁棒性

1. **Woods-Saxon 指数截断**：`arg = clip((r-R)/a, -500, 500)`，防止溢出。
2. **自旋-轨道核心正则化**：在 `r < 0.2 fm` 区域强制 `V_so = 0`，消除 `1/r` 奇点。
3. **BCS 占据数裁剪**：`u², v²` 被裁剪到 `[0,1]`，避免数值溢出。
4. **质量矩阵 Cholesky 失败回退**：在 HFB 中若 `B` 不正定，自动回退到伪逆求解。
5. **特征值排序与边界态筛选**：仅返回 `E < 0` 的束缚态；若无束缚态则返回最低连续态。
6. **有限差分稳定性**：径向求解器采用对称三对角离散化，条件数远优于全局谱微分矩阵。
7. **四面体体积正则化**：`abs(det(J))/6`，确保非负体积。

---

## 7. 合成后的项目能够解决的科学问题

- **滴线核的单粒子结构**：计算远离稳定线核素的能级、占据数与壳修正。
- **对关联与超流性**：通过 BCS 自洽求解获得能隙、化学势与配对能。
- **衰变性质预测**：β 衰变 Q 值、半衰期、分支比的不确定性量化。
- **统计衰变链**：蒙特卡洛模拟连续衰变至稳定核的种群演化。
- **核反应截面**：中子转移与 Coulomb 碎裂截面的半经典估算。
- **核子输运**：热核中核子的扩散系数与蒸发衰变率。
- **三维密度分布**：形变核的 RMS 半径与体密度积分。
- **不确定性传播**：稀疏网格积分评估核力参数不确定性的高维传播。

---

## 8. 修改记录

- 所有原 MATLAB 代码被完全改写为 Python（`numpy`/`scipy`）。
- 删除所有可视化内容（`mario` 绘图、`peaks_movie` 动画、`triangulation_svg` 图形输出、`usa_box_plot` 填充可视化）。
- 将 `r8ge_cg` 的 CG 逻辑移植到 `hfb_selfconsistent.py` 的密度矩阵更新中。
- 将 `fem1d_spectral_numeric` 的谱展开思想与 `quadrature_weights_vandermonde` 的求积框架融合为稳定的有限差分径向求解器。
- 将 `sparse_grid_cc` 的 Smolyak 构造完整移植到 `quadrature_engine.py`。
- 将 `tet_mesh_to_xml` 的拓扑结构保留为 XML 输出接口，数学上升级为二十面体球面剖分。
- 将 `disk01_integrals` 的 Gamma 函数解析公式直接用于角动量耦合相空间因子。
- 将 `discrete_pdf_sample_2d` 的逆 CDF 采样机制升级为任意维衰变链 MC。

---

*合成完成日期：2026-05-04*
