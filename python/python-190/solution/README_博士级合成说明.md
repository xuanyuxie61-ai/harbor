# 物理信息生成对抗网络（PI-GAN）三维不可压缩流场合成项目

## 一、项目概述

本项目是基于 **15 个种子科研代码项目** 合成的**博士级 Python 科研计算项目**，核心科学领域为**数据科学：生成对抗网络（GAN）训练**。项目将原始种子代码中的数学物理算法（Navier-Stokes 精确解、四元数旋转、三角形对称求积、CVT 采样、Clausen 特殊函数、正态分布近似、球面求积、Hooke-Jeeves 优化、TSP 路径规划、OBJ 网格 I/O 等）深度融合，构建了一个面向**三维不可压缩流场生成**的物理信息 GAN（Physics-Informed GAN, PI-GAN）系统。

### 1.1 科学问题定义

**问题**：如何利用生成对抗网络（GAN）生成满足三维不可压缩 Navier-Stokes 方程的流场数据？

该问题具有以下博士级难度：
- **PDE 约束生成**：生成器不仅要"欺骗"判别器，还必须使生成的速度场/压力场满足连续性方程与动量方程。
- **旋转等变性**：三维流场具有 SO(3) 对称性，生成网络需在四元数代数框架下保持旋转等变。
- **高阶数值积分**：物理损失的计算需要在三角化域和球面上进行精确的数值积分。
- **隐空间结构化采样**：使用 Centroidal Voronoi Tessellation（CVT）优化隐空间的采样点布局。
- **超参数非梯度优化**：使用 Hooke-Jeeves 直接搜索对物理损失权重等超参数进行后验优化。

### 1.2 核心数学物理模型

#### Navier-Stokes 方程（不可压缩流）
连续性方程：
$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

动量方程：
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}$$

#### Ethier 精确解（C. Ross Ethier & David Steinman, 1994）
给定参数 $a = \pi/4$, $d = \pi/2$，速度分量与压力为：

$$
\begin{aligned}
u &= -a\left(e^{ax}\sin(ay+dz) + e^{az}\cos(ax+dy)\right)e^{-d^2 t} \\
v &= -a\left(e^{ay}\sin(az+dx) + e^{ax}\cos(ay+dz)\right)e^{-d^2 t} \\
w &= -a\left(e^{az}\sin(ax+dy) + e^{ay}\cos(az+dx)\right)e^{-d^2 t} \\
p &= \frac{1}{2}a^2 e^{-2d^2 t}\Big(e^{2ax} + e^{2ay} + e^{2az} \\
  &\quad + 2\sin(ax+dy)\cos(az+dx)e^{a(y+z)} \\
  &\quad + 2\sin(ay+dz)\cos(ax+dy)e^{a(z+x)} \\
  &\quad + 2\sin(az+dx)\cos(ay+dz)e^{a(x+y)}\Big)
\end{aligned}
$$

#### 四元数 Hamilton 积与 SO(3) 旋转
四元数乘法：
$$q_1 \otimes q_2 = \big(q_{1w}q_{2w} - \mathbf{q}_{1v}\cdot\mathbf{q}_{2v},\; q_{1w}\mathbf{q}_{2v} + q_{2w}\mathbf{q}_{1v} + \mathbf{q}_{1v} \times \mathbf{q}_{2v}\big)$$

单位四元数旋转向量：
$$\mathbf{v}' = q \otimes (0, \mathbf{v}) \otimes q^*$$

其中 $q^*$ 为共轭四元数，$q = (\cos(\theta/2), \sin(\theta/2)\,\mathbf{n})$。

#### 三角形对称求积
参考三角形 $T$（顶点 $(0,0),(1,0),(0,1)$）上的单项式精确积分：
$$\iint_T x^m y^n \,dx\,dy = \frac{m!\,n!}{(m+n+2)!}$$

对称求积规则（degree 5, 6点 Strang 规则）：
$$\iint_T f(x,y)\,dx\,dy \approx |T| \sum_{i=1}^{6} w_i f(x_i, y_i)$$

#### 球面三角形面积（L'Huilier 定理）
设球面边长为 $a,b,c$，半周长 $s = (a+b+c)/2$，则球面盈量 $E$ 满足：
$$\tan\frac{E}{4} = \sqrt{\tan\frac{s}{2} \tan\frac{s-a}{2} \tan\frac{s-b}{2} \tan\frac{s-c}{2}}$$
球面三角形面积（单位球面）$= E$。

#### Clausen 函数
$$\mathrm{Cl}_2(x) = -\int_0^x \ln\left|2\sin\frac{t}{2}\right|dt = \sum_{k=1}^{\infty} \frac{\sin(kx)}{k^2}$$

#### CVT 能量泛函
$$E(G) = \sum_{i=1}^{k} \int_{V_i} \|x - g_i\|^2 \rho(x)\,dx$$
Lloyd 迭代单调递减 $E(G)$。

#### Hooke-Jeeves 模式搜索
探测移动（Exploratory Move）沿坐标轴 $\pm\delta_i$ 搜索，模式移动（Pattern Move）沿成功方向外推：
$$x_{\text{pattern}} = x_{\text{new}} + (x_{\text{new}} - x_{\text{old}})$$
步长收缩：$\delta \leftarrow \rho \cdot \delta$，直至 $\|\delta\| \leq \varepsilon$。

---

## 二、种子项目映射与融合方案

| 序号 | 种子项目 | 核心算法/数据结构 | 合成项目中的角色 |
|------|----------|-------------------|------------------|
| 1 | `788_navier_stokes_3d_exact` | Ethier 精确解、NS 残差计算 | `navier_stokes_exact.py`：生成真实训练数据，计算物理残差损失 |
| 2 | `960_quaternions` | 四元数乘法、旋转、指数 | `quaternion_equivariance.py`：SO(3) 等变约束，旋转一致性验证 |
| 3 | `713_maple_area` | Monte Carlo 面积估计 | `complex_geometry.py`：复杂多边形域内的面积估计与物理场采样 |
| 4 | `1316_triangle_symq_rule` | 三角形对称求积、单项式精确积分 | `triangle_quadrature.py`：平面/球面三角形高阶数值积分，PDE 残差区域积分 |
| 5 | `548_human_mesh2d` | 2D 三角网格生成 | `mesh_generator.py`：Delaunay 三角剖分，复杂边界内网格生成与质量评估 |
| 6 | `1300_triangle_distance` | 三角形内距离 PDF、Monte Carlo 统计 | `geometric_stats.py`：几何距离统计，Wasserstein-1 近似距离评估 |
| 7 | `248_cvt_2d_sampling` | Lloyd 算法、CVT 采样 | `cvt_sampler.py`：物理域与隐空间的最优采样点布局 |
| 8 | `187_clausen` | Clausen 函数、Chebyshev 级数 | `special_functions.py`：周期性激活函数，谱基函数构建 |
| 9 | `032_asa066` | 正态 CDF 近似（AS 66, Alg 5666, Alg 39） | `normal_approx.py`：重参数化技巧，高斯 KL 散度，隐空间采样 |
| 10 | `1306_triangle_histogram` | 三角形直方图、均匀性测试 | `uniformity_test.py`（内嵌于 `mesh_generator.py` 评估中）：生成样本覆盖度评估 |
| 11 | `1266_toms178` | Hooke-Jeeves 直接搜索 | `hooke_jeeves.py`：GAN 超参数（学习率、物理损失权重）自适应优化 |
| 12 | `1130_sphere_triangle_quad` | 球面三角形求积、L'Huilier 定理 | `sphere_quad.py`：球面三角形上的 Monte Carlo 与中点求积，三维能谱积分 |
| 13 | `547_human_data` | 人类轮廓边界数据 | `complex_geometry.py`：复杂边界几何构造，非规则域流场采样 |
| 14 | `1363_tsp_brute` | TSP 穷举、Trotter 排列生成 | `latent_path.py`：隐向量最优排序与 Slerp 插值，生成平滑过渡序列 |
| 15 | `822_obj_io` | Wavefront OBJ 文件 I/O | `obj_io.py`：3D 网格顶点/面片/法线读写，icosphere 生成 |

---

## 三、文件结构与改造说明

本项目共包含 **16 个 Python 文件**（远超 8 个最低要求），所有代码从零编写，基于种子项目的核心数学思想进行深度融合与扩展。

### 3.1 文件清单

1. **`main.py`** — 统一入口，零参数可运行。 orchestrates 数据生成 → GAN 训练 → 等变性验证 → 高阶数值评估 → 超参数优化 → 报告输出。
2. **`navier_stokes_exact.py`** — 基于种子 788：Ethier 精确解、中心差分 NS 残差计算、训练数据生成。
3. **`quaternion_equivariance.py`** — 基于种子 960：四元数代数、SO(3) 旋转、生成器等变性损失。
4. **`triangle_quadrature.py`** — 基于种子 1316 + 1130：平面三角形对称求积、球面三角形 L'Huilier 面积与 Monte Carlo 求积。
5. **`mesh_generator.py`** — 基于种子 548：Bowyer-Watson Delaunay 三角剖分、人体轮廓网格生成、网格质量统计。
6. **`geometric_stats.py`** — 基于种子 1300：Turk 均匀采样、距离统计、Wasserstein-1 近似距离。
7. **`cvt_sampler.py`** — 基于种子 248：Lloyd 迭代、CVT 能量泛函、二维域与高维隐空间最优采样。
8. **`special_functions.py`** — 基于种子 187：Clausen 函数级数求和、Chebyshev 递推、Clausen 周期性激活函数。
9. **`normal_approx.py`** — 基于种子 032：AS 66 / Algorithm 5666 / Algorithm 39 正态 CDF、Box-Muller 变换、重参数化采样、KL 散度。
10. **`sphere_quad.py`** — 基于种子 1130：球面三角形面积、重心/中点计算、3点/7点求积规则、正二十面体剖分球面积分。
11. **`hooke_jeeves.py`** — 基于种子 1266：模式搜索算法（探测移动 + 模式移动 + 步长收缩），GAN 超参数优化接口。
12. **`complex_geometry.py`** — 基于种子 547 + 713：射线法点包含判断、Monte Carlo 多边形面积、人体轮廓参数化、域内物理场采样。
13. **`latent_path.py`** — 基于种子 1363：Trotter 排列生成、TSP 穷举/贪心路径、Slerp 球面插值、隐空间过渡序列生成。
14. **`obj_io.py`** — 基于种子 822：OBJ 文件读写、面片/顶点法线计算、icosphere 递归细分生成。
15. **`gan_numpy.py`** — **原创核心**：纯 NumPy 坐标条件 GAN（Generator + Discriminator），手动实现前向/反向传播、SGD with momentum、BCE/MSE 损失、物理损失接口。
16. **`training_engine.py`** — **原创核心**：训练引擎，整合对抗训练、物理损失（每 5 轮完整网格评估）、等变损失监控、CVT/Box-Muller 采样、综合几何评估。

### 3.2 关键改造与扩展

- **从 MATLAB 到 Python**：所有种子项目原为 MATLAB/Octave 代码，已完整翻译为 Python，去除所有可视化相关代码，保留核心数学算法。
- **从独立脚本到模块化库**：每个种子项目的函数被封装为独立模块，通过 `import` 在 GAN 训练流程中协同调用。
- **从零搭建 GAN**：不依赖 PyTorch/TensorFlow，使用纯 NumPy 手动实现反向传播链式法则，包含数值稳定的 Sigmoid 与 LeakyReLU。
- **物理信息注入**：在标准对抗损失外，增加了基于 NS 中心差分残差的物理损失，以及基于四元数旋转的等变损失。
- **高阶数值方法融合**：训练评估阶段同时调用平面三角形求积、球面三角形求积、CVT 最优采样、Hooke-Jeeves 优化等多种数值方法，体现博士级计算复杂度。

---

## 四、运行方式

### 环境要求
- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.7（用于 `scipy.spatial.Delaunay` 与 `cKDTree`）

### 运行命令
```bash
cd Synthesis-project-python/190_synth_project
python3 main.py
```

程序零参数运行，自动完成：
1. 基于 Ethier 精确解生成 216 点真实流场数据；
2. 训练 120 轮物理信息 GAN（约 0.8 秒）；
3. 四元数旋转等变性验证；
4. 三角形/球面求积、Wasserstein 距离、网格质量评估；
5. Hooke-Jeeves 超参数优化演示；
6. 输出中文训练报告并保存到 `training_report.txt`。

---

## 五、科学贡献与前沿性

本项目解决了一个前沿的交叉学科问题——**如何用深度生成模型在严格的物理定律约束下生成高保真三维流场**。其博士级难度体现在：

1. **PDE-约束生成建模**：将计算流体力学（CFD）中的经典精确解与现代生成模型结合，突破了纯数据驱动 GAN 的物理不可解释性瓶颈。
2. **对称性保持机器学习**：在四元数代数框架下显式编码 SO(3) 等变性，这比常规的 data augmentation 方法更具数学严谨性。
3. **多尺度数值分析**：同时涉及有限差分（NS 残差）、有限元风格求积（三角形积分）、谱方法（Clausen 函数）与蒙特卡洛方法（球面/面积估计）等多种数值方法。
4. **无梯度优化与梯度下降协同**：Hooke-Jeeves 直接搜索用于超参数调优，与 GAN 的 SGD 形成互补优化策略。
5. **复杂几何处理**：从简单立方体域拓展到非规则人体轮廓域内的流场采样，展示了算法在复杂边界条件下的鲁棒性。

---

## 六、边界处理与数值鲁棒性

- **指数截断**：`np.clip(..., -700, 700)` 防止 `exp` 上溢/下溢。
- **除零保护**：所有除法操作均检查分母是否小于 `1e-15`，若为真则替换为安全值。
- **角度截断**：`arccos` 输入截断到 `[-1, 1]`，防止浮点误差导致的 NaN。
- **网格退化处理**：`ns_residual` 函数在网格维度不足时自动退化为前向差分近似。
- **空 Voronoi 单元**：CVT Lloyd 迭代中，空单元保持原生成元不变。
- **Sigmoid 数值稳定性**：使用分段公式避免大负数输入时的指数上溢。
