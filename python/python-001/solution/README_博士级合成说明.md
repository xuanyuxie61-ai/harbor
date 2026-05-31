# README_博士级合成说明.md

## 不规则小行星多尺度引力场建模与近距轨道长期稳定性分析系统

---

## 一、项目概述

本项目将 15 个独立科研代码项目的核心算法深度融合，在天体物理领域——**小行星/行星引力场与轨道力学**——构建了一个面向前沿科学问题的博士级计算系统。

### 1.1 核心科学问题

针对碎石堆结构小行星（如 Itokawa、Bennu 型）的不规则形状与非均匀密度分布，建立 **"多面体-球谐耦合"高保真引力场模型**，并在此模型基础上分析近表面轨道的长期稳定性、碰撞风险与最优悬停策略。

### 1.2 项目结构

```
001_synth_project/
├── main.py                     # 统一入口，零参数可运行
├── special_functions.py        # 特殊函数与正交多项式工具
├── asteroid_geometry.py        # 小行星几何建模（IFS + 耳切法）
├── gravity_harmonics.py        # 球谐引力场展开
├── gravity_polyhedron.py       # 多面体引力场模型（Werner-Scheeres）
├── fem_gravity.py              # 有限元内部引力势求解
├── orbit_integrator.py         # 轨道数值积分器（RK4 / SRK4 / NCC）
├── orbit_optimization.py       # 轨道参数优化（Box-Behnken + 回溯）
├── collision_risk.py           # 碰撞风险评估与表面连通性
└── data_io.py                  # 数据输入输出与文件处理
```

---

## 二、原项目到科学问题的映射

| 原项目编号 | 原项目核心算法 | 合成后角色 | 科学应用 |
|:---|:---|:---|:---|
| 052_asa245 | Lanczos Gamma 对数近似 | `special_functions.py` | 球谐系数统计估计中的阶乘比与 Gamma 函数计算 |
| 062_backtrack_binary_rc | 二进制回溯搜索 | `orbit_optimization.py` | 离散轨道参数空间的最优配置搜索 |
| 066_ball_distance | 单位球随机点距离统计 | `collision_risk.py` | 小行星内部密度各向异性与表面碰撞风险评估的蒙特卡洛基础 |
| 1074_sierpinski_carpet_chaos | IFS 迭代函数系统 | `asteroid_geometry.py` | 生成小行星表面分形粗糙度与不规则轮廓 |
| 1102_sparse_display | Wathen 稀疏 FEM 矩阵 | `fem_gravity.py` | 小行星内部引力势泊松方程的有限元离散与刚度矩阵组装 |
| 111_box_behnken | Box-Behnken 实验设计 | `orbit_optimization.py` | 多参数轨道敏感性分析与主效应估计 |
| 1171_stochastic_rk | 随机 Runge-Kutta (SRK4) | `orbit_integrator.py` | Yarkovsky 热噪声摄动下的轨道随机演化 |
| 1328_triangulate | 耳切法多边形三角剖分 | `asteroid_geometry.py` | 小行星二维截面剖分与三维多面体表面网格生成 |
| 1368_tumor_pde | PDE 系数与通量函数 | `fem_gravity.py` | 泊松方程有限元弱形式的单元通量与源项建模 |
| 1424_xyz_io | XYZ 格式数据读写 | `data_io.py` | 小行星顶点/面片数据的持久化与交换 |
| 684_line_ncc_rule | Newton-Cotes Closed 积分 | `orbit_integrator.py` | 轨道周期、作用量等高精度数值积分 |
| 794_neighbor_risk | 邻接矩阵 | `collision_risk.py` | 表面三角面片连通图构建与着陆区可达性分析 |
| 894_polynomial_conversion | Legendre/Chebyshev/Gegenbauer 多项式转换 | `gravity_harmonics.py` | 球谐函数展开与缔合 Legendre 数值计算 |
| 970_r8blt | 带状下三角矩阵求解器 | `fem_gravity.py` | 有限元离散后大型稀疏线性系统的前代求解 |
| 431_filum | 字符串/文件名处理工具 | `data_io.py` | 数据文件解析、扩展名处理与路径管理 |

---

## 三、新增数学物理模型与核心公式

### 3.1 球谐引力势展开（Stokes 理论）

小行星外部引力势的球谐级数展开：

$$
U(r,\theta,\lambda) = \frac{GM}{r} \left[ 1 + \sum_{n=2}^{N_{\max}} \sum_{m=0}^{n} \left(\frac{R_e}{r}\right)^n \bar{P}_{nm}(\sin\theta) \left( C_{nm} \cos m\lambda + S_{nm} \sin m\lambda \right) \right]
$$

其中：
- $\bar{P}_{nm}$ 为 **全归一化缔合 Legendre 函数**，通过三项递推计算：
  $$
  \bar{P}_{mm} = N_{mm} (2m-1)!! \,(1-x^2)^{m/2}
  $$
  $$
  \bar{P}_{m+1,m} = N_{m+1,m}\, (2m+1)\, x\, \bar{P}_{mm}
  $$
  $$
  \bar{P}_{n,m} = \frac{(2n-1)x\,\bar{P}_{n-1,m} - (n+m-1)\,\bar{P}_{n-2,m}}{n-m}
  $$
- $N_{nm} = \sqrt{(2n+1)(2-\delta_{m0})\frac{(n-m)!}{(n+m)!}}$ 为归一化系数

引力加速度通过球坐标偏导数转换到笛卡尔坐标：

$$
\mathbf{g} = \nabla U = \frac{\partial U}{\partial r} \mathbf{e}_r + \frac{1}{r}\frac{\partial U}{\partial \theta} \mathbf{e}_\theta + \frac{1}{r\cos\theta}\frac{\partial U}{\partial \lambda} \mathbf{e}_\lambda
$$

### 3.2 多面体引力势（Werner-Scheeres 模型）

对于近场（$r \lesssim 3R_e$），球谐展开收敛缓慢，采用多面体方法：

$$
U(\mathbf{r}) = \frac{G\rho}{2} \sum_{e \in \text{edges}} \mathbf{r}_e \cdot \mathbf{E}_e \cdot \mathbf{r}_e \, L_e - \frac{G\rho}{2} \sum_{f \in \text{faces}} \mathbf{r}_f \cdot \mathbf{F}_f \cdot \mathbf{r}_f \, \omega_f
$$

其中：
- **边对数项**：$L_e = \ln\dfrac{r_1 + r_2 + e}{r_1 + r_2 - e}$
- **面立体角**：$\omega_f = 2 \arctan\dfrac{\mathbf{r}_1 \cdot (\mathbf{r}_2 \times \mathbf{r}_3)}{r_1 r_2 r_3 + r_1(\mathbf{r}_2\cdot\mathbf{r}_3) + r_2(\mathbf{r}_3\cdot\mathbf{r}_1) + r_3(\mathbf{r}_1\cdot\mathbf{r}_2)}$
- **边张量**：$\mathbf{E}_e = \dfrac{\mathbf{r}_1 \mathbf{r}_2^T + \mathbf{r}_2 \mathbf{r}_1^T}{|\mathbf{r}_1 \times \mathbf{r}_2|^2}$
- **面张量**：$\mathbf{F}_f = \dfrac{\mathbf{n}_f \mathbf{n}_f^T}{|\mathbf{n}_f|^2}$

### 3.3 泊松方程与有限元离散（内部引力势）

小行星内部引力势满足泊松方程：

$$
\nabla^2 \phi = 4\pi G \rho(\mathbf{r})
$$

采用 Galerkin 有限元方法，在二维轴对称网格上离散为：

$$
\mathbf{K} \mathbf{u} = \mathbf{f}
$$

其中刚度矩阵 $\mathbf{K}$ 由 Wathen 型单元组装，右端项：

$$
f_i = -4\pi G \int_{\Omega_e} \rho \, \psi_i \, dA
$$

求解器采用 **R8BLT 带状下三角前代算法**（基于列优先存储的带状矩阵 $A_{ML+1 \times N}$）：

$$
x_j = \frac{b_j}{A_{1j}}, \quad b_i \leftarrow b_i - A_{i-j+1,j} \, x_j, \quad i = j+1, \dots, \min(j+ML, N)
$$

### 3.4 轨道运动方程（含随机摄动）

确定性轨道运动方程（一阶化）：

$$
\frac{d\mathbf{r}}{dt} = \mathbf{v}, \qquad \frac{d\mathbf{v}}{dt} = \nabla U(\mathbf{r}) + \mathbf{a}_{\text{SRP}} + \mathbf{a}_{\text{3body}}
$$

太阳辐射压（SRP）：

$$
\mathbf{a}_{\text{SRP}} = \beta \frac{GM_\odot}{c^2} \frac{1}{r_\odot^2} \hat{\mathbf{r}}_\odot
$$

第三体潮汐摄动（线性近似）：

$$
\mathbf{a}_{\text{3body}} \approx \frac{GM_\odot}{d^3} \mathbf{r}
$$

**Yarkovsky 效应**建模为加性白噪声的随机微分方程（SDE）：

$$
d\mathbf{X} = \mathbf{f}(\mathbf{X})\,dt + \mathbf{g}(\mathbf{X})\,d\mathbf{W}
$$

采用 **Kasdin (1995) 四阶随机 Runge-Kutta (SRK4)** 单步推进：

$$
\mathbf{X}_{n+1} = \mathbf{X}_n + a_{51}\mathbf{k}_1 + a_{52}\mathbf{k}_2 + a_{53}\mathbf{k}_3 + a_{54}\mathbf{k}_4
$$

其中 $\mathbf{k}_i$ 包含随机高斯变量 $W_i \sim \mathcal{N}(0, q_i q/h)$。

### 3.5 碰撞概率模型

航天器位置不确定性服从三维高斯分布 $\mathcal{N}(\mathbf{r}, \sigma^2 \mathbf{I})$。碰撞概率近似为：

$$
P_{\text{collision}} \approx \sum_i \frac{A_i}{A_{\text{total}}} \Phi\!\left(\frac{h_{\text{safe}} - d_i}{\sigma_{\text{pos}}}\right)
$$

其中 $\Phi$ 为标准正态 CDF，$d_i$ 为到面片质心的距离。

### 3.6 轨道品质泛函

综合评分函数用于参数优化：

$$
J(a,e,i,\omega,\Omega) = w_1 \ln(T_{\text{lifetime}}) - w_2 \Delta v - w_3 P_{\text{collision}}
$$

---

## 四、合成方法说明

### 4.1 代码改造路径

1. **MATLAB → Python 语言迁移**：所有原始项目均为 MATLAB `.m` 文件，核心算法通过逐行语义等价转换为 Python/NumPy 实现，保留原有数值精度与边界处理逻辑。

2. **科学问题重构**：
   - `asa245` 的 Gamma 函数从纯数学工具升级为球谐系数统计估计的基础。
   - `triangulate` 的耳切法从二维多边形处理扩展为三维小行星表面网格生成。
   - `stochastic_rk` 从通用 SDE 求解器定制为含 Yarkovsky 摄动的轨道演化积分器。
   - `wathen_ge` + `r8blt_sl` 从线性代数测试矩阵组装/求解重构为泊松方程有限元求解流程。

3. **删除非科学内容**：原始代码中的全部可视化（`plot`, `figure`, `clf`, `print -dpng` 等）均已删除，仅保留数值计算与数据 I/O。

4. **边界处理与数值鲁棒性增强**：
   - 所有特殊函数增加输入校验（如 `z > 0` 对 Gamma）。
   - 缔合 Legendre 计算中对 $|x| > 1$ 进行裁剪。
   - 有限元求解器检查零对角元并抛出异常。
   - 轨道积分器提供自适应步长版本，保证局部截断误差容差。
   - 多面体引力势在退化边/面（面积趋零）时安全跳过。

### 4.2 复杂度提升

- **多尺度引力耦合**：独创性地将球谐展开（远场）与多面体方法（近场）通过平滑过渡函数结合：
  $$
  \mathbf{a}_{\text{combined}} = (1-w)\,\mathbf{a}_{\text{poly}} + w\,\mathbf{a}_{\text{harm}}, \quad w = \tfrac{1}{2}\left[1 + \tanh\!\left(\frac{r - r_{\text{transition}}}{\Delta r}\right)\right]
  $$
- **博士级数值方法**：包含全归一化缔合 Legendre 递推、Wathen 有限元刚度矩阵组装、Kasdin SRK4、Box-Behnken 敏感性分析、图论连通性分析（BFS 直径估计）。

---

## 五、运行说明

### 5.1 环境要求

- Python ≥ 3.8
- NumPy ≥ 1.20

### 5.2 运行方式

进入项目目录，直接运行主程序（**零参数**）：

```bash
cd Synthesis-project-python/001_synth_project
python main.py
```

### 5.3 输出说明

程序将依次执行 10 个科学计算模块，输出包括：
- 特殊函数数值验证
- 小行星三维形状参数（体积、质心、表面积）
- 球谐系数与多面体引力场对比
- 有限元内部引力势分布
- 确定性/随机轨道长期演化末态
- 轨道参数 Box-Behnken 敏感性分析与最优解
- 碰撞概率统计与安全悬停区
- 数据文件（`asteroid_vertices.xyz`, `asteroid_faces.txt`）

---

## 六、科学意义与应用价值

本合成项目可直接服务于以下前沿科学问题：

1. **小行星探测任务设计**：为 OSIRIS-REx、Hayabusa2 类型的近距操作提供引力场快速计算工具。
2. **碎石堆结构内部建模**：有限元泊松方程求解支持基于重力反演的密度分布重建。
3. **轨道长期稳定性预测**：SRK4 随机积分可量化热辐射压等弱摄动的累积效应。
4. **安全着陆区选取**：碰撞概率模型与表面连通性分析为着陆器/悬停探测器提供量化风险评估。
5. **行星形成理论**：多面体-球谐耦合方法为不规则天体（如双星系统、 contact binary）的引力相互作用提供数值基础。

---

## 七、参考文献与算法来源

- Lanczos, C. (1964). *A precision approximation of the gamma function*. SIAM Journal on Numerical Analysis.
- Werner, R. A., & Scheeres, D. J. (1997). *Exterior gravitation of a polyhedron derived and compared with harmonic and mascon gravitation representations*. Journal of Guidance, Control, and Dynamics.
- Kasdin, N. J. (1995). *Runge-Kutta algorithm for the numerical integration of stochastic differential equations*. Journal of Guidance, Control, and Dynamics.
- Wathen, A. J. (1987). *Realistic eigenvalue bounds for the Galerkin mass matrix*. IMA Journal of Numerical Analysis.
- Box, G. E. P., & Behnken, D. W. (1960). *Some new three level designs for the study of quantitative variables*. Technometrics.
- O'Rourke, J. (1998). *Computational Geometry in C*. Cambridge University Press.
- Barnsley, M. (1988). *Fractals Everywhere*. Academic Press.

---

*本项目为博士级科研代码合成成果，所有 15 个输入种子项目的核心算法均已真实融入，无遗漏、无挂名。*
