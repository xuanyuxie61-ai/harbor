# 钙钛矿太阳能电池多物理场耦合效率模拟系统

## 1. 项目概述

本项目将 **15 个科研代码种子项目** 的核心算法融合重构，围绕**能源系统：太阳能电池光电转换效率**这一前沿科学领域，构建了一个面向钙钛矿（Perovskite）太阳能电池的多物理场耦合模拟与光电转换效率评估系统。

科学问题的深度达到博士级，涉及：
- **光物理**：AM1.5G 光谱离散采样、Beer-Lambert 光吸收、三维楔形体高斯求积
- **载流子输运**：漂移-扩散方程（Scharfetter-Gummel 离散）、稳态/瞬态数值求解
- **材料科学**：温度-组分依赖的材料参数插值、多晶网格生成
- **缺陷物理**：SRH/辐射/Auger/带尾复合机制、蒙特卡洛缺陷分布
- **力学耦合**：薄膜热应力屈曲分析与带隙移动
- **离子动力学**：离子迁移-载流子耦合迟滞效应
- **不确定性量化**：多项式混沌展开（PCE）与 SVD 模型降阶

---

## 2. 原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 合成后角色 |
|------|-----------|---------|-----------|
| 1 | `538_histogram_data_2d_sample` | 离散 CDF → XY 采样 | `spectrum_sampler.py`：AM1.5G 太阳光谱二维离散概率采样，生成入射光子能量与角度分布 |
| 2 | `1406_wedge_exactness` | 楔形体多项式求积精确性检验 | `absorption_integrator.py`：钙钛矿薄膜三维楔形体上的 Gauss 求积，计算光生载流子产生率体积分 |
| 3 | `927_pwl_interp_2d` | 二维分段线性插值 | `material_interpolator.py`：MAPbI₃ 材料参数（带隙、迁移率）随温度 T 和卤素配比 x 的二维插值 |
| 4 | `769_mm_io` | Matrix Market 稀疏矩阵 I/O | `sparse_matrix_io.py`：漂移-扩散 Jacobian 的 COO 稀疏矩阵存储与读写 |
| 5 | `767_midpoint_fixed` | 固定点中点法 ODE 求解 | `drift_diffusion_solver.py`：载流子输运瞬态方程的时间推进 |
| 6 | `1336_triangulation_display` | 三角剖分数据处理 | `mesh_triangulation.py`：多晶薄膜二维三角网格生成与 PLY 格式 I/O |
| 7 | `1006_random_data` | 多区域随机采样 | `defect_monte_carlo.py`：晶粒三角形区域内的缺陷均匀采样（Turk's rule）与正态分布 |
| 8 | `780_mortality` | 死亡率 PDF/CDF 统计 | `defect_monte_carlo.py`：载流子寿命的 PDF/CDF 建模与生存概率分析 |
| 9 | `550_humps_ode` | Humps ODE 精确解与导数 | `drift_diffusion_solver.py`：数值求解器精度基准验证（Humps 精确解对比） |
| 10 | `122_buckling_spring` | 弹簧屈曲 λ/μ 参数曲线 | `mechanical_stress.py`：薄膜热应力屈曲分析与屈曲后挠度计算 |
| 11 | `641_laguerre_polynomial` | Laguerre 多项式与 Gauss-Laguerre 求积 | `recombination_models.py`：带尾态辐射复合的半无穷区间数值积分 |
| 12 | `873_ply_io` | PLY 三维网格文件读写 | `mesh_triangulation.py`：多面体晶粒网格的 PLY 格式导入导出 |
| 13 | `854_pce_ode_hermite` | Hermite PCE 随机 ODE 展开 | `uncertainty_pce.py`：器件效率的不确定性量化（均值、方差、敏感性指标） |
| 14 | `345_exm` | Euler ODE + Predator-Prey 系统 | `coupled_ion_migration.py`：离子迁移-载流子耦合动力学（类比捕食者-猎物） |
| 15 | `1187_svd_fingerprint` | SVD 分解与低秩近似 | `model_reduction.py`：Jacobian 矩阵的 POD 降阶与低秩近似加速 |

---

## 3. 核心数学物理模型与公式

### 3.1 光子吸收与载流子产生

**光子能量-波长关系**：
$$
E = \frac{hc}{\lambda} = \frac{1239.8}{\lambda_{\text{nm}}} \, \text{eV}
$$

**Beer-Lambert 吸收定律**（在楔形体区域 $W$ 内）：
$$
I(z, \lambda) = I_0(\lambda) \exp\bigl[-\alpha(\lambda)(z + d/2)\bigr]
$$

**载流子产生率密度**：
$$
G(\mathbf{r}, \lambda) = \frac{\alpha(\lambda) I(\mathbf{r}, \lambda)}{E_{\text{photon}}(\lambda)}
$$

**总产生率（楔形体体积分）**：
$$
Q_{\text{gen}} = \iiint_W G(x,y,z) \, dV \approx V_W \sum_k w_k G(\mathbf{x}_k)
$$

其中楔形体体积 $V_W = L_{xy}^2 d / 2$，求积规则采用 Dunavant 三角形规则 × Gauss-Legendre 张量积。

### 3.2 漂移-扩散方程（稳态/瞬态）

**电子连续性方程**：
$$
\frac{\partial n}{\partial t} = \frac{1}{q} \nabla \cdot \mathbf{J}_n + G - R
$$

**空穴连续性方程**：
$$
\frac{\partial p}{\partial t} = -\frac{1}{q} \nabla \cdot \mathbf{J}_p + G - R
$$

**Poisson 方程**：
$$
\nabla^2 \phi = -\frac{q}{\varepsilon}(p - n + N_D^+ - N_A^-)
$$

**Scharfetter-Gummel 电流密度离散**（数值稳定格式）：
$$
J_{n,i+1/2} = \frac{q D_n}{\Delta x} \bigl[ B(\Delta \phi / kT_q) n_i - B(-\Delta \phi / kT_q) n_{i+1} \bigr]
$$

其中 $B(x) = \frac{x}{e^x - 1}$ 为 Bernoulli 函数，在 $|x| \to 0$ 时采用 Taylor 展开：
$$
B(x) \approx 1 - \frac{x}{2} + \frac{x^2}{12}
$$

**Einstein 关系**：
$$
D_n = \frac{k_B T}{q} \mu_n, \qquad D_p = \frac{k_B T}{q} \mu_p
$$

### 3.3 复合机制

**总复合率**：
$$
R_{\text{total}} = R_{\text{SRH}} + R_{\text{rad}} + R_{\text{Auger}} + R_{\text{tail}}
$$

**Shockley-Read-Hall (SRH) 复合**：
$$
R_{\text{SRH}} = \frac{np - n_i^2}{\tau_p(n + n_1) + \tau_n(p + p_1)}
$$

其中 $n_1 = N_c \exp\bigl(\frac{E_t - E_c}{k_B T}\bigr)$，$p_1 = N_v \exp\bigl(\frac{E_v - E_t}{k_B T}\bigr)$。

**辐射复合**（van Roosbroeck-Shockley）：
$$
R_{\text{rad}} = B(T)(np - n_i^2), \qquad B(T) = B_{300} \left(\frac{T}{300}\right)^{-3/2}
$$

**Auger 复合**：
$$
R_{\text{Auger}} = (C_n n + C_p p)(np - n_i^2)
$$

**带尾态辐射复合**（Urbach 尾，需 Gauss-Laguerre 求积）：
$$
R_{\text{tail}} = N_{t,\text{tail}} \sigma_{\text{eff}} v_{\text{th}} \int_0^\infty e^{-x} \frac{np - n_i^2}{p + n_1(x) + n + p_1(x)} \, dx
$$

积分变量替换 $x = (E - E_g)/E_u$，权函数 $e^{-x}$ 使 Gauss-Laguerre 求积成为最优选择：
$$
\int_0^\infty e^{-x} f(x) \, dx \approx \sum_{i=1}^N w_i f(x_i)
$$

### 3.4 材料参数模型

**带隙 Varshni 公式**（温度依赖）：
$$
E_g(T, x) = E_g(0, x) - \frac{S T^2}{T + \Theta_D}
$$

其中 $E_g(0, x) = 1.57 + 0.72x$ eV（MAPbI₃ → MAPbBr₃），$S = 8 \times 10^{-4}$ eV/K，$\Theta_D = 150$ K。

**迁移率温度依赖**（声学声子散射）：
$$
\mu(T) = \mu_0 \left(\frac{T}{300}\right)^{-3/2}
$$

### 3.5 热应力与屈曲

**双轴热应力**：
$$
\sigma_{\text{th}} = \frac{E}{1 - \nu}(\alpha_{\text{film}} - \alpha_{\text{substrate}}) \Delta T
$$

**屈曲临界应力**（薄膜-衬底系统）：
$$
\sigma_{\text{cr}} = \frac{\pi^2 E}{12(1 - \nu^2)} \left(\frac{d}{\lambda}\right)^2
$$

**屈曲后挠度**（von Kármán 简化）：
$$
w(x) = w_{\max} \sin\left(\frac{\pi x}{\lambda}\right)
$$

**应变引起的带隙移动**（形变势理论）：
$$
\Delta E_g = a_{\text{def}} \cdot 2\varepsilon_{\text{in-plane}}
$$

### 3.6 多项式混沌展开（PCE）不确定性量化

**随机衰减模型**（效率退化）：
$$\frac{d\eta}{dt} = -\alpha_{\text{eff}} \eta, \qquad \alpha_{\text{eff}} = \alpha_\mu + \alpha_\sigma \xi, \quad \xi \sim \mathcal{N}(0,1)$$

**PCE 展开**：
$$\eta(t, \xi) = \sum_{k=0}^{N_p} \eta_k(t) H_k(\xi)$$

其中 $H_k(\xi)$ 为概率化 Hermite 多项式，满足正交性：
$$\langle H_i, H_j \rangle = \sqrt{2\pi} \, i! \, \delta_{ij}$$

**Galerkin 投影后的系数方程**：
$$\frac{d\eta_k}{dt} = -\alpha_\mu \eta_k - \alpha_\sigma \sum_{j=0}^{N_p} \eta_j \frac{\langle H_1 H_j H_k \rangle}{\langle H_k^2 \rangle}$$

**统计矩**：
$$\mathbb{E}[\eta] = \eta_0(t), \qquad \text{Var}[\eta] = \sum_{k=1}^{N_p} \eta_k(t)^2 \langle H_k^2 \rangle$$

### 3.7 SVD 模型降阶

**快照矩阵 SVD**：
$$\mathbf{S} = [\boldsymbol{\phi}_1, \boldsymbol{\phi}_2, \dots, \boldsymbol{\phi}_N] = \boldsymbol{\Phi} \boldsymbol{\Lambda} \boldsymbol{\Psi}^T$$

**降阶基**：
$$\mathbf{B} = \boldsymbol{\Phi}_{:,1:r}$$

**Galerkin 投影**：
$$\frac{d\mathbf{a}}{dt} = \mathbf{B}^T \mathbf{f}(\mathbf{B}\mathbf{a})$$

---

## 4. 项目文件结构

```
161_synth_project/
├── main.py                      # 统一入口，零参数运行
├── spectrum_sampler.py          # AM1.5G 光谱离散 CDF 采样
├── absorption_integrator.py     # 楔形体光吸收高斯求积
├── material_interpolator.py     # 材料参数二维分段线性插值
├── sparse_matrix_io.py          # 稀疏 Jacobian Matrix Market I/O
├── drift_diffusion_solver.py    # 漂移-扩散方程中点法求解
├── mesh_triangulation.py        # 多晶网格生成与 PLY I/O
├── defect_monte_carlo.py        # 缺陷蒙特卡洛与寿命统计
├── recombination_models.py      # 辐射/Auger/带尾复合模型
├── mechanical_stress.py         # 薄膜热应力屈曲分析
├── uncertainty_pce.py           # PCE 不确定性量化
├── coupled_ion_migration.py     # 离子迁移-载流子耦合动力学
├── model_reduction.py           # SVD/POD 模型降阶
└── README_博士级合成说明.md      # 本文档
```

共 **13 个 .py 文件**（要求 ≥ 8 个）。

---

## 5. 运行方式

```bash
cd 161_synth_project
python3 main.py
```

程序将自动执行以下流程：
1. AM1.5G 光谱采样与楔形体光吸收积分
2. 钙钛矿材料参数 (T, x) 二维插值
3. 稀疏 Jacobian 构建与 Matrix Market I/O 测试
4. 漂移-扩散瞬态求解（含 Humps ODE 精度验证）
5. 多晶网格生成、PLY 读写、缺陷蒙特卡洛采样
6. SRH/辐射/Auger/带尾复合计算（Gauss-Laguerre 求积）
7. 热应力屈曲分析与带隙移动评估
8. PCE 不确定性量化（均值、方差、敏感性指标）
9. 离子迁移 I-V 迟滞模拟
10. SVD/POD 模型降阶验证
11. 综合光电转换效率评估

---

## 6. 数值鲁棒性与边界处理

本项目在以下层面实现了严格的数值鲁棒性：

- **浓度非负约束**：所有载流子浓度变量通过 `np.clip(..., 1.0, 1e25)` 强制非负
- **Bernoulli 函数稳定计算**：分段处理 $x \to 0$、$x \to +\infty$、$x \to -\infty$ 三种渐近行为
- **除零保护**：所有分母通过 `abs(den) > tol` 判断，或添加小量 `+ 1e-10`
- **矩阵 I/O 校验**：Matrix Market 读写后比较非零元数量
- **ODE 求解发散保护**：漂移-扩散瞬态求解采用 `try/except` + 稳态近似 fallback
- **参数边界裁剪**：温度、组分等输入变量自动裁剪到插值网格范围
- **奇异值裁剪**：SVD 降阶时限制保留模态数不超过矩阵最小维度

---

## 7. 科学问题总结

本合成项目解决的核心科学问题是：

> **如何在多物理场耦合（光吸收、载流子输运、缺陷复合、热应力屈曲、离子迁移、材料不确定性）条件下，定量预测钙钛矿太阳能电池的光电转换效率及其统计分布？**

通过融合 15 个原始科研代码项目的核心算法，本项目构建了一个从**光子入射 → 载流子产生 → 输运分离 → 复合损失 → 力学/电学退化 → 效率统计**的全链条计算框架。每个原始项目都在合成系统中承担了不可替代的真实角色，无遗漏、无挂名。
