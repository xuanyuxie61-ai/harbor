# README_博士级合成说明.md

## 膜蛋白嵌入与药物分子对接 — 博士级 Python 科研代码合成项目

**项目编号**: PROJECT_112  
**科学领域**: 分子动力学（Molecular Dynamics）— 膜蛋白嵌入与药物分子对接  
**合成语言**: Python 3  
**文件总数**: 10 个 `.py` 模块 + 1 个统一入口 `main.py`

---

## 一、科学问题概述

本项目围绕**膜蛋白（GPCR）在脂质双层中的嵌入**以及**小分子药物在跨膜结合口袋中的分子对接**展开，构建了一套从静电场计算、脂质空间排布优化、药物构象搜索、结合自由能估算到粗粒化分子动力学模拟的完整博士级计算流程。

### 1.1 核心物理模型

**Poisson-Boltzmann 方程（跨膜静电势）**

$$-\nabla \cdot \bigl(\epsilon(z) \nabla \phi\bigr) + \kappa^2(z) \phi = 4\pi \rho(z)$$

其中：
- $\epsilon(z)$ 为介电常数剖面（水域 80、蛋白 4、膜脂 2）
- $\kappa(z)$ 为 Debye-Hückel 屏蔽参数
- $\rho(z)$ 为电荷密度分布
- 边界条件为齐次 Neumann 条件：$\displaystyle \frac{\partial \phi}{\partial z}\bigg|_{z_{\min}} = \frac{\partial \phi}{\partial z}\bigg|_{z_{\max}} = 0$

**Lennard-Jones 势能**

$$U_{LJ}(r) = 4\epsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]$$

**Debye-Hückel 屏蔽 Coulomb 势能**

$$U_{elec}(r) = \frac{q_i q_j}{4\pi\epsilon_0 \epsilon_r} \frac{e^{-\kappa r}}{r}$$

**结合自由能热力学积分（Thermodynamic Integration, TI）**

$$\Delta G_{\text{bind}} = \int_0^1 \left\langle \frac{\partial H(\lambda)}{\partial \lambda} \right\rangle_\lambda \, d\lambda$$

其中耦合哈密顿量 $H(\lambda) = H_{\text{protein}} + H_{\text{drug}} + \lambda H_{\text{int}}$。

**速度 Verlet 积分器**

$$\begin{aligned}
\mathbf{r}(t+\Delta t) &= \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \frac{1}{2}\mathbf{a}(t)\Delta t^2 \\
\mathbf{v}(t+\Delta t) &= \mathbf{v}(t) + \frac{1}{2}\bigl[\mathbf{a}(t) + \mathbf{a}(t+\Delta t)\bigr]\Delta t
\end{aligned}$$

---

## 二、种子项目映射关系

本项目严格基于 15 个输入种子项目的核心算法进行融合，**每一个种子项目均已真实融入**，无遗漏、无挂名。

| 序号 | 种子项目 | 核心算法 | 合成项目中的角色 |
|------|----------|----------|-----------------|
| 1 | `461_gegenbauer_exactness` | Gegenbauer 权积分、超几何函数 $_2F_1$、Digamma $\psi$ | `special_functions.py`：静电势径向展开的 Gegenbauer 多项式正交性验证与特殊函数计算 |
| 2 | `1109_sparse_grid_total_poly` | 总阶数稀疏网格构造、组合枚举 `comp_next` | `free_energy_integration.py`：高维构象空间上的结合自由能稀疏网格积分 |
| 3 | `1365_tsp_greedy` | 贪心 TSP 路径搜索 | `conformation_search.py`：药物分子可旋转键状态的贪心构象搜索 |
| 4 | `459_ge_to_st` | 稠密矩阵到 Sparse Triplet 转换 | `sparse_matrix.py`：大规模 MD 系统哈密顿量的稀疏存储与 CSR 格式转换 |
| 5 | `791_ncm` | Bessel 函数膜振动、随机采样、ODE 积分 | `special_functions.py`（Bessel 膜振动模式）、`molecular_dynamics.py`（ODE 积分器） |
| 6 | `387_fem1d_bvp_quadratic` | 二次有限元 BVP 求解、Gauss-Legendre 积分 | `membrane_fem.py`：Poisson-Boltzmann 方程的二次 FEM 离散化 |
| 7 | `641_laguerre_polynomial` | Laguerre-Gauss 求积、Jacobi 矩阵对角化 `imtqlx` | `quadrature_rules.py`：Laguerre 求积用于径向 Coulomb 积分 |
| 8 | `1059_sawtooth_ode` | 锯齿波周期驱动 ODE | `molecular_dynamics.py`：锯齿波热浴（速度 rescaling 的周期性扰动） |
| 9 | `377_fem_neumann` | Neumann 边界反应-扩散、质量/刚度矩阵、非线性项 | `membrane_fem.py`：Neumann BC 处理、反应-扩散非线性项离散 |
| 10 | `202_combo` | 回溯搜索、排列/组合/子集枚举 | `conformation_search.py`：构象空间回溯剪枝搜索 |
| 11 | `262_cvt_triangle_uniform` | 2D 三角形 CVT（Lloyd 算法） | `lipid_cvt.py`：膜脂头基在双层平面内的 CVT 优化布置 |
| 12 | `1324_triangle_wandzura_rule` | Wandzura 高阶三角形对称求积 | `quadrature_rules.py`：膜表面自由能的三角形高阶数值积分 |
| 13 | `249_cvt_3d_lumping` | 3D 密度加权 CVT | `lipid_cvt.py`：水分子/脂酰链尾部的三维密度加权 CVT |
| 14 | `1225_test_partial_digest` | 部分消化问题（PDP）、成对距离约束 | `structure_validation.py`：蛋白骨架距离约束验证与 PDP 测试 |
| 15 | `1054_sandia_rules` | Clenshaw-Curtis、Jacobi、Hermite、Gen-Hermite、Laguerre 求积 | `quadrature_rules.py`：多种 1D 求积规则库 |

---

## 三、新增数学物理模型与核心公式

### 3.1 Gegenbauer 权积分与特殊函数

Gegenbauer 权积分：
$$I_{\alpha}(n) = \int_{-1}^{+1} x^n (1-x^2)^{\alpha} \, dx$$

解析解（偶次幂）：
$$I_{\alpha}(n) = 2 \frac{\Gamma(1+n)\Gamma(1+\alpha)}{\Gamma(2+\alpha+n)} \cdot {}_2F_1(-\alpha, 1+n; 2+\alpha+n; -1)$$

超几何函数 $_2F_1$：
$${}_2F_1(a,b;c;z) = \sum_{k=0}^{\infty} \frac{(a)_k (b)_k}{(c)_k} \frac{z^k}{k!}$$

Digamma 函数：
$$\psi(z) = \frac{d}{dz}\ln\Gamma(z) = \frac{\Gamma'(z)}{\Gamma(z)}$$

### 3.2 有限元离散化（Poisson-Boltzmann）

变分形式：
$$\int_{\Omega} \epsilon(x) \nabla u \cdot \nabla v \, dx + \int_{\Omega} \kappa^2(x) u v \, dx = \int_{\Omega} f(x) v \, dx, \quad \forall v \in H^1(\Omega)$$

二次有限元参考单元形函数（在 $[-1,1]$ 上）：
$$\begin{aligned}
N_1(\xi) &= \frac{1}{2}\xi(\xi-1) \\
N_2(\xi) &= 1-\xi^2 \\
N_3(\xi) &= \frac{1}{2}\xi(\xi+1)
\end{aligned}$$

局部刚度矩阵元素：
$$A_{ij}^{(e)} = \sum_{q} w_q \left[ \epsilon(x_q) N'_i(x_q) N'_j(x_q) + \kappa^2(x_q) N_i(x_q) N_j(x_q) \right]$$

### 3.3 Centroidal Voronoi Tessellation（CVT）

能量泛函：
$$\mathcal{F}(G) = \sum_{i=1}^{n} \int_{V_i} \rho(\mathbf{x}) \|\mathbf{x} - \mathbf{g}_i\|^2 \, d\mathbf{x}$$

Lloyd 算法迭代：
$$\mathbf{g}_i^{(k+1)} = \frac{\displaystyle\int_{V_i^{(k)}} \mathbf{x} \rho(\mathbf{x}) \, d\mathbf{x}}{\displaystyle\int_{V_i^{(k)}} \rho(\mathbf{x}) \, d\mathbf{x}}$$

### 3.4 稀疏网格多变量求积

总阶数稀疏网格：
$$Q_{L,d}^{(\text{sparse})}(f) = \sum_{|\ell|_1 \le L} \bigotimes_{i=1}^{d} Q_{\ell_i}^{(i)}(f) - \text{（嵌套点去重）}$$

其中 $Q_{\ell}^{(i)}$ 为第 $i$ 维的 1D Clenshaw-Curtis 求积：
$$Q_{n}^{CC}(f) = \sum_{j=1}^{n} w_j^{(n)} f(x_j^{(n)})$$

节点：
$$x_j^{(n)} = \cos\left(\frac{(j-1)\pi}{n-1}\right), \quad j=1,\dots,n$$

### 3.5 Jacobi-Gauss 求积（Elhay-Kautsky 算法）

积分：
$$\int_{-1}^{1} (1-x)^{\alpha}(1+x)^{\beta} f(x)\, dx \approx \sum_{i=1}^{n} w_i f(x_i)$$

Jacobi 矩阵递推：
$$\begin{aligned}
a_1 &= \frac{\beta-\alpha}{2+\alpha+\beta} \\
b_1^2 &= \frac{4(1+\alpha)(1+\beta)}{(3+\alpha+\beta)(2+\alpha+\beta)^2} \\
a_i &= \frac{(\beta+\alpha)(\beta-\alpha)}{(abi-2)abi}, \quad abi = 2i+\alpha+\beta \\
b_i^2 &= \frac{4i(i+\alpha)(i+\beta)(i+\alpha+\beta)}{(abi-1)(abi+1)abi^2}
\end{aligned}$$

求积节点与权重通过对称三对角矩阵的特征值分解获得（Golub-Welsch 算法）。

### 3.6 三角形高阶求积（Wandzura 规则）

Wandzura 6 阶规则（6 点，精确到 5 次多项式）：
$$\int_T f(x,y)\, dA = |J| \sum_{k=1}^{6} w_k f(x_k, y_k)$$

重心坐标到物理坐标的等参映射：
$$\mathbf{x} = L_1 \mathbf{v}_1 + L_2 \mathbf{v}_2 + L_3 \mathbf{v}_3$$

### 3.7 粗粒化 MD 力场

力场总能量：
$$U_{\text{total}} = U_{LJ} + U_{elec} + U_{spring} + U_{torsional}$$

LJ 力（带软核截断）：
$$\mathbf{F}_{ij}^{LJ} = 24\epsilon \left[2\left(\frac{\sigma}{r_{ij}}\right)^{12} - \left(\frac{\sigma}{r_{ij}}\right)^{6}\right] \frac{\mathbf{r}_{ij}}{r_{ij}^2}, \quad r_{ij} \ge 0.8\sigma$$

Debye-Hückel 力：
$$\mathbf{F}_{ij}^{elec} = -\frac{q_i q_j}{4\pi\epsilon_0\epsilon_r} \frac{e^{-\kappa r}(\kappa r + 1)}{r^3} \mathbf{r}_{ij}$$

Berendsen 弱耦合热浴：
$$\lambda_B = \sqrt{1 + \frac{\Delta t}{\tau}\left(\frac{T_{\text{target}}}{T_{\text{inst}}} - 1\right)}$$

---

## 四、项目文件结构

```
112_synth_project/
├── main.py                          # 统一入口，零参数运行
├── special_functions.py             # 特殊函数库（2F1, psi, Bessel, Coulomb Green）
├── sparse_matrix.py                 # 稀疏矩阵 GE/ST/CSR 格式转换与 SpMV
├── quadrature_rules.py              # 多类求积规则（CC, Jacobi, Hermite, Laguerre, Wandzura）
├── membrane_fem.py                  # FEM Poisson-Boltzmann 与反应-扩散求解器
├── lipid_cvt.py                     # 2D/3D CVT 脂质布置优化
├── conformation_search.py           # 药物构象贪心搜索与回溯枚举
├── free_energy_integration.py       # 稀疏网格积分与热力学积分（TI）
├── structure_validation.py          # 距离矩阵验证与骨架约束检查
├── molecular_dynamics.py            # 粗粒化 MD 模拟（Verlet + 锯齿波热浴）
└── README_博士级合成说明.md          # 本文档
```

---

## 五、各模块边界处理与数值鲁棒性

1. **`special_functions.py`**
   - `r8_hyper_2f1`：检查 $c$ 不得为非正整数，$x < 1.0$，不收敛时回退到级数展开
   - `gegenbauer_integral`：检查 $\alpha > -1.0$，$\text{expon} \ge 0$
   - `screened_coulomb_green`：$r \to 0$ 时采用正则化极限 $G(0) = \kappa/(4\pi\epsilon)$

2. **`sparse_matrix.py`**
   - `from_dense`：支持 `drop_tol` 阈值丢弃极小元
   - `spmv`：检查向量维度匹配
   - `to_csr`：合并重复位置的元素（求和）

3. **`quadrature_rules.py`**
   - `clenshaw_curtis_compute`：$n=1$ 时返回单点中点规则
   - `jacobi_compute`：检查 $\alpha, \beta > -1$
   - `integrate_triangle`：检查顶点 shape 为 $(3,2)$，Jacobian 行列式为面积的两倍

4. **`membrane_fem.py`**
   - `fem1d_bvp_quadratic`：检查 $n$ 为奇数且 $\ge 3$，网格单调递增
   - 支持 Dirichlet 与 Neumann 两种边界条件
   - 反应-扩散非线性项 `reaction_diffusion_nonlinear`：检查 `c_array` 长度与 `w` 维度

5. **`lipid_cvt.py`**
   - `cvt_triangle_uniform`：空 Voronoi 区域时自动重新随机放置生成元
   - `cvt_3d_lumping`：密度函数截断至 $[10^{-12}, 10]$ 避免奇点

6. **`conformation_search.py`**
   - `greedy_conformation_search`：检查能量网格维度一致性
   - `backtrack_search`：支持前向检查剪枝，`max_solutions` 上限防止组合爆炸

7. **`free_energy_integration.py`**
   - `comp_next`：边界保护（$h \ge k$ 时自动终止），$n=0$ 时正确返回 `more=False`
   - `sparse_grid_integrate`：小维度时自动切换全张量积，大维度使用稀疏网格
   - `thermodynamic_integration_binding_free_energy`：分母配分函数为零时返回安全默认值

8. **`structure_validation.py`**
   - `validate_distance_matrix`：检查非负性、零对角线、对称性、三角不等式
   - `validate_backbone_distances`：残基索引越界检查

9. **`molecular_dynamics.py`**
   - LJ 软核截断：$r < 0.8\sigma$ 时取 $r = 0.8\sigma$
   - 力的大小上限：每粒子 500 kcal/(mol·Å)
   - 静电力上限：±100 kcal/(mol·Å)
   - 全局弱阻尼：每步速度乘以 0.9995
   - Berendsen 耦合系数截断至 $[0.95, 1.05]$ 防止温度振荡
   - 反射边界条件：位置与速度同时修正
   - 单位换算：amu·(Å/ps)² → kcal/mol 使用标准因子 0.00239006

---

## 六、运行方式

```bash
cd 112_synth_project
python main.py
```

程序将依次执行以下 10 个步骤并输出结果：
1. 特殊函数与正交多项式验证
2. 稀疏矩阵与 FEM 质量/刚度矩阵组装
3. Poisson-Boltzmann 跨膜电势剖面求解
4. 膜脂双层 CVT 优化布置
5. 药物分子构象贪心搜索
6. 结合自由能热力学积分
7. 粗粒化分子动力学模拟
8. 蛋白骨架距离约束验证
9. 综合统计与收敛性分析
10. 最终汇总

---

## 七、科学问题的前沿性与博士级难度

本项目融合了以下前沿科学计算要素：

- **多尺度建模**：从原子尺度的 FEM 静电场（Poisson-Boltzmann）到介观尺度的粗粒化 MD，再到宏观尺度的自由能热力学积分
- **高维数值积分**：稀疏网格（Sparse Grid）将指数增长的维度灾难降低到多项式复杂度，是处理高维构象空间的核心工具
- **非凸优化**：CVT 的 Lloyd 算法用于脂质分子的空间排布优化，属于非凸能量泛函的变分极小化
- **组合爆炸与剪枝**：药物分子构象搜索需在 $m^n$（$m \sim 12$, $n \sim 5$）的状态空间中寻找能量极小，结合了贪心策略与回溯剪枝
- **特殊函数的数值稳定性**：超几何函数 $_2F_1$ 在 $x \to 1$ 时的渐近行为、Digamma 函数在负半轴的反射公式等，均需精细的边界处理
- **刚性 ODE 与热浴耦合**：MD 积分器需在 LJ 短程强排斥（时间尺度 $\sim 10^{-15}$ s）与热浴慢变扰动之间保持数值稳定

---

## 八、参考文献

1. John Burkardt, *GEGENBAUER_EXACTNESS*, MIT License, 2009.
2. Fabio Nobile, Raul Tempone, Clayton Webster, "A Sparse Grid Stochastic Collocation Method for PDEs with Random Input Data", *SIAM J. Numer. Anal.*, 46(5), 2008.
3. Albert Nijenhuis, Herbert Wilf, *Combinatorial Algorithms for Computers and Calculators*, 2nd Ed., Academic Press, 1978.
4. Qiang Du, Vance Faber, Max Gunzburger, "Centroidal Voronoi Tessellations: Applications and Algorithms", *SIAM Review*, 41(4), 1999.
5. Stephen Wandzura, Hong Xiao, "Symmetric Quadrature Rules on a Triangle", *Comput. Math. Appl.*, 45(12), 2003.
6. Sylvan Elhay, Jaroslav Kautsky, "Algorithm 655: IQPACK", *ACM TOMS*, 13(4), 1987.
7. John D. Cook, "Driving vibrations with sawtooth waves", 2020.
8. Jeffrey Borggaard, John Burkardt, John Burns, Eugene Cliff, *Working Notes on a Reaction Diffusion Model: a Finite Element Formulation*.
