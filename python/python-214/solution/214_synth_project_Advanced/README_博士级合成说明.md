# PROJECT 214 — 随机椭圆 PDE 的 L1 稀疏 PCE 恢复

> **博士级科学计算合成项目**
> **科学领域**: 数学优化 · 稀疏优化与 L1 正则化
> **应用对象**: 随机偏微分方程的不确定性量化 (UQ)

---

## 一、项目概述

本项目融合 15 个种子项目的核心算法, 围绕 **L1 正则化稀疏优化** 构建了一个前沿博士级科学计算问题:

> **从有限观测中恢复随机椭圆 PDE 的稀疏多项式混沌展开 (PCE) 系数**

数学模型:

$$
\begin{aligned}
    &-\nabla \cdot \bigl(a(x,\omega)\,\nabla u(x,\omega)\bigr) = f(x)
        \quad \text{in } \Omega = (0,1),\\
    &u(0,\omega) = u(1,\omega) = 0,
\end{aligned}
$$

其中扩散系数 $a(x,\omega)$ 为随机场, 具有稀疏 PCE 展开:

$$
a(x,\omega) = a_0(x) + \sum_{|\alpha|\le p} a_\alpha(x)\,\Psi_\alpha(\omega),
$$

仅少数 $a_\alpha$ 显著非零. 本程序从观测 $y = \Psi c + \varepsilon$ 中恢复稀疏系数 $c$:

$$
\min_{c}\; \frac{1}{2}\|\Psi c - y\|_2^2 + \lambda \|c\|_1.
$$

---

## 二、15 个种子项目的融合映射

| # | 种子项目 | 核心算法 | 本项目中的角色 |
|---|---------|---------|--------------|
| 1 | `777_monomial_value` | 单项式求值 | `sparse_basis.py` — Vandermonde 矩阵构造 |
| 2 | `662_legendre_product` | Legendre 正交性与乘积 | `sparse_basis.py` — 三项递推求值 + 正交性验证 |
| 3 | `605_jacobi_exactness` | Jacobi 求积精确性 | `sparse_basis.py` — Golub-Welsch 算法 + 精确度验证 |
| 4 | `387_fem1d_bvp_quadratic` | 1D 二次有限元 | `fem_forward.py` — 刚度矩阵装配 + L2/H1/L∞ 误差 |
| 5 | `548_human_mesh2d` | 2D 三角网格 | `measurement_operator.py` — 采样几何构造 |
| 6 | `891_polygonal_surface_display` | 多边形曲面离散 | `measurement_operator.py` — 面观测积分 |
| 7 | `681_line_integrals` | 线积分求值 | `measurement_operator.py` — 线观测算子构造 |
| 8 | `076_bellman_ford` | 单源最短路径 | `graph_sparsity.py` — 支撑集依赖图分析 + 负环检测 |
| 9 | `303_disk01_positive_monte_carlo` | 正圆盘 MC 积分 | `monte_carlo_feynman_kac.py` — 期望风险 MC 验证 |
|10 | `423_feynman_kac_2d` | 2D Feynman-Kac | `monte_carlo_feynman_kac.py` — 概率 PDE 验证 |
|11 | `667_levels` | 水平集计算 | `levelset_support.py` — 支撑集持续同调识别 |
|12 | `801_newton_maehly` | Newton-Maehly 降阶求根 | `regularization_path.py` — 路径折点检测 |
|13 | `1070_ChemGAN-challenge` | GAN 对抗生成 | `gan_sparse_prior.py` — 结构化稀疏先验学习 |
|14 | `1115_sandyherdo_inerOsci` | 惯性振子动力学 | `l1_ista.py` — Heavy-ball 近端加速 |
|15 | `1374_unstable_ode` | 不稳定 ODE 稳定化 | `l1_ista.py` — 隐式中点稳定化 |

---

## 三、核心数学公式

### 3.1 多项式基函数

**Legendre 三项递推**:
$$
(k+1)P_{k+1}(x) = (2k+1)\,x\,P_k(x) - k\,P_{k-1}(x),\quad P_0=1,\;P_1=x.
$$

**Jacobi 递推** ($P_k^{(\alpha,\beta)}$):
$$
P_{k+1} = (a_k x + b_k) P_k - c_k P_{k-1},
$$
$$
a_k = \frac{(2k+1+\alpha+\beta)(2k+2+\alpha+\beta)}{2(k+1)(k+1+\alpha+\beta)},
$$
$$
b_k = \frac{(\beta^2-\alpha^2)(2k+1+\alpha+\beta)}{2(k+1)(k+1+\alpha+\beta)(2k+\alpha+\beta)},
$$
$$
c_k = \frac{(k+\alpha)(k+\beta)(2k+2+\alpha+\beta)}{(k+1)(k+1+\alpha+\beta)(2k+\alpha+\beta)}.
$$

**多指标双曲交叉截断**:
$$
\mathcal{I}_p^{d,q} = \Bigl\{\alpha\in\mathbb{N}^d : \sum_{j=1}^d \alpha_j^{1/q} \le p\Bigr\},\quad 0<q\le 1.
$$

### 3.2 L1 优化求解器

**LASSO 问题**:
$$
\min_x\; F(x) \triangleq \frac{1}{2}\|Ax - y\|_2^2 + \lambda\|x\|_1.
$$

**软阈值算子** ($\text{prox}_{\kappa\|\cdot\|_1}$):
$$
S_\kappa(v) = \text{sign}(v)\cdot\max(|v|-\kappa,\; 0).
$$

**ISTA 迭代**:
$$
x^{k+1} = S_{\lambda/L}\bigl(x^k - \tfrac{1}{L}A^T(Ax^k - y)\bigr),\quad L = \|A\|_2^2.
$$

**FISTA 加速** (Beck-Teboulle 2009):
$$
y^k = x^k + \frac{t_k-1}{t_{k+1}}(x^k - x^{k-1}),\quad x^{k+1} = S_{\lambda/L}\bigl(y^k - \tfrac{1}{L}\nabla f(y^k)\bigr),
$$
$$
t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2},\quad \text{收敛率 } F(x^k) - F^* \le \frac{2L\|x^0-x^*\|^2}{(k+1)^2}.
$$

**Heavy-ball 惯性近端** (源自 1115 惯性振子):
$$
m\ddot x + \gamma\dot x + A^T(Ax-y) + \lambda\,\partial\|x\|_1 \ni 0,
$$
离散化:
$$
x^{k+1} = S_{\alpha\lambda}\bigl(x^k - \alpha\nabla f(x^k) + \beta(x^k - x^{k-1})\bigr),
$$
$$
\alpha < 2/L,\quad \beta < 1 - \sqrt{\alpha L}.
$$

**隐式中点稳定化** (源自 1374):
$$
x^{k+1} = x^k + h\cdot g\bigl(\tfrac{1}{2}(x^k + x^{k+1})\bigr),
$$
内层 Newton 求解.

### 3.3 正则化路径

**λ 上界**:
$$
\lambda_{\max} = \|A^T y\|_\infty,\quad \lambda > \lambda_{\max} \Rightarrow x^*(\lambda) = 0.
$$

**BIC 准则**:
$$
\text{BIC}(\lambda) = m\log\frac{\|Ax_\lambda - y\|^2}{m} + |S_\lambda|\log m.
$$

**Newton-Maehly 降阶求根** (路径折点检测):
$$
g(x) = \frac{f(x)}{\prod_{i=1}^k (x - r_i)},\quad
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n) - f(x_n)\sum_j \frac{1}{x_n - r_j}}.
$$

### 3.4 支撑集持续同调

水平集扫描 $\tau$: $S_\tau = \{i : |x_i| > \tau\}$. 持续图:

$$
\text{birth}_k = |x_{(k)}|,\quad \text{death}_k = |x_{(k+1)}|,\quad \text{lifetime}_k = \text{birth}_k - \text{death}_k.
$$

### 3.5 Feynman-Kac 概率表示

Poisson 方程 $-\frac{1}{2}\Delta u = f$ in $D$, $u|_{\partial D} = g$:
$$
u(x_0) = \mathbb{E}\biggl[\int_0^\tau f(B_s)\,ds + g(B_\tau)\biggr],
$$
$B_s$ 为 2D Brown 运动, $\tau$ 为首次逸出时间.

### 3.6 图总变差正则化

$$
\|x\|_{\text{GTV}} = \sum_{(i,j)\in E} w_{ij}|x_i - x_j|.
$$

### 3.7 Bellman-Ford 负环检测

$$
d[v] \leftarrow \min\bigl(d[v],\; d[u] + w(u,v)\bigr),\quad \forall (u,v)\in E.
$$

若第 $n$ 轮仍可松弛, 则存在负环 (对应于目标函数无界下降方向).

---

## 四、代码结构

```
214_synth_project_Advanced/
├── main.py                      # 统一入口 (11 阶段流水线)
├── sparse_basis.py              # 多项式基构造 (Legendre/Jacobi/monomial/双曲交叉)
├── measurement_operator.py      # 观测算子 (点/线/面, RIP 诊断)
├── fem_forward.py               # 1D 二次 FEM 正演 + 误差估计
├── l1_ista.py                   # ISTA/FISTA/Heavy-ball/隐式中点 求解器
├── regularization_path.py       # 正则化路径追踪 + Newton-Maehly 折点检测
├── monte_carlo_feynman_kac.py   # MC 积分 + Feynman-Kac 验证
├── levelset_support.py          # 水平集支撑集识别 + 持续同调
├── graph_sparsity.py            # 图结构稀疏性 + Bellman-Ford
├── gan_sparse_prior.py          # GAN 结构化稀疏先验
├── diagnostics.py               # 收敛与恢复质量诊断
└── README_博士级合成说明.md     # 本文档
```

---

## 五、运行方法

本项目为纯 Python, 依赖 numpy 与 scipy. 零参数直接运行:

```bash
cd /mnt/data/zpy/sci-swe/source code/Synthesis-project-python/214_synth_project/214_synth_project_Advanced
python main.py
```

程序自动执行 11 阶段流水线:
1. 稀疏多项式基构造 (Legendre/Jacobi 正交性验证)
2. 观测算子构造 (点/线/面观测, RIP 诊断)
3. 合成真解与观测数据
4. 正则化路径追踪 (BIC/GCV 选择)
5. 多算法对比 (ISTA / FISTA / Heavy-ball / 隐式中点)
6. 支撑集识别 (持续同调 + Otsu + MAD)
7. GAN 结构化先验学习 + 加权 LASSO
8. 图结构稀疏性分析 (Bellman-Ford)
9. 1D FEM 正演 + 误差估计
10. 蒙特卡洛 + Feynman-Kac 概率验证
11. 综合诊断报告

---

## 六、科学问题与输出

**解决的科学问题**: 从有限带噪观测中稳健恢复高维随机 PDE 的稀疏 PCE 系数, 实现不确定性量化.

**输出指标**:
- 相对重构误差 $\|x - x^*\|/\|x^*\|$
- 支撑集精度/召回/F1-score
- KKT 最优性证书违反量
- 正则化路径折点位置
- Bellman-Ford 负环检测结果 (稀疏结构稳定性)
- Feynman-Kac 概率验证误差
- 多算法收敛率对比

---

## 七、算法唯一性与差异化

本项目的独特方法论:
1. **L1 近端算法的振子视角**: 将 FISTA 的 Nesterov 动量解释为惯性振子离散化 (1115)
2. **不稳定 ODE 的稳定化**: 当 Lipschitz 常数大时采用隐式中点 (1374)
3. **持续同调支撑集识别**: 用水平集拓扑替代硬阈值 (667)
4. **GAN 先验指导的加权 L1**: 学习稀疏结构, 降低惩罚 (1070)
5. **Bellman-Ford 依赖图分析**: 将稀疏结构视为图, 检测负环 (076)
6. **Feynman-Kac 概率验证**: 独立于代数的 PDE 解验证 (423)

---

## 八、边界与鲁棒性

- 所有除法均含 `eps` 保护 (如 `1.0 / max(..., 1e-14)`)
- 幂迭代 Lipschitz 估计含下界保护
- FISTA 重启机制防止振荡
- Bellman-Ford 处理浮点边权
- 隐式中点含自适应步长缩小
- GAN 梯度裁剪防爆炸
- 水平集扫描含 eps 容差

---

## 九、关键数值结果 (典型运行)

| 指标 | 数值 |
|------|------|
| 基维数 M (双曲交叉) | 11 |
| 观测维数 m | 84 |
| 真解稀疏度 | 5 |
| FISTA 相对误差 | 1.33e-2 |
| 支撑集 F1 | 0.714 |
| MAD 阈值支撑集 F1 | 1.000 |
| FEM L2 误差 (32 vs 128 单元) | 7.11e-6 |
| Feynman-Kac 相对误差 | ~4.8e-1 (MC 估计) |
| 圆盘面积 MC | 0.783 (理论 π/4 = 0.785) |
