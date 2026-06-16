# PROJECT_268 博士级合成说明

## 计算凝聚态: 强关联 Hubbard 模型量子蒙特卡洛
### 高阶有限差分与稳定性分析 (小规模可复现实验)

---

## 一、科学问题与物理背景

### 1.1 核心物理模型

本项目聚焦 **强关联 Hubbard 模型** 在 **三角晶格** 上的量子蒙特卡洛求解:

$$
\hat{H} = -t \sum_{\langle i,j\rangle,\sigma} \hat{c}^{\dagger}_{i\sigma} \hat{c}_{j\sigma}
        + U \sum_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow}
        - \mu \sum_i (\hat{n}_{i\uparrow} + \hat{n}_{i\downarrow})
$$

其中:
- $t$: 最近邻跳跃积分 (取为能量单位 $t = 1$)
- $U$: 在位库仑排斥 (刻画电子关联强度)
- $\mu$: 化学势 (控制粒子数密度)
- $\hat{c}^{\dagger}_{i\sigma}$, $\hat{c}_{i\sigma}$: 格点 $i$ 自旋 $\sigma$ 的费米子产生/湮灭算符
- $\hat{n}_{i\sigma} = \hat{c}^{\dagger}_{i\sigma} \hat{c}_{i\sigma}$: 粒子数算符

**三角晶格** 是最简单的几何阻挫系统: 反铁磁海森堡模型在三角晶格上具有 120° 自旋序,
而 Hubbard 模型展现出丰富的量子相图, 包括 Mott 绝缘体、自旋液体、非常规超导等.

### 1.2 数值方法: 行列式量子蒙特卡洛 (DQMC)

DQMC 是求解 Hubbard 模型最严格的数值方法之一:

1. **Trotter-Suzuki 分解**:
   $$e^{-\beta \hat{H}} \approx \left[e^{-\Delta\tau \hat{H}}\right]^L, \quad \beta = L\Delta\tau$$

2. **Hubbard-Stratonovich 变换**:
   $$e^{-\frac{\Delta\tau U}{2}(\hat{n}_\uparrow - \hat{n}_\downarrow)^2}
     = \frac{1}{2} \sum_{s=\pm1} e^{\alpha s (\hat{n}_\uparrow - \hat{n}_\downarrow)}$$
   其中 $\cosh(\alpha) = e^{\Delta\tau U/2}$.

3. **费米子迹**: 对费米子自由度精确积分, 得到行列式表达式:
   $$Z = \sum_{\{\sigma\}} \det[M_\uparrow(\sigma)] \det[M_\downarrow(\sigma)]$$

4. **Metropolis-Hastings 采样**: 以行列式比为接受概率, 抽样 HS 辅助场构型.

### 1.3 本项目的独特贡献

- **高阶有限差分**: 实现 2/4/6/8 阶中心差分模板, 用于虚时格林函数导数的高精度近似
- **稳定性分析**: 系统分析 B 矩阵乘积的条件数、CFL 条件、von Neumann 稳定性
- **热化诊断**: 融合冰川学 AAR 非平衡概念, 诊断 DQMC 马尔可夫链的平衡性
- **自适应提议**: 受强化学习启发, 动态调整提议步长达到最优接受率

---

## 二、原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 本项目中的物理角色 |
|------|-----------|---------|-------------------|
| 01 | `1360_truncated_normal` | 截断正态分布 | HS 辅助场的有界高斯采样, 防止数值溢出 |
| 02 | `1408_wedge_grid` | 楔形体网格 | 三维布里渊区 (层状 Hubbard 的 $k_z$ 方向) |
| 03 | `660_legendre_fast_rule` | Gauss-Legendre 求积 | 虚时积分 $\int_0^\beta d\tau$ 的高精度数值求积 |
| 04 | `1014_GlacierWeilin_AAR_disequilibrium` | AAR 非平衡诊断 | DQMC 热化诊断, 检测马尔可夫链是否达平衡 |
| 05 | `1320_triangle_to_fem` | 三角网格→FEM | 三角晶格实空间构造, 跳跃连接表生成 |
| 06 | `764_midpoint` | 隐式中点法 | 虚时 Wegner 流方程 $dH_\lambda/d\lambda$ 积分 |
| 07 | `362_fd1d_heat_steady` | 一维稳态 FD | Dyson 方程实空间类比 $-\nabla^2 G + \Sigma G = \delta$ |
| 08 | `293_disk_grid` | Fibonacci 圆盘网格 | 布里渊区内切圆盘采样, 低偏差准蒙特卡洛 |
| 09 | `1235_hrl-team_OptimismPerseveration` | 乐观-坚持学习 | DQMC 提议分布自适应 (接受率 → 最优 35%) |
| 10 | `123_burgers_pde_etdrk4` | ETD RK4 | 矩阵指数 $e^{-\Delta\tau H}$ 的 Padé/Taylor 实现 |
| 11 | `1352_triangulation_svg` | 三角剖分 | 晶格图论拓扑 (邻接矩阵、图直径、METIS 分割) |
| 12 | `798_nested_sequence_display` | 嵌套序列 | 多尺度虚时网格 (Level 0 到 Level L) |
| 13 | `218_coordinate_search` | 坐标直接搜索 | Mott 转变临界 $U_c$ 的参数优化 |
| 14 | `301_disk01_monte_carlo` | 圆盘 MC 积分 | 布里渊区圆盘上的物理量积分 |
| 15 | `796_neighbors_to_metis_graph` | METIS 图格式 | 晶格分割 (并行 DQMC 域分解) |

---

## 三、新增数学物理模型与核心公式

### 3.1 紧束缚色散关系

三角晶格的紧束缚色散 (含次近邻跳跃 $t'$):

$$
\varepsilon(\mathbf{k}) = -2t\left[\cos(\mathbf{k}\cdot\mathbf{a}_1) + \cos(\mathbf{k}\cdot\mathbf{a}_2) + \cos(\mathbf{k}\cdot(\mathbf{a}_1-\mathbf{a}_2))\right]
$$
$$
-2t'\left[\cos(\mathbf{k}\cdot(\mathbf{a}_1+\mathbf{a}_2)) + \cos(\mathbf{k}\cdot(2\mathbf{a}_1-\mathbf{a}_2)) + \cos(\mathbf{k}\cdot(\mathbf{a}_1-2\mathbf{a}_2))\right]
$$

范霍夫奇点条件: $\nabla_\mathbf{k}\varepsilon(\mathbf{k}) = 0$, 态密度出现对数发散.

### 3.2 高阶有限差分模板

**4阶中心差分 (一阶导数)**:
$$
f'(x) \approx \frac{f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)}{12h} + O(h^4)
$$

**6阶中心差分**:
$$
f'(x) \approx \frac{-f(x-3h) + 9f(x-2h) - 45f(x-h) + 45f(x+h) - 9f(x+2h) + f(x+3h)}{60h} + O(h^6)
$$

**8阶中心差分**:
$$
f'(x) \approx \frac{f(x-4h) - \frac{32}{3}f(x-3h) + 56f(x-2h) - 224f(x-h) + 224f(x+h) - 56f(x+2h) + \frac{32}{3}f(x+3h) - f(x+4h)}{280h} + O(h^8)
$$

### 3.3 Dyson 方程与自能

虚时 Matsubara 频率空间的 Dyson 方程:
$$
G^{-1}(i\omega_n) = G_0^{-1}(i\omega_n) - \Sigma(i\omega_n)
$$
$$
G_0^{-1}(i\omega_n) = i\omega_n + \mu - \varepsilon_\mathbf{k}
$$
$$
i\omega_n = \frac{(2n+1)\pi}{\beta} \quad \text{(费米子 Matsubara 频率)}
$$

### 3.4 稳定性判据

**CFL 条件** (对显式扩散方程):
$$
\frac{D\Delta\tau}{(\Delta x)^2} \leq \frac{1}{2d}
$$
其中 $d$ 为空间维度.

**von Neumann 放大因子**:
$$
G(k) = 1 - 4\frac{D\Delta\tau}{(\Delta x)^2}\sin^2\left(\frac{k\Delta x}{2}\right)
$$
稳定性: $|G(k)| \leq 1$ 对所有 $k$.

**SVD 稳定化判据**:
$$
\kappa(B) = \frac{\sigma_{\max}}{\sigma_{\min}} < \frac{1}{\sqrt{\varepsilon_{\text{machine}}}} \approx 10^8
$$

### 3.5 热化诊断 (AAR Disequilibrium)

冰川学中的积累区比率 (AAR) 类比到 DQMC:
$$
\text{AAR}(t) = \frac{\#\{s \in \text{window}: |O(s) - \langle O\rangle| < \kappa\sigma\}}{N_{\text{window}}}
$$
$$
\text{Disequilibrium} = 1 - \overline{\text{AAR}}
$$

积分自相关时间:
$$
\tau_{\text{int}} = \frac{1}{2} + \sum_{t=1}^{M} \rho(t), \quad \rho(t) = \frac{C(t)}{C(0)}
$$

### 3.6 Mott 转变判据 (Brinkman-Rice / Gutzwiller)

$$
D(U) = \frac{1}{4}\left(1 - \frac{U^2}{U_c^2}\right) \quad \text{for } U < U_c
$$
$$
U_{c2} \approx 2.94 W \quad (W = \text{带宽} \approx 12t \text{ 对三角晶格})
$$

---

## 四、修改与合成的文件清单

| 文件名 | 功能 | 融合的种子项目 |
|--------|------|----------------|
| `lattice_geometry.py` | 晶格几何、布里渊区、METIS 图 | [02][05][08][11][15] |
| `finite_difference.py` | 高阶 FD 模板、稳态求解、中点积分 | [06][07] |
| `hubbard_stratonovich.py` | HS 变换、截断正态、圆盘 MC | [01][14] |
| `greens_function.py` | 虚时格林函数、B 矩阵、GL 积分 | [03][10] |
| `determinant_qmc.py` | DQMC 主循环、嵌套虚时、自适应 | [09][12] |
| `stability_analysis.py` | 数值稳定性、CFL、von Neumann | (新增) |
| `parameter_optimizer.py` | 坐标搜索、Mott $U_c$ 定位 | [13] |
| `thermalization_diagnostics.py` | 热化、AAR 非平衡、自相关 | [04] |
| `observable_estimator.py` | 物理可观测量估计 | (新增) |
| `main.py` | 统一入口 (零参数) | (综合) |
| `__init__.py` | 包初始化 | - |

**统计**: 11 个 Python 文件, 总代码约 3500+ 行.

---

## 五、项目能够解决的科学问题

1. **Mott 金属-绝缘体转变**: 定位三角晶格 Hubbard 模型的临界 $U_c(t, T, n)$
2. **量子临界现象**: 研究 $T \to 0$ 极限下的量子相变
3. **几何阻挫效应**: 三角晶格的阻挫如何影响反铁磁序和自旋液体
4. **有限温度相图**: 通过改变 $\beta = 1/T$ 绘制温度-掺杂相图
5. **符号问题诊断**: 量化非 bipartite 晶格上 DQMC 的平均符号衰减
6. **数值方法基准**: 高阶有限差分的精度验证, 矩阵指数方法的稳定性对比

---

## 六、运行方法

### 6.1 环境要求
- Python 3.7+
- NumPy
- 无其他第三方依赖 (无 matplotlib, 无可視化)

### 6.2 运行命令

```bash
cd 268_synth_project_Advanced
python main.py
```

**零参数运行**: 直接执行 `python main.py` 即可运行完整的 8 阶段 DQMC 流水线:

1. 晶格与布里渊区构造
2. 高阶有限差分与流方程
3. HS 变换与蒙特卡洛积分
4. DQMC 核心模拟
5. 数值稳定性分析
6. 热化诊断
7. 参数优化 (Mott $U_c$ 定位)
8. 物理可观测量估计

### 6.3 输出

程序输出 8 个阶段的详细结果到标准输出, 包括:
- 晶格参数、能带结构、态密度
- 有限差分精度验证
- HS 耦合常数、截断正态采样统计
- DQMC 能量、双占据、平均符号、接受率
- CFL 条件、von Neumann 稳定性、矩阵指数精度
- 自相关时间、AAR 非平衡度、Gelman-Rubin $R̂$
- Mott $U_c$ 估计、U 扫描相图
- 最终能量分解、态密度、Gauss-Legendre 积分精度

### 6.4 关键物理参数

可在 `main.py` 中调整:
- `U`: 在位库仑排斥 (典型值 0~16t)
- `beta`: 逆温度 (典型值 1~20)
- `L`: 虚时切片数 (Trotter 数)
- `Lx, Ly`: 簇尺寸 (典型 4×4, 6×6, 8×8)
- `n_sweeps`: MC 扫掠数

---

## 七、代码架构特色

### 7.1 物理深度
- 完整实现 Hubbard 模型从第一性原理到 Monte Carlo 的全流程
- 包含解析公式 (色散关系、态密度、自能、Dyson 方程)
- 实现多尺度方法 (嵌套虚时网格、Richardson 外推)

### 7.2 数值鲁棒性
- 矩阵指数的三种方法 (Taylor/Padé/特征值分解) 交叉验证
- SVD 稳定化防止 B 矩阵乘积溢出
- Thomas 算法的零主元检测
- 所有物理量的边界条件处理

### 7.3 工程复杂度
- 模块化解耦: 11 个文件各司其职
- 类型注解与完整 docstring
- 错误处理 (LinAlgError, ValueError)
- 无可视化依赖 (纯数值计算)

### 7.4 可复现性
- 固定随机种子 (`seed=42`, `seed=12345`)
- 小规模系统 (4×4 簇, 16 格点) 可在秒级完成
- 所有中间结果打印到标准输出

---

## 八、扩展方向

1. **更大规模簇**: 6×6, 8×8, 配合 METIS 分割实现并行 DQMC
2. **UDT 稳定化**: 替代 SVD, 计算效率更高
3. **连续时间 DQMC (CT-DQMC)**: 消除 Trotter 误差
4. **解析延拓**: 最大熵方法从 $G(i\omega_n)$ 提取谱函数 $A(\omega)$
5. **多轨道 Hubbard**: 包含轨道自由度, 研究轨道选择性 Mott 转变

---

**项目完成日期**: 2026/06/07
**语言**: Python 3
**代码行数**: ~3500 行 (不含文档)
**可执行性**: ✓ 零参数运行通过
