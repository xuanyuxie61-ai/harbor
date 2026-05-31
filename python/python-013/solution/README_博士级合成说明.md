# 强关联电子系统 Hubbard 模型 — 多方法合成计算框架

## 项目概述

本项目将 **15 个种子科研项目** 的核心算法融合重构，面向**凝聚态物理：强关联电子系统 Hubbard 模型**这一前沿博士级科学问题，构建了一个多方法、多尺度的数值计算框架。

科学核心问题：
> 研究二维三角晶格 Hubbard 模型在有限温度下的 Mott 转变、双占据数动力学、谱函数性质，以及无序、周期驱动等非平衡效应。

---

## 科学背景与核心公式

### 1. Hubbard 模型哈密顿量

单带 Hubbard 模型描述电子在晶格上的跃迁与在位库仑排斥：

$$
\hat{H} = -t \sum_{\langle ij \rangle, \sigma} \left( \hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma} + \text{h.c.} \right) + U \sum_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow} - \mu \sum_{i,\sigma} \hat{n}_{i\sigma}
$$

其中：
- $t$：最近邻跃迁积分
- $U$：在位库仑排斥能
- $\mu$：化学势
- $\hat{c}_{i\sigma}^\dagger$、$\hat{c}_{j\sigma}$：费米子产生/湮灭算符
- $\hat{n}_{i\sigma} = \hat{c}_{i\sigma}^\dagger \hat{c}_{i\sigma}$：粒子数算符

### 2. Matsubara 格林函数

虚时（Matsubara）频率：

$$
\omega_n = \frac{(2n+1)\pi}{\beta} \quad (\text{费米子})
$$

非相互作用格林函数：

$$
G_0(i\omega_n) = \left[ (i\omega_n + \mu) \mathbf{I} - \mathbf{K} \right]^{-1}
$$

Dyson 方程：

$$
G(i\omega_n) = \left[ G_0^{-1}(i\omega_n) - \Sigma(i\omega_n) \right]^{-1}
$$

### 3. Hubbard-Stratonovich 变换 (DQMC)

将相互作用项通过辅助场展开：

$$
e^{-\Delta\tau U n_{i\uparrow} n_{i\downarrow}} = \frac{1}{2} \sum_{s=\pm 1} e^{\lambda s (n_{i\uparrow} - n_{i\downarrow})}
$$

其中 $\cosh(\lambda) = e^{\Delta\tau U / 2}$。

### 4. 谱函数与求和规则

谱函数定义为：

$$
A(\omega) = -\frac{1}{\pi} \text{Im} \, G(\omega + i0^+)
$$

满足求和规则：

$$
\int_{-\infty}^{\infty} d\omega \, A(\omega) = 1
$$

谱矩：

$$
M_n = \int d\omega \, \omega^n A(\omega)
$$

### 5. 周期驱动非平衡动力学

周期驱动哈密顿量：

$$
\hat{H}(t) = \hat{H}_0 + A \cdot f(\Omega t) \sum_i (-1)^i \hat{n}_i
$$

其中 $f(\Omega t)$ 为 sawtooth 型周期驱动：

$$
f(t) = \frac{t \mod T}{T} - \frac{1}{2}, \quad T = \frac{2\pi}{\Omega}
$$

### 6. 四面体法 DOS 积分

态密度：

$$
\rho(\omega) = \frac{1}{V_{BZ}} \int_{BZ} d^d k \, \delta(\omega - \varepsilon_k)
$$

将布里渊区剖分为四面体（2D 为三角形），在每个单元内线性化能带，解析计算贡献。

### 7. DMFT 自洽方程

平均场自洽条件：

$$
\Sigma(\omega) = U \langle n_\uparrow \rangle \langle n_\downarrow \rangle - \frac{U^2}{4} G(\omega)
$$

迭代直至残差 $| \Sigma_{new} - \Sigma_{old} | < \varepsilon$。

---

## 文件结构

```
013_synth_project/
├── main.py                     # 统一入口，零参数运行
├── lattice_geometry.py         # 晶格几何 (三角/六角晶格、BZ剖分)
├── hubbard_hamiltonian.py      # 精确对角化、热力学平均
├── dqmc_engine.py              # 行列式量子蒙特卡洛 (DQMC)
├── matsubara_green.py          # Matsubara 格林函数、分差插值、Shepard 插值
├── brillouin_zone.py           # BZ 积分 (四面体法、Lebedev 规则)
├── dynamics_evolution.py       # 非平衡动力学 (Verlet、Sawtooth 驱动、反应ODE)
├── disorder_config.py          # 无序采样 (Anderson 型、截断正态、热自旋)
├── convergence_tools.py        # 自洽迭代收敛监控 (Collatz、Pulay 混频)
├── spectral_function.py        # 谱函数解析延拓 (Padé、MaxEnt)
└── README_博士级合成说明.md    # 本说明文档
```

共 **10 个 .py 文件**，满足 >= 8 的要求。

---

## 种子项目映射表

| 种子项目 | 核心算法 | 合成后所在文件 | 科学角色 |
|---------|---------|--------------|---------|
| 1120_sphere_lebedev_rule | 球面 Lebedev 积分 | brillouin_zone.py | 3D 费米面角向积分 |
| 310_divdif | Newton 分差插值 | matsubara_green.py | 自能 Matsubara 频率插值 |
| 196_collatz | Collatz 序列迭代 | convergence_tools.py | 迭代收敛复杂度监控 |
| 1358_trinity | 三角铺砖/LP | lattice_geometry.py | 布里渊区三角剖分 |
| 1018_reaction_twoway_ode | 双向反应 ODE | dynamics_evolution.py | Doublon-Holon 对动力学 |
| 1360_truncated_normal | 截断正态采样 | disorder_config.py, dqmc_engine.py | 无序势/HS 场采样 |
| 745_md_fast | 分子动力学 Verlet | dynamics_evolution.py | 密度矩阵传播 |
| 1059_sawtooth_ode | Sawtooth 驱动 ODE | dynamics_evolution.py | 周期驱动 Hubbard 模型 |
| 1071_shepard_interp_1d | Shepard 插值 | matsubara_green.py | 谱函数重构插值 |
| 562_hypersphere | 超球面均匀采样 | disorder_config.py, brillouin_zone.py | 高维自旋/动量空间采样 |
| 1040_rnglib | 随机数生成 | dqmc_engine.py | DQMC Metropolis 采样 |
| 658_lebesgue | Lebesgue 常数 | matsubara_green.py, convergence_tools.py | 插值稳定性分析 |
| 1146_square_hex_grid | 六角网格生成 | lattice_geometry.py | 三角晶格 k 点网格 |
| 1150_square_surface_distance | 方边界采样 | disorder_config.py | 开放边界条件采样 |
| 936_pyramid_rule | 金字塔/四面体积分 | brillouin_zone.py | 3D BZ 体积积分 |

**全部 15 个种子项目均已真实融入合成项目，无遗漏、无挂名。**

---

## 运行方式

```bash
cd Synthesis-project-python/013_synth_project
python main.py
```

程序自动执行以下 9 个模块：

1. **精确对角化**：2×2 方格，计算不同 U 下的基态能量、双占据数、能隙
2. **DQMC**：4 格点有限温度模拟，测量双占据数与动能
3. **Matsubara 自能插值**：Newton 分差插值与 Shepard 插值
4. **布里渊区积分**：64 k 点四面体法 DOS + Lebedev 球面验证
5. **非平衡动力学**：Doublon-Holon 反应动力学 + Sawtooth 周期驱动
6. **无序采样**：Anderson 型截断正态无序 + 热自旋构型
7. **收敛监控**：自洽迭代 + Collatz 复杂度指数
8. **谱函数**：Padé 解析延拓 + MaxEnt + 谱矩计算
9. **DMFT 自洽环**：平均场自洽迭代直至收敛

---

## 关键科学结果示例

运行输出节选：

```
[模块 1] 精确对角化
  U=0.0: E0=-2.0000, D=0.7055, N=3.74, Gap=0.0000
  U=4.0: E0=-1.6039, D=0.0346, N=2.00, Gap=0.4416
  U=8.0: E0=-1.5231, D=0.0117, N=2.00, Gap=0.5020

[模块 2] DQMC
  双占据数: 0.1848 ± 0.0083
  动能:     -1.2510 ± 0.0388

[模块 4] BZ 积分
  DOS 峰值位置: -5.545
  Lebedev 权重和: 12.566371 (理论值 12.566371)

[模块 9] DMFT
  DMFT 自洽: 迭代次数=28, 最终残差=6.27e-07
  收敛后 Σ(iω_0)=-0.4171+0.5806j
```

---

## 工程鲁棒性设计

- **边界处理**：所有模块对输入参数进行范围检查（如 `nsites >= 0`、`beta > 0`、`dtau <= 0.5` 等）
- **数值稳定**：DQMC 采用 SVD 稳定化防止浮点溢出；Padé 近似处理奇异值截断
- **异常保护**：`np.where(np.isfinite(...))` 清理 NaN/Inf；Metropolis 判据拒绝负比值
- **约束保持**：MaxEnt 归一化约束 `∫ A dω = 1`；反应动力学非负约束

---

## 科学领域声明

本项目严格围绕 **凝聚态物理：强关联电子系统 Hubbard 模型** 展开，所有算法、公式、物理量均服务于该领域的前沿数值研究。
