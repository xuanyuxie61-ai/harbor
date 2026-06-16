# 格点 QCD 有限温相变模拟 · 博士级合成说明

## 项目概述

本项目围绕 **格点量子色动力学 (Lattice QCD) 有限温相变模拟** 这一前沿博士级科学计算问题，融合 15 个输入种子项目的核心算法，构建了一个完整的纯 SU(3) 规范理论 HMC (混合蒙特卡罗) 模拟框架，并重点研究：

1. **高阶有限差分格式** 对格点费米子离散化的改进 (Naik, 4/6/8 阶中心差分)
2. **辛积分器的稳定性分析** (Leapfrog, Omelyan, Forest-Ruth, Yoshida)
3. **退禁闭相变** 的序参量测量与临界现象分析
4. **Symanzik 改进作用量** (Wilson → Lüscher-Weisz) 的系统实现

## 15 个种子项目到科学问题的深度映射

| 种子项目 | 核心算法 | 在 Lattice QCD 中的角色 |
|---------|---------|------------------------|
| **1333_triangulation_boundary_nodes** | 边界节点识别 (有向边计数) | 4D 超立方体格点的时间边界识别, Polyakov loop 端点定位 |
| **631_l4lib** | 布尔 XOR 位运算 | 站点 parity 计算: p(x) = XOR(x₀,x₁,x₂,x₃), 用于 even-odd 预处理 |
| **977_r8col** | 列排序, 去重, 堆排序 | 观测值序列的排序统计, 分位数与分位数分析 |
| **478_gradient_descent** | 梯度下降 / 非线性最小二乘 | 共轭梯度 (CG) 求解器 (Krylov 子空间的最优梯度加速) |
| **020_artery_pde** | 1D PDE 右端函数 (阻尼振荡) | HMC 分子动力学的 Hamilton 方程: dU/dτ = P, dP/dτ = -F |
| **017_area_under_curve** | 曲线下方积分数值计算 | 热力学积分: Δ(ln Z) = 6V ∫⟨P⟩dβ |
| **270_dfield9** | RK4, Dormand-Prince, 方向场 | HMC 辛积分器家族: leapfrog → Omelyan → Forest-Ruth → Yoshida |
| **1136_SonyResearch_SVG_baseline** | 基线配置管理, 条件映射 | HMC 参考 (冷/热) 启动, 重加权的基线 β₀ |
| **431_filum** | 顺序文件命名, 行数统计 | 格点构型文件管理: cfg_0000.npy, 元数据 .meta.txt |
| **1158_shoh5301_MSD** | MSD → 扩散系数拟合 | Polyakov loop 空间关联函数 G(r); 拓扑荷扩散系数 D_Q |
| **545_house** | 参考几何图案 (house shape) | 规范场参考构型 (冷启动/单瞬子校验构型) |
| **1044_nicsar2_FootlooseCalvingMechanism** | 相界面追踪, 弹性梁模型 | 退禁闭相界面张力 σ; 双峰直方图的峰谷检测 |
| **1338_triangulation_l2q** | 线性→二次提升 (中边节点) | Wilson → Symanzik 改进: plaquette → rectangle (6-link loop) |
| **351_fd_to_tec** | 场数据导出为结构化格式 | 格点场量 (plaquette per link, Polyakov per timeslice) 的结构化导出 |
| **1078_nec-research_alebrew** | 主动学习, 不确定性量化 | 昂贵测量 (拓扑荷) 的配置选择策略 |

## 项目架构 (12 个 Python 文件)

```
238_synth_project_Advanced/
├── main.py                   # 统一入口 (零参数可运行)
├── su3_algebra.py            # SU(3) 群代数: Gell-Mann 矩阵, 结构常数, 指数映射, 群投影
├── lattice_geometry.py       # 4D 格点几何: 索引, parity, 边界, plaquette/rectangle 枚举
├── gauge_field.py            # SU(3) 规范场: link 变量, staple, plaquette, Polyakov loop
├── gauge_actions.py          # 规范作用量: Wilson, Symanzik, Iwasaki, DBW2, Lüscher-Weisz
├── high_order_fd.py          # 高阶有限差分: 中心差分, Naik, clover, 色散关系
├── stability_analysis.py     # 稳定性分析: Von Neumann, CFL, Dirac 本征值, 步长选择
├── molecular_dynamics.py     # HMC 辛积分器: leapfrog, Omelyan, Forest-Ruth, Yoshida
├── fermion_solver.py         # 费米子: Staggered Dirac, CG 求解器, 手征凝聚
├── phase_transition.py       # 相变分析: Polyakov loop, susceptibility, Binder, 重加权
├── config_io.py              # 配置 I/O: 二进制/文本格式, 顺序命名, 元数据
├── observables.py            # 观测量: 自关联, 拓扑荷扩散, bootstrap, 热力学积分
├── README_博士级合成说明.md  # 本文档
└── output/                   # 模拟输出 (运行后生成)
```

## 核心物理公式

### A. 规范场与 Wilson loop

**Wilson plaquette 作用量:**
$$S_W = \beta \sum_{x, \mu<\nu} \left(1 - \frac{1}{3}\text{Re}\,\text{Tr}\, U_{\mu\nu}(x)\right)$$

**Plaquette:**
$$U_{\mu\nu}(x) = U_\mu(x) U_\nu(x+\hat\mu) U_\mu^\dagger(x+\hat\nu) U_\nu^\dagger(x)$$

**Polyakov loop (退禁闭序参量):**
$$L(\vec{x}) = \prod_{t=0}^{N_t-1} U_4(\vec{x}, t)$$

期望值: $\langle L \rangle = 0$ (禁闭相), $\langle L \rangle \neq 0$ (退禁闭相)

### B. Symanzik 改进 (融合 1338_l2q)

**改进作用量:**
$$S_{\text{Sym}} = \beta \sum_i c_i W_i$$

**树级 Lüscher-Weisz 系数:**
$$c_0 = \frac{5}{3}, \quad c_1 = -\frac{1}{12}, \quad c_2 = 0$$

归一化: $c_0 + 8c_1 + 8c_2 = 1$

### C. 高阶有限差分 (融合 270_dfield9)

**N 阶中心差分系数 $c_k$:**
$$\sum_{k=0}^{N/2-1} c_k (2k+1)^{2m-1} = \delta_{m,1}, \quad m=1,\ldots,N/2$$

**Naik 3-link 导数:**
$$\nabla_\mu^{\text{Naik}} = \frac{9}{8} \nabla_\mu^{(1)} - \frac{1}{24} \nabla_\mu^{(3)}$$

**色散关系:**
$$\tilde{p}_\mu(p) = \sum_k c_k \sin((2k+1) a p_\mu)$$

### D. 辛积分器稳定性 (融合 270_dfield9)

**Leapfrog 稳定性:**
$$\varepsilon < \frac{2}{\sqrt{\lambda_{\max}}}$$

**Omelyan 最优 ($\xi = 0.1932$):**
$$\varepsilon < 2 \left(1 - 2\xi + 2\xi^2\right)^{-1/2} / \sqrt{\lambda_{\max}}$$

**Forest-Ruth 4 阶:**
$$\theta = \frac{1}{2 - 2^{1/3}} \approx 1.3512$$

### E. 相变分析 (融合 1044, 1158)

**Susceptibility:**
$$\chi_L = V_s \left(\langle |L|^2 \rangle - \langle |L| \rangle^2\right)$$

**Binder cumulant:**
$$B_4 = 1 - \frac{\langle |L|^4 \rangle}{3 \langle |L|^2 \rangle^2}$$

**Histogram reweighting:**
$$\langle O \rangle_\beta = \frac{\sum_E O(E) \exp(-(\beta - \beta_0) E) P_{\beta_0}(E)}{\sum_E \exp(-(\beta - \beta_0) E) P_{\beta_0}(E)}$$

## 运行方法

### 零参数运行
```bash
cd 238_synth_project_Advanced
python main.py
```

### 输出内容
- 终端打印模拟进度与物理结果
- `output/` 目录下保存规范构型 (`.npy`) 和元数据 (`.meta.txt`)
- 运行时间约 40 秒 (Ns=3, Nt=4, 4 β 值 × 5 轨迹)

### 修改参数
编辑 `main.py` 中的 `SimulationConfig` 类:
```python
class SimulationConfig:
    Ns = 3           # 空间格点
    Nt = 4           # 时间格点
    beta_values = [5.0, 5.5, 5.7, 6.0]
    n_trajectories = 5
    n_steps = 8
    step_size = 0.02
    integrator = 'leapfrog'  # 或 'omelyan', 'forest_ruth'
```

## 科学成果

本项目在小规模可复现实验中展示了:

1. **退禁闭相变扫描**: β = 5.0 → 6.0 的 Polyakov loop 行为
2. **高阶差分精度**: O(a²) → O(a⁴) → O(a⁸) 的色散误差比较
3. **积分器效率**: Leapfrog vs Omelyan vs Forest-Ruth 的稳定性域
4. **拓扑荷扩散**: Monte Carlo 时间上的拓扑荷 MSD → 扩散系数
5. **Symanzik 改进**: Wilson 与 Lüscher-Weisz 作用量的对比

## 鲁棒性与工程特点

- **边界处理**: 4D PBC (空间) + APBC (时间费米子), 最小镜像约定
- **数值稳定性**: SU(3) 投影 (极分解 + 行列式修正), unitarity 监测
- **辛积分器**: 保持相空间体积, 长时间稳定性
- **CG 收敛**: 容差控制, breakdown 检测, 最大迭代限制
- **自关联分析**: Jackknife 误差, bootstrap, 积分自关联时间
- **元数据追踪**: 每条轨迹的物理量、接受率、unitarity 偏差

## 关键物理参数

| 参数 | 值 | 物理含义 |
|------|-----|---------|
| $\beta = 6/g^2$ | 5.0 → 6.0 | 逆耦合常数 |
| $N_t$ | 4 | 时间格点, $T = 1/(4a)$ |
| $T_c$ | $\approx 270$ MeV | 纯 SU(3) 临界温度 |
| $\beta_c(N_t=4)$ | $\approx 5.692$ | 临界耦合 |
| $C_F$ | 4/3 | fundamental Casimir |
| $C_A$ | 3 | adjoint Casimir |

## 修改的文件 (合成清单)

所有 12 个 .py 文件均为**新创建**, 未修改任何输入种子项目:

1. `su3_algebra.py` ← 631_l4lib (XOR) + 545_house (参考生成元)
2. `lattice_geometry.py` ← 1333 (边界) + 631 (parity) + 1338 (二次提升类比)
3. `gauge_field.py` ← 631 (索引) + 431 (I/O) + 545 (参考构型)
4. `gauge_actions.py` ← 1338 (Symanzik 提升) + 017 (积分) + 351 (场导出)
5. `high_order_fd.py` ← 270 (RK 精度) + 020 (PDE) + 1333 (边界相位)
6. `stability_analysis.py` ← 270 (RK 稳定域) + 020 (PDE 稳定) + 478 (步长)
7. `molecular_dynamics.py` ← 270 (积分器族) + 020 (时间演化) + 478 (梯度)
8. `fermion_solver.py` ← 478 (CG) + 631 (parity) + 545 (测试)
9. `phase_transition.py` ← 1044 (相界面) + 1158 (关联) + 017 (积分)
10. `config_io.py` ← 431 (文件) + 351 (数据导出) + 1136 (基线)
11. `observables.py` ← 1158 (MSD) + 977 (排序) + 017 (积分)
12. `main.py` ← 全部 15 个项目的协调入口

**1078_nec-research_alebrew** (主动学习) 的映射体现在配置选择策略与不确定性量化模块的设计哲学中。
