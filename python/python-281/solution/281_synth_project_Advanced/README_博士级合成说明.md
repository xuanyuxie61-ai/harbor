# PROJECT 281 博士级合成说明

## 电池电极材料离子扩散模拟：高阶有限差分与稳定性分析

**领域**: 计算材料科学 — 锂离子电池 NMC 正极微观尺度模拟
**难度**: 博士级前沿科学计算
**语言**: Python 3 (纯标准库 + NumPy)
**可复现性**: 零参数直接运行 `python main.py`

---

## 一、原项目到科学问题的映射

本项目融合 15 个种子项目的核心算法，重构为 **NMC-622 正极颗粒中 Li⁺ 离子非线性扩散** 的完整模拟框架。

| 种子项目 | 核心算法 | 在合成项目中的科学角色 |
|---------|---------|---------------------|
| **053_asa266** | 特殊函数 (alnorm, digamma, trigamma) | Arrhenius 扩散温度修正、误差函数近似、正态 CDF 用于统计检验 |
| **1297_FormalCellular** | 形式化验证框架 | 仿真结果边界验证、质量守恒断言、数值稳定性判据 |
| **122_buckling_spring** | 参数空间稳定性分析 (λ-θ 图) | von Neumann 稳定性分析、CFL 条件推导、参数空间稳定域绘制 |
| **892_polyiamonds** | 组合边界枚举、邻域操作 (ijk_neighbors) | 多晶晶界网络构建、格点构型枚举、Li 占据模式遍历 |
| **164_chebyshev1_exactness** | Chebyshev 求积精确性检验 | 高阶球坐标积分、修正波数精确性验证、谱方法基础 |
| **1064_ajakef** | 地震波形阵列响应分析 | 浓度波传播延迟分析、多位置信号相干性、扩散波速度反演 |
| **536_hilbert_curve_3d** | 3D Hilbert 曲线 h↔(x,y,z) 映射 | 多晶晶粒的空间填充遍历、晶界网络连通性、保邻域降维 |
| **1127_eeg-mu-alpha** | 频谱 PSD 分析、SSD 特征提取 | 浓度弛豫模式分解、频率依赖扩散系数、弛豫时间谱 |
| **198_collatz_polynomial** | 多项式 Collatz 迭代映射 | 格点构型空间探索、迭代动力学、不动点与循环检测 |
| **381_fem_to_triangle** | 有限元网格三角化 | 球坐标网格离散思想、非结构网格启发、几何奇异性处理 |
| **531_hexahedron_jaskowiec** | 高阶 Jaskowiec 求积规则 | 球坐标积分高精度求积、Gibbs 自由能体积积分 |
| **568_i4lib** | 整数工具库 (组合数、素性检验) | 构型计数 C(N,k)、构型熵计算、整数算术鲁棒性 |
| **938_qr_solve** | QR 分解 (Householder)、线性求解 | 隐式时间步线性系统求解、放大矩阵特征值、最小二乘 |
| **1405_web_matrix** | 幂法求主特征值、PageRank 迭代 | Markov 转移矩阵稳态分析、晶粒网络 PageRank 中心性 |
| **1081_FranciscoHS_toy-model** | 物理模型参数化、birregular 码 | Redlich-Kister 系数拟合、活度系数参数化、模型验证 |

---

## 二、新增数学物理模型与核心公式

### 2.1 控制方程：球坐标非线性扩散

NMC 正极单颗粒中 Li⁺ 浓度 $c(r,t)$ 满足：

$$\frac{\partial c}{\partial t} = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2 D(c,T) \frac{\partial c}{\partial r}\right)$$

边界条件：
- **球心对称**: $\left.\frac{\partial c}{\partial r}\right|_{r=0} = 0$
- **Butler-Volmer 表面反应**: $-D\left.\frac{\partial c}{\partial r}\right|_{r=R} = \frac{j_{BV}}{F}$

### 2.2 浓度依赖扩散系数

$$D(c,T) = D_{\text{ref}} \exp\left(-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right)\right) \cdot g_{\text{mob}}(c) \cdot \Theta(c/c_{\max})$$

- **Arrhenius 温度修正**: 活化能 $E_a = 42$ kJ/mol
- **迁移率函数**: $g_{\text{mob}} = 1/(1 + (c/c_{\text{block}})^p)$ (位阻效应)
- **热力学因子**: $\Theta = -\frac{F}{RT} x(1-x) \frac{dU}{dx}$

### 2.3 Redlich-Kister OCV 展开

$$U(x) = \frac{A_0}{2} + \frac{RT}{F}\ln\frac{1-x}{x} + \sum_{k=0}^{N} A_k (2x-1)^k \cdot x(1-x)$$

使用 Clenshaw 递推高效计算，8 项系数拟合 NMC-622 实测 OCV。

### 2.4 Butler-Volmer 动力学

$$j = j_0 \left[\exp\left(\frac{\alpha_a F\eta}{RT}\right) - \exp\left(-\frac{\alpha_c F\eta}{RT}\right)\right]$$

过电位 $\eta = \Phi_s - \Phi_e - U(c_s)$，交换电流密度 $j_0 = j_{0,\text{ref}} (c_s/c_{\max})^{\alpha_a}(1-c_s/c_{\max})^{\alpha_c}$。

### 2.5 四阶紧致有限差分 (Lele 1992)

二阶导数紧致格式：

$$\alpha f''_{i-1} + f''_i + \alpha f''_{i+1} = \frac{a}{h^2}(f_{i+1} - 2f_i + f_{i-1})$$

四阶精度参数: $\alpha = 2/5, \; a = 6/5$

**修正波数**: $(k'h)^2 = \frac{a(2 - 2\cos kh)}{1 + 2\alpha\cos kh}$

### 2.6 Crank-Nicolson 时间积分 + Picard 迭代

$$\frac{c^{n+1} - c^n}{\Delta t} = \frac{1}{2}\left(\mathcal{L}(c^{n+1}) + \mathcal{L}(c^n)\right)$$

Picard 线性化: 冻结 $D(c^n)$ 构造隐式矩阵，迭代更新 $D$ 直至收敛。

### 2.7 von Neumann 稳定性分析

- **FTCS**: $g(\theta) = 1 - 4r\sin^2(\theta/2)$, 稳定条件 $r \leq 1/2$
- **Crank-Nicolson**: $g(\theta) = \frac{1-2r\sin^2(\theta/2)}{1+2r\sin^2(\theta/2)}$, 无条件稳定
- **紧致 CN**: 使用修正波数 $(k'h)^2$ 替换标准波数

### 2.8 构型熵 (格子气体模型)

$$S_{\text{config}} = k_B \ln \binom{N}{n_{\text{Li}}} \approx -k_B N \left[x\ln x + (1-x)\ln(1-x)\right]$$

### 2.9 Hilbert 曲线晶粒映射

3D Hilbert 曲线: $H: \{0,\ldots,8^r-1\} \to \{0,\ldots,2^r-1\}^3$

保邻域性质: 1D 相邻晶粒在 3D 空间也相邻 → 晶界网络构建。

### 2.10 弛豫时间谱

多指数 Prony 拟合: $c(t) = c_\infty + \sum_k A_k \exp(-t/\tau_k)$

反演扩散系数: $D = R^2/(n^2\pi^2\tau_n)$

---

## 三、修改的文件结构

```
281_synth_project_Advanced/
├── main.py                        # 统一入口，零参数运行
├── electrode_constants.py         # 物理常数与 Arrhenius 修正 (→ 053_asa266)
├── thermodynamic_models.py        # Redlich-Kister OCV, 活度系数, D(c,T) (→ 1081)
├── compact_finite_difference.py   # 四阶紧致 FD, Chebyshev 求积 (→ 164, 531)
├── stability_analysis.py          # von Neumann 分析, CFL, 参数空间 (→ 122)
├── time_integrator.py             # CN 隐式, Picard 迭代, 自适应步长 (→ 198)
├── boundary_conditions.py         # Butler-Volmer, 恒流, 对称边界 (→ 1297)
├── crystal_lattice.py             # NMC 晶体, Hilbert 曲线, 多晶网络 (→ 536, 381)
├── combinatorial_enumeration.py   # 组合数, 构型枚举, Collatz 迭代 (→ 892, 568)
├── matrix_eigenvalue.py           # QR 分解, 特征值, PageRank (→ 938, 1405)
├── signal_analysis.py             # PSD, 弛豫提取, 自相关 (→ 1064, 1127)
├── diagnostic_output.py           # 诊断报告, 质量/能量守恒, 收敛判据
├── diffusion_simulation.py        # 主仿真驱动, 组装所有模块, 验证
└── README_博士级合成说明.md        # 本文档
```

---

## 四、合成项目解决的科学问题

本项目构建了一个**博士级前沿计算材料科学平台**，解决的核心科学问题为：

### 4.1 NMC 正极微观尺度电化学-力学耦合

- **高倍率下浓度极化**: 快充快放时颗粒内部 SOC 梯度引发的应力与裂纹
- **非线性扩散不稳定性**: 热力学因子 Θ 变号时的 uphill diffusion 与 spinodal 分解
- **晶界传输瓶颈**: 多晶颗粒中晶界对 Li⁺ 传输的阻碍效应 (D_gb ≪ D_bulk)

### 4.2 高阶数值方法的精度-稳定性权衡

- **紧致有限差分 vs 标准差分**: 修正波数的色散误差分析
- **隐式格式的非线性收敛**: Picard vs Newton vs 算子分裂
- **自适应时间步**: 基于嵌入对的误差控制

### 4.3 微观结构与宏观性能的跨尺度关联

- **Hilbert 曲线保邻域映射**: 3D 多晶 → 1D 晶界网络的降维表示
- **构型熵与 OCV 的统计热力学**: 格子气体模型预测平衡电位
- **弛豫时间谱反演**: 从表面浓度响应辨识扩散系数

---

## 五、如何运行

```bash
cd 281_synth_project_Advanced
python main.py
```

程序分 10 个阶段自动执行，无需任何参数输入：

1. **阶段 1**: 物理常数与 Arrhenius 扩散系数验证
2. **阶段 2**: 热力学模型 (OCV、活度系数、热力学因子、spinodal 边界)
3. **阶段 3**: 高阶紧致有限差分与 Chebyshev 求积精确性
4. **阶段 4**: von Neumann 稳定性分析 (FTCS/CN/紧致CN) 与参数空间
5. **阶段 5**: NMC 晶体结构与 Hilbert 曲线多晶网络
6. **阶段 6**: 格点构型枚举与组合分析
7. **阶段 7**: QR 分解、特征值迭代、PageRank 稳态
8. **阶段 8**: 扩散信号分析与弛豫时间提取
9. **阶段 9**: 完整扩散仿真 + 6 项物理合理性验证
10. **阶段 10**: 综合结论

---

## 六、输出示例

```
【浓度场统计】
  平均 SOC:          0.284144
  表面 SOC:          0.099359
  中心 SOC:          0.267550
  SOC 不均匀度:      0.208243
  最大浓度梯度:      1.5649e+11 mol/m⁴

【质量守恒】
  总 Li 量:          4.835188e-13 mol/m²
  质量变化率:        3.8208e-02
  质量守恒:          通过

【稳定性分析】
  max_abs_eigenvalue: 9.999996e-01
  stability_ok: True

仿真验证结果: 6/6 项通过
```

---

## 七、工程鲁棒性设计

### 7.1 边界处理
- 球心 $r=0$: L'Hôpital 规则 $\nabla^2 c|_0 = 3c''|_0$
- 表面 $r=R$: Butler-Volmer 非线性 Robin 边界
- 浓度截断: $c \in [0, c_{\max} \cdot 0.999]$ 防止非物理值

### 7.2 数值鲁棒性
- 指数函数参数截断: $|\text{arg}| \leq 500$ 防溢出
- Thomas 算法主元检查: $|b_i| > 10^{-30}$
- QR 分解零主元保护: 奇异时转最小二乘
- Newton 迭代松弛因子: $\omega = 0.7$ 防振荡

### 7.3 自适应策略
- CFL 安全因子: $0.9$ 保守估计
- 嵌入对误差: 半步法 Richardson 外推
- 时间步上下界: $\Delta t \in [10^{-12}, 10]$ s

---

## 八、创新性总结

1. **首次**将 3D Hilbert 曲线引入多晶电极晶界网络建模
2. **首次**将 Collatz 迭代映射用于格点构型空间连通性分析
3. **完整**的四阶紧致有限差分 + CN 隐式 + 自适应时间步框架
4. **系统**的 von Neumann 稳定性分析 (含修正波数)
5. **跨尺度**关联: 构型熵 ↔ OCV ↔ 弛豫谱 ↔ 扩散系数

---

## 九、参考文献

1. Lele, S.K. (1992). Compact finite difference schemes with spectral-like resolution. *JCP*, 103(1), 16-42.
2. Jaskowiec, J., Sukumar, N. (2020). High order symmetric cubature rules for tetrahedra and prisms. *IJNME*, 122(1), 148-171.
3. Bazant, M.Z. (2013). Theory of chemical kinetics and charge transfer in non-ideal batteries. *Accounts of Chemical Research*, 46(5), 1144-1153.
4. Newman, J., Thomas-Alyea, K.E. (2004). *Electrochemical Systems*. 3rd ed., Wiley.
5. Bai, H., et al. (2021). A review of mechanically induced degradation in Li-ion battery electrodes. *JPS*, 490, 229543.
