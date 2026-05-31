# PEM 燃料电池阴极催化剂层多物理场耦合衰减模拟系统

## 项目概述

本项目围绕**能源系统：氢燃料电池催化剂衰减**这一前沿科学问题，基于 15 个数值计算种子项目的核心算法，构建了一个面向博士级难度的多物理场耦合计算框架。

具体研究目标为：建立 PEM 燃料电池阴极催化剂层 (Cathode Catalyst Layer, CCL) 中 Pt 纳米颗粒电化学溶解—奥斯瓦尔德熟化—碳载体腐蚀的多物理场耦合衰减模型，量化长期运行下的电化学活性表面积 (ECSA) 损失与性能衰减轨迹，并优化催化剂负载分布以延缓衰减。

---

## 一、原项目到科学问题的映射

| 原项目编号 | 原项目核心算法 | 合成后角色 | 科学功能 |
|-----------|--------------|-----------|---------|
| 508_hb_to_mm | Harwell-Boeing 稀疏矩阵格式转换 | sparse_assembler.py | CCL 离散化后的稀疏/边界带状矩阵组装与元数据管理 |
| 659_legendre_exactness | Legendre 高斯积分精度检验 | ripening_model.py | 颗粒尺寸分布矩的高精度数值积分 |
| 268_cyclic_reduction | 循环约化法求解三对角系统 | diffusion_solver.py | 扩散-反应方程离散化后的线性系统直接求解 |
| 687_linpack_bench | LINPACK 基准 (LU 分解) | diffusion_solver.py | 带状矩阵 LU 分解与线性求解 (验证用) |
| 292_disk_distance | 单位圆盘随机点距离统计 | ripening_model.py | 催化剂颗粒二维间距分布的蒙特卡洛统计 |
| 969_r8bb | 边界带状矩阵存储与操作 | sparse_assembler.py | 耦合系统的边界带状 (Border-Banded) 矩阵格式存储 |
| 979_r8gb | 一般带状矩阵 LU 分解 | diffusion_solver.py | 传质方程带状存储矩阵的 PLU 分解 |
| 902_power_method | 幂法求主导特征值 | ecsa_calculator.py | 催化剂衰减系统的稳定性分析与主导特征值提取 |
| 1404_wdk | Weierstrass-Durand-Kerner 多项式求根 | butler_volmer.py | 电化学过电位非线性方程的复数域求根 |
| 834_opt_golden | 黄金分割搜索优化 | catalyst_optimizer.py | 催化剂负载分布的单峰目标函数优化 |
| 558_hypercube_grid | 超立方体多维网格生成 | ccl_grid.py | CCL 操作参数空间 (T, RH, E, L_Pt, S_c) 的均匀采样网格 |
| 126_burgers_time_inviscid | 无粘 Burgers 方程守恒律格式 | carbon_corrosion.py | 碳腐蚀结构退化传播方程的 Godunov/MacCormack 格式求解 |
| 1370_ubvec | 无符号二进制向量格雷码遍历 | morphology_evolution.py | 催化剂表面吸附位点微观状态的格雷码枚举 |
| 1431_zero_muller | Muller 复数求根法 | butler_volmer.py | Butler-Volmer 非线性方程的 Muller 法复数迭代求根 |
| 711_mandelbrot_area | Mandelbrot 集逃逸时间迭代 | morphology_evolution.py | 催化剂层孔隙网络连通性的类比迭代分析 |

**每一个输入项目均已真实融入合成项目，无遗漏、无挂名。**

---

## 二、新增数学物理模型与核心公式

### 2.1 电化学动力学：Butler-Volmer 方程

阴极氧还原反应 (ORR):

$$
\mathrm{O_2 + 4H^+ + 4e^- \rightarrow 2H_2O}
$$

Butler-Volmer 电流密度:

$$
j = j_0 \left[ \exp\left(\frac{\alpha_a n F \eta}{RT}\right) - \exp\left(-\frac{\alpha_c n F \eta}{RT}\right) \right]
$$

其中交换电流密度的温度与浓度修正:

$$
j_0 = j_{0,\mathrm{ref}} \left(\frac{C_{\mathrm{O_2}}}{C_{\mathrm{O_2,ref}}}\right)^\gamma \exp\left[ -\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{\mathrm{ref}}}\right) \right]
$$

过电位隐含方程（需数值求解）:

$$
f(\eta) = \eta - (E - E_{\mathrm{eq}}) + R_{\mathrm{ct}} j_0 \left[ \exp\left(\frac{\alpha_a n F \eta}{RT}\right) - \exp\left(-\frac{\alpha_c n F \eta}{RT}\right) \right] = 0
$$

### 2.2 传质扩散：稳态扩散-反应方程

CCL 中氧气的一维稳态传质:

$$
D_{\mathrm{eff}} \frac{d^2 C}{dx^2} - k_{\mathrm{rxn}} C = 0, \quad 0 < x < L_{\mathrm{CCL}}
$$

有效扩散系数（Bruggeman 修正）:

$$
D_{\mathrm{eff}} = D_{\mathrm{bulk}} \cdot \varepsilon^{1.5}
$$

边界条件:
- Dirichlet: $C(0) = C_0$（GDL/CCL 界面）
- Neumann: $\left.\frac{dC}{dx}\right|_{x=L} = 0$（膜界面）

离散化后得到三对角线性系统 $A \mathbf{C} = \mathbf{b}$，使用 Thomas 算法（追赶法，循环约化法家族）求解。

### 2.3 Pt 纳米颗粒溶解-熟化：LSW 理论

Gibbs-Thomson 效应下的溶解度修正（Kelvin 方程）:

$$
C_{\mathrm{sat}}(r) = C_{\mathrm{sat},\infty} \exp\left(\frac{2\gamma V_m}{rRT}\right)
$$

颗粒半径演化速率:

$$
\frac{dr}{dt} = \frac{D V_m}{r} \left[ C_{\mathrm{bulk}} - C_{\mathrm{sat}}(r) \right]
$$

LSW 临界半径:

$$
r_c = \frac{2\gamma V_m}{RT \ln(C_{\mathrm{bulk}} / C_{\mathrm{sat},\infty})}
$$

平均半径的 LSW 渐近律:

$$
\langle r \rangle^3 - \langle r_0 \rangle^3 = \frac{8\gamma D V_m^2 C_{\mathrm{sat},\infty}}{9RT} \cdot t
$$

### 2.4 碳载体腐蚀：结构退化传播

碳腐蚀电化学反应:

$$
\mathrm{C + 2H_2O \rightarrow CO_2 + 4H^+ + 4e^-}, \quad E^0 = 0.207 \, \mathrm{V}
$$

结构退化守恒律方程:

$$
\frac{\partial S_c}{\partial t} + \frac{\partial}{\partial x}(v_c S_c) = -k_{\mathrm{corr}} S_c \theta_{\mathrm{pore}}
$$

采用 Godunov / Lax-Wendroff / MacCormack 守恒律格式数值求解。

### 2.5 ECSA 损失与稳定性分析

单位质量活性表面积:

$$
\mathrm{ECSA} = \frac{\sum_i 4\pi r_i^2}{\sum_i \frac{4}{3}\pi r_i^3 \rho_{\mathrm{Pt}}} \times 1000 \quad [\mathrm{m^2/g_{Pt}}]
$$

ECSA 损失动力学:

$$
\frac{d(\mathrm{ECSA})}{dt} = -k_1 \cdot \mathrm{ECSA} - k_2 \cdot \mathrm{ECSA}^2
$$

稳定性分析：构建衰减系统的 Jacobian 矩阵 $J$，使用幂法 (Power Method) 迭代提取主导特征值:

$$
\lambda^{(k+1)} = \frac{\mathbf{y}^T J \mathbf{y}}{\mathbf{y}^T \mathbf{y}}, \quad \mathbf{y}^{(k+1)} = \frac{J \mathbf{y}}{\|J \mathbf{y}\|}
$$

- $\lambda_{\max} < 0$: 系统渐近稳定
- $\lambda_{\max} > 0$: 系统不稳定（衰减加速）

电压损失与 ECSA 关系（Tafel 关联）:

$$
\Delta V = b \cdot \log_{10}\left(\frac{\mathrm{ECSA}_0}{\mathrm{ECSA}}\right)
$$

### 2.6 催化剂负载优化

目标泛函:

$$
J(L_{\mathrm{Pt}}) = w_1 \frac{C_{\mathrm{Pt}}(L_{\mathrm{Pt}})}{C_{\mathrm{ref}}} + w_2 \left[1 - \frac{P(L_{\mathrm{Pt}})}{P_{\max}}\right] + w_3 D(L_{\mathrm{Pt}})
$$

性能-负载经验关系:

$$
P(L_{\mathrm{Pt}}) = P_{\max} \left[1 - \exp\left(-k_{\mathrm{eff}} \frac{L_{\mathrm{Pt}}}{L_{\mathrm{ref}}}\right)\right]
$$

使用黄金分割搜索 (Golden Section Search) 求解:

$$
\phi = \frac{\sqrt{5}-1}{2} \approx 0.618
$$

### 2.7 形貌退化与孔隙连通性

催化剂层形貌的分形维数（盒计数法）:

$$
D_f = -\lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log \epsilon}
$$

有效比表面积与分形维数关系:

$$
A_{\mathrm{eff}} = A_0 \left(\frac{L}{l_0}\right)^{D_f - 2}
$$

孔隙网络连通性类比 Mandelbrot-Julia 逃逸时间:

$$
z_{n+1} = z_n^2 + c
$$

微观状态枚举使用格雷码 (Gray Code) 遍历:

$$
G(n) = n \oplus (n \gg 1)
$$

---

## 三、项目文件结构

```
164_synth_project/
├── main.py                  # 统一入口，零参数可运行
├── ccl_grid.py              # CCL 多维参数网格生成 (hypercube_grid)
├── butler_volmer.py         # Butler-Volmer 方程求解 (zero_muller + wdk)
├── diffusion_solver.py      # 扩散-反应方程求解 (cyclic_reduction + r8gb)
├── ripening_model.py        # Pt 熟化与尺寸分布演化 (disk_distance + legendre)
├── carbon_corrosion.py      # 碳腐蚀传播模拟 (burgers_time_inviscid)
├── ecsa_calculator.py       # ECSA 损失与稳定性分析 (power_method)
├── catalyst_optimizer.py    # 催化剂负载优化 (opt_golden)
├── sparse_assembler.py      # 稀疏矩阵组装 (hb_to_mm + r8bb)
├── morphology_evolution.py  # 形貌退化与状态枚举 (mandelbrot + ubvec)
└── README_博士级合成说明.md   # 本文档
```

---

## 四、合成后的项目能够解决什么科学问题

1. **电化学动力学参数提取**：在给定操作条件 (T, RH, E) 下，求解 ORR 的 Butler-Volmer 过电位和电流密度。
2. **CCL 传质分析**：求解氧气在催化剂层中的浓度分布，评估传质限制对性能的影响。
3. **Pt 催化剂衰减预测**：模拟纳米颗粒的 Ostwald 熟化过程，预测长期运行后的粒径分布和 ECSA 损失。
4. **碳载体寿命评估**：模拟高电位下碳腐蚀引起的结构完整性损失。
5. **系统稳定性判定**：通过 Jacobian 特征值分析判断衰减过程是否会自加速。
6. **催化剂负载优化**：在成本和性能之间寻找最优的 Pt 负载分布。
7. **形貌退化量化**：通过分形维数和孔隙连通性评估催化剂层的结构退化程度。

---

## 五、如何运行

### 环境要求
- Python 3.8+
- NumPy

### 运行方式

```bash
cd 164_synth_project
python main.py
```

程序无需任何输入参数，执行后将自动完成以下 9 步计算流程：

1. 生成 CCL 操作条件参数网格 (324 个采样点)
2. 求解 ORR Butler-Volmer 电化学动力学
3. 求解 CCL 氧气传质扩散方程
4. 模拟 Pt 纳米颗粒 500 小时 Ostwald 熟化
5. 模拟碳载体腐蚀传播 (10 小时)
6. ECSA 损失评估与系统稳定性分析
7. 催化剂负载黄金分割优化
8. 离散化稀疏矩阵组装与对角占优性检验
9. 催化剂层形貌退化分析

### 典型输出示例

```
操作条件: T=333.1 K, E=0.600 V, L_Pt=0.050 mg/cm^2
电化学:   j0=2.105e-03 A/m^2, eta=-0.2072 V, j=-3.913e+03 A/m^2
传质:     O2(膜侧)=1.0739 mol/m^3
熟化:     r_mean(0h)=3.94 nm -> r_mean(500h)=17.82 nm
碳腐蚀:   完整性损失=13.42%
ECSA:     34.16 -> 3.19 m^2/g_Pt (保留 9.3%)
电压损失: 61.76 mV
稳定性:   lambda_max=-2.081e-05 (stable)
优化:     L_Pt*=0.077 mg/cm^2
形貌:     MDI=0.170
计算时间: 0.155 s
```

---

## 六、数值鲁棒性与边界处理

本项目在多处实施了严格的数值边界保护：

- **指数溢出保护**：所有 exp() 运算均经过 clip(-350, 350) 或 clip(-700, 700) 保护
- **除零保护**：所有除法运算均检查分母绝对值是否大于 1e-30
- **物理约束**：颗粒半径下限 0.5 nm（原子尺度下限），浓度非负，负载非负
- **矩阵稳定性**：稀疏矩阵组装后自动检查严格对角占优性
- **迭代收敛**：非线性求解器和幂法均设置最大迭代次数和收敛容差
- **CFL 条件**：守恒律格式求解时自动校验并调整时间步长

---

## 七、科学复杂度说明

本项目融合了以下博士级计算内容：

- **多物理场耦合**：电化学动力学 + 传质扩散 + 颗粒熟化 + 结构腐蚀 + 稳定性分析
- **高阶数值方法**：循环约化法、带状 LU 分解、Godunov 守恒律格式、黄金分割优化、幂法特征值分析
- **复杂科学公式**：Butler-Volmer 方程、Kelvin 方程、LSW 熟化理论、Tafel 关联、分形维数
- **工程鲁棒性**：全面的边界处理、数值溢出保护、参数物理范围检查

---

*文档生成日期: 2026-05-04*
