# 水声传播宽角抛物方程（WAPE）建模系统 — 博士级合成说明

## 一、项目概述

本项目为**声学工程**领域的前沿博士级科研代码合成项目，核心科学问题为：

> **深海复杂环境下的宽角抛物方程（Wide-Angle Parabolic Equation, WAPE）水声传播数值建模，涵盖非均匀声速剖面、变深度海底地形、体积散射、简正波模态分析、混响预测及传播损失后处理。**

项目基于 15 个种子科研代码项目的核心算法，融合重构为一个统一的 Python 科研计算系统。所有代码具备零参数运行能力，无需外部输入即可输出完整的数值模拟结果与诊断信息。

---

## 二、原项目到科学问题的映射

| 序号 | 原项目 | 核心算法 | 合成后角色 |
|:---:|:---|:---|:---|
| 1 | `905_pram` | 边界词编码（多边形边界追踪） | **海底地形边界多边形编码**：将 bathymetry 轮廓离散为有序闭合多边形，用于计算域掩码裁剪 |
| 2 | `301_disk01_monte_carlo` | 圆盘均匀采样、Gamma 解析积分 | **圆形声源孔径均匀采样**：声源初始场生成中的 piston 声源孔径采样，以及方位角积分中的均匀圆盘采样 |
| 3 | `1023_rigid_body_ode` | ODE 步进架构、守恒量监测 | **PE 范围步进架构**：借鉴 `deriv()` + `conserved()` + `parameters()` 的模块化设计，实现能量通量守恒监测与参数管理 |
| 4 | `382_fem_to_xml` | FEM 网格结构（节点/单元/连通性） | **计算网格生成**：构建 range-depth 平面上的结构化网格，管理节点掩码与三角形单元连通性 |
| 5 | `1426_xyzl_display` | 3D 点/线数据结构 | **接收器阵列几何管理**：VLA 垂直线阵的 3D 坐标结构、波束形成权值计算与信号提取 |
| 6 | `1265_toms112` | 点在多边形内判定（射线交叉） | **计算域边界掩码**：判断网格节点是否在有效水域内，排除陆地/海底以上无效区域 |
| 7 | `113_box_distance` | 3D 盒子内随机点距离统计 | **散射体积路径统计**：估算体积散射体内的随机路径长度均值，用于时延扩展与混响模型 |
| 8 | `498_hammersley` | Hammersley 准随机序列 | **QMC 散射积分**：利用低差异序列加速散射相位函数与参数扫描的多维积分，收敛速度 O(1/N) |
| 9 | `1082_sinc` | sinc 函数、Ci/Si 特殊函数 | **带限场插值与格林函数**：sinc 插值重建离散声压场，Ci/Si 用于柱面波远场渐近与吸收积分 |
| 10 | `942_quad_parfor` | 并行梯形积分 | **能量通量并行积分**：深度方向能量通量的梯形法积分，以及多频传播损失的批量计算 |
| 11 | `1307_triangle_integrals` | 三角形上多项式精确积分 | **FEM 基函数积分**：谱元离散中三角形参考元到物理元的映射与精确多项式积分 |
| 12 | `899_polyomino_parity` | Diophantine 方程与回溯求解 | **模态数离散约束**：将简正波截止条件视为 Diophantine 不等式，用回溯法求解多声道耦合模态数 |
| 13 | `894_polynomial_conversion` | 正交多项式基转换矩阵 | **Chebyshev/Legendre 谱元离散**：利用三项递推构建 Chebyshev→monomial 转换矩阵，实现谱精度微分算子 |
| 14 | `807_nonlin_fixed_point` | 固定点迭代 + Newton  safeguard | **海底阻抗边界自洽迭代**：非线性 Robin 边界条件的导纳参数 γ_b 通过固定点/Newton 混合迭代求解 |
| 15 | `044_asa152` | 超几何分布、log-gamma 防溢出 | **数值稳定性工具**：大数阶乘的 log-gamma 计算、正态 CDF 有理逼近，用于统计声学与组合数计算 |

---

## 三、新增数学物理模型与核心公式

### 3.1 宽角抛物方程（WAPE）

从 Helmholtz 方程出发，设声压 $p(r,z) = u(r,z) \cdot H_0^{(1)}(k_0 r)$，在远场近似 $\partial^2 u / \partial r^2 \ll 2ik_0 \cdot \partial u / \partial r$ 下得到标准 PE：

$$2i k_0 \frac{\partial u}{\partial r} = \frac{\partial^2 u}{\partial z^2} + k_0^2 \left(n^2(z) - 1\right) u$$

宽角修正采用 Claerbout Padé(1,1) 近似：

$$\frac{\partial}{\partial r} = i k_0 \left(\sqrt{1+X} - 1\right) \approx i k_0 \frac{X}{2+X}$$

其中 $X = \frac{1}{k_0^2}\frac{\partial^2}{\partial z^2} + \left(n^2(z)-1\right)$。

### 3.2 Munk 标准声速剖面

$$c(z) = c_0 \left[1 + \varepsilon \left(\eta + e^{-\eta} - 1\right)\right], \quad \eta = \frac{2(z-z_a)}{B}$$

该剖面在声道轴 $z=z_a$ 处取得最小值 $c_0$，形成 SOFAR 声道。

### 3.3 吸收模型（Thorp 公式）

$$\alpha(f) = \frac{0.11 f^2}{1+f^2} + \frac{44 f^2}{4100+f^2} + 2.75\times10^{-4} f^2 + 0.0033 \quad [\text{dB/km}]$$

复波数：$k(z) = \omega/c(z) + i\alpha(z)$。

### 3.4 Crank-Nicolson 隐式离散

$$\left[I - \frac{\Delta r}{4ik_0} L\right] u^{m+1} = \left[I + \frac{\Delta r}{4ik_0} L\right] u^m$$

其中 $L = D^2 + k_0^2 \cdot \text{diag}(n^2-1)$，$D^2$ 为变网格二阶差分算子。

### 3.5 海底阻抗边界（Robin）

$$\frac{\partial u}{\partial z} + \gamma_b \cdot u = 0, \quad \gamma_b = i k_0 \sqrt{n_b^2 - \cos^2\theta}$$

通过固定点迭代 $\gamma_{k+1} = g(\gamma_k)$ 自洽求解，辅以 Newton 修正加速收敛。

### 3.6 PML 吸收层

$$\tilde{z} = z + i \sigma(z)(z - z_{\text{PML}}), \quad \sigma(z) = \sigma_{\max}\left(\frac{z-z_{\text{PML}}}{L_{\text{PML}}}\right)^p$$

### 3.7 简正波本征值问题

$$\frac{d^2\phi_n}{dz^2} + \left[k^2(z) - k_{r,n}^2\right] \phi_n = 0$$

采用 Chebyshev tau 谱方法离散，边界条件通过 tau 行替换施加。

### 3.8 传播损失

$$\text{TL}(r,z) = -10\log_{10}|u(r,z)|^2 \quad [\text{dB}]$$

### 3.9 混响级

$$\text{RL} = \text{SL} - 2\cdot\text{TL} + S_v + 10\log_{10} V$$

散射体积：$V = \frac{c\tau}{2} R^2 \Delta\Omega$。

### 3.10 空间相关系数

$$C(\Delta r, \Delta z) = \frac{\langle u(r,z) \cdot u^*(r+\Delta r, z+\Delta z) \rangle}{\sqrt{\langle|u|^2\rangle \langle|u|^2\rangle}}$$

---

## 四、项目文件结构

```
093_synth_project/
├── main.py                          # 统一入口，零参数运行
├── environment.py                   # 海洋环境模型（Munk SSP、吸收、地形）
├── mesh_builder.py                  # 计算网格生成与域掩码管理
├── source_field.py                  # 声源初始场生成（高斯/Green/方向性）
├── boundary_conditions.py           # 边界条件处理（海面/海底/PML）
├── parabolic_solver.py              # WAPE 核心求解器（CN-FD / SSF）
├── modal_analysis.py                # 简正波模态分析与离散约束求解
├── scattering_model.py              # 体积散射、混响、QMC 积分、相关分析
├── propagation_loss.py              # 传播损失计算、阵列处理、多径统计
├── utils.py                         # 正交多项式转换、三角积分、数值稳定性
└── special_functions.py             # sinc、Ci/Si、alnorm、log-gamma
```

共 **11 个 Python 文件**（含 main.py），满足至少 8 个 .py 文件的要求。

---

## 五、合成后的项目能够解决的科学问题

1. **深海声道中的长距离声传播预测**：利用 WAPE 在 Munk 声速剖面下模拟 50 km 量级的声传播，预测传播损失随距离和深度的分布。

2. **复杂海底地形对声场的影响**：通过参数化 bathymetry 模型（高斯山丘 + 斜坡），研究海山、大陆坡等地形引起的声影区与会聚区。

3. **简正波模态提取与频散分析**：通过 Chebyshev 谱方法求解本征问题，提取各阶模态的相速度、群速度与深度函数，用于模态识别。

4. **体积散射与混响强度预测**：基于深度调制的散射强度模型，估算特定距离和脉冲条件下的混响级，为声纳系统设计提供依据。

5. **多径时延扩展与相干带宽估算**：利用随机路径统计方法，估算海洋信道的时间相干性与频率相干性。

6. **接收器阵列波束形成与空间相关分析**：模拟垂直线阵的波束输出，分析声场的空间相关系数与衰落特性。

---

## 六、如何运行

在项目目录下执行：

```bash
python main.py
```

程序将自动完成以下流程并输出诊断信息：
1. 海洋环境参数（Munk SSP、吸收、海底地形）
2. 计算网格生成与质量评估
3. 声源初始场构建与归一化
4. 边界条件初始化（Robin 导纳、PML）
5. Crank-Nicolson 有限差分步进求解 WAPE
6. 简正波模态分析（本征值、相速度、群速度）
7. 体积散射强度、混响级、QMC 积分验证
8. 传播损失、收敛区/影区检测、多径统计
9. 空间相关分析
10. 所有种子项目核心算法的数值验证

---

## 七、合成方法说明

### 7.1 科学问题重构

将原本离散、独立的 15 个科研算法项目，围绕**"深海宽角抛物方程声传播建模"**这一核心科学问题进行有机融合：
- 数值积分与采样项目（`301`、`498`、`942`、`1307`）转化为声源建模、散射积分与能量计算模块；
- ODE/PDE 求解相关项目（`1023`、`382`、`807`）转化为 PE 步进求解器、网格生成器与边界迭代器；
- 几何与特殊函数项目（`905`、`1265`、`113`、`1426`、`1082`）转化为地形编码、域掩码、接收器阵列与格林函数模块；
- 组合数学项目（`899`、`894`、`044`）转化为模态离散约束求解、谱元基转换与数值稳定性工具。

### 7.2 高难公式注入

在代码与文档中系统注入以下公式体系：
- **波动方程类**：Helmholtz → PE 的推导、WAPE 的 Padé 近似、SSF 分裂步公式；
- **环境模型类**：Munk SSP、Thorp 吸收、状态方程、地形参数化；
- **离散方法类**：Crank-Nicolson、变网格差分、Chebyshev tau 谱离散、三项递推；
- **边界条件类**：Dirichlet、Robin 导纳、PML 复坐标伸展、Snell 反射；
- **后处理类**：TL 定义、混响级、空间相关、时延扩展、相干带宽、WKB 模态估计。

### 7.3 复杂度升级

- **数值方法**：从简单显式格式升级为 Crank-Nicolson 隐式格式 + Chebyshev 谱元；
- **边界处理**：引入非线性 Robin 边界自洽迭代 + PML 吸收层；
- **物理模型**：整合 Munk 声道、体积散射、混响、海底地形等多物理场耦合；
- **工程鲁棒性**：所有除法操作均带安全掩码，Gamma/阶乘计算采用 log-domain，PML 避免反射污染，能量守恒实时监测。

---

## 八、质量检查清单

- [x] 原目录未被修改
- [x] 合成后的项目为 Python 语言
- [x] 新目录完整包含合成后的项目（11 个 .py 文件）
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] 15 个输入项目均已真实融入，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性（安全除法、log-gamma、PML、能量监测）
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
