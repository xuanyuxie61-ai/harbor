# 光纤非线性脉冲传输 — 博士级综合科学计算项目

## 1. 项目概述

本项目围绕**光学工程：光纤非线性脉冲传输**这一前沿科学领域，基于15个种子科研代码项目的核心算法，融合构建了一个面向光子晶体光纤中超短脉冲非线性动力学的高难度综合计算项目。

### 1.1 科学问题

研究光子晶体光纤（PCF）中超短光脉冲（~100 fs）在受激拉曼散射（SRS）、自相位调制（SPM）、交叉相位调制（XPM）及放大自发辐射（ASE）噪声共同作用下的传输特性。项目涵盖：

- **广义非线性薛定谔方程（GNLSE）**的数值求解
- **Jacobi多项式谱方法**的高精度频域展开
- **光纤截面三角剖分**与有效模场面积计算
- **球面/超球蒙特卡洛采样**用于参数不确定性量化
- **DREAM MCMC**贝叶斯参数反演
- **GMRES稀疏求解器**用于隐式色散算子系统
- **Laguerre根查找**用于光纤模式特征方程求解
- **分段线性重叠积分**用于脉冲间非线性相互作用

---

## 2. 原项目到科学问题的映射

| 原种子项目 | 核心算法 | 在合成项目中的角色 | 对应文件 |
|:---|:---|:---|:---|
| 607_jacobi_polynomial | Jacobi多项式计算与高斯求积 | 脉冲包络的谱展开、色散算子的谱离散化 | `jacobi_spectral.py` |
| 1308_triangle_integrands | 三角形区域高斯积分 | 光纤截面有效模场面积计算 | `fiber_geometry.py` |
| 119_brownian_motion_simulation | 布朗运动随机游走 | ASE噪声模型、相位噪声统计 | `noise_model.py` |
| 929_pwl_product_integral | 分段线性函数乘积积分 | 脉冲间XPM重叠积分、Raman卷积 | `pulse_overlap.py` |
| 554_hyperball_monte_carlo | 超球体均匀采样 | 光纤参数高维不确定性量化 | `monte_carlo_sampler.py` |
| 132_caesar | Caesar循环移位 | 频谱循环卷积相位编码算子 | `phase_coding.py` |
| 1113_sphere_cvt | 球面CVT | 远场辐射方向分布优化 | `monte_carlo_sampler.py` |
| 284_digital_dice | Parrondo概率悖论 | 噪声耦合模型、光子统计 | `noise_model.py` |
| 708_magic_matrix | 幻方矩阵生成 | 特殊相位掩码构造 | `phase_coding.py` |
| 319_dream | DREAM MCMC算法 | 光纤非线性系数贝叶斯反演 | `mcmc_inversion.py` |
| 1126_sphere_quad | 球面数值积分 | 远场辐射强度积分 | `monte_carlo_sampler.py` |
| 1430_zero_laguerre | Laguerre根查找 | 光纤模式传播常数求解 | `rootfinder.py` |
| 760_mgmres | 重启GMRES迭代 | 色散算子稀疏线性系统求解 | `sparse_solver.py` |
| 1333_triangulation_boundary_nodes | 三角剖分边界识别 | 光纤截面有限元边界条件 | `fiber_geometry.py` |
| 445_four_fifths | 整数组合搜索 | WDM信道色散失配优化 | `phase_coding.py` |

---

## 3. 新增数学物理模型与核心公式

### 3.1 广义非线性薛定谔方程（GNLSE）

$$\frac{\partial A}{\partial z} = -\frac{\alpha}{2} A + \sum_{m\geq 2} \frac{i^{m+1}}{m!} \beta_m \frac{\partial^m A}{\partial T^m} + i\gamma \left(1 + \frac{i}{\omega_0}\frac{\partial}{\partial T}\right)\left[ A(z,T) \int_0^\infty R(T')|A(z,T-T')|^2 \, dT' \right]$$

其中：
- $A(z,T)$：脉冲慢变包络（$\mathrm{W}^{1/2}$）
- $\alpha$：光纤损耗（$\mathrm{m}^{-1}$）
- $\beta_m$：$m$阶色散系数
- $\gamma = n_2 \omega_0 / (c A_{\mathrm{eff}})$：非线性系数
- $R(T) = (1-f_R)\delta(T) + f_R h_R(T)$：Raman响应函数
- $h_R(T) = \frac{\tau_1^2+\tau_2^2}{\tau_1\tau_2^2}\exp(-T/\tau_2)\sin(T/\tau_1)\cdot\Theta(T)$

### 3.2 分步傅里叶法（SSFM）

$$A(z+h,\omega) = \exp\left(\frac{h}{2}\hat{D}\right) \exp\left(h\hat{N}\right) \exp\left(\frac{h}{2}\hat{D}\right) A(z,\omega)$$

色散算子：$\hat{D}(\omega) = -\frac{\alpha}{2} + i\frac{\beta_2}{2}\omega^2 - i\frac{\beta_3}{6}\omega^3 + \frac{\beta_4}{24}\omega^4$

### 3.3 Jacobi多项式正交展开

将时域脉冲映射到 $[-1,1]$ 区间后展开：

$$A(t) \approx \sum_{n=0}^{N-1} c_n P_n^{(\alpha,\beta)}(\tau), \quad \tau = \frac{2(t-t_{\min})}{T_{\mathrm{window}}}-1$$

正交归一化常数：

$$h_n = \frac{2^{\alpha+\beta+1}\Gamma(n+\alpha+1)\Gamma(n+\beta+1)}{(2n+\alpha+\beta+1)n!\,\Gamma(n+\alpha+\beta+1)}$$

Gauss-Jacobi求积：$\int_{-1}^1 (1-\tau)^\alpha(1+\tau)^\beta f(\tau)\,d\tau \approx \sum_{i=1}^N w_i f(\tau_i)$

### 3.4 有效模场面积

$$A_{\mathrm{eff}} = \frac{\left(\iint |E(x,y)|^2 \,dxdy\right)^2}{\iint |E(x,y)|^4 \,dxdy}$$

利用三角形Wandzura 5阶高斯积分规则在三角剖分网格上数值计算。

### 3.5 光纤模式特征方程（标量弱导近似）

对于$LP_{lm}$模式，横向相位参数$u$满足：

$$u \frac{J_{\nu+1}(u)}{J_\nu(u)} = w \frac{K_{\nu+1}(w)}{K_\nu(w)}, \quad w = \sqrt{V^2-u^2}$$

其中$V = ak_0\sqrt{n_{\mathrm{core}}^2 - n_{\mathrm{clad}}^2}$为归一化频率。

### 3.6 ASE噪声功率谱密度

$$\rho_{\mathrm{ASE}} = n_{\mathrm{sp}}(G-1)h\nu$$

基于Wiener过程（布朗运动）的噪声演化模型：

$$\phi(z+\Delta z) = \phi(z) + \sqrt{2\alpha_{\mathrm{coh}}\Delta z}\,\xi, \quad \xi\sim\mathcal{N}(0,1)$$

### 3.7 DREAM MCMC参数反演

后验分布：$p(\boldsymbol{\theta}|\mathbf{y}_{\mathrm{obs}}) \propto p(\mathbf{y}_{\mathrm{obs}}|\boldsymbol{\theta})p(\boldsymbol{\theta})$

候选生成（差分进化）：$\boldsymbol{\theta}_p = \boldsymbol{\theta}_{\mathrm{chain}} + \gamma \sum_{j=1}^{\delta}(\boldsymbol{\theta}_{a_j} - \boldsymbol{\theta}_{b_j}) + \boldsymbol{\varepsilon}$

Metropolis比率：$\alpha = \min\left(1, \frac{L(\boldsymbol{\theta}_p)p(\boldsymbol{\theta}_p)}{L(\boldsymbol{\theta}_{\mathrm{old}})p(\boldsymbol{\theta}_{\mathrm{old}})}\right)$

### 3.8 GMRES Krylov子空间迭代

在Krylov子空间 $\mathcal{K}_m(\mathbf{A},\mathbf{r}_0) = \mathrm{span}\{\mathbf{r}_0, \mathbf{A}\mathbf{r}_0, \ldots, \mathbf{A}^{m-1}\mathbf{r}_0\}$ 中最小化残差：

$$\min_{\mathbf{x}\in\mathbf{x}_0+\mathcal{K}_m} \|\mathbf{b}-\mathbf{A}\mathbf{x}\|_2$$

---

## 4. 文件结构

```
103_synth_project/
├── main.py                  # 统一入口，零参数运行
├── jacobi_spectral.py       # Jacobi多项式谱方法（种子607）
├── pulse_overlap.py         # 分段线性重叠积分（种子929）
├── fiber_geometry.py        # 光纤几何三角剖分（种子1308, 1333）
├── monte_carlo_sampler.py   # 蒙特卡洛与球面积分（种子554, 1126, 1113）
├── noise_model.py           # 噪声与概率统计（种子119, 284）
├── sparse_solver.py         # GMRES稀疏求解器（种子760）
├── mcmc_inversion.py        # DREAM MCMC反演（种子319）
├── rootfinder.py            # Laguerre根查找与模式分析（种子1430）
├── phase_coding.py          # 相位编码与整数搜索（种子132, 708, 445）
├── gnlse_solver.py          # GNLSE核心求解器
└── README_博士级合成说明.md  # 本文档
```

---

## 5. 合成后的项目能够解决的科学问题

1. **超短脉冲非线性传输仿真**：基于SSFM的GNLSE数值求解，可分析SPM、SRS、自陡峭效应对脉冲时域/频域演化的影响。

2. **光纤模式分析**：求解标量波动方程特征值问题，计算各LP模式的传播常数和有效折射率。

3. **有效非线性系数计算**：通过三角剖分网格上的高斯积分精确计算$A_{\mathrm{eff}}$和$\gamma$。

4. **参数不确定性量化**：在5维超球参数空间中蒙特卡洛采样，评估制造公差对传输性能的影响。

5. **贝叶斯参数反演**：利用DREAM MCMC从观测频谱中同时反演$\gamma$、$\beta_2$、$\beta_3$、$\alpha$、$T_R$等参数。

6. **相位编码与WDM优化**：应用幻方相位掩码和整数搜索优化信道分配，最小化四波混频效率。

---

## 6. 运行方式

```bash
cd 103_synth_project
python main.py
```

程序无需任何输入参数，将自动执行以下10个计算演示模块：

1. Jacobi谱方法脉冲展开
2. 分段线性脉冲重叠积分
3. 光纤截面三角剖分与$A_{\mathrm{eff}}$计算
4. 高维蒙特卡洛采样与球面积分
5. ASE噪声模型与玻色-爱因斯坦统计
6. GMRES稀疏线性系统求解
7. DREAM MCMC参数贝叶斯反演
8. Laguerre根查找光纤模式分析
9. 幻方相位掩码与WDM整数优化
10. GNLSE超短脉冲非线性传输仿真

---

## 7. 边界处理与数值鲁棒性

- **Jacobi参数检查**：强制$\alpha,\beta > -1$，使用`gammaln`避免Gamma函数溢出
- **光纤几何验证**：芯径/包层半径必须满足$0 < r_{\mathrm{core}} < r_{\mathrm{cladding}}$
- **GMRES重正交化**：当更新过小（`av + delta*h(k+1,k) == av`）时触发重正交化
- **SSFM数值稳定性**：每步检查`isfinite`，能量守恒监控
- **根查找边界保护**：二分法保证在$(0,V)$区间内收敛，避免Laguerre方法的发散
- **MCMC边界截断**：所有候选参数强制截断到先验边界内

---

## 8. 关键物理常数与典型参数

| 参数 | 符号 | 数值 | 单位 |
|:---|:---|:---|:---|
| 光速 | $c$ | $2.99792458\times10^8$ | m/s |
| 普朗克常数 | $h$ | $6.62607015\times10^{-34}$ | J·s |
| 非线性折射率 | $n_2$ | $2.6\times10^{-20}$ | m²/W |
| 二阶色散 | $\beta_2$ | $-20\times10^{-27}$ | s²/m |
| 非线性系数 | $\gamma$ | $1.5\times10^{-3}$ | 1/(W·m) |
| Raman分数 | $f_R$ | $0.18$ | — |
| Raman时间常数 | $\tau_1,\tau_2$ | $12.2, 32.0$ | fs |
| 中心波长 | $\lambda_0$ | $1550$ | nm |
