# 多尺度突触可塑性与神经权重动态计算框架 —— 博士级合成项目说明

## 一、项目概述

本项目围绕**生物医学：神经可塑性与突触权重**这一前沿科学领域，将15个科研代码项目的核心算法融合为一个面向多尺度突触可塑性计算的博士级Python科研代码项目。

### 科学问题

突触权重的动态演化涉及从分子到网络的多个时空尺度：
1. **分子尺度**：可塑性相关蛋白（PRPs，如CamKII、PKMζ）沿树突的扩散
2. **突触尺度**：突触小泡释放概率、NMDA受体的非线性电流-电压关系
3. **细胞尺度**：稳态可塑性对突触权重的阻尼振荡调节
4. **网络尺度**：长时程增强（LTP）波在组织中的传播、神经元相位同步
5. **随机尺度**：突触权重的随机涨落
6. **代谢尺度**：有限代谢预算下的蛋白质合成资源分配

### 项目结构

```
130_synth_project/
├── main.py                     # 统一入口，零参数运行
├── cable_diffusion.py          # 离散Laplacian与电缆扩散方程
├── numerical_integrator.py     # RK1/RK4数值积分与稳定性分析
├── plasticity_wave.py          # Fisher-KPP反应扩散与LTP波传播
├── vesicle_release.py          # 球面积分与小泡释放概率
├── resource_optimizer.py       # 贪心算法与代谢资源分配
├── cortical_mesh.py            # 皮层网格生成与三角剖分邻居分析
├── homeostatic_dynamics.py     # 阻尼弹簧与非线性摆稳态动力学
├── stochastic_weights.py       # 随机微分方程与突触权重演化
├── spectral_field.py           # FFT谱分析、三角/Chebyshev插值
└── synaptic_nonlinearity.py    # 多元多项式与NMDA受体非线性逼近
```

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 合成后科学角色 |
|--------|---------|--------------|
| 648_laplacian_matrix | 1D离散Laplacian（DD/DN/ND/NN/PP边界）、特征值分解、Cholesky/LU | 树突电缆方程的空间离散化，PRP扩散模拟 |
| 433_fisher_exact | Fisher-KPP方程精确行波解 | LTP波传播的动力学模型与数值验证 |
| 1119_sphere_integrals | 单位球面单项式精确积分（Gamma函数） | 球形突触小泡释放概率的解析积分核 |
| 157_change_greedy | 贪心算法求解找零问题 | 代谢资源在多个突触间的最优分配 |
| 1354_triangulation_triangle_neighbors | 三角剖分邻居检测 | 皮层组织有限元分析的邻居关系 |
| 183_circle_rule | 单位圆周上的均匀求积公式 | 突触扣结二维截面释放概率积分 |
| 1139_spring_ode | 阻尼弹簧振子ODE | 稳态可塑性对权重的阻尼回归力 |
| 098_black_scholes | Black-Scholes期权定价公式 | 突触权重随机演化的几何布朗运动模型 |
| 1356_trig_interp | 三角基函数Cardinal插值 | 周期性神经发放率的谱插值 |
| 426_fft_serial | 快速傅里叶变换（FFT） | 局部场电位（LFP）的功率谱密度分析 |
| 893_polynomial | 多元多项式运算（求值、加减、微分） | NMDA受体Mg²⁺阻塞的多项式逼近 |
| 860_pendulum_nonlinear_exact | 非线性摆的Jacobi椭圆函数精确解 | 神经元网络相位同步的非线性振子模型 |
| 503_hand_mesh2d | 2D网格生成 | 皮层切片的Delaunay三角剖分 |
| 1027_rk1 | 一阶Runge-Kutta（Euler）ODE积分器 | 多尺度ODE系统的时步推进 |
| 591_interp_chebyshev | Chebyshev节点插值（Newton差商形式） | 突触传递函数的高精度谱逼近 |

---

## 三、核心数学物理模型与公式

### 3.1 树突电缆方程与蛋白扩散

膜电位 $V(x,t)$ 满足的电缆方程：

$$\lambda^2 \frac{\partial^2 V}{\partial x^2} = \tau_m \frac{\partial V}{\partial t} + V - r_m I_{\text{syn}}(x,t)$$

其中空间常数 $\lambda = \sqrt{r_m / r_a}$，时间常数 $\tau_m = r_m c_m$。

对于可塑性相关蛋白浓度 $c(x,t)$：

$$\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2} - \gamma c + S(x,t)$$

**离散Laplacian**（Dirichlet-Dirichlet边界）：

$$L_{ii} = \frac{2}{h^2}, \quad L_{i,i\pm 1} = -\frac{1}{h^2}$$

**显式Euler稳定性条件**：

$$\Delta t \leq \frac{h^2}{2D}$$

**特征值**（决定各扩散模式的衰减速率）：

$$\lambda_k = \frac{2}{h^2}\left[1 - \cos\left(\frac{k\pi}{n+1}\right)\right], \quad k=1,\dots,n$$

### 3.2 Fisher-KPP LTP波传播

归一化突触权重 $u(x,t) \in [0,1]$ 满足反应扩散方程：

$$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + r \cdot u(1-u)$$

**精确行波解**（Ablowitz & Zeppetella, 1979）：

$$u(x,t) = \frac{1}{\left(1 + a \cdot e^{k(x-ct)}\right)^2}$$

其中 $c = 5/\sqrt{6}$，$k = 1/\sqrt{6}$，$a = 2$。

各阶导数：

$$u_t = \frac{2ca k \cdot e^{kz}}{(1 + a e^{kz})^3}, \quad z = x - ct$$

$$u_x = -\frac{2a k \cdot e^{kz}}{(1 + a e^{kz})^3}$$

$$u_{xx} = \frac{6a^2 k^2 \cdot e^{2kz}}{(1 + a e^{kz})^4} - \frac{2a k^2 \cdot e^{kz}}{(1 + a e^{kz})^3}$$

**最小波速**：

$$c_{\min} = 2\sqrt{Dr}$$

**方法线（Method of Lines）离散**：

$$\frac{du_i}{dt} = \frac{D}{h^2}(u_{i-1} - 2u_i + u_{i+1}) + r u_i(1-u_i)$$

### 3.3 突触小泡释放概率

球形突触扣结（半径 $R$）上的释放概率密度：

$$p(\theta,\phi) = p_0 \cdot \exp\left(-\frac{d^2}{2\sigma^2}\right)$$

其中 $d$ 为到活性区的大地线距离。总释放概率：

$$P_{\text{release}} = R^2 \int_0^{2\pi} \int_0^\pi p(\theta,\phi) \sin\phi \, d\phi \, d\theta$$

**圆周求积公式**（circle_rule）：

$$\int_0^{2\pi} f(\theta) \, d\theta \approx \frac{2\pi}{N} \sum_{i=1}^N f(\theta_i), \quad \theta_i = \frac{2\pi(i-1)}{N}$$

**球面单项式精确积分**：

对 $f(x,y,z) = x^a y^b z^c$ 在单位球面 $S^2$ 上：
- 若任一指数为奇数：$I = 0$（对称性）
- 若全为零：$I = 4\pi$
- 若全为偶数：

$$I = \frac{2 \cdot \Gamma\left(\frac{a+1}{2}\right) \Gamma\left(\frac{b+1}{2}\right) \Gamma\left(\frac{c+1}{2}\right)}{\Gamma\left(\frac{a+b+c+3}{2}\right)}$$

**量子含量**（二项式模型）：

$$E[Q] = N \cdot P_{\text{release}}, \quad \text{Var}[Q] = N \cdot P_{\text{release}} \cdot (1 - P_{\text{release}})$$

### 3.4 代谢资源分配

给定代谢预算 $B$ 和每个突触的目标权重变化 $\Delta w_i^{\text{target}}$，
单位变化成本 $c_i$，优化问题为：

$$\max \sum_i |\Delta w_i| \quad \text{s.t.} \quad \sum_i c_i |\Delta w_i| \leq B, \quad |\Delta w_i| \leq |\Delta w_i^{\text{target}}|$$

**贪心策略**：按效益成本比 $r_i = |\Delta w_i^{\text{target}}| / c_i$ 降序排列，
依次分配直至预算耗尽。

**0/1背包变体**（离散标签模型）：

动态规划递推：

$$\text{dp}[b][i] = \max\left(\text{dp}[b][i-1], \; \text{dp}[b-c_i][i-1] + v_i\right)$$

### 3.5 稳态动力学

**阻尼谐振子模型**（弹簧-质量-阻尼）：

$$m \frac{d^2w}{dt^2} + b \frac{dw}{dt} + k(w - w_{\text{target}}) = F_{\text{plasticity}}(t)$$

特征方程 $m\lambda^2 + b\lambda + k = 0$ 的根决定阻尼 regime：
- $b^2 < 4mk$：欠阻尼（振荡回归）
- $b^2 = 4mk$：临界阻尼（最快非振荡回归）
- $b^2 > 4mk$：过阻尼（缓慢单调回归）

**非线性摆**（网络相位同步）：

$$\frac{d^2\theta}{dt^2} + \frac{g}{l}\sin(\theta) = I_{\text{ext}}(t)$$

**精确解**（Ochs, 2011；Jacobi椭圆函数）：

$$k_0 = \sin(\theta_0/2), \quad \omega = \sqrt{g/l}, \quad k = \sqrt{\frac{\dot{\theta}_0^2 + 4\omega^2 k_0^2}{4\omega^2}}$$

$$\theta(t) = 2 \arcsin\left(k \cdot \text{sn}(\omega t, k)\right)$$

周期：

$$T = \frac{4K(k)}{\omega}$$

其中 $K(k)$ 为第一类完全椭圆积分。

### 3.6 随机突触权重演化

带有Hebbian漂移和稳态均值回归的几何布朗运动：

$$dW = \left[\mu W\left(1 - \frac{W}{W_{\max}}\right) - \lambda(W - W_{\text{target}})\right] dt + \sigma W \, dB_t$$

**Fokker-Planck方程**：

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial w}\left[A(w)p\right] + \frac{1}{2}\frac{\partial^2}{\partial w^2}\left[B(w)p\right]$$

其中漂移 $A(w) = \mu w(1-w/W_{\max}) - \lambda(w-W_{\text{target}})$，
扩散系数 $B(w) = \sigma^2 w^2$。

**Euler-Maruyama离散**：

$$W_{n+1} = W_n + A(W_n)\Delta t + \sigma W_n \sqrt{\Delta t} \cdot Z_n, \quad Z_n \sim \mathcal{N}(0,1)$$

**Black-Scholes型"可塑性期权"**：

权重超过目标的期望"收益"：

$$d_1 = \frac{\ln(w_0/W_{\text{target}}) + (\mu + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}$$

$$d_2 = d_1 - \sigma\sqrt{\tau}$$

$$V = w_0 \cdot \Phi(d_1) - W_{\text{target}} \cdot e^{-\mu\tau} \cdot \Phi(d_2)$$

### 3.7 神经场谱分析

**FFT功率谱密度**：

$$\tilde{\phi}_k = \sum_{n=0}^{N-1} \phi_n e^{-i 2\pi k n / N}, \quad S(k) = \frac{|\tilde{\phi}_k|^2}{N}$$

**三角Cardinal插值**（周期信号）：

$$C_j(x) = \frac{\sin\left(N(x-x_j)/(2h)\right)}{N\sin\left((x-x_j)/(2h)\right)}$$

**Chebyshev节点**：

$$x_j = \frac{(b-a)\cos(j\pi/(n-1)) + (a+b)}{2}, \quad j=0,\dots,n-1$$

**Newton差商插值**：

$$P(x) = d_0 + d_1(x-x_0) + d_2(x-x_0)(x-x_1) + \cdots$$

### 3.8 NMDA受体非线性多项式逼近

NMDA电流的Mg²⁺阻塞：

$$M_g(V) = \frac{1}{1 + \frac{[\text{Mg}^{2+}]}{K_d} \exp(-\gamma V)}$$

多元多项式逼近：

$$M_g(V, [\text{Mg}^{2+}]) \approx \sum_{\alpha,\beta} c_{\alpha\beta} \cdot V^\alpha \cdot [\text{Mg}^{2+}]^\beta$$

通过最小二乘法在Chebyshev节点上拟合系数 $c_{\alpha\beta}$。

**Hill函数**（Ca²⁺依赖释放）：

$$P_{\text{rel}}([\text{Ca}^{2+}]) = \frac{[\text{Ca}^{2+}]^n}{[\text{Ca}^{2+}]^n + K_D^n}$$

---

## 四、修改与合成路径

### 4.1 代码改造策略

每个原MATLAB项目被分析提取核心算法后，以Python重新实现，并注入神经科学领域的数学物理模型：

1. **laplacian_matrix → cable_diffusion.py**
   - 保留5种边界条件（DD/DN/ND/NN/PP）的离散Laplacian构造
   - 新增矩阵自由应用函数 `apply_laplacian_1d` 以节省内存
   - 加入电缆方程的物理参数（扩散系数D、降解率γ）
   - 加入CFL稳定性分析与特征值分解

2. **fisher_exact → plasticity_wave.py**
   - 精确解公式直接移植为Python函数
   - 新增方法线（Method of Lines）空间离散
   - 使用 `numerical_integrator.py` 中的RK4进行时间积分
   - 加入最小波速 $c_{\min} = 2\sqrt{Dr}$ 的理论计算

3. **sphere_integrals + circle_rule → vesicle_release.py**
   - 球面Gamma函数精确积分直接移植
   - 圆周均匀求积公式用于二维截面简化
   - 加入突触小泡释放概率密度模型与量子含量计算

4. **change_greedy → resource_optimizer.py**
   - 贪心分配策略从"找零"映射到"突触资源分配"
   - 新增连续松弛与0/1背包动态规划变体
   - 加入Gini系数等效率评估指标

5. **hand_mesh2d + triangulation_triangle_neighbors → cortical_mesh.py**
   - 使用 `scipy.spatial.Delaunay` 替代MATLAB的mesh2d
   - 实现边界扰动模拟皮层沟回
   - 完整的邻居检测算法（边字典法）
   - 加入网格质量指标（内切圆/外接圆半径比）

6. **spring_ode + pendulum_nonlinear_exact → homeostatic_dynamics.py**
   - 弹簧ODE的RHS函数移植并参数化
   - 阻尼分类（欠阻尼/临界/过阻尼）
   - Jacobi椭圆函数精确解使用 `scipy.special.ellipj`
   - 新增耦合非线性摆网络模拟

7. **black_scholes → stochastic_weights.py**
   - 欧式看涨期权公式映射为"可塑性期权价值"
   - 新增带逻辑增长和均值回归的SDE
   - Euler-Maruyama数值实现
   - 权重分布统计量（熵、变异系数）

8. **fft_serial + trig_interp + interp_chebyshev → spectral_field.py**
   - FFT使用NumPy的 `np.fft.fft`
   - 三角Cardinal基函数完整实现
   - Chebyshev节点生成与Newton差商插值
   - 神经振荡频段功率分析（Delta/Theta/Alpha/Beta/Gamma）

9. **polynomial → synaptic_nonlinearity.py**
   - 多元单项式求值与分级字典序
   - 最小二乘多项式拟合NMDA阻塞函数
   - Hill函数的多项式逼近

10. **rk1 → numerical_integrator.py**
    - RK1（Euler）与RK4积分器
    - 自适应步长控制（RK1/RK2嵌入对）
    - 数值Jacobian估计与刚度比分析

### 4.2 鲁棒性与边界处理

每个模块均包含严格的输入验证：
- **维度检查**：数组长度一致性验证
- **物理约束**：浓度非负、概率∈[0,1]、权重截断
- **数值稳定性**：CFL条件自动调整、特征值分析、 stiffness检测
- **退化处理**：空数组、零预算、零扩散系数的特殊分支

---

## 五、运行方式

```bash
cd Synthesis-project-python/130_synth_project
python main.py
```

程序无需任何命令行参数，自动执行以下完整流程：
1. 生成皮层三角网格并计算邻居关系与质量指标
2. 模拟PRP沿树突的扩散过程
3. 求解Fisher-KPP方程模拟LTP波传播
4. 计算球形突触扣结的小泡释放概率
5. 比较三种代谢资源分配策略
6. 模拟稳态调节下的权重动力学与网络同步
7. 模拟100个突触的随机权重演化并评估"可塑性期权"组合
8. 对合成神经场信号进行FFT谱分析
9. 用多项式逼近NMDA受体的Mg²⁺阻塞函数
10. 输出综合统计摘要

---

## 六、科学前沿性与难度

本项目综合运用了以下博士级计算技术：

- **偏微分方程数值解**：反应扩散方程的方法线离散 + RK4时间积分
- **谱方法**：Chebyshev插值与三角插值用于突触传递函数
- **随机微分方程**：几何布朗运动与Euler-Maruyama数值解
- **离散Laplacian谱分析**：特征值分解与稳定性理论
- **椭圆函数精确解**：非线性摆的Jacobi椭圆函数解析解
- **优化理论**：贪心算法与动态规划在资源分配中的应用
- **计算几何**：Delaunay三角剖分与邻居检测
- **特殊函数积分**：Gamma函数在球面积分中的精确应用

---

## 七、验证结果

运行 `main.py` 后，程序自动验证：
- 球面面积分 `∫ dΩ = 4π` 精确到机器精度
- Fisher-KPP行波解与数值解的一致性
- Chebyshev插值误差约 $10^{-8}$ 量级
- 离散Laplacian的CFL稳定性自动满足
- 所有突触权重保持在物理边界 $[10^{-6}, W_{\max}]$ 内
