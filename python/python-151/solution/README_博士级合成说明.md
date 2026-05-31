# 变分量子本征求解器 (VQE) — 博士级科研代码合成项目

## 一、项目概述

本项目围绕**量子计算：变分量子本征求解器（Variational Quantum Eigensolver, VQE）**展开，基于15个科研代码项目的核心算法，融合构建了一个面向分子基态能量计算的博士级计算框架。

**科学问题**：使用自适应变分量子本征求解器求解分子电子哈密顿量的基态能量，并与全组态相互作用（FCI）精确解进行对比，验证算法精度与收敛性。

**核心公式**：

分子电子哈密顿量（第二量子化形式）：
$$\hat{H} = \sum_{pq} h_{pq} \hat{a}_p^\dagger \hat{a}_q + \frac{1}{2}\sum_{pqrs} g_{pqrs} \hat{a}_p^\dagger \hat{a}_q^\dagger \hat{a}_r \hat{a}_s$$

其中 $h_{pq}$ 为单电子积分（动能 + 核吸引），$g_{pqrs}$ 为双电子排斥积分。

VQE能量泛函：
$$E(\vec{\theta}) = \langle \psi(\vec{\theta}) | \hat{H} | \psi(\vec{\theta}) \rangle$$

参数位移规则（Parameter-Shift Rule）梯度：
$$\frac{\partial E}{\partial \theta_i} = \frac{1}{2}\left[E\left(\theta_i + \frac{\pi}{2}\right) - E\left(\theta_i - \frac{\pi}{2}\right)\right]$$

Jordan-Wigner变换（费米子 $\to$ Pauli字符串）：
$$\hat{a}_j^\dagger = \frac{1}{2}\left(\prod_{k<j} Z_k\right)(X_j - iY_j)$$

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 融入文件 | 科学角色 |
|--------|---------|---------|---------|
| 1368_tumor_pde | 肿瘤PDE参数管理、边界条件、初始条件 | `banded_solver.py` | PDE系数函数管理、稳态方程离散化作为VQE经典基准 |
| 987_r8pbl | 对称正定带状矩阵存储与运算 | `banded_solver.py` | 有限差分离散化后的带状线性系统求解 |
| 398_fem1d_sample | 一维有限元采样与插值 | `fem_sampler.py` | 能量-参数曲线的有限元插值 |
| 176_circle_arc_grid | 圆弧网格生成 | `ansatz_tree.py` | Bloch球大圆上的参数初始化采样 |
| 495_gyroscope_ode | 陀螺仪欧拉角动力学 | `optimizer_geodesic.py` | 参数空间测地线旋转流优化 |
| 1048_rref2 | 行最简形(RREF)计算 | `pauli_operator.py` | Pauli测量集合线性无关性提取 |
| 995_r8sm | Sherman-Morrison低秩更新 | `pauli_operator.py` | 迭代过程中耦合矩阵的低秩逆更新 |
| 1021_rejection_sample | Chebyshev/CVT拒绝采样 | `measurement_sampler.py` | 量子Born概率的统计采样与方差缩减 |
| 1206_test_eigen | 随机正交/对称矩阵生成 | `hamiltonian_builder.py` | 哈密顿量随机实例测试与谱分析 |
| 467_gen_laguerre_rule | Laguerre高斯求积、Jacobi矩阵 | `hamiltonian_builder.py` | 分子轨道径向积分的高精度数值计算 |
| 418_fem3d_project | 3D有限元四面体基础函数 | `fem_sampler.py` | 3D参数空间上的概率密度投影 |
| 558_hypercube_grid | 多维超立方体网格 | `molecular_grid.py` | 分子几何空间的多维笛卡尔积分网格 |
| 518_hermite_cubic | Hermite三次样条插值 | `optimizer_geodesic.py` | 能量曲面局部近似与自适应步长控制 |
| 1291_treepack | 完全二叉树遍历、图树检测 | `ansatz_tree.py` | 参数化量子电路的树结构表示与连通性分析 |
| 263_cvtm_1d | 一维镜像周期CVT优化 | `molecular_grid.py` | 电子密度分布适配的优化积分节点 |

## 三、新增数学物理模型与核心公式

### 3.1 Gross-Pitaevskii型非线性薛定谔方程（经典基准）

用于验证VQE的经典偏微分方程基准：
$$c(x,t,u)\frac{\partial u}{\partial t} = \nabla \cdot f(x,t,u,\nabla u) + s(x,t,u)$$

其中扩散-反应-对流耦合系数：
$$f_0 = \delta \partial_x u_0, \quad f_1 = \varepsilon \partial_x u_1 - k u_1 \partial_x u_0$$
$$s_0 = -\frac{\alpha u_0 u_1}{\gamma + u_0} - \lambda u_1, \quad s_1 = \mu u_1(1-u_1)\max(u_0-c^*,0) - \beta u_1$$

### 3.2 DIF2矩阵的解析谱

有限差分离散化的三对角矩阵：
$$A = \text{tridiag}(-1, 2, -1)$$

特征值与特征向量：
$$\lambda_i = 4\sin^2\left(\frac{i\pi}{2(n+1)}\right), \quad X_i(j) = \sqrt{\frac{2}{n+1}}\sin\left(\frac{ij\pi}{n+1}\right)$$

### 3.3 广义Laguerre高斯求积

权函数 $w(x) = x^\alpha e^{-x}$ 的数值积分：
$$\int_0^\infty x^\alpha e^{-x} f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)$$

节点 $x_i$ 为Jacobi矩阵特征值，权重 $w_i = z_{0i}^2$（$z_{0i}$为特征向量首分量）。
Jacobi矩阵元：
$$a_j = 2j + 1 + \alpha, \quad b_j = \sqrt{j(j+\alpha)}$$

### 3.4 参数位移规则（Parameter-Shift Rule）

对单量子比特旋转门 $R_P(\theta) = e^{-i\theta P/2}$（$P \in \{X,Y,Z\}$）：
$$\frac{\partial \langle H \rangle}{\partial \theta} = \frac{1}{2}\left[\langle H \rangle_{\theta+\pi/2} - \langle H \rangle_{\theta-\pi/2}\right]$$

### 3.5 Sherman-Morrison低秩更新

对秩1修正矩阵 $B = A - \mathbf{u}\mathbf{v}^T$：
$$B^{-1} = A^{-1} + \frac{A^{-1}\mathbf{u}\mathbf{v}^T A^{-1}}{1 - \mathbf{v}^T A^{-1}\mathbf{u}}$$

### 3.6 四面体线性基础函数

对四面体顶点 $\mathbf{t}_1,\mathbf{t}_2,\mathbf{t}_3,\mathbf{t}_4$，基础函数：
$$\phi_i(\mathbf{p}) = \frac{\det([\mathbf{p}, \mathbf{t}_{j\neq i}, \mathbf{1}])}{\det([\mathbf{t}_1,\mathbf{t}_2,\mathbf{t}_3,\mathbf{t}_4,\mathbf{1}])}, \quad \sum_{i=1}^4 \phi_i = 1$$

### 3.7 Hermite三次样条

在区间 $[x_1, x_2]$ 上：
$$H(x) = f_1 + \Delta x\left(d_1 + \Delta x\left(c_2 + \Delta x \cdot c_3\right)\right)$$
其中 $h = x_2 - x_1$，$\Delta f = (f_2-f_1)/h$，
$$c_2 = -\frac{2d_1 - 3\Delta f + d_2}{h}, \quad c_3 = \frac{d_1 - 2\Delta f + d_2}{h^2}$$

### 3.8 陀螺仪欧拉方程

刚体旋转动力学（用于参数空间测地线模拟）：
$$\dot{\psi} = \frac{\omega_1\sin\phi + \omega_2\cos\phi}{\sin\theta}, \quad \dot{\theta} = \omega_1\cos\phi - \omega_2\sin\phi$$
$$\dot{\phi} = \omega_3 - \cos\theta\,\dot{\psi}$$
$$A_1\dot{\omega}_1 = (A_2-A_3)\omega_2\omega_3 + M_1, \quad \text{etc.}$$

## 四、项目文件结构

```
151_synth_project/
├── main.py                     # 统一入口，零参数运行
├── banded_solver.py            # 带状矩阵与PDE离散化
├── pauli_operator.py           # Pauli算符代数与稀疏线性代数
├── ansatz_tree.py              # 自适应Ansatz树与圆弧参数化
├── optimizer_geodesic.py       # 测地线梯度流与Hermite样条
├── fem_sampler.py              # 有限元采样与3D基础函数
├── molecular_grid.py           # 分子积分网格与CVT优化
├── measurement_sampler.py      # 量子测量与拒绝采样
├── hamiltonian_builder.py      # 高斯求积与随机矩阵
├── vqe_core.py                 # VQE核心算法整合
└── README_博士级合成说明.md    # 中文说明文档
```

## 五、运行方式

```bash
python main.py
```

程序将依次执行以下验证流程：
1. 带状矩阵DIF2的特征值分析与稳态PDE求解
2. Pauli算符乘法、RREF线性无关性提取、Sherman-Morrison求解
3. HEA ansatz构建、圆弧参数初始化、状态向量验证
4. 陀螺仪ODE轨迹模拟、Hermite样条插值与积分
5. 1D FEM插值误差验证、四面体基础函数与体积计算
6. 3D超立方体网格生成、CVT径向网格优化、双电子积分近似
7. Chebyshev拒绝采样统计验证、Bell态量子测量模拟
8. Laguerre-Gauss求积精度验证、随机正交矩阵测试
9. **完整VQE计算**：构建H2模型 → FCI基准 → 精确VQE优化 → 含噪声VQE → 误差分析

## 六、边界处理与数值鲁棒性

1. **带状矩阵Cholesky分解**：检查正定性和零体积退化，对非正正定矩阵抛出明确异常。
2. **PDE系数函数**：$\max(u_1-c^*,0)$ 避免负值；分母 $\gamma+u_0$ 加入 $10^{-12}$ 正则化防止除零。
3. **陀螺仪ODE**：$\sin\theta \to 0$ 时正则化为 $\text{sgn}(\sin\theta)\times 10^{-10}$，避免欧拉角奇点。
4. **Hermite样条**：节点严格递增检查；区间长度小于 $10^{-14}$ 时退化到常数插值。
5. **四面体基础函数**：体积为零时抛出异常，防止退化单元。
6. **CVT优化**：权重为零的生成元保持原位；网格节点通过 `np.mod` 限制在周期边界内。
7. **参数优化**：参数限制在 $[-4\pi, 4\pi]$ 范围内；梯度范数低于阈值时自动终止。
8. **Sherman-Morrison**：检查 $1-\mathbf{v}^T A^{-1}\mathbf{u}$ 是否接近零，避免数值溢出。

## 七、科学问题总结

本项目实现的VQE框架能够：
- 构建第二量子化的分子电子哈密顿量（含单/双电子积分）
- 通过Jordan-Wigner变换映射为Pauli字符串表示
- 使用硬件高效ansatz（HEA）或ADAPT-VQE风格自适应电路准备试探态
- 利用参数位移规则计算解析梯度，避免有限差分的精度损失
- 通过测地线梯度流优化器在参数空间寻找能量极小值
- 与FCI精确对角化结果对比，评估VQE的绝对误差与相对误差
- 提供能隙估计、收敛速率分析、误差上界计算等诊断工具

这是一个完整的、可运行的量子化学计算原型系统，展示了经典数值方法（有限元、ODE积分、样条插值、矩阵算法）与量子计算理论的深度交叉融合。
