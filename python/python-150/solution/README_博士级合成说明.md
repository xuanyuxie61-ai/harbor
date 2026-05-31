# 神经计算：图神经网络分子性质预测 — 博士级合成说明

## 1. 项目概述

本项目围绕**神经计算：图神经网络分子性质预测**这一前沿领域，将 15 个原始科研代码项目的核心算法融合为一个完整的博士级科学计算系统。项目实现了从分子图构建、谱分析、静电势求解、物理信息损失到图神经网络训练与不确定性量化的端到端流程，全部使用 Python 编写，零参数可运行。

## 2. 种子项目映射

| 原始项目 | 核心算法 | 在本项目中的角色 |
|---------|---------|----------------|
| 756_mesh_vtoe | 顶点-单元 CSR 风格稀疏邻接 | `molecular_graph.py`：分子图的顶点-边关联映射与稀疏拉普拉斯构建 |
| 1405_web_matrix | 稀疏三元组、关联矩阵→转移矩阵、幂迭代 | `molecular_graph.py`：COO 格式稀疏矩阵、PageRank 风格原子重要性排序 |
| 160_chebyshev_interp_1d | Chebyshev 插值、Vandermonde 系统 | `chebyshev_conv.py`：Chebyshev 谱图卷积层与一维插值工具 |
| 869_pic | 粒子云网格电荷沉积、Gauss-Seidel、双线性插值 | `electrostatic_solver.py`：分子电荷沉积、Poisson 方程求解、电场插值 |
| 1304_triangle_felippa_rule | 三角形对称 Gauss 求积 | `numerical_quadrature.py`：分子表面三角形积分 |
| 685_line_nco_rule | Newton-Cotes Open 一维求积 | `numerical_quadrature.py`：沿化学键的线积分 |
| 893_polynomial | 多元多项式 grlex 排序、求值、微分 | `polynomial_basis.py`：分子内坐标多项式描述符 |
| 063_backward_euler | 隐式后向 Euler ODE 求解 | `dynamics_integrator.py`：分子结构梯度流隐式松弛 |
| 918_prob | 概率分布、特殊函数 (ERF、Gamma、Bessel、Digamma) | `uncertainty_model.py`：NIG 证据回归、不确定性量化 |
| 850_partition_greedy | 贪心图划分 | `graph_utils.py`：分子图 mini-batch 贪心划分 |
| 899_polyomino_parity | Diophantine 约束与奇偶性 | `graph_utils.py`：分子对称性阶数分解、奇偶性检验 |
| 556_hypercube_distance | 高维距离统计 | `graph_utils.py`：描述符空间距离分布与均匀性度量 |
| 420_fermat_factor | Fermat 整数分解 | `graph_utils.py`：图拓扑哈希指纹生成 |
| 185_circles | 参数化圆/球基函数 | `graph_utils.py`：原子轨道角度基、局部各向异性描述符 |
| 586_image_threshold | 阈值分割 | `feature_engineering.py`：多层阈值分子指纹、Coulomb 矩阵二值化 |

## 3. 核心科学模型与公式

### 3.1 图拉普拉斯与谱归一化

分子图 $G=(V,E)$ 的加权邻接矩阵 $A$ 满足 $A_{ij} = \text{order}_{ij} / r_{ij}^2$。度矩阵 $D_{ii}=\sum_j A_{ij}$。图拉普拉斯：

$$
L = D - A
$$

为将特征值缩放到 $[-1,1]$ 以适配 Chebyshev 多项式，构造归一化拉普拉斯：

$$
\tilde{L} = \frac{2L}{\lambda_{\max}} - I
$$

其中 $\lambda_{\max}$ 通过幂迭代（Rayleigh 商）估计：

$$
x_{k+1} = \frac{L x_k}{\|L x_k\|}, \quad \lambda_{\max} \approx x_k^T L x_k
$$

### 3.2 Chebyshev 谱图卷积

图卷积由谱域滤波器定义 $y = U g_\theta(\Lambda) U^T x$。采用 Chebyshev 多项式截断逼近：

$$
g_\theta(\tilde{\Lambda}) \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{L})
$$

其中 $T_k$ 满足递推关系：

$$
T_0(x) = 1, \quad T_1(x) = x, \quad T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
$$

计算复杂度为 $O(K|E|)$，避免了 $O(N^3)$ 的特征分解。

### 3.3 静电势 Poisson 方程

电荷密度 $\rho(\mathbf{r})$ 与静电势 $\phi(\mathbf{r})$ 满足：

$$
\nabla^2 \phi = -\frac{\rho}{\varepsilon_0}
$$

在均匀网格上采用 7 点有限差分 Laplacian：

$$
(\nabla^2 \phi)_{i,j,k} \approx \frac{\phi_{i+1,j,k}-2\phi_{i,j,k}+\phi_{i-1,j,k}}{\Delta x^2} + \frac{\phi_{i,j+1,k}-2\phi_{i,j,k}+\phi_{i,j-1,k}}{\Delta y^2} + \frac{\phi_{i,j,k+1}-2\phi_{i,j,k}+\phi_{i,j,k-1}}{\Delta z^2}
$$

通过 Gauss-Seidel 迭代配合 SOR 加速 ($\omega=1.5$) 求解。电场 $\mathbf{E}=-\nabla\phi$ 通过中心差分计算。

### 3.4 隐式结构松弛

分子几何优化可建模为梯度流：

$$
\frac{d\mathbf{R}}{dt} = -\nabla E(\mathbf{R})
$$

采用后向 Euler 隐式积分：

$$
\mathbf{R}_{n+1} = \mathbf{R}_n + h \cdot (-\nabla E(\mathbf{R}_{n+1}))
$$

即求解非线性残差方程 $g(\mathbf{R}_{n+1}) = \mathbf{R}_{n+1} - \mathbf{R}_n + h\nabla E(\mathbf{R}_{n+1}) = 0$，通过 Newton-Raphson 迭代：

$$
\mathbf{R}^{(k+1)} = \mathbf{R}^{(k)} - J_g^{-1} g(\mathbf{R}^{(k)}), \quad J_g = I + h \nabla^2 E(\mathbf{R}^{(k)})
$$

### 3.5 证据回归与不确定性

采用正态逆 Gamma (NIG) 分布建模预测不确定性。给定输入 $\mathbf{x}$，网络输出四元组 $(\gamma, \nu, \alpha, \beta)$，预测分布为：

$$
p(y \mid \mathbf{x}) = \text{St}\left(y \mid \gamma, \frac{\beta(\nu+1)}{\alpha\nu}, 2\alpha\right)
$$

负对数似然：

$$
\mathcal{L}_{\text{NIG}} = \frac{1}{2}\log\frac{\pi}{\nu} - \alpha\log(2\beta) + \left(\alpha+\frac{1}{2}\right)\log\bigl(\nu(y-\gamma)^2 + 2\beta\bigr) + \ln\Gamma(\alpha) - \ln\Gamma\left(\alpha+\frac{1}{2}\right)
$$

偶然不确定性与认知不确定性分别为：

$$
\sigma_{\text{ale}}^2 = \frac{\beta}{\alpha-1}, \quad \sigma_{\text{epi}}^2 = \frac{\beta}{\nu(\alpha-1)}
$$

### 3.6 物理信息损失

注入以下守恒约束：

- **能量旋转/平移不变性**：$\mathcal{L}_E = |E(\mathbf{R}) - E(Q\mathbf{R}+\mathbf{t})|^2$
- **力-能量一致性**：$\mathcal{L}_F = \|\mathbf{F} + \nabla_{\mathbf{R}} E\|^2$
- **电荷守恒**：$\mathcal{L}_Q = |\sum_i q_i - Q_{\text{target}}|^2$

总损失：$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_E \mathcal{L}_E + \lambda_F \mathcal{L}_F + \lambda_Q \mathcal{L}_Q$

### 3.7 分子能量模型

数据集采用量子力学启发的解析能量模型：

$$
E = \sum_i E_{\text{atom}}^{(i)} + \sum_{\langle i,j \rangle} D_e\bigl[1-e^{-a(r-r_e)}\bigr]^2 + \sum_{\angle} \frac{1}{2}k_\theta(\theta-\theta_0)^2 + E_{\text{Coulomb}}
$$

其中原子能为类氢近似 $E_{\text{atom}}^{(i)} = -0.5 Z_i^2$，键能为 Morse 势，角能为谐波势。

## 4. 文件结构

```
150_synth_project/
├── main.py                       # 统一入口
├── molecular_graph.py            # 分子图构建、拉普拉斯、PageRank 重要性
├── chebyshev_conv.py             # Chebyshev 谱图卷积与 1D 插值
├── electrostatic_solver.py       # PIC 风格电荷沉积、Poisson 求解、电场插值
├── numerical_quadrature.py       # 三角形 Felippa 求积、Newton-Cotes Open 线积分
├── polynomial_basis.py           # 多元多项式基、分子描述符
├── dynamics_integrator.py        # 隐式后向 Euler 结构松弛
├── uncertainty_model.py          # NIG 证据回归、特殊函数、ERF 核
├── graph_utils.py                # 图划分、Fermat 指纹、Diophantine、距离统计、角度基
├── feature_engineering.py        # 阈值指纹、Coulomb 矩阵、RDF 直方图
├── physics_informed_loss.py      # 旋转/平移不变性、力一致性、电荷守恒
├── gnn_model.py                  # MPNN + Chebyshev 滤波 + 不确定性头
├── dataset.py                    # 合成分子数据集与能量模型
├── train_eval.py                 # Adam 优化器、训练循环、评估指标
└── README_博士级合成说明.md       # 本文档
```

## 5. 运行方式

```bash
cd Synthesis-project-python/150_synth_project
python main.py
```

程序将依次执行 11 个模块的演示，包括分子图分析、Chebyshev 卷积、静电势求解、数值积分、多项式描述符、结构松弛、不确定性量化、图分析、特征工程、物理损失以及 GNN 训练与评估。

## 6. 边界处理与数值鲁棒性

- **除零保护**：所有距离计算均设置下限（如 `max(r, 0.5)`）。
- **奇异矩阵**：Newton-Raphson 中若 Hessian 奇异，自动回退到最小二乘求解。
- ** arccos 定义域**：通过 `np.clip` 限制在 $[-1,1]$。
- ** softplus 稳定**：`log(1+exp(x))` 对极大/极小值安全。
- ** Laplacian 估计**：幂迭代中加入 margin 防止 $\lambda_{\max}$ 低估。
- ** 电荷守恒**：网络输出电荷通过损失项约束总和。

## 7. 合成项目解决的前沿科学问题

本项目针对**分子性质预测**这一计算化学核心挑战，提出了一套融合谱图理论、静电学、分子动力学与概率不确定性量化的物理信息神经网络框架。与传统纯数据驱动方法相比，本项目的创新在于：

1. **谱图卷积物理化**：通过 Chebyshev 多项式在图谱域实现可学习的介电响应滤波。
2. **静电势嵌入**：将求解 Poisson 方程得到的电势/电场作为图节点初始条件，引入长程相互作用。
3. **隐式结构优化**：在预测前通过 backward Euler 松弛将分子几何推向能量极小点，保证输入构型的物理合理性。
4. **认知不确定性量化**：利用 NIG 分布区分数据噪声（偶然不确定）与模型外推（认知不确定），对药物发现中的分子筛选具有指导意义。
5. **多尺度守恒约束**：显式施加旋转/平移不变性、力-能量一致性与电荷守恒，提升外推能力。
