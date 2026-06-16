# PROJECT 209: 随机参数 PDE 模型的不确定性量化
# Uncertainty Quantification for Stochastic Parameter PDEs

## 一、合成项目概述

本项目是一个**博士级**科学计算项目，聚焦**不确定性量化 (Uncertainty Quantification, UQ)** 这一前沿科学计算领域。项目实现了**随机参数椭圆偏微分方程 (Stochastic Elliptic PDE)** 的完整不确定性量化流水线：

$$-\nabla \cdot (a(x,\omega) \nabla u(x,\omega)) = f(x) \quad \text{in } \Omega = [0,1]^2$$
$$u = 0 \quad \text{on } \partial\Omega$$

其中扩散系数 $a(x,\omega) = \exp(Y(x,\omega))$ 为**对数正态随机场**，$Y(x,\omega)$ 为高斯随机场，通过 **Karhunen-Loève 展开**进行参数化。

### 核心科学问题
- **高维随机参数空间的高效采样**: 稀疏网格 + 拟随机序列
- **随机场的低维参数化**: KL 展开 + Hankel-Cholesky 快速分解
- **PDE 解的统计矩提取**: 多项式混沌展开 + Sobol 全局灵敏度
- **多保真度信息融合**: 状态机驱动的自适应加密
- **贝叶斯反问题**: 后验推断与分布验证

---

## 二、科学问题的数学模型

### 2.1 随机场的 Karhunen-Loève 展开

对高斯随机场 $Y(x,\omega)$ 进行 KL 展开：

$$Y(x,\omega) = \sum_{k=1}^{K} \sqrt{\lambda_k} \, \varphi_k(x) \, \xi_k(\omega)$$

其中 $\{(\lambda_k, \varphi_k)\}$ 是协方差算子的特征对：

$$\int_D C(x,y) \varphi_k(y) \, dy = \lambda_k \varphi_k(x)$$

协方差函数采用平方指数核 (Squared Exponential)：

$$C(x,y) = \sigma^2 \exp\left(-\frac{\|x-y\|_2^2}{2\ell^2}\right)$$

或 Matérn 核 ($\nu = 3/2$)：

$$C(x,y) = \sigma^2 \left(1 + \frac{\sqrt{3}\|x-y\|}{\ell}\right) \exp\left(-\frac{\sqrt{3}\|x-y\|}{\ell}\right)$$

### 2.2 Smolyak 稀疏网格配置

对于 $d$ 维随机参数空间 $\Gamma \subset \mathbb{R}^d$，Smolyak 算子定义为：

$$\mathcal{A}(q,d) = \sum_{\mathbf{i} \in I(q,d)} (-1)^{q-|\mathbf{i}|} \binom{d-1}{q-|\mathbf{i}|} \left(Q^{i_1} \otimes \cdots \otimes Q^{i_d}\right)$$

其中多指标集：

$$I(q,d) = \{\mathbf{i} \in \mathbb{N}^d : q-d+1 \le |\mathbf{i}| \le q, \; i_k \ge 1\}$$

### 2.3 广义多项式混沌 (gPC) 展开

PDE 解的 gPC 展开：

$$u(x,\xi) \approx \sum_{\alpha \in \mathcal{J}_{p,d}} c_\alpha(x) \Psi_\alpha(\xi)$$

其中 $\Psi_\alpha(\xi) = \prod_{k=1}^d \phi_{\alpha_k}(\xi_k)$ 为正交多项式基，$\phi_n$ 为 Hermite/Legendre 多项式。

**Hermite 多项式递推**：
$$He_0(x) = 1, \quad He_1(x) = x, \quad He_{n+1}(x) = x \cdot He_n(x) - n \cdot He_{n-1}(x)$$

### 2.4 弱形式有限元离散

PDE 弱形式：求 $u \in H_0^1(\Omega)$ 使得

$$\int_\Omega a(x,\omega) \nabla u \cdot \nabla v \, dx = \int_\Omega f v \, dx \quad \forall v \in H_0^1(\Omega)$$

P1 有限元离散后得线性系统：

$$K(\xi) \mathbf{U}(\xi) = \mathbf{F}$$

其中刚度矩阵 $K_{ij}(\xi) = \int_\Omega a(x,\xi) \nabla \phi_i \cdot \nabla \phi_j \, dx$。

### 2.5 贝叶斯反问题

给定观测数据 $\mathbf{d}_{\text{obs}}$，后验分布：

$$p(\xi | \mathbf{d}_{\text{obs}}) = \frac{p(\mathbf{d}_{\text{obs}} | \xi) p(\xi)}{p(\mathbf{d}_{\text{obs}})}$$

MAP 估计：

$$\xi_{\text{MAP}} = \arg\max_\xi \left\{ \log p(\mathbf{d}_{\text{obs}}|\xi) + \log p(\xi) \right\}$$

---

## 三、种子项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 本项目映射 |
|------|---------|---------|-----------|
| 1 | 180_circle_map | 2×2 矩阵的 SVD / 椭圆映射 | PCE 系数空间的 SVD 分析 (长短轴比表征各向异性) |
| 2 | 558_hypercube_grid | 张量积超立方体网格 | 张量积配置点网格 + 直接乘积构造算法 |
| 3 | 1055_fperdigon_DeepHistoPathology | Inception 多尺度深度学习 | PDE 代理模型 (Inception-Depthwise 多尺度网络) |
| 4 | 306_distance_to_position | 多维标度 (MDS) | 解流形的低维嵌入 (距离矩阵 → 参数空间坐标) |
| 5 | 1380_van_der_corput | Van der Corput 拟随机序列 | QMC 采样器 (低差异序列用于参数空间采样) |
| 6 | 1371_ulam_spiral | Ulam 螺旋索引 | 稀疏网格多指标集的螺旋自适应枚举 |
| 7 | 865_percolation_simulation | BFS 洪水填充 + 渗流判断 | 随机介质连通性分析 (判断高导区域是否形成渗流通路) |
| 8 | 582_image_normalize | 图像标准化 | 随机场实现的标准化 (zscore/minmax/robust) |
| 9 | 801_newton_maehly | Newton-Maehly 同时求根 | 分岔检测：多项式特征方程的根 → 分岔类型识别 |
| 10 | 847_pariomino | Diophantine 有界解 + 奇偶荷 | gPC 多指标集的 Diophantine 枚举 + 对称性检测 |
| 11 | 1276_IzzetYoung_ConsistentMI | 状态机 + Dijkstra + 主题图 | 多保真度模型图 + 自适应加密状态机 + 最优路径搜索 |
| 12 | 504_hankel_cholesky | Hankel 矩阵 Cholesky 分解 | 协方差矩阵的快速 Hankel 结构分解 |
| 13 | 1111_sbi-benchmark_results | SBI 基准框架 + C2ST/MMD/KSD | 后验推断验证指标 (C2ST, MMD, KSD, 中位距离) |
| 14 | 1164_jenzenho_flame-ai | SSIM/Jaccard/centroid 误差 | 随机场解的误差度量 (MSE, SSIM, Jaccard, 质心误差) |
| 15 | 190_closest_pair_brute | 暴力最近点对搜索 | 网格质量评估 + 解流形自适应采样 (max-min 策略) |

---

## 四、项目文件结构

```
209_synth_project_Advanced/
├── main.py                       # 统一入口 (零参数运行)
├── random_field.py               # 随机场建模: KL 展开 + Hankel-Cholesky
├── sparse_grid_collocation.py    # 稀疏网格: Smolyak + Van der Corput + Ulam 螺旋
├── polynomial_chaos.py           # gPC 展开: Diophantine 多指标 + 奇偶荷 + SVD
├── fem_solver.py                 # 有限元求解: P1 FEM + 渗流拓扑 + 最近点对
├── deep_surrogate.py             # 深度学习代理: Inception 多尺度网络
├── solution_manifold.py          # 解流形: MDS + POD + 自适应采样
├── multi_fidelity.py             # 多保真度: 状态机 + Dijkstra + MFMC
├── bayesian_inverse.py           # 贝叶斯反问题: 拒绝采样 + MAP + C2ST/MMD
├── bifurcation.py                # 分岔检测: Newton-Maehly + 临界参数
├── error_analysis.py             # 误差分析: SSIM/Jaccard/KS/收敛率
└── README_博士级合成说明.md      # 本文档
```

共 **11 个 Python 源文件** + **1 个中文文档**。

---

## 五、运行说明

### 环境要求
- Python 3.8+
- NumPy >= 1.20
- SciPy >= 1.7

### 运行命令
```bash
cd 209_synth_project_Advanced
python main.py
```

**零参数运行**：无需任何命令行参数，程序自动执行完整的 10 步流水线：

1. **Step 1**: 随机场建模 (KL 展开, Hankel-Cholesky 分解)
2. **Step 2**: 稀疏网格构造 (Smolyak, Van der Corput, Halton, Ulam 螺旋)
3. **Step 3**: 多项式混沌展开 (多指标枚举, Sobol 灵敏度, SVD)
4. **Step 4**: 有限元求解 (P1 FEM, 渗流分析)
5. **Step 5**: 深度学习代理 (Inception 网络训练)
6. **Step 6**: 解流形分析 (MDS, POD, 自适应采样)
7. **Step 7**: 多保真度融合 (状态机, Dijkstra, MFMC)
8. **Step 8**: 贝叶斯反问题 (拒绝采样, MAP, 后验验证)
9. **Step 9**: 分岔检测 (Newton-Maehly, 临界参数搜索)
10. **Step 10**: 综合误差分析 (MSE, SSIM, Jaccard, KS)

---

## 六、核心科学计算能力

### 6.1 随机场生成与参数化
- 支持 3 种协方差核: Exponential, Squared-Exponential, Matérn (ν=1/2, 3/2, 5/2, 通用)
- KL 展开能量截断: 自动选择模态数使捕获能量 ≥ 99%
- 对数正态变换保证扩散系数正定性

### 6.2 高维随机配置
- 张量积网格: 5 种 centering 方案
- Smolyak 稀疏网格: 指数级减少配置点数
- Van der Corput 序列: 差异 $D_N = O(\log N / N)$
- Halton 序列: 多维拟随机采样

### 6.3 多项式混沌展开
- Diophantine 有界解枚举多指标集
- 全阶截断 + 双曲交叉截断
- 奇偶荷对称性检测
- 直接提取统计矩和 Sobol 主效应指数

### 6.4 有限元求解
- 结构化三角网格生成
- P1 有限元组装
- 共轭梯度法求解
- 渗流理论分析随机介质连通性

### 6.5 多保真度信息融合
- 多保真度模型图 (Dijkstra 最短路径)
- 自适应加密状态机 (状态转移决策)
- MFMC 估计器 (最优样本分配)
- 信息渗流连通性判断

### 6.6 贝叶斯推断
- 高斯/均匀先验
- 拒绝采样 + 重要性采样
- MAP 估计 (多起点梯度上升)
- 后验验证指标: C2ST, MMD², KSD, KS 距离

### 6.7 分岔分析
- Newton-Maehly 同时求根
- 沿参数路径的分岔检测
- 临界参数搜索

---

## 七、边界条件处理与数值鲁棒性

1. **协方差矩阵正定性保证**: 自动添加正则化 $\epsilon I$
2. **Cholesky 分解失败回退**: 特征值位移法
3. **PCE 最小二乘正则化**: 基于条件数自适应 Tikhonov 正则
4. **FEM 大数法施加 Dirichlet BC**: $\alpha = 10^{30}$
5. **共轭梯度法残差监控**: `tol=1e-8, max_iter=5000`
6. **渗流 BFS 边界检测**: 基于容差判断节点是否在边界
7. **MDS 负特征值处理**: $\max(\lambda, 0)$ 截断
8. **神经网络梯度裁剪**: ReLU/tanh/sigmoid/GELU 多种激活
9. **贝叶斯拒绝采样**: 对数空间计算避免下溢
10. **Newton-Maehly 除零保护**: $|denom| > 10^{-15}$ 检查

---

## 八、计算复杂度

| 步骤 | 复杂度 | 主要瓶颈 |
|------|--------|---------|
| 随机场 KL | $O(N_x^2 \cdot K)$ | 协方差矩阵特征分解 |
| 稀疏网格 | $O(N_{\text{sg}} \cdot d)$ | Smolyak 组合数 |
| PCE 配点 | $O(N_s \cdot P^2)$ | 最小二乘求解 |
| FEM | $O(N_h^2)$ | 刚度矩阵组装 |
| 代理训练 | $O(E \cdot N \cdot W^2)$ | 神经网络训练 |
| MDS/POD | $O(N_s^3)$ | 特征分解 |
| 多保真度 MC | $O(\sum N_\ell \cdot c_\ell)$ | 多级别模拟 |
| 贝叶斯推断 | $O(N_{\text{prop}} \cdot d)$ | 前向模拟 |

---

## 九、科学意义

本项目展示了**不确定性量化**这一前沿领域的核心计算方法，为以下科学问题提供工具支持：

1. **地下水 contaminant transport**: 随机渗透系数场的 UQ
2. **复合材料力学**: 随机材料参数的结构响应分析
3. **气候建模**: 参数化方案的不确定性传播
4. **金融工程**: 随机波动率模型的衍生品定价
5. **生物医学**: 组织属性的随机变异性分析

---

## 十、合成技术路线总结

```
15 个种子项目
      ↓ 核心算法抽取
      ↓
科学问题重构: 随机参数 PDE 的 UQ
      ↓
模块划分 (11 个 .py):
  ├─ 随机场建模 (KL + Hankel-Cholesky)
  ├─ 稀疏网格配置 (Smolyak + VDC + Ulam)
  ├─ PCE 展开 (Diophantine + 奇偶荷 + SVD)
  ├─ FEM 求解 (P1 + 渗流 + 最近点对)
  ├─ 代理模型 (Inception 多尺度网络)
  ├─ 解流形分析 (MDS + POD)
  ├─ 多保真度融合 (状态机 + Dijkstra)
  ├─ 贝叶斯推断 (SBI + C2ST/MMD)
  ├─ 分岔检测 (Newton-Maehly)
  └─ 误差分析 (SSIM/Jaccard/KS)
      ↓
统一入口 main.py (零参数)
      ↓
完整流水线运行通过 ✓
```

---

**作者**: DA 博士级科学合成工作流
**日期**: 2026-06-07
**领域**: 不确定性量化 / 随机偏微分方程 / 科学计算
