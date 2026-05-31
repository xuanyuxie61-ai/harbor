# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：多因子随机波动率模型下奇异期权定价与全局校准

本项目围绕金融工程中的衍生品定价与随机波动率问题，基于多个数值算法模块合成一个博士级研究系统。核心任务包括：Heston 随机波动率 PDE 的有限差分离散与求解、稀疏线性系统的高效求解、高维参数空间的蒙特卡洛方差缩减与稀疏网格积分、波动率曲面的主成分分析、非线性动力学与混沌分析、参数优化与延拓校准，以及一系列特殊数学工具。

## 文件与模块职责

### 1. `main.py`
项目统一入口。调用所有其他模块的函数，按顺序执行以下数值实验：
- 实验1：稀疏矩阵 CCS 格式与 GMRES 迭代求解器验证
- 实验2：拉丁超立方采样与蒙特卡洛方差缩减
- 实验3：波动率曲面主成分分析
- 实验4：稀疏网格随机配置法积分
- 实验5：Heston PDE 有限差分定价与 Greeks 计算
- 实验6：非线性动力学、混沌分析与 Riccati 方程求解
- 实验7：黄金分割搜索与延拓法参数校准
- 实验8：风险度量统计与网格数据管理

`main.py` 本身不包含算法实现，仅作为调用入口和组织者。后续复现过程中，`main.py` 将被保留，其他 `.py` 文件被删除，需要根据本描述重新实现这些模块。

### 2. `sparse_matrix_ccs.py`
提供压缩列存储（CCS）格式的稀疏矩阵类 `SparseMatrixCCS`。  
- 存储三个数组：列指针 `colptr`，行索引 `rowind`，非零元数值 `a`。  
- 支持矩阵-向量乘法、转置乘法、元素访问、转换为稠密矩阵。  
- 提供静态工厂方法 `dif2` 构造二阶差分算子矩阵，以及 `from_dense` 从稠密矩阵转换。  
- 内部包含完整性校验。

### 3. `gmres_iterative.py`
实现重启 GMRES 迭代法求解大型稀疏线性系统。  
- 核心函数 `restarted_gmres` 接受 COO 三元组格式的稀疏矩阵，进行多轮重启 Arnoldi 过程，构造 Krylov 子空间并通过 Givens 旋转更新残差。  
- 辅助函数 `sparse_mv` 实现 COO 格式的稀疏矩阵-向量乘。  
- 辅助函数 `mult_givens` 对向量施加 Givens 旋转。  
- 提供稠密矩阵包装器 `gmres_dense`，将稠密矩阵转换为 COO 三元组后调用 `restarted_gmres`，便于中小规模调试。

### 4. `heston_pde_engine.py`
Heston 随机波动率 PDE 的有限差分求解器。  
- 类 `HestonPDESolver` 负责生成非均匀空间网格（S 方向对数拉伸，v 方向指数拉伸），构造二维 PDE 离散算子（包含扩散、交叉导数、对流、反应项），并处理四种边界条件（S=0 的 Dirichlet，S=S_max 的时变 Dirichlet，v=0 的退化边界，v=v_max 的 Neumann）。  
- 使用隐式欧拉时间推进，每步求解线性系统，可选用直接求解器或 GMRES。  
- 提供欧式看涨期权定价方法 `solve_european_call`，返回价格曲面，以及双线性插值方法 `price_at_spot`。  
- 模块级函数 `heston_european_call_price` 和 `heston_pde_greeks` 提供便捷接口和有限差分 Greeks 计算。

### 5. `latin_hypercube_sampler.py`
拉丁超立方采样（LHS）模块，用于蒙特卡洛方差缩减。  
- 类 `LatinHypercubeSampler` 生成 [0,1]^d 上的均匀拉丁超立方样本，并通过逆正态变换和 Cholesky 分解生成具有指定协方差矩阵的多元正态样本。  
- 提供 `sample_for_heston` 方法专门生成二维相关标准正态向量，用于 Heston 模型中两个布朗运动的模拟。  
- 内部包含逆正态累计分布函数的有理近似实现（基于 Acklam 算法）。

### 6. `principal_component_analysis.py`
主成分分析（PCA）模块，用于波动率曲面降维和多因子建模。  
- 类 `PrincipalComponentAnalysis` 支持通过 Turk-Pentland 技巧高效计算高维低样本情况下的特征分解，拟合后保留主成分、解释方差比等。  
- 提供 `transform` 和 `inverse_transform` 方法进行投影与重建。  
- 模块级函数 `volatility_surface_pca` 对隐含波动率矩阵进行 PCA，返回主成分载荷、得分和重建曲面。  
- 函数 `correlated_volatility_factors` 从协方差矩阵提取正交因子载荷。

### 7. `sparse_grid_stochastic.py`
稀疏网格随机配置积分模块，用于高维参数空间的期望计算。  
- 实现 Smolyak 稀疏网格的多指数生成、组合系数计算（朴素算法），以及基于 Clenshaw-Curtis 节点的一维积分规则。  
- 类 `SparseGridIntegrator` 根据维度与层级预计算节点和权重，并提供 `integrate` 方法在任意超矩形域上积分任意
