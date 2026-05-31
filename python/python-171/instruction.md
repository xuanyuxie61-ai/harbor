# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目概述

本项目是一个面向稀疏线性系统迭代求解与预处理的科学计算工具集，围绕计算数学中的共轭梯度法（CG）及其变种展开，整合了多种稀疏矩阵存储格式、预处理技术、正交多项式与求积规则、随机化数值工具、特殊函数、自适应网格生成和测试问题构造等功能。所有代码均为 Python 语言，主要依赖 NumPy 和标准库 math，部分模块在精确积分验证中可能调用 SciPy，但核心算法不依赖 SciPy。

项目包含一个主入口文件 `main.py`（保留），以及若干功能模块（将被删除，需由 agent 根据本描述复现）。各模块之间通过函数调用和 NumPy 数组进行数据交换。

以下按文件逐一说明其作用和主要功能，以便复现。

---

## 文件清单及功能摘要

### `sparse_matrix.py`
**职责**：定义多种稀疏矩阵存储格式及其矩阵-向量乘法，并为 CG 类求解器提供统一调用接口。

主要内容：
- 若干独立函数，分别实现不同格式的矩阵-向量乘法（`r8ge_mv`、`r83_mv`、`r83s_mv`、`r83t_mv`、`r8pbu_mv`、`r8sd_mv`、`r8sp_mv`）。每种格式对应一种稀疏表示，例如 R83 是 (3,n) 的三对角存储，R8PBU 是针对对称正定带状矩阵的存储，COO 格式使用行、列、值三个数组。
- 工厂函数（如 `dif2_r8ge`、`dif2_r83`、`dif2_r83s` 等）用于生成一维离散 Laplacian（DIF2）矩阵在对应格式下的数据。
- 类 `SparseMatrixOperator`，接收格式标识符、矩阵数据、矩阵阶数和额外参数，对外提供统一的 `matvec` 方法（即矩阵乘向量），并可通过 `to_dense` 转换为稠密 NumPy 数组以便小型验证。

### `utils.py`
**职责**：通用数值工具，包括范数、残差计算、条件数估计、校验和机制以及辅助输出。

主要内容：
- 向量与矩阵范数计算函数（`vec_norm`、`mat_norm`），支持多种阶数，处理空数组。
- 多种格式下的残差计算函数（`residual_dense`、`residual_tridiag`、`residual_banded`、`residual_sd`、`residual_sparse`），用于计算给定近似解的残差向量 `b - A x`。
- 校验和函数 `checksum_vector` 基于加权和取模生成解向量的标量校验和，灵感来自 ISBN 校验，可用于验证迭代解的数值一致性；`verify_checksum` 用于比较期望值与计算值是否在一定容差内匹配。
- 基于幂法和反幂法的条件数估计函数 `condition_number_estimate`，返回估计条件数及最大、最小特征值近似。
- 对称正定性检查函数 `is_spd`，通过 Cholesky 分解尝试判断。
- 安全除法 `safe_divide` 以及打印格式化函数 `print_header`、`print_vec`。

### `conjugate_gradient.py`
**职责**：实现共轭梯度法及其变种（标准 CG、预处理 CG、灵活 CG、重启 CG），并提供面向多种稀疏格式的统一求解接口。

主要内容：
- 函数 `conjugate_gradient`：实现标准 CG 算法。接受一个矩阵-向量乘法函数（callable）、右端向量 `b`、初始猜测 `x0`、最大迭代次数、收敛容差等参数，返回数值解和含有迭代次数、残差历史、收敛标志等信息的字典。算法基于残差正交性和 A-范数优化，通过迭代更新搜索方向。
- 函数 `preconditioned_cg`：实现预处理共轭梯度法（PCG）。除矩阵-向量乘法外，还需接受一个预处理算子（callable，可对残差施加近似逆作用），核心迭代使用预处理后的残差。
- 函数 `flexible_cg`：灵活 CG，允许预处理算子在每次迭代中变化，适用于不固定或非线性的预处理子。
- 函数 `restarted_cg`：重启 CG，每运行若干步后以当前解为初值重新启动标准 CG，以减少舍入误差累积。
- 函数 `solve_with_format`：统一接口，根据稀疏格式标识符构造 `SparseMatrixOperator`，然后调用标准 CG 或 PCG 求解，方便对不同格式的矩阵进行对比。

### `preconditioner.py`
**职责**：实现多种预处理算子，每个预处理算子都是一个可调用对象，对残差向量施加近似逆矩阵的作用。

主要内容：
- `jacobi_preconditioner`：基于矩阵对角线构造 Jacobi 预处理（对角缩放），返回预处理函数。
- `ssor_preconditioner`：对称逐次超松弛（SSOR）预处理，基于矩阵的下三角和上三角部分，用前代和后代两步求解。
- `incomplete_cholesky_ic0`：零填充不完全 Cholesky 分解（IC(0)），仅保留输入矩阵的非零模式进行 Cholesky 分解，返回求解三角系统的预处理函数。
- `polynomial_spectral_preconditioner`：基于正交多项式零点的谱预处理。先估计矩阵的极端特征值，然后利用正交多项式（如 Laguerre 或 Hermite）的求积节点构造多项式逼近 A 的逆，结合对角缩放与几步 Richardon 式迭代实现近似逆作用。
- `two_grid_preconditioner`：两层网格预处理，利用粗网格矩阵和延拓/限制算子进行粗网格修正，辅以光滑步。
- `block_diagonal_preconditioner`：块对角预处理，对给定的块列表分别求逆并应用于相应段。

### `orthogonal_polynomials.py`
**职责**：提供标准/广义 Laguerre 多项式、Hermite 多项式的计算，以及基于 Jacobi 矩阵对角化的 Gauss 型求积规则生成。

主要内容：
- `laguerre_polynomial` 和 `generalized_laguerre_function`：在给定点集上计算标准 Laguerre 多项式或广义 Laguerre 函数的值，采用三项递推公式。
- `hermite_probabilist` 和 `hermite_physicist`：分别按概率学家或物理学家约定计算 Hermite 多项式值。
- `imtqlx`：隐式 QL 算法，用于对角化对称三对角矩阵（Jacobi 矩阵），输出升序特征值和变换后的向量。该算法是 Gauss 求积规则计算的核心。
- `gauss_laguerre_rule`、`gauss_generalized_laguerre_rule`、`gauss_hermite_rule`：分别生成 Gauss-Laguerre、广义 Gauss-Laguerre 和 Gauss-Hermite 求积节点与权重。前两者通过构造 Jacobi 矩阵并调用 `imtqlx` 对角化得到；后者直接使用 NumPy 提供的 `hermgauss` 函数。
- `build_polynomial_preconditioner_spectrum`：辅助函数，根据所选正交多项式族调用对应的求积规则函数，为预处理子提供节点和权重。

### `random_tools.py`
**职责**：提供随机化数值工具，包括 Halton 低差异序列、随机正交/SPD 矩阵生成、随机探测向量、Hutchinson 迹估计和随机 SVD。

主要内容：
- Halton 序列相关函数：`halton_value` 计算单个 Halton 点（基于进制逆函数）；`halton_sequence` 批量生成一段 Halton 序列；`halton_scrambled` 加入随机平移以降低高维相关性。
- `random_orthogonal_matrix`：利用 Stewart 算法生成随机正交矩阵，通过依次右乘 Householder 矩阵实现。
- `random_spd_matrix`：生成具有指定特征值范围（均匀分布）的随机对称正定矩阵，使用正交相似变换 `Q diag(λ) Q^T`，并强制对称。
- `random_spd_with_clustered_spectrum`：生成特征值呈聚类分布的 SPD
