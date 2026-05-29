"""
network_solver.py
大规模视网膜神经网络线性系统求解

基于以下种子项目合成：
- 974_r8cbb: 压缩边界带状矩阵线性代数

科学背景：
视网膜是一个大规模耦合神经网络，其中大部分神经元形成局部连接（带状结构），
少数神经元（如神经节细胞）与远处神经元形成长程连接（边界稠密块）。
这种结构可用压缩边界带状（Compressed Border Banded, CBB）矩阵表示。

矩阵结构：
    [ A1 | A2 ]
    [----+----]
    [ A3 | A4 ]

其中：
- A1: N1×N1 带状矩阵（局部连接）
- A2: N1×N2 稠密矩阵（局部→长程连接）
- A3: N2×N1 稠密矩阵（长程→局部连接）
- A4: N2×N2 稠密矩阵（长程神经元间连接）

求解使用块LU分解（Schur补方法）：
    S = A4 - A3 * A1^{-1} * A2
"""

import numpy as np
from typing import Tuple


# =============================================================================
# 带状矩阵LU分解与求解（基于974_r8cbb）
# =============================================================================

def band_lu_factorize(
    A_band: np.ndarray,
    n: int,
    ml: int,
    mu: int
) -> Tuple[np.ndarray, int]:
    """
    对带状矩阵进行无选主元的LU分解。
    
    带状矩阵A的存储格式：A_band[mu + i - j, j] = A[i,j]
    即第j列中，行i的元素存储在A_band[mu + i - j, j]。
    
    LU分解：A = L * U
    - L: 单位下三角，下带宽ml
    - U: 上三角，上带宽mu
    
    分解过程中，L和U共享带状存储。
    L的非对角元存储在下三角位置，U存储在上三角和对角线位置。
    
    参数:
        A_band: (ml+mu+1, n) 带状矩阵存储
        n: 矩阵维度
        ml: 下带宽
        mu: 上带宽
    
    返回:
        A_band: 覆盖存储的LU分解结果
        info: 0表示成功，非零表示奇异
    """
    info = 0
    nrow = ml + mu + 1
    
    for k in range(n):
        # 主元位置：对角线元素 A[k,k] 存储在 A_band[mu, k]
        pivot = A_band[mu, k]
        if abs(pivot) < 1e-14:
            info = k + 1
            pivot = 1e-14 if pivot >= 0 else -1e-14
            A_band[mu, k] = pivot
        
        # 对k列下方的元素（L部分）进行消去
        # i的范围：k+1 到 min(k+ml, n-1)
        for i in range(k + 1, min(k + ml + 1, n)):
            # A[i,k] 的存储位置
            row_ik = mu + (i - k)
            if row_ik >= nrow:
                continue
            
            factor = A_band[row_ik, k] / pivot
            A_band[row_ik, k] = factor  # 存储L[i,k]
            
            # 更新i行中k右侧的元素（U部分）
            # j的范围：k+1 到 min(k+mu, n-1)
            for j in range(k + 1, min(k + mu + 1, n)):
                # A[i,j] 的存储位置
                row_ij = mu + (i - j)
                # A[k,j] 的存储位置
                row_kj = mu + (k - j)
                
                if row_ij >= 0 and row_ij < nrow and row_kj >= 0 and row_kj < nrow:
                    A_band[row_ij, j] -= factor * A_band[row_kj, j]
    
    return A_band, info


def band_lu_solve(
    A_band_lu: np.ndarray,
    b: np.ndarray,
    n: int,
    ml: int,
    mu: int
) -> np.ndarray:
    """
    使用带状LU分解结果求解线性系统 A*x = b。
    
    分两步：
    1. 前向替换解 L*y = b
    2. 后向替换解 U*x = y
    
    参数:
        A_band_lu: LU分解后的带状矩阵
        b: (n,) 右端向量
        n: 矩阵维度
        ml: 下带宽
        mu: 上带宽
    
    返回:
        x: (n,) 解向量
    """
    x = b.copy().astype(np.float64)
    nrow = A_band_lu.shape[0]
    
    # 前向替换：L*y = b
    # L的对角线为1，下三角部分存储在 A_band[mu + i - j, j] for i>j
    for i in range(1, n):
        for j in range(max(0, i - ml), i):
            row_ij = mu + (i - j)
            if row_ij < nrow:
                x[i] -= A_band_lu[row_ij, j] * x[j]
    
    # 后向替换：U*x = y
    # U存储在上三角和对角线：A_band[mu + i - j, j] for i<=j
    for i in range(n - 1, -1, -1):
        pivot = A_band_lu[mu, i]
        if abs(pivot) < 1e-14:
            pivot = 1e-14
        
        for j in range(i + 1, min(i + mu + 1, n)):
            row_ij = mu + (i - j)
            if row_ij >= 0 and row_ij < nrow:
                x[i] -= A_band_lu[row_ij, j] * x[j]
        
        x[i] /= pivot
    
    return x


# =============================================================================
# 稠密矩阵LU分解与求解
# =============================================================================

def dense_lu_factorize(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    对稠密矩阵进行无选主元的LU分解。
    
    参数:
        A: (n, n) 稠密矩阵
    
    返回:
        L: (n, n) 单位下三角矩阵
        U: (n, n) 上三角矩阵
        info: 0表示成功
    """
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = A.copy()
    info = 0
    
    for k in range(n):
        if abs(U[k, k]) < 1e-14:
            info = k + 1
            U[k, k] = 1e-14 if U[k, k] >= 0 else -1e-14
        
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    
    return L, U, info


def dense_lu_solve(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    使用稠密LU分解求解线性系统。
    
    参数:
        L: 单位下三角矩阵
        U: 上三角矩阵
        b: 右端向量
    
    返回:
        x: 解向量
    """
    n = len(b)
    y = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    
    # 前向替换
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # 后向替换
    for i in range(n - 1, -1, -1):
        denom = U[i, i]
        if abs(denom) < 1e-14:
            denom = 1e-14
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / denom
    
    return x


# =============================================================================
# CBB矩阵求解（Schur补方法）
# =============================================================================

def solve_cbb_system(
    A1_band: np.ndarray,
    A2: np.ndarray,
    A3: np.ndarray,
    A4: np.ndarray,
    b: np.ndarray,
    n1: int,
    n2: int,
    ml: int,
    mu: int
) -> np.ndarray:
    """
    求解压缩边界带状（CBB）线性系统。
    
    系统结构：
        [ A1  A2 ] [ x1 ]   [ b1 ]
        [ A3  A4 ] [ x2 ] = [ b2 ]
    
    使用Schur补方法：
    1. 对A1进行LU分解
    2. 逐列求解 A1 * Y = A2，得到 Y = A1^{-1} * A2
    3. 计算 Schur补 S = A4 - A3 * Y
    4. 对S进行LU分解
    5. 解 S * x2 = b2 - A3 * A1^{-1} * b1
    6. 解 A1 * x1 = b1 - A2 * x2
    
    参数:
        A1_band: (ml+mu+1, n1) 带状矩阵存储
        A2: (n1, n2) 稠密矩阵
        A3: (n2, n1) 稠密矩阵
        A4: (n2, n2) 稠密矩阵
        b: (n1+n2,) 右端向量
        n1, n2: 块大小
        ml, mu: A1的带宽
    
    返回:
        x: (n1+n2,) 解向量
    """
    b1 = b[:n1].copy()
    b2 = b[n1:].copy()
    
    # 1. A1的LU分解
    A1_lu, info1 = band_lu_factorize(A1_band.copy(), n1, ml, mu)
    if info1 != 0:
        print(f"  Warning: A1 band LU factorization info={info1}")
    
    # 2. 计算 Y = A1^{-1} * A2 （逐列求解）
    Y = np.zeros((n1, n2), dtype=np.float64)
    for j in range(n2):
        Y[:, j] = band_lu_solve(A1_lu, A2[:, j].copy(), n1, ml, mu)
    
    # 3. 计算 Schur补 S = A4 - A3 * Y
    S = A4 - A3 @ Y
    
    # 4. S的LU分解
    L_s, U_s, info_s = dense_lu_factorize(S)
    if info_s != 0:
        print(f"  Warning: Schur complement LU factorization info={info_s}")
    
    # 5. 计算 A1^{-1} * b1
    A1_inv_b1 = band_lu_solve(A1_lu, b1.copy(), n1, ml, mu)
    
    # 6. 解 S * x2 = b2 - A3 * A1^{-1} * b1
    rhs2 = b2 - A3 @ A1_inv_b1
    x2 = dense_lu_solve(L_s, U_s, rhs2)
    
    # 7. 解 A1 * x1 = b1 - A2 * x2
    rhs1 = b1 - A2 @ x2
    x1 = band_lu_solve(A1_lu, rhs1, n1, ml, mu)
    
    return np.concatenate([x1, x2])


# =============================================================================
# 带状矩阵与稠密矩阵的转换
# =============================================================================

def band_to_dense(A_band: np.ndarray, n: int, ml: int, mu: int) -> np.ndarray:
    """
    将带状矩阵存储转换为稠密矩阵。
    
    存储格式：A_band[mu + i - j, j] = A[i,j]
    
    参数:
        A_band: (ml+mu+1, n) 带状存储
        n: 矩阵维度
        ml: 下带宽
        mu: 上带宽
    
    返回:
        A_dense: (n, n) 稠密矩阵
    """
    A_dense = np.zeros((n, n), dtype=np.float64)
    nrow = A_band.shape[0]
    
    for j in range(n):
        for i in range(max(0, j - mu), min(n, j + ml + 1)):
            row = mu + (i - j)
            if 0 <= row < nrow:
                A_dense[i, j] = A_band[row, j]
    
    return A_dense


# =============================================================================
# 视网膜网络连接矩阵构建
# =============================================================================

def build_retinal_network_matrix(
    n_local: int,
    n_long_range: int,
    connectivity_radius: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建视网膜神经网络的CBB格式连接矩阵。
    
    局部神经元（如双极细胞、无长突细胞）形成短程连接（带状结构），
    长程神经元（如某些神经节细胞类型）形成全连接（稠密块）。
    
    连接权重使用高斯衰减：
        w_{ij} = exp(-d_{ij}^2 / (2*σ^2))
    
    参数:
        n_local: 局部神经元数量
        n_long_range: 长程神经元数量
        connectivity_radius: 连接半径（以神经元索引距离为单位）
    
    返回:
        A1_band, A2, A3, A4: CBB矩阵的四个块
    """
    ml = int(connectivity_radius)
    mu = int(connectivity_radius)
    nrow = ml + mu + 1
    A1_band = np.zeros((nrow, n_local), dtype=np.float64)
    
    for j in range(n_local):
        for i in range(max(0, j - ml), min(n_local, j + mu + 1)):
            dist = abs(i - j)
            if dist == 0:
                val = 2.0  # 自连接（膜电容项）
            else:
                val = -0.3 * np.exp(-dist ** 2 / (2.0 * connectivity_radius ** 2))
            
            row = mu + (i - j)
            if 0 <= row < nrow:
                A1_band[row, j] = val
    
    # A2: 局部→长程连接（稠密）
    np.random.seed(123)
    A2 = np.random.random((n_local, n_long_range)) * 0.1
    
    # A3: 长程→局部连接（稠密）
    A3 = np.random.random((n_long_range, n_local)) * 0.05
    
    # A4: 长程神经元间连接（稠密对角占优）
    A4 = np.eye(n_long_range, dtype=np.float64) * 2.0
    for i in range(n_long_range):
        for j in range(n_long_range):
            if i != j:
                A4[i, j] = -0.05 * np.random.random()
    
    return A1_band, A2, A3, A4
