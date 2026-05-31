"""
稀疏 Hessian 矩阵工具箱
基于 hb_to_msm 和 r8ss 核心算法：
- Harwell-Boeing (HB) 稀疏矩阵格式解析
- 对称天际线 (Symmetric Skyline, R8SS) 存储格式
- 矩阵-向量乘法

在蛋白质折叠中的应用：
- 弹性网络模型 (Elastic Network Model, ENM) 的 Kirchhoff/Laplacian 矩阵
- 蛋白质力场在平衡构象处的 Hessian 矩阵存储与运算
- 正常模式分析 (Normal Mode Analysis, NMA)
- 大规模 MD 协方差矩阵的稀疏处理

数学基础:
    天际线存储: 对实对称矩阵，每列 j 只存储从第一个非零元到对角线的元素
    diag[j] = 列 j 对角元在压缩数组中的索引
    总非零元数: na = diag[n-1] + 1
    
    矩阵-向量乘法:
        对每列 j，遍历其非零条带中的元素 A[k]
        行号 i = j - (diag[j] - k)
        b[i] += A[k] * x[j]
        b[j] += A[k] * x[i]   (利用对称性)
"""

import numpy as np
from typing import Tuple, List


def hb_to_msm(hb_data_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 Harwell-Boeing 格式文本数据解析为稀疏矩阵和右端项。
    
    HB 格式结构:
        1. Header 行: 标题、维度、非零元数、格式说明
        2. COLPTR: 列指针数组 (CSC 格式)
        3. ROWIND: 行索引数组
        4. VALUES: 非零数值
        5. RHS: 右端项 (可选)
    
    Parameters
    ----------
    hb_data_lines : list of str
        HB 文件的文本行列表。
    
    Returns
    -------
    A : np.ndarray
        稠密矩阵（或稀疏矩阵的稠密表示，用于小规模验证）。
    rhs : np.ndarray
        右端项（若无则为空数组）。
    """
    if len(hb_data_lines) < 3:
        raise ValueError("Invalid HB data: too few lines")
    
    # 解析 Header (简化版，假设格式固定)
    header = hb_data_lines[0].strip()
    line2 = hb_data_lines[1].strip().split()
    line3 = hb_data_lines[2].strip().split()
    
    # 提取维度信息
    nrow = int(line2[0])
    ncol = int(line2[1])
    nnzero = int(line2[2])
    
    # 简化实现：假设数据在后续行中以空格分隔
    # 实际 HB 格式更复杂，这里用简化的逻辑提取数值
    all_numbers = []
    for line in hb_data_lines[3:]:
        parts = line.strip().split()
        for p in parts:
            try:
                all_numbers.append(float(p))
            except ValueError:
                pass
    
    # 构造 CSC 格式并转换为稠密
    A = np.zeros((nrow, ncol))
    ptr_start = 0
    # 简化的解析：若数值足够，则按顺序填充
    if len(all_numbers) >= nnzero:
        # 假设前 nnzero 个为值，然后为 colptr 和 rowind
        # 这里做一个非常简化的演示性解析
        vals = np.array(all_numbers[:nnzero])
        # 均匀分布到矩阵中作为演示
        count = 0
        for j in range(ncol):
            for i in range(nrow):
                if count < nnzero:
                    A[i, j] = vals[count]
                    count += 1
    
    rhs = np.array(all_numbers[nnzero:nnzero + nrow]) if len(all_numbers) > nnzero else np.array([])
    return A, rhs


def r8ss_mv(n: int, na: int, diag: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    对称天际线 (R8SS) 格式矩阵-向量乘法: b = A * x。
    
    存储说明:
        diag[j]: 第 j 列对角元在压缩数组 a 中的索引
        a[diag[j] - (j - i)]: 矩阵元素 A[i, j]，其中 i <= j
    
    Parameters
    ----------
    n : int
        矩阵维度。
    na : int
        压缩数组 a 的长度。
    diag : np.ndarray, shape (n,)
        对角元索引数组。
    a : np.ndarray, shape (na,)
        压缩的非零元素数组。
    x : np.ndarray, shape (n,)
        输入向量。
    
    Returns
    -------
    b : np.ndarray, shape (n,)
        结果向量。
    """
    if len(diag) != n:
        raise ValueError("diag length must equal n")
    if len(a) != na:
        raise ValueError("a length must equal na")
    if len(x) != n:
        raise ValueError("x length must equal n")
    
    b = np.zeros(n)
    for j in range(n):
        # 列 j 的非零元范围: 从行 j - bandwidth 到行 j
        bandwidth = diag[j] - (diag[j - 1] + 1) if j > 0 else diag[j]
        start_row = j - bandwidth
        
        for k in range(start_row, j + 1):
            idx = diag[j] - (j - k)
            if 0 <= idx < na:
                val = a[idx]
                b[k] += val * x[j]
                if k != j:
                    b[j] += val * x[k]
    return b


def r8ss_from_dense(dense: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    从稠密对称矩阵构造 R8SS 天际线存储。
    
    Parameters
    ----------
    dense : np.ndarray, shape (n, n)
        对称矩阵。
    
    Returns
    -------
    na : int
        压缩数组长度。
    diag : np.ndarray
        对角元索引。
    a : np.ndarray
        压缩数组。
    """
    n = dense.shape[0]
    diag = np.zeros(n, dtype=int)
    a_list = []
    
    for j in range(n):
        # 找到该列第一个非零元
        first_nonzero = j
        for i in range(j, -1, -1):
            if abs(dense[i, j]) > 1e-14:
                first_nonzero = i
        
        col_vals = dense[first_nonzero:j + 1, j]
        diag[j] = len(a_list) + len(col_vals) - 1
        a_list.extend(col_vals.tolist())
    
    a = np.array(a_list)
    na = len(a)
    return na, diag, a


def r8ss_to_r8ge(n: int, na: int, diag: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    将 R8SS 格式转换回稠密矩阵。
    
    Parameters
    ----------
    n : int
        维度。
    na : int
        压缩长度。
    diag : np.ndarray
        对角元索引。
    a : np.ndarray
        压缩数组。
    
    Returns
    -------
    dense : np.ndarray, shape (n, n)
        稠密矩阵。
    """
    dense = np.zeros((n, n))
    for j in range(n):
        bandwidth = diag[j] - (diag[j - 1] + 1) if j > 0 else diag[j]
        start_row = j - bandwidth
        for k in range(start_row, j + 1):
            idx = diag[j] - (j - k)
            if 0 <= idx < na:
                dense[k, j] = a[idx]
                dense[j, k] = a[idx]
    return dense


def build_elastic_network_matrix(coords: np.ndarray, cutoff: float = 1.5,
                                  spring_constant: float = 1.0) -> np.ndarray:
    """
    构建粗粒化弹性网络模型 (Elastic Network Model, ENM) 的 Kirchhoff 矩阵。
    
    模型定义 (Gaussian Network Model, GNM):
        若残基 i 和 j 的距离 < cutoff，则在它们之间放置一个弹簧 (力常数 k)。
        Kirchhoff 矩阵 Γ:
            Γ_{ii} = Σ_j k_{ij}
            Γ_{ij} = -k_{ij}  (i ≠ j)
    
    Parameters
    ----------
    coords : np.ndarray, shape (N, d)
        残基坐标。
    cutoff : float
        接触截断距离。
    spring_constant : float
        弹簧力常数。
    
    Returns
    -------
    gamma : np.ndarray, shape (N, N)
        Kirchhoff 矩阵。
    """
    N = coords.shape[0]
    gamma = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                gamma[i, j] = -spring_constant
                gamma[j, i] = -spring_constant
                gamma[i, i] += spring_constant
                gamma[j, j] += spring_constant
    
    return gamma


def normal_mode_analysis(gamma: np.ndarray, n_modes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    对弹性网络矩阵进行正常模式分析 (Normal Mode Analysis, NMA)。
    
    求解广义特征值问题:
        Γ * u = λ * u
    
    特征值 λ_i 对应模式的频率平方 ω_i²。
    最小的非零特征值对应最慢的集体运动模式 (低频模式)。
    
    Parameters
    ----------
    gamma : np.ndarray, shape (N, N)
        Kirchhoff 矩阵。
    n_modes : int
        计算的模式数。
    
    Returns
    -------
    eigenvalues : np.ndarray
        特征值（升序）。
    eigenvectors : np.ndarray
        特征向量（每列为一个模式）。
    """
    # 确保对称
    gamma = 0.5 * (gamma + gamma.T)
    eigvals, eigvecs = np.linalg.eigh(gamma)
    
    # 跳过零模式（平移/旋转）
    nonzero_mask = np.abs(eigvals) > 1e-8
    eigvals = eigvals[nonzero_mask]
    eigvecs = eigvecs[:, nonzero_mask]
    
    if len(eigvals) < n_modes:
        n_modes = len(eigvals)
    
    return eigvals[:n_modes], eigvecs[:, :n_modes]


def compute_mean_square_fluctuation(gamma: np.ndarray, kT: float = 1.0) -> np.ndarray:
    """
    计算每个残基的均方涨落 (Mean Square Fluctuation, MSF)。
    
    在 GNM 中:
        <Δr_i²> = (k_B T / γ) * [Γ^{-1}]_{ii}
    
    为避免求逆的数值问题，使用伪逆。
    
    Parameters
    ----------
    gamma : np.ndarray, shape (N, N)
        Kirchhoff 矩阵。
    kT : float
        热能量。
    
    Returns
    -------
    msf : np.ndarray, shape (N,)
        各残基的均方涨落。
    """
    # 使用伪逆处理零模式
    gamma_inv = np.linalg.pinv(gamma, rcond=1e-10)
    msf = kT * np.diag(gamma_inv)
    return msf
