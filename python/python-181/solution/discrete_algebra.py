"""
discrete_algebra.py
离散代数结构与布尔特征编码
融合原项目: 672_lights_out, 485_gray_code_display

核心科学思想:
利用模2线性代数(Boolean algebra over F_2)和Gray码的离散结构，
对高维数据的拓扑特征进行离散编码与压缩。

数学模型:
F_2 上的向量空间:
    V = F_2^n, 运算为模2加法和模2乘法

Gray码的邻接图:
    n维超立方体图 Q_n，顶点为Gray码，边连接Hamming距离为1的顶点
    Q_n 的邻接矩阵特征值: λ_k = n - 2k, k=0,...,n

特征编码:
    将连续特征x通过阈值化映射到F_2:
        encode(x) = (x > median(x)) mod 2
    然后利用Lights Out矩阵进行扩散编码
"""

import numpy as np
from typing import Tuple


def threshold_encode(data: np.ndarray, thresholds: np.ndarray = None) -> np.ndarray:
    """
    将连续数据阈值化为二元特征
    """
    if thresholds is None:
        thresholds = np.median(data, axis=0)
    binary = (data > thresholds).astype(int)
    return binary


def gray_code_hypercube_adjacency(n_dim: int) -> np.ndarray:
    """
    构建n维超立方体Gray码邻接矩阵
    """
    n_vertices = 1 << n_dim
    A = np.zeros((n_vertices, n_vertices), dtype=int)
    for i in range(n_vertices):
        gray_i = i ^ (i >> 1)
        for d in range(n_dim):
            j = i ^ (1 << d)
            gray_j = j ^ (j >> 1)
            if bin(gray_i ^ gray_j).count('1') == 1:
                A[i, j] = 1
                A[j, i] = 1
    return A


def boolean_pca(data_binary: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    布尔主成分分析 (在实数域上执行，但输入为二元)
    """
    # 中心化
    mean = np.mean(data_binary, axis=0)
    centered = data_binary - mean
    cov = centered.T @ centered / len(data_binary)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    components = eigvecs[:, idx[:n_components]]
    embedding = centered @ components
    return embedding


def mod2_rank(matrix: np.ndarray) -> int:
    """
    计算F_2上矩阵的秩 (高斯消元)
    """
    A = matrix.copy() % 2
    m, n = A.shape
    rank = 0
    row = 0
    for col in range(n):
        if row >= m:
            break
        # 找主元
        pivot = -1
        for r in range(row, m):
            if A[r, col] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        # 交换
        A[[row, pivot]] = A[[pivot, row]]
        # 消去
        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r] = (A[r] + A[row]) % 2
        row += 1
        rank += 1
    return rank


def binary_feature_hash(data_binary: np.ndarray, n_bits: int = 16) -> np.ndarray:
    """
    将二元特征矩阵哈希为紧凑的二进制编码
    使用随机投影的符号函数:
        h(x) = sign(R x) mod 2
    其中 R 为随机高斯矩阵
    """
    n, d = data_binary.shape
    np.random.seed(42)
    R = np.random.randn(n_bits, d)
    projected = R @ data_binary.T
    hash_codes = (projected > 0).astype(int).T
    return hash_codes


def hamming_distance_matrix(binary_data: np.ndarray) -> np.ndarray:
    """
    计算二元数据间的Hamming距离矩阵
    """
    n = len(binary_data)
    D = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = int(np.sum(np.abs(binary_data[i] - binary_data[j])))
            D[i, j] = d
            D[j, i] = d
    return D


def discrete_wasserstein_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    一维离散分布的Wasserstein-1距离 (Earth Mover's Distance)
    W_1(p, q) = ∫ |F_p(x) - F_q(x)| dx
    其中 F 为累积分布函数
    """
    p = p / (np.sum(p) + 1e-15)
    q = q / (np.sum(q) + 1e-15)
    cum_p = np.cumsum(p)
    cum_q = np.cumsum(q)
    return float(np.sum(np.abs(cum_p - cum_q)))


def lights_out_feature_transform(data: np.ndarray, grid_size: int = 5) -> np.ndarray:
    """
    利用Lights Out矩阵对数据进行特征变换
    将数据映射到5x5网格，求解Lights Out问题作为特征编码
    """
    n = len(data)
    from topological_invariants import lights_out_matrix
    A = lights_out_matrix(grid_size, grid_size)
    # 将数据投影到网格
    features = []
    for pt in data:
        # 取前25个维度或复制填充
        vec = np.zeros(grid_size * grid_size)
        d = min(len(pt), grid_size * grid_size)
        vec[:d] = pt[:d]
        # 阈值化
        median = np.median(vec)
        binary = (vec > median).astype(int)
        # 求解 A p = binary (mod 2)
        p = np.linalg.lstsq(A.astype(float), binary.astype(float), rcond=None)[0]
        p = (p > 0.5).astype(int)
        features.append(p)
    return np.array(features)
