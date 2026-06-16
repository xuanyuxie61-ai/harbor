"""
solution_manifold.py — 解流形分析与低维嵌入
==============================================

本模块对随机 PDE 的解集合进行流形分析:
  1. 多维标度 (MDS): 从距离矩阵重构解流形的低维坐标
  2. 最近邻搜索: 参数空间中的自适应采样引导
  3. 本征正交分解 (POD): 解流形的主成分分析

数学框架:
  给定 N 个解快照 {u(x,ξⁱ)}, 构造距离矩阵:
      D_{ij} = ||u(·,ξⁱ) - u(·,ξʲ)||_{L²}
  MDS: 寻找 Y ∈ ℝ^{d×N} 使 ||D_{ij} - ||Y_i - Y_j||||² 最小。
  等价于经典 MDS: Y = Λ^{1/2} Vᵀ 从 -J·D²·J/2 的特征分解。

映射种子项目:
  - 306_distance_to_position: 多维标度 (MDS) 非线性最小二乘
  - 190_closest_pair_brute: 最近点对搜索
"""

import numpy as np
from numpy.linalg import eigh


# ============================================================
# 第1部分: 距离矩阵计算
# ============================================================

def compute_solution_distance_matrix(snapshots, norm_type='L2'):
    """
    计算解快照间的距离矩阵:
        D_{ij} = ||uⁱ - uʲ||_{norm}

    支持:
      'L2':    L² 范数
      'H1':    H¹ Sobolev 范数 (含梯度)
      'energy': 能量范数 ||u||_a = √(a∇u·∇u)

    参数:
        snapshots: (N, M) N 个快照, 每个 M 维
        norm_type: 范数类型

    返回:
        D: (N, N) 对称距离矩阵
    """
    N = len(snapshots)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            diff = snapshots[i] - snapshots[j]
            if norm_type == 'L2':
                d = np.sqrt(np.sum(diff ** 2))
            elif norm_type == 'H1':
                # H1: L2 + 差分近似梯度
                if len(diff) > 1:
                    grad_diff = np.diff(diff)
                    d = np.sqrt(np.sum(diff ** 2) + np.sum(grad_diff ** 2))
                else:
                    d = np.abs(diff[0])
            elif norm_type == 'energy':
                # 能量范数近似: 加权 L2
                weights = np.linspace(0.5, 1.5, len(diff))
                d = np.sqrt(np.sum(weights * diff ** 2))
            else:
                d = np.sqrt(np.sum(diff ** 2))
            D[i, j] = d
            D[j, i] = d
    return D


# ============================================================
# 第2部分: 经典多维标度 (cMDS)
# (映射自 306_distance_to_position: MDS via lsqnonlin)
# ============================================================

def classical_mds(distance_matrix, n_components=2):
    """
    经典 MDS (Torgerson):
        1. 构造 B = -J·D²·J/2, J = I - 11ᵀ/N (中心化矩阵)
        2. 特征分解: B = V·Λ·Vᵀ
        3. 嵌入: Y = Λₖ^{1/2}·Vₖᵀ

    这比非线性最小二乘更稳定, 且有闭合解。
    非线性 MDS 的应力函数:
        Stress = √(Σ(D_ij - ||Y_i-Y_j||)² / Σ D_ij²)

    参数:
        distance_matrix: (N, N) 距离矩阵
        n_components: 嵌入维数

    返回:
        Y: (N, n_components) 嵌入坐标
        eigenvalues: 前 n_components 个特征值
        stress: 嵌入应力 (越小越好)
    """
    N = distance_matrix.shape[0]
    D2 = distance_matrix ** 2
    # 双中心化
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ D2 @ H
    # 对称化
    B = (B + B.T) / 2.0
    # 特征分解
    eigenvalues, eigenvectors = eigh(B)
    # 降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # 取前 k 个
    k = min(n_components, N)
    eig_k = eigenvalues[:k]
    vec_k = eigenvectors[:, :k]
    # 处理负特征值
    eig_k_pos = np.maximum(eig_k, 0.0)
    Y = vec_k * np.sqrt(eig_k_pos)[np.newaxis, :]
    # 应力计算
    if N > 1:
        D_embedded = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt(np.sum((Y[i] - Y[j]) ** 2))
                D_embedded[i, j] = d
                D_embedded[j, i] = d
        numerator = np.sum((distance_matrix - D_embedded) ** 2)
        denominator = np.sum(distance_matrix ** 2)
        stress = np.sqrt(numerator / max(denominator, 1e-30))
    else:
        stress = 0.0
    return Y, eig_k, stress


# ============================================================
# 第3部分: 非线性 MDS 迭代优化
# (映射自 306_distance_to_position: lsqnonlin 迭代)
# ============================================================

def stress_majorization_mds(distance_matrix, n_components=2, max_iter=100, tol=1e-6):
    """
    Stress Majorization MDS (SMACOF):
        迭代最小化 Kruskal 应力:
            Stress(Y) = √(Σ_{i<j} (D_ij - d_ij(Y))²)

        每次迭代:
            Y^{k+1} = V⁺·B(Y^k)·Y^k
        其中:
            V_{ij} = -1/D_ij(Y^k) (i≠j), V_{ii} = Σ_{j≠i} 1/D_ij(Y^k)
            B_{ij} = -D_ij/D_ij(Y^k) (i≠j), B_{ii} = Σ_{j≠i} D_ij/D_ij(Y^k)

    保证应力单调递减。
    """
    N = distance_matrix.shape[0]
    if N <= n_components:
        return np.zeros((N, n_components)), [0.0], 0.0
    # 随机初始化
    rng = np.random.default_rng(42)
    Y = rng.standard_normal((N, n_components)) * 0.1
    stresses = []

    for iteration in range(max_iter):
        # 计算当前嵌入距离
        D_emb = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt(np.sum((Y[i] - Y[j]) ** 2) + 1e-15)
                D_emb[i, j] = d
                D_emb[j, i] = d
        # 当前应力
        stress = np.sqrt(np.sum((distance_matrix - D_emb) ** 2))
        stresses.append(stress)
        # 收敛检测
        if len(stresses) > 1 and abs(stresses[-2] - stresses[-1]) < tol:
            break
        # 构造 B 和 V 矩阵
        B = np.zeros((N, N))
        V = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    d_emb = max(D_emb[i, j], 1e-15)
                    B[i, j] = -distance_matrix[i, j] / d_emb
                    V[i, j] = -1.0 / d_emb
            V[i, i] = -np.sum(V[i, :i].tolist() + V[i, i + 1:].tolist())
            B[i, i] = -np.sum(B[i, :i].tolist() + B[i, i + 1:].tolist())
        # 更新: Y = V⁺ B Y
        V_pinv = np.linalg.pinv(V)
        Y = V_pinv @ B @ Y
    return Y, stresses, stresses[-1] if stresses else float('inf')


# ============================================================
# 第4部分: POD / 本征正交分解
# ============================================================

def pod_decomposition(snapshots, n_modes=None, energy_threshold=0.999):
    """
    本征正交分解 (POD) /  Proper Orthogonal Decomposition:
        1. 计算均值: ū = (1/N) Σ uⁱ
        2. 脉动: u'ⁱ = uⁱ - ū
        3. 相关矩阵: C_{ij} = (u'ⁱ, u'ʲ)/N
        4. 特征分解: C = V·Λ·Vᵀ
        5. POD 模态: φₖ = (1/√(Nλₖ)) Σⱼ V_{jk} u'ʲ

    返回:
        mean_field: (M,) 均值场
        modes: (M, K) POD 模态
        eigenvalues: (K,) POD 特征值
        coefficients: (N, K) 投影系数
        energy_ratio: 累积能量比
    """
    N = len(snapshots)
    M = len(snapshots[0])
    mean_field = np.mean(snapshots, axis=0)
    fluctuations = snapshots - mean_field[np.newaxis, :]
    # 快照相关矩阵
    C = (fluctuations @ fluctuations.T) / N
    eigenvalues, eigvecs = eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigvecs = eigvecs[:, idx]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = np.sum(eigenvalues)
    if total < 1e-30:
        return mean_field, np.zeros((M, 1)), np.array([0.0]), \
               np.zeros((N, 1)), 0.0
    cumulative = np.cumsum(eigenvalues) / total
    if n_modes is None:
        n_modes = int(np.searchsorted(cumulative, energy_threshold) + 1)
        n_modes = min(n_modes, N)
    # POD 模态
    modes = np.zeros((M, n_modes))
    for k in range(n_modes):
        lam_k = eigenvalues[k]
        if lam_k > 1e-15:
            modes[:, k] = (fluctuations.T @ eigvecs[:, k]) / np.sqrt(N * lam_k)
    # 投影系数
    coefficients = fluctuations @ modes
    return mean_field, modes, eigenvalues[:n_modes], coefficients, cumulative[n_modes - 1]


# ============================================================
# 第5部分: 最近邻搜索
# (映射自 190_closest_pair_brute)
# ============================================================

def closest_pair_in_set(points):
    """
    暴力最近点对:
        d_min = min_{i<j} ||x_i - x_j||₂
    返回: (d_min, i, j)
    """
    n = len(points)
    d_min = float('inf')
    best_i, best_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            if d < d_min:
                d_min = d
                best_i, best_j = i, j
    return d_min, best_i, best_j


def nearest_neighbor_query(query, dataset):
    """
    对查询点找最近邻:
        j* = argmin_j ||query - dataset[j]||₂
    返回: (distance, index)
    """
    dists = np.sqrt(np.sum((dataset - query) ** 2, axis=1))
    j_star = np.argmin(dists)
    return dists[j_star], j_star


def adaptive_sampling_fill_gap(existing_points, manifold_dim=2, n_new=5, seed=42):
    """
    自适应采样: 在现有采样点最稀疏区域添加新点。
    策略:
      1. 找最近点对 → 该区域采样密度最高
      2. 在最远点对中间添加新点
      3. 使用 max-min 策略: 新点最大化到已有点集的最小距离

    返回:
        new_points: (n_new, d) 新增采样点
    """
    rng = np.random.default_rng(seed)
    d = existing_points.shape[1]
    new_points = []
    current_set = existing_points.copy()
    for _ in range(n_new):
        # max-min 策略: 候选点中选到已有点集最大最小距离的
        n_candidates = 50
        candidates = rng.standard_normal((n_candidates, d)) * 2.0
        best_candidate = None
        best_min_dist = -1.0
        for c in candidates:
            dists = np.sqrt(np.sum((current_set - c) ** 2, axis=1))
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_candidate = c
        if best_candidate is not None:
            new_points.append(best_candidate)
            current_set = np.vstack([current_set, best_candidate[np.newaxis, :]])
    return np.array(new_points) if new_points else np.zeros((0, existing_points.shape[1]))


# ============================================================
# 第6部分: 解流形维度估计
# ============================================================

def estimate_manifold_dimension(eigenvalues, threshold_ratio=0.01):
    """
    估计解流形的本征维度:
        d_eff = #{k : λ_k/λ_1 > threshold_ratio}

    物理含义: 随机 PDE 解在参数空间中的有效维数。
    """
    if len(eigenvalues) == 0 or eigenvalues[0] < 1e-30:
        return 0
    ratio = eigenvalues / eigenvalues[0]
    return int(np.sum(ratio > threshold_ratio))
