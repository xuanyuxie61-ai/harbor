"""
eigenmode_decomposition.py
==========================

SEI 浓度场本征模分解（PCA / SVD）模块。

融合种子项目：
    326_eigenfaces：主成分分析 (PCA) 提取本征向量

核心思想：
    将 SEI 演化过程中的浓度场时间序列视为高维数据矩阵，
    使用 PCA 提取主导本征模（本征浓度分布），
    分析各模态的能量占比和物理含义。

数学描述：
    数据矩阵 A ∈ R^{N×T}（N 空间点, T 时间步）
    均值场 Ψ = (1/T) Σ A(:,t)
    协方差矩阵 L = A_centered^T A_centered
    本征分解 L v = λ v
    本征模 u = A_centered v / ||A_centered v||

    能量占比：η_k = λ_k / Σ λ_i

作者: DA-Synthesis
"""

import math
try:
    from . import sei_parameters as P
except ImportError:
    import sei_parameters as P


def compute_mean_field(snapshots):
    """
    计算快照集的均值场（参考 326_eigenfaces pc_vectors）。

    Parameters
    ----------
    snapshots : list[list[float]]
        快照矩阵，每行是一个时间步的空间分布。

    Returns
    -------
    psi : list[float]
        均值场。
    """
    if not snapshots:
        return []
    n = len(snapshots[0])
    t = len(snapshots)
    psi = [0.0] * n
    for snap in snapshots:
        for i in range(n):
            psi[i] += snap[i]
    psi = [p / t for p in psi]
    return psi


def center_snapshots(snapshots, mean_field):
    """
    从快照中减去均值场。

    Parameters
    ----------
    snapshots : list[list[float]]
        原始快照。
    mean_field : list[float]
        均值场。

    Returns
    -------
    centered : list[list[float]]
        中心化后的快照。
    """
    centered = []
    for snap in snapshots:
        centered.append([snap[i] - mean_field[i] for i in range(len(snap))])
    return centered


def compute_covariance_matrix(centered):
    """
    计算协方差矩阵 L = A^T A（参考 326_eigenfaces）。

    Parameters
    ----------
    centered : list[list[float]]
        中心化快照矩阵（T × N）。

    Returns
    -------
    L : list[list[float]]
        协方差矩阵（T × T）。
    """
    t = len(centered)
    if t == 0:
        return []
    n = len(centered[0])

    L = [[0.0] * t for _ in range(t)]
    for i in range(t):
        for j in range(i, t):
            dot = sum(centered[i][k] * centered[j][k] for k in range(n))
            L[i][j] = dot
            L[j][i] = dot

    return L


def eigen_decomposition_2x2(a, b, c, d):
    """
    2×2 对称矩阵的本征分解（解析解）。

    用于快速验证和小规模问题。

    Parameters
    ----------
    a, b, c, d : float
        矩阵元素 [[a, b], [c, d]]（b = c for symmetric）。

    Returns
    -------
    values : list[float]
        本征值（降序）。
    vectors : list[list[float]]
        本征向量（列向量）。
    """
    trace = a + d
    det = a * d - b * c
    disc = trace * trace / 4.0 - det
    if disc < 0:
        disc = 0.0
    sqrt_disc = math.sqrt(disc)
    lam1 = trace / 2.0 + sqrt_disc
    lam2 = trace / 2.0 - sqrt_disc

    if abs(b) > 1.0e-30:
        v1 = [b, lam1 - a]
        v2 = [b, lam2 - a]
    else:
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]

    # 归一化
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if n1 > 1.0e-30:
        v1 = [v / n1 for v in v1]
    if n2 > 1.0e-30:
        v2 = [v / n2 for v in v2]

    return [lam1, lam2], [v1, v2]


def power_iteration_eigen(matrix, n_vectors, max_iter=100, tol=1.0e-8):
    """
    幂迭代法求前 n_vectors 个本征对（参考 326_eigenfaces 方法）。

    用于中等规模对称矩阵的本征分解。

    Parameters
    ----------
    matrix : list[list[float]]
        对称矩阵。
    n_vectors : int
        需要的本征向量数。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容限。

    Returns
    -------
    values : list[float]
        本征值（降序）。
    vectors : list[list[float]]
        本征向量列表。
    """
    n = len(matrix)
    if n == 0:
        return [], []

    n_vectors = min(n_vectors, n)
    values = []
    vectors = []

    # 深拷贝矩阵用于 deflation
    work = [row[:] for row in matrix]

    for k in range(n_vectors):
        # 随机初始向量（确定性种子）
        v = [1.0 / math.sqrt(n)] * n
        if k > 0:
            v = [(i + k + 1) % 3 - 1.0 for i in range(n)]
            norm_v = math.sqrt(sum(x * x for x in v))
            if norm_v > 1.0e-30:
                v = [x / norm_v for x in v]

        lam = 0.0
        for iteration in range(max_iter):
            # 矩阵-向量乘法
            w = [sum(work[i][j] * v[j] for j in range(n)) for i in range(n)]

            # Rayleigh 商
            lam_new = sum(v[i] * w[i] for i in range(n))

            # 归一化
            norm_w = math.sqrt(sum(x * x for x in w))
            if norm_w < 1.0e-30:
                break
            v_new = [x / norm_w for x in w]

            # 收敛检查
            if abs(lam_new - lam) < tol * (abs(lam) + 1.0):
                v = v_new
                lam = lam_new
                break

            v = v_new
            lam = lam_new

        values.append(lam)
        vectors.append(v)

        # Deflation: work = work - lam * v v^T
        for i in range(n):
            for j in range(n):
                work[i][j] -= lam * v[i] * v[j]

    return values, vectors


def project_onto_eigenmodes(centered, eigen_vectors):
    """
    将中心化快照投影到本征模上（参考 326_eigenfaces pgm_project）。

    Parameters
    ----------
    centered : list[list[float]]
        中心化快照。
    eigen_vectors : list[list[float]]
        本征向量（在快照空间中）。

    Returns
    -------
    coefficients : list[list[float]]
        投影系数矩阵。
    """
    t = len(centered)
    n_modes = len(eigen_vectors)
    coefficients = [[0.0] * n_modes for _ in range(t)]

    for t_idx in range(t):
        for m_idx in range(n_modes):
            dot = sum(centered[t_idx][i] * eigen_vectors[m_idx][i]
                     for i in range(t))
            coefficients[t_idx][m_idx] = dot

    return coefficients


def reconstruct_physical_modes(centered, eigen_vectors_t):
    """
    将快照空间的本征向量转换回物理空间（参考 326_eigenfaces）。

    u_k = A^T v_k / ||A^T v_k||

    Parameters
    ----------
    centered : list[list[float]]
        中心化快照 (T × N)。
    eigen_vectors_t : list[list[float]]
        快照空间的本征向量。

    Returns
    -------
    physical_modes : list[list[float]]
        物理空间的本征模（N 维）。
    """
    t = len(centered)
    if t == 0:
        return []
    n = len(centered[0])
    n_modes = len(eigen_vectors_t)

    physical_modes = []
    for m in range(n_modes):
        mode = [0.0] * n
        for i in range(n):
            for j in range(t):
                mode[i] += centered[j][i] * eigen_vectors_t[m][j]
        # 归一化
        norm = math.sqrt(sum(x * x for x in mode))
        if norm > 1.0e-30:
            mode = [x / norm for x in mode]
        physical_modes.append(mode)

    return physical_modes


def compute_energy_fractions(eigenvalues):
    """
    计算各本征模的能量占比。

    η_k = λ_k / Σ λ_i

    Parameters
    ----------
    eigenvalues : list[float]
        本征值。

    Returns
    -------
    fractions : list[float]
        能量占比。
    """
    total = sum(abs(lam) for lam in eigenvalues)
    if total < 1.0e-30:
        return [0.0] * len(eigenvalues)
    return [abs(lam) / total for lam in eigenvalues]


def run_pca_demo(snapshots=None, n_modes=3):
    """
    演示 SEI 浓度场的 PCA 本征模分解。

    Parameters
    ----------
    snapshots : list[list[float]] or None
        快照数据，None 时使用合成数据。
    n_modes : int
        提取的模态数。

    Returns
    -------
    dict
        PCA 分析结果。
    """
    if snapshots is None:
        # 合成快照：指数衰减 + 空间振荡
        n = P.N_GRID
        t_snap = 20
        snapshots = []
        for t in range(t_snap):
            snap = []
            for i in range(n):
                x = i * P.DX
                # 基态 + 衰减振荡
                val = (P.C_LI_INIT
                       * (1.0 - 0.2 * math.exp(-t / 5.0)
                          * math.sin(math.pi * x / P.L_DOMAIN)))
                snap.append(val)
            snapshots.append(snap)

    psi = compute_mean_field(snapshots)
    centered = center_snapshots(snapshots, psi)
    L = compute_covariance_matrix(centered)
    values, vectors_t = power_iteration_eigen(L, n_modes)
    fractions = compute_energy_fractions(values)

    return {
        "mean_field": psi,
        "eigenvalues": values,
        "eigen_vectors_t": vectors_t,
        "energy_fractions": fractions,
        "n_snapshots": len(snapshots),
        "n_modes": n_modes,
    }


if __name__ == "__main__":
    result = run_pca_demo()
    print(f"[eigenmode] PCA 本征模分解")
    print(f"  快照数: {result['n_snapshots']}")
    print(f"  提取模态: {result['n_modes']}")
    for i, (lam, frac) in enumerate(
            zip(result['eigenvalues'], result['energy_fractions'])):
        print(f"  模态 {i}: λ={lam:.6e}, 能量占比 η={frac:.4f}")
