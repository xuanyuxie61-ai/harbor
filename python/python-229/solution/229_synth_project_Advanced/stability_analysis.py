"""
stability_analysis.py
=====================

稳定性分析核心
--------------
unfold 的数值稳定性通过以下指标评估:

1. 条件数 κ(R):
       κ(R) = σ_max / σ_min
   κ 越大, 反演越不稳定. 典型探测器 κ(R) ∈ [10^3, 10^8].

2. 传播矩阵 H (hat matrix):
       T = H O,   H = V Σ^{-1} U^T  (SVD unfold)
   或 H = (R^T R + λ L^T L)^{-1} R^T  (Tikhonov)

3. 方差放大:
       Var(T_i) = Σ_j H_{ij}^2 Var(O_j)
   若 Var(O_j) = O_j (泊松), 则:
       σ(T_i) = √(Σ_j H_{ij}^2 O_j)

4. 高阶 FD 光滑度:
       S_p = (1/N) Σ_i |Δ^p T_i|^2 / h^{2p}
   p = 1, 2, 3, ...
   S_p 大 → 解不光滑 → 可能过拟合.

5. Figure of Merit (FOM):
       FOM = √(bias^2 + variance)
   bias = ‖T_unfold - T_true‖ / ‖T_true‖
   variance = (1/N) Σ_i σ(T_i)^2 / T_i^2

本模块计算上述所有指标.
"""

from __future__ import annotations
from typing import List, Tuple
import math
import finite_difference as fd


# ===========================================================================
#          条件数与奇异值谱
# ===========================================================================

def condition_number_from_singulars(sigma: List[float]) -> float:
    """
    从奇异值列表计算条件数:
        κ = σ_max / σ_min (非零最小奇异值)
    """
    non_zero = [s for s in sigma if s > 1e-14]
    if not non_zero:
        return float('inf')
    return max(non_zero) / min(non_zero)


def effective_rank(sigma: List[float], threshold: float = 1e-3) -> int:
    """
    有效秩: σ_i > threshold * σ_max 的个数.
    """
    if not sigma:
        return 0
    s_max = max(sigma)
    return sum(1 for s in sigma if s > threshold * s_max)


# ===========================================================================
#          传播矩阵 H (hat matrix)
# ===========================================================================

def hat_matrix_svd(
    U: List[List[float]],
    sigma: List[float],
    V_T: List[List[float]],
    k_cutoff: int = None,
) -> List[List[float]]:
    """
    SVD unfold 的传播矩阵:
        H = V_k Σ_k^{-1} U_k^T

    H 的大小: N_true × N_rec.

    若 k_cutoff 未指定, 使用全部非零奇异值.
    """
    m = len(U)        # N_rec
    n = len(V_T[0])   # N_true
    if k_cutoff is None:
        k_cutoff = min(m, n)
    k_cutoff = min(k_cutoff, len(sigma))

    # H[i][j] = Σ_{s=0}^{k-1} V_T[s][i] * (1/σ_s) * U[j][s]
    H = [[0.0] * m for _ in range(n)]
    for s in range(k_cutoff):
        if sigma[s] < 1e-14:
            continue
        inv_s = 1.0 / sigma[s]
        for i in range(n):
            v = V_T[s][i]
            for j in range(m):
                H[i][j] += v * inv_s * U[j][s]
    return H


def hat_matrix_tikhonov(
    R_dense: List[List[float]],
    lam: float,
    L_order: int = 1,
) -> List[List[float]]:
    """
    Tikhonov unfold 的传播矩阵:
        H = (R^T R + λ L^T L)^{-1} R^T
    """
    import regularization as reg
    N_true = len(R_dense[0])
    N_rec = len(R_dense)

    # R^T R
    RTR = [[0.0] * N_true for _ in range(N_true)]
    for i in range(N_true):
        for j in range(N_true):
            s = 0.0
            for k in range(N_rec):
                s += R_dense[k][i] * R_dense[k][j]
            RTR[i][j] = s

    # LTL
    LTL = reg._build_LtL(N_true, L_order)

    # A = RTR + λ LTL
    A = [[RTR[i][j] + lam * LTL[i][j] for j in range(N_true)] for i in range(N_true)]

    # A^{-1} via Gauss-Jordan
    Ainv = _matrix_inverse(A)

    # H = A^{-1} R^T
    H = [[0.0] * N_rec for _ in range(N_true)]
    for i in range(N_true):
        for j in range(N_rec):
            s = 0.0
            for k in range(N_true):
                s += Ainv[i][k] * R_dense[j][k]
            H[i][j] = s
    return H


def _matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    """Gauss-Jordan 求逆 (小矩阵)."""
    n = len(A)
    # 增广 [A | I]
    M = [A[i] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        # 部分主元
        max_row = col
        max_val = abs(M[col][col])
        for row in range(col + 1, n):
            if abs(M[row][col]) > max_val:
                max_val = abs(M[row][col])
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        if abs(pivot) < 1e-300:
            continue
        for j in range(2 * n):
            M[col][j] /= pivot
        for row in range(n):
            if row == col:
                continue
            factor = M[row][col]
            for j in range(2 * n):
                M[row][j] -= factor * M[col][j]
    return [M[i][n:] for i in range(n)]


# ===========================================================================
#          方差传播与 FOM
# ===========================================================================

def propagate_variance(
    H: List[List[float]],
    O: List[float],
) -> List[float]:
    """
    方差传播 (泊松噪声):
        Var(T_i) = Σ_j H_{ij}^2 Var(O_j) = Σ_j H_{ij}^2 O_j
        σ(T_i) = √(Var(T_i))

    返回 σ(T) 数组.
    """
    N_true = len(H)
    N_rec = len(O)
    sigma_T = [0.0] * N_true
    for i in range(N_true):
        s = 0.0
        for j in range(N_rec):
            s += H[i][j] * H[i][j] * max(O[j], 0.0)
        sigma_T[i] = math.sqrt(max(s, 0.0))
    return sigma_T


def figure_of_merit(
    T_unfold: List[float],
    T_true: List[float],
    sigma_T: List[float],
) -> Tuple[float, float, float]:
    """
    FOM 计算:
        bias      = ‖T_unfold - T_true‖ / ‖T_true‖
        variance  = (1/N) Σ_i (σ(T_i) / T_i)^2
        FOM       = √(bias^2 + variance)

    返回 (bias, variance, FOM).
    """
    N = len(T_true)
    if N == 0:
        return 0.0, 0.0, 0.0

    # bias
    norm_true = math.sqrt(sum(t * t for t in T_true))
    if norm_true < 1e-300:
        norm_true = 1e-300
    bias = math.sqrt(sum((T_unfold[i] - T_true[i]) ** 2 for i in range(N))) / norm_true

    # variance
    var_sum = 0.0
    n_valid = 0
    for i in range(N):
        if T_true[i] > 1e-10:
            var_sum += (sigma_T[i] / T_true[i]) ** 2
            n_valid += 1
    variance = var_sum / max(n_valid, 1)

    fom = math.sqrt(bias * bias + variance)
    return bias, variance, fom


# ===========================================================================
#          FD 光滑度指标
# ===========================================================================

def fd_smoothness(
    T: List[float],
    E_edges: List[float],
    max_order: int = 3,
) -> List[float]:
    """
    计算各阶 FD 光滑度:
        S_p = (1/N) Σ_i |Δ^p T_i|^2 / h^{2p}

    返回 [S_1, S_2, ..., S_{max_order}].
    """
    N = len(T)
    if N < max_order + 1:
        return [0.0] * max_order

    h = (E_edges[-1] - E_edges[0]) / max(N - 1, 1)
    result = []
    for p in range(1, max_order + 1):
        dp = fd.fd_derivative(T, h, deriv_order=p, stencil_order=2)
        s = sum(d * d for d in dp) / max(N, 1) / (h ** (2 * p))
        result.append(s)
    return result


# ===========================================================================
#          综合稳定性报告
# ===========================================================================

def stability_report(
    sigma: List[float],
    H: List[List[float]],
    O: List[float],
    T_unfold: List[float],
    T_true: List[float],
    E_edges: List[float],
) -> dict:
    """
    生成综合稳定性报告.
    """
    kappa = condition_number_from_singulars(sigma)
    k_eff = effective_rank(sigma)

    sigma_T = propagate_variance(H, O)
    bias, variance, fom = figure_of_merit(T_unfold, T_true, sigma_T)
    smooth = fd_smoothness(T_unfold, E_edges, max_order=3)

    return {
        "condition_number": kappa,
        "effective_rank": k_eff,
        "bias_rel": bias,
        "variance_mean": variance,
        "FOM": fom,
        "fd_smoothness_S1": smooth[0],
        "fd_smoothness_S2": smooth[1],
        "fd_smoothness_S3": smooth[2],
    }


__all__ = [
    "condition_number_from_singulars",
    "effective_rank",
    "hat_matrix_svd",
    "hat_matrix_tikhonov",
    "propagate_variance",
    "figure_of_merit",
    "fd_smoothness",
    "stability_report",
]
