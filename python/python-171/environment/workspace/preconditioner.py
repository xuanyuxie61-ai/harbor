# -*- coding: utf-8 -*-
"""
preconditioner.py
=================
稀疏线性系统预处理技术集合。

融合种子项目：
- 641_laguerre_polynomial / 525_hermite_rule : 正交多项式零点构造谱预处理
- 506_hankel_spd                            : Hankel 矩阵 Cholesky 分解思想
- 244_cvt_1d_lumping                        : CVT 自适应密度（用于构造变系数预处理）
"""

import numpy as np
import math
from orthogonal_polynomials import build_polynomial_preconditioner_spectrum


# ---------------------------------------------------------------------------
# 基础预处理子
# ---------------------------------------------------------------------------

def jacobi_preconditioner(A):
    """
    Jacobi 预处理子（对角缩放）：
        M^{-1} = diag(A)^{-1}
    返回函数 apply(r) = M^{-1} r。
    """
    A = np.asarray(A, dtype=float)
    diag = np.diag(A).copy()
    diag = np.where(np.abs(diag) < 1e-30, 1.0, diag)
    inv_diag = 1.0 / diag

    def apply(r):
        return inv_diag * np.asarray(r, dtype=float).flatten()

    return apply


def ssor_preconditioner(A, omega=1.5):
    """
    SSOR (Symmetric Successive Over-Relaxation) 预处理子。
    对 SPD 矩阵 A = D + L + L^T，其中 D 为对角，L 为严格下三角：
        M = (D + ωL) D^{-1} (D + ωL^T) / (ω(2-ω))
    本函数实现 M^{-1} r 的求解，通过前代-后代两步：
        (D + ωL) y = r
        (D + ωL^T) z = D y
        M^{-1} r = z * ω(2-ω)
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    D = np.diag(A).copy()
    D = np.where(np.abs(D) < 1e-30, 1.0, D)

    def apply(r):
        r = np.asarray(r, dtype=float).flatten()
        # 前代: (D + ωL) y = r
        y = np.zeros(n, dtype=float)
        for i in range(n):
            sum_val = r[i]
            for j in range(i):
                sum_val -= omega * A[i, j] * y[j]
            y[i] = sum_val / D[i]
        # 后代: (D + ωL^T) z = D y
        z = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            sum_val = D[i] * y[i]
            for j in range(i + 1, n):
                sum_val -= omega * A[j, i] * z[j]
            z[i] = sum_val / D[i]
        return z * omega * (2.0 - omega)

    return apply


def incomplete_cholesky_ic0(A, drop_tol=1e-12):
    """
    零填充不完全 Cholesky 分解（IC(0)）。
    对 A 的稀疏模式做精确 Cholesky，但仅保留 A 的非零模式位置。
    返回函数 apply(r) 解 L L^T x = r。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    nz_mask = np.abs(A) > drop_tol

    for i in range(n):
        for j in range(i + 1):
            if not nz_mask[i, j]:
                continue
            sum_val = A[i, j]
            for k in range(j):
                if nz_mask[i, k] and nz_mask[j, k]:
                    sum_val -= L[i, k] * L[j, k]
            if i == j:
                if sum_val <= 0:
                    sum_val = abs(sum_val) + 1e-10
                L[i, j] = math.sqrt(sum_val)
            else:
                if abs(L[j, j]) > 1e-30:
                    L[i, j] = sum_val / L[j, j]

    def apply(r):
        r = np.asarray(r, dtype=float).flatten()
        # 前代 L y = r
        y = np.zeros(n, dtype=float)
        for i in range(n):
            sum_val = r[i]
            for j in range(i):
                sum_val -= L[i, j] * y[j]
            if abs(L[i, i]) > 1e-30:
                y[i] = sum_val / L[i, i]
        # 后代 L^T x = y
        x = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            sum_val = y[i]
            for j in range(i + 1, n):
                sum_val -= L[j, i] * x[j]
            if abs(L[i, i]) > 1e-30:
                x[i] = sum_val / L[i, i]
        return x

    return apply


# ---------------------------------------------------------------------------
# 谱预处理（基于正交多项式零点）
# ---------------------------------------------------------------------------

def polynomial_spectral_preconditioner(A, poly_type='laguerre', n_nodes=16):
    """
    基于正交多项式零点的谱预处理子。
    思想：利用正交多项式（Laguerre/Hermite）的零点构造多项式 P(A) 近似 A^{-1}。

    对 SPD 矩阵 A，设其特征值在 [λ_min, λ_max] 上，定义缩放矩阵：
        Ã = (A - cI) / s
    其中 c, s 将 A 的谱映射到正交多项式的定义域。
    然后构造 n_nodes 阶多项式近似：
        P(A) ≈ Σ_{k=0}^{n_nodes-1} α_k T_k(Ã)
    其中 T_k 为 Chebyshev 多项式，系数 α_k 由正交多项式求积规则确定。

    本简化实现采用对角缩放 + 基于条件数的谱变换：
        M^{-1} = D^{-1/2} * P(D^{-1/2} A D^{-1/2}) * D^{-1/2}
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    D = np.diag(A).copy()
    D = np.where(np.abs(D) < 1e-30, 1.0, D)
    D_inv_sqrt = 1.0 / np.sqrt(D)

    # 估计条件数（幂法）
    lam_max, lam_min = _estimate_extreme_eigenvalues(A, max_iter=20)
    kappa = lam_max / max(lam_min, 1e-30)

    # 构造节点和权重
    nodes, weights = build_polynomial_preconditioner_spectrum(n_nodes, poly_type)

    # TODO: 将正交多项式节点映射到矩阵谱区间 [lam_min, lam_max]
    #       注意：Laguerre 节点定义在 [0, ∞)，Hermite 节点定义在 (-∞, ∞)
    #       映射方式必须与 build_polynomial_preconditioner_spectrum 返回的节点语义一致
    # TODO: 构造多项式平滑的 Richardson 迭代步长系数
    #       利用权重和映射后的节点构造 coeff，使得 coeff 之和为 1

    def apply(r):
        r = np.asarray(r, dtype=float).flatten()
        # TODO: 实现对角预处理 D^{-1/2} * r 和多项式平滑（Richardson 迭代）
        #       迭代步长由映射后的谱节点决定
        #       最终返回 D^{-1/2} * x_smoothed
        pass

    return apply


def _estimate_extreme_eigenvalues(A, max_iter=30):
    """幂法估计最大/最小特征值。"""
    n = A.shape[0]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    for _ in range(max_iter):
        y = A @ x
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-30:
            break
        x = y / norm_y
    lam_max = float(x @ (A @ x))

    # 最小特征值（用带移位的幂法近似）
    x2 = np.random.randn(n)
    x2 = x2 / np.linalg.norm(x2)
    shift = lam_max * 1.01
    for _ in range(max_iter):
        try:
            y2 = np.linalg.solve(A - shift * np.eye(n), x2)
        except np.linalg.LinAlgError:
            y2 = x2
        norm_y2 = np.linalg.norm(y2)
        if norm_y2 < 1e-30:
            break
        x2 = y2 / norm_y2
    lam_min = float(x2 @ (A @ x2))
    if lam_min <= 0:
        lam_min = 1e-6
    return lam_max, lam_min


# ---------------------------------------------------------------------------
# 多重网格型粗网格预处理（基于 mesh_refinement 思想）
# ---------------------------------------------------------------------------

def two_grid_preconditioner(A_coarse, prolongation, restriction):
    """
    两层网格预处理子。
    A_coarse: 粗网格矩阵
    P: 延拓算子 (fine -> coarse 的逆方向，即 coarse -> fine)
    R: 限制算子 (fine -> coarse)
    预处理步：
        1) 前光滑（Jacobi）
        2) 粗网格修正：e_c = A_coarse^{-1} R r
        3) 延拓：x += P e_c
        4) 后光滑（Jacobi）
    """
    P = np.asarray(prolongation, dtype=float)
    R = np.asarray(restriction, dtype=float)
    A_c = np.asarray(A_coarse, dtype=float)
    n_f = P.shape[0]

    # 粗网格 LU 或逆
    try:
        A_c_inv = np.linalg.inv(A_c)
    except np.linalg.LinAlgError:
        A_c_inv = np.eye(A_c.shape[0])

    def apply(r):
        r = np.asarray(r, dtype=float).flatten()
        x = np.zeros(n_f, dtype=float)
        # 前光滑：一步 Jacobi
        diag = np.diag(A_coarse if False else np.eye(n_f))  # placeholder
        # 简化为直接用粗网格修正
        r_c = R @ r
        e_c = A_c_inv @ r_c
        x = P @ e_c
        return x

    return apply


# ---------------------------------------------------------------------------
# 块对角预处理（对多维张量积矩阵）
# ---------------------------------------------------------------------------

def block_diagonal_preconditioner(blocks):
    """
    块对角预处理子。
    blocks: list of ndarray，每个为对角块。
    构造 M^{-1} = diag(blocks[0]^{-1}, blocks[1]^{-1}, ...)。
    """
    inverses = []
    for B in blocks:
        B = np.asarray(B, dtype=float)
        try:
            invB = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            invB = np.eye(B.shape[0])
        inverses.append(invB)

    def apply(r):
        r = np.asarray(r, dtype=float).flatten()
        x = np.array([])
        offset = 0
        for invB in inverses:
            m = invB.shape[0]
            x = np.concatenate([x, invB @ r[offset:offset + m]])
            offset += m
        return x

    return apply
