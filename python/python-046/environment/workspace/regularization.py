
import numpy as np
from utils import check_finite, ensure_positive_definite


def build_laplacian_1d(n, h=1.0):
    if n < 3:
        raise ValueError("build_laplacian_1d: n must be >= 3")
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i] = 1.0 / (h * h)
        L[i, i + 1] = -2.0 / (h * h)
        L[i, i + 2] = 1.0 / (h * h)
    return L


def build_laplacian_2d(nx, ny, hx=1.0, hy=1.0):
    N = nx * ny
    L = np.zeros((N, N))
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i

            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                L[idx, idx] = 1.0
                continue
            L[idx, idx] = -2.0 / (hx * hx) - 2.0 / (hy * hy)
            L[idx, idx - 1] = 1.0 / (hx * hx)
            L[idx, idx + 1] = 1.0 / (hx * hx)
            L[idx, idx - nx] = 1.0 / (hy * hy)
            L[idx, idx + nx] = 1.0 / (hy * hy)
    return L


def tikhonov_solve(G, W, d, lam, L):








    raise NotImplementedError("tikhonov_solve: 待实现 Tikhonov 正则化求解")


def l_curve_analysis(G, W, d, L, lam_list):
    res_norms = []
    reg_norms = []
    for lam in lam_list:
        m, _ = tikhonov_solve(G, W, d, lam, L)
        res = G @ m - d
        reg = L @ m
        res_norms.append(np.linalg.norm(res))
        reg_norms.append(np.linalg.norm(reg))
    return np.array(res_norms), np.array(reg_norms)


def compute_gcv_score(G, W, d, lam, L):
    M = len(d)
    m, cov = tikhonov_solve(G, W, d, lam, L)


    A = G.T @ W @ G + (lam ** 2) * (L.T @ L)

    H = G @ np.linalg.solve(A, G.T @ W)
    tr_H = np.trace(H)
    res = G @ m - d
    numerator = np.sum((np.sqrt(np.diag(W)) * res) ** 2)
    denominator = (M - tr_H) ** 2
    if denominator <= 0:
        return float('inf')
    return numerator / denominator


def find_optimal_lambda_gcv(G, W, d, L, lam_candidates):
    scores = []
    for lam in lam_candidates:
        score = compute_gcv_score(G, W, d, lam, L)
        scores.append(score)
    scores = np.array(scores)
    best_idx = np.argmin(scores)
    return lam_candidates[best_idx], scores
