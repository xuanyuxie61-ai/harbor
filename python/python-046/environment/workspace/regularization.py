"""
regularization.py
正则化矩阵运算与稳定化模块。

融合种子项目:
  - 056_asa314 (invmod): 模运算矩阵求逆的思想，转化为正则化矩阵的数值稳定求逆。

在 InSAR 形变反演中的应用:
  1. Tikhonov 正则化: (G^T W G + λ^2 L^T L)^{-1}
  2. 拉普拉斯平滑算子 L 的构造
  3. 截断 SVD / L-curve 分析
  4. 正则化矩阵的条件数改善
"""

import numpy as np
from utils import check_finite, ensure_positive_definite


def build_laplacian_1d(n, h=1.0):
    """
    构造一维离散拉普拉斯算子 L (n-2 × n)。
    采用二阶中心差分:
        (L u)_i = (u_{i+1} - 2u_i + u_{i-1}) / h^2,  i = 1, ..., n-2
    """
    if n < 3:
        raise ValueError("build_laplacian_1d: n must be >= 3")
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i] = 1.0 / (h * h)
        L[i, i + 1] = -2.0 / (h * h)
        L[i, i + 2] = 1.0 / (h * h)
    return L


def build_laplacian_2d(nx, ny, hx=1.0, hy=1.0):
    """
    构造二维离散拉普拉斯平滑矩阵 L (nx*ny × nx*ny)。
    使用五点差分格式:
        ∇^2 u_{i,j} = (u_{i+1,j} - 2u_{i,j} + u_{i-1,j}) / hx^2
                    + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1}) / hy^2
    边界点设为恒等（Dirichlet 条件）。
    """
    N = nx * ny
    L = np.zeros((N, N))
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            # 边界点不施加拉普拉斯平滑
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
    """
    Tikhonov 正则化最小二乘求解:
        min_m  || W^{1/2} (G m - d) ||_2^2 + λ^2 || L m ||_2^2

    正规方程:
        (G^T W G + λ^2 L^T L) m = G^T W d

    参数:
        G:  设计矩阵 (M × N)
        W:  权重矩阵 (M × M)，对角阵或稠密阵
        d:  观测数据 (M,)
        lam: 正则化参数 λ
        L:  正则化算子 (K × N)

    返回:
        m:  反演结果 (N,)
        cov: 近似后验方差矩阵 (N × N)
    """
    # HOLE 1: 需实现 Tikhonov 正则化最小二乘求解的核心算法。
    # 关键步骤：
    #   1. 构造正规方程左端矩阵 A = G^T W G + λ^2 L^T L
    #   2. 构造右端向量 b = G^T W d
    #   3. 确保 A 正定（特征值截断）
    #   4. 求解线性方程组 A m = b
    #   5. 计算近似协方差 cov = A^{-1}
    # 返回 (m, cov)
    raise NotImplementedError("tikhonov_solve: 待实现 Tikhonov 正则化求解")


def l_curve_analysis(G, W, d, L, lam_list):
    """
    L-curve 分析：计算不同 λ 下的残差范数 ||Gm-d|| 和解范数 ||Lm||。

    返回:
        res_norms: list of float
        reg_norms: list of float
    """
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
    """
    广义交叉验证 (GCV) 分数计算:
        GCV(λ) = || W^{1/2} (G m_λ - d) ||^2 / (M - tr(H))^2
    其中 H = G (G^T W G + λ^2 L^T L)^{-1} G^T W 是帽子矩阵。
    """
    M = len(d)
    m, cov = tikhonov_solve(G, W, d, lam, L)
    # 近似 trace(H) = trace(G cov G^T W) = trace(G^T W G cov)
    # 使用随机估计简化
    A = G.T @ W @ G + (lam ** 2) * (L.T @ L)
    # 直接计算 hat matrix 的对角线之和
    H = G @ np.linalg.solve(A, G.T @ W)
    tr_H = np.trace(H)
    res = G @ m - d
    numerator = np.sum((np.sqrt(np.diag(W)) * res) ** 2)
    denominator = (M - tr_H) ** 2
    if denominator <= 0:
        return float('inf')
    return numerator / denominator


def find_optimal_lambda_gcv(G, W, d, L, lam_candidates):
    """
    通过 GCV 选取最优正则化参数 λ。
    """
    scores = []
    for lam in lam_candidates:
        score = compute_gcv_score(G, W, d, lam, L)
        scores.append(score)
    scores = np.array(scores)
    best_idx = np.argmin(scores)
    return lam_candidates[best_idx], scores
