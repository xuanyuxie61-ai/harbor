"""
regularization.py
=================

来源: 834_opt_golden
--------------------
原项目实现了黄金分割搜索, 用于最小化单变量函数:
    区间 [a, b], 黄金比 g = (√5 - 1)/2 ≈ 0.618,
    每步缩小区间, 线性收敛.

物理重构: 正则化参数优选
--------------------------------
在 unfold 中, 正则化参数 λ 控制偏差-方差权衡:
    - λ 太小 → 过拟合 (噪声放大)
    - λ 太大 → 欠拟合 (偏置大)

优选 λ 的方法:
1. 黄金分割搜索最小化:
       f(λ) = χ²(λ) + α · ||T(λ)||^2   (L-curve 角点)
   或 f(λ) = -L''(λ) (L-curve 曲率)

2. 交叉验证:
       CV(λ) = (1/K) Σ_{k=1}^K χ²_k(λ)

3. L-curve 角点:
       绘制 (log ||T||, log ||R T - O||), 找最大曲率点.

本模块:
  - 黄金分割搜索求最优 λ
  - L-curve 构造与角点检测
  - Tikhonov 正则化解
"""

from __future__ import annotations
from typing import List, Callable, Tuple
import math


# ===========================================================================
#          黄金分割搜索 (移植自 834_opt_golden)
# ===========================================================================

def golden_section_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> Tuple[float, float]:
    """
    黄金分割搜索最小化 f(x) 在 [a, b] 上.

    算法:
        g = (√5 - 1) / 2 ≈ 0.618
        x1 = b - g(b - a),  x2 = a + g(b - a)
        若 f(x1) < f(x2):  新区间 [a, x2]
        否则:               新区间 [x1, b]
        重复直到 |b - a| < tol.

    返回 (x_min, f(x_min)).
    """
    if a >= b:
        raise ValueError(f"黄金分割: a={a} 必须 < b={b}")

    g = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = b - g * (b - a)
    x2 = a + g * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    for _ in range(max_iter):
        if (b - a) < tol:
            break
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - g * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + g * (b - a)
            f2 = f(x2)

    x_min = 0.5 * (a + b)
    return x_min, f(x_min)


# ===========================================================================
#          Tikhonov 正则化 unfolding
# ===========================================================================

def tikhonov_unfold(
    R_dense: List[List[float]],
    O: List[float],
    lam: float,
    L_order: int = 1,
) -> List[float]:
    """
    Tikhonov 正则化解:
        T_λ = argmin_T { ‖R T - O‖^2 + λ ‖L T‖^2 }

    正规方程:
        (R^T R + λ L^T L) T = R^T O

    L 为差分算子:
        L_order = 0: L = I (单位阵, 零阶正则)
        L_order = 1: L = 一阶差分 (T_{i+1} - T_i)
        L_order = 2: L = 二阶差分 (T_{i+1} - 2T_i + T_{i-1})

    解法: 共轭梯度.
    """
    N_true = len(R_dense[0])
    N_rec = len(R_dense)

    # 构造 R^T R
    RTR = [[0.0] * N_true for _ in range(N_true)]
    for i in range(N_true):
        for j in range(N_true):
            s = 0.0
            for k in range(N_rec):
                s += R_dense[k][i] * R_dense[k][j]
            RTR[i][j] = s

    # 构造 L^T L
    LTL = _build_LtL(N_true, L_order)

    # A = RTR + λ LTL
    A = [[RTR[i][j] + lam * LTL[i][j] for j in range(N_true)] for i in range(N_true)]

    # b = R^T O
    RTO = [sum(R_dense[k][i] * O[k] for k in range(N_rec)) for i in range(N_true)]

    # CG 求解
    import response_matrix as rm
    T, _, _ = rm.cg_solve(lambda x: [sum(A[i][j] * x[j] for j in range(N_true)) for i in range(N_true)], RTO, tol=1e-9, max_iter=500)
    return T


def _build_LtL(N: int, order: int) -> List[List[float]]:
    """构造 L^T L (N × N)."""
    if order == 0:
        return [[1.0 if i == j else 0.0 for j in range(N)] for i in range(N)]

    # L 矩阵 (差分)
    if order == 1:
        # L: (N-1) × N,  L[i][j] = δ_{j,i+1} - δ_{j,i}
        L_rows = N - 1
        L = [[0.0] * N for _ in range(L_rows)]
        for i in range(L_rows):
            L[i][i] = -1.0
            L[i][i + 1] = 1.0
    elif order == 2:
        # L: (N-2) × N
        L_rows = N - 2
        L = [[0.0] * N for _ in range(max(L_rows, 0))]
        for i in range(max(L_rows, 0)):
            L[i][i] = 1.0
            L[i][i + 1] = -2.0
            L[i][i + 2] = 1.0
    else:
        raise ValueError(f"L_order={order} 不支持 (>2)")

    if not L:
        return [[0.0] * N for _ in range(N)]

    # LTL = L^T L
    LTL = [[0.0] * N for _ in range(N)]
    M = len(L)
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(M):
                s += L[k][i] * L[k][j]
            LTL[i][j] = s
    return LTL


# ===========================================================================
#          L-curve 与角点检测
# ===========================================================================

def compute_l_curve(
    R_dense: List[List[float]],
    O: List[float],
    lam_values: List[float],
    L_order: int = 1,
) -> Tuple[List[float], List[float], List[float]]:
    """
    计算 L-curve:
        对每个 λ, 解 Tikhonov, 得到:
            η(λ) = ‖R T_λ - O‖   (残差范数)
            ξ(λ) = ‖L T_λ‖        (正则范数)

    返回 (log_xi, log_eta, lam_values).
    """
    log_xi = []
    log_eta = []
    for lam in lam_values:
        T = tikhonov_unfold(R_dense, O, lam, L_order)
        N_true = len(T)
        N_rec = len(R_dense)
        # 残差
        RT = [sum(R_dense[i][j] * T[j] for j in range(N_true)) for i in range(N_rec)]
        res = math.sqrt(sum((RT[i] - O[i]) ** 2 for i in range(N_rec)))
        # 正则
        LTL = _build_LtL(N_true, L_order)
        LT = [sum(LTL[i][j] * T[j] for j in range(N_true)) for i in range(N_true)]
        reg = math.sqrt(sum(T[i] * LT[i] for i in range(N_true)))
        log_xi.append(math.log(max(reg, 1e-300)))
        log_eta.append(math.log(max(res, 1e-300)))
    return log_xi, log_eta, lam_values


def l_curve_corner(
    log_xi: List[float],
    log_eta: List[float],
) -> int:
    """
    L-curve 角点检测 (最大曲率).

    曲率近似 (离散):
        κ_i ≈ |x'' y' - x' y''| / (x'^2 + y'^2)^{3/2}
    其中 x = log_xi, y = log_eta, 导数用中心差分.
    """
    n = len(log_xi)
    if n < 3:
        return 0
    curvatures = []
    for i in range(1, n - 1):
        dx = 0.5 * (log_xi[i + 1] - log_xi[i - 1])
        dy = 0.5 * (log_eta[i + 1] - log_eta[i - 1])
        ddx = log_xi[i + 1] - 2 * log_xi[i] + log_xi[i - 1]
        ddy = log_eta[i + 1] - 2 * log_eta[i] + log_eta[i - 1]
        denom = (dx * dx + dy * dy) ** 1.5
        if denom < 1e-300:
            curvatures.append(0.0)
        else:
            curvatures.append(abs(dx * ddy - dy * ddx) / denom)
    # 最大曲率
    max_k = -1.0
    idx = 1
    for i in range(n - 2):
        if curvatures[i] > max_k:
            max_k = curvatures[i]
            idx = i + 1
    return idx


def optimal_lambda_by_golden(
    R_dense: List[List[float]],
    O: List[float],
    lam_range: Tuple[float, float] = (1e-6, 1.0),
    L_order: int = 1,
) -> float:
    """
    黄金分割搜索最优 λ.

    目标: 最小化 GCV (Generalized Cross-Validation):
        GCV(λ) = ‖(I - H_λ) O‖^2 / (Tr(I - H_λ))^2
    其中 H_λ = R (R^T R + λ L^T L)^{-1} R^T 是 hat 矩阵.

    简化: 使用 L-curve 曲率的负值作为目标 (最大化曲率 = 最小化 -曲率).
    """
    def objective(lam):
        # 计算该 λ 处的 L-curve 曲率 (近似)
        dlam = lam * 0.01
        lam_vals = [max(lam - dlam, 1e-10), lam, lam + dlam]
        log_xi, log_eta, _ = compute_l_curve(R_dense, O, lam_vals, L_order)
        # 中心曲率
        dx = 0.5 * (log_xi[2] - log_xi[0])
        dy = 0.5 * (log_eta[2] - log_eta[0])
        ddx = log_xi[2] - 2 * log_xi[1] + log_xi[0]
        ddy = log_eta[2] - 2 * log_eta[1] + log_eta[0]
        denom = (dx * dx + dy * dy) ** 1.5
        if denom < 1e-300:
            return 0.0
        k = abs(dx * ddy - dy * ddx) / denom
        return -k  # 最小化负曲率 = 最大化曲率

    lam_opt, _ = golden_section_search(
        objective, lam_range[0], lam_range[1],
        tol=1e-3, max_iter=30,
    )
    return lam_opt


__all__ = [
    "golden_section_search",
    "tikhonov_unfold",
    "compute_l_curve",
    "l_curve_corner",
    "optimal_lambda_by_golden",
]
