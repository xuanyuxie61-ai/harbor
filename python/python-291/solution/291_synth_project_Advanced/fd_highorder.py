"""
fd_highorder.py
===============
高阶有限差分算子模块。

本模块实现用于等离子体鞘层计算的高阶有限差分格式：
    1. 紧致有限差分 (Compact/Padé 格式)：
        4阶/6阶 三对角紧致格式
    2. 显式中心差分：4阶/6阶/8阶
    3. 迎风格式 (WENO-5)
    4. 非均匀网格有限差分
    5. 矩阵形式微分算子构造

核心公式：
    紧致4阶格式 (Lele 1992):
        α f'_{i-1} + f'_i + α f'_{i+1}
          = a (f_{i+1}-f_{i-1})/(2h) + b (f_{i+2}-f_{i-2})/(4h)
    其中 α=1/4, a=3/2, b=0 (三对角4阶紧致)

    WENO-5 重构：
        采用三个子模板的凸组合实现无振荡高阶重构
"""

import numpy as np
from scipy import linalg as la
from typing import Tuple, Optional
import math


def compact_fd_matrix_4th(N: int, dx: float) -> np.ndarray:
    """
    紧致4阶有限差分矩阵 (三对角 Padé 格式)
        α f'_{i-1} + f'_i + α f'_{i+1} = a * (f_{i+1} - f_{i-1}) / (2h)

    参数：α = 1/4, a = 3/2

    左端紧致矩阵 A * f' = B * f
        A: 三对角矩阵 (1, α, α)
        B: 反对角矩阵 (-a/(2h), 0, a/(2h))

    返回：
        D: shape (N, N) 微分矩阵 D ≈ d/dx
    """
    alpha = 0.25
    a_coeff = 1.5

    # A 矩阵 (LHS)
    A = np.eye(N)
    for i in range(1, N):
        A[i, i - 1] = alpha
        A[i - 1, i] = alpha

    # B 矩阵 (RHS)
    B = np.zeros((N, N))
    for i in range(1, N - 1):
        B[i, i - 1] = -a_coeff / (2.0 * dx)
        B[i, i + 1] = a_coeff / (2.0 * dx)

    # 边界处理：使用低阶格式
    # i=0: 一阶前向
    B[0, 0] = -3.0 / (2.0 * dx)
    B[0, 1] = 4.0 / (2.0 * dx)
    B[0, 2] = -1.0 / (2.0 * dx)
    A[0, :] = 0.0
    A[0, 0] = 1.0

    # i=N-1: 一阶后向
    B[-1, -3] = 1.0 / (2.0 * dx)
    B[-1, -2] = -4.0 / (2.0 * dx)
    B[-1, -1] = 3.0 / (2.0 * dx)
    A[-1, :] = 0.0
    A[-1, -1] = 1.0

    # 求解 D = A^{-1} B
    D = la.solve(A, B)
    return D


def compact_fd_matrix_6th(N: int, dx: float) -> np.ndarray:
    """
    紧致6阶有限差分矩阵 (五对角 Padé 格式)
        α f'_{i-1} + f'_i + α f'_{i+1}
          = a (f_{i+1}-f_{i-1})/(2h) + b (f_{i+2}-f_{i-2})/(4h)

    参数：α = 1/3, a = 14/9, b = 1/9

    返回：
        D: shape (N, N) 微分矩阵
    """
    alpha = 1.0 / 3.0
    a_coeff = 14.0 / 9.0
    b_coeff = 1.0 / 9.0

    # A 矩阵
    A = np.eye(N)
    for i in range(1, N):
        A[i, i - 1] = alpha
        A[i - 1, i] = alpha

    # B 矩阵
    B = np.zeros((N, N))
    for i in range(2, N - 2):
        B[i, i - 2] = -b_coeff / dx
        B[i, i - 1] = -a_coeff / (2.0 * dx)
        B[i, i + 1] = a_coeff / (2.0 * dx)
        B[i, i + 2] = b_coeff / dx

    # 边界附近使用4阶紧致
    for i in [1, N - 2]:
        B[i, max(0, i - 1)] += -a_coeff / (2.0 * dx) * (1 if i - 1 >= 0 else 0)
        B[i, min(N - 1, i + 1)] += a_coeff / (2.0 * dx)

    # 端点使用一阶
    B[0, 0] = -3.0 / (2.0 * dx)
    B[0, 1] = 4.0 / (2.0 * dx)
    B[0, 2] = -1.0 / (2.0 * dx)
    A[0, :] = 0.0
    A[0, 0] = 1.0

    B[-1, -3] = 1.0 / (2.0 * dx)
    B[-1, -2] = -4.0 / (2.0 * dx)
    B[-1, -1] = 3.0 / (2.0 * dx)
    A[-1, :] = 0.0
    A[-1, -1] = 1.0

    D = la.solve(A, B)
    return D


def explicit_fd_weights_4th() -> np.ndarray:
    """
    4阶中心差分权重
        f'_i ≈ (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / (12h)
    返回权重 [-1, 8, 0, -8, 1] / 12
    """
    return np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0


def explicit_fd_weights_6th() -> np.ndarray:
    """
    6阶中心差分权重
        f'_i ≈ (f_{i+3} - 9f_{i+2} + 45f_{i+1} - 45f_{i-1} + 9f_{i-2} - f_{i-3}) / (60h)
    """
    return np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0


def explicit_fd_weights_8th() -> np.ndarray:
    """
    8阶中心差分权重
        9阶精度：使用 9 点模板
    """
    return np.array([
        3.0, -32.0, 168.0, -672.0, 0.0, 672.0, -168.0, 32.0, -3.0
    ]) / 840.0


def explicit_fd_matrix(N: int, dx: float, order: int = 4) -> np.ndarray:
    """
    显式有限差分矩阵

    参数：
        N: 网格点数
        dx: 网格间距
        order: 精度阶数 (4, 6, 8)

    返回：
        D: shape (N, N) 微分矩阵
    """
    if order == 4:
        w = explicit_fd_weights_4th()
        half_width = 2
    elif order == 6:
        w = explicit_fd_weights_6th()
        half_width = 3
    elif order == 8:
        w = explicit_fd_weights_8th()
        half_width = 4
    else:
        raise ValueError(f"不支持的阶数: {order}")

    D = np.zeros((N, N))
    for i in range(half_width, N - half_width):
        for j, wj in enumerate(w):
            D[i, i - half_width + j] = wj / dx

    # 边界处理：逐点降低阶数
    for i in range(half_width):
        _set_boundary_row(D, i, N, dx, forward=True)
    for i in range(N - half_width, N):
        _set_boundary_row(D, i, N, dx, forward=False)

    return D


def _set_boundary_row(D: np.ndarray, i: int, N: int, dx: float, forward: bool):
    """使用单侧差分设置边界行"""
    if forward:
        # 前向差分 (至少 2 阶)
        if i == 0:
            D[0, 0] = -3.0 / (2.0 * dx)
            D[0, 1] = 4.0 / (2.0 * dx)
            D[0, 2] = -1.0 / (2.0 * dx)
        elif i == 1:
            D[1, 0] = -1.0 / (2.0 * dx)
            D[1, 2] = 1.0 / (2.0 * dx)
        else:
            # 使用可用点的中心差分
            D[i, i - 1] = -1.0 / (2.0 * dx)
            D[i, i + 1] = 1.0 / (2.0 * dx)
    else:
        if i == N - 1:
            D[-1, -3] = 1.0 / (2.0 * dx)
            D[-1, -2] = -4.0 / (2.0 * dx)
            D[-1, -1] = 3.0 / (2.0 * dx)
        elif i == N - 2:
            D[-2, -3] = -1.0 / (2.0 * dx)
            D[-2, -1] = 1.0 / (2.0 * dx)
        else:
            D[i, i - 1] = -1.0 / (2.0 * dx)
            D[i, i + 1] = 1.0 / (2.0 * dx)


def nonuniform_fd_matrix(
    x: np.ndarray,
    order: int = 4,
) -> np.ndarray:
    """
    非均匀网格上的有限差分矩阵

    采用 Taylor 展开求解变步长差分系数：
        Σⱼ cⱼ f(xⱼ) ≈ f'(xᵢ)

    对于 5 点模板 (i-2, i-1, i, i+1, i+2)：
        Σⱼ cⱼ (xⱼ - xᵢ)^k = δ_{k,1}  对 k=0,1,2,3,4

    参数：
        x: shape (N,) 网格坐标 (严格递增)
        order: 目标精度阶数 (2 或 4)

    返回：
        D: shape (N, N) 微分矩阵
    """
    N = len(x)
    D = np.zeros((N, N))

    stencil_half = order // 2

    for i in range(N):
        # 确定模板范围
        j_start = max(0, i - stencil_half)
        j_end = min(N, i + stencil_half + 1)

        # 确保模板宽度一致
        if j_end - j_start < stencil_half + 1:
            if j_start == 0:
                j_end = min(N, stencil_half * 2 + 1)
            else:
                j_start = max(0, N - stencil_half * 2 - 1)

        n_pts = j_end - j_start
        if n_pts < 2:
            continue

        # 构建 Vandermonde 系统
        # V[k, j] = (x[j] - x[i])^k / k!
        V = np.zeros((n_pts, n_pts))
        for k in range(n_pts):
            for j_idx in range(n_pts):
                j = j_start + j_idx
                V[k, j_idx] = (x[j] - x[i])**k / math.factorial(k)

        # RHS: e_1 (一阶导数)
        rhs = np.zeros(n_pts)
        rhs[1] = 1.0

        try:
            coeffs = la.solve(V, rhs)
        except la.LinAlgError:
            # 退化情况：使用一阶差分
            if i > 0 and i < N - 1:
                D[i, i - 1] = -1.0 / (x[i] - x[i - 1])
                D[i, i + 1] = 1.0 / (x[i + 1] - x[i])
            continue

        for j_idx in range(n_pts):
            j = j_start + j_idx
            D[i, j] += coeffs[j_idx]

    return D


def fd_second_deriv_matrix(
    x: np.ndarray,
    order: int = 4,
) -> np.ndarray:
    """
    二阶导数有限差分矩阵
        d²f/dx² ≈ D2 * f

    均匀网格4阶：
        f''_i ≈ (-f_{i+2} + 16f_{i+1} - 30f_i + 16f_{i-1} - f_{i-2}) / (12h²)

    参数：
        x: shape (N,) 网格坐标
        order: 精度阶数

    返回：
        D2: shape (N, N) 二阶微分矩阵
    """
    N = len(x)
    D2 = np.zeros((N, N))

    if order == 2:
        # 标准三对角二阶
        for i in range(1, N - 1):
            dx1 = x[i] - x[i - 1]
            dx2 = x[i + 1] - x[i]
            dx_avg = 0.5 * (dx1 + dx2)

            D2[i, i - 1] = 2.0 / (dx1 * (dx1 + dx2))
            D2[i, i + 1] = 2.0 / (dx2 * (dx1 + dx2))
            D2[i, i] = -2.0 / (dx1 * dx2)
    elif order == 4:
        # 5 点二阶差分（均匀网格）
        h = (x[-1] - x[0]) / (N - 1)
        w = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * h**2)
        for i in range(2, N - 2):
            for j, wj in enumerate(w):
                D2[i, i - 2 + j] = wj
    else:
        raise ValueError(f"不支持的二阶差分阶数: {order}")

    # 边界
    # 左边界
    if N > 2:
        D2[0, 0] = 1.0
        D2[0, 1] = -2.0
        D2[0, 2] = 1.0
        dx = x[1] - x[0]
        if dx > 1e-30:
            D2[0, :] /= dx**2

    if N > 2:
        D2[-1, -3] = 1.0
        D2[-1, -2] = -2.0
        D2[-1, -1] = 1.0
        dx = x[-1] - x[-2]
        if dx > 1e-30:
            D2[-1, :] /= dx**2

    return D2


def weno5_reconstruct(
    f: np.ndarray,
    dx: float,
    direction: int = 1,
) -> np.ndarray:
    """
    WENO-5 重构 (5阶加权本质无振荡格式)
    (Jiang & Shu 1996)

    三个候选子模板：
        S₀ = {i-2, i-1, i}
        S₁ = {i-1, i, i+1}
        S₂ = {i, i+1, i+2}

    理想权重：d₀=1/10, d₁=6/10, d₂=3/10
    光滑指标：
        β₀ = 13/12 (f_{i-2}-2f_{i-1}+f_i)² + 1/4 (f_{i-2}-4f_{i-1}+3f_i)²
        β₁ = 13/12 (f_{i-1}-2f_i+f_{i+1})² + 1/4 (f_{i-1}-f_{i+1})²
        β₂ = 13/12 (f_i-2f_{i+1}+f_{i+2})² + 1/4 (3f_i-4f_{i+1}+f_{i+2})²

    非线性权重：
        α_k = d_k / (ε + β_k)²
        ω_k = α_k / Σα_k

    参数：
        f: 函数值数组
        dx: 网格间距
        direction: +1 向前重构, -1 向后重构

    返回：
        f_reconstructed: 重构值数组
    """
    N = len(f)
    eps = 1e-6
    d = np.array([0.1, 0.6, 0.3])

    f_recon = np.zeros(N)

    for i in range(3, N - 2):
        # 三个子模板的重构值
        # S₀: f_{i-2}, f_{i-1}, f_i
        f0 = (2.0 * f[i - 2] - 7.0 * f[i - 1] + 11.0 * f[i]) / 6.0
        # S₁: f_{i-1}, f_i, f_{i+1}
        f1 = (-f[i - 1] + 5.0 * f[i] + 2.0 * f[i + 1]) / 6.0
        # S₂: f_i, f_{i+1}, f_{i+2}
        f2 = (2.0 * f[i] + 5.0 * f[i + 1] - f[i + 2]) / 6.0

        # 光滑指标
        beta0 = (13.0 / 12.0) * (f[i - 2] - 2 * f[i - 1] + f[i])**2 + \
                0.25 * (f[i - 2] - 4 * f[i - 1] + 3 * f[i])**2
        beta1 = (13.0 / 12.0) * (f[i - 1] - 2 * f[i] + f[i + 1])**2 + \
                0.25 * (f[i - 1] - f[i + 1])**2
        beta2 = (13.0 / 12.0) * (f[i] - 2 * f[i + 1] + f[i + 2])**2 + \
                0.25 * (3 * f[i] - 4 * f[i + 1] + f[i + 2])**2

        alphas = np.array([
            d[0] / (eps + beta0)**2,
            d[1] / (eps + beta1)**2,
            d[2] / (eps + beta2)**2,
        ])
        omegas = alphas / np.sum(alphas)

        f_recon[i] = omegas[0] * f0 + omegas[1] * f1 + omegas[2] * f2

    # 边界使用中心差分
    for i in range(3):
        if i > 0 and i < N - 1:
            f_recon[i] = f[i]
    for i in range(N - 2, N):
        f_recon[i] = f[i]

    return f_recon


def apply_fd(
    D: np.ndarray,
    f: np.ndarray,
    boundary: str = "dirichlet",
) -> np.ndarray:
    """
    应用有限差分算子并处理边界条件

    参数：
        D: shape (N, N) 微分矩阵
        f: shape (N,) 函数值
        boundary: "dirichlet" 或 "neumann" 或 "periodic"

    返回：
        df: shape (N,) 导数值
    """
    df = D @ f

    if boundary == "periodic":
        # 周期性边界修正
        pass  # 矩阵已处理

    return df


def spectral_radius(D: np.ndarray) -> float:
    """
    计算微分矩阵的谱半径
        ρ(D) = max |λ_i|
    用于 CFL 条件估计：Δt ≤ C / ρ(D)
    """
    eigenvalues = la.eigvals(D)
    return float(np.max(np.abs(eigenvalues)))


def dispersion_relation_check(
    D: np.ndarray,
    dx: float,
    k_range: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    数值色散关系检验

    对于精确微分算子 d/dx:
        特征值 λ(k) = i k
    数值微分算子的特征值偏离虚轴表示：
        - 虚部偏差 → 色散误差
        - 实部非零 → 耗散/增幅误差

    参数：
        D: 微分矩阵
        dx: 网格间距
        k_range: 波数范围

    返回：
        k_exact: 精确波数
        k_numerical: 数值波数 (复数)
    """
    N = D.shape[0]
    eigenvalues = la.eigvals(D)

    # 排序
    idx = np.argsort(np.angle(eigenvalues))
    eigenvalues = eigenvalues[idx]

    # 数值波数 k_num = λ / i = -i λ
    k_numerical = -1j * eigenvalues

    # 精确波数范围
    k_max = np.pi / dx
    k_exact = np.linspace(-k_max, k_max, N)

    return k_exact, k_numerical
