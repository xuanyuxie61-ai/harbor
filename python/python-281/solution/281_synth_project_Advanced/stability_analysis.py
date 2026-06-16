"""
stability_analysis.py
=====================
数值稳定性分析模块。
融合项目: 122_buckling_spring (参数空间稳定性边界),
         1405_web_matrix (幂法求主特征值),
         938_qr_solve (QR 分解特征值分析)

核心内容:
  1. von Neumann 稳定性分析 (傅里叶模式分析)
  2. CFL 条件自动推导
  3. 参数空间稳定性边界图 (类似屈曲弹簧的 λ-μ 图)
  4. 刚性比分析与自适应时间步控制
  5. Crank-Nicolson 格式的无条件稳定性证明 (数值验证)

关键公式:
  放大因子: g(k) = (1 - 2r*sin²(kh/2)) / (1 + 2r*sin²(kh/2))  [FTCS]
  紧致格式: g(k) = (1 - r*(kh')²/2) / (1 + r*(kh')²/2)      [CN]
  稳定性条件: |g(k)| ≤ 1, ∀ k
"""

import math
import numpy as np
from compact_finite_difference import modified_wavenumber_compact
from electrode_constants import SAFETY_FACTOR


def von_neumann_ftcs(r_values, n_kh=200):
    """
    FTCS (Forward Time Centered Space) 格式的 von Neumann 稳定性分析。

    对于 ∂c/∂t = D*∂²c/∂x², FTCS 格式:
    c^{n+1}_j = c^n_j + r*(c^n_{j+1} - 2c^n_j + c^n_{j-1})
    其中 r = D*Δt/h²

    放大因子: g(θ) = 1 - 4r*sin²(θ/2),  θ = k*h

    稳定性: |g| ≤ 1  ⟺  r ≤ 1/2 (经典 CFL 条件)

    Parameters
    ----------
    r_values : ndarray or list
        扩散数 r = D*Δt/h² 的取值
    n_kh : int
        波数采样点数

    Returns
    -------
    results : dict
        每个 r 值对应的稳定性信息
    """
    theta = np.linspace(0, math.pi, n_kh)
    results = {}

    for r in r_values:
        # FTCS 放大因子
        g = 1.0 - 4.0 * r * np.sin(theta / 2.0) ** 2

        g_max = np.max(np.abs(g))
        g_min = np.min(g)

        results[r] = {
            'amplification_max': float(g_max),
            'amplification_min': float(g_min),
            'is_stable': g_max <= 1.0 + 1e-12 and g_min >= -1.0 - 1e-12,
            'spectral_radius': float(g_max),
            'cfl_ratio': r / 0.5  # 相对于 CFL 极限的比值
        }

    return results


def von_neumann_crank_nicolson(r_values, n_kh=200):
    """
    Crank-Nicolson 格式的 von Neumann 稳定性分析。

    CN 格式: (c^{n+1} - c^n)/Δt = (D/2)*(∇²c^{n+1} + ∇²c^n)

    放大因子: g(θ) = (1 - 2r*sin²(θ/2)) / (1 + 2r*sin²(θ/2))

    理论性质: |g| ≤ 1 对所有 r > 0 成立 → 无条件稳定

    但注意: 对于大 r, g → -1, 产生虚假振荡 (数值色散)

    Parameters
    ----------
    r_values : ndarray or list
        扩散数 r 的取值
    n_kh : int
        波数采样点数

    Returns
    -------
    results : dict
        每个 r 值的稳定性信息
    """
    theta = np.linspace(0, math.pi, n_kh)
    results = {}

    for r in r_values:
        s = np.sin(theta / 2.0) ** 2
        numerator = 1.0 - 2.0 * r * s
        denominator = 1.0 + 2.0 * r * s

        # 防止除零
        safe_denom = np.where(np.abs(denominator) > 1e-14, denominator, 1e-14)
        g = numerator / safe_denom

        g_max = np.max(np.abs(g))

        # 计算振荡指标: g 的负值区域比例
        oscillation_fraction = np.sum(g < 0) / len(g)

        results[r] = {
            'amplification_max': float(g_max),
            'is_stable': g_max <= 1.0 + 1e-12,
            'oscillation_fraction': float(oscillation_fraction),
            'max_negative_g': float(np.min(g)),
            'dissipation_rate': 1.0 - float(np.mean(np.abs(g)))
        }

    return results


def von_neumann_compact_crank_nicolson(r_values, alpha=0.4, n_kh=200):
    """
    紧致有限差分 + Crank-Nicolson 组合格式的 von Neumann 分析。

    紧致格式修正波数 (k*h)' 代替标准波数:
    (k*h)'² = a*(2-2cos(k*h)) / (1 + 2α*cos(k*h))

    CN 放大因子:
    g = (1 - r*(k*h)'²/2) / (1 + r*(k*h)'²/2)

    Parameters
    ----------
    r_values : list
        扩散数 r 的取值
    alpha : float
        紧致参数
    n_kh : int
        波数采样点数

    Returns
    -------
    results : dict
        稳定性分析结果
    """
    kh_values = np.linspace(0, math.pi, n_kh)
    kh_prime_sq, _ = modified_wavenumber_compact(kh_values, alpha)

    results = {}
    for r in r_values:
        g_num = 1.0 - 0.5 * r * kh_prime_sq
        g_den = 1.0 + 0.5 * r * kh_prime_sq
        safe_den = np.where(np.abs(g_den) > 1e-14, g_den, 1e-14)
        g = g_num / safe_den

        g_max = float(np.max(np.abs(g)))
        results[r] = {
            'amplification_max': g_max,
            'is_stable': g_max <= 1.0 + 1e-12,
            'max_modified_wavenumber_sq': float(np.max(kh_prime_sq)),
            'dispersion_error': float(np.max(np.abs(kh_prime_sq - kh_values**2)))
        }

    return results


def stability_boundary_parameter_space(n_lambda=50, n_mu=50):
    """
    参数空间稳定性边界分析。
    融合项目122: 屈曲弹簧的 λ(L,θ), μ(L,θ) 参数图思想。

    此处分析扩散问题中两个关键参数:
    - λ = D*Δt/h² (扩散数, 类似 CFL 数)
    - μ = Δt*∂D/∂c * Δc/h² (非线性强度参数)

    稳定性域: {(λ, μ) : 格式稳定}

    对于非线性扩散 ∂c/∂t = ∂/∂x(D(c)*∂c/∂x),
    有效扩散数 r_eff = λ * (1 + μ*f(k))

    Parameters
    ----------
    n_lambda : int
        λ 方向采样点数
    n_mu : int
        μ 方向采样点数

    Returns
    -------
    Lambda : ndarray, shape (n_lambda, n_mu)
    Mu : ndarray, shape (n_lambda, n_mu)
    stability_map : ndarray, shape (n_lambda, n_mu)
        1 = 稳定, 0 = 不稳定
    boundary_points : list
        稳定性边界上的 (λ, μ) 点
    """
    lambda_vals = np.linspace(0.01, 1.0, n_lambda)
    mu_vals = np.linspace(-0.5, 0.5, n_mu)
    Lambda, Mu = np.meshgrid(lambda_vals, mu_vals, indexing='ij')

    stability_map = np.zeros_like(Lambda)
    boundary_points = []
    n_kh = 100
    theta = np.linspace(0, math.pi, n_kh)

    for i in range(n_lambda):
        for j in range(n_mu):
            lam = Lambda[i, j]
            mu = Mu[i, j]

            # 有效放大因子 (考虑非线性修正)
            # r_eff(θ) = λ * (1 + μ*cos(θ))
            r_eff = lam * (1.0 + mu * np.cos(theta))
            r_eff = np.maximum(r_eff, 0.0)  # 保证非负

            g = 1.0 - 4.0 * r_eff * np.sin(theta / 2.0) ** 2
            g_max = np.max(np.abs(g))

            is_stable = g_max <= 1.0 + 1e-10
            stability_map[i, j] = 1.0 if is_stable else 0.0

    # 提取边界点
    for i in range(1, n_lambda - 1):
        for j in range(1, n_mu - 1):
            if stability_map[i, j] != stability_map[i+1, j]:
                boundary_points.append((float(Lambda[i, j]), float(Mu[i, j])))

    return Lambda, Mu, stability_map, boundary_points


def cfl_condition(D, h, safety=SAFETY_FACTOR, scheme='ftcs'):
    """
    自动计算 CFL 条件限制的最大时间步。

    FTCS: Δt_max = h² / (2*D)
    CN:   无条件稳定, 但 Δt_max ≈ h² / D 避免虚假振荡
    紧致CN: Δt_max 由修正波数决定

    Parameters
    ----------
    D : float
        扩散系数 [m²/s]
    h : float
        网格间距 [m]
    safety : float
        安全因子 (0 < safety ≤ 1)
    scheme : str
        'ftcs', 'cn', 或 'compact_cn'

    Returns
    -------
    dict
        包含 dt_max, cfl_number 等信息
    """
    if D <= 0 or h <= 0:
        raise ValueError(f"扩散系数和网格间距必须为正: D={D}, h={h}")

    if scheme == 'ftcs':
        dt_max = safety * h * h / (2.0 * D)
        cfl = D * dt_max / (h * h)
    elif scheme == 'cn':
        # CN 无条件稳定, 但精度要求限制时间步
        dt_max = safety * h * h / D
        cfl = D * dt_max / (h * h)
    elif scheme == 'compact_cn':
        # 紧致格式的修正 CFL
        alpha = 0.4
        # 最大修正波数平方 ≈ a*4/(1-2α) 在 kh=π
        a_coeff = 2.0 * (1.0 - alpha)
        kh_prime_sq_max = a_coeff * 4.0 / (1.0 - 2.0 * alpha)
        dt_max = safety * 2.0 * h * h / (D * kh_prime_sq_max)
        cfl = D * dt_max / (h * h)
    else:
        raise ValueError(f"未知格式: {scheme}")

    return {
        'dt_max': dt_max,
        'cfl_number': cfl,
        'scheme': scheme,
        'safety_factor': safety,
        'diffusion_number': D * dt_max / (h * h)
    }


def stiffness_ratio(eigenvalues):
    """
    计算刚性比 (stiffness ratio)。

    S = |λ_max| / |λ_min|

    高刚性比 (S >> 1) 表示需要隐式方法或自适应时间步。

    Parameters
    ----------
    eigenvalues : ndarray
        系统矩阵的特征值 (复数)

    Returns
    -------
    dict
        刚性比及相关信息
    """
    abs_eigs = np.abs(eigenvalues)
    nonzero = abs_eigs[abs_eigs > 1e-14]

    if len(nonzero) == 0:
        return {'stiffness_ratio': 0.0, 'lambda_max': 0.0, 'lambda_min': 0.0}

    lam_max = float(np.max(abs_eigs))
    lam_min = float(np.min(nonzero))
    ratio = lam_max / lam_min if lam_min > 0 else float('inf')

    return {
        'stiffness_ratio': ratio,
        'lambda_max_real': float(np.max(np.real(eigenvalues))),
        'lambda_min_real': float(np.min(np.real(eigenvalues))),
        'lambda_max_abs': lam_max,
        'lambda_min_abs': lam_min,
        'is_stiff': ratio > 100.0,
        'spectral_gap': lam_max - lam_min,
        'recommended_scheme': 'implicit' if ratio > 10 else 'explicit'
    }


def power_method_eigenvalue(A, n_iter=1000, tol=1e-12):
    """
    幂法求矩阵主特征值。
    融合项目1405: power_rank 的幂法迭代。

    算法: x_{k+1} = A*x_k / ||A*x_k||
    特征值: λ ≈ x_k^T * A * x_k / (x_k^T * x_k) (Rayleigh 商)

    Parameters
    ----------
    A : ndarray, shape (N, N)
        方阵
    n_iter : int
        最大迭代次数
    tol : float
        收敛容差

    Returns
    -------
    eigenvalue : complex
        主特征值
    eigenvector : ndarray
        对应的特征向量
    n_iterations : int
        实际迭代次数
    """
    N = A.shape[0]
    # 初始向量 (均匀分布)
    x = np.ones(N) / math.sqrt(N)
    eigenvalue_old = 0.0

    for k in range(n_iter):
        y = A @ x
        norm_y = np.linalg.norm(y)

        if norm_y < 1e-30:
            return 0.0, x, k

        x_new = y / norm_y

        # Rayleigh 商
        eigenvalue = float(np.dot(x_new, A @ x_new) / np.dot(x_new, x_new))

        # 收敛检查
        if abs(eigenvalue - eigenvalue_old) < tol * (abs(eigenvalue) + 1e-14):
            return eigenvalue, x_new, k + 1

        x = x_new
        eigenvalue_old = eigenvalue

    return eigenvalue, x, n_iter


def amplification_matrix(D, N, h, dt, scheme='cn'):
    """
    构造数值格式的放大矩阵 G。

    对于 c^{n+1} = G * c^n,
    稳定性 ⟺ ρ(G) ≤ 1 (谱半径 ≤ 1)

    Parameters
    ----------
    D : float
        扩散系数
    N : int
        内部网格点数
    h : float
        网格间距
    dt : float
        时间步长
    scheme : str
        'ftcs' 或 'cn'

    Returns
    -------
    G : ndarray, shape (N, N)
        放大矩阵
    """
    r = D * dt / (h * h)

    # 构造二阶差分矩阵
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0
        if i > 0:
            L[i, i-1] = 1.0
        if i < N - 1:
            L[i, i+1] = 1.0

    if scheme == 'ftcs':
        # G = I + r*L
        G = np.eye(N) + r * L
    elif scheme == 'cn':
        # G = (I - r/2*L)^{-1} * (I + r/2*L)
        lhs = np.eye(N) - 0.5 * r * L
        rhs = np.eye(N) + 0.5 * r * L
        G = np.linalg.solve(lhs, rhs)
    else:
        raise ValueError(f"未知格式: {scheme}")

    return G


def adaptive_timestep(D_current, h, cfl_target=0.4, dt_min=1e-10, dt_max=10.0):
    """
    基于 CFL 条件的自适应时间步计算。

    Δt = safety * h² / (2 * D_max)

    Parameters
    ----------
    D_current : float or ndarray
        当前扩散系数 (可以是场, 取最大值)
    h : float
        网格间距
    cfl_target : float
        目标 CFL 数
    dt_min : float
        最小允许时间步
    dt_max : float
        最大允许时间步

    Returns
    -------
    float
        推荐的时间步长 [s]
    """
    D_max = float(np.max(np.abs(D_current))) if hasattr(D_current, '__len__') else abs(D_current)
    if D_max < 1e-30:
        return dt_max

    dt = cfl_target * h * h / (2.0 * D_max)
    return max(dt_min, min(dt, dt_max))
