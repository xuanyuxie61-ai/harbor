"""
compact_finite_difference.py
============================
高阶紧致有限差分 (Compact Finite Difference) 模块。
融合项目: 164_chebyshev1_exactness (多项式精确性检验思想),
         531_hexahedron_jaskowiec_rule (高阶求积规则)

核心方法:
  Lele (1992) 紧致格式 — 三对角隐式差分格式
  第四精度阶: α*f'_{i-1} + f'_i + α*f'_{i+1}
              = a*(f_{i+1}-f_{i-1})/(2h) + b*(f_{i+2}-f_{i-2})/(4h)

  其中 α = 1/4, a = 3/2, b = 0 (Pade' 4阶紧致格式)

  对于二阶导数:
  α*f''_{i-1} + f''_i + α*f''_{i+1}
    = a*(f_{i+1}-2f_i+f_{i-1})/h² + b*(f_{i+2}-2f_i+f_{i-2})/(4h²)

  其中 α = 2/5, a = 6/5, b = 0 (三对角 4阶紧致)

球坐标扩散算子:
  ∇²c = (1/r²)*∂/∂r(r²*∂c/∂r)
       = ∂²c/∂r² + (2/r)*∂c/∂r
"""

import math
import numpy as np
from electrode_constants import COMPACT_FD_ORDER


def compact_fd_1st_derive_matrix(N, h, alpha=None):
    """
    构造紧致有限差分一阶导数矩阵。

    格式: α*f'_{i-1} + f'_i + α*f'_{i+1} = (3/2)*(f_{i+1}-f_{i-1})/(2h)

    这给出三对角系统 A*f' = B*f
    其中 A = tridiag(α, 1, α)
         B = tridiag(-3/(4h), 0, 3/(4h))

    Parameters
    ----------
    N : int
        内部网格点数
    h : float
        网格间距
    alpha : float, optional
        紧致参数 (默认 1/4, 对应 Pade' 4阶)

    Returns
    -------
    A : ndarray, shape (N, N)
        左端三对角矩阵
    B : ndarray, shape (N, N)
        右端差分矩阵
    """
    if alpha is None:
        # Pade' 4阶: α = 1/4
        alpha = 0.25

    # 左端矩阵 A: tridiag(α, 1, α)
    A = np.eye(N)
    for i in range(N):
        if i > 0:
            A[i, i-1] = alpha
        if i < N - 1:
            A[i, i+1] = alpha

    # 右端矩阵 B
    a_coeff = 1.5  # 3/2
    B = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            B[i, i-1] = -a_coeff / (2.0 * h)
        if i < N - 1:
            B[i, i+1] = a_coeff / (2.0 * h)

    return A, B


def compact_fd_2nd_derivative_matrix(N, h, alpha=None):
    """
    构造紧致有限差分二阶导数矩阵。

    格式: α*f''_{i-1} + f''_i + α*f''_{i+1}
         = a*(f_{i+1}-2f_i+f_{i-1})/h²

    其中 α = 2/5, a = 6/5 (三对角 4阶紧致, Lele 1992)

    Fourier 符号分析:
    修正波数 k*h 满足:
    (2/5)*cos(k*h) + 1) * (k*h)² = (6/5)*(2 - 2*cos(k*h))

    Parameters
    ----------
    N : int
        内部网格点数
    h : float
        网格间距
    alpha : float, optional
        紧致参数 (默认 2/5)

    Returns
    -------
    A : ndarray, shape (N, N)
        左端三对角矩阵
    B : ndarray, shape (N, N)
        右端差分矩阵
    """
    if alpha is None:
        alpha = 0.4  # 2/5

    # 左端矩阵 A
    A = np.eye(N)
    for i in range(N):
        if i > 0:
            A[i, i-1] = alpha
        if i < N - 1:
            A[i, i+1] = alpha

    # 右端矩阵 B
    a_coeff = 1.2  # 6/5
    inv_h2 = 1.0 / (h * h)
    B = np.zeros((N, N))
    for i in range(N):
        B[i, i] = -2.0 * a_coeff * inv_h2
        if i > 0:
            B[i, i-1] = a_coeff * inv_h2
        if i < N - 1:
            B[i, i+1] = a_coeff * inv_h2

    return A, B


def compact_fd_2nd_4th_order(N, h):
    """
    五节点 4阶紧致二阶导数格式。

    α*f''_{i-1} + f''_i + α*f''_{i+1}
    = a*(f_{i+1}-2f_i+f_{i-1})/h² + b*(f_{i+2}-2f_i+f_{i-2})/(4h²)

    4阶精度: α = 2/5, a = 6/5, b = 0
    6阶精度: α = 4/11, a = 12/11, b = 3/11

    Parameters
    ----------
    N : int
        内部网格点数
    h : float
        网格间距

    Returns
    -------
    A : ndarray
        左端三对角矩阵
    B : ndarray
        右端五对角矩阵
    """
    alpha = 2.0 / 5.0
    a_coeff = 6.0 / 5.0
    b_coeff = 0.0  # 保持4阶; 设为3/11可升至6阶

    A = np.eye(N)
    B = np.zeros((N, N))
    inv_h2 = 1.0 / (h * h)

    for i in range(N):
        A[i, i] = 1.0
        if i > 0:
            A[i, i-1] = alpha
        if i < N - 1:
            A[i, i+1] = alpha

        # 中心差分
        B[i, i] = -2.0 * (a_coeff + b_coeff) * inv_h2
        if i > 0:
            B[i, i-1] = a_coeff * inv_h2
        if i < N - 1:
            B[i, i+1] = a_coeff * inv_h2
        if i > 1:
            B[i, i-2] = b_coeff * inv_h2 / 4.0
        if i < N - 2:
            B[i, i+2] = b_coeff * inv_h2 / 4.0

    return A, B


def tridiag_solve(a, b, c, d):
    """
    Thomas 算法 (三对角矩阵求解器)。

    求解 A*x = d, 其中 A = tridiag(a, b, c)

    算法复杂度: O(N) (vs. 高斯消去 O(N³))

    稳定性条件: |b_i| > |a_i| + |c_i| (对角占优)

    Parameters
    ----------
    a : ndarray, shape (N,)
        下次对角线 (a[0] 未使用)
    b : ndarray, shape (N,)
        主对角线
    c : ndarray, shape (N,)
        上次对角线 (c[N-1] 未使用)
    d : ndarray, shape (N,)
        右端向量

    Returns
    -------
    x : ndarray, shape (N,)
        解向量
    """
    N = len(b)
    if N == 0:
        return np.array([])

    # 前向消去 (不修改输入)
    c_star = np.zeros(N)
    d_star = np.zeros(N)

    # 检查主元
    if abs(b[0]) < 1e-30:
        raise ValueError("Thomas 算法: 主元 b[0] ≈ 0, 矩阵可能奇异")

    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, N):
        denom = b[i] - a[i] * c_star[i-1]
        if abs(denom) < 1e-30:
            raise ValueError(f"Thomas 算法: 第 {i} 步主元 ≈ 0 ({denom:.2e})")
        c_star[i] = c[i] / denom if i < N - 1 else 0.0
        d_star[i] = (d[i] - a[i] * d_star[i-1]) / denom

    # 回代
    x = np.zeros(N)
    x[N-1] = d_star[N-1]
    for i in range(N - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i+1]

    return x


def spherical_laplacian_compact(c_vec, r_grid, h):
    """
    球坐标 Laplacian 的紧致有限差分实现。

    ∇²c = ∂²c/∂r² + (2/r)*∂c/∂r

    使用紧凑格式:
    1. 先通过紧致格式计算 c' (一阶导数)
    2. 再通过紧致格式计算 c'' (二阶导数)
    3. 组合: ∇²c_i = c''_i + (2/r_i)*c'_i

    球坐标几何奇异性处理:
    - r = 0: 使用 L'Hôpital 规则, ∇²c|_{r=0} = 3*c''|_{r=0}
    - 使用 r² 变换: u = r*c, 则 ∇²c = (1/r)*∂²u/∂r²

    Parameters
    ----------
    c_vec : ndarray, shape (N,)
        浓度场 (内部网格点)
    r_grid : ndarray, shape (N,)
        径向坐标 (内部网格点, r > 0)
    h : float
        网格间距

    Returns
    -------
    lap_c : ndarray, shape (N,)
        ∇²c 的值
    """
    N = len(c_vec)
    if N < 3:
        raise ValueError("紧致格式至少需要 3 个内部网格点")

    # 计算一阶导数
    A1, B1 = compact_fd_1st_derive_matrix(N, h)
    rhs1 = B1 @ c_vec
    # 使用 Thomas 算法求解 (A1 是三对角的)
    a_lower = np.concatenate(([0.0], np.full(N-1, 0.25)))
    b_main = np.ones(N)
    c_upper = np.concatenate((np.full(N-1, 0.25), [0.0]))
    dc_dr = tridiag_solve(a_lower, b_main, c_upper, rhs1)

    # 计算二阶导数
    A2, B2 = compact_fd_2nd_derivative_matrix(N, h)
    rhs2 = B2 @ c_vec
    a2_lower = np.concatenate(([0.0], np.full(N-1, 0.4)))
    b2_main = np.ones(N)
    c2_upper = np.concatenate((np.full(N-1, 0.4), [0.0]))
    d2c_dr2 = tridiag_solve(a2_lower, b2_main, c2_upper, rhs2)

    # 组合球坐标 Laplacian
    lap_c = np.zeros(N)
    for i in range(N):
        r_i = r_grid[i]
        if r_i < 1e-14:
            # r = 0: L'Hôpital 规则, ∇²c = 3*c''
            lap_c[i] = 3.0 * d2c_dr2[i]
        else:
            lap_c[i] = d2c_dr2[i] + 2.0 * dc_dr[i] / r_i

    return lap_c


def modified_wavenumber_compact(kh_values, alpha=0.4):
    """
    紧致格式的修正波数分析。

    对于紧致格式 α*f''_{i-1} + f''_i + α*f''_{i+1} = a*(f_{i+1}-2f_i+f_{i-1})/h²,
    修正波数 (k*h)' 满足:

    (k*h)'² = a * (2 - 2*cos(k*h)) / (1 + 2*α*cos(k*h))

    修正波数越接近真实波数 k*h, 格式精度越高。

    Parameters
    ----------
    kh_values : ndarray
        无量纲波数 k*h 的值
    alpha : float
        紧致参数

    Returns
    -------
    kh_prime_squared : ndarray
        修正波数的平方 (k*h)'²
    kh_exact_squared : ndarray
        精确波数的平方 (k*h)²
    """
    a_coeff = 2.0 * (1.0 - alpha)  # 对于二阶导数, 关系 a = 2(1-α)
    # 确保 a > 0
    a_coeff = max(a_coeff, 1e-10)

    denominator = 1.0 + 2.0 * alpha * np.cos(kh_values)
    numerator = a_coeff * (2.0 - 2.0 * np.cos(kh_values))

    # 防止除零
    safe_denom = np.where(np.abs(denominator) > 1e-14, denominator, 1e-14)
    kh_prime_squared = numerator / safe_denom
    kh_exact_squared = kh_values ** 2

    return kh_prime_squared, kh_exact_squared


def chebyshev_nodes_interval(a, b, n):
    """
    区间 [a, b] 上的 Gauss-Chebyshev 求积节点与权重。
    (项目164: Chebyshev 精确性检验)

    Gauss-Chebyshev (第一类) 求积:
    ∫_{-1}^{1} f(x)/√(1-x²) dx ≈ Σ w_k * f(x_k)

    节点: x_k = cos((2k+1)π/(2n)),  k=0,...,n-1
    权重: w_k = π/n (全部相等)

    精确度: 2n-1 次多项式 (带权函数)

    对于区间 [a, b], 做线性变换: x = (b-a)/2 * t + (a+b)/2

    Parameters
    ----------
    a, b : float
        区间端点
    n : int
        节点数

    Returns
    -------
    x_nodes : ndarray, shape (n,)
        Chebyshev-Gauss 节点
    w_weights : ndarray, shape (n,)
        对应的求积权重 (包含 Jacobian (b-a)/2)
    """
    k = np.arange(n)
    # Chebyshev-Gauss 节点 (标准区间 [-1, 1])
    theta = (2.0 * k + 1.0) * math.pi / (2.0 * n)
    t_nodes = np.cos(theta)

    # 映射到 [a, b]
    x_nodes = 0.5 * (a + b) + 0.5 * (b - a) * t_nodes

    # Gauss-Chebyshev 权重: π/n (对于带权积分)
    # 对于 ∫_a^b f(x) dx = ∫_{-1}^{1} f(x(t)) * (b-a)/2 dt
    # ≈ Σ (π/n) * f(x(t_k)) * sqrt(1-t_k²) * (b-a)/2
    # 但简化: 直接使用 π/n * (b-a)/2 作为不带宽函数的近似
    w_base = math.pi / n * (b - a) / 2.0
    # 补偿权重函数: w_k = w_base * sqrt(1-t_k²) 近似
    w_weights = w_base * np.sqrt(np.maximum(1.0 - t_nodes**2, 1e-14))

    return np.sort(x_nodes), w_weights


def chebyshev_quadrature_exactness(n_quad, degree_max=10):
    """
    检验 Gauss-Chebyshev 求积规则的多项式精确性。
    (项目164)

    Gauss-Chebyshev 第一类求积应精确积分带权函数 1/√(1-x²) 的多项式。
    即: ∫_{-1}^{1} x^n / √(1-x²) dx = Σ w_k * x_k^n

    精确值:
    - n 奇数: 0 (奇函数)
    - n 偶数: π * (n-1)!! / n!!  (双阶乘之比)
    - n=0: π
    - n=2: π/2
    - n=4: 3π/8

    Parameters
    ----------
    n_quad : int
        求积点数
    degree_max : int
        检验的最大多项式次数

    Returns
    -------
    results : list of dict
        每个次数的精确性检验结果
    """
    # 使用标准 Chebyshev-Gauss 节点和权重
    k = np.arange(n_quad)
    theta = (2.0 * k + 1.0) * math.pi / (2.0 * n_quad)
    x_nodes = np.cos(theta)
    w_weights = np.full(n_quad, math.pi / n_quad)

    # 预计算精确值
    def exact_integral(n):
        """∫_{-1}^{1} x^n / √(1-x²) dx 的精确值"""
        if n % 2 == 1:
            return 0.0
        # 偶数: π * (n-1)!! / n!!
        num = 1
        den = 1
        for i in range(1, n + 1, 2):
            num *= i  # 奇数乘积
        for i in range(2, n + 1, 2):
            den *= i  # 偶数乘积
        if n == 0:
            return math.pi
        return math.pi * num / den

    results = []
    for deg in range(degree_max + 1):
        exact = exact_integral(deg)
        approx = np.sum(w_weights * x_nodes ** deg)
        error = abs(approx - exact)

        results.append({
            'degree': deg,
            'exact': exact,
            'approximate': float(approx),
            'absolute_error': float(error),
            'is_exact': error < 1e-10
        })

    return results


def spectral_radius_diffusion_operator(D, N, h):
    """
    扩散算子的谱半径估计。

    对于 ∂c/∂t = D*∇²c 的标准中心差分离散,
    最大特征值 λ_max ≈ -4D/h² * sin²(π/(2N))

    紧致格式的修正波数:
    λ_max_compact ≈ -D*(k*h)'²_max / h²

    Parameters
    ----------
    D : float
        扩散系数 [m²/s]
    N : int
        网格点数
    h : float
        网格间距 [m]

    Returns
    -------
    float
        谱半径 |λ_max|
    """
    # 修正波数的最大频率对应 k*h = π
    kh_max = math.pi
    alpha = 0.4
    a_coeff = 2.0 * (1.0 - alpha)

    denom = 1.0 + 2.0 * alpha * math.cos(kh_max)
    numer = a_coeff * (2.0 - 2.0 * math.cos(kh_max))
    kh_prime_sq_max = numer / max(abs(denom), 1e-14)

    rho = D * kh_prime_sq_max / (h * h)
    return rho
