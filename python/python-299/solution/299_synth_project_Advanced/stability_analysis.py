"""
stability_analysis.py — 高阶有限差分格式稳定性分析
==================================================

种子项目映射:
  - Gershgorin 圆定理: 带状矩阵特征值定位
  - Von Neumann 稳定性分析: Fourier 模式增长因子

物理背景:
  Fokker-Planck 方程 ∂f/∂t = L[f] 的数值稳定性取决于:
  1. 空间离散算子 L_h 的特征值谱 (全部 Re(λ) ≤ 0)
  2. 时间积分方法的稳定域
  3. CFL 条件 (对于显式格式)

核心分析:
  - Gershgorin 圆盘定理: 特征值位于圆盘之并中
  - 谱半径 ρ(L_h) 决定最大允许时间步长
  - Von Neumann 分析: 对均匀网格, 放大因子 g(k) 的模
"""

import numpy as np
from scipy import linalg as la


# ===========================================================================
#  §1  Gershgorin 圆盘分析
# ===========================================================================
def gershgorin_disks(A):
    """计算矩阵 A 的 Gershgorin 圆盘.

    Gershgorin 定理: 矩阵 A 的每个特征值至少位于一个圆盘 D_i 中:
      D_i = {z ∈ C : |z - a_ii| ≤ R_i}
      R_i = Σ_{j≠i} |a_ij|  (第 i 行非对角元素绝对值之和)

    物理意义: 如果所有圆盘都在左半平面 (Re(z) ≤ 0),
    则算子是耗散的, 数值格式稳定.

    Parameters
    ----------
    A : ndarray(n, n)  方阵

    Returns
    -------
    centers : ndarray(n)  圆盘中心 (= 对角元素)
    radii : ndarray(n)    圆盘半径
    disks : list of tuples  (center, radius) 对
    """
    n = A.shape[0]
    centers = np.diag(A).copy()
    radii = np.zeros(n)

    for i in range(n):
        radii[i] = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])

    disks = [(centers[i], radii[i]) for i in range(n)]
    return centers, radii, disks


def gershgorin_stability_check(A):
    """检查 Gershgorin 圆盘是否全部在左半平面.

    Returns
    -------
    is_stable : bool  是否所有圆盘都在 Re(z) ≤ 0 区域
    max_real_upper : float  Re(λ) 的上界
    """
    centers, radii, _ = gershgorin_disks(A)
    # 每个圆盘的最右端
    right_edges = centers.real + radii
    max_real_upper = np.max(right_edges)
    is_stable = max_real_upper <= 1e-10
    return is_stable, max_real_upper


# ===========================================================================
#  §2  Von Neumann 稳定性分析
# ===========================================================================
def von_neumann_analysis(dx, dt, D_func, A_func, x_grid, n_modes=64):
    """对非均匀系数 FP 算子进行局部 Von Neumann 分析.

    标准 Von Neumann 分析 (均匀系数):
      ∂f/∂t = D ∂²f/∂x² + A ∂f/∂x
      f(x,t) = f̂(t) exp(ikx)
      → df̂/dt = (-D k² + iA k) f̂
      → 放大因子: g(k, dt) = exp((-D k² + iA k) dt)

    稳定性条件 (显式 Euler):
      |g| ≤ 1  ⟺  D k² dt ≤ 2
      ⟹  dt ≤ 2 / (D k_max²)
      k_max = π/dx (Nyquist)
      ⟹  dt ≤ 2 dx² / (π² D)

    对于 4 阶差分, 修正的修正波数:
      k_eff = (8 sin(k dx/2) - sin(k dx)) / (6 dx)  (4阶)
      vs k_eff = sin(k dx/2) / (dx/2)  (2阶)

    Parameters
    ----------
    dx : float  网格间距
    dt : float  时间步长
    D_func : callable  D(x) 扩散系数
    A_func : callable  A(x) 摩擦系数
    x_grid : ndarray  空间网格
    n_modes : int  Fourier 模式数

    Returns
    -------
    results : dict  稳定性诊断结果
    """
    k_values = np.linspace(0, np.pi / dx, n_modes)

    # 4 阶修正波数
    k_eff_4th = (8.0 * np.sin(k_values * dx / 2.0)
                  - np.sin(k_values * dx)) / (6.0 * dx)

    # 2 阶修正波数
    k_eff_2nd = np.sin(k_values * dx / 2.0) / (dx / 2.0)

    # 在每个网格点评估系数
    D_vals = D_func(x_grid)
    A_vals = A_func(x_grid)
    D_max = np.max(np.abs(D_vals))
    A_max = np.max(np.abs(A_vals))

    # 放大因子 (最坏情况: D=D_max)
    # g(k) = exp((-D k_eff² + i A k_eff) dt)
    # |g(k)| = exp(-D k_eff² dt)
    # 稳定性: D k_eff² dt ≤ 2 (对显式 Euler)
    # 或 D k_eff² dt ≤ 2 (对 Crank-Nicolson, 无条件稳定)

    amplification_explicit = np.exp(-D_max * k_eff_4th**2 * dt)
    amplification_cn = (1.0 - 0.5 * D_max * k_eff_4th**2 * dt) / \
                        (1.0 + 0.5 * D_max * k_eff_4th**2 * dt)

    # CFL 数
    cfl_diffusion = D_max * dt / dx**2
    cfl_advection = A_max * dt / dx

    # 最大允许 dt (显式)
    dt_max_explicit = 2.0 * dx**2 / (np.pi**2 * max(D_max, 1e-30))

    return {
        "D_max": D_max,
        "A_max": A_max,
        "cfl_diffusion": cfl_diffusion,
        "cfl_advection": cfl_advection,
        "dt_max_explicit": dt_max_explicit,
        "max_amplification_explicit": float(np.max(np.abs(amplification_explicit))),
        "max_amplification_cn": float(np.max(np.abs(amplification_cn))),
        "stable_explicit": bool(np.all(np.abs(amplification_explicit) <= 1.0 + 1e-10)),
        "stable_cn": True,  # Crank-Nicolson 对扩散方程无条件稳定
        "k_values": k_values,
        "k_eff_4th": k_eff_4th,
        "amplification_explicit": amplification_explicit,
    }


# ===========================================================================
#  §3  离散算子特征值分析
# ===========================================================================
def compute_operator_eigenvalues(x, f_background):
    """计算离散 Fokker-Planck 算子的特征值.

    构造矩阵 L_h 使得 (L_h f)_j ≈ C[f](x_j).

    使用 4 阶有限差分 + 给定背景分布 f_background 的系数.

    Parameters
    ----------
    x : ndarray  速度网格
    f_background : ndarray  背景分布 (用于计算系数)

    Returns
    -------
    eigenvalues : ndarray  特征值 (复数)
    eigenvectors : ndarray  特征向量
    spectral_radius : float  谱半径
    """
    from special_functions import chandrasekhar_G
    from fp_collision import compute_cumulative_M

    N = len(x)
    dx = x[1] - x[0] if N > 1 else 1.0

    # 计算系数
    M = compute_cumulative_M(x, f_background)
    G = chandrasekhar_G(x)
    A = np.zeros(N)
    D = np.zeros(N)
    for i in range(N):
        if x[i] > 1e-10:
            A[i] = 2.0 * G[i] * M[i] / x[i]**2
            D[i] = G[i] * M[i] / x[i]

    # 构造差分矩阵 (5 对角)
    # C[f] ≈ (2/x) J + dJ/dx, J = D f' + A f
    # 使用 4 阶差分, 模板宽度 5
    L = np.zeros((N, N))

    for j in range(2, N - 2):
        # ∂f/∂x: 4th order stencil
        # ∂f/∂x |_j = (-f_{j+2} + 8f_{j+1} - 8f_{j-1} + f_{j-2}) / (12 dx)
        df_stencil = [-1.0, 8.0, 0.0, -8.0, 1.0] / (12.0 * dx)
        # indices: j-2, j-1, j, j+1, j+2

        # J_j = D_j df_j + A_j f_j
        # L 的第 j 行:
        # L[j, j-2] = (2/x_j) D_j df_stencil[0] + dJ/dx 贡献
        # ... (完整构造)

        # 简化: 直接构造 C 的矩阵表示
        # C[f]_j = Σ_k L[j,k] f_k
        xj = x[j]
        Dj = D[j]
        Aj = A[j]

        if xj > 1e-10:
            # (2/x) J 部分
            coeff_J = 2.0 / xj
            # J_k 的贡献:
            # J_{j-2} = D_j * df_stencil[0] * f_{j-2}  (注意 D 在 j 处取值)
            # 但实际上 D 也依赖于 f, 所以我们固定 D 和 A

            # dJ/dx 部分 (4 阶)
            # dJ/dx |_j = Σ J_k * d_stencil[k]
            # J_k = D_k * df_k + A_k * f_k
            # 这导致 L 是 A 的复合函数

            # 为简化, 我们只构造扩散部分 D ∂²f/∂x²
            # ∂²f/∂x² = (-f_{j+2} + 16f_{j+1} - 30f_j + 16f_{j-1} - f_{j-2}) / (12 dx²)
            d2_stencil = [-1.0, 16.0, -30.0, 16.0, -1.0] / (12.0 * dx**2)

            for s, coeff in enumerate(d2_stencil):
                k = j - 2 + s
                if 0 <= k < N:
                    L[j, k] += Dj * coeff

            # 对流部分 A ∂f/∂x
            for s, coeff in enumerate(df_stencil):
                k = j - 2 + s
                if 0 <= k < N:
                    L[j, k] += Aj * coeff

            # (2/x) A f 部分
            L[j, j] += coeff_J * Aj

    # 边界条件 (简化: Dirichlet)
    L[0, 0] = -1.0 / dx**2
    L[0, 1] = 1.0 / dx**2
    L[1, 0] = 1.0 / dx**2
    L[1, 1] = -2.0 / dx**2
    L[1, 2] = 1.0 / dx**2
    L[N-1, N-1] = -1.0 / dx**2
    L[N-1, N-2] = 1.0 / dx**2
    L[N-2, N-2] = -2.0 / dx**2
    L[N-2, N-3] = 1.0 / dx**2
    L[N-2, N-1] = 1.0 / dx**2

    # 计算特征值
    try:
        eigenvalues, eigenvectors = la.eig(L)
    except la.LinAlgError:
        eigenvalues = np.zeros(N, dtype=complex)
        eigenvectors = np.eye(N)

    # 谱半径
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    return eigenvalues, eigenvectors, spectral_radius


# ===========================================================================
#  §4  综合稳定性诊断
# ===========================================================================
def full_stability_diagnosis(x, f, dt, dx):
    """执行完整的稳定性诊断.

    Returns
    -------
    report : dict  稳定性报告
    """
    from special_functions import chandrasekhar_G
    from fp_collision import compute_cumulative_M

    N = len(x)
    M = compute_cumulative_M(x, f)
    G = chandrasekhar_G(x)

    D = np.zeros(N)
    A = np.zeros(N)
    for i in range(N):
        if x[i] > 1e-10:
            D[i] = G[i] * M[i] / x[i]
            A[i] = 2.0 * G[i] * M[i] / x[i]**2

    # Von Neumann 分析
    D_func = lambda xx: np.interp(xx, x, D)
    A_func = lambda xx: np.interp(xx, x, A)
    vn = von_neumann_analysis(dx, dt, D_func, A_func, x)

    # 特征值分析 (对小网格)
    if N <= 200:
        eigenvalues, _, spec_rad = compute_operator_eigenvalues(x, f)
        max_real_part = float(np.max(eigenvalues.real))
    else:
        eigenvalues = None
        spec_rad = None
        max_real_part = None

    # Gershgorin 分析
    # 构造一个小矩阵来演示
    n_demo = min(N, 50)
    x_demo = x[:n_demo]
    f_demo = f[:n_demo]
    # 简化: 用三对角近似
    A_tri = np.zeros((n_demo, n_demo))
    for i in range(n_demo):
        A_tri[i, i] = -2.0 * D[i] / dx**2
        if i > 0:
            A_tri[i, i-1] = D[i] / dx**2 - A[i] / (2 * dx)
        if i < n_demo - 1:
            A_tri[i, i+1] = D[i] / dx**2 + A[i] / (2 * dx)
    gersh_stable, gersh_max = gershgorin_stability_check(A_tri)

    report = {
        "von_neumann": {
            "D_max": vn["D_max"],
            "A_max": vn["A_max"],
            "cfl_diffusion": vn["cfl_diffusion"],
            "cfl_advection": vn["cfl_advection"],
            "dt_max_explicit": vn["dt_max_explicit"],
            "stable_explicit": vn["stable_explicit"],
            "stable_cn": vn["stable_cn"],
        },
        "eigenvalue": {
            "spectral_radius": spec_rad,
            "max_real_part": max_real_part,
        },
        "gershgorin": {
            "is_stable": gersh_stable,
            "max_real_upper_bound": gersh_max,
        },
    }

    return report
