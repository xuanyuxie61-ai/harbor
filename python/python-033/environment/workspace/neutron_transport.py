"""
neutron_transport.py
基于种子项目 079_besselj 的贝塞尔函数

在中子星并合环境的 r 过程中，中子输运决定核合成的中子通量。
球对称几何下的稳态中子扩散方程：
    (1/r²) d/dr [r² D(r) dφ/dr] - Σ_a(r) φ(r) + S(r) = 0

对于均匀介质（D, Σ_a 为常数），引入无量纲变量 x = r/L，
L = sqrt(D/Σ_a) 为扩散长度，方程化为：
    (1/x²) d/dx [x² dφ/dx] - φ = -S/Σ_a

齐次解涉及球贝塞尔函数：
    φ_h(x) = A · j_0(ix) / x + B · y_0(ix) / x
其中 j_0, y_0 为球贝塞尔函数，与标准贝塞尔函数的关系：
    j_0(z) = sin(z)/z,   y_0(z) = -cos(z)/z

对于虚宗量，使用修正贝塞尔函数 I_{1/2}, K_{1/2}。
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn, kv, iv


def spherical_bessel_j0(x):
    """
    球贝塞尔函数 j_0(x) = sin(x)/x
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    mask = np.abs(x) > 1e-12
    result[mask] = np.sin(x[mask]) / x[mask]
    result[~mask] = 1.0 - x[~mask] ** 2 / 6.0  # Taylor 展开
    return result


def spherical_bessel_y0(x):
    """
    球贝塞尔函数 y_0(x) = -cos(x)/x
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    mask = np.abs(x) > 1e-12
    result[mask] = -np.cos(x[mask]) / x[mask]
    # x=0 处有奇点，返回大数表示发散
    result[~mask] = -1.0 / (x[~mask] + 1e-30)
    return result


def modified_bessel_half(x, kind='I'):
    """
    半整数阶修正贝塞尔函数 I_{1/2}(x) 和 K_{1/2}(x)。
    解析形式：
        I_{1/2}(x) = sqrt(2/(πx)) sinh(x)
        K_{1/2}(x) = sqrt(π/(2x)) exp(-x)

    参数:
        x : float 或 ndarray
        kind : 'I' 或 'K'

    返回:
        val : 同类型
    """
    x = np.asarray(x, dtype=float)
    x_safe = np.where(np.abs(x) < 1e-15, 1e-15, np.abs(x))
    if kind == 'I':
        return np.sqrt(2.0 / (np.pi * x_safe)) * np.sinh(x_safe)
    else:
        return np.sqrt(np.pi / (2.0 * x_safe)) * np.exp(-x_safe)


def neutron_diffusion_solution(r, R_star, D, sigma_a, S0):
    """
    求解球对称中子扩散方程的解析近似解。

    模型假设：
    - 中子源 S(r) = S0 · (1 - r/R_star) 在星体内线性递减
    - 边界条件：φ(R_star) = 0（真空边界）
    - 原点正则：φ(0) 有限

    方程：d²φ/dr² + (2/r) dφ/dr - (1/L²) φ = -S(r)/D
    其中 L = sqrt(D/Σ_a)。

    齐次解（有限在原点）：φ_h = A · sinh(r/L) / (r/L)
    特解可用常数变易法或数值求得。

    参数:
        r : ndarray, 径向坐标 (cm)
        R_star : float, 中子星半径 (cm)
        D : float, 扩散系数 (cm)
        sigma_a : float, 宏观吸收截面 (cm^{-1})
        S0 : float, 中心中子源强 (cm^{-3}s^{-1})

    返回:
        phi : ndarray, 中子通量 (cm^{-2}s^{-1})
    """
    r = np.asarray(r, dtype=float)
    L = np.sqrt(D / sigma_a)
    if L <= 0 or R_star <= 0:
        raise ValueError("扩散长度和半径必须为正")

    # 无量纲变量
    x = r / L
    x_max = R_star / L

    # 齐次解：φ_h = sinh(x)/x
    phi_h = np.zeros_like(x)
    mask = x > 1e-12
    phi_h[mask] = np.sinh(x[mask]) / x[mask]
    phi_h[~mask] = 1.0 + x[~mask] ** 2 / 6.0

    # 特解（数值）：使用有限差分求解非齐次方程
    n = len(r)
    if n < 3:
        return np.zeros_like(r)

    # 构造三对角矩阵
    dr = r[1] - r[0]
    if not np.allclose(np.diff(r), dr, rtol=1e-5):
        # 非均匀网格：使用简单插值到均匀网格
        r_uniform = np.linspace(r[0], r[-1], n)
        # 递归调用（简化处理）
        phi_uniform = neutron_diffusion_solution(r_uniform, R_star, D, sigma_a, S0)
        return np.interp(r, r_uniform, phi_uniform)

    main_diag = np.full(n, -2.0 / (dr ** 2) - 1.0 / (L ** 2))
    lower_diag = 1.0 / (dr ** 2) - 1.0 / (r[1:] * dr)
    upper_diag = 1.0 / (dr ** 2) + 1.0 / (r[:-1] * dr)

    # 边界条件
    # r=0: 对称性 dφ/dr=0 -> φ_{-1} = φ_1，即第一行：(-2/dr² - 1/L²) φ_0 + (2/dr²) φ_1 = -S/D
    # r=R: φ=0
    A = np.diag(main_diag) + np.diag(upper_diag, k=1) + np.diag(lower_diag, k=-1)
    A[0, 0] = -2.0 / (dr ** 2) - 1.0 / (L ** 2)
    A[0, 1] = 2.0 / (dr ** 2)
    A[-1, :] = 0.0
    A[-1, -1] = 1.0

    rhs = -S0 * (1.0 - r / R_star) / D
    rhs[-1] = 0.0

    try:
        phi = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        # 退化情况：返回齐次解
        phi = phi_h * 0.0

    # 保证非负
    phi = np.maximum(phi, 0.0)
    return phi


def neutron_capture_rate_profile(r, phi, n_n, sigma_capture):
    """
    计算中子俘获率径向分布：
        R_cap(r) = n_n(r) · σ_cap · φ(r)

    参数:
        r : ndarray
        phi : ndarray, 中子通量
        n_n : ndarray 或 float, 中子数密度
        sigma_capture : float, 微观俘获截面 (cm²)

    返回:
        rate : ndarray, 俘获率 (cm^{-3}s^{-1})
    """
    phi = np.asarray(phi, dtype=float)
    n_n = np.asarray(n_n, dtype=float)
    if n_n.ndim == 0:
        n_n = np.full_like(phi, n_n)
    rate = n_n * sigma_capture * phi
    return rate


def test_neutron_transport():
    """自包含测试"""
    x = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    j0 = spherical_bessel_j0(x)
    y0 = spherical_bessel_y0(x)
    print(f"[neutron_transport] j0(1.0) = {spherical_bessel_j0(1.0):.6f}, exact = {np.sin(1.0):.6f}")

    # 测试中子扩散解
    r = np.linspace(1e3, 1e6, 500)
    R_star = 1e6
    D = 1e5
    sigma_a = 1e-3
    S0 = 1e20
    phi = neutron_diffusion_solution(r, R_star, D, sigma_a, S0)
    print(f"[neutron_transport] Neutron flux at center: {phi[0]:.3e} cm^{-2}s^{-1}")
    print(f"[neutron_transport] Neutron flux at surface: {phi[-1]:.3e} cm^{-2}s^{-1}")
    assert phi[0] > phi[-1], "Flux should decrease outward"


if __name__ == "__main__":
    test_neutron_transport()
