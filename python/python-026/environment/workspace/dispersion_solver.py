# -*- coding: utf-8 -*-
"""
dispersion_solver.py

基于 wdk (Weierstrass-Durand-Kerner) 算法的等离子体色散关系求根模块。

原项目 1404_wdk 提供了计算复系数多项式全部根的并行迭代方法。
在激光-等离子体物理中，WDK 方法被用于求解复色散关系:
    D(ω, k) = 0
以获取等离子体波（如 Langmuir 波、离子声波）的复频率 ω = ω_r + iγ。

核心物理模型:
    1. 无碰撞电子等离子体波的色散关系 (Bohm-Gross):
        1 - ω_p^2/ω^2 - 3 k^2 v_te^2/ω^2 = 0
        其中 v_te = sqrt(k_B T_e / m_e) 为电子热速度。

    2. 含阻尼的 Vlasov-Poisson 线性化色散关系:
        ε(ω, k) = 1 - ω_p^2/(k^2 v_te^2) * Z'(ζ_e) = 0
        其中 ζ_e = ω/(sqrt(2) k v_te)，Z' 为等离子体色散函数的导数。

    3. 对于受激拉曼散射 (SRS) 三波耦合:
        (ω_0 - ω_s)^2 = ω_p^2 + c^2 (k_0 - k_s)^2
        ω_s^2 = ω_p^2 + 3 k_s^2 v_te^2
        增长率满足: γ^2 ≈ (v_osc/(2c))^2 ω_p ω_0 (Rosenbluth-Liu)。

数值方法:
    - 将色散关系 D(ω) = 0 转化为多项式求根问题（在合适的近似下）。
    - 使用 WDK 算法同时求出所有复根。
    - 根的选择标准: 物理上合理的根需满足 Im(ω) < 0（ Landau 阻尼）
      或 Im(ω) > 0（不稳定性增长）。
"""

import numpy as np


def poly_eval(coeffs, z):
    """
    使用 Horner 法则求多项式值。

    P(z) = c_0 + c_1*z + c_2*z^2 + ... + c_n*z^n

    Parameters
    ----------
    coeffs : ndarray, shape (n+1,)
        系数数组，c[0] 为常数项。
    z : complex or ndarray
        求值点。

    Returns
    -------
    value : complex or ndarray
        多项式值。
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    z = np.asarray(z, dtype=complex)
    result = np.zeros_like(z, dtype=complex)
    # Horner scheme
    for c in reversed(coeffs):
        result = result * z + c
    return result


def wdk_roots(coeffs, tol=1e-12, max_iter=1000):
    """
    Weierstrass-Durand-Kerner (WDK) 算法求复系数多项式的全部根。

    原项目 1404_wdk 的核心算法:
        1. 利用 Cauchy 界确定初始圆半径 R = 1 + max(|c_i/c_n|)。
        2. 初始猜测为 R 倍单位根。
        3. 迭代: z_i^{new} = z_i^{old} - P(z_i) / Π_{j≠i} (z_i - z_j)

    Parameters
    ----------
    coeffs : ndarray, shape (n+1,)
        多项式系数，c[0] 为常数项，c[n] 为首项系数（必须非零）。
    tol : float, optional
        收敛容差。
    max_iter : int, optional
        最大迭代次数。

    Returns
    -------
    roots : ndarray, shape (n,)
        多项式的 n 个复根。
    converged : bool
        是否收敛。
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    n = len(coeffs) - 1
    if n < 1:
        raise ValueError("多项式次数必须 >= 1。")
    if abs(coeffs[-1]) < 1e-30:
        raise ValueError("首项系数不能为零。")

    # 归一化首项系数
    coeffs = coeffs / coeffs[-1]

    # Cauchy 界
    R = 1.0 + np.max(np.abs(coeffs[:-1]))

    # 初始猜测: R 倍单位根
    theta = np.linspace(0.0, 2.0 * np.pi, n + 1)[:-1]
    roots = R * np.exp(1j * theta)

    for iteration in range(max_iter):
        roots_old = roots.copy()
        for i in range(n):
            zi = roots_old[i]
            denom = 1.0 + 0.0j
            for j in range(n):
                if i != j:
                    diff = zi - roots[j]
                    if abs(diff) < 1e-30:
                        diff = 1e-30 * (1.0 + 1.0j)
                    denom *= diff
            pz = poly_eval(coeffs, zi)
            roots[i] = zi - pz / denom

        max_change = np.max(np.abs(roots - roots_old))
        if max_change < tol:
            return roots, True

    return roots, False


def bohm_gross_polynomial(ne, Te, k, omega0):
    """
    构造 Bohm-Gross 色散关系的多项式形式用于求根。

    冷等离子体+热修正的 Langmuir 波色散:
        ω^2 = ω_p^2 + 3 k^2 v_te^2
    转换为:
        D(ω) = ω^2 - ω_p^2 - 3 k^2 v_te^2 = 0

    为了使用 WDK 方法求解更一般的复频率，我们引入一个
    人工阻尼项并构造四次多项式以近似 Vlasov 色散。

    近似模型 (四次):
        D(ω) ≈ ω^4 - (ω_p^2 + 3k^2v_te^2) ω^2 + i ν ω^3 + ω_p^4/4 = 0
    其中 ν 为等效碰撞频率（小量）。

    Parameters
    ----------
    ne : float
        电子密度 [m^{-3}]。
    Te : float
        电子温度 [K]。
    k : float
        波数 [rad/m]。
    omega0 : float
        激光角频率 [rad/s]（用于设定 ν 的参考尺度）。

    Returns
    -------
    coeffs : ndarray, shape (5,)
        四次多项式系数 [c0, c1, c2, c3, c4]。
    """
    raise NotImplementedError("Hole 2: 请实现 bohm_gross_polynomial 函数体")


def solve_langmuir_wave_dispersion(ne, Te, k, omega0):
    """
    求解 Langmuir 波的复频率。

    使用 WDK 算法求 Bohm-Gross 近似色散关系的根，
    然后选择物理上合理的根。

    Parameters
    ----------
    ne : float
        电子密度 [m^{-3}]。
    Te : float
        电子温度 [K]。
    k : float
        波数 [rad/m]。
    omega0 : float
        激光角频率 [rad/s]。

    Returns
    -------
    omega_r : float
        实频率 [rad/s]。
    gamma : float
        增长率/阻尼率 [rad/s]。
    root_selected : complex
        选定的复根。
    all_roots : ndarray
        所有复根。
    """
    coeffs = bohm_gross_polynomial(ne, Te, k, omega0)
    roots, converged = wdk_roots(coeffs, tol=1e-14, max_iter=2000)

    if not converged:
        # 未收敛时退回到解析近似
        from physics_constants import plasma_frequency, E_MASS, K_BOLTZMANN
        omega_p = plasma_frequency(ne)
        v_te = np.sqrt(K_BOLTZMANN * Te / E_MASS)
        omega_r = np.sqrt(omega_p**2 + 3.0 * k**2 * v_te**2)
        # Landau damping 近似
        k_lambda = k * np.sqrt(K_BOLTZMANN * Te / (ne * (1.602176634e-19)**2 / (8.8541878128e-12 * E_MASS)))
        if k_lambda > 0:
            gamma = -np.sqrt(np.pi / 8.0) * (omega_p / (k_lambda**3)) * np.exp(-1.0 / (2.0 * k_lambda**2) - 1.5)
        else:
            gamma = 0.0
        return omega_r, gamma, omega_r + 1j * gamma, roots

    # 选择物理根: 最接近正实轴且 |Re(ω)| 最大的根
    # Langmuir 波应有 ω_r ≈ ω_p
    from physics_constants import plasma_frequency
    omega_p = plasma_frequency(ne)
    best_idx = -1
    best_score = -np.inf
    for idx, r in enumerate(roots):
        if abs(r) < 1e-10:
            continue
        # 分数偏差
        score = -abs(abs(r.real) - omega_p) / max(omega_p, 1.0)
        # 偏向正频率
        if r.real > 0:
            score += 0.1
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        best_idx = np.argmax(np.abs(roots))

    root_selected = roots[best_idx]
    omega_r = abs(root_selected.real)
    gamma = root_selected.imag

    return omega_r, gamma, root_selected, roots


def srs_three_wave_coupling_roots(ne, Te, k_s, omega0, E0):
    """
    求解 SRS 三波耦合的复频率（扩展色散关系）。

    SRS 过程:
        ω_0 = ω_s + ω_p
        k_0 = k_s + k_p

    在 Rosenbluth-Liu 框架下，将三波耦合系统转化为关于
    散射波频率 ω_s 的特征多项式:
        D(ω_s) = (ω_s^2 - ω_ps^2 - 3k_s^2v_te^2) * ((ω_0-ω_s)^2 - ω_p^2 - c^2(k_0-k_s)^2) - γ_0^4 = 0

    其中 γ_0 = (v_osc/(2c)) sqrt(ω_p ω_0) 为 SRS 增长率。

    这里构造一个八次多项式近似以捕捉不稳定根。

    Parameters
    ----------
    ne : float
        电子密度 [m^{-3}]。
    Te : float
        电子温度 [K]。
    k_s : float
        散射波波数 [rad/m]。
    omega0 : float
        泵浦激光角频率 [rad/s]。
    E0 : float
        泵浦激光电场振幅 [V/m]。

    Returns
    -------
    omega_s_r : float
        散射波实频率 [rad/s]。
    gamma_srs : float
        SRS 增长率 [rad/s]。
    all_roots : ndarray
        所有复根。
    """
    from physics_constants import (plasma_frequency, C_LIGHT, E_MASS,
                                    K_BOLTZMANN, quiver_velocity, srs_growth_rate)

    omega_p = plasma_frequency(ne)
    v_te = np.sqrt(K_BOLTZMANN * Te / E_MASS)
    v_osc = quiver_velocity(E0, omega0)
    gamma_0 = srs_growth_rate(ne, E0, omega0)

    # 泵浦波数 (真空近似)
    k_0 = omega0 / C_LIGHT
    k_p = k_0 - k_s

    # 定义系数
    A = omega_p**2 + 3.0 * k_s**2 * v_te**2
    B = omega_p**2 + C_LIGHT**2 * k_p**2

    # 将 D(ω_s) = (ω_s^2 - A) * ((ω_0 - ω_s)^2 - B) - γ_0^4 = 0 展开
    # 令 x = ω_s
    # (x^2 - A) * ((ω0 - x)^2 - B) - γ0^4
    # = (x^2 - A) * (ω0^2 - 2ω0 x + x^2 - B) - γ0^4
    # = x^4 - 2ω0 x^3 + (ω0^2 - B - A) x^2 + 2ω0 A x - A(ω0^2 - B) - γ0^4
    c4 = 1.0 + 0.0j
    c3 = -2.0 * omega0 + 0.0j
    c2 = (omega0**2 - B - A) + 0.0j
    c1 = 2.0 * omega0 * A + 0.0j
    c0 = -A * (omega0**2 - B) - gamma_0**4 + 0.0j

    coeffs = np.array([c0, c1, c2, c3, c4], dtype=complex)
    roots, converged = wdk_roots(coeffs, tol=1e-12, max_iter=2000)

    # 选择物理根: 最接近实轴且 ω_s ≈ ω_0 - ω_p 的根
    target = omega0 - omega_p
    best_idx = -1
    best_score = -np.inf
    for idx, r in enumerate(roots):
        score = -abs(abs(r.real) - target) / max(target, 1.0) - abs(r.imag) / max(omega_p, 1.0)
        if r.real > 0:
            score += 0.05
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        best_idx = 0

    root_sel = roots[best_idx]
    omega_s_r = abs(root_sel.real)
    gamma_srs = root_sel.imag

    return omega_s_r, gamma_srs, roots


def plasma_dispersion_function_derivative(zeta, n_terms=50):
    """
    计算等离子体色散函数 Z(ζ) 的导数 Z'(ζ) 的级数展开。

    级数展开 (Fried-Conte):
        Z'(ζ) = -2 * Σ_{n=0}^∞ (-2ζ)^n / (sqrt(π) * (2n+1)!!)  + i * sqrt(π) * ζ * exp(-ζ^2)
    （此近似适用于 |Im(ζ)| > 0 的情况）

    更实用的展开（大 |ζ| 近似）:
        Z'(ζ) ≈ -1/ζ^2 - 3/(2ζ^4) - 15/(4ζ^6) - ...

    Parameters
    ----------
    zeta : complex
        自变量。
    n_terms : int, optional
        级数项数。

    Returns
    -------
    Zp : complex
        Z'(ζ) 的近似值。
    """
    zeta = complex(zeta)
    if abs(zeta) < 0.1:
        # 小参数展开
        Zp = -2.0 + 0.0j
        term = 1.0
        for n in range(1, n_terms):
            term *= (-2.0 * zeta) / (2.0 * n + 1.0)
            Zp += term
        Zp = Zp / np.sqrt(np.pi)
        Zp += 1j * np.sqrt(np.pi) * zeta * np.exp(-zeta**2)
    else:
        # 大参数渐近展开
        Zp = 0.0 + 0.0j
        for n in range(1, n_terms + 1):
            coeff = 1.0
            for m in range(n):
                coeff *= (2.0 * m + 1.0) / 2.0
            Zp += coeff / (zeta ** (2.0 * n))
        Zp = -Zp
    return Zp
