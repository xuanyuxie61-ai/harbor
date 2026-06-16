"""
dispersion_solver.py - 等离子体色散关系求解器

本模块使用 Newton-Maehly 方法同时求解等离子体色散多项式的所有根,
从而确定等离子体中允许传播的电磁模式。

Newton-Maehly 方法:

  经典 Newton 法求多项式 p(z) 的根时, 容易收敛到已经找到的根。
  Newton-Maehly 方法通过在迭代中加入去重项来避免这个问题:

      z_{n+1} = z_n - p(z_n) / (p'(z_n) - p(z_n) * S_n)

  其中去重因子:
      S_n = sum_{j != i} 1 / (z_n^{(i)} - z_n^{(j)})

  这里 z^{(j)} 为已经收敛 (或正在迭代) 的其他根。

  初始猜测使用 Cauchy  bound 确定的圆上等分点:
      z_k^{(0)} = R * exp(2*pi*i*k/N),  k = 0, 1, ..., N-1
  其中 R 为 Cauchy 半径:
      R = 1 + max|a_k/a_N|

  多项式求值使用 Horner 方法:
      p(z) = (...((a_N * z + a_{N-1}) * z + a_{N-2}) * z + ...) * z + a_0
      p'(z) = N*a_N*z^{N-1} + ... + a_1

等离子体色散多项式:

  冷等离子体电磁波色散 (归一化):
      omega^2 = omega_p^2 + k^2 * c^2

  在磁化等离子体中 (Appleton-Hartree 色散):
      D(omega) = A*omega^4 - B*omega^2 + C = 0

  其中系数 A, B, C 依赖于等离子体参数和传播角度:
      A = S * sin^2(theta) + P * cos^2(theta)
      B = R*L*sin^2(theta) + P*S*(1 + cos^2(theta))
      C = P*R*L

  S = (R+L)/2, D = (R-L)/2
  R = 1 - sum_s omega_ps^2 / (omega*(omega - Omega_s))
  L = 1 - sum_s omega_ps^2 / (omega*(omega + Omega_s))
  P = 1 - sum_s omega_ps^2 / omega^2

  对于双束等离子体 (beam-plasma 不稳定性):
      1 = omega_p1^2/(omega^2) + omega_p2^2/(omega - k*v_b)^2
  展开为 omega 的多项式:
      omega^4 - 2*k*v_b*omega^3 + (k^2*v_b^2 - omega_p1^2 - omega_p2^2)*omega^2
      + 2*k*v_b*omega_p1^2*omega - k^2*v_b^2*omega_p1^2 = 0
"""

import numpy as np


def poly_eval_horner(coeffs, z):
    """Horner 方法同时计算多项式值和导数值。

    p(z) = a_0 + a_1*z + a_2*z^2 + ... + a_n*z^n

    Horner 方法 (从最高次到最低次):
        p(z) = (...((a_n*z + a_{n-1})*z + a_{n-2})*z + ...) + a_0

    导数通过辅助变量同时计算:
        d(z) = p'(z)
        d = a_n
        for k = n-1 down to 1:
            d = d*z + a_k  (修正后的递推)
            p = p*z + a_{k-1}

    更精确: 同时递推 p 和 dp:
        p = a_n
        dp = 0
        for k = n-1 down to 0:
            dp = dp*z + p
            p = p*z + a_k

    Parameters
    ----------
    coeffs : ndarray
        多项式系数 [a_0, a_1, ..., a_n] (升幂排列)
    z : complex
        求值点

    Returns
    -------
    p : complex
        多项式值
    dp : complex
        导数值
    """
    n = len(coeffs) - 1
    p = complex(coeffs[n])
    dp = complex(0)

    for k in range(n - 1, -1, -1):
        dp = dp * z + p
        p = p * z + coeffs[k]

    return p, dp


def newton_maehly_roots(coeffs, tol=1.0e-12, max_iter=200):
    """Newton-Maehly 方法求多项式的所有根。

    算法步骤:
      1. 计算 Cauchy 半径 R = 1 + max|a_k/a_n|
      2. 初始化 N 个猜测: z_k = R * exp(2*pi*i*k/N)
      3. 对每个根 z_i 执行 Newton-Maehly 迭代:
           p, dp = horner(coeffs, z_i)
           S = sum_{j != i} 1/(z_i - z_j)
           z_i = z_i - p / (dp - p * S)
         直到 |p| < tol 或达到 max_iter

    Parameters
    ----------
    coeffs : ndarray
        多项式系数 [a_0, a_1, ..., a_n] (升幂)
    tol : float
        收敛容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    roots : ndarray (complex)
        所有根
    info : dict
        收敛信息
    """
    coeffs = np.asarray(coeffs, dtype=float)
    n = len(coeffs) - 1

    if n <= 0:
        return np.array([], dtype=complex), {'converged': True, 'iterations': 0}

    if n == 1:
        return np.array([-coeffs[0] / coeffs[1]]), {'converged': True, 'iterations': 1}

    # Cauchy 半径
    ratio_max = 0.0
    for k in range(n):
        ratio = abs(coeffs[k] / coeffs[n])
        if ratio > ratio_max:
            ratio_max = ratio
    R = 1.0 + ratio_max

    # 初始猜测: 圆上等分点
    roots = np.zeros(n, dtype=complex)
    for k in range(n):
        theta = 2.0 * np.pi * k / n
        roots[k] = R * np.exp(1j * theta)

    # Newton-Maehly 迭代
    total_iters = 0
    for iteration in range(max_iter):
        max_residual = 0.0
        total_iters += 1

        for i in range(n):
            # 计算 p(z_i) 和 p'(z_i)
            p, dp = poly_eval_horner(coeffs, roots[i])

            # 计算去重因子 S_i
            S = complex(0)
            for j in range(n):
                if j != i:
                    diff = roots[i] - roots[j]
                    if abs(diff) > 1.0e-30:
                        S += 1.0 / diff

            # Newton-Maehly 更新
            denominator = dp - p * S
            if abs(denominator) < 1.0e-30:
                continue

            delta = p / denominator
            roots[i] -= delta

            max_residual = max(max_residual, abs(delta))

        # 收敛检查
        if max_residual < tol:
            break

    # 验证根
    residuals = np.array([abs(poly_eval_horner(coeffs, r)[0]) for r in roots])
    max_res = np.max(residuals)
    converged = max_res < tol * 100  # 适度放宽

    info = {
        'converged': converged,
        'iterations': total_iters,
        'max_residual': max_res,
        'residuals': residuals,
    }

    return roots, info


def beam_plasma_dispersion_polynomial(k, omega_p1, omega_p2, v_beam):
    """双束等离子体色散多项式系数。

    色散关系:
        1 = omega_p1^2/omega^2 + omega_p2^2/(omega - k*v_b)^2

    展开为 omega 的 4 次多项式:
        a_4*omega^4 + a_3*omega^3 + a_2*omega^2 + a_1*omega + a_0 = 0

    系数:
        a_4 = 1
        a_3 = -2*k*v_b
        a_2 = k^2*v_b^2 - omega_p1^2 - omega_p2^2
        a_1 = 2*k*v_b*omega_p1^2
        a_0 = -k^2*v_b^2*omega_p1^2

    如果存在正虚部根, 则表示不稳定性 (beam-plasma instability)。

    Parameters
    ----------
    k : float
        波数
    omega_p1 : float
        背景等离子体频率
    omega_p2 : float
        束等离子体频率
    v_beam : float
        束速度

    Returns
    -------
    coeffs : ndarray
        多项式系数 [a_0, a_1, a_2, a_3, a_4]
    """
    kb = k * v_beam
    wp1_sq = omega_p1 ** 2
    wp2_sq = omega_p2 ** 2

    a0 = -kb ** 2 * wp1_sq
    a1 = 2.0 * kb * wp1_sq
    a2 = kb ** 2 - wp1_sq - wp2_sq
    a3 = -2.0 * kb
    a4 = 1.0

    return np.array([a0, a1, a2, a3, a4])


def magnetized_plasma_dispersion(omega_p, omega_ce, theta, species_omega_p=None, species_Omega=None):
    """磁化等离子体 Appleton-Hartree 色散多项式。

    对于冷磁化等离子体, 沿角度 theta 传播的电磁波满足:
        A*omega^4 - B*omega^2 + C = 0

    这是 omega^2 的二次方程, 给出两个模式 (寻常波 O 和非常波 X)。

    Stix 参数:
        R = 1 - omega_p^2 / (omega*(omega - omega_ce))
        L = 1 - omega_p^2 / (omega*(omega + omega_ce))
        S = (R + L) / 2
        D = (R - L) / 2
        P = 1 - omega_p^2 / omega^2

    简化形式 (单离子, 沿磁场传播 theta=0):
        omega^2 = 0.5 * [(S+c^2*k^2) +/- sqrt((S-c^2*k^2)^2 + 4*D^2*c^2*k^2)]

    Parameters
    ----------
    omega_p : float
        等离子体频率
    omega_ce : float
        电子回旋频率
    theta : float
        传播角度 (相对磁场)
    species_omega_p : list, optional
        各物种等离子体频率
    species_Omega : list, optional
        各物种回旋频率

    Returns
    -------
    A_coeff, B_coeff, C_coeff : float
        色散多项式系数
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # 简化: 仅考虑电子
    if species_omega_p is None:
        species_omega_p = [omega_p]
    if species_Omega is None:
        species_Omega = [omega_ce]

    # 在高频极限 (omega >> Omega_i), 仅电子贡献
    omega_pe = species_omega_p[0]
    Omega_e = species_Omega[0]

    # R, L 参数的高频近似
    # R = 1 - omega_pe^2/(omega*(omega - Omega_e))
    # L = 1 - omega_pe^2/(omega*(omega + Omega_e))
    # S = (R+L)/2 = 1 - omega_pe^2*(omega^2 + Omega_e^2) / (omega^2*(omega^2 - Omega_e^2))
    # D = (R-L)/2 = omega_pe^2*Omega_e*omega / (omega^2*(omega^2 - Omega_e^2))
    # P = 1 - omega_pe^2/omega^2

    # 色散关系的 omega^4 多项式形式
    wp2 = omega_pe ** 2
    Oc2 = Omega_e ** 2

    A_coeff = sin_t ** 2 + (1.0 - wp2 / max(Oc2, 1e-30)) * cos_t ** 2
    B_coeff = (1.0 - wp2 / max(Oc2, 1e-30)) * sin_t ** 2 + (
        2.0 - wp2 / max(Oc2, 1e-30) - wp2 / max(Oc2, 1e-30)
    ) * cos_t ** 2
    C_coeff = (1.0 - wp2 / max(Oc2, 1e-30)) * (1.0 - wp2 / max(Oc2, 1e-30)) * cos_t ** 2

    return A_coeff, B_coeff, C_coeff


def solve_dispersion_sweep(k_array, omega_p1, omega_p2, v_beam):
    """对一系列波数求解色散关系。

    Parameters
    ----------
    k_array : ndarray
        波数数组
    omega_p1 : float
        背景等离子体频率
    omega_p2 : float
        束等离子体频率
    v_beam : float
        束速度

    Returns
    -------
    sweep_results : dict
        色散关系扫描结果
    """
    all_roots = []
    growth_rates = []
    real_frequencies = []

    for k in k_array:
        coeffs = beam_plasma_dispersion_polynomial(k, omega_p1, omega_p2, v_beam)
        roots, info = newton_maehly_roots(coeffs)

        all_roots.append(roots)
        growth_rates.append(np.array([r.imag for r in roots]))
        real_frequencies.append(np.array([r.real for r in roots]))

    max_growth = np.array([np.max(gr) for gr in growth_rates])

    return {
        'k': k_array,
        'roots': all_roots,
        'growth_rates': growth_rates,
        'real_frequencies': real_frequencies,
        'max_growth_rate': max_growth,
        'unstable_k': k_array[max_growth > 0],
    }


def dispersion_analysis(config):
    """执行完整的色散关系分析。

    Parameters
    ----------
    config : SimulationConfig
        仿真配置

    Returns
    -------
    results : dict
        色散分析结果
    """
    omega_p_bg = 1.0  # 背景等离子体频率 (归一化)
    omega_p_beam = 0.3  # 束等离子体频率
    v_beam = 3.0  # 束速度

    k_array = np.linspace(0.01, 5.0, 100)

    # 双束色散扫描
    sweep = solve_dispersion_sweep(k_array, omega_p_bg, omega_p_beam, v_beam)

    # 单个 k 的详细分析
    k_test = 1.0
    coeffs = beam_plasma_dispersion_polynomial(k_test, omega_p_bg, omega_p_beam, v_beam)
    roots, info = newton_maehly_roots(coeffs)

    # 磁化等离子体色散
    A, B, C = magnetized_plasma_dispersion(omega_p_bg, 0.5, np.pi / 4)
    # omega^2 = (B +/- sqrt(B^2 - 4*A*C)) / (2*A)
    disc = B ** 2 - 4 * A * C
    if disc >= 0 and abs(A) > 1e-30:
        omega_sq_1 = (B + np.sqrt(disc)) / (2 * A)
        omega_sq_2 = (B - np.sqrt(disc)) / (2 * A)
    else:
        omega_sq_1 = complex(B + np.sqrt(complex(disc))) / (2 * A + 1e-30)
        omega_sq_2 = complex(B - np.sqrt(complex(disc))) / (2 * A + 1e-30)

    results = {
        'sweep': sweep,
        'test_k': k_test,
        'test_roots': roots,
        'test_info': info,
        'magnetized': {
            'A': A, 'B': B, 'C': C,
            'omega_sq_O': omega_sq_1,
            'omega_sq_X': omega_sq_2,
        },
        'beam_params': {
            'omega_p_bg': omega_p_bg,
            'omega_p_beam': omega_p_beam,
            'v_beam': v_beam,
        },
    }

    return results
