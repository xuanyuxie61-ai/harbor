"""
dispersion_analysis.py - 阿尔芬波色散关系与特殊函数计算模块

本模块计算阿尔芬波的色散关系，包括:
  - 等离子体色散函数 (Plasma dispersion function)
  - 阿尔芬连续谱结构
  - 高能粒子修正的色散关系
  - 间隙模 (TAE, EAE) 的频率和增长率

核心算法融合了以下种子项目:
  - clausen (187): Chebyshev 展开和 Clenshaw 递推
  - ellipse_distance (329): Monte Carlo 轨道采样统计
  - circle_positive_distance (182): 曲线几何上的距离统计

物理背景:
  阿尔芬波在均匀等离子体中的色散关系:
    ω² = k_∥² v_A²

  在非均匀环形等离子体中，由于磁场曲率和梯度，
  不同极向模耦合，形成阿尔芬连续谱:
    ω_A²(r) = (v_A/R₀)² (n - m/q(r))²

  在连续谱间隙处，存在离散本征模:
    TAE (Toroidicity-induced): 在 ω = (n/2)(v_A/qR₀) 附近
    EAE (Ellipticity-induced): 在 ω = (n)(v_A/qR₀) 附近

  含高能粒子的修正色散关系:
    D(ω, k) = 1 + χ_MHD(ω, k) + χ_EP(ω, k) = 0

  其中:
    χ_MHD = -k_∥² v_A² / ω²  (MHD 极化率)
    χ_EP = (ω_*^T / ω) β_fast F(ω, ω_d, ω_b)  (EP 极化率)

作者: DA 博士级合成项目 PROJECT_290
"""

import math

import numpy as np


# ========================================================================
# 等离子体色散函数 (改编自 clausen 项目的 Chebyshev 方法)
# ========================================================================

def plasma_dispersion_function(z, n_terms=20):
    """
    计算等离子体色散函数 Z(ζ) (Fried-Conte 函数)。

    Z(ζ) = (1/√π) ∫_{-∞}^{∞} exp(-t²)/(t - ζ) dt,  Im(ζ) > 0

    解析性质:
      Z(ζ) = -2ζ [1 - 2ζ²/3 + ...]  (|ζ| << 1)
      Z(ζ) ≈ i√π exp(-ζ²) - 1/ζ - 1/(2ζ³) - ...  (|ζ| >> 1)

    数值方法: 使用截断级数展开或连分式。
    改编自 clausen 项目的 Clenshaw 递推思想。

    参数:
      z: complex or ndarray, 参数 ζ
      n_terms: int, 级数截断项数

    返回:
      Z: complex or ndarray, 色散函数值
    """
    z = np.asarray(z, dtype=complex)
    scalar_input = z.ndim == 0
    z = np.atleast_1d(z)

    result = np.zeros_like(z, dtype=complex)

    for idx in np.ndindex(z.shape):
        z_val = z[idx]

        if abs(z_val) < 1e-15:
            result[idx] = complex(0.0, np.sqrt(np.pi))
        elif abs(z_val) < 4.0:
            # 小参数级数展开: Z(ζ) = i√π w(ζ) - 2ζ(1 - 2ζ²/3 + ...)
            # 使用 Faddeeva 函数的级数
            w_val = _faddeeva_series(z_val, n_terms)
            result[idx] = complex(0.0, np.sqrt(np.pi)) * w_val - 2.0 * z_val * _series_h(z_val, n_terms)
        else:
            # 大参数渐近展开: Z(ζ) ≈ i√π exp(-ζ²) - 1/ζ Σ (2n-1)!!/(2ζ²)^n
            asymptotic = 0.0 + 0.0j
            factorial2 = 1.0
            z_sq = z_val ** 2
            for n in range(min(n_terms, 15)):
                if n > 0:
                    factorial2 *= (2 * n - 1)
                term = factorial2 / ((-2.0 * z_sq) ** n)
                asymptotic += term
                if abs(term) < 1e-15 * abs(asymptotic + 1e-30):
                    break

            # 加上指数项 (仅对 Im(ζ) > 0)
            if z_val.imag > 0:
                exp_term = complex(0.0, np.sqrt(np.pi)) * np.exp(-z_val ** 2)
            else:
                exp_term = 0.0 + 0.0j

            result[idx] = exp_term + asymptotic / z_val

    if scalar_input:
        return complex(result[0])
    return result


def _faddeeva_series(z, n_terms):
    """
    Faddeeva 函数 w(z) = exp(-z²) erfc(-iz) 的级数计算。
    """
    # 简化: 使用 Taylor 级数
    result = np.exp(-z ** 2) * (1.0 + 2j / np.sqrt(np.pi) * _dawson_integral(z))
    return result


def _dawson_integral(x, n_terms=20):
    """
    Dawson 积分 F(x) = exp(-x²) ∫₀ˣ exp(t²) dt 的近似。
    """
    if abs(x) < 1e-10:
        return x
    elif abs(x) < 3.0:
        # 级数展开: F(x) = x - 2x³/3 + 4x⁵/15 - ...
        result = 0.0 + 0.0j
        x_power = x
        for k in range(n_terms):
            coeff = (-2.0) ** k / math.factorial(k) / (2 * k + 1)
            result += coeff * x_power
            x_power *= x * x
            if abs(coeff * x_power) < 1e-15:
                break
        return result * x
    else:
        # 渐近: F(x) ≈ 1/(2x)
        return 0.5 / x


def _series_h(z, n_terms):
    """
    H(ζ) = 1 - 2ζ²/3 + 4ζ⁴/15 - ... 的级数求和。
    """
    result = 0.0 + 0.0j
    z_power = 1.0
    for k in range(n_terms):
        coeff = (-1.0) ** k * 2.0 ** k / math.factorial(k) / (2 * k + 1)
        result += coeff * z_power
        z_power *= z ** 2
        if abs(coeff * z_power) < 1e-15:
            break
    return result


# ========================================================================
# Chebyshev 色散关系求解器 (改编自 clausen 的 Clenshaw 递推)
# ========================================================================

def chebyshev_dispersion_solver(coeffs, interval=(-1.0, 1.0), n_roots=10):
    """
    使用 Chebyshev 展开的伴随矩阵法求色散多项式的根。

    色散关系可以化为多项式形式:
      D(ω) = Σ c_k T_k(ω̃) = 0
    其中 T_k 为 Chebyshev 多项式，ω̃ ∈ [-1, 1] 为缩放频率。

    根通过构建伴随矩阵并计算特征值获得 (203_companion_matrix)。

    参数:
      coeffs: array_like, Chebyshev 系数 [c₀, c₁, ..., cₙ]
      interval: tuple, 频率区间 [ω_min, ω_max]
      n_roots: int, 最多返回的根数量

    返回:
      roots: ndarray, 色散关系的根 (复数频率)
    """
    coeffs = np.asarray(coeffs, dtype=float)
    n = len(coeffs) - 1

    if n < 1:
        return np.array([])

    # 构建 Chebyshev 伴随矩阵
    A = np.zeros((n, n), dtype=float)

    # 次对角和超对角 (Chebyshev 三项递推)
    for i in range(n - 1):
        A[i, i + 1] = 0.5
        if i > 0:
            A[i, i - 1] = 0.5
    if n > 1:
        A[0, 1] = 0.5

    # 最后一行由系数确定
    c_n = coeffs[-1]
    if abs(c_n) < 1e-30:
        # 降阶
        coeffs = coeffs[:-1]
        n = len(coeffs) - 1
        if n < 1:
            return np.array([])
        A = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            A[i, i + 1] = 0.5
            if i > 0:
                A[i, i - 1] = 0.5
        if n > 1:
            A[0, 1] = 0.5
        c_n = coeffs[-1]

    for i in range(n):
        A[n - 1, i] = -0.5 * coeffs[i] / c_n
    A[n - 1, n - 1] *= 2.0  # 修正

    # 计算特征值
    eigenvalues = np.linalg.eigvals(A)

    # 从 Chebyshev 域映射回物理域
    omega_min, omega_max = interval
    roots = omega_min + (eigenvalues + 1.0) * 0.5 * (omega_max - omega_min)

    # 选择实部在区间内的根
    valid = (roots.real >= omega_min - 0.1 * abs(omega_max - omega_min)) & \
            (roots.real <= omega_max + 0.1 * abs(omega_max - omega_min))
    roots = roots[valid]

    # 按频率排序
    roots = roots[np.argsort(roots.real)]

    return roots[:n_roots]


# ========================================================================
# 阿尔芬连续谱与间隙模分析
# ========================================================================

def alfven_continuum_gap_structure(q_profile_func, r_array, R0, v_alfven,
                                    n_toroidal=1, m_range=(1, 6)):
    """
    计算阿尔芬连续谱的间隙结构。

    连续谱分支: ω_m(r) = |k_∥,m| v_A = |(n - m/q(r))/R₀| v_A

    间隙出现在相邻分支交叉处:
      ω_m = ω_{m+1} → q = (m + 0.5) / n

    TAE 间隙中心频率:
      ω_TAE = (2m + 1) v_A / (2 n q R₀) ≈ v_A / (2 q R₀)

    参数:
      q_profile_func: callable, q(r) 函数
      r_array: ndarray, 径向坐标
      R0: float, 大半径
      v_alfven: float, 阿尔芬速度
      n_toroidal: int, 环向模数
      m_range: tuple, 极向模数范围

    返回:
      omega_branches: ndarray, shape (n_m, n_r)
      gap_locations: list of dict, 间隙位置信息
    """
    m_min, m_max = m_range
    n_m = m_max - m_min + 1
    n_r = len(r_array)
    omega_branches = np.zeros((n_m, n_r))

    for i_m, m in enumerate(range(m_min, m_max + 1)):
        for j, r in enumerate(r_array):
            q = q_profile_func(r)
            k_parallel = abs(n_toroidal / R0 * (1.0 - m / (n_toroidal * q + 1e-30)))
            omega_branches[i_m, j] = k_parallel * v_alfven

    # 检测间隙位置
    gap_locations = []
    for i_m in range(n_m - 1):
        diff = omega_branches[i_m] - omega_branches[i_m + 1]
        # 找交叉点 (符号变化)
        for j in range(n_r - 1):
            if diff[j] * diff[j + 1] < 0:
                # 线性插值找交叉
                r_cross = r_array[j] + (r_array[j + 1] - r_array[j]) * abs(diff[j]) / (
                    abs(diff[j]) + abs(diff[j + 1]) + 1e-30)
                q_cross = q_profile_func(r_cross)
                omega_cross = 0.5 * (omega_branches[i_m, j] + omega_branches[i_m + 1, j])

                gap_locations.append({
                    'r_norm': r_cross,
                    'q_rational': q_cross,
                    'omega_gap': omega_cross,
                    'm_modes': (m_min + i_m, m_min + i_m + 1),
                    'gap_width': abs(omega_branches[i_m, j] - omega_branches[i_m + 1, j]),
                })

    return omega_branches, gap_locations


def tae_frequency_estimate(r_res, q_res, R0, v_alfven, n_toroidal, epsilon):
    """
    估算 TAE (Toroidicity-induced Alfvén Eigenmode) 频率。

    TAE 频率近似:
      ω_TAE ≈ (2m + 1) / (2qR₀) n v_A

    有限反转比修正:
      ω_TAE ≈ ω_TAE₀ (1 - ε²/4 + ...)

    参数:
      r_res: float, 共振面归一化半径
      q_res: float, 共振面安全因子
      R0: float, 大半径 [m]
      v_alfven: float, 阿尔芬速度 [m/s]
      n_toroidal: int, 环向模数
      epsilon: float, 反转比 a/R₀

    返回:
      omega_tae: float, TAE 频率 [rad/s]
      f_tae: float, TAE 频率 [kHz]
    """
    m_res = int(round(n_toroidal * q_res))
    omega_0 = (2 * m_res + 1) * n_toroidal * v_alfven / (2.0 * q_res * R0)

    # 有限反转比修正
    correction = 1.0 - epsilon ** 2 / 4.0
    omega_tae = omega_0 * correction

    f_tae = omega_tae / (2.0 * np.pi * 1e3)

    return omega_tae, f_tae


def ep_dispersion_correction(omega, beta_fast, v_fast, v_alfven,
                              omega_d, k_parallel, gamma_damping=0.01):
    """
    计算高能粒子对色散关系的修正。

    高能粒子极化率:
      χ_EP = (ω_*^T / ω) β_fast * Z'(ζ_d)

    其中:
      ζ_d = (ω - ω_d) / (k_∥ v_th,fast)
      ω_*^T = (n T_fast) / (e B L_n) (温度梯度漂移频率)

    修正后的色散关系:
      D(ω) = ω² - k_∥² v_A² (1 + χ_EP) = 0

    参数:
      omega: complex, 波频率
      beta_fast: float, 高能粒子 β
      v_fast: float, 高能粒子特征速度
      v_alfven: float, 阿尔芬速度
      omega_d: float, 漂移频率
      k_parallel: float, 平行波数
      gamma_damping: float, 阻尼率

    返回:
      D: complex, 色散函数值
      chi_ep: complex, 高能粒子极化率
    """
    # 热速度
    v_th_fast = v_fast / np.sqrt(2.0)

    # 共振参数
    if abs(k_parallel * v_th_fast) > 1e-30:
        zeta_d = (omega - omega_d) / (k_parallel * v_th_fast)
    else:
        zeta_d = complex(1e10, 0)

    # 等离子体色散函数
    Z_val = plasma_dispersion_function(zeta_d)

    # Z'(ζ) = -2(1 + ζ Z(ζ))
    Z_prime = -2.0 * (1.0 + zeta_d * Z_val)

    # 温度梯度漂移频率 (简化)
    omega_star_T = 0.1 * omega_d  # 简化估计

    # 高能粒子极化率
    if abs(omega) > 1e-30:
        chi_ep = (omega_star_T / omega) * beta_fast * Z_prime
    else:
        chi_ep = 0.0 + 0.0j

    # 色散函数
    D = omega ** 2 - k_parallel ** 2 * v_alfven ** 2 * (1.0 + chi_ep) + \
        complex(0.0, gamma_damping * abs(omega))

    return D, chi_ep


def monte_carlo_orbit_statistics(n_samples, q_safety, R0, a_minor, v_fast, seed=42):
    """
    Monte Carlo 采样高能粒子轨道统计。

    改编自 ellipse_distance (329) 和 circle_positive_distance (182)
    的 Monte Carlo 距离统计方法。

    在环形几何中，高能粒子的漂移轨道为:
      r_d(t) = r₀ + δ_r cos(ω_d t + φ₀)
    其中 δ_r = v_d / ω_d 为漂移轨道宽度。

    统计轨道宽度分布和特征位移。

    参数:
      n_samples: int, 采样数
      q_safety: float, 安全因子
      R0: float, 大半径
      a_minor: float, 小半径
      v_fast: float, 高能粒子速度
      seed: int, 随机种子

    返回:
      stats: dict, 包含均值、方差、偏度等统计量
    """
    rng = np.random.RandomState(seed)

    # 采样初始位置 (均匀在等离子体截面)
    r0 = a_minor * np.sqrt(rng.uniform(0.0, 1.0, n_samples))
    theta0 = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    pitch = rng.uniform(0.0, 1.0, n_samples)  # 投掷角余弦

    # 漂移速度
    epsilon = r0 / R0
    omega_ci = 1.60217663e-19 * 5.0 / 1.67262192e-27  # 简化
    v_d = v_fast ** 2 * epsilon / (omega_ci * R0 + 1e-30)

    # 漂移频率 (简化)
    omega_d = v_d / (r0 + 1e-10) * q_safety

    # 漂移轨道宽度
    delta_r = np.where(
        np.abs(omega_d) > 1e-10,
        v_d / (np.abs(omega_d) + 1e-30),
        a_minor * 0.1
    )

    # 最大位移 = 初始位置 + 漂移宽度
    max_displacement = r0 + delta_r
    min_displacement = np.abs(r0 - delta_r)

    # 成对距离统计 (改编自 ellipse_distance_stats)
    idx1 = rng.randint(0, n_samples, n_samples)
    idx2 = rng.randint(0, n_samples, n_samples)
    distances = np.abs(max_displacement[idx1] - max_displacement[idx2])

    mean_dist = np.mean(distances)
    var_dist = np.var(distances, ddof=1) if n_samples > 1 else 0.0
    skewness = np.mean((distances - mean_dist) ** 3) / (var_dist ** 1.5 + 1e-30) if var_dist > 0 else 0.0

    return {
        'mean_drift_width': np.mean(delta_r),
        'var_drift_width': np.var(delta_r),
        'mean_displacement': np.mean(max_displacement),
        'mean_pairwise_distance': mean_dist,
        'var_pairwise_distance': var_dist,
        'skewness': skewness,
        'max_drift_width': np.max(delta_r),
        'n_samples': n_samples,
    }
