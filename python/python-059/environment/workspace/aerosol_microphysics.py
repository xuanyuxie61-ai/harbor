
import numpy as np
from math import sqrt, log, exp, pi
from numerical_utils import comb_lexicographic, binomial_coefficient


class AerosolMicrophysicsError(Exception):
    pass


def lognormal_size_distribution(r, N_total, r_median, sigma_g):
    r = np.asarray(r, dtype=np.float64)
    if np.any(r <= 0):
        raise AerosolMicrophysicsError("lognormal_size_distribution: 粒径必须为正")
    if r_median <= 0 or sigma_g <= 1.0:
        raise AerosolMicrophysicsError("lognormal_size_distribution: 参数非法")

    ln_r = np.log(r)
    ln_r_m = np.log(r_median)
    ln_sigma = np.log(sigma_g)

    coeff = N_total / (sqrt(2.0 * pi) * r * ln_sigma)
    exponent = -0.5 * ((ln_r - ln_r_m) / ln_sigma) ** 2
    return coeff * np.exp(exponent)


def multimode_lognormal(r, modes):
    total = np.zeros_like(np.asarray(r), dtype=np.float64)
    for N_total, r_median, sigma_g in modes:
        total += lognormal_size_distribution(r, N_total, r_median, sigma_g)
    return total


def count_mixing_state(m, n, C):
    if m <= 0 or n <= 0 or C <= 0:
        raise AerosolMicrophysicsError("count_mixing_state: 参数必须为正整数")

    a = m % C
    b = n % C
    N = (m * n - a * b) // C

    counts = np.zeros(C, dtype=int)
    for x in range(1, C + 1):
        if x <= a + b - 1 - C:
            counts[x - 1] = N + a + b - C
        elif a + b - 1 - C < x < min(a, b):
            counts[x - 1] = N + x
        elif min(a, b) <= x <= max(a, b):
            counts[x - 1] = N + min(a, b)
        elif max(a, b) < x <= a + b - 1:
            counts[x - 1] = N + a + b - x
        else:
            counts[x - 1] = N

    return counts


def mixing_state_index(counts):
    total = np.sum(counts)
    if total == 0:
        return 0.0
    p_bulk = counts / total

    chi = 1.0 - np.std(p_bulk) * len(p_bulk)
    return float(np.clip(chi, 0.0, 1.0))


def bruggeman_effective_medium(fractions, refractive_indices, tol=1e-12, max_iter=500):
    fractions = np.asarray(fractions, dtype=np.float64)
    m = np.asarray(refractive_indices, dtype=np.complex128)

    if not np.isclose(np.sum(fractions), 1.0):
        raise AerosolMicrophysicsError("bruggeman: 体积分数之和必须等于 1")
    if len(fractions) != len(m):
        raise AerosolMicrophysicsError("bruggeman: 数组长度不匹配")

    m2 = m ** 2

    m_eff2 = np.sum(fractions * m2)

    for _ in range(max_iter):
        sum_term = np.sum(fractions * (m2 - m_eff2) / (m2 + 2.0 * m_eff2))

        denom = np.sum(fractions * (-3.0 * m2) / ((m2 + 2.0 * m_eff2) ** 2))
        if abs(denom) < 1e-30:
            break
        delta = -sum_term / denom
        m_eff2_new = m_eff2 + delta
        if abs(m_eff2_new - m_eff2) < tol:
            m_eff2 = m_eff2_new
            break
        m_eff2 = m_eff2_new

    return np.sqrt(m_eff2)


def select_optimal_size_bins(N_total, r_median, sigma_g, num_bins, r_min=0.001, r_max=10.0):
    n_grid = max(num_bins * 4, 20)
    ln_r_grid = np.linspace(np.log(r_min), np.log(r_max), n_grid)
    r_grid = np.exp(ln_r_grid)


    idx = comb_lexicographic(n_grid, num_bins, 1)

    idx_arr = np.array([i - 1 for i in idx], dtype=int)
    r_bins = r_grid[idx_arr]


    N_bins = np.zeros(num_bins)
    for i in range(num_bins):
        if i == 0:
            r_low = r_min
        else:
            r_low = np.sqrt(r_bins[i - 1] * r_bins[i])
        if i == num_bins - 1:
            r_high = r_max
        else:
            r_high = np.sqrt(r_bins[i] * r_bins[i + 1])


        pts = np.linspace(r_low, r_high, 50)
        vals = lognormal_size_distribution(pts, N_total, r_median, sigma_g)
        N_bins[i] = np.trapezoid(vals, pts)

    return r_bins, N_bins


def extinction_efficiency_small(r, wavelength, m_eff):
    x = 2.0 * pi * r / wavelength
    if x > 0.5:

        return 2.0
    ratio = (m_eff ** 2 - 1.0) / (m_eff ** 2 + 2.0)
    q_scat = (8.0 / 3.0) * (x ** 4) * (abs(ratio) ** 2)
    q_abs = 4.0 * x * np.imag(ratio)
    return float(q_scat + q_abs)
