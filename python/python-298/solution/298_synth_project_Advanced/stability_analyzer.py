"""
稳定性分析模块

融合以下种子项目的思想:
1. prime: 素数检测 (用于网格点数选取的优化)
2. ccn_rule: 组合数计算 (用于高阶模板的系数)
3. ode_sweep: 参数扫描 (稳定性边界探索)

PIC 稳定性判据:
- CFL 条件: v_max * dt / dx <= 1
- 等离子体频率约束: omega_pe * dt < 2
- 德拜长度约束: dx < lambda_D * n_ppd
- 粒子数约束: N_p > N_grid * (lambda_D / L)
"""

import numpy as np
from typing import Dict, Tuple, List
try:
    from plasma_constants import (PI, plasma_frequency_electron, debye_length,
                                  thermal_velocity, ELECTRON_MASS, ELECTRON_CHARGE)
except ImportError:
    from plasma_constants import (PI, plasma_frequency_electron, debye_length,
                                  thermal_velocity, ELECTRON_MASS, ELECTRON_CHARGE)


# ============================================================================
# 素数检测 (from prime)
# ============================================================================
def is_prime(n: int) -> bool:
    """试除法素数检测"""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def prime_sieve(limit: int) -> List[int]:
    """
    Eratosthenes 筛法求 [2, limit] 内的所有素数
    """
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(limit + 1) if sieve[i]]


def is_coprime(a: int, b: int) -> bool:
    """互素检测 (用于多重网格层级选择)"""
    while b:
        a, b = b, a % b
    return a == 1


def factorize(n: int) -> Dict[int, int]:
    """整数因子分解 (用于 FFT 友好网格点选择)"""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def is_fft_friendly(n: int) -> bool:
    """
    FFT 友好数: 因子仅含 2, 3, 5
    """
    factors = factorize(n)
    return all(p in [2, 3, 5] for p in factors.keys())


def next_fft_friendly(n: int) -> int:
    """大于等于 n 的最小 FFT 友好数"""
    while not is_fft_friendly(n):
        n += 1
    return n


# ============================================================================
# 组合数计算 (from ccn_rule)
# ============================================================================
def binomial_coefficient(n: int, k: int) -> int:
    """组合数 C(n, k)"""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def central_binomial(n: int) -> int:
    """中心二项式系数 C(2n, n)"""
    return binomial_coefficient(2 * n, n)


def fd_stencil_coeff(order: int, derivative: int, offset: int) -> float:
    """
    计算有限差分模板系数 (Fornberg 算法)

    对于中心差分模板 [-m, ..., m] 上的 d 阶导数
    """
    m = order // 2
    points = list(range(-m, m + 1))
    n = len(points)

    # Fornberg 算法
    c = np.zeros((n, n + 1))
    c[0, 0] = 1.0
    c1 = 1.0
    for i in range(1, n):
        c2 = 1.0
        for j in range(i):
            c3 = points[i] - points[j]
            c2 *= c3
            if i <= derivative:
                c[j, derivative] = 0.0
            for k in range(min(i, derivative), 0, -1):
                c[j, k] = (c1 * k * c[j, k - 1] - (points[i] - points[0]) * c[j, k] +
                            (points[i] - points[0]) * c[j + 1, k] if j + 1 < n else 0)
                c[j, k] /= c3
            c[j, 0] = -c1 * (points[i] - points[0]) * c[j, 0] / c3
        if i <= derivative:
            c[i, derivative] = c1 * derivative * c[i - 1, derivative - 1]
        for k in range(min(i, derivative), 0, -1):
            c[i, k] = c1 * k * c[i - 1, k - 1] / c2
        c[i, 0] = 0.0
        c1 = c2

    # 查找 offset 对应的行
    idx = points.index(offset) if offset in points else -1
    if idx < 0:
        return 0.0
    return c[idx, derivative]


# ============================================================================
# PIC 稳定性判据
# ============================================================================
def cfl_condition(v_max: float, dx: float, dt: float,
                    safety_factor: float = 0.9) -> bool:
    """
    CFL 条件检查:
        v_max * dt / dx < safety_factor
    """
    if dx <= 0:
        raise ValueError("dx 必须为正")
    return v_max * dt / dx < safety_factor


def max_dt_cfl(v_max: float, dx: float, safety_factor: float = 0.9) -> float:
    """CFL 允许的最大时间步"""
    return safety_factor * dx / (v_max + 1e-300)


def plasma_frequency_condition(omega_pe: float, dt: float) -> bool:
    """
    等离子体频率条件:
        omega_pe * dt < 2 (leapfrog 稳定性)
    更严格: omega_pe * dt < 0.2 (避免数值加热)
    """
    return omega_pe * dt < 2.0


def max_dt_plasma_frequency(omega_pe: float, safety: float = 0.2) -> float:
    """等离子体频率约束的最大时间步"""
    return safety / (omega_pe + 1e-300)


def debye_resolution_condition(dx: float, lambda_D: float,
                                  n_ppd: int = 10) -> bool:
    """
    德拜长度分辨率条件:
        dx <= lambda_D / n_ppd
    """
    return dx <= lambda_D / n_ppd


def particle_number_condition(Np: int, N_grid: int, L: float,
                                  lambda_D: float) -> bool:
    """
    粒子数条件 (避免离散噪声主导):
        N_p > N_grid * (lambda_D / L)

    实际上是信噪比要求
    """
    return Np > N_grid * lambda_D / (L + 1e-300)


def check_all_stability_criteria(n_e: float, T_e: float, L: float,
                                    Np: int, N_grid: int,
                                    dx: float, dt: float) -> Dict:
    """
    综合检查所有稳定性条件

    Returns:
        dict with boolean flags for each criterion
    """
    omega_pe = plasma_frequency_electron(n_e)
    lambda_D = debye_length(T_e, n_e)
    v_th = thermal_velocity(T_e, ELECTRON_MASS)
    v_max = 4.0 * v_th  # 4 sigma 截断

    checks = {}
    checks['cfl'] = cfl_condition(v_max, dx, dt)
    checks['plasma_freq'] = plasma_frequency_condition(omega_pe, dt)
    checks['debye_resolution'] = debye_resolution_condition(dx, lambda_D)
    checks['particle_number'] = particle_number_condition(Np, N_grid, L, lambda_D)
    checks['all_passed'] = all(checks.values())

    checks['cfl_number'] = v_max * dt / dx
    checks['omega_pe_dt'] = omega_pe * dt
    checks['dx_over_lambda_D'] = dx / (lambda_D + 1e-300)
    checks['Np_per_cell'] = Np / (N_grid + 1e-300)

    return checks


def von_neumann_analysis(A: np.ndarray) -> Dict:
    """
    Von Neumann 稳定性分析 (矩阵形式)

    计算放大矩阵 A 的谱半径:
        rho(A) = max |eigenvalue|

    rho < 1: 稳定
    rho = 1: 边际稳定
    rho > 1: 不稳定
    """
    eigenvalues = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigenvalues))

    return {
        'spectral_radius': rho,
        'eigenvalues': eigenvalues,
        'is_stable': rho <= 1.0 + 1e-10,
        'growth_rate': np.log(max(rho, 1e-300))
    }


def leapfrog_amplification_matrix(omega_pe: float, k: float,
                                     v_th: float, dx: float, dt: float,
                                     fd_order: int = 2) -> np.ndarray:
    """
    Leapfrog 时间推进的放大矩阵 (2x2)

    对于 phi^{n+1} - 2*phi^n + phi^{n-1} = -omega_pe^2 * dt^2 * phi^n
    写成:
        [phi^{n+1}]   [2 - omega_pe^2*dt^2,  -1] [phi^n    ]
        [phi^n    ] = [1,                     0] [phi^{n-1}]
    """
    from high_order_fd import modified_wavenumber
    km_sq = modified_wavenumber(np.array([k]), dx, fd_order, derivative=2)
    omega_eff_sq = omega_pe**2 + 3.0 * float(np.real(km_sq[0])) * (v_th / np.sqrt(2.0))**2

    A = np.array([
        [2.0 - omega_eff_sq * dt**2, -1.0],
        [1.0, 0.0]
    ])
    return A


def stability_boundary_scan(k_array: np.ndarray, omega_pe: float,
                               v_th: float, dx: float,
                               fd_order: int = 2) -> float:
    """
    扫描 k 值, 找到最大稳定 dt
    """
    dt_max = np.inf
    for k in k_array:
        A = leapfrog_amplification_matrix(omega_pe, k, v_th, dx, 1.0, fd_order)
        # 找 dt 使谱半径 <= 1
        # omega_eff^2 * dt^2 <= 4  => dt <= 2 / omega_eff
        from high_order_fd import modified_wavenumber
        km_sq = modified_wavenumber(np.array([k]), dx, fd_order, derivative=2)
        omega_eff_sq = omega_pe**2 + 3.0 * float(np.real(km_sq[0])) * (v_th / np.sqrt(2.0))**2
        if omega_eff_sq > 0:
            dt_k = 2.0 / np.sqrt(omega_eff_sq)
            dt_max = min(dt_max, dt_k)
    return dt_max
