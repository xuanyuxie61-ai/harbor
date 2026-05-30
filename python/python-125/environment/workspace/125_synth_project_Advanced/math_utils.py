
import numpy as np
from typing import Tuple






def simpson_integration(f: callable, a: float, b: float, n: int = 1000) -> float:
    if n % 2 == 1:
        n += 1
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x], dtype=np.float64)
    
    integral = y[0] + y[-1]
    integral += 4.0 * np.sum(y[1:-1:2])
    integral += 2.0 * np.sum(y[2:-1:2])
    integral *= h / 3.0
    
    return float(integral)


def trapezoidal_integration(x: np.ndarray, y: np.ndarray) -> float:
    dx = np.diff(x)
    avg_y = (y[:-1] + y[1:]) / 2.0
    return float(np.sum(dx * avg_y))






def associated_legendre_value(n: int, m: int, x: float) -> float:
    x = np.clip(float(x), -1.0, 1.0)
    
    if m > n or m < 0:
        return 0.0
    

    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    
    if n == m:
        return float(pmm)
    

    pmmp1 = x * (2.0 * m + 1.0) * pmm
    
    if n == m + 1:
        return float(pmmp1)
    

    pll = 0.0
    for l in range(m + 2, n + 1):
        pll = (x * (2.0 * l - 1.0) * pmmp1 - (l + m - 1.0) * pmm) / (l - m)
        pmm = pmmp1
        pmmp1 = pll
    
    return float(pll)






def spherical_harmonic_y(l: int, m: int, theta: float, phi: float) -> complex:
    if abs(m) > l:
        return 0.0 + 0.0j
    


    import math
    

    fact_ratio = 1.0
    for i in range(l - abs(m) + 1, l + abs(m) + 1):
        fact_ratio /= i
    
    N = np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi) * fact_ratio)
    

    Plm = associated_legendre_value(l, abs(m), np.cos(theta))
    

    if m >= 0:
        Y = N * Plm * np.exp(1.0j * m * phi)
    else:

        phase = (-1) ** abs(m)
        Y = phase * N * Plm * np.exp(1.0j * m * phi)
    
    return complex(Y)






def matrix_condition_number_estimate(A: np.ndarray) -> float:
    n = A.shape[0]
    

    norm_A = np.max(np.sum(np.abs(A), axis=0))
    

    try:
        s = np.linalg.svd(A, compute_uv=False)
        if s[-1] < 1e-14:
            return 1e16
        return float(s[0] / s[-1])
    except Exception:
        return float(norm_A * 1e6)


def check_diagonal_dominance(A: np.ndarray) -> bool:
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag = np.sum(np.abs(A[i, :])) - diag
        if diag <= off_diag:
            return False
    return True






def relative_error(approx: float, exact: float) -> float:
    if abs(exact) < 1e-14:
        return abs(approx - exact)
    return abs(approx - exact) / abs(exact)


def convergence_order(errors: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    orders = np.zeros(len(errors) - 1)
    for i in range(len(errors) - 1):
        if errors[i] > 1e-14 and errors[i + 1] > 1e-14 and ratios[i] > 1e-14:
            orders[i] = np.log(errors[i + 1] / errors[i]) / np.log(ratios[i])
        else:
            orders[i] = 0.0
    return orders






def log_gamma_lanczos(x: float) -> float:
    if x <= 0:
        return np.inf
    

    c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    
    g = 7.0
    z = x - 1.0
    

    Ag = c[0]
    for k in range(1, len(c)):
        Ag += c[k] / (z + k)
    
    t = z + g + 0.5
    return 0.5 * np.log(2.0 * np.pi) + np.log(Ag) + (z + 0.5) * np.log(t) - t






def erf_approx(x: float) -> float:
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return sign * y
