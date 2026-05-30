
import numpy as np
from typing import Callable, Tuple


def chebyshev_nodes(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be positive")
    j = np.arange(n + 1)
    return np.cos(np.pi * j / n)


def chebyshev_coefficients(
    f: Callable[[np.ndarray], np.ndarray],
    n: int,
    a: float = -1.0,
    b: float = 1.0,
) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1")
    

    x_std = chebyshev_nodes(n)
    

    x_phys = 0.5 * (a + b) + 0.5 * (b - a) * x_std
    

    fx = f(x_phys)
    


    c = np.zeros(n + 1)
    for k in range(n + 1):
        j = np.arange(n + 1)
        cos_terms = np.cos(np.pi * k * j / n)

        weights = np.ones(n + 1)
        weights[0] = 0.5
        weights[-1] = 0.5
        c[k] = (2.0 / n) * np.sum(weights * fx * cos_terms)
    
    return c


def chebyshev_evaluate(x: np.ndarray, c: np.ndarray, a: float = -1.0, b: float = 1.0) -> np.ndarray:

    t = (2.0 * x - (a + b)) / (b - a)
    t = np.clip(t, -1.0, 1.0)
    
    N = len(c) - 1
    

    b_kp2 = np.zeros_like(t, dtype=float)
    b_kp1 = np.full_like(t, c[N])
    
    for k in range(N - 1, 0, -1):
        b_k = 2.0 * t * b_kp1 - b_kp2 + c[k]
        b_kp2 = b_kp1
        b_kp1 = b_k
    
    result = t * b_kp1 - b_kp2 + c[0]
    return result


def chebyshev_derivative_coefficients(c: np.ndarray) -> np.ndarray:
    N = len(c) - 1
    if N < 1:
        return np.zeros_like(c)
    
    d = np.zeros(N)
    d[N - 1] = 2.0 * N * c[N]
    if N >= 2:
        d[N - 2] = 2.0 * (N - 1) * c[N - 1]
    
    for k in range(N - 3, -1, -1):
        d[k] = d[k + 2] + 2.0 * (k + 1) * c[k + 1]
    



    return d


def chebyshev_derivative(
    x: np.ndarray,
    c: np.ndarray,
    a: float = -1.0,
    b: float = 1.0,
) -> np.ndarray:
    d = chebyshev_derivative_coefficients(c)

    dV_dt = chebyshev_evaluate(x, np.append(d, 0.0), a, b)

    dV_dx = dV_dt * 2.0 / (b - a)
    return dV_dx


def build_chebyshev_pes_approximation(
    mass_number: int,
    charge_number: int,
    beta2_min: float = -0.3,
    beta2_max: float = 2.5,
    n_terms: int = 32,
) -> Tuple[np.ndarray, float, float]:
    from potential_energy_surface import potential_energy
    
    def f(b_arr):
        vals = np.zeros(len(b_arr))
        for i in range(len(b_arr)):
            q = np.array([b_arr[i], 0.0, 0.0, 0.0, 0.0])
            vals[i] = potential_energy(q, mass_number, charge_number)
        return vals
    
    c = chebyshev_coefficients(f, n_terms, beta2_min, beta2_max)
    return c, beta2_min, beta2_max
