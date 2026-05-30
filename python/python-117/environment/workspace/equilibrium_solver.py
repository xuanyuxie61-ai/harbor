
import numpy as np
from typing import List, Tuple, Callable


def chebyshev_coefficients_cpr(f_vals: np.ndarray, N: int) -> np.ndarray:
    k = np.arange(N + 1)
    t = k * np.pi / N

    pj = np.ones(N + 1, dtype=np.float64)
    pj[0] = 2.0
    pj[N] = 2.0
    acoeff = np.zeros(N + 1, dtype=np.float64)
    for j in range(N + 1):
        acoeff[j] = np.sum(f_vals * np.cos(j * t) / pj) * (2.0 / (N * pj[j]))
    return acoeff


def chebyshev_companion_matrix_cpr(acoeff: np.ndarray, Nt: int) -> np.ndarray:
    A = np.zeros((Nt, Nt), dtype=np.float64)
    if Nt > 1:
        A[0, 1] = 1.0
        for j in range(1, Nt - 1):
            A[j, j - 1] = 0.5
            A[j, j + 1] = 0.5
        A[Nt - 1, :Nt] = -acoeff[:Nt] / (2.0 * acoeff[Nt])
        A[Nt - 1, Nt - 2] = A[Nt - 1, Nt - 2] + 0.5
    else:
        A[0, 0] = -acoeff[0] / (2.0 * acoeff[1])
    return A


def chebyshev_proxy_rootfinder(f: Callable[[float], float],
                               a: float,
                               b: float,
                               N: int = 64,
                               epscutoff: float = 1e-13,
                               tau: float = 1e-8,
                               sigma: float = 1e-6) -> List[float]:










    raise NotImplementedError("HOLE 3: 请补全 chebyshev_proxy_rootfinder 的 CPR 算法")


def find_equilibrium_distances(force_func: Callable[[float], float],
                                z_min: float = 0.1,
                                z_max: float = 10.0) -> Tuple[List[float], List[str]]:
    roots = chebyshev_proxy_rootfinder(force_func, z_min, z_max, N=80)
    stability = []
    h = 1e-4
    for z_eq in roots:
        df = (force_func(z_eq + h) - force_func(z_eq - h)) / (2.0 * h)
        if df < 0:
            stability.append("stable")
        else:
            stability.append("unstable")
    return roots, stability
