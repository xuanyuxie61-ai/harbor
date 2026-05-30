
import numpy as np
from typing import Callable, Tuple


def build_biharmonic_matrix(n: int) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n):
        if i - 2 >= 0:
            A[i, i - 2] = 1.0
        if i - 1 >= 0:
            A[i, i - 1] = -4.0
        A[i, i] = 6.0
        if i + 1 < n:
            A[i, i + 1] = -4.0
        if i + 2 < n:
            A[i, i + 2] = 1.0
    return A


def solve_biharmonic_fd1d(f_func: Callable,
                          n: int = 65,
                          xlim: Tuple[float, float] = (-1.0, 1.0),
                          bc_displacement: Tuple[float, float] = (0.0, 0.0),
                          bc_slope: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    x_left, x_right = xlim
    x = np.linspace(x_left, x_right, n)
    h = (x_right - x_left) / (n - 1)


    b = np.array([f_func(xi) for xi in x]) * (h ** 4)

    A = build_biharmonic_matrix(n)

    ul, ur = bc_displacement
    upl, upr = bc_slope



    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = ul


    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = ur





    if n > 2:
        A[1, :] = 0.0
        A[1, 0] = 0.0
        A[1, 1] = 7.0
        if n > 3:
            A[1, 2] = -4.0
        if n > 4:
            A[1, 3] = 1.0
        b[1] = b[1] + 2.0 * h * upl - ul



        b[1] = b[1] + 4.0 * ul





    if n > 3:
        A[-2, :] = 0.0
        if n > 5:
            A[-2, -4] = 1.0
        if n > 4:
            A[-2, -3] = -4.0
        A[-2, -2] = 7.0
        A[-2, -1] = 0.0
        b[-2] = b[-2] - 2.0 * h * upr


    try:
        u = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:

        u = np.linalg.lstsq(A, b, rcond=None)[0]

    return x, u


def compute_curvature(u: np.ndarray, h: float) -> np.ndarray:
    n = len(u)
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        kappa[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / (h * h)

    if n > 2:
        kappa[0] = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - u[3]) / (h * h)
        kappa[-1] = (2.0 * u[-1] - 5.0 * u[-2] + 4.0 * u[-3] - u[-4]) / (h * h)
    return kappa


def compute_strain_energy(u: np.ndarray, h: float,
                          young_modulus: float = 1.0,
                          moment_of_inertia: float = 1.0) -> float:
    kappa = compute_curvature(u, h)
    return 0.5 * young_modulus * moment_of_inertia * np.sum(kappa ** 2) * h


def thermal_load_from_gradient(temp_func: Callable,
                               x: np.ndarray,
                               alpha: float = 1.0,
                               young_modulus: float = 1.0,
                               thickness: float = 1.0,
                               poisson_ratio: float = 0.3) -> np.ndarray:
    n = len(x)
    h = x[1] - x[0] if n > 1 else 1.0
    T = np.array([temp_func(xi) for xi in x])


    d2T = np.zeros(n)
    for i in range(1, n - 1):
        d2T[i] = (T[i - 1] - 2.0 * T[i] + T[i + 1]) / (h * h)
    if n > 2:
        d2T[0] = d2T[1]
        d2T[-1] = d2T[-2]

    coeff = alpha * young_modulus / (1.0 - poisson_ratio)


    return -coeff * d2T * (thickness ** 3) / 12.0
