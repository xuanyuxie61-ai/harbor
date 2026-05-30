
import numpy as np
from typing import Callable, Tuple, List
from utils import validate_array_1d, tridiagonal_solve


def gegenbauer_weight(x: np.ndarray, lam: float) -> np.ndarray:
    x = validate_array_1d(x, "x")
    if lam <= -0.5:
        raise ValueError("lambda must be > -0.5")

    val = 1.0 - x ** 2
    val = np.where(val < 1e-14, 1e-14, val)
    return val ** (lam - 0.5)


def chebyshev_even_coefficients(n: int, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")
    s = n // 2
    sigma = n % 2
    a2 = np.zeros(s + 1, dtype=float)
    for r in range(0, 2 * s + 1, 2):
        rh = r // 2
        total = 0.5 * float(f(np.array([1.0]))[0])
        j_pts = np.arange(1, n)
        xj = np.cos(j_pts * np.pi / n)
        fj = f(xj)
        total += np.sum(fj * np.cos(r * j_pts * np.pi / n))
        total += 0.5 * ((-1.0) ** r) * float(f(np.array([-1.0]))[0])
        a2[rh] = (2.0 / n) * total
    return a2


def gegenbauer_cc_quadrature(n: int, lam: float, f: Callable[[np.ndarray], np.ndarray]) -> float:
    if n < 1:
        raise ValueError("n must be >= 1")
    if lam <= -0.5:
        raise ValueError("lambda must be > -0.5")
    a2 = chebyshev_even_coefficients(n, f)
    s = n // 2
    sigma = n % 2
    rh = s
    u = 0.5 * (sigma + 1.0) * a2[rh]
    for rh in range(s - 1, 0, -1):
        u = ((rh - lam) / (rh + lam + 1.0)) * u + a2[rh]
    u = -lam * u / (lam + 1.0) + 0.5 * a2[0]

    from math import lgamma, exp, sqrt, pi
    gamma_ratio = exp(lgamma(lam + 0.5) + 0.5 * np.log(np.pi) - lgamma(lam + 1.0))
    value = gamma_ratio * u
    return float(value)


def divided_differences(xd: np.ndarray, yd: np.ndarray) -> np.ndarray:
    xd = validate_array_1d(xd, "xd")
    yd = validate_array_1d(yd, "yd")
    n = xd.size
    if yd.size != n:
        raise ValueError("xd and yd must have same length")
    dd = np.array(yd, dtype=float)
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            denom = xd[j] - xd[j - i]
            if abs(denom) < 1e-14:
                denom = 1e-14
            dd[j] = (dd[j] - dd[j - 1]) / denom
    return dd


def newton_interpolation_eval(xd: np.ndarray, dd: np.ndarray, xp: np.ndarray) -> np.ndarray:
    xd = validate_array_1d(xd, "xd")
    dd = validate_array_1d(dd, "dd")
    xp = np.asarray(xp, dtype=float)
    nd = xd.size
    if dd.size != nd:
        raise ValueError("dd must have same length as xd")
    yp = dd[nd - 1] * np.ones_like(xp)
    for i in range(nd - 2, -1, -1):
        yp = dd[i] + (xp - xd[i]) * yp
    return yp


def interpolate_potential(
    x_nodes: np.ndarray, V_nodes: np.ndarray, x_fine: np.ndarray
) -> np.ndarray:
    x_nodes = validate_array_1d(x_nodes, "x_nodes")
    V_nodes = validate_array_1d(V_nodes, "V_nodes")
    dd = divided_differences(x_nodes, V_nodes)
    return newton_interpolation_eval(x_nodes, dd, x_fine)


def radial_wavefunction_shooting(
    r_grid: np.ndarray,
    V: np.ndarray,
    m_star_ratio: float,
    E_guess: float,
    l: int = 0,
) -> Tuple[np.ndarray, float]:
    r_grid = validate_array_1d(r_grid, "r_grid")
    V = validate_array_1d(V, "V")
    if r_grid.size != V.size:
        raise ValueError("r_grid and V must have same size")
    n = r_grid.size
    if n < 4:
        raise ValueError("Need at least 4 grid points")
    h = float(r_grid[1] - r_grid[0])
    if abs(h) < 1e-20:
        raise ValueError("Grid spacing too small")

    H_BAR = 1.054571817e-34
    M_E = 9.10938356e-31
    m_star = m_star_ratio * M_E
    prefactor = 2.0 * m_star / (H_BAR ** 2)


    V_eff = V.copy()
    if l > 0:
        centrifugal = np.zeros_like(r_grid)

        r_safe = np.where(r_grid < 1e-18, 1e-18, r_grid)
        centrifugal = l * (l + 1) * (H_BAR ** 2) / (2.0 * m_star * r_safe ** 2)
        V_eff += centrifugal

    k2 = prefactor * (E_guess - V_eff)


    u = np.zeros(n, dtype=float)
    u[0] = 0.0
    u[1] = 1e-6

    h2 = h ** 2
    fac = h2 / 12.0
    for i in range(1, n - 1):
        denom = 1.0 + fac * k2[i + 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        u[i + 1] = (
            2.0 * (1.0 - 5.0 * fac * k2[i]) * u[i]
            - (1.0 + fac * k2[i - 1]) * u[i - 1]
        ) / denom


    try:
        norm = np.sqrt(np.trapezoid(u ** 2, r_grid))
    except AttributeError:
        norm = np.sqrt(np.trapz(u ** 2, r_grid))
    if norm > 1e-20:
        u /= norm
    else:
        u[:] = 0.0


    residual = u[-1]
    return u, residual


def solve_radial_wavefunctions(
    r_grid: np.ndarray,
    V: np.ndarray,
    m_star_ratio: float,
    num_states: int = 3,
    l: int = 0,
    E_min: float = 0.0,
    E_max: float = 2.0,
) -> List[dict]:
    EV_TO_J = 1.602176634e-19
    energies_eV = np.linspace(E_min, E_max, 200)
    residuals = []
    for E_eV in energies_eV:
        _, res = radial_wavefunction_shooting(r_grid, V, m_star_ratio, E_eV * EV_TO_J, l)
        residuals.append(res)
    residuals = np.array(residuals)


    states = []
    for i in range(len(residuals) - 1):
        if residuals[i] * residuals[i + 1] < 0:
            E_low = energies_eV[i]
            E_high = energies_eV[i + 1]

            for _ in range(20):
                E_mid = 0.5 * (E_low + E_high)
                _, res_mid = radial_wavefunction_shooting(r_grid, V, m_star_ratio, E_mid * EV_TO_J, l)
                if residuals[i] * res_mid < 0:
                    E_high = E_mid
                else:
                    E_low = E_mid
            E_eig = 0.5 * (E_low + E_high)
            u_eig, _ = radial_wavefunction_shooting(r_grid, V, m_star_ratio, E_eig * EV_TO_J, l)
            states.append({
                "energy_eV": float(E_eig),
                "energy_J": float(E_eig * EV_TO_J),
                "wavefunction": u_eig,
                "r_grid": r_grid,
            })
            if len(states) >= num_states:
                break
    return states
