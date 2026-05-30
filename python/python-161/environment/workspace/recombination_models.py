
import numpy as np
from typing import Tuple


def imtqlx(n: int, d: np.ndarray, e: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = d.copy()
    e = e.copy()
    z = z.copy()
    e[n - 1] = 0.0
    for l in range(1, n + 1):
        j = 0
        while True:
            for m in range(l, n):
                if m == n - 1:
                    break
                if abs(e[m]) <= 1e-14 * (abs(d[m]) + abs(d[m + 1])):
                    break
            if m == l - 1:
                break
            if j >= 30:
                raise RuntimeError("imtqlx 未收敛")
            j += 1
            g = (d[l] - d[l - 1]) / (2.0 * e[l - 1])
            r = np.sqrt(g * g + 1.0)
            g = d[m - 1] - d[l - 1] + e[l - 1] / (g + np.copysign(r, g))
            s, c = 1.0, 1.0
            p = 0.0
            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                if abs(f) >= abs(g):
                    c = g / f
                    r = np.sqrt(c * c + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c *= s
                else:
                    s = f / g
                    r = np.sqrt(s * s + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s *= c
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b

                for k in range(n):
                    temp = z[k + n * (i + 1)]
                    z[k + n * (i + 1)] = s * z[k + n * i] + c * temp
                    z[k + n * i] = c * z[k + n * i] - s * temp
            d[l - 1] -= p
            e[l - 1] = g
            e[m - 1] = 0.0
    return d, z


def laguerre_quadrature_rule(n: int, alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("n 必须为正整数")
    if alpha < -1.0:
        raise ValueError("alpha 必须 ≥ -1")

    import math

    diag = np.zeros(n)
    offdiag = np.zeros(n - 1)
    for i in range(n):
        diag[i] = 2.0 * i + 1.0 + alpha
    for i in range(n - 1):
        offdiag[i] = np.sqrt((i + 1.0) * (i + 1.0 + alpha))

    jacobi = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(jacobi)

    x_out = eigenvalues
    v0 = eigenvectors[0, :]
    w_out = math.gamma(alpha + 1.0) * v0 ** 2
    return x_out, w_out


def lm_polynomial_values(n: int, m: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    v = np.zeros((n + 1, len(x)))
    if n >= 0:
        v[0, :] = 1.0
    if n >= 1:
        v[1, :] = (m + 1.0) - x
    for i in range(2, n + 1):
        v[i, :] = ((m + 2.0 * i - 1.0 - x) * v[i - 1, :] + (1.0 - m - i) * v[i - 2, :]) / i
    return v[n, :]


def radiative_recombination_rate(
    n: float, p: float, n_i: float, T: float = 300.0
) -> float:
    B_300 = 2.0e-10
    B = B_300 * (T / 300.0) ** (-1.5)
    return B * (n * p - n_i * n_i)


def auger_recombination_rate(
    n: float, p: float, n_i: float, T: float = 300.0
) -> float:
    C_n = 1.0e-29
    C_p = 1.0e-29
    return (C_n * n + C_p * p) * (n * p - n_i * n_i)


def band_to_tail_recombination_integral(
    E_g: float, T: float, n: float, p: float,
    n_i: float, N_t_tail: float = 1e16, E_u: float = 0.015,
    quadrature_order: int = 16,
) -> float:
    if E_g <= 0 or T <= 0 or E_u <= 0:
        return 0.0

    kT = 8.617333e-5 * T
    x_nodes, w_nodes = laguerre_quadrature_rule(quadrature_order, alpha=0.0)


    sigma_eff = 1e-14
    v_th = 1e7

    integral = 0.0
    for xi, wi in zip(x_nodes, w_nodes):


        trap_depth = xi * E_u


        n1 = n_i * np.exp(trap_depth / kT)
        p1 = n_i * np.exp(-trap_depth / kT)
        denom = p + n1 + n + p1
        if denom > 0:
            integrand = (n * p - n_i * n_i) / denom
            integral += wi * integrand


    if not np.isfinite(integral):
        integral = 0.0

    R_tail = N_t_tail * sigma_eff * v_th * integral
    return max(R_tail, 0.0)


def total_recombination_rate(
    n: float, p: float, n_i: float, T: float = 300.0,
    tau_n: float = 1e-9, tau_p: float = 1e-9,
    E_t: float = 0.0, E_g: float = 1.6,
    N_t_tail: float = 1e16, E_u: float = 0.015,
) -> dict:






    kT = 8.617333e-5 * T
    R_srh = 0.0

    R_rad = radiative_recombination_rate(n, p, n_i, T)
    R_aug = auger_recombination_rate(n, p, n_i, T)
    R_tail = band_to_tail_recombination_integral(E_g, T, n, p, n_i, N_t_tail, E_u)

    return {
        "SRH": max(R_srh, 0.0),
        "radiative": max(R_rad, 0.0),
        "auger": max(R_aug, 0.0),
        "tail": max(R_tail, 0.0),
        "total": max(R_srh + R_rad + R_aug + R_tail, 0.0),
    }


if __name__ == "__main__":
    x, w = laguerre_quadrature_rule(8)
    print(f"Gauss-Laguerre (n=8) 节点: {x}")
    print(f"权重和: {w.sum():.6f} (应≈1)")

    rates = total_recombination_rate(1e15, 1e15, 1e10, 300.0, 1e-9, 1e-9, 0.0, 1.6)
    print("复合率分量:", rates)
