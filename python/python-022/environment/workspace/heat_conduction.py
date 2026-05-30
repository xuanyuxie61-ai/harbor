
import numpy as np
from typing import Tuple
from icf_parameters import PC, NP, TP
from utils import log_mean, safe_divide, clamp_array


def spitzer_harm_conductivity(T_e: float, Z_eff: float, n_e: float) -> float:
    if T_e <= 0.0 or Z_eff <= 0.0 or n_e <= 0.0:
        return 0.0


    ln_lambda = 23.5 - np.log(np.sqrt(n_e) / max(T_e, 1.0))
    ln_lambda = max(ln_lambda, 2.0)

    K = 1.84e-5 * T_e**2.5 / (Z_eff * ln_lambda)
    return max(K, 0.0)


def flux_limited_conductivity(T_e: float, Z_eff: float, n_e: float,
                              grad_T: float) -> float:
    K_sh = spitzer_harm_conductivity(T_e, Z_eff, n_e)
    if abs(grad_T) < 1.0e-30:
        return K_sh


    v_th = np.sqrt(PC.BOLTZMANN * T_e / PC.ELECTRON_MASS)
    q_max = NP.FLUX_LIMITER * n_e * PC.BOLTZMANN * T_e * v_th

    q_sh = K_sh * abs(grad_T)
    if q_sh > q_max:
        return q_max / abs(grad_T)
    return K_sh


def build_tridiag_heat_matrix(r: np.ndarray, K_eff: np.ndarray,
                              rho: np.ndarray, cv: np.ndarray,
                              dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = len(r)
    n_cells = n_nodes - 1

    lower = np.zeros(n_nodes - 1)
    main = np.zeros(n_nodes)
    upper = np.zeros(n_nodes - 1)
    rhs = np.zeros(n_nodes)


    for i in range(1, n_nodes - 1):

        vol_left = 4.0 * np.pi * r[i]**2 * (r[i] - r[i - 1]) / 2.0 if i > 0 else 0.0
        vol_right = 4.0 * np.pi * r[i]**2 * (r[i + 1] - r[i]) / 2.0 if i < n_nodes - 1 else 0.0
        m_node = rho[i - 1] * vol_left + rho[min(i, n_cells - 1)] * vol_right
        cv_node = cv[i - 1] if i > 0 else cv[0]
        main[i] += m_node * cv_node
        rhs[i] += m_node * cv_node


    main[0] = 1.0
    main[-1] = 1.0
    rhs[0] = 0.0
    rhs[-1] = 0.0


    for i in range(n_cells):
        r_face = 0.5 * (r[i] + r[i + 1])
        dr = r[i + 1] - r[i]
        if dr < 1.0e-15:
            continue


        A_face = 4.0 * np.pi * r_face**2

        K_face = 0.5 * (K_eff[i] + K_eff[i]) if i >= len(K_eff) else K_eff[i]


        coeff = dt * A_face * K_face / dr

        if i == 0:

            main[1] += coeff
        elif i == n_cells - 1:

            main[i] += coeff
        else:
            main[i] += coeff
            main[i + 1] += coeff
            lower[i] -= coeff
            upper[i] -= coeff

    return lower, main, upper, rhs


def r83v_cg_solve(lower: np.ndarray, main: np.ndarray, upper: np.ndarray,
                  b: np.ndarray, x0: np.ndarray = None,
                  tol: float = 1.0e-12, max_iter: int = None) -> np.ndarray:
    n = len(main)
    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    def matvec(v):
        result = main * v
        if n > 1:
            result[:-1] += upper * v[1:]
            result[1:] += lower * v[:-1]
        return result


    Ax = matvec(x)
    r = b - Ax
    p = r.copy()

    rs_old = float(np.dot(r, r))
    if rs_old < tol**2:
        return x

    for it in range(max_iter):
        Ap = matvec(p)
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1.0e-30:
            break

        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap

        rs_new = float(np.dot(r, r))
        if np.sqrt(rs_new) < tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def solve_heat_conduction(r: np.ndarray, T_old: np.ndarray,
                          rho: np.ndarray, Z_eff: np.ndarray,
                          n_e: np.ndarray, dt: float,
                          source: np.ndarray = None) -> np.ndarray:
    n_nodes = len(r)
    n_cells = n_nodes - 1

    if source is None:
        source = np.zeros(n_cells)


    T_nodes = np.zeros(n_nodes)
    T_nodes[1:-1] = 0.5 * (T_old[:-1] + T_old[1:])
    T_nodes[0] = T_old[0]
    T_nodes[-1] = T_old[-1]


    K_eff = np.zeros(n_cells)
    for i in range(n_cells):
        dr = r[i + 1] - r[i]
        if dr < 1.0e-15:
            grad_T = 0.0
        else:
            grad_T = (T_nodes[i + 1] - T_nodes[i]) / dr
        K_eff[i] = flux_limited_conductivity(T_old[i], Z_eff[i], n_e[i], grad_T)


    cv_e = np.zeros(n_cells)
    for i in range(n_cells):
        cv_e[i] = 1.5 * PC.BOLTZMANN * n_e[i] / max(rho[i], 1.0e-30)


    lower, main, upper, rhs = build_tridiag_heat_matrix(r, K_eff, rho, cv_e, dt)


    for i in range(1, n_nodes - 1):
        rhs[i] *= T_nodes[i]

        if i < n_cells:
            rhs[i] += dt * source[i] * 0.5
        if i > 0:
            rhs[i] += dt * source[i - 1] * 0.5


    rhs[0] = T_nodes[0]
    rhs[-1] = T_nodes[-1]


    T_new_nodes = r83v_cg_solve(lower, main, upper, rhs, x0=T_nodes)


    T_new_cells = 0.5 * (T_new_nodes[:-1] + T_new_nodes[1:])
    T_new_cells = np.maximum(T_new_cells, 1.0)

    return T_new_cells
