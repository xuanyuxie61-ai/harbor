
import numpy as np
from typing import Tuple, Optional, Callable


def compute_lumped_mass_matrix(nodes: np.ndarray, elements: np.ndarray,
                                density: float = 1000.0) -> np.ndarray:
    n_nodes = nodes.shape[0]
    mass = np.zeros(n_nodes, dtype=np.float64)

    for e in elements:
        x0, x1, x2, x3 = nodes[e[0]], nodes[e[1]], nodes[e[2]], nodes[e[3]]
        mat = np.vstack([x1 - x0, x2 - x0, x3 - x0])
        vol = abs(np.linalg.det(mat)) / 6.0
        m_e = density * vol
        for n in e:
            mass[n] += m_e / 4.0

    M_diag = np.zeros(3 * n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        M_diag[3 * i:3 * i + 3] = mass[i]
    return M_diag


def compute_rayleigh_damping(M_diag: np.ndarray, K: np.ndarray,
                              alpha_ray: float = 0.0,
                              beta_ray: float = 0.0) -> np.ndarray:
    C_diag = alpha_ray * M_diag + beta_ray * np.diag(K)

    C_diag = np.maximum(C_diag, 0.0)
    return C_diag


def sawtooth_wave(t: float, period: float = 1.0,
                  amplitude: float = 1.0) -> float:
    if period <= 0:
        raise ValueError("周期必须为正")
    frac = (t / period) - np.floor(t / period)
    return amplitude * (2.0 * frac - 1.0)


def newmark_beta_step(u_n: np.ndarray, v_n: np.ndarray, a_n: np.ndarray,
                       dt: float, M_diag: np.ndarray, C_diag: np.ndarray,
                       compute_residual: Callable,
                       compute_stiffness: Callable,
                       beta_newmark: float = 0.25,
                       gamma_newmark: float = 0.5,
                       tol: float = 1e-8,
                       max_iter: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    n_dof = u_n.shape[0]

    u_pred = u_n + dt * v_n + (dt ** 2 / 2.0) * (1.0 - 2.0 * beta_newmark) * a_n
    v_pred = v_n + dt * (1.0 - gamma_newmark) * a_n


    u_new = u_pred.copy()
    a_new = np.zeros(n_dof, dtype=np.float64)

    converged = False
    for it in range(max_iter):

        a_new = (u_new - u_pred) / (beta_newmark * dt ** 2)
        v_new = v_pred + gamma_newmark * dt * a_new

        R_int, F_ext = compute_residual(u_new)

        R_dyn = F_ext - R_int - M_diag * a_new - C_diag * v_new

        if np.linalg.norm(R_dyn) < tol * (np.linalg.norm(F_ext) + 1.0):
            converged = True
            break

        K_T = compute_stiffness(u_new)

        K_eff = K_T + (1.0 / (beta_newmark * dt ** 2)) * np.diag(M_diag) \
                + (gamma_newmark / (beta_newmark * dt)) * np.diag(C_diag)

        try:
            du = np.linalg.solve(K_eff, R_dyn)
        except np.linalg.LinAlgError:

            du = np.linalg.lstsq(K_eff, R_dyn, rcond=None)[0]
        u_new += du

    if not converged:

        u_new = u_pred.copy()
        a_new = np.zeros(n_dof, dtype=np.float64)
        v_new = v_pred.copy()

    return u_new, v_new, a_new, converged


def trapezoidal_step(y_n: np.ndarray, t_n: float, dt: float,
                      f: Callable[[float, np.ndarray], np.ndarray],
                      max_inner_iter: int = 10,
                      tol: float = 1e-10) -> np.ndarray:
    t_next = t_n + dt
    f_n = f(t_n, y_n)
    y_next = y_n + dt * f_n

    for _ in range(max_inner_iter):
        f_next = f(t_next, y_next)
        y_new = y_n + 0.5 * dt * (f_n + f_next)
        if np.linalg.norm(y_new - y_next) < tol:
            y_next = y_new
            break
        y_next = y_new

    return y_next
