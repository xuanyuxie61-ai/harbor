
import numpy as np
from scipy.integrate import solve_ivp


def beta_function_su3(g: float, nf: int = 2) -> float:
    beta0 = 11.0 - (2.0 / 3.0) * nf
    beta1 = 102.0 - (38.0 / 3.0) * nf
    beta = -beta0 * g ** 3 / (16.0 * np.pi ** 2)
    beta -= beta1 * g ** 5 / (256.0 * np.pi ** 4)
    return beta


def alpha_s_running(mu: np.ndarray, lambda_qcd: float = 0.3,
                    nf: int = 2) -> np.ndarray:
    beta0 = 11.0 - (2.0 / 3.0) * nf
    log_term = np.log((mu ** 2) / (lambda_qcd ** 2))
    log_term = np.where(log_term < 1e-3, 1e-3, log_term)
    alpha = 4.0 * np.pi / (beta0 * log_term)
    return alpha


def rg_flow_equations(t: float, y: np.ndarray, nf: int = 2) -> np.ndarray:
    g, mq = y
    g = max(g, 1e-6)
    mq = max(mq, 1e-9)

    dgdt = beta_function_su3(g, nf)


    gamma_m = 12.0 / (16.0 * np.pi ** 2)
    dmqdt = -gamma_m * g ** 2 * mq

    return np.array([dgdt, dmqdt])


def solve_rg_flow(g0: float, mq0: float, t_span: tuple,
                  nf: int = 2, n_points: int = 200) -> tuple:
    sol = solve_ivp(
        lambda t, y: rg_flow_equations(t, y, nf),
        t_span, [g0, mq0], method='RK45',
        rtol=1e-9, atol=1e-12, dense_output=True
    )
    t = np.linspace(t_span[0], t_span[1], n_points)
    y = sol.sol(t)
    return t, y


def lattice_coupling_from_beta(beta_lat: float, nf: int = 2) -> float:
    if beta_lat <= 0:
        raise ValueError("beta must be positive")
    return np.sqrt(6.0 / beta_lat)


def beta_from_lattice_spacing(a_fm: float, beta0: float = 11.0 - 4.0 / 3.0) -> float:
    lambda_inv_fm = 1.0 / 0.5
    a_target = a_fm

    beta_est = (11.0 / (6.0 * np.pi ** 2)) * np.log(1.0 / (a_target * lambda_inv_fm))
    return beta_est


def rg_step_matrix(n_scales: int = 5) -> np.ndarray:
    R = np.zeros((n_scales, n_scales))
    for i in range(n_scales):
        R[i, i] = -0.5
        if i > 0:
            R[i, i - 1] = 0.3
    return R


def coupled_rg_reaction_network(g0_vec: np.ndarray, t_span: tuple,
                                n_points: int = 200) -> tuple:
    n = len(g0_vec)
    R = rg_step_matrix(n)

    def deriv(t, g):
        return R @ g

    sol = solve_ivp(deriv, t_span, g0_vec, method='BDF',
                    rtol=1e-8, atol=1e-10, dense_output=True)
    t = np.linspace(t_span[0], t_span[1], n_points)
    g = sol.sol(t)
    return t, g
