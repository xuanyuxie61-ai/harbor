
import numpy as np
from math import sqrt, pi, exp


class RTESolverError(Exception):
    pass


def build_rte_matrix(num_depth, num_angle, tau_total, omega, g, B=None):
    if not (0.0 <= omega <= 1.0):
        raise RTESolverError("build_rte_matrix: ω 必须在 [0,1]")
    if not (-1.0 < g < 1.0):
        raise RTESolverError("build_rte_matrix: g 必须在 (-1,1)")

    N = num_depth * num_angle

    dtau = None


    mu, w = _gauss_legendre_nodes_weights(num_angle)

    A = np.zeros((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)

    if B is None:
        B = np.zeros(num_depth, dtype=np.float64)

    def idx(t, a):

        pass

    for t in range(num_depth):
        for a in range(num_angle):
            i = idx(t, a)
            mu_a = mu[a]


            scatter_diag = 0.0
            for aj in range(num_angle):
                p_val = _hg_phase_function(mu_a, mu[aj], g)
                scatter_diag += 0.5 * omega * w[aj] * p_val

            if mu_a > 0:



                if t == num_depth - 1:

                    A[i, i] = 1.0
                    b[i] = 0.0
                else:



                    coeff = mu_a / dtau
                    A[i, i] = coeff + 1.0
                    A[i, idx(t + 1, a)] = -coeff

                    for aj in range(num_angle):
                        p_val = _hg_phase_function(mu_a, mu[aj], g)
                        A[i, idx(t, aj)] -= 0.5 * omega * w[aj] * p_val
                    b[i] = (1.0 - omega) * B[t]
            else:

                if t == 0:

                    A[i, i] = 1.0
                    b[i] = 1.0
                else:
                    coeff = -mu_a / dtau
                    A[i, i] = coeff + 1.0
                    A[i, idx(t - 1, a)] = -coeff
                    for aj in range(num_angle):
                        p_val = _hg_phase_function(mu_a, mu[aj], g)
                        A[i, idx(t, aj)] -= 0.5 * omega * w[aj] * p_val
                    b[i] = (1.0 - omega) * B[t]

    return A, b, mu, w


def _gauss_legendre_nodes_weights(n):
    if n == 2:
        x = np.array([-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)])
        w = np.array([1.0, 1.0])
    elif n == 4:
        x = np.array([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116])
        w = np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451])
    elif n == 6:
        x = np.array([-0.9324695142, -0.6612093865, -0.2386191861,
                      0.2386191861, 0.6612093865, 0.9324695142])
        w = np.array([0.1713244924, 0.3607615730, 0.4679139346,
                      0.4679139346, 0.3607615730, 0.1713244924])
    elif n == 8:
        x = np.array([-0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
                      0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565])
        w = np.array([0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
                      0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363])
    else:

        x = np.linspace(-1.0 + 1.0 / n, 1.0 - 1.0 / n, n)
        w = np.full(n, 2.0 / n)
    return x, w


def _hg_phase_function(mu1, mu2, g):
    cos_theta = mu1 * mu2
    denom = (1.0 + g ** 2 - 2.0 * g * cos_theta) ** 1.5
    if denom < 1e-15:
        denom = 1e-15
    return (1.0 - g ** 2) / denom


def sor_solve(A, b, x0=None, omega_sor=1.5, tol=1e-10, max_iter=2000):
    n = A.shape[0]
    if A.shape[1] != n or b.shape[0] != n:
        raise RTESolverError("sor_solve: 维度不匹配")
    if not (0.0 < omega_sor < 2.0):
        raise RTESolverError("sor_solve: ω_SOR 必须在 (0,2)")

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.array(x0, dtype=np.float64, copy=True)

    for it in range(1, max_iter + 1):
        x_new = x.copy()
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j == i:
                    continue
                if j < i:
                    sigma += A[i, j] * x_new[j]
                else:
                    sigma += A[i, j] * x[j]
            if abs(A[i, i]) < 1e-15:
                raise RTESolverError(f"sor_solve: A[{i},{i}] 为零，无法迭代")
            x_new[i] = (1.0 - omega_sor) * x[i] + omega_sor * (b[i] - sigma) / A[i, i]

        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        if diff < tol:
            residual = np.linalg.norm(A @ x - b, ord=np.inf)
            return x, it, residual


        if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.linalg.norm(x, ord=np.inf) > 1e30:
            residual = np.linalg.norm(A @ x - b, ord=np.inf)
            return x, it, residual

    residual = np.linalg.norm(A @ x - b, ord=np.inf)
    return x, max_iter, residual


def compute_radiative_flux(I, mu, w):
    integrand = I * mu
    return 2.0 * pi * np.trapezoid(integrand, mu)


def compute_heating_rate(F_up, F_down, dtau, rho_cp):
    dF_net = F_up - F_down

    heating = dF_net / dtau / rho_cp

    heating_k_day = heating * 86400.0
    return heating_k_day
