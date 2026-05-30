
import numpy as np
from math import gamma
from scipy.special import gammaln


def jacobi_polynomial(m, n, alpha, beta, x):
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("jacobi_polynomial: alpha and beta must be > -1")
    if n < 0:
        return np.empty((m, 0))

    x = np.asarray(x, dtype=float)
    v = np.ones((m, n + 1), dtype=float)

    if n == 0:
        return v

    v[:, 1] = (1.0 + 0.5 * (alpha + beta)) * x + 0.5 * (alpha - beta)

    for i in range(2, n + 1):
        c1 = 2.0 * i * (i + alpha + beta) * (2.0 * i - 2.0 + alpha + beta)
        c2 = (2.0 * i - 1.0 + alpha + beta) * (2.0 * i + alpha + beta) * (2.0 * i - 2.0 + alpha + beta)
        c3 = (2.0 * i - 1.0 + alpha + beta) * (alpha + beta) * (alpha - beta)
        c4 = -2.0 * (i - 1.0 + alpha) * (i - 1.0 + beta) * (2.0 * i + alpha + beta)

        v[:, i] = ((c3 + c2 * x) * v[:, i - 1] + c4 * v[:, i - 2]) / c1

    return v


def jacobi_quadrature_rule(n, alpha, beta):
    if n < 1:
        return np.array([]), np.array([])

    ab = alpha + beta
    abi = 2.0 + ab


    zemu = (2.0 ** (ab + 1.0)) * gamma(alpha + 1.0) * gamma(beta + 1.0) / gamma(abi)


    diag = np.zeros(n)
    offd = np.zeros(n - 1)

    diag[0] = (beta - alpha) / abi
    a2b2 = beta * beta - alpha * alpha

    for i in range(1, n):
        abi_val = 2.0 * (i + 1) + ab
        diag[i] = a2b2 / ((abi_val - 2.0) * abi_val)
        abi_sq = abi_val ** 2
        offd[i - 1] = np.sqrt(
            4.0 * (i + 1) * (i + 1 + alpha) * (i + 1 + beta) * (i + 1 + ab)
            / ((abi_sq - 1.0) * abi_sq)
        )



    jacobi_mat = np.diag(diag) + np.diag(offd, k=1) + np.diag(offd, k=-1)
    eigvals, eigvecs = np.linalg.eigh(jacobi_mat)

    x = eigvals
    w = zemu * (eigvecs[0, :] ** 2)

    return x, w


def spectral_expand_pulse(t_grid, pulse, alpha_jac=-0.5, beta_jac=-0.5, n_modes=32):
    if pulse.size != t_grid.size:
        raise ValueError("spectral_expand_pulse: t_grid and pulse must have same size")
    if t_grid.size < 2:
        raise ValueError("spectral_expand_pulse: grid too small")

    t_min, t_max = np.min(t_grid), np.max(t_grid)
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError("spectral_expand_pulse: invalid time grid")


    tau = 2.0 * (t_grid - t_min) / (t_max - t_min) - 1.0
    tau = np.clip(tau, -1.0, 1.0)


    x_q, w_q = jacobi_quadrature_rule(n_modes, alpha_jac, beta_jac)


    t_q = t_min + (x_q + 1.0) * 0.5 * (t_max - t_min)
    pulse_q = np.interp(t_q, t_grid, np.real(pulse)) + 1j * np.interp(t_q, t_grid, np.imag(pulse))


    v = jacobi_polynomial(n_modes, n_modes - 1, alpha_jac, beta_jac, x_q)


    hn = np.zeros(n_modes)
    for n in range(n_modes):
        log_num = (alpha_jac + beta_jac + 1.0) * np.log(2.0) + gammaln(n + alpha_jac + 1.0) + gammaln(n + beta_jac + 1.0)
        denom_arg = n + alpha_jac + beta_jac + 1.0
        if abs(denom_arg) < 1e-14:
            if abs(alpha_jac + 0.5) < 1e-10 and abs(beta_jac + 0.5) < 1e-10:
                hn[n] = np.pi if n == 0 else np.pi / 2.0
            else:
                hn[n] = 1.0
            continue
        log_den = np.log(abs(2.0 * n + alpha_jac + beta_jac + 1.0)) + gammaln(n + 1.0) + gammaln(denom_arg)
        hn[n] = np.exp(log_num - log_den)


    coeffs = np.zeros(n_modes, dtype=complex)
    for n in range(n_modes):
        integrand = pulse_q * v[:, n]
        coeffs[n] = np.sum(w_q * integrand) / hn[n]


    v_orig = jacobi_polynomial(t_grid.size, n_modes - 1, alpha_jac, beta_jac, tau)
    reconstructed = v_orig @ coeffs

    return coeffs, reconstructed


def dispersion_operator_spectral(coeffs, alpha_jac, beta_jac, n_modes, beta2, beta3, L):
    if coeffs.size != n_modes:
        raise ValueError("dispersion_operator_spectral: coefficient size mismatch")





    x_q, w_q = jacobi_quadrature_rule(n_modes + 2, alpha_jac, beta_jac)

    v0 = jacobi_polynomial(n_modes + 2, n_modes - 1, alpha_jac, beta_jac, x_q)

    v1 = jacobi_polynomial(n_modes + 2, n_modes - 1, alpha_jac + 1.0, beta_jac + 1.0, x_q)






    hn = np.zeros(n_modes)
    for n in range(n_modes):

        log_num = (alpha_jac + beta_jac + 1.0) * np.log(2.0) + gammaln(n + alpha_jac + 1.0) + gammaln(n + beta_jac + 1.0)

        denom_arg = n + alpha_jac + beta_jac + 1.0
        if abs(denom_arg) < 1e-14:



            if abs(alpha_jac + 0.5) < 1e-10 and abs(beta_jac + 0.5) < 1e-10:

                hn[n] = np.pi if n == 0 else np.pi / 2.0
            else:
                hn[n] = 1.0
            continue
        log_den = np.log(abs(2.0 * n + alpha_jac + beta_jac + 1.0)) + gammaln(n + 1.0) + gammaln(denom_arg)
        hn[n] = np.exp(log_num - log_den)


    D1 = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(1, n_modes):


            deriv_vals = 0.5 * (j + alpha_jac + beta_jac + 1.0) * v1[:, j - 1]
            D1[i, j] = np.sum(w_q * v0[:, i] * deriv_vals) / hn[i]


    D2 = D1 @ D1






    disp_coeffs = 1j * (beta2 / 2.0) * (D2 @ coeffs) - 1j * (beta3 / 6.0) * (D2 @ D1 @ coeffs)

    return disp_coeffs
