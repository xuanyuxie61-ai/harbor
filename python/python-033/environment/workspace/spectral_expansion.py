
import numpy as np


def shifted_legendre_polynomial_value(m, n, x):
    x = np.asarray(x, dtype=float)
    if x.shape[0] != m:
        raise ValueError("x 的长度必须与 m 一致")
    if np.any(x < 0.0) or np.any(x > 1.0):

        x = np.clip(x, 0.0, 1.0)

    v = np.zeros((m, n + 1), dtype=float)
    v[:, 0] = 1.0
    if n >= 1:
        v[:, 1] = 2.0 * x - 1.0
    for j in range(1, n):
        v[:, j + 1] = (
            (2 * j + 1) * (2.0 * x - 1.0) * v[:, j]
            - j * v[:, j - 1]
        ) / (j + 1)
    return v


def gauss_legendre_shifted_nodes_weights(n):
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (xi + 1.0)
    w = 0.5 * wi
    return x, w


def spectral_expand_reaction_rate(temperatures, rates, degree=8):
    temperatures = np.asarray(temperatures, dtype=float)
    rates = np.asarray(rates, dtype=float)
    t_min, t_max = np.min(temperatures), np.max(temperatures)
    if t_max <= t_min:
        raise ValueError("温度范围必须为正")

    tau = (temperatures - t_min) / (t_max - t_min)
    tau = np.clip(tau, 0.0, 1.0)


    x_quad, w_quad = gauss_legendre_shifted_nodes_weights(degree + 4)
    v_quad = shifted_legendre_polynomial_value(len(x_quad), degree, x_quad)


    rates_quad = np.interp(x_quad, tau, rates)

    coeffs = np.zeros(degree + 1, dtype=float)
    for k in range(degree + 1):

        coeffs[k] = (2 * k + 1) * np.sum(w_quad * rates_quad * v_quad[:, k])
    return coeffs, t_min, t_max


def spectral_evaluate_reaction_rate(tau, coeffs, t_min, t_max):
    tau = np.asarray(tau, dtype=float)
    tau = np.clip(tau, 0.0, 1.0)
    degree = len(coeffs) - 1
    m = tau.shape[0]
    v = shifted_legendre_polynomial_value(m, degree, tau)
    rates = v @ coeffs
    return rates


def test_spectral_expansion():
    T = np.linspace(1e9, 10e9, 200)

    kB = 1.380649e-16
    Q = 2.5e6
    nu = 0.5
    R = (T ** nu) * np.exp(-Q / (kB * T))
    coeffs, t_min, t_max = spectral_expand_reaction_rate(T, R, degree=10)
    tau_test = np.linspace(0, 1, 100)
    R_recon = spectral_evaluate_reaction_rate(tau_test, coeffs, t_min, t_max)
    T_test = tau_test * (t_max - t_min) + t_min
    R_exact = (T_test ** nu) * np.exp(-Q / (kB * T_test))
    rel_err = np.abs(R_recon - R_exact) / (np.abs(R_exact) + 1e-30)
    print(f"[spectral_expansion] Max relative reconstruction error: {np.max(rel_err):.3e}")
    assert np.max(rel_err) < 0.05, "Spectral expansion accuracy too low"


if __name__ == "__main__":
    test_spectral_expansion()
