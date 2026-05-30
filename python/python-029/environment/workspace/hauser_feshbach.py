
import numpy as np
from scipy.linalg import solve


def level_density_parameter(A, E_exc):
    if A <= 0:
        raise ValueError("质量数 A 必须为正")
    k_tau = 8.0

    a = A / k_tau * (1.0 + 0.05 * np.log1p(E_exc / 10.0))
    return a


def level_density(A, E_exc, J, spin_cutoff=5.0):
    if E_exc <= 0:
        return 1e-30
    a = level_density_parameter(A, E_exc)

    rho_total = np.exp(2.0 * np.sqrt(a * E_exc)) / (12.0 * np.sqrt(2.0) * spin_cutoff * (a ** 0.25) * (E_exc ** 1.25))

    spin_factor = (2.0 * J + 1.0) / (2.0 * np.sqrt(2.0 * np.pi) * (spin_cutoff ** 3))
    spin_factor *= np.exp(-((J + 0.5) ** 2) / (2.0 * spin_cutoff ** 2))
    return rho_total * spin_factor


def transmission_coefficient_integral(T_dict, l_max):
    total = 0.0
    for l in range(l_max + 1):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            key = (l, j)
            if key in T_dict:
                total += (2.0 * j + 1.0) * T_dict[key]
    return total


def compound_formation_cross_section(params, T_dict, l_max, I_target=0.0, I_proj=0.5):
    prefactor = np.pi / (params.k ** 2)
    spin_denom = (2.0 * I_proj + 1.0) * (2.0 * I_target + 1.0)
    T_sum = transmission_coefficient_integral(T_dict, l_max)

    return prefactor * T_sum / spin_denom


def decay_width(T_dict, l_max, E_gamma=1.0):
    T_sum = transmission_coefficient_integral(T_dict, l_max)

    Gamma_n = T_sum * 0.1
    Gamma_gamma = E_gamma ** 5 * 1e-6
    Gamma_total = Gamma_n + Gamma_gamma + 0.01

    return {
        'neutron': Gamma_n,
        'gamma': Gamma_gamma,
        'total': Gamma_total,
        'ratio_n': Gamma_n / Gamma_total,
        'ratio_gamma': Gamma_gamma / Gamma_total,
    }


def open_newton_cotes_weights(n, a, b):
    if n < 1:
        raise ValueError("n 必须 >= 1")
    i = np.arange(1, n + 1, dtype=float)
    x = ((n - i + 1.0) * a + i * b) / (n + 1.0)



    if n == 1:

        w = np.array([b - a], dtype=float)
    elif n == 2:
        h = (b - a) / 3.0
        w = np.array([h, h])
    elif n == 3:
        h = (b - a) / 4.0
        w = np.array([3.0 * h / 2.0, 3.0 * h / 2.0, 3.0 * h / 2.0])
    elif n == 4:
        h = (b - a) / 5.0
        w = np.array([2.0 * h, 2.0 * h, 2.0 * h, 2.0 * h])
    else:


        V = np.vander(x - a, increasing=True, N=n)

        exact_moments = np.array([((b ** (k + 1) - a ** (k + 1)) / (k + 1.0)) for k in range(n)])
        w = solve(V.T, exact_moments)

    return x, w


def energy_average_cross_section(E_min, E_max, n_points, sigma_func):
    x, w = open_newton_cotes_weights(n_points, E_min, E_max)
    vals = np.array([sigma_func(e) for e in x])
    integral = np.dot(w, vals)
    return integral / (E_max - E_min)


def decay_chain_bdf2(initial_populations, decay_matrix, t_span, n_steps):
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    n_species = len(initial_populations)

    t = np.linspace(t0, tf, n_steps + 1)
    N = np.zeros((n_steps + 1, n_species))
    N[0, :] = initial_populations


    I = np.eye(n_species)
    A1 = I - dt * decay_matrix
    N[1, :] = solve(A1, N[0, :])


    A_bdf2 = I - (2.0 * dt / 3.0) * decay_matrix
    for n in range(1, n_steps):
        rhs = (4.0 / 3.0) * N[n, :] - (1.0 / 3.0) * N[n - 1, :]
        N[n + 1, :] = solve(A_bdf2, rhs)

    return t, N


def width_fluctuation_correction(T_dict, l_max, nu=1.0):
    T_sum = transmission_coefficient_integral(T_dict, l_max)

    W = 1.0 / np.sqrt(1.0 + 2.0 * T_sum / (nu + 1.0))
    return W


if __name__ == "__main__":

    print("能级密度参数 (A=56, E=10MeV):", level_density_parameter(56, 10.0))
    print("能级密度 (A=56, E=10MeV, J=2):", level_density(56, 10.0, 2.0))

    x, w = open_newton_cotes_weights(4, 0.0, 10.0)
    print("NCO(4) 积分 x³:", np.dot(w, x ** 3), "期望 2500")


    M = np.array([[-0.5, 0.0, 0.0],
                  [0.3, -0.2, 0.0],
                  [0.2, 0.2, -0.1]])
    N0 = np.array([100.0, 0.0, 0.0])
    t, N = decay_chain_bdf2(N0, M, (0.0, 10.0), 100)
    print("衰变链最终布居:", N[-1, :])
