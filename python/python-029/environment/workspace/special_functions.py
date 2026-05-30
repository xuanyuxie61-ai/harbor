
import numpy as np



_BERNOULLI_NUMBERS = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
]


def log_gamma_stirling(z, n_terms=4):
    z = complex(z)
    if z.real <= 0:


        return np.log(np.pi) - np.log(np.sin(np.pi * z)) - log_gamma_stirling(1.0 - z, n_terms)


    result = (z - 0.5) * np.log(z) - z + 0.5 * np.log(2.0 * np.pi)


    for k in range(1, min(n_terms + 1, len(_BERNOULLI_NUMBERS) + 1)):
        B2k = _BERNOULLI_NUMBERS[k - 1]
        result += B2k / (2.0 * k * (2.0 * k - 1.0) * (z ** (2.0 * k - 1.0)))

    return result


def gamma_function(z):
    return np.exp(log_gamma_stirling(z))


def coulomb_wave_function_series(L, eta, rho, max_iter=1000, tol=1e-14):
    if rho <= 0:
        return 0.0


    log_cl = L * np.log(2.0) - 0.5 * np.pi * eta
    log_cl += log_gamma_stirling(L + 1.0 + 1j * eta).real
    log_cl -= log_gamma_stirling(2.0 * L + 2.0).real
    C_L = np.exp(log_cl)


    a_km2 = 1.0
    a_km1 = eta / (L + 1.0)

    sum_val = a_km2 + a_km1 * rho
    rho_pow = rho ** 2

    for k in range(2, max_iter + 1):
        denom = k * (k + 2 * L + 1)
        a_k = (2.0 * eta * a_km1 - a_km2) / denom
        term = a_k * rho_pow
        sum_val += term

        if abs(term) < tol * abs(sum_val):
            break

        a_km2 = a_km1
        a_km1 = a_k
        rho_pow *= rho

    F_L = C_L * (rho ** (L + 1.0)) * sum_val
    return F_L


def spherical_bessel_jn_highprecision(x, n, max_iter=1000):
    x = float(x)
    if x < 1e-15:
        return 1.0 if n == 0 else 0.0

    if n == 0:
        return np.sin(x) / x
    if n == 1:
        return np.sin(x) / (x ** 2) - np.cos(x) / x


    M = n + int(np.sqrt(10.0 * n)) + 20
    j = np.zeros(M + 2)
    j[M] = 1.0
    j[M + 1] = 0.0


    for k in range(M, 0, -1):
        j[k - 1] = (2.0 * k + 1.0) / x * j[k] - j[k + 1]


    scale = np.sin(x) / x / j[0]
    return j[n] * scale


def spherical_neumann_nn_highprecision(x, n):
    x = float(x)
    if x < 1e-15:
        return -1e10
    if n == 0:
        return -np.cos(x) / x
    if n == 1:
        return -np.cos(x) / (x ** 2) - np.sin(x) / x

    nm2 = -np.cos(x) / x
    nm1 = -np.cos(x) / (x ** 2) - np.sin(x) / x
    for k in range(2, n + 1):
        nn = (2.0 * k - 1.0) / x * nm1 - nm2
        nm2 = nm1
        nm1 = nn
    return nm1


def complex_error_function(z, n_terms=20):
    z = complex(z)
    result = 0.0 + 0.0j
    z2 = z * z
    term = z
    factorial_n = 1.0

    for n in range(n_terms):
        result += term / (factorial_n * (2.0 * n + 1.0))
        term *= -z2
        factorial_n *= (n + 1.0)

    return (2.0 / np.sqrt(np.pi)) * result


def coulomb_phase_shift(L, eta):
    sigma0 = log_gamma_stirling(1.0 + 1j * eta).imag
    sigma = sigma0
    for k in range(1, L + 1):
        sigma += np.arctan(eta / k)
    return sigma


def penetration_factor(l, k, R, eta=0.0):
    rho = k * R
    if eta == 0.0:

        jl = spherical_bessel_jn_highprecision(rho, l)
        nl = spherical_neumann_nn_highprecision(rho, l)
        denominator = (rho * jl) ** 2 + (rho * nl) ** 2
    else:

        FL = coulomb_wave_function_series(l, eta, rho)

        GL = 1.0 / FL if abs(FL) > 1e-10 else 1e10
        denominator = FL ** 2 + GL ** 2

    if denominator < 1e-30:
        denominator = 1e-30
    return rho / denominator


if __name__ == "__main__":

    print("ln Γ(5.5) =", log_gamma_stirling(5.5).real, "期望 ~3.958")
    print("Γ(5.5) =", gamma_function(5.5).real, "期望 ~52.34")

    print("j_5(3.0) =", spherical_bessel_jn_highprecision(3.0, 5))
    print("n_2(2.0) =", spherical_neumann_nn_highprecision(2.0, 2))

    print("erf(1+1j) =", complex_error_function(1.0 + 1.0j))
    print("Coulomb σ_2(η=1) =", coulomb_phase_shift(2, 1.0))

    F = coulomb_wave_function_series(0, 1.0, 2.0)
    print("F_0(η=1, ρ=2) =", F)
