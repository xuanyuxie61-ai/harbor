
import numpy as np
from scipy.special import legendre, eval_legendre


def compute_cross_sections(S_matrix_dict, k, l_max):
    prefactor = np.pi / (k ** 2)

    sigma_tot = 0.0
    sigma_react = 0.0
    sigma_el = 0.0

    for l in range(l_max + 1):

        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            key = (l, j)
            if key not in S_matrix_dict:
                continue
            S_l = S_matrix_dict[key]
            gj = int(2 * j + 1)





            sigma_tot += 0.0
            sigma_react += 0.0
            sigma_el += 0.0

    return {
        'sigma_total': prefactor * sigma_tot,
        'sigma_reaction': prefactor * sigma_react,
        'sigma_elastic': prefactor * sigma_el,
        'sigma_absorption': prefactor * sigma_react,
    }


def scattering_amplitude(theta, S_matrix_dict, k, l_max):
    theta = np.asarray(theta, dtype=float)
    mu = np.cos(theta)
    f = np.zeros_like(mu, dtype=complex)
    prefactor = 1.0 / (2.0j * k)

    for l in range(l_max + 1):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            key = (l, j)
            if key not in S_matrix_dict:
                continue
            S_l = S_matrix_dict[key]
            gj = int(2 * j + 1)
            P_l = eval_legendre(l, mu)
            f += gj * (S_l - 1.0) * P_l

    return prefactor * f


def differential_cross_section(theta, S_matrix_dict, k, l_max):
    f = scattering_amplitude(theta, S_matrix_dict, k, l_max)
    return np.abs(f) ** 2


def svd_analysis_smatrix(S_matrix_dict, l_max):

    dim = l_max + 1
    S_mat = np.zeros((dim, dim), dtype=complex)

    for l in range(dim):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for idx, j in enumerate(js):
            col = min(l + idx, dim - 1)
            key = (l, j)
            if key in S_matrix_dict:
                S_mat[l, col] = S_matrix_dict[key]


    U, s, Vh = np.linalg.svd(S_mat, full_matrices=False)


    total_energy = np.sum(s ** 2)
    cumulative = np.cumsum(s ** 2)
    relative_error = 1.0 - cumulative / total_energy


    rank_99 = np.searchsorted(cumulative / total_energy, 0.99) + 1

    return {
        'singular_values': s,
        'rank_99': rank_99,
        'relative_error': relative_error,
        'U': U,
        'Vh': Vh,
    }


def multipole_expansion_scattering(f_theta, theta, max_order):
    mu = np.cos(theta)
    f_vals = f_theta(theta)
    coeffs = np.zeros(max_order + 1, dtype=complex)



    dmu = np.diff(mu)

    for lam in range(max_order + 1):
        P_lam = eval_legendre(lam, mu)
        integrand = f_vals * P_lam

        integral = np.trapezoid(integrand, mu)
        coeffs[lam] = (2.0 * lam + 1.0) / 2.0 * integral

    return coeffs


def transmission_coefficients(S_matrix_dict, l_max):
    T = {}
    for l in range(l_max + 1):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            key = (l, j)
            if key in S_matrix_dict:
                T[key] = 1.0 - abs(S_matrix_dict[key]) ** 2
            else:
                T[key] = 0.0
    return T


def compound_formation_cross_section(params, T_dict, l_max):
    prefactor = np.pi / (params.k ** 2)
    sigma_cf = 0.0
    for l in range(l_max + 1):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            key = (l, j)
            if key in T_dict:
                sigma_cf += (2 * j + 1) * T_dict[key]
    return prefactor * sigma_cf


if __name__ == "__main__":

    k = 0.5
    l_max = 5
    S_dict = {}
    for l in range(l_max + 1):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            delta = 0.1 * (l + 1)
            eta = 0.9
            S_dict[(l, j)] = eta * np.exp(2.0j * delta)

    xs = compute_cross_sections(S_dict, k, l_max)
    print("截面 (fm²):", xs)

    theta = np.linspace(0.01, np.pi, 100)
    dsigma = differential_cross_section(theta, S_dict, k, l_max)
    print("微分截面范围:", dsigma.min(), dsigma.max())

    svd_res = svd_analysis_smatrix(S_dict, l_max)
    print("奇异值:", svd_res['singular_values'])
    print("99% 能量秩:", svd_res['rank_99'])
