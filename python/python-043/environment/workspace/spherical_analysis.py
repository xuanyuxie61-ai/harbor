
import numpy as np
from special_functions import associated_legendre


def svd_sphere_decomposition(data_matrix, n_modes=10):
    U, S, Vt = np.linalg.svd(data_matrix, full_matrices=False)

    total_energy = np.sum(S ** 2)
    retained_energy = np.sum(S[:n_modes] ** 2)
    energy_ratio = retained_energy / (total_energy + 1e-30)

    return U[:, :n_modes], S[:n_modes], Vt[:n_modes, :], energy_ratio


def spherical_harmonic_transform(theta, phi, values, L_max=8):
    N = len(values)
    coeffs = np.zeros((L_max + 1, 2 * L_max + 1), dtype=complex)

    for l in range(L_max + 1):
        for m in range(-l, l + 1):

            P_lm = associated_legendre(l, abs(m), np.cos(theta))
            N_lm = np.sqrt((2 * l + 1) * np.math.factorial(l - abs(m))
                           / (4 * np.pi * np.math.factorial(l + abs(m))))
            Y_lm = N_lm * P_lm * np.exp(1j * m * phi)


            integrand = values * np.conj(Y_lm) * np.sin(theta)

            f_lm = np.mean(integrand) * 4 * np.pi
            coeffs[l, m + L_max] = f_lm

    return coeffs


def inverse_spherical_harmonic_transform(coeffs, theta, phi, L_max=None):
    if L_max is None:
        L_max = coeffs.shape[0] - 1

    N = len(theta)
    values = np.zeros(N, dtype=complex)

    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            P_lm = associated_legendre(l, abs(m), np.cos(theta))
            N_lm = np.sqrt((2 * l + 1) * np.math.factorial(l - abs(m))
                           / (4 * np.pi * np.math.factorial(l + abs(m))))
            Y_lm = N_lm * P_lm * np.exp(1j * m * phi)
            values += coeffs[l, m + L_max] * Y_lm

    return np.real(values)


def task_division_spherical_harmonics(L_max, proc_first, proc_last):
    task_number = L_max + 1
    p = proc_last + 1 - proc_first

    divisions = []
    i_hi = -1
    task_remain = task_number
    proc_remain = p

    for proc in range(proc_first, proc_last + 1):
        task_proc = _div_rounded(task_remain, proc_remain)
        proc_remain -= 1
        task_remain -= task_proc

        i_lo = i_hi + 1
        i_hi = i_hi + task_proc

        divisions.append((proc, task_proc, i_lo, i_hi))

    return divisions


def _div_rounded(a, b):
    if b == 0:
        return 0
    value = a / b
    if value < 0:
        return int(value - 0.5)
    else:
        return int(value + 0.5)


def compute_gauss_spectral_coefficients(B_field, theta, phi, nodes, r,
                                         L_max=6, R_surface=1.0):
    n_points = len(theta)
    g_coeffs = np.zeros((L_max + 1, L_max + 1))
    h_coeffs = np.zeros((L_max + 1, L_max + 1))


    mask = np.abs(r - R_surface) < 0.1
    if not np.any(mask):
        mask = np.ones(n_points, dtype=bool)

    theta_s = theta[mask]
    phi_s = phi[mask]
    Br = B_field[mask, 0] if B_field.ndim > 1 else B_field[mask]

    for l in range(1, L_max + 1):
        for m in range(0, l + 1):
            P_lm = associated_legendre(l, m, np.cos(theta_s))
            N_lm = np.sqrt((2 * l + 1) * np.math.factorial(l - m)
                           / (4 * np.pi * np.math.factorial(l + m)))
            integrand_g = Br * P_lm * np.cos(m * phi_s) * np.sin(theta_s)
            integrand_h = Br * P_lm * np.sin(m * phi_s) * np.sin(theta_s)

            norm = (l + 1) / (R_surface * (2 - (m == 0)))
            g_coeffs[l, m] = np.mean(integrand_g) * 4 * np.pi * norm
            h_coeffs[l, m] = np.mean(integrand_h) * 4 * np.pi * norm

    return g_coeffs, h_coeffs


def spectral_dipole_tilt(g_coeffs, h_coeffs):
    g10 = g_coeffs[1, 0]
    g11 = g_coeffs[1, 1]
    h11 = h_coeffs[1, 1]
    tilt = np.arctan2(np.sqrt(g11 ** 2 + h11 ** 2), abs(g10))
    return np.degrees(tilt)


def field_gradient_on_triangulation(nodes, elements, values):
    if elements.size == 0:
        return np.zeros_like(nodes)

    n_nodes = len(nodes)
    grad = np.zeros((n_nodes, 3))
    count = np.zeros(n_nodes)

    for elem in elements:
        p0, p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]], nodes[elem[3]]
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = p3 - p0


        vol = abs(np.dot(v0, np.cross(v1, v2))) / 6.0
        if vol < 1e-15:
            continue

        val = values[elem]

        grad_elem = np.array([
            (val[1] - val[0]) / (np.linalg.norm(v0) + 1e-15),
            (val[2] - val[0]) / (np.linalg.norm(v1) + 1e-15),
            (val[3] - val[0]) / (np.linalg.norm(v2) + 1e-15),
        ])

        for idx in elem:
            grad[idx] += grad_elem
            count[idx] += 1

    for i in range(n_nodes):
        if count[i] > 0:
            grad[i] /= count[i]

    return grad
