
import numpy as np


def diff2_center(f_func, x, h=1e-5):
    x = float(x)
    h = float(h)
    if h <= 0.0:
        raise ValueError("Step size h must be positive")
    return (f_func(x + h) - 2.0 * f_func(x) + f_func(x - h)) / (h * h)


def e_spigot(n_digits):
    if n_digits <= 0:
        return [2]
    n = n_digits + 2
    a = np.ones(n, dtype=np.int64)
    digits = [2]
    
    for _ in range(n_digits):
        a *= 10
        carry = 0
        for i in range(n - 1, -1, -1):
            q = a[i] // (i + 2)
            a[i] = a[i] % (i + 2)
            if i > 0:
                a[i - 1] += q
            else:
                carry = q
        digits.append(int(carry))
    return digits[:n_digits + 1]


def yukawa_potential(r, Q_eff, lambda_D):
    eps0 = 8.854187817e-12
    if r <= 0.0:
        return 0.0
    return (Q_eff**2 / (4.0 * np.pi * eps0 * r)) * np.exp(-r / lambda_D)


def yukawa_force_magnitude(r, Q_eff, lambda_D):




    raise NotImplementedError("Hole 1: yukawa_force_magnitude is not implemented.")


def yukawa_force_vector(r_vec, Q_eff, lambda_D):
    r = np.linalg.norm(r_vec)
    if r < 1e-15:
        return np.zeros_like(r_vec)
    fm = yukawa_force_magnitude(r, Q_eff, lambda_D)
    return -fm * (r_vec / r)


def debye_length(n_e, T_e):
    eps0 = 8.854187817e-12
    k_B = 1.380649e-23
    e = 1.602176634e-19
    return np.sqrt(eps0 * k_B * T_e / (n_e * e**2))


def coupling_parameter(Q_eff, n_dust, T_dust, lambda_D):
    eps0 = 8.854187817e-12
    k_B = 1.380649e-23
    a_ws = (3.0 / (4.0 * np.pi * n_dust)) ** (1.0 / 3.0)
    kappa = a_ws / lambda_D
    gamma = (Q_eff**2 / (4.0 * np.pi * eps0 * a_ws * k_B * T_dust)) * np.exp(-kappa)
    return gamma


def wigner_seitz_radius(n_dust):
    return (3.0 / (4.0 * np.pi * n_dust)) ** (1.0 / 3.0)
