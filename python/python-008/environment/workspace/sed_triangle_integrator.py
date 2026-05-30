
import numpy as np


def triangle_area(vertices):
    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]
    area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1])
                     - (v2[0] - v0[0]) * (v1[1] - v0[1]))
    return area


def triangle_sample(vertices, n):
    v0 = vertices[0, :].reshape(2, 1)
    v1 = vertices[1, :].reshape(2, 1)
    v2 = vertices[2, :].reshape(2, 1)

    r1 = np.random.rand(n)
    r2 = np.random.rand(n)

    sqrt_r1 = np.sqrt(r1)
    lam1 = 1.0 - sqrt_r1
    lam2 = sqrt_r1 * (1.0 - r2)
    lam3 = sqrt_r1 * r2

    p = lam1 * v0 + lam2 * v1 + lam3 * v2
    return p


def synchrotron_function_approx(x):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)


    mask_small = x < 1e-3
    if np.any(mask_small):
        result[mask_small] = 1.808 * x[mask_small] ** (1.0 / 3.0) * np.exp(-x[mask_small])


    mask_large = x > 10.0
    if np.any(mask_large):
        result[mask_large] = np.sqrt(np.pi * x[mask_large] / 2.0) * np.exp(-x[mask_large])


    mask_mid = ~(mask_small | mask_large)
    if np.any(mask_mid):
        xi = x[mask_mid]

        result[mask_mid] = (1.808 * xi ** (1.0 / 3.0)
                            * np.exp(-xi)
                            * (1.0 + 0.16 * xi ** (2.0 / 3.0))
                            / (1.0 + 0.53 * xi ** (2.0 / 3.0)))

    return result


def integrand_sed(gamma, theta, nu_obs, B, N_gamma):
    m_e = 9.10938356e-28
    c = 2.99792458e10
    e = 4.80320427e-10
    p_index = 2.5

    sin_theta = np.sin(theta)
    sin_theta = np.clip(sin_theta, 1e-6, 1.0)

    nu_c = (3.0 * e * B * sin_theta) / (4.0 * np.pi * m_e * c) * gamma ** 2
    nu_c = np.clip(nu_c, 1e-20, None)

    x = nu_obs / nu_c
    F_x = synchrotron_function_approx(x)

    prefactor = (np.sqrt(3.0) * e ** 3 * B * sin_theta) / (4.0 * np.pi * m_e * c ** 2)
    n_e = N_gamma * gamma ** (-p_index)

    f = prefactor * F_x * n_e
    f = np.clip(f, 0.0, 1e200)
    return f


def monte_carlo_sed(vertices, n_samples, nu_obs, B, N_gamma):
    area = triangle_area(vertices)
    p = triangle_sample(vertices, n_samples)
    gamma = p[0, :]
    theta = p[1, :]


    gamma = np.clip(gamma, 1.0, 1e12)
    theta = np.clip(theta, 1e-6, np.pi - 1e-6)

    f_vals = integrand_sed(gamma, theta, nu_obs, B, N_gamma)

    mean_f = np.mean(f_vals)
    std_f = np.std(f_vals, ddof=1)

    flux = area * mean_f
    std_err = area * std_f / np.sqrt(n_samples)
    return flux, std_err
