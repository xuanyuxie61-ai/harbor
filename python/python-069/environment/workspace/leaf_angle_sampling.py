import numpy as np


def simplex_unit_sample(m, n):
    e = -np.log(np.random.rand(m + 1, n))
    s = np.sum(e, axis=0, keepdims=True)
    x = e[:m, :] / s
    return x


def simplex_general_sample(m, n, t):
    x1 = simplex_unit_sample(m, n)
    x = t[:, :m] @ x1 + t[:, m:m + 1] @ (1.0 - np.sum(x1, axis=0, keepdims=True))
    return x


def simplex_unit_volume(m):
    vol = 1.0
    for i in range(2, m + 1):
        vol /= float(i)
    return vol


def leaf_angle_monte_carlo(n_samples, theta_s, phi_s=0.0,
                           theta_l_mean=np.pi / 4, sigma_theta=np.pi / 6):

    theta_l = np.random.normal(theta_l_mean, sigma_theta, n_samples)
    theta_l = np.clip(theta_l, 0.01, np.pi / 2 - 0.01)
    phi_l = np.random.uniform(0.0, 2.0 * np.pi, n_samples)


    cos_xi = (np.cos(theta_l) * np.cos(theta_s)
              + np.sin(theta_l) * np.sin(theta_s) * np.cos(phi_l - phi_s))



    f_phi = 1.0 / (2.0 * np.pi)

    f_theta = np.exp(-0.5 * ((theta_l - theta_l_mean) / sigma_theta) ** 2)
    f_theta /= (sigma_theta * np.sqrt(2.0 * np.pi))

    jacobian = np.sin(theta_l)

    integrand = np.abs(cos_xi) * f_theta * f_phi * jacobian

    volume = (np.pi / 2.0) * (2.0 * np.pi)
    g_estimate = volume * np.mean(integrand)
    return g_estimate


def g_function_table(theta_s_range, n_samples=20000,
                     theta_l_mean=np.pi / 4, sigma_theta=np.pi / 6):
    theta_s_vals = np.asarray(theta_s_range, dtype=float)
    g_vals = np.array([leaf_angle_monte_carlo(n_samples, ts,
                                               theta_l_mean=theta_l_mean,
                                               sigma_theta=sigma_theta)
                       for ts in theta_s_vals])
    return theta_s_vals, g_vals
