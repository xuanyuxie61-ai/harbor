
import numpy as np
from utils import validate_positive


def quadrature_weights_vandermonde(n, a, b, x_nodes):
    x_nodes = np.asarray(x_nodes).flatten()
    if len(x_nodes) != n:
        raise ValueError("x_nodes length must equal n.")

    V = np.zeros((n, n))
    V[0, :] = 1.0
    for i in range(1, n):
        V[i, :] = V[i - 1, :] * x_nodes

    rhs = np.zeros(n)
    for i in range(n):
        rhs[i] = (b ** (i + 1) - a ** (i + 1)) / (i + 1.0)

    w = np.linalg.solve(V.T, rhs)
    return w


def gauss_legendre_nodes_weights(n, a, b):
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)

    x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
    w_mapped = 0.5 * (b - a) * w
    return x_mapped, w_mapped


def hermite_gauss_nodes_weights(n):
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)
    return x, w


def hermite_monomial_integral(n, option=1):
    if n < 0:
        return -np.inf
    if n % 2 == 1:
        return 0.0


    double_fact = 1.0
    k = n - 1
    while k > 0:
        double_fact *= k
        k -= 2

    if option == 0 or option == 1:
        return double_fact * np.sqrt(np.pi) / (2.0 ** (n / 2.0))
    elif option == 2:
        return double_fact * np.sqrt(2.0 * np.pi)
    elif option == 3:
        return double_fact / (2.0 ** (n / 2.0))
    elif option == 4:
        return double_fact
    else:
        raise ValueError("option must be 0-4.")


def compute_j_integral(stress_field, displacement_jump, crack_tip_pos,
                       integration_radius, n_quad=16):
    theta = np.linspace(0, np.pi, n_quad)
    x_path = crack_tip_pos[0] + integration_radius * np.cos(theta)
    y_path = crack_tip_pos[1] + integration_radius * np.sin(theta)


    t_nodes, t_weights = gauss_legendre_nodes_weights(n_quad, 0.0, np.pi)
    x_quad = crack_tip_pos[0] + integration_radius * np.cos(t_nodes)
    y_quad = crack_tip_pos[1] + integration_radius * np.sin(t_nodes)

    J_val = 0.0
    for i in range(n_quad):
        xq, yq = x_quad[i], y_quad[i]

        nx = np.cos(t_nodes[i])
        ny = np.sin(t_nodes[i])
        ds = integration_radius * t_weights[i]




        r = integration_radius

        sigma_11 = 1.0 / np.sqrt(2.0 * np.pi * r) * np.cos(t_nodes[i] / 2.0) * (
            1.0 - np.sin(t_nodes[i] / 2.0) * np.sin(1.5 * t_nodes[i]))
        sigma_22 = 1.0 / np.sqrt(2.0 * np.pi * r) * np.cos(t_nodes[i] / 2.0) * (
            1.0 + np.sin(t_nodes[i] / 2.0) * np.sin(1.5 * t_nodes[i]))
        sigma_12 = 1.0 / np.sqrt(2.0 * np.pi * r) * np.sin(t_nodes[i] / 2.0) * np.cos(
            t_nodes[i] / 2.0) * np.cos(1.5 * t_nodes[i])


        W = 0.5 * (sigma_11 ** 2 + sigma_22 ** 2 + 2.0 * sigma_12 ** 2)


        T1 = sigma_11 * nx + sigma_12 * ny
        T2 = sigma_12 * nx + sigma_22 * ny


        du1_dx = 0.1 / np.sqrt(r)
        du2_dx = 0.1 / np.sqrt(r)

        J_val += (W * nx - (T1 * du1_dx + T2 * du2_dx)) * ds

    return abs(J_val)


def compute_vcct_energy_release_rate(stress_at_crack_tip, displacement_jump,
                                     delta_a, n_quad=8):

    x_quad, w_quad = gauss_legendre_nodes_weights(n_quad, 0.0, delta_a)

    G_I = 0.0
    G_II = 0.0
    for i in range(n_quad):
        x = x_quad[i]
        w = w_quad[i]

        sigma_22 = stress_at_crack_tip / np.sqrt(1.0 + x / delta_a)
        sigma_12 = 0.5 * stress_at_crack_tip / np.sqrt(1.0 + x / delta_a)
        du2 = displacement_jump * np.sqrt(1.0 - x / delta_a)
        du1 = 0.3 * displacement_jump * np.sqrt(1.0 - x / delta_a)

        G_I += w * sigma_22 * du2
        G_II += w * sigma_12 * du1

    G_I /= (2.0 * delta_a)
    G_II /= (2.0 * delta_a)
    return G_I, G_II


def probabilistic_strength_integral(mean_strength, std_strength, n_hermite=12):
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n_hermite)


    sigma_ln = np.sqrt(np.log(1.0 + (std_strength / mean_strength) ** 2))
    mu_ln = np.log(mean_strength) - 0.5 * sigma_ln ** 2

    E_X = 0.0
    for i in range(n_hermite):

        E_X += w[i] * np.exp(np.sqrt(2.0) * sigma_ln * x[i] + mu_ln)

    E_X /= np.sqrt(np.pi)
    return E_X


def compute_strain_energy_release_rate_quadrature(stress, strain, damage, material,
                                                   thickness, n_quad=8):
    z_nodes, z_weights = gauss_legendre_nodes_weights(n_quad, -thickness / 2.0, thickness / 2.0)

    U = 0.0
    for i in range(n_quad):
        z = z_nodes[i]
        w = z_weights[i]

        eps_z = strain * (2.0 * z / thickness)
        sigma_z = stress * (1.0 - damage) * (2.0 * z / thickness)
        U += 0.5 * w * np.dot(sigma_z, eps_z)

    return U
