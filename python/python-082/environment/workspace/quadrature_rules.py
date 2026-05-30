
import numpy as np


def quadrature_weights_vandermonde(n, a, b, x):
    x = np.asarray(x, dtype=float).flatten()
    if len(x) != n:
        raise ValueError("x must have length n.")
    if np.any(np.diff(np.sort(x)) < 1e-12):
        raise ValueError("Quadrature nodes must be distinct.")

    V = np.zeros((n, n))
    V[0, :] = 1.0
    for i in range(1, n):
        V[i, :] = V[i - 1, :] * x

    rhs = np.zeros(n)
    for i in range(n):
        power = i + 1
        rhs[i] = (b ** power - a ** power) / power


    w = np.linalg.solve(V.T, rhs)
    return w


def gauss_legendre_nodes_weights(n):
    if n < 1:
        raise ValueError("n must be >= 1.")


    if n == 1:
        return np.array([0.0]), np.array([2.0])

    i = np.arange(1.0, n)
    beta = i / np.sqrt(4.0 * i * i - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)
    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    w = 2.0 * (eigvecs[0, :] ** 2)
    return x, w


def factorial2(n):
    if n < 0:
        return 1.0
    result = 1.0
    while n > 1:
        result *= n
        n -= 2
    return result


def hermite_monomial_integral(n, option=1):
    if n < 0:
        return -np.inf
    if n % 2 == 1:
        return 0.0

    if option in (0, 1):
        value = factorial2(n - 1) * np.sqrt(np.pi) / (2.0 ** (n / 2.0))
    elif option == 2:
        value = factorial2(n - 1) * np.sqrt(2.0 * np.pi)
    elif option == 3:
        value = factorial2(n - 1) / (2.0 ** (n / 2.0))
    elif option == 4:
        value = factorial2(n - 1)
    else:
        raise ValueError("Invalid option.")
    return value


def gauss_hermite_nodes_weights(n, option=1):
    if n < 1:
        raise ValueError("n must be >= 1.")


    alpha = np.zeros(n)
    beta = np.zeros(n - 1)
    for i in range(n - 1):
        beta[i] = np.sqrt((i + 1.0) / 2.0)

    J = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals


    w = np.sqrt(np.pi) * (eigvecs[0, :] ** 2)

    if option == 2:
        x *= np.sqrt(2.0)
        w *= np.sqrt(2.0)
    elif option == 3:
        w /= np.sqrt(np.pi)
    elif option == 4:
        x *= np.sqrt(2.0)
        w /= np.sqrt(2.0 * np.pi)
        w *= np.sqrt(2.0)

    return x, w


def stochastic_fiber_failure_probability(sigma_11, mu_strength, sigma_strength,
                                         n_hermite=16):
    if sigma_strength <= 0:
        return 1.0 if sigma_11 >= mu_strength else 0.0

    x, w = gauss_hermite_nodes_weights(n_hermite, option=4)


    z = mu_strength + sigma_strength * np.sqrt(2.0) * x
    indicator = (z <= sigma_11).astype(float)
    P_f = np.sum(w * indicator) / np.sqrt(np.pi)
    return np.clip(P_f, 0.0, 1.0)


def integrate_strain_energy(element_strain, C_matrix, quad_order=4):
    xi, wi = gauss_legendre_nodes_weights(quad_order)
    energy = 0.0
    for i in range(quad_order):
        for j in range(quad_order):

            w_ij = wi[i] * wi[j]
            if callable(element_strain):
                eps = element_strain(xi[i], xi[j])
            else:
                eps = element_strain[i * quad_order + j]
            eps = np.asarray(eps, dtype=float).flatten()[:3]
            energy += w_ij * 0.5 * eps @ C_matrix @ eps
    return energy
