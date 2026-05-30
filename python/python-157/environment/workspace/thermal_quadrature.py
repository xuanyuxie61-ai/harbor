import numpy as np
from combustion_utils import check_positive, check_nonnegative


def square_symq_rule(degree):
    if not (0 <= degree <= 20):
        raise ValueError("degree must be in [0, 20]")

    n = degree + 1

    xi_1d, wi_1d = np.polynomial.legendre.leggauss(n)


    x = np.zeros(n * n)
    y = np.zeros(n * n)
    w = np.zeros(n * n)
    idx = 0
    for i in range(n):
        for j in range(n):
            x[idx] = xi_1d[i]
            y[idx] = xi_1d[j]
            w[idx] = wi_1d[i] * wi_1d[j]
            idx += 1
    return x, y, w


def integrate_square(func, degree=5):
    x, y, w = square_symq_rule(degree)
    vals = func(x, y)
    return np.sum(w * vals)


def integrate_thermal_source(lambda_field, T_field, rho_field,
                             dx, dy, degree=5,
                             A=1.0e8, Ea=8.314e4, Q=2.5e6,
                             R=8.314462618):
    check_positive(dx, "dx")
    check_positive(dy, "dy")
    lambda_field = np.asarray(lambda_field, dtype=float)
    T_field = np.asarray(T_field, dtype=float)
    rho_field = np.asarray(rho_field, dtype=float)

    if lambda_field.shape != T_field.shape or lambda_field.shape != rho_field.shape:
        raise ValueError("Field arrays must have the same shape")

    nx, ny = lambda_field.shape

    x_nodes, y_nodes, w_nodes = square_symq_rule(degree)

    total = 0.0
    for i in range(nx):
        for j in range(ny):

            xc = (i + 0.5) * dx
            yc = (j + 0.5) * dy

            lam = max(0.0, min(1.0, lambda_field[i, j]))
            T = max(T_field[i, j], 1.0e-6)
            rho = max(rho_field[i, j], 1.0e-9)
            k = A * np.exp(-Ea / (R * T))
            rate = rho * k * ((1.0 - lam) ** 1.0)
            cell_area = dx * dy
            total += Q * rate * cell_area

    return total


def average_temperature_profile(T_field, dx, dy, degree=5):
    nx, ny = T_field.shape
    area = nx * dx * ny * dy
    integral = np.sum(T_field) * dx * dy
    return integral / area if area > 0.0 else 0.0
