
import numpy as np


def hat_function(x, x_left, x_center, x_right):
    x = np.asarray(x, dtype=float)
    val = np.zeros_like(x)
    mask1 = (x >= x_left) & (x <= x_center)
    if x_center > x_left:
        val[mask1] = (x[mask1] - x_left) / (x_center - x_left)
    mask2 = (x > x_center) & (x <= x_right)
    if x_right > x_center:
        val[mask2] = (x_right - x[mask2]) / (x_right - x_center)
    return val


def data_bracket(mesh, x_data):
    mesh = np.asarray(mesh, dtype=float)
    x_data = np.asarray(x_data, dtype=float)
    indices = np.searchsorted(mesh, x_data, side='right') - 1
    indices = np.clip(indices, 0, len(mesh) - 2)
    return indices


def fem1d_approximate(mesh, x_data, y_data, weight_approx=1.0,
                       weight_deriv=0.01, weight_boundary=1e6,
                       boundary_values=None):
    mesh = np.asarray(mesh, dtype=float)
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    n_nodes = len(mesh)
    n_data = len(x_data)

    if n_data == 0:
        return np.zeros(n_nodes)


    A = np.zeros((n_data, n_nodes))
    for i in range(n_nodes):
        x_l = mesh[max(0, i - 1)]
        x_c = mesh[i]
        x_r = mesh[min(n_nodes - 1, i + 1)]
        A[:, i] = hat_function(x_data, x_l, x_c, x_r)


    D = np.zeros((n_nodes - 2, n_nodes))
    for i in range(n_nodes - 2):
        h1 = mesh[i + 1] - mesh[i]
        h2 = mesh[i + 2] - mesh[i + 1]
        if h1 > 0 and h2 > 0:
            D[i, i] = 1.0 / h1
            D[i, i + 1] = -1.0 / h1 - 1.0 / h2
            D[i, i + 2] = 1.0 / h2


    M = weight_approx * (A.T @ A) + weight_deriv * (D.T @ D)
    rhs = weight_approx * (A.T @ y_data)


    if boundary_values is not None:
        y_left, y_right = boundary_values
        B = np.zeros((2, n_nodes))
        B[0, 0] = 1.0
        B[1, -1] = 1.0
        b_vec = np.array([y_left, y_right])
        M += weight_boundary * (B.T @ B)
        rhs += weight_boundary * (B.T @ b_vec)


    try:
        coeffs = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(M, rhs, rcond=None)[0]

    return coeffs


def fem1d_evaluate(x, mesh, coeffs):
    x = np.asarray(x, dtype=float)
    mesh = np.asarray(mesh, dtype=float)
    y = np.zeros_like(x)
    for i in range(len(mesh)):
        x_l = mesh[max(0, i - 1)]
        x_c = mesh[i]
        x_r = mesh[min(len(mesh) - 1, i + 1)]
        y += coeffs[i] * hat_function(x, x_l, x_c, x_r)
    return y


def test_fem_approximation():
    mesh = np.linspace(0, 1, 21)
    x_data = np.random.rand(100)
    y_data = np.sin(2 * np.pi * x_data) + 0.1 * np.random.randn(100)
    coeffs = fem1d_approximate(mesh, x_data, y_data,
                                weight_approx=1.0, weight_deriv=0.1,
                                weight_boundary=1e4,
                                boundary_values=(0.0, 0.0))
    x_test = np.linspace(0, 1, 200)
    y_fit = fem1d_evaluate(x_test, mesh, coeffs)
    y_exact = np.sin(2 * np.pi * x_test)
    err = np.mean((y_fit - y_exact) ** 2)
    print(f"[fem_approximation] FEM fit MSE = {err:.3e}")


if __name__ == "__main__":
    test_fem_approximation()
