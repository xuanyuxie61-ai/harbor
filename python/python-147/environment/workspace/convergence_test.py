
import numpy as np


def manufactured_solution_1(t, x):
    c = 1.0
    return np.sin(x - c * t) * np.exp(-0.1 * t)


def manufactured_forcing_1(t, x):
    c = 1.0
    alpha = 0.1
    s = np.sin(x - c * t)
    co = np.cos(x - c * t)
    e = np.exp(-alpha * t)

    u = s * e
    u_t = (-c * co - alpha * s) * e
    u_x = co * e
    u_xx = -s * e
    u_xxx = -co * e
    u_xxxx = s * e

    return u_t + u * u_x + u_xx + u_xxxx


def manufactured_solution_2(t, x):
    return np.exp(-(x - 16.0 * np.pi - t) ** 2 / 8.0)


def manufactured_forcing_2(t, x):
    c = 1.0
    sigma2 = 8.0
    xi = x - 16.0 * np.pi - c * t
    u = np.exp(-xi ** 2 / sigma2)
    u_t = u * (2.0 * c * xi / sigma2)
    u_x = u * (-2.0 * xi / sigma2)
    u_xx = u * ((4.0 * xi ** 2 / sigma2 ** 2) - (2.0 / sigma2))
    u_xxx = u * ((-8.0 * xi ** 3 / sigma2 ** 3) + (12.0 * xi / sigma2 ** 2))
    u_xxxx = u * ((16.0 * xi ** 4 / sigma2 ** 4)
                  - (48.0 * xi ** 2 / sigma2 ** 3)
                  + (12.0 / sigma2 ** 2))
    return u_t + u * u_x + u_xx + u_xxxx


def manufactured_solution_3(t, x):
    return np.sin(x) * np.cos(t) * np.exp(-0.05 * t)


def manufactured_forcing_3(t, x):
    alpha = 0.05
    s = np.sin(x)
    ct = np.cos(t)
    st = np.sin(t)
    e = np.exp(-alpha * t)

    u = s * ct * e
    u_t = s * (-st - alpha * ct) * e
    u_x = np.cos(x) * ct * e
    u_xx = -s * ct * e
    u_xxxx = s * ct * e

    return u_t + u * u_x + u_xx + u_xxxx


def compute_pin_error(u_pred, u_exact):
    diff = u_pred - u_exact
    l2_exact = np.sqrt(np.mean(u_exact ** 2))
    l2_abs = np.sqrt(np.mean(diff ** 2))
    l2_rel = l2_abs / (l2_exact + 1e-12)
    linf_abs = np.max(np.abs(diff))
    linf_rel = linf_abs / (np.max(np.abs(u_exact)) + 1e-12)
    mse = np.mean(diff ** 2)
    return {
        'l2_abs': l2_abs,
        'l2_rel': l2_rel,
        'linf_abs': linf_abs,
        'linf_rel': linf_rel,
        'mse': mse,
    }


def run_convergence_test(network, solver, test_id=1, nt_test=20, nx_test=64):
    L_domain = 32.0 * np.pi
    tmax = 5.0

    if test_id == 1:
        u_exact_fn = manufactured_solution_1
        f_fn = manufactured_forcing_1
    elif test_id == 2:
        u_exact_fn = manufactured_solution_2
        f_fn = manufactured_forcing_2
    elif test_id == 3:
        u_exact_fn = manufactured_solution_3
        f_fn = manufactured_forcing_3
    else:
        raise ValueError("test_id must be 1, 2, or 3")

    t_test = np.linspace(0.0, tmax, nt_test)
    x_test = np.linspace(0.0, L_domain, nx_test, endpoint=False)
    T_grid, X_grid = np.meshgrid(t_test, x_test, indexing='ij')
    X_query = np.column_stack([T_grid.ravel(), X_grid.ravel()])

    u_pred = network.predict(X_query).ravel()
    u_exact = u_exact_fn(X_query[:, 0], X_query[:, 1])

    errors = compute_pin_error(u_pred, u_exact)
    return errors, u_pred, u_exact, X_query


def dg1d_reference_comparison(u_dg, u_pinn, x_grid):
    if len(u_dg) != len(u_pinn) or len(u_dg) != len(x_grid):
        raise ValueError("Length mismatch")
    diff = u_dg - u_pinn
    l2 = np.sqrt(np.mean(diff ** 2))
    linf = np.max(np.abs(diff))
    return {'l2': l2, 'linf': linf}
