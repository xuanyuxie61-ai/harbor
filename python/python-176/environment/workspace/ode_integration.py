
import numpy as np


def explicit_euler(f, y0, t_span, n_steps):
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    dim = len(np.atleast_1d(y0))
    t_array = np.linspace(t0, t1, n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim), dtype=float)
    y_array[0] = y0

    for n in range(n_steps):
        y_array[n + 1] = y_array[n] + h * f(t_array[n], y_array[n])

    return t_array, y_array


def implicit_euler_linear(M, A, b_fn, y0, t_span, n_steps):
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    dim = len(np.atleast_1d(y0))
    t_array = np.linspace(t0, t1, n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim), dtype=float)
    y_array[0] = y0

    LHS = M + h * A
    try:

        import scipy.linalg as la
        for n in range(n_steps):
            rhs = M @ y_array[n] + h * b_fn(t_array[n + 1])
            y_array[n + 1] = la.solve(LHS, rhs, assume_a='pos')
    except Exception:

        for n in range(n_steps):
            rhs = M @ y_array[n] + h * b_fn(t_array[n + 1])
            y_array[n + 1] = np.linalg.solve(LHS, rhs)

    return t_array, y_array


def crank_nicolson_linear(M, A, b_fn, y0, t_span, n_steps):
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    dim = len(np.atleast_1d(y0))
    t_array = np.linspace(t0, t1, n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim), dtype=float)
    y_array[0] = y0

    LHS = M + 0.5 * h * A
    RHS_mat = M - 0.5 * h * A
    for n in range(n_steps):
        rhs = RHS_mat @ y_array[n] + 0.5 * h * (b_fn(t_array[n]) + b_fn(t_array[n + 1]))
        y_array[n + 1] = np.linalg.solve(LHS, rhs)

    return t_array, y_array


def trapezoid_integrate(f_vals, h):
    n = len(f_vals) - 1
    if n < 1:
        return 0.0
    val = 0.5 * f_vals[0] + 0.5 * f_vals[-1] + np.sum(f_vals[1:-1])
    return h * val


def sensitive_ode_rhs(t, y):
    y = np.atleast_1d(y)
    return np.array([y[1], y[0]], dtype=float)


def sensitive_ode_exact(t, epsilon=0.0):
    t = np.atleast_1d(t)
    c1 = 0.5 * epsilon
    c2 = 1.0 + 0.5 * epsilon
    y1 = c1 * np.exp(t) + c2 * np.exp(-t)
    y2 = c1 * np.exp(t) - c2 * np.exp(-t)
    return np.column_stack((y1, y2))


def verify_adjoint_consistency(y_state, p_adjoint, M, dt):
    n_steps = len(y_state) - 1
    lhs = 0.0
    for n in range(n_steps):
        dy = y_state[n + 1] - y_state[n]
        dp = p_adjoint[n] - p_adjoint[n + 1]
        lhs += np.dot(p_adjoint[n + 1], M @ dy) + np.dot(y_state[n], M @ dp)

    rhs = np.dot(p_adjoint[-1], M @ y_state[-1]) - np.dot(p_adjoint[0], M @ y_state[0])
    return abs(lhs - rhs)
