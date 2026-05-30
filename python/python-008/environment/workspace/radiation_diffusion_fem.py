
import numpy as np
from scipy.integrate import solve_ivp


def basic_hat(x):
    x = np.asarray(x, dtype=float)
    f = ((x + 1.0) * ((-1.0 <= x) & (x < 0.0))
         + (1.0 - x) * ((0.0 <= x) & (x <= 1.0)))
    return f


def assemble_mass_matrix(n):
    h = 1.0 / n
    main = np.full(n + 1, 4.0)
    main[0] = 2.0
    main[-1] = 2.0
    off = np.ones(n)
    M = h / 6.0 * (np.diag(main, 0) + np.diag(off, -1) + np.diag(off, 1))
    return M


def assemble_stiffness_matrix(n):
    h = 1.0 / n
    main = np.full(n + 1, 2.0)
    off = np.full(n, -1.0)
    K = (1.0 / h) * (np.diag(main, 0) + np.diag(off, -1) + np.diag(off, 1))
    return K


def project_initial_condition(w0_func, n):
    h = 1.0 / n
    b = np.zeros(n + 1)


    x_left = np.linspace(0.0, h, 50)
    phi = basic_hat(n * x_left)
    b[0] = np.trapezoid(w0_func(x_left) * phi, x_left)


    x_right = np.linspace((n - 1) * h, 1.0, 50)
    phi = basic_hat(n * x_right - n)
    b[-1] = np.trapezoid(w0_func(x_right) * phi, x_right)


    for jj in range(1, n):
        x_seg = np.linspace((jj - 1) * h, (jj + 1) * h, 100)
        phi = basic_hat(n * x_seg + 1 - jj)
        b[jj] = np.trapezoid(w0_func(x_seg) * phi, x_seg)

    M = assemble_mass_matrix(n)
    u0 = np.linalg.solve(M, b)
    return u0


def nonlinear_source(u, c_array, n, M):
    c1, c2, c3, c4 = c_array


    h = 1.0 / n
    const_vec = np.full(n + 1, 1.0)
    const_vec[0] = 0.5
    const_vec[-1] = 0.5
    term_const = (c1 / n) * const_vec


    term_lin = c2 * (M @ u)


    u2 = u ** 2
    wx = (u[:-1] + u[1:]) ** 2
    term_quad = np.zeros(n + 1)
    term_quad[0] = 2 * u2[0] + wx[0]
    term_quad[-1] = wx[-1] + 2 * u2[-1]
    if n > 1:
        term_quad[1:-1] = wx[:-1] + 4 * u2[1:-1] + wx[1:]
    term_quad = c3 * term_quad / (12.0 * n)


    u3 = u ** 3
    wx3 = (u[:-1] + u[1:]) ** 3
    term_cubic = np.zeros(n + 1)
    term_cubic[0] = 3 * u3[0] + wx3[0] - u[0] * u[1] ** 2
    term_cubic[-1] = wx3[-1] + 3 * u3[-1] - u[-1] * u[-2] ** 2
    if n > 1:
        term_cubic[1:-1] = (wx3[:-1] + 6 * u3[1:-1] + wx3[1:]
                            - u[1:-1] * (u[:-2] ** 2 + u[2:] ** 2))
    term_cubic = c4 * term_cubic / (20.0 * n)

    return term_const + term_lin + term_quad + term_cubic


def solve_radiation_diffusion(n=64, t_span=(0.0, 4.0),
                              c_array=None,
                              w0_func=None):
    if c_array is None:
        c_array = np.array([0.0, -0.5, 0.0, 0.0])
    if w0_func is None:
        w0_func = lambda x: np.sin(np.pi * x)

    M = assemble_mass_matrix(n)
    K = assemble_stiffness_matrix(n)
    u0 = project_initial_condition(w0_func, n)

    def rhs(t, u):
        return np.linalg.solve(M, -K @ u + nonlinear_source(u, c_array, n, M))

    sol = solve_ivp(rhs, t_span, u0, method='BDF',
                    dense_output=True, max_step=0.05)

    t_eval = np.linspace(t_span[0], t_span[1], 100)
    u_eval = sol.sol(t_eval).T
    return t_eval, u_eval
