
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve


def musiela_drift(sigma_funcs, T, t=0.0):
    if T <= t:
        return 0.0


    gl_nodes = np.array([
        0.013046735741414139, 0.06746831665550773, 0.1602952158504878,
        0.2833023029353765, 0.4255628305091844, 0.5744371694908156,
        0.7166976970646235, 0.8397047841495122, 0.9325316833444923,
        0.9869532642585859
    ])
    gl_weights = np.array([
        0.03333567215434411, 0.0747256745752903, 0.109543181257991,
        0.1346333596549982, 0.1477621123573764, 0.1477621123573764,
        0.1346333596549982, 0.109543181257991, 0.0747256745752903,
        0.03333567215434411
    ])

    alpha = 0.0
    scale = T - t
    for sigma in sigma_funcs:
        sigma_T = sigma(t, T)
        u = t + scale * gl_nodes
        integrand = np.array([sigma(t, ui) for ui in u])
        integral = scale * np.dot(gl_weights, integrand)
        alpha += sigma_T * integral
    return alpha


def forward_rate_pde_rhs(t, T_grid, f_current, nu, mu_func, forcing_func,
                         sigma_funcs):
    T_grid = np.asarray(T_grid, dtype=float)
    f_current = np.asarray(f_current, dtype=float)
    N = len(T_grid)

    if N < 3:
        raise ValueError("forward_rate_pde_rhs: T_grid 至少要有 3 个点")
    if nu < 0.0:
        raise ValueError("forward_rate_pde_rhs: 扩散系数 nu 必须非负")

    dT = np.diff(T_grid)
    if not np.allclose(dT, dT[0], atol=1e-12):

        dfdt = np.zeros(N, dtype=float)
        for j in range(1, N - 1):
            h_m = T_grid[j] - T_grid[j - 1]
            h_p = T_grid[j + 1] - T_grid[j]
            h_avg = (h_m + h_p) / 2.0

            df_dT = (f_current[j + 1] - f_current[j - 1]) / (h_m + h_p)
            d2f_dT2 = (f_current[j + 1] / (h_p * h_avg) -
                       f_current[j] / (h_m * h_p) +
                       f_current[j - 1] / (h_m * h_avg))

            mu_j = mu_func(T_grid[j])
            forcing_j = forcing_func(t, T_grid[j])


            alpha_j = musiela_drift(sigma_funcs, T_grid[j], t)

            dfdt[j] = -df_dT + nu * d2f_dT2 + mu_j * df_dT + forcing_j + alpha_j
    else:







        dfdt = np.zeros(N, dtype=float)
        raise NotImplementedError("HOLE_1: 均匀网格 PDE 空间离散尚未实现")





    A_fd = sp.coo_matrix((N, N)).tocsr()



    rhs_forcing = np.zeros(N, dtype=float)

    return dfdt, A_fd, rhs_forcing


def bond_price_from_forward(f_curve, T_grid, t, T):
    if T < t:
        raise ValueError("bond_price_from_forward: T 必须 >= t")
    if T == t:
        return 1.0

    f_curve = np.asarray(f_curve, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)


    mask = (T_grid >= t - 1e-14) & (T_grid <= T + 1e-14)
    T_sub = T_grid[mask]
    f_sub = f_curve[mask]

    if len(T_sub) < 2:

        if len(T_sub) == 1:
            integral = f_sub[0] * (T - t)
        else:

            idx = np.argmin(np.abs(T_grid - (t + T) / 2.0))
            integral = f_curve[idx] * (T - t)
    else:
        integral = np.trapezoid(f_sub, T_sub)

    price = np.exp(-integral)
    price = np.clip(price, 0.0, 1.0)
    return price


def zero_yield_from_forward(f_curve, T_grid, t, T):
    if T <= t:
        raise ValueError("zero_yield_from_forward: T 必须 > t")

    P = bond_price_from_forward(f_curve, T_grid, t, T)
    if P <= 0.0:
        P = 1e-300
    y = -np.log(P) / (T - t)
    return max(y, 0.0)


def instantaneous_short_rate(f_curve, T_grid):
    f_curve = np.asarray(f_curve, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    if len(T_grid) < 2:
        return float(f_curve[0]) if len(f_curve) > 0 else 0.0

    r = f_curve[0]

    return max(r, 0.0)


def solve_term_structure_pde(T_grid, f_init, t_max, dt,
                             nu, mu_func, forcing_func, sigma_funcs,
                             r_short_func, r_long_func):
    T_grid = np.asarray(T_grid, dtype=float)
    f_init = np.asarray(f_init, dtype=float)
    N = len(T_grid)
    n_steps = int(np.ceil(t_max / dt))
    dt = t_max / n_steps

    if N < 3:
        raise ValueError("solve_term_structure_pde: T_grid 至少要有 3 个点")

    t_history = np.zeros(n_steps + 1, dtype=float)
    f_history = np.zeros((n_steps + 1, N), dtype=float)
    bond_prices = np.zeros((n_steps + 1, N), dtype=float)

    f = f_init.copy()
    t_history[0] = 0.0
    f_history[0, :] = f


    for j in range(N):
        bond_prices[0, j] = bond_price_from_forward(f, T_grid, 0.0, T_grid[j])

    for step in range(n_steps):
        t = step * dt
        t_new = (step + 1) * dt


        _, A_fd, rhs_f = forward_rate_pde_rhs(t, T_grid, f, nu, mu_func,
                                               forcing_func, sigma_funcs)


        I = sp.eye(N, format='csr')
        lhs = I - dt * A_fd
        rhs = f + dt * rhs_f


        lhs = lhs.tolil()
        lhs[0, :] = 0.0
        lhs[0, 0] = 1.0
        rhs[0] = r_short_func(t_new)

        lhs[N - 1, :] = 0.0
        lhs[N - 1, N - 1] = 1.0
        rhs[N - 1] = r_long_func(t_new)
        lhs = lhs.tocsr()

        f_new = spsolve(lhs, rhs)
        if f_new is None:
            raise RuntimeError("solve_term_structure_pde: 稀疏求解失败")

        f = np.asarray(f_new, dtype=float)

        f = np.clip(f, 0.0, None)

        t_history[step + 1] = t_new
        f_history[step + 1, :] = f

        for j in range(N):
            bond_prices[step + 1, j] = bond_price_from_forward(f, T_grid, t_new, T_grid[j])

    return t_history, f_history, bond_prices
