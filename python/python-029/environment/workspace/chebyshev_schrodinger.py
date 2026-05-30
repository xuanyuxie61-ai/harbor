
import numpy as np
from scipy.integrate import solve_ivp


def chebyshev_points(N):
    j = np.arange(N + 1)
    x = np.cos(np.pi * j / N)
    return x


def chebyshev_differentiation_matrix(N):
    x = chebyshev_points(N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c = c * ((-1.0) ** np.arange(N + 1))

    X = np.tile(x, (N + 1, 1))
    dX = X - X.T

    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D


def chebyshev_second_derivative_matrix(N):
    D = chebyshev_differentiation_matrix(N)
    return D @ D


def numerov_integration(f_func, r_grid, l, u_start=1e-10):
    n = len(r_grid)
    h = r_grid[1] - r_grid[0]
    u = np.zeros(n, dtype=complex)



    u[0] = 0.0
    u[1] = u_start * (r_grid[1] ** (l + 1))

    f = np.array([f_func(ri) for ri in r_grid])


    h2_12 = h ** 2 / 12.0
    for i in range(1, n - 1):
        denom = 1.0 + h2_12 * f[i + 1]
        if abs(denom) < 1e-30:
            denom = 1e-30
        u[i + 1] = (2.0 * (1.0 - 5.0 * h2_12 * f[i]) * u[i]
                    - (1.0 + h2_12 * f[i - 1]) * u[i - 1]) / denom

    return u


def spherical_bessel_ln(x, l):
    x = float(x)
    if x < 1e-12:
        return (1.0 if l == 0 else 0.0), -1e12

    j = np.zeros(l + 1)
    j[0] = np.sin(x) / x
    if l >= 1:
        j[1] = np.sin(x) / (x ** 2) - np.cos(x) / x
    for ll in range(2, l + 1):
        j[ll] = (2.0 * ll - 1.0) / x * j[ll - 1] - j[ll - 2]

    n = np.zeros(l + 1)
    n[0] = -np.cos(x) / x
    if l >= 1:
        n[1] = -np.cos(x) / (x ** 2) - np.sin(x) / x
    for ll in range(2, l + 1):
        n[ll] = (2.0 * ll - 1.0) / x * n[ll - 1] - n[ll - 2]

    return j[l], n[l]


def spherical_bessel_derivative(x, l, kind='j'):
    x = float(x)
    if x < 1e-12:
        return 0.0
    if kind == 'j':
        fl, _ = spherical_bessel_ln(x, l)
        f_prev = np.sin(x) / x if l == 0 else spherical_bessel_ln(x, l - 1)[0]
    else:
        _, fl = spherical_bessel_ln(x, l)
        f_prev = -np.cos(x) / x if l == 0 else spherical_bessel_ln(x, l - 1)[1]
    return f_prev - (l + 1.0) / x * fl


def solve_radial_schrodinger(params, l, j=None, n_points=2000, r_max=15.0, r_match=10.0):
    from optical_potential import effective_potential


    t = np.linspace(0.0, 1.0, n_points)

    r = r_max * (t ** 1.2)
    r[0] = 1e-8


    def f_func(ri):
        V_eff = effective_potential(np.array([ri]), params, l, j)[0]
        return params.k ** 2 - V_eff


    u = numerov_integration(f_func, r, l)


    idx_match = np.argmin(np.abs(r - r_match))
    if idx_match < 5:
        idx_match = 5
    if idx_match >= n_points - 2:
        idx_match = n_points - 3


    h_m = r[idx_match + 1] - r[idx_match - 1]
    u_deriv_match = (u[idx_match + 1] - u[idx_match - 1]) / h_m
    u_match = u[idx_match]



    idx_cheb_start = max(0, idx_match - 15)
    idx_cheb_end = min(n_points, idx_match + 16)
    r_cheb = r[idx_cheb_start:idx_cheb_end]
    u_cheb = u[idx_cheb_start:idx_cheb_end]
    if len(r_cheb) >= 4:
        N_cheb = len(r_cheb) - 1
        D_cheb = chebyshev_differentiation_matrix(N_cheb)


        r_min_c, r_max_c = r_cheb[0], r_cheb[-1]
        x_cheb = 2.0 * (r_cheb - r_min_c) / (r_max_c - r_min_c) - 1.0

        du_dx = D_cheb @ u_cheb
        du_dr_cheb = du_dx * 2.0 / (r_max_c - r_min_c)

        idx_local = np.argmin(np.abs(r_cheb - r_match))
        cheb_deriv_check = du_dr_cheb[idx_local]
    else:
        cheb_deriv_check = u_deriv_match


    kr = params.k * r_match
    jl, nl = spherical_bessel_ln(kr, l)
    Rl = kr * jl
    Sl = kr * nl
    Rl_deriv = jl + kr * spherical_bessel_derivative(kr, l, 'j')
    Sl_deriv = nl + kr * spherical_bessel_derivative(kr, l, 'n')






    L_int = u_deriv_match / u_match
    S_l = 1.0 + 0j
    delta_l = 0.0



    norm_factor = 1.0 / np.max(np.abs(u))
    u_norm = u * norm_factor

    return {
        'r': r,
        'u': u_norm,
        'phase_shift': delta_l,
        'S_matrix': S_l,
        'absorption': abs(S_l),
        'log_derivative': L_int,
        'cheb_deriv_check': cheb_deriv_check,
        'k': params.k,
        'l': l,
        'j': j,
    }


def riccati_bessel_functions(kr, l_max):
    Rl = np.zeros(l_max + 1)
    Rl_prime = np.zeros(l_max + 1)
    Sl = np.zeros(l_max + 1)
    Sl_prime = np.zeros(l_max + 1)

    for l in range(l_max + 1):
        jl, nl = spherical_bessel_ln(kr, l)
        Rl[l] = kr * jl
        Sl[l] = kr * nl
        Rl_prime[l] = jl + kr * spherical_bessel_derivative(kr, l, 'j')
        Sl_prime[l] = nl + kr * spherical_bessel_derivative(kr, l, 'n')

    return Rl, Rl_prime, Sl, Sl_prime


def chebyshev_spectral_verify(u_func, r_interval, N=32):
    r_min, r_max = r_interval
    x_cheb = chebyshev_points(N)

    r_cheb = 0.5 * (r_max - r_min) * (x_cheb + 1.0) + r_min
    u_vals = u_func(r_cheb)

    D = chebyshev_differentiation_matrix(N)
    du_dx = D @ u_vals
    du_dr = du_dx * 2.0 / (r_max - r_min)


    du_dr_fd = np.zeros_like(du_dr)
    for i in range(1, N):
        du_dr_fd[i] = (u_vals[i + 1] - u_vals[i - 1]) / (r_cheb[i + 1] - r_cheb[i - 1])
    du_dr_fd[0] = (u_vals[1] - u_vals[0]) / (r_cheb[1] - r_cheb[0])
    du_dr_fd[N] = (u_vals[N] - u_vals[N - 1]) / (r_cheb[N] - r_cheb[N - 1])

    return np.max(np.abs(du_dr - du_dr_fd))


if __name__ == "__main__":
    from optical_potential import OpticalPotentialParameters
    params = OpticalPotentialParameters('n', 56, 26, 14.0)
    res = solve_radial_schrodinger(params, l=0, n_points=1500)
    print(f"l=0: δ = {res['phase_shift']:.6f} rad, |S| = {res['absorption']:.6f}")
    res2 = solve_radial_schrodinger(params, l=2, j=2.5, n_points=1500)
    print(f"l=2,j=2.5: δ = {res2['phase_shift']:.6f} rad, |S| = {res2['absorption']:.6f}")


    u_func = lambda r: np.sin(params.k * r) * np.exp(-r / 5.0)
    err = chebyshev_spectral_verify(u_func, (0.5, 8.0), N=24)
    print(f"Chebyshev 谱导数最大误差: {err:.2e}")
