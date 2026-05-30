
import numpy as np
from math import sqrt, exp, fabs


HBARC = 197.3269804
M_NUCLEON = 939.0


def numerov_integrate(r_grid, f_values, u0, u1):
    N = len(r_grid)
    h = r_grid[1] - r_grid[0]
    h2 = h * h
    h12 = h2 / 12.0

    u = np.zeros(N)
    u[0] = u0
    u[1] = u1

    for n in range(1, N - 1):
        denom = 1.0 + h12 * f_values[n + 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        u[n + 1] = ((2.0 * (1.0 - 5.0 * h12 * f_values[n]) * u[n] -
                     (1.0 + h12 * f_values[n - 1]) * u[n - 1]) / denom)
    return u


def compute_radial_wavefunction(r_grid, potential, E, l, mass=M_NUCLEON):
    N = len(r_grid)
    f_values = np.zeros(N)
    prefactor = 2.0 * mass / (HBARC ** 2)

    for i in range(N):
        r = r_grid[i]



        centrifugal = 0.0
        f_values[i] = prefactor * (potential[i] - E) + centrifugal


    h = r_grid[1] - r_grid[0]
    u0 = 0.0
    u1 = h ** (l + 1)

    u = numerov_integrate(r_grid, f_values, u0, u1)
    return u, f_values


def wavefunction_logarithmic_derivative(u, r_grid):
    N = len(r_grid)
    if abs(u[N - 1]) < 1e-30:
        return 1e30

    h = r_grid[1] - r_grid[0]
    dudr = (3.0 * u[N - 1] - 4.0 * u[N - 2] + u[N - 3]) / (2.0 * h)
    return dudr / u[N - 1]


def brent_root_find(a, b, t, func):
    calls = 0
    sa, sb = a, b
    fa = func(sa)
    calls += 1
    fb = func(sb)
    calls += 1

    if fa * fb > 0:
        raise ValueError("区间端点必须变号")

    c, fc = sa, fa
    e = sb - sa
    d = e

    while True:
        if abs(fc) < abs(fb):
            sa, sb, c = sb, c, sa
            fa, fb, fc = fb, fc, fa

        tol = 2.0 * np.finfo(float).eps * abs(sb) + t
        m = 0.5 * (c - sb)

        if abs(m) <= tol or fb == 0.0:
            break

        if abs(e) < tol or abs(fa) <= abs(fb):
            e = m
            d = e
        else:
            s = fb / fa
            if sa == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (sb - sa) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q
            else:
                p = -p

            s = e
            e = d

            if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(0.5 * s * q):
                d = p / q
            else:
                e = m
                d = e

        sa = sb
        fa = fb

        if abs(d) > tol:
            sb = sb + d
        elif m > 0.0:
            sb = sb + tol
        else:
            sb = sb - tol

        fb = func(sb)
        calls += 1

        if (fb > 0.0 and fc > 0.0) or (fb <= 0.0 and fc <= 0.0):
            c = sa
            fc = fa
            e = sb - sa
            d = e

    return sb, calls


def find_bound_state_energy(r_grid, potential, l, E_min, E_max, tol=1e-6):
    def mismatch(E):
        u, _ = compute_radial_wavefunction(r_grid, potential, E, l)

        return u[-1]


    f_min = mismatch(E_min)
    f_max = mismatch(E_max)

    if f_min * f_max > 0:


        E_test = np.linspace(E_min, E_max, 200)
        best_E = E_min
        best_val = abs(f_min)
        for Et in E_test:
            val = abs(mismatch(Et))
            if val < best_val:
                best_val = val
                best_E = Et
        u_best, _ = compute_radial_wavefunction(r_grid, potential, best_E, l)
        return best_E, u_best, 200

    E_bound, n_calls = brent_root_find(E_min, E_max, tol, mismatch)
    u_bound, _ = compute_radial_wavefunction(r_grid, potential, E_bound, l)


    h = r_grid[1] - r_grid[0]
    norm = sqrt(np.trapezoid(u_bound ** 2, r_grid))
    if norm > 0:
        u_bound = u_bound / norm

    return E_bound, u_bound, n_calls


def solve_all_bound_states(r_grid, potential, l, n_max_states=5,
                           E_search_min=-60.0, E_search_max=-1.0):
    energies = []
    wavefunctions = []

    n_probe = 300
    E_probe = np.linspace(E_search_min, E_search_max, n_probe)


    u_ends = []
    for E in E_probe:
        u, _ = compute_radial_wavefunction(r_grid, potential, E, l)
        u_ends.append(u[-1])
    u_ends = np.array(u_ends)


    sign_changes = []
    for i in range(n_probe - 1):
        if u_ends[i] * u_ends[i + 1] < 0:
            sign_changes.append((E_probe[i], E_probe[i + 1]))

    for (E_a, E_b) in sign_changes[:n_max_states]:
        try:
            E_bnd, u_bnd, _ = find_bound_state_energy(r_grid, potential, l,
                                                        E_a, E_b, tol=1e-5)
            energies.append(E_bnd)
            wavefunctions.append(u_bnd)
        except ValueError:
            continue

    return energies, wavefunctions
