
import numpy as np


def midpoint_quad_2d(nx, ny, a, b, c, d, f):
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    if b <= a or d <= c:
        raise ValueError("Integration limits must satisfy a < b and c < d")

    estimate = 0.0
    for i in range(1, nx + 1):
        x = ((2 * nx - 2 * i + 1) * a + (2 * i - 1) * b) / (2 * nx)
        for j in range(1, ny + 1):
            y = ((2 * ny - 2 * j + 1) * c + (2 * j - 1) * d) / (2 * ny)
            estimate += f(x, y)

    estimate = (b - a) * (d - c) * estimate / (nx * ny)
    return estimate


def circle_rule(nt):
    if nt <= 0:
        raise ValueError("nt must be positive")
    weights = np.ones(nt, dtype=np.float64) / nt
    angles = 2.0 * np.pi * np.arange(nt) / nt
    return weights, angles


def integrate_over_circle(radius, nt, f_polar):
    weights, angles = circle_rule(nt)

    r_nodes, r_weights = np.polynomial.legendre.leggauss(5)

    r_nodes = 0.5 * radius * (r_nodes + 1.0)
    r_weights = 0.5 * radius * r_weights

    integral = 0.0
    for i in range(nt):
        theta = angles[i]
        for j in range(5):
            r = r_nodes[j]
            w = weights[i] * r_weights[j] * r
            integral += w * f_polar(r, theta)

    integral *= 2.0 * np.pi
    return integral


def gauss_legendre_integral_1d(f, a, b, n=16):
    if n <= 0:
        raise ValueError("n must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(n)

    t = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    w = 0.5 * (b - a) * weights
    return np.sum(w * f(t))


def oscillation_probability_integral_2d(
        E_min, E_max, L_min, L_max,
        nx=32, ny=32,
        theta12=None, theta23=None, theta13=None,
        delta_cp=None, delta_m2_21=None, delta_m2_31=None,
        hierarchy='normal', initial_flavor=0, final_flavor=0
):
    from pmns_matrix import build_pmns_matrix, build_mass_matrix

    U = build_pmns_matrix(theta12, theta23, theta13, delta_cp)
    M2 = build_mass_matrix(delta_m2_21, delta_m2_31, hierarchy)

    def prob_func(E, L):
        if E <= 0 or L < 0:
            return 0.0
        H = (1.0 / (2.0 * E * 1e9)) * (U @ M2 @ U.conj().T)
        L_ev_inv = L * 5.067730889e9

        eigenvalues, eigenvectors = np.linalg.eigh(H)
        D = np.diag(np.exp(-1j * eigenvalues * L_ev_inv))
        U_prop = eigenvectors @ D @ eigenvectors.conj().T

        psi0 = np.zeros(3, dtype=np.complex128)
        psi0[initial_flavor] = 1.0
        psi_L = U_prop @ psi0
        return abs(psi_L[final_flavor]) ** 2

    area = (E_max - E_min) * (L_max - L_min)
    integral = midpoint_quad_2d(nx, ny, E_min, E_max, L_min, L_max, prob_func)
    P_avg = integral / area

    return float(P_avg)


def integrate_over_delta_cp(f_delta, n_points=64):
    weights, angles = circle_rule(n_points)


    values = np.array([f_delta(a) for a in angles])
    return 2.0 * np.pi * np.mean(values)


def simpson_integral_1d(y, dx):
    n = len(y)
    if n < 3 or n % 2 == 0:

        return np.trapezoid(y, dx=dx)

    integral = y[0] + y[-1]
    integral += 4.0 * np.sum(y[1:-1:2])
    integral += 2.0 * np.sum(y[2:-1:2])
    integral *= dx / 3.0
    return integral


def adaptive_integral_1d(f, a, b, tol=1e-6, max_depth=20):
    def simpson(f, a, b):
        c = 0.5 * (a + b)
        h = b - a
        return h / 6.0 * (f(a) + 4.0 * f(c) + f(b))

    def recursive(f, a, b, eps, S, fa, fb, fc, depth):
        c = 0.5 * (a + b)
        d = 0.5 * (a + c)
        e = 0.5 * (c + b)
        fd = f(d)
        fe = f(e)
        Sleft = (c - a) / 6.0 * (fa + 4.0 * fd + fc)
        Sright = (b - c) / 6.0 * (fc + 4.0 * fe + fb)
        S2 = Sleft + Sright
        if depth >= max_depth or abs(S2 - S) <= 15 * eps:
            return S2 + (S2 - S) / 15.0
        return (recursive(f, a, c, eps / 2.0, Sleft, fa, fc, fd, depth + 1) +
                recursive(f, c, b, eps / 2.0, Sright, fc, fb, fe, depth + 1))

    c = 0.5 * (a + b)
    fa, fb, fc = f(a), f(b), f(c)
    S = simpson(f, a, b)
    return recursive(f, a, b, tol, S, fa, fb, fc, 0)
