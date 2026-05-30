
import numpy as np
from scipy.special import jv, jvp, kv, kvp


def zero_laguerre(x0, degree, abserr, kmax, f):
    if degree < 2:
        degree = 2

    x = x0
    ierror = 0
    k = 0
    beta = 1.0 / (degree - 1.0)

    while True:
        fx = f(x, 0)
        if abs(fx) <= abserr:
            break

        dfx = f(x, 1)
        d2fx = f(x, 2)

        k += 1
        if k > kmax:
            ierror = 2
            return x, ierror, k

        z = dfx ** 2 - (beta + 1.0) * fx * d2fx
        z = max(z, 0.0)

        bot = beta * dfx + np.sqrt(z)
        if abs(bot) < 1e-30:
            ierror = 3
            return x, ierror, k

        dx = -(beta + 1.0) * fx / bot
        x = x + dx

        if not np.isfinite(x):
            ierror = 4
            return x, ierror, k

    return x, ierror, k


def fiber_mode_characteristic_eq(u, V, l_mode, n_core, n_clad):
    if u <= 0 or u >= V:
        return 1e10, 0.0, 0.0

    w = np.sqrt(max(V ** 2 - u ** 2, 0.0))
    if w < 1e-15:
        return 1e10, 0.0, 0.0


    Ju = jv(l_mode, u)
    Jpu = jvp(l_mode, u, 1)
    Kw = kv(l_mode, w)
    Kpw = kvp(l_mode, w, 1)

    if abs(Ju) < 1e-30 or abs(Kw) < 1e-30:
        return 1e10, 0.0, 0.0








    term1 = u * Jpu / Ju
    term2 = w * Kpw / Kw
    f_val = term1 - term2


    du = max(1e-8 * u, 1e-10)
    fu_plus = fiber_mode_characteristic_eq_scalar(u + du, V, l_mode)
    fu_minus = fiber_mode_characteristic_eq_scalar(u - du, V, l_mode)
    fu_plus2 = fiber_mode_characteristic_eq_scalar(u + 2 * du, V, l_mode)
    fu_minus2 = fiber_mode_characteristic_eq_scalar(u - 2 * du, V, l_mode)

    f_prime = (fu_plus - fu_minus) / (2.0 * du)
    f_double_prime = (fu_plus2 - 2.0 * f_val + fu_minus2) / (du ** 2)

    return f_val, f_prime, f_double_prime


def fiber_mode_characteristic_eq_scalar(u, V, l_mode):
    if u <= 0 or u >= V:
        return 1e10
    w = np.sqrt(max(V ** 2 - u ** 2, 0.0))
    if w < 1e-15:
        return 1e10
    Ju = jv(l_mode, u)
    Jpu = jvp(l_mode, u, 1)
    Kw = kv(l_mode, w)
    Kpw = kvp(l_mode, w, 1)
    if abs(Ju) < 1e-30 or abs(Kw) < 1e-30:
        return 1e10
    return u * Jpu / Ju - w * Kpw / Kw


def find_fiber_mode_roots(V, l_mode, n_core, n_clad, n_roots=5):
    roots = []
    betas = []
    k0 = 2.0 * np.pi / 1550e-9


    n_scan = max(200, int(V * 50))
    u_scan = np.linspace(0.01 * V, 0.99 * V, n_scan)
    f_scan = np.array([fiber_mode_characteristic_eq_scalar(u, V, l_mode) for u in u_scan])


    for i in range(n_scan - 1):
        if np.isfinite(f_scan[i]) and np.isfinite(f_scan[i + 1]):
            if f_scan[i] * f_scan[i + 1] < 0:
                x0 = 0.5 * (u_scan[i] + u_scan[i + 1])


                u_left = u_scan[i]
                u_right = u_scan[i + 1]
                f_left = f_scan[i]
                f_right = f_scan[i + 1]
                root = None
                for _ in range(60):
                    u_mid = 0.5 * (u_left + u_right)
                    f_mid = fiber_mode_characteristic_eq_scalar(u_mid, V, l_mode)
                    if f_mid == 0:
                        root = u_mid
                        break
                    if f_left * f_mid < 0:
                        u_right = u_mid
                        f_right = f_mid
                    else:
                        u_left = u_mid
                        f_left = f_mid
                if root is None:
                    root = 0.5 * (u_left + u_right)

                if 0 < root < V:

                    is_new = True
                    for r in roots:
                        if abs(r - root) < 1e-6:
                            is_new = False
                            break
                    if is_new:
                        roots.append(root)

                        a = 4e-6
                        w = np.sqrt(V ** 2 - root ** 2)
                        beta = np.sqrt((k0 * n_core) ** 2 - (root / a) ** 2)
                        betas.append(beta)

        if len(roots) >= n_roots:
            break

    return roots, betas


def degree_estimation(V):
    return min(max(int(V), 2), 20)
