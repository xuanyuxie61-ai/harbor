
import numpy as np


def detq_orthogonal(a: np.ndarray, n: int) -> tuple:
    ifault = 0
    tol = 1e-10

    if n <= 0:
        return 0.0, 1

    a2 = a.flatten().copy()
    d = 1.0
    r = 0

    for k in range(1, n + 1):
        q = r
        x = a2[r]
        y = np.sign(x)
        d *= y
        y = -1.0 / (x + y)
        x = abs(x) - 1.0

        if tol < abs(x):
            if x > 0:
                ifault = 1
                return d, ifault
            if k == n:
                ifault = 1
                return d, ifault

            for i in range(k, n):
                q += n
                x = a2[q] * y
                p = r
                s = q
                for j in range(k, n):
                    p += 1
                    s += 1
                    a2[s] += x * a2[p]

        r += n + 1


    if abs(abs(d) - 1.0) > tol:
        d = np.sign(d)

    return d, ifault


def bisection_root_find(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 100) -> tuple:
    fa = f(a)
    fb = f(b)


    if fa == 0.0:
        return float(a), 0, True
    if fb == 0.0:
        return float(b), 0, True
    if fa * fb > 0.0:

        for scale in [2.0, 5.0, 10.0, 50.0, 100.0]:
            b_new = b + scale * abs(b - a)
            fb_new = f(b_new)
            if fa * fb_new <= 0.0:
                b = b_new
                fb = fb_new
                break
        else:

            return (a + b) / 2.0, 0, False

    it = 0
    while abs(b - a) > tol and it < max_iter:
        c = (a + b) / 2.0
        fc = f(c)
        it += 1

        if fc == 0.0:
            return float(c), it, True
        elif np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc

    root = (a + b) / 2.0
    converged = it < max_iter or abs(b - a) <= tol
    return float(root), it, converged


def safe_sqrt(x: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    return np.sqrt(np.maximum(x, eps))


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    return a / np.where(np.abs(b) < eps, np.sign(b + eps) * eps, b)


def check_cfl(dx: float, dy: float, u: float, v: float, c: float, nu: float, CFL_max: float = 0.8) -> float:
    dt_conv_x = dx / (abs(u) + c + 1e-14)
    dt_conv_y = dy / (abs(v) + c + 1e-14)
    dt_visc_x = dx * dx / (4.0 * nu + 1e-14)
    dt_visc_y = dy * dy / (4.0 * nu + 1e-14)

    dt = CFL_max * min(dt_conv_x, dt_conv_y, dt_visc_x, dt_visc_y)
    return float(dt)


def limiter_minmod(r: np.ndarray, theta: float = 2.0) -> np.ndarray:

    raise NotImplementedError("TODO: implement minmod limiter")
