
import numpy as np


def lambert_w_approx(x, branch=0):
    x = np.atleast_1d(x).astype(float)
    w = np.zeros_like(x)


    em1 = -1.0 / np.e

    for idx in range(x.size):
        xv = x.flat[idx]

        if branch == 0:

            if xv < em1:

                w.flat[idx] = np.nan
                continue
            if xv < -0.323581708061267:

                p = np.sqrt(2.0 * (np.e * xv + 1.0))
                w.flat[idx] = -1.0 + p - p * p / 3.0 + 11.0 * p ** 3 / 72.0
            elif xv < 1.857183860207835:

                w.flat[idx] = (0.665 * (1.0 + 0.0195 * xv) * np.log(1.0 + xv) +
                               0.04 * xv)
                if xv > 0:
                    w.flat[idx] = np.log(1.0 + xv) * (1.0 - np.log(1.0 + xv) / 3.0)
            else:

                lx = np.log(xv)
                llx = np.log(lx)
                w.flat[idx] = lx - llx + llx / lx
        else:

            if xv < em1 or xv > 0.0:
                w.flat[idx] = np.nan
                continue
            if xv < -0.323581708061267:
                p = -np.sqrt(2.0 * (np.e * xv + 1.0))
                w.flat[idx] = -1.0 + p - p * p / 3.0 + 11.0 * p ** 3 / 72.0
            else:

                lx = np.log(-xv)
                llx = np.log(-lx)
                w.flat[idx] = lx - llx + llx / lx


    for idx in range(x.size):
        if not np.isfinite(w.flat[idx]):
            continue
        wv = w.flat[idx]
        ew = np.exp(wv)
        we = wv * ew - x.flat[idx]
        g = (wv + 2.0) * we / (2.0 * wv + 2.0)
        denom = (wv + 1.0) * ew - g
        if abs(denom) > 1.0e-30:
            w.flat[idx] -= we / denom

    return w


def lambert_w_newton(x, branch=0, tol=1.0e-12, max_iter=50):
    x = float(x)
    w = float(lambert_w_approx(np.array([x]), branch)[0])
    if not np.isfinite(w):
        return w

    for _ in range(max_iter):
        ew = np.exp(w)
        f = w * ew - x
        df = ew * (w + 1.0)
        if abs(df) < 1.0e-30:
            break
        dw = f / df
        w -= dw
        if abs(dw) < tol:
            break
    return w


def newton_solve(f, df, x0, tol=1.0e-10, max_iter=50):
    x = float(x0)
    for k in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1.0e-30:
            return x, False, k
        dx = fx / dfx
        x -= dx
        if abs(dx) < tol:
            return x, True, k + 1
    return x, False, max_iter


def solve_nonlinear_reaction(y_old, dt, c, f_rhs, M=None):
    rhs = float(f_rhs)
    if abs(c) < 1.0e-15:
        return rhs


    def g(y):
        return y + dt * c * y ** 3 - rhs

    def dg(y):
        return 1.0 + 3.0 * dt * c * y ** 2

    x0 = rhs if abs(rhs) < 10.0 else np.cbrt(rhs / (dt * c))
    root, conv, iters = newton_solve(g, dg, x0)
    if not conv:

        y = x0
        for _ in range(100):
            y_new = rhs / (1.0 + dt * c * y ** 2)
            if abs(y_new - y) < 1.0e-12:
                return y_new
            y = y_new
    return root


def nonlinear_rhs_cubic(y, c):
    return c * y ** 3


def nonlinear_rhs_cubic_derivative(y, c):
    return 3.0 * c * y ** 2
