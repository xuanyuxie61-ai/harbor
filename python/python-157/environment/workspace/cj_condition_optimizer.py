import numpy as np
from combustion_utils import check_positive, check_interval


def zero_brent(f, a, b, tol=1.0e-10, max_iter=100):
    a, b = check_interval(a, b)
    fa = f(a)
    fb = f(b)
    if fa * fb > 0.0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    c = a
    fc = fa
    for _ in range(max_iter):
        if abs(fc) < abs(fb):
            a, b, c = b, c, a
            fa, fb, fc = fb, fc, fa

        tol_act = 2.0 * np.finfo(float).eps * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)
        if abs(m) <= tol_act or fb == 0.0:
            return b

        if abs(a - c) < tol_act or abs(fa - fc) < tol_act or abs(fb - fc) < tol_act:

            d = e = m
        else:

            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            s = e
            e = d
            if 2.0 * p < 3.0 * m * q - abs(tol_act * q) and p < abs(0.5 * s * q):
                d = p / q
            else:
                d = e = m

        a = b
        fa = fb
        if abs(d) > tol_act:
            b += d
        elif m > 0.0:
            b += tol_act
        else:
            b -= tol_act
        fb = f(b)
        if (fb > 0.0 and fc > 0.0) or (fb <= 0.0 and fc <= 0.0):
            c = a
            fc = fa
            d = e = b - a
    return b


def local_min_brent(f, a, b, tol=1.0e-10, max_iter=100):
    a, b = check_interval(a, b)
    c = 0.5 * (3.0 - np.sqrt(5.0))
    v = a + c * (b - a)
    w = v
    x = v
    e = 0.0
    fx = f(x)
    fv = fx
    fw = fx

    for _ in range(max_iter):
        midpoint = 0.5 * (a + b)
        tol1 = np.sqrt(np.finfo(float).eps) * abs(x) + tol / 3.0
        tol2 = 2.0 * tol1
        if abs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
            return x, fx

        if abs(e) <= tol1:

            if x >= midpoint:
                e = a - x
            else:
                e = b - x
            d = c * e
        else:

            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = d
            if abs(p) >= abs(0.5 * q * r) or p <= q * (a - x) or p >= q * (b - x):
                if x >= midpoint:
                    e = a - x
                else:
                    e = b - x
                d = c * e
            else:
                d = p / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if midpoint >= x else -tol1

        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + np.sign(d) * tol1
        fu = f(u)

        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    return x, fx


def opt_quadratic_interpolation(f, x1, x2, x3, n_iter=50,
                                x_tol=1.0e-12, y_tol=1.0e-12):
    x = np.array([float(x1), float(x2), float(x3)])
    for i in range(n_iter):
        y = np.array([f(xi) for xi in x])

        V = np.vstack([x ** 2, x, np.ones(3)]).T
        try:
            p_coeff = np.linalg.solve(V, y)
        except np.linalg.LinAlgError:
            break
        a_coeff = p_coeff[0]
        b_coeff = p_coeff[1]
        if abs(a_coeff) < 1.0e-14:
            break
        x_new = -b_coeff / (2.0 * a_coeff)
        x = np.array([x[1], x[2], x_new])
        if abs(x[2] - x[1]) < x_tol and abs(y[2] - y[1]) < y_tol:
            return x_new, f(x_new)
    return x[2], f(x[2])


class CJConditionSolver:

    def __init__(self, gamma=1.4, Q=2.5e6, p0=101325.0, rho0=1.225):
        self.gamma = gamma
        self.Q = Q
        self.p0 = p0
        self.rho0 = rho0

    def hugoniot_pressure(self, v):
        v0 = 1.0 / self.rho0
        gp1 = self.gamma + 1.0
        gm1 = self.gamma - 1.0
        if abs(v - v0 * gm1 / gp1) < 1.0e-14:
            return np.inf
        p = self.p0 * (gp1 / gm1 * v0 / v - 1.0) / (gp1 / gm1 - v0 / v)

        p += (2.0 * self.Q * self.rho0 / gp1) / (1.0 - gm1 / gp1 * v / v0)
        return p

    def rayleigh_line(self, v, D):
        v0 = 1.0 / self.rho0
        return self.p0 + (self.rho0 ** 2) * (D ** 2) * (v0 - v)

    def cj_velocity_iterative(self):
        v0 = 1.0 / self.rho0

        D_low = np.sqrt(self.gamma * self.p0 / self.rho0) * 1.01
        D_high = np.sqrt(8.0 * self.Q) * 2.0

        def residual(D):

            vs = np.linspace(0.1 * v0, 0.9 * v0, 500)
            diffs = []
            for v in vs:
                try:
                    pH = self.hugoniot_pressure(v)
                    pR = self.rayleigh_line(v, D)
                    diffs.append(abs(pH - pR))
                except Exception:
                    continue
            if not diffs:
                return np.inf
            return min(diffs)


        try:
            Ds = np.linspace(D_low, D_high, 20)
            Rs = np.array([residual(D) for D in Ds])
            idx_min = np.argmin(Rs)
            if idx_min == 0:
                idx_min = 1
            if idx_min == len(Ds) - 1:
                idx_min = len(Ds) - 2
            D_est, _ = opt_quadratic_interpolation(
                lambda D: residual(D),
                Ds[idx_min - 1], Ds[idx_min], Ds[idx_min + 1],
                n_iter=20
            )
        except Exception:
            D_est = 0.5 * (D_low + D_high)


        try:
            D_cj, _ = local_min_brent(lambda D: residual(D), D_low, D_high, tol=1.0e-8)
        except Exception:
            D_cj = D_est

        return D_cj

    def exact_cj_velocity(self):
        from combustion_utils import cj_detonation_velocity
        return cj_detonation_velocity(self.gamma, self.Q, self.p0, self.rho0)
