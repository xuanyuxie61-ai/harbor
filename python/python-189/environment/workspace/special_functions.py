
import numpy as np
from math import gamma, lgamma, log, exp, sqrt, cos, sin, floor, fabs, pi, isinf






def sine_integral(x: float) -> float:
    if not np.isfinite(x):
        raise ValueError("sine_integral: x must be finite")

    p2 = 1.570796326794897
    el = 0.5772156649015329
    eps = 1.0e-15
    x2 = x * x
    xabs = abs(x)
    xsign = -1.0 if x < 0.0 else 1.0

    if xabs == 0.0:
        return 0.0

    if xabs <= 16.0:
        xr = xabs
        val = xabs
        for k in range(1, 41):
            xr = -0.5 * xr * (2 * k - 1) / k / (4 * k * k + 4 * k + 1) * x2
            val = val + xr
            if abs(xr) < abs(val) * eps:
                return xsign * val
        return xsign * val

    elif xabs <= 32.0:
        m = int(floor(47.2 + 0.82 * xabs))
        bj = np.zeros(m)
        xa1 = 0.0
        xa0 = 1.0e-100
        for k in range(m - 1, -1, -1):
            xa = 4.0 * (k + 1) * xa0 / xabs - xa1
            bj[k] = xa
            xa1 = xa0
            xa0 = xa
        xs = bj[0]
        for k in range(2, m, 2):
            xs = xs + 2.0 * bj[k]
        bj[0] = bj[0] / xs
        for k in range(1, m):
            bj[k] = bj[k] / xs

        xr = 1.0
        xg1 = bj[0]
        for k in range(1, m):
            xr = 0.25 * xr * (2.0 * (k + 1) - 3.0) ** 2 \
                 / ((k) * (2.0 * (k + 1) - 1.0) ** 2) * xabs
            xg1 = xg1 + bj[k] * xr

        xr = 1.0
        xg2 = bj[0]
        for k in range(1, m):
            xr = 0.25 * xr * (2.0 * (k + 1) - 5.0) ** 2 \
                 / ((k) * (2.0 * (k + 1) - 3.0) ** 2) * xabs
            xg2 = xg2 + bj[k] * xr

        xcs = cos(xabs / 2.0)
        xss = sin(xabs / 2.0)
        val = xsign * (xabs * xcs * xg1 + 2.0 * xss * xg2 - sin(xabs))
        return val

    else:
        xr = 1.0
        xf = 1.0
        for k in range(1, 10):
            xr = -2.0 * xr * k * (2 * k - 1) / x2
            xf = xf + xr
        xr = 1.0 / xabs
        xg = xr
        for k in range(1, 9):
            xr = -2.0 * xr * (2 * k + 1) * k / x2
            xg = xg + xr
        val = xsign * (p2 - xf * cos(xabs) / xabs - xg * sin(xabs) / xabs)
        return val






def _besselj(n, x):
    from scipy.special import jv
    return jv(n, x)


def _bessely(n, x):
    from scipy.special import yv
    return yv(n, x)


def bessel_zero(n: float, k: int, kind: int = 1, max_iter: int = 100) -> float:
    if not np.isfinite(n):
        raise ValueError("bessel_zero: n must be finite")
    if k < 1 or not isinstance(k, int):
        raise ValueError("bessel_zero: k must be positive integer")
    if kind not in (1, 2):
        raise ValueError("bessel_zero: kind must be 1 or 2")

    n_abs = abs(n)
    if kind == 1:
        order_max = 146222.16674537213
    else:
        order_max = 370030.762407380
    if n_abs > order_max:
        raise ValueError(f"bessel_zero: |n| too large for kind={kind}")


    if kind == 1:
        if k == 1:
            c = [0.411557013144507, 0.999986723293410,
                 0.698028985524484, 1.06977507291468]
            e = [0.335300369843979, 0.339671493811664]
        elif k == 2:
            c = [1.93395115137444, 1.00007656297072,
                 -0.805720018377132, 3.38764629174694]
            e = [0.456215294517928, 0.388380341189200]
        elif k == 3:
            c = [5.40770803992613, 1.00093850589418,
                 2.66926179799040, -0.174925559314932]
            e = [0.429702214054531, 0.633480051735955]
        else:

            z2 = bessel_zero(n, 2, kind)
            z3 = bessel_zero(n, 3, kind)
            return 2.0 * z3 - z2
        x0 = c[0] + c[1] * n_abs + c[2] * (n_abs + 1) ** e[0] + c[3] * (n_abs + 1) ** e[1]
    else:
        if k == 1:
            c = [0.0795046982450635, 0.999998378297752,
                 0.890380645613825, 0.0270604048106402]
            e = [0.335377217953294, 0.308720059086699]
        elif k == 2:
            c = [1.04502538172394, 1.00002054874161,
                 -0.437921325402985, 2.70113114990400]
            e = [0.434823025111322, 0.366245194174671]
        elif k == 3:
            c = [3.72777931751914, 1.00035294977757,
                 2.68566718444899, -0.112980454967090]
            e = [0.398247585896959, 0.604770035236606]
        else:
            z2 = bessel_zero(n, 2, kind)
            z3 = bessel_zero(n, 3, kind)
            return 2.0 * z3 - z2
        x0 = c[0] + c[1] * n_abs + c[2] * (n_abs + 1) ** e[0] + c[3] * (n_abs + 1) ** e[1]

    if k >= 4:
        z2 = bessel_zero(n, k - 1, kind)
        z3 = bessel_zero(n, k - 2, kind)
        x0 = 2.0 * z2 - z3


    x = float(x0)
    tol_relative = 1.0e4
    for _ in range(max_iter):
        if kind == 1:
            a = _besselj(n, x)
            b = _besselj(n + 1, x)
        else:
            a = _bessely(n, x)
            b = _bessely(n + 1, x)
        denom = (2.0 * b * b * x * x - a * b * x * (4.0 * n + 1.0)
                 + (n * (n + 1.0) + x * x) * a * a)
        if abs(denom) < 1.0e-300:
            break
        dx = 2.0 * a * x * (n * a - b * x) / denom
        x = x - dx
        if abs(dx) <= np.finfo(float).eps * abs(x) * tol_relative:
            break
    return x


def bessel_zeros(n: float, k_max: int, kind: int = 1) -> np.ndarray:
    return np.array([bessel_zero(n, k, kind) for k in range(1, k_max + 1)])






def log_gamma(z: float) -> float:
    if z <= 0.0:
        raise ValueError("log_gamma: z must be positive")
    return lgamma(z)


def incomplete_beta(x: float, p: float, q: float) -> tuple:
    if x < 0.0 or x > 1.0:
        return 0.0, 1
    if p <= 0.0 or q <= 0.0:
        return 0.0, 2

    if x == 0.0:
        return 0.0, 0
    if x == 1.0:
        return 1.0, 0

    aleps = -179.6016
    eps = 2.2e-16
    eps1 = 1.0e-78


    if x <= 0.5:
        interval = 0
        y = x
        pp, qq = p, q
    else:
        interval = 1
        pp, qq = q, p
        y = 1.0 - x

    ib = int(floor(qq))
    ps = qq - ib
    if abs(qq - ib) < 1.0e-14:
        ps = 1.0

    px = pp * log(y)
    pq_val = log_gamma(pp + qq)
    p1 = log_gamma(pp)
    c_val = log_gamma(qq)
    d4 = log(pp)
    xb = px + log_gamma(ps + pp) - log_gamma(ps) - d4 - p1

    ib = int(floor(xb / aleps))
    infsum = 0.0

    if ib == 0:
        infsum = exp(xb)
        cnt = infsum * pp
        wh = 0.0
        while True:
            wh = wh + 1.0
            cnt = cnt * (wh - ps) * y / wh
            xb_term = cnt / (pp + wh)
            infsum = infsum + xb_term
            if xb_term / eps < infsum:
                break
            if wh > 1.0e6:
                break

    finsum = 0.0
    if qq <= 1.0:
        prob = finsum + infsum
        if interval != 0:
            prob = 1.0 - prob
        return prob, 0

    xb = px + qq * log(1.0 - y) + pq_val - p1 - log(qq) - c_val
    ib = int(floor(xb / aleps))
    if ib < 0:
        ib = 0
    c_factor = 1.0 / (1.0 - y)
    cnt = exp(xb - ib * aleps)
    ps = qq
    wh = qq

    while True:
        wh = wh - 1.0
        if wh <= 0.0:
            prob = finsum + infsum
            if interval != 0:
                prob = 1.0 - prob
            break
        px2 = (ps * c_factor) / (pp + wh)
        if px2 <= 1.0:
            if cnt / eps <= finsum or cnt <= eps1 / px2:
                prob = finsum + infsum
                if interval != 0:
                    prob = 1.0 - prob
                break
        cnt = cnt * px2
        if cnt > 1.0:
            ib = ib - 1
            cnt = cnt * eps1
        ps = wh
        if ib == 0:
            finsum = finsum + cnt

    return prob, 0


def beta_cdf(x: float, p: float, q: float) -> float:
    prob, ier = incomplete_beta(x, p, q)
    if ier != 0:
        if ier == 1:
            if x < 0.0:
                return 0.0
            else:
                return 1.0
        raise ValueError(f"beta_cdf: invalid parameters p={p}, q={q}")
    return prob
