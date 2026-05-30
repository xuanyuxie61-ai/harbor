
import numpy as np
from typing import Callable, Tuple


def flin(
    n: int,
    jsearch: int,
    l: float,
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    v: np.ndarray,
) -> float:
    x_try = x.copy()
    if 1 <= jsearch <= n:
        x_try = x_try + l * v[:, jsearch - 1]
    elif jsearch == 0:
        x_try = x_try + l * x_try
    return f(x_try)


def minny(
    n: int,
    jsearch: int,
    nits: int,
    d2: float,
    x1: float,
    x2: float,
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    v: np.ndarray,
    h: float,
) -> Tuple[float, float, float, np.ndarray]:
    small = np.finfo(float).eps ** 2
    m2 = np.sqrt(np.finfo(float).eps)

    def _eval(l: float) -> float:
        return flin(n, jsearch, l, f, x, v)


    a = min(x1, x2)
    b = max(x1, x2)
    fa = _eval(a)
    fb = _eval(b)

    if fa > fb:
        a, b = b, a
        fa, fb = fb, fa

    c = b
    fc = fb
    tol = m2 * abs(b - a)

    for _ in range(nits * 10):
        mid = 0.5 * (a + c)
        if abs(c - a) < tol:
            break

        denom = 2.0 * (fa - 2.0 * fb + fc)
        if abs(denom) > small:
            t = 0.5 * (a + c) + 0.5 * (fa - fc) * (c - a) / denom
            if a < t < c:
                ft = _eval(t)
                if ft < fb:
                    b, fb = t, ft
                    continue

        if b - a > c - b:
            t = b - 0.381966 * (b - a)
        else:
            t = b + 0.381966 * (c - b)
        ft = _eval(t)
        if ft < fb:
            a, fa = b, fb
            b, fb = t, ft
        elif ft < fc:
            c, fc = t, ft
        else:
            if t < b:
                a, fa = t, ft
            else:
                c, fc = t, ft

    lds = b
    x_new = x.copy()
    if 1 <= jsearch <= n:
        x_new = x_new + lds * v[:, jsearch - 1]
    return d2, lds, fb, x_new


def svsort(n: int, d: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(-np.abs(d))
    return d[idx], v[:, idx]


def praxis(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    tol: float = 1e-6,
    h0: float = 1.0,
    max_iter: int = 500,
) -> Tuple[float, np.ndarray, int]:
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    machep = np.finfo(float).eps
    small = machep * machep
    m2 = np.sqrt(machep)
    t = small + abs(tol)
    h = max(h0, 100.0 * t)

    v = np.eye(n)
    d = np.zeros(n)
    nf = 1
    fx = f(x)

    ldt = h
    ktm = 1
    kt = 0

    while nf < max_iter:
        sf = d[0]
        d[0] = 0.0


        x_prev = x.copy()
        d2, s, fx_new, x = minny(n, 1, 2, d[0], 0.0, 0.0, f, x, v, h)
        d[0] = d2


        for k in range(2, n + 1):
            y = x.copy()
            sf = fx_new

            for k2 in range(k, n + 1):
                d2, s, fx_new, x = minny(n, k2, 2, d[k2 - 1], 0.0, 0.0, f, x, v, h)
                d[k2 - 1] = d2


            for k2 in range(1, k):
                d2, s, fx_new, x = minny(n, k2, 2, d[k2 - 1], 0.0, 0.0, f, x, v, h)
                d[k2 - 1] = d2


            lds = np.linalg.norm(x - y)
            if lds > small:
                v[:, k - 1] = (x - y) / lds
                d2, lds, fx_new, x = minny(n, k, 4, 0.0, lds, 0.0, f, x, v, h)
                d[k - 1] = d2

        ldt = 0.01 * ldt
        ldt = max(ldt, lds)
        t2 = m2 * np.linalg.norm(x) + t
        if 0.5 * t2 < ldt:
            kt = -1
        kt += 1
        if kt > ktm:
            break


        vt = v.T
        try:
            u_mat, s_vals, _ = np.linalg.svd(vt, full_matrices=False)
            d = 1.0 / (s_vals + small)
            d, v = svsort(n, d, u_mat)
        except np.linalg.LinAlgError:
            break


        if np.linalg.norm(x - x_prev) < tol * (1.0 + np.linalg.norm(x)):
            break

    return fx_new, x, nf


def optimize_turbine_array(
    n_turbines: int = 5,
    domain_size: float = 500.0,
    min_spacing: float = 50.0,
) -> Tuple[np.ndarray, float]:
    rho = 1025.0
    A = 20.0
    cp = 16.0 / 27.0
    Ct = 0.8
    D = np.sqrt(4.0 * A / np.pi)
    k_wake = 0.05
    U_inf = 2.5

    def objective(x_flat: np.ndarray) -> float:
        if x_flat.size != 2 * n_turbines:
            return 1e10
        pos = x_flat.reshape((n_turbines, 2))

        penalty = 0.0
        for i in range(n_turbines):
            for j in range(i + 1, n_turbines):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < min_spacing:
                    penalty += 1e6 * (min_spacing - dist) ** 2


        speeds = np.full(n_turbines, U_inf)
        for i in range(n_turbines):
            deficit = 0.0
            for j in range(n_turbines):
                if j == i:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dx > 0:
                    wake_diam = D + 2.0 * k_wake * dist
                    if abs(dy) < 0.5 * wake_diam:
                        deficit += Ct * (D / wake_diam) ** 2
            speeds[i] = max(U_inf * (1.0 - deficit), 0.1)





        power = np.zeros(n_turbines)
        return 0.0


    x0 = np.random.rand(2 * n_turbines) * domain_size
    fmin, xmin, _ = praxis(objective, x0, tol=1e-4, h0=domain_size * 0.1, max_iter=2000)
    positions = xmin.reshape((n_turbines, 2))


    positions = np.clip(positions, 0.0, domain_size)
    total_power = -objective(xmin)

    return total_power, positions
