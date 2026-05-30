
import numpy as np
from typing import List, Tuple


def circle_arc_grid(cx: float, cy: float, r: float, theta_start: float,
                    theta_end: float, n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2")
    theta_start_rad = np.deg2rad(theta_start)
    theta_end_rad = np.deg2rad(theta_end)
    thetas = np.linspace(theta_start_rad, theta_end_rad, n)
    x = cx + r * np.cos(thetas)
    y = cy + r * np.sin(thetas)
    return np.column_stack((x, y))


def polynomial_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return np.convolve(p, q)


def diff2_center(f: callable, x: float, h: float = 1e-5) -> float:
    if h <= 0:
        raise ValueError("step size h must be positive")
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


def triangle_area_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1])
                     - (c[0] - a[0]) * (b[1] - a[1]))


def triangle_angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    lab = np.linalg.norm(b - a)
    lbc = np.linalg.norm(c - b)
    lca = np.linalg.norm(a - c)
    alpha = np.arccos(np.clip((lab**2 + lca**2 - lbc**2)
                      / (2 * lab * lca + 1e-15), -1.0, 1.0))
    beta = np.arccos(np.clip((lab**2 + lbc**2 - lca**2)
                     / (2 * lab * lbc + 1e-15), -1.0, 1.0))
    gamma = np.pi - alpha - beta
    return np.array([alpha, beta, gamma])


def i4mat_rref(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=np.int64)
    m, n = a.shape
    lead = 0
    for r in range(m):
        if lead >= n:
            break
        i = r
        while i < m and a[i, lead] == 0:
            i += 1
        if i == m:
            lead += 1
            continue
        a[[r, i]] = a[[i, r]]

        piv = a[r, lead]
        g = int(np.gcd.reduce(np.abs(a[r])))
        if g > 1:
            a[r] //= g
            piv //= g
        for j in range(m):
            if j != r and a[j, lead] != 0:
                factor = a[j, lead]
                a[j] = piv * a[j] - factor * a[r]

                g2 = int(np.gcd.reduce(np.abs(a[j])))
                if g2 > 1:
                    a[j] //= g2
        lead += 1
    return a


def rk4_step(f: callable, t: float, y: np.ndarray, h: float,
             args: Tuple = ()) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    k1 = h * np.asarray(f(t, y, *args), dtype=np.float64)
    k2 = h * np.asarray(f(t + 0.5 * h, y + 0.5 * k1, *args), dtype=np.float64)
    k3 = h * np.asarray(f(t + 0.5 * h, y + 0.5 * k2, *args), dtype=np.float64)
    k4 = h * np.asarray(f(t + h, y + k3, *args), dtype=np.float64)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def sort_heap_external(n: int, indx: int, i: int, j: int, isgn: int):
    if indx < 0:
        i = (n - 1) // 2
        j = n - 1
        indx = 1
        return indx, i, j, isgn
    if isgn <= 0:
        j -= 1
        if j == 0:
            indx = 0
        else:
            i = 0
            indx = 2
        return indx, i, j, isgn
    i += 1
    if i == j:
        indx = 3
    else:
        indx = 2
    return indx, i, j, isgn
