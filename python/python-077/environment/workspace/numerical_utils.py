
import numpy as np
from typing import Tuple, Callable, Optional






def trigcardinal(xi: np.ndarray, xdj: float, nd: int, h: float) -> np.ndarray:
    eps = 1e-14
    tau = np.zeros_like(xi, dtype=float)
    diff = xi - xdj


    mask_eq = np.abs(diff) < eps
    tau[mask_eq] = 1.0

    mask_ne = ~mask_eq
    if nd % 2 == 1:
        denom = nd * np.sin(np.pi * diff[mask_ne] / (nd * h))
    else:
        denom = nd * np.tan(np.pi * diff[mask_ne] / (nd * h))


    safe_denom = np.where(np.abs(denom) < eps, np.sign(denom + eps) * eps, denom)
    tau[mask_ne] = np.sin(np.pi * diff[mask_ne] / h) / safe_denom
    return tau


def trig_interpolant(xd: np.ndarray, yd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    nd = len(xd)
    if nd < 2:
        raise ValueError("至少需要 2 个插值节点")
    h = xd[1] - xd[0]
    if h <= 0:
        raise ValueError("节点间距必须为正")

    if not np.allclose(np.diff(xd), h, rtol=1e-10):
        raise ValueError("xd 必须是等距节点")

    yi = np.zeros_like(xi, dtype=float)
    for j in range(nd):
        yi += yd[j] * trigcardinal(xi, xd[j], nd, h)
    return yi






def pwl_approx_1d_matrix(nd: int, xd: np.ndarray, yd: np.ndarray, nc: int, xc: np.ndarray) -> np.ndarray:
    xd = np.asarray(xd).ravel()
    xc = np.asarray(xc).ravel()
    if len(xd) != nd or len(xc) != nc:
        raise ValueError("维度不匹配")
    if nc < 2:
        raise ValueError("至少需要 2 个控制点")
    if not np.all(np.diff(xc) > 0):
        raise ValueError("控制点 xc 必须严格单调递增")

    A = np.zeros((nd, nc), dtype=float)
    for i in range(nd):
        x = xd[i]
        if x <= xc[0]:
            A[i, 0] = 1.0
        elif x >= xc[-1]:
            A[i, -1] = 1.0
        else:

            j = np.searchsorted(xc, x, side='right') - 1
            j = max(0, min(j, nc - 2))
            dx = xc[j + 1] - xc[j]
            if dx < 1e-14:
                A[i, j] = 1.0
            else:
                t = (x - xc[j]) / dx
                A[i, j] = 1.0 - t
                A[i, j + 1] = t
    return A


def pwl_approx_1d(nd: int, xd: np.ndarray, yd: np.ndarray, nc: int, xc: np.ndarray) -> np.ndarray:
    A = pwl_approx_1d_matrix(nd, xd, yd, nc, xc)
    yd_vec = np.asarray(yd).ravel()

    ATA = A.T @ A
    ATy = A.T @ yd_vec

    ATA += 1e-12 * np.eye(nc)
    yc = np.linalg.solve(ATA, ATy)
    return yc






def alnorm(x: float, upper: bool = False) -> float:
    a1 = 5.75885480458
    a2 = 2.62433121679
    a3 = 5.92885724438
    b1 = -29.8213557807
    b2 = 48.6959930692
    c1 = -0.000000038052
    c2 = 0.000398064794
    c3 = -0.151679116635
    c4 = 4.8385912808
    c5 = 0.742380924027
    c6 = 3.99019417011
    con = 1.28
    d1 = 1.00000615302
    d2 = 1.98615381364
    d3 = 5.29330324926
    d4 = -15.1508972451
    d5 = 30.789933034
    ltone = 7.0
    p = 0.39894228044
    q = 0.39990348504
    r = 0.398942280385
    utzero = 18.66

    up = upper
    z = x
    if z < 0.0:
        up = not up
        z = -z

    if ltone < z and (not up or utzero < z):
        return 0.0 if up else 1.0

    y = 0.5 * z * z
    if z <= con:
        value = 0.5 - z * (p - q * y / (y + a1 + b1 / (y + a2 + b2 / (y + a3))))
    else:
        value = r * np.exp(-y) / (z + c1 + d1 / (z + c2 + d2 / (z + c3 + d3 / (z + c4 + d4 / (z + c5 + d5 / (z + c6))))))

    return 1.0 - value if not up else value


def alnorm_array(x: np.ndarray, upper: bool = False) -> np.ndarray:
    return np.array([alnorm(float(v), upper) for v in x.ravel()]).reshape(x.shape)






def r8mat_fs(n: int, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True).ravel()
    if A.shape != (n, n):
        raise ValueError(f"A 必须是 {n}×{n} 矩阵")
    if len(b) != n:
        raise ValueError(f"b 长度必须为 {n}")

    x = b.copy()
    for jcol in range(n):

        piv = abs(A[jcol, jcol])
        ipiv = jcol
        for i in range(jcol + 1, n):
            if piv < abs(A[i, jcol]):
                piv = abs(A[i, jcol])
                ipiv = i
        if piv < 1e-15:
            raise ValueError(f"第 {jcol} 步主元为零，矩阵奇异")


        if jcol != ipiv:
            A[[jcol, ipiv], :] = A[[ipiv, jcol], :]
            x[jcol], x[ipiv] = x[ipiv], x[jcol]


        temp = A[jcol, jcol]
        A[jcol, jcol] = 1.0
        A[jcol, jcol + 1:] /= temp
        x[jcol] /= temp


        for i in range(jcol + 1, n):
            if abs(A[i, jcol]) > 1e-15:
                factor = -A[i, jcol]
                A[i, jcol] = 0.0
                A[i, jcol + 1:] += factor * A[jcol, jcol + 1:]
                x[i] += factor * x[jcol]


    for jcol in range(n - 1, 0, -1):
        x[:jcol] -= A[:jcol, jcol] * x[jcol]

    return x






def sor1(n: int, A: np.ndarray, b: np.ndarray, x: np.ndarray, w: float) -> np.ndarray:
    if not (0.0 < w < 2.0):
        raise ValueError("SOR 松弛因子 ω 必须在 (0, 2) 区间内")
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    x_new = x.copy()
    for i in range(n):
        sigma = 0.0
        for j in range(n):
            if j != i:
                if j < i:
                    sigma += A[i, j] * x_new[j]
                else:
                    sigma += A[i, j] * x[j]
        if abs(A[i, i]) < 1e-14:
            raise ValueError(f"第 {i} 个对角元为零")
        x_new[i] = (1.0 - w) * x[i] + w * (b[i] - sigma) / A[i, i]
    return x_new


def sor_solve(A: np.ndarray, b: np.ndarray, w: float = 1.5,
              tol: float = 1e-8, max_iter: int = 10000) -> Tuple[np.ndarray, int]:
    n = len(b)
    x = np.zeros(n, dtype=float)
    for it in range(max_iter):
        x_old = x.copy()
        x = sor1(n, A, b, x, w)
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, it + 1
    return x, max_iter






def langford_parameters() -> Tuple[float, float, float, float, float, float,
                                    float, np.ndarray, float]:
    a = 0.95
    b = 0.7
    c = 0.6
    d = 3.5
    e = 0.25
    f = 0.1
    t0 = 0.0
    xyz0 = np.array([0.1, 0.1, 0.1])
    tstop = 50.0
    return a, b, c, d, e, f, t0, xyz0, tstop


def langford_deriv(t: float, xyz: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f, _, _, _ = langford_parameters()
    x, y, z = xyz[0], xyz[1], xyz[2]
    dxdt = (z - b) * x - d * y
    dydt = d * x + (z - b) * y
    dzdt = c + a * z - z**3 / 3.0 - (x**2 + y**2) * (1.0 + e * z) + f * z * x**3
    return np.array([dxdt, dydt, dzdt])


def rk4_step(f: Callable[[float, np.ndarray], np.ndarray],
             t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_ode(f: Callable, y0: np.ndarray, t_span: Tuple[float, float],
                  n_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    y = np.zeros((n_steps + 1, len(y0)), dtype=float)
    t = np.linspace(t0, t1, n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        y[i + 1] = rk4_step(f, t[i], y[i], h)
    return t, y






def weibull_pdf(u: np.ndarray, A: float, k: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    pdf = np.zeros_like(u)
    mask = u > 0
    pdf[mask] = (k / A) * (u[mask] / A)**(k - 1) * np.exp(-(u[mask] / A)**k)
    return pdf


def weibull_cdf(u: np.ndarray, A: float, k: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    cdf = np.zeros_like(u)
    mask = u > 0
    cdf[mask] = 1.0 - np.exp(-(u[mask] / A)**k)
    return cdf
