
import numpy as np
from math import gamma, sqrt, exp, cos, pi






def gauss_laguerre_nodes_weights(n: int, alpha: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(n, int) or n < 1:
        raise ValueError("求积点数 n 必须为正整数")
    if alpha <= -1.0:
        raise ValueError("广义 Laguerre 参数 alpha 必须大于 -1")


    i = np.arange(1, n + 1, dtype=float)
    diag = 2.0 * i - 1.0 + alpha
    offdiag = np.sqrt(i[:-1] * (i[:-1] + alpha))

    J = np.diag(diag) + np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)

    eigenvalues, eigenvectors = np.linalg.eigh(J)
    x = eigenvalues


    w = np.zeros(n)
    for k in range(n):
        v0 = eigenvectors[0, k]
        w[k] = (v0 ** 2) * gamma(alpha + 1.0)


    x = np.maximum(x, 0.0)
    w = np.abs(w)
    return x, w


def integrate_laguerre(f, n: int = 64, alpha: float = 0.0, args=()) -> float:
    x, w = gauss_laguerre_nodes_weights(n, alpha)
    fx = np.array([f(xi, *args) for xi in x])
    return float(np.dot(w, fx))


def integrate_decay_convolution(C_history: np.ndarray, dt: float, lam: float,
                                n_quad: int = 32) -> float:
    if lam <= 0.0:
        raise ValueError("衰变常数 λ 必须为正")
    if dt <= 0.0:
        raise ValueError("时间步长 dt 必须为正")
    if len(C_history) == 0:
        raise ValueError("浓度历史不能为空")

    s_nodes, s_weights = gauss_laguerre_nodes_weights(n_quad, alpha=0.0)

    tau_nodes = s_nodes / lam


    t_max = (len(C_history) - 1) * dt
    vals = []
    for tau in tau_nodes:
        t_query = t_max - tau
        if t_query <= 0.0:
            vals.append(C_history[0])
        elif t_query >= t_max:
            vals.append(C_history[-1])
        else:
            idx = int(t_query / dt)
            frac = (t_query - idx * dt) / dt
            idx = min(idx, len(C_history) - 2)
            vals.append(C_history[idx] * (1.0 - frac) + C_history[idx + 1] * frac)

    return float(np.dot(s_weights, vals) / lam)






def gauss_legendre_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n 必须 ≥ 1")

    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def integrate_2d_rectangle(f, xlim: tuple[float, float],
                           ylim: tuple[float, float],
                           nx: int = 8, ny: int = 8) -> float:
    xa, xb = xlim
    yc, yd = ylim
    if xa >= xb or yc >= yd:
        raise ValueError("积分区间必须满足 a < b 且 c < d")

    xi, wi = gauss_legendre_nodes_weights(nx)
    eta, wj = gauss_legendre_nodes_weights(ny)


    X = 0.5 * (xb - xa) * xi + 0.5 * (xb + xa)
    Y = 0.5 * (yd - yc) * eta + 0.5 * (yd + yc)
    Wx = 0.5 * (xb - xa) * wi
    Wy = 0.5 * (yd - yc) * wj

    total = 0.0
    for i in range(nx):
        for j in range(ny):
            total += Wx[i] * Wy[j] * f(X[i], Y[j])
    return float(total)






def padua_points(level: int) -> tuple[np.ndarray, np.ndarray]:
    if level < 0:
        raise ValueError("Padua 级别必须 ≥ 0")
    n = level + 1
    z = np.cos(np.arange(n + 1) * np.pi / n)

    pts_x = []
    pts_y = []
    for i in range(n + 1):
        for j in range(n + 1):
            if (i + j) % 2 == 0:
                pts_x.append(z[i])
                pts_y.append(z[j])
    return np.array(pts_x), np.array(pts_y)


def padua_weights(level: int) -> np.ndarray:
    x, y = padua_points(level)
    n = level + 1

    w = np.ones(len(x))
    z = np.cos(np.arange(n + 1) * np.pi / n)
    tol = 1e-12
    for k in range(len(x)):

        if abs(abs(x[k]) - 1.0) < tol or abs(abs(y[k]) - 1.0) < tol:
            w[k] *= 0.5

    if np.sum(w) > 0:
        w = w / np.sum(w) * 4.0
    return w






def test_quadrature_exactness() -> dict:
    results = {}


    n = 8
    alpha = 0.5
    max_m = 2 * n - 1
    laguerre_pass = True
    for m in range(max_m + 1):
        exact = gamma(alpha + m + 1.0)
        approx = integrate_laguerre(lambda x: x ** m, n=n, alpha=alpha)
        rel_err = abs(approx - exact) / (abs(exact) + 1e-15)
        if rel_err > 1e-10:
            laguerre_pass = False
            break
    results["laguerre_exactness"] = laguerre_pass


    def monomial_integral(p, q):
        def f(x, y):
            return (x ** p) * (y ** q)
        return integrate_2d_rectangle(f, (-1.0, 1.0), (-1.0, 1.0), nx=n, ny=n)

    legendre_pass = True
    for p in range(n):
        for q in range(n):
            if p + q > 2 * n - 1:
                continue
            exact = ((1 - (-1) ** (p + 1)) / (p + 1)) * ((1 - (-1) ** (q + 1)) / (q + 1)) if p >= 0 and q >= 0 else 0.0
            approx = monomial_integral(p, q)
            if abs(exact) < 1e-12:
                err = abs(approx)
            else:
                err = abs(approx - exact) / abs(exact)
            if err > 1e-10:
                legendre_pass = False
                break
        if not legendre_pass:
            break
    results["legendre2d_exactness"] = legendre_pass

    return results


if __name__ == "__main__":
    r = test_quadrature_exactness()
    print("quadrature_engine 自测试:", r)
