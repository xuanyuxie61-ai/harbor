
import numpy as np
from scipy.special import gamma as scipy_gamma


def jacobi_polynomial(x, n_max, alpha, beta):
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("jacobi_polynomial: alpha 和 beta 必须大于 -1。")

    x = np.asarray(x, dtype=float)
    if np.any(np.abs(x) > 1.0 + 1e-12):

        pass

    m = x.size
    if n_max < 0:
        return np.empty((m, 0))

    P = np.zeros((m, n_max + 1))
    P[:, 0] = 1.0

    if n_max == 0:
        return P

    P[:, 1] = 0.5 * (alpha - beta) + (1.0 + 0.5 * (alpha + beta)) * x

    for n in range(1, n_max):
        c1 = 2.0 * (n + 1.0) * (n + alpha + beta + 1.0) * (2.0 * n + alpha + beta)
        c2 = (2.0 * n + alpha + beta + 1.0) * (alpha * alpha - beta * beta)
        c3 = (2.0 * n + alpha + beta + 1.0) * (2.0 * n + alpha + beta + 2.0) * (2.0 * n + alpha + beta)
        c4 = 2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0)

        P[:, n + 1] = ((c2 + c3 * x) * P[:, n] - c4 * P[:, n - 1]) / c1

    return P


def imtqlx(n, d, e, z):
    d = np.asarray(d, dtype=float).copy()
    e = np.asarray(e, dtype=float).copy()
    z = np.asarray(z, dtype=float).copy()

    itn = 30
    prec = np.finfo(float).eps

    if n == 1:
        return d, z

    e[n - 1] = 0.0

    for l in range(n):
        j = 0
        while True:
            m = l
            while m < n - 1:
                if abs(e[m]) <= prec * (abs(d[m]) + abs(d[m + 1])):
                    break
                m += 1

            p = d[l]
            if m == l:
                break

            if j == itn:
                raise RuntimeError("imtqlx: 迭代次数超限，三对角矩阵对角化失败。")

            j += 1
            g = (d[l + 1] - p) / (2.0 * e[l])
            r = np.sqrt(g * g + 1.0)
            g = d[m] - p + e[l] / (g + np.sign(g) * abs(r))
            s = 1.0
            c = 1.0
            p = 0.0
            mml = m - l

            for ii in range(1, mml + 1):
                i = m - ii
                f = s * e[i]
                b = c * e[i]

                if abs(f) >= abs(g):
                    c = g / f
                    r = np.sqrt(c * c + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c = c * s
                else:
                    s = f / g
                    r = np.sqrt(s * s + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s = s * c

                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                f = z[i + 1]
                z[i + 1] = s * z[i] + c * f
                z[i] = c * z[i] - s * f

            d[l] = d[l] - p
            e[l] = g
            e[m] = 0.0


    for ii in range(1, n):
        i = ii - 1
        k = i
        p = d[i]
        for j in range(ii, n):
            if d[j] < p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            p = z[i]
            z[i] = z[k]
            z[k] = p

    return d, z


def gauss_jacobi_rule(n, alpha, beta):
    if n < 1:
        return np.array([]), np.array([])

    ab = alpha + beta
    abi = 2.0 + ab


    zemu = (2.0 ** (ab + 1.0)) * scipy_gamma(alpha + 1.0) * scipy_gamma(beta + 1.0) / scipy_gamma(abi)


    diag = np.zeros(n)
    off_diag = np.zeros(n)

    diag[0] = (beta - alpha) / abi
    off_diag[0] = np.sqrt(
        4.0 * (1.0 + alpha) * (1.0 + beta) / ((abi + 1.0) * abi * abi)
    )
    a2b2 = beta * beta - alpha * alpha

    for i in range(1, n):
        abi_i = 2.0 * (i + 1) + ab
        diag[i] = a2b2 / ((abi_i - 2.0) * abi_i)
        abi_sq = abi_i * abi_i
        off_diag[i] = np.sqrt(
            4.0 * (i + 1.0) * (i + 1.0 + alpha) * (i + 1.0 + beta) * (i + 1.0 + ab)
            / ((abi_sq - 1.0) * abi_sq)
        )


    w = np.zeros(n)
    w[0] = np.sqrt(zemu)

    nodes, weights = imtqlx(n, diag, off_diag, w)
    weights = weights ** 2

    return nodes, weights


def spectral_differentiation_matrix(x_nodes):
    N = len(x_nodes)
    D = np.zeros((N, N))



    b = np.ones(N)
    for j in range(N):
        for k in range(N):
            if k != j:
                b[j] *= 1.0 / (x_nodes[j] - x_nodes[k])

    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (b[j] / b[i]) / (x_nodes[i] - x_nodes[j])
            else:
                s = 0.0
                for k in range(N):
                    if k != i:
                        s += 1.0 / (x_nodes[i] - x_nodes[k])
                D[i, i] = s

    return D


def boundary_layer_map(eta, delta, alpha=0.0, beta=0.0):
    eta = np.asarray(eta, dtype=float)
    if delta <= 0:
        raise ValueError("boundary_layer_map: delta 必须为正。")


    eta_clipped = np.clip(eta, 0.0, delta)

    t = np.sqrt(eta_clipped / delta)
    xi = 2.0 * t - 1.0
    dy_dxi = delta * (xi + 1.0) * 0.5


    dy_dxi = np.where(np.abs(dy_dxi) < 1e-15, 1e-15, dy_dxi)

    return xi, dy_dxi


def integrate_boundary_layer(f_values, nodes, weights, delta):
    xi = np.asarray(nodes, dtype=float)
    _, dy_dxi = boundary_layer_map(delta * 0.5 * (xi + 1.0), delta)


    y = delta * 0.25 * (xi + 1.0) ** 2
    dy_dxi_direct = delta * 0.5 * (xi + 1.0)
    dy_dxi_direct = np.where(np.abs(dy_dxi_direct) < 1e-15, 1e-15, dy_dxi_direct)

    return np.sum(weights * f_values * dy_dxi_direct)


def test_jacobi_spectral():
    alpha, beta = 0.5, -0.3
    n = 8
    x, w = gauss_jacobi_rule(n, alpha, beta)




    f = x ** 6
    numerical = np.sum(w * f)


    from scipy.special import beta as beta_func
    exact = (2.0 ** (alpha + beta + 1.0)) * beta_func(alpha + 1.0, beta + 1.0)


    from scipy.integrate import fixed_quad
    def integrand(t):
        return ((1.0 - t) ** alpha) * ((1.0 + t) ** beta) * (t ** 6)
    exact_ref, _ = fixed_quad(integrand, -1.0, 1.0, n=40)

    rel_err = abs(numerical - exact_ref) / (abs(exact_ref) + 1e-15)
    print(f"[jacobi_spectral] Gauss-Jacobi 自检: n={n}, rel_err={rel_err:.3e}")
    return rel_err < 1e-10


if __name__ == "__main__":
    test_jacobi_spectral()
