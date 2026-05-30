
import numpy as np


def jacobi_polynomial(x, alpha, beta, N):
    x = np.asarray(x, dtype=float)
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("Jacobi parameters must be > -1.")
    if N < 0:
        return np.zeros_like(x)


    try:
        from math import gamma as _gamma
    except ImportError:
        def _gamma(z):
            return np.exp(_gammaln(z))

    def _gammaln(z):

        if z < 0.5:
            return np.log(np.pi) - np.log(np.sin(np.pi * z)) - _gammaln(1.0 - z)
        p = [76.18009172947146, -86.50532032941677,
             24.01409824083091, -1.231739572450155,
             0.1208650973866179e-2, -0.5395239384953e-5]
        y = z
        x_lanczos = y + 5.5
        x_lanczos -= (y + 0.5) * np.log(x_lanczos)
        ser = 1.000000000190015
        for i, p_i in enumerate(p):
            y += 1.0
            ser += p_i / y
        return -x_lanczos + np.log(2.5066282746310005 * ser / z)

    gamma0 = (2.0 ** (alpha + beta + 1.0) / (alpha + beta + 1.0)
              * _gamma(alpha + 1.0) * _gamma(beta + 1.0)
              / _gamma(alpha + beta + 1.0))

    PL = np.zeros((N + 1, x.size))
    PL[0, :] = 1.0 / np.sqrt(gamma0)
    if N == 0:
        return PL[0, :]

    gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0
    PL[1, :] = (((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0)
                / np.sqrt(gamma1))
    if N == 1:
        return PL[1, :]

    a_old = (2.0 / (2.0 + alpha + beta)
             * np.sqrt((alpha + 1.0) * (beta + 1.0)
                       / (alpha + beta + 3.0)))

    for i in range(1, N):
        h1 = 2.0 * i + alpha + beta
        a_new = (2.0 / (h1 + 2.0)
                 * np.sqrt((i + 1.0) * (i + 1.0 + alpha + beta)
                           * (i + 1.0 + alpha) * (i + 1.0 + beta)
                           / (h1 + 1.0) / (h1 + 3.0)))
        b_new = - (alpha ** 2 - beta ** 2) / h1 / (h1 + 2.0)
        PL[i + 1, :] = (1.0 / a_new
                        * (-a_old * PL[i - 1, :]
                           + (x - b_new) * PL[i, :]))
        a_old = a_new

    return PL[N, :]


def grad_jacobi_polynomial(x, alpha, beta, N):
    if N == 0:
        return np.zeros_like(np.asarray(x, dtype=float))
    x = np.asarray(x, dtype=float)
    coeff = np.sqrt(N * (N + alpha + beta + 1.0))
    return coeff * jacobi_polynomial(x, alpha + 1.0, beta + 1.0, N - 1)


def vandermonde_1d(N, r):
    r = np.asarray(r, dtype=float)
    V = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        V[:, j] = jacobi_polynomial(r, 0.0, 0.0, j)
    return V


def grad_vandermonde_1d(N, r):
    r = np.asarray(r, dtype=float)
    Vr = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        Vr[:, j] = grad_jacobi_polynomial(r, 0.0, 0.0, j)
    return Vr


def differentiation_matrix_1d(N, r, V):
    Vr = grad_vandermonde_1d(N, r)


    try:
        Dr = Vr @ np.linalg.inv(V)
    except np.linalg.LinAlgError:
        Dr = Vr @ np.linalg.pinv(V)
    return Dr


def jacobi_gauss_lobatto(alpha, beta, N):
    if N == 0:
        return np.array([-1.0, 1.0])
    if N == 1:
        return np.array([-1.0, 0.0, 1.0])



    from math import cos, pi
    x = np.zeros(N + 1)
    x[0] = -1.0
    x[N] = 1.0


    for i in range(1, N):
        x[i] = -cos(pi * i / N)


    eps = 1e-14
    for _ in range(100):
        P = jacobi_polynomial(x[1:N], 1.0, 1.0, N - 1)
        dP = grad_jacobi_polynomial(x[1:N], 1.0, 1.0, N - 1)
        dx = -P / (dP + eps)
        x[1:N] += dx
        if np.max(np.abs(dx)) < eps:
            break

    return x


def vandermonde_interp_2d_matrix(n, m, x, y):
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    if len(x) != n or len(y) != n:
        raise ValueError("x and y must have length n.")

    tmp1 = (m + 1) * (m + 2) // 2
    if n != tmp1:
        raise ValueError(f"For interpolation, need n = T(M+1) = {tmp1}, got n={n}")

    A = np.zeros((n, n))
    j = 0
    for s in range(m + 1):
        for ex in range(s, -1, -1):
            ey = s - ex
            A[:, j] = (x ** ex) * (y ** ey)
            j += 1
    return A


def polynomial_value_2d(n, c, m, x, y):
    c = np.asarray(c, dtype=float).flatten()
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if len(c) != n:
        raise ValueError("Length of c must equal n.")

    value = np.zeros_like(x, dtype=float)
    j = 0
    for s in range(m + 1):
        for ex in range(s, -1, -1):
            ey = s - ex
            value += c[j] * (x ** ex) * (y ** ey)
            j += 1
    return value
