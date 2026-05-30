import numpy as np


def polynomial_multiply(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)


    pn = len(p)
    qn = len(q)
    while pn > 1 and abs(p[pn - 1]) < 1e-15:
        pn -= 1
    while qn > 1 and abs(q[qn - 1]) < 1e-15:
        qn -= 1

    if pn == 0 or qn == 0:
        return np.array([0.0])

    p = p[:pn]
    q = q[:qn]


    r = np.zeros(pn + qn - 1, dtype=np.float64)
    for i in range(pn):
        for j in range(qn):
            r[i + j] += p[i] * q[j]

    return r


def chebyshev_polynomial(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])

    T_prev2 = np.array([1.0])
    T_prev1 = np.array([0.0, 1.0])

    for k in range(1, n):

        T_cur = np.zeros(len(T_prev1) + 1, dtype=np.float64)
        T_cur[1:] += 2.0 * T_prev1

        T_cur[:len(T_prev2)] -= T_prev2
        T_prev2, T_prev1 = T_prev1, T_cur

    return T_prev1


def legendre_polynomial(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])

    P_prev2 = np.array([1.0])
    P_prev1 = np.array([0.0, 1.0])

    for k in range(1, n):

        temp = np.zeros(len(P_prev1) + 1, dtype=np.float64)
        temp[1:] = (2 * k + 1) * P_prev1

        temp[:len(P_prev2)] -= k * P_prev2

        P_cur = temp / (k + 1)
        P_prev2, P_prev1 = P_prev1, P_cur

    return P_prev1


def spectral_differentiation_matrix(N):
    if N < 1:
        raise ValueError("N must be at least 1")

    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1, dtype=np.float64)
    c[0] = 2.0
    c[N] = 2.0

    D = np.zeros((N + 1, N + 1), dtype=np.float64)

    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (x[i] - x[j])
            else:
                if i == 0:
                    D[i, i] = (2.0 * N * N + 1.0) / 6.0
                elif i == N:
                    D[i, i] = -(2.0 * N * N + 1.0) / 6.0
                else:
                    D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))

    return D, x


def apply_angular_spectral_derivative(f_phi, N_modes=16):
    f_phi = np.asarray(f_phi, dtype=np.complex128)
    n = len(f_phi)
    if n < 2:
        return np.zeros_like(f_phi)


    f_hat = np.fft.fft(f_phi)


    k = np.fft.fftfreq(n, d=1.0 / n) * (2.0 * np.pi)


    k_max = N_modes
    mask = np.abs(k) > k_max * (2.0 * np.pi / n) * (n // 2)

    idx = np.arange(n)
    idx = np.where(idx > n // 2, idx - n, idx)
    mask = np.abs(idx) > N_modes
    f_hat[mask] = 0.0


    df_hat = 1j * k * f_hat
    df_dphi = np.fft.ifft(df_hat).real

    return df_dphi
