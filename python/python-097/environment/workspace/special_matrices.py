
import numpy as np


def hankel_matrix(n, x):
    x = np.asarray(x).flatten()
    if len(x) < 2 * n - 1:
        raise ValueError(f"向量长度至少为{2*n-1}，当前为{len(x)}")

    H = np.zeros((n, n))
    for j in range(n):
        H[:, j] = x[j:j + n]
    return H


def toeplitz_matrix(n, t):
    t = np.asarray(t).flatten()
    if len(t) < 2 * n - 1:
        raise ValueError(f"向量长度至少为{2*n-1}，当前为{len(t)}")

    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            T[i, j] = t[n - 1 + j - i]
    return T


def hankel_inverse_fiedler(n, x):
    x = np.asarray(x).flatten()
    if len(x) < 2 * n - 1:
        raise ValueError(f"向量长度至少为{2*n-1}")

    A = hankel_matrix(n, x)


    p = np.zeros(n)
    p[:n - 1] = x[n:2 * n - 1]

    q = np.zeros(n)
    q[-1] = 1.0


    try:
        u = np.linalg.solve(A, p)
        v = np.linalg.solve(A, q)
    except np.linalg.LinAlgError:

        A_inv = np.linalg.pinv(A)
        return A_inv, A


    z1 = np.zeros(n)
    w1 = np.concatenate([v[1:], z1])
    M1 = hankel_matrix(n, w1)

    z2 = np.zeros(n - 1)
    w2 = np.concatenate([z2, u])
    M2 = toeplitz_matrix(n, w2)

    z3 = np.zeros(n)
    z3[0] = -1.0
    w3 = np.concatenate([u[1:], z3])
    M3 = hankel_matrix(n, w3)

    z4 = np.zeros(n - 1)
    w4 = np.concatenate([z4, v])
    M4 = toeplitz_matrix(n, w4)

    A_inv = M1 @ M2 - M3 @ M4
    return A_inv, A


def circulant_matrix(n, c):
    c = np.asarray(c).flatten()
    if len(c) < n:
        raise ValueError(f"向量长度至少为{n}")

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = c[(j - i) % n]
    return C


def solve_toeplitz_system(n, t, b):
    t = np.asarray(t).flatten()
    b = np.asarray(b).flatten()

    if len(t) < 2 * n - 1:
        raise ValueError("t向量长度不足")
    if len(b) < n:
        raise ValueError("b向量长度不足")



    r = t[n - 1:]
    c = t[n - 1::-1]

    x = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)


    x[0] = b[0] / r[0]
    f[0] = 1.0 / r[0]
    g[0] = 1.0 / r[0]

    for k in range(1, n):

        alpha = np.dot(c[1:k + 1], f[:k])
        beta = 1.0 / (1.0 - alpha ** 2)


        f_new = np.zeros(k + 1)
        g_new = np.zeros(k + 1)
        f_new[:k] = beta * (f[:k] - alpha * g[k - 1::-1])
        g_new[:k] = beta * (g[k - 1::-1] - alpha * f[:k])
        f_new[k] = -beta * alpha * f[0]
        g_new[k] = beta * (1.0 - alpha) * f[0]

        f[:k + 1] = f_new
        g[:k + 1] = g_new


        delta = b[k] - np.dot(r[1:k + 1][::-1], x[:k])
        x[:k + 1] += delta * f[:k + 1]

    return x


def antenna_array_impedance_matrix(n_elements, spacing_wavelength, ka=1.0):
    d = spacing_wavelength * 2.0 * np.pi

    Z = np.zeros((n_elements, n_elements))
    Z_self = 50.0 + 20.0j

    for i in range(n_elements):
        for j in range(n_elements):
            if i == j:
                Z[i, j] = Z_self.real
            else:
                dist = abs(i - j) * d

                if dist > 0.01:
                    Z[i, j] = 30.0 * np.sin(dist) / dist
                else:
                    Z[i, j] = 30.0

    return Z
