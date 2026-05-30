
import numpy as np


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
                raise RuntimeError("IMTQLX: iteration limit exceeded.")

            j += 1
            g = (d[l + 1] - p) / (2.0 * e[l])
            r = np.sqrt(g * g + 1.0)
            if g < 0.0:
                t = g - r
            else:
                t = g + r
            g = d[m] - p + e[l] / (g + t)

            s = 1.0
            c = 1.0
            p_val = 0.0
            mml = m - l

            for ii in range(1, mml + 1):
                i = m - ii
                f = s * e[i]
                b = c * e[i]

                if abs(g) <= abs(f):
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

                g = d[i + 1] - p_val
                r = (d[i] - g) * s + 2.0 * c * b
                p_val = s * r
                d[i + 1] = g + p_val
                g = c * r - b
                f = z[i + 1]
                z[i + 1] = s * z[i] + c * f
                z[i] = c * z[i] - s * f

            d[l] = d[l] - p_val
            e[l] = g
            e[m] = 0.0


    for i in range(n - 1):
        k = i
        p = d[i]
        for j in range(i + 1, n):
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


def legendre_ek_compute(n):
    if n < 1:
        raise ValueError("n must be >= 1.")
    zemu = 2.0
    bj = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        bj[i - 1] = (i * i) / (4.0 * i * i - 1.0)
    bj = np.sqrt(bj)

    d = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)
    z[0] = np.sqrt(zemu)

    x, w = imtqlx(n, d, bj, z)
    w = w ** 2
    return x, w


def gauss_legendre_map(a, b, n):
    if a >= b:
        raise ValueError("Require a < b.")
    x, w = legendre_ek_compute(n)
    x_mapped = ((1.0 - x) * a + (x + 1.0) * b) / 2.0
    w_mapped = w * (b - a) / 2.0
    return x_mapped, w_mapped


def integrate_spectral_interferogram(integrand_func, k_min, k_max, n_gl=64):
    k_nodes, k_weights = gauss_legendre_map(k_min, k_max, n_gl)
    fk = integrand_func(k_nodes)
    fk = np.asarray(fk)
    if fk.ndim == 1:
        I = np.dot(k_weights, fk)
    else:

        I = np.dot(k_weights, fk)
    return I


def integrate_depth_resolved_signal(reflectivity_func, z_array, k_min, k_max, n_gl=64):
    z_array = np.asarray(z_array, dtype=float)
    k_nodes, k_weights = gauss_legendre_map(k_min, k_max, n_gl)
    A = np.zeros_like(z_array)
    for i, z in enumerate(z_array):
        vals = reflectivity_func(k_nodes, z) * np.exp(1j * 2.0 * k_nodes * z)
        A[i] = np.abs(np.dot(k_weights, vals))
    return A
