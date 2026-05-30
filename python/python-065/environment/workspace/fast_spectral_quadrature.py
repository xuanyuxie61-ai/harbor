
import numpy as np


def fejer1_integrate_fast(f, n):
    if n < 1:
        raise ValueError("n 必须 >= 1")
    N = np.arange(1, n, 2)
    L = len(N)
    m = n - L
    k = np.arange(m)
    temp1 = np.exp(1j * np.pi * k / n)
    temp2 = np.zeros(L + 1)
    v0 = np.concatenate([2.0 * temp1 / (1.0 - 4.0 * k ** 2), temp2])
    v1 = v0[:-1] + np.conj(v0[:0:-1])
    w = np.real(np.fft.ifft(v1))

    x = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    fx = f(x)
    quad = float(np.dot(w, fx))
    return quad


def gauss_legendre_integrate_fast(f, n):
    if n < 1:
        raise ValueError("n 必须 >= 1")
    beta = 0.5 / np.sqrt(1.0 - (2.0 * np.arange(1, n + 1)) ** (-2))

    tridiag = np.diag(beta, 1) + np.diag(beta, -1)
    eigval, eigvec = np.linalg.eigh(tridiag)
    x = eigvec[0, :]
    idx = np.argsort(x)
    x = x[idx]
    w = 2.0 * eigval[0, idx] ** 2

    fx = f(x)
    quad = float(np.dot(w, fx))
    return quad


def clenshaw_curtis_rule_compute(n):
    if n < 1:
        raise ValueError("n 必须 >= 1")
    x = np.cos(np.pi * np.arange(n + 1) / n)

    c = np.ones(n + 1)
    c[0] = 2.0
    c[n] = 2.0
    k = np.arange(n + 1)

    w = np.zeros(n + 1)
    for j in range(n + 1):
        if j == 0 or j == n:
            w[j] = 1.0 / (n ** 2 - 1.0)
        else:
            s = 0.0
            for m in range(1, n // 2):
                s += np.cos(2.0 * m * j * np.pi / n) / (4.0 * m ** 2 - 1.0)
            if n % 2 == 0:
                s += 0.5 * np.cos(n * j * np.pi / n) / (n ** 2 - 1.0)
            w[j] = (1.0 - 2.0 * s) / n
    w[0] *= 0.5
    w[n] *= 0.5
    return x, w


def integrate_vertical_column(z_levels, values, method="gauss_legendre"):
    z_levels = np.asarray(z_levels)
    values = np.asarray(values)

    if method == "gauss_legendre":
        n = len(z_levels)

        z_min, z_max = z_levels[0], z_levels[-1]
        scale = 0.5 * (z_max - z_min)
        shift = 0.5 * (z_max + z_min)

        beta = 0.5 / np.sqrt(1.0 - (2.0 * np.arange(1, n + 1)) ** (-2))
        tridiag = np.diag(beta, 1) + np.diag(beta, -1)
        eigval, eigvec = np.linalg.eigh(tridiag)
        x = eigvec[0, :]
        idx = np.argsort(x)
        x = x[idx]
        w = 2.0 * eigval[idx] ** 2

        z_phys = scale * x + shift

        f_vals = np.interp(z_phys, z_levels, values)
        integral = scale * np.dot(w, f_vals)
        return float(integral)
    else:

        return float(np.trapezoid(values, z_levels))


def compute_total_column_water_vapor(z, q, rho):
    f_vals = rho * q
    return integrate_vertical_column(z, f_vals, method="gauss_legendre")


def test_fast_quad():

    def f(x):
        return x ** 2
    val = gauss_legendre_integrate_fast(f, 16)
    assert abs(val - 2.0 / 3.0) < 1e-10
    val2 = fejer1_integrate_fast(f, 64)
    assert abs(val2 - 2.0 / 3.0) < 1e-10
    print("fast_spectral_quadrature 自测试通过")


if __name__ == "__main__":
    test_fast_quad()
