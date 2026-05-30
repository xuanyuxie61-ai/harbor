
import numpy as np


def legendre_nodes_weights_1d(n):
    if n < 1:
        raise ValueError("n must be positive.")
    if n <= 100:
        x, w = np.polynomial.legendre.leggauss(n)
        return x, w
    else:
        raise ValueError("n too large for direct Gauss-Legendre computation.")


def gauss_legendre_3d_set(a, b, nx, ny, nz):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.any(b <= a):
        raise ValueError("Upper limits must exceed lower limits.")

    xx, wx = legendre_nodes_weights_1d(nx)
    yy, wy = legendre_nodes_weights_1d(ny)
    zz, wz = legendre_nodes_weights_1d(nz)


    xx = ((1.0 - xx) * a[0] + (1.0 + xx) * b[0]) / 2.0
    wx = wx * (b[0] - a[0]) / 2.0

    yy = ((1.0 - yy) * a[1] + (1.0 + yy) * b[1]) / 2.0
    wy = wy * (b[1] - a[1]) / 2.0

    zz = ((1.0 - zz) * a[2] + (1.0 + zz) * b[2]) / 2.0
    wz = wz * (b[2] - a[2]) / 2.0

    n_total = nx * ny * nz
    x = np.zeros(n_total)
    y = np.zeros(n_total)
    z = np.zeros(n_total)
    w = np.zeros(n_total)

    idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x[idx] = xx[i]
                y[idx] = yy[j]
                z[idx] = zz[k]
                w[idx] = wx[i] * wy[j] * wz[k]
                idx += 1

    return x, y, z, w


def integrate_over_box(f, a, b, nx=6, ny=6, nz=6):
    x, y, z, w = gauss_legendre_3d_set(a, b, nx, ny, nz)
    values = f(x, y, z)
    return float(np.sum(w * values))


def electromagnetic_energy_density_integral(epsilon, mu, E_field_func, H_field_func,
                                            a, b, nx=6, ny=6, nz=6):
    x, y, z, w = gauss_legendre_3d_set(a, b, nx, ny, nz)
    e2 = E_field_func(x, y, z)
    h2 = H_field_func(x, y, z)
    integrand = 0.5 * (np.real(epsilon) * e2 + mu * h2)
    return float(np.sum(w * integrand))


def absorbed_power_integral(omega, epsilon, E_field_func,
                            a, b, nx=6, ny=6, nz=6):
    eps0 = 8.854187817e-12
    x, y, z, w = gauss_legendre_3d_set(a, b, nx, ny, nz)
    e2 = E_field_func(x, y, z)
    integrand = 0.5 * omega * eps0 * np.imag(epsilon) * e2

    integrand = np.maximum(integrand, 0.0)
    return float(np.sum(w * integrand))


def test_exactness_monomial(a, b, max_total_degree, nx=6, ny=6, nz=6):
    errors = []
    for t in range(max_total_degree + 1):
        for k in range(t + 1):
            for j in range(t - k + 1):
                i = t - j - k
                p = np.array([i, j, k])

                exact = (
                    (b[0] ** (p[0] + 1) - a[0] ** (p[0] + 1)) / (p[0] + 1) *
                    (b[1] ** (p[1] + 1) - a[1] ** (p[1] + 1)) / (p[1] + 1) *
                    (b[2] ** (p[2] + 1) - a[2] ** (p[2] + 1)) / (p[2] + 1)
                )

                def monomial(xx, yy, zz):
                    return (xx ** p[0]) * (yy ** p[1]) * (zz ** p[2])

                approx = integrate_over_box(monomial, a, b, nx, ny, nz)
                if abs(exact) < 1e-30:
                    err = abs(approx)
                else:
                    err = abs(approx - exact) / abs(exact)
                errors.append(err)
    return errors
