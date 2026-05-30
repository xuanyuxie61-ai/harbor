
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import legendre as legendre_poly


def chebyshev2_nodes_weights(n, a=-1.0, b=1.0):
    i = np.arange(1, n + 1, dtype=float)
    x_std = np.cos(i * np.pi / (n + 1))
    w_std = (np.pi / (n + 1)) * np.sin(i * np.pi / (n + 1)) ** 2


    shift = (a + b) / 2.0
    scale = (b - a) / 2.0
    x = shift + scale * x_std
    w = scale * w_std
    return x, w


def gauss_legendre_3d(nx, ny, nz, ax, bx, ay, by, az, bz):
    x1d, wx = leggauss(nx)
    y1d, wy = leggauss(ny)
    z1d, wz = leggauss(nz)


    x1d = 0.5 * (bx - ax) * x1d + 0.5 * (bx + ax)
    y1d = 0.5 * (by - ay) * y1d + 0.5 * (by + ay)
    z1d = 0.5 * (bz - az) * z1d + 0.5 * (bz + az)

    wx *= 0.5 * (bx - ax)
    wy *= 0.5 * (by - ay)
    wz *= 0.5 * (bz - az)

    X, Y, Z = np.meshgrid(x1d, y1d, z1d, indexing='ij')
    Wx, Wy, Wz = np.meshgrid(wx, wy, wz, indexing='ij')

    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()
    w = (Wx * Wy * Wz).ravel()
    return x, y, z, w


def bubble_surface_area_quadrature(r_func, theta_nodes=32, phi_nodes=32):

    t_nodes, t_weights = leggauss(theta_nodes)
    theta = 0.5 * np.pi * (t_nodes + 1.0)
    w_theta = 0.5 * np.pi * t_weights


    phi = np.linspace(0, 2 * np.pi, phi_nodes, endpoint=False)
    dphi = 2.0 * np.pi / phi_nodes

    area = 0.0
    for i, th in enumerate(theta):
        for j, ph in enumerate(phi):
            r = r_func(th, ph)
            if r <= 0:
                continue

            h_theta = 1e-6
            h_phi = 1e-6
            r_plus_theta = r_func(th + h_theta, ph)
            r_minus_theta = r_func(max(th - h_theta, 0.0), ph)
            dr_dtheta = (r_plus_theta - r_minus_theta) / (2.0 * h_theta + 1e-15)

            r_plus_phi = r_func(th, ph + h_phi)
            r_minus_phi = r_func(th, ph - h_phi)
            dr_dphi = (r_plus_phi - r_minus_phi) / (2.0 * h_phi + 1e-15)

            sin_th = np.sin(th)
            metric = np.sqrt(1.0 + (dr_dtheta / (r + 1e-15)) ** 2 +
                             (dr_dphi / ((r + 1e-15) * sin_th + 1e-15)) ** 2)
            integrand = r ** 2 * sin_th * metric
            area += integrand * w_theta[i] * dphi

    return area


def bubble_volume_quadrature(r_func, theta_nodes=24, phi_nodes=24):
    t_nodes, t_weights = leggauss(theta_nodes)
    theta = 0.5 * np.pi * (t_nodes + 1.0)
    w_theta = 0.5 * np.pi * t_weights

    phi = np.linspace(0, 2 * np.pi, phi_nodes, endpoint=False)
    dphi = 2.0 * np.pi / phi_nodes

    volume = 0.0
    for i, th in enumerate(theta):
        for j, ph in enumerate(phi):
            r = r_func(th, ph)
            r = max(r, 0.0)
            integrand = (r ** 3 / 3.0) * np.sin(th)
            volume += integrand * w_theta[i] * dphi

    return volume


def surface_tension_energy(r_func, sigma, theta_nodes=24, phi_nodes=24):
    S = bubble_surface_area_quadrature(r_func, theta_nodes, phi_nodes)
    return sigma * S


def kinetic_energy_integral(R, dRdt, rho, theta_nodes=16):
    E_k = 2.0 * np.pi * rho * (R ** 3) * (dRdt ** 2)
    return E_k


def pressure_work_integral(p_in, p_out, r_func, theta_nodes=16, phi_nodes=16):
    V = bubble_volume_quadrature(r_func, theta_nodes, phi_nodes)
    return (p_in - p_out) * V


def legendre_3d_exactness_test(n_points, max_degree=4):
    x, y, z, w = gauss_legendre_3d(n_points, n_points, n_points, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    errors = []

    for tt in range(max_degree + 1):
        for k in range(tt + 1):
            for j in range(tt - k + 1):
                i = tt - j - k

                if i % 2 == 0 and j % 2 == 0 and k % 2 == 0:
                    exact = 8.0 / ((i + 1) * (j + 1) * (k + 1))
                else:
                    exact = 0.0

                v = (x ** i) * (y ** j) * (z ** k)
                approx = np.dot(w, v)
                if abs(exact) > 1e-15:
                    err = abs(approx - exact) / abs(exact)
                else:
                    err = abs(approx)
                errors.append((i, j, k, err))

    return errors


def chebyshev_surface_integral(f, a, b, n=32):
    x, w = chebyshev2_nodes_weights(n, a, b)
    fx = np.array([f(xi) for xi in x])
    return np.dot(w, fx)
