
import numpy as np


def generate_grf_1d(n_points, length_scale=0.1, sigma=1.0):
    x = np.linspace(0, 1, n_points)

    dx = np.abs(x[:, None] - x[None, :])
    C = sigma ** 2 * np.exp(-dx / length_scale)


    C += 1e-10 * np.eye(n_points)


    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:

        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    z = np.random.randn(n_points)
    return L @ z


def generate_grf_spherical(n_lat=36, n_lon=72, length_scale_km=1000.0, sigma=1.0):
    R = 6371.0
    L_rad = length_scale_km / R

    lats = np.deg2rad(np.linspace(-90, 90, n_lat))
    lons = np.deg2rad(np.linspace(0, 360, n_lon))

    n = n_lat * n_lon
    points = np.zeros((n, 3))
    idx = 0
    for i in range(n_lat):
        for j in range(n_lon):
            points[idx] = [
                np.cos(lats[i]) * np.cos(lons[j]),
                np.cos(lats[i]) * np.sin(lons[j]),
                np.sin(lats[i])
            ]
            idx += 1


    C = np.zeros((n, n))
    for i in range(n):

        dot = np.dot(points, points[i])
        dot = np.clip(dot, -1.0, 1.0)
        chord = np.sqrt(2.0 * (1.0 - dot))
        C[i, :] = sigma ** 2 * np.exp(-chord ** 2 / (2.0 * L_rad ** 2))

    C += 1e-10 * np.eye(n)

    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    z = np.random.randn(n)
    grf = L @ z
    return grf.reshape((n_lat, n_lon))


def ar1_noise(n, phi=0.85, sigma=1.0, x0=0.0):
    phi = np.clip(phi, -0.999, 0.999)
    sigma_e = sigma * np.sqrt(1.0 - phi ** 2)
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.random.randn() * sigma_e
    return x


def fbm_noise(n, hurst=0.8, sigma=1.0):
    hurst = np.clip(hurst, 0.01, 0.99)

    k = np.arange(n)
    r = 0.5 * (np.abs(k + 1) ** (2 * hurst) + np.abs(k - 1) ** (2 * hurst) -
               2 * np.abs(k) ** (2 * hurst))
    r[0] = 1.0


    n_fft = 2 * n
    r_ext = np.zeros(n_fft)
    r_ext[0:n] = r
    r_ext[n_fft - n + 1:n_fft] = r[1:n][::-1]

    lambda_vals = np.real(np.fft.fft(r_ext))
    lambda_vals = np.maximum(lambda_vals, 0.0)

    z = np.random.randn(n_fft) + 1j * np.random.randn(n_fft)
    y = np.fft.ifft(np.sqrt(lambda_vals) * z)
    fbm = np.real(y[0:n])
    fbm = sigma * (fbm - np.mean(fbm)) / max(np.std(fbm), 1e-15)
    return fbm


def seasonal_noise(n_years, annual_amplitude=5.0, phase=0.0,
                   interannual_variability=1.0):
    n_months = n_years * 12
    t = np.linspace(0, 2.0 * np.pi * n_years, n_months)


    seasonal = annual_amplitude * np.sin(t + phase)


    interannual = ar1_noise(n_months, phi=0.95, sigma=interannual_variability)

    return seasonal + interannual
