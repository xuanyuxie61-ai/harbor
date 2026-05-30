
import numpy as np

def hermite_polynomial_value(n, x):
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()

    H_prev2 = np.ones_like(x)
    H_prev1 = x.copy()
    for k in range(2, n + 1):
        H_curr = x * H_prev1 - (k - 1) * H_prev2
        H_prev2 = H_prev1
        H_prev1 = H_curr
    return H_prev1

def gaussian_value(x, mu=0.0, sigma=1.0):
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2.0 * np.pi))

def gaussian_derivative(x, n, mu=0.0, sigma=1.0):
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    z = (x - mu) / (sigma * np.sqrt(2.0))
    G = gaussian_value(x, mu, sigma)
    Hn = hermite_polynomial_value(n, z)
    return ((-1.0)**n / (sigma * np.sqrt(2.0))**n) * Hn * G

def initialize_gaussian_vortex_2d(Nx, Ny, Lx, Ly, x0, y0, A, sigma,
                                  vorticity_sign=1.0):
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')


    dx = X - x0
    dy = Y - y0
    dx = np.mod(dx + Lx/2, Lx) - Lx/2
    dy = np.mod(dy + Ly/2, Ly) - Ly/2
    r2 = dx**2 + dy**2

    psi = A * np.exp(-r2 / (2.0 * sigma**2))
    zeta = vorticity_sign * (A / sigma**2) * (r2 / sigma**2 - 2.0) * np.exp(-r2 / (2.0 * sigma**2))

    return psi, zeta

def hermite_spectral_filter(spec_field, KX, KY, k_cutoff, order=4):
    k = np.sqrt(KX**2 + KY**2)
    alpha = np.log(2.0)
    filter_mask = np.exp(-alpha * (k / k_cutoff)**(2 * order))
    return spec_field * filter_mask
