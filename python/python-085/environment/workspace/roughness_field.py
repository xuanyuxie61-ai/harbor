import numpy as np
from typing import Tuple, Optional


def ran1f_step(b: int, u: np.ndarray, q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if b > 31:
        raise ValueError("b must be <= 31 for ran1f")
    z = 0.0
    j = 1
    for i in range(b):

        if q[i] <= 0:
            u[i] = np.random.randn()
            q[i] = j
        q[i] -= 1
        y = u[i]
        z += y
        j *= 2
    if b > 0:
        z /= b
    return z, u, q


def correlation_function(x: np.ndarray, m: int) -> np.ndarray:
    n = len(x)
    m = min(m, n - 1)
    xbar = np.mean(x)
    r = np.zeros(m + 1)
    for k in range(m + 1):
        for j in range(n - k):
            r[k] += (x[j + k] - xbar) * (x[j] - xbar)
    r /= n
    return r


def generate_pink_noise_profile(n_points: int, length: float = 1.0,
                                 beta: float = 1.8,
                                 b_levels: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, length, n_points)
    dx = length / (n_points - 1)

    freqs = np.fft.rfftfreq(n_points, d=dx)
    freqs[0] = 1e-6
    spectrum = freqs ** (-beta / 2.0)
    spectrum[0] = 0.0

    phases = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
    fft_vals = spectrum * phases
    h = np.fft.irfft(fft_vals, n=n_points)

    h = (h - np.mean(h)) / (np.std(h) + 1e-20)
    return x, h


def generate_2d_fractal_surface(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
                                 hurst: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(x, y)

    fx = np.fft.fftfreq(nx, d=lx / nx)
    fy = np.fft.fftfreq(ny, d=ly / ny)
    FX, FY = np.meshgrid(fx, fy)
    k2 = FX ** 2 + FY ** 2
    k2[0, 0] = 1e-12
    spectrum = k2 ** (-(hurst + 1.0) / 2.0)
    spectrum[0, 0] = 0.0
    random_phase = np.random.randn(ny, nx) + 1j * np.random.randn(ny, nx)
    fft_surface = spectrum * random_phase
    h = np.real(np.fft.ifft2(fft_surface))
    h = (h - np.mean(h)) / (np.std(h) + 1e-20)
    return X, Y, h


def apply_roughness_to_mesh(mesh_nodes: np.ndarray, roughness_1d: np.ndarray,
                             contact_mask: np.ndarray, scale: float = 1e-5) -> np.ndarray:
    nodes_new = mesh_nodes.copy()
    x_contact = mesh_nodes[contact_mask, 0]
    n_contact = len(x_contact)
    if n_contact != len(roughness_1d):

        x_src = np.linspace(np.min(x_contact), np.max(x_contact), len(roughness_1d))
        h_interp = np.interp(x_contact, x_src, roughness_1d)
    else:
        h_interp = roughness_1d
    for idx, node in enumerate(np.where(contact_mask)[0]):
        nodes_new[node, 1] += scale * h_interp[idx]
    return nodes_new
