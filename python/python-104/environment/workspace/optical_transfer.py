
import numpy as np
from math import factorial




def tetrahedron01_monomial_integral(e1, e2, e3):
    if e1 < 0 or e2 < 0 or e3 < 0:
        raise ValueError("Exponents must be non-negative.")
    num = factorial(e1) * factorial(e2) * factorial(e3)
    den = factorial(e1 + e2 + e3 + 3)
    return float(num) / float(den)


def tetrahedron01_volume():
    return 1.0 / 6.0


def tetrahedron01_sample(n_samples, seed=None):
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    if seed is not None:
        np.random.seed(seed)
    U = np.random.rand(n_samples, 4)
    E = -np.log(np.clip(U, 1e-30, 1.0))
    S = np.sum(E, axis=1, keepdims=True)
    points = E[:, :3] / S
    return points


def integrate_over_tetrahedral_mesh(integrand_func, n_samples_per_cell=1000, n_cells=8):
    total_integral = 0.0
    for _ in range(n_cells):
        pts = tetrahedron01_sample(n_samples_per_cell)
        vals = np.array([integrand_func(p[0], p[1], p[2]) for p in pts])
        total_integral += np.mean(vals) * tetrahedron01_volume()
    return total_integral / n_cells




def compute_pupil_function(grid_size, aperture_mask, phase):
    if aperture_mask.shape != phase.shape:
        raise ValueError("aperture_mask and phase must have the same shape.")
    P = np.zeros_like(phase, dtype=np.complex128)
    P[aperture_mask] = np.exp(1j * phase[aperture_mask])
    return P


def compute_otf_from_pupil(P):
    F = np.fft.fft2(P)
    otf = np.fft.ifft2(np.abs(F) ** 2)
    return np.fft.fftshift(otf)


def compute_psf_from_pupil(P, pixel_scale, wavelength, focal_length):
    if pixel_scale <= 0 or wavelength <= 0 or focal_length <= 0:
        raise ValueError("Physical parameters must be positive.")
    F = np.fft.fftshift(np.fft.fft2(P))
    psf = np.abs(F) ** 2
    psf = psf / np.max(psf)
    N = P.shape[0]
    freq = np.fft.fftfreq(N, d=pixel_scale)
    x_coords = wavelength * focal_length * freq
    return psf, x_coords


def compute_mtf_from_otf(otf):
    if otf.ndim == 2:
        center = (otf.shape[0] // 2, otf.shape[1] // 2)
        otf0 = np.abs(otf[center[0], center[1]])
    else:
        otf0 = np.abs(otf[len(otf) // 2])
    otf0 = max(otf0, 1e-30)
    return np.abs(otf) / otf0


def compute_wavefront_variance(phase, mask):
    phi = phase[mask]
    if len(phi) == 0:
        return 0.0
    phi = phi - np.mean(phi)

    coords = np.argwhere(mask)
    if len(coords) > 2:
        x = coords[:, 1] - np.mean(coords[:, 1])
        y = coords[:, 0] - np.mean(coords[:, 0])
        A = np.column_stack([x, y])
        tilt, _, _, _ = np.linalg.lstsq(A, phi, rcond=None)
        phi = phi - A @ tilt
    return float(np.var(phi))


def compute_encircled_energy(psf, x_coords, radius):
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    N = psf.shape[0]
    center = N // 2
    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    total = np.sum(psf)
    if total < 1e-30:
        return 0.0

    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    r_pix = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) * abs(dx)
    encircled = np.sum(psf[r_pix <= radius])
    return float(encircled / total)


def tetrahedral_phase_moments(phase, mask, max_order=3):
    if max_order < 0:
        raise ValueError("max_order must be non-negative.")
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return {}

    y_idx = coords[:, 0]
    x_idx = coords[:, 1]
    phi_vals = phase[mask]
    area = len(phi_vals)

    moments = {}
    for p in range(max_order + 1):
        for q in range(max_order + 1 - p):
            if p == 0 and q == 0:
                moments[(p, q)] = float(np.mean(phi_vals))
            else:
                x_norm = (x_idx - np.mean(x_idx)) / max(np.std(x_idx), 1.0)
                y_norm = (y_idx - np.mean(y_idx)) / max(np.std(y_idx), 1.0)
                moments[(p, q)] = float(np.mean((x_norm ** p) * (y_norm ** q) * phi_vals))

    return moments
