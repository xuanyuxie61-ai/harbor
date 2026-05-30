
import numpy as np


def mandelbrot_escape_time(cx, cy, count_max=50, escape_radius=2.0):
    is_scalar = np.isscalar(cx)
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    zr = np.zeros_like(cx)
    zi = np.zeros_like(cx)
    escape = np.full_like(cx, count_max + 1, dtype=int)
    for i in range(count_max):

        zr = np.clip(zr, -1e6, 1e6)
        zi = np.clip(zi, -1e6, 1e6)
        zr_new = zr * zr - zi * zi + cx
        zi_new = 2.0 * zr * zi + cy
        zr, zi = zr_new, zi_new
        mag = zr * zr + zi * zi
        mask = (mag > escape_radius ** 2) & (escape == count_max + 1)
        escape[mask] = i + 1
    if is_scalar:
        return int(escape.item())
    return escape


def compute_scattering_strength(x_grid, y_grid, center_x=0.0, center_y=0.0,
                                 scale=1.0, count_max=30):
    cx = (x_grid - center_x) / scale
    cy = (y_grid - center_y) / scale
    escape = mandelbrot_escape_time(cx, cy, count_max=count_max)
    tau = count_max / 3.0


    strength = np.exp(-escape / tau)

    r2 = cx ** 2 + cy ** 2
    envelope = np.exp(-r2 / 4.0)
    return strength * envelope


def ifs_leaf_fractal(n_points=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    A = np.array([
        [[0.80, 0.00], [0.00, 0.80]],
        [[0.50, 0.00], [0.00, 0.50]],
        [[0.355, -0.355], [0.355, 0.355]],
        [[0.355, 0.355], [-0.355, 0.355]]
    ])
    b = np.array([
        [0.10, 0.04],
        [0.25, 0.40],
        [0.266, 0.078],
        [0.378, 0.434]
    ])

    probs = np.array([0.25, 0.25, 0.25, 0.25])
    points = np.zeros((n_points, 2))
    x = rng.random(2)

    for _ in range(100):
        j = rng.choice(4, p=probs)
        x = A[j] @ x + b[j]
    for i in range(n_points):
        j = rng.choice(4, p=probs)
        x = A[j] @ x + b[j]
        points[i] = x
    return points


def fractal_porosity_field(nx, ny, fractal_dim=1.8, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    H = 2.0 - fractal_dim

    phase = rng.random((ny, nx)) * 2.0 * np.pi

    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    KX, KY = np.meshgrid(kx, ky)
    k_mag = np.sqrt(KX ** 2 + KY ** 2)
    k_mag[0, 0] = 1e-10

    spectrum = k_mag ** (-(2.0 * H + 1.0) / 2.0)
    spectrum[0, 0] = 0.0

    noise = spectrum * np.exp(1j * phase)
    field = np.real(np.fft.ifft2(noise))

    f_min, f_max = np.min(field), np.max(field)
    if abs(f_max - f_min) < 1e-14:
        return np.zeros((ny, nx))
    porosity = (field - f_min) / (f_max - f_min)
    return porosity
