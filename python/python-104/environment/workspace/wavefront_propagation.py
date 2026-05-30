
import numpy as np


def opd_to_complex_amplitude(phase, wavelength):
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    k = 2.0 * np.pi / wavelength
    amplitude = np.exp(1j * k * phase)
    return amplitude


def fresnel_propagation(amplitude_in, pixel_scale, z, wavelength):
    if z < 0:
        raise ValueError("Propagation distance z must be non-negative.")
    if pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive.")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")

    N = amplitude_in.shape[0]
    if N != amplitude_in.shape[1]:
        raise ValueError("Input amplitude must be square.")

    freq = np.fft.fftfreq(N, d=pixel_scale)
    fx, fy = np.meshgrid(freq, freq)
    k = 2.0 * np.pi / wavelength

    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (fx ** 2 + fy ** 2))

    U_fft = np.fft.fft2(amplitude_in)
    U_out = np.fft.ifft2(U_fft * H)
    return U_out


def compute_wavefront_curvature_hessian(phase, pixel_scale):
    if pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive.")
    dx2 = pixel_scale ** 2
    W = phase

    W_xx = np.zeros_like(W)
    W_yy = np.zeros_like(W)
    W_xy = np.zeros_like(W)

    W_xx[1:-1, 1:-1] = (W[2:, 1:-1] - 2.0 * W[1:-1, 1:-1] + W[:-2, 1:-1]) / dx2
    W_yy[1:-1, 1:-1] = (W[1:-1, 2:] - 2.0 * W[1:-1, 1:-1] + W[1:-1, :-2]) / dx2
    W_xy[1:-1, 1:-1] = (
        W[2:, 2:] - W[2:, :-2] - W[:-2, 2:] + W[:-2, :-2]
    ) / (4.0 * pixel_scale ** 2)

    return W_xx, W_yy, W_xy


def detect_caustic_singularities(phase, pixel_scale, mask):
    W_xx, W_yy, W_xy = compute_wavefront_curvature_hessian(phase, pixel_scale)
    det_H = W_xx * W_yy - W_xy ** 2


    caustic_region = (det_H < 0) & mask


    laplacian = np.zeros_like(det_H)
    laplacian[1:-1, 1:-1] = (
        det_H[2:, 1:-1] + det_H[:-2, 1:-1] + det_H[1:-1, 2:] + det_H[1:-1, :-2]
        - 4.0 * det_H[1:-1, 1:-1]
    )

    singularity = caustic_region & (laplacian > 0) & mask
    return singularity, det_H


def caustic_line_density(n, m, num_points=500):
    if n < 3:
        raise ValueError("n must be >= 3.")
    if m < 1:
        raise ValueError("m must be >= 1.")

    theta = np.linspace(0, 2.0 * np.pi, n + 1)[:-1]
    z = np.exp(1j * theta)

    lines = []
    for j in range(n):
        j_next = (j + 1) % n
        j_conn = ((j * m) % n)
        p1 = np.array([z[j_next].real, z[j_next].imag])
        p2 = np.array([z[j_conn].real, z[j_conn].imag])
        lines.append((p1, p2))

    return lines


def pyramid_ray_tracing_grid(n_layers, aperture_radius=1.0):
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1.")

    rays = []
    for k in range(n_layers, -1, -1):
        z = k / n_layers
        r_layer = aperture_radius * (1.0 - z)
        if r_layer <= 0:
            continue
        n_pts = max(2 * k + 1, 3)
        x_vals = np.linspace(-r_layer, r_layer, n_pts)
        y_vals = np.linspace(-r_layer, r_layer, n_pts)
        for xv in x_vals:
            for yv in y_vals:
                origin = np.array([xv, yv, z])
                direction = np.array([0.0, 0.0, 1.0])
                rays.append((origin, direction))
    return rays


def compute_strehl_ratio(phase_corrected, wavelength, mask):
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    phi = phase_corrected[mask]
    if len(phi) == 0:
        return 0.0
    area = np.sum(mask)
    integral = np.sum(np.exp(1j * phi))
    S = (np.abs(integral) ** 2) / (area ** 2)
    return float(S)


def compute_modulation_transfer_function(phase, wavelength, pixel_scale, mask):
    if wavelength <= 0 or pixel_scale <= 0:
        raise ValueError("wavelength and pixel_scale must be positive.")

    N = phase.shape[0]
    k = 2.0 * np.pi / wavelength
    pupil = np.zeros_like(phase, dtype=np.complex128)
    pupil[mask] = np.exp(1j * k * phase[mask])


    otf = np.fft.ifft2(np.abs(np.fft.fft2(pupil)) ** 2)
    otf = np.fft.fftshift(otf)
    otf_max = np.max(np.abs(otf))
    if otf_max < 1e-30:
        return np.zeros_like(otf)
    mtf = np.abs(otf) / otf_max
    return mtf
