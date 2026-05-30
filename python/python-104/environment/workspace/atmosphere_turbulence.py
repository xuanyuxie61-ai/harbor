
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift




def hufnagel_valley_cnsquared(h, v_wind=21.0, A_ground=1.7e-14):
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    term1 = 0.00594 * (v_wind / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
    term2 = 2.7e-16 * np.exp(-h / 1500.0)
    term3 = A_ground * np.exp(-h / 100.0)
    return term1 + term2 + term3


def fried_parameter(wavelength, Cn2_integral):
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    if Cn2_integral < 0:
        raise ValueError("Cn2_integral must be non-negative.")
    k = 2.0 * np.pi / wavelength
    r0 = (0.423 * k ** 2 * Cn2_integral) ** (-3.0 / 5.0)
    return r0


def kolmogorov_psd(fx, fy, r0):
    f2 = fx ** 2 + fy ** 2
    f2 = np.where(f2 < 1e-20, 1e-20, f2)
    Phi = 0.023 * (r0 ** (-5.0 / 3.0)) * (f2 ** (-11.0 / 6.0))
    return Phi


def generate_phase_screen(grid_size, pixel_scale, r0, L0=30.0, seed=None):
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2.")
    if pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive.")
    if r0 <= 0:
        raise ValueError("r0 must be positive.")

    if seed is not None:
        np.random.seed(seed)


    freq = np.fft.fftfreq(grid_size, d=pixel_scale)
    fx, fy = np.meshgrid(freq, freq)
    f2 = fx ** 2 + fy ** 2


    f0 = 1.0 / L0
    f2_safe = np.where(f2 < 1e-20, 1e-20, f2)
    Phi = 0.023 * (r0 ** (-5.0 / 3.0)) * (f2_safe + f0 ** 2) ** (-11.0 / 6.0)


    W_real = np.random.normal(0, 1, (grid_size, grid_size))
    W_imag = np.random.normal(0, 1, (grid_size, grid_size))
    W = W_real + 1j * W_imag


    if grid_size % 2 == 0:
        W[0, grid_size // 2] = np.real(W[0, grid_size // 2])
        W[grid_size // 2, 0] = np.real(W[grid_size // 2, 0])
        W[grid_size // 2, grid_size // 2] = np.real(W[grid_size // 2, grid_size // 2])

    spectrum = W * np.sqrt(Phi)
    phase = np.fft.ifft2(spectrum).real * (grid_size ** 2)


    x = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X ** 2 + Y ** 2)
    mask = rho <= 1.0


    phase -= np.mean(phase[mask])

    return phase, mask




def barenblatt_pme_solution(x, t, m=3.0, c=None, delta=1.0):
    if m <= 1.0:
        raise ValueError("m must be > 1 for PME.")
    if t + delta <= 0:
        raise ValueError("t + delta must be positive.")
    if c is None:
        c = np.sqrt(3.0) / 15.0

    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))

    factor = c - gamma * (x / ((t + delta) ** beta)) ** 2
    u = np.where(factor > 0, (t + delta) ** (-beta) * (factor ** alpha), 0.0)
    return u


def pme_diffusion_phase_correction(phase, pixel_scale, D_eff=1e-4, m=3.0, dt=1e-3, n_steps=10):
    if D_eff < 0:
        raise ValueError("D_eff must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive.")

    u = phase.copy()
    dx2 = pixel_scale ** 2
    for _ in range(n_steps):
        v = u ** m
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            v[2:, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2] - 4.0 * v[1:-1, 1:-1]
        ) / dx2
        u = u + dt * D_eff * lap
    return u




def photochemical_refractive_index_ode(y, t, td=86400.0, k2=1e-2, k3=1e-12, q_heat=1e-6):
    if len(y) != 4:
        raise ValueError("State vector y must have length 4.")
    k1 = 0.01 * max(0.0, np.sin(2.0 * np.pi * t / td))
    k_thermal = 1e-3
    k_absorb = 1e-6

    dn, dT, o3, uv = y
    ddn_dt = q_heat * k1 * o3 - k2 * dn - k3 * dn * dT
    ddT_dt = q_heat * k1 * o3 - k_thermal * dT
    do3_dt = -k1 * o3 + k3 * dn * dT
    duv_dt = -k_absorb * uv * o3 + 0.01 * max(0.0, np.sin(2.0 * np.pi * t / td))

    return np.array([ddn_dt, ddT_dt, do3_dt, duv_dt], dtype=np.float64)


def rk4_integrate_thermal(y0, t_span, n_steps=1000):
    if len(y0) != 4:
        raise ValueError("y0 must have length 4.")
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("t_span[1] must be > t_span[0].")
    h = (tf - t0) / n_steps

    y = np.array(y0, dtype=np.float64)
    t = t0
    trajectory = [y.copy()]

    for _ in range(n_steps):
        k1 = h * photochemical_refractive_index_ode(y, t)
        k2 = h * photochemical_refractive_index_ode(y + 0.5 * k1, t + 0.5 * h)
        k3 = h * photochemical_refractive_index_ode(y + 0.5 * k2, t + 0.5 * h)
        k4 = h * photochemical_refractive_index_ode(y + k3, t + h)
        y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t += h
        trajectory.append(y.copy())

    return np.array(trajectory)


def apply_thermal_photochemical_phase(phase_base, grid_size, t_sim=100.0):
    y0 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    traj = rk4_integrate_thermal(y0, (0.0, t_sim), n_steps=500)
    avg_dn = np.mean(traj[:, 0])

    x = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X ** 2 + Y ** 2)
    mask = rho <= 1.0


    gaussian_weight = np.exp(-rho ** 2 / 0.5)
    phase_thermal = phase_base + avg_dn * gaussian_weight * mask
    return phase_thermal




def generate_turbulent_phase_screen(grid_size=256, D_aperture=1.0,
                                     wavelength=500e-9, seeing=1.0,
                                     apply_pme_correction=True,
                                     apply_thermal_perturbation=True,
                                     seed=None):
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4.")
    if D_aperture <= 0:
        raise ValueError("D_aperture must be positive.")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")

    pixel_scale = D_aperture / grid_size


    r0 = wavelength / (seeing / 206265.0)
    r0 = max(r0, 1e-3)

    phase, mask = generate_phase_screen(grid_size, pixel_scale, r0, seed=seed)

    if apply_pme_correction:
        phase = pme_diffusion_phase_correction(phase, pixel_scale, D_eff=1e-4, m=3.0, dt=1e-4, n_steps=5)

    if apply_thermal_perturbation:
        phase = apply_thermal_photochemical_phase(phase, grid_size, t_sim=50.0)


    phase[mask] -= np.mean(phase[mask])

    return phase, r0, mask
