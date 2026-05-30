
import numpy as np






def brownian_motion_simulation(m_dim: int, n_steps: int,
                               diffusion_coeff: float,
                               total_time: float,
                               seed: int = None) -> np.ndarray:
    if m_dim < 1 or n_steps < 2:
        raise ValueError("Invalid simulation parameters")
    if diffusion_coeff <= 0 or total_time <= 0:
        raise ValueError("Physical parameters must be positive")

    if seed is not None:
        np.random.seed(seed)

    dt = total_time / (n_steps - 1)

    step_std = np.sqrt(2.0 * m_dim * diffusion_coeff * dt)


    if m_dim == 1:
        dx = step_std * np.random.randn(1, n_steps - 1)
    else:

        a = np.random.randn(m_dim, n_steps - 1)
        norms = np.linalg.norm(a, axis=0, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        directions = a / norms
        step_sizes = step_std * np.random.randn(1, n_steps - 1)
        dx = directions * step_sizes

    x = np.zeros((m_dim, n_steps))
    x[:, 1:] = np.cumsum(dx, axis=1)
    return x


def brownian_displacement_simulation(k_trials: int, n_steps: int,
                                     m_dim: int, diffusion_coeff: float,
                                     total_time: float,
                                     seed: int = None) -> np.ndarray:
    if k_trials < 1:
        raise ValueError("k_trials must be positive")

    if seed is not None:
        np.random.seed(seed)

    dsq = np.zeros((k_trials, n_steps))
    for k in range(k_trials):
        traj = brownian_motion_simulation(m_dim, n_steps, diffusion_coeff,
                                          total_time, seed=None)
        dsq[k, :] = np.sum(traj ** 2, axis=0)
    return dsq


def verify_einstein_relation(dsq: np.ndarray, m_dim: int,
                             diffusion_coeff: float, total_time: float) -> dict:
    k_trials, n_steps = dsq.shape
    mean_dsq = np.mean(dsq, axis=0)
    t = np.linspace(0, total_time, n_steps)


    valid = t > 1e-12
    if np.count_nonzero(valid) < 2:
        return {"slope": 0.0, "theoretical_slope": 0.0, "relative_error": 1.0}

    slope, intercept = np.polyfit(t[valid], mean_dsq[valid], 1)
    theoretical = 2.0 * m_dim * diffusion_coeff
    rel_err = abs(slope - theoretical) / (theoretical + 1e-15)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "theoretical_slope": float(theoretical),
        "relative_error": float(rel_err)
    }






def effective_diffusion_plasma(temperature_kelvin: float = 310.15,
                               particle_radius_nm: float = 100.0,
                               plasma_viscosity_pa_s: float = 0.0012) -> float:
    k_B = 1.380649e-23
    R_m = particle_radius_nm * 1e-9
    if R_m <= 0 or plasma_viscosity_pa_s <= 0 or temperature_kelvin <= 0:
        raise ValueError("Physical parameters must be positive")
    return k_B * temperature_kelvin / (6.0 * np.pi * plasma_viscosity_pa_s * R_m)


def einstein_viscosity_correction(hematocrit: float) -> float:
    if not (0.0 <= hematocrit <= 1.0):
        raise ValueError("Hematocrit must be in [0, 1]")
    phi = hematocrit
    return 1.0 + 2.5 * phi + 6.2 * phi * phi


def peclet_number(shear_rate: float, particle_radius_nm: float,
                  diffusion_coeff: float) -> float:
    R = particle_radius_nm * 1e-9
    if diffusion_coeff <= 0:
        raise ValueError("Diffusion coefficient must be positive")
    return shear_rate * R * R / diffusion_coeff


def ldl_wall_flux_estimate(wss_pa: float, diffusion_coeff: float,
                           wall_permeability: float = 1e-8) -> float:
    k_wss = 0.05
    P_eff = wall_permeability * (1.0 + k_wss * wss_pa)
    return max(P_eff, 0.0)
