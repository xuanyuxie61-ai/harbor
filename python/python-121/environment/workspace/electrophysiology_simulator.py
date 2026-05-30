
import numpy as np
from math import sqrt, exp, pi


def create_stimulus_mask(nx, ny, x_range, y_range, dx, dy):
    mask = np.zeros((nx, ny), dtype=bool)
    
    ix_min = int(x_range[0] * nx)
    ix_max = int(x_range[1] * nx)
    iy_min = int(y_range[0] * ny)
    iy_max = int(y_range[1] * ny)
    
    mask[ix_min:ix_max, iy_min:iy_max] = True
    return mask


def square_wave_stimulus(t, amplitude=-1.0, duration=2.0, period=300.0):
    if (t % period) < duration:
        return amplitude
    return 0.0


def compute_wavefront_velocity(u_history, dx, dt, threshold=0.5):
    n_time, nx, ny = u_history.shape
    positions = []
    times = []
    
    for t_idx in range(n_time):

        above_threshold = u_history[t_idx] > threshold
        if np.any(above_threshold):

            indices = np.argwhere(above_threshold)
            if len(indices) > 0:
                cx = np.mean(indices[:, 0]) * dx
                cy = np.mean(indices[:, 1]) * dx
                positions.append((cx, cy))
                times.append(t_idx)
    
    if len(positions) < 2:
        return 0.0, positions
    
    positions = np.array(positions)
    times = np.array(times) * dt
    


    if len(times) > 1:
        vx = np.polyfit(times, positions[:, 0], 1)[0] if len(times) > 1 else 0.0
        vy = np.polyfit(times, positions[:, 1], 1)[0] if len(times) > 1 else 0.0
        velocity = sqrt(vx ** 2 + vy ** 2)
    else:
        velocity = 0.0
    
    return velocity, positions


def detect_reentrant_activity(u_history, threshold=0.5, min_cycle_length=50):
    n_time, nx, ny = u_history.shape
    phase_singularities = []
    

    for t_idx in range(0, n_time, max(1, n_time // 20)):
        u = u_history[t_idx]
        


        grad_x = np.zeros((nx, ny))
        grad_y = np.zeros((nx, ny))
        
        grad_x[1:nx - 1, :] = u[2:nx, :] - u[0:nx - 2, :]
        grad_y[:, 1:ny - 1] = u[:, 2:ny] - u[:, 0:ny - 2]
        

        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        

        for i in range(2, nx - 2):
            for j in range(2, ny - 2):
                local = grad_mag[i - 1:i + 2, j - 1:j + 2]
                if grad_mag[i, j] == np.min(local) and grad_mag[i, j] < 0.1:
                    phase_singularities.append((t_idx, i, j))
    
    reentrant_detected = len(phase_singularities) > 0
    
    return reentrant_detected, phase_singularities


def compute_action_potential_duration(u_history, dt, threshold_up=0.3, threshold_down=0.3):
    if u_history.ndim == 3:

        nx, ny = u_history.shape[1], u_history.shape[2]
        u_series = u_history[:, nx // 2, ny // 2]
    else:
        u_series = u_history
    
    n_time = len(u_series)
    

    upstrokes = []
    downstrokes = []
    
    for i in range(1, n_time):
        if u_series[i - 1] < threshold_up <= u_series[i]:
            upstrokes.append(i)
        if u_series[i - 1] > threshold_down >= u_series[i]:
            downstrokes.append(i)
    

    apds = []
    for up in upstrokes:
        for down in downstrokes:
            if down > up:
                apds.append((down - up) * dt)
                break
    
    if len(apds) > 0:
        return np.mean(apds)
    return 0.0


def compute_refractory_period(u_history, dt, threshold=0.1):
    apd = compute_action_potential_duration(u_history, dt)

    return apd + 20.0


def compute_wavelength(velocity, apd):
    if velocity <= 0 or apd <= 0:
        return 0.0
    return velocity * apd


def arrhythmia_risk_index(u_history, dx, dt, threshold=0.5):
    velocity, _ = compute_wavefront_velocity(u_history, dx, dt, threshold)
    apd = compute_action_potential_duration(u_history, dt)
    


    

    v_normal = 0.07
    v_abnormality = abs(velocity - v_normal) / v_normal if v_normal > 0 else 0.0
    v_abnormality = min(1.0, v_abnormality)
    

    apd_normal = 300.0
    apd_abnormality = abs(apd - apd_normal) / apd_normal if apd_normal > 0 else 0.0
    apd_abnormality = min(1.0, apd_abnormality)
    

    risk = 0.5 * v_abnormality + 0.5 * apd_abnormality
    
    return risk


def run_full_simulation(nx=64, ny=64, T=500.0, dt=0.05, dx=0.025,
                         D_f=0.001, D_t=0.0002,
                         a=0.1, k=8.0, mu1=0.2, mu2=0.3, eps=0.002,
                         solver='adi',
                         n_stimuli=3, stim_period=150.0,
                         fiber_model='parallel',
                         add_noise=False, noise_level=0.01):
    from ion_channel_dynamics import aliev_panfilov_reaction, generate_ion_channel_noise
    from tissue_reaction_diffusion import (solve_reaction_diffusion_2d,
                                            generate_fiber_angle_field,
                                            build_diffusion_tensor)
    from mesh_generator import generate_cardiac_mesh
    from linear_algebra_core import stability_eigenvalue_analysis
    

    fiber_angle = generate_fiber_angle_field(nx, ny, fiber_model)
    Dxx, Dxy, Dyy = build_diffusion_tensor(D_f, D_t, fiber_angle)
    

    u0 = np.zeros((nx, ny))
    v0 = np.zeros((nx, ny))
    

    u0 += 0.01 * np.random.randn(nx, ny)
    

    stim_mask = create_stimulus_mask(nx, ny, (0.0, 0.15), (0.0, 0.15), dx, dx)
    
    def stimulus(t):
        for i in range(n_stimuli):
            t_stim = i * stim_period
            if t_stim <= t < t_stim + 2.0:
                return -1.0
        return 0.0
    

    reaction_params = {'a': a, 'k': k, 'mu1': mu1, 'mu2': mu2, 'eps': eps}
    

    D_eff = (Dxx, Dxy, Dyy)
    

    print(f"  Running reaction-diffusion simulation: nx={nx}, ny={ny}, T={T}ms, solver={solver}")
    u_hist, v_hist, t_hist = solve_reaction_diffusion_2d(
        u0, v0, D_eff, dx, dx, dt, T,
        aliev_panfilov_reaction, reaction_params,
        solver=solver,
        stimulus_func=stimulus,
        stimulus_region=stim_mask
    )
    

    if add_noise:
        noise = generate_ion_channel_noise(1, dt, T, D_ion=noise_level, alpha=1.5)
        for t_idx in range(min(len(u_hist), len(noise))):
            u_hist[t_idx] += 0.01 * noise[t_idx, 0]
    

    print("  Computing wavefront velocity...")
    velocity, positions = compute_wavefront_velocity(u_hist, dx, dt)
    
    print("  Computing APD...")
    apd = compute_action_potential_duration(u_hist, dt)
    
    print("  Computing wavelength...")
    wavelength = compute_wavelength(velocity, apd)
    
    print("  Computing refractory period...")
    erp = compute_refractory_period(u_hist, dt)
    
    print("  Detecting reentrant activity...")
    reentrant, singularities = detect_reentrant_activity(u_hist)
    
    print("  Computing arrhythmia risk index...")
    risk = arrhythmia_risk_index(u_hist, dx, dt)
    









    raise NotImplementedError("Hole 3: stability analysis 待实现")
    
    results = {
        'u_history': u_hist,
        'v_history': v_hist,
        't_history': t_hist,
        'nx': nx,
        'ny': ny,
        'dx': dx,
        'dt': dt,
        'T': T,
        'velocity': velocity,
        'apd': apd,
        'wavelength': wavelength,
        'erp': erp,
        'reentrant_detected': reentrant,
        'phase_singularities': singularities,
        'risk_index': risk,
        'lambda_max': lambda_max,
        'is_stable': is_stable,
        'solver': solver,
        'fiber_model': fiber_model
    }
    
    return results
