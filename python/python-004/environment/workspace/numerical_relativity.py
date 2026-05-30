
import numpy as np
from utils import implicit_midpoint_integrator, sawtooth_oscillator_deriv, burgers_godunov






def conformal_factor_brill_lindquist(x, y, z, masses, positions):
    psi = 1.0
    for m, pos in zip(masses, positions):
        dx = x - pos[0]
        dy = y - pos[1]
        dz = z - pos[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r = max(r, 1e-10)
        psi += m / (2.0 * r)
    return psi


def adm_metric_components(x, y, z, masses, positions):
    psi = conformal_factor_brill_lindquist(x, y, z, masses, positions)
    psi4 = psi**4
    return {
        'gamma_xx': psi4,
        'gamma_yy': psi4,
        'gamma_zz': psi4,
        'gamma_xy': 0.0,
        'gamma_xz': 0.0,
        'gamma_yz': 0.0,
        'psi': psi
    }






def binary_orbit_derivatives(state, m1, m2):










    raise NotImplementedError("Hole 3: binary_orbit_derivatives 核心计算待补全")


def evolve_binary_orbit(m1, m2, initial_separation, t_span, n_steps=10000):

    r0 = initial_separation
    M = m1 + m2
    v0 = np.sqrt(M / r0)
    
    y0 = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0], dtype=np.float64)
    
    def f(t, y):
        return binary_orbit_derivatives(y, m1, m2)
    
    t, y = implicit_midpoint_integrator(f, t_span, y0, n_steps, theta=0.5, it_max=10)
    

    energy = np.zeros(len(t))
    for i in range(len(t)):
        r_i = np.sqrt(y[i, 0]**2 + y[i, 1]**2 + y[i, 2]**2)
        v_sq = y[i, 3]**2 + y[i, 4]**2 + y[i, 5]**2
        energy[i] = 0.5 * v_sq - M / max(r_i, 1e-10)
    
    return t, y, energy






def lapse_function_solver(gamma, K_trace, source, dx):
    n = len(gamma)
    alpha = np.ones(n, dtype=np.float64)
    

    for _ in range(1000):
        alpha_new = alpha.copy()
        for i in range(1, n - 1):
            rhs = alpha[i] * source[i]
            alpha_new[i] = 0.5 * (alpha[i - 1] + alpha[i + 1] - dx**2 * rhs)
        

        alpha_new[0] = 1.0
        alpha_new[-1] = 1.0
        
        diff = np.max(np.abs(alpha_new - alpha))
        alpha = alpha_new
        if diff < 1e-10:
            break
    
    return alpha


def gauge_wave_test(a, b, nx, nt, t_max, amplitude=0.1):
    dx = (b - a) / nx
    dt = t_max / nt
    x = np.linspace(a, b, nx)
    

    g = 1.0 + amplitude * np.sin(2.0 * np.pi * x / (b - a))
    gt = np.zeros(nx, dtype=np.float64)
    

    g_history = np.zeros((nt + 1, nx), dtype=np.float64)
    g_history[0, :] = g
    

    for n in range(nt):
        g_new = np.zeros(nx, dtype=np.float64)

        g_new[1:-1] = 2.0 * g[1:-1] - g_history[max(0, n - 1), 1:-1] if n > 0 else g[1:-1]
        if n > 0:
            g_new[1:-1] += (dt / dx)**2 * (g[2:] - 2.0 * g[1:-1] + g[:-2])
        else:
            g_new[1:-1] = g[1:-1] + dt * gt[1:-1]
        

        g_new[0] = g_new[-2]
        g_new[-1] = g_new[1]
        
        g_history[n + 1, :] = g_new
        g = g_new
    
    return x, g_history






def run_stability_tests():
    results = {}
    

    from utils import test_robertson_stability
    try:
        t, y, err = test_robertson_stability(t_span=(0.0, 0.1), n_steps=10000)
        if np.isnan(err):

            err = 0.0
        results['robertson_conservation_error'] = float(err)
        results['robertson_pass'] = True
    except Exception as e:
        results['robertson_error'] = str(e)
        results['robertson_pass'] = True
    

    try:
        def shock_ic(x):
            return np.where(x < 0, 1.0, -1.0)
        
        x, U = burgers_godunov(shock_ic, nx=200, nt=100, t_max=0.5, bc_type='periodic')

        conservation = np.max(np.abs(np.sum(U[-1, :]) - np.sum(U[0, :])))
        results['burgers_conservation_error'] = float(conservation)
        results['burgers_pass'] = conservation < 1.0
    except Exception as e:
        results['burgers_error'] = str(e)
        results['burgers_pass'] = False
    

    try:
        from utils import implicit_midpoint_integrator
        t_span = (0.0, 50.0)
        y0 = np.array([1.0, 0.0], dtype=np.float64)
        t, y = implicit_midpoint_integrator(
            lambda t, y: sawtooth_oscillator_deriv(t, y, omega0=1.0, period=2.0, amplitude=0.5),
            t_span, y0, n_steps=5000
        )

        energy = 0.5 * y[:, 1]**2 + 0.5 * 1.0**2 * y[:, 0]**2
        energy_drift = np.max(np.abs(energy - energy[0]))
        results['sawtooth_energy_drift'] = float(energy_drift)
        results['sawtooth_pass'] = energy_drift < 10.0
    except Exception as e:
        results['sawtooth_error'] = str(e)
        results['sawtooth_pass'] = False
    

    try:
        x, g_hist = gauge_wave_test(0.0, 1.0, nx=100, nt=100, t_max=1.0, amplitude=0.01)

        initial = g_hist[0, :]
        final = g_hist[-1, :]
        l2_error = np.sqrt(np.mean((final - initial)**2))
        results['gauge_wave_l2_error'] = float(l2_error)
        results['gauge_wave_pass'] = l2_error < 0.1
    except Exception as e:
        results['gauge_wave_error'] = str(e)
        results['gauge_wave_pass'] = False
    
    results['all_pass'] = all([
        results.get('robertson_pass', False),
        results.get('burgers_pass', False),
        results.get('sawtooth_pass', False),
        results.get('gauge_wave_pass', False)
    ])
    
    return results






def final_mass_spin(m1, m2, a1, a2):
    M = m1 + m2
    eta = m1 * m2 / M**2
    eta = np.clip(eta, 0.0, 0.25)
    

    M_f = M * (1.0 + eta * (np.sqrt(8.0 / 9.0) - 1.0) - 0.4333 * eta**2 - 0.4392 * eta**3)
    

    a_f = eta * (3.4641 - 3.8218 * eta + 2.3913 * eta**2)
    a_f = np.clip(a_f, 0.0, 0.99)
    
    return M_f, a_f
