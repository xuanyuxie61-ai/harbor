
import numpy as np
from photon_transport_ode import ode_midpoint_solve, fitzhugh_nagumo_deriv, glycolysis_deriv






def integrate_fitzhugh_nagumo(y0, t_span, n_steps=1000, a=0.7, b=0.8, c=12.5, d=0.5):
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span

    def f(t, y):
        return fitzhugh_nagumo_deriv(t, y, a, b, c, d)

    t, y = ode_midpoint_solve(f, t0, t1, y0, n_steps, theta=0.5, it_max=5)
    return t, y






def integrate_glycolysis(y0, t_span, n_steps=1000, a=0.08, b=0.6):
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span

    def f(t, y):
        return glycolysis_deriv(t, y, a, b)

    t, y = ode_midpoint_solve(f, t0, t1, y0, n_steps, theta=0.5, it_max=5)
    return t, y






def compute_refractive_index_timeseries(t, v_trace, u_trace,
                                        n0=1.33, alpha_eo=5e-4, alpha_thermo=2e-3):
    v_trace = np.asarray(v_trace, dtype=float)
    u_trace = np.asarray(u_trace, dtype=float)
    n_t = n0 + alpha_eo * v_trace + alpha_thermo * u_trace

    n_t = np.clip(n_t, 1.30, 1.50)
    return n_t






def phase_shift_from_dn(dn, optical_path_length, lambda0=0.84):
    if lambda0 <= 0:
        raise ValueError("lambda0 must be positive.")
    delta_phi = (4.0 * np.pi / lambda0) * optical_path_length * dn
    return delta_phi






def simulate_functional_oct_signal(t_bio, bio_params, oct_params):
    n0 = oct_params.get('n0', 1.33)
    alpha_eo = oct_params.get('alpha_eo', 5e-4)
    alpha_thermo = oct_params.get('alpha_thermo', 2e-3)
    lambda0 = oct_params.get('lambda0', 0.84)
    path_length = oct_params.get('path_length', 100.0)

    bio_type = bio_params.get('type', 'FHN')
    y0 = np.asarray(bio_params.get('y0', [0.0, 0.0]), dtype=float)
    t0 = float(t_bio[0])
    t1 = float(t_bio[-1])
    n_steps = min(len(t_bio) - 1, 5000)

    if bio_type.upper() == 'FHN':
        mp = bio_params.get('model_params', {})
        t, y = integrate_fitzhugh_nagumo(
            y0, (t0, t1), n_steps,
            a=mp.get('a', 0.7), b=mp.get('b', 0.8),
            c=mp.get('c', 12.5), d=mp.get('d', 0.5)
        )
        v_trace = y[:, 0]
        u_trace = np.zeros_like(v_trace)
    elif bio_type.upper() == 'GLYCOLYSIS':
        mp = bio_params.get('model_params', {})
        t, y = integrate_glycolysis(
            y0, (t0, t1), n_steps,
            a=mp.get('a', 0.08), b=mp.get('b', 0.6)
        )
        v_trace = np.zeros_like(y[:, 0])
        u_trace = y[:, 0]
    else:
        raise ValueError(f"Unknown bio_type: {bio_type}")

    n_t = compute_refractive_index_timeseries(t, v_trace, u_trace, n0, alpha_eo, alpha_thermo)
    dn = n_t - n0
    delta_phi = phase_shift_from_dn(dn, path_length, lambda0)


    intensity_mod = 0.5 * (1.0 + np.cos(delta_phi))

    return {
        't': t,
        'n_t': n_t,
        'phase_shift': delta_phi,
        'intensity_modulation': intensity_mod,
        'v_trace': v_trace,
        'u_trace': u_trace
    }
