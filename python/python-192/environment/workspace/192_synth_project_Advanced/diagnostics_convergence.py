
import numpy as np
from utils_numerical import safe_divide


def estimate_convergence_order(residuals: list) -> dict:
    if len(residuals) < 10:
        return {'order': None, 'R_squared': 0.0}

    log_r = np.log(np.maximum(residuals, 1e-16))
    x = log_r[:-1]
    y = log_r[1:]


    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx < 1e-14:
        return {'order': None, 'R_squared': 0.0}

    p = ss_xy / ss_xx
    r_sq = ss_xy ** 2 / (ss_xx * np.sum((y - y_mean) ** 2) + 1e-14)

    return {
        'order': float(p),
        'R_squared': float(r_sq),
        'asymptotic_rate': float(10 ** p) if p < 0 else None
    }


def compute_gci(fine: float, medium: float, coarse: float,
                r: float = 2.0, p: float = None, Fs: float = 1.25) -> dict:
    if p is None:

        if abs(fine - medium) < 1e-14 or abs(medium - coarse) < 1e-14:
            p = 2.0
        else:
            p = np.log(abs(coarse - medium) / abs(medium - fine)) / np.log(r)
            p = np.clip(p, 0.5, 4.0)

    epsilon = safe_divide(fine - medium, fine)
    gci = Fs * abs(epsilon) / (r ** p - 1.0)


    asymptotic_range = safe_divide(
        (r ** p - 1.0) * abs(coarse - medium),
        (r ** p * (r ** p - 1.0)) * abs(medium - fine) + 1e-14
    )

    return {
        'p_observed': float(p),
        'gci_fine_medium': float(gci),
        'epsilon': float(epsilon),
        'asymptotic_range': float(asymptotic_range),
        'mesh_acceptable': asymptotic_range > 0.8 and gci < 0.05
    }


def check_energy_conservation(Q_history: list, gamma: float = 1.4) -> dict:
    total_energy = []
    for Q in Q_history:
        E_total = np.sum(Q[..., 3])
        total_energy.append(float(E_total))

    energy = np.array(total_energy)
    if len(energy) < 2:
        return {'energy': energy, 'drift': 0.0, 'max_relative_error': 0.0}


    E0 = energy[0]
    drift = (energy[-1] - E0) / (abs(E0) + 1e-14)


    rel_changes = np.abs(np.diff(energy)) / (np.abs(energy[:-1]) + 1e-14)
    max_rel_error = np.max(rel_changes) if len(rel_changes) > 0 else 0.0

    return {
        'energy': energy,
        'drift': float(drift),
        'max_relative_error': float(max_rel_error),
        'energy_conserved': abs(drift) < 0.01
    }


def compute_mass_flow_rate(Q: np.ndarray, y: np.ndarray, gamma: float = 1.4) -> dict:
    rho = Q[:, :, 0]
    u = safe_divide(Q[:, :, 1], rho)

    ny = len(y)
    dy = np.diff(y)
    dy = np.concatenate([dy, [dy[-1]]])


    m_dot_in = np.sum(rho[:, 0] * u[:, 0] * dy)


    m_dot_out = np.sum(rho[:, -1] * u[:, -1] * dy)


    error = abs(m_dot_in - m_dot_out) / (abs(m_dot_in) + 1e-14)

    return {
        'mass_flow_in': float(m_dot_in),
        'mass_flow_out': float(m_dot_out),
        'relative_error': float(error),
        'mass_conserved': error < 0.05
    }


def monitor_cfl_stability(u: np.ndarray, v: np.ndarray, c: np.ndarray,
                          dx: float, dy: np.ndarray, dt: float,
                          gamma: float = 1.4) -> dict:
    cfl_x = (np.abs(u) + c) * dt / dx
    cfl_y = (np.abs(v) + c) * dt / dy[:, None]

    cfl_x_max = np.max(cfl_x)
    cfl_y_max = np.max(cfl_y)


    nu = 1.0 / 1000.0
    cfl_visc = nu * dt / (dx ** 2)

    stable = (cfl_x_max < 1.0) and (cfl_y_max < 1.0) and (cfl_visc < 0.5)

    return {
        'cfl_x_max': float(cfl_x_max),
        'cfl_y_max': float(cfl_y_max),
        'cfl_viscous': float(cfl_visc),
        'stable': stable,
        'recommendation': 'reduce dt' if not stable else 'stable'
    }


def print_diagnostics_header():
    print("=" * 80)
    print(f"{'Step':>6} {'Time':>10} {'Resid':>12} {'CFL_x':>8} {'CFL_y':>8} {'Energy':>12} {'MassErr':>10}")
    print("-" * 80)


def print_diagnostics_row(step: int, time: float, residual: float,
                          cfl_x: float, cfl_y: float, energy: float, mass_err: float):
    print(f"{step:>6} {time:>10.4f} {residual:>12.4e} {cfl_x:>8.3f} {cfl_y:>8.3f} {energy:>12.4e} {mass_err:>10.4e}")
