
import numpy as np


def l1_norm_discrete(field, dx, dy=None):
    if dy is None:
        vol = dx
    else:
        vol = dx * dy
    return np.sum(np.abs(field)) * vol


def l2_norm_discrete(field, dx, dy=None):
    if dy is None:
        vol = dx
    else:
        vol = dx * dy
    return np.sqrt(np.sum(field ** 2) * vol)


def linfty_norm_discrete(field):
    return np.max(np.abs(field))


def h1_seminorm_discrete(field, dx, dy):
    ny, nx = field.shape
    if nx < 3 or ny < 3:
        return 0.0


    dfdx = np.zeros_like(field)
    dfdy = np.zeros_like(field)
    dfdx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dx)
    dfdy[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dy)

    grad_sq = dfdx ** 2 + dfdy ** 2
    return np.sqrt(np.sum(grad_sq) * dx * dy)


def richardson_error_estimate(u_fine, u_coarse, ratio=2.0, order=2.0):
    denom = ratio ** order - 1.0
    if denom < 1e-15:
        raise ValueError("ratio^order 过于接近 1，无法估计。")
    return (u_fine - u_coarse) / denom


def convergence_order(errors, grid_sizes):
    n = len(errors)
    if n != len(grid_sizes) or n < 2:
        return []

    orders = []
    for i in range(n - 1):
        e1, e2 = errors[i], errors[i + 1]
        h1, h2 = grid_sizes[i], grid_sizes[i + 1]
        if e1 <= 0 or e2 <= 0 or h1 <= 0 or h2 <= 0:
            orders.append(np.nan)
        else:
            p = np.log(e1 / e2) / np.log(h1 / h2)
            orders.append(p)
    return orders


def estimate_temporal_error(state_history, dt_values, order=2.0):
    if len(dt_values) < 2 or len(state_history) < 2:
        return None, None


    final_states = [s[-1] if hasattr(s, '__len__') else s for s in state_history]

    errors = []
    valid_dt = []
    for i in range(len(final_states) - 1):
        diff = np.abs(final_states[i] - final_states[-1])
        if hasattr(diff, '__len__'):
            err = np.max(diff)
        else:
            err = diff
        if err > 1e-15:
            errors.append(err)
            valid_dt.append(dt_values[i])

    if len(errors) < 2:
        return None, None

    log_dt = np.log(valid_dt)
    log_err = np.log(errors)


    A = np.vstack([np.ones_like(log_dt), log_dt]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_err, rcond=None)
    p_est = coeffs[1]
    C_est = np.exp(coeffs[0])

    return p_est, C_est


def compute_solution_quality_metrics(omega, psi, u, v, dx, dy,
                                     solid_mask=None):
    if solid_mask is not None:

        omega_f = np.where(solid_mask, 0.0, omega)
        u_f = np.where(solid_mask, 0.0, u)
        v_f = np.where(solid_mask, 0.0, v)
    else:
        omega_f = omega
        u_f = u
        v_f = v

    omega_l2 = l2_norm_discrete(omega_f, dx, dy)
    psi_l2 = l2_norm_discrete(psi, dx, dy)

    vol = dx * dy
    ke = 0.5 * np.sum(u_f ** 2 + v_f ** 2) * vol
    ens = 0.5 * np.sum(omega_f ** 2) * vol


    div = np.zeros_like(u)
    if u.shape[0] > 2 and u.shape[1] > 2:
        div[1:-1, 1:-1] = (
            (u_f[1:-1, 2:] - u_f[1:-1, :-2]) / (2.0 * dx)
            + (v_f[2:, 1:-1] - v_f[:-2, 1:-1]) / (2.0 * dy)
        )
    div_rms = np.sqrt(np.mean(div ** 2))

    return {
        'omega_l2': omega_l2,
        'psi_l2': psi_l2,
        'kinetic_energy': ke,
        'enstrophy': ens,
        'divergence_rms': div_rms,
    }
