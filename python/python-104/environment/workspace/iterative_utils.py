
import numpy as np


def collatz_step_size(residual_norm, base_step=1.0, min_step=1e-6, max_step=1.0):
    if base_step <= 0:
        raise ValueError("base_step must be positive.")
    if residual_norm < 0:
        raise ValueError("residual_norm must be non-negative.")

    eps = np.finfo(float).eps
    ratio = residual_norm / max(eps, base_step)
    if ratio <= 1.0:
        k = 0
    else:
        k = int(np.floor(np.log2(ratio)))
    denominator = 1.0 + 2.0 ** min(k, 20)
    step = base_step / denominator
    return np.clip(step, min_step, max_step)


def collatz_sequence_length(n0):
    if n0 < 1:
        raise ValueError("n0 must be a positive integer.")
    if not np.isfinite(n0):
        raise ValueError("n0 must be finite.")
    n = int(n0)
    length = 1
    max_iter = 100000
    while n != 1 and length < max_iter:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        length += 1
    return length


def adaptive_relaxation(residual_history, omega_min=0.1, omega_max=1.9):
    if len(residual_history) < 2:
        return 0.5 * (omega_min + omega_max)
    r_k = residual_history[-1]
    r_km1 = residual_history[-2]
    if r_k <= 0:
        return omega_min
    delta = abs(r_k - r_km1) / r_k
    omega = omega_max - (omega_max - omega_min) * np.tanh(delta)
    return float(np.clip(omega, omega_min, omega_max))


def robust_convergence_check(residual, tol_abs=1e-10, tol_rel=1e-8, max_iter=1000, iter_count=0):
    if not np.isfinite(residual):
        return False, "Diverged (non-finite residual)", False
    if residual < 0:
        return False, "Invalid negative residual", False
    if iter_count >= max_iter:
        return False, f"Max iteration {max_iter} reached", False
    if residual <= tol_abs:
        return True, f"Absolute tolerance reached: {residual:.3e}", False
    if iter_count > 0 and residual <= tol_rel:
        return True, f"Relative tolerance reached: {residual:.3e}", False
    return False, "Continuing", True


def anderson_acceleration_iterate(x_new, x_old, m=3, history_F=None, history_X=None):
    if history_F is None:
        history_F = []
    if history_X is None:
        history_X = []

    fk = x_new - x_old
    history_F.append(fk)
    history_X.append(x_new)

    if len(history_F) < 2:
        return x_new, history_F, history_X

    m_eff = min(m, len(history_F) - 1)
    n = len(fk)
    F_mat = np.column_stack([history_F[-1] - history_F[-(j + 2)] for j in range(m_eff)])
    rhs = -history_F[-1]

    try:
        gamma, _, _, _ = np.linalg.lstsq(F_mat, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return x_new, history_F, history_X

    alpha = np.zeros(m_eff + 1)
    alpha[0] = 1.0 - np.sum(gamma)
    for j in range(m_eff):
        alpha[j + 1] = gamma[j]

    x_aa = np.zeros(n)
    for j in range(m_eff + 1):
        x_aa += alpha[j] * history_X[-(j + 1)]

    if len(history_F) > m + 5:
        history_F = history_F[-(m + 5):]
        history_X = history_X[-(m + 5):]

    return x_aa, history_F, history_X
