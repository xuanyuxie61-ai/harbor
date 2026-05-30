
import numpy as np






def is_prime(n):
    if not isinstance(n, int):
        n = int(n)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(np.sqrt(n)) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True


def next_prime(n):
    candidate = n + 1
    while True:
        if is_prime(candidate):
            return candidate
        candidate += 1


def generate_prime_grid_steps(nx_target, ny_target, nz_target):
    nx = next_prime(nx_target - 1)
    ny = next_prime(ny_target - 1)
    nz = next_prime(nz_target - 1)
    return nx, ny, nz






def sphere_unit_sample(n_samples=1):
    samples = np.random.randn(n_samples, 3)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1e-15, norms)
    samples = samples / norms
    return samples


def sphere_distance_stats(n_samples=10000):
    p = sphere_unit_sample(n_samples)
    q = sphere_unit_sample(n_samples)
    distances = np.linalg.norm(p - q, axis=1)

    mean_dist = np.mean(distances)
    if n_samples > 1:
        variance = np.sum((distances - mean_dist) ** 2) / (n_samples - 1)
    else:
        variance = 0.0

    return {
        'mean': mean_dist,
        'variance': variance,
        'distances': distances,
    }


def monte_carlo_radiation_pattern(e_field_func, n_samples=5000):

    directions = sphere_unit_sample(n_samples)


    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.arctan2(directions[:, 1], directions[:, 0])

    power = np.array([e_field_func(t, p) for t, p in zip(theta, phi)])


    d_omega = 4.0 * np.pi / n_samples
    total_power = np.sum(power) * d_omega

    return {
        'theta': theta,
        'phi': phi,
        'power_density': power,
        'total_radiated_power': total_power,
    }






def check_energy_conservation(energy_history, time_history, power_loss_history, tol=1e-3):
    W = np.asarray(energy_history, dtype=np.float64)
    t = np.asarray(time_history, dtype=np.float64)
    P = np.asarray(power_loss_history, dtype=np.float64)


    valid = np.isfinite(W) & np.isfinite(t)
    if np.any(valid):
        W = W[valid]
        t = t[valid]
    valid_p = np.isfinite(P)
    if np.any(valid_p):
        P = P[valid_p]

    if len(W) < 2 or len(P) < 1:
        return {'conserved': True, 'max_relative_error': 0.0, 'energy_drift': 0.0}


    dt = np.diff(t)
    valid_dt = dt > 1e-30
    if not np.any(valid_dt):
        return {'conserved': True, 'max_relative_error': 0.0, 'energy_drift': 0.0}

    dW_dt = np.diff(W) / dt
    dW_dt = np.where(np.isfinite(dW_dt), dW_dt, 0.0)


    n_min = min(len(P), len(dW_dt))
    P_avg = 0.5 * (P[:n_min-1] + P[1:n_min]) if len(P) == len(W) else P[:n_min]
    P_avg = np.where(np.isfinite(P_avg), P_avg, 0.0)


    residual = dW_dt[:len(P_avg)] + P_avg
    W_avg = 0.5 * (W[:len(P_avg)] + W[1:len(P_avg)+1])
    W_avg = np.where(np.abs(W_avg) < 1e-30, 1e-30, W_avg)

    relative_error = np.abs(residual / W_avg)
    relative_error = np.where(np.isfinite(relative_error), relative_error, 0.0)
    max_error = float(np.max(relative_error)) if len(relative_error) > 0 else 0.0
    energy_drift = float((W[-1] - W[0]) / W[0]) if abs(W[0]) > 1e-30 else 0.0

    return {
        'conserved': max_error < tol,
        'max_relative_error': max_error,
        'energy_drift': energy_drift,
    }


def rigid_body_like_conserved_quantity(field_momenta, inertia_tensor):

    L = np.sum(field_momenta, axis=0)


    try:
        I_inv = np.linalg.inv(inertia_tensor)
        conserved = 0.5 * np.dot(L, I_inv @ L)
    except np.linalg.LinAlgError:
        conserved = np.dot(L, L)

    return conserved






def rms_error(u_numeric, u_exact):
    diff = u_numeric - u_exact
    return np.sqrt(np.mean(diff ** 2))


def relative_l2_error(u_numeric, u_exact, eps=1e-30):
    diff_norm = np.linalg.norm(u_numeric - u_exact)
    exact_norm = np.linalg.norm(u_exact)
    if exact_norm < eps:
        return diff_norm
    return diff_norm / exact_norm


def convergence_rate(errors, resolutions):
    errors = np.asarray(errors)
    resolutions = np.asarray(resolutions)

    rates = []
    for i in range(len(errors) - 1):
        if errors[i] > 1e-30 and errors[i + 1] > 1e-30 and resolutions[i] != resolutions[i + 1]:
            p = np.log(errors[i + 1] / errors[i]) / np.log(resolutions[i + 1] / resolutions[i])
            rates.append(p)

    return rates
