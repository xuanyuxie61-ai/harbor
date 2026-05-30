
import numpy as np





SOBOL_DIRECTIONS = {
    1: [1], 2: [1, 3], 3: [1, 3, 1], 4: [1, 1, 1],
    5: [1, 3, 3], 6: [1, 3, 5, 15], 7: [1, 1, 5, 17],
    8: [1, 5, 5, 13], 9: [1, 5, 7, 11], 10: [1, 7, 11, 19]
}
SOBOL_POLY_DEGREE = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4,
    8: 5, 9: 5, 10: 5
}
SOBOL_POLY = {
    1: 0, 2: 1, 3: 3, 4: 3, 5: 5, 6: 3, 7: 3,
    8: 9, 9: 9, 10: 5
}


def i4_bit_lo0(n):
    if n <= 0:
        return 1
    bit = 0
    while True:
        bit += 1
        mask = 1 << (bit - 1)
        if (n & mask) == 0:
            return bit


def sobol_generate(dim_num, n, skip=0):
    if dim_num < 1 or dim_num > 10:
        raise ValueError("dim_num must be between 1 and 10")
    max_bits = 30
    recipd = 1.0 / (1 << max_bits)

    v = np.zeros((dim_num, max_bits), dtype=np.uint32)
    for j in range(max_bits):
        v[0, j] = 1 << (max_bits - 1 - j)

    for dim in range(1, dim_num):
        poly = SOBOL_POLY[dim + 1]
        degree = SOBOL_POLY_DEGREE[dim + 1]
        directions = SOBOL_DIRECTIONS[dim + 1]
        for j in range(degree):
            v[dim, j] = directions[j] << (max_bits - 1 - j)
        for j in range(degree, max_bits):
            new_v = v[dim, j - degree] >> degree
            for k in range(degree):
                if (poly >> k) & 1:
                    new_v ^= v[dim, j - k - 1]
            v[dim, j] = new_v

    lastq = np.zeros(dim_num, dtype=np.uint32)
    points = np.zeros((n, dim_num), dtype=np.float64)
    seed = skip
    for i in range(n):
        seed += 1
        l = i4_bit_lo0(seed)
        if l > max_bits:
            l = max_bits
        for dim in range(dim_num):
            lastq[dim] ^= v[dim, l - 1]
            points[i, dim] = lastq[dim] * recipd
    return points




def euler_maruyama_sde(X0, t_span, dt, drift_func, diffusion_func, n_ensemble=1):
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_array = np.linspace(t_start, t_end, n_steps)
    n_state = X0.shape[1] if len(X0.shape) > 1 else 1
    if len(X0.shape) == 1:
        X0 = X0.reshape(1, -1)

    X_history = np.zeros((n_steps, n_ensemble, n_state), dtype=np.float64)
    X_history[0] = X0
    X = X0.copy()

    for i in range(1, n_steps):
        t = t_array[i - 1]
        mu = drift_func(X, t)
        sigma = diffusion_func(X, t)
        dW = np.random.normal(0.0, np.sqrt(dt), size=(n_ensemble, n_state))
        X = X + mu * dt + sigma * dW
        X = np.clip(X, 50.0, 400.0)
        X_history[i] = X

    return t_array, X_history


def generate_ensemble_perturbations(n_ensemble, n_nodes, t_span, dt, perturbation_scale=0.5):
    lambda_ou = 1.0 / 2.0
    sigma_ou = perturbation_scale
    X0 = np.zeros((n_ensemble, n_nodes), dtype=np.float64)

    def drift(X, t):
        return -lambda_ou * X

    def diffusion(X, t):
        return sigma_ou * np.ones_like(X)

    t_array, perturbations = euler_maruyama_sde(
        X0, t_span, dt, drift, diffusion, n_ensemble
    )
    return t_array, perturbations


def generate_initial_ensemble(n_ensemble, n_nodes, T_climatology=288.0, T_amplitude=5.0):
    nd = min(10, n_ensemble)
    sobol_pts = sobol_generate(nd, n_ensemble, skip=100)
    ensemble = np.zeros((n_ensemble, n_nodes), dtype=np.float64)

    for i in range(n_ensemble):
        base = T_climatology + T_amplitude * (2.0 * sobol_pts[i, 0] - 1.0)
        spatial = np.zeros(n_nodes)
        for j in range(1, min(5, nd)):
            phase = 2.0 * np.pi * sobol_pts[i, j]
            spatial += np.sin(np.arange(n_nodes) * 0.1 + phase) / j
        ensemble[i] = base + 0.5 * spatial

    return np.clip(ensemble, 200.0, 350.0)
