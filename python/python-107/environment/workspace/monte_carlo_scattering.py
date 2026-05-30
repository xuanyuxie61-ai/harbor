
import numpy as np






def sphere01_sample(n):
    if n < 1:
        raise ValueError("n must be >= 1.")
    x = np.random.randn(3, n)
    norms = np.sqrt(np.sum(x ** 2, axis=0))
    norms = np.where(norms < 1e-14, 1.0, norms)
    x = x / norms
    return x


def sphere01_monomial_integral(e):
    e = np.asarray(e, dtype=int)
    if e.shape != (3,):
        raise ValueError("e must have shape (3,).")
    if np.any(e < 0):
        raise ValueError("All exponents must be non-negative.")
    if np.all(e == 0):
        val = 2.0 * np.sqrt(np.pi ** 3) / sp_gamma(1.5)
    elif np.any(e % 2 == 1):
        val = 0.0
    else:
        val = 2.0
        for i in range(3):
            val *= sp_gamma(0.5 * (e[i] + 1))
        val /= sp_gamma(0.5 * np.sum(e + 1))
    return val






def hypercube_surface_sample(n, d):
    if n < 1 or d < 1:
        raise ValueError("n and d must be >= 1.")
    p = np.random.rand(n, d)
    i = np.random.randint(0, d, size=n)
    s = np.random.randint(0, 2, size=n)
    k = np.arange(n) + i * n
    p.flat[k] = s
    return p


def hypercube_surface_distance_stats(n, d):
    p1 = hypercube_surface_sample(n, d)
    p2 = hypercube_surface_sample(n, d)
    dists = np.linalg.norm(p1 - p2, axis=1)
    dmu = np.mean(dists)
    dvar = np.var(dists, ddof=1)
    return dmu, dvar






def hg_sample_cos_theta(n, g):
    if not (-1.0 < g < 1.0):
        raise ValueError("g must be in (-1, 1).")
    u = np.random.rand(n)
    if abs(g) < 1e-8:
        return 2.0 * u - 1.0
    g2 = g * g
    numerator = 1.0 - g2
    denom = 1.0 - g + 2.0 * g * u
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
    cos_theta = (1.0 + g2 - (numerator / denom) ** 2) / (2.0 * g)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return cos_theta






def track_photon_packet(initial_position, initial_direction, max_steps,
                        mu_s, mu_a, g, layer_z_boundaries,
                        n_medium=1.33, step_size=None):
    pos = np.asarray(initial_position, dtype=float).copy()
    direc = np.asarray(initial_direction, dtype=float)
    norm = np.linalg.norm(direc)
    if norm < 1e-14:
        direc = np.array([0.0, 0.0, 1.0])
    else:
        direc = direc / norm

    layer_boundaries = np.asarray(layer_z_boundaries, dtype=float)
    if np.isscalar(n_medium):
        n_layers = len(layer_boundaries) - 1
        n_vals = np.full(n_layers, n_medium, dtype=float)
    else:
        n_vals = np.asarray(n_medium, dtype=float)

    def get_layer_idx(z):
        for idx in range(len(layer_boundaries) - 1):
            if layer_boundaries[idx] <= z < layer_boundaries[idx + 1]:
                return idx
        return len(layer_boundaries) - 2

    def get_coeff(z, coeff):
        if callable(coeff):
            return float(coeff(z))
        return float(coeff)

    path = [pos.copy()]
    weight = 1.0
    weights = [weight]

    for _ in range(max_steps):
        layer_idx = get_layer_idx(pos[2])
        mu_s_val = get_coeff(pos[2], mu_s)
        mu_a_val = get_coeff(pos[2], mu_a)
        mu_t = mu_s_val + mu_a_val
        if mu_t <= 1e-14:
            break

        if step_size is None:
            s = -np.log(max(np.random.rand(), 1e-14)) / mu_t
        else:
            s = step_size

        new_pos = pos + s * direc


        crossed = False
        for bz in layer_boundaries:
            if (pos[2] - bz) * (new_pos[2] - bz) < 0:

                if abs(direc[2]) > 1e-14:
                    s_boundary = (bz - pos[2]) / direc[2]
                    new_pos = pos + s_boundary * direc

                    n1 = n_vals[min(layer_idx, len(n_vals) - 1)]
                    n2 = n_vals[min(layer_idx + 1, len(n_vals) - 1)]
                    if n1 != n2:

                        direc[2] = -direc[2]
                        new_pos = pos + s_boundary * direc
                    crossed = True
                    break

        pos = new_pos.copy()
        path.append(pos.copy())


        albedo = mu_s_val / mu_t if mu_t > 0 else 0.0
        if np.random.rand() > albedo:
            weight = 0.0
            weights.append(weight)
            break
        weight *= albedo
        weights.append(weight)


        g_val = get_coeff(pos[2], g)
        cos_theta = hg_sample_cos_theta(1, g_val)[0]
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi_angle = 2.0 * np.pi * np.random.rand()


        if abs(direc[2]) > 0.99999:
            ux = sin_theta * np.cos(phi_angle)
            uy = sin_theta * np.sin(phi_angle)
            uz = cos_theta * np.sign(direc[2])
        else:
            denom = np.sqrt(1.0 - direc[2] ** 2)
            ux = (sin_theta * (direc[0] * direc[2] * np.cos(phi_angle) - direc[1] * np.sin(phi_angle))
                  / denom + direc[0] * cos_theta)
            uy = (sin_theta * (direc[1] * direc[2] * np.cos(phi_angle) + direc[0] * np.sin(phi_angle))
                  / denom + direc[1] * cos_theta)
            uz = -denom * sin_theta * np.cos(phi_angle) + direc[2] * cos_theta
        direc = np.array([ux, uy, uz])
        direc = direc / np.linalg.norm(direc)

    return path, weights


def simulate_oct_signal_mc(n_photons, source_z, detector_z,
                           layer_boundaries, layer_props, max_steps=100):
    detected = 0
    depths = []
    mu_s_arr = np.array([p['mu_s'] for p in layer_props])
    mu_a_arr = np.array([p['mu_a'] for p in layer_props])
    g_arr = np.array([p['g'] for p in layer_props])
    n_arr = np.array([p['n'] for p in layer_props])

    def coeff_func(z, arr):
        for idx in range(len(layer_boundaries) - 1):
            if layer_boundaries[idx] <= z < layer_boundaries[idx + 1]:
                return arr[idx]
        return arr[-1]

    for _ in range(n_photons):
        pos0 = np.array([0.0, 0.0, source_z])
        dir0 = np.array([0.0, 0.0, 1.0])
        path, weights = track_photon_packet(
            pos0, dir0, max_steps,
            lambda z: coeff_func(z, mu_s_arr),
            lambda z: coeff_func(z, mu_a_arr),
            lambda z: coeff_func(z, g_arr),
            layer_boundaries,
            n_arr
        )
        if len(path) > 1 and weights[-1] > 0:
            last_z = path[-1][2]
            depths.append(last_z)

            if abs(last_z - detector_z) < 1.0:
                detected += weights[-1]

    signal = detected / n_photons if n_photons > 0 else 0.0
    return signal, np.array(depths)





try:
    from scipy.special import gamma as sp_gamma
except Exception:

    import math
    def sp_gamma(x):
        return math.gamma(x)
