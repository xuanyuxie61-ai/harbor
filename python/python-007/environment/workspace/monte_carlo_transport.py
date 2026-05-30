import numpy as np






def wedge01_volume():
    return 1.0


def wedge01_sample(n_samples, seed=None):
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if seed is not None:
        np.random.seed(seed)

    samples = np.zeros((n_samples, 3), dtype=np.float64)

    for i in range(n_samples):
        e = -np.log(np.random.rand(3) + 1e-15)
        s = np.sum(e)
        if s < 1e-15:
            s = 1e-15
        samples[i, 0] = e[0] / s
        samples[i, 1] = e[1] / s
        samples[i, 2] = 2.0 * np.random.rand() - 1.0

    return samples


def wedge01_monomial_integral(exponents):
    a, b, c = int(exponents[0]), int(exponents[1]), int(exponents[2])

    if a < 0 or b < 0 or c < 0:
        raise ValueError("Exponents must be non-negative")


    from math import factorial
    xy_val = factorial(a) * factorial(b) / factorial(a + b + 2)


    if c % 2 == 1:
        z_val = 0.0
    else:
        z_val = 2.0 / (c + 1)

    return xy_val * z_val


def wedge_monte_carlo_integral(n_samples, integrand_func, seed=None):
    samples = wedge01_sample(n_samples, seed)
    vals = np.array([integrand_func(s) for s in samples], dtype=np.float64)

    V = wedge01_volume()
    estimate = V * np.mean(vals)
    std_error = V * np.std(vals, ddof=1) / np.sqrt(n_samples) if n_samples > 1 else 0.0

    return estimate, std_error






def ball_unit_sample(n_samples, dim=3, seed=None):
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if seed is not None:
        np.random.seed(seed)

    dirs = np.random.randn(n_samples, dim)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    dirs = dirs / norms

    u = np.random.rand(n_samples)
    r = u ** (1.0 / dim)

    return dirs * r.reshape(-1, 1)


def ball_distance_pdf(d):
    d = np.asarray(d, dtype=np.float64)
    result = np.zeros_like(d)
    mask = (d >= 0) & (d <= 2)
    result[mask] = (3.0 / 16.0) * (d[mask] - 2) ** 2 * d[mask] ** 2 * (d[mask] + 4)
    return result


def ball_distance_stats(n_samples, seed=None):
    p1 = ball_unit_sample(n_samples, dim=3, seed=seed)
    p2 = ball_unit_sample(n_samples, dim=3, seed=(seed + 1 if seed is not None else None))

    dists = np.linalg.norm(p1 - p2, axis=1)

    mean = np.mean(dists)
    variance = np.var(dists, ddof=1) if n_samples > 1 else 0.0

    return {
        'mean': float(mean),
        'variance': float(variance),
        'distances': dists
    }






def sample_jet_particles(n_particles, r_launch, theta_opening, v_jet, seed=None):
    if n_particles <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    if seed is not None:
        np.random.seed(seed)




    cos_theta_max = np.cos(theta_opening)
    cos_theta = np.random.uniform(cos_theta_max, 1.0, n_particles)
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, n_particles)


    x = r_launch * np.sin(theta) * np.cos(phi)
    y = r_launch * np.sin(theta) * np.sin(phi)
    z = r_launch * np.cos(theta)

    positions = np.column_stack([x, y, z])


    vx = v_jet * np.sin(theta) * np.cos(phi)
    vy = v_jet * np.sin(theta) * np.sin(phi)
    vz = v_jet * np.cos(theta)

    velocities = np.column_stack([vx, vy, vz])

    return positions, velocities


def mc_jet_energy_transport(n_photons, r_disk, T_disk, seed=None):
    if seed is not None:
        np.random.seed(seed)



    energies = np.random.exponential(scale=1.0, size=n_photons)


    escaped = np.random.rand(n_photons) > 0.3

    return energies, escaped


def compute_correlation_function(points, r_bins):
    points = np.asarray(points, dtype=np.float64)
    n = len(points)

    if n < 2:
        return np.zeros(len(r_bins) - 1)


    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)


    dists = dists[np.triu_indices(n, k=1)]


    counts, _ = np.histogram(dists, bins=r_bins)


    volumes = (4.0 / 3.0) * np.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    volumes = np.where(volumes < 1e-15, 1e-15, volumes)


    box_size = np.max(points) - np.min(points)
    if box_size < 1e-15:
        box_size = 1.0
    total_volume = box_size ** 3
    mean_density = n * (n - 1) / 2.0 / total_volume

    xi = counts / volumes / mean_density - 1.0

    return xi
