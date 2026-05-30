
import numpy as np
from math import sin, cos, sqrt, pi, exp


def uniform_sphere_sample(n_points, R_max, center=(0.0, 0.0, 0.0), seed=42):
    rng = np.random.default_rng(seed)
    u = rng.random(n_points)
    v = rng.random(n_points)
    w = rng.random(n_points)

    r = R_max * u ** (1.0 / 3.0)
    theta = np.arccos(2.0 * v - 1.0)
    phi = 2.0 * pi * w

    x = center[0] + r * np.sin(theta) * np.cos(phi)
    y = center[1] + r * np.sin(theta) * np.sin(phi)
    z = center[2] + r * np.cos(theta)

    return np.column_stack([x, y, z])


def deformed_fermi_sample(n_points, A, beta2=0.0, gamma=0.0,
                          R0=1.2, a=0.52, seed=42):
    rng = np.random.default_rng(seed)
    R = R0 * (A ** (1.0 / 3.0))
    box_size = 1.5 * R * (1.0 + abs(beta2) + 0.1)

    points = []
    n_trial = 0
    max_trials = n_points * 50

    while len(points) < n_points and n_trial < max_trials:

        xyz = rng.uniform(-box_size, box_size, size=3)
        x, y, z = xyz
        r = sqrt(x * x + y * y + z * z)
        if r < 1e-10:
            n_trial += 1
            continue

        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)


        Y20 = sqrt(5.0 / (16.0 * pi)) * (3.0 * cos(theta) ** 2 - 1.0)
        Y22_real = sqrt(15.0 / (32.0 * pi)) * sin(theta) ** 2 * cos(2.0 * phi)
        R_def = R * (1.0 + beta2 * (cos(gamma) * Y20 + sin(gamma) * Y22_real))


        prob = 1.0 / (1.0 + exp((r - R_def) / a))

        if rng.random() < prob:
            points.append([x, y, z])

        n_trial += 1

    if len(points) < n_points:

        extra = uniform_sphere_sample(n_points - len(points), box_size, seed=seed + 1)
        points.extend(extra.tolist())

    return np.array(points[:n_points])


def pairwise_distance_statistics(points):
    N = len(points)
    if N < 2:
        return {'mean': 0.0, 'variance': 0.0, 'min': 0.0, 'max': 0.0, 'rms': 0.0}


    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    triu_indices = np.triu_indices(N, k=1)
    d = dists[triu_indices]

    return {
        'mean': float(np.mean(d)),
        'variance': float(np.var(d)),
        'min': float(np.min(d)),
        'max': float(np.max(d)),
        'rms': float(np.sqrt(np.mean(d ** 2)))
    }


def pair_correlation_function(points, dr, r_max):
    N = len(points)
    if N < 2:
        return np.array([0.0]), np.array([0.0])

    n_bins = int(r_max / dr)
    r_bins = np.linspace(dr / 2.0, r_max - dr / 2.0, n_bins)
    g_r = np.zeros(n_bins)


    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    triu_indices = np.triu_indices(N, k=1)
    all_dists = dists[triu_indices]


    counts, _ = np.histogram(all_dists, bins=np.linspace(0, r_max, n_bins + 1))


    V = (4.0 / 3.0) * pi * r_max ** 3
    rho0 = N / V

    for i in range(n_bins):
        r = r_bins[i]
        shell_volume = 4.0 * pi * r * r * dr
        ideal_count = 0.5 * N * (N - 1) * shell_volume / V
        if ideal_count > 0:
            g_r[i] = counts[i] / ideal_count

    return r_bins, g_r


def monte_carlo_nuclear_radius(A, n_samples=100000, beta2=0.0, gamma=0.0,
                                R0=1.2, seed=123):
    points = deformed_fermi_sample(n_samples, A, beta2, gamma, R0, seed=seed)
    radii = np.sqrt(np.sum(points ** 2, axis=1))

    R_rms = float(np.sqrt(np.mean(radii ** 2)))
    R_eff = float(np.percentile(radii, 90))
    r10 = float(np.percentile(radii, 10))
    t_surface = R_eff - r10

    return R_eff, t_surface, R_rms


def triangular_deformation_analysis(n_theta, n_phi, beta2, gamma, R0):
    from nuclear_grid import deformed_nuclear_surface_grid
    grid, _ = deformed_nuclear_surface_grid(beta2, gamma, R0, n_theta, n_phi)

    areas = []
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):

            idx00 = i * n_phi + j
            idx01 = i * n_phi + (j + 1)
            idx10 = (i + 1) * n_phi + j
            idx11 = (i + 1) * n_phi + (j + 1)


            tri1 = [grid[idx00], grid[idx10], grid[idx01]]
            tri2 = [grid[idx01], grid[idx10], grid[idx11]]

            for tri in [tri1, tri2]:
                a_vec = tri[1] - tri[0]
                b_vec = tri[2] - tri[0]
                cross = np.cross(a_vec, b_vec)
                area = 0.5 * np.linalg.norm(cross)
                areas.append(area)

    total_area = float(np.sum(areas))
    area_variance = float(np.var(areas))
    return total_area, area_variance, areas
