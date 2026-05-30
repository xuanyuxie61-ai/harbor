
import numpy as np




def triangle_area(v0, v1, v2):
    v0, v1, v2 = np.array(v0), np.array(v1), np.array(v2)
    cross = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
    return 0.5 * abs(cross)


def sample_triangle_uniform(v0, v1, v2, n_samples, seed=None):
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    if seed is not None:
        np.random.seed(seed)
    v0, v1, v2 = np.array(v0), np.array(v1), np.array(v2)
    u1 = np.random.rand(n_samples)
    u2 = np.random.rand(n_samples)
    alpha = np.sqrt(u1)
    points = ((1.0 - alpha)[:, None] * v0[None, :]
              + alpha[:, None] * (1.0 - u2)[:, None] * v1[None, :]
              + alpha[:, None] * u2[:, None] * v2[None, :])
    return points


def triangle_centroid(points):
    if len(points) == 0:
        return np.array([0.0, 0.0])
    return np.mean(points, axis=0)




def cvt_triangle_uniform(triangle_vertices, n_generators, n_samples_per_iter=5000,
                          n_iterations=50, seed=None):
    if n_generators < 1:
        raise ValueError("n_generators must be >= 1.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1.")

    v0, v1, v2 = np.array(triangle_vertices[0]), np.array(triangle_vertices[1]), np.array(triangle_vertices[2])


    if seed is not None:
        np.random.seed(seed)
    generators = sample_triangle_uniform(v0, v1, v2, n_generators)

    for it in range(n_iterations):

        samples = sample_triangle_uniform(v0, v1, v2, n_samples_per_iter)


        assignments = np.zeros(n_samples_per_iter, dtype=int)
        for i, samp in enumerate(samples):
            dists = np.sum((generators - samp) ** 2, axis=1)
            assignments[i] = np.argmin(dists)


        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators, dtype=int)
        for k in range(n_generators):
            mask = assignments == k
            if np.sum(mask) > 0:
                new_generators[k] = np.mean(samples[mask], axis=0)
                counts[k] = np.sum(mask)
            else:

                new_generators[k] = sample_triangle_uniform(v0, v1, v2, 1)[0]
                counts[k] = 1


        shift = np.max(np.linalg.norm(new_generators - generators, axis=1))
        generators = new_generators
        if shift < 1e-8:
            break

    return generators


def cvt_disk_uniform(n_generators, radius=1.0, n_iterations=30, seed=None):
    if n_generators < 1:
        raise ValueError("n_generators must be >= 1.")

    n_sectors = max(6, int(np.sqrt(n_generators)))
    generators_per_sector = max(1, n_generators // n_sectors)
    all_generators = []

    angles = np.linspace(0, 2 * np.pi, n_sectors + 1)
    for s in range(n_sectors):
        theta0, theta1 = angles[s], angles[s + 1]
        v0 = np.array([0.0, 0.0])
        v1 = np.array([radius * np.cos(theta0), radius * np.sin(theta0)])
        v2 = np.array([radius * np.cos(theta1), radius * np.sin(theta1)])

        gens = cvt_triangle_uniform([v0, v1, v2], generators_per_sector,
                                     n_iterations=n_iterations, seed=seed)
        all_generators.append(gens)
        if seed is not None:
            seed += 1

    generators = np.vstack(all_generators)

    if len(generators) > n_generators:
        idx = np.random.choice(len(generators), n_generators, replace=False)
        generators = generators[idx]
    return generators




def adaptive_phase_sampling(phase, mask, n_target_points, n_iterations=20):
    if n_target_points < 1:
        raise ValueError("n_target_points must be >= 1.")

    grid_size = phase.shape[0]
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)


    dphidx = np.zeros_like(phase)
    dphidy = np.zeros_like(phase)
    dphidx[:, 1:-1] = (phase[:, 2:] - phase[:, :-2]) / (2.0 * (x[1] - x[0]))
    dphidy[1:-1, :] = (phase[2:, :] - phase[:-2, :]) / (2.0 * (y[1] - y[0]))
    grad_amp = np.sqrt(dphidx ** 2 + dphidy ** 2)
    grad_amp[~mask] = 0.0
    grad_max = np.max(grad_amp)
    if grad_max < 1e-20:
        grad_amp = np.ones_like(grad_amp) * mask


    coords = np.column_stack([X[mask].ravel(), Y[mask].ravel()])
    weights = grad_amp[mask].ravel()
    weights = weights / np.sum(weights)

    if len(coords) < n_target_points:
        n_target_points = len(coords)


    indices = np.random.choice(len(coords), size=n_target_points, p=weights)
    generators = coords[indices].copy()


    for _ in range(n_iterations):

        assignments = np.zeros(len(coords), dtype=int)
        for i, pt in enumerate(coords):
            dists = np.sum((generators - pt) ** 2, axis=1)
            assignments[i] = np.argmin(dists)


        new_gens = np.zeros_like(generators)
        for k in range(n_target_points):
            mask_k = assignments == k
            if np.sum(mask_k) > 0:
                w = weights[mask_k]
                pts = coords[mask_k]
                new_gens[k] = np.sum(pts * w[:, None], axis=0) / np.sum(w)
            else:
                new_gens[k] = coords[np.random.randint(len(coords))]

        generators = new_gens

    return generators




def log_sampling_info(filepath, generators, iteration, residual=None):
    with open(filepath, 'w') as f:
        f.write("# Adaptive Sampling Log\n")
        f.write(f"# Iteration: {iteration}\n")
        if residual is not None:
            f.write(f"# Residual: {residual:.6e}\n")
        f.write("# x y\n")
        for pt in generators:
            f.write(f"{pt[0]:.12e} {pt[1]:.12e}\n")
