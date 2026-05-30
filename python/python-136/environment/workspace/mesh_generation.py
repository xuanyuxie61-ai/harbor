
import numpy as np


class MeshGenerationError(Exception):
    pass


def cvt_1d_lloyd(n_generators, n_iterations, n_samples,
                 density_func=None, domain=(0.0, 1.0)):
    a, b = domain
    if a >= b:
        raise MeshGenerationError("domain 必须满足 a < b")
    if n_generators < 2:
        raise MeshGenerationError("生成器数量至少为 2")

    if density_func is None:
        density_func = lambda x: np.ones_like(x)


    generators = np.linspace(a + 0.01 * (b - a), b - 0.01 * (b - a),
                             n_generators)
    generators = np.sort(generators)
    energy_history = []

    for _ in range(n_iterations):

        samples = np.empty(0)
        batch = min(n_samples * 5, 1000000)
        while samples.size < n_samples:
            cand = np.random.uniform(a, b, size=batch)
            rho_cand = density_func(cand)
            rho_max = np.max(rho_cand)
            if rho_max <= 0:
                raise MeshGenerationError("密度函数在非零测集上为零")
            accept = np.random.uniform(0, rho_max, size=batch) < rho_cand
            accepted = cand[accept]
            samples = np.concatenate([samples, accepted])
        samples = samples[:n_samples]
        samples = np.sort(samples)


        boundaries = np.empty(n_generators + 1)
        boundaries[0] = a
        boundaries[-1] = b
        if n_generators > 1:
            boundaries[1:-1] = 0.5 * (generators[:-1] + generators[1:])


        new_generators = np.zeros(n_generators)
        counts = np.zeros(n_generators)
        energy = 0.0


        idx = np.searchsorted(samples, boundaries)
        for i in range(n_generators):
            lo = idx[i]
            hi = idx[i + 1]
            cell_samples = samples[lo:hi]
            counts[i] = cell_samples.size
            if counts[i] > 0:
                new_generators[i] = np.mean(cell_samples)
                energy += np.sum((cell_samples - generators[i]) ** 2)
            else:

                new_generators[i] = generators[i]

        energy = energy / n_samples
        energy_history.append(energy)
        generators = np.sort(new_generators)

    return generators, energy_history


def cvt_square_uniform_2d(n_generators, n_iterations, n_samples,
                          domain=(0.0, 1.0, 0.0, 1.0)):
    xmin, xmax, ymin, ymax = domain
    if n_generators < 2:
        raise MeshGenerationError("生成器数量至少为 2")


    generators = np.random.rand(n_generators, 2)
    generators[:, 0] = xmin + generators[:, 0] * (xmax - xmin)
    generators[:, 1] = ymin + generators[:, 1] * (ymax - ymin)

    energy_history = []

    for _ in range(n_iterations):

        samples = np.random.rand(n_samples, 2)
        samples[:, 0] = xmin + samples[:, 0] * (xmax - xmin)
        samples[:, 1] = ymin + samples[:, 1] * (ymax - ymin)



        dx = samples[:, 0:1] - generators[:, 0].reshape(1, -1)
        dy = samples[:, 1:2] - generators[:, 1].reshape(1, -1)
        dists = dx ** 2 + dy ** 2
        nearest = np.argmin(dists, axis=1)


        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators)
        energy = 0.0

        for i in range(n_generators):
            mask = nearest == i
            cell_samples = samples[mask]
            counts[i] = cell_samples.shape[0]
            if counts[i] > 0:
                new_generators[i] = np.mean(cell_samples, axis=0)
                energy += np.sum(
                    np.sum((cell_samples - generators[i]) ** 2, axis=1)
                )
            else:
                new_generators[i] = generators[i]

        energy = energy / n_samples
        energy_history.append(energy)
        generators = new_generators.copy()

    return generators, energy_history


def adaptive_radial_mesh(R, n_nodes, reaction_steepness=5.0):
    if R <= 0:
        raise MeshGenerationError("R 必须为正")

    def density(r):
        return 1.0 + reaction_steepness * (r / R) ** 2

    generators, _ = cvt_1d_lloyd(
        n_generators=n_nodes - 2,
        n_iterations=30,
        n_samples=50000,
        density_func=density,
        domain=(0.0, R)
    )
    nodes = np.concatenate([[0.0], generators, [R]])
    nodes = np.sort(nodes)

    nodes = np.unique(nodes)

    if nodes.size < n_nodes:
        nodes = np.linspace(0.0, R, n_nodes)
    return nodes
