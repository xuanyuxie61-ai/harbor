
import numpy as np
from math import sin, cos, sqrt, pi


def deformed_nuclear_surface_grid(beta2, gamma, R0, n_theta, n_phi):
    theta = np.linspace(0, pi, n_theta)
    phi = np.linspace(0, 2 * pi, n_phi)
    dtheta = pi / (n_theta - 1) if n_theta > 1 else pi
    dphi = 2 * pi / (n_phi - 1) if n_phi > 1 else 2 * pi

    grid = []
    areas = []

    for i in range(n_theta):
        for j in range(n_phi):
            th = theta[i]
            ph = phi[j]

            Y20 = sqrt(5.0 / (16.0 * pi)) * (3.0 * cos(th) ** 2 - 1.0)

            Y22_real = sqrt(15.0 / (32.0 * pi)) * sin(th) ** 2 * cos(2.0 * ph)

            R = R0 * (1.0 + beta2 * (cos(gamma) * Y20 + sin(gamma) * Y22_real))
            x = R * sin(th) * cos(ph)
            y = R * sin(th) * sin(ph)
            z = R * cos(th)
            grid.append([x, y, z])


            dA = max(R, 0.0) ** 2 * sin(th) * dtheta * dphi
            areas.append(dA)

    return np.array(grid), np.array(areas)


def annular_shell_grid(n_r, n_theta, r_inner, r_outer, center=(0.0, 0.0)):
    if r_inner < 0 or r_outer <= r_inner:
        raise ValueError("必须满足 0 ≤ r_inner < r_outer")

    dr = (r_outer - r_inner) / n_r
    dtheta = 2 * pi / n_theta
    points = []
    weights = []

    for i in range(n_r):
        r = r_inner + (i + 0.5) * dr
        for j in range(n_theta):
            theta = (j + 0.5) * dtheta
            x = center[0] + r * cos(theta)
            y = center[1] + r * sin(theta)
            points.append([x, y])
            weights.append(r * dr * dtheta)

    return np.array(points), np.array(weights)


def cvt_3d_sample(n_generators, n_iterations, n_samples, density_fn=None,
                  bounds=((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)), seed=42):
    rng = np.random.default_rng(seed)


    generators = rng.random((n_generators, 3))
    for d in range(3):
        lo, hi = bounds[d]
        generators[:, d] = lo + generators[:, d] * (hi - lo)

    if density_fn is None:
        def density_fn(x, y, z):
            return 1.0

    energy_history = []

    for it in range(n_iterations):

        samples = rng.random((n_samples, 3))
        for d in range(3):
            lo, hi = bounds[d]
            samples[:, d] = lo + samples[:, d] * (hi - lo)


        rho = np.array([density_fn(s[0], s[1], s[2]) for s in samples])


        nearest = np.zeros(n_samples, dtype=int)
        for k in range(n_samples):
            dists = np.sum((generators - samples[k]) ** 2, axis=1)
            nearest[k] = np.argmin(dists)


        energy = 0.0
        for k in range(n_samples):
            energy += rho[k] * np.sum((samples[k] - generators[nearest[k]]) ** 2)
        energy /= n_samples
        energy_history.append(energy)


        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators)
        for k in range(n_samples):
            i = nearest[k]
            new_generators[i] += rho[k] * samples[k]
            counts[i] += rho[k]


        for i in range(n_generators):
            if counts[i] > 0:
                new_generators[i] /= counts[i]
            else:
                new_generators[i] = generators[i]

        generators = new_generators

    return generators, energy_history


def nuclear_volume_cvt_quadrature(A, beta2=0.0, gamma=0.0, n_points=200,
                                   n_iter=30, R0=1.2):
    R = R0 * (A ** (1.0 / 3.0))
    a_diff = 0.52

    def fermi_density(x, y, z):

        r = sqrt(x * x + y * y + z * z)
        return 1.0 / (1.0 + np.exp((r - R) / a_diff))

    bounds = ((-1.5 * R, 1.5 * R),) * 3
    generators, _ = cvt_3d_sample(
        n_generators=n_points,
        n_iterations=n_iter,
        n_samples=50000,
        density_fn=fermi_density,
        bounds=bounds,
        seed=42
    )



    rng = np.random.default_rng(123)
    n_mc = 200000
    mc_samples = rng.random((n_mc, 3))
    for d in range(3):
        lo, hi = bounds[d]
        mc_samples[:, d] = lo + mc_samples[:, d] * (hi - lo)

    weights = np.zeros(n_points)
    total_vol = 0.0
    for k in range(n_mc):
        x, y, z = mc_samples[k]
        rho = fermi_density(x, y, z)
        if rho < 1e-6:
            continue
        dists = np.sum((generators - mc_samples[k]) ** 2, axis=1)
        i = np.argmin(dists)
        weights[i] += rho
        total_vol += rho

    if total_vol > 0:
        weights *= ((3.0 * R) ** 3) / n_mc

    return generators, weights
