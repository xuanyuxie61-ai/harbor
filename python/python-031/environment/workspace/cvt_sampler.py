# -*- coding: utf-8 -*-

import numpy as np


def cvt_2d_lumping(n_generators, it_num, s_num, density_func):
    if n_generators < 3:
        raise ValueError("生成点数量至少为3")


    g = 2.0 * np.random.rand(n_generators, 2) - 1.0

    x_min = -1.0 + 1e-10
    x_max = 1.0 - 1e-10
    s_1d = np.linspace(x_min, x_max, s_num)
    sx, sy = np.meshgrid(s_1d, s_1d)
    sx_flat = sx.flatten()
    sy_flat = sy.flatten()


    mu_mat = np.zeros_like(sx)
    for i in range(s_num):
        for j in range(s_num):
            mu_mat[i, j] = density_func(sx[i, j], sy[i, j])


    mu_mat = np.clip(mu_mat, 0.0, 10.0)
    r_mat = mu_mat**4
    r_flat = r_mat.flatten()

    energy_history = []
    motion_history = []

    for _ in range(it_num):

        s_points = np.column_stack((sx_flat, sy_flat))

        dists = np.sum((s_points[:, None, :] - g[None, :, :])**2, axis=2)
        nearest = np.argmin(dists, axis=1)


        g_new = np.zeros_like(g)
        mass = np.zeros(n_generators)
        for k in range(n_generators):
            mask = nearest == k
            if np.any(mask):
                mass[k] = np.sum(r_flat[mask])
                if mass[k] > 0:
                    g_new[k, 0] = np.sum(r_flat[mask] * sx_flat[mask]) / mass[k]
                    g_new[k, 1] = np.sum(r_flat[mask] * sy_flat[mask]) / mass[k]
                else:
                    g_new[k] = g[k]
            else:
                g_new[k] = g[k]


        e = np.sum(r_flat * np.min(dists, axis=1)) / s_num
        energy_history.append(e)


        motion = np.mean(np.sum((g_new - g)**2, axis=1))
        motion_history.append(motion)

        g = g_new

    return g, np.array(energy_history), np.array(motion_history)


def sphere_delaunay(n, xyz):

    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(xyz)
        face = hull.simplices
        face_num = len(face)
    except Exception:

        face = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
        face_num = 4
    return face_num, face


def uniform_on_sphere01_map(n):
    xyz = np.zeros((n, 3))
    for i in range(n):
        while True:
            x1 = 2.0 * np.random.rand() - 1.0
            x2 = 2.0 * np.random.rand() - 1.0
            r2 = x1**2 + x2**2
            if r2 < 1.0:
                break
        xyz[i, 0] = 2.0 * x1 * np.sqrt(1.0 - r2)
        xyz[i, 1] = 2.0 * x2 * np.sqrt(1.0 - r2)
        xyz[i, 2] = 1.0 - 2.0 * r2
    return xyz


def sphere_cvt_step(n, xyz):
    face_num, face = sphere_delaunay(n, xyz)


    centroid = np.zeros((n, 3))
    counts = np.zeros(n)


    n_samples = min(10000, n * 500)
    samples = uniform_on_sphere01_map(n_samples)


    dots = samples @ xyz.T
    nearest = np.argmax(dots, axis=1)

    for k in range(n):
        mask = nearest == k
        if np.any(mask):
            c = np.mean(samples[mask], axis=0)
            norm = np.linalg.norm(c)
            if norm > 1e-15:
                centroid[k] = c / norm
            else:
                centroid[k] = xyz[k]
        else:
            centroid[k] = xyz[k]

    return centroid


def voronoi_areas_direct(n, xyz, centroid):

    n_samples = min(20000, n * 1000)
    samples = uniform_on_sphere01_map(n_samples)
    dots = samples @ xyz.T
    nearest = np.argmax(dots, axis=1)

    area = np.zeros(n)
    for k in range(n):
        count = np.sum(nearest == k)
        area[k] = count / n_samples * 4.0 * np.pi

    return area


def nd_integrand_gaussian(dim_num, point_num, x):
    value = np.exp(-np.sum(x**2, axis=0))
    return value


def nd_integrand_coulomb(dim_num, point_num, x, charge_center=None):
    if charge_center is None:
        charge_center = np.zeros(dim_num)
    charge_center = np.array(charge_center).reshape(-1, 1)
    r = np.sqrt(np.sum((x - charge_center)**2, axis=0))
    r = np.maximum(r, 1e-15)
    value = 1.0 / r
    return value


def monte_carlo_nd_integral(integrand, dim_num, a, b, n_samples=100000):
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    if len(a) == 1:
        a = np.full(dim_num, a[0])
    if len(b) == 1:
        b = np.full(dim_num, b[0])

    volume = np.prod(b - a)
    x = np.random.rand(dim_num, n_samples)
    for d in range(dim_num):
        x[d, :] = a[d] + x[d, :] * (b[d] - a[d])

    values = integrand(dim_num, n_samples, x)
    integral = volume * np.mean(values)
    error = volume * np.std(values) / np.sqrt(n_samples)

    return integral, error


def pasta_density_profile(x, y, phase_centers, phase_radii, rho_bulk, rho_gas):
    rho = rho_gas
    for center, radius in zip(phase_centers, phase_radii):
        r2 = (x - center[0])**2 + (y - center[1])**2
        if r2 <= radius**2:
            rho = rho_bulk
            break
    return rho


def optimize_pasta_cvt(density, proton_fraction, phase_id, n_generators=20,
                       it_num=50):
    from geometry_pasta import create_pasta_phase

    phase = create_pasta_phase(phase_id, density, proton_fraction)
    a_ws = phase.a_WS


    def density_func(x, y):
        r = np.sqrt(x**2 + y**2)
        return np.exp(-r**2 / (2 * (0.5 * a_ws)**2)) + 0.1

    generators, energy, motion = cvt_2d_lumping(
        n_generators, it_num, 50, density_func
    )


    generators = generators * a_ws


    n_samples = min(20000, n_generators * 1000)
    sx = np.random.rand(n_samples) * 2 * a_ws - a_ws
    sy = np.random.rand(n_samples) * 2 * a_ws - a_ws
    s_points = np.column_stack((sx, sy))
    dists = np.sum((s_points[:, None, :] - generators[None, :, :])**2, axis=2)
    nearest = np.argmin(dists, axis=1)
    areas = np.zeros(n_generators)
    for k in range(n_generators):
        count = np.sum(nearest == k)
        areas[k] = count / n_samples * (2 * a_ws)**2

    return generators, areas, energy, motion


if __name__ == '__main__':

    integral, error = monte_carlo_nd_integral(
        nd_integrand_gaussian, 2, -3, 3, n_samples=50000
    )
    exact = np.pi
    print(f"2D Gaussian MC: {integral:.6f} +/- {error:.6f}, exact={exact:.6f}")

    g, e, m = cvt_2d_lumping(10, 20, 30, lambda x, y: 1.0)
    print(f"CVT test: energy_final={e[-1]:.4f}, motion_final={m[-1]:.6f}")
