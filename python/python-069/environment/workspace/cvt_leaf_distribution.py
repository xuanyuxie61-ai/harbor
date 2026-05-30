import numpy as np
from scipy.spatial import Delaunay, cKDTree


def cvt_3d_lloyd(n_generators, it_num, s_num, rho_func, domain=((-1, 1), (-1, 1), (-1, 1))):

    g = np.zeros((n_generators, 3), dtype=float)
    for dim in range(3):
        g[:, dim] = np.random.uniform(domain[dim][0], domain[dim][1], n_generators)


    lin = [np.linspace(domain[d][0] + 1e-6, domain[d][1] - 1e-6, s_num) for d in range(3)]
    sx, sy, sz = np.meshgrid(lin[0], lin[1], lin[2], indexing='ij')
    s_points = np.column_stack((sx.ravel(), sy.ravel(), sz.ravel()))


    rho_vals = rho_func(s_points[:, 0], s_points[:, 1], s_points[:, 2])
    rho_vals = np.clip(rho_vals, 1e-6, 1e6)

    energy_history = []
    motion_history = []

    for it in range(it_num):

        tree = cKDTree(g)
        _, nearest = tree.query(s_points)


        m = np.zeros(n_generators, dtype=float)
        cx = np.zeros(n_generators, dtype=float)
        cy = np.zeros(n_generators, dtype=float)
        cz = np.zeros(n_generators, dtype=float)

        np.add.at(m, nearest, rho_vals)
        np.add.at(cx, nearest, rho_vals * s_points[:, 0])
        np.add.at(cy, nearest, rho_vals * s_points[:, 1])
        np.add.at(cz, nearest, rho_vals * s_points[:, 2])


        m_safe = np.maximum(m, 1e-14)
        g_new = np.column_stack((cx / m_safe, cy / m_safe, cz / m_safe))


        dist2 = np.sum((s_points - g[nearest, :]) ** 2, axis=1)
        energy = np.sum(rho_vals * dist2) / s_num
        energy_history.append(energy)


        motion = np.mean(np.sum((g_new - g) ** 2, axis=1))
        motion_history.append(motion)

        g = g_new

    return g, energy_history, motion_history


def canopy_cvt_optimization(canopy_height, crown_radius, n_clusters=50,
                            it_num=20, s_num=20, lai_max=4.5):
    def rho_func(x, y, z):

        z = np.asarray(z)
        r = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2)
        decay_r = np.maximum(0.0, 1.0 - (r / crown_radius) ** 2)
        decay_z = np.clip(z / canopy_height, 0.0, 1.0)
        return decay_r * decay_z + 0.1

    domain = ((-crown_radius, crown_radius),
              (-crown_radius, crown_radius),
              (0.0, canopy_height))

    g, e_hist, m_hist = cvt_3d_lloyd(n_clusters, it_num, s_num, rho_func, domain)
    return g, e_hist, m_hist
