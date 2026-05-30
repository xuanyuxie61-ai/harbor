
import numpy as np






_TETRA_RULES = {
    3: {
        'points': np.array([
            [0.25, 0.25, 0.25],
            [0.5, 1.0/6.0, 1.0/6.0],
            [1.0/6.0, 0.5, 1.0/6.0],
            [1.0/6.0, 1.0/6.0, 0.5],
            [1.0/6.0, 1.0/6.0, 1.0/6.0],
        ]),
        'weights': np.array([-0.8, 0.45, 0.45, 0.45, 0.45]) / 6.0,
    },
    5: {

        'points': np.array([
            [0.25, 0.25, 0.25],
            [0.091971078052723, 0.091971078052723, 0.091971078052723],
            [0.72408676584183,  0.091971078052723, 0.091971078052723],
            [0.091971078052723, 0.72408676584183,  0.091971078052723],
            [0.091971078052723, 0.091971078052723, 0.72408676584183],
            [0.31979362782963,  0.31979362782963,  0.31979362782963],
            [0.04061911651111,  0.31979362782963,  0.31979362782963],
            [0.31979362782963,  0.04061911651111,  0.31979362782963],
            [0.31979362782963,  0.31979362782963,  0.04061911651111],
            [0.44364916731037,  0.44364916731037,  0.05635083268963],
            [0.44364916731037,  0.05635083268963,  0.44364916731037],
            [0.05635083268963,  0.44364916731037,  0.44364916731037],
            [0.44364916731037,  0.05635083268963,  0.05635083268963],
            [0.05635083268963,  0.44364916731037,  0.05635083268963],
            [0.05635083268963,  0.05635083268963,  0.44364916731037],
        ]),
        'weights': np.array([
            -0.013155555555556,
             0.007622222222222,
             0.007622222222222,
             0.007622222222222,
             0.007622222222222,
             0.024888888888889,
             0.024888888888889,
             0.024888888888889,
             0.024888888888889,
             0.009851111111111,
             0.009851111111111,
             0.009851111111111,
             0.009851111111111,
             0.009851111111111,
             0.009851111111111,
        ]) / 6.0,
    }
}


def reference_to_physical_t4(ref_points, tetra):



    N = ref_points.shape[0]
    phys = np.zeros((N, 3))
    for i in range(N):
        x, y, z = ref_points[i]
        w = 1.0 - x - y - z
        phys[i] = x * tetra[0] + y * tetra[1] + z * tetra[2] + w * tetra[3]
    return phys


def tetrahedron_volume(tetra):
    M = np.ones((4, 4))
    M[:, :3] = tetra
    vol = abs(np.linalg.det(M)) / 6.0
    return vol


def integrate_over_tetrahedron(f, tetra, degree=5):
    if degree not in _TETRA_RULES:
        raise ValueError(f"Unsupported degree {degree}. Choose 3 or 5.")
    rule = _TETRA_RULES[degree]
    pts = rule['points']
    wts = rule['weights']
    phys_pts = reference_to_physical_t4(pts, tetra)
    vol = tetrahedron_volume(tetra)
    total = 0.0 + 0.0j
    for p, w in zip(phys_pts, wts):
        total += w * f(p[0], p[1], p[2])
    return total * vol * 6.0


def partition_bz_into_tetrahedra(n_k=4):
    edges = np.linspace(-np.pi, np.pi, n_k + 1)
    tetra_list = []
    for ix in range(n_k):
        for iy in range(n_k):
            for iz in range(n_k):
                x0, x1 = edges[ix], edges[ix + 1]
                y0, y1 = edges[iy], edges[iy + 1]
                z0, z1 = edges[iz], edges[iz + 1]

                v000 = np.array([x0, y0, z0])
                v100 = np.array([x1, y0, z0])
                v010 = np.array([x0, y1, z0])
                v110 = np.array([x1, y1, z0])
                v001 = np.array([x0, y0, z1])
                v101 = np.array([x1, y0, z1])
                v011 = np.array([x0, y1, z1])
                v111 = np.array([x1, y1, z1])

                tetra_list.append(np.array([v000, v100, v010, v001]))
                tetra_list.append(np.array([v111, v011, v101, v110]))
                tetra_list.append(np.array([v100, v010, v110, v001]))
                tetra_list.append(np.array([v100, v110, v101, v001]))
                tetra_list.append(np.array([v010, v110, v011, v001]))
                tetra_list.append(np.array([v110, v101, v011, v001]))
    return tetra_list


def integrate_bz_3d(f, n_k=4, degree=5):
    tetras = partition_bz_into_tetrahedra(n_k)
    total = 0.0 + 0.0j
    for tetra in tetras:
        total += integrate_over_tetrahedron(f, tetra, degree=degree)
    return total


def bz_average_energy(H_func, n_k=4, degree=5):
    def f(kx, ky, kz):
        H = H_func(kx, ky, kz)
        E = np.linalg.eigvals(H)
        return E[np.argmin(E.real)]

    V_BZ = (2.0 * np.pi) ** 3
    integral = integrate_bz_3d(f, n_k=n_k, degree=degree)
    return integral / V_BZ
