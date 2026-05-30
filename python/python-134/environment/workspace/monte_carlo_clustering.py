#!/usr/bin/env python3

import numpy as np






def tetrahedron01_sample(n, seed=42):
    rng = np.random.default_rng(seed)

    e = rng.exponential(scale=1.0, size=(n, 4))
    s = np.sum(e, axis=1, keepdims=True)
    p = e / s


    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    v3 = np.array([0.0, 0.0, 1.0])

    xyz = (p[:, 0:1] * v0 + p[:, 1:2] * v1 +
           p[:, 2:3] * v2 + p[:, 3:4] * v3)
    return xyz


def reference_to_physical_tet4(nodes_ref, tet_nodes, n_samples, seed=42):
    p0 = tet_nodes[0]
    p1 = tet_nodes[1]
    p2 = tet_nodes[2]
    p3 = tet_nodes[3]

    J = np.array([
        [p1[0] - p0[0], p2[0] - p0[0], p3[0] - p0[0]],
        [p1[1] - p0[1], p2[1] - p0[1], p3[1] - p0[1]],
        [p1[2] - p0[2], p2[2] - p0[2], p3[2] - p0[2]],
    ], dtype=float)

    xyz_ref = tetrahedron01_sample(n_samples, seed)
    xyz_phys = p0 + xyz_ref @ J.T
    return xyz_phys


def estimate_effective_diffusivity_monte_carlo(nodes, elements, params, n_samples_per_tet=50):
    n_tets = elements.shape[0]
    rng = np.random.default_rng(42)

    D_0 = params.get('D_gdl_ref', 1.0e-6)
    epsilon = params.get('epsilon_gdl', 0.4)
    tau = 1.5

    total_volume = 0.0
    weighted_D = 0.0

    for e in range(min(n_tets, 200)):
        tet = elements[e]
        tet_nodes = nodes[tet]


        M = np.array([
            [tet_nodes[1, 0] - tet_nodes[0, 0],
             tet_nodes[1, 1] - tet_nodes[0, 1],
             tet_nodes[1, 2] - tet_nodes[0, 2]],
            [tet_nodes[2, 0] - tet_nodes[0, 0],
             tet_nodes[2, 1] - tet_nodes[0, 1],
             tet_nodes[2, 2] - tet_nodes[0, 2]],
            [tet_nodes[3, 0] - tet_nodes[0, 0],
             tet_nodes[3, 1] - tet_nodes[0, 1],
             tet_nodes[3, 2] - tet_nodes[0, 2]],
        ], dtype=float)
        vol = abs(np.linalg.det(M)) / 6.0
        if vol < 1e-15:
            continue


        samples = reference_to_physical_tet4(nodes, tet_nodes, n_samples_per_tet, seed=rng.integers(0, 1e9))



        z_centers = samples[:, 2]
        z_mid = 0.15
        sigma_z = 0.05
        I_conn = np.exp(-((z_centers - z_mid) ** 2) / (2.0 * sigma_z ** 2))
        I_conn = np.clip(I_conn, 0.2, 1.0)


        D_local = D_0 * (epsilon ** tau) * np.mean(I_conn)

        weighted_D += D_local * vol
        total_volume += vol

    if total_volume < 1e-15:
        return D_0 * (epsilon ** tau)

    D_eff = weighted_D / total_volume
    return D_eff


def estimate_water_cluster_distribution(nodes, elements, params, n_samples=5000):
    rng = np.random.default_rng(123)
    n_tets = min(elements.shape[0], 100)

    samples_all = []
    for e in range(n_tets):
        tet = elements[e]
        tet_nodes = nodes[tet]
        n_samp = max(5, n_samples // n_tets)
        samples = reference_to_physical_tet4(nodes, tet_nodes, n_samp, seed=rng.integers(0, 1e9))
        samples_all.append(samples)

    if not samples_all:
        return np.array([0.0]), np.array([1.0])

    samples_all = np.vstack(samples_all)


    lambda_local = 10.0 + 5.0 * np.sin(2.0 * np.pi * samples_all[:, 0])
    lambda_local += rng.normal(0.0, 1.0, size=lambda_local.shape)
    lambda_local = np.clip(lambda_local, 0.0, 22.0)


    thresholds = np.linspace(8.0, 18.0, 20)
    cluster_fractions = []
    for th in thresholds:
        cluster_fractions.append(np.mean(lambda_local > th))

    return thresholds, np.array(cluster_fractions)


if __name__ == '__main__':
    from mesh_generator import generate_pemfc_mesh, refine_mesh
    nodes, elements = generate_pemfc_mesh()
    nodes_r, elements_r = refine_mesh(nodes, elements)
    p = {'D_gdl_ref': 1e-6, 'epsilon_gdl': 0.4}
    D_eff = estimate_effective_diffusivity_monte_carlo(nodes_r, elements_r, p)
    print("D_eff =", D_eff)
