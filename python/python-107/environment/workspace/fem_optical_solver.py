
import numpy as np






def triangle_xy_to_barycentric(xy):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (m, 2).")
    m = xy.shape[0]
    xyz = np.zeros((m, 3), dtype=float)
    xyz[:, 0:2] = xy
    xyz[:, 2] = 1.0 - xy[:, 0] - xy[:, 1]

    xyz = np.clip(xyz, 0.0, 1.0)
    xyz = xyz / np.sum(xyz, axis=1, keepdims=True)
    return xyz


def barycentric_symmetry(xyz, tol=1e-10):
    xyz = np.asarray(xyz, dtype=float)
    m = xyz.shape[0]
    symmetry = np.zeros(m, dtype=int)
    for i in range(m):
        vals = xyz[i, :]
        unique_num = len(np.unique(np.round(vals / tol).astype(int)))
        if unique_num == 1:
            if np.allclose(vals, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=tol):
                symmetry[i] = 1
            else:
                symmetry[i] = 0
        elif unique_num == 2:
            symmetry[i] = 3
        else:
            symmetry[i] = 6
    return symmetry






def triangle_gauss_rule(order=7):
    if order == 3:

        bary = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        weights = np.array([0.5])
    elif order == 7:

        a = 1.0 / 3.0
        b = (6.0 + np.sqrt(15.0)) / 21.0
        c = (6.0 - np.sqrt(15.0)) / 21.0
        d = (9.0 - 2.0 * np.sqrt(15.0)) / 21.0
        e = (9.0 + 2.0 * np.sqrt(15.0)) / 21.0
        w1 = 9.0 / 80.0
        w2 = (155.0 - np.sqrt(15.0)) / 2400.0
        w3 = (155.0 + np.sqrt(15.0)) / 2400.0
        bary = np.array([
            [a, a, a],
            [b, b, d],
            [b, d, b],
            [d, b, b],
            [c, c, e],
            [c, e, c],
            [e, c, c]
        ])
        weights = np.array([w1, w2, w2, w2, w3, w3, w3])
    else:

        bary = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        weights = np.array([0.5])
    return bary, weights


def barycentric_to_cartesian(bary, vertices):
    return np.dot(bary, vertices)






def assemble_fem_2d_diffusion(nodes, elements, diffusivity, absorption, source_func):
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)


    bary_qp, w_qp = triangle_gauss_rule(order=7)
    n_qp = len(w_qp)

    for e in range(n_elements):
        idx = elements[e, :]
        verts = nodes[idx, :]



        J_mat = np.array([
            [verts[1, 0] - verts[0, 0], verts[2, 0] - verts[0, 0]],
            [verts[1, 1] - verts[0, 1], verts[2, 1] - verts[0, 1]]
        ])
        detJ = np.linalg.det(J_mat)
        if abs(detJ) < 1e-14:
            continue
        invJ = np.linalg.inv(J_mat)


        K_local = np.zeros((3, 3), dtype=float)
        M_local = np.zeros((3, 3), dtype=float)
        F_local = np.zeros(3, dtype=float)


        grad_N_ref = np.array([
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float)


        grad_N_phy = np.dot(grad_N_ref, invJ.T)

        for q in range(n_qp):
            xi = bary_qp[q, 1]
            eta = bary_qp[q, 2]
            w = w_qp[q] * abs(detJ)

            xy_phys = verts[0, :] + xi * (verts[1, :] - verts[0, :]) + eta * (verts[2, :] - verts[0, :])
            D_val = diffusivity(xy_phys[0], xy_phys[1]) if callable(diffusivity) else float(diffusivity)
            mu_val = absorption(xy_phys[0], xy_phys[1]) if callable(absorption) else float(absorption)
            S_val = source_func(xy_phys[0], xy_phys[1]) if callable(source_func) else float(source_func)


            for i in range(3):
                for j in range(3):
                    K_local[i, j] += D_val * np.dot(grad_N_phy[i, :], grad_N_phy[j, :]) * w
                    M_local[i, j] += mu_val * w / 3.0
                F_local[i] += S_val * w / 3.0


        for i in range(3):
            for j in range(3):
                K[idx[i], idx[j]] += K_local[i, j] + M_local[i, j]
            F[idx[i]] += F_local[i]

    return K, F


def solve_fem_2d_diffusion(nodes, elements, diffusivity, absorption, source_func,
                           dirichlet_nodes=None, dirichlet_values=None):
    nodes = np.asarray(nodes, dtype=float)
    n_nodes = nodes.shape[0]
    K, F = assemble_fem_2d_diffusion(nodes, elements, diffusivity, absorption, source_func)

    phi = np.zeros(n_nodes, dtype=float)

    if dirichlet_nodes is not None and dirichlet_values is not None:
        dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=int)
        dirichlet_values = np.asarray(dirichlet_values, dtype=float)
        phi[dirichlet_nodes] = dirichlet_values

        for idx in dirichlet_nodes:
            F -= K[:, idx] * phi[idx]
            K[idx, :] = 0.0
            K[:, idx] = 0.0
            K[idx, idx] = 1.0
            F[idx] = phi[idx]

    phi = np.linalg.solve(K, F)
    return phi






def triangle_area(vertices):
    v = np.asarray(vertices, dtype=float)
    area = 0.5 * abs((v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1])
                     - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1]))
    return area


def mesh_quality_min_angle(nodes, elements):
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    min_angle = np.inf
    for e in elements:
        v = nodes[e, :]
        for i in range(3):
            a = v[(i + 1) % 3, :] - v[i, :]
            b = v[(i + 2) % 3, :] - v[i, :]
            cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-14)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            min_angle = min(min_angle, angle)
    return np.degrees(min_angle)
