
import numpy as np
from scipy.spatial import Delaunay


def simpvol(p, t):
    dim = p.shape[1]
    n_simplices = t.shape[0]
    vol = np.zeros(n_simplices)
    if dim == 1:
        for i in range(n_simplices):
            vol[i] = np.abs(p[t[i, 1], 0] - p[t[i, 0], 0])
    elif dim == 2:
        for i in range(n_simplices):
            v0, v1, v2 = p[t[i, 0]], p[t[i, 1]], p[t[i, 2]]
            vol[i] = 0.5 * np.abs((v1[0] - v0[0]) * (v2[1] - v0[1]) -
                                   (v2[0] - v0[0]) * (v1[1] - v0[1]))
    elif dim == 3:
        for i in range(n_simplices):
            v0, v1, v2, v3 = p[t[i, 0]], p[t[i, 1]], p[t[i, 2]], p[t[i, 3]]
            M = np.array([v1 - v0, v2 - v0, v3 - v0])
            vol[i] = np.abs(np.linalg.det(M)) / 6.0
    return vol


def surftri(p, t):

    faces = np.vstack([
        t[:, [1, 2, 3]],
        t[:, [0, 3, 2]],
        t[:, [0, 1, 3]],
        t[:, [0, 2, 1]]
    ])

    faces_sorted = np.sort(faces, axis=1)

    face_dict = {}
    for f in faces_sorted:
        key = (int(f[0]), int(f[1]), int(f[2]))
        face_dict[key] = face_dict.get(key, 0) + 1

    boundary_keys = [k for k, v in face_dict.items() if v == 1]
    boundary_faces = np.array(boundary_keys, dtype=int)
    return boundary_faces


def simp_qual_3d(p, t):
    n_tet = t.shape[0]
    qual = np.zeros(n_tet)
    for i in range(n_tet):
        v0, v1, v2, v3 = p[t[i, 0]], p[t[i, 1]], p[t[i, 2]], p[t[i, 3]]

        e01 = v1 - v0
        e02 = v2 - v0
        e03 = v3 - v0
        e12 = v2 - v1
        e13 = v3 - v1
        e23 = v3 - v2


        vol = np.abs(np.linalg.det(np.array([e01, e02, e03]))) / 6.0
        if vol < 1e-14:
            qual[i] = 0.0
            continue

        a1 = 0.5 * np.linalg.norm(np.cross(e02, e03))
        a2 = 0.5 * np.linalg.norm(np.cross(e01, e03))
        a3 = 0.5 * np.linalg.norm(np.cross(e01, e02))
        e21 = -e12
        e31 = -e13
        a4 = 0.5 * np.linalg.norm(np.cross(e21, e31))

        r_in = 3.0 * vol / (a1 + a2 + a3 + a4)

        a = np.linalg.norm(e01)
        b = np.linalg.norm(e02)
        c = np.linalg.norm(e03)
        d = np.linalg.norm(e23)
        e = np.linalg.norm(e13)
        f = np.linalg.norm(e12)

        CM = np.array([
            [0, 1, 1, 1, 1],
            [1, 0, a * a, b * b, c * c],
            [1, a * a, 0, f * f, e * e],
            [1, b * b, f * f, 0, d * d],
            [1, c * c, e * e, d * d, 0]
        ], dtype=float)
        det_cm = np.linalg.det(CM)
        if det_cm < 1e-14:
            qual[i] = 0.0
            continue
        r_circ = np.sqrt(np.abs(det_cm)) / (288.0 * vol * vol)
        if r_circ < 1e-14:
            qual[i] = 0.0
        else:
            qual[i] = 3.0 * r_in / r_circ
    return qual


def distmesh_3d(fd, fh, h0, box, iteration_max=100, pfix=None):
    dptol = 0.001
    ttol = 0.1
    Fscale = 1.2
    deltat = 0.2
    geps = 0.1 * h0


    x_min, x_max, y_min, y_max, z_min, z_max = box
    x_vec = np.arange(x_min, x_max, h0)
    y_vec = np.arange(y_min, y_max, h0)
    z_vec = np.arange(z_min, z_max, h0)
    xx, yy, zz = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')
    p = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


    d = fd(p)
    p = p[d < geps]

    if pfix is not None and pfix.size > 0:
        p = np.vstack((pfix, p))

        p = np.unique(p, axis=0)

    N = p.shape[0]
    p_old = np.inf * np.ones_like(p)

    for iteration in range(iteration_max):

        if np.max(np.sqrt(np.sum((p - p_old) ** 2, axis=1))) > ttol * h0:
            p_old = p.copy()

            tri = Delaunay(p)
            t = tri.simplices

            pc = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]] + p[t[:, 3]]) / 4.0
            t = t[fd(pc) < -geps]



        edges = np.vstack([
            t[:, [0, 1]], t[:, [0, 2]], t[:, [0, 3]],
            t[:, [1, 2]], t[:, [1, 3]], t[:, [2, 3]]
        ])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)


        bar_vec = p[edges[:, 1]] - p[edges[:, 0]]
        L = np.sqrt(np.sum(bar_vec ** 2, axis=1))
        L = np.maximum(L, 1e-14)

        pc_mid = (p[edges[:, 0]] + p[edges[:, 1]]) / 2.0
        h_mid = fh(pc_mid)
        L0 = h_mid * Fscale * np.sqrt(np.sum(L ** 2) / np.sum(h_mid ** 2))
        L0 = np.maximum(L0, 1e-14)

        F = np.maximum(L0 - L, 0.0)

        F_vec = (F / L)[:, None] * bar_vec


        Ftot = np.zeros_like(p)
        for i in range(edges.shape[0]):
            Ftot[edges[i, 0]] -= F_vec[i]
            Ftot[edges[i, 1]] += F_vec[i]


        p += deltat * Ftot


        d = fd(p)
        is_boundary = np.abs(d) < geps
        if np.any(is_boundary):

            eps_proj = 0.01 * h0
            for idx in np.where(is_boundary)[0]:
                pt = p[idx]

                gx = (fd(np.array([[pt[0] + eps_proj, pt[1], pt[2]]]))[0] -
                      fd(np.array([[pt[0] - eps_proj, pt[1], pt[2]]]))[0]) / (2 * eps_proj)
                gy = (fd(np.array([[pt[0], pt[1] + eps_proj, pt[2]]]))[0] -
                      fd(np.array([[pt[0], pt[1] - eps_proj, pt[2]]]))[0]) / (2 * eps_proj)
                gz = (fd(np.array([[pt[0], pt[1], pt[2] + eps_proj]]))[0] -
                      fd(np.array([[pt[0], pt[1], pt[2] - eps_proj]]))[0]) / (2 * eps_proj)
                grad = np.array([gx, gy, gz])
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-14:
                    p[idx] -= d[idx] * grad / grad_norm


        max_move = np.max(np.sqrt(np.sum(Ftot ** 2, axis=1)))
        if max_move < dptol * h0:
            break


    tri = Delaunay(p)
    t = tri.simplices
    pc = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]] + p[t[:, 3]]) / 4.0
    t = t[fd(pc) < -geps]
    return p, t


def refine_mesh_near_boundary(p, t, fd, h0, n_refinements=2):
    for _ in range(n_refinements):

        pc = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]] + p[t[:, 3]]) / 4.0
        d_centers = fd(pc)
        near_boundary = np.abs(d_centers) < 2.0 * h0
        if not np.any(near_boundary):
            break

        refine_tets = t[near_boundary]
        edges = np.vstack([
            refine_tets[:, [0, 1]], refine_tets[:, [0, 2]], refine_tets[:, [0, 3]],
            refine_tets[:, [1, 2]], refine_tets[:, [1, 3]], refine_tets[:, [2, 3]]
        ])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        bar_vec = p[edges[:, 1]] - p[edges[:, 0]]
        L = np.sqrt(np.sum(bar_vec ** 2, axis=1))
        long_edges = edges[L > 1.2 * h0]
        if long_edges.shape[0] == 0:
            break

        new_points = []
        edge_to_mid = {}
        for i, e in enumerate(long_edges):
            mid = (p[e[0]] + p[e[1]]) / 2.0
            new_points.append(mid)
            edge_to_mid[(e[0], e[1])] = p.shape[0] + i
        if len(new_points) == 0:
            break
        p = np.vstack((p, np.array(new_points)))

        tri = Delaunay(p)
        t = tri.simplices
        pc = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]] + p[t[:, 3]]) / 4.0
        t = t[fd(pc) < -0.1 * h0]
    return p, t


def mesh_statistics(p, t):
    vols = simpvol(p, t)
    qual = simp_qual_3d(p, t)
    stats = {
        'node_num': p.shape[0],
        'tet_num': t.shape[0],
        'volume_min': float(np.min(vols)),
        'volume_mean': float(np.mean(vols)),
        'volume_max': float(np.max(vols)),
        'quality_min': float(np.min(qual)),
        'quality_mean': float(np.mean(qual)),
        'quality_max': float(np.max(qual)),
    }
    return stats


def write_mesh_files(p, t, prefix):
    np.savetxt(f"{prefix}_nodes.txt", p, fmt='%.6f')
    np.savetxt(f"{prefix}_elements.txt", t, fmt='%d')
