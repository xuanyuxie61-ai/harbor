
import numpy as np
from typing import Tuple, Optional, Callable


def sphere_llt_grid_points(r: float, pc: np.ndarray,
                           lat_num: int, long_num: int) -> np.ndarray:
    pc = np.asarray(pc, dtype=float).flatten()
    if pc.size != 3:
        raise ValueError("pc 必须为三维坐标")
    if lat_num < 0 or long_num < 1:
        raise ValueError("lat_num >= 0 且 long_num >= 1")

    point_num = 2 + lat_num * long_num
    p = np.zeros((point_num, 3), dtype=float)
    n = 0


    p[n, :] = pc + np.array([0.0, 0.0, r])
    n += 1


    for lat in range(1, lat_num + 1):
        phi = np.pi * lat / (lat_num + 1)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        for long_idx in range(long_num):
            theta = 2.0 * np.pi * long_idx / long_num
            p[n, 0] = pc[0] + r * sin_phi * np.cos(theta)
            p[n, 1] = pc[1] + r * sin_phi * np.sin(theta)
            p[n, 2] = pc[2] + r * cos_phi
            n += 1


    p[n, :] = pc + np.array([0.0, 0.0, -r])
    n += 1

    return p


def sphere_llt_grid_line_count(lat_num: int, long_num: int) -> int:
    if lat_num < 0 or long_num < 1:
        return 0
    return long_num * (lat_num + 1) + long_num * lat_num + long_num * max(lat_num - 1, 0)


def distmesh_2d_simple(fd: Callable[[np.ndarray], np.ndarray],
                       fh: Callable[[np.ndarray], np.ndarray],
                       h0: float,
                       box: np.ndarray,
                       iteration_max: int = 100,
                       pfix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if pfix is None:
        pfix = np.zeros((0, 2), dtype=float)

    dptol = 0.001
    ttol = 0.1
    Fscale = 1.2
    deltat = 0.2
    geps = 0.001 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0


    x_grid = np.arange(box[0, 0], box[1, 0] + h0, h0)
    y_grid = np.arange(box[0, 1], box[1, 1] + h0 * np.sqrt(3.0) / 2.0,
                       h0 * np.sqrt(3.0) / 2.0)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    x_mesh[1::2, :] += h0 / 2.0
    p = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T


    d_val = fd(p)
    p = p[d_val < geps, :]
    if p.shape[0] == 0:
        return pfix.copy(), np.zeros((0, 3), dtype=int)

    r0 = 1.0 / (fh(p) ** 2)
    r0_max = np.max(r0) if r0.size > 0 else 1.0
    keep = np.random.rand(p.shape[0]) < (r0 / r0_max)
    p = np.vstack([pfix, p[keep, :]])


    p_unique, idx = np.unique(p, axis=0, return_index=True)

    order = np.argsort(idx)
    p = p_unique[order, :]
    N = p.shape[0]

    if iteration_max <= 0:

        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(p)
            t = tri.simplices
            pmid = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3.0
            t = t[fd(pmid) < -geps, :]
            return p, t
        except Exception:
            return p, np.zeros((0, 3), dtype=int)

    pold = np.full_like(p, np.inf)
    iteration = 0
    triangulation_count = 0

    try:
        from scipy.spatial import Delaunay
    except Exception:
        return p, np.zeros((0, 3), dtype=int)

    t = None
    while iteration < iteration_max:
        iteration += 1
        displacement = np.sqrt(np.sum((p - pold) ** 2, axis=1)) / h0
        if np.max(displacement) > ttol or t is None:
            pold = p.copy()
            tri = Delaunay(p)
            triangulation_count += 1
            t = tri.simplices
            pmid = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3.0
            t = t[fd(pmid) < -geps, :]
            if t.size == 0:
                break
            bars = np.vstack([t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]])
            bars = np.unique(np.sort(bars, axis=1), axis=0)
        else:
            bars = np.vstack([t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]])
            bars = np.unique(np.sort(bars, axis=1), axis=1)


        barvec = p[bars[:, 0], :] - p[bars[:, 1], :]
        L = np.sqrt(np.sum(barvec ** 2, axis=1))
        L = np.maximum(L, 1e-12)
        hbars = fh((p[bars[:, 0], :] + p[bars[:, 1], :]) / 2.0)
        L0 = hbars * Fscale * np.sqrt(np.sum(L ** 2) / np.sum(hbars ** 2))
        F = np.maximum(L0 - L, 0.0)
        Fvec = (F / L)[:, None] * barvec

        Ftot = np.zeros((N, 2), dtype=float)
        np.add.at(Ftot, bars[:, 0], Fvec)
        np.add.at(Ftot, bars[:, 1], -Fvec)
        if pfix.shape[0] > 0:
            Ftot[:pfix.shape[0], :] = 0.0
        p = p + deltat * Ftot


        d_val = fd(p)
        ix = d_val > 0
        if np.any(ix):
            px = p[ix, :].copy()
            dgradx = (fd(px + np.array([deps, 0.0])[None, :]) - d_val[ix]) / deps
            dgrady = (fd(px + np.array([0.0, deps])[None, :]) - d_val[ix]) / deps
            p[ix, 0] -= d_val[ix] * dgradx
            p[ix, 1] -= d_val[ix] * dgrady


        interior = d_val < -geps
        if not np.any(interior):
            break
        max_move = np.max(np.sqrt(np.sum((deltat * Ftot[interior, :]) ** 2, axis=1))) / h0
        if max_move < dptol:
            break

    return p, t


def tet_mesh_quality_metrics(node_xyz: np.ndarray,
                             tetra_node: np.ndarray) -> dict:
    node_xyz = np.asarray(node_xyz, dtype=float)
    tetra_node = np.asarray(tetra_node, dtype=int)
    nt = tetra_node.shape[0]
    if nt == 0:
        return {}

    q1 = np.zeros(nt, dtype=float)
    q2 = np.zeros(nt, dtype=float)
    q3 = np.zeros(nt, dtype=float)
    q4 = np.zeros(nt, dtype=float)
    q5 = np.zeros(nt, dtype=float)

    for e in range(nt):
        idx = tetra_node[e, :]
        v0 = node_xyz[idx[0], :]
        v1 = node_xyz[idx[1], :]
        v2 = node_xyz[idx[2], :]
        v3 = node_xyz[idx[3], :]


        e1 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0


        vol = abs(np.linalg.det(np.vstack([e1, e2, e3]))) / 6.0
        vol = max(vol, 1e-18)


        edges = [
            np.sum((v0 - v1) ** 2),
            np.sum((v0 - v2) ** 2),
            np.sum((v0 - v3) ** 2),
            np.sum((v1 - v2) ** 2),
            np.sum((v1 - v3) ** 2),
            np.sum((v2 - v3) ** 2),
        ]
        l_sum = sum(edges)
        l_max = max(edges)


        q1[e] = 216.0 * np.sqrt(3.0) * vol / (l_sum ** 1.5)

        q2[e] = 12.0 * np.sqrt(6.0) * vol / (l_max ** 1.5)

        mat = np.vstack([e1, e2, e3])
        s = np.linalg.svd(mat, compute_uv=False)
        cond = s[0] / max(s[-1], 1e-18)
        q3[e] = 1.0 / cond

        q4[e] = 3.0 * vol / (np.sqrt(l_max) * l_sum)


        a_len = np.sqrt(np.sum(e1 ** 2))
        b_len = np.sqrt(np.sum(e2 ** 2))
        c_len = np.sqrt(np.sum(e3 ** 2))
        R = a_len * b_len * c_len / (12.0 * vol)

        face_areas = []
        for (a, b, c) in [(v0, v1, v2), (v0, v1, v3), (v0, v2, v3), (v1, v2, v3)]:
            u = b - a
            w = c - a
            cross = np.cross(u, w)
            face_areas.append(0.5 * np.linalg.norm(cross))
        A_sum = sum(face_areas)
        r_in = 3.0 * vol / max(A_sum, 1e-18)
        q5[e] = r_in / max(R, 1e-18)

    def stats(arr):
        return {
            'min': float(np.min(arr)),
            'mean': float(np.mean(arr)),
            'max': float(np.max(arr)),
            'var': float(np.var(arr))
        }

    return {
        'quality1': stats(q1),
        'quality2': stats(q2),
        'quality3': stats(q3),
        'quality4': stats(q4),
        'quality5': stats(q5),
    }


def generate_planar_array(nx: int, ny: int, dx: float, dy: float,
                          aperture_type: str = 'rectangular') -> np.ndarray:
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    xv, yv = np.meshgrid(x, y)
    pos = np.vstack([xv.ravel(), yv.ravel(), np.zeros(xv.size)]).T
    if aperture_type == 'circular':
        r_max = min(nx * dx, ny * dy) / 2.0
        r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        pos = pos[r <= r_max, :]
    return pos


def generate_conformal_array(r: float, lat_num: int, long_num: int) -> np.ndarray:
    return sphere_llt_grid_points(r, np.array([0.0, 0.0, 0.0]), lat_num, long_num)
