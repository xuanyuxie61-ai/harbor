
import numpy as np


def distance_cylinder(p, R=0.05, H=0.20):
    if p.ndim == 1:
        p = p.reshape(1, -1)
    r = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    d1 = r - R
    d2 = np.abs(p[:, 2]) - H / 2.0
    return np.maximum(d1, d2)


def target_edge_length(p, h0, R=0.05):
    if p.ndim == 1:
        p = p.reshape(1, -1)
    d = distance_cylinder(p, R)

    h = h0 * (1.0 + 0.5 * np.clip(np.abs(d) / R, 0.0, 2.0))
    return h


def generate_mesh_3d(h0=0.015, R=0.05, H=0.20, iteration_max=20, pfix=None):
    dim = 3
    ptol = 0.001
    ttol = 0.1
    L0mult = 1.0 + 0.4 / 2.0 ** (dim - 1)
    deltat = 0.1
    geps = 0.1 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0


    box = np.array([[-R, -R, -H / 2.0],
                    [R, R, H / 2.0]])


    grids = [np.arange(box[0, i], box[1, i] + h0, h0) for i in range(dim)]
    mesh = np.meshgrid(*grids, indexing='ij')
    p = np.vstack([m.ravel() for m in mesh]).T


    p = p[distance_cylinder(p, R, H) < geps]


    r0 = target_edge_length(p, h0, R)
    if len(p) > 0:
        prob = np.min(r0) ** dim / (r0 ** dim)
        p = p[np.random.rand(len(p)) < prob]


    if pfix is not None and len(pfix) > 0:
        p = np.vstack([pfix, p])
    else:
        pfix = np.zeros((0, 3))


    p = np.unique(np.round(p / (h0 * 1.0e-6)) * (h0 * 1.0e-6), axis=0)

    if iteration_max <= 0:
        try:
            from scipy.spatial import Delaunay
            t = Delaunay(p).simplices
        except Exception:
            t = np.zeros((0, 4), dtype=int)
        return p, t

    N = len(p)
    count = 0
    p0 = np.inf * np.ones_like(p)
    t = np.zeros((0, 4), dtype=int)

    for iteration in range(iteration_max):

        if ttol * h0 < np.max(np.sqrt(np.sum((p - p0) ** 2, axis=1))):
            p0 = p.copy()
            try:
                from scipy.spatial import Delaunay
                t_new = Delaunay(p).simplices
            except Exception:
                break


            pmid = np.zeros((len(t_new), dim))
            for ii in range(dim + 1):
                pmid += p[t_new[:, ii], :] / (dim + 1)
            t_new = t_new[distance_cylinder(pmid, R, H) < -geps]
            t = t_new
            count += 1

        if len(t) == 0:
            break


        edges = []
        for i in range(dim + 1):
            for j in range(i + 1, dim + 1):
                edges.append(t[:, [i, j]])
        edges = np.vstack(edges)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)


        bars = p[edges[:, 0], :] - p[edges[:, 1], :]
        L = np.sqrt(np.sum(bars ** 2, axis=1))

        mid = (p[edges[:, 0], :] + p[edges[:, 1], :]) / 2.0
        L0 = target_edge_length(mid, h0, R)
        L0 = L0 * L0mult * (np.sum(L ** dim) / np.sum(L0 ** dim)) ** (1.0 / dim)

        F = np.maximum(L0 - L, 0.0)
        Fbar = np.hstack([bars, -bars]) * np.tile(F / np.maximum(L, 1.0e-12), (1, 2 * dim)).reshape(-1, 2 * dim)


        dp = np.zeros((N, dim))
        for idx in range(len(edges)):
            for d in range(dim):
                dp[edges[idx, 0], d] += Fbar[idx, d]
                dp[edges[idx, 1], d] += Fbar[idx, dim + d]


        if len(pfix) > 0:
            dp[:len(pfix), :] = 0.0

        p = p + deltat * dp


        for _ in range(2):
            d = distance_cylinder(p, R, H)
            ix = d > 0
            if not np.any(ix):
                break
            gradd = np.zeros((np.sum(ix), dim))
            for ii in range(dim):
                a = np.zeros((1, dim))
                a[0, ii] = deps
                d1x = distance_cylinder(p[ix, :] + np.ones((np.sum(ix), 1)) * a, R, H)
                gradd[:, ii] = (d1x - d[ix]) / deps

            grad_norm = np.sqrt(np.sum(gradd ** 2, axis=1))
            grad_norm = np.where(grad_norm < 1.0e-12, 1.0, grad_norm)
            p[ix, :] -= d[ix][:, None] * gradd / grad_norm[:, None]


        maxdp = np.max(deltat * np.sqrt(np.sum(dp[d < -geps] ** 2, axis=1))) if np.any(d < -geps) else 0.0
        if maxdp < ptol * h0:
            break

    return p, t
