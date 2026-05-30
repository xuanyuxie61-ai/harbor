
import numpy as np
from typing import Tuple, Optional


def sammon_mapping(
    X: np.ndarray,
    n_components: int = 2,
    max_iter: int = 300,
    alpha: float = 0.3,
    tol: float = 1e-5,
    random_state: int = 42,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    if n_samples < 2:
        raise ValueError("need at least 2 samples")

    rng = np.random.RandomState(random_state)


    dist_high = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = np.linalg.norm(X[i, :] - X[j, :])
            dist_high[i, j] = d
            dist_high[j, i] = d


    dist_high = np.where(dist_high < 1e-12, 1e-12, dist_high)

    sum_dist = np.sum(dist_high)
    if sum_dist < 1e-12:
        raise ValueError("all pairwise distances are zero")


    X_centered = X - np.mean(X, axis=0)
    cov = X_centered.T @ X_centered / n_samples
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    Y = X_centered @ eigvecs[:, idx[:n_components]]


    scale = np.mean(dist_high) / (np.mean(np.linalg.norm(Y[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)) + 1e-12)
    Y *= scale


    for iteration in range(max_iter):
        dist_low = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = np.linalg.norm(Y[i, :] - Y[j, :])
                dist_low[i, j] = max(d, 1e-12)
                dist_low[j, i] = dist_low[i, j]


        delta = np.zeros_like(Y)
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    continue
                diff = Y[i, :] - Y[j, :]
                d_low = dist_low[i, j]
                d_high = dist_high[i, j]
                factor = -2.0 / sum_dist * (d_high - d_low) / (d_high * d_low)
                delta[i, :] += factor * diff


        second_order = np.zeros(n_samples)
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    continue
                d_low = dist_low[i, j]
                d_high = dist_high[i, j]
                second_order[i] += -2.0 / sum_dist / (d_high * d_low)

        second_order = np.abs(second_order) + 1e-12


        Y_new = Y.copy()
        for i in range(n_samples):
            Y_new[i, :] -= alpha * delta[i, :] / second_order[i]


        stress_change = np.linalg.norm(Y_new - Y) / (np.linalg.norm(Y) + 1e-12)
        Y = Y_new

        if stress_change < tol:
            break

    return Y


def pwl_interp_2d(
    xd: np.ndarray,
    yd: np.ndarray,
    zd: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
) -> np.ndarray:
    xd = np.asarray(xd, dtype=np.float64)
    yd = np.asarray(yd, dtype=np.float64)
    zd = np.asarray(zd, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    yi = np.asarray(yi, dtype=np.float64)

    nxd = len(xd)
    nyd = len(yd)
    ni = len(xi)

    if zd.shape != (nxd, nyd):
        raise ValueError(f"zd shape {zd.shape} does not match ({nxd}, {nyd})")

    zi = np.full(ni, np.inf, dtype=np.float64)

    for k in range(ni):

        i = np.searchsorted(xd, xi[k], side='right') - 1
        if i < 0 or i >= nxd - 1:
            continue

        j = np.searchsorted(yd, yi[k], side='right') - 1
        if j < 0 or j >= nyd - 1:
            continue



        slope = (yd[j + 1] - yd[j]) / (xd[i + 1] - xd[i] + 1e-18)
        diag_y = yd[j] + slope * (xi[k] - xd[i])

        if yi[k] < diag_y:

            dxa = xd[i + 1] - xd[i]
            dya = 0.0
            dxb = 0.0
            dyb = yd[j + 1] - yd[j]
            dxi = xi[k] - xd[i]
            dyi = yi[k] - yd[j]
            det = dxa * dyb - dya * dxb
            if abs(det) < 1e-18:
                continue
            alpha = (dxi * dyb - dyi * dxb) / det
            beta = (dxa * dyi - dya * dxi) / det
            gamma = 1.0 - alpha - beta
            if alpha < -1e-9 or beta < -1e-9 or gamma < -1e-9:
                continue
            zi[k] = alpha * zd[i + 1, j] + beta * zd[i, j + 1] + gamma * zd[i, j]
        else:

            dxa = xd[i] - xd[i + 1]
            dya = yd[j + 1] - yd[j + 1]
            dxb = xd[i + 1] - xd[i + 1]
            dyb = yd[j] - yd[j + 1]
            dxi = xi[k] - xd[i + 1]
            dyi = yi[k] - yd[j + 1]
            det = dxa * dyb - dya * dxb
            if abs(det) < 1e-18:
                continue
            alpha = (dxi * dyb - dyi * dxb) / det
            beta = (dxa * dyi - dya * dxi) / det
            gamma = 1.0 - alpha - beta
            if alpha < -1e-9 or beta < -1e-9 or gamma < -1e-9:
                continue
            zi[k] = alpha * zd[i, j + 1] + beta * zd[i + 1, j] + gamma * zd[i + 1, j + 1]

    return zi


def build_free_energy_surface(
    dihedral_angles: np.ndarray,
    temperature: float = 310.0,
    grid_n: int = 40,
) -> dict:

    Y = sammon_mapping(dihedral_angles, n_components=2, max_iter=200, alpha=0.2)


    q1, q2 = Y[:, 0], Y[:, 1]


    q1_min, q1_max = np.min(q1), np.max(q1)
    q2_min, q2_max = np.min(q2), np.max(q2)


    margin1 = 0.1 * (q1_max - q1_min)
    margin2 = 0.1 * (q2_max - q2_min)
    q1_min -= margin1
    q1_max += margin1
    q2_min -= margin2
    q2_max += margin2

    H, edges1, edges2 = np.histogram2d(
        q1, q2, bins=grid_n,
        range=[[q1_min, q1_max], [q2_min, q2_max]]
    )


    dx1 = edges1[1] - edges1[0]
    dx2 = edges2[1] - edges2[0]
    P = H / (np.sum(H) * dx1 * dx2 + 1e-18)










    raise NotImplementedError("Hole 2: 待实现自由能玻尔兹曼反转计算")



    xc = 0.5 * (edges1[:-1] + edges1[1:])
    yc = 0.5 * (edges2[:-1] + edges2[1:])



    local_min = []
    local_max = []
    for i in range(1, grid_n - 1):
        for j in range(1, grid_n - 1):
            neighborhood = F[i - 1:i + 2, j - 1:j + 2]
            if F[i, j] == np.min(neighborhood):
                local_min.append((xc[i], yc[j], F[i, j]))
            if F[i, j] == np.max(neighborhood):
                local_max.append((xc[i], yc[j], F[i, j]))


    fine_n = 80
    xi_fine = np.linspace(q1_min, q1_max, fine_n)
    yi_fine = np.linspace(q2_min, q2_max, fine_n)
    XI, YI = np.meshgrid(xi_fine, yi_fine)
    ZI = pwl_interp_2d(xc, yc, F.T, XI.ravel(), YI.ravel())
    ZI = ZI.reshape(fine_n, fine_n)


    ZI = np.where(np.isfinite(ZI), ZI, np.max(F))

    return {
        "sammon_coords": Y,
        "free_energy_ev": F,
        "grid_x": xc,
        "grid_y": yc,
        "fine_x": xi_fine,
        "fine_y": yi_fine,
        "fine_z": ZI,
        "local_minima": local_min,
        "local_maxima": local_max,
        "barrier_height_ev": float(np.max(F) - np.min(F)) if len(local_max) > 0 else 0.0,
    }


def simplex_vertex_coordinates(n_dim: int) -> np.ndarray:
    if n_dim < 1:
        raise ValueError("n_dim must be >= 1")

    x = np.zeros((n_dim, n_dim + 1), dtype=np.float64)
    for j in range(n_dim):
        x[j, j] = 1.0

    a = (1.0 - np.sqrt(1.0 + n_dim)) / n_dim
    x[:, n_dim] = a


    centroid = np.mean(x, axis=1, keepdims=True)
    x -= centroid


    s = np.linalg.norm(x[:, 0])
    if s < 1e-15:
        raise ValueError("degenerate simplex")
    x /= s

    return x
