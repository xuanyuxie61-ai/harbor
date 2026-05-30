
import numpy as np
from typing import Tuple


def epicycloid_xy(k: float, s: float, n: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    rsmall = 1.0
    t = np.linspace(0.0, 2.0 * np.pi * s, n)
    x = rsmall * (k + 1.0) * np.cos(t) - rsmall * np.cos((k + 1.0) * t)
    y = rsmall * (k + 1.0) * np.sin(t) - rsmall * np.sin((k + 1.0) * t)
    return x, y


def epicycloid_arc_length(k: float, s: float, n: int = 1000) -> float:
    t = np.linspace(0.0, 2.0 * np.pi * s, n)
    dx_dt = -(k + 1.0) * np.sin(t) + (k + 1.0) * np.sin((k + 1.0) * t)
    dy_dt = (k + 1.0) * np.cos(t) - (k + 1.0) * np.cos((k + 1.0) * t)
    ds = np.sqrt(dx_dt ** 2 + dy_dt ** 2)
    return float(np.trapz(ds, t))


def epicycloid_curvature(k: float, s: float, n: int = 500) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi * s, n)
    dx = -(k + 1.0) * np.sin(t) + (k + 1.0) * np.sin((k + 1.0) * t)
    dy = (k + 1.0) * np.cos(t) - (k + 1.0) * np.cos((k + 1.0) * t)
    d2x = -(k + 1.0) * np.cos(t) + (k + 1.0) ** 2 * np.cos((k + 1.0) * t)
    d2y = -(k + 1.0) * np.sin(t) + (k + 1.0) ** 2 * np.sin((k + 1.0) * t)
    num = np.abs(dx * d2y - dy * d2x)
    den = (dx ** 2 + dy ** 2) ** 1.5
    den = np.where(den < 1e-15, 1e-15, den)
    return num / den


def embed_epicycloid_high_dim(k: float, s: float, D: int = 10,
                               n: int = 200) -> np.ndarray:
    x, y = epicycloid_xy(k, s, n)
    data = np.zeros((n, D))
    data[:, 0] = x
    data[:, 1] = y

    np.random.seed(42)
    for d in range(2, D):
        freq = 0.5 + d * 0.3
        amp = 0.1 / d
        data[:, d] = amp * np.sin(freq * np.linspace(0, 2 * np.pi, n))
    return data


def christoffel_symbols(metric: np.ndarray, h: float = 1e-5) -> np.ndarray:
    D = metric(np.zeros(2)).shape[0]
    Gamma = np.zeros((D, D, D))

    x0 = np.zeros(2)
    g0 = metric(x0)
    g_inv = np.linalg.inv(g0 + 1e-6 * np.eye(D))
    for mu in range(D):
        for nu in range(D):
            for rho in range(D):

                dg_nu = np.zeros((D, D))
                dg_rho = np.zeros((D, D))
                dg_sigma = np.zeros((D, D))

                Gamma[mu, nu, rho] = 0.5 * (
                    g_inv[mu, 0] * (dg_nu[0, rho] + dg_rho[0, nu] - dg_sigma[nu, rho]) +
                    g_inv[mu, 1] * (dg_nu[1, rho] + dg_rho[1, nu] - dg_sigma[nu, rho])
                )
    return Gamma


def geodesic_distance_estimate(data: np.ndarray, i: int, j: int,
                                k: int = 10) -> float:
    N = len(data)
    dists = np.linalg.norm(data - data[i], axis=1)
    visited = np.zeros(N, dtype=bool)
    distances = np.full(N, np.inf)
    distances[i] = 0.0
    prev = -1 * np.ones(N, dtype=int)
    for _ in range(N):
        u = -1
        min_dist = np.inf
        for v in range(N):
            if not visited[v] and distances[v] < min_dist:
                min_dist = distances[v]
                u = v
        if u == -1:
            break
        visited[u] = True
        if u == j:
            break

        neigh_dists = np.linalg.norm(data - data[u], axis=1)
        knn_idx = np.argsort(neigh_dists)[1:k + 1]
        for v in knn_idx:
            if not visited[v]:
                alt = distances[u] + neigh_dists[v]
                if alt < distances[v]:
                    distances[v] = alt
                    prev[v] = u
    return distances[j]


def isometric_embedding_quality(data_high: np.ndarray,
                                 data_low: np.ndarray) -> float:
    N = len(data_high)
    num = 0.0
    den = 0.0
    count = 0
    for i in range(min(N, 50)):
        for j in range(i + 1, min(N, 50)):
            d_h = np.linalg.norm(data_high[i] - data_high[j])
            d_l = np.linalg.norm(data_low[i] - data_low[j])
            num += (d_h - d_l) ** 2
            den += d_h ** 2
            count += 1
    if den < 1e-15:
        return 1.0
    return max(0.0, 1.0 - num / den)
