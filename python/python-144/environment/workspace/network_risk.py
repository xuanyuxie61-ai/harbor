
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay


def build_asset_digraph(n: int, threshold: float, corr: np.ndarray) -> np.ndarray:
    if corr.shape != (n, n):
        raise ValueError("build_asset_digraph: 相关性矩阵维度不匹配。")
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and abs(corr[i, j]) >= threshold:
                adj[i, j] = abs(corr[i, j])

    row_sums = adj.sum(axis=1)
    isolated = row_sums == 0
    adj[np.diag_indices(n)] = np.where(isolated, 1.0, adj[np.diag_indices(n)])
    return adj


def pagerank_systemic_risk(adj: np.ndarray, damping: float = 0.85,
                            max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    n = adj.shape[0]
    if n == 0:
        return np.array([])

    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    P = adj / row_sums

    x = np.ones(n) / n
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        x_new = damping * (P.T @ x) + teleport
        x_new = x_new / np.sum(x_new)
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        x = x_new
    return x


def delaunay_similarity_triangulation(positions: np.ndarray) -> np.ndarray:
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("delaunay_similarity_triangulation: 输入必须是 N×2 数组。")
    if positions.shape[0] < 3:
        return np.zeros((positions.shape[0], positions.shape[0]))
    tri = Delaunay(positions)
    n = positions.shape[0]
    adj = np.zeros((n, n))
    for simplex in tri.simplices:
        i, j, k = simplex
        adj[i, j] = 1.0
        adj[j, i] = 1.0
        adj[i, k] = 1.0
        adj[k, i] = 1.0
        adj[j, k] = 1.0
        adj[k, j] = 1.0
    return adj


def stochastic_risk_diffusion(network_adj: np.ndarray,
                               initial_risk: np.ndarray,
                               omega: np.ndarray,
                               nx: int = 20, ny: int = 20) -> np.ndarray:
    n = nx * ny
    if len(initial_risk) != n:
        raise ValueError("stochastic_risk_diffusion: initial_risk 长度必须等于 nx*ny。")
    if len(omega) != 4:
        raise ValueError("stochastic_risk_diffusion: omega 必须包含4个参数。")


    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]


    def diffusivity(xi, yj):
        return omega[0] + omega[1] * np.sin(np.pi * xi) * np.sin(np.pi * yj)


    row_ind = []
    col_ind = []
    data = []
    rhs = np.zeros(n)

    def idx(i, j):
        return i * nx + j

    for i in range(ny):
        for j in range(nx):
            k = idx(i, j)
            if i == 0 or i == ny - 1 or j == 0 or j == nx - 1:

                row_ind.append(k)
                col_ind.append(k)
                data.append(1.0)
                rhs[k] = omega[2]
            else:
                a_ip = diffusivity(x[j], 0.5 * (y[i] + y[i + 1]))
                a_im = diffusivity(x[j], 0.5 * (y[i] + y[i - 1]))
                a_jp = diffusivity(0.5 * (x[j] + x[j + 1]), y[i])
                a_jm = diffusivity(0.5 * (x[j] + x[j - 1]), y[i])

                coeff = 0.0

                center = (a_ip + a_im) / dy ** 2 + (a_jp + a_jm) / dx ** 2
                row_ind.append(k)
                col_ind.append(k)
                data.append(center)
                coeff += center


                row_ind.append(k)
                col_ind.append(idx(i + 1, j))
                data.append(-a_ip / dy ** 2)


                row_ind.append(k)
                col_ind.append(idx(i - 1, j))
                data.append(-a_im / dy ** 2)


                row_ind.append(k)
                col_ind.append(idx(i, j + 1))
                data.append(-a_jp / dx ** 2)


                row_ind.append(k)
                col_ind.append(idx(i, j - 1))
                data.append(-a_jm / dx ** 2)


                rhs[k] = omega[3] * initial_risk[k]

    A = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    u = spsolve(A, rhs)
    if np.any(np.isnan(u)) or np.any(np.isinf(u)):
        raise RuntimeError("stochastic_risk_diffusion: 求解失败，矩阵可能奇异。")
    return u.reshape((ny, nx))


def network_risk_contribution(adj: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    cov = np.cov(asset_returns.T)
    vol = np.sqrt(np.diag(cov))
    beta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if cov[j, j] > 1e-12:
                beta[i, j] = cov[i, j] / cov[j, j]
    rc = vol * (adj * beta).sum(axis=1)
    return rc
