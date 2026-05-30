
import numpy as np
from typing import Tuple, Optional, List


def bracket4(t: np.ndarray, s: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    nt = t.shape[0]
    if nt < 2:
        raise ValueError("t必须至少包含2个元素")
    left = np.zeros(s.shape[0], dtype=int)
    for idx, val in enumerate(s):

        lo, hi = 0, nt - 2
        if val <= t[0]:
            left[idx] = 0
            continue
        if val >= t[-1]:
            left[idx] = nt - 2
            continue
        while lo < hi:
            mid = (lo + hi) // 2
            if t[mid] <= val:
                if val <= t[mid + 1]:
                    lo = mid
                    break
                lo = mid + 1
            else:
                hi = mid - 1
        left[idx] = lo
    return left


def fem1d_interpolate(node_x: np.ndarray, node_val: np.ndarray,
                      sample_x: np.ndarray) -> np.ndarray:
    node_x = np.asarray(node_x, dtype=float)
    node_val = np.asarray(node_val, dtype=float)
    sample_x = np.asarray(sample_x, dtype=float)
    left = bracket4(node_x, sample_x)
    sample_val = np.zeros(sample_x.shape[0])
    for i, l in enumerate(left):
        r = min(l + 1, node_x.shape[0] - 1)
        h = node_x[r] - node_x[l]
        if abs(h) < 1e-14:
            sample_val[i] = node_val[l]
        else:
            w_l = (node_x[r] - sample_x[i]) / h
            w_r = (sample_x[i] - node_x[l]) / h
            sample_val[i] = node_val[l] * w_l + node_val[r] * w_r
    return sample_val


def tetrahedron_volume(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if t.shape != (3, 4):
        raise ValueError("t必须是3x4矩阵")
    v1 = t[:, 1] - t[:, 0]
    v2 = t[:, 2] - t[:, 0]
    v3 = t[:, 3] - t[:, 0]
    vol = abs(np.linalg.det(np.vstack([v1, v2, v3]))) / 6.0
    return vol


def basis_mn_tet4(t: np.ndarray, n: int, p: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if p.ndim == 1 and t.shape[0] == 3:
        p = p.reshape(3, 1)
        n = 1
    phi = np.zeros((4, n))


    M = np.vstack([t, np.ones(4)])
    volume = np.linalg.det(M)
    if abs(volume) < 1e-14:
        raise ValueError("四面体体积为零")

    for k in range(n):
        pk = p[:, k] if n > 1 else p[:, 0]
        for i in range(4):
            Mi = M.copy()
            Mi[:3, i] = pk
            phi[i, k] = np.linalg.det(Mi) / volume
    return phi


def project_sample_to_fem3d(sample_nodes: np.ndarray, sample_vals: np.ndarray,
                            fem_nodes: np.ndarray, fem_elements: np.ndarray) -> np.ndarray:
    n_fem = fem_nodes.shape[0]
    fem_vals = np.zeros(n_fem)
    weights = np.zeros(n_fem)

    for elem in fem_elements:
        t = fem_nodes[elem, :].T
        try:
            vol = tetrahedron_volume(t)
        except ValueError:
            continue

        centroid = np.mean(t, axis=1)

        dists = np.linalg.norm(sample_nodes - centroid, axis=1)
        nearest = int(np.argmin(dists))
        val = sample_vals[nearest]


        for i, node_idx in enumerate(elem):
            w = vol * 0.25
            fem_vals[node_idx] += w * val
            weights[node_idx] += w


    mask = weights > 1e-14
    fem_vals[mask] /= weights[mask]
    fem_vals[~mask] = 0.0
    return fem_vals


class FEMExpectationSampler:
    def __init__(self, n_qubits: int, n_grid_1d: int = 32):
        self.n_qubits = n_qubits
        self.n_grid_1d = n_grid_1d
        self.dim = 2 ** n_qubits

        self.theta_grid = np.linspace(-np.pi, np.pi, n_grid_1d)
        self.cache: dict = {}

    def estimate_energy_1d(self, theta_vals: np.ndarray, energy_vals: np.ndarray,
                           target_theta: float) -> float:
        return float(fem1d_interpolate(theta_vals, energy_vals,
                                       np.array([target_theta]))[0])

    def build_probability_density(self, bitstrings: np.ndarray,
                                   probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        x_vals = bitstrings / (2 ** self.n_qubits - 1 + 1e-14)

        nodes = np.linspace(0, 1, self.n_grid_1d)
        vals = np.zeros(self.n_grid_1d)
        counts = np.zeros(self.n_grid_1d)
        for x, p in zip(x_vals, probabilities):
            idx = int(np.clip(x * (self.n_grid_1d - 1), 0, self.n_grid_1d - 2))
            vals[idx] += p
            counts[idx] += 1
        mask = counts > 0
        vals[mask] /= counts[mask]
        return nodes, vals

    def integrate_expectation(self, nodes: np.ndarray, vals: np.ndarray,
                               observable_func: callable) -> float:
        dx = np.diff(nodes)
        integrand = observable_func(nodes) * vals

        result = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dx)
        return float(result)
