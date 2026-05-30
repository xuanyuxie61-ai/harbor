
import numpy as np
from typing import Tuple, List, Optional


def hypercube_grid(m: int, ns: np.ndarray, a: np.ndarray,
                   b: np.ndarray, c: Optional[np.ndarray] = None) -> np.ndarray:
    ns = np.asarray(ns, dtype=int)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if c is None:
        c = np.ones(m, dtype=int)
    else:
        c = np.asarray(c, dtype=int)

    if ns.shape[0] != m or a.shape[0] != m or b.shape[0] != m:
        raise ValueError("维度参数长度不一致")
    n = int(np.prod(ns))
    x = np.zeros((m, n))

    for i in range(m):
        s = ns[i]
        xs = np.zeros(s)
        for j in range(s):
            if c[i] == 1:
                if s == 1:
                    xs[j] = 0.5 * (a[i] + b[i])
                else:
                    xs[j] = ((s - 1 - j) * a[i] + j * b[i]) / (s - 1)
            elif c[i] == 2:
                xs[j] = ((s - j) * a[i] + (j + 1) * b[i]) / (s + 1)
            elif c[i] == 3:
                xs[j] = ((s - j) * a[i] + (j - 1) * b[i]) / s
            elif c[i] == 4:
                xs[j] = ((s - 1 - j) * a[i] + (j + 1) * b[i]) / s
            elif c[i] == 5:
                xs[j] = ((2 * s - 2 * j - 1) * a[i] + (2 * j + 1) * b[i]) / (2 * s)

        if i == 0:
            x[i, :] = np.repeat(xs, int(np.prod(ns[i + 1:])))
            if i + 1 < m:
                tile_size = int(np.prod(ns[i + 1:]))
                x[i, :] = np.tile(np.repeat(xs, tile_size), int(np.prod(ns[:i])))
        else:
            repeats = int(np.prod(ns[i + 1:]))
            tiles = int(np.prod(ns[:i]))
            x[i, :] = np.tile(np.repeat(xs, repeats), tiles)
    return x


def cvtm_1d_optimize(g_num: int, it_num: int = 50, s_num: int = 5000,
                     density_func: Optional[callable] = None) -> np.ndarray:
    if g_num <= 0:
        raise ValueError("生成元数量必须为正")
    if density_func is None:
        density_func = lambda x: 1.0


    g = np.sort(np.random.rand(g_num))

    for it in range(it_num):
        s = np.random.rand(s_num)
        sa = 0.0 - s
        sb = 2.0 - s

        g_new = np.zeros(g_num)
        w_new = np.zeros(g_num)

        for i in range(s_num):

            d0 = np.abs(s[i] - g)
            d1 = np.abs(sa[i] - g)
            d2 = np.abs(sb[i] - g)
            d_min = np.minimum(np.minimum(d0, d1), d2)
            idx = int(np.argmin(d_min))


            if d1[idx] < d0[idx] and d1[idx] < d2[idx]:
                si = sa[i]
            elif d2[idx] < d0[idx] and d2[idx] < d1[idx]:
                si = sb[i]
            else:
                si = s[i]


            rho = max(density_func(si), 1e-10)
            g_new[idx] += rho * si
            w_new[idx] += rho


        mask = w_new > 1e-14
        g[mask] = g_new[mask] / w_new[mask]

        g = np.sort(np.mod(g, 1.0))

    return g


class MolecularIntegralGrid:
    def __init__(self, n_orbitals: int = 4, grid_level: int = 3):
        self.n_orbitals = n_orbitals
        self.grid_level = grid_level

        self.zeta = np.array([1.0, 1.0, 0.8, 0.8])
        self.centers = np.array([[0.0, 0.0, 0.0],
                                  [1.4, 0.0, 0.0],
                                  [0.7, 0.7, 0.0],
                                  [0.7, -0.7, 0.0]])

    def build_3d_grid(self, n_per_dim: int = 8) -> np.ndarray:
        m = 3
        ns = np.array([n_per_dim, n_per_dim, n_per_dim])
        a = np.array([-2.0, -2.0, -2.0])
        b = np.array([3.0, 2.0, 2.0])
        c = np.array([5, 5, 5])
        grid = hypercube_grid(m, ns, a, b, c)
        return grid.T

    def slater_orbital(self, r: np.ndarray, center: np.ndarray,
                       zeta: float) -> float:
        dist = np.linalg.norm(r - center)
        norm = np.sqrt(zeta ** 3 / np.pi)
        return norm * np.exp(-zeta * dist)

    def one_electron_integral(self, i: int, j: int,
                               grid_points: np.ndarray,
                               weights: np.ndarray) -> float:
        val = 0.0
        for gp, w in zip(grid_points, weights):
            phi_i = self.slater_orbital(gp, self.centers[i], self.zeta[i])
            phi_j = self.slater_orbital(gp, self.centers[j], self.zeta[j])

            V = 0.0
            for center in self.centers[:2]:
                dist = np.linalg.norm(gp - center) + 1e-10
                V -= 1.0 / dist

            T = 0.5 * self.zeta[i] * self.zeta[j] * phi_i * phi_j
            val += w * phi_i * (T + V) * phi_j
        return val

    def two_electron_integral(self, i: int, j: int, k: int, l: int,
                               grid_points: np.ndarray,
                               weights: np.ndarray) -> float:
        val = 0.0
        for gp, w in zip(grid_points, weights):
            phi_i = self.slater_orbital(gp, self.centers[i], self.zeta[i])
            phi_j = self.slater_orbital(gp, self.centers[j], self.zeta[j])
            phi_k = self.slater_orbital(gp, self.centers[k], self.zeta[k])
            phi_l = self.slater_orbital(gp, self.centers[l], self.zeta[l])
            val += w * phi_i * phi_j * phi_k * phi_l
        return val

    def optimize_radial_grid(self, n_points: int = 16) -> np.ndarray:
        rho_func = lambda r: np.exp(-2.0 * r) if isinstance(r, (int, float)) else np.exp(-2.0 * np.array(r))
        generators = cvtm_1d_optimize(n_points, it_num=30, s_num=2000,
                                      density_func=rho_func)

        R_max = 10.0
        return generators * R_max
