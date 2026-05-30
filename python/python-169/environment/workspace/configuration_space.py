
import numpy as np
from typing import List, Tuple, Optional, Callable






def simplex_grid_size(m: int, n: int) -> int:
    if m < 0 or n < 0:
        return 0
    ng = 1
    for i in range(1, m + 1):
        ng = ng * (n + i) // i
    return ng


def comp_next_grlex(kc: int, xc: np.ndarray) -> Optional[np.ndarray]:
    xc = np.asarray(xc, dtype=int)
    if xc.size == 0:
        return None

    for i in range(xc.size - 2, -1, -1):
        if xc[i] > 0:
            xc[i] -= 1
            xc[i + 1] += 1
            t = np.sum(xc[i + 2:])
            xc[-1] += t
            if i + 2 < xc.size:
                xc[i + 2:-1] = 0
            return xc.copy()
    return None


def simplex_grid_index_all(m: int, n: int) -> np.ndarray:
    ng = simplex_grid_size(m, n)
    g = np.zeros((ng, m + 1), dtype=int)
    g[0, 0] = n
    for idx in range(1, ng):
        prev = g[idx - 1].copy()
        nxt = comp_next_grlex(m + 1, prev)
        if nxt is None:
            break
        g[idx] = nxt
    return g


def simplex_grid_index_to_point(m: int, n: int, g: np.ndarray,
                                 vertices: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    if n == 0:
        return vertices[0].copy()
    return (g @ vertices) / float(n)






def pdf_to_histogram(pdf_func: Callable, n_bins: int,
                     a: float = -1.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(a, b, n_bins + 1)
    b_l = edges[:-1]
    b_r = edges[1:]
    mids = 0.5 * (b_l + b_r)
    b_p = np.array([pdf_func(m) for m in mids], dtype=float)

    b_p = np.maximum(b_p, 0.0)
    widths = b_r - b_l
    total = np.sum(b_p * widths)
    if total > 1e-14:
        b_p = b_p / total
    else:
        b_p = np.ones(n_bins) / (widths.sum())
    return b_l, b_r, b_p


def histogram_to_cdf(b_l: np.ndarray, b_r: np.ndarray,
                     b_p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    widths = b_r - b_l
    masses = b_p * widths
    c_y = np.zeros(b_p.size + 1, dtype=float)
    c_y[1:] = np.cumsum(masses)
    c_x = np.zeros(b_p.size + 1, dtype=float)
    c_x[0] = b_l[0]
    c_x[1:] = b_r

    if c_y[-1] < 1e-14:
        c_y[-1] = 1.0
    else:
        c_y = c_y / c_y[-1]
    return c_x, c_y


def cdf_sample(c_x: np.ndarray, c_y: np.ndarray, n_samples: int,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(seed=42)
    rvals = rng.random(n_samples)
    samples = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        r = rvals[i]

        left = 0
        right = c_y.size - 1
        while left < right:
            mid = (left + right) // 2
            if c_y[mid] <= r:
                left = mid
            else:
                right = mid
            if right - left <= 1:
                break

        if left >= c_y.size - 1:
            left = c_y.size - 2
        if c_y[left + 1] - c_y[left] < 1e-14:
            samples[i] = c_x[left]
        else:

            t = (r - c_y[left]) / (c_y[left + 1] - c_y[left])
            t = np.clip(t, 0.0, 1.0)
            samples[i] = c_x[left] + t * (c_x[left + 1] - c_x[left])
    return samples






class ConfigurationSampler:

    def __init__(self, q_min: np.ndarray, q_max: np.ndarray,
                 n_dof: int = 7, seed: int = 42):
        self.q_min = np.asarray(q_min, dtype=float).reshape(-1)
        self.q_max = np.asarray(q_max, dtype=float).reshape(-1)
        self.n_dof = self.q_min.size
        self.rng = np.random.default_rng(seed=seed)

    def uniform_random(self, n_samples: int) -> np.ndarray:
        return self.rng.uniform(self.q_min, self.q_max, size=(n_samples, self.n_dof))

    def simplex_grid_sample(self, n_per_dim: int = 4) -> np.ndarray:

        grids = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(self.n_dof)]

        from itertools import product
        total = n_per_dim ** self.n_dof
        if total > 5000:

            samples = self.rng.random(size=(5000, self.n_dof))
        else:
            samples = np.array(list(product(*grids)), dtype=float)

        q_samples = self.q_min + samples * (self.q_max - self.q_min)
        return q_samples

    def pdf_weighted_sample(self, pdf_func: Callable, n_bins: int,
                            n_samples: int, dim: int = 0) -> np.ndarray:
        dim = int(dim) % self.n_dof
        b_l, b_r, b_p = pdf_to_histogram(
            pdf_func, n_bins,
            a=float(self.q_min[dim]), b=float(self.q_max[dim])
        )
        c_x, c_y = histogram_to_cdf(b_l, b_r, b_p)
        weighted = cdf_sample(c_x, c_y, n_samples, rng=self.rng)
        others = self.rng.uniform(self.q_min, self.q_max, size=(n_samples, self.n_dof))
        others[:, dim] = weighted
        return others

    def gaussian_mixture_sample(self, n_samples: int,
                                 means: List[np.ndarray],
                                 covs: List[np.ndarray],
                                 weights: Optional[np.ndarray] = None) -> np.ndarray:
        if weights is None:
            weights = np.ones(len(means)) / len(means)
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        samples = []
        for _ in range(n_samples):
            k = self.rng.choice(len(means), p=weights)
            mean = np.asarray(means[k])
            cov = np.asarray(covs[k])

            cov = cov + 1e-4 * np.eye(self.n_dof)
            s = self.rng.multivariate_normal(mean, cov)
            s = np.clip(s, self.q_min, self.q_max)
            samples.append(s)
        return np.array(samples, dtype=float)
