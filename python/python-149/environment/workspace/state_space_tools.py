
import numpy as np
from typing import Tuple, Optional, Callable, List


def metric_tensor(x: np.ndarray, metric_type: str = "fisher") -> np.ndarray:
    d = len(np.atleast_1d(x))
    x_safe = np.clip(np.abs(np.atleast_1d(x)), 1e-6, 1e6)

    if metric_type == "euclidean":
        return np.eye(d)
    elif metric_type == "fisher":
        return np.diag(1.0 / x_safe)
    elif metric_type == "anisotropic":
        A = np.eye(d)
        if d >= 2:
            A[0, 0] = 1.0
            A[1, 1] = 10.0
        return A
    else:
        return np.eye(d)


def cvt_lloyd_iterate(
    generators: np.ndarray,
    n_samples: int = 5000,
    n_iter: int = 10,
    metric_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(seed=42)
    if metric_fn is None:
        metric_fn = lambda x: np.eye(len(np.atleast_1d(x)))

    n, d = generators.shape

    if bounds is None:
        xmin = np.zeros(d)
        xmax = np.ones(d)
    else:
        xmin, xmax = bounds

    for it in range(n_iter):

        samples = rng.uniform(0, 1, (n_samples, d))
        for dd in range(d):
            samples[:, dd] = xmin[dd] + samples[:, dd] * (xmax[dd] - xmin[dd])


        labels = np.zeros(n_samples, dtype=int)
        for s in range(n_samples):
            best_dist = np.inf
            best_g = 0
            xs = samples[s, :]
            for g in range(n):
                zg = generators[g, :]
                mid = 0.5 * (xs + zg)
                A = metric_fn(mid)
                diff = xs - zg

                try:
                    dist2 = diff @ A @ diff
                except Exception:
                    dist2 = np.sum(diff ** 2)
                if dist2 < best_dist:
                    best_dist = dist2
                    best_g = g
            labels[s] = best_g


        new_generators = np.zeros_like(generators)
        counts = np.zeros(n)
        for s in range(n_samples):
            g = labels[s]
            new_generators[g, :] += samples[s, :]
            counts[g] += 1

        for g in range(n):
            if counts[g] > 0:
                new_generators[g, :] /= counts[g]
            else:

                new_generators[g, :] = xmin + rng.uniform(0, 1, d) * (xmax - xmin)

        generators = new_generators.copy()

    return generators


def state_to_index(
    x: np.ndarray,
    generators: np.ndarray,
    metric_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> int:
    if metric_fn is None:
        metric_fn = lambda x: np.eye(len(np.atleast_1d(x)))

    n = generators.shape[0]
    best_dist = np.inf
    best_idx = 0

    for i in range(n):
        diff = x - generators[i, :]
        mid = 0.5 * (x + generators[i, :])
        A = metric_fn(mid)
        try:
            dist2 = float(diff @ A @ diff)
        except Exception:
            dist2 = float(np.sum(diff ** 2))
        if dist2 < best_dist:
            best_dist = dist2
            best_idx = i

    return best_idx






class StateEncoder:

    def __init__(self, n_states: int, rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng(seed=42)
        self.n_states = n_states

        self.perm = rng.permutation(n_states)
        self.inv_perm = np.argsort(self.perm)

    def encode(self, idx: int) -> int:
        if not (0 <= idx < self.n_states):
            idx = idx % self.n_states
        return int(self.perm[idx])

    def decode(self, code_idx: int) -> int:
        if not (0 <= code_idx < self.n_states):
            code_idx = code_idx % self.n_states
        return int(self.inv_perm[code_idx])






def serialize_state_trajectory(
    t: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    metadata: Optional[dict] = None,
) -> dict:
    data = {
        "version": "1.0",
        "n_steps": len(t),
        "dim": y.shape[1] if y.ndim > 1 else 1,
        "time": t.tolist(),
        "state": y.tolist(),
        "control": u.tolist(),
    }
    if metadata is not None:
        data["metadata"] = metadata
    return data


def deserialize_state_trajectory(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.array(data["time"])
    y = np.array(data["state"])
    u = np.array(data["control"])
    return t, y, u
