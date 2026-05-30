
import numpy as np
import math
from numerics_core import clenshaw_curtis_nodes_weights
from typing import Tuple, List, Dict, Generator






def comp_next(n: int, k: int) -> Generator[np.ndarray, None, None]:
    if n < 0 or k < 1:
        return
    if k == 1:
        yield np.array([n])
        return


    def _recurse(remaining: int, parts: int, prefix: List[int]):
        if parts == 1:
            yield np.array(prefix + [remaining])
            return
        for val in range(remaining + 1):
            yield from _recurse(remaining - val, parts - 1, prefix + [val])

    yield from _recurse(n, k, [])


def comp_all(n: int, k: int) -> np.ndarray:
    return np.array(list(comp_next(n, k)))






def cc_nested_rule(level: int) -> Tuple[np.ndarray, np.ndarray]:
    if level < 0:
        raise ValueError("level must be non-negative")
    if level == 0:
        return np.array([0.0]), np.array([2.0])
    n = 2 ** level
    return clenshaw_curtis_nodes_weights(n)


def trapezoidal_rule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be >= 1")
    x = np.linspace(-1.0, 1.0, n + 1)
    w = np.full(n + 1, 2.0 / n)
    w[0] = 1.0 / n
    w[-1] = 1.0 / n
    return x, w


def newton_cotes_rule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        return trapezoidal_rule(1)
    x = np.linspace(-1.0, 1.0, n + 1)
    w = np.zeros(n + 1)
    for j in range(n + 1):


        coeff = np.zeros(n + 1)
        coeff[0] = 1.0
        for m in range(n + 1):
            if m == j:
                continue

            p = np.array([1.0, -x[m]]) / (x[j] - x[m])
            coeff = np.convolve(coeff, p)


        for k in range(len(coeff)):
            w[j] += coeff[k] * (1.0 - (-1.0) ** (k + 1)) / (k + 1.0)
    return x, w






class SparseGrid:

    def __init__(self, dim: int, level: int, rule: str = "cc"):
        self.dim = dim
        self.level = max(level, dim)
        self.rule = rule
        self.points = None
        self.weights = None
        self._construct()

    def _oned_rule(self, lev: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.rule == "cc":
            return cc_nested_rule(lev)
        elif self.rule == "nc":
            n = max(1, 2 ** lev)
            return newton_cotes_rule(n)
        elif self.rule == "trap":
            n = max(1, 2 ** lev)
            return trapezoidal_rule(n)
        else:
            raise ValueError(f"Unknown rule: {self.rule}")

    def _construct(self):
        from itertools import product
        d = self.dim
        q = self.level

        all_points = []
        all_weights = []


        for s in range(d, q + 1):
            for lev_vec in comp_next(s, d):

                coeff = (-1) ** (q - s) * math.comb(d - 1, q - s)
                if coeff == 0:
                    continue


                nodes_list = []
                weights_list = []
                for l in lev_vec:
                    x, w = self._oned_rule(int(l))
                    nodes_list.append(x)
                    weights_list.append(w)


                for idx in product(*[range(len(nl)) for nl in nodes_list]):
                    pt = np.array([nodes_list[i][idx[i]] for i in range(d)])
                    w = coeff * np.prod([weights_list[i][idx[i]] for i in range(d)])
                    all_points.append(pt)
                    all_weights.append(w)

        if not all_points:
            self.points = np.zeros((0, d))
            self.weights = np.array([])
            return


        pts = np.array(all_points)
        wts = np.array(all_weights)


        unique_dict = {}
        for i in range(pts.shape[0]):
            key = tuple(np.round(pts[i], decimals=12))
            if key in unique_dict:
                unique_dict[key] += wts[i]
            else:
                unique_dict[key] = wts[i]

        self.points = np.array([np.array(k) for k in unique_dict.keys()])
        self.weights = np.array(list(unique_dict.values()))

    def integrate(self, f: callable) -> float:
        if self.points.shape[0] == 0:
            return 0.0
        vals = np.array([f(p) for p in self.points])
        return float(np.dot(self.weights, vals))

    def size(self) -> int:
        return self.points.shape[0]






class ParameterizedUQ:

    def __init__(self, dim: int = 4, level: int = 3):
        self.dim = dim
        self.level = level
        self.grid = SparseGrid(dim, level, rule="cc")

    def map_to_physical(self, p_norm: np.ndarray) -> np.ndarray:
        p = np.zeros(self.dim)

        p[0] = 1.25 + 0.75 * p_norm[0]

        p[1] = 10.0 ** (-1.0 + 1.1 * (p_norm[1] + 1.0) / 2.0)

        p[2] = 1.25 + 0.75 * p_norm[2]

        p[3] = 1.0 + 1.0 * p_norm[3]
        return p

    def estimate_statistics(self, model_evaluator: callable) -> Dict[str, float]:
        vals = []
        for pt in self.grid.points:
            try:
                v = float(model_evaluator(pt))
                if not np.isfinite(v):
                    v = np.nan
            except Exception:
                v = np.nan
            vals.append(v)
        vals = np.array(vals)

        w = self.grid.weights

        valid = np.isfinite(vals)
        if not np.any(valid):
            return {
                "mean": np.nan, "variance": np.nan, "std": np.nan,
                "min": np.nan, "max": np.nan, "grid_size": int(self.grid.size())
            }

        vals_valid = vals[valid]
        w_valid = w[valid]
        w_sum = np.sum(w_valid)
        if abs(w_sum) < 1e-15:
            w_sum = 1.0

        mean = np.dot(w_valid, vals_valid) / w_sum
        variance = np.dot(w_valid, (vals_valid - mean) ** 2) / w_sum

        return {
            "mean": float(mean),
            "variance": float(variance),
            "std": float(np.sqrt(max(variance, 0.0))),
            "min": float(np.min(vals_valid)),
            "max": float(np.max(vals_valid)),
            "grid_size": int(self.grid.size())
        }


if __name__ == "__main__":

    sg = SparseGrid(dim=2, level=3, rule="cc")
    print(f"Sparse grid 2D level 3: {sg.size()} points")


    f_test = lambda p: np.exp(-(p[0]**2 + p[1]**2))
    val = sg.integrate(f_test)
    print("Integral approx:", val)


    combos = comp_all(4, 3)
    print("Combinations of 4 into 3 parts:", len(combos))
