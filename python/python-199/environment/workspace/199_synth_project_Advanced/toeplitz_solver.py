
import math
from typing import List, Tuple, Optional


class R8LTTSolver:

    def __init__(self, first_col: List[float]):
        if not first_col:
            raise ValueError("First column must not be empty.")
        self.n = len(first_col)
        self.t = list(first_col)

        if abs(self.t[0]) < 1e-15:
            raise ValueError(
                f"Diagonal element t_0 = {self.t[0]} is too close to zero; "
                "matrix is singular or ill-conditioned."
            )

    def matvec(self, x: List[float]) -> List[float]:
        if len(x) != self.n:
            raise ValueError(f"Dimension mismatch: x has {len(x)}, expected {self.n}")
        y = [0.0] * self.n
        for i in range(self.n):
            s = 0.0
            for j in range(i + 1):
                s += self.t[i - j] * x[j]
            y[i] = s
        return y

    def solve(self, b: List[float]) -> List[float]:
        if len(b) != self.n:
            raise ValueError(f"Dimension mismatch: b has {len(b)}, expected {self.n}")
        x = [0.0] * self.n
        x[0] = b[0] / self.t[0]
        for i in range(1, self.n):
            s = 0.0
            for j in range(i):
                s += self.t[i - j] * x[j]
            x[i] = (b[i] - s) / self.t[0]
        return x

    def determinant(self) -> float:
        return self.t[0] ** self.n

    def inverse(self) -> 'R8LTTSolver':
        s = [0.0] * self.n
        s[0] = 1.0 / self.t[0]
        for i in range(1, self.n):
            accum = 0.0
            for k in range(i):
                accum += self.t[i - k] * s[k]
            s[i] = -accum / self.t[0]
        return R8LTTSolver(s)

    def condition_estimate(self) -> float:
        norm_t = sum(abs(v) for v in self.t)
        inv = self.inverse()
        norm_inv = sum(abs(v) for v in inv.t)
        return norm_t * norm_inv


def build_prefix_toeplitz(n: int, decay: float = 0.5) -> R8LTTSolver:
    if n <= 0:
        raise ValueError("n must be positive.")
    first_col = [decay ** i for i in range(n)]
    return R8LTTSolver(first_col)


def toeplitz_transform_keys(keys: List[float], decay: float = 0.9) -> List[float]:
    n = len(keys)
    if n == 0:
        return []
    mu = sum(keys) / n
    centered = [k - mu for k in keys]
    solver = build_prefix_toeplitz(n, decay)
    return solver.matvec(centered)
