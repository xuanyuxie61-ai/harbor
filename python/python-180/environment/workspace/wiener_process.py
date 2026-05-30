
import numpy as np
from typing import Tuple, Optional


class LEcuyerRNG:

    def __init__(self, seed1: int = 12345, seed2: int = 67890):
        if not isinstance(seed1, int) or not isinstance(seed2, int):
            raise TypeError("Seeds must be integers")
        self.m1 = 2147483647
        self.a1 = 40014
        self.m2 = 2145483479
        self.a2 = 40692

        self.x1 = max(1, abs(seed1) % self.m1)
        self.x2 = max(1, abs(seed2) % self.m2)

    def _advance(self) -> float:
        self.x1 = (self.a1 * self.x1) % self.m1
        self.x2 = (self.a2 * self.x2) % self.m2
        y = (self.x1 - self.x2) % self.m1
        if y <= 0:
            y += self.m1
        return y / self.m1

    def uniform(self, size: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        if size is None:
            return np.array(self._advance(), dtype=np.float64)
        arr = np.empty(size, dtype=np.float64)
        for idx in np.ndindex(size):
            arr[idx] = self._advance()
        return arr

    def gaussian(self, size: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        if size is None:
            size = (1,)
            squeeze = True
        else:
            squeeze = False

        n = np.prod(size)
        u1 = self.uniform(size=(n,))
        u2 = self.uniform(size=(n,))

        eps = np.finfo(np.float64).eps
        u1 = np.clip(u1, eps, 1.0 - eps)
        z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        result = z.reshape(size)
        if squeeze:
            result = result.item()
        return result


class QWienerProcess:

    def __init__(self,
                 spatial_grid: np.ndarray,
                 n_modes: int,
                 alpha: float = 1.0,
                 sigma: float = 1.0,
                 rng: Optional[LEcuyerRNG] = None,
                 use_antithetic: bool = False):
        if spatial_grid.ndim != 1:
            raise ValueError("spatial_grid must be 1D")
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if alpha <= 0.5:
            raise ValueError("alpha must be > 0.5 for trace-class covariance")

        self.x = spatial_grid.copy()
        self.L = float(spatial_grid[-1] - spatial_grid[0])
        if self.L <= 0:
            raise ValueError("Domain length must be positive")
        self.n_modes = n_modes
        self.alpha = alpha
        self.sigma = sigma
        self.rng = rng if rng is not None else LEcuyerRNG()
        self.use_antithetic = use_antithetic


        k = np.arange(1, n_modes + 1, dtype=np.float64)
        self.qk = (sigma ** 2) * np.power(k, -2.0 * alpha)

        q_min = np.finfo(np.float64).eps * 10.0
        self.qk = np.where(self.qk < q_min, q_min, self.qk)


        nx = len(spatial_grid)
        self.eigenfuncs = np.zeros((n_modes, nx), dtype=np.float64)
        for idx, kk in enumerate(k):
            self.eigenfuncs[idx, :] = np.sqrt(2.0 / self.L) * np.sin(kk * np.pi * spatial_grid / self.L)


        self._cached_normal: Optional[np.ndarray] = None
        self._cache_valid = False

    def increment(self, dt: float) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be positive")

        if self.use_antithetic and self._cache_valid:
            dbeta = -self._cached_normal
            self._cache_valid = False
        else:
            dbeta = np.sqrt(dt) * self.rng.gaussian(size=(self.n_modes,))
            if self.use_antithetic:
                self._cached_normal = dbeta.copy()
                self._cache_valid = True


        coeffs = np.sqrt(self.qk) * dbeta
        dW = self.eigenfuncs.T @ coeffs
        return dW

    def strong_error_estimate(self, dt: float, p: int = 2) -> float:
        from math import gamma
        if p < 1:
            raise ValueError("p must be >= 1")
        Cp = (gamma(p + 1) / np.power(2.0, p / 2.0) / gamma(p / 2.0 + 1.0)) ** (1.0 / p)
        return Cp * np.sqrt(dt)

    def spectral_truncation_error(self, t: float) -> float:
        if self.alpha <= 0.5:
            return np.inf
        tail = np.power(self.n_modes, 1.0 - 2.0 * self.alpha) / (2.0 * self.alpha - 1.0)
        return t * (self.sigma ** 2) * tail
