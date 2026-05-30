# -*- coding: utf-8 -*-

import numpy as np


class MiddleSquareGenerator:

    def __init__(self, seed: int = 1234, d: int = 4):
        if d < 1 or d > 9:
            raise ValueError("d 必须在 [1, 9] 范围内")
        max_seed = 10 ** (2 * d)
        if not (0 <= seed < max_seed):
            raise ValueError(f"seed 必须在 [0, {max_seed}) 范围内")
        self._s = int(seed)
        self._d = int(d)
        self._mod = 10 ** (2 * d)
        self._div = 10 ** d

    def _next(self) -> int:
        r = self._s * self._s
        r = r // self._div
        r = r % self._mod
        self._s = r
        return r

    def cycle_length(self, max_iter: int = 100000) -> int:
        seen = {}
        s0 = self._s
        for i in range(max_iter):
            if self._s in seen:
                length = i - seen[self._s]
                self._s = s0
                return length
            seen[self._s] = i
            self._next()
        self._s = s0
        return -1

    def random(self) -> float:
        return self._next() / self._mod

    def randn(self) -> float:
        u1 = max(self.random(), 1e-10)
        u2 = self.random()
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        if low >= high:
            raise ValueError("low 必须小于 high")
        return low + self.random() * (high - low)

    def lognormal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        if sigma <= 0:
            raise ValueError("sigma 必须为正")
        z = self.randn()
        return np.exp(mu + sigma * z)

    def exponential(self, scale: float = 1.0) -> float:
        if scale <= 0:
            raise ValueError("scale 必须为正")
        u = max(self.random(), 1e-10)
        return -scale * np.log(1.0 - u)

    def generate_array(self, n: int, dist: str = "uniform", **kwargs) -> np.ndarray:
        if n < 0:
            raise ValueError("n 必须为非负整数")
        arr = np.zeros(n)
        for i in range(n):
            if dist == "uniform":
                arr[i] = self.uniform(kwargs.get("low", 0.0), kwargs.get("high", 1.0))
            elif dist == "normal":
                arr[i] = kwargs.get("mu", 0.0) + kwargs.get("sigma", 1.0) * self.randn()
            elif dist == "lognormal":
                arr[i] = self.lognormal(kwargs.get("mu", 0.0), kwargs.get("sigma", 1.0))
            elif dist == "exponential":
                arr[i] = self.exponential(kwargs.get("scale", 1.0))
            else:
                raise ValueError(f"不支持的分布类型: {dist}")
        return arr
