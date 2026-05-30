import math
import numpy as np


class WichmannHill:

    def __init__(self, s1: int = 12345, s2: int = 67890, s3: int = 13579):
        if not (1 <= s1 <= 30268):
            raise ValueError(f"Wichmann-Hill seed s1 must be in [1, 30268], got {s1}")
        if not (1 <= s2 <= 30306):
            raise ValueError(f"Wichmann-Hill seed s2 must be in [1, 30306], got {s2}")
        if not (1 <= s3 <= 30322):
            raise ValueError(f"Wichmann-Hill seed s3 must be in [1, 30322], got {s3}")
        self.s1 = int(s1)
        self.s2 = int(s2)
        self.s3 = int(s3)

    def uniform(self) -> float:
        self.s1 = (171 * self.s1) % 30269
        self.s2 = (172 * self.s2) % 30307
        self.s3 = (170 * self.s3) % 30323
        r = (self.s1 / 30269.0 + self.s2 / 30307.0 + self.s3 / 30323.0) % 1.0

        if r <= 0.0:
            r = 1e-15
        if r >= 1.0:
            r = 1.0 - 1e-15
        return r

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.uniform() for _ in range(n)], dtype=float)

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        u1 = self.uniform()
        u2 = self.uniform()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def normals(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
        out = np.empty(n, dtype=float)
        for i in range(0, n, 2):
            u1 = self.uniform()
            u2 = self.uniform()
            mag = sigma * math.sqrt(-2.0 * math.log(u1))
            z0 = mag * math.cos(2.0 * math.pi * u2) + mu
            z1 = mag * math.sin(2.0 * math.pi * u2) + mu
            out[i] = z0
            if i + 1 < n:
                out[i + 1] = z1
        return out


class LEcuyer:

    def __init__(self, s1: int = 12345, s2: int = 67890):
        if not (1 <= s1 <= 2147483562):
            raise ValueError(f"L'Ecuyer seed s1 must be in [1, 2147483562], got {s1}")
        if not (1 <= s2 <= 2147483398):
            raise ValueError(f"L'Ecuyer seed s2 must be in [1, 2147483398], got {s2}")
        self.s1 = int(s1)
        self.s2 = int(s2)

    def uniform(self) -> float:
        k = self.s1 // 53668
        self.s1 = 40014 * (self.s1 - k * 53668) - k * 12211
        if self.s1 < 0:
            self.s1 += 2147483563

        k = self.s2 // 52774
        self.s2 = 40692 * (self.s2 - k * 52774) - k * 3791
        if self.s2 < 0:
            self.s2 += 2147483399

        z = self.s1 - self.s2
        if z < 1:
            z += 2147483562

        r = z / 2147483563.0
        if r <= 0.0:
            r = 1e-15
        if r >= 1.0:
            r = 1.0 - 1e-15
        return r

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.uniform() for _ in range(n)], dtype=float)

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        u1 = self.uniform()
        u2 = self.uniform()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def normals(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
        out = np.empty(n, dtype=float)
        for i in range(0, n, 2):
            u1 = self.uniform()
            u2 = self.uniform()
            mag = sigma * math.sqrt(-2.0 * math.log(u1))
            z0 = mag * math.cos(2.0 * math.pi * u2) + mu
            z1 = mag * math.sin(2.0 * math.pi * u2) + mu
            out[i] = z0
            if i + 1 < n:
                out[i + 1] = z1
        return out
