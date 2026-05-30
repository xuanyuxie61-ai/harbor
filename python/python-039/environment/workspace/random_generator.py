
import numpy as np
from typing import Tuple, List, Optional


class MiddleSquareHybrid:

    def __init__(self, seed: int = 123456789, d: int = 4):
        self.d = d
        self.modulus = 10 ** (2 * d)
        self.state = seed % self.modulus

        self.a = 1664525
        self.c = 1013904223
        self.m = 2 ** 32

    def _middle_square_step(self, s: int) -> int:
        sq = s * s

        sq_mod = sq % (10 ** (4 * self.d))
        mid = sq_mod // (10 ** self.d)
        return mid % self.modulus

    def _lcg_step(self, s: int) -> int:
        return (self.a * s + self.c) % self.m

    def next_int(self) -> int:
        self.state = self._lcg_step(self.state)
        ms = self._middle_square_step(self.state % self.modulus)
        mixed = (self.state + ms) % self.m
        return mixed

    def random(self) -> float:
        return self.next_int() / self.m

    def random_array(self, size: Tuple[int, ...]) -> np.ndarray:
        return np.array([self.random() for _ in range(int(np.prod(size)))])

    def cycle_length(self, max_steps: int = 100000) -> Tuple[int, int]:
        seen = {}
        value = self.state
        steps = 0
        while steps < max_steps:
            if value in seen:
                cycle_len = steps - seen[value]
                return cycle_len, seen[value]
            seen[value] = steps
            self.state = value
            value = self.next_int()
            steps += 1
        return -1, max_steps


class QGPEventSampler:

    def __init__(self, rng: Optional[MiddleSquareHybrid] = None):
        self.rng = rng if rng is not None else MiddleSquareHybrid()

    def sample_impact_parameter(self, b_max: float = 15.0,
                                n_samples: int = 1000) -> np.ndarray:
        samples = []
        while len(samples) < n_samples:
            b = b_max * np.sqrt(self.rng.random())
            samples.append(b)
        return np.array(samples)

    def sample_thermal_momentum(self, T: float, m: float,
                                n_samples: int = 1000) -> np.ndarray:
        samples = []

        T_eff = max(T, 1e-6)
        p_max = 10.0 * T_eff + 3.0 * m
        while len(samples) < n_samples:
            p = -T_eff * np.log(self.rng.random() + 1e-20)
            if p > p_max:
                continue

            E = np.sqrt(p ** 2 + m ** 2)
            f_target = p ** 2 * np.exp(-E / T_eff)
            f_proposal = np.exp(-p / T_eff) / T_eff
            ratio = f_target / (f_proposal + 1e-20)
            if self.rng.random() < ratio / (p_max ** 2 * T_eff + 1e-20):
                samples.append(p)
        return np.array(samples)

    def sample_azimuthal_angle(self, v2: float, n_samples: int = 1000) -> np.ndarray:
        samples = []
        v2_clip = np.clip(v2, -0.5, 0.5)
        while len(samples) < n_samples:
            phi = 2.0 * np.pi * self.rng.random()
            pdf = 1.0 + 2.0 * v2_clip * np.cos(2.0 * phi)
            if self.rng.random() < pdf / (1.0 + 2.0 * abs(v2_clip)):
                samples.append(phi)
        return np.array(samples)

    def sample_fluctuation(self, mean: float, std: float,
                           n_samples: int = 1000) -> np.ndarray:
        samples = []
        while len(samples) < n_samples:
            u1 = self.rng.random()
            u2 = self.rng.random()
            if u1 < 1e-20:
                continue
            z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            samples.append(mean + std * z0)
        return np.array(samples)
