
import numpy as np
from typing import Tuple, Optional


class QuantumRandomnessEngine:


    IA = 16807
    IM = 2147483647
    IQ = 127773
    IR = 2836
    AM = 1.0 / IM

    def __init__(self, seed: int = 1):
        if seed <= 0 or seed >= self.IM:
            raise ValueError(f"Seed must be in [1, {self.IM - 1}], got {seed}")
        self._seed = int(seed)
        self._initial_seed = int(seed)

    def reset(self) -> None:
        self._seed = self._initial_seed

    def _advance(self) -> int:
        k = self._seed // self.IQ
        self._seed = self.IA * (self._seed - k * self.IQ) - k * self.IR
        if self._seed < 0:
            self._seed += self.IM
        return self._seed

    def uniform_01(self) -> float:
        return self._advance() * self.AM

    def uniform_ab(self, a: float, b: float) -> float:
        if a >= b:
            raise ValueError(f"Invalid interval: a={a} >= b={b}")
        return a + (b - a) * self.uniform_01()

    def uniform_int(self, low: int, high: int) -> int:
        if low > high:
            raise ValueError(f"Invalid range: low={low} > high={high}")
        return low + int(self.uniform_01() * (high - low + 1)) % (high - low + 1)

    def uniform_disk(self) -> complex:
        u = self.uniform_01()
        theta = 2.0 * np.pi * self.uniform_01()
        r = np.sqrt(u)
        return complex(r * np.cos(theta), r * np.sin(theta))

    def uniform_sphere_nd(self, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError(f"Dimension n must be positive, got {n}")


        np.random.seed(self._seed % (2**32))
        g = np.random.randn(n)
        self._advance()
        norm = np.linalg.norm(g)
        if norm < 1e-15:
            g[0] = 1.0
            norm = 1.0
        return g / norm

    def power_mod(self, a: int, n: int, m: int) -> int:
        if m <= 0:
            raise ValueError("Modulus m must be positive")
        if n < 0:
            raise ValueError("Exponent n must be non-negative")
        result = 1 % m
        base = a % m
        exp = n
        while exp > 0:
            if exp & 1:
                result = (result * base) % m
            base = (base * base) % m
            exp >>= 1
        return result

    def extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = self.extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return (g, x, y)

    def mod_inverse(self, a: int, m: int) -> Optional[int]:
        g, x, _ = self.extended_gcd(a % m, m)
        if g != 1:
            return None
        return x % m

    def jump_ahead(self, n_steps: int) -> None:
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        an = self.power_mod(self.IA, n_steps, self.IM)
        self._seed = (an * self._seed) % self.IM
        if self._seed == 0:
            self._seed = self._initial_seed

    def generate_sequence(self, length: int) -> np.ndarray:
        if length < 0:
            raise ValueError("Length must be non-negative")
        return np.array([self.uniform_01() for _ in range(length)], dtype=np.float64)


def box_muller_transform(u1: float, u2: float) -> Tuple[float, float]:
    eps = 1e-15
    u1 = max(u1, eps)
    u1 = min(u1, 1.0 - eps)
    magnitude = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2
    z1 = magnitude * np.cos(angle)
    z2 = magnitude * np.sin(angle)
    return z1, z2


def quantum_random_hermitian(n: int, engine: QuantumRandomnessEngine) -> np.ndarray:
    if n <= 0:
        raise ValueError("Matrix dimension must be positive")
    A = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            u1, u2 = engine.uniform_01(), engine.uniform_01()
            z1, z2 = box_muller_transform(u1, u2)
            A[i, j] = complex(z1, z2) / np.sqrt(2.0)
    H = (A + A.conj().T) / 2.0
    return H


def quantum_random_unitary(n: int, engine: QuantumRandomnessEngine) -> np.ndarray:
    if n <= 0:
        raise ValueError("Matrix dimension must be positive")
    A = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            u1, u2 = engine.uniform_01(), engine.uniform_01()
            z1, z2 = box_muller_transform(u1, u2)
            A[i, j] = complex(z1, z2) / np.sqrt(2.0)
    Q, R = np.linalg.qr(A)

    D = np.diag(np.diag(R))
    D = np.diag(np.exp(1j * np.angle(np.diag(D))))
    U = Q @ D
    return U
