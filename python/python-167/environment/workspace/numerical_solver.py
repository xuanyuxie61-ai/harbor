
import numpy as np
from typing import Tuple, List
from utils import robust_sqrt, check_numerical_singularity


class CholeskySolver:

    def __init__(self, eps: float = 1e-13):
        self.eps = eps

    def decompose(self, A: np.ndarray) -> np.ndarray:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square matrix")
        n = A.shape[0]
        L = np.zeros_like(A)
        for i in range(n):
            diag_sum = sum(L[i, k] ** 2 for k in range(i))
            val = A[i, i] - diag_sum

            L[i, i] = robust_sqrt(val, self.eps)
            for j in range(i + 1, n):
                off_sum = sum(L[j, k] * L[i, k] for k in range(i))
                L[j, i] = (A[j, i] - off_sum) / L[i, i]
        return L

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        L = self.decompose(A)
        n = L.shape[0]
        b = np.asarray(b, dtype=float).flatten()

        y = np.zeros(n)
        for i in range(n):
            y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(L[i + 1:, i], x[i + 1:])) / L[i, i]
        return x


class BlockTridiagonalSolver:

    def __init__(self, block_size: int = 3):
        self.m = block_size

    def solve(self, lower_blocks: List[np.ndarray], diag_blocks: List[np.ndarray],
              upper_blocks: List[np.ndarray], rhs: List[np.ndarray]) -> List[np.ndarray]:
        N = len(diag_blocks)
        if N == 0:
            return []

        L = [np.zeros((self.m, self.m)) for _ in range(N)]
        U = [np.zeros((self.m, self.m)) for _ in range(N)]
        y = [np.zeros(self.m) for _ in range(N)]

        L[0] = diag_blocks[0].copy()
        y[0] = rhs[0].copy()
        for i in range(1, N):

            U[i - 1] = np.linalg.solve(L[i - 1], upper_blocks[i - 1])

            L[i] = diag_blocks[i] - lower_blocks[i - 1] @ U[i - 1]

            temp = np.linalg.solve(L[i - 1], y[i - 1])
            y[i] = rhs[i] - lower_blocks[i - 1] @ temp


        x = [np.zeros(self.m) for _ in range(N)]
        x[-1] = np.linalg.solve(L[-1], y[-1])
        for i in range(N - 2, -1, -1):
            x[i] = np.linalg.solve(L[i], y[i] - U[i] @ x[i + 1])
        return x


class Radix2FFT:

    def __init__(self):
        pass

    def _bit_reverse(self, n: int, bits: int) -> int:
        rev = 0
        for i in range(bits):
            rev = (rev << 1) | ((n >> i) & 1)
        return rev

    def fft(self, x: np.ndarray) -> np.ndarray:
        N = len(x)
        if N == 0 or (N & (N - 1)) != 0:
            raise ValueError("FFT input length must be power of 2")
        bits = int(np.log2(N))
        X = np.array(x, dtype=complex)

        for i in range(N):
            j = self._bit_reverse(i, bits)
            if i < j:
                X[i], X[j] = X[j], X[i]

        length = 2
        while length <= N:
            half = length // 2
            for start in range(0, N, length):
                for k in range(half):
                    twiddle = np.exp(-2j * np.pi * k / length)
                    even = X[start + k]
                    odd = twiddle * X[start + k + half]
                    X[start + k] = even + odd
                    X[start + k + half] = even - odd
            length *= 2
        return X

    def ifft(self, X: np.ndarray) -> np.ndarray:
        x_conj = self.fft(np.conj(X))
        return np.conj(x_conj) / len(X)

    def power_spectral_density(self, signal: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        N = len(signal)
        X = self.fft(signal)
        psd = np.abs(X) ** 2 / (N * dt)
        freqs = np.fft.fftfreq(N, dt)
        return freqs, psd


class VandermondeSolver:

    def __init__(self):
        pass

    def solve(self, x_nodes: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, int]:
        x = np.asarray(x_nodes, dtype=float).flatten()
        b_arr = np.asarray(b, dtype=float).flatten()
        n = x.size
        if b_arr.size != n:
            raise ValueError("b length must match x_nodes length")

        for j in range(n - 1):
            for i in range(j + 1, n):
                if np.isclose(x[i], x[j]):
                    return np.zeros(n), 1
        y = b_arr.copy()

        for j in range(n - 1):
            for i in range(n - 1, j, -1):
                y[i] = y[i] - x[j] * y[i - 1]

        for j in range(n - 2, -1, -1):
            for i in range(j + 1, n):
                y[i] = y[i] / (x[i] - x[i - j - 1])
            for i in range(j, n - 1):
                y[i] = y[i] - y[i + 1]
        return y, 0

    def evaluate(self, x_nodes: np.ndarray, coeffs: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float)
        x_query = np.asarray(x_query, dtype=float)
        result = np.zeros_like(x_query)
        for c in reversed(coeffs):
            result = result * x_query + c
        return result


class MatrixMultiplyBenchmark:

    @staticmethod
    def multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be 2D arrays")
        if A.shape[1] != B.shape[0]:
            raise ValueError("Incompatible shapes for matrix multiplication")
        return A @ B
