# -*- coding: utf-8 -*-

import numpy as np
from typing import Optional, Tuple


class LaplacianStencils:

    @staticmethod
    def laplacian5_2d(u: np.ndarray, h: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if u.ndim != 2:
            raise ValueError("u 必须是二维数组")
        ny, nx = u.shape
        Lu = np.zeros_like(u)

        Lu[1:-1, 1:-1] = (
            u[2:, 1:-1] + u[:-2, 1:-1] +
            u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
        ) / (h * h)
        if mask is not None:
            Lu *= mask
        return Lu

    @staticmethod
    def laplacian5_torus(u: np.ndarray, h: float) -> np.ndarray:
        if u.ndim != 2:
            raise ValueError("u 必须是二维数组")
        ny, nx = u.shape
        Lu = np.zeros_like(u)

        uxp = np.roll(u, shift=-1, axis=1)
        uxm = np.roll(u, shift=1, axis=1)
        uyp = np.roll(u, shift=-1, axis=0)
        uym = np.roll(u, shift=1, axis=0)
        Lu = (uxp + uxm + uyp + uym - 4.0 * u) / (h * h)
        return Lu

    @staticmethod
    def laplacian9_torus(u: np.ndarray, h: float) -> np.ndarray:
        if u.ndim != 2:
            raise ValueError("u 必须是二维数组")

        uxp = np.roll(u, shift=-1, axis=1)
        uxm = np.roll(u, shift=1, axis=1)
        uyp = np.roll(u, shift=-1, axis=0)
        uym = np.roll(u, shift=1, axis=0)

        uxpyp = np.roll(np.roll(u, shift=-1, axis=1), shift=-1, axis=0)
        uxmyp = np.roll(np.roll(u, shift=1, axis=1), shift=-1, axis=0)
        uxpym = np.roll(np.roll(u, shift=-1, axis=1), shift=1, axis=0)
        uxmym = np.roll(np.roll(u, shift=1, axis=1), shift=1, axis=0)
        Lu = (-20.0 * u
              + 4.0 * (uxp + uxm + uyp + uym)
              + 1.0 * (uxpyp + uxmyp + uxpym + uxmym)) / (6.0 * h * h)
        return Lu

    @staticmethod
    def laplacian3_uneven_1d(u: np.ndarray, x: np.ndarray) -> np.ndarray:
        if len(u) != len(x):
            raise ValueError("u 与 x 长度不匹配")
        n = len(u)
        Lu = np.zeros_like(u)
        for i in range(1, n - 1):
            hL = x[i] - x[i - 1]
            hR = x[i + 1] - x[i]
            if hL <= 0 or hR <= 0:
                raise ValueError("网格坐标必须严格递增")
            denom = hL * hR * (hL + hR)
            Lu[i] = 2.0 * (hR * u[i - 1] - (hL + hR) * u[i] + hL * u[i + 1]) / denom

        return Lu


class BandedMatrixSolver:

    def __init__(self, n: int, ml: int, mu: int):
        self.n = n
        self.ml = ml
        self.mu = mu
        self.lda = ml + mu + 1
        self._A_band = None
        self._factorized = False

    def from_dense(self, A_dense: np.ndarray):
        if A_dense.shape != (self.n, self.n):
            raise ValueError("稠密矩阵维度不匹配")
        self._A_band = np.zeros((self.lda, self.n), dtype=float)
        for j in range(self.n):
            for i in range(max(0, j - self.mu), min(self.n, j + self.ml + 1)):
                self._A_band[self.mu + i - j, j] = A_dense[i, j]
        self._factorized = False

    def set_element(self, i: int, j: int, value: float):
        if abs(i - j) > self.mu and abs(i - j) > self.ml:
            raise ValueError("(i,j) 超出带宽范围")
        if self._A_band is None:
            self._A_band = np.zeros((self.lda, self.n), dtype=float)
        self._A_band[self.mu + i - j, j] = value

    def factorize_np(self):
        if self._A_band is None:
            raise ValueError("矩阵未初始化")
        A = self._A_band.copy()
        n = self.n
        ml = self.ml
        mu = self.mu
        lda = self.lda

        for k in range(n - 1):

            pivot = A[mu, k]
            if abs(pivot) < 1e-30:
                raise ValueError(f"零主元出现在第 {k} 行，LU 分解失败")

            n_rows = min(ml, n - k - 1)
            for i in range(n_rows):
                A[mu + i + 1, k] /= pivot

            n_cols = min(mu + ml, n - k - 1)
            for j in range(n_cols):
                factor = A[mu - j - 1, k + j + 1]
                for i in range(n_rows):
                    A[mu + i - j, k + j + 1] -= A[mu + i + 1, k] * factor
        self._A_band = A
        self._factorized = True

    def solve(self, b: np.ndarray) -> np.ndarray:
        if not self._factorized:
            raise ValueError("矩阵未分解，请先调用 factorize_np()")
        A = self._A_band
        n = self.n
        ml = self.ml
        mu = self.mu
        x = b.copy().astype(float)


        for k in range(n - 1):
            n_rows = min(ml, n - k - 1)
            for i in range(n_rows):
                x[k + i + 1] -= A[mu + i + 1, k] * x[k]


        for k in range(n - 1, -1, -1):
            pivot = A[mu, k]
            if abs(pivot) < 1e-30:
                raise ValueError("回代时遇到零主元")
            x[k] /= pivot
            n_rows = min(mu, k)
            for i in range(n_rows):
                x[k - i - 1] -= A[mu - i - 1, k] * x[k]
        return x

    def matvec(self, x: np.ndarray) -> np.ndarray:
        if self._A_band is None:
            raise ValueError("矩阵未初始化")
        A = self._A_band
        n = self.n
        ml = self.ml
        mu = self.mu
        b = np.zeros(n, dtype=float)
        x = np.asarray(x)
        for i in range(n):
            j_min = max(0, i - ml)
            j_max = min(n - 1, i + mu)
            for j in range(j_min, j_max + 1):
                b[i] += A[mu + i - j, j] * x[j]
        return b


class BorderedBandedSolver:

    def __init__(self, n1: int, n2: int, ml: int, mu: int):
        self.n1 = n1
        self.n2 = n2
        self.ml = ml
        self.mu = mu
        self._A1_solver = BandedMatrixSolver(n1, ml, mu)
        self._A2 = None
        self._A3 = None
        self._A4 = None
        self._S_factor = None

    def set_blocks(self, A1_band: np.ndarray, A2: np.ndarray, A3: np.ndarray, A4: np.ndarray):
        self._A1_solver._A_band = A1_band.copy()
        self._A1_solver._factorized = False
        if A2.shape != (self.n1, self.n2):
            raise ValueError("A2 维度错误")
        if A3.shape != (self.n2, self.n1):
            raise ValueError("A3 维度错误")
        if A4.shape != (self.n2, self.n2):
            raise ValueError("A4 维度错误")
        self._A2 = A2.copy()
        self._A3 = A3.copy()
        self._A4 = A4.copy()

    def factorize(self):
        self._A1_solver.factorize_np()

        X = np.zeros_like(self._A2)
        for j in range(self.n2):
            rhs = -self._A2[:, j]
            X[:, j] = self._A1_solver.solve(rhs)
        self._A2 = X

        S = self._A4 + self._A3 @ self._A2

        self._S_factor = self._dense_lu_factor(S)

    @staticmethod
    def _dense_lu_factor(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = A.shape[0]
        LU = A.copy()
        piv = np.arange(n)
        for k in range(n):

            max_row = np.argmax(np.abs(LU[k:, k])) + k
            if abs(LU[max_row, k]) < 1e-30:
                raise ValueError("稠密矩阵 LU 分解遇到奇异主元")
            if max_row != k:
                LU[[k, max_row], :] = LU[[max_row, k], :]
                piv[[k, max_row]] = piv[[max_row, k]]
            for i in range(k + 1, n):
                LU[i, k] /= LU[k, k]
                for j in range(k + 1, n):
                    LU[i, j] -= LU[i, k] * LU[k, j]
        return LU, piv

    @staticmethod
    def _dense_lu_solve(LU: np.ndarray, piv: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = LU.shape[0]
        x = b[piv].copy().astype(float)

        for i in range(1, n):
            for j in range(i):
                x[i] -= LU[i, j] * x[j]

        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                x[i] -= LU[i, j] * x[j]
            x[i] /= LU[i, i]
        return x

    def solve(self, b: np.ndarray) -> np.ndarray:
        b1 = b[:self.n1]
        b2 = b[self.n1:]
        x1 = self._A1_solver.solve(b1)
        b2p = b2 - self._A3 @ x1
        x2 = self._dense_lu_solve(self._S_factor[0], self._S_factor[1], b2p)
        x1 = x1 + self._A2 @ x2
        return np.concatenate([x1, x2])


class HelmholtzSolver:

    def __init__(self, nx: int, ny: int, Lx: float, Ly: float,
                 k0: float, n_profile: np.ndarray,
                 boundary_value: float = 0.0,
                 damping: float = 0.05):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.hx = Lx / (nx - 1)
        self.hy = Ly / (ny - 1)
        self.k0 = k0
        self.n_profile = n_profile.copy()
        self.boundary_value = boundary_value
        self.damping = damping
        self._N = nx * ny
        self._band_solver = None

    def _build_banded_system(self) -> BandedMatrixSolver:
        n = self._N
        ml = self.nx
        mu = self.nx
        solver = BandedMatrixSolver(n, ml, mu)
        solver._A_band = np.zeros((solver.lda, n), dtype=float)
        hx2 = self.hx ** 2
        hy2 = self.hy ** 2

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i

                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                    solver._A_band[mu, idx] = 1.0
                    continue





                raise NotImplementedError("Hole 1: 请补全 Helmholtz 5 点 stencil 离散化")
        solver._factorized = False
        return solver

    def solve_for_rhs(self, rhs: np.ndarray) -> np.ndarray:
        if rhs.shape != (self.ny, self.nx):
            raise ValueError("rhs 形状必须与网格一致")
        rhs_flat = rhs.flatten().copy()

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                    rhs_flat[idx] = self.boundary_value

        if self._band_solver is None:
            self._band_solver = self._build_banded_system()
            self._band_solver.factorize_np()
        E_flat = self._band_solver.solve(rhs_flat)
        return E_flat.reshape(self.ny, self.nx)

    def compute_optical_intensity(self, E: np.ndarray) -> np.ndarray:
        return np.abs(E) ** 2

    def apply_source_and_solve(self, source_mask: np.ndarray, source_amplitude: complex = 1.0) -> np.ndarray:
        rhs = np.zeros((self.ny, self.nx), dtype=complex)
        rhs[source_mask] = source_amplitude

        E_real = self.solve_for_rhs(rhs.real)
        E_imag = self.solve_for_rhs(rhs.imag)
        return E_real + 1j * E_imag
