
import numpy as np
from typing import Tuple, Optional


class BandedLU:

    def __init__(self, n: int, ml: int, mu: int):
        self.n = n
        self.ml = ml
        self.mu = mu
        self.kv = mu + ml
        self.kd = mu + ml + 1

    def _full_to_band(self, A_full: np.ndarray) -> np.ndarray:
        n = self.n
        ml = self.ml
        mu = self.mu
        rows = 2 * ml + mu + 1
        A_band = np.zeros((rows, n), dtype=np.float64)

        for j in range(n):
            i1 = max(0, j - mu)
            i2 = min(n - 1, j + ml)
            for i in range(i1, i2 + 1):
                k = i - j + ml + mu
                A_band[k, j] = A_full[i, j]
        return A_band

    def factorize(self, A_band: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        n = self.n
        ml = self.ml
        mu = self.mu
        kv = self.kv

        A_lu = np.array(A_band, dtype=np.float64, copy=True)
        pivot = np.zeros(n, dtype=np.int64)
        info = 0


        for j in range(mu + 1, min(kv + 1, n)):
            for i in range(kv - j + 1, ml):
                A_lu[i, j] = 0.0

        ju = 0
        for j in range(min(n, n)):

            if j + kv < n:
                A_lu[:ml, j + kv] = 0.0


            km = min(ml, n - j - 1)
            piv = abs(A_lu[kv, j])
            jp = 0
            for i in range(1, km + 1):
                if abs(A_lu[kv + i, j]) > piv:
                    piv = abs(A_lu[kv + i, j])
                    jp = i

            pivot[j] = jp + j + 1

            if abs(A_lu[kv + jp, j]) < 1e-18:
                if info == 0:
                    info = j + 1
                continue


            ju = max(ju, min(j + mu + jp, n - 1))


            if jp != 0:
                for i in range(ju - j + 1):
                    t = A_lu[kv + jp - i, j + i]
                    A_lu[kv + jp - i, j + i] = A_lu[kv - i, j + i]
                    A_lu[kv - i, j + i] = t


            if km > 0:
                A_lu[kv + 1:kv + km + 1, j] /= A_lu[kv, j]


                if j < ju:
                    for k in range(1, ju - j + 1):
                        if abs(A_lu[kv - k, j + k]) > 1e-18:
                            for i in range(1, km + 1):
                                A_lu[kv + i - k, j + k] -= A_lu[kv + i, j] * A_lu[kv - k, j + k]

        return A_lu, pivot, info

    def solve(self, A_lu: np.ndarray, pivot: np.ndarray, b: np.ndarray, trans: str = 'N') -> np.ndarray:
        n = self.n
        ml = self.ml
        mu = self.mu
        kd = self.kd

        b = np.asarray(b, dtype=np.float64)
        if b.ndim == 1:
            b = b[:, np.newaxis]
            squeeze = True
        else:
            squeeze = False

        nrhs = b.shape[1]
        x = b.copy()

        if trans.upper() == 'N':

            if ml > 0:
                for j in range(n - 1):
                    lm = min(ml, n - j - 1)
                    l = int(pivot[j] - 1)
                    if l != j:
                        for i in range(nrhs):
                            t = x[l, i]
                            x[l, i] = x[j, i]
                            x[j, i] = t

                    for k in range(nrhs):
                        if x[j, k] != 0.0:
                            for i in range(1, lm + 1):
                                x[j + i, k] -= A_lu[kd + i - 1, j] * x[j, k]


            for k in range(nrhs):
                for j in range(n - 1, -1, -1):
                    if x[j, k] != 0.0:
                        x[j, k] /= A_lu[kd - 1, j]
                        for i in range(max(0, j - ml - mu), j):
                            x[i, k] -= A_lu[kd - 1 - j + i, j] * x[j, k]

        else:

            for k in range(nrhs):
                for j in range(n):
                    temp = x[j, k]
                    for i in range(max(0, j - ml - mu), j):
                        temp -= A_lu[kd - 1 - j + i, j] * x[i, k]
                    x[j, k] = temp / A_lu[kd - 1, j]


            if ml > 0:
                for j in range(n - 2, -1, -1):
                    lm = min(ml, n - j - 1)
                    for k in range(nrhs):
                        for i in range(1, lm + 1):
                            x[j, k] -= A_lu[kd + i - 1, j] * x[j + i, k]

                    l = int(pivot[j] - 1)
                    if l != j:
                        for k in range(nrhs):
                            t = x[l, k]
                            x[l, k] = x[j, k]
                            x[j, k] = t

        if squeeze:
            return x.ravel()
        return x


def assemble_poisson_boltzmann_jacobian(
    n: int,
    phi: np.ndarray,
    rho: np.ndarray,
    h: float,
    epsilon: float = 80.0,
    kappa: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:










    raise NotImplementedError("Hole 1: 待实现 PB Jacobian 与残差组装")


def solve_nonlinear_pb(
    n: int = 129,
    domain_length: float = 20.0,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> dict:
    h = domain_length / (n - 1)
    x = np.linspace(-domain_length / 2, domain_length / 2, n)


    sigma = 1.5
    rho = -1.0 * np.exp(-x ** 2 / (2.0 * sigma ** 2))


    phi = np.zeros(n, dtype=np.float64)

    solver = BandedLU(n, ml=1, mu=1)

    for it in range(max_iter):
        J, F = assemble_poisson_boltzmann_jacobian(n, phi, rho, h)


        A_band = solver._full_to_band(J)
        A_lu, pivot, info = solver.factorize(A_band)

        if info != 0:
            raise RuntimeError(f"LU factorization failed at iteration {it}, info={info}")

        delta_phi = solver.solve(A_lu, pivot, -F)
        phi += delta_phi

        res_norm = float(np.linalg.norm(F))
        if res_norm < tol:
            return {
                "x": x,
                "phi": phi,
                "rho": rho,
                "iterations": it + 1,
                "residual_norm": res_norm,
                "success": True,
            }

    return {
        "x": x,
        "phi": phi,
        "rho": rho,
        "iterations": max_iter,
        "residual_norm": float(np.linalg.norm(F)),
        "success": False,
    }
