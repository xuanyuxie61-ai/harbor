
import numpy as np
from typing import Tuple, Optional, Callable


class SparseMatrixCRS:

    def __init__(self, n: int, nz_num: int):
        self.n = n
        self.nz_num = nz_num
        self.a = np.zeros(nz_num)
        self.ia = np.zeros(nz_num, dtype=int)
        self.ja = np.zeros(nz_num, dtype=int)

    def from_dense(self, A: np.ndarray, threshold: float = 1e-14) -> None:
        if A.shape[0] != self.n or A.shape[1] != self.n:
            raise ValueError("矩阵维度不匹配")

        idx = 0
        for i in range(self.n):
            for j in range(self.n):
                if abs(A[i, j]) > threshold:
                    if idx >= self.nz_num:
                        raise ValueError("非零元素数量超过预分配")
                    self.a[idx] = A[i, j]
                    self.ia[idx] = i
                    self.ja[idx] = j
                    idx += 1


        self.nz_num = idx
        self.a = self.a[:idx]
        self.ia = self.ia[:idx]
        self.ja = self.ja[:idx]


def sparse_matvec(a: np.ndarray, ia: np.ndarray, ja: np.ndarray,
                  x: np.ndarray, n: int, nz_num: int) -> np.ndarray:
    y = np.zeros(n)
    for k in range(nz_num):
        i = ia[k]
        j = ja[k]
        y[i] += a[k] * x[j]
    return y


def mult_givens(c: float, s: float, k: int, g: np.ndarray) -> np.ndarray:
    g1 = c * g[k] - s * g[k + 1]
    g2 = s * g[k] + c * g[k + 1]
    g = g.copy()
    g[k] = g1
    g[k + 1] = g2
    return g


def mgmres(a: np.ndarray, ia: np.ndarray, ja: np.ndarray,
           x: np.ndarray, rhs: np.ndarray, n: int, nz_num: int,
           itr_max: int = 100, mr: int = 20,
           tol_abs: float = 1e-10, tol_rel: float = 1e-8) -> np.ndarray:
    if mr <= 0 or mr > n:
        mr = min(20, n)

    delta = 0.001
    verbose = 0

    rho_tol = tol_rel * np.linalg.norm(rhs) if np.linalg.norm(rhs) > 0 else tol_abs

    for itr in range(itr_max):
        r = rhs - sparse_matvec(a, ia, ja, x, n, nz_num)
        rho = np.linalg.norm(r)

        if rho <= tol_abs or rho <= rho_tol:
            break

        v = np.zeros((n, mr + 1))
        v[:, 0] = r / (rho + 1e-30)

        g = np.zeros(mr + 1)
        g[0] = rho

        h = np.zeros((mr + 1, mr))
        c = np.zeros(mr)
        s = np.zeros(mr)

        for k in range(mr):
            v[:, k + 1] = sparse_matvec(a, ia, ja, v[:, k], n, nz_num)
            av = np.linalg.norm(v[:, k + 1])

            for j in range(k + 1):
                h[j, k] = np.dot(v[:, j], v[:, k + 1])
                v[:, k + 1] -= h[j, k] * v[:, j]

            h[k + 1, k] = np.linalg.norm(v[:, k + 1])


            if av + delta * h[k + 1, k] == av:
                for j in range(k + 1):
                    htmp = np.dot(v[:, j], v[:, k + 1])
                    h[j, k] += htmp
                    v[:, k + 1] -= htmp * v[:, j]
                h[k + 1, k] = np.linalg.norm(v[:, k + 1])

            if h[k + 1, k] != 0.0:
                v[:, k + 1] /= h[k + 1, k]


            if k > 0:
                y = h[:k + 2, k].copy()
                for j in range(k):
                    y = mult_givens(c[j], s[j], j, y)
                h[:k + 2, k] = y

            mu = np.sqrt(h[k, k] ** 2 + h[k + 1, k] ** 2)
            if mu > 0:
                c[k] = h[k, k] / mu
                s[k] = -h[k + 1, k] / mu
                h[k, k] = c[k] * h[k, k] - s[k] * h[k + 1, k]
                h[k + 1, k] = 0.0
                g = mult_givens(c[k], s[k], k, g)

            rho = abs(g[k + 1])
            if rho <= tol_abs and rho <= rho_tol:

                k_copy = k
                y = np.zeros(k_copy + 1)
                y[k_copy] = g[k_copy] / (h[k_copy, k_copy] + 1e-30)
                for i in range(k_copy - 1, -1, -1):
                    y[i] = (g[i] - np.dot(h[i, i + 1:k_copy + 1], y[i + 1:k_copy + 1])) / \
                           (h[i, i] + 1e-30)
                x += v[:, :k_copy + 1] @ y
                break
        else:

            k = mr - 1
            y = np.zeros(mr)
            y[mr - 1] = g[mr - 1] / (h[mr - 1, mr - 1] + 1e-30)
            for i in range(mr - 2, -1, -1):
                y[i] = (g[i] - np.dot(h[i, i + 1:mr], y[i + 1:mr])) / \
                       (h[i, i] + 1e-30)
            x += v[:, :mr] @ y

    return x


def conjugate_gradient(A: np.ndarray, b: np.ndarray,
                       x0: Optional[np.ndarray] = None,
                       max_iter: int = None,
                       tol: float = 1e-10) -> np.ndarray:
    n = len(b)
    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    for _ in range(max_iter):
        Ap = A @ p
        pAp = np.dot(p, Ap)

        if abs(pAp) < 1e-30:
            break

        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)

        if np.sqrt(rs_new) < tol:
            break

        beta = rs_new / (rs_old + 1e-30)
        p = r + beta * p
        rs_old = rs_new

    return x


class VerticalTransportSolver:

    def __init__(self, z: np.ndarray, Kzz: np.ndarray,
                 w: Optional[np.ndarray] = None):
        if len(z) < 3:
            raise ValueError("高度网格至少需要3个点")
        if len(Kzz) != len(z):
            raise ValueError("Kzz 和 z 长度必须相同")
        if not np.all(np.diff(z) > 0):
            raise ValueError("高度网格必须严格单调递增")

        self.nz = len(z)
        self.z = z.copy()
        self.dz = np.diff(z)
        self.Kzz = np.clip(Kzz, 1e-6, 1e4)

        if w is None:
            self.w = np.zeros(self.nz)
        else:
            self.w = w.copy()


        self.K_face = np.zeros(self.nz - 1)
        for i in range(self.nz - 1):
            self.K_face[i] = 0.5 * (self.Kzz[i] + self.Kzz[i + 1])

    def _build_diffusion_matrix(self) -> np.ndarray:
        nz = self.nz
        L = np.zeros((nz, nz))

        for i in range(1, nz - 1):
            dz_i = self.z[i] - self.z[i - 1]
            dz_ip1 = self.z[i + 1] - self.z[i]
            dz_mid = 0.5 * (dz_i + dz_ip1)

            K_m = self.K_face[i - 1]
            K_p = self.K_face[i]

            a_m = K_m / (dz_i * dz_mid)
            a_p = K_p / (dz_ip1 * dz_mid)

            L[i, i - 1] = a_m
            L[i, i] = -(a_m + a_p)
            L[i, i + 1] = a_p



        dz_0 = self.z[1] - self.z[0]
        L[0, 0] = -self.K_face[0] / (dz_0 ** 2)
        L[0, 1] = self.K_face[0] / (dz_0 ** 2)


        dz_n = self.z[-1] - self.z[-2]
        L[-1, -2] = self.K_face[-1] / (dz_n ** 2)
        L[-1, -1] = -self.K_face[-1] / (dz_n ** 2)

        return L

    def _build_advection_matrix(self) -> np.ndarray:
        nz = self.nz
        A = np.zeros((nz, nz))

        for i in range(1, nz):
            dz_i = self.z[i] - self.z[i - 1]
            w_i = self.w[i]

            if w_i > 0:
                A[i, i - 1] = -w_i / dz_i
                A[i, i] = w_i / dz_i
            elif w_i < 0:
                if i < nz - 1:
                    dz_ip1 = self.z[i + 1] - self.z[i]
                    A[i, i] = -w_i / dz_ip1
                    A[i, i + 1] = w_i / dz_ip1


        A[0, 0] = 0.0

        return A

    def build_implicit_matrix(self, dt: float,
                              jac_diag: Optional[np.ndarray] = None) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt 必须为正")

        nz = self.nz
        L_diff = self._build_diffusion_matrix()
        L_adv = self._build_advection_matrix()

        A = np.eye(nz) - dt * (L_diff + L_adv)

        if jac_diag is not None:
            if len(jac_diag) != nz:
                raise ValueError("jac_diag 长度必须与 nz 相同")
            for i in range(nz):
                A[i, i] -= dt * jac_diag[i]

        return A

    def solve_implicit(self, n_old: np.ndarray, dt: float,
                       source: Optional[np.ndarray] = None,
                       jac_diag: Optional[np.ndarray] = None,
                       use_cg: bool = True) -> np.ndarray:
        nz = self.nz
        A = self.build_implicit_matrix(dt, jac_diag)
        b = n_old.copy()

        if source is not None:
            b += dt * source


        b = np.clip(b, -1e25, 1e25)

        if use_cg and self._is_symmetric_positive_definite(A):
            n_new = conjugate_gradient(A, b, max_iter=min(nz * 2, 1000),
                                       tol=1e-12)
        else:

            nz_est = nz * 3
            sparse_A = SparseMatrixCRS(nz, nz_est)
            sparse_A.from_dense(A)
            n_new = mgmres(sparse_A.a, sparse_A.ia, sparse_A.ja,
                           n_old.copy(), b, nz, sparse_A.nz_num,
                           itr_max=200, mr=min(30, nz),
                           tol_abs=1e-12, tol_rel=1e-10)

        return np.clip(n_new, 0.0, 1e25)

    def _is_symmetric_positive_definite(self, A: np.ndarray,
                                         tol: float = 1e-10) -> bool:
        if A.shape[0] != A.shape[1]:
            return False

        asym = np.max(np.abs(A - A.T))
        if asym > tol * np.max(np.abs(A)):
            return False

        try:
            np.linalg.cholesky(A + 1e-12 * np.eye(A.shape[0]))
            return True
        except np.linalg.LinAlgError:
            return False

    def compute_flux_divergence(self, n: np.ndarray) -> np.ndarray:
        nz = self.nz
        div = np.zeros(nz)


        for i in range(1, nz - 1):
            dz_i = self.z[i] - self.z[i - 1]
            dz_ip1 = self.z[i + 1] - self.z[i]
            dz_mid = 0.5 * (dz_i + dz_ip1)

            flux_p = self.K_face[i] * (n[i + 1] - n[i]) / dz_ip1
            flux_m = self.K_face[i - 1] * (n[i] - n[i - 1]) / dz_i
            div[i] = (flux_p - flux_m) / dz_mid


        for i in range(1, nz):
            dz_i = self.z[i] - self.z[i - 1]
            if self.w[i] > 0:
                div[i] -= self.w[i] * (n[i] - n[i - 1]) / dz_i
            elif i < nz - 1 and self.w[i] < 0:
                dz_ip1 = self.z[i + 1] - self.z[i]
                div[i] -= self.w[i] * (n[i + 1] - n[i]) / dz_ip1

        return div
