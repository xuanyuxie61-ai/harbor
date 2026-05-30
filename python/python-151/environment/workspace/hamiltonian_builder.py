
import numpy as np
import math
from typing import Tuple, Optional






def class_matrix_laguerre(m: int, alpha: float) -> Tuple[np.ndarray, np.ndarray, float]:
    if alpha <= -1.0:
        raise ValueError("alpha 必须 > -1")
    aj = np.zeros(m)
    bj = np.zeros(m)
    for i in range(m):
        aj[i] = 2.0 * (i + 1) - 1.0 + alpha
        bj[i] = np.sqrt((i + 1) * (i + 1 + alpha))
    zemu = np.exp(math.lgamma(alpha + 1.0))
    return aj, bj, zemu


def imtqlx(n: int, d: np.ndarray, e: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = d.astype(float).copy()
    e = e.astype(float).copy()
    z = z.astype(float).copy()
    itn = 30
    prec = np.finfo(float).eps

    if n == 1:
        return d, z
    e[n - 1] = 0.0
    for l in range(n):
        j = 0
        while True:
            m_val = l
            while m_val < n - 1:
                if abs(e[m_val]) <= prec * (abs(d[m_val]) + abs(d[m_val + 1])):
                    break
                m_val += 1
            p = d[l]
            if m_val == l:
                break
            if j == itn:
                raise RuntimeError("IMTQLX迭代次数超限")
            j += 1
            g = (d[l + 1] - p) / (2.0 * e[l])
            r_val = np.sqrt(g * g + 1.0)
            g = d[m_val] - p + e[l] / (g + np.sign(g) * abs(r_val))
            s_val = 1.0
            c_val = 1.0
            p_local = 0.0
            mml = m_val - l
            for ii in range(1, mml + 1):
                i = m_val - ii
                f = s_val * e[i]
                b_val = c_val * e[i]
                if abs(f) >= abs(g):
                    c_val = g / f
                    r2 = np.sqrt(c_val * c_val + 1.0)
                    e[i + 1] = f * r2
                    s_val = 1.0 / r2
                    c_val *= s_val
                else:
                    s_val = f / g
                    r2 = np.sqrt(s_val * s_val + 1.0)
                    e[i + 1] = g * r2
                    c_val = 1.0 / r2
                    s_val *= c_val
                g = d[i + 1] - p_local
                r_val = (d[i] - g) * s_val + 2.0 * c_val * b_val
                p_local = s_val * r_val
                d[i + 1] = g + p_local
                g = c_val * r_val - b_val
                f = z[i + 1]
                z[i + 1] = s_val * z[i] + c_val * f
                z[i] = c_val * z[i] - s_val * f
            d[l] = d[l] - p_local
            e[l] = g
            e[m_val] = 0.0

    for ii in range(1, n):
        i = ii - 1
        k = i
        p = d[i]
        for j in range(ii, n):
            if d[j] < p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            p = z[i]
            z[i] = z[k]
            z[k] = p
    return d, z


def gauss_laguerre_rule(order: int, alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    aj, bj, zemu = class_matrix_laguerre(order, alpha)
    z = np.zeros(order)
    z[0] = np.sqrt(zemu)
    x, w = imtqlx(order, aj, bj, z)
    w = w ** 2
    return x, w






def r8vec_house_column(n: int, x: np.ndarray, j: int) -> np.ndarray:
    v = np.zeros(n)
    v[j:n] = x[j:n]
    s = np.linalg.norm(v)
    if s != 0.0:
        if v[j] < 0:
            s = -s
        v[j] += s
        v[j:n] /= np.sqrt(v[j] * s)
    return v


def r8mat_house_axh(n: int, a: np.ndarray, v: np.ndarray) -> np.ndarray:
    beta = -2.0 / np.dot(v, v)
    w = beta * a.T @ v
    return a + np.outer(v, w)


def r8mat_orth_uniform(n: int) -> np.ndarray:
    a = np.eye(n)
    for j in range(n - 1):
        x = np.zeros(n)
        x[j:n] = np.random.randn(n - j)
        v = r8vec_house_column(n, x, j)
        if np.linalg.norm(v) > 1e-14:
            a = r8mat_house_axh(n, a, v)

        if np.random.rand() > 0.5:
            k = np.random.randint(0, n)
            a[k, :] = -a[k, :]
    return a


def r8symm_gen(n: int, lambda_mean: float = 0.0,
               lambda_dev: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lambda_vec = lambda_mean + lambda_dev * np.random.randn(n)
    q = r8mat_orth_uniform(n)
    a = q @ np.diag(lambda_vec) @ q.T
    return a, q, lambda_vec






class MolecularHamiltonian:
    def __init__(self, n_orbitals: int = 4):
        self.n_orbitals = n_orbitals
        self.h_core = np.zeros((n_orbitals, n_orbitals))
        self.eri = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
        self._build_integrals()

    def _build_integrals(self):
        n = self.n_orbitals


        for i in range(n):
            self.h_core[i, i] = -0.5 - 0.1 * i

        for i in range(n - 1):
            self.h_core[i, i + 1] = -0.2
            self.h_core[i + 1, i] = -0.2



        for i in range(n):
            for j in range(n):
                self.eri[i, i, j, j] = 0.3 / (abs(i - j) + 1.0)
                self.eri[i, j, j, i] = 0.1 / (abs(i - j) + 1.0)

    def get_nuclear_repulsion(self) -> float:
        return 0.7

    def compute_exact_ground_state(self) -> Tuple[float, np.ndarray]:
        from itertools import combinations
        n_electrons = self.n_orbitals // 2
        n_spin_orbitals = self.n_orbitals

        configs = list(combinations(range(n_spin_orbitals), n_electrons))
        dim = len(configs)
        H_mat = np.zeros((dim, dim))

        def idx(config):
            try:
                return configs.index(tuple(sorted(config)))
            except ValueError:
                return -1

        for ic, c in enumerate(configs):

            for p in range(n_spin_orbitals):
                if p in c:


                    spatial_p = p % self.n_orbitals
                    spin_p = p // self.n_orbitals
                    H_mat[ic, ic] += self.h_core[spatial_p, spatial_p]
                    for q in range(p + 1, n_spin_orbitals):
                        if q in c:
                            spatial_q = q % self.n_orbitals
                            spin_q = q // self.n_orbitals
                            if spin_p == spin_q:
                                H_mat[ic, ic] += self.eri[spatial_p, spatial_p, spatial_q, spatial_q]
                                H_mat[ic, ic] -= self.eri[spatial_p, spatial_q, spatial_q, spatial_p]

            for p in range(n_spin_orbitals):
                if p in c:
                    for q in range(n_spin_orbitals):
                        if q not in c:
                            c_new = list(c)
                            c_new.remove(p)
                            c_new.append(q)
                            jc = idx(c_new)
                            if jc >= 0:
                                spatial_p = p % self.n_orbitals
                                spatial_q = q % self.n_orbitals
                                H_mat[ic, jc] += self.h_core[spatial_p, spatial_q]


        eigvals = np.linalg.eigvalsh(H_mat)
        E0 = eigvals[0] + self.get_nuclear_repulsion()
        return float(E0), H_mat

    def to_pauli_strings(self) -> dict:

        raise NotImplementedError("Hole 1: 请实现Jordan-Wigner变换")



def compute_radial_integral_gauss_laguerre(n: int, alpha: float,
                                            f: callable) -> float:
    x, w = gauss_laguerre_rule(n, alpha)







    result = 0.0
    for xi, wi in zip(x, w):
        ri = np.sqrt(xi)

        jac = 0.5 / ri
        result += wi * f(ri) * jac
    return result
