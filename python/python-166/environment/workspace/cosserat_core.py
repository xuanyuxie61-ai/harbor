
import numpy as np
from typing import Tuple, List
from mesh_utils import line_grid, chebyshev_grid


def hat_map(v: np.ndarray) -> np.ndarray:
    if len(v) != 3:
        raise ValueError("v must have length 3")
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def vee_map(R: np.ndarray) -> np.ndarray:
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")
    return np.array([R[2, 1], R[0, 2], R[1, 0]])


def rodrigues_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-14:
        return np.eye(3)
    axis = axis / norm

    K = hat_map(axis)
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R


def compute_curvature(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    N = len(s)
    if N < 3:
        return np.zeros(N), np.zeros(N)


    ds = np.diff(s)
    if np.any(ds <= 0):
        raise ValueError("s must be strictly increasing")


    dr = np.zeros_like(r)
    dr[0] = (r[1] - r[0]) / ds[0]
    dr[-1] = (r[-1] - r[-2]) / ds[-1]
    for i in range(1, N - 1):
        dr[i] = (r[i + 1] - r[i - 1]) / (ds[i - 1] + ds[i])


    d2r = np.zeros_like(r)
    d2r[0] = (r[2] - 2.0 * r[1] + r[0]) / (ds[0] ** 2)
    d2r[-1] = (r[-1] - 2.0 * r[-2] + r[-3]) / (ds[-2] ** 2)
    for i in range(1, N - 1):
        d2r[i] = (r[i + 1] - 2.0 * r[i] + r[i - 1]) / (0.5 * (ds[i - 1] + ds[i])) ** 2


    kappa = np.zeros(N)
    for i in range(N):
        rp = dr[i]
        rpp = d2r[i]
        cross = np.cross(rp, rpp)
        denom = np.linalg.norm(rp) ** 3
        if denom > 1e-14:
            kappa[i] = np.linalg.norm(cross) / denom


    tau = np.zeros(N)
    return kappa, tau


def r8blt_mv(a: np.ndarray, ml: int, x: np.ndarray) -> np.ndarray:
    if a.ndim != 2:
        raise ValueError("a must be 2D")
    ml1, N = a.shape
    ml_actual = ml1 - 1
    if len(x) != N:
        raise ValueError("dimension mismatch")

    y = np.zeros(N)
    for i in range(N):
        j_lo = max(0, i - ml_actual)
        for j in range(j_lo, i + 1):

            y[i] += a[i - j, j] * x[j]
    return y


def r8blt_sl(a: np.ndarray, ml: int, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2:
        raise ValueError("a must be 2D")
    ml1, N = a.shape
    ml_actual = ml1 - 1
    if len(b) != N:
        raise ValueError("dimension mismatch")

    x = np.zeros(N)
    for i in range(N):
        x[i] = b[i]
        j_lo = max(0, i - ml_actual)
        for j in range(j_lo, i):
            x[i] -= a[i - j, j] * x[j]
        diag = a[0, i]
        if abs(diag) < 1e-14:
            diag = 1e-14
        x[i] /= diag
    return x


def r8blt_det(a: np.ndarray, ml: int) -> float:
    if a.ndim != 2:
        raise ValueError("a must be 2D")
    N = a.shape[1]
    det = 1.0
    for j in range(N):
        diag = a[0, j]
        if abs(diag) < 1e-14:
            return 0.0
        det *= diag
    return det


def assemble_banded_stiffness(N: int, EI: float, EA: float, ds: float, ml: int = 3) -> np.ndarray:
    if ml < 1:
        raise ValueError("ml must be >= 1")



    a = np.zeros((ml + 1, N))



    for i in range(N):
        a[0, i] = 6.0 * EI / ds ** 4
        if i >= 1:
            a[1, i - 1] = -4.0 * EI / ds ** 4
        if i >= 2:
            a[2, i - 2] = EI / ds ** 4

    return a


def forward_kinematics_cosserat(L: float, Ns: int,
                                kappa_base: np.ndarray,
                                epsilon_base: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = line_grid(Ns + 1, 0.0, L, c=1)
    ds = L / Ns

    n_nodes = Ns + 1
    r = np.zeros((n_nodes, 3))
    R = np.zeros((n_nodes, 3, 3))
    R[0] = np.eye(3)

    for i in range(n_nodes - 1):
        kappa = kappa_base[i]
        eps = epsilon_base






        raise NotImplementedError("Hole 1: 实现RK4积分核心")

    return s, r, R


def compute_strain_measures(r: np.ndarray, R: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = len(s)
    v = np.zeros((N, 3))
    u = np.zeros((N, 3))


    dr = np.zeros_like(r)
    dr[0] = (r[1] - r[0]) / (s[1] - s[0])
    dr[-1] = (r[-1] - r[-2]) / (s[-1] - s[-2])
    for i in range(1, N - 1):
        dr[i] = (r[i + 1] - r[i - 1]) / (s[i + 1] - s[i - 1])

    dR = np.zeros_like(R)
    dR[0] = (R[1] - R[0]) / (s[1] - s[0])
    dR[-1] = (R[-1] - R[-2]) / (s[-1] - s[-2])
    for i in range(1, N - 1):
        dR[i] = (R[i + 1] - R[i - 1]) / (s[i + 1] - s[i - 1])

    for i in range(N):
        Ri_T = R[i].T
        v[i] = Ri_T @ dr[i]
        u[i] = vee_map(Ri_T @ dR[i])

    return v, u
