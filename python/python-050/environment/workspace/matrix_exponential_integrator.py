
import numpy as np
from typing import Callable, Optional
from scipy.linalg import expm


def phi1_function(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128 if np.iscomplexobj(z) else np.float64)

    if np.isscalar(z) or z.size == 1:
        z_val = float(z) if not np.iscomplexobj(z) else complex(z)
        if abs(z_val) < 1e-8:
            return np.array(1.0 + z_val / 2.0 + z_val ** 2 / 6.0)
        return np.array((np.exp(z_val) - 1.0) / z_val)


    if z.ndim == 2 and z.shape[0] == z.shape[1]:

        nz = z.shape[0]
        I = np.eye(nz, dtype=z.dtype)
        if nz <= 50:


            eA = expm(z)
            diff = eA - I

            phi1 = np.linalg.solve(z + 1e-14 * I, diff)
            return phi1
        else:
            raise NotImplementedError("Large matrix phi1 requires Krylov method.")


    result = np.empty_like(z, dtype=np.float64)
    small = np.abs(z) < 1e-8
    result[small] = 1.0 + z[small] / 2.0 + z[small] ** 2 / 6.0 + z[small] ** 3 / 24.0
    result[~small] = (np.exp(z[~small]) - 1.0) / z[~small]
    return result


def exponential_euler_step(u: np.ndarray,
                           dt: float,
                           L: np.ndarray,
                           N_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)

    if L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix.")
    if len(u) != L.shape[0]:
        raise ValueError("u and L dimensions must match.")
    if dt <= 0:
        raise ValueError("dt must be positive.")


    dtL = dt * L
    exp_dtL = expm(dtL)


    Nu = np.asarray(N_func(u), dtype=np.float64)


    if len(u) <= 50:
        phi1_dtL = phi1_function(dtL)
        phi_N = phi1_dtL @ Nu
    else:

        I = np.eye(len(u))

        rhs = (expm(dtL) - I) @ Nu
        phi_N = np.linalg.solve(dtL + 1e-12 * I, rhs)

    u_new = exp_dtL @ u + dt * phi_N
    return u_new


def build_1d_diffusion_matrix(n: int, dx: float,
                               diffusivity: float) -> np.ndarray:
    if n < 3:
        raise ValueError("n must be >= 3")
    coef = diffusivity / (dx ** 2)
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n - 1):
        L[i, i - 1] = coef
        L[i, i] = -2.0 * coef
        L[i, i + 1] = coef
    L[0, 0] = -coef
    L[0, 1] = coef
    L[-1, -2] = coef
    L[-1, -1] = -coef
    return L


def exponential_integrator_ice_thickness(H: np.ndarray,
                                         dt: float,
                                         dx: float,
                                         diffusivity_func: Callable,
                                         accumulation: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    n = len(H)
    D_eff = float(diffusivity_func(H))

    L = build_1d_diffusion_matrix(n, dx, D_eff)

    def N_func(u: np.ndarray) -> np.ndarray:


        return np.asarray(accumulation, dtype=np.float64)

    H_new = exponential_euler_step(H, dt, L, N_func)
    H_new = np.maximum(H_new, 0.0)
    return H_new


def krylov_phi1_approximation(A: np.ndarray, v: np.ndarray,
                              m_krylov: int = 30) -> np.ndarray:
    n = len(v)
    m = min(m_krylov, n)


    V = np.zeros((n, m + 1), dtype=np.float64)
    H_arnoldi = np.zeros((m + 1, m), dtype=np.float64)

    beta = np.linalg.norm(v)
    if beta < 1e-15:
        return np.zeros(n, dtype=np.float64)
    V[:, 0] = v / beta

    for j in range(m):
        w = A @ V[:, j]
        for i in range(j + 1):
            H_arnoldi[i, j] = np.dot(V[:, i], w)
            w = w - H_arnoldi[i, j] * V[:, i]
        H_arnoldi[j + 1, j] = np.linalg.norm(w)
        if H_arnoldi[j + 1, j] < 1e-14:
            m = j + 1
            break
        V[:, j + 1] = w / H_arnoldi[j + 1, j]


    Hm = H_arnoldi[:m, :m]
    e1 = np.zeros(m, dtype=np.float64)
    e1[0] = 1.0


    phi1_Hm = phi1_function(Hm)
    y = phi1_Hm @ e1

    approx = beta * V[:, :m] @ y
    return approx
