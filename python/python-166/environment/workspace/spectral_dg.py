
import numpy as np
from typing import Tuple
from scipy.special import jacobi, roots_jacobi


def chebyshev_matrix(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be >= 1")

    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1)
    c[0] = 2.0
    c[n] = 2.0
    c = c * ((-1.0) ** np.arange(n + 1))

    X = np.tile(x.reshape(-1, 1), (1, n + 1))
    dX = X - X.T

    D = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                D[i, j] = c[i] / (c[j] * (x[i] - x[j]))


    for i in range(n + 1):
        D[i, i] = -np.sum(D[:, i])

        if i == 0:
            D[i, i] = (2.0 * n * n + 1.0) / 6.0
        elif i == n:
            D[i, i] = -(2.0 * n * n + 1.0) / 6.0
        else:
            D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))

    return x, D


def jacobi_polynomial(x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    x = np.atleast_1d(x)
    if N == 0:
        return np.ones_like(x)
    if N == 1:
        return 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x


    P = np.zeros((N + 1, len(x)))
    P[0] = 1.0
    P[1] = 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x

    for n in range(1, N):
        a1 = 2.0 * (n + 1.0) * (n + alpha + beta + 1.0) * (2.0 * n + alpha + beta)
        a2 = (2.0 * n + alpha + beta + 1.0) * (alpha ** 2 - beta ** 2)
        a3 = (2.0 * n + alpha + beta) * (2.0 * n + alpha + beta + 1.0) * (2.0 * n + alpha + beta + 2.0)
        a4 = 2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0)

        P[n + 1] = ((a2 + a3 * x) * P[n] - a4 * P[n - 1]) / a1

    return P[N]


def vandermonde_1d(N: int, r: np.ndarray) -> np.ndarray:
    r = np.atleast_1d(r)
    V = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        V[:, j] = jacobi_polynomial(r, 0.0, 0.0, j)
    return V


def dmatrix_1d(N: int, r: np.ndarray, V: np.ndarray = None) -> np.ndarray:
    if V is None:
        V = vandermonde_1d(N, r)


    Vr = np.zeros_like(V)
    for j in range(N + 1):
        if j == 0:
            Vr[:, j] = 0.0
        else:


            Vr[:, j] = 0.5 * (j + 1.0) * jacobi_polynomial(r, 1.0, 1.0, j - 1)

    Dr = Vr @ np.linalg.inv(V)
    return Dr


def jacobi_gauss_lobatto(alpha: float, beta: float, N: int) -> np.ndarray:
    if N == 0:
        return np.array([-1.0])
    if N == 1:
        return np.array([-1.0, 1.0])


    x_int, _ = roots_jacobi(N - 1, alpha + 1.0, beta + 1.0)
    x = np.sort(np.concatenate([[-1.0], x_int, [1.0]]))
    return x


def lift_1d(N: int, V: np.ndarray = None) -> np.ndarray:
    if V is None:

        r = jacobi_gauss_lobatto(0.0, 0.0, N)
        V = vandermonde_1d(N, r)

    Np = N + 1
    Nfaces = 2
    Nfp = 1

    Emat = np.zeros((Np, Nfaces * Nfp))
    Emat[0, 0] = 1.0
    Emat[Np - 1, 1] = 1.0

    invV = np.linalg.inv(V)
    LIFT = V @ (V.T @ Emat)
    return LIFT


def dg_derivative_1d(u: np.ndarray, Dr: np.ndarray, rx: np.ndarray,
                     LIFT: np.ndarray, nx: np.ndarray,
                     vmapM: np.ndarray, vmapP: np.ndarray) -> np.ndarray:
    Np, K = u.shape


    dudr = Dr @ u


    uM = u.flatten()[vmapM]
    uP = u.flatten()[vmapP]
    flux = 0.5 * (uM + uP)






    du = uM - uP
    flux = nx.flatten() * du * 0.5


    Nfaces = 2
    fluxmat = flux.reshape(Nfaces, K, order='F')
    lifted = LIFT @ fluxmat


    dudx = rx * dudr - rx * lifted
    return dudx


def build_maps_1d(Np: int, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Nfaces = 2
    Nfp = 1
    NfacesK = Nfaces * K


    vmapM = np.zeros(NfacesK, dtype=int)
    vmapP = np.zeros(NfacesK, dtype=int)
    nx = np.zeros((Nfaces, K))

    for k in range(K):

        vmapM[0 * K + k] = k * Np

        vmapM[1 * K + k] = k * Np + Np - 1


        if k == 0:
            vmapP[0 * K + k] = vmapM[1 * K + k]
        else:
            vmapP[0 * K + k] = vmapM[1 * K + (k - 1)]

        if k == K - 1:
            vmapP[1 * K + k] = vmapM[0 * K + k]
        else:
            vmapP[1 * K + k] = vmapM[0 * K + (k + 1)]

        nx[0, k] = -1.0
        nx[1, k] = 1.0

    return vmapM, vmapP, nx


def meshgen_1d(xmin: float, xmax: float, K: int) -> Tuple[np.ndarray, np.ndarray]:
    if K < 1:
        raise ValueError("K must be >= 1")
    VX = np.linspace(xmin, xmax, K + 1)
    EToV = np.zeros((K, 2), dtype=int)
    for k in range(K):
        EToV[k] = [k, k + 1]
    return VX, EToV


def geometric_factors_1d(x: np.ndarray, Dr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    J = Dr @ x
    if np.any(np.abs(J) < 1e-14):

        J = np.where(np.abs(J) < 1e-14, np.sign(J + 1e-14) * 1e-14, J)
    rx = 1.0 / J
    return rx, J
