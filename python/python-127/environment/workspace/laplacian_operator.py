
import numpy as np
from scipy.sparse import diags, csr_matrix


def laplacian_5point(V, dx, dy):
    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise ValueError("V 必须为二维数组")

    nx, ny = V.shape
    L = np.zeros_like(V)


    L[1:-1, 1:-1] = (
        (V[2:, 1:-1] - 2.0 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / (dx**2)
        + (V[1:-1, 2:] - 2.0 * V[1:-1, 1:-1] + V[1:-1, :-2]) / (dy**2)
    )


    L[0, :] = L[1, :]
    L[-1, :] = L[-2, :]
    L[:, 0] = L[:, 1]
    L[:, -1] = L[:, -2]

    return L


def laplacian_9point(V, dx, dy):
    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise ValueError("V 必须为二维数组")

    nx, ny = V.shape
    L = np.zeros_like(V)







    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            L[i, j] = (

                0.0
            )


    L[0, :] = L[1, :]
    L[-1, :] = L[-2, :]
    L[:, 0] = L[:, 1]
    L[:, -1] = L[:, -2]

    return L


def build_differ_matrix(n, stencil):
    stencil = np.asarray(stencil, dtype=float)
    if len(stencil) != n:
        raise ValueError("stencil 长度必须等于 n")
    if len(np.unique(stencil)) != n:
        raise ValueError("stencil 中的点必须互异")
    if np.any(np.isclose(stencil, 0.0)):
        raise ValueError("stencil 中不能包含零点（边界处理请单独处理）")

    A = np.zeros((n, n))
    A[0, :] = stencil
    for i in range(1, n):
        A[i, :] = A[i - 1, :] * stencil
    return A


def high_order_derivative_coefficients(order, stencil):
    n = len(stencil)
    A = build_differ_matrix(n, stencil)
    rhs = np.zeros(n)
    if 1 <= order <= n - 1:
        rhs[order] = np.math.factorial(order)
    else:
        raise ValueError("order 必须在 1 到 n-1 之间")
    coeffs = np.linalg.solve(A, rhs)
    return coeffs


def laplacian_matrix_1d(n, dx, stencil_type='central'):
    if stencil_type == 'central':
        main = -2.0 * np.ones(n)
        off = np.ones(n - 1)
        L = diags([off, main, off], [-1, 0, 1], format='csr') / (dx**2)

        L[0, 0] = -1.0 / (dx**2)
        L[0, 1] = 1.0 / (dx**2)
        L[-1, -1] = -1.0 / (dx**2)
        L[-1, -2] = 1.0 / (dx**2)
    elif stencil_type == 'compact4':


        main = -2.5 * np.ones(n)
        off1 = (4.0 / 3.0) * np.ones(n - 1)
        off2 = (-1.0 / 12.0) * np.ones(n - 2)
        L = diags([off2, off1, main, off1, off2],
                  [-2, -1, 0, 1, 2], format='csr') / (dx**2)

        L[0, 0] = -2.0 / (dx**2)
        L[0, 1] = 1.0 / (dx**2)
        L[-1, -1] = -2.0 / (dx**2)
        L[-1, -2] = 1.0 / (dx**2)
    else:
        raise ValueError(f"未知的 stencil_type: {stencil_type}")

    return L


def anisotropic_laplacian_5point(V, dx, dy, sigma_xx, sigma_yy):
    V = np.asarray(V, dtype=float)
    nx, ny = V.shape
    L = np.zeros_like(V)


    L[1:-1, 1:-1] = (
        sigma_xx * (V[2:, 1:-1] - 2.0 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / (dx**2)
        + sigma_yy * (V[1:-1, 2:] - 2.0 * V[1:-1, 1:-1] + V[1:-1, :-2]) / (dy**2)
    )


    L[0, :] = L[1, :]
    L[-1, :] = L[-2, :]
    L[:, 0] = L[:, 1]
    L[:, -1] = L[:, -2]

    return L
