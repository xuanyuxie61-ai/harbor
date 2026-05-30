
import numpy as np
from numpy.polynomial import polynomial as P


def vandermonde_like(stencil):
    n = len(stencil)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = stencil[j] ** i
    return A


def differ_stencil(x0, order, precision, x):
    n = order + precision
    if len(x) != n:
        raise ValueError("模板点数量必须等于 order + precision")
    dx = x - x0
    A = vandermonde_like(dx)
    b = np.zeros(n)
    b[order] = 1.0

    c = np.linalg.solve(A, b)
    c = c * np.math.factorial(order)
    return c


def build_laplacian_1d(N, dx, bc_type='dirichlet'):
    A = np.zeros((N, N))
    coeff = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * dx ** 2)
    for i in range(2, N - 2):
        A[i, i - 2:i + 3] = coeff


    bc_coeff = np.array([1.0, -2.0, 1.0]) / (dx ** 2)
    A[0, 0:3] = bc_coeff
    A[1, 0:3] = bc_coeff
    A[N - 2, N - 3:N] = bc_coeff
    A[N - 1, N - 3:N] = bc_coeff

    if bc_type == 'dirichlet':

        A[0, :] = 0.0
        A[0, 0] = 1.0
        A[N - 1, :] = 0.0
        A[N - 1, N - 1] = 1.0

    return A


def build_laplacian_3d(Nx, Ny, Nz, dx, dy, dz):
    Lx = build_laplacian_1d(Nx, dx, bc_type='neumann')
    Ly = build_laplacian_1d(Ny, dy, bc_type='neumann')
    Lz = build_laplacian_1d(Nz, dz, bc_type='neumann')


    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    Iz = np.eye(Nz)


    L = (np.kron(np.kron(Lx, Iy), Iz) +
         np.kron(np.kron(Ix, Ly), Iz) +
         np.kron(np.kron(Ix, Iy), Lz))
    return L


def apply_laplacian_3d(phi, dx, dy, dz):
    Nx, Ny, Nz = phi.shape
    out = np.zeros_like(phi)





    raise NotImplementedError("Hole 1: 请实现 apply_laplacian_3d 的内部点计算")


    out[0, :, :] = out[1, :, :]
    out[-1, :, :] = out[-2, :, :]
    out[:, 0, :] = out[:, 1, :]
    out[:, -1, :] = out[:, -2, :]
    out[:, :, 0] = out[:, :, 1]
    out[:, :, -1] = out[:, :, -2]

    return out
