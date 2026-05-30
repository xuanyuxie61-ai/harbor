
import numpy as np
from typing import Tuple, Optional
from sparse_matrix import R8NCFSparseMatrix


def sor_solve(
    A: R8NCFSparseMatrix,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    omega: float = 1.5,
    tol: float = 1e-8,
    max_iter: int = 5000,
) -> Tuple[np.ndarray, int, float]:
    b = np.asarray(b, dtype=float)
    n = A.n_rows
    if b.shape[0] != n:
        raise ValueError("b 的长度必须与矩阵行数一致")
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
        if x.shape[0] != n:
            raise ValueError("x0 的长度必须与矩阵行数一致")
    if omega <= 0.0 or omega >= 2.0:
        raise ValueError("松弛因子 omega 必须在 (0, 2) 区间内")


    diag_vals = np.zeros(n, dtype=float)
    row_entries = [[] for _ in range(n)]
    for k in range(A.nz_num):
        i = A.rowcol[0, k]
        j = A.rowcol[1, k]
        val = A.values[k]
        if i == j:
            diag_vals[i] += val
        else:
            row_entries[i].append((j, val))


    for i in range(n):
        if abs(diag_vals[i]) < 1e-15:
            raise ValueError(f"第 {i} 行对角元接近零，SOR 不适用")

    for it in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = 0.0
            for j, val in row_entries[i]:
                sigma += val * x[j]
            x[i] = (1.0 - omega) * x[i] + (omega / diag_vals[i]) * (b[i] - sigma)

        diff = np.linalg.norm(x - x_old)
        if diff < tol:
            res = np.linalg.norm(A.mv(x) - b)
            return x, it + 1, res
    res = np.linalg.norm(A.mv(x) - b)
    return x, max_iter, res


def jacobi_solve_2d_poisson(
    nx: int,
    ny: int,
    rhs: np.ndarray,
    u_exact: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 20000,
) -> Tuple[np.ndarray, float]:
    if nx < 3 or ny < 3:
        raise ValueError("nx 和 ny 至少为 3")
    rhs = np.asarray(rhs, dtype=float)
    if rhs.shape != (nx, ny):
        raise ValueError(f"rhs 形状必须为 ({nx}, {ny})")

    hx = 1.0 / (nx - 1)
    hy = 1.0 / (ny - 1)
    factor = hx * hy

    u = np.zeros((nx, ny), dtype=float)

    if u_exact is not None:
        u[0, :] = u_exact[0, :]
        u[-1, :] = u_exact[-1, :]
        u[:, 0] = u_exact[:, 0]
        u[:, -1] = u_exact[:, -1]

    for it in range(max_iter):
        u_new = u.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_new[i, j] = 0.25 * (
                    u[i - 1, j]
                    + u[i + 1, j]
                    + u[i, j - 1]
                    + u[i, j + 1]
                    + rhs[i, j] * factor
                )
        diff = np.max(np.abs(u_new - u))
        u = u_new
        if diff < tol:
            break

    error = -1.0
    if u_exact is not None:
        error = np.sqrt(np.mean((u - u_exact) ** 2))
    return u, error


def solve_pressure_poisson_sor(
    velocity_field: np.ndarray,
    dx: float,
    dy: float,
    rho: float = 1025.0,
    omega: float = 1.6,
    tol: float = 1e-7,
    max_iter: int = 8000,
) -> np.ndarray:
    if velocity_field.ndim != 2:
        raise ValueError("速度场必须是二维数组")
    nx, ny = velocity_field.shape
    if nx < 3 or ny < 3:
        raise ValueError("速度场维度至少为 3×3")
    if dx <= 0 or dy <= 0:
        raise ValueError("网格步长必须为正")


    rhs = np.zeros((nx, ny), dtype=float)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dudx = (velocity_field[i + 1, j] - velocity_field[i - 1, j]) / (2.0 * dx)
            dudy = (velocity_field[i, j + 1] - velocity_field[i, j - 1]) / (2.0 * dy)
            rhs[i, j] = rho * (dudx ** 2 + dudy ** 2)


    from sparse_matrix import build_laplacian_2d_sparse
    n = nx * ny
    A = build_laplacian_2d_sparse(nx, ny, dx, dy)
    b = rhs.flatten()


    x, iters, res = sor_solve(A, b, omega=omega, tol=tol, max_iter=max_iter)
    pressure = x.reshape((nx, ny))
    return pressure


def solve_laplace_velocity_potential(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    body_bc: np.ndarray,
    free_surface_bc: Optional[np.ndarray] = None,
    tol: float = 1e-7,
    max_iter: int = 10000,
) -> np.ndarray:
    if nx < 3 or ny < 3:
        raise ValueError("网格维度至少为 3×3")
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    factor = dx * dy

    phi = np.zeros((nx, ny), dtype=float)

    phi[:, 0] = 0.0
    if body_bc is not None:
        if body_bc.shape != (nx,):
            raise ValueError("body_bc 长度必须与 nx 一致")
        phi[:, -1] = body_bc
    else:
        phi[:, -1] = 0.0


    k = 2.0 * np.pi / Lx
    for j in range(ny):
        z = j * dy
        phi[0, j] = np.sin(k * 0.0) * np.cosh(k * z)

    for it in range(max_iter):
        phi_new = phi.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                phi_new[i, j] = 0.25 * (
                    phi[i - 1, j] + phi[i + 1, j] + phi[i, j - 1] + phi[i, j + 1]
                )

        for j in range(1, ny - 1):
            phi_new[-1, j] = phi_new[-2, j] / (1.0 + k * dx)
        diff = np.max(np.abs(phi_new - phi))
        phi = phi_new
        if diff < tol:
            break
    return phi


def compute_velocity_from_potential(
    phi: np.ndarray, dx: float, dy: float
) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny = phi.shape
    u = np.zeros_like(phi)
    v = np.zeros_like(phi)
    for i in range(nx):
        for j in range(ny):
            if i == 0:
                u[i, j] = (phi[i + 1, j] - phi[i, j]) / dx
            elif i == nx - 1:
                u[i, j] = (phi[i, j] - phi[i - 1, j]) / dx
            else:
                u[i, j] = (phi[i + 1, j] - phi[i - 1, j]) / (2.0 * dx)
            if j == 0:
                v[i, j] = (phi[i, j + 1] - phi[i, j]) / dy
            elif j == ny - 1:
                v[i, j] = (phi[i, j] - phi[i, j - 1]) / dy
            else:
                v[i, j] = (phi[i, j + 1] - phi[i, j - 1]) / (2.0 * dy)
    return u, v
