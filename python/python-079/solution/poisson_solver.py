"""
泊松方程求解模块

基于种子项目：
  - 877_poisson_2d：二维五点差分离散 + Jacobi 迭代
  - 1099_sor：逐次超松弛 (SOR) 线性求解器

核心物理模型：
  1. 压力泊松方程（海洋平台周围流场压力修正）：
       ∇²p = ρ (∂u_i/∂x_j)(∂u_j/∂x_i) - ∂F_i/∂x_i
     其中 ρ = 1025 kg/m³ 为海水密度，u 为速度场，F 为体积力。

  2. 速度势 Laplace 方程（线性波浪理论）：
       ∇²φ = 0
     边界条件：
       - 自由表面：∂φ/∂z = (1/g) ∂²φ/∂t²   (线性化动力学条件)
       - 物面：∂φ/∂n = U_n                 (不可穿透条件)
       - 底部：∂φ/∂n = 0                   (固壁条件)

  3. 离散化：
       - 二维区域 [0, Lx] × [0, Ly] 上均匀网格，步长 dx, dy。
       - 五点差分格式：
           (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_{i,j}) / (dx·dy) = f_{i,j}
       - SOR 迭代加速收敛：
           u^{new}_i = (1-ω) u_i + (ω/a_{ii}) [b_i - Σ_{j<i} a_{ij} u^{new}_j - Σ_{j>i} a_{ij} u^{old}_j]
"""

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
    """
    使用 SOR 迭代求解线性方程组 A x = b。
    矩阵 A 通过坐标格式访问，Gauss-Seidel 风格逐分量更新。

    参数
    ----
    A : R8NCFSparseMatrix，系数矩阵
    b : 右端项向量
    x0 : 初始猜测（默认零向量）
    omega : 松弛因子 (0, 2)
    tol : 残差收敛容差
    max_iter : 最大迭代次数

    返回
    ----
    x : 解向量
    iters : 实际迭代次数
    residual : 最终残差范数
    """
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

    # 预处理：提取每行的对角元位置及非对角元列表
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

    # 检查对角元非零
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
        # 收敛判断
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
    """
    使用 Jacobi 迭代求解二维泊松方程 -∇²u = f 在 [0,1]×[0,1] 上。
    五点差分格式，均匀网格，步长 hx = 1/(nx-1), hy = 1/(ny-1)。

    参数
    ----
    nx, ny : x 和 y 方向的网格点数（含边界）
    rhs : 右端项 f 的 (nx, ny) 数组
    u_exact : 精确解（用于误差估计，可选）
    tol : 收敛容差
    max_iter : 最大迭代次数

    返回
    ----
    u : (nx, ny) 的解数组
    error : 若提供 u_exact 则返回 L2 误差，否则返回 -1
    """
    if nx < 3 or ny < 3:
        raise ValueError("nx 和 ny 至少为 3")
    rhs = np.asarray(rhs, dtype=float)
    if rhs.shape != (nx, ny):
        raise ValueError(f"rhs 形状必须为 ({nx}, {ny})")

    hx = 1.0 / (nx - 1)
    hy = 1.0 / (ny - 1)
    factor = hx * hy

    u = np.zeros((nx, ny), dtype=float)
    # 若提供精确解，设置 Dirichlet 边界条件
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
    """
    求解海洋平台绕流的压力泊松方程：
        ∇²p = ρ * [ (∂u/∂x)² + 2(∂u/∂y)(∂v/∂x) + (∂v/∂y)² ]
    假设给定二维速度场 (u, v) 的合速度标量场。
    使用 SOR 迭代求解离散化后的线性系统。
    """
    if velocity_field.ndim != 2:
        raise ValueError("速度场必须是二维数组")
    nx, ny = velocity_field.shape
    if nx < 3 or ny < 3:
        raise ValueError("速度场维度至少为 3×3")
    if dx <= 0 or dy <= 0:
        raise ValueError("网格步长必须为正")

    # 计算右端项：速度梯度张量的 Frobenius 范数平方
    rhs = np.zeros((nx, ny), dtype=float)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dudx = (velocity_field[i + 1, j] - velocity_field[i - 1, j]) / (2.0 * dx)
            dudy = (velocity_field[i, j + 1] - velocity_field[i, j - 1]) / (2.0 * dy)
            rhs[i, j] = rho * (dudx ** 2 + dudy ** 2)

    # 构建二维 Laplacian 稀疏矩阵
    from sparse_matrix import build_laplacian_2d_sparse
    n = nx * ny
    A = build_laplacian_2d_sparse(nx, ny, dx, dy)
    b = rhs.flatten()

    # SOR 求解
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
    """
    求解海洋平台周围二维速度势 Laplace 方程 ∇²φ = 0。
    边界条件：
      - 左边界 (x=0)：入射波势 φ = sin(kx) cosh(k(z+h)) / cosh(kh)
      - 右边界 (x=Lx)：辐射边界 ∂φ/∂x = -ikφ
      - 下边界 (y=0)：固壁 ∂φ/∂y = 0
      - 上边界 (y=Ly)：物面不可穿透条件，由 body_bc 给定

    使用 Jacobi 迭代求解。
    """
    if nx < 3 or ny < 3:
        raise ValueError("网格维度至少为 3×3")
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    factor = dx * dy

    phi = np.zeros((nx, ny), dtype=float)
    # 设置边界条件
    phi[:, 0] = 0.0  # 底部固壁
    if body_bc is not None:
        if body_bc.shape != (nx,):
            raise ValueError("body_bc 长度必须与 nx 一致")
        phi[:, -1] = body_bc  # 物面边界
    else:
        phi[:, -1] = 0.0

    # 入射波边界（左边界）
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
        # 辐射边界（右边界）：一阶吸收边界条件 ∂φ/∂x ≈ -kφ
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
    """
    由速度势 φ 计算速度分量 (u, v) = (∂φ/∂x, ∂φ/∂y)。
    使用中心差分，边界处使用前/后差分。
    """
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
