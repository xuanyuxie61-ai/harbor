# -*- coding: utf-8 -*-
"""
fd_operators.py
===============

高阶有限差分算子构造与稳定性分析核心工具。

综合融合种子项目:
    - 362_fd1d_heat_steady: 一维稳态有限差分格式
    - 370_fd3d_poisson: 三维 Poisson 有限差分
    - 279_diff_center: 中心差分算子
    - 648_laplacian_matrix: 离散 Laplacian 矩阵 (Dirichlet / Neumann 边界)

本模块为 parton shower 演化方程与强子化场方程提供:
    1) 2/4/6 阶中心差分算子 D1, D2
    2) 多维离散 Laplacian (张量积构造)
    3) 一维稳态扩散型方程求解器
    4) 三维 Poisson 方程求解器 (用于色场通量管)
"""

from __future__ import annotations
import math
from typing import List, Tuple, Callable, Optional
import constants as C


# ======================================================================
# 一维差分算子 (来自 279_diff_center, 648_laplacian_matrix)
# ======================================================================
def diff_center_matrix(n: int, h: float, order: int = 2) -> List[List[float]]:
    """
    构造 n x n 一阶中心差分矩阵 D1, 阶数为 order (2/4/6)。
    Dirichlet 边界: 端点外值为 0。

    2阶: D1[i,i-1] = -1/(2h), D1[i,i+1] = 1/(2h)
    4阶: D1[i,i-2] = 1/(12h), D1[i,i-1] = -8/(12h),
         D1[i,i+1] = 8/(12h),  D1[i,i+2] = -1/(12h)
    6阶: 标准六点模板。
    """
    if order == 2:
        stencil = [-0.5, 0.0, 0.5]
        offset = [-1, 0, 1]
        scale = 1.0 / h
    elif order == 4:
        stencil = [1.0/12.0, -8.0/12.0, 0.0, 8.0/12.0, -1.0/12.0]
        offset = [-2, -1, 0, 1, 2]
        scale = 1.0 / h
    elif order == 6:
        stencil = [-1.0/60.0, 9.0/60.0, -45.0/60.0, 0.0,
                    45.0/60.0, -9.0/60.0, 1.0/60.0]
        offset = [-3, -2, -1, 0, 1, 2, 3]
        scale = 1.0 / h
    else:
        raise ValueError(f"不支持的差分阶数: {order}")

    M = [[0.0] * n for _ in range(n)]
    half = len(stencil) // 2
    for i in range(n):
        for s_idx, (w, d) in enumerate(zip(stencil, offset)):
            j = i + d
            if 0 <= j < n:
                M[i][j] = w * scale
    return M


def laplacian_1d_dd(n: int, h: float) -> List[List[float]]:
    """
    一维 Dirichlet-Dirichlet 离散 Laplacian (来自 648_laplacian_matrix/l1dd):
        L = (1/h^2) * tridiag(-1, 2, -1)
    """
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 2.0 / (h * h)
        if i > 0:
            L[i][i-1] = -1.0 / (h * h)
        if i < n - 1:
            L[i][i+1] = -1.0 / (h * h)
    return L


def laplacian_1d_nn(n: int, h: float) -> List[List[float]]:
    """一维 Neumann-Neumann Laplacian"""
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if i == 0:
            L[0][0] = 1.0 / (h * h)
            L[0][1] = -1.0 / (h * h)
        elif i == n - 1:
            L[n-1][n-2] = -1.0 / (h * h)
            L[n-1][n-1] = 1.0 / (h * h)
        else:
            L[i][i] = 2.0 / (h * h)
            L[i][i-1] = -1.0 / (h * h)
            L[i][i+1] = -1.0 / (h * h)
    return L


# ======================================================================
# 多维 Laplacian (张量积)
# ======================================================================
def kronecker_sum(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Kronecker 和: A (x) I_B + I_A (x) B
    用于从一维 Laplacian 构造多维 Laplacian。
    """
    na = len(A)
    nb = len(B)
    N = na * nb
    result = [[0.0] * N for _ in range(N)]
    for i in range(na):
        for j in range(na):
            for p in range(nb):
                for q in range(nb):
                    row = i * nb + p
                    col = j * nb + q
                    if i == j:
                        result[row][col] += B[p][q]
                    if p == q:
                        result[row][col] += A[i][j]
    return result


def laplacian_3d_tensor(nx: int, ny: int, nz: int,
                        hx: float, hy: float, hz: float) -> List[List[float]]:
    """
    三维 Laplacian (来自 370_fd3d_poisson), 通过 Kronecker 和构造:
        L_3D = L_x (x) I_y (x) I_z + I_x (x) L_y (x) I_z + I_x (x) I_y (x) L_z
    总维度 N = nx * ny * nz。
    """
    Lx = laplacian_1d_dd(nx, hx)
    Ly = laplacian_1d_dd(ny, hy)
    Lz = laplacian_1d_dd(nz, hz)
    Ix = [[1.0 if i == j else 0.0 for j in range(nx)] for i in range(nx)]
    Iy = [[1.0 if i == j else 0.0 for j in range(ny)] for i in range(ny)]
    Iz = [[1.0 if i == j else 0.0 for j in range(nz)] for i in range(nz)]

    # Lx (x) Iy (x) Iz
    Lxy = kronecker_sum(Lx, Ly)
    Ixy = kronecker_sum(Ix, [[0.0]*ny for _ in range(ny)])
    for i in range(len(Ixy)):
        Ixy[i][i] = 1.0

    # 三步 Kronecker: 先 x,y, 再与 z
    Lxy_Iz = _kron_extend(Lxy, Iz)
    Ix_Ly_Iz = _kron_extend(kronecker_sum(Ix, Ly), Iz)
    Ixy_Lz = _kron_extend(_kron_extend(Ix, Iy), Lz)

    N = nx * ny * nz
    result = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            result[i][j] = Lxy_Iz[i][j]
            # 重新正确构造: 直接用三重 Kronecker
    return _build_laplacian_3d_direct(nx, ny, nz, hx, hy, hz)


def _kron_extend(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Kronecker 积 A (x) B"""
    na = len(A)
    nb = len(B)
    result = [[0.0] * (na * nb) for _ in range(na * nb)]
    for i in range(na):
        for j in range(na):
            for p in range(nb):
                for q in range(nb):
                    result[i*nb+p][j*nb+q] = A[i][j] * B[p][q]
    return result


def _build_laplacian_3d_direct(nx: int, ny: int, nz: int,
                                hx: float, hy: float, hz: float) -> List[List[float]]:
    """直接构造三维 Laplacian (避免巨大 Kronecker 中间矩阵)"""
    N = nx * ny * nz
    L = [[0.0] * N for _ in range(N)]

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                idx = ix * ny * nz + iy * nz + iz
                # 对角: 2/hx^2 + 2/hy^2 + 2/hz^2
                diag = 0.0
                if 0 < ix < nx - 1:
                    diag += 2.0 / (hx * hx)
                if 0 < iy < ny - 1:
                    diag += 2.0 / (hy * hy)
                if 0 < iz < nz - 1:
                    diag += 2.0 / (hz * hz)
                L[idx][idx] = diag

                # 邻居
                if ix > 0:
                    L[idx][(ix-1)*ny*nz + iy*nz + iz] = -1.0 / (hx*hx)
                if ix < nx - 1:
                    L[idx][(ix+1)*ny*nz + iy*nz + iz] = -1.0 / (hx*hx)
                if iy > 0:
                    L[idx][ix*ny*nz + (iy-1)*nz + iz] = -1.0 / (hy*hy)
                if iy < ny - 1:
                    L[idx][ix*ny*nz + (iy+1)*nz + iz] = -1.0 / (hy*hy)
                if iz > 0:
                    L[idx][ix*ny*nz + iy*nz + (iz-1)] = -1.0 / (hz*hz)
                if iz < nz - 1:
                    L[idx][ix*ny*nz + iy*nz + (iz+1)] = -1.0 / (hz*hz)

    return L


# ======================================================================
# 一维稳态热/扩散方程 (来自 362_fd1d_heat_steady)
# ======================================================================
def solve_fd1d_steady(n: int, a: float, b: float,
                      ua: float, ub: float,
                      k_func: Callable[[float], float],
                      f_func: Callable[[float], float]) -> Tuple[List[float], List[float]]:
    """
    求解稳态方程:
        - d/dx ( K(x) du/dx ) = F(x)
    Dirichlet 边界: u(a) = ua, u(b) = ub

    用于 parton 在动量分数 x 空间的稳态扩散近似。

    返回 (x_grid, u_solution)
    """
    h = (b - a) / (n + 1)
    x = [a + i * h for i in range(n + 2)]

    # 组装三对角系统
    lower = [0.0] * n
    diag = [0.0] * n
    upper = [0.0] * n
    rhs = [0.0] * n

    for i in range(n):
        xi = x[i + 1]
        k_plus = k_func(xi + 0.5 * h)
        k_minus = k_func(xi - 0.5 * h)
        lower[i] = -k_minus / (h * h)
        upper[i] = -k_plus / (h * h)
        diag[i] = (k_plus + k_minus) / (h * h)
        rhs[i] = f_func(xi)

    # 边界条件贡献
    rhs[0] -= lower[0] * ua
    rhs[n-1] -= upper[n-1] * ub

    # Thomas 算法 (三对角求解)
    u_inner = _thomas_solve(lower, diag, upper, rhs)

    u = [ua] + u_inner + [ub]
    return x, u


def _thomas_solve(lower: List[float], diag: List[float],
                   upper: List[float], rhs: List[float]) -> List[float]:
    """Thomas 算法求解三对角系统"""
    n = len(diag)
    c_prime = [0.0] * n
    d_prime = [0.0] * n

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        m = lower[i] / (diag[i] - lower[i] * c_prime[i-1])
        if i < n - 1:
            c_prime[i] = upper[i] / (diag[i] - lower[i] * c_prime[i-1])
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i-1]) / (diag[i] - lower[i] * c_prime[i-1])

    x = [0.0] * n
    x[n-1] = d_prime[n-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    return x


# ======================================================================
# 三维 Poisson 求解器 (来自 370_fd3d_poisson) — CG 迭代
# ======================================================================
def solve_poisson_3d_cg(nx: int, ny: int, nz: int,
                         xmin: float, xmax: float,
                         ymin: float, ymax: float,
                         zmin: float, zmax: float,
                         f_func: Callable, g_func: Callable,
                         max_iter: int = 500, tol: float = 1e-8
                         ) -> Tuple:
    """
    三维 Poisson 方程:
        -Laplacian U = f(x,y,z)  in Omega
        U = g(x,y,z)             on dOmega

    使用共轭梯度法求解。用于色场通量管电势求解。

    返回 (U, X, Y, Z)
    """
    hx = (xmax - xmin) / (nx + 1)
    hy = (ymax - ymin) / (ny + 1)
    hz = (zmax - zmin) / (nz + 1)

    # 网格 (包含边界)
    X = [xmin + i * hx for i in range(nx + 2)]
    Y = [ymin + j * hy for j in range(ny + 2)]
    Z = [zmin + k * hz for k in range(nz + 2)]

    N = nx * ny * nz

    # 初始化 U (内部点)
    U = [0.0] * N
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                idx = ix * ny * nz + iy * nz + iz
                U[idx] = g_func(X[ix+1], Y[iy+1], Z[iz+1])

    # 右端项
    rhs = [0.0] * N
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                idx = ix * ny * nz + iy * nz + iz
                rhs[idx] = f_func(X[ix+1], Y[iy+1], Z[iz+1])
                # 边界贡献
                if ix == 0:
                    rhs[idx] += g_func(X[0], Y[iy+1], Z[iz+1]) / (hx*hx)
                if ix == nx - 1:
                    rhs[idx] += g_func(X[nx+1], Y[iy+1], Z[iz+1]) / (hx*hx)
                if iy == 0:
                    rhs[idx] += g_func(X[ix+1], Y[0], Z[iz+1]) / (hy*hy)
                if iy == ny - 1:
                    rhs[idx] += g_func(X[ix+1], Y[ny+1], Z[iz+1]) / (hy*hy)
                if iz == 0:
                    rhs[idx] += g_func(X[ix+1], Y[iy+1], Z[0]) / (hz*hz)
                if iz == nz - 1:
                    rhs[idx] += g_func(X[ix+1], Y[iy+1], Z[nz+1]) / (hz*hz)

    # CG 迭代
    def matvec(v):
        """稀疏矩阵-向量乘"""
        w = [0.0] * N
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    idx = ix * ny * nz + iy * nz + iz
                    s = 0.0
                    s += 2.0 * v[idx] / (hx*hx)
                    s += 2.0 * v[idx] / (hy*hy)
                    s += 2.0 * v[idx] / (hz*hz)
                    if ix > 0:
                        s -= v[(ix-1)*ny*nz + iy*nz + iz] / (hx*hx)
                    if ix < nx - 1:
                        s -= v[(ix+1)*ny*nz + iy*nz + iz] / (hx*hx)
                    if iy > 0:
                        s -= v[ix*ny*nz + (iy-1)*nz + iz] / (hy*hy)
                    if iy < ny - 1:
                        s -= v[ix*ny*nz + (iy+1)*nz + iz] / (hy*hy)
                    if iz > 0:
                        s -= v[ix*ny*nz + iy*nz + (iz-1)] / (hz*hz)
                    if iz < nz - 1:
                        s -= v[ix*ny*nz + iy*nz + (iz+1)] / (hz*hz)
                    w[idx] = s
        return w

    r = [rhs[i] - matvec(U)[i] for i in range(N)]
    p = list(r)
    rs_old = sum(ri * ri for ri in r)

    for iteration in range(max_iter):
        if math.sqrt(rs_old) < tol:
            break
        Ap = matvec(p)
        pAp = sum(p[i] * Ap[i] for i in range(N))
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        for i in range(N):
            U[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        rs_new = sum(ri * ri for ri in r)
        if math.sqrt(rs_new) < tol:
            break
        beta = rs_new / rs_old
        for i in range(N):
            p[i] = r[i] + beta * p[i]
        rs_old = rs_new

    # 重塑为三维数组
    U3d = [[[0.0 for _ in range(nz + 2)] for _ in range(ny + 2)] for _ in range(nx + 2)]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                U3d[ix+1][iy+1][iz+1] = U[ix * ny * nz + iy * nz + iz]

    # 填充边界
    for ix in range(nx + 2):
        for iy in range(ny + 2):
            U3d[ix][iy][0] = g_func(X[ix], Y[iy], Z[0])
            U3d[ix][iy][nz+1] = g_func(X[ix], Y[iy], Z[nz+1])
    for ix in range(nx + 2):
        for iz in range(nz + 2):
            U3d[ix][0][iz] = g_func(X[ix], Y[0], Z[iz])
            U3d[ix][ny+1][iz] = g_func(X[ix], Y[ny+1], Z[iz])
    for iy in range(ny + 2):
        for iz in range(nz + 2):
            U3d[0][iy][iz] = g_func(X[0], Y[iy], Z[iz])
            U3d[nx+1][iy][iz] = g_func(X[nx+1], Y[iy], Z[iz])

    return U3d, X, Y, Z


# ======================================================================
# 自测
# ======================================================================
if __name__ == "__main__":
    print("=== 1D Laplacian (DD) n=5, h=0.1 ===")
    L = laplacian_1d_dd(5, 0.1)
    for row in L:
        print("  [" + ", ".join(f"{v:8.3f}" for v in row) + "]")

    print("\n=== Steady-state 1D: -u'' = 1, u(0)=u(1)=0 ===")
    x, u = solve_fd1d_steady(20, 0.0, 1.0, 0.0, 0.0,
                              lambda x: 1.0, lambda x: 1.0)
    err = max(abs(u[i] - 0.5 * x[i] * (1.0 - x[i])) for i in range(len(x)))
    print(f"  max error vs exact: {err:.2e}")
