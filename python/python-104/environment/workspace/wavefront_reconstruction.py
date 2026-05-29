"""
wavefront_reconstruction.py — 波前重构算法库

融合原项目: 967_r83v (R83V三对角矩阵存储格式的线性代数求解器)

功能:
  - 从Shack-Hartmann斜率向量重构波前相位
  - 基于R83V格式的三对角系统快速求解 (CG, Jacobi, GS, CR)
  - Southwell型离散化: 相位与斜率的关系
  -  modal重构 (Zernike基) 和 zonal重构 (直接相位恢复)

物理模型:
  1. Southwell离散化:
       (phi_{i+1,j} - phi_{i,j}) / dx = s_x(i+0.5, j)
       (phi_{i,j+1} - phi_{i,j}) / dy = s_y(i, j+0.5)

  2. 最小二乘重构问题:
       minimize || G * phi - s ||^2
     其正规方程为:
       (G^T G) phi = G^T s
     矩阵 A = G^T G 为稀疏对称正定矩阵, 具有三对角块结构.

  3. R83V格式求解:
       将A存储为三对角形式 (sub-diag, main-diag, super-diag),
       使用共轭梯度法 (CG)、Jacobi迭代、Gauss-Seidel迭代或循环约化 (CR) 求解.
"""

import numpy as np


# --- R83V格式线性代数操作 (源自967_r83v) ---

class R83VOperator:
    """
    R83V格式的三对角矩阵操作类.

    对于N x N三对角矩阵 A, 存储为:
      a: 次对角线 (长度 N-1),  a[i] = A[i+1, i]
      b: 主对角线 (长度 N),    b[i] = A[i, i]
      c: 超对角线 (长度 N-1),  c[i] = A[i, i+1]
    """

    def __init__(self, a, b, c):
        self.a = np.array(a, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.c = np.array(c, dtype=np.float64)
        self.N = len(b)
        if len(self.a) != self.N - 1 or len(self.c) != self.N - 1:
            raise ValueError("Inconsistent diagonal lengths for R83V format.")

    def matvec(self, x):
        """矩阵-向量乘法 y = A @ x."""
        if len(x) != self.N:
            raise ValueError("Dimension mismatch in R83V matvec.")
        y = self.b * x
        if self.N > 1:
            y[1:] += self.a * x[:-1]
            y[:-1] += self.c * x[1:]
        return y

    def transpose_matvec(self, x):
        """转置矩阵-向量乘法 y = A^T @ x (对称时同matvec)."""
        return self.matvec(x)

    def residual(self, x, b_vec):
        """计算残差 r = b - A @ x."""
        return b_vec - self.matvec(x)

    def jacobi_iterate(self, x, b_vec, omega=1.0):
        """
        Jacobi迭代一步:
          x_new[i] = (1-omega)*x[i] + omega*(b[i] - a[i-1]*x[i-1] - c[i]*x[i+1]) / b[i]
        """
        x_new = np.zeros_like(x)
        for i in range(self.N):
            sigma = 0.0
            if i > 0:
                sigma += self.a[i - 1] * x[i - 1]
            if i < self.N - 1:
                sigma += self.c[i] * x[i + 1]
            if abs(self.b[i]) < 1e-30:
                x_new[i] = x[i]
            else:
                x_new[i] = (1.0 - omega) * x[i] + omega * (b_vec[i] - sigma) / self.b[i]
        return x_new

    def gauss_seidel_iterate(self, x, b_vec, omega=1.0):
        """
        Gauss-Seidel迭代一步 (SOR):
          使用前向扫描, 已更新的分量立即参与计算.
        """
        x_new = x.copy()
        for i in range(self.N):
            sigma = 0.0
            if i > 0:
                sigma += self.a[i - 1] * x_new[i - 1]
            if i < self.N - 1:
                sigma += self.c[i] * x[i + 1]
            if abs(self.b[i]) < 1e-30:
                continue
            x_new[i] = (1.0 - omega) * x[i] + omega * (b_vec[i] - sigma) / self.b[i]
        return x_new

    def conjugate_gradient_solve(self, b_vec, x0=None, tol=1e-10, max_iter=None):
        """
        共轭梯度法 (CG) 求解 A x = b.

        CG算法:
          r_0 = b - A x_0,  p_0 = r_0
          alpha_k = (r_k^T r_k) / (p_k^T A p_k)
          x_{k+1} = x_k + alpha_k p_k
          r_{k+1} = r_k - alpha_k A p_k
          beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
          p_{k+1} = r_{k+1} + beta_k p_k
        """
        N = self.N
        if max_iter is None:
            max_iter = N
        if x0 is None:
            x = np.zeros(N, dtype=np.float64)
        else:
            x = np.array(x0, dtype=np.float64)

        r = b_vec - self.matvec(x)
        p = r.copy()
        rsold = np.dot(r, r)

        for _ in range(max_iter):
            Ap = self.matvec(p)
            pAp = np.dot(p, Ap)
            if abs(pAp) < 1e-30:
                break
            alpha = rsold / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = np.dot(r, r)
            if np.sqrt(rsnew) < tol:
                break
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew

        return x

    def cyclic_reduction_solve(self, b_vec):
        """
        循环约化法 (Cyclic Reduction, CR) 求解三对角系统.

        基于Hockney (1965) 算法:
          对三对角系统, 通过奇偶重排消去约半数未知数,
          递归直到系统规模足够小, 然后回代.

        要求矩阵严格对角占优.
        """
        N = self.N
        if N <= 1:
            if abs(self.b[0]) < 1e-30:
                return np.array([0.0])
            return np.array([b_vec[0] / self.b[0]])

        # 拷贝系数
        a = np.concatenate([[0.0], self.a])
        c = np.concatenate([self.c, [0.0]])
        d = self.b.copy()
        rhs = b_vec.copy()

        n = N
        systems = [(a.copy(), d.copy(), c.copy(), rhs.copy(), n)]

        # 前向约化
        while n > 1:
            a_new = np.zeros(n // 2)
            d_new = np.zeros(n // 2)
            c_new = np.zeros(n // 2)
            rhs_new = np.zeros(n // 2)

            for i in range(1, n, 2):
                idx = i // 2
                if abs(d[i]) < 1e-30:
                    d[i] = 1e-30
                alpha = a[i] / d[i]
                gamma = c[i] / d[i]

                d_new[idx] = d[i - 1] - alpha * a[i]
                if i + 1 < n:
                    d_new[idx] -= gamma * c[i]
                    c_new[idx] = c[i - 1]
                if i - 2 >= 0:
                    a_new[idx] = a[i - 1]
                rhs_new[idx] = rhs[i - 1] - alpha * rhs[i]
                if i + 1 < n:
                    rhs_new[idx] -= gamma * rhs[i + 1]

            systems.append((a_new.copy(), d_new.copy(), c_new.copy(), rhs_new.copy(), n // 2))
            a, d, c, rhs = a_new, d_new, c_new, rhs_new
            n = n // 2

        # 求解最小系统
        if abs(d[0]) < 1e-30:
            d[0] = 1e-30
        x = np.array([rhs[0] / d[0]])

        # 回代
        for level in range(len(systems) - 2, -1, -1):
            a_lvl, d_lvl, c_lvl, rhs_lvl, n_lvl = systems[level]
            x_new = np.zeros(n_lvl)
            for i in range(0, n_lvl, 2):
                x_new[i] = (rhs_lvl[i] - (c_lvl[i] * x[i // 2] if i + 1 < n_lvl else 0.0)) / d_lvl[i]
                if i + 1 < n_lvl:
                    val = rhs_lvl[i + 1]
                    if i >= 0:
                        val -= a_lvl[i + 1] * x_new[i]
                    if i + 2 < len(x):
                        val -= c_lvl[i + 1] * x[(i + 2) // 2]
                    if abs(d_lvl[i + 1]) < 1e-30:
                        d_lvl[i + 1] = 1e-30
                    x_new[i + 1] = val / d_lvl[i + 1]
            x = x_new

        return x[:N]


# --- 波前重构核心 ---

def build_southwell_matrix_1d(N, dx):
    """
    构建一维Southwell离散化的三对角矩阵.

    对于一维相位 phi[0..N-1], 斜率 s[i] = (phi[i+1] - phi[i]) / dx,
    最小二乘目标: minimize sum_i ( (phi[i+1]-phi[i])/dx - s[i] )^2
    正规方程矩阵 A 为三对角:
      b[0] = 1/dx^2,  b[N-1] = 1/dx^2
      b[i] = 2/dx^2  (i=1..N-2)
      a[i] = -1/dx^2
      c[i] = -1/dx^2
    """
    if N < 2:
        raise ValueError("N must be >= 2.")
    if dx <= 0:
        raise ValueError("dx must be positive.")
    inv_dx2 = 1.0 / (dx ** 2)
    a = -inv_dx2 * np.ones(N - 1, dtype=np.float64)
    b = 2.0 * inv_dx2 * np.ones(N, dtype=np.float64)
    b[0] = inv_dx2
    b[-1] = inv_dx2
    c = -inv_dx2 * np.ones(N - 1, dtype=np.float64)
    return R83VOperator(a, b, c)


def reconstruct_wavefront_1d(slopes, dx, method='cg'):
    """
    一维波前重构.

    方法:
      'cg': 共轭梯度法
      'jacobi': Jacobi迭代
      'gs': Gauss-Seidel迭代
      'cr': 循环约化法
    """
    N = len(slopes) + 1
    A = build_southwell_matrix_1d(N, dx)
    # 右端项: b[i] = (s[i] - s[i-1]) / dx, 边界处理
    b_vec = np.zeros(N, dtype=np.float64)
    b_vec[0] = slopes[0] / dx
    b_vec[-1] = -slopes[-1] / dx
    for i in range(1, N - 1):
        b_vec[i] = (slopes[i] - slopes[i - 1]) / dx

    if method == 'cg':
        return A.conjugate_gradient_solve(b_vec)
    elif method == 'cr':
        return A.cyclic_reduction_solve(b_vec)
    elif method == 'jacobi':
        x = np.zeros(N, dtype=np.float64)
        for _ in range(5000):
            x_new = A.jacobi_iterate(x, b_vec, omega=1.0)
            if np.linalg.norm(x_new - x) < 1e-10:
                break
            x = x_new
        return x
    elif method == 'gs':
        x = np.zeros(N, dtype=np.float64)
        for _ in range(5000):
            x_new = A.gauss_seidel_iterate(x, b_vec, omega=1.5)
            if np.linalg.norm(x_new - x) < 1e-10:
                break
            x = x_new
        return x
    else:
        raise ValueError("Unknown method.")


def reconstruct_wavefront_zonal(sx, sy, subaps, grid_size, pixel_scale, method='cg'):
    """
    二维zonal波前重构 (基于Southwell离散化).

    将二维相位展平为 Npix = grid_size^2 维向量,
    使用R83V格式求解正规方程.

    为了效率, 采用逐行/逐列扫描的交替方向隐式 (ADI) 策略,
    每一维的求解退化为三对角系统.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2.")

    # 先在每个子孔径位置建立粗网格相位
    n_subap = int(np.sqrt(len(subaps)))
    if n_subap < 1:
        n_subap = 1

    phi_coarse_x = np.zeros((grid_size, grid_size), dtype=np.float64)
    phi_coarse_y = np.zeros((grid_size, grid_size), dtype=np.float64)

    # 将斜率映射回粗网格
    for idx, (rs, re, cs, ce) in enumerate(subaps):
        cx = (rs + re) // 2
        cy = (cs + ce) // 2
        if cx < grid_size and cy < grid_size:
            phi_coarse_x[cx, cy] = sx[idx]
            phi_coarse_y[cx, cy] = sy[idx]

    # 逐行积分 (x方向)
    phi_rows = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(grid_size):
        # 从粗网格采样斜率
        s_row = np.zeros(grid_size - 1, dtype=np.float64)
        for j in range(grid_size - 1):
            s_row[j] = phi_coarse_x[i, min(j, grid_size - 1)]
        if np.all(np.abs(s_row) < 1e-20):
            s_row[0] = 1e-10
        row_recon = reconstruct_wavefront_1d(s_row, pixel_scale, method=method)
        phi_rows[i, :] = row_recon

    # 逐列积分 (y方向), 修正
    phi = phi_rows.copy()
    for j in range(grid_size):
        s_col = np.zeros(grid_size - 1, dtype=np.float64)
        for i in range(grid_size - 1):
            s_col[i] = phi_coarse_y[min(i, grid_size - 1), j]
        if np.all(np.abs(s_col) < 1e-20):
            s_col[0] = 1e-10
        col_recon = reconstruct_wavefront_1d(s_col, pixel_scale, method=method)
        phi[:, j] = 0.5 * (phi[:, j] + col_recon)

    # 移除 piston
    phi -= np.mean(phi)
    return phi


def reconstruct_wavefront_modal(sx, sy, subaps, basis_flat, mask, pixel_scale):
    """
    Modal波前重构: 将斜率投影到Zernike基上.

    斜率向量 s = [sx; sy].
    对于每个模式 j, 其理论斜率为:
      s_x^{(j)}(k) = (1/A_k) * integral_{subap_k} dZ_j/dx  dA
      s_y^{(j)}(k) = (1/A_k) * integral_{subap_k} dZ_j/dy  dA

    求解最小二乘:  c = argmin || D @ c - s ||^2
    其中 D 为斜率-模式响应矩阵.
    """
    # TODO: 实现Modal波前重构
    # 需要:
    #   1. 将 basis_flat 重塑为网格上的模式图像
    #   2. 对每个模式数值计算梯度 (dZ/dx, dZ/dy)
    #   3. 在每个子孔径内平均梯度, 构建响应矩阵 D
    #   4. 最小二乘求解系数 c = lstsq(D, [sx; sy])
    # 注意: basis_flat 的形状与 zernike_modes.compute_zernike_basis 的返回格式一致
    raise NotImplementedError("Hole 2: 请实现 reconstruct_wavefront_modal 函数体.")
