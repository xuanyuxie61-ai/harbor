
import numpy as np




class R83VOperator:

    def __init__(self, a, b, c):
        self.a = np.array(a, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.c = np.array(c, dtype=np.float64)
        self.N = len(b)
        if len(self.a) != self.N - 1 or len(self.c) != self.N - 1:
            raise ValueError("Inconsistent diagonal lengths for R83V format.")

    def matvec(self, x):
        if len(x) != self.N:
            raise ValueError("Dimension mismatch in R83V matvec.")
        y = self.b * x
        if self.N > 1:
            y[1:] += self.a * x[:-1]
            y[:-1] += self.c * x[1:]
        return y

    def transpose_matvec(self, x):
        return self.matvec(x)

    def residual(self, x, b_vec):
        return b_vec - self.matvec(x)

    def jacobi_iterate(self, x, b_vec, omega=1.0):
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
        N = self.N
        if N <= 1:
            if abs(self.b[0]) < 1e-30:
                return np.array([0.0])
            return np.array([b_vec[0] / self.b[0]])


        a = np.concatenate([[0.0], self.a])
        c = np.concatenate([self.c, [0.0]])
        d = self.b.copy()
        rhs = b_vec.copy()

        n = N
        systems = [(a.copy(), d.copy(), c.copy(), rhs.copy(), n)]


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


        if abs(d[0]) < 1e-30:
            d[0] = 1e-30
        x = np.array([rhs[0] / d[0]])


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




def build_southwell_matrix_1d(N, dx):
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
    N = len(slopes) + 1
    A = build_southwell_matrix_1d(N, dx)

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
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2.")


    n_subap = int(np.sqrt(len(subaps)))
    if n_subap < 1:
        n_subap = 1

    phi_coarse_x = np.zeros((grid_size, grid_size), dtype=np.float64)
    phi_coarse_y = np.zeros((grid_size, grid_size), dtype=np.float64)


    for idx, (rs, re, cs, ce) in enumerate(subaps):
        cx = (rs + re) // 2
        cy = (cs + ce) // 2
        if cx < grid_size and cy < grid_size:
            phi_coarse_x[cx, cy] = sx[idx]
            phi_coarse_y[cx, cy] = sy[idx]


    phi_rows = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(grid_size):

        s_row = np.zeros(grid_size - 1, dtype=np.float64)
        for j in range(grid_size - 1):
            s_row[j] = phi_coarse_x[i, min(j, grid_size - 1)]
        if np.all(np.abs(s_row) < 1e-20):
            s_row[0] = 1e-10
        row_recon = reconstruct_wavefront_1d(s_row, pixel_scale, method=method)
        phi_rows[i, :] = row_recon


    phi = phi_rows.copy()
    for j in range(grid_size):
        s_col = np.zeros(grid_size - 1, dtype=np.float64)
        for i in range(grid_size - 1):
            s_col[i] = phi_coarse_y[min(i, grid_size - 1), j]
        if np.all(np.abs(s_col) < 1e-20):
            s_col[0] = 1e-10
        col_recon = reconstruct_wavefront_1d(s_col, pixel_scale, method=method)
        phi[:, j] = 0.5 * (phi[:, j] + col_recon)


    phi -= np.mean(phi)
    return phi


def reconstruct_wavefront_modal(sx, sy, subaps, basis_flat, mask, pixel_scale):







    raise NotImplementedError("Hole 2: 请实现 reconstruct_wavefront_modal 函数体.")
