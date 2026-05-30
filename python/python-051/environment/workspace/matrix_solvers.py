
import numpy as np






def bicg_solve(A, b, x0=None, tol=1e-8, max_iter=1000):
    n = b.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    bnrm = np.linalg.norm(b)
    if bnrm == 0.0:
        bnrm = 1.0

    r = b - A.dot(x)
    error = np.linalg.norm(r) / bnrm
    if error < tol:
        return x

    r_tld = r.copy()
    rho_old = 1.0
    p = np.zeros(n)
    p_tld = np.zeros(n)

    for it in range(1, max_iter + 1):
        z = r.copy()
        z_tld = r_tld.copy()
        rho = np.dot(z, r_tld)
        if abs(rho) < 1e-30:
            break

        if it == 1:
            p = z
            p_tld = z_tld
        else:
            beta = rho / rho_old
            p = z + beta * p
            p_tld = z_tld + beta * p_tld

        q = A.dot(p)
        q_tld = A.T.dot(p_tld)
        denom = np.dot(p_tld, q)
        if abs(denom) < 1e-30:
            break
        alpha = rho / denom

        x = x + alpha * p
        r = r - alpha * q
        r_tld = r_tld - alpha * q_tld

        error = np.linalg.norm(r) / bnrm
        if error <= tol:
            break
        rho_old = rho

    return x






def r8sto_yw_sl(n, a):
    if n == 0:
        return np.array([])
    x = np.zeros(n)
    x[0] = -a[0]
    if n == 1:
        return x

    alpha = -a[0]
    beta = 1.0

    for i in range(1, n):

        num = -(a[i] + np.dot(a[:i][::-1], x[:i]))
        den = beta
        if abs(den) < 1e-30:
            den = 1e-30
        kappa = num / den


        x_old = x[:i].copy()
        x[:i] = x_old + kappa * x_old[::-1]
        x[i] = kappa


        beta = beta * (1.0 - kappa ** 2)
        if abs(beta) < 1e-30:
            beta = 1e-30

    return x


def r8sto_inverse(n, a_row):
    if n < 1:
        raise ValueError("n >= 1")
    if a_row.shape[0] < n:
        raise ValueError("a_row 长度不足")

    a0 = a_row[0]
    if abs(a0) < 1e-30:
        raise ValueError("对角元 a0 不能为零")

    if n == 1:
        return np.array([[1.0 / a0]])

    a2 = a_row[1:n] / a0
    v = r8sto_yw_sl(n - 1, a2)


    v_n = 1.0 / (1.0 + np.dot(a2, v))

    v_rev = v[::-1]
    v_full = np.zeros(n)
    v_full[0] = v_n
    v_full[1:n] = v_n * v_rev

    B = np.zeros((n, n))
    B[0, :] = v_full[::-1]
    B[n - 1, :] = v_full
    B[1:n - 1, 0] = v_full[n - 1:1:-1]
    B[1:n - 1, n - 1] = v_full[1:n - 1]


    for i in range(1, (n + 1) // 2 + 1):
        for j in range(i, n - i + 1):
            val = B[i - 1, j - 1] + (v_full[n - j] * v_full[n - i] - v_full[i - 1] * v_full[j - 1]) / v_full[n - 1]
            B[i - 1, j - 1] = val
            B[j - 1, i - 1] = val
            B[n - i, n - j] = val
            B[n - j, n - i] = val

    B /= a0
    return B


def toeplitz_solve(n, a_row, b):
    Tinv = r8sto_inverse(n, a_row)
    return Tinv.dot(b)






class SparseNCF:

    def __init__(self, m, n, nz_num, rowcol, a):
        self.m = m
        self.n = n
        self.nz_num = nz_num
        self.rowcol = rowcol.astype(int)
        self.a = a

    def mv(self, x):
        if x.shape[0] != self.n:
            raise ValueError("维度不匹配")
        y = np.zeros(self.m)
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            y[i] += self.a[k] * x[j]
        return y

    def mtv(self, x):
        if x.shape[0] != self.m:
            raise ValueError("维度不匹配")
        y = np.zeros(self.n)
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            y[j] += self.a[k] * x[i]
        return y

    def to_dense(self):
        A = np.zeros((self.m, self.n))
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            A[i, j] += self.a[k]
        return A






def create_poisson_stencil(nx, nz, dx, dz):
    N = nx * nz
    A = np.zeros((N, N))

    coeff_x = 1.0 / (dx ** 2)
    coeff_z = 1.0 / (dz ** 2)
    coeff_c = -2.0 * (coeff_x + coeff_z)

    for j in range(nz):
        for i in range(nx):
            idx = j * nx + i

            if i == 0 or i == nx - 1 or j == 0 or j == nz - 1:
                A[idx, idx] = 1.0
                continue

            A[idx, idx] = coeff_c
            A[idx, idx - 1] = coeff_x
            A[idx, idx + 1] = coeff_x
            A[idx, idx - nx] = coeff_z
            A[idx, idx + nx] = coeff_z

    return A


def create_sparse_poisson_ncf(nx, nz, dx, dz):
    N = nx * nz
    rowcol_list = []
    a_list = []

    coeff_x = 1.0 / (dx ** 2)
    coeff_z = 1.0 / (dz ** 2)
    coeff_c = -2.0 * (coeff_x + coeff_z)

    for j in range(nz):
        for i in range(nx):
            idx = j * nx + i
            if i == 0 or i == nx - 1 or j == 0 or j == nz - 1:
                rowcol_list.append([idx, idx])
                a_list.append(1.0)
                continue

            rowcol_list.append([idx, idx])
            a_list.append(coeff_c)
            rowcol_list.append([idx, idx - 1])
            a_list.append(coeff_x)
            rowcol_list.append([idx, idx + 1])
            a_list.append(coeff_x)
            rowcol_list.append([idx, idx - nx])
            a_list.append(coeff_z)
            rowcol_list.append([idx, idx + nx])
            a_list.append(coeff_z)

    rowcol = np.array(rowcol_list).T
    a = np.array(a_list)
    return SparseNCF(N, N, len(a_list), rowcol, a)
