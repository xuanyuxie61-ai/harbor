import numpy as np






class R8SDMatrix:

    def __init__(self, n, ndiag, offset):
        self.n = n
        self.ndiag = ndiag
        self.offset = np.asarray(offset, dtype=np.int64)
        self.a = np.zeros((n, ndiag), dtype=np.float64)

    def mv(self, x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) != self.n:
            raise ValueError("Dimension mismatch")

        y = np.zeros(self.n, dtype=np.float64)

        for j in range(self.ndiag):
            off = self.offset[j]
            for i in range(self.n):
                aij = self.a[i, j]
                if aij == 0.0:
                    continue
                y[i] += aij * x[i + off]
                if off != 0 and i + off < self.n:
                    y[i + off] += aij * x[i]

        return y

    def residual(self, x, b):
        return b - self.mv(x)

    def to_dense(self):
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for j in range(self.ndiag):
            off = self.offset[j]
            for i in range(self.n):
                if i + off < self.n:
                    A[i, i + off] = self.a[i, j]
                    if off != 0:
                        A[i + off, i] = self.a[i, j]
        return A


def r8sd_cg(A, b, x0=None, tol=1e-10, max_iter=None):
    b = np.asarray(b, dtype=np.float64)
    n = A.n

    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).copy()

    r = A.residual(x, b)
    p = r.copy()
    rsold = np.dot(r, r)
    rs0 = rsold

    if rs0 < 1e-30:
        return x, {'iterations': 0, 'residual': 0.0, 'converged': True}

    for k in range(max_iter):
        Ap = A.mv(p)
        pAp = np.dot(p, Ap)

        if abs(pAp) < 1e-30:
            break

        alpha = rsold / pAp
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)

        if np.sqrt(rsnew / rs0) < tol or rsnew < 1e-12:
            return x, {'iterations': k + 1, 'residual': np.sqrt(rsnew), 'converged': True}

        if rsold < 1e-30:
            break

        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x, {'iterations': max_iter, 'residual': np.sqrt(rsnew), 'converged': False}






class R8PBLMatrix:

    def __init__(self, n, ml):
        self.n = n
        self.ml = ml
        self.a = np.zeros((ml + 1, n), dtype=np.float64)

    def set_diagonal(self, values):
        self.a[0, :] = np.asarray(values, dtype=np.float64)

    def set_subdiagonal(self, k, values):
        if k < 1 or k > self.ml:
            raise ValueError(f"k must be in [1, {self.ml}]")
        self.a[k, :self.n - k] = np.asarray(values, dtype=np.float64)

    def mv(self, x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) != self.n:
            raise ValueError("Dimension mismatch")

        y = self.a[0, :] * x

        for k in range(1, self.ml + 1):
            for j in range(self.n - k):
                a_val = self.a[k, j]
                y[j] += a_val * x[j + k]
                y[j + k] += a_val * x[j]

        return y

    def to_dense(self):
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for j in range(self.n):
            A[j, j] = self.a[0, j]
        for k in range(1, self.ml + 1):
            for j in range(self.n - k):
                A[j, j + k] = self.a[k, j]
                A[j + k, j] = self.a[k, j]
        return A

    def cholesky_band_solve(self, b):
        b = np.asarray(b, dtype=np.float64)
        n = self.n
        ml = self.ml


        A = self.to_dense()


        L = np.zeros_like(A)
        for j in range(n):

            diag_sum = np.sum(L[j, max(0, j - ml):j] ** 2)
            val = A[j, j] - diag_sum
            if val <= 1e-15:
                val = 1e-15
            L[j, j] = np.sqrt(val)


            for i in range(j + 1, min(n, j + ml + 1)):
                off_sum = np.sum(L[i, max(0, i - ml):j] * L[j, max(0, i - ml):j])
                L[i, j] = (A[i, j] - off_sum) / L[j, j]


        y = np.zeros(n, dtype=np.float64)
        for i in range(n):
            y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]


        x = np.zeros(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(L[i + 1:, i], x[i + 1:])) / L[i, i]

        return x


def build_poisson_r8sd(n, dr):

    offset = np.array([0, 1], dtype=np.int64)
    A = R8SDMatrix(n, 2, offset)

    for i in range(n):
        r_i = (i + 1) * dr
        rp = r_i + 0.5 * dr
        rm = r_i - 0.5 * dr

        if rm < 0:
            rm = 0.0


        if i == 0:
            A.a[i, 0] = rp / (r_i * dr * dr)
        elif i == n - 1:
            A.a[i, 0] = (rp + rm) / (r_i * dr * dr)
        else:
            A.a[i, 0] = (rp + rm) / (r_i * dr * dr)


        if i < n - 1:
            A.a[i, 1] = -rp / (r_i * dr * dr)

    return A


def build_band_spd_matrix(n, ml, condition_hint=1.0):
    A = R8PBLMatrix(n, ml)


    for k in range(1, ml + 1):
        vals = np.random.rand(n - k) * 0.5
        A.set_subdiagonal(k, vals)


    for i in range(n):
        off_sum = 0.0
        for k in range(1, ml + 1):
            if i - k >= 0:
                off_sum += abs(A.a[k, i - k])
            if i + k < n:
                off_sum += abs(A.a[k, i])
        A.a[0, i] = (1.0 + np.random.rand()) * off_sum + 0.01 * condition_hint

    return A
