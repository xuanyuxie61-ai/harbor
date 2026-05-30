
import numpy as np






class NASRandom:
    
    def __init__(self, seed=None):
        self.f7 = 78125.0
        self.t30 = 1073741824.0
        if seed is None:
            seed = 0.5
        self.state = seed
    
    def next(self):
        self.state = (self.f7 * self.state) % 1.0
        return self.state
    
    def rand(self, shape=None):
        if shape is None:
            return self.next()
        
        result = np.zeros(shape)
        for idx in np.ndindex(*shape if hasattr(shape, '__iter__') else (shape,)):
            result[idx] = self.next()
        return result
    
    def fill_array_2d(self, rows, cols):
        arr = np.zeros((rows, cols))
        for j in range(cols):
            for i in range(rows):
                arr[i, j] = self.next()
        return arr






def mxm_optimized(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("输入必须为二维矩阵")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"矩阵维度不匹配: {A.shape} 和 {B.shape}")
    
    return A.dot(B)






def cholsky_tridiagonal(a, b, c, n):
    L = np.zeros((n, n))
    
    for j in range(n):

        sum_sq = 0.0
        for k in range(j):
            sum_sq += L[j, k] ** 2
        
        diag_val = b[j] - sum_sq
        if diag_val <= 1e-15:
            return L, j + 1
        
        L[j, j] = np.sqrt(diag_val)
        

        if j < n - 1:

            sum_prod = 0.0
            for k in range(j):
                sum_prod += L[j + 1, k] * L[j, k]
            
            if abs(L[j, j]) > 1e-15:
                L[j + 1, j] = (c[j] - sum_prod) / L[j, j]
    
    return L, 0


def solve_tridiagonal(a, b, c, d, n):
    if n < 1:
        raise ValueError("维数必须 >= 1")
    

    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    
    dp[0] = d[0] / b[0]
    if n > 1:
        cp[0] = c[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i - 1] * cp[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    
    return x


def solve_pentadiagonal(a2, a1, b, c1, c2, d, n):
    if n < 5:

        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = b[i]
            if i > 0:
                A[i, i - 1] = a1[i - 1]
            if i > 1:
                A[i, i - 2] = a2[i - 2]
            if i < n - 1:
                A[i, i + 1] = c1[i]
            if i < n - 2:
                A[i, i + 2] = c2[i]
        return np.linalg.solve(A, d)
    


    a2_p = a2.copy()
    a1_p = a1.copy()
    b_p = b.copy()
    c1_p = c1.copy()
    c2_p = c2.copy()
    d_p = d.copy()
    
    for i in range(n):

        if i >= 2:
            factor = a2_p[i - 2] / b_p[i - 2]
            if abs(factor) > 1e-15:
                a1_p[i - 1] -= factor * c1_p[i - 2]
                b_p[i] -= factor * c2_p[i - 2]
                d_p[i] -= factor * d_p[i - 2]
        

        if i >= 1:
            factor = a1_p[i - 1] / b_p[i - 1]
            if abs(factor) > 1e-15:
                b_p[i] -= factor * c1_p[i - 1]
                if i < n - 1:
                    c1_p[i] -= factor * c2_p[i - 1]
                d_p[i] -= factor * d_p[i - 1]
    

    x = np.zeros(n)
    x[-1] = d_p[-1] / b_p[-1]
    if n > 1:
        x[-2] = (d_p[-2] - c1_p[-2] * x[-1]) / b_p[-2]
    
    for i in range(n - 3, -1, -1):
        rhs = d_p[i]
        if i + 1 < n:
            rhs -= c1_p[i] * x[i + 1]
        if i + 2 < n:
            rhs -= c2_p[i] * x[i + 2]
        x[i] = rhs / b_p[i]
    
    return x






def dft_1d_naive(x, inverse=False):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    sign = 1 if inverse else -1
    
    for k in range(N):
        for n in range(N):
            angle = sign * 2.0 * np.pi * k * n / N
            X[k] += x[n] * (np.cos(angle) + 1j * np.sin(angle))
    
    if inverse:
        X /= N
    
    return X


def fft_2d_photonic(field, dx, dy):
    nx, ny = field.shape
    

    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(field)) ** 2)
    
    kx = np.fft.fftshift(np.fft.fftfreq(nx, dx)) * 2.0 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dy)) * 2.0 * np.pi
    
    return spectrum, kx, ky






def benchmark_kernels():
    import time
    
    results = {}
    

    n = 128
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    t0 = time.time()
    for _ in range(10):
        C = mxm_optimized(A, B)
    results['mxm_128'] = time.time() - t0
    

    n = 1000
    a = np.random.rand(n - 1)
    b = np.abs(np.random.rand(n)) + 2.0
    c = np.random.rand(n - 1)
    d = np.random.rand(n)
    t0 = time.time()
    for _ in range(100):
        x = solve_tridiagonal(a, b, c, d, n)
    results['tridiag_1000'] = time.time() - t0
    

    nx, ny = 256, 256
    field = np.random.rand(nx, ny)
    t0 = time.time()
    for _ in range(10):
        spectrum, kx, ky = fft_2d_photonic(field, 1e-9, 1e-9)
    results['fft2d_256'] = time.time() - t0
    
    return results
