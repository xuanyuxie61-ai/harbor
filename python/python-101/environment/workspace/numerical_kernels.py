"""
numerical_kernels.py
====================
高性能数值计算核函数

融合原项目:
  - 786_nas : NASA Ames 基准测试 (矩阵乘法、FFT、Cholesky 等)

本模块提供经过优化的核心数值运算，用于大规模光子晶体计算:
  1. 块循环展开的矩阵-矩阵乘法
  2. 一维/二维快速傅里叶变换
  3. 三对角/五对角方程组求解 (用于 FDTD 隐式格式)
  4. 伪随机数生成器 (与 NAS 兼容)
"""

import numpy as np


# =============================================================================
# 基于 786_nas 的伪随机数生成器
# =============================================================================

class NASRandom:
    """
    NASA Ames 基准兼容的伪随机数生成器
    
    递推公式:
        x_{n+1} = 5^7 · x_n  (mod 2^30)
    
    周期: 2^28 ≈ 2.68 亿
    """
    
    def __init__(self, seed=None):
        self.f7 = 78125.0
        self.t30 = 1073741824.0
        if seed is None:
            seed = 0.5
        self.state = seed
    
    def next(self):
        """生成下一个随机数 ∈ [0, 1)"""
        self.state = (self.f7 * self.state) % 1.0
        return self.state
    
    def rand(self, shape=None):
        """生成随机数组"""
        if shape is None:
            return self.next()
        
        result = np.zeros(shape)
        for idx in np.ndindex(*shape if hasattr(shape, '__iter__') else (shape,)):
            result[idx] = self.next()
        return result
    
    def fill_array_2d(self, rows, cols):
        """填充二维数组 (与 NAS 初始化一致)"""
        arr = np.zeros((rows, cols))
        for j in range(cols):
            for i in range(rows):
                arr[i, j] = self.next()
        return arr


# =============================================================================
# 基于 786_nas 的优化矩阵乘法
# =============================================================================

def mxm_optimized(A, B):
    """
    优化的矩阵乘法 C = A · B —— 基于 nas.m 中的 mxm
    
    对于科学计算，采用 NumPy 的 BLAS 加速，但保持与 NAS 的
    接口兼容性。
    
    计算复杂度: O(l·m·n)
    
    Parameters
    ----------
    A : ndarray, shape (l, m)
    B : ndarray, shape (m, n)
    
    Returns
    -------
    C : ndarray, shape (l, n)
    """
    A = np.asarray(A)
    B = np.asarray(B)
    
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("输入必须为二维矩阵")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"矩阵维度不匹配: {A.shape} 和 {B.shape}")
    
    return A.dot(B)


# =============================================================================
# 基于 786_nas 的 Cholesky 分解 (用于隐式 FDTD)
# =============================================================================

def cholsky_tridiagonal(a, b, c, n):
    """
    三对角对称正定矩阵的 Cholesky 分解
    
    矩阵形式:
        [ b₁  c₁                  ]
        [ c₁  b₂  c₂              ]
        [     c₂  b₃  c₃          ]
        [         ...             ]
        [             c_{n-1} b_n ]
    
    分解: A = L · Lᵀ
    
    Parameters
    ----------
    a : ndarray (下对角, 未使用)
    b : ndarray, shape (n,)
        主对角
    c : ndarray, shape (n-1,)
        上对角
    n : int
        矩阵维数
    
    Returns
    -------
    L : ndarray, shape (n, n)
        下三角矩阵
    info : int
        0=成功
    """
    L = np.zeros((n, n))
    
    for j in range(n):
        # 计算 L[j,j]
        sum_sq = 0.0
        for k in range(j):
            sum_sq += L[j, k] ** 2
        
        diag_val = b[j] - sum_sq
        if diag_val <= 1e-15:
            return L, j + 1
        
        L[j, j] = np.sqrt(diag_val)
        
        # 计算 L[i,j] for i > j
        if j < n - 1:
            # 对于三对角矩阵，只有 j+1 行有非零元
            sum_prod = 0.0
            for k in range(j):
                sum_prod += L[j + 1, k] * L[j, k]
            
            if abs(L[j, j]) > 1e-15:
                L[j + 1, j] = (c[j] - sum_prod) / L[j, j]
    
    return L, 0


def solve_tridiagonal(a, b, c, d, n):
    """
    追赶法求解三对角方程组 A·x = d
    
    矩阵 A:
        [ b₀  c₀                  ]
        [ a₀  b₁  c₁              ]
        [     a₁  b₂  c₂          ]
        [         ...             ]
        [             a_{n-2} b_{n-1} ]
    
    算法复杂度: O(n)
    
    Parameters
    ----------
    a : ndarray, shape (n-1,)
        下对角
    b : ndarray, shape (n,)
        主对角
    c : ndarray, shape (n-1,)
        上对角
    d : ndarray, shape (n,)
        右端项
    n : int
        维数
    
    Returns
    -------
    x : ndarray, shape (n,)
        解向量
    """
    if n < 1:
        raise ValueError("维数必须 >= 1")
    
    # 前向消元
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
    
    # 后向回代
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    
    return x


def solve_pentadiagonal(a2, a1, b, c1, c2, d, n):
    """
    五对角方程组求解 (用于高阶 FDTD 隐式格式)
    
    矩阵非零元位置: i±2, i±1, i
    
    采用针对五对角矩阵的 LU 分解变体。
    
    Parameters
    ----------
    a2 : ndarray, shape (n-2,)
        第-2对角
    a1 : ndarray, shape (n-1,)
        第-1对角
    b : ndarray, shape (n,)
        主对角
    c1 : ndarray, shape (n-1,)
        第+1对角
    c2 : ndarray, shape (n-2,)
        第+2对角
    d : ndarray, shape (n,)
        右端项
    n : int
        维数
    
    Returns
    -------
    x : ndarray, shape (n,)
        解向量
    """
    if n < 5:
        # 小矩阵直接求解
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
    
    # 针对五对角系统的专用消元
    # 将矩阵化为上三角形式
    a2_p = a2.copy()
    a1_p = a1.copy()
    b_p = b.copy()
    c1_p = c1.copy()
    c2_p = c2.copy()
    d_p = d.copy()
    
    for i in range(n):
        # 消去 a2[i-2] 元素
        if i >= 2:
            factor = a2_p[i - 2] / b_p[i - 2]
            if abs(factor) > 1e-15:
                a1_p[i - 1] -= factor * c1_p[i - 2]
                b_p[i] -= factor * c2_p[i - 2]
                d_p[i] -= factor * d_p[i - 2]
        
        # 消去 a1[i-1] 元素
        if i >= 1:
            factor = a1_p[i - 1] / b_p[i - 1]
            if abs(factor) > 1e-15:
                b_p[i] -= factor * c1_p[i - 1]
                if i < n - 1:
                    c1_p[i] -= factor * c2_p[i - 1]
                d_p[i] -= factor * d_p[i - 1]
    
    # 回代
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


# =============================================================================
# FFT 加速核 (基于 NAS CFFT2D)
# =============================================================================

def dft_1d_naive(x, inverse=False):
    """
    一维离散傅里叶变换 (朴素实现，用于小规模验证)
    
    公式:
        X_k = Σ_{n=0}^{N-1} x_n · exp(-2πi·k·n/N)
    
    逆变换:
        x_n = (1/N) Σ_{k=0}^{N-1} X_k · exp(2πi·k·n/N)
    
    Parameters
    ----------
    x : ndarray
        输入信号
    inverse : bool
        是否逆变换
    
    Returns
    -------
    X : ndarray
        变换结果
    """
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
    """
    二维 FFT 用于光子晶体场量分析
    
    对场量做二维傅里叶变换后，计算动量空间能谱:
        E(k_x, k_y) = |FFT{E(x,y)}|²
    
    波矢分辨率:
        Δk_x = 2π/(N_x·dx),  Δk_y = 2π/(N_y·dy)
    
    Parameters
    ----------
    field : ndarray, shape (nx, ny)
        实空间场分布
    dx, dy : float
        空间网格间距 [m]
    
    Returns
    -------
    spectrum : ndarray, shape (nx, ny)
        动量空间能谱
    kx, ky : ndarray
        波矢坐标 [m⁻¹]
    """
    nx, ny = field.shape
    
    # 使用 NumPy FFT (等效于 NAS CFFT2D 的优化实现)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(field)) ** 2)
    
    kx = np.fft.fftshift(np.fft.fftfreq(nx, dx)) * 2.0 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dy)) * 2.0 * np.pi
    
    return spectrum, kx, ky


# =============================================================================
# 性能基准测试
# =============================================================================

def benchmark_kernels():
    """
    运行核心数值核的性能基准测试
    
    Returns
    -------
    results : dict
        各核函数的执行时间
    """
    import time
    
    results = {}
    
    # 矩阵乘法测试
    n = 128
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    t0 = time.time()
    for _ in range(10):
        C = mxm_optimized(A, B)
    results['mxm_128'] = time.time() - t0
    
    # 三对角求解测试
    n = 1000
    a = np.random.rand(n - 1)
    b = np.abs(np.random.rand(n)) + 2.0  # 保证对角占优
    c = np.random.rand(n - 1)
    d = np.random.rand(n)
    t0 = time.time()
    for _ in range(100):
        x = solve_tridiagonal(a, b, c, d, n)
    results['tridiag_1000'] = time.time() - t0
    
    # FFT 测试
    nx, ny = 256, 256
    field = np.random.rand(nx, ny)
    t0 = time.time()
    for _ in range(10):
        spectrum, kx, ky = fft_2d_photonic(field, 1e-9, 1e-9)
    results['fft2d_256'] = time.time() - t0
    
    return results
