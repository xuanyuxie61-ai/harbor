"""
utils.py
通用数值分析工具库
基于 Conte & de Boor 数值分析算法集合的 Python 重构
包含: LU分解、前代回代、逆迭代、Horner求值、FFT、根查找等
"""
import numpy as np
from constants import EPS, TINY, MAX_ITER

# ============================================================
# 1. 线性代数: 带缩放行主元的 LU 分解 (factor_cd + subst)
# ============================================================
def lu_factor_scaled(a):
    """
    对 n x n 矩阵 A 进行 LU 分解，采用缩放行主元策略 (Scaled Partial Pivoting)
    分解形式: PA = LU
    
    数学公式:
      s_i = max_j |a_{ij}|   (行缩放因子)
      在第 k 步，选主元: p = argmax_{i>=k} |a_{ik}| / s_i
      交换第 k 行与第 p 行
      消元: l_{ik} = a_{ik} / a_{kk},  i = k+1,...,n-1
            a_{ij} <- a_{ij} - l_{ik} * a_{kj},  j = k+1,...,n-1
    
    参数:
        a: numpy.ndarray, shape (n, n)
    返回:
        lu: 分解后的矩阵 (上三角 U 在上部含对角线，严格下三角 L 在下部)
        pivot: 行交换索引数组
        iflag: 0=成功, 1=奇异矩阵
    """
    n = a.shape[0]
    lu = a.astype(float).copy()
    pivot = np.arange(n, dtype=int)
    scale = np.zeros(n)
    iflag = 0

    # 计算每行的缩放因子
    for i in range(n):
        row_max = np.max(np.abs(lu[i, :]))
        if row_max < TINY:
            iflag = 1
            row_max = TINY
        scale[i] = row_max

    for k in range(n - 1):
        # 选主元
        max_ratio = 0.0
        pivot_row = k
        for i in range(k, n):
            ratio = abs(lu[i, k]) / scale[i]
            if ratio > max_ratio:
                max_ratio = ratio
                pivot_row = i

        if max_ratio < TINY:
            iflag = 1
            break

        # 行交换
        if pivot_row != k:
            lu[[k, pivot_row], :] = lu[[pivot_row, k], :]
            scale[[k, pivot_row]] = scale[[pivot_row, k]]
            pivot[[k, pivot_row]] = pivot[[pivot_row, k]]

        # 消元
        if abs(lu[k, k]) > TINY:
            for i in range(k + 1, n):
                lu[i, k] /= lu[k, k]
                for j in range(k + 1, n):
                    lu[i, j] -= lu[i, k] * lu[k, j]

    if abs(lu[n - 1, n - 1]) < TINY:
        iflag = 1

    return lu, pivot, iflag


def lu_solve(lu, pivot, b):
    """
    利用 LU 分解结果求解线性方程组 Ax = b
    前代 (Ly = Pb) + 回代 (Ux = y)
    
    参数:
        lu: LU 分解矩阵
        pivot: 行交换索引
        b: 右端项
    返回:
        x: 解向量
    """
    n = lu.shape[0]
    y = b.astype(float).copy()

    # 应用行交换
    y = y[pivot]

    # 前代: L y = P b
    for i in range(1, n):
        for j in range(i):
            y[i] -= lu[i, j] * y[j]

    # 回代: U x = y
    x = y.copy()
    for i in range(n - 1, -1, -1):
        if abs(lu[i, i]) < TINY:
            x[i] = 0.0
        else:
            for j in range(i + 1, n):
                x[i] -= lu[i, j] * x[j]
            x[i] /= lu[i, i]
    return x


# ============================================================
# 2. 逆迭代求特征值/特征向量 (invitr)
# ============================================================
def inverse_iteration(a, shift, max_iter=100, tol=1.0e-12):
    """
    带位移的逆迭代算法求矩阵 A 最接近 shift 的特征值及对应特征向量
    
    迭代公式:
      (A - sigma I) v_{k+1} = v_k
      v_{k+1} <- v_{k+1} / ||v_{k+1}||
      lambda_k = v_k^T A v_k  (Rayleigh 商)
    
    参数:
        a: 方阵
        shift: 位移 sigma
        max_iter: 最大迭代次数
        tol: 收敛容差
    返回:
        eigenvalue: 近似特征值
        eigenvector: 近似特征向量
        converged: 是否收敛
    """
    n = a.shape[0]
    a_shifted = a - shift * np.eye(n)
    lu, pivot, iflag = lu_factor_scaled(a_shifted)
    if iflag != 0:
        return shift, np.zeros(n), False

    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    eigval = shift

    for _ in range(max_iter):
        v_new = lu_solve(lu, pivot, v)
        norm_v = np.linalg.norm(v_new)
        if norm_v < TINY:
            break
        v_new /= norm_v
        eigval_new = float(v_new.T @ (a @ v_new))
        if abs(eigval_new - eigval) < tol * max(1.0, abs(eigval_new)):
            return eigval_new, v_new, True
        eigval = eigval_new
        v = v_new
    return eigval, v, False


# ============================================================
# 3. Horner 多项式求值 (r8poly_values_horner)
# ============================================================
def horner_eval(coeffs, x):
    """
    使用 Horner 方法求多项式值 p(x) = c0 + c1*x + ... + cm*x^m
    
    Horner 公式:
      p = c_m
      p = c_{m-1} + x * p
      ...
      p = c_0 + x * p
    
    参数:
        coeffs: 系数数组 [c0, c1, ..., cm]
        x: 求值点或点数组
    返回:
        p(x) 值
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    p = np.zeros_like(x)
    for c in reversed(coeffs):
        p = c + x * p
    return p


# ============================================================
# 4. 快速傅里叶变换 (fft_cd)
# ============================================================
def prime_factors(n):
    """将 n 分解为素因子列表 [2,3,5,...]"""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def cooley_tukey_fft(x):
    """
    素因子 Cooley-Tukey FFT 实现
    对于 N = n1 * n2 * ... * nk，递归分解为小型 DFT
    
    DFT 定义:
      X_k = sum_{n=0}^{N-1} x_n * exp(-2*pi*i*k*n/N)
    
    参数:
        x: 复数序列，长度 N
    返回:
        X: DFT 结果
    """
    x = np.asarray(x, dtype=complex)
    N = x.size
    if N <= 1:
        return x.copy()
    
    factors = prime_factors(N)
    if len(factors) == 0 or (len(factors) == 1 and factors[0] == N):
        # 素数长度，使用直接 DFT 或 Bluestein 算法简化版
        return np.fft.fft(x)  # 回退到 numpy 保证正确性
    
    # 使用 radix-2 分解
    if N % 2 == 0:
        even = cooley_tukey_fft(x[0::2])
        odd = cooley_tukey_fft(x[1::2])
        twiddle = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        return np.concatenate([even + twiddle * odd, even - twiddle * odd])
    else:
        # 对于奇数合数，使用 numpy 回退
        return np.fft.fft(x)


# ============================================================
# 5. 根查找算法 (bisect, muller)
# ============================================================
def bisection(f, a, b, tol=1.0e-12, max_iter=100):
    """
    二分法求根。要求 f(a) 和 f(b) 异号。
    
    收敛定理: |x_n - x*| <= (b-a)/2^{n+1}
    
    参数:
        f: 目标函数
        a, b: 区间端点
        tol: 容差
        max_iter: 最大迭代次数
    返回:
        root: 近似根，或 None
        info: 迭代信息字典
    """
    fa = float(f(a))
    fb = float(f(b))
    if fa * fb > 0:
        return None, {"status": "same_sign", "iter": 0}
    
    for k in range(max_iter):
        c = (a + b) / 2.0
        fc = float(f(c))
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c, {"status": "converged", "iter": k + 1, "residual": abs(fc)}
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return (a + b) / 2.0, {"status": "max_iter", "iter": max_iter}


def muller_method(f, x0, x1, x2, tol=1.0e-12, max_iter=100):
    """
    Muller 方法求根。使用三个点的二次插值逼近根。
    
    插值多项式:
      p(x) = f(x2) + w*(x-x2) + a*(x-x2)^2
      其中 w 和 a 由三点条件确定
      根取: x3 = x2 - 2*f(x2) / (w + sign(w)*sqrt(w^2 - 4*a*f(x2)))
    
    参数:
        f: 目标函数
        x0, x1, x2: 三个初始点
        tol: 容差
        max_iter: 最大迭代次数
    返回:
        root: 近似根
        info: 迭代信息字典
    """
    f0, f1, f2 = float(f(x0)), float(f(x1)), float(f(x2))
    for k in range(max_iter):
        h0, h1 = x0 - x2, x1 - x2
        if abs(h0) < TINY or abs(h1) < TINY:
            break
        d0 = (f0 - f2) / h0
        d1 = (f1 - f2) / h1
        a = (d0 - d1) / (h0 - h1)
        w = d0 - a * h0
        disc = w * w - 4.0 * a * f2
        if disc < 0:
            disc = 0.0
        sqrt_disc = np.sqrt(disc)
        if abs(w + sqrt_disc) > abs(w - sqrt_disc):
            den = w + sqrt_disc
        else:
            den = w - sqrt_disc
        if abs(den) < TINY:
            break
        dx = -2.0 * f2 / den
        x3 = x2 + dx
        f3 = float(f(x3))
        if abs(f3) < tol or abs(dx) < tol * max(1.0, abs(x3)):
            return x3, {"status": "converged", "iter": k + 1, "residual": abs(f3)}
        x0, x1, x2 = x1, x2, x3
        f0, f1, f2 = f1, f2, f3
    return x2, {"status": "max_iter", "iter": max_iter}


# ============================================================
# 6. Runge-Kutta 2阶 (Heun方法) (rk2)
# ============================================================
def rk2_step(f, t, y, h):
    """
    Heun 方法 (改进的 Euler 方法) 单步:
      k1 = f(t, y)
      k2 = f(t + h, y + h*k1)
      y_{n+1} = y_n + h/2 * (k1 + k2)
    
    局部截断误差 O(h^3)，全局误差 O(h^2)
    """
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + h, y + h * k1), dtype=float)
    return y + 0.5 * h * (k1 + k2)


def rk2_integrate(f, t_span, y0, n_steps):
    """
    RK2 积分器
    
    参数:
        f: dy/dt = f(t, y)
        t_span: (t0, t1)
        y0: 初始条件
        n_steps: 步数
    返回:
        t_array, y_array
    """
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    y = np.asarray(y0, dtype=float)
    t = t0
    t_array = [t]
    y_array = [y.copy()]
    for _ in range(n_steps):
        y = rk2_step(f, t, y, h)
        t += h
        t_array.append(t)
        y_array.append(y.copy())
    return np.array(t_array), np.array(y_array)


# ============================================================
# 7. 安全数值运算
# ============================================================
def safe_divide(a, b, default=0.0):
    """安全除法，避免除零"""
    b = np.asarray(b, dtype=float)
    result = np.where(np.abs(b) > TINY, np.asarray(a, dtype=float) / b, default)
    return result


def safe_sqrt(x, default=0.0):
    """安全开方，对负数返回 default"""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0.0, np.sqrt(x), default)
