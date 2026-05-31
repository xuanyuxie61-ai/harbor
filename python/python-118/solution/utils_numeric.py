"""
utils_numeric.py
数值工具模块：边界处理、随机数生成、正交多项式、特殊函数、收敛判断

融合种子项目：
- 1373_uniform: 均匀伪随机数生成思想
- 081_besselzero: Bessel函数零点计算
- 641_laguerre_polynomial: 拉盖尔多项式递推
- 463_gegenbauer_rule: 正交多项式参数检验
"""

import numpy as np
from scipy.special import jv, yv, gamma, gammaln
from scipy.optimize import newton
import warnings


class RandomState:
    """基于线性同余生成器(LCG)思想的伪随机数状态管理，融合1373_uniform。"""

    def __init__(self, seed=None):
        if seed is None:
            seed = 123456789
        self._state = int(seed) & 0xFFFFFFFF
        self._a = 1103515245
        self._b = 12345
        self._mod = 2 ** 31

    def _lcg_next(self):
        self._state = (self._a * self._state + self._b) % self._mod
        return self._state

    def uniform_ab(self, n, a, b):
        """生成区间[a,b]上的均匀分布随机数。"""
        if n <= 0:
            return np.array([])
        vals = np.array([self._lcg_next() for _ in range(n)], dtype=np.float64)
        vals = vals / self._mod
        return a + (b - a) * vals

    def maxwell_boltzmann(self, n, T, m):
        """Maxwell-Boltzmann速度分布采样：P(v) ~ v^2 exp(-mv^2/2kT)。
        使用Box-Muller变换生成高斯分布后组合。"""
        if n <= 0 or T <= 0 or m <= 0:
            raise ValueError("n, T, m must be positive for Maxwell-Boltzmann sampling.")
        sigma = np.sqrt(T / m)
        u1 = self.uniform_ab(n, 1e-12, 1.0)
        u2 = self.uniform_ab(n, 0.0, 2.0 * np.pi)
        u3 = self.uniform_ab(n, 0.0, 1.0)
        r = np.sqrt(-2.0 * np.log(u1)) * sigma
        theta = np.arccos(2.0 * u3 - 1.0)
        vx = r * np.sin(theta) * np.cos(u2)
        vy = r * np.sin(theta) * np.sin(u2)
        vz = r * np.cos(theta)
        return np.stack([vx, vy, vz], axis=1)


def bessel_zero_newton(n, k, kind=1, tol=1e-14, max_iter=100):
    """基于Halley-Newton迭代的Bessel函数零点计算，融合081_besselzero。
    
    参数:
        n: Bessel阶数 (实数)
        k: 第k个正零点
        kind: 1为J_n, 2为Y_n
    返回:
        zero: 零点估计值
    """
    n = abs(n)
    if k <= 0:
        raise ValueError("k must be positive integer.")

    # 初始猜测：利用渐近公式 + 最小二乘拟合系数
    if kind == 1:
        if k == 1:
            x0 = 0.411557 + 0.999987 * n + 0.698029 * (n + 1) ** 0.335300 + 1.069775 * (n + 1) ** 0.339671
        elif k == 2:
            x0 = 1.933951 + 1.000077 * n - 0.805720 * (n + 1) ** 0.456215 + 3.387646 * (n + 1) ** 0.388380
        elif k == 3:
            x0 = 5.407708 + 1.000939 * n + 2.669262 * (n + 1) ** 0.429702 - 0.174926 * (n + 1) ** 0.633480
        else:
            # 对k>=4，利用间距外推
            z2 = bessel_zero_newton(n, 2, kind)
            z3 = bessel_zero_newton(n, 3, kind)
            spacing = z3 - z2
            x0 = z3 + (k - 3) * spacing
    else:
        if k == 1:
            x0 = 0.079505 + 0.999998 * n + 0.890381 * (n + 1) ** 0.335377 + 0.027060 * (n + 1) ** 0.308720
        elif k == 2:
            x0 = 1.045025 + 1.000021 * n - 0.437921 * (n + 1) ** 0.434823 + 2.701131 * (n + 1) ** 0.366245
        elif k == 3:
            x0 = 3.727779 + 1.000353 * n + 2.685667 * (n + 1) ** 0.398248 - 0.112980 * (n + 1) ** 0.604770
        else:
            z2 = bessel_zero_newton(n, 2, kind)
            z3 = bessel_zero_newton(n, 3, kind)
            spacing = z3 - z2
            x0 = z3 + (k - 3) * spacing

    def f(x):
        if kind == 1:
            return float(jv(n, x))
        else:
            return float(yv(n, x))

    try:
        zero = newton(f, x0, tol=tol, maxiter=max_iter)
    except RuntimeError:
        zero = x0
    return zero


def laguerre_polynomial_alpha(x, n, alpha=0.0):
    """广义拉盖尔多项式 L_n^{(alpha)}(x) 的递推计算，融合641_laguerre_polynomial和lf_function。
    
    递推关系:
        L_0^{(alpha)}(x) = 1
        L_1^{(alpha)}(x) = 1 + alpha - x
        n L_n^{(alpha)} = (2n - 1 + alpha - x) L_{n-1}^{(alpha)} - (n - 1 + alpha) L_{n-2}^{(alpha)}
    
    正交性:
        \int_0^\infty x^{alpha} e^{-x} L_n^{(alpha)}(x) L_m^{(alpha)}(x) dx
        = \Gamma(n + alpha + 1) / n! \delta_{nm}
    """
    x = np.atleast_1d(x)
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1 for Laguerre polynomials.")
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 1.0 + alpha - x

    L_prev2 = np.ones_like(x)
    L_prev1 = 1.0 + alpha - x
    for i in range(2, n + 1):
        L_curr = ((2.0 * i - 1.0 + alpha - x) * L_prev1 - (i - 1.0 + alpha) * L_prev2) / i
        L_prev2 = L_prev1
        L_prev1 = L_curr
    return L_prev1


def gegenbauer_polynomial(x, n, lambda_):
    """盖根堡尔多项式 C_n^{(lambda)}(x) 递推计算，融合463_gegenbauer_rule的数学基础。
    
    递推:
        C_0^{(lambda)}(x) = 1
        C_1^{(lambda)}(x) = 2 lambda x
        (n+1) C_{n+1}^{(lambda)}(x) = 2(n+lambda) x C_n^{(lambda)}(x) - (n+2lambda-1) C_{n-1}^{(lambda)}(x)
    """
    x = np.atleast_1d(x)
    if lambda_ <= -0.5:
        raise ValueError("lambda must be > -0.5 for Gegenbauer polynomials.")
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * lambda_ * x

    C_prev2 = np.ones_like(x)
    C_prev1 = 2.0 * lambda_ * x
    for i in range(1, n):
        C_curr = (2.0 * (i + lambda_) * x * C_prev1 - (i + 2.0 * lambda_ - 1.0) * C_prev2) / (i + 1.0)
        C_prev2 = C_prev1
        C_prev1 = C_curr
    return C_prev1


def check_bounds(x, lower, upper, name="variable"):
    """边界检查与裁剪，确保数值鲁棒性。"""
    x = np.atleast_1d(x)
    if np.any(x < lower) or np.any(x > upper):
        warnings.warn(f"{name} out of bounds [{lower}, {upper}], clipping applied.")
        x = np.clip(x, lower, upper)
    return x


def relative_convergence_check(val_new, val_old, rtol=1e-6, atol=1e-12):
    """相对收敛判据。"""
    diff = np.abs(val_new - val_old)
    scale = 0.5 * (np.abs(val_new) + np.abs(val_old)) + atol
    return np.all(diff < rtol * scale)


def safe_sqrt(x, eps=1e-30):
    """安全开方，避免负值导致NaN。"""
    return np.sqrt(np.maximum(x, eps))


def compute_radial_grid(r_min, r_max, n_r, grid_type="legendre"):
    """构造径向积分网格，支持Legendre和Laguerre映射。"""
    if grid_type == "uniform":
        return np.linspace(r_min, r_max, n_r)
    elif grid_type == "legendre":
        # 将[-1,1]上的Legendre点映射到[r_min, r_max]
        from numpy.polynomial.legendre import leggauss
        xi, wi = leggauss(n_r)
        r = 0.5 * (r_max - r_min) * (xi + 1.0) + r_min
        w = 0.5 * (r_max - r_min) * wi
        return r, w
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")
