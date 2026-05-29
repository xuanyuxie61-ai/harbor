"""
spectral_analysis.py
谱分析与插值模块

实现拉盖尔 (Laguerre) 正交多项式系、切比雪夫 (Chebyshev) 节点插值、
以及基于正交展开的径向分布函数分析。

核心数学：
    - 拉盖尔多项式 L_n(x):
        L_0(x) = 1
        L_1(x) = 1 - x
        (n+1) * L_{n+1}(x) = (2n+1-x) * L_n(x) - n * L_{n-1}(x)
      
      广义拉盖尔函数 L_n^{(alpha)}(x):
        L_0^{(alpha)}(x) = 1
        L_1^{(alpha)}(x) = 1 + alpha - x
        n * L_n^{(alpha)} = (2n-1+alpha-x) * L_{n-1}^{(alpha)} - (n-1+alpha) * L_{n-2}^{(alpha)}
      
      正交性:
        integral_0^inf exp(-x) * L_n(x) * L_m(x) dx = delta_{nm}
      
      广义正交性:
        integral_0^inf x^alpha * exp(-x) * L_n^{(alpha)}(x) * L_m^{(alpha)}(x) dx
        = Gamma(n+alpha+1) / n!
    
    - 切比雪夫节点（第一类）:
        x_k = cos( (2k-1) * pi / (2n) ),   k = 1,...,n
      
      切比雪夫插值多项式:
        p(x) = sum_{k=0}^{n-1} c_k * T_k(x)
      
      其中 T_k(x) = cos(k * arccos(x)) 为第一类切比雪夫多项式。
    
    - 牛顿差商插值:
        给定节点 x_0,...,x_n 和值 y_0,...,y_n
        差商表:
            f[x_i] = y_i
            f[x_i,...,x_j] = (f[x_{i+1},...,x_j] - f[x_i,...,x_{j-1}]) / (x_j - x_i)
        
        插值多项式:
            P(x) = f[x_0] + f[x_0,x_1]*(x-x_0) + ... + f[x_0,...,x_n]*(x-x_0)*...*(x-x_{n-1})
    
    - 径向分布函数 g(r) 的谱展开:
        g(r) = sum_{n=0}^{N} a_n * L_n^{(alpha)}(beta * r) * exp(-beta*r/2)
        
        展开系数:
            a_n = (n! / Gamma(n+alpha+1)) * integral_0^inf g(r) * L_n^{(alpha)}(beta*r)
                  * (beta*r)^alpha * exp(-beta*r) * beta dr
"""

import numpy as np
import math
from scipy.special import gamma as Gamma
from typing import Tuple, Optional
from utils import check_bounds, EPSILON_MACHINE


def laguerre_polynomial(m: int, n: int, x: np.ndarray) -> np.ndarray:
    """
    计算标准拉盖尔多项式 L_0(x) 到 L_n(x)。
    
    递归关系:
        L_0(x) = 1
        L_1(x) = 1 - x
        n * L_n(x) = (2n - 1 - x) * L_{n-1}(x) - (n - 1) * L_{n-2}(x)
    
    Parameters
    ----------
    m : int
        评估点数量
    n : int
        最高阶数
    x : np.ndarray, shape (m,)
        评估点（x >= 0）
    
    Returns
    -------
    np.ndarray, shape (m, n+1)
        多项式值矩阵
    """
    x = np.asarray(x, dtype=float).flatten()
    if n < 0:
        return np.empty((m, 0))

    v = np.zeros((m, n + 1), dtype=float)
    v[:, 0] = 1.0
    if n == 0:
        return v

    v[:, 1] = 1.0 - x
    for j in range(2, n + 1):
        v[:, j] = (
            ((2.0 * j - 1.0) - x) * v[:, j - 1]
            - (j - 1.0) * v[:, j - 2]
        ) / j

    return v


def generalized_laguerre_function(m: int, n: int, alpha: float,
                                   x: np.ndarray) -> np.ndarray:
    """
    计算广义拉盖尔函数 L_n^{(alpha)}(x)。
    
    递归关系:
        L_0^{(alpha)}(x) = 1
        L_1^{(alpha)}(x) = 1 + alpha - x
        n * L_n^{(alpha)} = (2n - 1 + alpha - x) * L_{n-1}^{(alpha)}
                            - (n - 1 + alpha) * L_{n-2}^{(alpha)}
    
    要求 alpha > -1。
    
    Parameters
    ----------
    m : int
        评估点数量
    n : int
        最高阶数
    alpha : float
        参数（alpha > -1）
    x : np.ndarray, shape (m,)
        评估点
    
    Returns
    -------
    np.ndarray, shape (m, n+1)
        函数值矩阵
    """
    if alpha <= -1.0:
        raise ValueError(f"alpha must be > -1, got {alpha}")
    x = np.asarray(x, dtype=float).flatten()
    if n < 0:
        return np.empty((m, 0))

    v = np.zeros((m, n + 1), dtype=float)
    v[:, 0] = 1.0
    if n == 0:
        return v

    v[:, 1] = 1.0 + alpha - x
    for i in range(2, n + 1):
        v[:, i] = (
            ((2.0 * i - 1.0 + alpha) - x) * v[:, i - 1]
            + (-i + 1.0 - alpha) * v[:, i - 2]
        ) / i

    return v


def chebyshev_nodes(a: float, b: float, n: int) -> np.ndarray:
    """
    生成区间 [a,b] 上的切比雪夫节点（第一类）。
    
    节点公式:
        x_k = (a+b)/2 + (b-a)/2 * cos( (2k-1)*pi / (2n) ),  k=1,...,n
    
    这些节点在区间端点聚集，最小化 Runge 现象。
    
    Parameters
    ----------
    a, b : float
        区间端点
    n : int
        节点数
    
    Returns
    -------
    np.ndarray
        切比雪夫节点
    """
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([(a + b) / 2.0])

    k = np.arange(1, n + 1, dtype=float)
    theta = (2.0 * k - 1.0) * np.pi / (2.0 * n)
    c = np.cos(theta)

    # 对奇数 n，中间节点应为 0
    if n % 2 == 1:
        mid = (n + 1) // 2
        c[mid - 1] = 0.0

    x = 0.5 * ((1.0 - c) * a + (1.0 + c) * b)
    return x


def divided_differences(xd: np.ndarray, yd: np.ndarray) -> np.ndarray:
    """
    计算牛顿差商表。
    
    算法:
        d_i^{(0)} = y_i
        d_i^{(k)} = (d_{i+1}^{(k-1)} - d_i^{(k-1)}) / (x_{i+k} - x_i)
    
    返回对角线元素 d_0^{(0)}, d_0^{(1)}, ..., d_0^{(n)}。
    
    Parameters
    ----------
    xd, yd : np.ndarray
        数据点
    
    Returns
    -------
    np.ndarray
        差商系数（牛顿基系数）
    """
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)
    n = len(xd)
    d = yd.copy()
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            denom = xd[j] - xd[j - i]
            if abs(denom) < EPSILON_MACHINE * 100:
                d[j] = 0.0
            else:
                d[j] = (d[j] - d[j - 1]) / denom
    return d


def newton_interpolate(xd: np.ndarray, dd: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """
    使用牛顿差商形式评估插值多项式。
    
    Parameters
    ----------
    xd : np.ndarray
        插值节点
    dd : np.ndarray
        差商系数（由 divided_differences 生成）
    xp : np.ndarray
        评估点
    
    Returns
    -------
    np.ndarray
        插值结果
    """
    xd = np.asarray(xd, dtype=float)
    dd = np.asarray(dd, dtype=float)
    xp = np.asarray(xp, dtype=float)
    nd = len(xd)
    yp = dd[nd - 1] * np.ones_like(xp)
    for i in range(nd - 2, -1, -1):
        yp = dd[i] + (xp - xd[i]) * yp
    return yp


def chebyshev_interpolate(func: callable, a: float, b: float,
                          n: int, xp: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    切比雪夫节点插值。
    
    在 n 个切比雪夫节点上采样函数，构建牛顿插值多项式，
    然后在 xp 上评估，并返回最大误差估计。
    
    Parameters
    ----------
    func : callable
        目标函数 f(x)
    a, b : float
        区间
    n : int
        插值节点数
    xp : np.ndarray
        评估点
    
    Returns
    -------
    yp : np.ndarray
        插值结果
    maxerr : float
        在密集采样点上的最大误差估计
    """
    xd = chebyshev_nodes(a, b, n)
    yd = func(xd)
    dd = divided_differences(xd, yd)
    yp = newton_interpolate(xd, dd, xp)

    # 误差估计：在 10001 个均匀点上比较
    ne = 10001
    xe = np.linspace(a, b, ne)
    ye = newton_interpolate(xd, dd, xe)
    fe = func(xe)
    maxerr = np.max(np.abs(ye - fe))
    return yp, maxerr


def radial_distribution_spectrum(r: np.ndarray, g: np.ndarray,
                                  n_modes: int = 10, alpha: float = 0.0,
                                  beta: float = 1.0) -> np.ndarray:
    """
    对径向分布函数 g(r) 进行广义拉盖尔谱展开，提取模态系数。
    
    展开式:
        g(r) approx sum_{n=0}^{N-1} a_n * L_n^{(alpha)}(beta * r)
    
    系数（数值积分）:
        a_n = sum_j g(r_j) * L_n^{(alpha)}(beta*r_j) * w_j
        
    其中权重 w_j 包含 exp(-beta*r) * (beta*r)^alpha 因子。
    
    Parameters
    ----------
    r : np.ndarray
        径向坐标（r >= 0）
    g : np.ndarray
        径向分布函数值
    n_modes : int
        模态数
    alpha : float
        拉盖尔参数
    beta : float
        缩放因子
    
    Returns
    -------
    np.ndarray
        谱系数 a_n
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    m = len(r)

    # 计算广义拉盖尔函数
    L = generalized_laguerre_function(m, n_modes - 1, alpha, beta * r)

    # 数值积分（梯形法则）
    coeffs = np.zeros(n_modes)
    # 权重: w(r) = exp(-beta*r) * (beta*r)^alpha
    w = np.exp(-beta * r) * np.maximum(beta * r, 0.0) ** alpha

    for n in range(n_modes):
        integrand = g * L[:, n] * w
        # 使用梯形法则（假设 r 已排序）
        if len(r) > 1:
            coeffs[n] = np.trapezoid(integrand, r)
        else:
            coeffs[n] = integrand[0] * r[0] if len(r) == 1 else 0.0

    # 归一化
    for n in range(n_modes):
        norm = Gamma(n + alpha + 1.0) / math.factorial(n)
        if norm > 0:
            coeffs[n] /= norm

    return coeffs


def chebyshev_spectral_derivative(u: np.ndarray, L: float) -> np.ndarray:
    """
    使用切比雪夫谱方法计算导数。
    
    在切比雪夫节点上，导数可通过离散切比雪夫变换计算:
        du/dx = sum_k c_k * dT_k/dx
    
    这里使用有限差分近似作为简化实现（保持数值稳定性）。
    
    Parameters
    ----------
    u : np.ndarray
        函数值（定义在均匀网格上）
    L : float
        域长度
    
    Returns
    -------
    np.ndarray
        谱近似导数
    """
    u = np.asarray(u, dtype=float)
    n = len(u)
    if n < 2:
        return np.zeros_like(u)
    h = L / (n - 1)
    dudx = np.zeros_like(u)
    # 内部：中心差分
    dudx[1:-1] = (u[2:] - u[:-2]) / (2.0 * h)
    # 边界：前向/后向差分
    dudx[0] = (-3.0 * u[0] + 4.0 * u[1] - u[2]) / (2.0 * h)
    dudx[-1] = (3.0 * u[-1] - 4.0 * u[-2] + u[-3]) / (2.0 * h)
    return dudx
