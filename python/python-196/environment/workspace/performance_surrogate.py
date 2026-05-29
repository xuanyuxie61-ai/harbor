"""
performance_surrogate.py
性能代理模型模块

包含：
- Chebyshev插值逼近（源自 approx_chebyshev）
- 最小二乘多项式拟合（源自 least_squares_approximant_coef）
- 代理模型驱动的调度决策

科学背景：
在异构HPC调度中，精确预测每个任务-处理器配对的执行时间
需要昂贵的基准测试。代理模型通过少量采样点建立快速预测器：

Chebyshev插值:
    在Chebyshev节点 x_k = cos((2k+1)pi/(2n)) 上构造多项式插值 p(x)，
    使得 max|f(x)-p(x)| 最小化。

最小二乘拟合:
    给定数据 (x_i, y_i)，寻找多项式 p(x) = sum c_j x^j 使得
        ||A c - y||_2 -> min
    其中 A_{i,j} = x_i^{j-1}。

在热-电耦合调度中，输入特征 x 可以是 (处理器频率, 任务flops, 内存带宽)，
输出 y 是预测执行时间。
"""

import numpy as np
from utils import safe_log


def cheby_nodes(a, b, n):
    """
    生成 [a, b] 上的 Chebyshev 节点。
    源自 chebyspace。

    x_k = (a+b)/2 + (b-a)/2 * cos((2k+1)pi/(2n)), k=0,...,n-1
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    k = np.arange(n)
    theta = (2.0 * k + 1.0) * np.pi / (2.0 * n)
    c = np.cos(theta)
    return 0.5 * ((1.0 - c) * a + (1.0 + c) * b)


def divided_differences(xd, yd):
    """
    计算差商表（源自 divdif）。

    dd[j] 表示第 j 阶差商，递推:
        dd_j = (dd_j - dd_{j-1}) / (x_j - x_{j-i+1})
    """
    xd = np.array(xd, dtype=float)
    yd = np.array(yd, dtype=float)
    n = xd.size
    dd = yd.copy()
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            dd[j] = (dd[j] - dd[j - 1]) / (xd[j] - xd[j - i])
    return dd


def newton_interp_eval(xd, dd, x):
    """
    用Newton差商形式求插值多项式值（源自 interp）。

    p(x) = dd_0 + (x-x_0)[dd_1 + (x-x_1)[dd_2 + ...]]
    """
    xd = np.array(xd, dtype=float)
    dd = np.array(dd, dtype=float)
    x = np.array(x, dtype=float)
    n = dd.size
    y = dd[-1] * np.ones_like(x)
    for i in range(n - 2, -1, -1):
        y = dd[i] + (x - xd[i]) * y
    return y


def chebyshev_approximate_1d(func, a, b, n, ne=10001):
    """
    1D Chebyshev插值逼近（源自 approx_chebyshev）。

    在Chebyshev节点采样，构造Newton插值多项式，
    并估计最大误差。

    参数:
        func: callable
        a, b: float, 区间
        n: int, 插值点数
        ne: int, 误差估计采样数

    返回:
        xd: Chebyshev节点
        dd: 差商系数
        maxerr: 估计最大误差
    """
    xd = cheby_nodes(a, b, n)
    yd = func(xd)
    dd = divided_differences(xd, yd)
    # 误差估计
    xe = np.linspace(a, b, ne)
    ye = newton_interp_eval(xd, dd, xe)
    fe = func(xe)
    maxerr = float(np.max(np.abs(ye - fe)))
    return xd, dd, maxerr


def least_squares_approximant_matrix(nd, xd, m):
    """
    构造最小二乘范德蒙德矩阵（源自 least_squares_approximant_matrix）。

    A_{i,j} = xd_i^{j-1},  i=0,...,nd-1, j=0,...,m-1
    """
    xd = np.array(xd, dtype=float)
    A = np.zeros((nd, m), dtype=float)
    A[:, 0] = 1.0
    for j in range(1, m):
        A[:, j] = A[:, j - 1] * xd
    return A


def least_squares_fit(xd, yd, m):
    """
    最小二乘多项式拟合（源自 least_squares_approximant_coef）。

    当 nd >= m 时，使用正规方程或SVD求解。
    当 nd < m 时，返回 nd 阶拟合并补零。

    参数:
        xd: ndarray, 数据点
        yd: ndarray, 数据值
        m: int, 多项式阶数+1（即基函数个数）

    返回:
        c: ndarray, shape (m,), 多项式系数
        residual: float, 拟合残差
    """
    xd = np.array(xd, dtype=float)
    yd = np.array(yd, dtype=float)
    nd = xd.size
    if nd < m:
        A = least_squares_approximant_matrix(nd, xd, nd)
        c1 = np.linalg.lstsq(A, yd, rcond=None)[0]
        # SVD伪逆
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        s_inv = np.where(s < np.sqrt(np.finfo(float).eps), 0.0, 1.0 / s)
        c2 = Vh.T @ np.diag(s_inv) @ U.T @ yd
        c = np.zeros(m, dtype=float)
        c[:nd] = c1
        residual = float(np.linalg.norm(A @ c1 - yd))
    else:
        A = least_squares_approximant_matrix(nd, xd, m)
        c1 = np.linalg.lstsq(A, yd, rcond=None)[0]
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        s_inv = np.where(s < np.sqrt(np.finfo(float).eps), 0.0, 1.0 / s)
        c2 = Vh.T @ np.diag(s_inv) @ U.T @ yd
        c = c1
        residual = float(np.linalg.norm(A @ c - yd))
    return c, residual


def poly_value(c, x):
    """
    Horner法求多项式值（源自 r8poly_value_horner）。

    p(x) = c_0 + c_1 x + ... + c_{m-1} x^{m-1}
    """
    x = np.array(x, dtype=float)
    c = np.array(c, dtype=float)
    y = c[-1] * np.ones_like(x)
    for i in range(len(c) - 2, -1, -1):
        y = y * x + c[i]
    return y


class PerformanceSurrogate:
    """
    性能代理模型：用于快速预测任务在不同处理器上的执行时间。
    """
    def __init__(self, model_type='chebyshev'):
        self.model_type = model_type
        self.xd = None
        self.dd = None
        self.coeffs = None
        self.maxerr = None
        self.residual = None
        self.a = None
        self.b = None

    def train(self, feature_range, func, n_nodes, m_poly=None):
        """
        训练代理模型。

        参数:
            feature_range: (a, b), 特征区间
            func: callable, 真实性能函数 f(x)
            n_nodes: int, 采样/节点数
            m_poly: int, 最小二乘多项式阶数+1（仅用于lsq模式）
        """
        a, b = feature_range
        self.a = a
        self.b = b
        if self.model_type == 'chebyshev':
            self.xd, self.dd, self.maxerr = chebyshev_approximate_1d(
                func, a, b, n_nodes, ne=5001
            )
        elif self.model_type == 'least_squares':
            if m_poly is None:
                m_poly = n_nodes
            xd = np.linspace(a, b, n_nodes)
            yd = func(xd)
            self.coeffs, self.residual = least_squares_fit(xd, yd, m_poly)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def predict(self, x):
        """
        预测输出。
        """
        x = np.clip(x, self.a, self.b)
        if self.model_type == 'chebyshev':
            return newton_interp_eval(self.xd, self.dd, x)
        elif self.model_type == 'least_squares':
            return poly_value(self.coeffs, x)
        else:
            raise ValueError("Model not trained")
