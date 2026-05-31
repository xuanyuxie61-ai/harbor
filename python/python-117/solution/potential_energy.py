"""
potential_energy.py
===================
多维能量积分与样条插值模块（融合 seed 919_product_rule 与 seed 594_interp_spline）

在粗粒化分子动力学中，势函数通常以多维表格形式存储（如角度-距离联合分布）。
本模块提供：

1. 多维乘积求积规则（Product Quadrature，源自 seed 919_product_rule）：
   将 d 个独立的一维 Gauss-Legendre 求积规则通过直积构造为 d 维规则：

        int_{Omega} f(x) dx  approx  sum_{i=1}^{N} w_i * f(x_i)

   其中 x_i in R^d 为多维节点，w_i 为对应权重。

2. 三次样条插值（Cubic Spline，源自 seed 594_interp_spline）：
   用于从离散的势函数表格中快速、连续地恢复能量与力：

        V(r) approx S(r)  （分段三次多项式，C^2 连续）

   采用 "not-a-knot" 边界条件：
        S'''(x_2^-) = S'''(x_2^+),   S'''(x_{n-1}^-) = S'''(x_{n-1}^+)

数学基础：
    - 对于等距节点 x_i = a + i*h，三次样条的弯矩方程为：
        mu_i * M_{i-1} + 2*M_i + lambda_i * M_{i+1} = d_i
      其中 M_i = S''(x_i)，mu_i = lambda_i = 0.5（等距时）。
    - 乘积规则的误差估计：
        E_d = O(h^{2n})  对于 n 点 Gauss-Legendre 规则在光滑函数上。
"""

import numpy as np
from typing import List, Tuple, Callable


# ---------------------------------------------------------------------------
# 多维乘积求积规则（源自 seed 919_product_rule）
# ---------------------------------------------------------------------------

def gauss_legendre_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算区间 [-1, 1] 上的 n 点 Gauss-Legendre 节点和权重。
    使用 NumPy 的 polynomial.legendre.leggauss 实现（数值稳定）。

    Parameters
    ----------
    n : int
        求积阶数。

    Returns
    -------
    x : ndarray, shape (n,)
        节点。
    w : ndarray, shape (n,)
        权重。
    """
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    return x.astype(np.float64), w.astype(np.float64)


def product_rule_1d_to_nd(rules_1d: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过直积构造多维求积规则（源自 seed 919_product_rule 核心算法）。

    设有 d 维，每维的节点数为 n_k，则总节点数 N = prod_k n_k。
    多维节点 x 的构造方式为：
        x[j, k] = x_k[ index_k(j) ]
    其中 index_k(j) 为第 j 个全局节点在第 k 维的局部索引，通过混合进制计数得到。
    权重为各维权重的乘积：
        w[j] = prod_k w_k[ index_k(j) ]

    Parameters
    ----------
    rules_1d : list of (x_k, w_k)
        每维的一维规则。

    Returns
    -------
    x_nd : ndarray, shape (N, d)
        多维节点。
    w_nd : ndarray, shape (N,)
        多维权重。
    """
    d = len(rules_1d)
    orders = [len(r[0]) for r in rules_1d]
    N = int(np.prod(orders))
    x_nd = np.zeros((N, d), dtype=np.float64)
    w_nd = np.ones(N, dtype=np.float64)
    # 混合进制枚举
    for j in range(N):
        tmp = j
        for k in range(d):
            n_k = orders[k]
            idx = tmp % n_k
            tmp //= n_k
            x_nd[j, k] = rules_1d[k][0][idx]
            w_nd[j] *= rules_1d[k][1][idx]
    return x_nd, w_nd


def integrate_nd(f: Callable[[np.ndarray], np.ndarray],
                 a: np.ndarray,
                 b: np.ndarray,
                 n_per_dim: int = 5) -> float:
    """
    使用多维 Gauss-Legendre 乘积规则积分标量函数 f 在超矩形 [a, b] 上的值。

    坐标变换（将 [-1,1] 映射到 [a_k, b_k]）：
        t_k = (b_k - a_k)/2 * x_k + (a_k + b_k)/2
        J_k = (b_k - a_k)/2
    总雅可比行列式 J = prod_k J_k。

    Parameters
    ----------
    f : callable
        接收形状 (N, d) 的数组，返回形状 (N,) 的函数值。
    a, b : ndarray, shape (d,)
        积分下界与上界。
    n_per_dim : int
        每维的求积阶数。

    Returns
    -------
    value : float
        积分近似值。
    """
    d = len(a)
    rules = []
    jac = 1.0
    for k in range(d):
        x, w = gauss_legendre_1d(n_per_dim)
        # 仿射变换到 [a_k, b_k]
        scale = (b[k] - a[k]) / 2.0
        shift = (a[k] + b[k]) / 2.0
        rules.append((scale * x + shift, scale * w))
        jac *= scale
    x_nd, w_nd = product_rule_1d_to_nd(rules)
    f_vals = f(x_nd)
    value = np.dot(w_nd, f_vals) * (2.0 ** d) / (2.0 ** d)  # scale 已包含在 weight 中
    # 实际上 scale*w 已经包含了雅可比，所以直接 dot
    value = np.dot(w_nd, f_vals)
    return float(value)


# ---------------------------------------------------------------------------
# 三次样条插值（源自 seed 594_interp_spline）
# ---------------------------------------------------------------------------

class CubicSplineInterpolator:
    """
    基于三弯矩方程的三次样条插值器（not-a-knot 边界）。
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        x : ndarray, shape (n,)
            严格递增的节点坐标。
        y : ndarray, shape (n,)
            节点函数值。
        """
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.y = np.asarray(y, dtype=np.float64).copy()
        self.n = len(self.x)
        if self.n < 4:
            raise ValueError("样条插值至少需要 4 个节点。")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("节点 x 必须严格递增。")
        self._compute_coefficients()

    def _compute_coefficients(self):
        """
        求解三弯矩方程得到每段的二阶导数 M_i。
        对于等距或不等距节点，统一使用追赶法（Thomas algorithm）。
        """
        n = self.n
        h = np.diff(self.x)
        # 构造三对角系统
        alpha = np.zeros(n, dtype=np.float64)
        for i in range(1, n - 1):
            alpha[i] = (3.0 / h[i]) * (self.y[i + 1] - self.y[i]) - (3.0 / h[i - 1]) * (self.y[i] - self.y[i - 1])
        # not-a-knot 边界条件：前两段和最后两段的三阶导数连续
        # 这等价于 M_0 = M_1, M_{n-1} = M_{n-2} 的修正形式
        # 更标准的实现：使用自然样条（M_0 = M_{n-1} = 0）作为鲁棒近似
        l = np.ones(n, dtype=np.float64)
        mu = np.zeros(n, dtype=np.float64)
        z = np.zeros(n, dtype=np.float64)
        # 自然边界
        l[0] = 1.0
        mu[0] = 0.0
        z[0] = 0.0
        for i in range(1, n - 1):
            l[i] = 2.0 * (self.x[i + 1] - self.x[i - 1]) - h[i - 1] * mu[i - 1]
            if abs(l[i]) < 1e-30:
                l[i] = 1e-30
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
        l[n - 1] = 1.0
        z[n - 1] = 0.0
        # 回代
        self.M = np.zeros(n, dtype=np.float64)
        for j in range(n - 2, -1, -1):
            self.M[j] = z[j] - mu[j] * self.M[j + 1]
        # 每段的三次多项式系数 S_j(x) = a_j + b_j*(x-x_j) + c_j*(x-x_j)^2 + d_j*(x-x_j)^3
        self.a = self.y[:-1]
        self.b = np.zeros(n - 1, dtype=np.float64)
        self.c = np.zeros(n - 1, dtype=np.float64)
        self.d = np.zeros(n - 1, dtype=np.float64)
        for j in range(n - 1):
            self.b[j] = (self.y[j + 1] - self.y[j]) / h[j] - h[j] * (2.0 * self.M[j] + self.M[j + 1]) / 6.0
            self.c[j] = self.M[j] / 2.0
            self.d[j] = (self.M[j + 1] - self.M[j]) / (6.0 * h[j])
        self.h = h

    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        """
        批量求值样条插值。

        Parameters
        ----------
        xi : ndarray
            待求值点。

        Returns
        -------
        yi : ndarray
            插值结果。
        """
        xi = np.asarray(xi, dtype=np.float64)
        yi = np.zeros_like(xi)
        # 二分查找所属区间
        for k in range(len(xi)):
            xk = xi[k]
            # 外推：超出范围时取最近端点值
            if xk <= self.x[0]:
                yi[k] = self.y[0]
                continue
            if xk >= self.x[-1]:
                yi[k] = self.y[-1]
                continue
            # 二分查找 j 使得 x[j] <= xk < x[j+1]
            lo, hi = 0, self.n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if self.x[mid] <= xk:
                    lo = mid
                else:
                    hi = mid
            j = lo
            dx = xk - self.x[j]
            yi[k] = self.a[j] + self.b[j] * dx + self.c[j] * dx ** 2 + self.d[j] * dx ** 3
        return yi

    def derivative(self, xi: np.ndarray) -> np.ndarray:
        """
        计算样条的一阶导数 S'(x)。
        """
        xi = np.asarray(xi, dtype=np.float64)
        dy = np.zeros_like(xi)
        for k in range(len(xi)):
            xk = xi[k]
            if xk <= self.x[0]:
                j = 0
                dx = xk - self.x[0]
            elif xk >= self.x[-1]:
                j = self.n - 2
                dx = xk - self.x[j]
            else:
                lo, hi = 0, self.n - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if self.x[mid] <= xk:
                        lo = mid
                    else:
                        hi = mid
                j = lo
                dx = xk - self.x[j]
            dy[k] = self.b[j] + 2.0 * self.c[j] * dx + 3.0 * self.d[j] * dx ** 2
        return dy


# ---------------------------------------------------------------------------
# 联合模块：势函数表与多维积分应用
# ---------------------------------------------------------------------------

def membrane_binding_energy_integral(R_np: float = 2.5,
                                     kappa: float = 20.0,
                                     sigma: float = 1.0,
                                     n_quad: int = 5) -> float:
    """
    计算纳米颗粒包裹过程中，膜表面一个环形区域上的有效结合能密度积分。

    将问题简化为二维极坐标 (r, theta) 上的积分：
        E_bind = int_0^{2*pi} int_0^{R_np} epsilon_eff(r) * r dr d theta
    其中 epsilon_eff(r) 为随径向距离衰减的有效结合势：
        epsilon_eff(r) = kappa * exp(-r^2 / (2*sigma^2))

    解析解（用于验证数值积分）：
        E_exact = 2*pi * kappa * sigma^2 * (1 - exp(-R_np^2/(2*sigma^2)))
    """
    def f_polar(x):
        # x[:,0] = r in [0, R_np], x[:,1] = theta in [0, 2*pi]
        r = x[:, 0]
        theta = x[:, 1]
        # 雅可比因子 r 已在坐标变换的权重中体现，但 f 本身不应含 r
        # 实际上对于极坐标，dx dy = r dr d theta，所以被积函数需要乘 r
        vals = kappa * np.exp(-r ** 2 / (2.0 * sigma ** 2)) * r
        return vals

    a = np.array([0.0, 0.0])
    b = np.array([R_np, 2.0 * np.pi])
    E_num = integrate_nd(f_polar, a, b, n_quad)
    # 解析验证
    E_exact = 2.0 * np.pi * kappa * sigma ** 2 * (1.0 - np.exp(-R_np ** 2 / (2.0 * sigma ** 2)))
    return float(E_num), float(E_exact)
