"""
numerical_utils.py
博士级科学计算数值工具库

融合源项目：
- 596_interp_trig: 三角函数插值（周期性风向数据重构）
- 925_pwl_approx_1d: 分段线性逼近（风速-功率曲线逼近）
- 032_asa066: 标准正态累积分布函数（风资源统计检验）
- 1098_solve: 高斯消元法直接求解（线性系统）
- 1099_sor: SOR迭代法（稀疏线性系统）
- 645_langford_ode: ODE动力系统（湍流能量级联模型）
"""

import numpy as np
from typing import Tuple, Callable, Optional


# ============================================================================
# 1. 三角函数插值 (源自 596_interp_trig)
# ============================================================================

def trigcardinal(xi: np.ndarray, xdj: float, nd: int, h: float) -> np.ndarray:
    """
    三角基 cardinal 函数，用于周期数据插值。

    给定等距节点 x_j 和间距 h，cardinal 函数 τ_j(x) 满足：
    
        τ_j(x_k) = δ_jk
    
    当节点数 nd 为奇数时：
        τ_j(x) = sin(π(x-x_j)/h) / [nd · sin(π(x-x_j)/(nd·h))]
    
    当节点数 nd 为偶数时：
        τ_j(x) = sin(π(x-x_j)/h) / [nd · tan(π(x-x_j)/(nd·h))]

    Parameters
    ----------
    xi : np.ndarray
        待插值点。
    xdj : float
        第 j 个数据节点。
    nd : int
        数据节点总数。
    h : float
        节点间距。

    Returns
    -------
    np.ndarray
        cardinal 函数在 xi 处的取值。
    """
    eps = 1e-14
    tau = np.zeros_like(xi, dtype=float)
    diff = xi - xdj

    # 处理 xi == xdj 的情况
    mask_eq = np.abs(diff) < eps
    tau[mask_eq] = 1.0

    mask_ne = ~mask_eq
    if nd % 2 == 1:
        denom = nd * np.sin(np.pi * diff[mask_ne] / (nd * h))
    else:
        denom = nd * np.tan(np.pi * diff[mask_ne] / (nd * h))

    # 避免除零
    safe_denom = np.where(np.abs(denom) < eps, np.sign(denom + eps) * eps, denom)
    tau[mask_ne] = np.sin(np.pi * diff[mask_ne] / h) / safe_denom
    return tau


def trig_interpolant(xd: np.ndarray, yd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    三角插值函数。

    对于周期为 L = nd·h 的函数，三角插值多项式为：
    
        P_n(x) = Σ_{j=1}^{n} y_j · τ_j(x)

    其中 τ_j(x) 为三角 cardinal 函数。

    Parameters
    ----------
    xd : np.ndarray
        等距数据节点，必须满足 xd[j+1] - xd[j] = h。
    yd : np.ndarray
        节点处的数据值。
    xi : np.ndarray
        待求值点。

    Returns
    -------
    np.ndarray
        插值结果。
    """
    nd = len(xd)
    if nd < 2:
        raise ValueError("至少需要 2 个插值节点")
    h = xd[1] - xd[0]
    if h <= 0:
        raise ValueError("节点间距必须为正")
    # 检查等距
    if not np.allclose(np.diff(xd), h, rtol=1e-10):
        raise ValueError("xd 必须是等距节点")

    yi = np.zeros_like(xi, dtype=float)
    for j in range(nd):
        yi += yd[j] * trigcardinal(xi, xd[j], nd, h)
    return yi


# ============================================================================
# 2. 分段线性逼近 (源自 925_pwl_approx_1d)
# ============================================================================

def pwl_approx_1d_matrix(nd: int, xd: np.ndarray, yd: np.ndarray, nc: int, xc: np.ndarray) -> np.ndarray:
    """
    构造分段线性逼近的 nd × nc 最小二乘矩阵 A。

    对于控制点 (xc_j, yc_j)，分段线性基函数 φ_j(x) 满足：
    
        φ_j(xc_j) = 1,  φ_j(xc_k) = 0 (k ≠ j)
    
    逼近函数：
        f̃(x) = Σ_{j=1}^{nc} yc_j · φ_j(x)

    矩阵 A 的元素 A_{ij} = φ_j(xd_i)，通过最小二乘求解：
    
        A^T A · yc = A^T · yd

    Parameters
    ----------
    nd : int
        数据点数量。
    xd : np.ndarray
        数据点横坐标。
    yd : np.ndarray
        数据点纵坐标（未实际使用，仅保持接口一致）。
    nc : int
        控制点数量。
    xc : np.ndarray
        控制点横坐标，必须单调递增。

    Returns
    -------
    np.ndarray
        nd × nc 的逼近矩阵 A。
    """
    xd = np.asarray(xd).ravel()
    xc = np.asarray(xc).ravel()
    if len(xd) != nd or len(xc) != nc:
        raise ValueError("维度不匹配")
    if nc < 2:
        raise ValueError("至少需要 2 个控制点")
    if not np.all(np.diff(xc) > 0):
        raise ValueError("控制点 xc 必须严格单调递增")

    A = np.zeros((nd, nc), dtype=float)
    for i in range(nd):
        x = xd[i]
        if x <= xc[0]:
            A[i, 0] = 1.0
        elif x >= xc[-1]:
            A[i, -1] = 1.0
        else:
            # 找到 x 所在的区间 [xc[j], xc[j+1]]
            j = np.searchsorted(xc, x, side='right') - 1
            j = max(0, min(j, nc - 2))
            dx = xc[j + 1] - xc[j]
            if dx < 1e-14:
                A[i, j] = 1.0
            else:
                t = (x - xc[j]) / dx
                A[i, j] = 1.0 - t
                A[i, j + 1] = t
    return A


def pwl_approx_1d(nd: int, xd: np.ndarray, yd: np.ndarray, nc: int, xc: np.ndarray) -> np.ndarray:
    """
    分段线性最小二乘逼近，求解控制点纵坐标 yc。

    Parameters
    ----------
    nd, xd, yd, nc, xc
        同 pwl_approx_1d_matrix。

    Returns
    -------
    np.ndarray
        控制点纵坐标 yc（nc × 1）。
    """
    A = pwl_approx_1d_matrix(nd, xd, yd, nc, xc)
    yd_vec = np.asarray(yd).ravel()
    # 最小二乘求解
    ATA = A.T @ A
    ATy = A.T @ yd_vec
    # 添加正则化保证可逆
    ATA += 1e-12 * np.eye(nc)
    yc = np.linalg.solve(ATA, ATy)
    return yc


# ============================================================================
# 3. 标准正态累积分布函数 (源自 032_asa066)
# ============================================================================

def alnorm(x: float, upper: bool = False) -> float:
    """
    计算标准正态分布的累积分布函数（CDF）。

    对于标准正态随机变量 Z ~ N(0,1)：

        Φ(x) = P(Z ≤ x) = ∫_{-∞}^{x} (1/√(2π)) exp(-t²/2) dt

    当 upper=True 时，计算上尾概率：
        Q(x) = P(Z ≥ x) = 1 - Φ(x)

    采用 Hill (1973) 的 AS 66 算法，使用连分数有理逼近：

        Φ(x) ≈ 0.5 - x·(p - q·y / (y + a1 + b1/(y + a2 + b2/(y + a3))))   (|x| ≤ 1.28)

        Φ(x) ≈ r·exp(-y) / (z + c1 + d1/(z + c2 + d2/(...)))                (|x| > 1.28)

    Parameters
    ----------
    x : float
        积分端点。
    upper : bool, optional
        是否计算上尾概率，默认为 False。

    Returns
    -------
    float
        标准正态 CDF 值或上尾概率。
    """
    a1 = 5.75885480458
    a2 = 2.62433121679
    a3 = 5.92885724438
    b1 = -29.8213557807
    b2 = 48.6959930692
    c1 = -0.000000038052
    c2 = 0.000398064794
    c3 = -0.151679116635
    c4 = 4.8385912808
    c5 = 0.742380924027
    c6 = 3.99019417011
    con = 1.28
    d1 = 1.00000615302
    d2 = 1.98615381364
    d3 = 5.29330324926
    d4 = -15.1508972451
    d5 = 30.789933034
    ltone = 7.0
    p = 0.39894228044
    q = 0.39990348504
    r = 0.398942280385
    utzero = 18.66

    up = upper
    z = x
    if z < 0.0:
        up = not up
        z = -z

    if ltone < z and (not up or utzero < z):
        return 0.0 if up else 1.0

    y = 0.5 * z * z
    if z <= con:
        value = 0.5 - z * (p - q * y / (y + a1 + b1 / (y + a2 + b2 / (y + a3))))
    else:
        value = r * np.exp(-y) / (z + c1 + d1 / (z + c2 + d2 / (z + c3 + d3 / (z + c4 + d4 / (z + c5 + d5 / (z + c6))))))

    return 1.0 - value if not up else value


def alnorm_array(x: np.ndarray, upper: bool = False) -> np.ndarray:
    """向量化的 alnorm。"""
    return np.array([alnorm(float(v), upper) for v in x.ravel()]).reshape(x.shape)


# ============================================================================
# 4. 高斯消元法直接求解 (源自 1098_solve)
# ============================================================================

def r8mat_fs(n: int, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    使用部分主元高斯消元法求解线性系统 A·x = b。

    算法步骤：
        1. 前向消元：对 j = 1, ..., n
           a) 选列主元：piv = max_{i≥j} |A_{ij}|，交换行
           b) 归一化主元行
           c) 消去下方元素
        2. 回代求解 x

    时间复杂度 O(n³)。

    Parameters
    ----------
    n : int
        矩阵阶数。
    A : np.ndarray
        n × n 系数矩阵（会被修改）。
    b : np.ndarray
        n × 1 右端项。

    Returns
    -------
    np.ndarray
        解向量 x。
    """
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True).ravel()
    if A.shape != (n, n):
        raise ValueError(f"A 必须是 {n}×{n} 矩阵")
    if len(b) != n:
        raise ValueError(f"b 长度必须为 {n}")

    x = b.copy()
    for jcol in range(n):
        # 选主元
        piv = abs(A[jcol, jcol])
        ipiv = jcol
        for i in range(jcol + 1, n):
            if piv < abs(A[i, jcol]):
                piv = abs(A[i, jcol])
                ipiv = i
        if piv < 1e-15:
            raise ValueError(f"第 {jcol} 步主元为零，矩阵奇异")

        # 行交换
        if jcol != ipiv:
            A[[jcol, ipiv], :] = A[[ipiv, jcol], :]
            x[jcol], x[ipiv] = x[ipiv], x[jcol]

        # 归一化
        temp = A[jcol, jcol]
        A[jcol, jcol] = 1.0
        A[jcol, jcol + 1:] /= temp
        x[jcol] /= temp

        # 消元
        for i in range(jcol + 1, n):
            if abs(A[i, jcol]) > 1e-15:
                factor = -A[i, jcol]
                A[i, jcol] = 0.0
                A[i, jcol + 1:] += factor * A[jcol, jcol + 1:]
                x[i] += factor * x[jcol]

    # 回代
    for jcol in range(n - 1, 0, -1):
        x[:jcol] -= A[:jcol, jcol] * x[jcol]

    return x


# ============================================================================
# 5. SOR 迭代法 (源自 1099_sor)
# ============================================================================

def sor1(n: int, A: np.ndarray, b: np.ndarray, x: np.ndarray, w: float) -> np.ndarray:
    """
    执行一步 SOR (Successive Over-Relaxation) 迭代。

    SOR 迭代公式：
        x_i^{(k+1)} = (1 - ω)·x_i^{(k)} + (ω/a_ii)·[b_i - Σ_{j<i} a_{ij}·x_j^{(k+1)} - Σ_{j>i} a_{ij}·x_j^{(k)}]

    收敛条件：
        - A 严格对角占优或对称正定
        - 0 < ω < 2

    Parameters
    ----------
    n : int
        矩阵阶数。
    A : np.ndarray
        n × n 系数矩阵。
    b : np.ndarray
        右端项。
    x : np.ndarray
        当前解估计。
    w : float
        松弛因子，必须在 (0, 2) 内。

    Returns
    -------
    np.ndarray
        更新后的解估计。
    """
    if not (0.0 < w < 2.0):
        raise ValueError("SOR 松弛因子 ω 必须在 (0, 2) 区间内")
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    x_new = x.copy()
    for i in range(n):
        sigma = 0.0
        for j in range(n):
            if j != i:
                if j < i:
                    sigma += A[i, j] * x_new[j]
                else:
                    sigma += A[i, j] * x[j]
        if abs(A[i, i]) < 1e-14:
            raise ValueError(f"第 {i} 个对角元为零")
        x_new[i] = (1.0 - w) * x[i] + w * (b[i] - sigma) / A[i, i]
    return x_new


def sor_solve(A: np.ndarray, b: np.ndarray, w: float = 1.5,
              tol: float = 1e-8, max_iter: int = 10000) -> Tuple[np.ndarray, int]:
    """
    完整 SOR 求解器。

    Parameters
    ----------
    A, b, w
        同 sor1。
    tol : float
        收敛容差。
    max_iter : int
        最大迭代次数。

    Returns
    -------
    x : np.ndarray
        解向量。
    iters : int
        实际迭代次数。
    """
    n = len(b)
    x = np.zeros(n, dtype=float)
    for it in range(max_iter):
        x_old = x.copy()
        x = sor1(n, A, b, x, w)
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, it + 1
    return x, max_iter


# ============================================================================
# 6. Langford ODE 与 Runge-Kutta 积分 (源自 645_langford_ode)
# ============================================================================

def langford_parameters() -> Tuple[float, float, float, float, float, float,
                                    float, np.ndarray, float]:
    """
    Langford 系统参数。

    返回 (a, b, c, d, e, f, t0, xyz0, tstop)
    """
    a = 0.95
    b = 0.7
    c = 0.6
    d = 3.5
    e = 0.25
    f = 0.1
    t0 = 0.0
    xyz0 = np.array([0.1, 0.1, 0.1])
    tstop = 50.0
    return a, b, c, d, e, f, t0, xyz0, tstop


def langford_deriv(t: float, xyz: np.ndarray) -> np.ndarray:
    """
    Langford ODE 右端项。

    Langford 系统描述三维混沌吸引子：

        dx/dt = (z - b)·x - d·y
        dy/dt = d·x + (z - b)·y
        dz/dt = c + a·z - z³/3 - (x² + y²)·(1 + e·z) + f·z·x³

    参数：
        a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1

    Parameters
    ----------
    t : float
        时间。
    xyz : np.ndarray
        状态向量 [x, y, z]。

    Returns
    -------
    np.ndarray
        导数 [dx/dt, dy/dt, dz/dt]。
    """
    a, b, c, d, e, f, _, _, _ = langford_parameters()
    x, y, z = xyz[0], xyz[1], xyz[2]
    dxdt = (z - b) * x - d * y
    dydt = d * x + (z - b) * y
    dzdt = c + a * z - z**3 / 3.0 - (x**2 + y**2) * (1.0 + e * z) + f * z * x**3
    return np.array([dxdt, dydt, dzdt])


def rk4_step(f: Callable[[float, np.ndarray], np.ndarray],
             t: float, y: np.ndarray, h: float) -> np.ndarray:
    """
    经典四阶 Runge-Kutta 单步。

        k1 = h·f(t, y)
        k2 = h·f(t + h/2, y + k1/2)
        k3 = h·f(t + h/2, y + k2/2)
        k4 = h·f(t + h, y + k3)
        y_{n+1} = y_n + (k1 + 2·k2 + 2·k3 + k4) / 6

    局部截断误差 O(h⁵)。
    """
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_ode(f: Callable, y0: np.ndarray, t_span: Tuple[float, float],
                  n_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    ODE 数值积分。

    Parameters
    ----------
    f : Callable
        右端项函数 f(t, y)。
    y0 : np.ndarray
        初始条件。
    t_span : Tuple[float, float]
        时间区间 [t0, t1]。
    n_steps : int
        步数。

    Returns
    -------
    t : np.ndarray
        时间序列。
    y : np.ndarray
        状态序列，形状 (n_steps+1, len(y0))。
    """
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    y = np.zeros((n_steps + 1, len(y0)), dtype=float)
    t = np.linspace(t0, t1, n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        y[i + 1] = rk4_step(f, t[i], y[i], h)
    return t, y


# ============================================================================
# 7. 辅助数值函数
# ============================================================================

def weibull_pdf(u: np.ndarray, A: float, k: float) -> np.ndarray:
    """
    Weibull 概率密度函数：

        f(u) = (k/A)·(u/A)^{k-1}·exp(-(u/A)^k),  u ≥ 0

    Parameters
    ----------
    u : np.ndarray
        风速。
    A : float
        尺度参数。
    k : float
        形状参数。

    Returns
    -------
    np.ndarray
        PDF 值。
    """
    u = np.asarray(u, dtype=float)
    pdf = np.zeros_like(u)
    mask = u > 0
    pdf[mask] = (k / A) * (u[mask] / A)**(k - 1) * np.exp(-(u[mask] / A)**k)
    return pdf


def weibull_cdf(u: np.ndarray, A: float, k: float) -> np.ndarray:
    """
    Weibull 累积分布函数：

        F(u) = 1 - exp(-(u/A)^k),  u ≥ 0
    """
    u = np.asarray(u, dtype=float)
    cdf = np.zeros_like(u)
    mask = u > 0
    cdf[mask] = 1.0 - np.exp(-(u[mask] / A)**k)
    return cdf
