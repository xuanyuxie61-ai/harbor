"""
dispersion_calculus.py
色散曲线的高精度插值、Chebyshev谱展开与数值稳定性分析

融合原项目:
  - 159_chebyshev: Chebyshev多项式插值系数计算
  - 1388_vanloan (CubicSpline): 三次样条插值
  - 658_lebesgue: Lebesgue常数估计

科学背景:
  光子晶体光纤的色散特性 D(lambda) 或 beta(omega) 通常由全矢量有限元
  计算得到离散数据点。在广义非线性薛定谔方程（GNLSE）中，需要极高精度
  的色散算子 
      hat{D} = sum_{m>=2} i^{m+1} beta_m / m! * d^m/dT^m
  其中 beta_m 为在中心频率 omega_0 处的泰勒展开系数。
  本模块提供:
    1. Chebyshev谱展开用于全局光滑色散曲线的近似；
    2. 分段三次样条插值用于局部精细结构；
    3. Lebesgue常数监控插值稳定性，防止Runge现象。
"""

import numpy as np
from typing import Callable, Tuple, Optional


def chebyshev_zeros(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """
    计算区间 [a, b] 上的 n 阶 Chebyshev 节点（第一类）。

    公式:
        x_k = (a+b)/2 + (b-a)/2 * cos( (2k-1)*pi / (2n) ),  k = 1,...,n

    Chebyshev节点在区间两端密集分布，可有效抑制多项式插值的Runge振荡。

    Parameters
    ----------
    n : int
        节点数，n >= 1。
    a, b : float
        区间端点。

    Returns
    -------
    np.ndarray
        形状为 (n,) 的节点坐标。
    """
    if n < 1:
        raise ValueError("chebyshev_zeros: n must be >= 1")
    angles = (2.0 * np.arange(1, n + 1) - 1.0) * np.pi / (2.0 * n)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(angles)
    return x


def chebyshev_coefficients(a: float, b: float, n: int, f: Callable) -> np.ndarray:
    """
    计算函数 f 在区间 [a, b] 上的 Chebyshev 插值系数。

    算法（Broucke ACM 446）:
        1. 在 Chebyshev 节点 x_k 上采样 f(x_k)。
        2. 通过离散余弦变换（DCT）计算系数 c_j:
           c_j = (2/n) * sum_{k=1}^{n} f(x_k) * cos( pi * j * (2k-1) / (2n) )

    插值多项式表示为:
        P_n(x) = sum_{j=0}^{n-1} c_j * T_j( (2x - a - b)/(b - a) )
    其中 T_j 为第一类 Chebyshev 多项式:
        T_0(t) = 1
        T_1(t) = t
        T_{j+1}(t) = 2t * T_j(t) - T_{j-1}(t)

    Parameters
    ----------
    a, b : float
        区间端点。
    n : int
        插值阶数。
    f : callable
        目标函数 f(x)，接受数组输入。

    Returns
    -------
    np.ndarray
        形状为 (n,) 的 Chebyshev 系数。
    """
    if n < 1:
        raise ValueError("chebyshev_coefficients: n must be >= 1")
    if b <= a:
        raise ValueError("chebyshev_coefficients: must have b > a")
    x = chebyshev_zeros(n, a, b)
    fx = f(x)
    c = np.zeros(n)
    for j in range(n):
        s = 0.0
        for k in range(n):
            s += fx[k] * np.cos(np.pi * j * (2.0 * k + 1.0) / (2.0 * n))
        c[j] = 2.0 * s / n
    return c


def chebyshev_interpolant(c: np.ndarray, a: float, b: float, x: np.ndarray) -> np.ndarray:
    """
    使用 Clenshaw 递推算法求值 Chebyshev 插值多项式。

    对于 t = (2x - a - b)/(b - a) in [-1, 1]，Clenshaw 递推:
        b_n = c_n
        b_{n-1} = c_{n-1} + 2t * b_n
        ...
        b_0 = c_0 + t * b_1
        P(x) = b_0 - t * b_1   （修正最后一步）

    实际标准 Clenshaw 算法:
        y_{n+1} = y_n = 0
        y_{k} = 2t * y_{k+1} - y_{k+2} + c_k
        P = y_0 - t * y_1   （当 c_0 未减半时）
        或 P = 0.5*(y_0 - y_2) （Clenshaw 标准形式）

    此处采用简化但稳定的形式:
        P = sum_{j=0}^{n-1} c_j * T_j(t)
    通过递推 T_j 实现。

    Parameters
    ----------
    c : np.ndarray
        Chebyshev 系数，形状 (n,)。
    a, b : float
        原始区间。
    x : np.ndarray
        求值点。

    Returns
    -------
    np.ndarray
        插值结果。
    """
    if b <= a:
        raise ValueError("chebyshev_interpolant: must have b > a")
    n = len(c)
    t = (2.0 * x - a - b) / (b - a)
    # 限制在 [-1, 1]
    t = np.clip(t, -1.0, 1.0)
    # 递推计算所有 T_j
    if n == 1:
        return np.full_like(x, c[0])
    T0 = np.ones_like(x)
    T1 = t.copy()
    y = c[0] * T0 + c[1] * T1
    for j in range(2, n):
        T2 = 2.0 * t * T1 - T0
        y += c[j] * T2
        T0, T1 = T1, T2
    return y


def cubic_spline_coefficients(x: np.ndarray, y: np.ndarray,
                              derivative: int = 1,
                              muL: Optional[float] = None,
                              muR: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算三次样条插值系数。

    在区间 [x[i], x[i+1]] 上，样条表示为:
        S_i(z) = a_i + b_i*(z-x_i) + c_i*(z-x_i)^2 + d_i*(z-x_i)^2*(z-x_{i+1})

    等价于标准形式:
        S_i(z) = a_i + b_i*(z-x_i) + c_i*(z-x_i)^2 + d_i*(z-x_i)^3
    其中 d_i_std = c_i - d_i * dx_i （需要转换）。

    本实现采用 Van Loan 教材中的 not-a-knot 边界条件或导数边界条件。

    Parameters
    ----------
    x : np.ndarray
        节点坐标，严格递增，长度 n >= 4。
    y : np.ndarray
        节点函数值，长度 n。
    derivative : int
        边界条件类型：1 表示一阶导数给定，2 表示二阶导数给定。
        若 muL, muR 为 None，则使用 not-a-knot 条件。
    muL, muR : float, optional
        左、右端点的指定导数值。

    Returns
    -------
    tuple
        (a, b, c, d)，各为长度 n-1 的数组。
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    n = len(x)
    if n < 4:
        raise ValueError("cubic_spline_coefficients: need at least 4 points")
    if len(y) != n:
        raise ValueError("cubic_spline_coefficients: x and y must have same length")
    if np.any(np.diff(x) <= 0):
        raise ValueError("cubic_spline_coefficients: x must be strictly increasing")
    Dx = np.diff(x)
    yp = np.diff(y) / Dx
    # 构建三对角系统求内部斜率 s[1:n-1]
    T = np.zeros((n - 2, n - 2))
    r = np.zeros(n - 2)
    for i in range(1, n - 3):
        T[i, i] = 2.0 * (Dx[i] + Dx[i + 1])
        T[i, i - 1] = Dx[i + 1]
        T[i, i + 1] = Dx[i]
        r[i] = 3.0 * (Dx[i + 1] * yp[i] + Dx[i] * yp[i + 1])
    if muL is not None and muR is not None:
        if derivative == 1:
            T[0, 0] = 2.0 * (Dx[0] + Dx[1])
            T[0, 1] = Dx[0]
            r[0] = 3.0 * (Dx[1] * yp[0] + Dx[0] * yp[1]) - Dx[1] * muL
            T[n - 3, n - 3] = 2.0 * (Dx[n - 3] + Dx[n - 2])
            T[n - 3, n - 4] = Dx[n - 2]
            r[n - 3] = 3.0 * (Dx[n - 2] * yp[n - 3] + Dx[n - 3] * yp[n - 2]) - Dx[n - 3] * muR
            s = np.concatenate([[muL], np.linalg.solve(T, r), [muR]])
        elif derivative == 2:
            T[0, 0] = 2.0 * Dx[0] + 1.5 * Dx[1]
            T[0, 1] = Dx[0]
            r[0] = 1.5 * Dx[1] * yp[0] + 3.0 * Dx[0] * yp[1] + Dx[0] * Dx[1] * muL / 4.0
            T[n - 3, n - 3] = 1.5 * Dx[n - 3] + 2.0 * Dx[n - 2]
            T[n - 3, n - 4] = Dx[n - 2]
            r[n - 3] = (3.0 * Dx[n - 2] * yp[n - 3] + 1.5 * Dx[n - 3] * yp[n - 2]
                        - Dx[n - 2] * Dx[n - 3] * muR / 4.0)
            stilde = np.linalg.solve(T, r)
            s1 = (3.0 * yp[0] - stilde[0] - muL * Dx[0] / 2.0) / 2.0
            sn = (3.0 * yp[n - 2] - stilde[n - 3] + muR * Dx[n - 2] / 2.0) / 2.0
            s = np.concatenate([[s1], stilde, [sn]])
        else:
            raise ValueError("cubic_spline_coefficients: derivative must be 1 or 2")
    else:
        # not-a-knot 条件
        q = Dx[0] * Dx[0] / Dx[1]
        T[0, 0] = 2.0 * Dx[0] + Dx[1] + q
        T[0, 1] = Dx[0] + q
        r[0] = Dx[1] * yp[0] + Dx[0] * yp[1] + 2.0 * yp[1] * (q + Dx[0])
        q = Dx[n - 2] * Dx[n - 2] / Dx[n - 3]
        T[n - 3, n - 3] = 2.0 * Dx[n - 2] + Dx[n - 3] + q
        T[n - 3, n - 4] = Dx[n - 2] + q
        r[n - 3] = (Dx[n - 2] * yp[n - 3] + Dx[n - 3] * yp[n - 2]
                    + 2.0 * yp[n - 2] * (Dx[n - 2] + q))
        stilde = np.linalg.solve(T, r)
        s1 = -stilde[0] + 2.0 * yp[0]
        s1 = s1 + ((Dx[0] / Dx[1]) ** 2) * (stilde[0] + stilde[1] - 2.0 * yp[1])
        sn = -stilde[n - 3] + 2.0 * yp[n - 2]
        sn = sn + ((Dx[n - 2] / Dx[n - 3]) ** 2) * (stilde[n - 4] + stilde[n - 3] - 2.0 * yp[n - 3])
        s = np.concatenate([[s1], stilde, [sn]])
    a = y[:-1].copy()
    b = s[:-1].copy()
    c = (yp - s[:-1]) / Dx
    d = (s[1:] + s[:-1] - 2.0 * yp) / (Dx * Dx)
    return a, b, c, d


def cubic_spline_eval(x: np.ndarray, a: np.ndarray, b: np.ndarray,
                      c: np.ndarray, d: np.ndarray, xk: np.ndarray) -> np.ndarray:
    """
    在指定点 x 上求值三次样条。

    Parameters
    ----------
    x : np.ndarray
        求值点。
    a, b, c, d : np.ndarray
        样条系数（长度为 n-1）。
    xk : np.ndarray
        原始节点坐标（长度为 n）。

    Returns
    -------
    np.ndarray
        求值结果。
    """
    x = np.asarray(x, dtype=float)
    n = len(xk)
    m = len(a)
    if m != n - 1:
        raise ValueError("cubic_spline_eval: coefficient length mismatch")
    y = np.zeros_like(x)
    for i in range(len(x)):
        xi = x[i]
        # 二分查找所属区间
        if xi <= xk[0]:
            idx = 0
        elif xi >= xk[n - 2]:
            idx = n - 2
        else:
            idx = int(np.searchsorted(xk[1:-1], xi))
        dx = xi - xk[idx]
        y[i] = a[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx
    return y


def lebesgue_function(n: int, x_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    计算一组插值节点的 Lebesgue 函数。

    对于 Lagrange 基函数 l_j(x)，Lebesgue 函数定义为:
        L(x) = sum_{j=1}^{n} |l_j(x)|
    其中
        l_j(x) = prod_{k!=j} (x - x_k) / (x_j - x_k)

    Lebesgue 函数的上确界即为 Lebesgue 常数 Lambda_n，它量化了
    插值过程的稳定性: ||P - P_exact||_inf <= (1 + Lambda_n) * epsilon。

    Parameters
    ----------
    n : int
        节点数。
    x_nodes : np.ndarray
        插值节点，形状 (n,)。
    x_eval : np.ndarray
        求值点，形状 (m,)。

    Returns
    -------
    np.ndarray
        Lebesgue 函数值，形状 (m,)。
    """
    if len(x_nodes) != n:
        raise ValueError("lebesgue_function: node count mismatch")
    m = len(x_eval)
    lfun = np.zeros(m)
    for j in range(n):
        lj = np.ones(m)
        for k in range(n):
            if k != j:
                denom = x_nodes[j] - x_nodes[k]
                if abs(denom) < 1e-15:
                    continue
                lj *= (x_eval - x_nodes[k]) / denom
        lfun += np.abs(lj)
    return lfun


def lebesgue_constant(n: int, x_nodes: np.ndarray, x_eval: np.ndarray) -> float:
    """
    估计 Lebesgue 常数（Lebesgue 函数的上确界）。

    对于 Chebyshev 节点，Lebesgue 常数以 O(log n) 增长；
    对于等距节点，则以 2^n / (n log n) 指数增长。
    因此，Lebesgue 常数是判断插值节点优劣的关键指标。

    Parameters
    ----------
    n : int
        节点数。
    x_nodes : np.ndarray
        插值节点。
    x_eval : np.ndarray
        密集的求值点，用于估计上确界。

    Returns
    -------
    float
        Lebesgue 常数的估计值。
    """
    lfun = lebesgue_function(n, x_nodes, x_eval)
    return float(np.max(lfun))


def dispersion_taylor_coefficients(omega: np.ndarray, beta: np.ndarray,
                                    omega0: float, order: int = 6) -> np.ndarray:
    """
    从离散色散曲线 beta(omega) 提取中心频率 omega0 处的泰勒展开系数 beta_m。

    GNLSE 中的色散算子:
        hat{D} = sum_{m=2}^{M} (i^{m+1} / m!) * beta_m * d^m/dT^m

    其中 beta_m = d^m beta / d omega^m |_{omega=omega0}。

    算法:
        1. 对 beta(omega) 做 Chebyshev 全局拟合或样条局部拟合。
        2. 在 omega0 附近选取邻域，用多项式最小二乘拟合求各阶导数。
        3. 通过 Lebesgue 常数评估插值稳定性。

    Parameters
    ----------
    omega : np.ndarray
        角频率采样点（rad/ps）。
    beta : np.ndarray
        传播常数（1/m）。
    omega0 : float
        中心角频率。
    order : int
        泰勒展开最高阶数（>= 2）。

    Returns
    -------
    np.ndarray
        beta_coeffs，形状 (order+1,)，其中 beta_coeffs[m] = beta_m / m!。
    """
    if order < 2:
        raise ValueError("dispersion_taylor_coefficients: order must be >= 2")
    omega = np.asarray(omega, dtype=float)
    beta = np.asarray(beta, dtype=float)
    # 选取 omega0 附近的邻域（避免全局外推误差）
    delta = np.max(np.abs(omega - omega0)) * 0.3
    mask = np.abs(omega - omega0) <= delta
    if np.sum(mask) < order + 2:
        # 如果不够点，扩大邻域
        mask = np.abs(omega - omega0) <= delta * 2.0
    x_local = omega[mask] - omega0
    y_local = beta[mask]
    # 数值稳定性：对 x_local 做无量纲缩放
    # 使用 x_scaled = x_local / x_scale，拟合后再转换系数
    x_scale = np.max(np.abs(x_local))
    if x_scale < 1e-30:
        x_scale = 1.0
    x_scaled = x_local / x_scale
    # 用最小二乘多项式拟合求导
    # 设计矩阵: [1, x, x^2, ..., x^order]
    V = np.vander(x_scaled, order + 1, increasing=True)
    coeffs_scaled, _, _, _ = np.linalg.lstsq(V, y_local, rcond=None)
    # 转换回原始单位: beta_m / m! = coeffs_scaled[m] / (x_scale^m)
    coeffs = np.zeros_like(coeffs_scaled)
    for m in range(len(coeffs_scaled)):
        coeffs[m] = coeffs_scaled[m] / (x_scale ** m)
    return coeffs


def sellmeier_equation_silica(wavelength_um: np.ndarray) -> np.ndarray:
    """
    石英的 Sellmeier 色散方程。

    公式（标准三组 Sellmeier）:
        n^2(lambda) = 1 + sum_{i=1}^{3} B_i * lambda^2 / (lambda^2 - C_i)

    其中 lambda 单位为 um，系数为:
        B1 = 6.961663e-1,  C1 = 4.679148e-3
        B2 = 4.079426e-1,  C2 = 1.351206e-2
        B3 = 8.974794e-1,  C3 = 9.896161e+1

    Parameters
    ----------
    wavelength_um : np.ndarray
        波长（um）。

    Returns
    -------
    np.ndarray
        折射率 n。
    """
    lam2 = wavelength_um ** 2
    B = np.array([0.6961663, 0.4079426, 0.8974794])
    C = np.array([0.004679148, 0.01351206, 98.96161])
    n2 = np.ones_like(wavelength_um)
    for Bi, Ci in zip(B, C):
        n2 += Bi * lam2 / (lam2 - Ci)
    return np.sqrt(n2)


def beta_from_sellmeier(wavelength_um: np.ndarray) -> np.ndarray:
    """
    由 Sellmeier 方程计算传播常数 beta = n * omega / c = 2*pi*n / lambda。

    Parameters
    ----------
    wavelength_um : np.ndarray
        波长（um）。

    Returns
    -------
    np.ndarray
        beta（1/m）。
    """
    n = sellmeier_equation_silica(wavelength_um)
    # lambda in um -> m: multiply by 1e-6
    beta = 2.0 * np.pi * n / (wavelength_um * 1e-6)
    return beta
