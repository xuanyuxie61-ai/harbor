"""
legendre_fd.py - 基于Legendre多项式的高阶有限差分格式构造

本模块融合以下种子项目的核心算法：
  - 664_legendre_product_polynomial → Legendre多项式递推与求值
  - 077_bernstein_approximation → Bernstein多项式逼近（用于光滑重建）

功能：
  1. Legendre多项式的递推计算与Gauss-Lobatto节点
  2. 基于Legendre展开的高阶有限差分格式（紧凑格式/Pade格式）
  3. 一阶/二阶导数的紧致差分格式系数
  4. 边界处单侧高阶格式
  5. Bernstein多项式用于等离子体剖面光滑重建
"""

import numpy as np
from typing import Tuple, Optional
from scipy.special import comb


# =============================================================================
# Legendre多项式（来自664_legendre_product_polynomial）
# =============================================================================
def legendre_value(n: int, x: float) -> float:
    """
    计算n阶Legendre多项式P_n(x)在x处的值

    使用三项递推关系:
      P_0(x) = 1
      P_1(x) = x
      (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)

    参数:
        n: 多项式阶数
        x: 求值点 [-1, 1]
    返回:
        P_n(x)
    """
    if n == 0:
        return 1.0
    if n == 1:
        return float(x)

    p_prev = 1.0
    p_curr = float(x)

    for k in range(1, n):
        p_next = ((2 * k + 1) * x * p_curr - k * p_prev) / (k + 1)
        p_prev = p_curr
        p_curr = p_next

    return p_curr


def legendre_values_vectorized(N: int, x: np.ndarray) -> np.ndarray:
    """
    计算0到N阶Legendre多项式在多个点的值

    参数:
        N: 最高阶数
        x: 求值点数组
    返回:
        P: shape (N+1, len(x)) P[k,:] = P_k(x)
    """
    nx = len(x)
    P = np.zeros((N + 1, nx))
    P[0, :] = 1.0
    if N >= 1:
        P[1, :] = x

    for n in range(1, N):
        P[n + 1, :] = ((2 * n + 1) * x * P[n, :] - n * P[n - 1, :]) / (n + 1)

    return P


def legendre_derivative_value(n: int, x: float) -> float:
    """
    计算Legendre多项式导数 P'_n(x)

    使用公式:
      P'_n(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)

    对于 |x| = 1 的特殊点:
      P'_n(1) = n*(n+1)/2
      P'_n(-1) = (-1)^{n+1} * n*(n+1)/2

    参数:
        n: 多项式阶数
        x: 求值点
    返回:
        P'_n(x)
    """
    if n == 0:
        return 0.0

    if abs(abs(x) - 1.0) < 1e-14:
        if abs(x - 1.0) < 1e-14:
            return n * (n + 1) / 2.0
        else:
            return (-1.0)**(n + 1) * n * (n + 1) / 2.0

    Pn = legendre_value(n, x)
    Pn_1 = legendre_value(n - 1, x)
    return n * (x * Pn - Pn_1) / (x**2 - 1.0 + 1e-30)


# =============================================================================
# Gauss-Lobatto-Legendre (GLL) 节点
# =============================================================================
def gauss_lobatto_legendre_nodes(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 N+1 个 Gauss-Lobatto-Legendre (GLL) 节点和权重

    GLL节点是 (1-x^2)*P'_N(x) = 0 的根
    包括端点 x = ±1

    使用Newton迭代求根

    参数:
        N: 多项式阶数（节点数 = N+1）
    返回:
        nodes: GLL节点（升序）
        weights: GLL积分权重
    """
    if N <= 0:
        return np.array([0.0]), np.array([2.0])
    if N == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    # 初始猜测：Chebyshev节点
    nodes = np.zeros(N + 1)
    nodes[0] = -1.0
    nodes[N] = 1.0
    for j in range(1, N):
        nodes[j] = -np.cos(np.pi * j / N)

    # Newton迭代
    for iteration in range(50):
        max_change = 0.0
        for j in range(1, N):
            x = nodes[j]
            # 计算 P_N(x) 和 P'_N(x)
            Pn = legendre_value(N, x)
            Pn_prime = legendre_derivative_value(N, x)

            # f(x) = (1-x^2) * P'_N(x) = 0
            f = (1.0 - x**2) * Pn_prime
            # f'(x) = -2x * P'_N(x) + (1-x^2) * P''_N(x)
            # 使用 P''_N = (2x*P'_N - N*(N+1)*P_N) / (1-x^2)
            f_prime = -2.0 * x * Pn_prime - N * (N + 1) * Pn

            if abs(f_prime) < 1e-30:
                continue

            dx = -f / f_prime
            nodes[j] += dx
            max_change = max(max_change, abs(dx))

        if max_change < 1e-15:
            break

    nodes.sort()

    # 计算权重: w_j = 2 / (N*(N+1)*P_N(x_j)^2)
    weights = np.zeros(N + 1)
    for j in range(N + 1):
        Pn = legendre_value(N, nodes[j])
        weights[j] = 2.0 / (N * (N + 1) * Pn**2 + 1e-30)

    return nodes, weights


# =============================================================================
# Vandermonde矩阵与微分矩阵
# =============================================================================
def vandermonde_matrix(N: int, x: np.ndarray) -> np.ndarray:
    """
    构造Legendre Vandermonde矩阵

    V_{ij} = P_j(x_i)

    参数:
        N: 最高阶数
        x: 节点坐标
    返回:
        V: shape (len(x), N+1)
    """
    return legendre_values_vectorized(N, x).T


def differentiation_matrix(N: int, x: np.ndarray) -> np.ndarray:
    """
    构造多项式微分矩阵 D

    使得 (D * f)_i ≈ f'(x_i)

    使用 D = V' * V^{-1}，其中 V 是Vandermonde矩阵

    参数:
        N: 多项式阶数
        x: GLL节点
    返回:
        D: shape (N+1, N+1) 微分矩阵
    """
    # 构造 V 和 V'
    V = vandermonde_matrix(N, x)

    # V'_{ij} = P'_j(x_i)
    Vp = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            Vp[i, j] = legendre_derivative_value(j, x[i])

    # D = V' * V^{-1}
    try:
        D = Vp @ np.linalg.inv(V)
    except np.linalg.LinAlgError:
        D = Vp @ np.linalg.pinv(V)

    return D


# =============================================================================
# 高阶紧致有限差分格式（Pade格式）
# =============================================================================
def compact_fd_coefficients_1st(order: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构造高阶紧致（compact/Pade）一阶导数格式

    紧致格式:
      alpha * f'_{i-1} + f'_i + alpha * f'_{i+1} =
        a * (f_{i+1} - f_{i-1}) / (2h) +
        b * (f_{i+2} - f_{i-2}) / (4h)

    对于第四阶:
      alpha = 1/4, a = 3/2, b = 0
    对于第六阶:
      alpha = 1/3, a = 14/9, b = 1/9

    通过匹配Taylor展开系数确定

    参数:
        order: 精度阶数（4或6）
    返回:
        alpha, a, b: 紧致格式系数
    """
    if order == 2:  # 标准中心差分
        return 0.0, 1.0, 0.0
    elif order == 4:
        return 0.25, 1.5, 0.0
    elif order == 6:
        return 1.0 / 3.0, 14.0 / 9.0, 1.0 / 9.0
    elif order == 8:
        # 八阶紧致格式
        alpha = 0.4
        a = 4.0 / 3.0 * (1.0 + alpha / 2.0)
        b = 1.0 / 3.0 * (alpha / 2.0 - 1.0 / 6.0)
        return alpha, a, b
    else:
        # 默认四阶
        return 0.25, 1.5, 0.0


def compact_fd_coefficients_2nd(order: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构造高阶紧致二阶导数格式

    紧致格式:
      alpha * f''_{i-1} + f''_i + alpha * f''_{i+1} =
        a * (f_{i+1} - 2*f_i + f_{i-1}) / h^2 +
        b * (f_{i+2} - 2*f_i + f_{i-2}) / (4*h^2)

    参数:
        order: 精度阶数（4或6）
    返回:
        alpha, a, b: 紧致格式系数
    """
    if order == 2:
        return 0.0, 1.0, 0.0
    elif order == 4:
        return 1.0 / 10.0, 6.0 / 5.0, 0.0
    elif order == 6:
        return 2.0 / 11.0, 12.0 / 11.0, 3.0 / 11.0
    else:
        return 1.0 / 10.0, 6.0 / 5.0, 0.0


def solve_tridiagonal_compact(rhs: np.ndarray, alpha: float) -> np.ndarray:
    """
    求解三对角系统: alpha*x_{i-1} + x_i + alpha*x_{i+1} = rhs_i

    使用Thomas算法（追赶法）

    参数:
        rhs: 右端项
        alpha: 非对角系数
    返回:
        solution: 解向量
    """
    n = len(rhs)
    if n < 3:
        return rhs.copy()

    # Thomas算法
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    # 前向消元
    c_prime[0] = alpha / (1.0 + 1e-30)
    d_prime[0] = rhs[0] / (1.0 + 1e-30)

    for i in range(1, n):
        m = 1.0 - alpha * c_prime[i - 1]
        if abs(m) < 1e-30:
            m = 1e-30
        if i < n - 1:
            c_prime[i] = alpha / m
        d_prime[i] = (rhs[i] - alpha * d_prime[i - 1]) / m

    # 回代
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def apply_compact_fd_1st(f: np.ndarray, h: float, order: int = 4) -> np.ndarray:
    """
    应用紧致有限差分计算一阶导数

    包含边界处理：使用单侧格式

    参数:
        f: 函数值数组
        h: 网格间距
        order: 内部格式精度
    返回:
        df: 一阶导数近似
    """
    n = len(f)
    df = np.zeros(n)
    alpha, a, b = compact_fd_coefficients_1st(order)

    # 内部点：构造右端项
    rhs = np.zeros(n)
    for i in range(2, n - 2):
        rhs[i] = a * (f[i + 1] - f[i - 1]) / (2.0 * h)
        if abs(b) > 1e-15:
            rhs[i] += b * (f[i + 2] - f[i - 2]) / (4.0 * h)

    # 边界处理（单侧高阶格式）
    # 左边界：二阶单侧
    if n >= 3:
        df[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * h)
        df[1] = (f[2] - f[0]) / (2.0 * h)
    # 右边界
    if n >= 3:
        df[n - 1] = (3.0 * f[n - 1] - 4.0 * f[n - 2] + f[n - 3]) / (2.0 * h)
        df[n - 2] = (f[n - 1] - f[n - 3]) / (2.0 * h)

    # 内部点求解紧致格式
    if alpha > 1e-15 and n > 4:
        rhs_inner = rhs[2:n - 2].copy()
        df_inner = solve_tridiagonal_compact(rhs_inner, alpha)
        df[2:n - 2] = df_inner

    return df


def apply_compact_fd_2nd(f: np.ndarray, h: float, order: int = 4) -> np.ndarray:
    """
    应用紧致有限差分计算二阶导数

    参数:
        f: 函数值数组
        h: 网格间距
        order: 内部格式精度
    返回:
        d2f: 二阶导数近似
    """
    n = len(f)
    d2f = np.zeros(n)
    alpha, a, b = compact_fd_coefficients_2nd(order)

    # 内部点
    rhs = np.zeros(n)
    for i in range(2, n - 2):
        rhs[i] = a * (f[i + 1] - 2.0 * f[i] + f[i - 1]) / (h**2)
        if abs(b) > 1e-15:
            rhs[i] += b * (f[i + 2] - 2.0 * f[i] + f[i - 2]) / (4.0 * h**2)

    # 边界处理
    if n >= 3:
        d2f[0] = (f[2] - 2.0 * f[1] + f[0]) / (h**2)
        d2f[n - 1] = (f[n - 1] - 2.0 * f[n - 2] + f[n - 3]) / (h**2)

    # 内部点求解
    if alpha > 1e-15 and n > 4:
        rhs_inner = rhs[2:n - 2].copy()
        d2f_inner = solve_tridiagonal_compact(rhs_inner, alpha)
        d2f[2:n - 2] = d2f_inner

    return d2f


# =============================================================================
# Bernstein多项式重建（来自077_bernstein_approximation）
# =============================================================================
def bernstein_basis(n: int, i: int, x: float, a: float = 0.0, b: float = 1.0) -> float:
    """
    计算Bernstein基函数 B_{i,n}(x) 在 [a,b] 上:

      B_{i,n}(x) = C(n,i) * ((b-x)/(b-a))^(n-i) * ((x-a)/(b-a))^i

    其中 C(n,i) = n! / (i! * (n-i)!) 是二项式系数

    参数:
        n: 多项式总阶数
        i: 基函数指标 (0 <= i <= n)
        x: 求值点
        a, b: 区间端点
    返回:
        B_{i,n}(x)
    """
    if x < a or x > b:
        return 0.0

    t = (x - a) / (b - a + 1e-30)
    coeff = comb(n, i, exact=True)
    return float(coeff * t**i * (1.0 - t)**(n - i))


def bernstein_approximation(f_values: np.ndarray, x_eval: np.ndarray,
                              a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """
    计算函数的Bernstein多项式逼近

    BPAB(F)(x) = sum_{i=0}^{n} F(x_i) * B_{i,n}(x)

    Bernstein多项式的优点：
    1. 一致收敛性（Weierstrass逼近定理）
    2. 保形性（单调性、凸性保持）
    3. 数值稳定性高（避免Runge现象）

    在等离子体物理中用于：
    - 光滑重建边界等离子体密度/温度剖面
    - 消除测量噪声的平滑效果

    参数:
        f_values: 等距节点上的函数值
        x_eval: 求值点
        a, b: 原始区间
    返回:
        逼近值数组
    """
    n = len(f_values) - 1
    x_nodes = np.linspace(a, b, n + 1)

    result = np.zeros(len(x_eval))
    for i in range(n + 1):
        for k, x in enumerate(x_eval):
            b_val = bernstein_basis(n, i, x, a, b)
            result[k] += f_values[i] * b_val

    return result


def bernstein_approximation_vectorized(f_values: np.ndarray, x_eval: np.ndarray,
                                         a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """
    Bernstein多项式逼近的向量化实现

    使用动态规划递推计算基函数避免大数溢出
    """
    n = len(f_values) - 1
    t = (x_eval - a) / (b - a + 1e-30)

    # 使用递推关系计算所有Bernstein基函数
    B = np.zeros((n + 1, len(x_eval)))
    B[0, :] = (1.0 - t)**n

    for i in range(1, n + 1):
        # 递推: B_{i,n} = C(n,i)/C(n,i-1) * t/(1-t) * B_{i-1,n}
        # 为避免除零，直接使用二项式系数
        coeff = comb(n, i, exact=True)
        B[i, :] = coeff * t**i * (1.0 - t)**(n - i)

    result = np.zeros(len(x_eval))
    for i in range(n + 1):
        result += f_values[i] * B[i, :]

    return result


# =============================================================================
# 格式精度验证（收敛阶测试）
# =============================================================================
def verify_fd_convergence(func, dfunc, h_values: np.ndarray,
                            x_domain: Tuple[float, float] = (0.0, 1.0),
                            order: int = 4) -> np.ndarray:
    """
    验证有限差分格式的收敛阶

    使用已知函数检验 df/dx 的计算精度

    参数:
        func: 测试函数 f(x)
        dfunc: 精确导数 f'(x)
        h_values: 网格间距数组
        x_domain: 计算域
        order: 紧致格式阶数
    返回:
        errors: 每个h对应的L2误差
    """
    errors = np.zeros(len(h_values))

    for k, h in enumerate(h_values):
        x = np.arange(x_domain[0], x_domain[1] + h * 0.5, h)
        f = func(x)
        df_exact = dfunc(x)

        df_computed = apply_compact_fd_1st(f, h, order)

        # 去掉边界点
        interior = slice(2, -2)
        errors[k] = np.sqrt(np.mean((df_computed[interior] - df_exact[interior])**2))

    return errors


def convergence_rate(errors: np.ndarray, h_values: np.ndarray) -> float:
    """
    计算收敛阶: rate = log(e1/e2) / log(h1/h2)
    """
    if len(errors) < 2:
        return 0.0

    rate = np.log(errors[0] / (errors[-1] + 1e-30)) / np.log(h_values[0] / (h_values[-1] + 1e-30))
    return float(rate)


# =============================================================================
# 多维Legendre乘积多项式（来自664_legendre_product_polynomial的高维推广）
# =============================================================================
def legendre_product_2d(n1: int, n2: int, x1: float, x2: float) -> float:
    """
    计算二维Legendre乘积多项式:

      L_{n1,n2}(x1, x2) = P_{n1}(x1) * P_{n2}(x2)

    在SOL输运方程中用于张量积基函数构造

    参数:
        n1, n2: 各方向阶数
        x1, x2: 坐标
    返回:
        L_{n1,n2}(x1, x2)
    """
    return legendre_value(n1, x1) * legendre_value(n2, x2)


def graded_lex_order_2d(max_degree: int) -> list:
    """
    生成二维多项式的分级字典序（grlex ordering）

    按总阶数递增排序，同阶数按字典序排列

    参数:
        max_degree: 最大总阶数
    返回:
        二元组列表 [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    """
    terms = []
    for total in range(max_degree + 1):
        for i in range(total + 1):
            terms.append((total - i, i))

    return terms
