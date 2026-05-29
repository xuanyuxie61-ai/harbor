"""
utils.py — 数学与数值工具箱
=====================================
为脑机接口神经信号解码系统提供底层数学支持。
包含 Clenshaw 递推、Chebyshev 级数求值、特殊函数及矩阵/向量运算工具。
"""

import numpy as np
from numpy.linalg import norm


def clenshaw_chebyshev_eval(x, coeffs):
    """
    Clenshaw 递推求 Chebyshev 级数值。
    对系数向量 a = [a_0, a_1, ..., a_n]，计算
        S(x) = sum_{k=0}^{n} a_k * T_k(x)
    其中 T_k(x) 为第一类 Chebyshev 多项式。
    递推关系：
        b_n     = a_n
        b_{n-1} = a_{n-1} + 2x * b_n
        b_{n-2} = a_{n-2} + 2x * b_{n-1} - b_n
        ...
        b_0     = a_0 + x * b_1 - b_2   （注意首项系数只乘 1 而非 2）
    结果：S(x) = b_0 - b_2  （对首项归一化后的标准形式）。
    这里采用更稳健的实现：将首项单独处理。
    """
    a = np.asarray(coeffs, dtype=float)
    n = len(a) - 1
    if n < 0:
        return 0.0
    if n == 0:
        return float(a[0])
    # 标准 Clenshaw
    b2 = 0.0
    b1 = 0.0
    for k in range(n, 0, -1):
        b0 = a[k] + 2.0 * x * b1 - b2
        b2 = b1
        b1 = b0
    # k=0 时系数不乘 2
    return float(a[0] + x * b1 - b2)


def chebyshev_coefficients_from_function(f, n, a=-1.0, b=1.0):
    """
    通过离散 Chebyshev 变换求函数 f 在 [a,b] 上的 n 次 Chebyshev 逼近系数。
    使用 Chebyshev-Gauss-Lobatto 节点：
        x_j = cos(pi * j / n),  j=0,...,n
    变换公式：
        c_k = (2/n) * sum_{j=0}^{n}'' f(x_j) * T_k(x_j)
    其中双撇号表示首末项权重为 1/2。
    """
    # 在 [-1,1] 上的节点
    j = np.arange(n + 1)
    x_tilde = np.cos(np.pi * j / n)
    # 映射到 [a,b]
    x = 0.5 * (b - a) * x_tilde + 0.5 * (b + a)
    fx = np.array([f(xi) for xi in x], dtype=float)
    coeffs = np.zeros(n + 1, dtype=float)
    for k in range(n + 1):
        Tk = np.cos(k * np.arccos(x_tilde))
        if k == 0:
            coeffs[k] = np.sum(fx * Tk) / n
        else:
            coeffs[k] = 2.0 * np.sum(fx * Tk) / n
    # 修正端点权重（在离散余弦变换中已自动体现，这里直接按标准公式）
    coeffs[0] *= 0.5
    coeffs[n] *= 0.5
    return coeffs


def sawtooth_wave(t, omega, amplitude=1.0):
    """
    周期锯齿波：s(t) = amplitude * (mod(t + pi/omega, 2*pi/omega) - pi/omega)
    对应角频率 omega 的标准锯齿波。
    """
    period = 2.0 * np.pi / omega
    return amplitude * (np.mod(t + np.pi / omega, period) - np.pi / omega)


def sigmoid_activation(x, theta=0.0, sigma=1.0):
    """
    神经激活函数：S(x) = 1 / (1 + exp(-(x - theta)/sigma))
    theta 为阈值，sigma 为增益参数的倒数。
    边界处理：对大 |x| 使用饱和值避免 overflow。
    """
    x = np.asarray(x, dtype=float)
    z = -(x - theta) / sigma
    # 截断防止 exp 溢出
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(z))


def softplus(x):
    """Softplus 激活：log(1+exp(x))，带截断。"""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mask = x > 100
    out[mask] = x[mask]
    out[~mask] = np.log1p(np.exp(x[~mask]))
    return out


def rk4_step(f, t, y, h):
    """
    经典四阶 Runge-Kutta 单步积分。
    y' = f(t, y)
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def gauss_legendre_nodes_weights(n):
    """
    使用 numpy 的 legendre 多项式根计算 Gauss-Legendre 节点与权重。
    对标准区间 [-1, 1]：节点为 P_n(x)=0 的根 x_i，
    权重 w_i = 2 / [(1 - x_i^2) * (P_n'(x_i))^2]
    利用 numpy.polynomial.legendre.leggauss 保证数值稳定。
    """
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    return x, w


def barycentric_lagrange_interpolate(x_nodes, y_nodes, x_eval):
    """
    重心坐标 Lagrange 插值。对 Chebyshev 节点具有 O(n) 复杂度和良好稳定性。
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    n = len(x_nodes)
    # 计算重心权重 b_j = (-1)^j （对 Chebyshev 第一类节点的近似）
    # 更一般地，用标准公式
    b = np.ones(n)
    b[0] = 0.5
    b[-1] = 0.5
    b[1::2] *= -1.0
    # 若节点不是 Chebyshev，可用精确权重（这里假设 Chebyshev-Gauss-Lobatto）
    diff = x_eval[:, None] - x_nodes[None, :]
    # 避免除以零
    exact = np.isclose(diff, 0.0)
    if np.any(exact):
        # 在节点处直接返回原值
        result = np.zeros(len(x_eval), dtype=float)
        for i, xe in enumerate(x_eval):
            mask = np.isclose(x_nodes, xe)
            if np.any(mask):
                result[i] = y_nodes[np.argmax(mask)]
            else:
                num = np.sum(b * y_nodes / (xe - x_nodes))
                den = np.sum(b / (xe - x_nodes))
                result[i] = num / den
        return result
    num = np.sum(b * y_nodes / diff, axis=1)
    den = np.sum(b / diff, axis=1)
    return num / den


def sparse_adjacency_to_laplacian(A):
    """
    从稀疏邻接矩阵 A 计算图拉普拉斯 L = D - A，其中 D 为度矩阵。
    """
    A = np.asarray(A, dtype=float)
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    return D - A


def matrix_exponential_power_series(A, t, terms=20):
    """
    通过截断幂级数计算 exp(A*t)：
        exp(At) = sum_{k=0}^{infty} (At)^k / k!
    用于小 t 值的近似。对大脑网络动力学的小时间步近似有效。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    I = np.eye(n)
    result = I.copy()
    term = I.copy()
    for k in range(1, terms):
        term = term @ (A * t) / k
        result += term
        if norm(term, ord='fro') < 1e-15:
            break
    return result


def safe_log1p_exp(x):
    """数值稳定的 log(1+exp(x))。"""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x > 20
    neg = x < -20
    mid = ~(pos | neg)
    out[pos] = x[pos]
    out[neg] = np.exp(x[neg])
    out[mid] = np.log1p(np.exp(x[mid]))
    return out


def softmax_stable(x):
    """数值稳定的 softmax。"""
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    ex = np.exp(x - m)
    return ex / np.sum(ex)
