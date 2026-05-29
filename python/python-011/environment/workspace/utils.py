# -*- coding: utf-8 -*-
"""
utils.py
--------
通用工具模块：集成多项式根界、Box-Behnken 实验设计、自然三次样条插值、
边界词多连方几何、数值鲁棒性边界处理等。

对应种子项目：
  - 897_polynomial_root_bound：Cauchy 根界 + 二分法
  - 111_box_behnken：实验设计矩阵生成
  - 593_interp_ncs：自然三次样条插值
  - 110_boundary_word_square：边界词几何（面积、质心、转动惯量）
"""

import numpy as np
from scipy.interpolate import CubicSpline


def polynomial_root_bound(coeffs):
    """
    计算复多项式的 Cauchy 根界。

    给定多项式
        p(z) = c[0] z^n + c[1] z^{n-1} + ... + c[n]
    构造关联实多项式
        q(x) = |c[0]| x^n - |c[1]| x^{n-1} - ... - |c[n]|.
    q(x)=0 的唯一正根 b 即为所有根的上界：|z_i| <= b。

    Parameters
    ----------
    coeffs : array_like, shape (n+1,)
        复多项式系数，按降幂排列。

    Returns
    -------
    b : float
        Cauchy 根界。若为零多项式则返回 0.0。
    """
    c = np.asarray(coeffs, dtype=complex).flatten()
    if c.size == 0:
        return 0.0
    # 去除尾部零系数（对应 z=0 的根）
    while c.size > 1 and np.isclose(c[-1], 0, atol=1e-15):
        c = c[:-1]
    if c.size == 0:
        return 0.0
    if c.size == 1:
        return 0.0

    # 构造 q(x) = |c0| x^n - |c1| x^{n-1} - ... - |cn|
    q = np.abs(c).astype(float)
    q[1:] = -q[1:]

    def qval(x):
        # Horner 法则求值
        res = q[0]
        for k in range(1, q.size):
            res = res * x + q[k]
        return res

    # 寻找 bracket: 从 x=1 开始倍增直到 q(x) > 0
    xpos = 1.0
    max_iter_bracket = 100
    for _ in range(max_iter_bracket):
        if qval(xpos) > 0:
            break
        xpos *= 2.0
        if xpos > 1e18:
            raise RuntimeError(" polynomial_root_bound: bracket 搜索失败，系数可能病态。")
    else:
        raise RuntimeError(" polynomial_root_bound: bracket 搜索迭代超限。")

    xneg = 0.0
    tol = 1e-12
    max_iter_bisect = 100
    for _ in range(max_iter_bisect):
        mid = (xneg + xpos) * 0.5
        if (xpos - xneg) < tol:
            break
        fm = qval(mid)
        if fm > 0:
            xpos = mid
        else:
            xneg = mid
    return xpos


def box_behnken_size(dim_num):
    """
    Box-Behnken 设计点数量：1 + dim_num * 2^(dim_num-1)。
    """
    if dim_num < 1:
        return 0
    return 1 + dim_num * (2 ** (dim_num - 1))


def box_behnken(dim_num, ranges):
    """
    生成 Box-Behnken 实验设计矩阵。

    对于 dim_num 个因子，每个因子取 low/high 两个水平，
    固定某一因子为中点，其余因子取全部 2^(dim_num-1) 种 low/high 组合。
    加上一个中心点。

    Parameters
    ----------
    dim_num : int
        因子维数，>=1。
    ranges : ndarray, shape (dim_num, 2)
        每行给出 [low, high]。

    Returns
    -------
    x : ndarray, shape (dim_num, x_num)
        设计矩阵，每列一个实验点。
    """
    if dim_num < 1:
        return np.zeros((0, 0))
    ranges = np.asarray(ranges, dtype=float)
    if ranges.shape != (dim_num, 2):
        raise ValueError("ranges 形状必须是 (dim_num, 2)。")
    x_num = box_behnken_size(dim_num)
    x = np.zeros((dim_num, x_num))
    col = 0
    # 中心点
    midpoint = (ranges[:, 0] + ranges[:, 1]) * 0.5
    x[:, col] = midpoint
    col += 1
    # 对每个因子 i，固定为中点，其余取全部组合
    for i in range(dim_num):
        others = [j for j in range(dim_num) if j != i]
        n_others = len(others)
        n_combo = 2 ** n_others
        # 二进制枚举
        vals = np.zeros((dim_num, n_combo))
        vals[i, :] = midpoint[i]
        for combo in range(n_combo):
            bits = combo
            for idx, j in enumerate(others):
                bit = bits & 1
                bits >>= 1
                if bit == 0:
                    vals[j, combo] = ranges[j, 0]
                else:
                    vals[j, combo] = ranges[j, 1]
        x[:, col:col + n_combo] = vals
        col += n_combo
    return x


def natural_cubic_spline(xd, yd, xs):
    """
    自然三次样条插值（边界二阶导数为零）。

    求解三弯矩方程 M_i，满足 M_0 = M_{n-1} = 0，
    其中 h_i = x_{i+1} - x_i，
        mu_i M_{i-1} + 2 M_i + lambda_i M_{i+1} = d_i，
    然后逐段用 Hermite 形式求值。

    Parameters
    ----------
    xd : ndarray, shape (n,)
        插值节点，要求严格递增。
    yd : ndarray, shape (n,)
        节点函数值。
    xs : ndarray, shape (m,)
        待求点。

    Returns
    -------
    ys : ndarray, shape (m,)
        插值结果。
    """
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)
    xs = np.asarray(xs, dtype=float)
    n = xd.size
    if n < 2:
        raise ValueError("自然三次样条至少需要 2 个节点。")
    if not np.all(np.diff(xd) > 0):
        raise ValueError("插值节点 xd 必须严格递增。")
    if yd.shape[0] != n:
        raise ValueError("xd 与 yd 长度不一致。")

    # 使用 scipy 的 CubicSpline 设置边界条件为自然（bc_type='natural'）
    cs = CubicSpline(xd, yd, bc_type='natural')
    return cs(xs)


# ---------------------------------------------------------------------------
# 边界词多连方几何（来自 110_boundary_word_square）
# ---------------------------------------------------------------------------

def _dir_delta(ch):
    """方向字符到位移向量。"""
    mapping = {'u': (0, 1), 'd': (0, -1), 'l': (-1, 0), 'r': (1, 0)}
    return mapping.get(ch, (0, 0))


def boundary_word_check(word):
    """
    检查边界词是否构成闭合不自交的正交回路。
    返回 (is_valid, msg)。
    """
    word = str(word)
    if len(word) == 0:
        return False, "空边界词。"
    dx = sum(_dir_delta(ch)[0] for ch in word)
    dy = sum(_dir_delta(ch)[1] for ch in word)
    if dx != 0 or dy != 0:
        return False, "边界词未闭合。"
    # 检查无立即回溯
    opp = {'u': 'd', 'd': 'u', 'l': 'r', 'r': 'l'}
    for i in range(len(word)):
        if word[i] == opp.get(word[(i + 1) % len(word)], ''):
            return False, "存在相邻回溯步。"
    return True, "OK"


def boundary_word_area(word):
    """
    由边界词计算多连方面积（离散 Green 定理）。
    沿边界词行走，顶点为 (x_k, y_k)，面积为
        A = (1/2) |Σ (x_k y_{k+1} - x_{k+1} y_k)|。
    """
    word = str(word)
    x, y = [0], [0]
    for ch in word:
        dx, dy = _dir_delta(ch)
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    # 闭合
    A = 0.0
    for i in range(len(x) - 1):
        A += x[i] * y[i + 1] - x[i + 1] * y[i]
    return abs(A) * 0.5


def boundary_word_perimeter(word):
    """边界词周长 = 步数。"""
    return len(str(word))


def boundary_word_centroid(word):
    """
    由边界词计算多连方质心（利用边界积分）。
    Cx = (1/(6A)) Σ (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
    Cy = (1/(6A)) Σ (y_i + y_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
    """
    word = str(word)
    x, y = [0], [0]
    for ch in word:
        dx, dy = _dir_delta(ch)
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    A = 0.0
    cx_num = 0.0
    cy_num = 0.0
    for i in range(len(x) - 1):
        cross = x[i] * y[i + 1] - x[i + 1] * y[i]
        A += cross
        cx_num += (x[i] + x[i + 1]) * cross
        cy_num += (y[i] + y[i + 1]) * cross
    A *= 0.5
    if abs(A) < 1e-15:
        return 0.0, 0.0
    return cx_num / (6.0 * A), cy_num / (6.0 * A)


def boundary_word_moment(word):
    """
    计算多连方关于质心的转动惯量（离散 Green 定理 + 平行轴定理）。
    I = Σ∫ x^2 + y^2 dA，利用多边形公式先算关于原点的 I0，再减去 A*(Cx^2+Cy^2)。
    """
    word = str(word)
    x, y = [0], [0]
    for ch in word:
        dx, dy = _dir_delta(ch)
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    A = 0.0
    I0 = 0.0
    for i in range(len(x) - 1):
        cross = x[i] * y[i + 1] - x[i + 1] * y[i]
        A += cross
        # 多边形关于原点的极转动惯量公式（使用绝对值保证正定性）
        I0 += (x[i] ** 2 + x[i] * x[i + 1] + x[i + 1] ** 2 +
               y[i] ** 2 + y[i] * y[i + 1] + y[i + 1] ** 2) * abs(cross)
    A = abs(A) * 0.5
    if A < 1e-15:
        return 0.0
    I0 /= 12.0
    cx, cy = boundary_word_centroid(word)
    return I0 - A * (cx ** 2 + cy ** 2)


# ---------------------------------------------------------------------------
# 通用数值鲁棒性工具
# ---------------------------------------------------------------------------

def safe_sqrt(x, eps=1e-15):
    """保证非负的平方根。"""
    return np.sqrt(np.maximum(x, eps))


def safe_divide(a, b, eps=1e-15):
    """安全除法，防止除以零。"""
    return np.where(np.abs(b) > eps, a / b, 0.0)


def fermi_dirac(energy, beta, mu=0.0):
    """
    Fermi-Dirac 分布函数：
        f(ε) = 1 / (exp(β(ε-μ)) + 1)
    对大正指数截断防止溢出。
    """
    arg = beta * (energy - mu)
    # 截断
    arg = np.clip(arg, -700.0, 700.0)
    return 1.0 / (np.exp(arg) + 1.0)


def kron_delta(i, j):
    """Kronecker delta。"""
    return 1.0 if i == j else 0.0
