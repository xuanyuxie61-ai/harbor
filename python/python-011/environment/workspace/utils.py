# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import CubicSpline


def polynomial_root_bound(coeffs):
    c = np.asarray(coeffs, dtype=complex).flatten()
    if c.size == 0:
        return 0.0

    while c.size > 1 and np.isclose(c[-1], 0, atol=1e-15):
        c = c[:-1]
    if c.size == 0:
        return 0.0
    if c.size == 1:
        return 0.0


    q = np.abs(c).astype(float)
    q[1:] = -q[1:]

    def qval(x):

        res = q[0]
        for k in range(1, q.size):
            res = res * x + q[k]
        return res


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
    if dim_num < 1:
        return 0
    return 1 + dim_num * (2 ** (dim_num - 1))


def box_behnken(dim_num, ranges):
    if dim_num < 1:
        return np.zeros((0, 0))
    ranges = np.asarray(ranges, dtype=float)
    if ranges.shape != (dim_num, 2):
        raise ValueError("ranges 形状必须是 (dim_num, 2)。")
    x_num = box_behnken_size(dim_num)
    x = np.zeros((dim_num, x_num))
    col = 0

    midpoint = (ranges[:, 0] + ranges[:, 1]) * 0.5
    x[:, col] = midpoint
    col += 1

    for i in range(dim_num):
        others = [j for j in range(dim_num) if j != i]
        n_others = len(others)
        n_combo = 2 ** n_others

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


    cs = CubicSpline(xd, yd, bc_type='natural')
    return cs(xs)






def _dir_delta(ch):
    mapping = {'u': (0, 1), 'd': (0, -1), 'l': (-1, 0), 'r': (1, 0)}
    return mapping.get(ch, (0, 0))


def boundary_word_check(word):
    word = str(word)
    if len(word) == 0:
        return False, "空边界词。"
    dx = sum(_dir_delta(ch)[0] for ch in word)
    dy = sum(_dir_delta(ch)[1] for ch in word)
    if dx != 0 or dy != 0:
        return False, "边界词未闭合。"

    opp = {'u': 'd', 'd': 'u', 'l': 'r', 'r': 'l'}
    for i in range(len(word)):
        if word[i] == opp.get(word[(i + 1) % len(word)], ''):
            return False, "存在相邻回溯步。"
    return True, "OK"


def boundary_word_area(word):
    word = str(word)
    x, y = [0], [0]
    for ch in word:
        dx, dy = _dir_delta(ch)
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)

    A = 0.0
    for i in range(len(x) - 1):
        A += x[i] * y[i + 1] - x[i + 1] * y[i]
    return abs(A) * 0.5


def boundary_word_perimeter(word):
    return len(str(word))


def boundary_word_centroid(word):
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

        I0 += (x[i] ** 2 + x[i] * x[i + 1] + x[i + 1] ** 2 +
               y[i] ** 2 + y[i] * y[i + 1] + y[i + 1] ** 2) * abs(cross)
    A = abs(A) * 0.5
    if A < 1e-15:
        return 0.0
    I0 /= 12.0
    cx, cy = boundary_word_centroid(word)
    return I0 - A * (cx ** 2 + cy ** 2)






def safe_sqrt(x, eps=1e-15):
    return np.sqrt(np.maximum(x, eps))


def safe_divide(a, b, eps=1e-15):
    return np.where(np.abs(b) > eps, a / b, 0.0)


def fermi_dirac(energy, beta, mu=0.0):
    arg = beta * (energy - mu)

    arg = np.clip(arg, -700.0, 700.0)
    return 1.0 / (np.exp(arg) + 1.0)


def kron_delta(i, j):
    return 1.0 if i == j else 0.0
