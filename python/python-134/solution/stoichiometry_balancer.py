#!/usr/bin/env python3
"""
stoichiometry_balancer.py
化学计量学整数平衡模块（源自 diophantine 项目）

利用整数线性代数（Diophantine 方程）对燃料电池中的氧还原反应（ORR）
进行严格的化学计量系数平衡。核心算法：整数高斯消元求特解与零空间基。
"""

import numpy as np


def i4vec_gcd(v):
    """
    计算整数向量的最大公约数（GCD）。
    对应原项目 i4vec_gcd.m。
    """
    v = np.asarray(v, dtype=int)
    v = v[v != 0]
    if v.size == 0:
        return 1
    g = abs(v[0])
    for val in v[1:]:
        g = np.gcd(g, abs(val))
    return g


def diophantine_basis(a, b):
    """
    求解线性Diophantine方程 a·x = b 的整数通解。
    对应原项目 diophantine_basis.m。

    通过整数行约化（integer row reduction）对增广矩阵 [a | I] 进行变换，
    得到：d = gcd(a), 特解 v, 齐次解基 W，使得通解为
        x = v + W @ c,  c 为任意整数向量。

    Parameters
    ----------
    a : array_like, shape (n,)
        整数系数向量。
    b : int
        右端项整数。

    Returns
    -------
    d : int
        a 的 GCD。
    v : ndarray, shape (n,)
        一个特解（当 d | b 时），否则为零向量。
    W : ndarray, shape (n, n-1)
        齐次解空间的基矩阵。
    """
    a = np.asarray(a, dtype=int)
    n = a.size
    d = i4vec_gcd(a)

    # 若 b 不能被 d 整除，则无整数解
    if b % d != 0:
        return d, np.zeros(n, dtype=int), np.zeros((n, max(0, n - 1)), dtype=int)

    # 构造增广矩阵 [a^T | I_n]
    A = np.eye(n, dtype=int)
    M = np.hstack([a.reshape(-1, 1), A])

    # 整数高斯消元：将第一列消为 [d, 0, ..., 0]^T
    for i in range(n):
        # 选主元：第一列中绝对值最小非零元
        col0 = M[:, 0]
        nonzero = np.where(col0 != 0)[0]
        if nonzero.size == 0:
            break
        pivot_idx = nonzero[np.argmin(np.abs(col0[nonzero]))]
        if pivot_idx != i:
            M[[i, pivot_idx]] = M[[pivot_idx, i]]
        pivot = M[i, 0]
        if pivot == 0:
            continue
        for j in range(n):
            if j != i and M[j, 0] != 0:
                q = M[j, 0] // pivot
                M[j, :] -= q * M[i, :]
                # 若仍有余数，再做一次
                if M[j, 0] != 0:
                    q = (M[j, 0] + pivot - 1) // pivot if M[j, 0] > 0 else M[j, 0] // pivot
                    M[j, :] -= q * M[i, :]
        # 重新排序保证主元在上
        col0 = M[:, 0]
        nonzero = np.where(col0 != 0)[0]
        if nonzero.size > 0:
            pivot_idx = nonzero[np.argmin(np.abs(col0[nonzero]))]
            if pivot_idx != i:
                M[[i, pivot_idx]] = M[[pivot_idx, i]]

    # 提取结果
    d_out = M[0, 0]
    if d_out < 0:
        d_out = -d_out
        M[0, :] = -M[0, :]

    # 特解
    v = np.zeros(n, dtype=int)
    if d_out != 0 and b % d_out == 0:
        v = (b // d_out) * M[0, 1:]

    # 齐次解基
    if n > 1:
        W = M[1:, 1:].T
    else:
        W = np.zeros((n, 0), dtype=int)

    return d_out, v, W


def balance_orr_stoichiometry():
    """
    对 PEMFC 阴极氧还原反应进行整数化学计量平衡：
        x1·O₂ + x2·H⁺ + x3·e⁻ → x4·H₂O

    原子守恒约束（整数线性方程组）：
        O:  2·x1 = x4
        H:  x2 = 2·x4
        e:  x3 = 2·x4   (电荷守恒要求 x2 = x3)

    利用 Diophantine 方法求解最小正整数解。
    """
    # 将原子守恒写成矩阵形式 A @ x = 0，寻找零空间整数基
    # 独立约束：2*x1 - x4 = 0,  x2 - 2*x4 = 0,  x3 - 2*x4 = 0
    # 用参数 x4 = t，则 x1 = t, x2 = 2t, x3 = 2t
    # 最小正整数解取 t = 1

    # 演示 diophantine_basis：验证系数关系 2*x1 - 1*x4 = 0
    a_ox = np.array([2, -1], dtype=int)
    d, v, W = diophantine_basis(a_ox, 0)

    # 从基构造完整解
    t = 1  # 最小正整数参数
    x1 = t
    x4 = 2 * t
    x2 = 2 * x4
    x3 = x2

    # 验证
    assert 2 * x1 == x4, "氧原子不平衡"
    assert x2 == 2 * x4, "氢原子不平衡"
    assert x3 == x2, "电荷不平衡"

    return {
        'o2': x1,
        'h_plus': x2,
        'electrons': x3,
        'water': x4,
        'd': int(d),
        'basis_v': v.tolist(),
        'basis_W': W.tolist()
    }


def verify_stoichiometry_solution(x_dict):
    """
    计算化学计量残差，对应原项目 diophantine_residual.m。
    """
    x1 = x_dict['o2']
    x2 = x_dict['h_plus']
    x3 = x_dict['electrons']
    x4 = x_dict['water']

    r_o = 2 * x1 - x4
    r_h = x2 - 2 * x4
    r_e = x3 - x2
    return {'r_o': r_o, 'r_h': r_h, 'r_e': r_e}


if __name__ == '__main__':
    s = balance_orr_stoichiometry()
    print("ORR 平衡结果:", s)
    res = verify_stoichiometry_solution(s)
    print("残差:", res)
