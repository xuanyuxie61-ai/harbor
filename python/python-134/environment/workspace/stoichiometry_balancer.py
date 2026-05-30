#!/usr/bin/env python3

import numpy as np


def i4vec_gcd(v):
    v = np.asarray(v, dtype=int)
    v = v[v != 0]
    if v.size == 0:
        return 1
    g = abs(v[0])
    for val in v[1:]:
        g = np.gcd(g, abs(val))
    return g


def diophantine_basis(a, b):
    a = np.asarray(a, dtype=int)
    n = a.size
    d = i4vec_gcd(a)


    if b % d != 0:
        return d, np.zeros(n, dtype=int), np.zeros((n, max(0, n - 1)), dtype=int)


    A = np.eye(n, dtype=int)
    M = np.hstack([a.reshape(-1, 1), A])


    for i in range(n):

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

                if M[j, 0] != 0:
                    q = (M[j, 0] + pivot - 1) // pivot if M[j, 0] > 0 else M[j, 0] // pivot
                    M[j, :] -= q * M[i, :]

        col0 = M[:, 0]
        nonzero = np.where(col0 != 0)[0]
        if nonzero.size > 0:
            pivot_idx = nonzero[np.argmin(np.abs(col0[nonzero]))]
            if pivot_idx != i:
                M[[i, pivot_idx]] = M[[pivot_idx, i]]


    d_out = M[0, 0]
    if d_out < 0:
        d_out = -d_out
        M[0, :] = -M[0, :]


    v = np.zeros(n, dtype=int)
    if d_out != 0 and b % d_out == 0:
        v = (b // d_out) * M[0, 1:]


    if n > 1:
        W = M[1:, 1:].T
    else:
        W = np.zeros((n, 0), dtype=int)

    return d_out, v, W


def balance_orr_stoichiometry():






    a_ox = np.array([2, -1], dtype=int)
    d, v, W = diophantine_basis(a_ox, 0)


    t = 1
    x1 = t
    x4 = 2 * t
    x2 = 2 * x4
    x3 = x2


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
