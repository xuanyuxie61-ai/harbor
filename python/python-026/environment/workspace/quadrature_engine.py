# -*- coding: utf-8 -*-

import numpy as np
from itertools import product



_LEGENDRE_TABLE = {
    1: {
        'x': np.array([0.0]),
        'w': np.array([2.0])
    },
    2: {
        'x': np.array([-0.5773502691896258, 0.5773502691896258]),
        'w': np.array([1.0, 1.0])
    },
    3: {
        'x': np.array([-0.7745966692414834, 0.0, 0.7745966692414834]),
        'w': np.array([0.5555555555555556, 0.8888888888888889, 0.5555555555555556])
    },
    4: {
        'x': np.array([-0.8611363115940526, -0.3399810435848563,
                       0.3399810435848563, 0.8611363115940526]),
        'w': np.array([0.3478548451374539, 0.6521451548625461,
                       0.6521451548625461, 0.3478548451374539])
    },
    5: {
        'x': np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                       0.5384693101056831, 0.9061798459386640]),
        'w': np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                       0.4786286704993665, 0.2369268850561891])
    },
    6: {
        'x': np.array([-0.9324695142031520, -0.6612093864662645, -0.2386191860831969,
                       0.2386191860831969, 0.6612093864662645, 0.9324695142031520]),
        'w': np.array([0.1713244923791703, 0.3607615730481386, 0.4679139345726910,
                       0.4679139345726910, 0.3607615730481386, 0.1713244923791703])
    },
    7: {
        'x': np.array([-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0,
                       0.4058451513773972, 0.7415311855993945, 0.9491079123427585]),
        'w': np.array([0.1294849661688697, 0.2797053914892767, 0.3818300505051189, 0.4179591836734694,
                       0.3818300505051189, 0.2797053914892767, 0.1294849661688697])
    },
    8: {
        'x': np.array([-0.9602898564975362, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                       0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975362]),
        'w': np.array([0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
                       0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763])
    },
    9: {
        'x': np.array([-0.9681602395076261, -0.8360311073266358, -0.6133714327005904,
                       -0.3242534234038089, 0.0, 0.3242534234038089,
                       0.6133714327005904, 0.8360311073266358, 0.9681602395076261]),
        'w': np.array([0.0812743883615744, 0.1806481606948574, 0.2606106964029355,
                       0.3123470770400028, 0.3302393550012598, 0.3123470770400028,
                       0.2606106964029355, 0.1806481606948574, 0.0812743883615744])
    },
    10: {
        'x': np.array([-0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
                       -0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
                       0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717]),
        'w': np.array([0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
                       0.2692667193099964, 0.2955242247147529, 0.2955242247147529,
                       0.2692667193099964, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881])
    }
}


def gauss_legendre_rule(n):
    if n not in _LEGENDRE_TABLE:
        raise ValueError(f"Gauss-Legendre 规则仅支持 n=1..10，收到 n={n}。")
    return _LEGENDRE_TABLE[n]['x'].copy(), _LEGENDRE_TABLE[n]['w'].copy()


def integrate_1d_gauss_legendre(f, a, b, n=8):
    if a >= b:
        return 0.0
    t, w = gauss_legendre_rule(n)
    x = 0.5 * (b - a) * t + 0.5 * (b + a)
    fx = f(x)
    fx = np.asarray(fx, dtype=float)
    result = 0.5 * (b - a) * np.sum(w * fx)
    return float(result)


def integrate_along_ray_gauss(f_s, s_nodes, n_quad=8):
    s_nodes = np.asarray(s_nodes, dtype=float)
    if len(s_nodes) < 2:
        return 0.0
    if np.any(np.diff(s_nodes) <= 0):
        raise ValueError("s_nodes 必须严格递增。")

    if callable(f_s):
        t, w = gauss_legendre_rule(n_quad)
        integral = 0.0
        for i in range(len(s_nodes) - 1):
            a = s_nodes[i]
            b = s_nodes[i + 1]
            if b <= a:
                continue
            s_quad = 0.5 * (b - a) * t + 0.5 * (b + a)
            fs = f_s(s_quad)
            integral += 0.5 * (b - a) * np.sum(w * fs)
    else:
        f_vals = np.asarray(f_s, dtype=float)
        if len(f_vals) != len(s_nodes):
            raise ValueError("f_s 作为数组时长度必须与 s_nodes 一致。")
        integral = np.trapezoid(f_vals, s_nodes)

    return float(integral)



_HERMITE_TABLE = {
    1: {
        'x': np.array([0.0]),
        'w': np.array([np.sqrt(np.pi)])
    },
    2: {
        'x': np.array([-0.7071067811865475, 0.7071067811865475]),
        'w': np.array([0.8862269254527580, 0.8862269254527580])
    },
    3: {
        'x': np.array([-1.224744871391589, 0.0, 1.224744871391589]),
        'w': np.array([0.2954089751509193, 1.181635900603677, 0.2954089751509193])
    },
    4: {
        'x': np.array([-1.650680123885785, -0.5246476232752900,
                       0.5246476232752900, 1.650680123885785]),
        'w': np.array([0.0813128354472452, 0.8049140900055135,
                       0.8049140900055135, 0.0813128354472452])
    },
    5: {
        'x': np.array([-2.020182870456086, -0.9585724646138185, 0.0,
                       0.9585724646138185, 2.020182870456086]),
        'w': np.array([0.0199532420590459, 0.3936193231522402, 0.9453087204829419,
                       0.3936193231522402, 0.0199532420590459])
    },
    6: {
        'x': np.array([-2.350604973674492, -1.335849074013697, -0.4360774119276165,
                       0.4360774119276165, 1.335849074013697, 2.350604973674492]),
        'w': np.array([0.0045300099055088, 0.1570673203228566, 0.7246295952243924,
                       0.7246295952243924, 0.1570673203228566, 0.0045300099055088])
    },
    7: {
        'x': np.array([-2.651961356835233, -1.673551628767471, -0.8162878828589647, 0.0,
                       0.8162878828589647, 1.673551628767471, 2.651961356835233]),
        'w': np.array([0.0009717812450995, 0.0545155828191270, 0.4256072526101278,
                       0.8102646175568073, 0.4256072526101278, 0.0545155828191270, 0.0009717812450995])
    }
}


def hermite_rule(order):
    if order not in _HERMITE_TABLE:
        raise ValueError(f"Gauss-Hermite 规则仅支持 order=1..7，收到 order={order}。")
    return _HERMITE_TABLE[order]['x'].copy(), _HERMITE_TABLE[order]['w'].copy()


def level_to_order_open(level):
    if level < 0:
        return 1
    order = 2 ** (level + 1) - 1
    max_avail = max(_HERMITE_TABLE.keys())
    if order > max_avail:
        order = max_avail
    return order


def comp_next(n, k, a, more, h, t):
    a = np.asarray(a, dtype=int)
    if not more:
        a[:] = 0
        a[0] = n
        more = True
        h = 0
        t = n if k > 1 else 0
    else:
        if 1 < t:
            h = 0
        h = h + 1
        t = a[h - 1]
        a[h - 1] = 0
        a[0] = t - 1
        a[h] = a[h] + 1
        if t - 1 != 0:
            h = 1
            t = a[0]
        more = (a[k - 1] != n)
    return a, more, h, t


def sparse_grid_hermite_size(dim_num, level_max):
    if level_max == 0:
        return 1
    level_min = max(0, level_max + 1 - dim_num)
    point_num = 0
    for level in range(level_min, level_max + 1):
        level_1d = np.zeros(dim_num, dtype=int)
        more = False
        h = 0
        t = 0
        while True:
            level_1d, more, h, t = comp_next(level, dim_num, level_1d, more, h, t)
            order_1d = np.array([level_to_order_open(l) for l in level_1d])
            for dim in range(dim_num):
                if level_min < level and 1 < order_1d[dim]:
                    order_1d[dim] -= 1
            point_num += int(np.prod(order_1d))
            if not more:
                break
    return point_num


def sparse_grid_hermite_index(dim_num, level_max):
    if level_max == 0:
        return [(tuple([0] * dim_num),)], [1.0]

    level_min = max(0, level_max + 1 - dim_num)
    points_data = []

    for level in range(level_min, level_max + 1):
        level_1d = np.zeros(dim_num, dtype=int)
        more = False
        h = 0
        t = 0
        while True:
            level_1d, more, h, t = comp_next(level, dim_num, level_1d, more, h, t)
            order_1d = np.array([level_to_order_open(l) for l in level_1d])
            for dim in range(dim_num):
                if level_min < level and 1 < order_1d[dim]:
                    order_1d[dim] -= 1


            coeff = ((-1) ** (level_max - level)) * \
                    comb(dim_num - 1, level_max - level)


            ranges = [range(o) for o in order_1d]
            for idx_tuple in product(*ranges):
                points_data.append((idx_tuple, order_1d, coeff))

            if not more:
                break


    unique = {}
    for idx_tuple, order_1d, coeff in points_data:
        key = tuple(int(i) for i in idx_tuple)
        if key not in unique:
            unique[key] = {'orders': order_1d, 'weight': 0.0}


        unique[key]['weight'] += float(coeff)

    indices = []
    weights = []
    for key, val in unique.items():
        indices.append(key)
        weights.append(val['weight'])

    return indices, weights


def comb(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result


def integrate_nd_sparse_hermite(f, dim_num, level_max):
    indices, comb_weights = sparse_grid_hermite_index(dim_num, level_max)
    result = 0.0
    for idx_tuple, cw in zip(indices, comb_weights):
        point = np.zeros(dim_num, dtype=float)
        w_prod = 1.0
        for d in range(dim_num):
            order = level_to_order_open(0)

            if idx_tuple[d] >= 1:

                for lev in range(1, level_max + 1):
                    ord_lev = level_to_order_open(lev)
                    if idx_tuple[d] < ord_lev:
                        order = ord_lev
                        break
                else:
                    order = level_to_order_open(level_max)
            else:
                order = level_to_order_open(0)
            x_1d, w_1d = hermite_rule(order)
            point[d] = x_1d[idx_tuple[d]]
            w_prod *= w_1d[idx_tuple[d]]
        result += cw * w_prod * f(point)
    return float(result)


def integrate_energy_deposition_along_ray(s_vals, intensity_vals, ne_vals, Te_val, omega0, Z=1):
    from physics_constants import plasma_frequency, electron_ion_collision_frequency, C_LIGHT

    s_vals = np.asarray(s_vals, dtype=float)
    intensity_vals = np.asarray(intensity_vals, dtype=float)
    ne_vals = np.asarray(ne_vals, dtype=float)

    if len(s_vals) < 2 or len(intensity_vals) != len(s_vals) or len(ne_vals) != len(s_vals):
        raise ValueError("输入数组长度不一致或不足。")

    kappa_vals = np.zeros_like(ne_vals)
    for i, ne in enumerate(ne_vals):
        if ne <= 0:
            kappa_vals[i] = 0.0
            continue
        nu_ei = electron_ion_collision_frequency(ne, Te_val, Z)
        omega_p = plasma_frequency(ne)
        ratio = (omega_p / omega0) ** 2
        ratio = np.clip(ratio, 0.0, 1.0)
        eta = np.sqrt(1.0 - ratio)
        eta_safe = max(eta, 1e-6)
        kappa_vals[i] = (nu_ei / C_LIGHT) * ratio * (1.0 / eta_safe)
        if not np.isfinite(kappa_vals[i]):
            kappa_vals[i] = 0.0


    integrand = kappa_vals * intensity_vals
    energy_dep = np.trapezoid(integrand, s_vals)
    return float(energy_dep)
