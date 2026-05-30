# -*- coding: utf-8 -*-

import numpy as np
import math
from itertools import combinations_with_replacement





_FEKETE_RULES = {}


def _register_fekete(degree, points, weights):
    _FEKETE_RULES[degree] = {
        'points': np.array(points, dtype=float),
        'weights': np.array(weights, dtype=float)
    }



_register_fekete(1, [[1.0 / 3.0, 1.0 / 3.0]], [0.5])


_register_fekete(2,
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])


_register_fekete(3,
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
     [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
    [1.0 / 30.0, 1.0 / 30.0, 1.0 / 30.0,
     1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0])


_register_fekete(4,
    [[1.0 / 3.0, 1.0 / 3.0],
     [0.059715871789770, 0.470142064105115],
     [0.470142064105115, 0.059715871789770],
     [0.470142064105115, 0.470142064105115],
     [0.797426985353087, 0.101286507323456],
     [0.101286507323456, 0.797426985353087],
     [0.101286507323456, 0.101286507323456]],
    [0.1125,
     0.066197076394253, 0.066197076394253, 0.066197076394253,
     0.062969590272413, 0.062969590272413, 0.062969590272413])


_register_fekete(5,
    [[0.501426509658179, 0.249286745170910],
     [0.249286745170910, 0.249286745170910],
     [0.249286745170910, 0.501426509658179],
     [0.873821971016996, 0.063089014491502],
     [0.063089014491502, 0.063089014491502],
     [0.063089014491502, 0.873821971016996],
     [0.053145049844817, 0.310352451033784],
     [0.310352451033784, 0.636502499121399],
     [0.636502499121399, 0.053145049844817],
     [0.310352451033784, 0.053145049844817],
     [0.636502499121399, 0.310352451033784],
     [0.053145049844817, 0.636502499121399]],
    [0.058393137863189, 0.058393137863189, 0.058393137863189,
     0.025422453185103, 0.025422453185103, 0.025422453185103,
     0.041425537809187, 0.041425537809187, 0.041425537809187,
     0.041425537809187, 0.041425537809187, 0.041425537809187])


_register_fekete(6,
    [[1.0 / 3.0, 1.0 / 3.0],
     [0.816847572980459, 0.091576213509771],
     [0.091576213509771, 0.091576213509771],
     [0.091576213509771, 0.816847572980459],
     [0.108103018168070, 0.445948490915965],
     [0.445948490915965, 0.445948490915965],
     [0.445948490915965, 0.108103018168070],
     [0.0, 0.5],
     [0.5, 0.0],
     [0.5, 0.5],
     [0.0, 0.25],
     [0.25, 0.0],
     [0.25, 0.75],
     [0.75, 0.0],
     [0.75, 0.25]],
    [0.072157803783908,
     0.047545817133642, 0.047545817133642, 0.047545817133642,
     0.051608685267359, 0.051608685267359, 0.051608685267359,
     0.016229248811599, 0.016229248811599, 0.016229248811599,
     0.013615157087217, 0.013615157087217, 0.013615157087217,
     0.013615157087217, 0.013615157087217])


_register_fekete(7,
    [[0.333333333333333, 0.333333333333333],
     [0.459292588292723, 0.270353705853638],
     [0.270353705853638, 0.270353705853638],
     [0.270353705853638, 0.459292588292723],
     [0.869739794195568, 0.065130102902216],
     [0.065130102902216, 0.065130102902216],
     [0.065130102902216, 0.869739794195568],
     [0.048690315425316, 0.312865496004874],
     [0.312865496004874, 0.638444188569810],
     [0.638444188569810, 0.048690315425316],
     [0.312865496004874, 0.048690315425316],
     [0.638444188569810, 0.312865496004874],
     [0.048690315425316, 0.638444188569810],
     [0.0, 0.5],
     [0.5, 0.0],
     [0.5, 0.5]],
    [0.072157803783908,
     0.047545817133642, 0.047545817133642, 0.047545817133642,
     0.016229248811599, 0.016229248811599, 0.016229248811599,
     0.013615157087217, 0.013615157087217, 0.013615157087217,
     0.013615157087217, 0.013615157087217, 0.013615157087217,
     0.016229248811599, 0.016229248811599, 0.016229248811599])


def fekete_triangle_quadrature(degree):
    if degree not in _FEKETE_RULES:
        available = sorted(_FEKETE_RULES.keys())
        raise ValueError(f"Fekete degree {degree} not available. Choose from {available}.")
    rule = _FEKETE_RULES[degree]
    return rule['points'].copy(), rule['weights'].copy()


def integrate_on_triangle(f, degree=5, vertices=None):
    if vertices is None:
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    xi, w = fekete_triangle_quadrature(degree)

    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    jac = np.array([[v1[0] - v0[0], v2[0] - v0[0]],
                    [v1[1] - v0[1], v2[1] - v0[1]]])
    detJ = abs(np.linalg.det(jac))
    x_phys = v0[np.newaxis, :] + xi @ jac.T
    vals = np.array([f(x_phys[i, 0], x_phys[i, 1]) for i in range(x_phys.shape[0])])
    return detJ * np.dot(w, vals)






def clenshaw_curtis_abscissas(n):
    if n == 1:
        return np.array([0.0])
    i = np.arange(n)
    return np.cos(i * np.pi / (n - 1))


def clenshaw_curtis_weights(n):
    if n == 1:
        return np.array([2.0])
    theta = np.arange(n) * np.pi / (n - 1)
    w = np.ones(n)
    for j in range(1, n // 2 + 1):
        b = 1.0 if (2 * j == n - 1) else 2.0
        w -= b * np.cos(2 * j * theta) / (4 * j * j - 1)

    w[0] = 1.0 / (n * n - 1 + (n % 2))
    w[-1] = w[0]

    w = 2.0 * w / np.sum(w)
    return w


def _level_to_order_closed(level):
    if level == 0:
        return 1
    return 2 ** level + 1


def sparse_grid_cc_size(dim_num, level_max):
    if dim_num < 1 or level_max < 0:
        return 0

    count = 0
    for l in range(level_max + 1):

        count += math.comb(l + dim_num - 1, dim_num - 1)


    return min(count * 2, 100000)


def sparse_grid_cc(dim_num, level_max):
    if dim_num < 1:
        raise ValueError("dim_num must be >= 1")
    if level_max < 0:
        raise ValueError("level_max must be >= 0")


    max_order = _level_to_order_closed(level_max)
    rules = {}
    for lvl in range(level_max + 1):
        order = _level_to_order_closed(lvl)
        rules[lvl] = {
            'x': clenshaw_curtis_abscissas(order),
            'w': clenshaw_curtis_weights(order)
        }





    from itertools import product

    point_dict = {}
    L = level_max


    def multi_indices(dim, max_sum):
        if dim == 1:
            for s in range(max_sum + 1):
                yield (s,)
        else:
            for s in range(max_sum + 1):
                for tail in multi_indices(dim - 1, max_sum - s):
                    yield (s,) + tail

    for l_vec in multi_indices(dim_num, L):
        if sum(l_vec) < L - dim_num + 1:
            continue

        s = sum(l_vec)
        if s > L:
            continue
        k = L - s
        coeff = ((-1) ** k) * math.comb(dim_num - 1, k)


        dim_rules = []
        for d, ld in enumerate(l_vec):
            order = _level_to_order_closed(ld)
            x = rules[ld]['x']
            w = rules[ld]['w']
            dim_rules.append((x, w))

        for idx in product(*[range(len(r[0])) for r in dim_rules]):
            pt = tuple(dim_rules[d][0][idx[d]] for d in range(dim_num))
            wt = coeff * np.prod([dim_rules[d][1][idx[d]] for d in range(dim_num)])
            point_dict[pt] = point_dict.get(pt, 0.0) + wt

    points = np.array([p for p in point_dict.keys()], dtype=float)
    weights = np.array([point_dict[p] for p in point_dict.keys()], dtype=float)


    mask = np.abs(weights) > 1e-14
    points = points[mask]
    weights = weights[mask]

    return points, weights


def sparse_grid_integrate(f, dim_num, level_max):
    pts, wts = sparse_grid_cc(dim_num, level_max)
    if pts.size == 0:
        return 0.0
    vals = np.array([f(pts[i, :]) for i in range(pts.shape[0])])
    return np.dot(wts, vals)


def integrate_deformation_pdf(beta2_min, beta2_max, beta3_min, beta3_max,
                              pdf_func, degree=5):


    v1 = np.array([[beta2_min, beta3_min],
                   [beta2_max, beta3_min],
                   [beta2_max, beta3_max]])
    v2 = np.array([[beta2_min, beta3_min],
                   [beta2_max, beta3_max],
                   [beta2_min, beta3_max]])

    def f1(x, y):
        return pdf_func(x, y)

    def f2(x, y):
        return pdf_func(x, y)

    return integrate_on_triangle(f1, degree, v1) + integrate_on_triangle(f2, degree, v2)
