
import numpy as np
from utils import NumericalConfig






def line_unit_o01():
    x = np.array([0.0], dtype=float)
    w = np.array([2.0], dtype=float)
    return x, w


def line_unit_o02():
    x = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    w = np.array([1.0, 1.0], dtype=float)
    return x, w


def line_unit_o03():
    x = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)], dtype=float)
    w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
    return x, w


def line_unit_o04():
    a = np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0))
    b = np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0))
    wa = (18.0 + np.sqrt(30.0)) / 36.0
    wb = (18.0 - np.sqrt(30.0)) / 36.0
    x = np.array([-b, -a, a, b], dtype=float)
    w = np.array([wb, wa, wa, wb], dtype=float)
    return x, w


def line_unit_o05():
    a = 1.0 / 3.0 * np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0))
    b = 1.0 / 3.0 * np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0))
    wa = (322.0 + 13.0 * np.sqrt(70.0)) / 900.0
    wb = (322.0 - 13.0 * np.sqrt(70.0)) / 900.0
    wc = 128.0 / 225.0
    x = np.array([-b, -a, 0.0, a, b], dtype=float)
    w = np.array([wb, wa, wc, wa, wb], dtype=float)
    return x, w


def line_rule(a, b, order):
    if order == 1:
        x0, w0 = line_unit_o01()
    elif order == 2:
        x0, w0 = line_unit_o02()
    elif order == 3:
        x0, w0 = line_unit_o03()
    elif order == 4:
        x0, w0 = line_unit_o04()
    elif order == 5:
        x0, w0 = line_unit_o05()
    else:
        raise ValueError("order must be between 1 and 5")


    scale = 0.5 * (b - a)
    shift = 0.5 * (a + b)
    x = scale * x0 + shift
    w = scale * w0
    return x, w


def line_monomial_integral(a, b, alpha):
    if alpha < 0:

        if a <= NumericalConfig.EPS:
            a = NumericalConfig.EPS
    return (b ** (alpha + 1.0) - a ** (alpha + 1.0)) / (alpha + 1.0)






def cube_arbq_size(degree):
    size_table = {
        1: 1, 2: 4, 3: 6, 4: 10, 5: 13, 6: 22, 7: 26,
        8: 42, 9: 50, 10: 73, 11: 84, 12: 116, 13: 130, 14: 172, 15: 190
    }
    if degree not in size_table:
        raise ValueError(f"Degree {degree} not supported. Use 1-15.")
    return size_table[degree]


def _cube_arbq_rule_low_degree(degree):
    if degree <= 1:
        n1d = 1
    elif degree <= 3:
        n1d = 2
    elif degree <= 5:
        n1d = 3
    else:
        raise ValueError("Use tensor product for degree > 5 in this simplified implementation")

    x1d, w1d = line_unit_o03() if n1d == 3 else (line_unit_o02() if n1d == 2 else line_unit_o01())

    nodes = []
    weights = []
    for i in range(n1d):
        for j in range(n1d):
            for k in range(n1d):
                nodes.append([x1d[i], x1d[j], x1d[k]])
                weights.append(w1d[i] * w1d[j] * w1d[k])

    return np.array(nodes, dtype=float), np.array(weights, dtype=float)


def cube_arbq(degree):
    if degree < 1 or degree > 5:

        degree = 5

    x, w = _cube_arbq_rule_low_degree(degree)

    vol = np.sum(w)
    if abs(vol - 8.0) > NumericalConfig.TOL:
        w = w * (8.0 / vol)
    return x, w






def pyramid_unit_volume():
    return 4.0 / 3.0


def pyramid_unit_o01():
    x = np.array([[0.0, 0.0, 0.75]], dtype=float)
    w = np.array([4.0 / 3.0], dtype=float)
    return x, w


def pyramid_unit_o05():

    a = 0.5
    b = 0.25
    c = 1.0
    nodes = np.array([
        [0.0, 0.0, b],
        [a, a, b],
        [-a, a, b],
        [a, -a, b],
        [-a, -a, b]
    ], dtype=float)

    w = np.array([4.0 / 3.0 / 5.0] * 5, dtype=float)
    return nodes, w


def pyramid_unit_o08():

    a = np.sqrt(3.0) / 3.0
    h1 = 0.2
    h2 = 0.8
    nodes = np.array([
        [a, a, h1], [-a, a, h1], [a, -a, h1], [-a, -a, h1],
        [a, a, h2], [-a, a, h2], [a, -a, h2], [-a, -a, h2]
    ], dtype=float)
    w = np.array([4.0 / 3.0 / 8.0] * 8, dtype=float)
    return nodes, w


def pyramid_unit_monomial(expon):
    a, b_exp, c = expon
    if a < 0 or b_exp < 0 or c < 0:
        raise ValueError("Exponents must be non-negative")

    from math import gamma
    beta_part = gamma(a + b_exp + 3.0) * gamma(c + 1.0) / gamma(a + b_exp + c + 4.0)
    return 4.0 * beta_part / ((a + 1.0) * (b_exp + 1.0))






def integrate_cube_domain(func, degree=5, scale=1.0, shift=None):
    if shift is None:
        shift = np.zeros(3, dtype=float)

    x, w = cube_arbq(degree)

    x_scaled = x * scale + shift
    fx = func(x_scaled)



    return np.sum(w * fx) * (scale ** 3)


def integrate_pyramid_domain(func, degree=5):
    if degree <= 1:
        x, w = pyramid_unit_o01()
    elif degree <= 3:
        x, w = pyramid_unit_o05()
    else:
        x, w = pyramid_unit_o08()

    fx = func(x)

    vol = np.sum(w)
    target_vol = pyramid_unit_volume()
    if abs(vol - target_vol) > NumericalConfig.TOL:
        w = w * (target_vol / vol)

    return np.sum(w * fx)


def integrate_line_profile(func, a, b, order=5):
    x, w = line_rule(a, b, order)
    fx = func(x)
    return np.sum(w * fx)


def estimate_total_biomass_cube(density_func, domain_bounds, degree=5):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain_bounds


    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    sx = 0.5 * (xmax - xmin)
    sy = 0.5 * (ymax - ymin)
    sz = 0.5 * (zmax - zmin)

    def mapped_func(pts):


        real_pts = np.zeros_like(pts)
        real_pts[:, 0] = pts[:, 0] * sx + cx
        real_pts[:, 1] = pts[:, 1] * sy + cy
        real_pts[:, 2] = pts[:, 2] * sz + cz
        return density_func(real_pts)

    x, w = cube_arbq(degree)
    fx = mapped_func(x)

    jacobian = sx * sy * sz
    return np.sum(w * fx) * jacobian
