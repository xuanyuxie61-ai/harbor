
import numpy as np



_1D_ABSCISSA = {
    1: np.array([0.0]),
    2: np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]),
    3: np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]),
    4: np.array([
        -np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
        -np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
        np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
        np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
    ]),
    5: np.array([
        -np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
        -np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
        0.0,
        np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
        np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
    ]),
}

_1D_WEIGHTS = {
    1: np.array([2.0]),
    2: np.array([1.0, 1.0]),
    3: np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]),
    4: np.array([
        (18.0 - np.sqrt(30.0)) / 36.0,
        (18.0 + np.sqrt(30.0)) / 36.0,
        (18.0 + np.sqrt(30.0)) / 36.0,
        (18.0 - np.sqrt(30.0)) / 36.0,
    ]),
    5: np.array([
        (322.0 - 13.0 * np.sqrt(70.0)) / 900.0,
        (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
        128.0 / 225.0,
        (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
        (322.0 - 13.0 * np.sqrt(70.0)) / 900.0,
    ]),
}


def line_rule(a, b, order):
    if order < 1 or order > 5:
        raise ValueError("Quadrature order must be in [1, 5].")
    x_ref = _1D_ABSCISSA[order].copy()
    w_ref = _1D_WEIGHTS[order].copy()


    jac = (b - a) / 2.0
    w = w_ref * jac
    x = ((1.0 - x_ref) * a + (1.0 + x_ref) * b) / 2.0
    return w, x


def line_monomial_integral(a, b, alpha):
    if alpha == -1:
        raise ValueError("Alpha = -1 is not integrable as a monomial.")
    return (b ** (alpha + 1) - a ** (alpha + 1)) / (alpha + 1)



_TRIANGLE_RULES = {
    1: {
        "w": np.array([0.5]),
        "x": np.array([1.0 / 3.0]),
        "y": np.array([1.0 / 3.0]),
    },
    3: {
        "w": np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]),
        "x": np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]),
        "y": np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]),
    },
    4: {
        "w": np.array([-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0]),
        "x": np.array([1.0 / 3.0, 1.0 / 5.0, 3.0 / 5.0, 1.0 / 5.0]),
        "y": np.array([1.0 / 3.0, 1.0 / 5.0, 1.0 / 5.0, 3.0 / 5.0]),
    },
    6: {
        "w": np.array([
            0.109951743655322, 0.109951743655322, 0.109951743655322,
            0.223381589678011, 0.223381589678011, 0.223381589678011,
        ]),
        "x": np.array([
            0.816847572980459, 0.091576213509771, 0.091576213509771,
            0.108103018168070, 0.445948490915965, 0.445948490915965,
        ]),
        "y": np.array([
            0.091576213509771, 0.816847572980459, 0.091576213509771,
            0.445948490915965, 0.108103018168070, 0.445948490915965,
        ]),
    },
}


def triangle_rule(order):
    if order not in _TRIANGLE_RULES:
        raise ValueError(f"Triangle quadrature order {order} not supported. Use 1, 3, 4, or 6.")
    data = _TRIANGLE_RULES[order]
    return data["w"].copy(), data["x"].copy(), data["y"].copy()


def map_triangle_quad(points, w_ref, xi_ref, eta_ref):
    p0, p1, p2 = points[0], points[1], points[2]
    J = np.array([
        [p1[0] - p0[0], p2[0] - p0[0]],
        [p1[1] - p0[1], p2[1] - p0[1]],
    ])
    detJ = abs(np.linalg.det(J))


    nq = len(w_ref)
    x_phys = np.zeros(nq)
    y_phys = np.zeros(nq)
    for q in range(nq):
        x_phys[q] = p0[0] + xi_ref[q] * (p1[0] - p0[0]) + eta_ref[q] * (p2[0] - p0[0])
        y_phys[q] = p0[1] + xi_ref[q] * (p1[1] - p0[1]) + eta_ref[q] * (p2[1] - p0[1])

    w_phys = w_ref * detJ
    return w_phys, x_phys, y_phys, detJ, J



_TET_RULES = {
    1: {
        "w": np.array([1.0 / 6.0]),
        "x": np.array([0.25]),
        "y": np.array([0.25]),
        "z": np.array([0.25]),
    },
    4: {

        "w": np.array([1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0]),
        "x": np.array([0.5, 0.5, 0.5, 0.0]),
        "y": np.array([0.5, 0.5, 0.0, 0.5]),
        "z": np.array([0.5, 0.0, 0.5, 0.5]),
    },
}


def tetrahedron_rule(order):
    if order not in _TET_RULES:
        raise ValueError(f"Tetrahedron quadrature order {order} not supported.")
    data = _TET_RULES[order]
    return data["w"].copy(), data["x"].copy(), data["y"].copy(), data["z"].copy()
