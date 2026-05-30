
import numpy as np
from typing import Tuple
from utils import compute_triangle_area


def triangle_quad_rule_degree1() -> Tuple[np.ndarray, np.ndarray]:
    weights = np.array([0.5])
    points = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    return weights, points


def triangle_quad_rule_degree2() -> Tuple[np.ndarray, np.ndarray]:
    w = 1.0 / 6.0
    weights = np.array([w, w, w])
    points = np.array([
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5]
    ])
    return weights, points


def triangle_quad_rule_degree3() -> Tuple[np.ndarray, np.ndarray]:
    w1 = -9.0 / 32.0
    w2 = 25.0 / 96.0
    weights = np.array([w1, w2, w2, w2])
    points = np.array([
        [1.0 / 3.0, 1.0 / 3.0],
        [3.0 / 5.0, 1.0 / 5.0],
        [1.0 / 5.0, 3.0 / 5.0],
        [1.0 / 5.0, 1.0 / 5.0]
    ])
    return weights, points


def triangle_quad_rule_degree5() -> Tuple[np.ndarray, np.ndarray]:

    a1 = (6.0 + np.sqrt(15.0)) / 21.0
    a2 = (6.0 - np.sqrt(15.0)) / 21.0
    b1 = (4.0 + np.sqrt(15.0)) / 7.0
    b2 = (4.0 - np.sqrt(15.0)) / 7.0

    w1 = (155.0 - np.sqrt(15.0)) / 1200.0
    w2 = (155.0 + np.sqrt(15.0)) / 1200.0
    w3 = 9.0 / 40.0

    weights = np.array([
        w1, w1, w1,
        w2, w2, w2,
        w3
    ])

    points = np.array([
        [a1, a1],
        [b1, a1],
        [a1, b1],
        [a2, a2],
        [b2, a2],
        [a2, b2],
        [1.0 / 3.0, 1.0 / 3.0]
    ])
    return weights, points


def integrate_over_triangle(vertices: np.ndarray,
                            f: callable,
                            degree: int = 3) -> float:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2)")

    area = abs(compute_triangle_area(vertices[0], vertices[1], vertices[2]))
    if area < 1e-14:
        return 0.0

    if degree <= 1:
        weights, points_ref = triangle_quad_rule_degree1()
    elif degree == 2:
        weights, points_ref = triangle_quad_rule_degree2()
    elif degree == 3:
        weights, points_ref = triangle_quad_rule_degree3()
    else:
        weights, points_ref = triangle_quad_rule_degree5()




    x1, y1 = vertices[0]
    dx2, dy2 = vertices[1] - vertices[0]
    dx3, dy3 = vertices[2] - vertices[0]

    total = 0.0
    for i in range(len(weights)):
        r, s = points_ref[i]
        x = x1 + dx2 * r + dx3 * s
        y = y1 + dy2 * r + dy3 * s
        total += weights[i] * float(f(x, y))


    return 2.0 * area * total


def integrate_over_mesh(nodes: np.ndarray, triangles: np.ndarray,
                        f: callable, degree: int = 3) -> float:
    total = 0.0
    for e in range(triangles.shape[0]):
        verts = nodes[triangles[e] - 1]
        total += integrate_over_triangle(verts, f, degree)
    return total


def compute_moment_over_mesh(nodes: np.ndarray, triangles: np.ndarray,
                              order_x: int = 0, order_y: int = 0,
                              degree: int = 3) -> float:
    def f(x, y):
        return (x ** order_x) * (y ** order_y)
    return integrate_over_mesh(nodes, triangles, f, degree)
