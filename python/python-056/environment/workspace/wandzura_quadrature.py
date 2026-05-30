
import numpy as np
from typing import Tuple


def i4_wrap(i: int, i1: int, i2: int) -> int:
    n = i2 - i1 + 1
    j = i1 + ((i - i1) % n)
    return j


def wandzura_suborder_num(rule: int) -> int:
    suborder_nums = {1: 2, 2: 3, 3: 4}
    return suborder_nums.get(rule, 2)


def wandzura_suborder(rule: int, suborder_num: int) -> np.ndarray:
    if rule == 1:
        return np.array([1, 3])
    elif rule == 2:
        return np.array([1, 3, 6])
    else:
        return np.array([1, 3, 6, 6])


def wandzura_subrule(rule: int, suborder_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rule == 1:

        xyz = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.059715871789770, 0.797426985353087, 0.142857142857143],
        ])
        w = np.array([
            0.225000000000000,
            0.132394152788506,
        ])
        types = np.array([1, 3])
    elif rule == 2:

        xyz = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.028844733232685, 0.942263611675977, 0.028891655091338],
            [0.143228964696565, 0.713522865095785, 0.143248170207650],
            [0.322343506604364, 0.355324285987699, 0.322332207407937],
        ])
        w = np.array([
            0.090817990382754,
            0.036725957098437,
            0.045321059435528,
            0.072757916845516,
        ])
        types = np.array([1, 3, 3, 6])
    else:
        xyz = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.059715871789770, 0.797426985353087, 0.142857142857143],
        ])
        w = np.array([
            0.225000000000000,
            0.132394152788506,
        ])
        types = np.array([1, 3])
    return xyz, w, types


def wandzura_rule(rule: int, order_num: int) -> Tuple[np.ndarray, np.ndarray]:
    suborder_xyz, suborder_w, suborder_types = wandzura_subrule(rule, 2)

    xy = np.zeros((2, order_num))
    w = np.zeros(order_num)
    o = 0

    for s in range(len(suborder_types)):
        stype = suborder_types[s]
        if stype == 1:
            xy[:, o] = suborder_xyz[s, :2]
            w[o] = suborder_w[s]
            o += 1
        elif stype == 3:

            for k in range(3):
                xy[0, o] = suborder_xyz[s, k]
                xy[1, o] = suborder_xyz[s, (k + 1) % 3]
                w[o] = suborder_w[s]
                o += 1
        elif stype == 6:
            for k in range(3):
                xy[0, o] = suborder_xyz[s, k]
                xy[1, o] = suborder_xyz[s, (k + 1) % 3]
                w[o] = suborder_w[s]
                o += 1
            for k in range(3):
                xy[0, o] = suborder_xyz[s, (k + 1) % 3]
                xy[1, o] = suborder_xyz[s, k]
                w[o] = suborder_w[s]
                o += 1
        else:
            raise ValueError(f"wandzura_rule: 非法子阶数类型 {stype}")

    return xy[:, :o], w[:o]


def integrate_triangle(
    f: callable,
    vertices: np.ndarray,
    rule: int = 1,
) -> float:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("integrate_triangle: vertices 必须为 (3,2) 数组")

    order_nums = {1: 6, 2: 25}
    order_num = order_nums.get(rule, 6)
    xy_ref, w = wandzura_rule(rule, order_num)
    nq = w.size


    v0 = vertices[0, :]
    J = np.array([
        [vertices[1, 0] - vertices[0, 0], vertices[2, 0] - vertices[0, 0]],
        [vertices[1, 1] - vertices[0, 1], vertices[2, 1] - vertices[0, 1]],
    ])
    detJ = abs(np.linalg.det(J))

    total = 0.0
    for i in range(nq):
        xi = xy_ref[0, i]
        eta = xy_ref[1, i]

        x_phys = v0[0] + J[0, 0] * xi + J[0, 1] * eta
        y_phys = v0[1] + J[1, 0] * xi + J[1, 1] * eta
        total += w[i] * f(x_phys, y_phys)

    return detJ * total


def compute_hydrofoil_lift(
    chord: float = 2.0,
    span: float = 10.0,
    angle_of_attack: float = 8.0,
    velocity: float = 2.5,
    rho: float = 1025.0,
) -> float:
    alpha_rad = np.deg2rad(angle_of_attack)

    cl = 2.0 * np.pi * alpha_rad
    dp = 0.5 * rho * velocity ** 2 * cl
    area = chord * span
    return dp * area
