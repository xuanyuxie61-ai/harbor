
import numpy as np
from typing import Tuple


class PolyhedronGravityError(Exception):
    pass


def edge_factor(r1: np.ndarray, r2: np.ndarray) -> float:
    len_r1 = np.linalg.norm(r1)
    len_r2 = np.linalg.norm(r2)
    e_vec = r2 - r1
    e_len = np.linalg.norm(e_vec)
    if e_len < 1e-14:
        return 0.0
    numerator = len_r1 + len_r2 + e_len
    denominator = len_r1 + len_r2 - e_len
    if denominator <= 0.0:
        return 0.0
    return np.log(numerator / denominator)


def face_solid_angle(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> float:
    len_r1 = np.linalg.norm(r1)
    len_r2 = np.linalg.norm(r2)
    len_r3 = np.linalg.norm(r3)
    cross = np.cross(r2, r3)
    numerator = np.dot(r1, cross)
    D = len_r1 * len_r2 * len_r3 + len_r1 * np.dot(r2, r3) + len_r2 * np.dot(r3, r1) + len_r3 * np.dot(r1, r2)
    if abs(D) < 1e-14:
        return 0.0
    return 2.0 * np.arctan2(numerator, D)


def polyhedron_gravity_potential(
    pos: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    density: float = 2000.0,
    g_const: float = 6.67430e-11
) -> float:
    n_faces = faces.shape[0]
    potential = 0.0

    for fi in range(n_faces):
        v1 = vertices[faces[fi, 0]]
        v2 = vertices[faces[fi, 1]]
        v3 = vertices[faces[fi, 2]]

        r1 = v1 - pos
        r2 = v2 - pos
        r3 = v3 - pos


        n_vec = np.cross(v2 - v1, v3 - v1)
        area = 0.5 * np.linalg.norm(n_vec)
        if area < 1e-14:
            continue
        n_hat = n_vec / (2.0 * area)









        raise NotImplementedError("Hole 1: 请实现多面体引力势的核心面/边贡献公式")

    return potential


def polyhedron_gravity_acceleration(
    pos: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    density: float = 2000.0,
    g_const: float = 6.67430e-11,
    fd_step: float = 1.0
) -> np.ndarray:
    acc = np.zeros(3)
    for i in range(3):
        pos_p = pos.copy()
        pos_m = pos.copy()
        pos_p[i] += fd_step
        pos_m[i] -= fd_step
        u_p = polyhedron_gravity_potential(pos_p, vertices, faces, density, g_const)
        u_m = polyhedron_gravity_potential(pos_m, vertices, faces, density, g_const)
        acc[i] = -(u_p - u_m) / (2.0 * fd_step)
    return acc


def combined_gravity_model(
    pos: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    gm: float,
    r_ref: float,
    c_coeff: np.ndarray,
    s_coeff: np.ndarray,
    n_max: int = 8,
    density: float = 2000.0,
    g_const: float = 6.67430e-11,
    transition_radius: float = 3.0
) -> np.ndarray:
    r = np.linalg.norm(pos)
    from gravity_harmonics import SphericalHarmonicGravity

    sh_model = SphericalHarmonicGravity(gm, r_ref, c_coeff, s_coeff, n_max)
    a_harm = sh_model.acceleration(pos)





    raise NotImplementedError("Hole 2: 请实现组合模型中的单位转换与过渡权重")
