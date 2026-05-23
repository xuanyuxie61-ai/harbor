"""
gravity_polyhedron.py

基于 triangulate (多边形剖分) 与 polynomial_conversion (正交多项式展开)
的扩展应用，实现小行星多面体引力场模型。

科学背景：
对于近小行星轨道（距离与天体尺寸相当），球谐展开收敛缓慢甚至发散。
此时需采用多面体方法 (Werner & Scheeres, 1997)：

    U(r) = (1/2) Gρ Σ_{e∈edges} r_e · E_e · r_e · L_e
           − (1/2) Gρ Σ_{f∈faces} r_f · F_f · r_f · ω_f

其中：
- E_e: 边张量积 (r1 r2^T + r2 r1^T) / |r1 × r2|²
- F_f: 面张量积 (n_f n_f^T) / |n_f|²
- L_e: 边对数项 log( (r1 + r2 + e) / (r1 + r2 − e) )
- ω_f: 面立体角 2 arctan( (r1 · (r2 × r3)) / (r1 r2 r3 + r1(r2·r3) + r2(r3·r1) + r3(r1·r2)) )

该方法在目标点位于多面体外部时精确成立，内部时存在奇异性（可用于撞击检测）。
"""

import numpy as np
from typing import Tuple


class PolyhedronGravityError(Exception):
    pass


def edge_factor(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    计算边的对数项 L_e：
        L_e = ln( (r1 + r2 + e) / (r1 + r2 − e) )
    其中 e = |r2 − r1|，r1 = |r1|，r2 = |r2|。
    """
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
    """
    计算由位置向量 r1, r2, r3 张成的三角面的立体角 ω_f。
    公式：
        ω_f = 2 arctan( r1·(r2×r3) / D )
    其中 D = r1 r2 r3 + r1(r2·r3) + r2(r3·r1) + r3(r1·r2)
    """
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
    """
    使用多面体方法计算外部引力势。

    参数:
        pos: 场点位置 (m)
        vertices: (n_v, 3) 多面体顶点 (m)
        faces: (n_f, 3) 三角面片顶点索引
        density: 密度 (kg/m³)
        g_const: 万有引力常数

    返回:
        potential: 引力势 (m²/s²)
    """
    n_faces = faces.shape[0]
    potential = 0.0

    for fi in range(n_faces):
        v1 = vertices[faces[fi, 0]]
        v2 = vertices[faces[fi, 1]]
        v3 = vertices[faces[fi, 2]]

        r1 = v1 - pos
        r2 = v2 - pos
        r3 = v3 - pos

        # 面法向量
        n_vec = np.cross(v2 - v1, v3 - v1)
        area = 0.5 * np.linalg.norm(n_vec)
        if area < 1e-14:
            continue
        n_hat = n_vec / (2.0 * area)

        # TODO: Hole 1 — 实现多面体引力势的面贡献与边贡献
        # 科学背景：Werner-Scheeres 多面体引力模型
        #   U(r) = (1/2) Gρ Σ_e r_e · E_e · r_e · L_e
        #          − (1/2) Gρ Σ_f r_f · F_f · r_f · ω_f
        # 其中 ω_f 为面立体角，L_e 为边对数项，E_e 和 F_f 为边/面张量积。
        # 需要同时处理面贡献和边贡献，并累加到 potential。
        # 提示：omega 已由 face_solid_angle(r1, r2, r3) 计算；
        #       le 已由 edge_factor(re1, re2) 计算。
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
    """
    使用数值梯度计算多面体引力加速度：
        a = −∇U
    采用中心差分保证二阶精度。
    """
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
    """
    组合引力模型：在近场使用多面体方法，远场使用球谐展开，中间区域加权过渡。

    过渡权重（平滑函数）：
        w = 0.5 * (1 + tanh( (r − r_transition) / Δr ))
    w=0 时完全使用多面体，w=1 时完全使用球谐。
    """
    r = np.linalg.norm(pos)
    from gravity_harmonics import SphericalHarmonicGravity

    sh_model = SphericalHarmonicGravity(gm, r_ref, c_coeff, s_coeff, n_max)
    a_harm = sh_model.acceleration(pos)

    # TODO: Hole 2 — 调用多面体引力加速度并处理单位转换与过渡权重
    # 注意：polyhedron_gravity_acceleration 使用 m, kg, s 单位制，
    #       而球谐模型使用 km 单位制。必须统一单位后才能线性组合。
    #       同时需要设计平滑过渡权重，使近场使用多面体、远场使用球谐。
    raise NotImplementedError("Hole 2: 请实现组合模型中的单位转换与过渡权重")
