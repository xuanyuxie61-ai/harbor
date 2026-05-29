"""
geometry_laser.py
=================
多束激光入射几何计算模块。

基于原项目 1258_theodolite（三维测地线与点线距离）的核心思想，
计算多束激光束与球形靶丸的三维交点、入射角及能量沉积分布。

在ICF中，NIF使用192束激光从多个锥角（23.5°, 30°, 44.5°, 50°）
对称入射。本模块建立各激光束的参数化直线方程，并计算其与靶丸
表面的几何关系。
"""

import numpy as np
from typing import List, Tuple
from utils import vector_norm, normalize_vector
from icf_parameters import LP, TP


class LaserBeam:
    """
    单束激光的几何描述。
    参数化直线: P(t) = P0 + t * D, 其中 D 为单位方向向量。
    """

    def __init__(self, origin: np.ndarray, direction: np.ndarray, beam_id: int):
        self.origin = np.array(origin, dtype=float).reshape(3)
        self.direction = normalize_vector(np.array(direction, dtype=float).reshape(3))
        self.beam_id = beam_id
        # 验证方向向量有效性
        if vector_norm(self.direction) < 1.0e-15:
            raise ValueError(f"激光束 {beam_id} 的方向向量无效")

    def point_at(self, t: float) -> np.ndarray:
        """返回参数 t 处的点坐标。"""
        return self.origin + t * self.direction

    def distance_to_point(self, p: np.ndarray) -> float:
        """
        计算空间点 p 到激光直线的垂直距离。
        公式: d = | (P - P0) x D | / |D|
        其中 D 已归一化，故 d = | (P - P0) x D |
        """
        diff = np.array(p, dtype=float).reshape(3) - self.origin
        cross = np.cross(diff, self.direction)
        return vector_norm(cross)

    def closest_point_on_line(self, p: np.ndarray) -> np.ndarray:
        """计算点 p 在激光直线上的垂足。"""
        diff = np.array(p, dtype=float).reshape(3) - self.origin
        t = np.dot(diff, self.direction)
        return self.point_at(t)

    def intersect_sphere(self, center: np.ndarray, radius: float) -> List[Tuple[float, np.ndarray]]:
        """
        计算激光直线与球面的交点。
        球面方程: |P - C|^2 = R^2
        代入参数方程: |P0 + t*D - C|^2 = R^2
        得到二次方程: a*t^2 + b*t + c = 0
        """
        oc = self.origin - np.array(center, dtype=float).reshape(3)
        a = 1.0  # |D|^2 = 1
        b = 2.0 * np.dot(oc, self.direction)
        c = np.dot(oc, oc) - radius * radius

        discriminant = b * b - 4.0 * a * c
        intersections = []
        if discriminant < 0.0:
            return intersections  # 无交点

        sqrt_disc = np.sqrt(discriminant)
        for t in [(-b - sqrt_disc) / (2.0 * a),
                  (-b + sqrt_disc) / (2.0 * a)]:
            if t >= 0.0:  # 只考虑正向传播
                pt = self.point_at(t)
                intersections.append((t, pt))
        return intersections

    def incidence_angle_on_sphere(self, center: np.ndarray, hit_point: np.ndarray) -> float:
        """
        计算激光在球面交点处的入射角（与法向的夹角）。
        法向量 n = (hit_point - center) / |...|
        cos(theta) = - D · n  （取负号因为激光射向靶丸）
        """
        normal = normalize_vector(hit_point - np.array(center, dtype=float).reshape(3))
        cos_theta = -np.dot(self.direction, normal)
        return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def create_nif_beam_geometry(num_cones: int = 4, beams_per_cone: int = 48) -> List[LaserBeam]:
    """
    构造NIF-like激光束几何排布。
    使用四个锥角（近环与远环），每个锥角均匀分布 beam_per_cone 束激光。

    锥角列表（度）: 23.5, 30.0, 44.5, 50.0
    对应 NIF 的内环(inner)、中环(middle)、外环(outer)及极区(polar)。
    """
    cone_angles_deg = [23.5, 30.0, 44.5, 50.0]
    if num_cones != len(cone_angles_deg):
        raise ValueError("锥角数量不匹配")

    beams: List[LaserBeam] = []
    chamber_radius = 5.0  # 激光入口到靶丸中心的典型距离 [m]

    beam_id = 0
    for cone_idx, angle_deg in enumerate(cone_angles_deg):
        theta = np.radians(angle_deg)
        # 每个锥角均匀分布
        for i in range(beams_per_cone):
            phi = 2.0 * np.pi * i / beams_per_cone

            # 激光入口位置（球坐标）
            x0 = chamber_radius * np.sin(theta) * np.cos(phi)
            y0 = chamber_radius * np.sin(theta) * np.sin(phi)
            z0 = chamber_radius * np.cos(theta) * (1.0 if cone_idx % 2 == 0 else -1.0)
            origin = np.array([x0, y0, z0])

            # 方向指向原点
            direction = -origin / vector_norm(origin)

            beams.append(LaserBeam(origin, direction, beam_id))
            beam_id += 1

    return beams


def compute_deposition_profile(beams: List[LaserBeam],
                               r_grid: np.ndarray,
                               center: np.ndarray = np.zeros(3)) -> np.ndarray:
    """
    计算激光能量在径向网格上的几何沉积权重。
    基于各激光束与球壳的交点，给出每层的相对能量沉积量。

    参数
    ----
    beams : List[LaserBeam]
        激光束列表
    r_grid : np.ndarray
        径向网格坐标（递增）
    center : np.ndarray
        靶丸中心

    返回
    ----
    deposition : np.ndarray
        每个网格层的归一化能量沉积权重
    """
    n = len(r_grid) - 1
    deposition = np.zeros(n)

    for beam in beams:
        for i in range(n):
            r_inner = r_grid[i]
            r_outer = r_grid[i + 1]
            # 计算该层与激光束的交线长度
            # 简化模型：使用球壳厚度与入射角关系
            hits_inner = beam.intersect_sphere(center, r_inner)
            hits_outer = beam.intersect_sphere(center, r_outer)
            if hits_inner and hits_outer:
                # 取最近的有效交点
                t_in = min([h[0] for h in hits_inner])
                t_out = min([h[0] for h in hits_outer])
                path_length = abs(t_out - t_in)
                # 权重反比于表面积（球壳层）
                area = 4.0 * np.pi * (r_outer**2 - r_inner**2)
                if area > 0.0:
                    deposition[i] += path_length / area

    # 归一化
    total = np.sum(deposition)
    if total > 0.0:
        deposition /= total
    return deposition


def laser_beam_characteristics(beams: List[LaserBeam]) -> dict:
    """统计激光束几何特征。"""
    n = len(beams)
    angles = []
    for beam in beams:
        # 计算与z轴夹角
        cos_z = abs(beam.direction[2])
        angles.append(np.degrees(np.arccos(np.clip(cos_z, 0.0, 1.0))))

    return {
        "num_beams": n,
        "mean_polar_angle_deg": float(np.mean(angles)),
        "std_polar_angle_deg": float(np.std(angles)),
        "min_polar_angle_deg": float(np.min(angles)),
        "max_polar_angle_deg": float(np.max(angles)),
    }
