
import numpy as np
from typing import List, Tuple
from utils import vector_norm, normalize_vector
from icf_parameters import LP, TP


class LaserBeam:

    def __init__(self, origin: np.ndarray, direction: np.ndarray, beam_id: int):
        self.origin = np.array(origin, dtype=float).reshape(3)
        self.direction = normalize_vector(np.array(direction, dtype=float).reshape(3))
        self.beam_id = beam_id

        if vector_norm(self.direction) < 1.0e-15:
            raise ValueError(f"激光束 {beam_id} 的方向向量无效")

    def point_at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction

    def distance_to_point(self, p: np.ndarray) -> float:
        diff = np.array(p, dtype=float).reshape(3) - self.origin
        cross = np.cross(diff, self.direction)
        return vector_norm(cross)

    def closest_point_on_line(self, p: np.ndarray) -> np.ndarray:
        diff = np.array(p, dtype=float).reshape(3) - self.origin
        t = np.dot(diff, self.direction)
        return self.point_at(t)

    def intersect_sphere(self, center: np.ndarray, radius: float) -> List[Tuple[float, np.ndarray]]:
        oc = self.origin - np.array(center, dtype=float).reshape(3)
        a = 1.0
        b = 2.0 * np.dot(oc, self.direction)
        c = np.dot(oc, oc) - radius * radius

        discriminant = b * b - 4.0 * a * c
        intersections = []
        if discriminant < 0.0:
            return intersections

        sqrt_disc = np.sqrt(discriminant)
        for t in [(-b - sqrt_disc) / (2.0 * a),
                  (-b + sqrt_disc) / (2.0 * a)]:
            if t >= 0.0:
                pt = self.point_at(t)
                intersections.append((t, pt))
        return intersections

    def incidence_angle_on_sphere(self, center: np.ndarray, hit_point: np.ndarray) -> float:
        normal = normalize_vector(hit_point - np.array(center, dtype=float).reshape(3))
        cos_theta = -np.dot(self.direction, normal)
        return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def create_nif_beam_geometry(num_cones: int = 4, beams_per_cone: int = 48) -> List[LaserBeam]:
    cone_angles_deg = [23.5, 30.0, 44.5, 50.0]
    if num_cones != len(cone_angles_deg):
        raise ValueError("锥角数量不匹配")

    beams: List[LaserBeam] = []
    chamber_radius = 5.0

    beam_id = 0
    for cone_idx, angle_deg in enumerate(cone_angles_deg):
        theta = np.radians(angle_deg)

        for i in range(beams_per_cone):
            phi = 2.0 * np.pi * i / beams_per_cone


            x0 = chamber_radius * np.sin(theta) * np.cos(phi)
            y0 = chamber_radius * np.sin(theta) * np.sin(phi)
            z0 = chamber_radius * np.cos(theta) * (1.0 if cone_idx % 2 == 0 else -1.0)
            origin = np.array([x0, y0, z0])


            direction = -origin / vector_norm(origin)

            beams.append(LaserBeam(origin, direction, beam_id))
            beam_id += 1

    return beams


def compute_deposition_profile(beams: List[LaserBeam],
                               r_grid: np.ndarray,
                               center: np.ndarray = np.zeros(3)) -> np.ndarray:
    n = len(r_grid) - 1
    deposition = np.zeros(n)

    for beam in beams:
        for i in range(n):
            r_inner = r_grid[i]
            r_outer = r_grid[i + 1]


            hits_inner = beam.intersect_sphere(center, r_inner)
            hits_outer = beam.intersect_sphere(center, r_outer)
            if hits_inner and hits_outer:

                t_in = min([h[0] for h in hits_inner])
                t_out = min([h[0] for h in hits_outer])
                path_length = abs(t_out - t_in)

                area = 4.0 * np.pi * (r_outer**2 - r_inner**2)
                if area > 0.0:
                    deposition[i] += path_length / area


    total = np.sum(deposition)
    if total > 0.0:
        deposition /= total
    return deposition


def laser_beam_characteristics(beams: List[LaserBeam]) -> dict:
    n = len(beams)
    angles = []
    for beam in beams:

        cos_z = abs(beam.direction[2])
        angles.append(np.degrees(np.arccos(np.clip(cos_z, 0.0, 1.0))))

    return {
        "num_beams": n,
        "mean_polar_angle_deg": float(np.mean(angles)),
        "std_polar_angle_deg": float(np.std(angles)),
        "min_polar_angle_deg": float(np.min(angles)),
        "max_polar_angle_deg": float(np.max(angles)),
    }
