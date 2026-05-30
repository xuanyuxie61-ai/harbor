
import numpy as np
from typing import Tuple, List, Optional


class AtmosphericMesh:

    def __init__(self, n_layers: int, P_top: float, P_bot: float,
                 planet_radius_m: float):
        self.n_layers = n_layers
        self.P_top = P_top
        self.P_bot = P_bot
        self.R_p = planet_radius_m


        self.P_interface = self._log_pressure_grid(n_layers + 1, P_top, P_bot)
        self.P_center = 0.5 * (self.P_interface[:-1] + self.P_interface[1:])
        self.dP = np.diff(self.P_interface)

    def _log_pressure_grid(self, n: int, P_min: float, P_max: float) -> np.ndarray:
        logP = np.linspace(np.log10(P_min), np.log10(P_max), n)
        return 10.0**logP

    def adaptive_refinement(self, error_estimator: np.ndarray,
                            max_layers: int = 200,
                            tol: float = 0.1) -> "AtmosphericMesh":
        error_estimator = np.asarray(error_estimator, dtype=np.float64)
        if error_estimator.shape[0] != self.n_layers:
            raise ValueError("误差估计维度与层数不匹配")

        P_int = self.P_interface.copy()
        err = error_estimator.copy()

        while len(P_int) - 1 < max_layers:
            err_max = np.max(err)
            if err_max < 1e-30:
                break

            rel_err = err / err_max
            refine_mask = rel_err > tol

            if not np.any(refine_mask):
                break


            refine_idx = np.where(refine_mask)[0]
            if len(refine_idx) == 0:
                break

            new_P_int = [P_int[0]]
            new_err = []
            for i in range(len(P_int) - 1):
                new_P_int.append(P_int[i + 1])
                new_err.append(err[i])
                if i in refine_idx and len(new_P_int) < max_layers + 1:

                    P_mid = np.sqrt(P_int[i] * P_int[i + 1])
                    new_P_int.insert(-1, P_mid)

                    new_err.insert(-1, err[i] * 0.25)

            P_int = np.array(new_P_int)
            err = np.array(new_err)

        mesh = AtmosphericMesh.__new__(AtmosphericMesh)
        mesh.n_layers = len(P_int) - 1
        mesh.P_top = P_int[0]
        mesh.P_bot = P_int[-1]
        mesh.R_p = self.R_p
        mesh.P_interface = P_int
        mesh.P_center = 0.5 * (P_int[:-1] + P_int[1:])
        mesh.dP = np.diff(P_int)
        return mesh

    def generate_2d_shell_mesh(self, n_angular: int = 32,
                                r_min_factor: float = 1.0,
                                r_max_factor: float = 1.1) -> Tuple[np.ndarray, np.ndarray]:
        if n_angular < 3:
            raise ValueError("角度方向节点数至少为 3")


        n_radial = self.n_layers + 1


        xi = np.linspace(0.0, 1.0, n_radial)

        xi = xi**1.5
        r_nodes = self.R_p * (r_min_factor + (r_max_factor - r_min_factor) * xi)

        theta_nodes = np.linspace(0.0, np.pi, n_angular)


        nodes = []
        for j in range(n_radial):
            for i in range(n_angular):
                nodes.append([r_nodes[j], theta_nodes[i]])
        nodes = np.array(nodes, dtype=np.float64)


        elements = []
        for j in range(n_radial - 1):
            for i in range(n_angular - 1):
                n0 = j * n_angular + i
                n1 = j * n_angular + (i + 1)
                n2 = (j + 1) * n_angular + i
                n3 = (j + 1) * n_angular + (i + 1)

                elements.append([n0, n1, n2])
                elements.append([n1, n3, n2])
        elements = np.array(elements, dtype=np.int64)

        return nodes, elements

    def mesh_quality_metrics(self, nodes: np.ndarray, elements: np.ndarray) -> dict:
        if elements.shape[1] != 3:
            raise ValueError("仅支持三角形单元")

        qualities = []
        min_angles = []
        max_angles = []
        aspect_ratios = []

        for tri in elements:
            p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]


            a = np.linalg.norm(p1 - p2)
            b = np.linalg.norm(p0 - p2)
            c = np.linalg.norm(p0 - p1)


            if a < 1e-15 or b < 1e-15 or c < 1e-15:
                continue


            ang0 = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0))
            ang1 = np.arccos(np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0))
            ang2 = np.arccos(np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0))

            angles = np.array([ang0, ang1, ang2]) * 180.0 / np.pi
            min_angles.append(np.min(angles))
            max_angles.append(np.max(angles))
            aspect_ratios.append(np.max([a, b, c]) / np.min([a, b, c]))


            s = 0.5 * (a + b + c)
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-30))
            r_in = area / s
            r_circ = a * b * c / (4.0 * max(area, 1e-30))
            quality = 2.0 * r_in / max(r_circ, 1e-30)
            qualities.append(quality)

        return {
            'min_angle_deg': np.min(min_angles) if min_angles else 0.0,
            'max_angle_deg': np.max(max_angles) if max_angles else 180.0,
            'mean_quality': np.mean(qualities) if qualities else 0.0,
            'min_quality': np.min(qualities) if qualities else 0.0,
            'max_aspect_ratio': np.max(aspect_ratios) if aspect_ratios else 1e10
        }


def distance_function_sphere_shell(points: np.ndarray, R_inner: float,
                                    R_outer: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(points, axis=1)
    d_inner = R_inner - r
    d_outer = r - R_outer
    return np.maximum(d_inner, d_outer)


def mesh_size_function(points: np.ndarray, R_p: float,
                        h_min: float = 1e3, h_max: float = 1e5) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(points, axis=1)
    delta_r = np.maximum(r - R_p, 0.0)
    H_atm = 5e6
    alpha = 0.8
    h = h_min + (h_max - h_min) * (delta_r / H_atm)**alpha
    return np.clip(h, h_min, h_max)
