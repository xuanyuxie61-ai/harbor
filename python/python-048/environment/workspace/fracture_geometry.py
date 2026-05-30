
import numpy as np
from typing import Tuple, List


def bernstein_basis_3(u: float) -> np.ndarray:
    if not (0.0 <= u <= 1.0):
        u = max(0.0, min(1.0, u))
    u2 = u * u
    u3 = u2 * u
    om = 1.0 - u
    om2 = om * om
    om3 = om2 * om
    return np.array([om3,
                     3.0 * u * om2,
                     3.0 * u2 * om,
                     u3])


def bernstein_derivative_3(u: float) -> np.ndarray:
    if not (0.0 <= u <= 1.0):
        u = max(0.0, min(1.0, u))
    return np.array([-3.0 * (1.0 - u) ** 2,
                     3.0 * (1.0 - u) ** 2 - 6.0 * u * (1.0 - u),
                     6.0 * u * (1.0 - u) - 3.0 * u ** 2,
                     3.0 * u ** 2])


def bezier_patch_evaluate(control_points: np.ndarray, u: float, v: float) -> np.ndarray:
    if control_points.shape != (4, 4, 3):
        raise ValueError("control_points 必须是 (4,4,3) 数组")
    bu = bernstein_basis_3(u)
    bv = bernstein_basis_3(v)
    point = np.zeros(3)
    for i in range(4):
        for j in range(4):
            point += bu[i] * bv[j] * control_points[i, j, :]
    return point


def bezier_patch_normal(control_points: np.ndarray, u: float, v: float) -> np.ndarray:
    du = bernstein_derivative_3(u)
    dv = bernstein_derivative_3(v)
    bv = bernstein_basis_3(v)
    bu = bernstein_basis_3(u)

    Su = np.zeros(3)
    Sv = np.zeros(3)
    for i in range(4):
        for j in range(4):
            Su += du[i] * bv[j] * control_points[i, j, :]
            Sv += bu[i] * dv[j] * control_points[i, j, :]

    nvec = np.cross(Su, Sv)
    norm = np.linalg.norm(nvec)
    if norm < 1.0e-14:
        return np.array([0.0, 0.0, 1.0])
    return nvec / norm


def bezier_patch_area(control_points: np.ndarray, n_quad: int = 8) -> float:
    u_vals = np.linspace(0.0, 1.0, n_quad)
    v_vals = np.linspace(0.0, 1.0, n_quad)
    du = 1.0 / (n_quad - 1) if n_quad > 1 else 1.0
    dv = du
    area = 0.0
    for ui in u_vals:
        for vj in v_vals:
            du_basis = bernstein_derivative_3(ui)
            dv_basis = bernstein_derivative_3(vj)
            bv = bernstein_basis_3(vj)
            bu = bernstein_basis_3(ui)
            Su = np.zeros(3)
            Sv = np.zeros(3)
            for i in range(4):
                for j in range(4):
                    Su += du_basis[i] * bv[j] * control_points[i, j, :]
                    Sv += bu[i] * dv_basis[j] * control_points[i, j, :]
            area += np.linalg.norm(np.cross(Su, Sv)) * du * dv
    return float(area)


class FracturePatch:

    def __init__(self, control_points: np.ndarray, patch_id: int = 0):
        if control_points.shape != (4, 4, 3):
            raise ValueError("控制点维度必须为 (4,4,3)")
        self.cp = np.array(control_points, dtype=float)
        self.id = patch_id
        self._area = None

    def evaluate(self, u: float, v: float) -> np.ndarray:
        return bezier_patch_evaluate(self.cp, u, v)

    def normal(self, u: float, v: float) -> np.ndarray:
        return bezier_patch_normal(self.cp, u, v)

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = bezier_patch_area(self.cp, n_quad=12)
        return self._area

    def centroid(self) -> np.ndarray:
        return np.mean(self.cp.reshape(-1, 3), axis=0)


def create_planar_fracture_patch(center: np.ndarray, normal: np.ndarray,
                                  length: float, height: float,
                                  patch_id: int = 0) -> FracturePatch:
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1.0e-12:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal /= n_norm



    if abs(normal[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(normal, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)


    cp = np.zeros((4, 4, 3))
    for i in range(4):
        ui = (i / 3.0) - 0.5
        for j in range(4):
            vj = (j / 3.0) - 0.5
            p = center + ui * length * e1 + vj * height * e2

            bend = 0.02 * length * np.sin(2.0 * np.pi * ui) * np.cos(2.0 * np.pi * vj)
            p += bend * normal
            cp[i, j, :] = p

    return FracturePatch(cp, patch_id)
