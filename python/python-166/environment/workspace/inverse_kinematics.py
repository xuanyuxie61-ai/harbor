
import numpy as np
from typing import Tuple, Optional, List


def barycentric_coordinates(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[float, float, float]:
    def tri_area(p1, p2, p3):
        return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

    total_area = tri_area(a, b, c)
    if total_area < 1e-14:
        return -1.0, -1.0, -1.0

    alpha = tri_area(p, b, c) / total_area
    beta = tri_area(p, a, c) / total_area
    gamma = 1.0 - alpha - beta
    return alpha, beta, gamma


def point_in_triangle(alpha: float, beta: float, gamma: float, tol: float = -1e-10) -> bool:
    return alpha >= tol and beta >= tol and gamma >= tol


def delaunay_triangulation_2d(points: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        return tri.simplices
    except ImportError:

        from mesh_utils import compute_polygon_area

        n = points.shape[0]
        if n < 3:
            return np.array([])


        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        order = np.argsort(angles)

        triangles = []
        for i in range(1, n - 1):
            triangles.append([order[0], order[i], order[i + 1]])
        return np.array(triangles, dtype=int)


def pwl_interp_2d_scattered(xyd: np.ndarray, zd: np.ndarray,
                            xyi: np.ndarray) -> np.ndarray:
    nd = xyd.shape[0]
    ni = xyi.shape[0]

    if len(zd) != nd:
        raise ValueError("zd must have same length as xyd")


    triangles = delaunay_triangulation_2d(xyd)

    if len(triangles) == 0:
        return np.zeros(ni)


    n_tri = len(triangles)

    zi = np.zeros(ni)
    for qi in range(ni):
        p = xyi[qi]
        found = False


        for tri in triangles:
            a = xyd[tri[0]]
            b = xyd[tri[1]]
            c = xyd[tri[2]]
            alpha, beta, gamma = barycentric_coordinates(p, a, b, c)
            if point_in_triangle(alpha, beta, gamma):
                zi[qi] = alpha * zd[tri[0]] + beta * zd[tri[1]] + gamma * zd[tri[2]]
                found = True
                break

        if not found:

            dists = np.sum((xyd - p) ** 2, axis=1)
            nearest = np.argmin(dists)
            zi[qi] = zd[nearest]

    return zi


def shape_reconstruction_from_sensors(sensor_positions: np.ndarray,
                                      sensor_readings: np.ndarray,
                                      query_points: np.ndarray,
                                      reconstruction_type: str = 'pwl') -> np.ndarray:
    if reconstruction_type == 'pwl':
        return pwl_interp_2d_scattered(sensor_positions, sensor_readings, query_points)
    elif reconstruction_type == 'rbf':

        return rbf_interp_2d(sensor_positions, sensor_readings, query_points)
    else:
        raise ValueError(f"Unknown reconstruction type: {reconstruction_type}")


def rbf_interp_2d(xyd: np.ndarray, zd: np.ndarray, xyi: np.ndarray,
                  epsilon: float = 1.0) -> np.ndarray:
    nd = xyd.shape[0]
    ni = xyi.shape[0]


    Phi = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(nd):
            r = np.linalg.norm(xyd[i] - xyd[j])
            Phi[i, j] = np.exp(-(epsilon * r) ** 2)


    try:
        w = np.linalg.solve(Phi, zd)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Phi, zd, rcond=None)[0]


    zi = np.zeros(ni)
    for i in range(ni):
        for j in range(nd):
            r = np.linalg.norm(xyi[i] - xyd[j])
            zi[i] += w[j] * np.exp(-(epsilon * r) ** 2)

    return zi


def inverse_kinematics_soft_robot(target_tip: np.ndarray,
                                  L: float, Ns: int,
                                  material_params: dict,
                                  max_iter: int = 100,
                                  tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    from cosserat_core import forward_kinematics_cosserat

    n_nodes = Ns + 1

    kappa = np.zeros((n_nodes, 3))

    for iteration in range(max_iter):
        s, r, R = forward_kinematics_cosserat(L, Ns, kappa)
        error = r[-1] - target_tip
        err_norm = np.linalg.norm(error)

        if err_norm < tol:
            break







        raise NotImplementedError("Hole 2: 实现Newton迭代核心")


        kappa_max = 2.0 * np.pi / L
        kappa = np.clip(kappa, -kappa_max, kappa_max)

    s, r_final, R = forward_kinematics_cosserat(L, Ns, kappa)
    return kappa, r_final
