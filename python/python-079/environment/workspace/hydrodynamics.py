
import numpy as np
from typing import Tuple, List, Optional
from mesh_geometry import polygon_area_2d, polygon_centroid_2d, polygon_solid_angle_3d






def green_function_3d(
    x: np.ndarray, xi: np.ndarray, use_image: bool = True
) -> float:
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    r = np.linalg.norm(x - xi)
    if r < 1e-12:
        return 0.0
    val = 1.0 / r
    if use_image:
        xi_star = xi.copy()
        xi_star[2] = -xi[2]
        r_star = np.linalg.norm(x - xi_star)
        if r_star > 1e-12:
            val += 1.0 / r_star
    return val


def panel_normal_3d(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0])
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-15:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def panel_area_3d(vertices: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=float)
    n = panel_normal_3d(vertices)

    abs_n = np.abs(n)
    proj_axis = np.argmax(abs_n)

    coords = np.delete(vertices, proj_axis, axis=1)
    return abs(polygon_area_2d(coords))


def panel_centroid_3d(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    n = panel_normal_3d(vertices)
    abs_n = np.abs(n)
    proj_axis = np.argmax(abs_n)
    coords = np.delete(vertices, proj_axis, axis=1)
    c2d = polygon_centroid_2d(coords)

    centroid = np.zeros(3)
    other_axes = [i for i in range(3) if i != proj_axis]
    centroid[other_axes[0]] = c2d[0]
    centroid[other_axes[1]] = c2d[1]
    centroid[proj_axis] = np.mean(vertices[:, proj_axis])
    return centroid






def compute_hydrodynamic_coefficients_panel_method(
    panels: List[np.ndarray],
    omega: float,
    rho: float = 1025.0,
    h: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_panels = len(panels)
    if n_panels == 0:
        return np.zeros((6, 6)), np.zeros((6, 6)), np.zeros(6)

    g = 9.80665
    k = omega * omega / g


    areas = np.zeros(n_panels)
    centroids = np.zeros((n_panels, 3))
    normals = np.zeros((n_panels, 3))
    for i, p in enumerate(panels):
        areas[i] = panel_area_3d(p)
        centroids[i] = panel_centroid_3d(p)
        normals[i] = panel_normal_3d(p)







    A = np.zeros((6, 6))
    B = np.zeros((6, 6))
    for i in range(n_panels):
        nvec = normals[i]
        area = areas[i]

        for m in range(3):
            A[m, m] += rho * area * nvec[m] ** 2 / k
            B[m, m] += rho * omega * area * nvec[m] ** 2 / (k ** 2)

        r = centroids[i]
        rxn = np.cross(r, nvec)
        for m in range(3):
            A[m + 3, m + 3] += rho * area * rxn[m] ** 2 / k
            B[m + 3, m + 3] += rho * omega * area * rxn[m] ** 2 / (k ** 2)


    F_tf = np.zeros(6)
    for i in range(n_panels):
        r = centroids[i]
        nvec = normals[i]
        area = areas[i]
        z = r[2]

        p_amp = rho * g * np.cosh(k * (z + h)) / np.cosh(k * h)

        F_tf[:3] += p_amp * area * nvec

        F_tf[3:] += p_amp * area * np.cross(r, nvec)

    return A, B, F_tf






def morison_force_1d(
    u: float,
    u_dot: float,
    xi_dot: float,
    xi_ddot: float,
    D: float,
    C_d: float = 1.0,
    C_m: float = 2.0,
    rho: float = 1025.0,
    dz: float = 1.0,
) -> Tuple[float, float]:
    rel_vel = u - xi_dot
    drag = 0.5 * rho * C_d * D * abs(rel_vel) * rel_vel * dz
    inertia = rho * C_m * (np.pi * D * D / 4.0) * (u_dot - xi_ddot) * dz
    return drag, inertia


def morison_force_on_platform_column(
    wave_kinematics: dict,
    platform_motion: dict,
    column_diameter: float = 15.0,
    draft: float = 20.0,
    z_nodes: Optional[np.ndarray] = None,
) -> np.ndarray:
    if z_nodes is None:
        z_nodes = np.linspace(-draft, 0.0, 41)
    dz = z_nodes[1] - z_nodes[0] if len(z_nodes) > 1 else 1.0
    F = np.zeros(6)
    u = wave_kinematics.get("u", np.zeros_like(z_nodes))
    u_dot = wave_kinematics.get("u_dot", np.zeros_like(z_nodes))
    xi_dot = platform_motion.get("xi_dot", 0.0)
    xi_ddot = platform_motion.get("xi_ddot", 0.0)
    for idx, z in enumerate(z_nodes):
        ui = u[idx] if idx < len(u) else 0.0
        udi = u_dot[idx] if idx < len(u_dot) else 0.0
        fd, fi = morison_force_1d(ui, udi, xi_dot, xi_ddot, column_diameter, dz=abs(dz))
        F[0] += fd + fi

        F[4] += (fd + fi) * z
    return F






def generate_semi_submersible_panels(
    col_spacing_x: float = 55.0,
    col_spacing_y: float = 40.0,
    col_diameter: float = 15.0,
    col_height: float = 20.0,
    pontoon_width: float = 10.0,
    pontoon_height: float = 8.0,
    n_azimuth: int = 16,
) -> List[np.ndarray]:
    panels = []

    col_positions = [
        (-col_spacing_x * 0.5, -col_spacing_y * 0.5, -col_height * 0.5),
        (-col_spacing_x * 0.5, col_spacing_y * 0.5, -col_height * 0.5),
        (col_spacing_x * 0.5, -col_spacing_y * 0.5, -col_height * 0.5),
        (col_spacing_x * 0.5, col_spacing_y * 0.5, -col_height * 0.5),
    ]

    theta = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    dtheta = theta[1] - theta[0]
    for cx, cy, cz in col_positions:
        r = col_diameter * 0.5
        for t in theta:
            t2 = t + dtheta
            x1 = cx + r * np.cos(t)
            y1 = cy + r * np.sin(t)
            x2 = cx + r * np.cos(t2)
            y2 = cy + r * np.sin(t2)

            panel = np.array(
                [
                    [x1, y1, cz - col_height * 0.5],
                    [x2, y2, cz - col_height * 0.5],
                    [x2, y2, cz + col_height * 0.5],
                    [x1, y1, cz + col_height * 0.5],
                ],
                dtype=float,
            )
            panels.append(panel)


    connections = [
        (0, 1),
        (2, 3),
        (0, 2),
        (1, 3),
    ]
    for i, j in connections:
        c1 = np.array(col_positions[i])
        c2 = np.array(col_positions[j])
        mid = 0.5 * (c1 + c2)
        length = np.linalg.norm(c2[:2] - c1[:2])
        direction = (c2[:2] - c1[:2]) / length
        normal = np.array([-direction[1], direction[0]])
        w = pontoon_width * 0.5
        h = pontoon_height * 0.5

        corners = [
            mid[:2] + w * normal + np.array([0, 0]),
            mid[:2] - w * normal + np.array([0, 0]),
        ]
        for sign in [-1, 1]:
            z_bottom = mid[2] - h
            z_top = mid[2] + h
            for ci in range(2):
                c = corners[ci]
                c_next = corners[(ci + 1) % 2]
                panel = np.array(
                    [
                        [c[0], c[1], z_bottom],
                        [c_next[0], c_next[1], z_bottom],
                        [c_next[0], c_next[1], z_top],
                        [c[0], c[1], z_top],
                    ],
                    dtype=float,
                )
                panels.append(panel)
    return panels






def airy_wave_kinematics(
    x: float,
    z: float,
    t: float,
    A: float,
    T: float,
    h: float,
    beta: float = 0.0,
) -> dict:







    raise NotImplementedError("airy_wave_kinematics 需要实现")


rho = 1025.0
