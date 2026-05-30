
import numpy as np
from typing import Tuple






def hashin_shtrikman_bounds_2d(E_m: float, nu_m: float, f: float) -> Tuple[float, float]:
    K_m = E_m / (2.0 * (1.0 - nu_m))
    G_m = E_m / (2.0 * (1.0 + nu_m))


    if f >= 1.0:
        return 0.0, 0.0
    K_upper = K_m + f / (-1.0 / K_m + (1.0 - f) / (K_m + G_m))
    G_upper = G_m + f / (-1.0 / G_m + (1.0 - f) * (K_m + 2.0 * G_m) / (2.0 * G_m * (K_m + G_m)))


    K_voigt = (1.0 - f) * K_m
    G_voigt = (1.0 - f) * G_m
    K_reuss = 0.0 if f > 0 else K_m
    G_reuss = 0.0 if f > 0 else G_m


    K_eff = max(0.0, min(K_upper, K_voigt))
    G_eff = max(0.0, min(G_upper, G_voigt))
    return K_eff, G_eff


def mori_tanaka_circle_holes_2d(E_m: float, nu_m: float, f: float) -> Tuple[float, float]:
    if f < 0.0:
        f = 0.0
    if f > 0.99:
        f = 0.99

    denom = 1.0 + 3.0 * f / (1.0 - (3.0 - nu_m) / (1.0 + nu_m) * f + 1e-14)
    if denom <= 0.0:
        E_eff = 0.0
    else:
        E_eff = E_m / denom
    E_eff = max(0.0, E_eff)

    nu_eff = max(0.0, min(0.5, nu_m * (1.0 - f)))
    return E_eff, nu_eff


def self_consistent_circle_holes(E_m: float, nu_m: float, f: float,
                                  max_iter: int = 50, tol: float = 1e-10) -> Tuple[float, float]:
    E_eff = E_m * (1.0 - f)
    nu_eff = nu_m
    for _ in range(max_iter):
        denom_E = 1.0 + f * (3.0 - nu_eff) / (1.0 + nu_eff + 1e-14)
        E_new = E_m * (1.0 - f) / denom_E
        denom_nu = 1.0 + f * (1.0 - 3.0 * nu_eff) / (1.0 + nu_eff + 1e-14)
        nu_new = nu_m * (1.0 - f) / denom_nu
        if abs(E_new - E_eff) < tol * E_m and abs(nu_new - nu_eff) < tol:
            break

        E_eff = 0.5 * E_eff + 0.5 * E_new
        nu_eff = 0.5 * nu_eff + 0.5 * nu_new
    return E_eff, nu_eff






def circle_monomial_integral(e1: int, e2: int) -> float:
    import math
    if e1 < 0 or e2 < 0:
        return 0.0
    if e1 % 2 == 1 or e2 % 2 == 1:
        return 0.0


    def gamma_half_int(k):


        if abs(k - round(k)) < 1e-12:
            return math.gamma(int(round(k)))
        else:
            return math.gamma(k)

    val = 2.0 * gamma_half_int((e1 + 1) / 2.0) * gamma_half_int((e2 + 1) / 2.0) / gamma_half_int((e1 + e2 + 2) / 2.0)
    return val


def effective_property_by_boundary_integral(porosity: float, n_harmonics: int = 6) -> float:

    f = porosity
    if f < 0.0:
        f = 0.0
    if f > 0.99:
        f = 0.99
    k_ratio_maxwell = 1.0 - 2.0 * f / (2.0 - f)



    Kt = 3.0



    integral_sum = 0.0
    for n in range(0, n_harmonics + 1, 2):


        Ix = circle_monomial_integral(n, 0)

        Iy = circle_monomial_integral(0, n)
        integral_sum += (Ix + Iy) / (2.0 * np.pi)


    numerical_factor = 1.0 + 0.01 * (integral_sum - 1.0)
    k_ratio = k_ratio_maxwell * numerical_factor
    return max(0.0, min(1.0, k_ratio))






def build_periodic_cell_mesh(lcell: float = 1.0, n_div: int = 20,
                              hole_radius: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    from fem_core import generate_rectangular_mesh
    node_xy_all, element_node_all = generate_rectangular_mesh(lcell, lcell, n_div, n_div)

    cx, cy = lcell / 2.0, lcell / 2.0

    inside_hole = np.sum((node_xy_all - np.array([cx, cy]))**2, axis=1) < hole_radius**2

    keep = ~inside_hole
    old_to_new = np.full(node_xy_all.shape[0], -1, dtype=np.int32)
    new_nodes = node_xy_all[keep]
    old_to_new[keep] = np.arange(new_nodes.shape[0])


    valid_elements = []
    for e in range(element_node_all.shape[0]):
        en = element_node_all[e]
        if keep[en[0]] and keep[en[1]] and keep[en[2]]:
            valid_elements.append([old_to_new[en[0]], old_to_new[en[1]], old_to_new[en[2]]])

    if len(valid_elements) == 0:
        raise RuntimeError("No valid elements after hole removal; reduce hole radius.")

    element_node_new = np.array(valid_elements, dtype=np.int32)


    tol = lcell / n_div * 0.5
    on_left = np.abs(new_nodes[:, 0]) < tol
    on_right = np.abs(new_nodes[:, 0] - lcell) < tol
    on_bottom = np.abs(new_nodes[:, 1]) < tol
    on_top = np.abs(new_nodes[:, 1] - lcell) < tol
    on_outer_boundary = on_left | on_right | on_bottom | on_top


    dist_to_center = np.linalg.norm(new_nodes - np.array([cx, cy]), axis=1)
    on_hole_boundary = np.abs(dist_to_center - hole_radius) < tol * 1.5

    boundary_nodes = np.where(on_outer_boundary | on_hole_boundary)[0]
    return new_nodes, element_node_new, boundary_nodes


def numerical_homogenization_2d(node_xy: np.ndarray, element_node: np.ndarray,
                                 E_m: float, nu_m: float,
                                 macro_strains: np.ndarray) -> np.ndarray:
    from fem_core import solve_fem_system
    n_nodes = node_xy.shape[0]
    n_dof = n_nodes * 2




    eps_xx, eps_yy, eps_xy = macro_strains


    tol = 1e-6
    lx = np.max(node_xy[:, 0]) - np.min(node_xy[:, 0])
    ly = np.max(node_xy[:, 1]) - np.min(node_xy[:, 1])
    x0, y0 = np.min(node_xy[:, 0]), np.min(node_xy[:, 1])

    bc_nodes = []
    bc_vals = []
    for i in range(n_nodes):
        x, y = node_xy[i]

        if (abs(x - x0) < tol and abs(y - y0) < tol) or \
           (abs(x - (x0+lx)) < tol and abs(y - y0) < tol) or \
           (abs(x - x0) < tol and abs(y - (y0+ly)) < tol) or \
           (abs(x - (x0+lx)) < tol and abs(y - (y0+ly)) < tol):
            bc_nodes.extend([2*i, 2*i+1])

            u_x = eps_xx * (x - x0) + eps_xy * (y - y0)
            u_y = eps_xy * (x - x0) + eps_yy * (y - y0)
            bc_vals.extend([u_x, u_y])

    bc_nodes = np.array(bc_nodes, dtype=np.int32)
    bc_vals = np.array(bc_vals, dtype=np.float64)


    F = np.zeros(n_dof, dtype=np.float64)


    U = solve_fem_system(node_xy, element_node, E_m, nu_m, F,
                          bc_nodes, bc_vals, plane_stress=True, element_type="T3")


    from fem_core import compute_element_stress
    stress = compute_element_stress(node_xy, element_node, U, E_m, nu_m,
                                     plane_stress=True, element_type="T3")
    sigma_avg = np.mean(stress, axis=0)
    return sigma_avg


def compute_effective_tensor_numerical(E_m: float, nu_m: float,
                                        hole_radius: float = 0.2,
                                        n_div: int = 16) -> np.ndarray:
    node_xy, element_node, _ = build_periodic_cell_mesh(
        lcell=1.0, n_div=n_div, hole_radius=hole_radius)

    C = np.zeros((3, 3), dtype=np.float64)


    s1 = numerical_homogenization_2d(node_xy, element_node, E_m, nu_m,
                                      np.array([1.0, 0.0, 0.0]))

    s2 = numerical_homogenization_2d(node_xy, element_node, E_m, nu_m,
                                      np.array([0.0, 1.0, 0.0]))

    s3 = numerical_homogenization_2d(node_xy, element_node, E_m, nu_m,
                                      np.array([0.0, 0.0, 1.0]))


    C[0, 0] = s1[0]
    C[1, 0] = s1[1]

    C[0, 1] = s2[0]
    C[1, 1] = s2[1]

    C[2, 2] = s3[2]


    C = 0.5 * (C + C.T)
    return C






def compute_effective_properties(E_m: float, nu_m: float, porosity: float,
                                  method: str = "mori_tanaka") -> Tuple[float, float]:
    if method == "mori_tanaka":
        return mori_tanaka_circle_holes_2d(E_m, nu_m, porosity)
    elif method == "self_consistent":
        return self_consistent_circle_holes(E_m, nu_m, porosity)
    elif method == "hashin_shtrikman":
        K_eff, G_eff = hashin_shtrikman_bounds_2d(E_m, nu_m, porosity)

        if K_eff < 1e-14 or G_eff < 1e-14:
            return 0.0, 0.0
        E_eff = 4.0 * K_eff * G_eff / (K_eff + G_eff)
        nu_eff = (K_eff - G_eff) / (K_eff + G_eff)
        return E_eff, nu_eff
    elif method == "numerical":


        r = np.sqrt(porosity / np.pi)
        r = min(r, 0.49)
        C = compute_effective_tensor_numerical(E_m, nu_m, hole_radius=r, n_div=12)

        if abs(C[0, 0]) < 1e-14:
            return 0.0, 0.0
        nu_eff = C[0, 1] / C[0, 0]
        E_eff = C[0, 0] * (1.0 - nu_eff * nu_eff)
        return E_eff, nu_eff
    else:
        raise ValueError(f"Unknown homogenization method: {method}")
