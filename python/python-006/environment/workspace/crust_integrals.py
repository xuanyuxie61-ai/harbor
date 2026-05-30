
import numpy as np
import math
from typing import Tuple





def triangle_area(t: np.ndarray) -> float:
    area = 0.5 * (
        t[0, 0] * (t[1, 1] - t[1, 2])
        + t[0, 1] * (t[1, 2] - t[1, 0])
        + t[0, 2] * (t[1, 0] - t[1, 1])
    )
    return abs(area)


def triangle_reference_to_physical(t: np.ndarray, n: int, ref: np.ndarray) -> np.ndarray:
    phy = np.zeros((2, n))
    phy[0, :] = t[0, 0] + (t[0, 1] - t[0, 0]) * ref[0, :] + (t[0, 2] - t[0, 0]) * ref[1, :]
    phy[1, :] = t[1, 0] + (t[1, 1] - t[1, 0]) * ref[0, :] + (t[1, 2] - t[1, 0]) * ref[1, :]
    return phy


def triangle_unit_monomial_integral(expon: Tuple[int, int]) -> float:
    m, n = expon
    if m < 0 or n < 0:
        raise ValueError("Exponents must be non-negative.")

    value = 1.0
    k = m
    for i in range(1, n + 1):
        k += 1
        value *= i / k
    k += 1
    value /= k
    k += 1
    value /= k
    return value


def monomial_value(m_dim: int, n_pts: int, e: Tuple[int, ...], x: np.ndarray) -> np.ndarray:
    v = np.ones(n_pts)
    for i in range(m_dim):
        if e[i] != 0:
            v *= x[i, :] ** e[i]
    return v


def triangle_quadrature_error(dim_num: int, expon: Tuple[int, ...],
                               point_num: int, x_ref: np.ndarray,
                               w: np.ndarray) -> float:
    if dim_num != 2:
        raise ValueError("Only 2D triangles supported.")

    exact = triangle_unit_monomial_integral(expon)
    vals = monomial_value(dim_num, point_num, expon, x_ref)
    quad = np.dot(w, vals)
    return abs(quad - exact)





def quadrilateral_unit_area() -> float:
    return 1.0


def quadrilateral_unit_monomial_integral(e: Tuple[int, int]) -> float:
    a, b = e
    if a < 0 or b < 0:
        raise ValueError("Exponents must be non-negative.")
    return 1.0 / ((a + 1) * (b + 1))


def quadrilateral_witherden_rule(p: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if p < 0:
        raise ValueError("Precision p must be non-negative.")

    if p <= 1:

        n = 1
        x = np.array([0.5])
        y = np.array([0.5])
        w = np.array([1.0])
    elif p <= 3:

        n = 4
        pts = np.array([0.211324865405187, 0.788675134594813])
        wg = np.array([0.5, 0.5])
        x, y = np.meshgrid(pts, pts)
        x = x.ravel()
        y = y.ravel()
        w = np.outer(wg, wg).ravel()
    elif p <= 5:

        n = 9
        pts = np.array([0.112701665379258, 0.5, 0.887298334620742])
        wg = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])
        x, y = np.meshgrid(pts, pts)
        x = x.ravel()
        y = y.ravel()
        w = np.outer(wg, wg).ravel()
    else:

        n = 25
        pts = np.array([
            0.046910077030668, 0.230765344947158, 0.5,
            0.769234655052842, 0.953089922969332
        ])
        wg = np.array([
            0.118463442528095, 0.239314335249683, 0.284444444444444,
            0.239314335249683, 0.118463442528095
        ])
        x, y = np.meshgrid(pts, pts)
        x = x.ravel()
        y = y.ravel()
        w = np.outer(wg, wg).ravel()

    return n, x, y, w





def wedge01_volume() -> float:
    return 1.0


def wedge01_monomial_integral(e: Tuple[int, int, int]) -> float:
    ex, ey, ez = e
    if ex < 0 or ey < 0:
        raise ValueError("Exponents ex, ey must be non-negative.")
    if ez == -1:
        raise ValueError("ez = -1 is not a legal input.")
    if ez % 2 == 1:
        return 0.0

    value = 1.0
    k = ex
    for i in range(1, ey + 1):
        k += 1
        value *= i / k
    k += 1
    value /= k
    k += 1
    value /= k
    value *= 2.0 / (ez + 1)
    return value





def pasta_free_energy_triangle_lattice(
    lattice_spacing: float,
    surface_tension: float,
    nuclear_radius: float
) -> float:
    if lattice_spacing <= 0.0 or nuclear_radius <= 0.0 or surface_tension < 0.0:
        raise ValueError("All physical parameters must be positive.")


    a = lattice_spacing
    t = np.array([
        [0.0, a, a / 2.0],
        [0.0, 0.0, a * math.sqrt(3.0) / 2.0]
    ])
    area = triangle_area(t)



    ref_pts = np.array([[1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]])
    weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])

    phy_pts = triangle_reference_to_physical(t, 3, ref_pts)

    F = 0.0
    center = np.array([a / 2.0, a * math.sqrt(3.0) / 6.0])
    for i in range(3):
        r_vec = phy_pts[:, i] - center
        r_dist = math.sqrt(np.sum(r_vec**2))

        eps_vol = -15.0

        surf = surface_tension if abs(r_dist - nuclear_radius) < 0.1 else 0.0
        F += weights[i] * (eps_vol + surf)

    return F * area


def pasta_free_energy_quadrilateral_sheet(
    sheet_width: float,
    sheet_spacing: float,
    surface_tension: float
) -> float:
    if sheet_width <= 0.0 or sheet_spacing <= 0.0:
        raise ValueError("Physical parameters must be positive.")

    n, x, y, w = quadrilateral_witherden_rule(5)

    F = 0.0
    for i in range(n):

        xi = x[i] * sheet_spacing
        yi = y[i] * sheet_spacing
        eps_vol = -15.0

        in_sheet = abs(xi - sheet_spacing / 2.0) < sheet_width / 2.0
        surf = surface_tension if in_sheet else 0.0
        F += w[i] * (eps_vol + surf)

    return F * sheet_spacing**2


def pasta_free_energy_wedge_cylinder(
    cylinder_radius: float,
    cell_size: float,
    surface_tension: float
) -> float:
    if cylinder_radius <= 0.0 or cell_size <= 0.0:
        raise ValueError("Physical parameters must be positive.")



    n_tri, x_tri, y_tri, w_tri = quadrilateral_witherden_rule(3)

    x_tri = x_tri[:3]
    y_tri = y_tri[:3]
    w_tri = w_tri[:3]

    z_pts = np.array([-0.577350269189626, 0.577350269189626])
    z_w = np.array([1.0, 1.0])

    F = 0.0
    for i in range(len(x_tri)):
        for j in range(len(z_pts)):
            xi = x_tri[i] * cell_size
            yi = y_tri[i] * cell_size
            zi = z_pts[j] * cell_size
            eps_vol = -15.0
            r_dist = math.sqrt(xi**2 + yi**2)
            in_cyl = r_dist < cylinder_radius
            surf = surface_tension if abs(r_dist - cylinder_radius) < 0.05 else 0.0
            F += w_tri[i] * z_w[j] * (eps_vol + surf) * in_cyl

    return F * cell_size**3
