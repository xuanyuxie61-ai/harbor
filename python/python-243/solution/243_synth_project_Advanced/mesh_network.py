"""
mesh_network.py - 2D nuclide mesh construction and polygonal boundary.

Adapted from 580_image_mesh2d and 891_polygonal_surface_display.
Constructs a structured (A, Z) mesh for the nuclear network and defines
the polygonal boundary (drip lines, stability line).
"""
from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple

def beta_stability_line(A: int) -> int:
    """Return Z on the beta-stability line for mass A.
    Z ~ A / (1.98 + 0.0155 A^{2/3})  (empirical).
    """
    if A <= 0:
        return 1
    return max(1, min(A, int(round(A / (1.98 + 0.0155 * A ** (2.0 / 3.0))))))


def neutron_drip_line(A: int) -> int:
    """Return approximate Z at the neutron drip line for mass A.
    Very neutron-rich: Z ~ A * 0.3 for A < 100, lower for heavier.
    """
    if A <= 0:
        return 1
    return max(1, int(A * (0.35 - 0.001 * A)))


def proton_drip_line(A: int) -> int:
    """Return approximate Z at proton drip line."""
    if A <= 0:
        return 1
    return min(A, beta_stability_line(A) + 5)


def build_nuclide_mesh(A_min: int, A_max: int,
                       Z_offset_min: int = -30, Z_offset_max: int = 30
                       ) -> List[Tuple[int, int, float]]:
    """
    Build a list of (A, Z, weight) nuclides in a band around stability.
    weight = 1 inside allowed region, 0 outside.
    """
    mesh = []
    for A in range(A_min, A_max + 1):
        Z_center = beta_stability_line(A)
        Z_lo = max(1, Z_center + Z_offset_min)
        Z_hi = min(A, Z_center + Z_offset_max)
        for Z in range(Z_lo, Z_hi + 1):
            if Z < neutron_drip_line(A):
                continue
            if Z > proton_drip_line(A):
                continue
            mesh.append((A, Z, 1.0))
    return mesh


def polygon_area(vertices: List[Tuple[float, float]]) -> float:
    """Shoelace formula for area of a polygon."""
    n = len(vertices)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * abs(s)


def polygon_contains(vertices: List[Tuple[float, float]],
                     p: Tuple[float, float]) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    x, y = p
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi):
            inside = not inside
        j = i
    return inside


def image_to_mesh2d(image: np.ndarray, threshold: float = 0.5
                    ) -> List[Tuple[int, int]]:
    """
    Convert a 2D binary image to a list of (i, j) pixel coordinates where
    image > threshold.  Adapted from 580_image_mesh2d.m.
    """
    coords = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > threshold:
                coords.append((i, j))
    return coords


def display_polygonal_boundary(vertices: List[Tuple[float, float]]) -> str:
    """
    Return a string representation of the polygonal boundary in the (A, Z) plane.
    Adapted from 891_polygonal_surface_display.m.
    """
    lines = []
    lines.append(f"Polygon with {len(vertices)} vertices:")
    for i, v in enumerate(vertices):
        lines.append(f"  V{i}: A = {v[0]:.2f}, Z = {v[1]:.2f}")
    lines.append(f"Area = {polygon_area(vertices):.4f}")
    return "\n".join(lines)


def nuclear_chart_boundary(A_min: int, A_max: int) -> List[Tuple[float, float]]:
    """
    Return the polygonal boundary of the nuclear chart (A, Z) region
    encompassing stability, neutron and proton drip lines.
    """
    verts = []
    # Bottom: neutron drip line
    for A in range(A_min, A_max + 1, 5):
        verts.append((float(A), float(neutron_drip_line(A))))
    # Right: proton drip at A_max
    verts.append((float(A_max), float(proton_drip_line(A_max))))
    # Top: proton drip line (going back)
    for A in range(A_max, A_min - 1, -5):
        verts.append((float(A), float(proton_drip_line(A))))
    verts.append((float(A_min), float(neutron_drip_line(A_min))))
    return verts


__all__ = [
    "beta_stability_line", "neutron_drip_line", "proton_drip_line",
    "build_nuclide_mesh", "polygon_area", "polygon_contains",
    "image_to_mesh2d", "display_polygonal_boundary",
    "nuclear_chart_boundary",
]
