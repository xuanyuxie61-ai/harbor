
import numpy as np


def circle_segment_area_from_angle(r: float, theta: float) -> float:
    if r <= 0.0:
        raise ValueError("circle_segment_area_from_angle: r must be > 0")
    if theta < 0.0 or theta > 2.0 * np.pi:
        raise ValueError("circle_segment_area_from_angle: theta must be in [0, 2*pi]")
    area = r * r * (theta - np.sin(theta)) / 2.0
    return float(area)


def circle_segment_height_from_angle(r: float, theta: float) -> float:
    if r <= 0.0:
        raise ValueError("circle_segment_height_from_angle: r must be > 0")
    if theta < 0.0 or theta > 2.0 * np.pi:
        raise ValueError("circle_segment_height_from_angle: theta must be in [0, 2*pi]")
    h = r * (1.0 - np.cos(theta / 2.0))
    return float(h)


def circle_segment_centroid_from_angle(r: float, theta: float) -> tuple:
    if r <= 0.0:
        raise ValueError("circle_segment_centroid_from_angle: r must be > 0")
    if theta <= 0.0 or theta > 2.0 * np.pi:
        raise ValueError("circle_segment_centroid_from_angle: theta must be in (0, 2*pi]")
    num = 4.0 * r * (np.sin(theta / 2.0) ** 3)
    den = 3.0 * (theta - np.sin(theta))
    if abs(den) < 1e-15:
        return (0.0, 0.0)
    d = num / den
    return (float(d), 0.0)


def hexagonal_lattice_points(pitch: float, n_rings: int) -> np.ndarray:
    if pitch <= 0.0:
        raise ValueError("hexagonal_lattice_points: pitch must be > 0")
    if n_rings < 1:
        raise ValueError("hexagonal_lattice_points: n_rings must be >= 1")
    points = []
    a1 = np.array([pitch, 0.0])
    a2 = np.array([pitch * 0.5, pitch * np.sqrt(3.0) / 2.0])
    for i in range(-n_rings, n_rings + 1):
        for j in range(-n_rings, n_rings + 1):
            if abs(i) + abs(j) + abs(-i - j) <= 2 * n_rings:
                p = i * a1 + j * a2
                points.append(p)
    pts = np.array(points)

    pts = np.unique(np.round(pts, 12), axis=0)
    return pts


def pcf_air_holes_geometry(pitch: float, n_rings: int, hole_radius: float) -> dict:
    if hole_radius >= pitch / 2.0:
        raise ValueError("pcf_air_holes_geometry: hole_radius must be < pitch/2")
    points = hexagonal_lattice_points(pitch, n_rings)
    n_holes = points.shape[0]
    unit_cell_area = np.sqrt(3.0) / 2.0 * pitch * pitch
    hole_area = np.pi * hole_radius * hole_radius
    filling_fraction = n_holes * hole_area / (unit_cell_area * n_holes) if n_holes > 0 else 0.0

    r_core = pitch - hole_radius

    silica_fraction = 1.0 - filling_fraction
    return {
        "n_holes": int(n_holes),
        "pitch": float(pitch),
        "hole_radius": float(hole_radius),
        "filling_fraction": float(filling_fraction),
        "silica_fraction": float(silica_fraction),
        "r_core": float(r_core),
        "hole_centers": points,
    }


def triangle_grid(n: int, t: np.ndarray) -> np.ndarray:
    if n < 1:
        raise ValueError("triangle_grid: n must be >= 1")
    if t.shape != (2, 3):
        raise ValueError("triangle_grid: t must have shape (2, 3)")
    ng = ((n + 1) * (n + 2)) // 2
    tg = np.zeros((2, ng))
    p = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            tg[:, p] = (i * t[:, 0] + j * t[:, 1] + k * t[:, 2]) / n
            p += 1
    return tg


def pcf_transverse_triangle_grid(pitch: float, n_rings: int, subdivisions: int = 8) -> np.ndarray:
    t = np.array([
        [0.0, pitch, pitch * 0.5],
        [0.0, 0.0, pitch * np.sqrt(3.0) / 2.0]
    ])
    return triangle_grid(subdivisions, t)


def effective_mode_area(pitch: float, hole_radius: float, n_rings: int) -> float:
    geo = pcf_air_holes_geometry(pitch, n_rings, hole_radius)
    f = geo["filling_fraction"]
    r_core = geo["r_core"]
    c1 = 2.5
    c2 = 1.8
    a_eff = np.pi * r_core * r_core * (1.0 - c1 * f + c2 * f * f)
    return float(max(a_eff, 1e-15))


def nonlinear_coefficient(pitch: float, hole_radius: float, n_rings: int,
                          n2: float = 2.6e-20, wavelength: float = 1.55e-6) -> float:
    if wavelength <= 0.0:
        raise ValueError("nonlinear_coefficient: wavelength must be > 0")
    a_eff = effective_mode_area(pitch, hole_radius, n_rings)

    gamma = (2.0 * np.pi / wavelength) * (n2 / a_eff)
    return float(gamma)
