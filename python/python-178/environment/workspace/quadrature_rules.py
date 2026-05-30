
import numpy as np
from typing import Tuple, Callable
from integer_utils import i4_factorial






TETRAHEDRON_QUADRATURE_RULES = {
    1: {
        'points': np.array([[0.25, 0.25, 0.25]], dtype=np.float64),
        'weights': np.array([1.0 / 6.0], dtype=np.float64),
    },
    2: {
        'points': np.array([
            [0.58541020, 0.13819660, 0.13819660],
            [0.13819660, 0.58541020, 0.13819660],
            [0.13819660, 0.13819660, 0.58541020],
            [0.13819660, 0.13819660, 0.13819660],
        ], dtype=np.float64),
        'weights': np.array([1.0 / 24.0] * 4, dtype=np.float64),
    },
    3: {
        'points': np.array([
            [0.25, 0.25, 0.25],
            [1.0/6.0, 1.0/6.0, 1.0/6.0],
            [1.0/2.0, 1.0/6.0, 1.0/6.0],
            [1.0/6.0, 1.0/2.0, 1.0/6.0],
            [1.0/6.0, 1.0/6.0, 1.0/2.0],
        ], dtype=np.float64),
        'weights': np.array([
            -2.0/15.0,
            3.0/40.0,
            3.0/40.0,
            3.0/40.0,
            3.0/40.0,
        ], dtype=np.float64),
    },
    4: {

        'points': np.array([
            [0.25, 0.25, 0.25],
            [0.071428571428571, 0.071428571428571, 0.071428571428571],
            [0.785714285714286, 0.071428571428571, 0.071428571428571],
            [0.071428571428571, 0.785714285714286, 0.071428571428571],
            [0.071428571428571, 0.071428571428571, 0.785714285714286],
            [0.399403576166799, 0.399403576166799, 0.100596423833201],
            [0.399403576166799, 0.100596423833201, 0.399403576166799],
            [0.100596423833201, 0.399403576166799, 0.399403576166799],
            [0.399403576166799, 0.100596423833201, 0.100596423833201],
            [0.100596423833201, 0.399403576166799, 0.100596423833201],
            [0.100596423833201, 0.100596423833201, 0.399403576166799],
        ], dtype=np.float64),
        'weights': np.array([
            -0.013155555555556,
            0.007622222222222,
            0.007622222222222,
            0.007622222222222,
            0.007622222222222,
            0.024888888888889,
            0.024888888888889,
            0.024888888888889,
            0.024888888888889,
            0.024888888888889,
            0.024888888888889,
        ], dtype=np.float64) * (1.0/6.0),
    },
}


def integrate_tetrahedron_deterministic(f: Callable,
                                        physical_vertices: np.ndarray,
                                        order: int = 3) -> float:
    from tetrahedron_geometry import jacobian_tet4
    if order not in TETRAHEDRON_QUADRATURE_RULES:
        raise ValueError(f"Quadrature order {order} not available.")
    rule = TETRAHEDRON_QUADRATURE_RULES[order]
    pts = rule['points']
    wts = rule['weights']
    J = jacobian_tet4(physical_vertices)
    detJ = abs(np.linalg.det(J))
    if detJ < 1e-30:
        raise ValueError("Degenerate tetrahedron in quadrature.")
    result = 0.0
    for i in range(len(wts)):
        xi, eta, zeta = pts[i]

        x = (physical_vertices[0] +
             (physical_vertices[1] - physical_vertices[0]) * xi +
             (physical_vertices[2] - physical_vertices[0]) * eta +
             (physical_vertices[3] - physical_vertices[0]) * zeta)
        result += wts[i] * f(x[0], x[1], x[2])
    return result * detJ






def sample_unit_simplex(m: int, n_samples: int) -> np.ndarray:
    if m < 0:
        raise ValueError("Dimension must be non-negative.")
    E = -np.log(np.random.rand(n_samples, m + 1))
    s = E.sum(axis=1, keepdims=True)
    s[s < 1e-30] = 1.0
    return E / s


def sample_tetrahedron_uniform(physical_vertices: np.ndarray,
                               n_samples: int) -> np.ndarray:

    bary = sample_unit_simplex(3, n_samples)
    pts = np.zeros((n_samples, 3), dtype=np.float64)
    for i in range(4):
        pts += bary[:, i:i+1] * physical_vertices[i:i+1, :]
    return pts


def integrate_tetrahedron_monte_carlo(f: Callable,
                                      physical_vertices: np.ndarray,
                                      n_samples: int = 10000) -> Tuple[float, float]:
    from tetrahedron_geometry import tetrahedron_volume
    vol = tetrahedron_volume(physical_vertices)
    pts = sample_tetrahedron_uniform(physical_vertices, n_samples)
    vals = np.array([f(p[0], p[1], p[2]) for p in pts], dtype=np.float64)
    mean = vol * np.mean(vals)
    stderr = vol * np.std(vals, ddof=1) / np.sqrt(n_samples) if n_samples > 1 else 0.0
    return mean, stderr


def exact_monomial_integral_tetrahedron(exponents: Tuple[int, int, int]) -> float:
    a, b, c = exponents
    num = i4_factorial(a) * i4_factorial(b) * i4_factorial(c)
    den = i4_factorial(a + b + c + 3)
    return float(num) / float(den)






def sphere01_triangle_angles_to_area(a: float, b: float, c: float) -> float:
    return a + b + c - np.pi


def sphere01_triangle_vertices_to_sides(v1: np.ndarray, v2: np.ndarray,
                                        v3: np.ndarray) -> Tuple[float, float, float]:
    def arc(u, w):
        dot = np.clip(np.dot(u, w), -1.0, 1.0)
        return np.arccos(dot)
    return arc(v1, v2), arc(v2, v3), arc(v3, v1)


def sphere01_triangle_sides_to_angles(a: float, b: float, c: float) -> Tuple[float, float, float]:
    s = 0.5 * (a + b + c)

    if s >= np.pi or a < 0 or b < 0 or c < 0:
        raise ValueError("Invalid spherical triangle sides.")

    tan_A2 = np.sqrt(np.sin(s - b) * np.sin(s - c) /
                     (np.sin(s) * np.sin(s - a) + 1e-30))
    tan_B2 = np.sqrt(np.sin(s - a) * np.sin(s - c) /
                     (np.sin(s) * np.sin(s - b) + 1e-30))
    tan_C2 = np.sqrt(np.sin(s - a) * np.sin(s - b) /
                     (np.sin(s) * np.sin(s - c) + 1e-30))
    A = 2.0 * np.arctan(tan_A2)
    B = 2.0 * np.arctan(tan_B2)
    C = 2.0 * np.arctan(tan_C2)
    return A, B, C


def sample_spherical_triangle(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                              n_samples: int) -> np.ndarray:
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    v3 = np.asarray(v3, dtype=np.float64)

    v1 /= np.linalg.norm(v1) + 1e-30
    v2 /= np.linalg.norm(v2) + 1e-30
    v3 /= np.linalg.norm(v3) + 1e-30
    a, b, c = sphere01_triangle_vertices_to_sides(v1, v2, v3)
    A, B, C = sphere01_triangle_sides_to_angles(a, b, c)
    area = sphere01_triangle_angles_to_area(A, B, C)

    e1 = v1.copy()
    e2 = v2 - np.dot(v2, e1) * e1
    nrm = np.linalg.norm(e2)
    if nrm < 1e-14:

        pts = []
        for _ in range(n_samples):
            t = np.random.rand()
            u = np.random.rand()
            ang = 2 * np.pi * u
            pts.append(np.cos(ang) * v1 + np.sin(ang) * np.cross(v1, v3))
        return np.array(pts)
    e2 /= nrm
    e3 = np.cross(e1, e2)

    pts = np.zeros((n_samples, 3), dtype=np.float64)
    for i in range(n_samples):
        xi1 = np.random.rand()
        xi2 = np.random.rand()
        area_hat = xi1 * area

        s = 0.5 * (a + b + c)




        alpha = area_hat

        z = np.cos(alpha)

        phi = xi2 * 2.0 * np.pi

        sin_theta = np.sqrt(max(0.0, 1.0 - z * z))
        p = z * e1 + sin_theta * (np.cos(phi) * e2 + np.sin(phi) * e3)

        p /= np.linalg.norm(p) + 1e-30
        pts[i] = p
    return pts


def integrate_spherical_triangle_monte_carlo(f: Callable,
                                              v1: np.ndarray, v2: np.ndarray,
                                              v3: np.ndarray, n_samples: int = 5000) -> float:
    a, b, c = sphere01_triangle_vertices_to_sides(v1, v2, v3)
    A, B, C = sphere01_triangle_sides_to_angles(a, b, c)
    area = sphere01_triangle_angles_to_area(A, B, C)
    pts = sample_spherical_triangle(v1, v2, v3, n_samples)
    vals = np.array([f(p[0], p[1], p[2]) for p in pts], dtype=np.float64)
    return area * np.mean(vals)
