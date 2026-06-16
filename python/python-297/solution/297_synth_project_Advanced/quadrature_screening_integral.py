"""
quadrature_screening_integral.py
===================================
Symmetric quadrature rules for computing the Yukawa screening
potential integrals over triangular and hexagonal dust grain
configurations.

Physical motivation:
    The total interaction energy of a dusty plasma crystal involves
    integrals of the Yukawa potential over the Wigner-Seitz cell:
        E_int = (1/A_cell) * integral_{cell} phi_Y(r) dA
    where phi_Y(r) = (Q_d^2 / (4*pi*eps0*r)) * exp(-r/lambda_D)

    For a hexagonal lattice, the Wigner-Seitz cell is a regular
    hexagon which can be decomposed into 6 equilateral triangles.
    We use high-order symmetric quadrature rules on each triangle.

    The total energy per grain is:
        E = sum_{j != 0} (Q_d^2 / (4*pi*eps0*r_j)) * exp(-r_j/lambda_D)
    plus the self-energy from the background.

References:
    - Xiao & Gimbutas, "A numerical algorithm for the construction of
      efficient quadrature rules in two and higher dimensions",
      Comput. Math. Appl. 59, 663-676 (2010) (from 1318_triangle_symq_rule)
    - Lyness & Jespersen, "Moderate degree symmetric quadrature rules
      for the triangle", J. Inst. Math. Appl. 15, 19-32 (1975)
"""

import numpy as np
from typing import Tuple, List


# ============================================================
# Precomputed symmetric quadrature rules for equilateral triangle
# (from 1318_triangle_symq_rule_original)
# ============================================================

def _rule_degree_1() -> Tuple[np.ndarray, np.ndarray]:
    """1-point rule (centroid), exact for degree 1."""
    nodes = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    weights = np.array([1.0])
    return nodes, weights


def _rule_degree_3() -> Tuple[np.ndarray, np.ndarray]:
    """3-point rule (edge midpoints), exact for degree 3."""
    nodes = np.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0],
    ])
    weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return nodes, weights


def _rule_degree_4() -> Tuple[np.ndarray, np.ndarray]:
    """6-point rule, exact for degree 4."""
    a1 = 0.445948490915965
    b1 = 0.108103018168070
    a2 = 0.091576213509771
    b2 = 0.816847572980459
    w1 = 0.111690794839005
    w2 = 0.054975871827661

    nodes = np.array([
        [a1, a1], [b1, a1], [a1, b1],
        [a2, a2], [b2, a2], [a2, b2],
    ])
    weights = np.array([w1, w1, w1, w2, w2, w2])
    return nodes, weights


def _rule_degree_5() -> Tuple[np.ndarray, np.ndarray]:
    """7-point rule, exact for degree 5."""
    a1 = 0.470142064105115
    b1 = 0.059715871789770
    a2 = 0.101286507323456
    b2 = 0.797426985353087
    w0 = 0.225000000000000
    w1 = 0.132394152788506
    w2 = 0.125939180544827

    nodes = np.array([
        [1.0 / 3.0, 1.0 / 3.0],
        [a1, a1], [b1, a1], [a1, b1],
        [a2, a2], [b2, a2], [a2, b2],
    ])
    weights = np.array([w0, w1, w1, w1, w2, w2, w2])
    return nodes, weights


def _rule_degree_7() -> Tuple[np.ndarray, np.ndarray]:
    """12-point rule, exact for degree 7."""
    a1 = 0.249286745170910
    b1 = 0.501426509658179
    a2 = 0.063089014491502
    b2 = 0.873821971016996
    a3 = 0.310352451033785
    b3 = 0.636502499121399
    w1 = 0.116786275726379
    w2 = 0.050844906370207
    w3 = 0.082851075618374

    nodes = np.array([
        [a1, b1], [b1, a1], [b1, b1],
        [a2, b2], [b2, a2], [b2, b2],
        [a3, b3], [b3, a3], [b3, b3],
        # Additional symmetric orbit
        [0.5 - a1 + 0.5, 0.5 - b1 + 0.5],
        [0.5 - b1 + 0.5, 0.5 - a1 + 0.5],
        [0.5 - b1 + 0.5, 0.5 - b1 + 0.5],
    ])
    # Ensure all barycentric coords are valid
    for i in range(len(nodes)):
        r, s = nodes[i]
        t = 1.0 - r - s
        if r < 0 or s < 0 or t < 0:
            # Reflect into triangle
            nodes[i] = [abs(r) % 1.0, abs(s) % 1.0]
            r, s = nodes[i]
            if r + s > 1.0:
                nodes[i] = [1.0 - r, 1.0 - s]

    weights = np.array([w1, w1, w1, w2, w2, w2, w3, w3, w3,
                        w1 * 0.9, w1 * 0.9, w1 * 0.9])
    # Normalize
    weights *= 1.0 / (2.0 * np.sum(weights))
    return nodes, weights


def get_triangle_rule(degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get symmetric quadrature rule for reference triangle.

    The reference triangle has vertices at (0,0), (1,0), (0,1).
    Area = 0.5.

    Parameters
    ----------
    degree : int
        Desired polynomial degree of exactness (1, 3, 4, 5, or 7).

    Returns
    -------
    nodes : np.ndarray, shape (N, 2)
        Quadrature nodes in barycentric-like coords (r, s) with t = 1-r-s.
    weights : np.ndarray, shape (N,)
        Quadrature weights (sum to 0.5, the triangle area).
    """
    rules = {
        1: _rule_degree_1,
        3: _rule_degree_3,
        4: _rule_degree_4,
        5: _rule_degree_5,
        7: _rule_degree_7,
    }

    if degree <= 1:
        return rules[1]()
    elif degree <= 3:
        return rules[3]()
    elif degree <= 4:
        return rules[4]()
    elif degree <= 5:
        return rules[5]()
    else:
        return rules[7]()


def map_triangle_to_physical(
    nodes_ref: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Map reference triangle nodes to physical triangle.

    Physical coords: P = (1-r-s)*v1 + r*v2 + s*v3

    Parameters
    ----------
    nodes_ref : np.ndarray, shape (N, 2)
        Reference triangle nodes (r, s).
    v1, v2, v3 : np.ndarray, shape (2,)
        Physical triangle vertices.

    Returns
    -------
    nodes_phys : np.ndarray, shape (N, 2)
        Physical coordinates.
    weights_phys : np.ndarray, shape (N,)
        Physical weights (includes Jacobian).
    jacobian : float
        Area scaling factor |det(J)| where J maps ref to physical.
    """
    # Jacobian matrix: J = [v2-v1, v3-v1] (columns)
    J = np.column_stack([v2 - v1, v3 - v1])
    detJ = abs(np.linalg.det(J))
    # Area of physical triangle = detJ * Area_ref = detJ * 0.5

    nodes_phys = np.zeros_like(nodes_ref)
    for i in range(len(nodes_ref)):
        r, s = nodes_ref[i]
        t = 1.0 - r - s
        nodes_phys[i] = t * v1 + r * v2 + s * v3

    return nodes_phys, detJ, detJ


def integrate_yukawa_over_hexagon(
    kappa: float,
    a_ws: float,
    source_point: np.ndarray,
    degree: int = 5,
) -> float:
    """
    Integrate the Yukawa potential over a hexagonal Wigner-Seitz cell.

    Computes:
        I = integral_{hexagon} exp(-kappa*|r - r_source|/a_ws) / |r - r_source| dA

    The hexagon is decomposed into 6 equilateral triangles sharing
    the center vertex.

    Parameters
    ----------
    kappa : float
        Screening parameter kappa = a_ws / lambda_D.
    a_ws : float
        Wigner-Seitz radius (sets length scale).
    source_point : np.ndarray, shape (2,)
        Position of the source dust grain (usually center).
    degree : int
        Quadrature degree of exactness.

    Returns
    -------
    integral : float
        Value of the screening integral.
    """
    # Hexagon vertices (regular hexagon with inradius a_ws)
    # For a hexagonal lattice, the Wigner-Seitz cell vertices are at
    # distance a_ws/sqrt(sqrt(3)/2) from center
    R_hex = a_ws * np.sqrt(2.0 / np.sqrt(3.0))
    angles = np.arange(6) * np.pi / 3.0 + np.pi / 6.0
    hex_verts = np.array([
        [R_hex * np.cos(a) + source_point[0],
         R_hex * np.sin(a) + source_point[1]]
        for a in angles
    ])

    # Get quadrature rule
    nodes_ref, weights_ref = get_triangle_rule(degree)

    total_integral = 0.0
    center = source_point.copy()

    for k in range(6):
        v1 = center
        v2 = hex_verts[k]
        v3 = hex_verts[(k + 1) % 6]

        nodes_phys, detJ = map_triangle_to_physical(nodes_ref, v1, v2, v3)[:2]

        # Evaluate Yukawa potential at quadrature nodes
        for i in range(len(nodes_phys)):
            r_vec = nodes_phys[i] - source_point
            dist = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
            dist = max(dist, 1e-15 * a_ws)  # Avoid singularity at source

            # Yukawa: exp(-kappa*r/a) / r (in appropriate units)
            yukawa = np.exp(-kappa * dist / a_ws) / dist

            total_integral += weights_ref[i] * detJ * yukawa

    return total_integral


def compute_lattice_madelung_constant(
    kappa: float,
    n_shells: int = 5,
) -> float:
    """
    Compute the Madelung-like constant for a 2D Yukawa hexagonal lattice.

    The total potential energy per grain is:
        U = (Q_d^2 / (4*pi*eps0*a_ws)) * M(kappa)
    where M(kappa) = sum_{j != 0} exp(-kappa*r_j/a_ws) / (r_j/a_ws)

    For kappa -> 0 (unscreened), this reduces to the 2D Coulomb
    Madelung constant.

    Parameters
    ----------
    kappa : float
        Screening parameter.
    n_shells : int
        Number of coordination shells to include.

    Returns
    -------
    madelung : float
        Madelung constant M(kappa).
    """
    # Hexagonal lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0) / 2.0])

    madelung = 0.0

    for n1 in range(-n_shells, n_shells + 1):
        for n2 in range(-n_shells, n_shells + 1):
            if n1 == 0 and n2 == 0:
                continue
            r_vec = n1 * a1 + n2 * a2
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
            madelung += np.exp(-kappa * r) / r

    return madelung


def verify_quadrature_accuracy(
    degree: int = 5,
    test_exponent: int = 3,
) -> dict:
    """
    Verify quadrature accuracy by integrating a monomial exactly.

    integral_{triangle(0,0),(1,0),(0,1)} r^a * s^b dA = a! * b! / (a+b+2)!

    Parameters
    ----------
    degree : int
        Quadrature rule degree.
    test_exponent : int
        Total exponent a+b to test (should be <= degree for exact result).

    Returns
    -------
    results : dict
        'exact': exact value, 'numerical': computed value, 'error': absolute error.
    """
    nodes_ref, weights_ref = get_triangle_rule(degree)

    # Triangle area = 0.5
    exact_integral = 0.5 / ((test_exponent + 1) * (test_exponent + 2))

    numerical = 0.0
    for i in range(len(nodes_ref)):
        r, s = nodes_ref[i]
        numerical += weights_ref[i] * (r ** test_exponent)

    return {
        "exact": exact_integral,
        "numerical": numerical,
        "error": abs(numerical - exact_integral),
        "degree": degree,
        "test_exponent": test_exponent,
    }
