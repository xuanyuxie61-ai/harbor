
import numpy as np
from typing import Tuple


def sylvester_matrix(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = len(p) - 1
    n = len(q) - 1
    if m < 0 or n < 0:
        raise ValueError("Polynomial degrees must be non-negative")
    
    S = np.zeros((m + n, m + n))

    for i in range(n):
        S[i, i:i + m + 1] = p

    for i in range(m):
        S[n + i, i:i + n + 1] = q
    return S


def polynomial_resultant_sylvester(p: np.ndarray, q: np.ndarray) -> float:
    S = sylvester_matrix(p, q)
    return float(np.linalg.det(S))


def polynomial_resultant_roots(p: np.ndarray, q: np.ndarray) -> float:
    m = len(p) - 1
    n = len(q) - 1
    
    if m == 0 or n == 0:

        return p[0] ** n * q[0] ** m if len(p) > 0 and len(q) > 0 else 0.0
    
    roots_p = np.roots(p)
    roots_q = np.roots(q)
    
    lead_p = p[0]
    lead_q = q[0]
    
    prod = 1.0
    for alpha in roots_p:
        for beta in roots_q:
            prod *= (alpha - beta)
    
    resultant = (lead_p ** n) * (lead_q ** m) * prod
    return float(np.real(resultant))


def find_critical_points_polynomial(potential_poly: np.ndarray) -> np.ndarray:

    n = len(potential_poly) - 1
    if n < 1:
        return np.array([])
    
    deriv = np.array([potential_poly[i] * (n - i) for i in range(n)])
    roots = np.roots(deriv)

    real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-8])
    return np.sort(real_roots)


def detect_bifurcation_points(poly1: np.ndarray, poly2: np.ndarray,
                               x_range: Tuple[float, float] = (-2.0, 2.0)) -> np.ndarray:
    max_len = max(len(poly1), len(poly2))
    p1 = np.zeros(max_len)
    p2 = np.zeros(max_len)
    p1[max_len - len(poly1):] = poly1
    p2[max_len - len(poly2):] = poly2
    
    diff = p1 - p2

    while len(diff) > 1 and abs(diff[0]) < 1e-14:
        diff = diff[1:]
    
    roots = np.roots(diff)
    real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-8])

    intersections = real_roots[(real_roots >= x_range[0]) & (real_roots <= x_range[1])]
    return np.sort(intersections)


def construct_dihedral_potential_polynomial(coeffs: np.ndarray) -> np.ndarray:


    poly = coeffs[::-1].copy()
    return poly


def analyze_potential_landscape_criticality(coeffs: np.ndarray) -> dict:
    cp = find_critical_points_polynomial(coeffs)
    
    n = len(coeffs) - 1

    d1 = np.array([coeffs[i] * (n - i) for i in range(n)])

    d2 = np.array([d1[i] * (n - 1 - i) for i in range(n - 1)])
    
    types = []
    energies = []
    for x in cp:
        v = np.polyval(coeffs, x)
        v2 = np.polyval(d2, x)
        energies.append(v)
        if v2 > 1e-6:
            types.append("minimum")
        elif v2 < -1e-6:
            types.append("maximum")
        else:
            types.append("degenerate")
    

    barriers = []
    minima_indices = [i for i, t in enumerate(types) if t == "minimum"]
    for i in range(len(minima_indices) - 1):
        idx1 = minima_indices[i]
        idx2 = minima_indices[i + 1]

        max_energy = -np.inf
        for j in range(idx1 + 1, idx2):
            if types[j] == "maximum" and energies[j] > max_energy:
                max_energy = energies[j]
        if max_energy > -np.inf:
            barriers.append(max_energy - energies[idx1])
    
    return {
        "critical_points": cp,
        "types": types,
        "energies": energies,
        "barrier_heights": barriers,
    }
