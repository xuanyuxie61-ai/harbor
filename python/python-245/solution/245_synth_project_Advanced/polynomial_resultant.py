"""
polynomial_resultant.py
========================
Polynomial resultant methods for locating critical points of the
fission potential energy surface.

Maps from: 896_polynomial_resultant (Sylvester matrix and roots-based
           resultant computation)

Physical context
----------------
Critical points of the PES V(c, h, alpha) satisfy:
  dV/dc = 0, dV/dh = 0, dV/dalpha = 0

Along a 1D cut (fixed h, alpha), the PES V(c) can be approximated
by a polynomial.  The critical points are roots of V'(c) = 0.

For two polynomial conditions (e.g., V'(c) = 0 AND V''(c) = 0),
the resultant vanishes iff the two polynomials share a common root
(i.e., a degenerate critical point / cusp).

The Sylvester resultant of two polynomials p1(x) of degree n1 and
p2(x) of degree n2 is det(S) where S is the (n1+n2) x (n1+n2)
Sylvester matrix formed by shifting coefficients.
"""

import math
from typing import List, Tuple, Dict, Optional
import numpy as np

from potential_energy_surface import FissionPES


def sylvester_matrix(coeffs1: List[float],
                     coeffs2: List[float]) -> np.ndarray:
    """
    Construct the Sylvester matrix for two polynomials.

    For p1 of degree n1 and p2 of degree n2, the Sylvester matrix is
    (n1+n2) x (n1+n2), where:
    - First n2 rows: shifted coefficients of p1
    - Last n1 rows: shifted coefficients of p2

    From 896_polynomial_resultant project: sylvester_matrix function.
    """
    n1 = len(coeffs1) - 1  # degree of p1
    n2 = len(coeffs2) - 1  # degree of p2

    if n1 < 0 or n2 < 0:
        return np.array([[]])

    size = n1 + n2
    S = np.zeros((size, size))

    # Fill in shifted coefficients of p1
    for i in range(n2):
        for j in range(n1 + 1):
            S[i, i + j] = coeffs1[j]

    # Fill in shifted coefficients of p2
    for i in range(n1):
        for j in range(n2 + 1):
            S[n2 + i, i + j] = coeffs2[j]

    return S


def resultant_sylvester(coeffs1: List[float],
                        coeffs2: List[float]) -> float:
    """
    Compute the resultant of two polynomials via Sylvester matrix determinant.

    Res(p1, p2) = det(S)

    The resultant is zero iff p1 and p2 share a common root.

    From 896_polynomial_resultant: polynomial_resultant_sylvester.
    """
    S = sylvester_matrix(coeffs1, coeffs2)
    if S.size == 0:
        return 0.0
    return float(np.linalg.det(S))


def resultant_roots(coeffs1: List[float],
                    coeffs2: List[float]) -> float:
    """
    Compute resultant via polynomial roots:
      Res(p1, p2) = a1^n2 * a2^n1 * prod_{i,j} (r1_i - r2_j)

    where r1_i, r2_j are roots of p1, p2 and a1, a2 are leading coefficients.

    From 896_polynomial_resultant: polynomial_resultant_roots.
    """
    if len(coeffs1) <= 1 or len(coeffs2) <= 1:
        return 0.0

    roots1 = np.roots(coeffs1)
    roots2 = np.roots(coeffs2)

    a1 = coeffs1[0]
    a2 = coeffs2[0]
    n1 = len(coeffs1) - 1
    n2 = len(coeffs2) - 1

    product = 1.0
    for r1 in roots1:
        for r2 in roots2:
            diff = r1 - r2
            product *= abs(diff)

    return float(abs(a1 ** n2 * a2 ** n1 * product))


def pes_polynomial_fit(pes: FissionPES, h: float, alpha: float,
                       c_range: Tuple[float, float] = (0.9, 2.5),
                       n_points: int = 20,
                       degree: int = 6) -> List[float]:
    """
    Fit the PES V(c, h, alpha) to a polynomial of given degree
    along the c-direction (fixed h, alpha).

    Uses least-squares polynomial fitting.

    Returns polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0].
    """
    c_min, c_max = c_range
    c_vals = np.linspace(c_min, c_max, n_points)
    v_vals = np.array([pes.total_potential(c, h, alpha) for c in c_vals])

    # Polynomial fit
    coeffs = np.polyfit(c_vals, v_vals, degree)
    return coeffs.tolist()


def derivative_polynomial(coeffs: List[float]) -> List[float]:
    """
    Compute derivative of polynomial.
    If p(x) = a_n*x^n + ... + a_1*x + a_0
    then p'(x) = n*a_n*x^{n-1} + ... + a_1
    """
    n = len(coeffs) - 1
    if n <= 0:
        return [0.0]

    deriv = []
    for i in range(n):
        power = n - i
        deriv.append(coeffs[i] * power)

    return deriv


def find_critical_points(pes: FissionPES, h: float, alpha: float,
                         c_range: Tuple[float, float] = (0.9, 2.5)) -> List[Dict[str, float]]:
    """
    Find critical points of PES along c-direction by:
    1. Fit V(c) to a polynomial
    2. Compute V'(c) polynomial
    3. Find roots of V'(c)
    4. Classify as max/min/saddle using V''(c)

    Uses resultant to check for degenerate critical points.
    """
    # Fit polynomial
    coeffs_v = pes_polynomial_fit(pes, h, alpha, c_range, 30, 8)

    # Derivative
    coeffs_dv = derivative_polynomial(coeffs_v)

    # Second derivative
    coeffs_d2v = derivative_polynomial(coeffs_dv)

    # Find roots of V'(c)
    roots_dv = np.roots(coeffs_dv)

    c_min, c_max = c_range
    critical_points = []

    for r in roots_dv:
        if abs(r.imag) > 1e-8:
            continue  # complex root
        c_val = float(r.real)
        if c_val < c_min or c_val > c_max:
            continue

        # Evaluate V and V'' at this point
        v_val = pes.total_potential(c_val, h, alpha)

        # V'' from polynomial
        d2v_val = np.polyval(coeffs_d2v, c_val)

        # Classification
        if d2v_val > 0.1:
            ctype = 'minimum'
        elif d2v_val < -0.1:
            ctype = 'maximum'
        else:
            ctype = 'inflection'

        critical_points.append({
            'c': c_val,
            'energy': v_val,
            'd2v': float(d2v_val),
            'type': ctype,
        })

    # Check for degeneracy using resultant of V' and V''
    if len(coeffs_dv) > 1 and len(coeffs_d2v) > 1:
        res_syl = resultant_sylvester(coeffs_dv, coeffs_d2v)
        res_rts = resultant_roots(coeffs_dv, coeffs_d2v)
        degenerate = abs(res_syl) < 1e-4 * max(1.0, abs(res_rts))
    else:
        degenerate = False
        res_syl = 0.0
        res_rts = 0.0

    return {
        'critical_points': sorted(critical_points, key=lambda x: x['c']),
        'resultant_sylvester': res_syl,
        'resultant_roots': res_rts,
        'is_degenerate': degenerate,
    }


def test_resultant_exactness() -> Dict[str, any]:
    """
    Test resultant computation with known polynomials.

    p1(x) = x^2 - 1  (roots: +1, -1)
    p2(x) = x^2 - 4  (roots: +2, -2)
    Res(p1, p2) = (1-2)(1+2)(-1-2)(-1+2) = (-1)(3)(-3)(1) = 9
    """
    p1 = [1.0, 0.0, -1.0]  # x^2 - 1
    p2 = [1.0, 0.0, -4.0]  # x^2 - 4

    res_syl = resultant_sylvester(p1, p2)
    res_rts = resultant_roots(p1, p2)
    expected = 9.0

    # Test with shared root:
    # p3(x) = x^2 - 1 (roots: +1, -1)
    # p4(x) = x - 1  (root: +1)
    # Res = 0 (common root at x=1)
    p3 = [1.0, 0.0, -1.0]
    p4 = [1.0, -1.0]
    res_shared = resultant_sylvester(p3, p4)

    return {
        'test_no_common_root': {
            'resultant_sylvester': res_syl,
            'resultant_roots': res_rts,
            'expected': expected,
            'error': abs(res_syl - expected),
        },
        'test_common_root': {
            'resultant_sylvester': res_shared,
            'expected_zero': True,
            'is_near_zero': abs(res_shared) < 1e-10,
        },
    }
