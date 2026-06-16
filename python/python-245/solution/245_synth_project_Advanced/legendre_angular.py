"""
legendre_angular.py
====================
Shifted Legendre polynomial expansion of the angular distribution
of fission fragments.

Maps from: 666_legendre_shifted_polynomial (shifted Legendre polynomials
           on [0,1] via three-term recurrence)

Physical context
----------------
The angular distribution of fission fragments relative to the beam axis
is characterised by anisotropy:

  W(theta) = sum_l a_l * P_l(cos(theta))

where P_l are Legendre polynomials and a_l are expansion coefficients.

For the transition-state model:
  W(theta) = sum_K (2I+1) * |d^I_{MK}(theta)|^2 * T_K / sum_K T_K

where K is the projection of angular momentum on the symmetry axis,
d^I_{MK} are Wigner rotation matrix elements, and T_K are transmission
coefficients.

The anisotropy is:
  A = W(0) / W(pi/2)

We expand W(cos(theta)) in shifted Legendre polynomials P_l^{01}(x)
on [0, 1] (from the 666_legendre_shifted_polynomial project), where
x = (cos(theta) + 1) / 2 maps [-1, 1] to [0, 1].

The three-term recurrence is:
  P01(0, x) = 1
  P01(1, x) = 2x - 1
  P01(n, x) = ((2n-1)(2x-1)*P01(n-1,x) - (n-1)*P01(n-2,x)) / n
"""

import math
from typing import List, Tuple, Dict, Optional


def shifted_legendre_value(n: int, x: float) -> float:
    """
    Evaluate shifted Legendre polynomial P01(n, x) at point x in [0, 1].

    Uses the three-term recurrence from 666_legendre_shifted_polynomial:
      P01(0, x) = 1
      P01(1, x) = 2x - 1
      P01(n, x) = ((2n-1)(2x-1)*P01(n-1,x) - (n-1)*P01(n-2,x)) / n

    Also known as Legendre polynomials on [0, 1] instead of [-1, 1].
    """
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x - 1.0

    p_prev2 = 1.0       # P01(0, x)
    p_prev1 = 2.0 * x - 1.0  # P01(1, x)

    for k in range(2, n + 1):
        p_curr = ((2.0 * k - 1.0) * (2.0 * x - 1.0) * p_prev1
                  - (k - 1.0) * p_prev2) / k
        p_prev2 = p_prev1
        p_prev1 = p_curr

    return p_prev1


def shifted_legendre_array(max_n: int, x: float) -> List[float]:
    """
    Evaluate all shifted Legendre polynomials P01(0..max_n, x).
    """
    result = [0.0] * (max_n + 1)
    result[0] = 1.0
    if max_n >= 1:
        result[1] = 2.0 * x - 1.0

    for k in range(2, max_n + 1):
        result[k] = ((2.0 * k - 1.0) * (2.0 * x - 1.0) * result[k - 1]
                     - (k - 1.0) * result[k - 2]) / k

    return result


def standard_legendre_value(n: int, x: float) -> float:
    """
    Standard Legendre polynomial P_n(x) on [-1, 1].

    Recurrence: (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x

    p_prev2 = 1.0
    p_prev1 = x

    for k in range(1, n):
        p_curr = ((2.0 * k + 1.0) * x * p_prev1 - k * p_prev2) / (k + 1.0)
        p_prev2 = p_prev1
        p_prev1 = p_curr

    return p_prev1


class FissionAngularDistribution:
    """
    Angular distribution of fission fragments using Legendre expansion.

    W(theta) = sum_{l=0}^{L_max} a_l * P_l(cos(theta))

    The coefficients a_l are determined by the spin distribution
    of the compound nucleus and the K-distribution at the saddle point.
    """

    def __init__(self, spin_i: float = 0.0, k_variance: float = 6.0,
                 max_legendre: int = 8):
        """
        Parameters
        ----------
        spin_i : float
            Compound nucleus spin (in units of hbar).
        k_variance : float
            Variance of K-distribution at saddle point: K_0^2.
        max_legendre : int
            Maximum Legendre order for expansion.
        """
        self.I = spin_i
        self.K0_sq = k_variance
        self.L_max = max_legendre
        self.coefficients: List[float] = []

    def compute_coefficients(self) -> List[float]:
        """
        Compute Legendre expansion coefficients for the angular distribution.

        For the transition-state model with spin I and K-distribution:
          a_l = (2l+1) * [d^l_{00}(pi/2)]^2 * exp(-l*(l+1)/(4*K0^2))

        Simplified formula (Bohr-Mottel):
          a_l/a_0 = (-1)^{l/2} * (2l+1)!! / (2^l * l!) * Pi_l
        where Pi_l involves K0^2.

        For this implementation:
          a_l = (2l+1) * exp(-l*(l+1)/(4*K0^2)) * F(I, l)
        """
        coeffs = []
        for l in range(0, self.L_max + 1):
            if l % 2 == 1 and self.I < 0.5:
                coeffs.append(0.0)  # odd terms vanish for I=0
                continue

            # K-distribution damping
            if self.K0_sq > 0:
                k_factor = math.exp(-l * (l + 1.0) / (4.0 * self.K0_sq))
            else:
                k_factor = 1.0 if l == 0 else 0.0

            # Spin factor (simplified)
            if self.I < 0.5:
                spin_factor = 1.0
            else:
                spin_factor = 1.0 + 0.1 * l * self.I / (self.I + 1.0)

            coeff = (2.0 * l + 1.0) * k_factor * spin_factor
            coeffs.append(coeff)

        # Normalise so a_0 = 1
        if len(coeffs) > 0 and abs(coeffs[0]) > 1e-30:
            a0 = coeffs[0]
            coeffs = [c / a0 for c in coeffs]

        self.coefficients = coeffs
        return coeffs

    def evaluate_w(self, theta: float) -> float:
        """
        Evaluate W(theta) = sum a_l * P_l(cos(theta)).
        """
        if not self.coefficients:
            self.compute_coefficients()

        cos_theta = math.cos(theta)
        w = 0.0
        for l, a_l in enumerate(self.coefficients):
            p_l = standard_legendre_value(l, cos_theta)
            w += a_l * p_l

        return w

    def evaluate_shifted(self, theta: float) -> float:
        """
        Evaluate using shifted Legendre polynomials on [0, 1].

        x = (cos(theta) + 1) / 2 maps [-1, 1] -> [0, 1]
        W(theta) = sum b_l * P01(l, x)
        """
        if not self.coefficients:
            self.compute_coefficients()

        cos_theta = math.cos(theta)
        x_shifted = 0.5 * (cos_theta + 1.0)  # in [0, 1]

        w = 0.0
        for l, b_l in enumerate(self.coefficients):
            p_l = shifted_legendre_value(l, x_shifted)
            w += b_l * p_l

        return w

    def anisotropy(self) -> float:
        """
        Compute angular anisotropy A = W(0) / W(pi/2).
        """
        w_0 = self.evaluate_w(0.0)
        w_90 = self.evaluate_w(math.pi / 2.0)

        if abs(w_90) < 1e-30:
            return float('inf')
        return w_0 / w_90

    def angular_distribution_table(self, n_angles: int = 19) -> List[Dict[str, float]]:
        """
        Tabulate W(theta) at evenly spaced angles.
        """
        results = []
        for i in range(n_angles):
            theta_deg = i * 180.0 / (n_angles - 1)
            theta_rad = theta_deg * math.pi / 180.0

            w_std = self.evaluate_w(theta_rad)
            w_shifted = self.evaluate_shifted(theta_rad)

            results.append({
                'theta_deg': theta_deg,
                'theta_rad': theta_rad,
                'W_standard': w_std,
                'W_shifted': w_shifted,
                'cos_theta': math.cos(theta_rad),
            })

        return results


def convergence_test_shifted_legendre(n_max: int = 12,
                                       x_test: float = 0.5) -> List[Dict]:
    """
    Test convergence of shifted Legendre polynomial evaluation.
    """
    results = []
    for n in range(n_max + 1):
        val = shifted_legendre_value(n, x_test)
        # Exact value at x=0.5: P01(n, 0.5) = P_n(0) (standard Legendre at 0)
        val_exact = standard_legendre_value(n, 0.0)
        results.append({
            'n': n,
            'shifted_value': val,
            'standard_at_0': val_exact,
            'difference': abs(val - val_exact),
        })
    return results
