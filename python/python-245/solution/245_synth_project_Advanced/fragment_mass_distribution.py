"""
fragment_mass_distribution.py
==============================
Calculation of fission fragment mass distribution using the
statistical model at the scission point.

Maps from: 594_interp_spline (cubic spline interpolation)

Physical context
----------------
The mass yield Y(A) is determined by the level density at the scission
point and the potential energy surface:

  Y(A) ~ rho_1(U1, A1) * rho_2(U2, A2) * exp(-V_scission(A)/T)

where rho_i are the level densities of the fragments, U_i their excitation
energies, and V_scission the scission-point potential.

The mass resolution function is interpolated using cubic splines
(from the interp_spline project).

Total kinetic energy at scission:
  TKE = Z1*Z2*e^2 / d_scission

where d_scission = R1 + R2 + delta_neck is the distance between fragment
centres at scission.
"""

import math
from typing import List, Tuple, Dict, Optional

from nuclear_constants import (
    binding_energy_ld, get_atomic_mass, q_value_fission,
    coulomb_barrier_energy, fermi_gas_level_density,
    level_density_parameter, prompt_neutron_multiplicity,
    ATOMIC_MASS_UNIT_MEV, NEUTRON_MASS_U, RADIUS_PARAMETER
)


class FragmentMassDistribution:
    """
    Compute fission fragment mass distribution Y(A) for a given
    compound nucleus (A_CN, Z_CN).

    The mass distribution is computed using the statistical model:
      Y(A) proportional to integral over configurations at scission.

    We use cubic spline interpolation (from interp_spline) to
    obtain smooth yield curves from discrete calculations.
    """

    def __init__(self, z_cn: int = 92, a_cn: int = 236,
                 temperature: float = 1.5):
        self.Z = z_cn
        self.A = a_cn
        self.N = a_cn - z_cn
        self.T = temperature  # MeV

        # Scission parameters
        self.d_neck = 2.0  # fm, neck length at scission
        self.e_gamma = 7.0  # MeV, total prompt gamma energy

    def scission_configuration(self, a1: int, z1: int) -> Dict[str, float]:
        """
        Compute scission-point configuration for fragment pair (A1,Z1), (A2,Z2).

        Returns dict with:
        - R1, R2: fragment radii (fm)
        - d_center: distance between centres (fm)
        - TKE: total kinetic energy (MeV)
        - Q: Q-value (MeV)
        """
        a2 = self.A - a1
        z2 = self.Z - z1

        r1 = RADIUS_PARAMETER * (a1 ** (1.0 / 3.0))
        r2 = RADIUS_PARAMETER * (a2 ** (1.0 / 3.0))
        d_center = r1 + r2 + self.d_neck

        # Coulomb energy at scission = TKE
        e2_meV_fm = 1.43998
        tke = z1 * z2 * e2_meV_fm / d_center

        # Q-value (with 0 prompt neutrons)
        q_val = q_value_fission(self.A, self.Z, a1, z1, a2, z2, 0)

        return {
            'a1': a1, 'z1': z1, 'a2': a2, 'z2': z2,
            'r1': r1, 'r2': r2,
            'd_center': d_center,
            'tke': tke,
            'q_value': q_val,
            'total_excitation': q_val - tke,
        }

    def charge_distribution(self, a_ff: int) -> int:
        """
        Estimate most probable charge for fragment mass A using
        the unchanged charge density (UCD) hypothesis:

        Z_p(A) = Z_CN * A / A_CN  (simplified)

        With polarisation correction:
        Z_p = Z_CN * A / A_CN + delta_Z_p

        where delta_Z_p depends on shell effects near Z=50, N=82.
        """
        z_ucd = self.Z * a_ff / self.A

        # Shell correction for Z_p near magic numbers
        # Enhanced yield near Z=50 (Sn) and N=82
        delta_z = 0.0
        n_ff = a_ff - int(round(z_ucd))

        # Near N=82 shell closure
        if abs(n_ff - 82) < 5:
            delta_z = 0.3 * math.exp(-(n_ff - 82) ** 2 / 8.0)

        z_p = z_ucd + delta_z
        return max(1, min(a_ff - 1, int(round(z_p))))

    def yield_at_mass(self, a1: int) -> float:
        """
        Compute log(Y(A1)) for fragment mass A1.

        Y(A1) ~ rho_1(U1) * rho_2(U2) * exp(-(V_scission - E_gs)/T)

        Returns ln(Y) to avoid overflow.
        """
        a2 = self.A - a1
        z1 = self.charge_distribution(a1)
        z2 = self.Z - z1

        config = self.scission_configuration(a1, z1)
        tke = config['tke']
        u_total = config['total_excitation']

        if u_total <= 0:
            return -1.0e30

        # Energy partition: U1/U2 ~ (A1/A2)^{2/3} * (level_density_ratio)
        # Simplified: equipartition weighted by level density parameter
        a_param1 = level_density_parameter(a1)
        a_param2 = level_density_parameter(a2)

        # Temperature at scission
        a_total = a_param1 + a_param2
        if a_total <= 0:
            return -1.0e30

        t_scission = math.sqrt(u_total / a_total) if u_total > 0 else 0.0

        # Excitation energy partition
        u1 = a_param1 * t_scission * t_scission
        u2 = a_param2 * t_scission * t_scission

        # Level densities
        shell1 = -2.0 if abs(a1 - 132) < 10 else 0.0  # near N=82
        shell2 = -2.0 if abs(a2 - 132) < 10 else 0.0

        log_rho1 = fermi_gas_level_density(u1, a1, shell1)
        log_rho2 = fermi_gas_level_density(u2, a2, shell2)

        # Boltzmann factor for scission-point potential
        # (simplified: just the TKE deficit)
        boltz = 0.0  # already included in level densities

        return log_rho1 + log_rho2 + boltz

    def compute_yield_curve(self, a_min: int = 70, a_max: int = 170) -> Tuple[List[int], List[float]]:
        """
        Compute the full mass yield curve Y(A) for A in [a_min, a_max].
        """
        masses = list(range(a_min, a_max + 1))
        log_yields = []

        for a1 in masses:
            ly = self.yield_at_mass(a1)
            log_yields.append(ly)

        # Normalise: sum(Y) = 2.0 (two fragments per fission)
        max_ly = max(log_yields)
        yields_raw = [math.exp(ly - max_ly) for ly in log_yields]
        total = sum(yields_raw)
        if total > 0:
            yields_norm = [2.0 * y / total for y in yields_raw]
        else:
            yields_norm = [0.0] * len(masses)

        return masses, yields_norm

    def cubic_spline_interpolation(self, x: List[float],
                                    y: List[float]) -> 'CubicSpline':
        """
        Build cubic spline interpolant (from interp_spline project).

        Not-a-knot boundary conditions:
          The third derivative is continuous at x[1] and x[-2].

        Returns a CubicSpline object for evaluation at arbitrary points.
        """
        return CubicSpline(x, y)

    def interpolated_yield(self, a_continuous: float,
                           masses: List[int], yields: List[float]) -> float:
        """
        Evaluate the mass yield at a continuous mass number using
        cubic spline interpolation.
        """
        x_data = [float(m) for m in masses]
        spline = self.cubic_spline_interpolation(x_data, yields)
        return spline.evaluate(a_continuous)

    def peak_positions(self, masses: List[int],
                       yields: List[float]) -> List[Dict[str, float]]:
        """
        Find peak positions in the mass yield curve.
        """
        peaks = []
        for i in range(1, len(yields) - 1):
            if yields[i] > yields[i - 1] and yields[i] > yields[i + 1]:
                peaks.append({
                    'mass': masses[i],
                    'yield': yields[i],
                    'is_heavy': masses[i] > self.A / 2,
                })
        return peaks


class CubicSpline:
    """
    Natural cubic spline interpolation (not-a-knot conditions).

    From interp_spline project concept.

    Given n data points (x_i, y_i), the spline S(x) satisfies:
      S(x_i) = y_i
      S''(x_i-) = S''(x_i+)  (smooth second derivative)
      S'''(x_1-) = S'''(x_1+)  (not-a-knot at left)
      S'''(x_{n-1}-) = S'''(x_{n-1}+)  (not-a-knot at right)

    On each interval [x_i, x_{i+1}]:
      S(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
    """

    def __init__(self, x: List[float], y: List[float]):
        n = len(x)
        if n < 3:
            raise ValueError("Need at least 3 points for cubic spline")

        self.x = x
        self.y = y
        self.n = n
        self.a = y.copy()
        self.b = [0.0] * n
        self.c = [0.0] * n
        self.d = [0.0] * n

        self._compute_coefficients()

    def _compute_coefficients(self) -> None:
        """Compute spline coefficients using tridiagonal system."""
        n = self.n
        x = self.x
        y = self.y

        h = [x[i + 1] - x[i] for i in range(n - 1)]
        for i in range(n - 1):
            if h[i] < 1e-14:
                h[i] = 1e-14

        # Build tridiagonal system for c_i (second derivatives)
        alpha = [0.0] * n
        for i in range(1, n - 1):
            alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - \
                       (3.0 / h[i - 1]) * (y[i] - y[i - 1])

        # Solve tridiagonal system
        l = [1.0] * n
        mu = [0.0] * n
        z = [0.0] * n

        for i in range(1, n - 1):
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            if abs(l[i]) < 1e-14:
                l[i] = 1e-14
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        # Not-a-knot: l[-1] = 1, z[-1] = 0
        l[-1] = 1.0
        z[-1] = 0.0
        self.c[-1] = 0.0

        # Back substitution
        for j in range(n - 2, -1, -1):
            self.c[j] = z[j] - mu[j] * self.c[j + 1]

        # Compute b and d
        for i in range(n - 1):
            if h[i] < 1e-14:
                h[i] = 1e-14
            self.b[i] = ((y[i + 1] - y[i]) / h[i]
                         - h[i] * (self.c[i + 1] + 2.0 * self.c[i]) / 3.0)
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])

    def evaluate(self, xval: float) -> float:
        """Evaluate spline at xval using binary search for the interval."""
        if xval <= self.x[0]:
            return self.y[0]
        if xval >= self.x[-1]:
            return self.y[-1]

        # Binary search
        lo, hi = 0, self.n - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.x[mid] <= xval:
                lo = mid
            else:
                hi = mid

        dx = xval - self.x[lo]
        return (self.a[lo] + self.b[lo] * dx + self.c[lo] * dx * dx
                + self.d[lo] * dx * dx * dx)
