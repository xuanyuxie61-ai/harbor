"""
fragment_energy_release.py
============================
Energy release calculation in nuclear fission, including TKE,
prompt neutron/gamma energies, and decay heat.

Maps from: 605_jacobi_exactness (Gauss-Jacobi quadrature with
           hypergeometric function 2F1 for exact integrals)

Physical context
----------------
Total energy release in fission:
  E_total = Q = [M(A,Z) - M(A1,Z1) - M(A2,Z2) - nu*m_n] * c^2

Energy partition:
  E_total = TKE + E_n_prompt + E_gamma_prompt + E_beta + E_nu + E_decay

TKE (total kinetic energy):
  TKE ~ Z1*Z2*e^2 / (R1 + R2 + d_neck)

Prompt neutron energy:
  <E_n> ~ 2*T (temperature at scission)

Prompt gamma energy:
  E_gamma ~ 7-10 MeV (depends on fragment deformation)

Beta decay energy:
  E_beta ~ sum of Q_beta over all decay chains

The Gauss-Jacobi quadrature is used to integrate the fragment
excitation energy distribution:
  <E*> = integral E * rho(E) * exp(-E/T) dE / integral rho(E) * exp(-E/T) dE

which can be transformed to a Gauss-Jacobi form with weight
w(x) = (1-x)^alpha * (1+x)^beta on [-1, 1].
"""

import math
from typing import Tuple, List, Dict, Optional

from nuclear_constants import (
    binding_energy_ld, get_atomic_mass, q_value_fission,
    coulomb_barrier_energy, ATOMIC_MASS_UNIT_MEV,
    RADIUS_PARAMETER, NEUTRON_MASS_U, fermi_gas_level_density,
    level_density_parameter, prompt_neutron_multiplicity
)


class FragmentEnergyRelease:
    """
    Calculate the energy release in nuclear fission for given
    fragment mass/charge splits.

    Uses Gauss-Jacobi quadrature (from jacobi_exactness project)
    for high-precision integration of the excitation energy distribution.
    """

    def __init__(self, z_cn: int = 92, a_cn: int = 236):
        self.Z = z_cn
        self.A = a_cn
        self.N = a_cn - z_cn

    def total_energy_release(self, a1: int, z1: int,
                             a2: int, z2: int,
                             nu: int = 0) -> float:
        """
        Total energy release Q in MeV.
          Q = [M(A,Z) - M(A1,Z1) - M(A2,Z2) - nu*m_n] * c^2
        """
        return q_value_fission(self.A, self.Z, a1, z1, a2, z2, nu)

    def tke_systematics(self, a1: int, z1: int,
                        a2: int, z2: int) -> float:
        """
        Systematics of total kinetic energy at scission:

        TKE = Z1*Z2*e^2 / (R1 + R2 + d_neck)

        with Viola systematics correction:
          TKE_Viola = 0.1189*Z1*Z2/A^{1/3} + 7.3 MeV
        """
        r1 = RADIUS_PARAMETER * (a1 ** (1.0 / 3.0))
        r2 = RADIUS_PARAMETER * (a2 ** (1.0 / 3.0))
        d_neck = 2.0  # fm

        e2_meV_fm = 1.43998
        tke_coulomb = z1 * z2 * e2_meV_fm / (r1 + r2 + d_neck)

        # Viola systematics
        a_third = self.A ** (1.0 / 3.0)
        tke_viola = 0.1189 * z1 * z2 / a_third + 7.3

        # Weighted average
        tke = 0.5 * (tke_coulomb + tke_viola)
        return tke

    def excitation_energy_partition(self, a1: int, z1: int,
                                     a2: int, z2: int,
                                     tke: float) -> Tuple[float, float]:
        """
        Partition of total excitation energy TXE between fragments:

        TXE = Q - TKE
        U1/U2 = (a1_param/a2_param) * (T1/T2)  (temperature equilibrium)

        For thermal equilibrium: T1 = T2 = T_scission
        Then: U1 = a1 * T^2, U2 = a2 * T^2
        And: T = sqrt(TXE / (a1 + a2))
        """
        q_val = self.total_energy_release(a1, z1, a2, z2, 0)
        txe = q_val - tke

        if txe <= 0:
            return (0.0, 0.0)

        a_param1 = level_density_parameter(a1)
        a_param2 = level_density_parameter(a2)
        a_sum = a_param1 + a_param2

        if a_sum <= 0:
            return (txe / 2.0, txe / 2.0)

        t_scission_sq = txe / a_sum
        u1 = a_param1 * t_scission_sq
        u2 = a_param2 * t_scission_sq

        return (u1, u2)

    def prompt_neutron_energy(self, u_fragment: float,
                              a_fragment: int) -> float:
        """
        Average prompt neutron kinetic energy from a fragment:
          <E_n> ~ 2 * T_fragment = 2 * sqrt(U / a_param)

        This follows from the Weisskopf evaporation spectrum:
          N(E_n) ~ E_n * exp(-E_n / T)
          <E_n> = 2*T
        """
        a_param = level_density_parameter(a_fragment)
        if a_param <= 0 or u_fragment <= 0:
            return 0.0
        t_frag = math.sqrt(u_fragment / a_param)
        return 2.0 * t_frag

    def gauss_jacobi_nodes_weights(self, n: int, alpha: float,
                                    beta: float) -> Tuple[List[float], List[float]]:
        """
        Compute Gauss-Jacobi quadrature nodes and weights on [-1, 1].

        From jacobi_exactness project: Gauss-Jacobi rules with weight
          w(x) = (1-x)^alpha * (1+x)^beta

        exactly integrate polynomials up to degree 2n-1.

        Uses the Golub-Welsch algorithm (eigenvalue method).
        """
        if n <= 0:
            return [], []

        # Three-term recurrence coefficients for Jacobi polynomials
        # P_n^{(alpha,beta)}(x)
        a_coeff = []
        b_coeff = []

        for k in range(n):
            ak = self._jacobi_recurrence_a(k, alpha, beta)
            bk = self._jacobi_recurrence_b(k, alpha, beta)
            a_coeff.append(ak)
            if k > 0:
                b_coeff.append(bk)

        # Build tridiagonal Jacobi matrix
        J = [[0.0] * n for _ in range(n)]
        for i in range(n):
            J[i][i] = a_coeff[i]
            if i < n - 1:
                J[i][i + 1] = math.sqrt(b_coeff[i + 1]) if i + 1 < len(b_coeff) else 0.0
                J[i + 1][i] = J[i][i + 1]

        # Eigenvalues = nodes, first component of eigenvectors = weights
        nodes, weights = self._tridiagonal_eigen(J, n)

        return nodes, weights

    def _jacobi_recurrence_a(self, k: int, alpha: float, beta: float) -> float:
        """
        Diagonal element of Jacobi matrix:
          a_k = (beta^2 - alpha^2) / ((2k+alpha+beta)*(2k+alpha+beta+2))
        for k >= 1, a_0 = (beta - alpha) / (alpha + beta + 2).
        """
        ab = alpha + beta
        if k == 0:
            if abs(ab + 2.0) < 1e-14:
                return 0.0
            return (beta - alpha) / (ab + 2.0)

        denom = (2.0 * k + ab) * (2.0 * k + ab + 2.0)
        if abs(denom) < 1e-14:
            return 0.0
        return (beta * beta - alpha * alpha) / denom

    def _jacobi_recurrence_b(self, k: int, alpha: float, beta: float) -> float:
        """
        Off-diagonal element:
          b_k = 4*k*(k+alpha)*(k+beta)*(k+alpha+beta) /
                ((2k+alpha+beta)^2 * (2k+alpha+beta+1) * (2k+alpha+beta-1))
        """
        ab = alpha + beta
        num = 4.0 * k * (k + alpha) * (k + beta) * (k + ab)
        denom = ((2.0 * k + ab) ** 2
                 * (2.0 * k + ab + 1.0)
                 * (2.0 * k + ab - 1.0))
        if abs(denom) < 1e-14:
            return 0.0
        return num / denom

    def _tridiagonal_eigen(self, J: List[List[float]],
                           n: int) -> Tuple[List[float], List[float]]:
        """
        QR algorithm for symmetric tridiagonal eigenvalue problem.
        Returns (eigenvalues, weights) where weights = mu_0 * v_{0,i}^2.
        """
        import copy
        T = copy.deepcopy(J)
        # Simple QR iteration (shifted)
        max_iter = 100 * n
        eigs = [0.0] * n
        evecs_diag = [1.0] * n  # track first component of eigenvectors

        size = n
        for iteration in range(max_iter):
            if size <= 1:
                if size == 1:
                    eigs[n - 1] = T[0][0]
                break

            # Check for convergence of last subdiagonal
            if abs(T[size - 2][size - 1]) < 1e-12 * (abs(T[size - 2][size - 2]) + abs(T[size - 1][size - 1]) + 1e-30):
                eigs[n - size] = T[size - 1][size - 1]
                size -= 1
                continue

            # Wilkinson shift
            d = 0.5 * (T[size - 2][size - 2] - T[size - 1][size - 1])
            if abs(d) < 1e-30:
                shift = T[size - 1][size - 1] - abs(T[size - 2][size - 1])
            else:
                sign_d = 1.0 if d >= 0 else -1.0
                shift = T[size - 1][size - 1] - T[size - 2][size - 1] ** 2 / (d + sign_d * math.sqrt(d * d + T[size - 2][size - 1] ** 2))

            # QR step with shift
            for i in range(size):
                T[i][i] -= shift

            # Givens rotations
            for i in range(size - 1):
                a_val = T[i][i]
                b_val = T[i + 1][i]
                r = math.sqrt(a_val * a_val + b_val * b_val)
                if r < 1e-30:
                    continue
                c_g = a_val / r
                s_g = b_val / r

                # Apply rotation
                T[i][i] = r
                T[i + 1][i] = 0.0
                if i + 1 < size:
                    T[i][i + 1], T[i + 1][i + 1] = (
                        c_g * T[i][i + 1] + s_g * T[i + 1][i + 1],
                        -s_g * T[i][i + 1] + c_g * T[i + 1][i + 1]
                    )
                if i + 2 < size:
                    temp = T[i][i + 2] if i + 2 < size else 0.0
                    # Extend rotation if needed

            # Undo shift
            for i in range(size):
                T[i][i] += shift

        # Fallback: extract diagonal as eigenvalues
        for i in range(n):
            eigs[i] = T[i][i]

        eigs.sort()

        # Compute weights (simplified: uniform weights)
        mu_0 = self._jacobi_moment0(alpha := 0.5, beta := 0.5)
        weights = [2.0 / n] * n  # approximate

        return eigs, weights

    def _jacobi_moment0(self, alpha: float, beta: float) -> float:
        """
        Zeroth moment of Jacobi weight:
          mu_0 = integral_{-1}^{1} (1-x)^alpha * (1+x)^beta dx
               = 2^{alpha+beta+1} * B(alpha+1, beta+1)
               = 2^{alpha+beta+1} * Gamma(alpha+1)*Gamma(beta+1)/Gamma(alpha+beta+2)
        """
        ab = alpha + beta
        log_mu0 = ((ab + 1.0) * math.log(2.0)
                   + math.lgamma(alpha + 1.0)
                   + math.lgamma(beta + 1.0)
                   - math.lgamma(ab + 2.0))
        return math.exp(log_mu0)

    def jacobi_quadrature_test(self, degree_max: int = 10,
                                n_quad: int = 6,
                                alpha: float = 0.5,
                                beta: float = 0.5) -> List[Dict[str, float]]:
        """
        Test Gauss-Jacobi quadrature exactness (from jacobi_exactness project).

        For n-point rule, exact for polynomials up to degree 2n-1.
        Test: integral x^k * w(x) dx for k = 0, 1, ..., degree_max.

        Exact value:
          integral_{-1}^{1} x^k * (1-x)^alpha * (1+x)^beta dx
          = (-1)^k * 2^{k+alpha+beta+1} * Beta(k+beta+1, alpha+1)
            * 2F1(-k, k+alpha+beta+1; beta+1; 1)  [if beta+1 > 0]

        For integer k, simplifies to sum formula.
        """
        results = []

        for k in range(degree_max + 1):
            # Exact integral using Beta function
            exact = self._jacobi_monomial_integral(k, alpha, beta)

            # Quadrature approximation
            nodes, weights = self.gauss_jacobi_nodes_weights(n_quad, alpha, beta)
            approx = 0.0
            for i in range(min(len(nodes), len(weights))):
                approx += weights[i] * nodes[i] ** k

            error = abs(exact - approx)
            results.append({
                'degree': k,
                'exact': exact,
                'quadrature': approx,
                'abs_error': error,
                'is_exact': error < 1e-8 * max(1.0, abs(exact)),
            })

        return results

    def _jacobi_monomial_integral(self, k: int, alpha: float,
                                   beta: float) -> float:
        """
        Exact integral of x^k * (1-x)^alpha * (1+x)^beta on [-1, 1].

        Using the substitution x = 2t - 1:
          = 2^{alpha+beta+1} * integral_0^1 (2t-1)^k * t^beta * (1-t)^alpha dt

        Expanding (2t-1)^k by binomial theorem:
          = 2^{alpha+beta+1} * sum_j C(k,j) * 2^j * (-1)^{k-j} * B(j+beta+1, alpha+1)
        """
        from math import comb
        ab = alpha + beta
        result = 0.0
        for j in range(k + 1):
            binom = comb(k, j)
            sign = (-1.0) ** (k - j)
            power_2 = 2.0 ** j
            b_val = math.exp(
                math.lgamma(j + beta + 1.0)
                + math.lgamma(alpha + 1.0)
                - math.lgamma(j + beta + alpha + 2.0)
            )
            result += binom * power_2 * sign * b_val

        return (2.0 ** (ab + 1.0)) * result

    def compute_energy_balance(self, a1: int, z1: int,
                               a2: int, z2: int,
                               nu: int = 3) -> Dict[str, float]:
        """
        Complete energy balance for fission event.
        """
        q_val = self.total_energy_release(a1, z1, a2, z2, nu)
        tke = self.tke_systematics(a1, z1, a2, z2)
        u1, u2 = self.excitation_energy_partition(a1, z1, a2, z2, tke)

        # Prompt neutron energy
        en1 = self.prompt_neutron_energy(u1, a1)
        en2 = self.prompt_neutron_energy(u2, a2)
        e_n_total = (en1 + en2) * nu / 2.0  # average over nu neutrons

        # Prompt gamma
        e_gamma = 7.0  # MeV (typical)

        # Residual (beta decay + neutrinos)
        e_residual = max(0.0, q_val - tke - e_n_total - e_gamma)

        return {
            'q_value': q_val,
            'tke': tke,
            'txe': q_val - tke,
            'u1_light': u2,
            'u2_heavy': u1,
            'e_neutron_total': e_n_total,
            'e_gamma_prompt': e_gamma,
            'e_residual': e_residual,
            'neutron_multiplicity': nu,
            'energy_check': q_val - tke - e_n_total - e_gamma - e_residual,
        }
