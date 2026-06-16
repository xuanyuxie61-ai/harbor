"""
potential_energy_surface.py
============================
Potential energy surface (PES) for the fissioning nucleus.

Maps from: 846_paraheat_functional (2D FEM with piecewise diffusivity)

Physical context
----------------
The PES V(q) is a function of collective coordinates q = (c, h, alpha) describing
the nuclear shape.  We implement:

1. Liquid Drop Model energy:
   E_LD = E_surface + E_Coulomb + E_proximity

2. Shell correction: delta_E_shell (Strutinsky method)

3. Pairing correction: delta_E_pair (BCS approximation)

4. Total PES: V(q) = E_LD(q) + delta_E_shell(q) + delta_E_pair(q)

The surface energy is:
  E_surf = a_s * (1 - k_s * I^2) * B_s(q) * A^{2/3}

where B_s(q) is the surface energy relative to sphere, and
  I = (N - Z) / A  (isospin parameter)

The Coulomb energy:
  E_Coul = (3/5) * (e^2 * Z^2 / R_0) * B_c(q)

where B_c(q) is the Coulomb energy relative to sphere.

The proximity energy:
  E_prox = 4*pi*gamma*b_bar * d_bar * integral of phi(d/d_bar) over neck surface

For this implementation, we use a simplified five-point paraboloid
approximation for the fission barrier, which captures the double-humped
barrier structure.
"""

import math
from typing import Tuple, List, Dict, Optional

from nuclear_constants import (
    LD_SURFACE, LD_COULOMB, LD_ASYMMETRY, LD_PAIRING_A, LD_WIGNER,
    RADIUS_PARAMETER, HBAR_C_MEV_FM, BOLTZMANN_MEV_K
)


class FissionPES:
    """
    Potential Energy Surface for fission.

    The PES uses a multi-dimensional collective space q = (c, h, alpha).
    The barrier structure follows a five-point paraboloid model:

    V(c, h, alpha) = V_gs(q) + V_inner(q) + V_outer(q)

    where:
    - V_gs is the ground-state deformation energy
    - V_inner is the inner (first) barrier
    - V_outer is the outer (second) barrier

    Inner barrier: Gaussian centred at c_inner
    Outer barrier: Gaussian centred at c_outer

    V_inner = B1 * exp(-(c - c1)^2 / (2*sigma1^2)) * f_h(h, alpha)
    V_outer = B2 * exp(-(c - c2)^2 / (2*sigma2^2)) * g_h(h, alpha)
    """

    def __init__(self, z_cn: int = 92, a_cn: int = 236):
        self.Z = z_cn
        self.A = a_cn
        self.N = a_cn - z_cn
        self.isospin = (self.N - self.Z) / self.A
        self.r0 = RADIUS_PARAMETER * (self.A ** (1.0 / 3.0))

        # Five-point paraboloid parameters (typical for U-236)
        # Ground state at c=1.0
        self.c_gs = 1.0
        self.c_inner = 1.35    # inner barrier
        self.c_outer = 1.75    # outer barrier
        self.c_scission = 2.20 # scission point

        # Barrier heights (MeV)
        self.b_inner = 5.8     # inner barrier height
        self.b_outer = 5.2     # outer barrier height

        # Barrier widths
        self.sigma1 = 0.15     # inner barrier width
        self.sigma2 = 0.20     # outer barrier width

        # Shell corrections at barriers (MeV)
        self.shell_inner = -2.5
        self.shell_outer = -1.8
        self.shell_gs = -4.2   # ground state shell correction

        # Pairing gap parameter (MeV)
        self.pairing_gap = 12.0 / math.sqrt(self.A)

    def surface_energy_ratio(self, c: float, h: float, alpha: float) -> float:
        """
        Surface energy relative to sphere: B_s(q) = S(q) / S(sphere).

        For a sphere, B_s = 1 by construction.  Deformation increases
        surface area, so B_s >= 1 always.

        Using the sharp-surface approximation for the parameterised shape:
          B_s(c, h, alpha) ~ c^{1/3} * g(h) * (1 + alpha^2 correction)

        Volume conservation requires c^3 * (1 + (2/5)*(1-h)) = 1, giving
        c_vc = (1 + (2/5)*(1-h))^{-1/3}.  The reference sphere has B_s = 1
        when c_vc = 1 and h = 1.

        Cohen-Swiatecki parametrisation gives B_s ~ 1.0 at sphere and
        increases to ~1.3-1.5 at scission.
        """
        if c < 0.5:
            c = 0.5

        # Volume-conserving elongation factor
        # For c > 1 the surface grows like c^{1/3} to first order
        shape_factor = c ** (1.0 / 3.0)

        # Neck correction: h < 1 means a neck is forming, increasing surface
        if h < 1.0:
            neck_factor = 1.0 + 0.35 * (1.0 - h) * (1.0 - h)
        else:
            neck_factor = 1.0 + 0.1 * (h - 1.0)

        # Asymmetry correction
        asym_factor = 1.0 + 0.5 * alpha * alpha

        return shape_factor * neck_factor * asym_factor

    def coulomb_energy_ratio(self, c: float, h: float, alpha: float) -> float:
        """
        Coulomb energy relative to uniformly charged sphere: B_c(q).

        For elongated shapes the Coulomb energy decreases because the
        charge distribution becomes more spread out:
          B_c(c, h, alpha) ~ c^{-1/3} * f(h)

        At the sphere (c=1, h=1), B_c = 1 by construction.
        """
        if c < 0.5:
            c = 0.5

        # Elongation reduces Coulomb energy
        elong_factor = c ** (-1.0 / 3.0)

        # Neck formation slightly increases Coulomb (charge separation)
        if h < 1.0:
            neck_factor = 1.0 + 0.1 * (1.0 - h)
        else:
            neck_factor = 1.0

        # Asymmetry correction (small)
        asym_factor = 1.0 - 0.1 * alpha * alpha

        b_c = elong_factor * neck_factor * max(asym_factor, 0.5)
        return b_c

    def proximity_energy(self, c: float, h: float) -> float:
        """
        Proximity energy between forming fragments:

        E_prox = 4*pi*gamma*b_bar * R1*R2/(R1+R2) * Phi(d/b_bar)

        where:
        - gamma = surface energy coefficient ~ 1.0 MeV/fm^2
        - b_bar ~ 1.0 fm (surface width)
        - d = surface-to-surface distance
        - Phi(x) = -1.36*x - 0.6 for x < 0, approx exp(-x) for x > 0

        This is significant only when h < 1 (neck is forming).
        """
        if h >= 1.0:
            return 0.0

        # Neck parameter determines gap
        gamma_prox = 1.0  # MeV/fm^2
        b_bar = 1.0       # fm

        # Effective radii of forming fragments (volume-conserving split)
        # For mass-symmetric split: R1 = R2 = R0 * 2^{-1/3}
        r_fragment = self.r0 * (0.5 ** (1.0 / 3.0))

        # Gap distance depends on neck parameter and elongation
        d_gap = self.r0 * c * (1.0 - h) * 0.5

        if d_gap < 0.0:
            d_gap = 0.0

        # Proximity function Phi(s)
        s = d_gap / b_bar
        if s < 0:
            phi_s = 0.0
        elif s < 1.5:
            phi_s = -0.5 * (s - 1.5) * (s - 1.5)
        else:
            phi_s = -0.1 * math.exp(-1.5 * s)

        # Geometric factor
        r1r2_over_sum = r_fragment * r_fragment / (2.0 * r_fragment)

        e_prox = 4.0 * math.pi * gamma_prox * b_bar * r1r2_over_sum * phi_s
        return e_prox

    def liquid_drop_energy(self, c: float, h: float, alpha: float) -> float:
        """
        Liquid-drop model deformation energy:
          E_LD = a_s * A^{2/3} * (1 - k_s * I^2) * (B_s - 1)
                 + a_c * Z^2 / A^{1/3} * (B_c - 1)
                 + E_proximity

        where k_s = 1.7826 (surface asymmetry coefficient).
        """
        k_s = 1.7826
        i_sq = self.isospin * self.isospin

        a_surf_eff = LD_SURFACE * (self.A ** (2.0 / 3.0)) * (1.0 - k_s * i_sq)
        a_coul = 0.7103 * self.Z * self.Z / (self.A ** (1.0 / 3.0))

        b_s = self.surface_energy_ratio(c, h, alpha)
        b_c = self.coulomb_energy_ratio(c, h, alpha)
        e_prox = self.proximity_energy(c, h)

        return (a_surf_eff * (b_s - 1.0)
                + a_coul * (b_c - 1.0)
                + e_prox)

    def shell_correction(self, c: float, h: float, alpha: float) -> float:
        """
        Shell correction using five-point paraboloid model:

        delta_E_shell = delta_E_gs * exp(-(c-1)^2/(2*sigma_gs^2))
                      + delta_E_inner * exp(-(c-c_inner)^2/(2*sigma1^2))
                      + delta_E_outer * exp(-(c-c_outer)^2/(2*sigma2^2))

        The shell correction washes out at large deformations (h -> 0).
        """
        # Damping factor for shell correction at large deformation
        damping = math.exp(-0.5 * (c - 1.0) ** 2 / 0.3) if c > 1.5 else 1.0

        # Ground-state shell correction
        shell_gs = self.shell_gs * math.exp(
            -(c - self.c_gs) ** 2 / (2.0 * 0.1 * 0.1)
        )

        # Inner barrier shell correction
        shell_in = self.shell_inner * math.exp(
            -(c - self.c_inner) ** 2 / (2.0 * self.sigma1 * self.sigma1)
        )

        # Outer barrier shell correction
        shell_out = self.shell_outer * math.exp(
            -(c - self.c_outer) ** 2 / (2.0 * self.sigma2 * self.sigma2)
        )

        # Mass asymmetry dependence of shell effects
        asym_factor = math.exp(-alpha * alpha / 0.1)

        return (shell_gs + (shell_in + shell_out) * asym_factor) * damping

    def pairing_energy(self, c: float, h: float) -> float:
        """
        Pairing correlation energy (BCS approximation):

        E_pair = -Delta^2 / G_eff

        where Delta is the pairing gap and G_eff is the effective pairing
        strength.  Near scission, pairing correlations are enhanced in the
        neck region.

        Simplified: E_pair ~ -Delta_0 * (1 + delta_pair(c,h))
        where delta_pair increases near scission (h -> 0, c > 1.5).
        """
        delta_0 = self.pairing_gap

        # Pairing enhancement near scission
        if h < 0.7 and c > 1.5:
            scission_factor = 1.0 + 0.3 * (1.0 - h / 0.7) * ((c - 1.5) / 0.5)
        else:
            scission_factor = 1.0

        # Pairing reduction with excitation energy (not tracked here)
        e_pair = -delta_0 * delta_0 / 1.0 * scission_factor
        return e_pair

    def five_point_barrier(self, c: float, h: float, alpha: float) -> float:
        """
        Five-point paraboloid barrier model:

        V_barrier = B1 * exp(-(c-c1)^2/(2*sigma1^2))
                  + B2 * exp(-(c-c2)^2/(2*sigma2^2))

        with dependence on h and alpha through modification factors.
        """
        v_inner = self.b_inner * math.exp(
            -(c - self.c_inner) ** 2 / (2.0 * self.sigma1 * self.sigma1)
        )
        v_outer = self.b_outer * math.exp(
            -(c - self.c_outer) ** 2 / (2.0 * self.sigma2 * self.sigma2)
        )

        # Neck dependence
        neck_factor = 1.0 + 0.2 * (1.0 - h)
        # Asymmetry dependence
        asym_factor = 1.0 - 0.5 * alpha * alpha

        return (v_inner + v_outer) * neck_factor * max(asym_factor, 0.1)

    def total_potential(self, c: float, h: float, alpha: float) -> float:
        """
        Total potential energy surface:

        V(c, h, alpha) = E_LD(c, h, alpha) + delta_E_shell(c, h, alpha)
                         + E_pair(c, h) + V_barrier(c, h, alpha)
        """
        e_ld = self.liquid_drop_energy(c, h, alpha)
        e_shell = self.shell_correction(c, h, alpha)
        e_pair = self.pairing_energy(c, h)
        v_barrier = self.five_point_barrier(c, h, alpha)

        return e_ld + e_shell + e_pair + v_barrier

    def gradient(self, c: float, h: float, alpha: float,
                 dc: float = 1e-4, dh: float = 1e-4,
                 da: float = 1e-4) -> Tuple[float, float, float]:
        """
        Gradient of PES via central differences:
          dV/dc = [V(c+dc, h, a) - V(c-dc, h, a)] / (2*dc)
        """
        v_cp = self.total_potential(c + dc, h, alpha)
        v_cm = self.total_potential(c - dc, h, alpha)
        v_hp = self.total_potential(c, h + dh, alpha)
        v_hm = self.total_potential(c, h - dh, alpha)
        v_ap = self.total_potential(c, h, alpha + da)
        v_am = self.total_potential(c, h, alpha - da)

        grad_c = (v_cp - v_cm) / (2.0 * dc)
        grad_h = (v_hp - v_hm) / (2.0 * dh)
        grad_a = (v_ap - v_am) / (2.0 * da)

        return (grad_c, grad_h, grad_a)

    def hessian_diagonal(self, c: float, h: float, alpha: float,
                         dc: float = 1e-3) -> Tuple[float, float, float]:
        """
        Diagonal elements of Hessian (second derivatives):
          d2V/dc2 = [V(c+dc) - 2*V(c) + V(c-dc)] / dc^2
        """
        v0 = self.total_potential(c, h, alpha)
        v_cp = self.total_potential(c + dc, h, alpha)
        v_cm = self.total_potential(c - dc, h, alpha)
        v_hp = self.total_potential(c, h + dc, alpha)
        v_hm = self.total_potential(c, h - dc, alpha)
        v_ap = self.total_potential(c, h, alpha + dc)
        v_am = self.total_potential(c, h, alpha - dc)

        d2_dc2 = (v_cp - 2.0 * v0 + v_cm) / (dc * dc)
        d2_dh2 = (v_hp - 2.0 * v0 + v_hm) / (dc * dc)
        d2_da2 = (v_ap - 2.0 * v0 + v_am) / (dc * dc)

        return (d2_dc2, d2_dh2, d2_da2)

    def find_saddle_points(self, alpha: float = 0.0,
                           h: float = 1.0) -> List[Dict[str, float]]:
        """
        Find saddle points of the PES along the c-direction.

        Scans V(c, h, alpha) and identifies local maxima (barriers)
        and minima (isomeric states).
        """
        saddles = []
        c_min, c_max = 0.9, 2.5
        nc_scan = 200
        dc = (c_max - c_min) / nc_scan

        v_prev = self.total_potential(c_min, h, alpha)
        v_curr = self.total_potential(c_min + dc, h, alpha)

        for i in range(1, nc_scan):
            c_curr = c_min + i * dc
            c_next = c_curr + dc
            v_next = self.total_potential(c_next, h, alpha)

            # Local maximum (inner or outer barrier)
            if v_curr > v_prev and v_curr > v_next:
                # Refine with golden section
                c_opt = self._refine_extremum(
                    c_curr - dc, c_curr + dc, h, alpha, is_max=True
                )
                v_opt = self.total_potential(c_opt, h, alpha)
                saddles.append({
                    'type': 'saddle',
                    'c': c_opt,
                    'h': h,
                    'alpha': alpha,
                    'energy': v_opt
                })

            # Local minimum (isomeric state)
            if v_curr < v_prev and v_curr < v_next:
                c_opt = self._refine_extremum(
                    c_curr - dc, c_curr + dc, h, alpha, is_max=False
                )
                v_opt = self.total_potential(c_opt, h, alpha)
                saddles.append({
                    'type': 'minimum',
                    'c': c_opt,
                    'h': h,
                    'alpha': alpha,
                    'energy': v_opt
                })

            v_prev = v_curr
            v_curr = v_next

        return saddles

    def _refine_extremum(self, c_lo: float, c_hi: float,
                         h: float, alpha: float,
                         is_max: bool, tol: float = 1e-6) -> float:
        """
        Golden section search for extremum of V(c, h, alpha) in [c_lo, c_hi].
        """
        gr = (math.sqrt(5.0) + 1.0) / 2.0

        a_c, b_c = c_lo, c_hi
        for _ in range(50):
            if abs(b_c - a_c) < tol:
                break
            c_test = b_c - (b_c - a_c) / gr
            d_test = a_c + (b_c - a_c) / gr

            v_c = self.total_potential(c_test, h, alpha)
            v_d = self.total_potential(d_test, h, alpha)

            if is_max:
                if v_c > v_d:
                    b_c = d_test
                else:
                    a_c = c_test
            else:
                if v_c < v_d:
                    b_c = d_test
                else:
                    a_c = c_test

        return 0.5 * (a_c + b_c)
