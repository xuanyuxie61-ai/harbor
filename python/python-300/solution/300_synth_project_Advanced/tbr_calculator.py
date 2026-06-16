"""
tbr_calculator.py
=================
Tritium Breeding Ratio (TBR) calculation for the breeding blanket.

The TBR is the key figure of merit for a fusion reactor blanket: it counts
the number of tritium atoms produced per fusion neutron born in the plasma.
A self-sustaining D-T reactor requires  TBR >= 1.05  (the "tritium self-
sufficiency" margin accounting for processing losses).

Two reactions produce tritium:

    6Li + n  ->  4He + T   + 4.78 MeV     (exothermic, 1/v at low E)
    7Li + n  ->  4He + T + n' - 2.47 MeV  (threshold at 2.47 MeV)

The TBR is computed as

    TBR = integral_V [ Sigma_{a,6}(x) phi_6(x) + Sigma_{a,7}(x) phi_7(x) ] dV
        / integral_V Q_source(x) dV

where phi_g is the scalar flux and Sigma_{a,6}, Sigma_{a,7} are the
macroscopic (n,t) cross sections for Li-6 and Li-7.

Adapted from seed projects:
    * 073_basketball_dynamic -> projectile kinematics (neutron ballistics)
    * 1265_pgoelz_fluid      -> flow continuity (T production rate balance)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import physics_constants as pc


# ---------------------------------------------------------------------------
# Macroscopic (n,t) cross sections
# ---------------------------------------------------------------------------
def sigma_li6_nt_macro(
    group: int,
    temperature_K: float = 600.0,
    li6_atom_density: Optional[float] = None,
) -> float:
    """Return Sigma_{n,t}^{Li6}(g, T) in cm^{-1}.

    Uses the 1/v scaling with Doppler correction:

        sigma(E, T) = sigma_0 sqrt(E_0 / E) * (1 + alpha (sqrt(T) - sqrt(T0)))

    where sigma_0 = 940 barn at E_0 = 2.53e-8 MeV (thermal) and
    alpha = Gamma / (2 E_r) with Gamma = 3 eV, E_r = group midpoint energy.
    """
    if li6_atom_density is None:
        li6_atom_density = pc.ATOM_DENSITY_LI * pc.FRAC_LI6
    if not 0 <= group < pc.N_GROUPS:
        raise ValueError("group out of range")
    E_mid = pc.group_midpoint_energy(group)
    if E_mid <= 0.0:
        return 0.0
    # 1/v scaling from thermal
    sigma_thermal = 940.0     # barn at 2.53e-5 eV
    E_thermal = 2.53e-8       # MeV
    sigma_micro = sigma_thermal * math.sqrt(E_thermal / E_mid)
    # Doppler correction
    alpha = 3.0e-6 / (2.0 * E_mid)   # Gamma in MeV
    T0 = 300.0
    scale = 1.0 + alpha * (math.sqrt(temperature_K) - math.sqrt(T0))
    sigma_micro *= max(scale, 1.0)
    return sigma_micro * li6_atom_density     # cm^{-1}


def sigma_li7_nnt_macro(
    group: int,
    temperature_K: float = 600.0,
    li7_atom_density: Optional[float] = None,
) -> float:
    """Return Sigma_{n,n't}^{Li7}(g, T) in cm^{-1}.

    The (n,n't) reaction has threshold E_th = 2.468 MeV and the cross
    section follows  sigma = sigma_0 (1 - E_th/E)^2 H(E - E_th)  with
    sigma_0 = 0.50 barn (ENDF fit).
    """
    if li7_atom_density is None:
        li7_atom_density = pc.ATOM_DENSITY_LI * pc.FRAC_LI7
    if not 0 <= group < pc.N_GROUPS:
        raise ValueError("group out of range")
    E_mid = pc.group_midpoint_energy(group)
    E_th = pc.Q_LI7_NNT * -1.0      # 2.468 MeV (threshold is positive)
    E_th = abs(E_th)
    if E_mid <= E_th:
        return 0.0
    sigma_micro = 0.50 * (1.0 - E_th / E_mid) ** 2
    # Doppler smearing near threshold
    kT_mev = pc.K_BOLTZMANN_MEV * temperature_K
    smear = 4.0 * math.sqrt(kT_mev * E_th) / max(E_th, pc.EPS_NUMERICAL)
    sigma_micro *= (1.0 + 0.5 * smear)
    return sigma_micro * li7_atom_density


# ---------------------------------------------------------------------------
# TBR integrator
# ---------------------------------------------------------------------------
def compute_tbr(
    phi: List[List[float]],
    dx: float,
    pebble_fraction: List[float],
    temperature_K: float = 600.0,
) -> Dict[str, float]:
    """Compute the TBR from the multigroup scalar flux.

    Parameters
    ----------
    phi : list of list of float, shape (N_GROUPS, n_cells)
        Scalar flux phi_g(x_i) in neutrons / cm^2 / s.
    dx : cell width (cm).
    pebble_fraction : list of float, length n_cells
        Fraction of each cell that is Li2O pebble material.
    temperature_K : blanket operating temperature (K).

    Returns
    -------
    Dict with keys:
        tbr_total : total TBR (Li-6 + Li-7 contributions).
        tbr_li6 : contribution from 6Li(n,t).
        tbr_li7 : contribution from 7Li(n,n't).
        production_rate : total T atoms produced per cm^2 per s (1D).
        source_rate : total source neutrons per cm^2 per s.
    """
    G = len(phi)
    nx = len(phi[0]) if phi else 0
    if G != pc.N_GROUPS:
        raise ValueError("phi must have N_GROUPS rows")

    tbr_li6 = 0.0
    tbr_li7 = 0.0
    source_total = 0.0
    for i in range(nx):
        f_p = pebble_fraction[i]
        for g in range(G):
            # only count T production in pebble regions
            sigma_6 = sigma_li6_nt_macro(g, temperature_K) * f_p
            sigma_7 = sigma_li7_nnt_macro(g, temperature_K) * f_p
            tbr_li6 += sigma_6 * phi[g][i] * dx
            tbr_li7 += sigma_7 * phi[g][i] * dx
            # approximate source from first group (14 MeV neutrons)
            if g == 0:
                source_total += phi[g][i] * pc.SIGMA_T_LI6.get(g, 2.4) * dx

    tbr_total = tbr_li6 + tbr_li7
    # normalise by source
    if source_total > 0.0:
        tbr_li6_norm = tbr_li6 / source_total
        tbr_li7_norm = tbr_li7 / source_total
        tbr_total_norm = tbr_total / source_total
    else:
        tbr_li6_norm = 0.0
        tbr_li7_norm = 0.0
        tbr_total_norm = 0.0

    return {
        "tbr_total": tbr_total_norm,
        "tbr_li6": tbr_li6_norm,
        "tbr_li7": tbr_li7_norm,
        "production_rate": tbr_total,
        "source_rate": source_total,
        "self_sufficient": tbr_total_norm >= 1.05,
    }


# ---------------------------------------------------------------------------
# TBR sensitivity to Li-6 enrichment
# ---------------------------------------------------------------------------
def tbr_vs_enrichment(
    phi: List[List[float]],
    dx: float,
    pebble_fraction: List[float],
    enrichments: List[float],
    temperature_K: float = 600.0,
) -> List[Dict[str, float]]:
    """Compute TBR for a range of Li-6 enrichments.

    Parameters
    ----------
    enrichments : list of float in [0, 1]
        Li-6 atom fractions to test.
    """
    results = []
    for enr in enrichments:
        pc_local_fracs = (pc.FRAC_LI6, pc.FRAC_LI7)
        # temporarily override
        pc.FRAC_LI6 = enr
        pc.FRAC_LI7 = 1.0 - enr
        res = compute_tbr(phi, dx, pebble_fraction, temperature_K)
        res["enrichment"] = enr
        results.append(res)
        # restore
        pc.FRAC_LI6 = pc_local_fracs[0]
        pc.FRAC_LI7 = pc_local_fracs[1]
    return results


# ---------------------------------------------------------------------------
# Flow continuity check (from pgoelz_fluid)
# ---------------------------------------------------------------------------
def tritium_production_balance(
    phi: List[List[float]],
    dx: float,
    pebble_fraction: List[float],
    temperature_K: float = 600.0,
) -> Dict[str, float]:
    """Verify that the total T production equals the integral of the
    (n,t) reaction rate over the blanket volume.

    This is the *flow continuity* check: the rate of T atoms created must
    equal the rate at which they are extracted (in steady state).
    """
    res = compute_tbr(phi, dx, pebble_fraction, temperature_K)
    production = res["production_rate"]
    # check by re-integrating
    G = len(phi)
    nx = len(phi[0])
    recompute = 0.0
    for i in range(nx):
        f_p = pebble_fraction[i]
        for g in range(G):
            sigma_6 = sigma_li6_nt_macro(g, temperature_K) * f_p
            sigma_7 = sigma_li7_nnt_macro(g, temperature_K) * f_p
            recompute += (sigma_6 + sigma_7) * phi[g][i] * dx
    return {
        "production": production,
        "recomputed": recompute,
        "imbalance": abs(production - recompute),
        "consistent": abs(production - recompute) < 1.0e-10 * max(production, 1.0),
    }
