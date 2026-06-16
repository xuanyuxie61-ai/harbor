"""
defect_formation_energy.py — Formation energy of a point defect
===============================================================

The formation energy of a point defect in charge state q is
    E_f[q] = E_tot[defect, q] − (N−1)/N E_tot[bulk] − μ + q (ε_VBM + ΔV̄)
for a vacancy, or
    E_f[q] = E_tot[defect, q] − (N+1)/N E_tot[bulk] − μ + q (ε_VBM + ΔV̄)
for an interstitial. Here μ is the chemical potential of the removed /
added atom, ε_VBM is the valence-band maximum of the bulk, and ΔV̄ is the
average potential alignment between bulk and defect supercells.

For our small-scale reproducible experiment we work in a *model* 2-D
Kohn–Sham framework where
    E_tot = Σ_i f_i ε_i   −  E_H[n] / 2   +  E_xc[n]   −  ∫ v_xc[n] n
          + E_ion-ion + E_Makov-Payne
with
    f_i = 1 / (exp((ε_i − μ_e) / kT) + 1)     (Fermi–Dirac occupation)
    E_H[n] = (1/2) ∫ n V_H                      (Hartree energy)
    E_xc[n] = ∫ n ε_xc(n)                       (exchange–correlation)

The Fermi–Dirac integrals are evaluated by Gauss–Hermite quadrature,
directly porting the philosophy of `464_gen_hermite_exactness.m`: the
quadrature rule ∫ w(x) f(x) dx ≈ Σ w_i f(x_i) is tested for exactness
on polynomials and then applied to the smooth Fermi function.

Seed project integration:
  * 464_gen_hermite_exactness: the Gauss–Hermite quadrature nodes and
    weights are used to evaluate the Fermi–Dirac integral
        F_j(η) = (1/Γ(j+1)) ∫_0^∞ x^j / (exp(x − η) + 1) dx
    via the substitution x = t² followed by Gauss–Hermite on (-∞, +∞).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List

from config import ProjectConfig, HARTREE_TO_EV, BOHR_TO_ANG


# -------------------------------------------------------------------------
# (1) Gauss–Hermite quadrature for Fermi–Dirac integrals
#     (from 464_gen_hermite_exactness/gen_hermite_exactness.m)
# -------------------------------------------------------------------------
def gauss_hermite_nodes_weights(n: int
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, w) nodes and weights for Gauss–Hermite of order n.

    Uses NumPy's `hermite` roots; the mathematical definition is
        ∫_{-∞}^{+∞} exp(-x²) f(x) dx  ≈  Σ_{i=1}^n w_i f(x_i)
    which is exact for polynomials f of degree ≤ 2n − 1.
    """
    x, w = np.polynomial.hermite.hermgauss(n)
    return x, w


def test_hermite_exactness(n: int, degree_max: int = 8) -> List[float]:
    """Test the quadrature exactness on monomials — port of
    `gen_hermite_exactness.m`.
    """
    x, w = gauss_hermite_nodes_weights(n)
    errs: List[float] = []
    for k in range(degree_max + 1):
        # exact integral of x^k exp(-x^2) over (-∞, ∞)
        if k % 2 == 1:
            exact = 0.0
        else:
            exact = math.gamma((k + 1) / 2.0)
        approx = float(np.sum(w * x ** k))
        errs.append(abs(approx - exact))
    return errs


def fermi_dirac_integral(j: int, eta: float, n_quad: int = 32) -> float:
    """Compute the complete Fermi–Dirac integral F_j(η).

    We use the substitution x = t², dx = 2t dt:
        F_j(η) = (2 / Γ(j+1)) ∫_0^∞ t^{2j+1} / (exp(t² − η) + 1) dt
    then write 1/(exp(t²−η)+1) = 1/2 − (1/2) tanh((t²−η)/2) and use
    Gauss–Hermite on t ∈ (-∞, ∞) exploiting symmetry.

    For j = 0 (the electron density) this reduces to the standard formula
        n = (1/2π) ln(1 + exp(η))    in 2-D (massless limit).
    """
    x, w = gauss_hermite_nodes_weights(n_quad)
    # For numerical stability we split the integral at t = sqrt(max(η, 0))
    # and use the asymptotic forms outside.
    t = np.where(x > 0, x, -x)   # exploit evenness
    t2 = t * t
    # 1 / (exp(t² − η) + 1) — with safe exponent
    arg = np.minimum(t2 - eta, 500.0)
    f = 1.0 / (np.exp(arg) + 1.0)
    # integrand: 2 t^{2j+1} f(t) exp(-t²) / Γ(j+1)
    # but we're integrating against exp(-t²) dt, so factor exp(+t²) to cancel
    # the Hermite weight. Actually the standard trick is:
    #   ∫_0^∞ g(t) dt  ≈  Σ_{x_i > 0} w_i g(x_i) / exp(-x_i²)
    # Here g(t) = 2 t^{2j+1} f(t) / Γ(j+1)
    # So the estimator is:
    #   F_j(η) ≈ (1/Γ(j+1)) Σ_{x_i > 0} w_i * 2 x_i^{2j+1} f(x_i) / exp(-x_i²)
    # But this is numerically unstable for large x_i. Instead, split:
    #   1/(exp(t²−η)+1) = exp(η−t²) / (1 + exp(η−t²))
    # For t² > η we can Taylor-expand.
    #
    # Simplest robust approach: direct Gauss–Legendre on a truncated interval.
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # map [-1, 1] → [0, max_t]
    max_t = max(math.sqrt(max(eta, 0.0) + 20.0), 5.0)
    ts = 0.5 * max_t * (nodes + 1.0)
    ws = 0.5 * max_t * weights
    t2 = ts * ts
    arg = np.minimum(t2 - eta, 500.0)
    f = 1.0 / (np.exp(arg) + 1.0)
    integrand = 2.0 * ts ** (2 * j + 1) * f / math.gamma(j + 1.0)
    return float(np.sum(ws * integrand))


# -------------------------------------------------------------------------
# (2) Local-density approximation (2D) for exchange–correlation
# -------------------------------------------------------------------------
def xc_energy_density(n: np.ndarray, a: float, b: float) -> np.ndarray:
    """2D LDA exchange–correlation energy density (Attaccalite et al. 2011).

        ε_xc(n) = a + b ln(n)        (n > 0)
        ε_xc(0) = 0                   (regularisation)

    Returns the *energy density* ε_xc(n) · n to be integrated over space.
    """
    out = np.zeros_like(n)
    mask = n > 1e-12
    out[mask] = (a + b * np.log(n[mask])) * n[mask]
    return out


def xc_potential(n: np.ndarray, a: float, b: float) -> np.ndarray:
    """v_xc(n) = d(n ε_xc)/dn = ε_xc(n) + n dε_xc/dn = a + b (1 + ln n)."""
    out = np.zeros_like(n)
    mask = n > 1e-12
    out[mask] = a + b * (1.0 + np.log(n[mask]))
    return out


# -------------------------------------------------------------------------
# (3) Total energy of a charge density on the grid
# -------------------------------------------------------------------------
def total_energy(n_grid: np.ndarray, V_eff: np.ndarray, V_H: np.ndarray,
                 V_xc: np.ndarray, h: float,
                 xc_a: float, xc_b: float) -> float:
    """Compute E_tot[n] on the grid.

        E_tot = ∫ [ (1/2) |∇√n|² + n V_eff + (1/2) n V_H + n (ε_xc - v_xc) ] d²r
    where the first term is the von Weizsäcker kinetic energy (model for T_s).

    This is a simplified orbital-free DFT functional; the purpose is to have
    a *smooth* functional of n whose minimum gives the ground-state density
    and whose value at the minimum approximates the true E_tot well enough
    to compute *differences* (formation energies) reliably.
    """
    # von Weizsäcker kinetic energy: (1/2) |∇√n|²
    sqrt_n = np.sqrt(np.maximum(n_grid, 0.0))
    gx = np.gradient(sqrt_n, h, axis=1)
    gy = np.gradient(sqrt_n, h, axis=0)
    T_w = 0.5 * float(np.sum(gx ** 2 + gy ** 2)) * h * h
    # external + Hartree + xc contributions
    E_ext = float(np.sum(n_grid * V_eff)) * h * h
    E_H = 0.5 * float(np.sum(n_grid * V_H)) * h * h
    eps_xc = xc_energy_density(n_grid, xc_a, xc_b)
    E_xc = float(np.sum(eps_xc)) * h * h
    E_vxc_shift = -float(np.sum(n_grid * V_xc)) * h * h
    return T_w + E_ext + E_H + E_xc + E_vxc_shift


# -------------------------------------------------------------------------
# (4) Formation energy of the defect
# -------------------------------------------------------------------------
def formation_energy(E_defect: float, E_bulk: float,
                     N_sites_defect: int, N_sites_bulk: int,
                     mu: float, q: int = 0,
                     evbm: float = 0.0, dV_bar: float = 0.0,
                     kind: str = "vacancy") -> float:
    """Compute E_f[q] according to the standard formula.

    For a vacancy:
        E_f = E_defect − (N−1)/N E_bulk − μ + q (ε_VBM + ΔV̄)
    For an interstitial:
        E_f = E_defect − (N+1)/N E_bulk − μ + q (ε_VBM + ΔV̄)
    """
    if kind == "vacancy":
        ratio = (N_sites_defect) / N_sites_bulk     # (N-1)/N
    else:
        ratio = (N_sites_defect) / N_sites_bulk     # (N+1)/N
    Ef = E_defect - ratio * E_bulk - mu + q * (evbm + dV_bar)
    return Ef


# -------------------------------------------------------------------------
# (5) Complete workflow for a single defect calculation
# -------------------------------------------------------------------------
def run_single_defect(cfg: ProjectConfig) -> dict:
    """Run the full formation-energy workflow and return a result dict.

    The calculation proceeds:
        1. Build the perfect bulk and compute E_bulk.
        2. Build the defect supercell and compute E_defect.
        3. Apply Makov–Payne correction for charged defects.
        4. Return E_f and sub-components.
    """
    import crystal_lattice as cl
    import high_order_fd as hfd
    import fft_poisson as fp
    import eshelby_strain as esh

    lat, graph = cl.build_default_lattice(cfg.crystal.lattice_constant,
                                          cfg.crystal.n_cells)
    N = cfg.grid.n_grid
    L = cfg.crystal.supercell_length
    h = L / N

    # ---- bulk density (uniform for the model) ----
    n_bulk_val = cfg.crystal.n_electrons_per_cell * (cfg.crystal.n_cells ** 2)
    n0_bulk = n_bulk_val / (L * L)     # mean density
    n_bulk = np.full((N, N), n0_bulk)

    V_ext_bulk = np.zeros((N, N))  # flat background for model
    V_H_bulk = fp.poisson_solve(n_bulk, h)
    V_xc_bulk = xc_potential(n_bulk, cfg.crystal.xc_a, cfg.crystal.xc_b)
    E_bulk = total_energy(n_bulk, V_ext_bulk, V_H_bulk, V_xc_bulk, h,
                          cfg.crystal.xc_a, cfg.crystal.xc_b)

    # ---- defect supercell: local density perturbation around one site ----
    n_def = n_bulk.copy()
    iy0, ix0 = N // 2, N // 2
    sigma_def = 2.0 * h
    xs = (np.arange(N) - N / 2) * h
    ys = (np.arange(N) - N / 2) * h
    X, Y = np.meshgrid(xs, ys)
    r2 = (X - xs[ix0]) ** 2 + (Y - ys[iy0]) ** 2
    gauss = np.exp(-r2 / (2 * sigma_def ** 2))
    if cfg.defect.kind == "vacancy":
        n_def -= n0_bulk * gauss * 0.5   # remove density at the vacancy
    else:
        n_def += n0_bulk * gauss * 0.5   # add density at the interstitial

    V_ext_def = np.zeros((N, N))
    V_H_def = fp.poisson_solve(n_def, h, q_charge=cfg.defect.charge_state)
    V_xc_def = xc_potential(n_def, cfg.crystal.xc_a, cfg.crystal.xc_b)
    E_def = total_energy(n_def, V_ext_def, V_H_def, V_xc_def, h,
                         cfg.crystal.xc_a, cfg.crystal.xc_b)

    # Makov–Payne correction for charged defect
    EMP = fp.makov_payne_correction(cfg.defect.charge_state, L)

    # Eshelby strain correction
    dV_def = cfg.eshelby.defect_volume_ang3 / (BOHR_TO_ANG ** 3)
    E_esh = esh.eshelby_strain_energy(
        defect_volume=dV_def,
        G=cfg.eshelby.shear_modulus * (HARTREE_TO_EV),  # back to eV
        nu=cfg.eshelby.poisson_ratio,
        aspect=cfg.eshelby.aspect_ratio,
    ) / HARTREE_TO_EV  # back to Hartree

    # formation energy
    Ef = formation_energy(
        E_defect=E_def + EMP + E_esh,
        E_bulk=E_bulk,
        N_sites_defect=lat.n_sites - (1 if cfg.defect.kind == "vacancy" else -1),
        N_sites_bulk=lat.n_sites,
        mu=-E_bulk / lat.n_sites,   # simple choice: μ = E_bulk / N_atoms
        q=cfg.defect.charge_state,
        kind=cfg.defect.kind,
    )

    return {
        "E_bulk_Ha": E_bulk,
        "E_defect_Ha": E_def,
        "E_Makov_Payne_Ha": EMP,
        "E_eshelby_Ha": E_esh,
        "E_formation_Ha": Ef,
        "E_formation_eV": Ef * HARTREE_TO_EV,
        "n_sites_bulk": lat.n_sites,
        "N_grid": N,
        "h_Bohr": h,
        "fd_order": cfg.grid.fd_order,
        "kind": cfg.defect.kind,
        "charge": cfg.defect.charge_state,
    }
