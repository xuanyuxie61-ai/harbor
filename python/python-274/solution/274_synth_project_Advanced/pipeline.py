# -*- coding: utf-8 -*-
"""
pipeline.py
-----------
Orchestration layer that glues together the thirteen scientific modules
into a single reproducible end-to-end calculation of the superconducting
transition temperature Tc from first-principles-style parameters.

Scientific origin of the fused algorithms
-----------------------------------------
* Bio-Inspired Navigation pipeline  (seed project 1057)
    -> clean separation of *configuration*, *data generators*, and
       *orchestration* via a main class that chains the stages.
    -> adapted: the pipeline has six stages
        1. Build adaptive k-point mesh (lattice_geometry)
        2. Build Brillouin zone (brillouin_zone)
        3. Construct phonon-shell spectrum (diophantine_phonon)
        4. Build the Eliashberg kernel (special_functions)
        5. Solve for Tc (ema_optimizer + tridiagonal_fd + polynomial_basis)
        6. Quantify uncertainty (monte_carlo_fluctuations + kmeans_gap_quantization)
        7. Find optimal phonon subset (graph_optimization)
        8. Self-consistent kinetic check (ion_transport_kinetics)
        9. Numerical-stability report (stability_analysis)

Core physics / mathematics
--------------------------
The pipeline implements the full workflow of a computational condensed-matter
study of a simple superconductor:

    lattice  ->  BZ  ->  phonon DOS  ->  Eliashberg kernel  ->  Tc
                                                      |
                                                      v
                                          stability + uncertainty
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1.  Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """All tunable parameters of the end-to-end Tc calculation."""
    # Lattice / BZ
    a1: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    a2: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    n_kpoints: int = 32
    cvt_density_index: int = 2      # rho(s) = s^(1/3)
    seed: int = 274

    # Phonon model
    n_einstein_branches: int = 6
    branch_multiplicities: tuple = (1, 2, 3, 4, 5, 6)
    omega_E: float = 0.025          # eV (Einstein frequency)
    max_phonon_shell: int = 50
    V_ph: float = 0.30              # eV  (pairing potential)
    mu_star: float = 0.10           # Coulomb pseudo-potential

    # Eliashberg solver
    n_matsubara: int = 64
    fd_order: int = 4               # 4 or 6
    T_search_low: float = 1.0       # K
    T_search_high: float = 20.0     # K
    n_T_search: int = 24
    ema_decay: float = 0.95

    # Gap quantisation
    n_gap_levels: int = 8

    # Monte-Carlo uncertainty
    n_mc_samples: int = 5000
    sigma_lambda: float = 0.05
    sigma_mu: float = 0.02

    # Phonon-mode selection
    n_selected_modes: int = 3


# ---------------------------------------------------------------------------
# 2.  Pipeline class
# ---------------------------------------------------------------------------
class TcPredictionPipeline:
    """End-to-end Tc prediction pipeline.

    Usage
    -----
        cfg = PipelineConfig()
        pipeline = TcPredictionPipeline(cfg)
        report = pipeline.run()
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.results: Dict[str, Any] = {}
        self._timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Stage 1: adaptive k-point mesh
    # ------------------------------------------------------------------
    def stage1_kpoint_mesh(self) -> Dict[str, Any]:
        t0 = time.time()
        from lattice_geometry import build_kpoint_mesh
        k, w, twb = build_kpoint_mesh(
            n_kpoints=self.config.n_kpoints,
            density_index=self.config.cvt_density_index,
            seed=self.config.seed,
        )
        out = dict(k_points=k, weights=w, twb_rule=twb)
        self._timings["stage1"] = time.time() - t0
        self.results["kpoint_mesh"] = out
        return out

    # ------------------------------------------------------------------
    # Stage 2: Brillouin zone
    # ------------------------------------------------------------------
    def stage2_brillouin_zone(self) -> Dict[str, Any]:
        t0 = time.time()
        from brillouin_zone import build_brillouin_zone
        bz = build_brillouin_zone(a1=self.config.a1, a2=self.config.a2)
        out = dict(bz=bz)
        self._timings["stage2"] = time.time() - t0
        self.results["brillouin_zone"] = out
        return out

    # ------------------------------------------------------------------
    # Stage 3: phonon-shell spectrum
    # ------------------------------------------------------------------
    def stage3_phonon_shells(self) -> Dict[str, Any]:
        t0 = time.time()
        from diophantine_phonon import phonon_shell_spectrum, migdal_lambda_from_shells
        spec = phonon_shell_spectrum(
            branch_multiplicities=self.config.branch_multiplicities,
            max_shell=self.config.max_phonon_shell,
        )
        lam_eff, lam_per_shell = migdal_lambda_from_shells(
            spec.degeneracy, self.config.V_ph, self.config.mu_star
        )
        out = dict(spectrum=spec, lambda_eff=lam_eff, lambda_per_shell=lam_per_shell)
        self._timings["stage3"] = time.time() - t0
        self.results["phonon_shells"] = out
        return out

    # ------------------------------------------------------------------
    # Stage 4: Eliashberg kernel
    # ------------------------------------------------------------------
    def stage4_eliashberg_kernel(self) -> Dict[str, Any]:
        t0 = time.time()
        from special_functions import build_retarded_kernel
        ker = build_retarded_kernel(
            n_matsubara=self.config.n_matsubara,
            temperature=0.5 * (self.config.T_search_low + self.config.T_search_high),
            omega_E=self.config.omega_E,
            g_ep=self.config.V_ph,
        )
        out = dict(kernel=ker)
        self._timings["stage4"] = time.time() - t0
        self.results["eliashberg_kernel"] = out
        return out

    # ------------------------------------------------------------------
    # Stage 5: spectral expansion of the gap
    # ------------------------------------------------------------------
    def stage5_gap_expansion(self) -> Dict[str, Any]:
        t0 = time.time()
        from polynomial_basis import build_gap_expansion
        exp = build_gap_expansion(n_order=12, basis="chebyshev")
        out = dict(expansion=exp)
        self._timings["stage5"] = time.time() - t0
        self.results["gap_expansion"] = out
        return out

    # ------------------------------------------------------------------
    # Stage 6: Tc search
    # ------------------------------------------------------------------
    def stage6_tc_search(self) -> Dict[str, Any]:
        t0 = time.time()
        from ema_optimizer import search_tc_by_bracket

        def kernel_at_T(T):
            from special_functions import build_retarded_kernel
            return build_retarded_kernel(
                n_matsubara=self.config.n_matsubara,
                temperature=T,
                omega_E=self.config.omega_E,
                g_ep=self.config.V_ph,
            ).K_ret

        res = search_tc_by_bracket(
            T_low=self.config.T_search_low,
            T_high=self.config.T_search_high,
            n_matsubara=self.config.n_matsubara,
            build_kernel=kernel_at_T,
            mu_star=self.config.mu_star,
            n_steps=self.config.n_T_search,
            decay=self.config.ema_decay,
        )
        out = dict(tc_result=res)
        self._timings["stage6"] = time.time() - t0
        self.results["tc_search"] = out
        return out

    # ------------------------------------------------------------------
    # Stage 7: gap quantisation
    # ------------------------------------------------------------------
    def stage7_gap_quantization(self) -> Dict[str, Any]:
        t0 = time.time()
        from kmeans_gap_quantization import quantise_gap
        tc_res = self.results["tc_search"]["tc_result"]
        # Build a model gap from the eigenvalue solver
        from ema_optimizer import search_tc_by_bracket

        def kernel_at_T(T):
            from special_functions import build_retarded_kernel
            return build_retarded_kernel(
                n_matsubara=self.config.n_matsubara,
                temperature=T,
                omega_E=self.config.omega_E,
                g_ep=self.config.V_ph,
            ).K_ret

        K = kernel_at_T(max(tc_res.Tc_estimate, 0.1))
        M = K.shape[0]
        omega = math.pi * max(tc_res.Tc_estimate, 0.1) * (2.0 * np.arange(M) + 1.0)
        # Use a trial gap  Delta_n = 1 / (2n+1)
        Delta = 1.0 / (2.0 * np.arange(M) + 1.0)
        gq = quantise_gap(omega, Delta, n_levels=self.config.n_gap_levels, seed=self.config.seed)
        out = dict(gap_quantization=gq)
        self._timings["stage7"] = time.time() - t0
        self.results["gap_quantization"] = out
        return out

    # ------------------------------------------------------------------
    # Stage 8: Monte-Carlo uncertainty
    # ------------------------------------------------------------------
    def stage8_monte_carlo(self) -> Dict[str, Any]:
        t0 = time.time()
        from monte_carlo_fluctuations import monte_carlo_tc
        lam0 = self.results["phonon_shells"]["lambda_eff"] + self.config.mu_star
        mc = monte_carlo_tc(
            lambda_0=lam0,
            sigma_lambda=self.config.sigma_lambda,
            mu_0=self.config.mu_star,
            sigma_mu=self.config.sigma_mu,
            omega_log=self.config.omega_E,
            omega_2=self.config.omega_E * 1.5,
            n_samples=self.config.n_mc_samples,
            seed=self.config.seed,
        )
        out = dict(monte_carlo=mc)
        self._timings["stage8"] = time.time() - t0
        self.results["monte_carlo"] = out
        return out

    # ------------------------------------------------------------------
    # Stage 9: optimal phonon subset
    # ------------------------------------------------------------------
    def stage9_optimal_subset(self) -> Dict[str, Any]:
        t0 = time.time()
        from graph_optimization import build_mode_graph, optimal_phonon_subset
        mg = build_mode_graph(
            n_modes=self.config.n_einstein_branches,
            frequencies=np.array(self.config.branch_multiplicities, dtype=float) * self.config.omega_E,
            coupling_strength=self.config.V_ph,
            decay_rate=0.5,
        )
        opt = optimal_phonon_subset(
            mode_graph=mg,
            budget=self.config.n_selected_modes,
            mu_star=self.config.mu_star,
            V_ph=self.config.V_ph,
        )
        out = dict(mode_graph=mg, optimal_subset=opt)
        self._timings["stage9"] = time.time() - t0
        self.results["optimal_subset"] = out
        return out

    # ------------------------------------------------------------------
    # Stage 10: kinetic steady-state check
    # ------------------------------------------------------------------
    def stage10_kinetic_check(self) -> Dict[str, Any]:
        t0 = time.time()
        from ion_transport_kinetics import (
            KineticParameters,
            sweep_kinetic_vs_temperature,
        )
        sweep = sweep_kinetic_vs_temperature(
            T_range=(1.0, 15.0),
            n_T=12,
            base_params=KineticParameters(T_c0=max(self.results["tc_search"]["tc_result"].Tc_estimate, 2.0)),
        )
        out = dict(kinetic_sweep=sweep)
        self._timings["stage10"] = time.time() - t0
        self.results["kinetic_check"] = out
        return out

    # ------------------------------------------------------------------
    # Stage 11: numerical stability report
    # ------------------------------------------------------------------
    def stage11_stability(self) -> Dict[str, Any]:
        t0 = time.time()
        from stability_analysis import (
            von_neumann_check,
            energy_method_check,
            cfl_check,
            matrix_stability_of_eliashberg,
        )
        vn = von_neumann_check(fd_order=self.config.fd_order, dx=0.1)
        em = energy_method_check(
            fd_order=self.config.fd_order,
            n=self.config.n_matsubara,
            dx=0.1,
        )
        cfl = cfl_check(fd_order=self.config.fd_order, dx=0.1)
        K = self.results["eliashberg_kernel"]["kernel"].K_ret
        ms = matrix_stability_of_eliashberg(
            T=max(self.results["tc_search"]["tc_result"].Tc_estimate, 0.1),
            n_matsubara=self.config.n_matsubara,
            lambda_kernel=K,
            mu_star=self.config.mu_star,
        )
        out = dict(von_neumann=vn, energy_method=em, cfl=cfl, matrix_stability=ms)
        self._timings["stage11"] = time.time() - t0
        self.results["stability"] = out
        return out

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Execute all 11 stages in order and return the collected results."""
        stages = [
            ("Stage 1 : adaptive k-point mesh       ", self.stage1_kpoint_mesh),
            ("Stage 2 : Brillouin zone              ", self.stage2_brillouin_zone),
            ("Stage 3 : phonon-shell spectrum       ", self.stage3_phonon_shells),
            ("Stage 4 : Eliashberg kernel           ", self.stage4_eliashberg_kernel),
            ("Stage 5 : spectral gap expansion      ", self.stage5_gap_expansion),
            ("Stage 6 : Tc search                   ", self.stage6_tc_search),
            ("Stage 7 : gap quantisation            ", self.stage7_gap_quantization),
            ("Stage 8 : Monte-Carlo uncertainty     ", self.stage8_monte_carlo),
            ("Stage 9 : optimal phonon subset       ", self.stage9_optimal_subset),
            ("Stage 10: kinetic steady-state check  ", self.stage10_kinetic_check),
            ("Stage 11: numerical-stability report  ", self.stage11_stability),
        ]
        t_all = time.time()
        for label, fn in stages:
            print(f"  [pipeline] {label} ...", end=" ", flush=True)
            try:
                fn()
                print(f"OK  ({self._timings.get(list(self._timings.keys())[-1], 0.0):.3f}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                raise
        self._timings["total"] = time.time() - t_all
        return self.results
