"""
dusty_plasma_simulator.py
==========================
Top-level simulation driver for the dusty plasma crystal.

Integrates all modules into a coherent simulation pipeline:
    1. Initialize plasma regime and crystal geometry
    2. Build the dynamical matrix (block Toeplitz)
    3. Perform eigenvalue stability analysis
    4. Compute DLW dispersion relation
    5. Benchmark with KdV soliton solutions
    6. Run Monte Carlo thermodynamics
    7. Reconstruct sheath potential from measurements
    8. Export results in standard formats

This is the unified entry point for the dusty plasma crystal
simulation package.
"""

import numpy as np
import os
import sys

# Import all modules
from dusty_plasma_physics_constants import DustyPlasmaRegime, PI
from fd_high_order_stencil import (
    fd_weights_1d, fd_laplacian_2d, fd_gradient_2d,
    apply_fd_1d, stencil_consistency_check,
)
from sparse_matrix_io import (
    HarwellBoeingMatrix, write_matrix_market, read_matrix_market,
)
from vandermonde_charge_interpolation import (
    vandermonde_matrix, bjorck_pereyra_solve,
    interpolate_dust_charge, vandermonde_condition_number,
)
from quadrature_screening_integral import (
    integrate_yukawa_over_hexagon, compute_lattice_madelung_constant,
    verify_quadrature_accuracy, get_triangle_rule,
)
from hermite_spectral_modes import (
    hermiteH_value, hermite_function_value, hermite_gauss_quadrature,
    dust_oscillation_spectrum,
)
from hexagonal_lattice_geometry import (
    HexagonalLattice, compute_arc_length, compute_arc_length_spectral,
)
from block_toeplitz_dynamical import (
    DustyPlasmaDynamicalMatrix, r8bto_mv, r8bto_sl,
)
from lagrange_potential_reconstruction import (
    lagrange_interpolation, lagrange_derivative,
    reconstruct_sheath_potential, chebyshev_nodes,
    lebesgue_constant, lagrange_basis_summary,
)
from dust_lattice_wave import DustLatticeWaveField
from kdv_soliton_benchmark import (
    kdv_parameters_dusty_plasma, kdv_exact_sech,
    kdv_residual, kdv_conserved_quantities,
)
from monte_carlo_thermodynamics import (
    DustyPlasmaMonteCarlo, melting_criterion,
)
from stability_eigenvalue_analysis import CrystalStabilityAnalyzer


class DustyPlasmaSimulation:
    """
    Complete simulation pipeline for dusty plasma crystal.
    """

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        self.results = {}

    def run(self, verbose: bool = True) -> dict:
        """
        Run the complete simulation pipeline.

        Parameters
        ----------
        verbose : bool
            If True, print progress to stdout.

        Returns
        -------
        results : dict
            All simulation results.
        """
        if verbose:
            print("=" * 72)
            print("  DUSTY PLASMA CRYSTAL SIMULATION")
            print("  High-Order Finite Difference & Stability Analysis")
            print("=" * 72)

        # Step 1: Initialize regime
        if verbose:
            print("\n[Step 1] Initializing dusty plasma regime...")
        regime = self._step1_regime(verbose)

        # Step 2: Build lattice
        if verbose:
            print("\n[Step 2] Building hexagonal lattice...")
        lattice = self._step2_lattice(regime, verbose)

        # Step 3: High-order FD validation
        if verbose:
            print("\n[Step 3] Validating high-order finite difference stencils...")
        self._step3_fd_validation(verbose)

        # Step 4: Quadrature and Madelung constant
        if verbose:
            print("\n[Step 4] Computing screening integrals and Madelung constant...")
        self._step4_quadrature(regime, verbose)

        # Step 5: Vandermonde charge interpolation
        if verbose:
            print("\n[Step 5] Vandermonde charge interpolation...")
        self._step5_vandermonde(regime, verbose)

        # Step 6: Lagrange potential reconstruction
        if verbose:
            print("\n[Step 6] Lagrange potential reconstruction...")
        self._step6_lagrange(regime, verbose)

        # Step 7: Hermite spectral modes
        if verbose:
            print("\n[Step 7] Hermite spectral mode analysis...")
        self._step7_hermite(regime, verbose)

        # Step 8: Block Toeplitz dynamical matrix
        if verbose:
            print("\n[Step 8] Building block Toeplitz dynamical matrix...")
        dyn_matrix = self._step8_dynamical_matrix(regime, verbose)

        # Step 9: Stability analysis
        if verbose:
            print("\n[Step 9] Eigenvalue stability analysis...")
        self._step9_stability(dyn_matrix, verbose)

        # Step 10: Dust lattice wave propagation
        if verbose:
            print("\n[Step 10] Dust lattice wave propagation...")
        self._step10_dlw(regime, verbose)

        # Step 11: KdV soliton benchmark
        if verbose:
            print("\n[Step 11] KdV soliton benchmark...")
        self._step11_kdv(regime, verbose)

        # Step 12: Monte Carlo thermodynamics
        if verbose:
            print("\n[Step 12] Monte Carlo thermodynamics...")
        self._step12_monte_carlo(regime, verbose)

        # Step 13: Sparse matrix I/O
        if verbose:
            print("\n[Step 13] Sparse matrix I/O test...")
        self._step13_sparse_io(dyn_matrix, verbose)

        # Step 14: Arc length computation
        if verbose:
            print("\n[Step 14] Trajectory arc length...")
        self._step14_arc_length(lattice, verbose)

        if verbose:
            print("\n" + "=" * 72)
            print("  SIMULATION COMPLETE")
            print("=" * 72)

            # Print results summary
            print("\n" + "-" * 72)
            print("  RESULTS SUMMARY")
            print("-" * 72)

            key_results = {
                "Plasma regime valid": self.results.get("regime_valid", "N/A"),
                "Crystal stable": self.results.get("stability", {}).get("is_stable", "N/A"),
                "Gamma": self.results.get("regime", {}).get("Gamma", "N/A"),
                "kappa": self.results.get("regime", {}).get("kappa", "N/A"),
                "Madelung constant": self.results.get("quadrature", {}).get("madelung_constant", "N/A"),
                "Condition number": self.results.get("stability", {}).get("condition_number", "N/A"),
                "DLW max freq": self.results.get("dlw", {}).get("omega_L_range", ["N/A", "N/A"])[1],
                "KdV residual": self.results.get("kdv", {}).get("residual_Linf", "N/A"),
                "MC acceptance": self.results.get("monte_carlo", {}).get("acceptance_rate", "N/A"),
                "FD Laplacian error": self.results.get("fd_validation", {}).get("laplacian_error", "N/A"),
            }

            for k, v in key_results.items():
                print(f"  {k:30s}: {v}")

            print("-" * 72)
            print("  All simulation steps completed successfully.")
            print("-" * 72)

        return self.results

    def _step1_regime(self, verbose: bool) -> DustyPlasmaRegime:
        """Initialize the dusty plasma regime."""
        regime = DustyPlasmaRegime(
            n_e=1.0e15,
            T_e_eV=2.5,
            T_i_eV=0.03,
            T_d_eV=0.025,
            r_d_um=0.5,    # Smaller grains to ensure physical validity
            n_d=1.0e8,     # Lower density for larger Wigner-Seitz radius
        )

        is_valid = regime.validate_regime()
        summary = regime.regime_summary()

        if verbose:
            print(f"  Plasma regime: {'VALID' if is_valid else 'INVALID'}")
            for k, v in summary.items():
                print(f"    {k:25s} = {v}")

        self.results["regime"] = summary
        self.results["regime_valid"] = is_valid
        return regime

    def _step2_lattice(
        self, regime: DustyPlasmaRegime, verbose: bool
    ) -> HexagonalLattice:
        """Build the hexagonal lattice."""
        lattice = HexagonalLattice(
            a=regime.a_ws * 1.0,
            n_cells_x=4,
            n_cells_y=4,
        )

        packing = lattice.packing_fraction(regime.r_d)
        fill_ok = lattice.fill_factor_check(regime.r_d)
        recip = lattice.get_reciprocal_lattice()

        if verbose:
            print(f"  Number of grains: {lattice.n_grains}")
            print(f"  Lattice constant: {lattice.a:.4e} m")
            print(f"  Wigner-Seitz radius: {lattice.a_ws:.4e} m")
            print(f"  Packing fraction: {packing:.6f}")
            print(f"  Fill factor valid: {fill_ok}")
            print(f"  Reciprocal lattice |b1|: {np.linalg.norm(recip[0]):.4e} 1/m")

        self.results["lattice"] = {
            "n_grains": lattice.n_grains,
            "a": lattice.a,
            "a_ws": lattice.a_ws,
            "packing_fraction": packing,
        }
        return lattice

    def _step3_fd_validation(self, verbose: bool) -> None:
        """Validate high-order FD stencils."""
        # Test 2nd-order stencil
        w2 = fd_weights_1d(1, 2, 1.0)
        # Test 4th-order stencil
        w4 = fd_weights_1d(2, 2, 1.0)
        # Test 6th-order stencil
        w6 = fd_weights_1d(3, 2, 1.0)

        if verbose:
            print(f"  Order-2 stencil (3 points): {w2}")
            print(f"  Order-4 stencil (5 points): {[f'{w:.6f}' for w in w4]}")
            print(f"  Order-6 stencil (7 points): {[f'{w:.6f}' for w in w6]}")

        # Consistency check
        result = stencil_consistency_check(1, 2)
        if verbose:
            print(f"  Convergence order (p=1, 2nd deriv): {result['observed_orders']}")
            print(f"  Expected order: {result['expected_order']}")

        # Test Laplacian on a known function
        Nx, Ny = 32, 32
        x = np.linspace(0, 2 * PI, Nx, endpoint=False)
        y = np.linspace(0, 2 * PI, Ny, endpoint=False)
        X, Y = np.meshgrid(x, y)
        phi = np.sin(X) * np.cos(Y)

        lap = fd_laplacian_2d(phi, x[1] - x[0], y[1] - y[0], order=4, boundary="periodic")
        exact_lap = -2.0 * np.sin(X) * np.cos(Y)
        error = np.max(np.abs(lap - exact_lap))

        if verbose:
            print(f"  Laplacian test (sin*cos, order 4): max error = {error:.6e}")

        self.results["fd_validation"] = {
            "laplacian_error": float(error),
            "stencil_orders": [2, 4, 6],
        }

    def _step4_quadrature(
        self, regime: DustyPlasmaRegime, verbose: bool
    ) -> None:
        """Compute quadrature and Madelung constant."""
        # Verify quadrature accuracy
        qa = verify_quadrature_accuracy(degree=5, test_exponent=3)
        if verbose:
            print(f"  Quadrature accuracy: exact={qa['exact']:.10f}, "
                  f"numerical={qa['numerical']:.10f}, error={qa['error']:.2e}")

        # Madelung constant
        madelung = compute_lattice_madelung_constant(regime.kappa, n_shells=5)
        if verbose:
            print(f"  Madelung constant (kappa={regime.kappa:.3f}): {madelung:.6f}")

        # Yukawa screening integral
        integral = integrate_yukawa_over_hexagon(
            regime.kappa, regime.a_ws, np.array([0.0, 0.0]), degree=5
        )
        if verbose:
            print(f"  Screening integral: {integral:.6e}")

        self.results["quadrature"] = {
            "madelung_constant": float(madelung),
            "screening_integral": float(integral),
            "quadrature_error": float(qa["error"]),
        }

    def _step5_vandermonde(
        self, regime: DustyPlasmaRegime, verbose: bool
    ) -> None:
        """Vandermonde charge interpolation."""
        # Create synthetic charge data
        N_nodes = 8
        positions = np.linspace(0, 1, N_nodes)
        # Simulated charge profile: Z_d(r) ~ Z_d0 * (1 + alpha*r^2)
        alpha = 0.5
        true_charges = regime.Z_d * (1.0 + alpha * positions**2)

        # Interpolate to finer grid
        eval_pos = np.linspace(0, 1, 20)
        interp_charges = interpolate_dust_charge(
            positions, true_charges, eval_pos, degree=min(7, N_nodes - 1)
        )

        # Exact values for comparison
        exact_charges = regime.Z_d * (1.0 + alpha * eval_pos**2)
        max_error = np.max(np.abs(interp_charges - exact_charges))

        # Condition number
        cond = vandermonde_condition_number(positions)

        # Bjorck-Pereyra test
        V = vandermonde_matrix(positions)
        coeffs_bp = bjorck_pereyra_solve(positions, true_charges)
        residual_bp = np.max(np.abs(V @ coeffs_bp - true_charges))

        if verbose:
            print(f"  Charge interpolation max error: {max_error:.6e}")
            print(f"  Vandermonde condition number: {cond:.2e}")
            print(f"  BP solve residual: {residual_bp:.2e}")

        self.results["vandermonde"] = {
            "interpolation_error": float(max_error),
            "condition_number": float(cond),
            "bp_residual": float(residual_bp),
        }

    def _step6_lagrange(
        self, regime: DustyPlasmaRegime, verbose: bool
    ) -> None:
        """Lagrange potential reconstruction."""
        # Create synthetic sheath potential measurements
        N_grains = 10
        z_grains = np.linspace(0.001, 0.01, N_grains)  # 1-10 mm
        # Child-Langmuir sheath: phi ~ (z/d)^(5/3)
        d_sheath = 0.01
        phi_0 = 100.0  # V
        phi_grains = phi_0 * (z_grains / d_sheath)**(5.0 / 3.0)

        z_eval = np.linspace(0.001, 0.01, 50)
        phi_interp, E_field, rho = reconstruct_sheath_potential(
            z_grains, phi_grains, z_eval, use_chebyshev=False
        )

        # Lebesgue constant
        leb = lebesgue_constant(z_grains)

        # Basis summary
        summary = lagrange_basis_summary(z_grains)

        if verbose:
            print(f"  Sheath potential range: [{phi_interp.min():.2f}, {phi_interp.max():.2f}] V")
            print(f"  Max E-field: {np.max(np.abs(E_field)):.2f} V/m")
            print(f"  Lebesgue constant: {leb:.4f}")
            print(f"  Vandermonde cond: {summary['condition_number']:.2e}")

        self.results["lagrange"] = {
            "phi_range": [float(phi_interp.min()), float(phi_interp.max())],
            "max_E_field": float(np.max(np.abs(E_field))),
            "lebesgue_constant": float(leb),
        }

    def _step7_hermite(
        self, regime: DustyPlasmaRegime, verbose: bool
    ) -> None:
        """Hermite spectral mode analysis."""
        n_modes = 8
        spectrum = dust_oscillation_spectrum(
            omega_conf=regime.omega_pd,
            n_modes=n_modes,
            temperature_ratio=0.01,
        )

        # Hermite-Gauss quadrature
        nodes, weights = hermite_gauss_quadrature(10)

        # Hermite function orthogonality check
        x_test = np.linspace(-5, 5, 200)
        psi = hermite_function_value(x_test, 5, sigma=1.0)
        # Check orthogonality: integral psi_m * psi_n dx ~ delta_{mn}
        dx = x_test[1] - x_test[0]
        overlap = np.zeros((6, 6))
        for m in range(6):
            for n in range(6):
                overlap[m, n] = np.sum(psi[:, m] * psi[:, n]) * dx

        ortho_error = np.max(np.abs(overlap - np.eye(6)))

        if verbose:
            print(f"  Mode frequencies: {spectrum['frequencies'][:4]}")
            print(f"  Populations: {spectrum['populations'][:4]}")
            print(f"  GH quadrature nodes: {nodes[:5]}")
            print(f"  Orthogonality error: {ortho_error:.4e}")

        self.results["hermite"] = {
            "n_modes": n_modes,
            "frequencies": spectrum["frequencies"][:4].tolist(),
            "populations": spectrum["populations"][:4].tolist(),
            "orthogonality_error": float(ortho_error),
        }

    def _step8_dynamical_matrix(
        self, regime: DustyPlasmaRegime, verbose: bool
    ) -> DustyPlasmaDynamicalMatrix:
        """Build the dynamical matrix."""
        dyn = DustyPlasmaDynamicalMatrix(
            kappa=regime.kappa,
            omega_pd=regime.omega_pd,
            n_cells=8,
            block_size=2,
        )

        D_dense = dyn.to_dense()

        # Dispersion relation
        q_vals, omega_vals = dyn.dispersion_relation(n_q=50)

        # Test matrix-vector product
        x_test = np.random.RandomState(42).random(D_dense.shape[0])
        y_bt = r8bto_mv(dyn.blocks, dyn.n_cells, dyn.block_size, x_test)
        y_dense = D_dense @ x_test
        mv_error = np.max(np.abs(y_bt - y_dense))

        if verbose:
            print(f"  Dynamical matrix size: {D_dense.shape}")
            print(f"  Block Toeplitz MV error: {mv_error:.2e}")
            print(f"  Dispersion: omega_L range [{omega_vals[:, 0].min():.2f}, {omega_vals[:, 0].max():.2f}]")
            print(f"  Dispersion: omega_T range [{omega_vals[:, 1].min():.2f}, {omega_vals[:, 1].max():.2f}]")

        self.results["dynamical_matrix"] = {
            "size": D_dense.shape[0],
            "mv_error": float(mv_error),
            "omega_L_max": float(omega_vals[:, 0].max()),
            "omega_T_max": float(omega_vals[:, 1].max()),
        }
        return dyn

    def _step9_stability(
        self, dyn: DustyPlasmaDynamicalMatrix, verbose: bool
    ) -> None:
        """Eigenvalue stability analysis."""
        D = dyn.to_dense()
        analyzer = CrystalStabilityAnalyzer(D)

        report = analyzer.stability_report()
        born = analyzer.born_stability_criteria_2d()

        # Power iteration
        lambda_max, v_max = analyzer.power_iteration(n_iter=200)

        # Mode decomposition of random perturbation
        rng = np.random.RandomState(42)
        perturbation = rng.random(D.shape[0])
        decomp = analyzer.mode_decomposition(perturbation)

        if verbose:
            print(f"  Crystal stable: {report['is_stable']}")
            print(f"  Negative modes: {report['n_negative_modes']}")
            print(f"  Min eigenvalue: {report['min_eigenvalue']:.6e}")
            print(f"  Max eigenvalue: {report['max_eigenvalue']:.6e}")
            print(f"  Condition number: {report['condition_number']:.2e}")
            print(f"  Born criteria satisfied: {born['all_satisfied']}")
            print(f"  Power iteration lambda_max: {lambda_max:.6e}")
            print(f"  Dominant mode: {decomp['dominant_mode']}")

        self.results["stability"] = {
            "is_stable": report["is_stable"],
            "n_negative_modes": report["n_negative_modes"],
            "min_eigenvalue": report["min_eigenvalue"],
            "max_eigenvalue": report["max_eigenvalue"],
            "condition_number": report["condition_number"],
            "born_satisfied": born["all_satisfied"],
        }

    def _step10_dlw(
        self, regime: DustyPlasmaRegime, verbose: bool
    ) -> None:
        """Dust lattice wave propagation."""
        dlw = DustLatticeWaveField(
            kappa=regime.kappa,
            omega_pd=regime.omega_pd,
            N=32,
            n_modes=16,
        )

        # Initialize wave packet
        q_center = dlw.q_values[len(dlw.q_values) // 3]
        sigma_q = (dlw.q_values[1] - dlw.q_values[0]) * 3
        dlw.initialize_wave_packet(q_center, sigma_q, amplitude=1.0)

        # Evolve
        dt = 0.01 / regime.omega_pd
        n_steps = 100
        result = dlw.evolve_time(dt, n_steps, damping=0.01)

        # Group velocity
        v_g_L, v_g_T = dlw.group_velocity()

        # Density of states
        omega_bins, g_L, g_T = dlw.density_of_states(n_bins=20)

        # Check energy conservation (should be nearly constant for small damping)
        E_initial = result["energy_history"][0]
        E_final = result["energy_history"][-1]
        E_drift = abs(E_final - E_initial) / max(abs(E_initial), 1e-30)

        if verbose:
            print(f"  DLW modes: {dlw.n_modes}")
            print(f"  omega_L range: [{dlw.omega_L.min():.2f}, {dlw.omega_L.max():.2f}]")
            print(f"  Max group velocity (L): {np.max(np.abs(v_g_L)):.4e}")
            print(f"  Energy drift: {E_drift:.2e}")

        self.results["dlw"] = {
            "n_modes": dlw.n_modes,
            "omega_L_range": [float(dlw.omega_L.min()), float(dlw.omega_L.max())],
            "energy_drift": float(E_drift),
            "max_group_velocity": float(np.max(np.abs(v_g_L))),
        }

    def _step11_kdv(
        self, regime: DustyPlasmaRegime, verbose: bool
    ) -> None:
        """KdV soliton benchmark."""
        params = kdv_parameters_dusty_plasma(
            kappa=regime.kappa,
            omega_pd=regime.omega_pd,
            C_DA=regime.C_DA,
            lambda_D=regime.lambda_D,
        )

        # Soliton on a grid
        Nx = 200
        x = np.linspace(-0.01, 0.01, Nx)
        amplitude = 0.5  # V

        phi, phi_x, phi_t = kdv_exact_sech(x, 0.0, params, amplitude, 0.0)

        # Residual check
        residual = kdv_residual(x, 0.0, params, amplitude, 0.0)

        # Conserved quantities
        invariants = kdv_conserved_quantities(x, phi)

        if verbose:
            print(f"  KdV A (nonlinearity): {params['A']:.4e}")
            print(f"  KdV B (dispersion): {params['B']:.4e}")
            print(f"  Soliton amplitude: {np.max(phi):.4f} V")
            print(f"  KdV residual L_inf: {residual['L_inf']:.4e}")
            print(f"  KdV residual L2: {residual['L2']:.4e}")
            print(f"  I1 (mass): {invariants['I1']:.6e}")
            print(f"  I2 (energy): {invariants['I2']:.6e}")

        self.results["kdv"] = {
            "A": params["A"],
            "B": params["B"],
            "residual_Linf": residual["L_inf"],
            "residual_L2": residual["L2"],
            "invariants": invariants,
        }

    def _step12_monte_carlo(
        self, regime: DustyPlasmaRegime, verbose: bool
    ) -> None:
        """Monte Carlo thermodynamics."""
        mc = DustyPlasmaMonteCarlo(
            N=16,
            kappa=regime.kappa,
            Gamma=min(regime.Gamma, 200.0),  # Cap for speed
            box_size=5.0,
            seed=42,
        )

        mc_results = mc.run_simulation(
            n_steps=2000,
            n_equil=500,
            max_displacement=0.1,
        )

        # Pair correlation
        r_bins, g_r = mc.pair_correlation(n_bins=20)

        # Melting criterion
        melt = melting_criterion(regime.kappa)

        if verbose:
            print(f"  MC energy: {mc_results['mean_energy']:.4f}")
            print(f"  MC specific heat: {mc_results['specific_heat']:.4f}")
            print(f"  MC acceptance rate: {mc_results['acceptance_rate']:.3f}")
            print(f"  Lindemann ratio: {mc_results['lindemann']:.4f}")
            print(f"  Melting Gamma: {melt['Gamma_melt']:.1f}")
            print(f"  Phase: {regime.regime_summary()['Crystal regime']}")

        self.results["monte_carlo"] = {
            "mean_energy": mc_results["mean_energy"],
            "specific_heat": mc_results["specific_heat"],
            "acceptance_rate": mc_results["acceptance_rate"],
            "lindemann": mc_results["lindemann"],
            "melting_Gamma": melt["Gamma_melt"],
        }

    def _step13_sparse_io(
        self, dyn: DustyPlasmaDynamicalMatrix, verbose: bool
    ) -> None:
        """Test sparse matrix I/O."""
        D = dyn.to_dense()

        # Harwell-Boeing format
        hb = HarwellBoeingMatrix.from_scipy_sparse(D, title="DustyPlasmaDynamical")
        hb_path = os.path.join(self.output_dir, "dynamical_matrix.hb")
        hb.write_hb(hb_path)

        # Read back
        hb_read = HarwellBoeingMatrix.read_hb(hb_path)
        D_read = hb_read.to_dense()
        hb_error = np.max(np.abs(D - D_read))

        # Matrix Market format
        mm_path = os.path.join(self.output_dir, "dynamical_matrix.mm")
        write_matrix_market(mm_path, D, comment="Dusty plasma dynamical matrix")

        # Read back
        D_mm, mm_info = read_matrix_market(mm_path)
        mm_error = np.max(np.abs(D - D_mm))

        # Cleanup temp files
        for path in [hb_path, mm_path]:
            if os.path.exists(path):
                os.remove(path)

        if verbose:
            print(f"  HB write/read roundtrip error: {hb_error:.2e}")
            print(f"  MM write/read roundtrip error: {mm_error:.2e}")

        self.results["sparse_io"] = {
            "hb_error": float(hb_error),
            "mm_error": float(mm_error),
        }

    def _step14_arc_length(
        self, lattice: HexagonalLattice, verbose: bool
    ) -> None:
        """Compute trajectory arc lengths."""
        # Simulate a grain trajectory (Lissajous-like)
        N_points = 100
        t = np.linspace(0, 2 * PI, N_points)
        # Circular orbit
        r = 0.3 * lattice.a
        trajectory = np.column_stack([
            r * np.cos(t) + lattice.positions[0, 0],
            r * np.sin(t) + lattice.positions[0, 1],
        ])

        arc_len = compute_arc_length(trajectory)
        exact_circumference = 2 * PI * r

        # Spectral arc length test
        def x_func(t): return r * np.cos(t) + lattice.positions[0, 0]
        def y_func(t): return r * np.sin(t) + lattice.positions[0, 1]
        def dx_func(t): return -r * np.sin(t)
        def dy_func(t): return r * np.cos(t)

        arc_len_spectral = compute_arc_length_spectral(
            x_func, y_func, dx_func, dy_func, 0, 2 * PI, n_points=200
        )

        if verbose:
            print(f"  Discrete arc length: {arc_len:.6f}")
            print(f"  Spectral arc length: {arc_len_spectral:.6f}")
            print(f"  Exact circumference: {exact_circumference:.6f}")
            print(f"  Discrete error: {abs(arc_len - exact_circumference):.2e}")
            print(f"  Spectral error: {abs(arc_len_spectral - exact_circumference):.2e}")

        self.results["arc_length"] = {
            "discrete": float(arc_len),
            "spectral": float(arc_len_spectral),
            "exact": float(exact_circumference),
        }


def main():
    """
    Main entry point: run the dusty plasma crystal simulation.

    Zero-parameter execution: simply run `python main.py`.
    """
    sim = DustyPlasmaSimulation(output_dir=".")
    results = sim.run(verbose=True)

    # Print summary
    print("\n" + "-" * 72)
    print("  RESULTS SUMMARY")
    print("-" * 72)

    key_results = {
        "Plasma regime valid": results.get("regime_valid", "N/A"),
        "Crystal stable": results.get("stability", {}).get("is_stable", "N/A"),
        "Gamma": results.get("regime", {}).get("Gamma", "N/A"),
        "kappa": results.get("regime", {}).get("kappa", "N/A"),
        "Madelung constant": results.get("quadrature", {}).get("madelung_constant", "N/A"),
        "Condition number": results.get("stability", {}).get("condition_number", "N/A"),
        "DLW max freq": results.get("dlw", {}).get("omega_L_range", ["N/A", "N/A"])[1],
        "KdV residual": results.get("kdv", {}).get("residual_Linf", "N/A"),
        "MC acceptance": results.get("monte_carlo", {}).get("acceptance_rate", "N/A"),
        "FD Laplacian error": results.get("fd_validation", {}).get("laplacian_error", "N/A"),
    }

    for k, v in key_results.items():
        print(f"  {k:30s}: {v}")

    print("-" * 72)
    print("  All simulation steps completed successfully.")
    print("-" * 72)

    return results


if __name__ == "__main__":
    main()
