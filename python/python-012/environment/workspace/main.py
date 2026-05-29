"""
Topological Insulator Surface State Transport: Main Simulation
===============================================================

This is the unified entry point for the quantum transport simulation
of 3D topological insulator surface states with magnetic doping and disorder.

Scientific Problem:
-------------------
Compute the anomalous Hall conductivity, spin Hall conductivity, and
longitudinal conductivity of magnetically doped topological insulator
surface states (e.g., Cr-doped (Bi,Sb)2Te3), including:

1. Gapped Dirac cone band structure with hexagonal warping
2. Berry curvature and intrinsic anomalous Hall effect
3. Disorder scattering (Born approximation, self-consistent T-matrix)
4. Skew scattering and side-jump contributions
5. Finite-size tight-binding verification
6. Thermoelectric coefficients

The code runs with zero parameters and prints all results to stdout
and output files.
"""

import numpy as np
import os
import sys

# Ensure all modules can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dirac_surface import DiracSurfaceHamiltonian, effective_mass_tensor
from berry_curvature import BerryCurvatureCalculator
from disorder_scattering import DisorderScattering
from kubo_conductivity import KuboConductivity
from tight_binding_surface import TightBindingSurface
from spectral_integrator import LatticeIntegrator, JacobiQuadrature, MonteCarloIntegrator
from fermi_surface import FermiSurface
from nonlinear_solver import NonlinearSolver
from utils_special import TrigammaFunction, CarlsonEllipticIntegrals, Interpolation2D
from geometry_utils import SampleGeometry
from io_manager import IOManager


def run_simulation():
    """
    Execute the full transport simulation pipeline.
    """
    print("=" * 70)
    print("  Topological Insulator Surface State Transport Simulation")
    print("  Quantum Anomalous Hall Effect & Spin Transport")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    io = IOManager(output_dir)

    # =====================================================================
    # Section 1: Material Parameters and Hamiltonian
    # =====================================================================
    print("\n[1] Initializing Dirac Surface Hamiltonian...")

    # Parameters for Cr-doped (Bi,Sb)2Te3 (QAH insulator)
    v_F = 4.5e5          # m/s
    Delta = 0.025        # eV (exchange gap from magnetic doping)
    lambda_w = 25.0      # eV·nm^3 (hexagonal warping for Bi2Te3)

    H = DiracSurfaceHamiltonian(v_F=v_F, Delta=Delta, lambda_w=lambda_w)

    params = {
        "Fermi_velocity_vF_m/s": v_F,
        "Exchange_gap_Delta_eV": Delta,
        "Hex_warping_lambda_eVnm3": lambda_w,
    }
    io.write_simulation_header("simulation_params.dat", params)
    print(f"  v_F = {v_F:.2e} m/s")
    print(f"  Delta = {Delta:.3f} eV")
    print(f"  lambda_w = {lambda_w:.1f} eV·nm^3")

    # =====================================================================
    # Section 2: Fermi Surface Geometry
    # =====================================================================
    print("\n[2] Computing Fermi Surface Geometry...")

    E_F = 0.12  # eV, Fermi level above the gap
    fs = FermiSurface(hamiltonian=H, E_F=E_F)

    k_F = fs.fermi_wavevector()
    v_F_star = fs.fermi_velocity()
    n_carrier = fs.carrier_density()
    m_c = fs.cyclotron_mass()
    omega_c = fs.cyclotron_frequency(B=1.0)

    print(f"  E_F = {E_F:.3f} eV")
    print(f"  k_F = {k_F:.4e} 1/m")
    print(f"  v_F* = {v_F_star:.4e} m/s")
    print(f"  Carrier density n = {n_carrier:.4e} m^-2")
    print(f"  Cyclotron mass m_c = {m_c:.4e} kg")
    print(f"  Cyclotron frequency (B=1T) = {omega_c:.4e} rad/s")

    # Test ellipsoid Fermi surface (project 332 connection)
    kx_ell, ky_ell = fs.sample_momentum_ellipsoid(a_ratio=1.0, b_ratio=1.15, n_samples=100)
    print(f"  Ellipsoidal FS anisotropy: b/a = 1.15")

    # Chord length distributions (projects 178, 230)
    chord_circle = fs.chord_length_distribution_circle(n_samples=2000)
    mean_chord = np.mean(chord_circle)
    print(f"  Mean chord length on circular FS = {mean_chord:.4e} 1/m")

    # =====================================================================
    # Section 3: Berry Curvature and Anomalous Hall Effect
    # =====================================================================
    print("\n[3] Computing Berry Curvature and Chern Number...")

    berry = BerryCurvatureCalculator(H)

    # Compute Berry curvature at a few points
    k_test = np.linspace(0.0, 1e9, 5)
    for k in k_test:
        omega = berry.berry_curvature_analytical(k, 0.0, band='lower')
        print(f"  Omega({k:.2e}, 0) = {omega:.4e} m^2")

    # Chern number
    C = berry.chern_number(k_max=5e9, n_k=300, method='analytical')
    print(f"  Chern number C = {C:.4f} (expected ≈ -0.5 for lower band)")

    # Berry phase around a loop
    theta_loop = np.linspace(0.0, 2.0 * np.pi, 100)
    k_path = np.column_stack((1e9 * np.cos(theta_loop), 1e9 * np.sin(theta_loop)))
    gamma = berry.berry_phase_1d(k_path)
    print(f"  Berry phase (loop at k=1e9) = {gamma:.4f} rad")

    # =====================================================================
    # Section 4: Disorder Scattering
    # =====================================================================
    print("\n[4] Computing Disorder Scattering Rates...")

    n_imp = 5e14       # m^-2
    V0 = 0.3           # eV·nm^2
    disorder = DisorderScattering(
        hamiltonian=H, n_imp=n_imp, V0=V0, disorder_type='delta'
    )

    E_F_J = E_F * 1.602176634e-19
    rate_born = disorder.born_scattering_rate(E_F_J, n_k=200)
    tau_tr = disorder.transport_scattering_time(E_F_J, n_k=200)
    l_mfp = disorder.mean_free_path(E_F_J)
    D_diff = disorder.diffusivity(E_F_J)
    skew_rate = disorder.skew_scattering_rate(E_F_J, n_k=200)

    print(f"  Impurity concentration n_i = {n_imp:.2e} m^-2")
    print(f"  Born scattering rate = {rate_born:.4e} 1/s")
    print(f"  Transport scattering time tau_tr = {tau_tr:.4e} s")
    print(f"  Mean free path l_mfp = {l_mfp:.4e} m")
    print(f"  Diffusion constant D = {D_diff:.4e} m^2/s")
    print(f"  Skew scattering rate = {skew_rate:.4e} 1/s")

    # Self-energy
    sigma_re, sigma_im = disorder.self_energy_born(E_F_J, n_k=300)
    print(f"  Self-energy Re[Sigma] = {sigma_re:.4e} J")
    print(f"  Self-energy Im[Sigma] = {sigma_im:.4e} J")

    # =====================================================================
    # Section 5: Transport Coefficients (Kubo Formula)
    # =====================================================================
    print("\n[5] Computing Transport Coefficients...")

    # TODO HOLE 3: Instantiate KuboConductivity and compute transport coefficients.
    # Required steps:
    #   1. Create kubo = KuboConductivity(H, disorder)
    #   2. Compute sigma_xx = kubo.dc_conductivity_semicalassical(E_F)
    #   3. Compute sigma_ah = kubo.intrinsic_anomalous_hall(E_F, n_k=300, k_max=2e10)
    #   4. Compute sigma_total, sigma_int, sigma_skew, sigma_sj from
    #      kubo.total_hall_conductivity(E_F, n_k=300, k_max=2e10)
    #   5. Compute sigma_spin = kubo.spin_hall_conductivity(E_F, n_k=300, k_max=2e10)
    #   6. Compute thermoelectric S, L = kubo.thermoelectric_coefficients(E_F, T=10.0)
    #   7. Print all results (use e2_over_h = E_CHARGE^2 / H_PLANCK for unit conversion)
    #   8. Assign all variables so they are available for the results dict below.
    raise NotImplementedError("HOLE 3: Kubo transport calculation in main not implemented")

    # =====================================================================
    # Section 6: Tight-Binding Finite-Size Verification
    # =====================================================================
    print("\n[6] Tight-Binding Lattice Model (Finite Size)...")

    tb = TightBindingSurface(Nx=20, Ny=20, a=2.0, v_F=v_F, Delta=Delta,
                              boundary='open')
    energies_tb, evecs_tb = tb.diagonalize()

    print(f"  Lattice size: {tb.Nx} x {tb.Ny}")
    print(f"  Total states: {tb.N_states}")
    print(f"  Energy range: [{np.min(energies_tb)/1.602176634e-19:.4f}, "
          f"{np.max(energies_tb)/1.602176634e-19:.4f}] eV")

    # Check for edge states
    profile, edge_weight = tb.edge_state_probability(evecs_tb, band_index=0)
    print(f"  Edge state weight (lowest band) = {edge_weight:.4f}")

    # Finite-size conductivity
    sigma_tb = tb.finite_size_conductivity(E_F, T=0.0, eta=1e-22)
    print(f"  Finite-size conductivity = {sigma_tb:.4e} S")

    # =====================================================================
    # Section 7: Spectral Integration Tests
    # =====================================================================
    print("\n[7] Spectral Integration Methods...")

    lat = LatticeIntegrator(dim=2)
    jq = JacobiQuadrature()
    mc = MonteCarloIntegrator(seed=42)

    # Test Fibonacci lattice rule: integrate k^2 over circular FS
    def test_func(x):
        kx = x[0] * 2e10 - 1e10
        ky = x[1] * 2e10 - 1e10
        return (kx ** 2 + ky ** 2) * np.exp(-(kx ** 2 + ky ** 2) / (1e10 ** 2))

    q_fib = lat.fibonacci_lattice_rule(8, test_func, bounds=[(0.0, 1.0), (0.0, 1.0)])
    print(f"  Fibonacci lattice integral = {q_fib:.4e}")

    # Test Jacobi quadrature: DOS integral
    dos_func = lambda e: max(0.0, abs(e)) / (2.0 * np.pi * (H.hbar * H.v_F) ** 2)
    q_jac = jq.integrate(dos_func, n=32, alpha=0.0, beta=0.0,
                         a=-0.2 * 1.602176634e-19, b=0.2 * 1.602176634e-19)
    print(f"  Gauss-Jacobi DOS integral = {q_jac:.4e}")

    # Monte Carlo over BZ
    def mc_func(kx, ky):
        return np.exp(-(kx ** 2 + ky ** 2) / (1e10 ** 2))
    q_mc, err_mc = mc.integrate_2d_brillouin_zone(mc_func, k_max=2e10, n_samples=10000)
    print(f"  Monte Carlo BZ integral = {q_mc:.4e} ± {err_mc:.4e}")

    # =====================================================================
    # Section 8: Self-Consistent Nonlinear Solver
    # =====================================================================
    print("\n[8] Self-Consistent Nonlinear Solutions...")

    solver = NonlinearSolver(max_iter=100, tol=1e-10)

    # Example: find root of f(x) = x^2 - 2
    f_test = lambda x: x ** 2 - 2.0
    root_snyder, it_snyder = solver.snyder_method(f_test, 1.0, 2.0)
    print(f"  Snyder method: sqrt(2) = {root_snyder:.10f} (iter={it_snyder})")

    # Find Fermi level from carrier density
    target_n = 2e16  # m^-2
    E_F_solved = solver.find_fermi_level(
        carrier_density=target_n, temperature=0.0, hamiltonian=H, method='snyder'
    )
    print(f"  Fermi level for n={target_n:.2e} m^-2: E_F = {E_F_solved:.4f} eV")

    # Self-consistent scattering time
    tau_sc = solver.self_consistent_scattering_time(E_F, disorder)
    print(f"  Self-consistent tau = {tau_sc:.4e} s")

    # =====================================================================
    # Section 9: Special Functions
    # =====================================================================
    print("\n[9] Special Function Evaluations...")

    tg = TrigammaFunction()
    val_tg, _ = tg.evaluate(1.0)
    print(f"  Trigamma(1) = {val_tg:.10f} (expected pi^2/6 = {np.pi**2/6:.10f})")

    # Carlson elliptic integrals for anisotropic Fermi surface
    carlson = CarlsonEllipticIntegrals(errtol=1e-6)
    rf_val, _ = carlson.rf(1.0, 2.0, 3.0)
    rd_val, _ = carlson.rd(1.0, 2.0, 3.0)
    print(f"  R_F(1,2,3) = {rf_val:.10f}")
    print(f"  R_D(1,2,3) = {rd_val:.10f}")

    # Ellipsoid surface area
    area_ell = carlson.ellipsoid_surface_area(3.0, 2.0, 1.0)
    print(f"  Ellipsoid area (a=3,b=2,c=1) = {area_ell:.4f}")

    # 2D interpolation test
    points = np.random.rand(50, 2) * 100
    values = np.sin(points[:, 0] / 10.0) * np.cos(points[:, 1] / 10.0)
    interp = Interpolation2D(points, values)
    val_idw = float(interp.inverse_distance_weighting(50.0, 50.0).flat[0])
    val_rbf = float(interp.radial_basis_function(50.0, 50.0, epsilon=0.1).flat[0])
    val_exact = float(np.sin(5.0) * np.cos(5.0))
    print(f"  IDW interpolation at (50,50) = {val_idw:.6f}")
    print(f"  RBF interpolation at (50,50) = {val_rbf:.6f}")
    print(f"  Exact value = {val_exact:.6f}")

    # =====================================================================
    # Section 10: Geometry and Sample Shape
    # =====================================================================
    print("\n[10] Sample Geometry Analysis...")

    geom_hex = SampleGeometry(size=50.0, shape='hexagon')
    area_hex = geom_hex.area()
    edge_hex = geom_hex.edge_length()
    print(f"  Hexagon area = {area_hex:.2f} nm^2")
    print(f"  Hexagon perimeter = {edge_hex:.2f} nm")

    # Tortoise boundary
    geom_tort = SampleGeometry(size=1.0, shape='tortoise')
    area_tort = geom_tort.area()
    edge_tort = geom_tort.edge_length()
    print(f"  Tortoise boundary area = {area_tort:.4f}")
    print(f"  Tortoise boundary length = {edge_tort:.4f}")

    # =====================================================================
    # Section 11: Write All Results
    # =====================================================================
    print("\n[11] Writing Output Files...")

    results = {
        "Fermi_level_eV": E_F,
        "Fermi_wavevector_1/m": k_F,
        "Carrier_density_m-2": n_carrier,
        "Cyclotron_mass_kg": m_c,
        "Chern_number": C,
        "Born_scattering_rate_1/s": rate_born,
        "Transport_time_s": tau_tr,
        "Mean_free_path_m": l_mfp,
        "Diffusivity_m2/s": D_diff,
        "Longitudinal_conductivity_S": sigma_xx,
        "Intrinsic_AHC_S": sigma_ah,
        "Skew_Hall_S": sigma_skew,
        "Side_jump_Hall_S": sigma_sj,
        "Total_Hall_S": sigma_total,
        "Spin_Hall_S": sigma_spin,
        "Seebeck_V_per_K": S,
        "Lorenz_number_W_Ohm_K2": L,
        "Finite_size_conductivity_S": sigma_tb,
        "Self_consistent_tau_s": tau_sc,
        "Fermi_level_from_density_eV": E_F_solved,
    }

    io.write_transport_results("transport_results.dat", results)
    io.write_vector("tb_energies_eV.dat",
                    energies_tb / 1.602176634e-19, title="Tight-binding energies")
    io.write_matrix("berry_curvature_sample.dat",
                    np.column_stack((k_test,
                                     [berry.berry_curvature_analytical(k, 0.0, band='lower')
                                      for k in k_test])),
                    title="k(1/m)  Omega(m^2)")

    print(f"  Output written to: {output_dir}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Key Results:")
    print(f"    Chern number (lower band):     {C:.4f}")
    print(f"    Longitudinal conductivity:     {sigma_xx:.4e} S")
    print(f"    Anomalous Hall conductivity:   {sigma_ah:.4e} S")
    print(f"    Total Hall conductivity:       {sigma_total:.4e} S")
    print(f"    Spin Hall conductivity:        {sigma_spin:.4e} S")
    print(f"    Mean free path:                {l_mfp:.4e} m")
    print(f"    Seebeck coefficient:           {S:.4e} V/K")
    print("=" * 70)

    return results


if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
