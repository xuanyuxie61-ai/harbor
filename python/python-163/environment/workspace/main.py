"""
main.py
=======
Unified entry point for the Geothermal Reservoir Thermal-Hydro-Mechanical (THM)
Coupled Simulation.

Scientific Domain: Energy Systems — Geothermal Reservoir Thermal-Hydro-Mechanical Coupling

This program performs a fully coupled THM simulation of an enhanced geothermal
system (EGS) including:
  1. Stochastic permeability field generation
  2. Fluid flow simulation (Darcy's law + Forchheimer correction)
  3. Thermal transport (advection-diffusion with ETD-RK4 time stepping)
  4. Poroelastic mechanical deformation
  5. Fracture network Markov evolution
  6. Monte Carlo uncertainty quantification
  7. Numerical convergence and error analysis

The simulation runs with zero parameters and uses synthetic data.
"""

import numpy as np
import os
import sys

# Ensure imports work from the project directory
_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from thm_model import (
    THMParameters, THMState, fluid_density_temperature,
    fluid_viscosity_temperature, thermal_diffusivity,
    effective_heat_capacity, effective_thermal_conductivity
)
from spline_properties import (
    default_rock_thermal_conductivity_spline,
    default_fluid_viscosity_spline
)
from stochastic_fields import (
    StochasticDiffusivity2D, LogNormalPermeabilityField,
    RandomSampler, generate_stochastic_permeability_realization
)
from orthogonal_fit import fit_permeability_field_1d, fit_thermal_conductivity_field_1d
from mesh_geometry import (
    Triangle, triangle_mesh_quality, reservoir_boundary_polygon,
    triangulate_polygon_simple, reservoir_tetrahedral_mesh,
    generate_structured_hex_mesh
)
from quadrature_rules import (
    hexahedron_witherden_rule, cube_rule,
    integrate_scalar_field_hexahedron
)
from etdrk4_solver import ETDRK4ThermalSolver, ETDRK4Solver1D
from lambert_flow import (
    lambert_w_approx, wellbore_pressure_drop_lambert,
    injection_well_pressure
)
from risk_fracture import (
    FractureMarkovModel, fracture_aperture_markov_evolution,
    effective_permeability_from_fracture_network
)
from convergence_analysis import (
    ConvergenceMonitor, l2_error, convergence_rate,
    newton_raphson, bisection
)
from poroelastic_solver import PoroelasticSolver2D
from monte_carlo_uq import MonteCarloIntegrator, MonteCarloUQ, mc_integral_thermal_energy
from io_utils import SimulationIO, generate_parameter_table


def run_simulation():
    print("=" * 70)
    print("Geothermal Reservoir THM Coupled Simulation")
    print("Domain: Energy Systems — Geothermal Reservoir Thermal-Hydro-Mechanical Coupling")
    print("=" * 70)

    # ================================================================
    # 1. Initialize parameters
    # ================================================================
    print("\n[1/9] Initializing THM parameters...")
    params = THMParameters()
    print(f"  Grid: {params.nx} x {params.nz} x {params.ny}")
    print(f"  Domain: {params.reservoir_length} x {params.reservoir_height} x {params.reservoir_width} m")
    print(f"  Initial T: {params.T_initial} K, Initial p: {params.p_initial/1e6:.2f} MPa")

    # ================================================================
    # 2. Spline-based material properties
    # ================================================================
    print("\n[2/9] Building spline-interpolated material properties...")
    spline_kappa_rock = default_rock_thermal_conductivity_spline()
    spline_mu_fluid = default_fluid_viscosity_spline()
    T_test = np.linspace(300, 600, 10)
    kappa_vals = spline_kappa_rock.evaluate(T_test)
    mu_vals = spline_mu_fluid.evaluate(T_test)
    print(f"  Rock thermal conductivity at 400K: {spline_kappa_rock.evaluate(400.0):.4f} W/(m·K)")
    print(f"  Fluid viscosity at 400K: {spline_mu_fluid.evaluate(400.0):.6f} Pa·s")

    # ================================================================
    # 3. Stochastic fields and random sampling
    # ================================================================
    print("\n[3/9] Generating stochastic permeability and diffusivity fields...")
    np.random.seed(42)
    field_gen = LogNormalPermeabilityField(
        k_mean=params.matrix_permeability,
        sigma_ln_k=0.5,
        L_c=params.reservoir_length / 4.0
    )
    x_1d = np.linspace(0, params.reservoir_length, params.nx)
    k_1d = field_gen.evaluate_1d(x_1d, xi=np.random.randn(6), M=6)
    print(f"  1D permeability range: [{np.min(k_1d):.3e}, {np.max(k_1d):.3e}] m^2")

    # 2D stochastic diffusivity (from seed project 1169)
    diffusivity_gen = StochasticDiffusivity2D(D0=params.lambda_eff, sigma=0.3)
    omega = np.random.randn(4)
    x_2d = np.linspace(0, params.reservoir_length, params.nx)
    y_2d = np.linspace(0, params.reservoir_width, params.ny)
    X2, Y2 = np.meshgrid(x_2d, y_2d, indexing='ij')
    D_2d = diffusivity_gen.evaluate(omega, X2.flatten(), Y2.flatten()).reshape(params.nx, params.ny)
    print(f"  2D stochastic diffusivity range: [{np.min(D_2d):.4f}, {np.max(D_2d):.4f}] W/(m·K)")

    # Random sampling for parameter bounds (from seed project 837)
    sampler = RandomSampler(seed=42)
    xmin, fmin, xmax, fmax = sampler.sample_min_max(
        lambda x: np.sin(x) * np.exp(-0.01 * x),
        0.0, 100.0, n=5000
    )
    print(f"  Sampled function min at x={xmin:.2f}, f={fmin:.6f}")

    # ================================================================
    # 4. Orthogonal polynomial fits
    # ================================================================
    print("\n[4/9] Fitting orthogonal polynomial basis for property fields...")
    weights_poly = np.ones_like(x_1d)
    fit_perm = fit_permeability_field_1d(x_1d, np.log10(k_1d), n_terms=5)
    k_fit_log = fit_perm.evaluate(x_1d)
    k_fit_log = np.clip(k_fit_log, -20.0, -8.0)
    k_fit = 10.0 ** k_fit_log
    fit_error = np.max(np.abs(k_fit - k_1d))
    print(f"  Permeability fit max error: {fit_error:.3e} m^2")

    lambda_rock_vals = spline_kappa_rock.evaluate(373.15 + 50.0 * np.sin(x_1d / params.reservoir_length * np.pi))
    fit_lambda = fit_thermal_conductivity_field_1d(x_1d, lambda_rock_vals, n_terms=5)
    lambda_fit = fit_lambda.evaluate(x_1d)
    print(f"  Thermal conductivity fit max error: {np.max(np.abs(lambda_fit - lambda_rock_vals)):.4f} W/(m·K)")

    # ================================================================
    # 5. Mesh generation and quality analysis
    # ================================================================
    print("\n[5/9] Generating reservoir mesh and analyzing quality...")
    polygon = reservoir_boundary_polygon()
    triangles = triangulate_polygon_simple(polygon)
    tri_quality = triangle_mesh_quality(triangles)
    print(f"  2D boundary triangulation: {tri_quality['num_triangles']} triangles")
    print(f"  Mean quality: {tri_quality['mean_quality']:.4f}, Min quality: {tri_quality['min_quality']:.4f}")

    nodes, hex_elements = generate_structured_hex_mesh(
        params.reservoir_length, params.reservoir_width, params.reservoir_height,
        params.nx, params.ny, params.nz
    )
    print(f"  Structured hex mesh: {nodes.shape[0]} nodes, {hex_elements.shape[0]} elements")

    # Tetrahedral mesh
    nodes_tet, tet_elements = reservoir_tetrahedral_mesh(params)
    print(f"  Tetrahedral mesh: {nodes_tet.shape[0]} nodes, {tet_elements.shape[1]} tets")

    # ================================================================
    # 6. Quadrature rules for element integration
    # ================================================================
    print("\n[6/9] Testing quadrature rules...")
    n_pts, xq, yq, zq, wq = hexahedron_witherden_rule(5)
    print(f"  Hexahedron Witherden rule (precision 5): {n_pts} points")

    # Integrate constant over unit cube -> should be 1.0
    integral_const = np.sum(wq * 1.0)
    print(f"  Integral of 1 over [0,1]^3: {integral_const:.12f} (exact=1.0)")

    # Integrate x^2 y^2 z^2 over [0,1]^3 -> 1/27
    integral_poly = np.sum(wq * (xq**2 * yq**2 * zq**2))
    print(f"  Integral of x^2 y^2 z^2: {integral_poly:.12f} (exact=1/27={1.0/27.0:.12f})")

    # Cube rule (from seed project 232)
    w_cube, xyz_cube = cube_rule(
        np.array([0.0, 0.0, 0.0]),
        np.array([params.reservoir_length, params.reservoir_width, params.reservoir_height]),
        [3, 3, 3]
    )
    print(f"  Cube Gauss product rule (3x3x3): {w_cube.size} points")

    # Volume integration
    vol_exact = params.reservoir_length * params.reservoir_width * params.reservoir_height
    vol_numeric = np.sum(w_cube)
    print(f"  Numeric volume: {vol_numeric:.2f} m^3, Exact: {vol_exact:.2f} m^3")

    # ================================================================
    # 7. ETD-RK4 thermal solver
    # ================================================================
    print("\n[7/9] Running thermal transport simulation with ETD-RK4...")
    nx, nz = params.nx, params.nz
    Lx, Lz = params.reservoir_length, params.reservoir_height
    dx = Lx / nx
    dz = Lz / nz

    # TODO: Compute Darcy flux qx_flux and thermal diffusivity kappa_field,
    # then initialize ETDRK4ThermalSolver.
    # Scientific knowledge required:
    #   1. Darcy's law: q = -(k/\mu) * dp/dx
    #   2. Thermal diffusivity: \kappa = \lambda_{eff} / (\rho_{eff} c_{eff})
    #   3. ETDRK4ThermalSolver needs (nx, nz, Lx, Lz, dt, kappa, qx_flux, T_injection)
    # Placeholders below will produce physically incorrect results and must be replaced.
    qx_flux = np.zeros((nx, nz))
    kappa_field = 1.0e-12  # placeholder — physically incorrect, must be replaced
    thermal_solver = ETDRK4ThermalSolver(
        nx, nz, Lx, Lz, params.dt / 10.0, kappa_field, qx_flux,
        T_injection=params.T_injection
    )

    # Initial temperature: hot reservoir with cold injection on left
    T_field = np.full((nx, nz), params.T_initial, dtype=np.float64)
    T_field[:nx//4, :] = params.T_injection  # cold injection zone

    num_thermal_steps = 20
    T_history = [T_field.copy()]
    for step in range(num_thermal_steps):
        T_field = thermal_solver.step(T_field)
        T_history.append(T_field.copy())

    T_final = T_history[-1]
    print(f"  Thermal steps: {num_thermal_steps}")
    print(f"  Final T range: [{np.min(T_final):.2f}, {np.max(T_final):.2f}] K")
    print(f"  Mean T: {np.mean(T_final):.2f} K")

    # 1D ETD-RK4 spectral solver test (from seed project 630)
    etd1d = ETDRK4Solver1D(nx=64, L_domain=100.0, dt=0.25,
                           kappa=1.0e-6, advection_coeff=0.0, M=16)
    u0 = np.cos(np.linspace(0, 2*np.pi, 64, endpoint=False)) * (1.0 + np.sin(np.linspace(0, 2*np.pi, 64, endpoint=False)))
    t_hist, u_hist = etd1d.solve(u0, num_steps=50, save_interval=10)
    print(f"  1D spectral ETD-RK4: {t_hist.size} saved states, final mean u={np.mean(u_hist[-1]):.6f}")

    # ================================================================
    # 8. Poroelastic mechanical solver
    # ================================================================
    print("\n[8/9] Solving poroelastic mechanical equilibrium...")
    p_field = np.full((nx, nz), params.p_initial, dtype=np.float64)
    p_field[:nx//4, :] = params.p_production
    p_field[-nx//4:, :] = params.p_production

    poro_solver = PoroelasticSolver2D(
        nx, nz, dx, dz,
        E=params.young_modulus,
        nu=params.poisson_ratio,
        alpha=params.biot_coefficient,
        beta=params.thermal_expansion_rock,
        rho_b=params.rho_eff,
        g=9.81
    )
    u_x, u_z = poro_solver.solve_displacement(
        p_field, T_final, params.T_initial,
        num_iterations=200, tol=1.0e-7,
        fixed_bottom=True, fixed_sides=False
    )
    strain, stress = poro_solver.compute_strain_stress(u_x, u_z, p_field, T_final, params.T_initial)
    svm = poro_solver.von_mises_stress(stress)
    print(f"  Max displacement: ux={np.max(np.abs(u_x)):.6e} m, uz={np.max(np.abs(u_z)):.6e} m")
    print(f"  Max von Mises stress: {np.max(svm):.4e} Pa")
    print(f"  Max principal stress: {np.max(stress['sxx']):.4e} Pa")

    # ================================================================
    # 9. Fracture Markov model, Lambert W wellbore, Monte Carlo UQ
    # ================================================================
    print("\n[9/9] Running fracture, wellbore, and uncertainty quantification models...")

    # Fracture Markov evolution (from seed project 1026)
    fm = FractureMarkovModel(num_states=20)
    pi_ss = fm.steady_state()
    print(f"  Fracture Markov steady-state entropy: {-np.sum(pi_ss * np.log(pi_ss + 1e-30)):.4f}")

    aperture_history = fracture_aperture_markov_evolution(
        a_initial=1.0e-4, thermal_cycles=100, delta_a=1.0e-5,
        closure_prob=0.25, opening_prob=0.35
    )
    k_eff_frac = effective_permeability_from_fracture_network(
        aperture_history, fracture_density=2.0,
        matrix_perm=params.matrix_permeability
    )
    print(f"  Final fracture aperture: {aperture_history[-1]:.3e} m")
    print(f"  Effective permeability with fractures: {k_eff_frac:.3e} m^2")

    # Lambert W wellbore pressure (from seed project 644)
    dp_well = wellbore_pressure_drop_lambert(
        m_dot=5.0, D_well=0.2, L_well=1500.0, T_well=400.0,
        rho_f=950.0, mu_ref=1.0e-3
    )
    p_inj = injection_well_pressure(
        m_dot=5.0, k_inj=k_eff_frac, mu_inj=1.0e-3,
        h_inj=100.0, r_e=500.0, r_w=0.1, p_res=params.p_initial
    )
    print(f"  Wellbore pressure drop: {dp_well/1e6:.4f} MPa")
    print(f"  Injection well pressure: {p_inj/1e6:.4f} MPa")

    # Lambert W test
    w_test = lambert_w_approx(1.0, branch=0)
    print(f"  Lambert W(1.0) = {w_test:.8f} (exact={0.56714329:.8f})")

    # Monte Carlo UQ (from seed project 941)
    mc = MonteCarloUQ(params, n_samples=200)
    mean_ext, std_ext, ci_lo, ci_up = mc.estimate_heat_extraction(
        lambda s: T_final * (1.0 - 0.1 * np.random.rand())
    )
    print(f"  Monte Carlo mean heat extraction index: {mean_ext:.4f} ± {std_ext:.4f}")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_up:.4f}]")

    # Monte Carlo thermal energy
    mean_E, std_E = mc_integral_thermal_energy(
        n_samples=2000, T_mean=np.mean(T_final), T_std=np.std(T_final),
        rho_eff=params.rho_eff, cp_eff=params.cp_eff,
        volume=params.reservoir_length * params.reservoir_height * params.reservoir_width
    )
    print(f"  Monte Carlo thermal energy: {mean_E:.4e} J ± {std_E:.4e} J")

    # Convergence analysis (from seed project 338)
    monitor = ConvergenceMonitor(tol=1.0e-8, max_iter=1000)
    for it in range(10):
        residual = 1.0 / (it + 1.0) ** 2
        converged, reason = monitor.check(residual)
        if converged:
            break
    print(f"  Convergence monitor: residual after {len(monitor.residuals)} iterations = {monitor.residuals[-1]:.2e}")

    # Newton-Raphson and bisection tests
    root_nr, conv_nr, it_nr = newton_raphson(
        lambda x: x**3 - 2.0, lambda x: 3.0*x**2, x0=1.5
    )
    print(f"  Newton-Raphson: root={root_nr:.8f}, converged={conv_nr}, iterations={it_nr}")

    root_bi, conv_bi, it_bi = bisection(
        lambda x: x**3 - 2.0, 1.0, 2.0
    )
    print(f"  Bisection: root={root_bi:.8f}, converged={conv_bi}, iterations={it_bi}")

    # ================================================================
    # 10. Output results
    # ================================================================
    print("\n[10/9] Writing output files...")
    io = SimulationIO(output_dir=os.path.join(_project_dir, "thm_output"))
    io.write_field("temperature_final.txt", T_final, "Final temperature field (K)")
    io.write_field("pressure_final.txt", p_field, "Final pressure field (Pa)")
    io.write_field("displacement_x.txt", u_x, "X-displacement field (m)")
    io.write_field("displacement_z.txt", u_z, "Z-displacement field (m)")
    io.write_field("von_mises_stress.txt", svm, "Von Mises stress (Pa)")

    io.write_table("thermal_history.txt",
                   [np.arange(len(T_history)),
                    np.array([np.mean(T) for T in T_history])],
                   ["step", "mean_temperature_K"])

    summary = {
        "reservoir_volume_m3": params.reservoir_length * params.reservoir_height * params.reservoir_width,
        "initial_temperature_K": params.T_initial,
        "final_mean_temperature_K": float(np.mean(T_final)),
        "final_max_temperature_K": float(np.max(T_final)),
        "final_min_temperature_K": float(np.min(T_final)),
        "max_ux_m": float(np.max(np.abs(u_x))),
        "max_uz_m": float(np.max(np.abs(u_z))),
        "max_von_mises_Pa": float(np.max(svm)),
        "effective_permeability_m2": k_eff_frac,
        "wellbore_dp_MPa": dp_well / 1e6,
        "injection_pressure_MPa": p_inj / 1e6,
        "monte_carlo_heat_extraction": mean_ext,
        "monte_carlo_thermal_energy_J": mean_E,
    }
    io.write_summary("simulation_summary.txt", summary)
    generate_parameter_table(os.path.join(io.output_dir, "parameters.txt"))

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Output directory: {io.output_dir}")
    print(f"Files written:")
    for f in os.listdir(io.output_dir):
        print(f"  - {f}")

    return summary


if __name__ == "__main__":
    summary = run_simulation()
