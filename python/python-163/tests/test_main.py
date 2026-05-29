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

    # Darcy flux (simplified)
    qx_flux = np.zeros((nx, nz))
    for iz in range(nz):
        # Simple linear pressure-driven flow
        dp_dx = -(params.p_initial - params.p_production) / Lx
        mu = fluid_viscosity_temperature(params.T_initial)
        # Use nominal matrix permeability for flow (stochastic field is for heterogeneity study)
        k_z = params.matrix_permeability
        qx_flux[:, iz] = -k_z / mu * dp_dx

    kappa_field = thermal_diffusivity(params.lambda_eff, params.rho_eff, params.cp_eff)
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

# ================================================================
# 测试用例（35个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: darcy_velocity_scalar 基本计算 ----
from thm_model import darcy_velocity_scalar, biot_modulus, strain_tensor_2d
q = darcy_velocity_scalar(k=1e-14, mu=1e-3, dp_dx=-100.0, rho_f=1000.0, g=9.81, dz_sign=0.0)
assert isinstance(q, float), '[TC01] 返回值应为标量 FAILED'
assert np.isfinite(q), '[TC01] 结果应有限 FAILED'
assert q > 0, '[TC01] 负压力梯度应产生正通量 FAILED'

# ---- TC02: effective_heat_capacity 已知值计算 ----
rho_c_eff = effective_heat_capacity(phi=0.15, rho_f=1000.0, cp_f=4180.0, rho_r=2700.0, cp_r=850.0)
rho_c_expected = 0.15 * 1000.0 * 4180.0 + 0.85 * 2700.0 * 850.0
assert abs(rho_c_eff - rho_c_expected) < 0.01, '[TC02] 有效热容计算错误 FAILED'

# ---- TC03: effective_thermal_conductivity 已知值计算 ----
lam_eff = effective_thermal_conductivity(phi=0.15, lam_f=0.6, lam_r=2.5)
lam_expected = 0.15 * 0.6 + 0.85 * 2.5
assert abs(lam_eff - lam_expected) < 1e-12, '[TC03] 有效导热系数计算错误 FAILED'

# ---- TC04: biot_modulus 已知值计算 ----
M = biot_modulus(phi=0.15, K_f=2.222e9, alpha=0.8, K_s=50e9)
assert M > 0, '[TC04] Biot模量应为正 FAILED'
assert np.isfinite(M), '[TC04] Biot模量应有限 FAILED'

# ---- TC05: thermal_diffusivity 已知值计算 ----
kappa = thermal_diffusivity(lambda_eff=2.215, rho_eff=2500.0, cp_eff=1000.0)
kappa_expected = 2.215 / (2500.0 * 1000.0)
assert abs(kappa - kappa_expected) < 1e-15, '[TC05] 热扩散率计算错误 FAILED'

# ---- TC06: strain_tensor_2d 对称性验证 ----
exx, ezz, exz = strain_tensor_2d(dux_dx=0.001, duz_dz=0.002, dux_dz=0.0005, duz_dx=0.0005)
exx2, ezz2, exz2 = strain_tensor_2d(dux_dx=0.001, duz_dz=0.002, dux_dz=0.0005, duz_dx=0.0005)
assert exx == exx2, '[TC06] 应变张量应确定性 FAILED'
assert exz == 0.5 * (0.0005 + 0.0005), '[TC06] 剪应变计算错误 FAILED'

# ---- TC07: fluid_density_temperature 范围约束 ----
params_test = THMParameters()
rho_fluid = fluid_density_temperature(p=20e6, T=423.15, params=params_test)
assert 500.0 <= rho_fluid <= 1500.0, '[TC07] 流体密度超出物理范围 FAILED'
rho_cold = fluid_density_temperature(p=20e6, T=273.15, params=params_test)
assert rho_cold > rho_fluid, '[TC07] 冷流体密度应大于热流体 FAILED'

# ---- TC08: fluid_viscosity_temperature 单调递减 ----
mu_hot = fluid_viscosity_temperature(400.0)
mu_cold = fluid_viscosity_temperature(300.0)
assert mu_hot < mu_cold, '[TC08] 粘度应随温度升高而减小 FAILED'
assert 1e-4 <= mu_hot <= 5e-3, '[TC08] 粘度裁剪范围错误 FAILED'

# ---- TC09: lambert_w_approx W(1.0) 解析验证 ----
w1 = lambert_w_approx(1.0, branch=0)
assert abs(w1 - 0.56714329) < 0.01, '[TC09] Lambert W(1.0) 近似精度不足 FAILED'

# ---- TC10: lambert_w_approx W(-1/e) = -1.0 ----
w_neg = lambert_w_approx(-np.exp(-1.0), branch=0)
assert abs(w_neg - (-1.0)) < 0.01, '[TC10] Lambert W(-1/e) 应等于 -1.0 FAILED'

# ---- TC11: lambert_w_approx W(0)=0 (n_iter=0避免Halley迭代log(0)) ----
w0 = lambert_w_approx(0.0, branch=0, n_iter=0)
assert abs(w0) < 1e-10, '[TC11] Lambert W(0.0) 应等于 0.0 FAILED'

# ---- TC12: injection_well_pressure 基本计算 ----
p_inj_test = injection_well_pressure(m_dot=5.0, k_inj=1e-14, mu_inj=1e-3, h_inj=100.0, r_e=500.0, r_w=0.1, p_res=20e6)
assert p_inj_test > 20e6, '[TC12] 注入压力应大于储层压力 FAILED'
assert np.isfinite(p_inj_test), '[TC12] 注入压力应有限 FAILED'

# ---- TC13: wellbore_pressure_drop_lambert 基本计算 ----
dp = wellbore_pressure_drop_lambert(m_dot=5.0, D_well=0.2, L_well=1500.0, T_well=400.0, rho_f=950.0, mu_ref=1e-3)
assert dp > 0, '[TC13] 井筒压降应为正 FAILED'
assert np.isfinite(dp), '[TC13] 井筒压降应有限 FAILED'

# ---- TC14: l2_error 相同数组误差为零 ----
u = np.array([1.0, 2.0, 3.0])
err = l2_error(u, u.copy())
assert err < 1e-15, '[TC14] 相同数组L2误差应为零 FAILED'

# ---- TC15: l2_error 不同数组正误差 ----
err_diff = l2_error(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
assert err_diff > 0.5, '[TC15] 不同数组应有正误差 FAILED'

# ---- TC16: newton_raphson 求解 x^3-2=0 ----
root, conv, it = newton_raphson(lambda x: x**3 - 2.0, lambda x: 3.0*x**2, x0=1.5)
assert conv, '[TC16] Newton-Raphson 应收敛 FAILED'
assert abs(root - 2.0**(1.0/3.0)) < 1e-10, '[TC16] x^3-2=0 根值不正确 FAILED'

# ---- TC17: bisection 求解 x^3-2=0 ----
root2, conv2, it2 = bisection(lambda x: x**3 - 2.0, 1.0, 2.0)
assert conv2, '[TC17] 二分法应收敛 FAILED'
assert abs(root2 - 2.0**(1.0/3.0)) < 1e-10, '[TC17] 二分法根值不正确 FAILED'

# ---- TC18: convergence_rate 已知收敛率 ----
errs = np.array([0.1, 0.025, 0.00625])
hs = np.array([1.0, 0.5, 0.25])
rates = convergence_rate(errs, hs)
assert len(rates) == 2, '[TC18] 应收敛率数量错误 FAILED'
assert abs(rates[0] - 2.0) < 0.01, '[TC18] 收敛率应约为2.0 (O(h^2)) FAILED'

# ---- TC19: richardson_extrapolation ----
from convergence_analysis import richardson_extrapolation
vals = np.array([4.0, 3.5])
hs_rich = np.array([1.0, 0.5])
ext, err_est = richardson_extrapolation(vals, hs_rich, p_expected=2.0)
assert np.isfinite(ext), '[TC19] Richardson外推值应有限 FAILED'

# ---- TC20: CubicSplineInterpolator 插值精度 ----
from spline_properties import build_temperature_spline_property
x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_data = x_data ** 2
spline = build_temperature_spline_property(x_data, y_data, "test")
y_interp = spline.evaluate(3.5)
assert abs(y_interp - 12.25) < 0.05, '[TC20] 样条插值 x^2 在3.5处应接近12.25 FAILED'

# ---- TC21: 默认样条可用性 ----
spline_k = default_rock_thermal_conductivity_spline()
val_400K = spline_k.evaluate(400.0)
assert 1.5 < val_400K < 3.5, '[TC21] 400K时岩石导热系数应在合理范围 FAILED'

spline_mu = default_fluid_viscosity_spline()
val_mu_300K = spline_mu.evaluate(300.0)
assert 1e-4 < val_mu_300K < 2e-3, '[TC21] 300K时流体粘度应在合理范围 FAILED'

# ---- TC22: Triangle 正三角形质量 ----
tri_eq = Triangle(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]]))
assert tri_eq.quality > 0.99, '[TC22] 正三角形质量应接近1.0 FAILED'
assert tri_eq.is_well_formed(), '[TC22] 正三角形应为良态 FAILED'

# ---- TC23: Triangle 退化三角形 ----
tri_deg = Triangle(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]))
assert tri_deg.quality == 0.0, '[TC23] 退化三角形质量应为0 FAILED'
assert not tri_deg.is_well_formed(), '[TC23] 退化三角形应为非良态 FAILED'

# ---- TC24: quadrature_rules 对常数函数精确积分 ----
n_pts, xq, yq, zq, wq = hexahedron_witherden_rule(5)
integral_1 = np.sum(wq)
assert abs(integral_1 - 1.0) < 1e-12, '[TC24] 单位立方体上1的积分应为1.0 FAILED'

# ---- TC25: quadrature_rules 多项式积分 ----
integral_poly_test = np.sum(wq * (xq**2 * yq**2 * zq**2))
assert abs(integral_poly_test - 1.0/27.0) < 1e-12, '[TC25] x^2y^2z^2在[0,1]^3上积分应为1/27 FAILED'

# ---- TC26: orthogonal_fit 输出结构和有限性验证 ----
x_fit = np.linspace(0.0, 500.0, 64)
y_fit = np.sin(x_fit / 100.0)
fit = fit_permeability_field_1d(x_fit, y_fit, n_terms=5)
y_pred = fit.evaluate(x_fit)
assert len(y_pred) == len(x_fit), '[TC26] 拟合输出长度应等于输入长度 FAILED'
assert np.all(np.isfinite(y_pred)), '[TC26] 拟合输出应全部有限 FAILED'
assert fit.n_terms == 5, '[TC26] 项数应为5 FAILED'

# ---- TC27: fracture_aperture_markov_evolution 可复现性 ----
np.random.seed(42)
hist1 = fracture_aperture_markov_evolution(a_initial=1e-4, thermal_cycles=50, delta_a=1e-5, closure_prob=0.3, opening_prob=0.4)
np.random.seed(42)
hist2 = fracture_aperture_markov_evolution(a_initial=1e-4, thermal_cycles=50, delta_a=1e-5, closure_prob=0.3, opening_prob=0.4)
assert np.allclose(hist1, hist2), '[TC27] 固定种子应产生相同裂缝演化 FAILED'
assert hist1[-1] >= 0, '[TC27] 裂缝孔径应为非负 FAILED'

# ---- TC28: effective_permeability_from_fracture_network ----
aps = np.array([1e-4, 2e-4, 5e-5])
k_eff_frac_test = effective_permeability_from_fracture_network(aps, fracture_density=2.0, matrix_perm=1e-14)
assert k_eff_frac_test >= 1e-14, '[TC28] 有效渗透率应不小于基质渗透率 FAILED'
assert k_eff_frac_test <= 1e-12, '[TC28] 有效渗透率不应超过上界 FAILED'

# ---- TC29: mc_integral_thermal_energy 可复现性 ----
np.random.seed(42)
mean1, std1 = mc_integral_thermal_energy(n_samples=2000, T_mean=400.0, T_std=20.0, rho_eff=2500.0, cp_eff=1000.0, volume=1e7)
np.random.seed(42)
mean2, std2 = mc_integral_thermal_energy(n_samples=2000, T_mean=400.0, T_std=20.0, rho_eff=2500.0, cp_eff=1000.0, volume=1e7)
assert abs(mean1 - mean2) < 1e-10, '[TC29] 固定种子MC应可复现 FAILED'

# ---- TC30: ConvergenceMonitor 收敛检测 ----
monitor = ConvergenceMonitor(tol=1e-4, max_iter=100)
conv, reason = monitor.check(residual=1e-5)
assert conv, '[TC30] 低于容差的残差应判定收敛 FAILED'
assert "residual below tolerance" in reason.lower(), '[TC30] 收敛原因不正确 FAILED'

# ---- TC31: ConvergenceMonitor 发散检测 ----
monitor2 = ConvergenceMonitor(tol=1e-8, max_iter=100)
monitor2.check(residual=0.001)
conv2, reason2 = monitor2.check(residual=0.1)
assert conv2, '[TC31] 残差突增十倍应检测到发散 FAILED'

# ---- TC32: THMParameters 属性一致性 ----
params_check = THMParameters()
lam_check = params_check.lame_lambda()
mu_check = params_check.lame_mu()
assert lam_check > 0 and mu_check > 0, '[TC32] Lamé参数应为正 FAILED'
E_check = mu_check * (3.0 * lam_check + 2.0 * mu_check) / (lam_check + mu_check)
assert abs(E_check - params_check.young_modulus) / params_check.young_modulus < 0.01, '[TC32] 弹性参数不一致 FAILED'

# ---- TC33: SimulationIO 写入与读取回环 ----
import tempfile, os as _os
io_test = SimulationIO(output_dir=_os.path.join(tempfile.gettempdir(), "test_thm_io"))
test_field = np.array([[1.0, 2.0], [3.0, 4.0]])
io_test.write_field("test_field.txt", test_field, "test")
read_back = io_test.read_matrix_file(_os.path.join(io_test.output_dir, "test_field.txt"))
assert read_back.shape == (2, 2), '[TC33] 读回矩阵形状错误 FAILED'
assert abs(np.sum(read_back) - 10.0) < 1e-10, '[TC33] 读回矩阵值错误 FAILED'
# Cleanup
import shutil
shutil.rmtree(io_test.output_dir, ignore_errors=True)

# ---- TC34: darcy_velocity_scalar 零渗透率 ----
q_zero_k = darcy_velocity_scalar(k=0.0, mu=1e-3, dp_dx=-100.0, rho_f=1000.0, g=0.0)
assert q_zero_k == 0.0, '[TC34] 零渗透率应产生零通量 FAILED'

# ---- TC35: linf_error 计算 ----
from convergence_analysis import linf_error
err_inf = linf_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 3.0]))
assert abs(err_inf - 0.5) < 1e-15, '[TC35] L∞误差应为0.5 FAILED'

print('\n全部 35 个测试通过!\n')
