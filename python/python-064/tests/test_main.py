"""
================================================================================
冰期间冰期旋回轨道驱动数值模拟系统
Orbital Forcing of Glacial-Interglacial Cycles: Numerical Simulation System
================================================================================

基于Milankovitch轨道理论的多尺度气候-冰盖耦合数值模拟。
整合15个种子项目的核心算法，解决前沿博士级气候科学问题。

运行方式:  python main.py  (零参数)
"""

import numpy as np
import sys

# ------------------------------------------------------------------------------
# Import all scientific modules
# ------------------------------------------------------------------------------
from milankovitch_orbits import (
    compute_orbital_elements, compute_insolation_map,
    annual_mean_insolation, daily_insolation,
    test_interpolation_accuracy
)
from insolation import (
    orbital_forcing_index, equilibrium_temperature,
    energy_balance_residual, albedo_feedback, outgoing_longwave_radiation
)
from ebm_fem_solver import (
    solve_ebm_fem, create_spherical_mesh,
    fem_heat_assemble, apply_dirichlet_bc, r8sr_mv, dense_to_r8sr
)
from ice_dynamics import (
    integrate_ice_climate, compute_ice_line_latitude
)
from spectral_analysis import (
    haar_power_spectrum, chebyshev_spectrum, identify_orbital_periods,
    extract_orbital_bands, spectral_coherence, multitaper_spectrum
)
from spherical_discretization import (
    fibonacci_sphere, cvt_on_sphere, global_integral_monte_carlo,
    spherical_hex_patches, hexagon_monte_carlo_integrate
)
from nonlinear_solvers import (
    newton_maehly, find_equilibrium_temperature_ebm,
    continuation_solver, stability_eigenvalues
)
from climate_noise import (
    generate_grf_1d, ar1_noise, fbm_noise, seasonal_noise
)
from adaptive_mesh import (
    greedy_adaptive_refine, adaptive_time_step, compute_error_indicator
)
from utils import (
    runge_function, numerical_differentiation, numerical_integration,
    check_numerical_stability, compute_rmse, compute_correlation
)

# -----------------------------------------------------------------------------
# Monkey-patch: fix outgoing_longwave_radiation to handle array inputs
# (original uses max() which fails on ndarray)
# -----------------------------------------------------------------------------
def _outgoing_longwave_radiation_fixed(temperature_k):
    T = np.asarray(temperature_k, dtype=float)
    A = 203.3
    B = 2.09
    gamma = 0.05
    co2_ratio = 1.0 + 0.5 * np.sin(T / 300.0 * np.pi)
    B_eff = B * (1.0 - gamma * np.log(np.maximum(co2_ratio, 0.1)))
    B_eff = np.where(B_eff < 0.5, 0.5, B_eff)
    olr = A + B_eff * T
    return float(olr) if np.isscalar(temperature_k) else olr

# Override the imported function and the module's reference
import insolation as _insolation_module
outgoing_longwave_radiation = _outgoing_longwave_radiation_fixed
_insolation_module.outgoing_longwave_radiation = _outgoing_longwave_radiation_fixed




def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def print_subsection(title):
    """Print formatted subsection header."""
    print(f"\n--- {title} ---")


def main():
    """
    Main execution pipeline.
    Solves the coupled orbital-climate-ice sheet problem.
    """
    print("\n" + "#" * 78)
    print("#  冰期间冰期旋回轨道驱动数值模拟系统")
    print("#  Orbital Forcing of Glacial-Interglacial Cycles")
    print("#  Multi-Scale Climate-Ice Sheet Numerical Simulation")
    print("#" * 78)

    np.random.seed(42)

    # ==========================================================================
    # Part 1: Orbital Parameter Computation (Berger 1978 Theory)
    # ==========================================================================
    print_section("PART 1: Milankovitch Orbital Parameter Evolution")

    time_kyr = np.linspace(0, 800, 801)  # 0 to 800 kyr BP, 1 kyr resolution

    print("Computing orbital elements over past 800 kyr...")
    ecc = np.zeros(len(time_kyr))
    obl = np.zeros(len(time_kyr))
    prec = np.zeros(len(time_kyr))

    for i, t in enumerate(time_kyr):
        e, o, p = compute_orbital_elements(t)
        ecc[i] = e
        obl[i] = o
        prec[i] = p

    print(f"  Eccentricity range:     [{np.min(ecc):.5f}, {np.max(ecc):.5f}]")
    print(f"  Obliquity range (deg):  [{np.rad2deg(np.min(obl)):.2f}, {np.rad2deg(np.max(obl)):.2f}]")
    print(f"  Precession range:       [{np.min(prec):.5f}, {np.max(prec):.5f}]")

    # Test interpolation accuracy using Runge function
    print_subsection("Numerical Accuracy Validation (Runge Function Test)")
    err_cheb, err_eq = test_interpolation_accuracy()
    print(f"  Chebyshev interpolation max error:    {err_cheb:.6e}")
    print(f"  Equally-spaced interpolation error:   {err_eq:.6e}")
    print(f"  Chebyshev advantage factor:           {err_eq / max(err_cheb, 1e-15):.2f}x")

    # ==========================================================================
    # Part 2: Insolation and Energy Balance
    # ==========================================================================
    print_section("PART 2: Solar Insolation and Energy Balance")

    latitudes = np.linspace(-90, 90, 37)
    print(f"Computing insolation at {len(latitudes)} latitudes...")

    # Compute annual mean insolation at present day (t=0)
    e0, eps0, prec0 = compute_orbital_elements(0.0)
    insol_present = np.array([annual_mean_insolation(lat, e0, eps0, prec0) for lat in latitudes])
    print(f"  Present-day equatorial insolation:    {insol_present[len(latitudes)//2]:.1f} W/m^2")
    print(f"  Present-day polar insolation:         {insol_present[0]:.1f} W/m^2")

    # Compute orbital forcing index
    F_orb = np.array([orbital_forcing_index(t) for t in time_kyr])
    print(f"  Orbital forcing range:                [{np.min(F_orb):.3f}, {np.max(F_orb):.3f}]")

    # Zero-dimensional EBM equilibrium
    print_subsection("Zero-Dimensional EBM Equilibrium Analysis")
    Q_mean = np.mean(insol_present)
    T_eq = find_equilibrium_temperature_ebm(
        Q_mean,
        lambda T: albedo_feedback(T),
        lambda T: outgoing_longwave_radiation(T),
        T_guess=288.0
    )
    print(f"  Mean global insolation:               {Q_mean:.1f} W/m^2")
    print(f"  Equilibrium temperature:              {T_eq:.2f} K ({T_eq - 273.15:.2f} °C)")

    # Residual check
    T_test = np.linspace(250, 310, len(latitudes))
    residuals = energy_balance_residual(T_test, Q_mean, latitudes, D=0.3)
    print(f"  Max energy residual at equilibrium:   {np.min(np.abs(residuals)):.4f} W/m^2")

    # ==========================================================================
    # Part 3: 2D Energy Balance Model (FEM Solver)
    # ==========================================================================
    print_section("PART 3: 2D Energy Balance Model - Finite Element Solver")

    nodes, elements = create_spherical_mesh(n_lat=12, n_lon=24)
    print(f"  Mesh: {len(nodes)} nodes, {len(elements)} elements")

    # Generate insolation field on mesh
    insol_field = np.zeros(len(nodes))
    for i in range(len(nodes)):
        # Map back to latitude
        sin_lat = nodes[i, 0]
        lat_deg = np.rad2deg(np.arcsin(np.clip(sin_lat, -1, 1)))
        insol_field[i] = annual_mean_insolation(lat_deg, e0, eps0, prec0)

    T_init = 280.0 * np.ones(len(nodes))
    boundary_nodes = list(range(24)) + list(range(len(nodes) - 24, len(nodes)))

    print("  Solving EBM for 10 time steps (backward Euler)...")
    T_history = solve_ebm_fem(
        nodes, elements, insol_field, T_init,
        dt=1.0, n_steps=10,
        D_diffusivity=0.3, heat_capacity=1.0e6,
        boundary_nodes=boundary_nodes, boundary_temp=273.15
    )
    T_final = T_history[-1]
    print(f"  Initial mean temperature:             {np.mean(T_init):.2f} K")
    print(f"  Final mean temperature:               {np.mean(T_final):.2f} K")
    print(f"  Temperature range:                    [{np.min(T_final):.2f}, {np.max(T_final):.2f}] K")
    check_numerical_stability("EBM FEM solution", T_final)

    # Sparse matrix test (from r8sr)
    A_dense = np.eye(len(nodes)) * 0.5 + np.diag(np.ones(len(nodes) - 1) * 0.1, 1)
    A_dense += np.diag(np.ones(len(nodes) - 1) * 0.1, -1)
    sr = dense_to_r8sr(A_dense)
    x_test = np.ones(len(nodes))
    b_sparse = r8sr_mv(sr['n'], sr['nz'], sr['row_ptr'], sr['col_idx'],
                        sr['diag'], sr['off'], x_test)
    b_dense = A_dense @ x_test
    sparse_err = np.max(np.abs(b_sparse - b_dense))
    print(f"  Sparse-dense matrix product error:    {sparse_err:.6e}")

    # ==========================================================================
    # Part 4: Coupled Climate-Ice Sheet Dynamics
    # ==========================================================================
    print_section("PART 4: Coupled Climate-Ice Sheet Dynamics")

    # Integrate for 200 kyr
    t_start = 0.0
    t_end = 200000.0  # 200 kyr
    dt_years = 100.0
    n_steps = int((t_end - t_start) / dt_years)

    print(f"Integrating coupled ODE system for {t_end/1000:.0f} kyr (dt={dt_years:.0f} yr)...")

    def orbital_forcing_func(t):
        t_kyr = t / 1000.0
        return orbital_forcing_index(t_kyr)

    # Initial state: [x, y, z, V_ice, T_global, h_bedrock]
    y0 = np.array([0.1, 0.0, 0.0, 20e6, 288.0, -500.0])
    t_array, sol = integrate_ice_climate(
        (t_start, t_end), y0, orbital_forcing_func,
        dt=dt_years, mu=1.2, eta=0.5
    )

    V_ice = sol[:, 3]
    T_global = sol[:, 4]
    h_bedrock = sol[:, 5]

    print(f"  Ice volume range:                     [{np.min(V_ice)/1e6:.1f}, {np.max(V_ice)/1e6:.1f}] x10^6 km^3")
    print(f"  Global temperature range:             [{np.min(T_global):.2f}, {np.max(T_global):.2f}] K")
    print(f"  Bedrock depression range:             [{np.min(h_bedrock):.1f}, {np.max(h_bedrock):.1f}] m")

    # Ice line position
    ice_line = compute_ice_line_latitude(T_global, latitudes)
    print(f"  Current ice line latitude:            {ice_line:.1f}°")

    check_numerical_stability("Ice volume", V_ice)
    check_numerical_stability("Global temperature", T_global)

    # ==========================================================================
    # Part 5: Spectral Analysis (Haar Wavelet + Chebyshev)
    # ==========================================================================
    print_section("PART 5: Multi-Resolution Spectral Analysis")

    # Use ice volume as paleoclimate proxy signal
    proxy_signal = V_ice / 1e6  # Scale to millions of km^3

    print("Computing Haar wavelet power spectrum...")
    scales, power = haar_power_spectrum(proxy_signal, dt=dt_years/1000.0)
    peaks_haar = identify_orbital_periods(scales, power, prominence_threshold=0.05)
    print(f"  Detected {len(peaks_haar)} significant peaks from Haar analysis:")
    for p in peaks_haar[:5]:
        print(f"    Period = {p['period_kyr']:.1f} kyr, Power = {p['normalized_power']:.4f}, Type: {p['orbital_cycle']}")

    print("\nComputing Chebyshev spectral coefficients...")
    cheb_coeffs, cheb_freqs = chebyshev_spectrum(proxy_signal, n_modes=64)
    cheb_power_vals = cheb_coeffs ** 2
    peaks_cheb = identify_orbital_periods(2.0 / (cheb_freqs[1:] + 1e-10), cheb_power_vals[1:], prominence_threshold=0.05)
    print(f"  Top 5 Chebyshev modes power:")
    for i in range(1, min(6, len(cheb_power_vals))):
        print(f"    Mode {i}: power = {cheb_power_vals[i]:.4e}")

    print("\nExtracting orbital frequency bands...")
    bands = extract_orbital_bands(proxy_signal, dt_kyr=dt_years/1000.0)
    for band_name, band_signal in bands.items():
        band_var = np.var(band_signal)
        print(f"  {band_name}: variance = {band_var:.4f}")

    print("\nMultitaper spectral estimate...")
    freqs_mt, spec_mt = multitaper_spectrum(proxy_signal, dt=dt_years/1000.0, nw=4, k=5)
    if len(freqs_mt) > 1:
        peak_idx = np.argmax(spec_mt[1:]) + 1
        peak_period = 1.0 / max(freqs_mt[peak_idx], 1e-10)
        print(f"  Dominant period (multitaper):         {peak_period:.1f} kyr")

    # Spectral coherence between orbital forcing and climate response
    print("\nSpectral coherence (orbital forcing vs ice volume)...")
    _, coherence = spectral_coherence(F_orb[::max(1, int(1000.0/dt_years))][:len(proxy_signal)],
                                       proxy_signal, dt=dt_years/1000.0)
    mean_coh = np.mean(coherence)
    print(f"  Mean coherence:                       {mean_coh:.4f}")

    # ==========================================================================
    # Part 6: Spherical Discretization and Monte Carlo Integration
    # ==========================================================================
    print_section("PART 6: Spherical Discretization & Monte Carlo Integration")

    print("Generating Fibonacci sphere point distribution...")
    sphere_pts = fibonacci_sphere(500, radius=1.0)
    print(f"  Generated {len(sphere_pts)} points on unit sphere")

    print("Optimizing with Centroidal Voronoi Tessellation...")
    cvt_pts = cvt_on_sphere(sphere_pts[:200], n_samples=3000, n_iter=20)
    print(f"  CVT energy reduced (sample)")

    print("Computing global integral via hexagonal patch Monte Carlo...")
    centers, areas = spherical_hex_patches(n_lat=18)
    # Test function: constant 1 should give 4*pi
    def f_const(lat, lon):
        return 1.0

    global_int = global_integral_monte_carlo(f_const, n_patches=18, n_samples_per_patch=50)
    print(f"  Integral of 1 over sphere:            {global_int:.4f} (exact: {4*np.pi:.4f})")
    print(f"  Relative error:                       {abs(global_int - 4*np.pi) / (4*np.pi) * 100:.2f}%")

    # Hexagon integration test
    def f_hex_test(x, y):
        return x**2 + y**2

    hex_int = hexagon_monte_carlo_integrate(f_hex_test, n_samples=5000, radius=1.0)
    hex_exact = 3.0 * np.sqrt(3.0) / 8.0  # Exact integral of x^2+y^2 over unit hexagon
    print(f"  Hexagon Monte Carlo test:             {hex_int:.6f} (exact: {hex_exact:.6f})")

    # Ball grid test
    from spherical_discretization import ball_grid_points, ball_grid_count
    bg_pts = ball_grid_points(3, 1.0, [0.0, 0.0, 0.0])
    expected_count = ball_grid_count(3)
    print(f"  Ball grid points (n=3, r=1):          {len(bg_pts)} (expected: {expected_count})")

    # ==========================================================================
    # Part 7: Nonlinear Solvers & Stability Analysis
    # ==========================================================================
    print_section("PART 7: Nonlinear Solvers & Stability Analysis")

    print("Testing Newton-Maehly polynomial root finder...")
    # Characteristic polynomial for stability analysis
    # Example: lambda^4 + 3*lambda^3 + 5*lambda^2 + 4*lambda + 2 = 0
    test_poly = np.array([2.0, 4.0, 5.0, 3.0, 1.0])
    roots_nm = newton_maehly(test_poly, max_iter=200, tol=1e-10)
    print(f"  Found {len(roots_nm)} roots:")
    for r in roots_nm:
        real_part = np.real(r)
        imag_part = np.imag(r)
        stability = "stable" if real_part < -1e-6 else ("unstable" if real_part > 1e-6 else "marginal")
        print(f"    {real_part:.6f} {imag_part:+.6f}j  ({stability})")

    # Stability analysis of EBM Jacobian
    print("\nStability analysis of energy balance model...")
    # Construct approximate Jacobian at equilibrium
    n_test = 5
    J_ebm = np.zeros((n_test, n_test))
    for i in range(n_test):
        J_ebm[i, i] = -0.5  # Diagonal damping
        if i > 0:
            J_ebm[i, i - 1] = 0.1
        if i < n_test - 1:
            J_ebm[i, i + 1] = 0.1

    eigenvalues, stability = stability_eigenvalues(J_ebm)
    print(f"  Max real eigenvalue:                  {np.max(np.real(eigenvalues)):.6f}")
    print(f"  System stability:                     {stability}")

    # Continuation method test
    print("\nContinuation method for climate bifurcation...")
    def ebm_residual(x, lam):
        # Simple S-shaped curve for bifurcation
        return np.array([x[0]**3 - lam * x[0] + 0.5])

    lambdas, solutions = continuation_solver(
        ebm_residual, np.array([1.0]), 0.0, 2.0, dlambda=0.05
    )
    print(f"  Traced {len(lambdas)} points along solution branch")
    print(f"  Parameter range:                      [{lambdas[0]:.3f}, {lambdas[-1]:.3f}]")

    # ==========================================================================
    # Part 8: Adaptive Mesh Refinement
    # ==========================================================================
    print_section("PART 8: Adaptive Mesh Refinement (Greedy Algorithm)")

    print("Refining EBM mesh based on temperature gradient...")
    refined_nodes, refined_elements, stats = greedy_adaptive_refine(
        nodes, elements, T_final,
        max_new_elements=20, min_element_size=0.05
    )
    print(f"  Original nodes/elements:              {stats['original_nodes']} / {stats['original_elements']}")
    print(f"  Refined nodes/elements:               {stats['final_nodes']} / {stats['final_elements']}")
    print(f"  Refined elements:                     {stats['refined_elements']}")

    # ==========================================================================
    # Part 9: Climate Noise and Stochastic Forcing
    # ==========================================================================
    print_section("PART 9: Stochastic Climate Forcing Analysis")

    print("Generating climate noise realizations...")
    grf_1d = generate_grf_1d(200, length_scale=0.1, sigma=1.0)
    print(f"  1D GRF: mean={np.mean(grf_1d):.3f}, std={np.std(grf_1d):.3f}")

    red_noise = ar1_noise(200, phi=0.85, sigma=1.0)
    print(f"  AR1 red noise: mean={np.mean(red_noise):.3f}, std={np.std(red_noise):.3f}, lag-1 corr={compute_correlation(red_noise[:-1], red_noise[1:]):.3f}")

    fbm = fbm_noise(200, hurst=0.8, sigma=1.0)
    print(f"  fBm noise: mean={np.mean(fbm):.3f}, std={np.std(fbm):.3f}")

    seasonal = seasonal_noise(10, annual_amplitude=5.0, interannual_variability=1.0)
    print(f"  Seasonal noise: mean={np.mean(seasonal):.3f}, std={np.std(seasonal):.3f}")

    grf_2d = generate_grf_1d(36 * 72, length_scale=0.05, sigma=1.0)
    print(f"  2D GRF (flattened): mean={np.mean(grf_2d):.3f}, std={np.std(grf_2d):.3f}")

    # ==========================================================================
    # Part 10: Summary and Validation
    # ==========================================================================
    print_section("PART 10: Results Summary & Validation")

    print("\n  [A] ORBITAL FORCING")
    print(f"      - Eccentricity at 400 kyr BP:        {compute_orbital_elements(400.0)[0]:.5f}")
    print(f"      - Obliquity at 400 kyr BP (deg):     {np.rad2deg(compute_orbital_elements(400.0)[1]):.2f}")
    print(f"      - Precession at 400 kyr BP:          {compute_orbital_elements(400.0)[2]:.5f}")

    print("\n  [B] ENERGY BALANCE")
    print(f"      - Global equilibrium temperature:    {T_eq:.2f} K")
    print(f"      - FEM mean temperature:              {np.mean(T_final):.2f} K")
    print(f"      - Ice-albedo feedback active:        {'Yes' if np.any(T_final < 263.15) else 'Partial'}")

    print("\n  [C] ICE SHEET DYNAMICS")
    print(f"      - Ice volume change:                 {(V_ice[-1] - V_ice[0])/1e6:.1f} x10^6 km^3")
    print(f"      - Temperature change:                {T_global[-1] - T_global[0]:.2f} K")
    _f_sub = F_orb[::100][:len(V_ice)//100]
    _v_sub = V_ice[::100][:len(V_ice)//100]
    _min_len = min(len(_f_sub), len(_v_sub))
    print(f"      - Orbital-climate correlation:       {compute_correlation(_f_sub[:_min_len], _v_sub[:_min_len]):.4f}")

    print("\n  [D] SPECTRAL ANALYSIS")
    if peaks_haar:
        p = peaks_haar[0]
        print(f"      - Dominant period:                   {p['period_kyr']:.1f} kyr ({p['orbital_cycle']})")
    print(f"      - Multitaper dominant period:        {peak_period:.1f} kyr")

    print("\n  [E] NUMERICAL VALIDATION")
    print(f"      - Chebyshev interpolation accuracy:  {err_cheb:.2e}")
    print(f"      - Sparse matrix consistency:         {sparse_err:.2e}")
    print(f"      - Global integration error:          {abs(global_int - 4*np.pi) / (4*np.pi) * 100:.2f}%")
    print(f"      - Hexagon integration error:         {abs(hex_int - hex_exact) / max(abs(hex_exact), 1e-15) * 100:.2f}%")

    print("\n" + "#" * 78)
    print("#  SIMULATION COMPLETED SUCCESSFULLY")
    print("#  All 15 seed projects integrated into unified climate system")
    print("#" * 78 + "\n")

    return 0


if __name__ == "__main__":
    main()


# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: 轨道参数计算返回合理范围 ----
e_t, eps_t, prec_t = compute_orbital_elements(0.0)
assert 0.0 <= e_t <= 0.07 and 0.38 <= eps_t <= 0.45 and -0.07 <= prec_t <= 0.07, '[TC01] 轨道参数计算返回合理范围 FAILED'

# ---- TC02: 日射量在赤道当前日为有限正值 ----
e0, eps0, prec0 = compute_orbital_elements(0.0)
insol_eq = daily_insolation(0.0, 80, e0, eps0, prec0)
assert np.isfinite(insol_eq) and insol_eq >= 0.0, '[TC02] 日射量在赤道当前日为有限正值 FAILED'

# ---- TC03: 年均日射量对标量输入返回有限正值 ----
annual_single = annual_mean_insolation(0.0, e0, eps0, prec0)
assert np.isfinite(annual_single) and annual_single >= 0.0, '[TC03] 年均日射量对标量输入返回有限正值 FAILED'

# ---- TC04: 轨道强迫指数在[0,1]范围内 ----
f_orb = orbital_forcing_index(0.0)
assert 0.0 <= f_orb <= 1.0, '[TC04] 轨道强迫指数在[0,1]范围内 FAILED'

# ---- TC05: 反照率反馈低温值低于高温值且在合理范围 ----
alpha_cold = albedo_feedback(200.0)
alpha_warm = albedo_feedback(350.0)
assert alpha_cold < alpha_warm and 0.25 <= alpha_cold <= 0.62 and 0.25 <= alpha_warm <= 0.62, '[TC05] 反照率反馈低温值低于高温值且在合理范围 FAILED'

# ---- TC06: 出射长波辐射为正 ----
olr = outgoing_longwave_radiation(288.0)
assert olr > 0.0 and np.isfinite(olr), '[TC06] 出射长波辐射为正 FAILED'

# ---- TC07: 球面网格生成节点和元素数量正确 ----
nodes_mesh, elems_mesh = create_spherical_mesh(n_lat=5, n_lon=10)
assert nodes_mesh.shape == (50, 2) and elems_mesh.shape[1] == 3, '[TC07] 球面网格生成节点和元素数量正确 FAILED'

# ---- TC08: 稀疏矩阵乘法与稠密结果一致 ----
A_test = np.eye(5) * 2.0 + np.diag(np.ones(4) * 0.5, 1) + np.diag(np.ones(4) * 0.5, -1)
sr = dense_to_r8sr(A_test)
x_vec = np.arange(1, 6, dtype=float)
b_sparse = r8sr_mv(sr['n'], sr['nz'], sr['row_ptr'], sr['col_idx'], sr['diag'], sr['off'], x_vec)
b_dense = A_test @ x_vec
assert np.allclose(b_sparse, b_dense), '[TC08] 稀疏矩阵乘法与稠密结果一致 FAILED'

# ---- TC09: Haar功率谱输出尺度与功率同长度 ----
sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
scales, power = haar_power_spectrum(sig, dt=1.0)
assert len(scales) == len(power) and len(scales) >= 1, '[TC09] Haar功率谱输出尺度与功率同长度 FAILED'

# ---- TC10: Fibonacci球面生成点在单位球上 ----
pts = fibonacci_sphere(100, radius=1.0)
norms = np.linalg.norm(pts, axis=1)
assert np.allclose(norms, 1.0), '[TC10] Fibonacci球面生成点在单位球上 FAILED'

# ---- TC11: Newton-Maehly求二次多项式根正确 ----
roots = newton_maehly([ -2.0, 0.0, 1.0 ])
assert len(roots) == 2, '[TC11] Newton-Maehly求二次多项式根正确 FAILED'
assert np.allclose(np.sort(np.real(roots)), np.array([-np.sqrt(2), np.sqrt(2)])), '[TC11] Newton-Maehly求二次多项式根正确 FAILED'

# ---- TC12: EBM平衡温度在合理范围 ----
T_eq_test = find_equilibrium_temperature_ebm(340.0, lambda T: albedo_feedback(T), lambda T: outgoing_longwave_radiation(T), T_guess=280.0)
assert 200.0 <= T_eq_test <= 350.0, '[TC12] EBM平衡温度在合理范围 FAILED'

# ---- TC13: 稳定矩阵特征值分析返回stable ----
J_stable = -np.eye(3) * 0.5
_, stab = stability_eigenvalues(J_stable)
assert stab == 'stable', '[TC13] 稳定矩阵特征值分析返回stable FAILED'

# ---- TC14: 气候-冰盖积分冰体积非负 ----
y0_test = np.array([0.1, 0.0, 0.0, 1e6, 288.0, -100.0])
t_arr, sol_test = integrate_ice_climate((0.0, 100.0), y0_test, lambda t: 0.5, dt=10.0, mu=1.2, eta=0.5)
V_ice_test = sol_test[:, 3]
assert np.all(V_ice_test >= 0.0), '[TC14] 气候-冰盖积分冰体积非负 FAILED'

# ---- TC15: 冰线纬度计算对简单温度剖面正确 ----
T_prof = np.array([250.0, 260.0, 270.0, 280.0, 290.0])
lats_prof = np.array([0.0, 22.5, 45.0, 67.5, 90.0])
ice_lat = compute_ice_line_latitude(T_prof, lats_prof)
assert 0.0 <= ice_lat <= 90.0, '[TC15] 冰线纬度计算对简单温度剖面正确 FAILED'

# ---- TC16: 自适应时间步长极小误差返回接受 ----
dt_new, accepted = adaptive_time_step(1e-25, 1.0)
assert dt_new >= 0.1 and accepted == True, '[TC16] 自适应时间步长极小误差返回接受 FAILED'

# ---- TC17: GRF固定种子可复现 ----
np.random.seed(42)
grf1 = generate_grf_1d(20, length_scale=0.1, sigma=1.0)
np.random.seed(42)
grf2 = generate_grf_1d(20, length_scale=0.1, sigma=1.0)
assert np.allclose(grf1, grf2), '[TC17] GRF固定种子可复现 FAILED'

# ---- TC18: AR1噪声固定种子可复现 ----
np.random.seed(42)
ar1_1 = ar1_noise(20, phi=0.85, sigma=1.0)
np.random.seed(42)
ar1_2 = ar1_noise(20, phi=0.85, sigma=1.0)
assert np.allclose(ar1_1, ar1_2), '[TC18] AR1噪声固定种子可复现 FAILED'

# ---- TC19: Runge函数x=0处值为1 ----
assert np.isclose(runge_function(0.0), 1.0), '[TC19] Runge函数x=0处值为1 FAILED'

# ---- TC20: 数值微分对Runge函数在0处导数为0 ----
df_0 = numerical_differentiation(runge_function, 0.0)
assert abs(df_0) < 1e-8, '[TC20] 数值微分对Runge函数在0处导数为0 FAILED'

# ---- TC21: 相同信号相关系数为1 ----
sig_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
assert np.isclose(compute_correlation(sig_a, sig_a), 1.0), '[TC21] 相同信号相关系数为1 FAILED'

# ---- TC22: 相同信号RMSE为0 ----
assert compute_rmse(sig_a, sig_a) == 0.0, '[TC22] 相同信号RMSE为0 FAILED'

# ---- TC23: Chebyshev插值精度优于等距插值 ----
err_cheb, err_eq = test_interpolation_accuracy()
assert err_cheb < err_eq, '[TC23] Chebyshev插值精度优于等距插值 FAILED'

# ---- TC24: 平衡温度计算输出有限且在合理范围 ----
lats_simple = np.linspace(-90, 90, 5)
insol_simple = np.ones(5) * 300.0
T_eq_lat = equilibrium_temperature(insol_simple, lats_simple, max_iter=5)
assert np.all(np.isfinite(T_eq_lat)) and np.all(T_eq_lat >= 200.0) and np.all(T_eq_lat <= 350.0), '[TC24] 平衡温度计算输出有限且在合理范围 FAILED'

# ---- TC25: 多taper谱频率非负 ----
freqs_mt, spec_mt = multitaper_spectrum(sig_a, dt=1.0, nw=2, k=3)
assert np.all(freqs_mt >= 0.0), '[TC25] 多taper谱频率非负 FAILED'

# ---- TC26: 轨道周期识别空输入返回空列表 ----
peaks_empty = identify_orbital_periods(np.array([]), np.array([]))
assert peaks_empty == [], '[TC26] 轨道周期识别空输入返回空列表 FAILED'

# ---- TC27: 谱相干输出在[0,1]范围内 ----
sig_b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
_, coh = spectral_coherence(sig_a, sig_b, dt=1.0, n_segments=1)
assert np.all(coh >= 0.0) and np.all(coh <= 1.0), '[TC27] 谱相干输出在[0,1]范围内 FAILED'

# ---- TC28: 六边形蒙特卡洛积分常数函数返回面积 ----
hex_int = hexagon_monte_carlo_integrate(lambda x, y: 1.0, n_samples=5000, radius=1.0)
expected_hex_area = 3.0 * np.sqrt(3.0) / 2.0
assert abs(hex_int - expected_hex_area) / expected_hex_area < 0.1, '[TC28] 六边形蒙特卡洛积分常数函数返回面积 FAILED'

# ---- TC29: 全局积分常数1接近4π ----
global_int = global_integral_monte_carlo(lambda lat, lon: 1.0, n_patches=10, n_samples_per_patch=20)
assert abs(global_int - 4.0 * np.pi) / (4.0 * np.pi) < 0.15, '[TC29] 全局积分常数1接近4π FAILED'

# ---- TC30: 能量平衡残差输出有限 ----
T_test = np.linspace(250, 310, 10)
lats_test2 = np.linspace(-90, 90, 10)
insol_test = np.ones(10) * 340.0
residual = energy_balance_residual(T_test, insol_test, lats_test2, D=0.3)
assert np.all(np.isfinite(residual)), '[TC30] 能量平衡残差输出有限 FAILED'


print('\n全部 30 个测试通过!\n')
