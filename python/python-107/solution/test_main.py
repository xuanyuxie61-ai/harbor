"""
main.py

Unified entry point for the Spectral-Domain Optical Coherence Tomography (SD-OCT)
Computational Framework for Quantitative Tissue Characterization.

This script executes the full simulation pipeline:
1. Define multi-layered tissue optical properties
2. Generate scan patterns and tissue meshes
3. Solve forward models: DG diffusion, FEM 2D, Monte Carlo photon tracking
4. Compute spectral-domain OCT interferograms with dispersion compensation
5. Simulate biological dynamics (FHN / glycolysis) for functional OCT
6. Solve inverse problem: reconstruct optical properties from A-scans
7. Perform sensitivity analysis and parameter space exploration
8. Output quantitative metrics and convergence analysis

All parameters are self-contained; zero user input required.
"""

import numpy as np
import os
import sys

# Ensure local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from oct_physics import (
    source_spectrum_gaussian,
    dispersion_phase,
    oct_interferogram_fd,
    henvey_greenstein_phase_function,
    mie_scattering_cross_section,
    scattering_coefficients,
    diffusion_coefficient,
    transport_length,
    coherence_length_gaussian,
    signal_to_noise_ratio_oct,
    speckle_contrast,
    doppler_phase_shift
)
from spectral_integration import integrate_spectral_interferogram, integrate_depth_resolved_signal
from photon_transport_ode import (
    sensitive_photon_deriv,
    sensitive_photon_exact,
    layered_photon_transport,
    fitzhugh_nagumo_deriv,
    glycolysis_deriv,
    glycolysis_equilibrium,
    refractive_index_from_bio_state
)
from monte_carlo_scattering import (
    sphere01_sample,
    sphere01_monomial_integral,
    hypercube_surface_sample,
    hypercube_surface_distance_stats,
    hg_sample_cos_theta,
    track_photon_packet,
    simulate_oct_signal_mc
)
from dg_radiative_transfer import (
    jacobi_p,
    grad_jacobi_p,
    jacobi_gl_nodes,
    vandermonde_1d,
    d_matrix_1d,
    dg_diffusion_solve_1d,
    solve_tissue_diffusion_dg
)
from fem_optical_solver import (
    triangle_xy_to_barycentric,
    barycentric_symmetry,
    triangle_gauss_rule,
    barycentric_to_cartesian,
    assemble_fem_2d_diffusion,
    solve_fem_2d_diffusion,
    triangle_area,
    mesh_quality_min_angle
)
from nonlinear_inverse_solver import (
    givapp,
    gmres_solve,
    cg_solve,
    diffjac,
    nsol_solve,
    build_forward_model_oct,
    inverse_problem_oct
)
from biological_oscillators import (
    integrate_fitzhugh_nagumo,
    integrate_glycolysis,
    compute_refractive_index_timeseries,
    phase_shift_from_dn,
    simulate_functional_oct_signal
)
from tissue_mesh_io import (
    gmsh_mesh2d_write,
    gmsh_mesh2d_read,
    freefem_msh_read,
    freefem_msh_write,
    xy_header_write,
    xy_data_write,
    xy_data_read,
    generate_layered_tissue_mesh
)
from scan_pattern import (
    generate_rectilinear_scan,
    generate_radial_scan,
    generate_spiral_scan,
    scan_pattern_to_ascan_coords,
    sort_scan_for_bscan,
    scan_uniformity_metric,
    scan_coverage_metric
)
from utils import (
    ParameterManager,
    safe_divide,
    clip_to_finite,
    robust_mean_std,
    save_array_with_header,
    load_array_skip_header,
    convergence_rate,
    relative_error
)


def run_pipeline():
    print("=" * 72)
    print("SD-OCT Computational Framework for Quantitative Tissue Characterization")
    print("=" * 72)

    # =====================================================================
    # 1. System and tissue parameters
    # =====================================================================
    print("\n[1] Setting up OCT system and tissue parameters...")
    params = ParameterManager({
        'lambda0': 0.84,          # Central wavelength (micron)
        'delta_lambda': 0.05,     # Spectral bandwidth FWHM (micron)
        'n_medium': 1.33,         # Refractive index of water/tissue
        'k0': 2.0 * np.pi * 1.33 / 0.84,  # Central wavenumber (rad/micron)
        'delta_k': 2.0 * np.pi * 1.33 * 0.05 / (0.84 ** 2),
        'scan_x_range': (-100.0, 100.0),  # Scan range (micron)
        'scan_y_range': (-100.0, 100.0),
        'n_x_scan': 5,
        'n_y_scan': 5,
        'z_max': 500.0,           # Maximum imaging depth (micron)
        'n_z': 200,
    })

    # Layered tissue: epithelium | basement membrane | stroma
    layer_boundaries = np.array([0.0, 50.0, 55.0, 500.0])
    layer_props = [
        {'mu_a': 0.01, 'mu_s': 10.0, 'g': 0.90, 'n': 1.37},   # Epithelium
        {'mu_a': 0.05, 'mu_s': 15.0, 'g': 0.85, 'n': 1.40},   # Basement membrane
        {'mu_a': 0.005, 'mu_s': 8.0, 'g': 0.92, 'n': 1.35},   # Stroma
    ]

    print(f"  Central wavelength: {params.get('lambda0')} micron")
    print(f"  Coherence length: {coherence_length_gaussian(params.get('delta_lambda'), params.get('lambda0')):.2f} micron")
    print(f"  Layers: {len(layer_props)} (z = {list(layer_boundaries)})")

    # =====================================================================
    # 2. Scan pattern generation
    # =====================================================================
    print("\n[2] Generating OCT scan patterns...")
    scan_xy = generate_rectilinear_scan(
        params.get('scan_x_range'),
        params.get('scan_y_range'),
        params.get('n_x_scan'),
        params.get('n_y_scan')
    )
    z_depth = np.linspace(0.0, params.get('z_max'), params.get('n_z'))
    coords_3d, scan_indices = scan_pattern_to_ascan_coords(scan_xy, z_depth)
    bscan_indices = sort_scan_for_bscan(scan_xy, fast_axis='x')
    print(f"  Scan positions: {scan_xy.shape[0]}")
    print(f"  A-scans per position: {len(z_depth)}")
    print(f"  B-scan frames: {len(bscan_indices)}")
    print(f"  Scan uniformity metric: {scan_uniformity_metric(scan_xy):.4f}")
    print(f"  Scan coverage: {scan_coverage_metric(scan_xy, (params.get('scan_x_range'), params.get('scan_y_range'))):.4f}")

    # =====================================================================
    # 3. Mie scattering and optical coefficients
    # =====================================================================
    print("\n[3] Computing Mie scattering parameters...")
    sigma_s, sigma_a = mie_scattering_cross_section(
        radius=0.5, n_particle=1.45, n_medium=1.33, wavelength=params.get('lambda0')
    )
    print(f"  Mie sigma_s = {sigma_s:.6f} micron^2")
    print(f"  Mie sigma_a = {sigma_a:.6f} micron^2")

    mu_s_bulk, mu_a_bulk, g_bulk = scattering_coefficients(
        volume_fraction=0.1, radius=0.5, n_particle=1.45,
        n_medium=1.33, wavelength=params.get('lambda0')
    )
    print(f"  Bulk mu_s = {mu_s_bulk:.4f} 1/micron")
    print(f"  Bulk mu_a = {mu_a_bulk:.4f} 1/micron")
    print(f"  Anisotropy g = {g_bulk:.4f}")
    print(f"  Diffusion coefficient D = {diffusion_coefficient(mu_s_bulk, mu_a_bulk, g_bulk):.4f} micron")
    print(f"  Transport length l_tr = {transport_length(mu_s_bulk, mu_a_bulk, g_bulk):.2f} micron")

    # =====================================================================
    # 4. Spectral-domain OCT interferogram simulation
    # =====================================================================
    print("\n[4] Simulating spectral-domain OCT interferogram...")
    k_min = 2.0 * np.pi * 1.33 / (params.get('lambda0') + 0.5 * params.get('delta_lambda'))
    k_max = 2.0 * np.pi * 1.33 / (params.get('lambda0') - 0.5 * params.get('delta_lambda'))

    # Simulate A-scan from a reflector at z = 100 micron with dispersion
    z_reflector = 100.0
    phi_coeffs = [0.0, 0.0, 2.5e-4]  # quadratic dispersion

    def integrand(k):
        return oct_interferogram_fd(
            k, z_reflector, params.get('k0'), params.get('delta_k'),
            phi_coeffs, reflectivity_sample=0.01, reflectivity_reference=0.1
        )

    I_total = integrate_spectral_interferogram(integrand, k_min, k_max, n_gl=64)
    print(f"  Spectral integral (interferogram power): {I_total:.6f}")

    # Depth-resolved A-scan
    def reflectivity_model(k, z):
        # Simple layered reflectivity model
        if z < 50.0:
            mu_s = layer_props[0]['mu_s']
            mu_a = layer_props[0]['mu_a']
        elif z < 55.0:
            mu_s = layer_props[1]['mu_s']
            mu_a = layer_props[1]['mu_a']
        else:
            mu_s = layer_props[2]['mu_s']
            mu_a = layer_props[2]['mu_a']
        mu_t = mu_a + mu_s
        return mu_s * np.exp(-2.0 * mu_t * z)

    A_scan = integrate_depth_resolved_signal(reflectivity_model, z_depth, k_min, k_max, n_gl=32)
    A_scan = clip_to_finite(A_scan)
    print(f"  A-scan peak at z={z_depth[np.argmax(A_scan)]:.1f} micron")
    print(f"  A-scan max amplitude: {np.max(A_scan):.6f}")
    print(f"  A-scan SNR: {signal_to_noise_ratio_oct(np.max(A_scan), 1e-6, 1e-8):.2f} dB")

    # =====================================================================
    # 5. DG radiative transfer solver for 1D tissue diffusion
    # =====================================================================
    print("\n[5] Solving 1D diffusion equation with DG method...")
    z_dg, phi_dg = solve_tissue_diffusion_dg(
        layer_boundaries, layer_props,
        source_profile='gaussian', poly_order=4, n_elements_per_layer=4
    )
    phi_dg = clip_to_finite(phi_dg)
    print(f"  DG DOFs: {len(z_dg)}")
    print(f"  Fluence max: {np.max(phi_dg):.6f}")
    print(f"  Fluence at surface: {phi_dg[0]:.6f}")
    print(f"  Fluence at deepest point: {phi_dg[-1]:.6f}")

    # =====================================================================
    # 6. FEM 2D optical solver on triangular mesh
    # =====================================================================
    print("\n[6] Solving 2D FEM diffusion on tissue cross-section...")
    nodes_2d, elements_2d = generate_layered_tissue_mesh(
        layer_boundaries, radial_extent=50.0, n_r=8, n_z_per_layer=4
    )
    print(f"  Mesh nodes: {nodes_2d.shape[0]}, elements: {elements_2d.shape[0]}")
    print(f"  Mesh min angle: {mesh_quality_min_angle(nodes_2d, elements_2d):.2f} deg")

    def D_func_2d(x, y):
        z = y
        for i in range(len(layer_boundaries) - 1):
            if layer_boundaries[i] <= z <= layer_boundaries[i + 1]:
                mu_s_p = (1.0 - layer_props[i]['g']) * layer_props[i]['mu_s']
                mu_a = layer_props[i]['mu_a']
                denom = 3.0 * (mu_s_p + mu_a)
                return 1.0 / denom if denom > 0 else 1e6
        return 1e6

    def mu_a_func_2d(x, y):
        z = y
        for i in range(len(layer_boundaries) - 1):
            if layer_boundaries[i] <= z <= layer_boundaries[i + 1]:
                return layer_props[i]['mu_a']
        return 0.0

    def source_func_2d(x, y):
        return 1.0 if 0.0 <= y <= layer_boundaries[-1] else 0.0

    # Dirichlet on top boundary
    top_nodes = np.where(np.abs(nodes_2d[:, 1] - layer_boundaries[0]) < 1e-6)[0]
    phi_2d = solve_fem_2d_diffusion(
        nodes_2d, elements_2d, D_func_2d, mu_a_func_2d, source_func_2d,
        dirichlet_nodes=top_nodes, dirichlet_values=np.ones(len(top_nodes))
    )
    phi_2d = clip_to_finite(phi_2d)
    print(f"  2D FEM solution range: [{np.min(phi_2d):.6f}, {np.max(phi_2d):.6f}]")

    # =====================================================================
    # 7. Monte Carlo photon tracking
    # =====================================================================
    print("\n[7] Monte Carlo photon transport simulation...")
    n_photons = 200
    signal_mc, depths_mc = simulate_oct_signal_mc(
        n_photons, source_z=0.0, detector_z=0.0,
        layer_boundaries=layer_boundaries, layer_props=layer_props, max_steps=50
    )
    print(f"  Photons simulated: {n_photons}")
    print(f"  Detected signal fraction: {signal_mc:.6f}")
    if len(depths_mc) > 0:
        print(f"  Mean scattering depth: {np.mean(depths_mc):.2f} micron")
        print(f"  Depth std: {np.std(depths_mc):.2f} micron")

    # Speckle contrast
    if len(depths_mc) > 1:
        # Use intensity proxy from photon weights at each depth bin
        hist, bin_edges = np.histogram(depths_mc, bins=20)
        C = speckle_contrast(hist.astype(float))
        print(f"  Speckle contrast: {C:.4f}")

    # =====================================================================
    # 8. Biological oscillators for functional OCT
    # =====================================================================
    print("\n[8] Simulating biological dynamics for functional OCT...")
    t_bio = np.linspace(0.0, 100.0, 500)

    # FitzHugh-Nagumo
    fhn_result = simulate_functional_oct_signal(
        t_bio,
        bio_params={'type': 'FHN', 'y0': [0.1, 0.0], 'model_params': {'a': 0.7, 'b': 0.8, 'c': 12.5, 'd': 0.5}},
        oct_params={'lambda0': 0.84, 'path_length': 100.0, 'n0': 1.33, 'alpha_eo': 5e-4, 'alpha_thermo': 2e-3}
    )
    print(f"  FHN refractive index range: [{np.min(fhn_result['n_t']):.6f}, {np.max(fhn_result['n_t']):.6f}]")
    print(f"  FHN max phase shift: {np.max(np.abs(fhn_result['phase_shift'])):.6f} rad")

    # Glycolysis
    gly_result = simulate_functional_oct_signal(
        t_bio,
        bio_params={'type': 'glycolysis', 'y0': [0.9, 0.7], 'model_params': {'a': 0.08, 'b': 0.6}},
        oct_params={'lambda0': 0.84, 'path_length': 100.0, 'n0': 1.33, 'alpha_eo': 5e-4, 'alpha_thermo': 2e-3}
    )
    print(f"  Glycolysis refractive index range: [{np.min(gly_result['n_t']):.6f}, {np.max(gly_result['n_t']):.6f}]")

    # =====================================================================
    # 9. Inverse problem: reconstruct optical properties
    # =====================================================================
    print("\n[9] Solving inverse problem: tissue property reconstruction...")
    z_scan_inv = np.linspace(0.0, 200.0, 50)
    # Create synthetic measurement from known parameters
    # Use first 2 layers for the inverse problem (6 parameters)
    inv_boundaries = layer_boundaries[:3]
    forward_known = build_forward_model_oct(z_scan_inv, inv_boundaries, k_min, k_max, n_gl=16)
    true_params = np.array([
        layer_props[0]['mu_a'], layer_props[0]['mu_s'], layer_props[0]['g'],
        layer_props[1]['mu_a'], layer_props[1]['mu_s'], layer_props[1]['g'],
    ])
    A_measured = forward_known(true_params)
    # Add small noise
    A_measured = A_measured * (1.0 + 0.01 * np.random.randn(len(A_measured)))
    A_measured = clip_to_finite(A_measured)

    param_guess = np.array([0.02, 5.0, 0.5, 0.02, 5.0, 0.5])
    try:
        params_recon, residuals = inverse_problem_oct(
            z_scan_inv, A_measured, inv_boundaries, param_guess, k_min, k_max, max_iter=10
        )
        print(f"  True params L0:   mu_a={true_params[0]:.4f}, mu_s={true_params[1]:.4f}, g={true_params[2]:.4f}")
        print(f"  True params L1:   mu_a={true_params[3]:.4f}, mu_s={true_params[4]:.4f}, g={true_params[5]:.4f}")
        print(f"  Recon params L0:  mu_a={params_recon[0]:.4f}, mu_s={params_recon[1]:.4f}, g={params_recon[2]:.4f}")
        print(f"  Recon params L1:  mu_a={params_recon[3]:.4f}, mu_s={params_recon[4]:.4f}, g={params_recon[5]:.4f}")
        print(f"  Initial residual: {residuals[0]:.6f}")
        if len(residuals) > 1:
            print(f"  Final residual:   {residuals[-1]:.6f}")
            print(f"  Convergence rate: {convergence_rate(residuals):.4f}")
    except Exception as e:
        print(f"  Inverse problem encountered issue: {e}")
        print(f"  True params L0:   mu_a={true_params[0]:.4f}, mu_s={true_params[1]:.4f}, g={true_params[2]:.4f}")
        print(f"  True params L1:   mu_a={true_params[3]:.4f}, mu_s={true_params[4]:.4f}, g={true_params[5]:.4f}")

    # =====================================================================
    # 10. Sensitivity and parameter space exploration
    # =====================================================================
    print("\n[10] Sensitivity analysis and parameter space exploration...")
    # Sensitive ODE analysis
    y0_sens = np.array([1.01, -1.0])
    t_sens = np.linspace(0.0, 5.0, 100)
    y_exact = sensitive_photon_exact(t_sens, y0_sens, growth_rate=0.5)
    print(f"  Sensitive ODE: initial density {y0_sens[0]:.4f}")
    print(f"  After 5 microns: density {y_exact[-1,0]:.4f}, flux {y_exact[-1,1]:.4f}")

    # Hypercube surface parameter exploration
    dmu, dvar = hypercube_surface_distance_stats(n=500, d=5)
    print(f"  Hypercube surface distance (d=5): mean={dmu:.4f}, var={dvar:.6f}")

    # Sphere sampling for scattering direction validation
    sphere_pts = sphere01_sample(100)
    # Verify normalization
    norms = np.linalg.norm(sphere_pts, axis=0)
    print(f"  Sphere sample mean norm: {np.mean(norms):.6f} (should be 1.0)")
    # Integral of x^2 over sphere should be 4*pi/3
    int_x2 = sphere01_monomial_integral([2, 0, 0])
    print(f"  Sphere integral x^2: {int_x2:.6f} (exact: {4*np.pi/3:.6f})")

    # =====================================================================
    # 11. Mesh I/O validation
    # =====================================================================
    print("\n[11] Validating mesh I/O formats...")
    gmsh_file = os.path.join(current_dir, "test_mesh.msh")
    freefem_file = os.path.join(current_dir, "test_mesh.msh_ff")
    xy_file = os.path.join(current_dir, "test_scan.xy")

    gmsh_mesh2d_write(gmsh_file, nodes_2d, elements_2d)
    nodes_read, elems_read = gmsh_mesh2d_read(gmsh_file)
    print(f"  Gmsh I/O: nodes {nodes_2d.shape[0]} -> {nodes_read.shape[0]}, elems {elements_2d.shape[0]} -> {elems_read.shape[0]}")

    freefem_msh_write(freefem_file, nodes_2d, elements_2d)
    nodes_ff, elems_ff, labels_ff, edges_ff = freefem_msh_read(freefem_file)
    print(f"  FreeFem++ I/O: nodes {nodes_2d.shape[0]} -> {nodes_ff.shape[0]}")

    xy_data_write(xy_file, scan_xy)
    scan_read = xy_data_read(xy_file)
    print(f"  XY I/O: points {scan_xy.shape[0]} -> {scan_read.shape[0]}")

    # Cleanup temporary files
    for f in [gmsh_file, freefem_file, xy_file]:
        if os.path.exists(f):
            os.remove(f)

    # =====================================================================
    # 12. Final summary metrics
    # =====================================================================
    print("\n" + "=" * 72)
    print("SIMULATION SUMMARY")
    print("=" * 72)
    print(f"Tissue layers:                {len(layer_props)}")
    print(f"Scan positions:               {scan_xy.shape[0]}")
    print(f"A-scan depth samples:         {len(z_depth)}")
    print(f"DG polynomial order:          4")
    print(f"2D FEM mesh elements:         {elements_2d.shape[0]}")
    print(f"MC photons tracked:           {n_photons}")
    print(f"Spectral quadrature points:   64 (Gauss-Legendre)")
    print(f"Coherence length:             {coherence_length_gaussian(params.get('delta_lambda'), params.get('lambda0')):.2f} micron")
    print(f"A-scan peak depth:            {z_depth[np.argmax(A_scan)]:.1f} micron")
    print(f"DG fluence at surface:        {phi_dg[0]:.6f}")
    print(f"FHN max phase shift:          {np.max(np.abs(fhn_result['phase_shift'])):.6f} rad")
    print(f"Speckle contrast:             {speckle_contrast(A_scan):.4f}")
    print("=" * 72)
    print("Pipeline completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    np.random.seed(42)
    run_pipeline()

# ================================================================
# 测试用例（68个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: safe_divide normal division ----
result = safe_divide(np.array([10.0, 5.0]), np.array([2.0, 1.0]))
assert np.allclose(result, [5.0, 5.0]), '[TC01] safe_divide normal FAILED'

# ---- TC02: safe_divide zero denominator uses fill_value ----
result = safe_divide(np.array([5.0, 3.0]), np.array([0.0, 0.0]), fill_value=-1.0)
assert np.allclose(result, [-1.0, -1.0]), '[TC02] safe_divide zero FAILED'

# ---- TC03: clip_to_finite replaces NaN/Inf and clips ----
arr = np.array([1.0, np.nan, np.inf, -np.inf, -2e7, 5e7])
clipped = clip_to_finite(arr, bounds=(-10.0, 10.0))
assert np.all(np.isfinite(clipped)), '[TC03] clip_to_finite finite FAILED'
assert np.all((clipped >= -10.0) & (clipped <= 10.0)), '[TC03] clip_to_finite bounds FAILED'

# ---- TC04: convergence_rate from geometric sequence ~ 0.5 ----
errors = np.array([1.0, 0.5, 0.25, 0.125])
rate = convergence_rate(errors)
assert 0.4 < rate < 0.6, '[TC04] convergence_rate FAILED'

# ---- TC05: relative_error for identical vectors is zero ----
a = np.array([1.0, 2.0, 3.0])
assert relative_error(a, a) < 1e-14, '[TC05] relative_error self FAILED'

# ---- TC06: ParameterManager get/set/get_all ----
pm = ParameterManager({'a': 1.0, 'b': 2.0})
pm.set(c=3.0)
assert pm.get('a') == 1.0, '[TC06] ParameterManager get FAILED'
assert pm.get('c') == 3.0, '[TC06] ParameterManager set FAILED'
assert pm.get_all()['b'] == 2.0, '[TC06] ParameterManager get_all FAILED'

# ---- TC07: ParameterManager validate_ranges ----
pm2 = ParameterManager({'x': 5.0})
ok, viol = pm2.validate_ranges({'x': (0.0, 10.0)})
assert ok, '[TC07] validate_ranges ok FAILED'
ok2, viol2 = pm2.validate_ranges({'x': (0.0, 3.0)})
assert not ok2, '[TC07] validate_ranges violate FAILED'

# ---- TC08: source_spectrum_gaussian peak higher than off-peak ----
k0 = 10.0; dk = 1.0
S_peak = source_spectrum_gaussian(np.array([k0]), k0, dk)
S_off = source_spectrum_gaussian(np.array([k0 + 3.0*dk]), k0, dk)
assert S_peak[0] > S_off[0], '[TC08] gaussian spectrum peak FAILED'

# ---- TC09: source_spectrum_gaussian all non-negative ----
k = np.linspace(5, 15, 100)
S = source_spectrum_gaussian(k, 10.0, 1.0)
assert np.all(S >= 0), '[TC09] gaussian spectrum non-negative FAILED'

# ---- TC10: dispersion_phase with all-zero coefficients is zero ----
k = np.linspace(5, 15, 100)
phi = dispersion_phase(k, 10.0, [0.0, 0.0, 0.0])
assert np.allclose(phi, 0.0), '[TC10] dispersion_phase zero FAILED'

# ---- TC11: oct_interferogram_fd non-negative ----
k = np.linspace(8, 12, 50)
I = oct_interferogram_fd(k, 100.0, 10.0, 1.0, [0.0, 0.0, 0.0], 0.01, 0.1)
assert np.all(I >= 0), '[TC11] interferogram non-negative FAILED'

# ---- TC12: henvey_greenstein_phase_function g=0 isotropic => 0.5 ----
cos_th = np.linspace(-1, 1, 100)
p = henvey_greenstein_phase_function(cos_th, 0.0)
assert np.allclose(p, 0.5, atol=1e-10), '[TC12] HG isotropic FAILED'

# ---- TC13: diffusion_coefficient known computation ----
D = diffusion_coefficient(10.0, 0.01, 0.9)
mu_s_prime = 0.1 * 10.0
expected_D = 1.0 / (3.0 * (mu_s_prime + 0.01))
assert abs(D - expected_D) < 1e-10, '[TC13] diffusion_coefficient FAILED'

# ---- TC14: transport_length known computation ----
ltr = transport_length(10.0, 0.01, 0.9)
mu_s_prime = 0.1 * 10.0
expected_ltr = 1.0 / (mu_s_prime + 0.01)
assert abs(ltr - expected_ltr) < 1e-10, '[TC14] transport_length FAILED'

# ---- TC15: coherence_length_gaussian exact formula ----
lc = coherence_length_gaussian(0.05, 0.84)
expected_lc = (2.0 * np.log(2.0) / np.pi) * (0.84**2 / 0.05)
assert abs(lc - expected_lc) < 1e-10, '[TC15] coherence_length FAILED'

# ---- TC16: speckle_contrast constant signal is zero ----
C = speckle_contrast(np.array([1.0, 1.0, 1.0, 1.0]))
assert C < 1e-14, '[TC16] speckle_contrast constant FAILED'

# ---- TC17: doppler_phase_shift zero velocity is zero ----
df = doppler_phase_shift(0.0, 1.33, 0.84)
assert df == 0.0, '[TC17] doppler zero velocity FAILED'

# ---- TC18: signal_to_noise_ratio_oct zero denominator returns large ----
snr = signal_to_noise_ratio_oct(0.01, 0.0, 0.0)
assert snr > 100, '[TC18] SNR large value FAILED'

# ---- TC19: legendre_ek_compute weights sum to 2.0 ----
from spectral_integration import legendre_ek_compute
x_gl, w_gl = legendre_ek_compute(10)
assert abs(np.sum(w_gl) - 2.0) < 1e-12, '[TC19] GL weights sum FAILED'

# ---- TC20: gauss_legendre_map scaled weights sum to (b-a) ----
from spectral_integration import gauss_legendre_map
x_m, w_m = gauss_legendre_map(-1.0, 3.0, 10)
assert abs(np.sum(w_m) - 4.0) < 1e-12, '[TC20] mapped weights sum FAILED'

# ---- TC21: integrate_spectral_interferogram constant integrand ----
import numpy as np
def const_func(k):
    return np.ones_like(k)
I = integrate_spectral_interferogram(const_func, 0.0, 1.0, n_gl=16)
assert abs(I - 1.0) < 1e-10, '[TC21] spectral integral constant FAILED'

# ---- TC22: sensitive_photon_exact returns finite values of correct shape ----
import numpy as np
t = np.array([0.0, 1.0, 2.0])
y0 = np.array([1.01, -1.0])
y = sensitive_photon_exact(t, y0, growth_rate=0.5)
assert y.shape == (3, 2), '[TC22] exact shape FAILED'
assert np.all(np.isfinite(y)), '[TC22] exact finite FAILED'
# At t=0, analytic formula yields phi(0) = 1.0 by design
assert abs(y[0, 0] - 1.0) < 1e-14, '[TC22] exact at t=0 returns 1.0 FAILED'

# ---- TC23: fitzhugh_nagumo_deriv returns shape (2,) and finite ----
import numpy as np
dydt = fitzhugh_nagumo_deriv(0.0, np.array([0.1, 0.0]))
assert dydt.shape == (2,), '[TC23] FHN deriv shape FAILED'
assert np.all(np.isfinite(dydt)), '[TC23] FHN deriv finite FAILED'

# ---- TC24: glycolysis_equilibrium analytic formula ----
import numpy as np
yeq = glycolysis_equilibrium(0.08, 0.6)
u_star = 0.6
v_star = 0.6 / (0.08 + 0.36)
assert abs(yeq[0] - u_star) < 1e-10, '[TC24] glycolysis eq u FAILED'
assert abs(yeq[1] - v_star) < 1e-10, '[TC24] glycolysis eq v FAILED'

# ---- TC25: glycolysis_deriv at equilibrium is approx zero ----
import numpy as np
yeq = glycolysis_equilibrium(0.08, 0.6)
dydt = glycolysis_deriv(0.0, yeq, 0.08, 0.6)
assert np.linalg.norm(dydt) < 1e-10, '[TC25] glycolysis at eq FAILED'

# ---- TC26: refractive_index_from_bio_state baseline ----
import numpy as np
n = refractive_index_from_bio_state(0.0, 0.0, n0=1.33, alpha_eo=1e-4, alpha_thermo=1e-3)
assert abs(n - 1.33) < 1e-10, '[TC26] refractive index baseline FAILED'

# ---- TC27: sphere01_monomial_integral x^2 exact = 4*pi/3 ----
import numpy as np
int_x2 = sphere01_monomial_integral([2, 0, 0])
expected_x2 = 4.0 * np.pi / 3.0
assert abs(int_x2 - expected_x2) < 1e-10, '[TC27] sphere integral x^2 FAILED'

# ---- TC28: sphere01_monomial_integral odd exponent => 0 ----
import numpy as np
int_odd = sphere01_monomial_integral([1, 0, 0])
assert int_odd == 0.0, '[TC28] sphere integral odd FAILED'

# ---- TC29: sphere01_monomial_integral all zero = sphere area 4*pi ----
import numpy as np
int_all0 = sphere01_monomial_integral([0, 0, 0])
assert abs(int_all0 - 4.0 * np.pi) < 1e-10, '[TC29] sphere area FAILED'

# ---- TC30: hg_sample_cos_theta reproducibility with fixed seed ----
import numpy as np
np.random.seed(42)
c1 = hg_sample_cos_theta(100, 0.8)
np.random.seed(42)
c2 = hg_sample_cos_theta(100, 0.8)
assert np.allclose(c1, c2), '[TC30] HG reproducibility FAILED'

# ---- TC31: hg_sample_cos_theta returns values in [-1,1] ----
import numpy as np
np.random.seed(123)
ct = hg_sample_cos_theta(200, 0.5)
assert np.all((ct >= -1.0) & (ct <= 1.0)), '[TC31] HG range FAILED'

# ---- TC32: jacobi_p P_0(x) = 1 ----
import numpy as np
P0 = jacobi_p(np.array([0.0, 0.5, -0.5]), 0.0, 0.0, 0)
assert np.allclose(P0, 1.0), '[TC32] jacobi P0 FAILED'

# ---- TC33: jacobi_gl_nodes endpoints are -1 and 1 ----
import numpy as np
r = jacobi_gl_nodes(0.0, 0.0, 4)
assert abs(r[0] - (-1.0)) < 1e-14, '[TC33] GL nodes left FAILED'
assert abs(r[-1] - 1.0) < 1e-14, '[TC33] GL nodes right FAILED'

# ---- TC34: triangle_area unit right triangle = 0.5 ----
import numpy as np
v = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
area = triangle_area(v)
assert abs(area - 0.5) < 1e-14, '[TC34] triangle area FAILED'

# ---- TC35: triangle_xy_to_barycentric sums to 1 ----
import numpy as np
xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.25, 0.25]])
xyz = triangle_xy_to_barycentric(xy)
assert np.allclose(np.sum(xyz, axis=1), 1.0), '[TC35] barycentric sum FAILED'

# ---- TC36: givapp identity rotation (c=1, s=0) ----
import numpy as np
c = np.array([1.0])
s = np.array([0.0])
vin = np.array([3.0, 7.0])
vout = givapp(c, s, vin, 1)
assert np.allclose(vout, vin), '[TC36] givapp identity FAILED'

# ---- TC37: givapp 90-degree rotation (c=0, s=1) ----
import numpy as np
c = np.array([0.0])
s = np.array([1.0])
vin = np.array([2.0, 5.0])
vout = givapp(c, s, vin, 1)
assert abs(vout[0] - (-5.0)) < 1e-14, '[TC37] givapp rotate 0 FAILED'
assert abs(vout[1] - 2.0) < 1e-14, '[TC37] givapp rotate 1 FAILED'

# ---- TC38: phase_shift_from_dn zero change => zero shift ----
import numpy as np
dphi = phase_shift_from_dn(0.0, 100.0, 0.84)
assert dphi == 0.0, '[TC38] phase shift zero FAILED'

# ---- TC39: generate_rectilinear_scan correct shape ----
import numpy as np
coords = generate_rectilinear_scan((-100, 100), (-100, 100), 5, 5)
assert coords.shape == (25, 2), '[TC39] rectilinear scan shape FAILED'

# ---- TC40: generate_radial_scan returns points within radius ----
import numpy as np
coords_r = generate_radial_scan((0, 0), 50.0, 8, 10)
dists = np.linalg.norm(coords_r, axis=1)
assert np.all(dists <= 50.0 + 1e-10), '[TC40] radial scan radius FAILED'

# ---- TC41: generate_spiral_scan returns n_points ----
import numpy as np
coords_s = generate_spiral_scan((0, 0), 100.0, 60)
assert coords_s.shape == (60, 2), '[TC41] spiral scan shape FAILED'

# ---- TC42: scan_uniformity_metric non-negative ----
import numpy as np
coords_u = generate_rectilinear_scan((-100, 100), (-100, 100), 10, 10)
u = scan_uniformity_metric(coords_u)
assert u >= 0, '[TC42] uniformity non-negative FAILED'

# ---- TC43: scan_coverage_metric in [0,1] ----
import numpy as np
coords_cvg = generate_rectilinear_scan((-100, 100), (-100, 100), 10, 10)
cov = scan_coverage_metric(coords_cvg, ((-100, 100), (-100, 100)))
assert 0.0 <= cov <= 1.0, '[TC43] coverage range FAILED'

# ---- TC44: generate_layered_tissue_mesh output sizes ----
import numpy as np
nodes, elems = generate_layered_tissue_mesh(np.array([0.0, 50.0, 500.0]), 50.0, 4, 4)
assert nodes.shape[0] > 0, '[TC44] mesh nodes positive FAILED'
assert elems.shape[0] > 0, '[TC44] mesh elements positive FAILED'
assert nodes.shape[1] == 2, '[TC44] mesh node dim FAILED'
assert elems.shape[1] == 3, '[TC44] mesh elem dim FAILED'

# ---- TC45: save/load roundtrip preserves values ----
import numpy as np
import os
test_arr = np.array([1.5, -3.2, 0.0])
save_array_with_header('_test_roundtrip.txt', test_arr, ['test header'])
loaded = load_array_skip_header('_test_roundtrip.txt')
assert loaded.shape == test_arr.shape, '[TC45] roundtrip shape FAILED'
assert np.allclose(loaded.flatten(), test_arr), '[TC45] roundtrip values FAILED'
os.remove('_test_roundtrip.txt')

# ---- TC46: robust_mean_std filters outliers ----
import numpy as np
arr = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 100.0, -100.0])
m, s = robust_mean_std(arr)
assert 1.0 < m < 3.0, '[TC46] robust mean FAILED'
assert s < 2.0, '[TC46] robust std FAILED'

# ---- TC47: ode_midpoint_solve for y'=0 returns constant ----
import numpy as np
from photon_transport_ode import ode_midpoint_solve
def zero_deriv(t, y):
    return np.array([0.0])
t_mid, y_mid = ode_midpoint_solve(zero_deriv, 0.0, 1.0, np.array([5.0]), 10)
assert np.allclose(y_mid, 5.0), '[TC47] midpoint constant FAILED'

# ---- TC48: layered_photon_transport returns finite values ----
import numpy as np
lb = np.array([0.0, 50.0, 500.0])
lp = [{'mu_a': 0.01, 'mu_s': 10.0, 'g': 0.9}, {'mu_a': 0.005, 'mu_s': 8.0, 'g': 0.92}]
z_all, y_all = layered_photon_transport(lb, lp, np.array([1.0, 0.0]), n_steps_per_layer=20)
assert len(z_all) > 0, '[TC48] layered transport z FAILED'
assert y_all.shape[0] == len(z_all), '[TC48] layered transport shape FAILED'
assert np.all(np.isfinite(y_all)), '[TC48] layered transport finite FAILED'

# ---- TC49: integrate_depth_resolved_signal returns non-negative amplitudes ----
import numpy as np
def simple_refl(k, z):
    return 0.01 * np.ones_like(k)
z_arr = np.linspace(0, 200, 30)
A = integrate_depth_resolved_signal(simple_refl, z_arr, 8.0, 12.0, n_gl=16)
assert np.all(A >= 0), '[TC49] A-scan non-negative FAILED'

# ---- TC50: triangle_gauss_rule order=7 returns 7 points ----
import numpy as np
from fem_optical_solver import triangle_gauss_rule
bary_7, w_7 = triangle_gauss_rule(order=7)
assert len(w_7) == 7, '[TC50] gauss rule 7 pts FAILED'
assert abs(np.sum(w_7) - 0.5) < 1e-10, '[TC50] gauss rule sum to 0.5 FAILED'

# ---- TC51: barycentric_to_cartesian maps centroid ----
import numpy as np
from fem_optical_solver import barycentric_to_cartesian
verts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
centroid_bary = np.array([[1.0/3.0, 1.0/3.0, 1.0/3.0]])
centroid_xy = barycentric_to_cartesian(centroid_bary, verts)
assert abs(centroid_xy[0, 0] - 2.0/3.0) < 1e-14, '[TC51] bary_to_cart x FAILED'
assert abs(centroid_xy[0, 1] - 2.0/3.0) < 1e-14, '[TC51] bary_to_cart y FAILED'

# ---- TC52 : gmsh_mesh2d_write / read roundtrip ----
import numpy as np
import os
n_test = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
e_test = np.array([[0, 1, 2]])
gmsh_mesh2d_write('_test_gmsh.msh', n_test, e_test)
nr, er = gmsh_mesh2d_read('_test_gmsh.msh')
assert nr.shape == n_test.shape, '[TC52] gmsh read nodes shape FAILED'
assert er.shape == e_test.shape, '[TC52] gmsh read elems shape FAILED'
os.remove('_test_gmsh.msh')

# ---- TC53 : xy_data_write / read roundtrip ----
import numpy as np
import os
xy_test = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
xy_data_write('_test_xy.txt', xy_test)
xy_read = xy_data_read('_test_xy.txt')
assert xy_read.shape == xy_test.shape, '[TC53] xy read shape FAILED'
assert np.allclose(xy_read, xy_test), '[TC53] xy read values FAILED'
os.remove('_test_xy.txt')

# ---- TC54: integrate_fitzhugh_nagumo returns correct shapes ----
import numpy as np
from biological_oscillators import integrate_fitzhugh_nagumo
t_fhn, y_fhn = integrate_fitzhugh_nagumo([0.1, 0.0], (0.0, 10.0), n_steps=100)
assert len(t_fhn) == 101, '[TC54] FHN t length FAILED'
assert y_fhn.shape == (101, 2), '[TC54] FHN y shape FAILED'
assert np.all(np.isfinite(y_fhn)), '[TC54] FHN y finite FAILED'

# ---- TC55: integrate_glycolysis returns correct shapes ----
import numpy as np
from biological_oscillators import integrate_glycolysis
t_gly, y_gly = integrate_glycolysis([0.9, 0.7], (0.0, 10.0), n_steps=100)
assert len(t_gly) == 101, '[TC55] glycolysis t length FAILED'
assert y_gly.shape == (101, 2), '[TC55] glycolysis y shape FAILED'
assert np.all(np.isfinite(y_gly)), '[TC55] glycolysis y finite FAILED'

# ---- TC56: compute_refractive_index_timeseries in physical range ----
import numpy as np
from biological_oscillators import compute_refractive_index_timeseries
t_tmp = np.linspace(0, 1, 10)
v_tmp = np.zeros(10)
u_tmp = np.zeros(10)
n_ts = compute_refractive_index_timeseries(t_tmp, v_tmp, u_tmp)
assert np.all((n_ts >= 1.30) & (n_ts <= 1.50)), '[TC56] n timeseries range FAILED'

# ---- TC57: simulate_functional_oct_signal returns all keys ----
import numpy as np
np.random.seed(42)
t_bio_test = np.linspace(0, 50, 200)
result = simulate_functional_oct_signal(
    t_bio_test,
    bio_params={'type': 'FHN', 'y0': [0.1, 0.0], 'model_params': {'a': 0.7, 'b': 0.8, 'c': 12.5, 'd': 0.5}},
    oct_params={'lambda0': 0.84, 'path_length': 100.0, 'n0': 1.33, 'alpha_eo': 5e-4, 'alpha_thermo': 2e-3}
)
assert 't' in result, '[TC57] functional OCT missing t FAILED'
assert 'n_t' in result, '[TC57] functional OCT missing n_t FAILED'
assert 'phase_shift' in result, '[TC57] functional OCT missing phase_shift FAILED'
assert 'intensity_modulation' in result, '[TC57] functional OCT missing intensity_mod FAILED'

# ---- TC58: gmsh_mesh2d_read of generate_layered_tissue_mesh roundtrip ----
import numpy as np
import os
nodes_m, elems_m = generate_layered_tissue_mesh(np.array([0.0, 50.0, 500.0]), 50.0, 4, 4)
gmsh_mesh2d_write('_test_mesh2.msh', nodes_m, elems_m)
nr2, er2 = gmsh_mesh2d_read('_test_mesh2.msh')
assert nr2.shape[0] == nodes_m.shape[0], '[TC58] gmsh mesh roundtrip nodes FAILED'
assert er2.shape[0] == elems_m.shape[0], '[TC58] gmsh mesh roundtrip elems FAILED'
os.remove('_test_mesh2.msh')

# ---- TC59: freefem_msh_write/read roundtrip ----
import numpy as np
import os
n_ff = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
e_ff = np.array([[0, 1, 3], [0, 3, 2]])
freefem_msh_write('_test_ff.msh', n_ff, e_ff)
n_ff_r, e_ff_r, labels_r, edges_r = freefem_msh_read('_test_ff.msh')
assert n_ff_r.shape[0] == n_ff.shape[0], '[TC59] freefem read nodes FAILED'
assert e_ff_r.shape[0] == e_ff.shape[0], '[TC59] freefem read elems FAILED'
os.remove('_test_ff.msh')

# ---- TC60: sphere01_sample reproducibility ----
import numpy as np
np.random.seed(42)
s1 = sphere01_sample(50)
np.random.seed(42)
s2 = sphere01_sample(50)
assert np.allclose(s1, s2), '[TC60] sphere sample reproducibility FAILED'

# ---- TC61: sphere01_sample unit norm ----
import numpy as np
np.random.seed(7)
pts = sphere01_sample(100)
norms = np.linalg.norm(pts, axis=0)
assert np.allclose(norms, 1.0), '[TC61] sphere sample norm FAILED'

# ---- TC62: hypercube_surface_sample points in [0,1] ----
import numpy as np
np.random.seed(99)
hp = hypercube_surface_sample(50, 5)
assert np.all((hp >= 0) & (hp <= 1)), '[TC62] hypercube bounds FAILED'

# ---- TC63: hypercube_surface_distance_stats returns positive mean ----
import numpy as np
np.random.seed(42)
dmu, dvar = hypercube_surface_distance_stats(200, 5)
assert dmu > 0, '[TC63] hypercube distance mean positive FAILED'
assert dvar >= 0, '[TC63] hypercube distance var non-negative FAILED'

# ---- TC64: sort_scan_for_bscan returns list of indices ----
import numpy as np
scan_xy_test = np.array([[0., 0.], [1., 0.], [2., 0.], [0., 1.], [1., 1.], [2., 1.]])
bscans = sort_scan_for_bscan(scan_xy_test, fast_axis='x', tol=1e-8)
assert len(bscans) == 2, '[TC64] bscan count FAILED'

# ---- TC65: scan_pattern_to_ascan_coords correct shape ----
import numpy as np
scan_xy_small = np.array([[0.0, 0.0], [10.0, 0.0]])
z_d = np.linspace(0, 100, 5)
c3d, si = scan_pattern_to_ascan_coords(scan_xy_small, z_d)
assert c3d.shape == (10, 3), '[TC65] ascan coords shape FAILED'
assert len(si) == 10, '[TC65] ascan indices len FAILED'

# ---- TC66: convergence_rate with single element returns 0 ----
import numpy as np
rate_single = convergence_rate(np.array([1.0]))
assert rate_single == 0.0, '[TC66] convergence single FAILED'

# ---- TC67: relative_error of different vectors positive ----
import numpy as np
a = np.array([1.0, 0.0])
b = np.array([0.0, 1.0])
err = relative_error(a, b)
assert err > 0, '[TC67] relative_error positive FAILED'

# ---- TC68: safe_divide mixed zero/nonzero ----
import numpy as np
r = safe_divide(np.array([10.0, 5.0, 3.0]), np.array([2.0, 0.0, 1.0]), fill_value=0.0)
assert r[0] == 5.0, '[TC68] safe_divide mixed a FAILED'
assert r[1] == 0.0, '[TC68] safe_divide mixed b FAILED'
assert r[2] == 3.0, '[TC68] safe_divide mixed c FAILED'

print('\n全部 68 个测试通过!\n')
