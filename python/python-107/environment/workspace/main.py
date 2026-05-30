
import numpy as np
import os
import sys


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




    print("\n[1] Setting up OCT system and tissue parameters...")
    params = ParameterManager({
        'lambda0': 0.84,
        'delta_lambda': 0.05,
        'n_medium': 1.33,
        'k0': 2.0 * np.pi * 1.33 / 0.84,
        'delta_k': 2.0 * np.pi * 1.33 * 0.05 / (0.84 ** 2),
        'scan_x_range': (-100.0, 100.0),
        'scan_y_range': (-100.0, 100.0),
        'n_x_scan': 5,
        'n_y_scan': 5,
        'z_max': 500.0,
        'n_z': 200,
    })


    layer_boundaries = np.array([0.0, 50.0, 55.0, 500.0])
    layer_props = [
        {'mu_a': 0.01, 'mu_s': 10.0, 'g': 0.90, 'n': 1.37},
        {'mu_a': 0.05, 'mu_s': 15.0, 'g': 0.85, 'n': 1.40},
        {'mu_a': 0.005, 'mu_s': 8.0, 'g': 0.92, 'n': 1.35},
    ]

    print(f"  Central wavelength: {params.get('lambda0')} micron")
    print(f"  Coherence length: {coherence_length_gaussian(params.get('delta_lambda'), params.get('lambda0')):.2f} micron")
    print(f"  Layers: {len(layer_props)} (z = {list(layer_boundaries)})")




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




    print("\n[4] Simulating spectral-domain OCT interferogram...")
    k_min = 2.0 * np.pi * 1.33 / (params.get('lambda0') + 0.5 * params.get('delta_lambda'))
    k_max = 2.0 * np.pi * 1.33 / (params.get('lambda0') - 0.5 * params.get('delta_lambda'))


    z_reflector = 100.0
    phi_coeffs = [0.0, 0.0, 2.5e-4]

    def integrand(k):
        return oct_interferogram_fd(
            k, z_reflector, params.get('k0'), params.get('delta_k'),
            phi_coeffs, reflectivity_sample=0.01, reflectivity_reference=0.1
        )

    I_total = integrate_spectral_interferogram(integrand, k_min, k_max, n_gl=64)
    print(f"  Spectral integral (interferogram power): {I_total:.6f}")


    def reflectivity_model(k, z):

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




    print("\n[6] Solving 2D FEM diffusion on tissue cross-section...")
    nodes_2d, elements_2d = generate_layered_tissue_mesh(
        layer_boundaries, radial_extent=50.0, n_r=8, n_z_per_layer=4
    )
    print(f"  Mesh nodes: {nodes_2d.shape[0]}, elements: {elements_2d.shape[0]}")
    print(f"  Mesh min angle: {mesh_quality_min_angle(nodes_2d, elements_2d):.2f} deg")

    def D_func_2d(x, y):



        raise NotImplementedError("Hole 3: D_func_2d in main.py needs to be implemented.")

    def mu_a_func_2d(x, y):
        z = y
        for i in range(len(layer_boundaries) - 1):
            if layer_boundaries[i] <= z <= layer_boundaries[i + 1]:
                return layer_props[i]['mu_a']
        return 0.0

    def source_func_2d(x, y):
        return 1.0 if 0.0 <= y <= layer_boundaries[-1] else 0.0


    top_nodes = np.where(np.abs(nodes_2d[:, 1] - layer_boundaries[0]) < 1e-6)[0]
    phi_2d = solve_fem_2d_diffusion(
        nodes_2d, elements_2d, D_func_2d, mu_a_func_2d, source_func_2d,
        dirichlet_nodes=top_nodes, dirichlet_values=np.ones(len(top_nodes))
    )
    phi_2d = clip_to_finite(phi_2d)
    print(f"  2D FEM solution range: [{np.min(phi_2d):.6f}, {np.max(phi_2d):.6f}]")




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


    if len(depths_mc) > 1:

        hist, bin_edges = np.histogram(depths_mc, bins=20)
        C = speckle_contrast(hist.astype(float))
        print(f"  Speckle contrast: {C:.4f}")




    print("\n[8] Simulating biological dynamics for functional OCT...")
    t_bio = np.linspace(0.0, 100.0, 500)


    fhn_result = simulate_functional_oct_signal(
        t_bio,
        bio_params={'type': 'FHN', 'y0': [0.1, 0.0], 'model_params': {'a': 0.7, 'b': 0.8, 'c': 12.5, 'd': 0.5}},
        oct_params={'lambda0': 0.84, 'path_length': 100.0, 'n0': 1.33, 'alpha_eo': 5e-4, 'alpha_thermo': 2e-3}
    )
    print(f"  FHN refractive index range: [{np.min(fhn_result['n_t']):.6f}, {np.max(fhn_result['n_t']):.6f}]")
    print(f"  FHN max phase shift: {np.max(np.abs(fhn_result['phase_shift'])):.6f} rad")


    gly_result = simulate_functional_oct_signal(
        t_bio,
        bio_params={'type': 'glycolysis', 'y0': [0.9, 0.7], 'model_params': {'a': 0.08, 'b': 0.6}},
        oct_params={'lambda0': 0.84, 'path_length': 100.0, 'n0': 1.33, 'alpha_eo': 5e-4, 'alpha_thermo': 2e-3}
    )
    print(f"  Glycolysis refractive index range: [{np.min(gly_result['n_t']):.6f}, {np.max(gly_result['n_t']):.6f}]")




    print("\n[9] Solving inverse problem: tissue property reconstruction...")
    z_scan_inv = np.linspace(0.0, 200.0, 50)


    inv_boundaries = layer_boundaries[:3]
    forward_known = build_forward_model_oct(z_scan_inv, inv_boundaries, k_min, k_max, n_gl=16)
    true_params = np.array([
        layer_props[0]['mu_a'], layer_props[0]['mu_s'], layer_props[0]['g'],
        layer_props[1]['mu_a'], layer_props[1]['mu_s'], layer_props[1]['g'],
    ])
    A_measured = forward_known(true_params)

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




    print("\n[10] Sensitivity analysis and parameter space exploration...")

    y0_sens = np.array([1.01, -1.0])
    t_sens = np.linspace(0.0, 5.0, 100)
    y_exact = sensitive_photon_exact(t_sens, y0_sens, growth_rate=0.5)
    print(f"  Sensitive ODE: initial density {y0_sens[0]:.4f}")
    print(f"  After 5 microns: density {y_exact[-1,0]:.4f}, flux {y_exact[-1,1]:.4f}")


    dmu, dvar = hypercube_surface_distance_stats(n=500, d=5)
    print(f"  Hypercube surface distance (d=5): mean={dmu:.4f}, var={dvar:.6f}")


    sphere_pts = sphere01_sample(100)

    norms = np.linalg.norm(sphere_pts, axis=0)
    print(f"  Sphere sample mean norm: {np.mean(norms):.6f} (should be 1.0)")

    int_x2 = sphere01_monomial_integral([2, 0, 0])
    print(f"  Sphere integral x^2: {int_x2:.6f} (exact: {4*np.pi/3:.6f})")




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


    for f in [gmsh_file, freefem_file, xy_file]:
        if os.path.exists(f):
            os.remove(f)




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
