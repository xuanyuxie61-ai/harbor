"""
main.py
=======
Unified entry point for the nanophotonics plasmonics synthesis project.

Zero-parameter execution performs a full multi-scale computation:

    1. Nanoparticle layout optimization (CVT)
    2. Single-particle Mie scattering spectrum
    3. Coupled-dipole collective response
    4. Spectral analysis via FFT
    5. Wavelet-based hotspot multiresolution analysis
    6. Connected-component hotspot detection
    7. 3D Gauss-Legendre EM energy integration
    8. Collective resonance identification (bisection)
    9. Hot-electron random-walk transport
   10. 1D effective waveguide mode solver
   11. Domain partitioning for parallel load balance
   12. Tensor-grid generation with CFL stability check

All parameters are self-contained; no external input required.
"""

import numpy as np
import sys

# Import all project modules
from mie_theory import (
    drude_permittivity,
    mie_cross_sections,
    generate_sphere_surface_grid
)
from dipole_coupling import (
    build_coupling_matrix,
    incident_plane_wave,
    solve_dipole_moments,
    build_coupling_graph,
    polarizability_clausius_mossotti
)
from spectral_analysis import power_spectral_density, spectral_response_dipole
from wavelet_field import haar_2d_transform, extract_multiresolution_hotspots
from hotspot_detector import (
    find_connected_components_2d,
    extract_hotspot_polygons_2d,
    hot_carrier_generation_rate
)
from volume_integrator import (
    gauss_legendre_3d_set,
    electromagnetic_energy_density_integral,
    absorbed_power_integral,
    test_exactness_monomial
)
from resonance_finder import (
    find_single_sphere_resonance,
    find_collective_resonance
)
from effective_wave import (
    solve_waveguide_modes,
    effective_permittivity_mim_waveguide
)
from hotcarrier_transport import (
    random_walk_1d,
    random_walk_3d,
    calculate_collection_efficiency
)
from nanoparticle_layout import place_nanoparticles_2d_cvt
from domain_partition import greedy_partition, estimate_workload
from tensor_grid import (
    uniform_tensor_grid_3d,
    cfl_time_step,
    grid_points_in_sphere
)


def run_full_simulation():
    print("=" * 70)
    print("  Nanophotonics Plasmonics Synthesis Project (PROJECT_106)")
    print("  Domain: Optical Engineering – Nanophotonics & Plasmonics")
    print("=" * 70)

    # ================================================================
    # Physical constants and material parameters
    # ================================================================
    c = 2.99792458e8          # speed of light (m/s)
    hbar = 1.054571817e-34    # reduced Planck constant (J·s)
    eV = 1.602176634e-19      # electron volt (J)
    eps0 = 8.854187817e-12    # vacuum permittivity (F/m)

    # Gold-like Drude parameters
    omega_p = 1.37e16         # rad/s (Au bulk plasma frequency)
    gamma = 4.05e13           # rad/s (Au electron collision rate)
    eps_inf = 9.84
    eps_medium = 1.0          # air/vacuum background

    # Nanoparticle geometry
    a_nm = 30.0               # sphere radius (nm)
    a = a_nm * 1e-9           # sphere radius (m)
    volume = (4.0 / 3.0) * np.pi * a ** 3

    print("\n[1] MATERIAL & GEOMETRY PARAMETERS")
    print(f"    Gold Drude: ω_p = {omega_p:.3e} rad/s, γ = {gamma:.3e} rad/s")
    print(f"    Nanosphere radius: {a_nm:.1f} nm")

    # ================================================================
    # 1. Nanoparticle layout via CVT optimization
    # ================================================================
    print("\n[2] NANOPARTICLE LAYOUT (CVT OPTIMIZATION)")
    region = (0.0, 200e-9, 0.0, 200e-9)
    positions = place_nanoparticles_2d_cvt(
        num_particles=12,
        region=region,
        it_num=15,
        s_num=2000,
        seed=42
    )
    # Add small z-offsets to make 3D positions
    Np = positions.shape[0]
    z_offsets = np.random.RandomState(42).rand(Np) * 20e-9
    positions_3d = np.column_stack([positions, z_offsets])
    print(f"    Number of nanoparticles placed: {Np}")
    print(f"    Simulation domain: {region[0]*1e9:.0f}–{region[1]*1e9:.0f} nm × "
          f"{region[2]*1e9:.0f}–{region[3]*1e9:.0f} nm")

    # ================================================================
    # 2. Single-particle Mie spectrum
    # ================================================================
    print("\n[3] MIE SCATTERING SPECTRUM (Single Nanosphere)")
    omega_min = 2.0e15
    omega_max = 8.0e15
    n_omega = 80
    omegas = np.linspace(omega_min, omega_max, n_omega)
    sigma_ext, sigma_sca = mie_cross_sections(
        omegas, a, eps_medium,
        omega_p=omega_p, gamma=gamma, eps_inf=eps_inf
    )
    idx_peak = np.argmax(sigma_ext)
    omega_peak_single = omegas[idx_peak]
    print(f"    Peak extinction: {sigma_ext[idx_peak]:.3e} m² at "
          f"ω = {omega_peak_single:.3e} rad/s")
    print(f"    Corresponding wavelength: {2*np.pi*c/omega_peak_single*1e9:.1f} nm")

    # ================================================================
    # 3. Coupled-dipole collective response
    # ================================================================
    print("\n[4] COUPLED DIPOLE MODEL (Collective Response)")
    omega_probe = omega_peak_single
    eps_metal = drude_permittivity(omega_probe, omega_p, gamma, eps_inf)
    alphas = np.full(Np, polarizability_clausius_mossotti(eps_metal, eps_medium, volume))

    A = build_coupling_matrix(positions_3d, alphas, omega_probe, eps_medium)
    E0 = 1.0e5  # V/m incident field
    kvec = np.array([0.0, 0.0, omega_probe * np.sqrt(eps_medium) / c])
    pol = np.array([1.0, 0.0, 0.0])
    b = incident_plane_wave(positions_3d, E0, kvec, pol)
    p = solve_dipole_moments(A, b)

    dipole_magnitudes = np.array([np.linalg.norm(p[3*i:3*i+3]) for i in range(Np)])
    print(f"    Interaction matrix size: {A.shape[0]}×{A.shape[1]}")
    print(f"    Mean induced dipole magnitude: {np.mean(dipole_magnitudes):.3e} C·m")
    print(f"    Max induced dipole magnitude:  {np.max(dipole_magnitudes):.3e} C·m")

    # Coupling graph
    adjacency, arc_list = build_coupling_graph(positions_3d, omega_probe, eps_medium,
                                                threshold=1.0e25)
    print(f"    Coupling graph arcs: {len(arc_list)}")

    # ================================================================
    # 4. Spectral analysis (FFT of synthetic time series)
    # ================================================================
    print("\n[5] SPECTRAL ANALYSIS (FFT)")
    dt = 1.0e-16  # s
    N_steps = 512
    t = np.arange(N_steps) * dt
    # Synthetic dipole moment: damped oscillation at plasmon frequency
    decay = np.exp(-gamma * t)
    pz_t = dipole_magnitudes[0] * decay * np.cos(omega_probe * t)
    p_t = np.column_stack([np.zeros(N_steps), np.zeros(N_steps), pz_t])
    freqs, intensity = spectral_response_dipole(p_t, dt)
    idx_max = np.argmax(intensity[1:]) + 1
    print(f"    FFT peak frequency: {freqs[idx_max]:.3e} rad/s")
    print(f"    Relative error vs input ω: {abs(freqs[idx_max]-omega_probe)/omega_probe:.3e}")

    # ================================================================
    # 5. Wavelet multiresolution analysis
    # ================================================================
    print("\n[6] WAVELET MULTIRESOLUTION ANALYSIS")
    # Create a synthetic 2D near-field map on a 128×128 grid
    nx_f, ny_f = 128, 128
    x_f = np.linspace(region[0], region[1], nx_f)
    y_f = np.linspace(region[2], region[3], ny_f)
    X_f, Y_f = np.meshgrid(x_f, y_f, indexing='ij')
    field_map = np.zeros((nx_f, ny_f))
    for i in range(Np):
        r2 = (X_f - positions_3d[i, 0]) ** 2 + (Y_f - positions_3d[i, 1]) ** 2
        field_map += np.exp(-r2 / (2 * (8e-9) ** 2))
    field_map *= E0

    coeffs = haar_2d_transform(field_map)
    scales, coeff_dict = extract_multiresolution_hotspots(field_map, threshold_factor=1.5)
    print(f"    Detected hotspot scales (pixels): {scales if scales else 'none above threshold'}")
    print(f"    Number of wavelet levels analyzed: {len(coeff_dict)}")

    # ================================================================
    # 6. Hotspot detection via connected components
    # ================================================================
    print("\n[7] HOTSPOT DETECTION (Connected Components)")
    intensity_map = np.abs(field_map) ** 2
    threshold = 1.5 * np.mean(intensity_map)
    labels, num_comp = find_connected_components_2d(intensity_map, threshold)
    print(f"    Threshold intensity: {threshold:.3e} (V/m)²")
    print(f"    Number of connected hotspots: {num_comp}")

    polygons, poly_intensities = extract_hotspot_polygons_2d(
        field_map, dx=(region[1]-region[0])/nx_f, dy=(region[3]-region[2])/ny_f,
        threshold_factor=1.5
    )
    print(f"    Hotspot polygons extracted: {len(polygons)}")
    if poly_intensities:
        print(f"    Mean hotspot intensity: {np.mean(poly_intensities):.3e} (V/m)²")

    # Hot-carrier generation rate in a representative hotspot
    G_hc = hot_carrier_generation_rate(
        np.sqrt(intensity_map), omega_probe, eps_metal,
        dx=(region[1]-region[0])/nx_f,
        dy=(region[3]-region[2])/ny_f,
        dz=10e-9
    )
    print(f"    Total hot-carrier generation rate: {np.sum(G_hc):.3e} s⁻¹")

    # ================================================================
    # 7. 3D Gauss-Legendre volume integration
    # ================================================================
    print("\n[8] 3D GAUSS-LEGENDRE VOLUME INTEGRATION")
    a_box = np.array([region[0], region[2], 0.0])
    b_box = np.array([region[1], region[3], 100e-9])

    def E2_func(x, y, z):
        # Synthetic |E|² decaying away from each nanoparticle center
        val = np.zeros_like(x)
        for i in range(Np):
            r2 = (x - positions_3d[i, 0]) ** 2 + (y - positions_3d[i, 1]) ** 2 + (z - positions_3d[i, 2]) ** 2
            val += np.exp(-r2 / (2 * (30e-9) ** 2))
        return val * (E0 ** 2)

    def H2_func(x, y, z):
        return np.zeros_like(x)

    U_em = electromagnetic_energy_density_integral(
        eps_medium, 4.0 * np.pi * 1e-7,
        E2_func, H2_func,
        a_box, b_box, nx=8, ny=8, nz=8
    )
    P_abs = absorbed_power_integral(
        omega_probe, eps_metal, E2_func,
        a_box, b_box, nx=8, ny=8, nz=8
    )
    print(f"    Total EM energy in box: {U_em:.3e} J")
    print(f"    Absorbed power:         {P_abs:.3e} W")

    # Exactness test
    errors = test_exactness_monomial(a_box, b_box, max_total_degree=3, nx=6, ny=6, nz=6)
    max_err = max(errors)
    print(f"    Quadrature exactness max relative error: {max_err:.3e}")

    # ================================================================
    # 8. Resonance identification
    # ================================================================
    print("\n[9] RESONANCE IDENTIFICATION")
    omega_res_single = find_single_sphere_resonance(
        eps_medium, omega_p=omega_p, gamma=gamma, eps_inf=eps_inf
    )
    print(f"    Single-sphere LSPR (Drude): {omega_res_single:.3e} rad/s")
    print(f"    Single-sphere wavelength:   {2*np.pi*c/omega_res_single*1e9:.1f} nm")

    # Collective resonance via scan
    def pol_func(omg):
        eps = drude_permittivity(omg, omega_p, gamma, eps_inf)
        alpha = polarizability_clausius_mossotti(eps, eps_medium, volume)
        return np.full(Np, alpha)

    omega_res_coll = find_collective_resonance(
        positions_3d, pol_func,
        omega_min=2.0e15, omega_max=7.0e15,
        eps_medium=eps_medium, num_points=60
    )
    print(f"    Collective resonance (scan): {omega_res_coll:.3e} rad/s")
    shift = (omega_res_coll - omega_res_single) / omega_res_single
    print(f"    Collective shift: {shift*100:.2f} %")

    # ================================================================
    # 9. Hot-electron transport (random walk)
    # ================================================================
    print("\n[10] HOT-ELECTRON TRANSPORT (Random Walk)")
    x2_ave, x2_max = random_walk_1d(step_num=1000, walk_num=500, step_length=1e-10)
    r2_ave = random_walk_3d(step_num=1000, walk_num=500, step_length=1e-10)
    D_eff = r2_ave[-1] / (6.0 * 1000 * 1e-14)  # D = <r²>/(6 t), t = N_step * τ_scatt
    print(f"    1D MSD after 1000 steps: {x2_ave[-1]:.3e} m²")
    print(f"    3D MSD after 1000 steps: {r2_ave[-1]:.3e} m²")
    print(f"    Effective diffusion coeff: {D_eff:.3e} m²/s")

    eff = calculate_collection_efficiency(
        particle_radius=a, mfp=10e-9, tau=10e-15,
        barrier_height=0.8, plasmon_energy=2.0,
        num_walkers=200
    )
    print(f"    Monte-Carlo collection efficiency: {eff*100:.2f} %")

    # ================================================================
    # 10. 1D Effective waveguide mode solver
    # ================================================================
    print("\n[11] 1D EFFECTIVE WAVEGUIDE MODE SOLVER")
    N_wave = 128
    h_wave = 1e-9
    eps_metal_wave = drude_permittivity(omega_probe, omega_p, gamma, eps_inf)
    eps_eff = effective_permittivity_mim_waveguide(
        eps_metal_wave, eps_medium,
        width_metal=10e-9, width_dielectric=5e-9,
        wavelength=2 * np.pi * c / omega_probe
    )
    epsilon_eff_profile = np.full(N_wave, np.real(eps_eff))
    # Add a dielectric gap in the center
    gap_start = N_wave // 2 - 5
    gap_end = N_wave // 2 + 5
    epsilon_eff_profile[gap_start:gap_end] = eps_medium

    k0 = omega_probe / c
    betas, modes = solve_waveguide_modes(
        epsilon_eff_profile, k0, h_wave, num_modes=3, boundary='PEC'
    )
    for m in range(len(betas)):
        print(f"    Mode {m}: β = {betas[m]:.3e} m⁻¹, "
              f"n_eff = {betas[m]/k0:.3f}")

    # ================================================================
    # 11. Domain partitioning
    # ================================================================
    print("\n[12] DOMAIN PARTITIONING (Load Balance)")
    interaction_radius = 80e-9
    weights = estimate_workload(positions_3d, interaction_radius)
    assignment, subset_sums, discrepancy = greedy_partition(weights, num_partitions=4)
    print(f"    4-way partition discrepancy: {discrepancy:.3e}")
    for p in range(4):
        count = np.sum(assignment == p)
        print(f"    Subset {p}: {count} particles, load = {subset_sums[p]:.3e}")

    # ================================================================
    # 12. Tensor grid generation & CFL check
    # ================================================================
    print("\n[13] TENSOR GRID & CFL STABILITY")
    Xg, Yg, Zg, dxg, dyg, dzg = uniform_tensor_grid_3d(
        (0.0, 200e-9), (0.0, 200e-9), (0.0, 100e-9),
        nx=41, ny=41, nz=21
    )
    dt_cfl = cfl_time_step(dxg, dyg, dzg, c=c, courant_factor=0.95)
    print(f"    Grid: {Xg.shape[0]}×{Xg.shape[1]}×{Xg.shape[2]} = {Xg.size} points")
    print(f"    Grid spacing: dx={dxg*1e9:.2f} nm, dy={dyg*1e9:.2f} nm, dz={dzg*1e9:.2f} nm")
    print(f"    CFL-limited Δt: {dt_cfl*1e15:.3f} fs")

    # Sphere mask for one nanoparticle
    mask = grid_points_in_sphere(Xg, Yg, Zg, positions_3d[0], a)
    print(f"    Grid points inside nanoparticle 0: {np.sum(mask)}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)
    print("  Key results:")
    print(f"    • Single-sphere LSPR:     {2*np.pi*c/omega_res_single*1e9:.1f} nm")
    print(f"    • Collective resonance:   {2*np.pi*c/omega_res_coll*1e9:.1f} nm")
    print(f"    • Hotspots detected:      {num_comp}")
    print(f"    • Collection efficiency:  {eff*100:.2f} %")
    print(f"    • CFL time step:          {dt_cfl*1e15:.2f} fs")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_full_simulation()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ================================================================
# 测试用例（50个，assert模式，涉及随机值均使用固定种子）
# ================================================================
from dipole_coupling import dyadic_green_tensor
from wavelet_field import haar_2d_inverse, haar_1d_transform
from volume_integrator import legendre_nodes_weights_1d, integrate_over_box
from resonance_finder import bisection_method
from hotcarrier_transport import hot_electron_escape_probability, effective_diffusion_coefficient
from nanoparticle_layout import cvtp_1d_optimize
from effective_wave import r83s_matvec, conjugate_gradient_r83s

# ---- TC01: drude_permittivity returns complex type ----
eps_drude = drude_permittivity(5.0e15)
assert isinstance(eps_drude, complex) or np.iscomplexobj(eps_drude), '[TC01] drude_permittivity complex type FAILED'

# ---- TC02: drude_permittivity has positive imaginary part (lossy medium absorption) ----
eps_drude2 = drude_permittivity(5.0e15, omega_p=1.37e16, gamma=4.05e13, eps_inf=9.84)
assert np.imag(eps_drude2) > 0, '[TC02] drude_permittivity positive imaginary part FAILED'

# ---- TC03: drude_permittivity array input yields same-length output ----
omegas_test = np.linspace(2.0e15, 8.0e15, 10)
eps_arr = drude_permittivity(omegas_test)
assert len(eps_arr) == len(omegas_test), '[TC03] drude_permittivity array output length FAILED'

# ---- TC04: drude_permittivity real part converges to eps_inf at high frequency ----
eps_high = drude_permittivity(1.0e20, omega_p=1.37e16, gamma=4.05e13, eps_inf=9.84)
assert abs(np.real(eps_high) - 9.84) < 1.0, '[TC04] drude_permittivity high-frequency limit FAILED'

# ---- TC05: mie_cross_sections scalar omega returns scalar sigma ----
sigma_ext, sigma_sca = mie_cross_sections(np.array([4.0e15]), 30e-9, eps_medium=1.0)
assert sigma_ext.shape == (1,), '[TC05] mie_cross_sections scalar output FAILED'
assert sigma_ext[0] >= sigma_sca[0] >= 0, '[TC05] mie_cross_sections ext>=sca>=0 FAILED'

# ---- TC06: polarizability_clausius_mossotti is complex ----
alpha_cm = polarizability_clausius_mossotti(complex(-20.0 + 0.5j, 0.5), 1.0, 1.13e-22)
assert np.iscomplexobj(alpha_cm) or isinstance(alpha_cm, complex), '[TC06] polarizability complex type FAILED'

# ---- TC07: dyadic_green_tensor zero distance returns zero matrix ----
G_zero = dyadic_green_tensor(np.array([0.0, 0.0, 0.0]), k=1.0e7)
assert np.allclose(G_zero, 0.0), '[TC07] dyadic_green_tensor zero distance FAILED'

# ---- TC08: dyadic_green_tensor is symmetric G = G^T ----
r_test = np.array([10e-9, 20e-9, 30e-9])
k_test = 1.0e7
G_sym = dyadic_green_tensor(r_test, k_test)
assert np.allclose(G_sym, G_sym.T), '[TC08] dyadic_green_tensor symmetry FAILED'

# ---- TC09: incident_plane_wave output shape matches positions ----
N_test = 5
pos_test = np.random.RandomState(123).rand(N_test, 3) * 100e-9
kvec_test = np.array([0.0, 0.0, 2.0e7])
pol_test = np.array([1.0, 0.0, 0.0])
b_test = incident_plane_wave(pos_test, 1.0e5, kvec_test, pol_test)
assert len(b_test) == 3 * N_test, '[TC09] incident_plane_wave output shape FAILED'

# ---- TC10: power_spectral_density non-negative ----
np.random.seed(42)
ts_test = np.sin(2.0 * np.pi * np.arange(128) / 128.0) + 0.1 * np.random.randn(128)
freqs_test, psd_test = power_spectral_density(ts_test, 1.0)
assert np.all(psd_test >= 0), '[TC10] PSD non-negative FAILED'

# ---- TC11: haar_2d_transform + haar_2d_inverse perfect reconstruction ----
np.random.seed(42)
field_orig = np.random.randn(64, 64)
field_trans = haar_2d_transform(field_orig)
field_recon = haar_2d_inverse(field_trans)
assert np.allclose(field_orig, field_recon, atol=1e-12), '[TC11] Haar 2D round-trip FAILED'

# ---- TC12: haar_1d_transform preserves norm approximately ----
np.random.seed(42)
u_test = np.random.randn(128)
v_test = haar_1d_transform(u_test)
assert abs(np.linalg.norm(v_test) - np.linalg.norm(u_test)) < 1e-10, '[TC12] Haar 1D norm preservation FAILED'

# ---- TC13: find_connected_components_2d basic detection ----
field_cc = np.zeros((8, 8))
field_cc[2:4, 2:4] = 10.0
field_cc[5:7, 5:7] = 10.0
labels_cc, num_cc = find_connected_components_2d(field_cc, 5.0)
assert num_cc == 2, '[TC13] connected_components_2d count FAILED'

# ---- TC14: legendre_nodes_weights_1d sum of weights = 2 ----
x_gl, w_gl = legendre_nodes_weights_1d(5)
assert abs(np.sum(w_gl) - 2.0) < 1e-12, '[TC14] legendre sum of weights FAILED'

# ---- TC15: gauss_legendre_3d_set total weight = box volume ----
a_box = np.array([0.0, 0.0, 0.0])
b_box = np.array([1.0, 2.0, 3.0])
_, _, _, w_gl3d = gauss_legendre_3d_set(a_box, b_box, 4, 4, 4)
expected_vol = 1.0 * 2.0 * 3.0
assert abs(np.sum(w_gl3d) - expected_vol) < 1e-12, '[TC15] GL 3D total weight = volume FAILED'

# ---- TC16: test_exactness_monomial small error for low-degree monomials ----
errors_test = test_exactness_monomial(a_box, b_box, max_total_degree=1, nx=6, ny=6, nz=6)
assert max(errors_test) < 1e-10, '[TC16] quadrature exactness FAILED'

# ---- TC17: bisection_method finds known root of f(x)=x ----
f_lin = lambda x: x
root_bis, it_bis, _, _ = bisection_method(f_lin, -1.0, 1.0)
assert abs(root_bis) < 1e-10, '[TC17] bisection root of f(x)=x FAILED'

# ---- TC18: find_single_sphere_resonance returns positive frequency ----
omega_res = find_single_sphere_resonance(1.0, omega_p=1.37e16, gamma=4.05e13, eps_inf=9.84)
assert omega_res > 0, '[TC18] single sphere resonance positive FAILED'

# ---- TC19: find_single_sphere_resonance approximate analytical limit ----
# Drude model resonance: omega_res ≈ omega_p / sqrt(eps_inf + 2*eps_medium)
omega_analytical = 1.37e16 / np.sqrt(9.84 + 2.0)
assert abs(omega_res - omega_analytical) / omega_analytical < 0.1, '[TC19] resonance analytical limit FAILED'

# ---- TC20: effective_diffusion_coefficient positive ----
D_eff_test = effective_diffusion_coefficient(1e-9, 1e-14)
assert D_eff_test > 0, '[TC20] diffusion coefficient positive FAILED'

# ---- TC21: effective_diffusion_coefficient known value ----
D_known = effective_diffusion_coefficient(1e-10, 1e-15)
D_expected = (1e-10)**2 / (3.0 * 1e-15)
assert abs(D_known - D_expected) < 1e-20, '[TC21] diffusion coefficient exact value FAILED'

# ---- TC22: hot_electron_escape_probability above barrier returns 1 ----
prob_escape = hot_electron_escape_probability(energy=2.0, theta=0.0, barrier_height=0.5)
assert prob_escape == 1.0, '[TC22] escape probability above barrier FAILED'

# ---- TC23: hot_electron_escape_probability below barrier returns 0 ----
prob_stay = hot_electron_escape_probability(energy=2.0, theta=np.pi/2, barrier_height=0.5)
assert prob_stay == 0.0, '[TC23] escape probability below barrier FAILED'

# ---- TC24: random_walk_1d MSD approximates analytical value (fixed seed) ----
np.random.seed(42)
step_num_1d = 100
x2_ave, x2_max = random_walk_1d(step_num=step_num_1d, walk_num=1000, step_length=1.0)
expected_msd_1d = step_num_1d * 1.0  # <x²(N)> = N * L² for 1D
assert abs(x2_ave[-1] - expected_msd_1d) / expected_msd_1d < 0.15, '[TC24] 1D random walk MSD analytical FAILED'

# ---- TC25: random_walk_1d reproducibility with fixed seed ----
np.random.seed(42)
x2_a, _ = random_walk_1d(step_num=50, walk_num=100, step_length=1.0)
np.random.seed(42)
x2_b, _ = random_walk_1d(step_num=50, walk_num=100, step_length=1.0)
assert np.array_equal(x2_a, x2_b), '[TC25] 1D random walk reproducibility FAILED'

# ---- TC26: random_walk_3d MSD approximates analytical value (fixed seed) ----
np.random.seed(42)
step_num_3d = 100
r2_ave = random_walk_3d(step_num=step_num_3d, walk_num=1000, step_length=1.0)
expected_msd_3d = step_num_3d * 1.0  # <r²(N)> = N * L² (one axis per step)
assert abs(r2_ave[-1] - expected_msd_3d) / expected_msd_3d < 0.15, '[TC26] 3D random walk MSD analytical FAILED'

# ---- TC27: random_walk_3d reproducibility with fixed seed ----
np.random.seed(42)
r2_a = random_walk_3d(step_num=50, walk_num=100, step_length=1.0)
np.random.seed(42)
r2_b = random_walk_3d(step_num=50, walk_num=100, step_length=1.0)
assert np.array_equal(r2_a, r2_b), '[TC27] 3D random walk reproducibility FAILED'

# ---- TC28: cvtp_1d_optimize generators in [0,1) range ----
np.random.seed(42)
gens, _, _ = cvtp_1d_optimize(g_num=8, it_num=5, s_num=500, seed=42)
assert np.all(gens >= 0) and np.all(gens < 1), '[TC28] CVT generators in [0,1) FAILED'

# ---- TC29: cvtp_1d_optimize returns correct output shapes ----
np.random.seed(42)
gens2, energies2, motions2 = cvtp_1d_optimize(g_num=6, it_num=3, s_num=200, seed=42)
assert len(gens2) == 6, '[TC29] CVT generators count FAILED'
assert len(energies2) == 3 and len(motions2) == 3, '[TC29] CVT energy/motion length FAILED'

# ---- TC30: greedy_partition basic balance ----
weights_test = np.array([5.0, 3.0, 2.0, 8.0, 1.0, 4.0])
assignment, subset_sums, disc = greedy_partition(weights_test, 3)
assert len(assignment) == len(weights_test), '[TC30] greedy partition assignment length FAILED'
assert disc >= 0, '[TC30] greedy partition discrepancy non-negative FAILED'

# ---- TC31: uniform_tensor_grid_3d output shapes ----
Xg, Yg, Zg, dxg, dyg, dzg = uniform_tensor_grid_3d((0, 2), (0, 3), (0, 4), 11, 13, 15)
assert Xg.shape == (11, 13, 15), '[TC31] tensor grid X shape FAILED'
assert Yg.shape == (11, 13, 15), '[TC31] tensor grid Y shape FAILED'
assert Zg.shape == (11, 13, 15), '[TC31] tensor grid Z shape FAILED'

# ---- TC32: cfl_time_step positive ----
dt_cfl_test = cfl_time_step(5e-9, 5e-9, 5e-9)
assert dt_cfl_test > 0, '[TC32] CFL time step positive FAILED'

# ---- TC33: grid_points_in_sphere center is inside ----
np.random.seed(42)
Xgg, Ygg, Zgg, _, _, _ = uniform_tensor_grid_3d((0, 2), (0, 2), (0, 2), 5, 5, 5)
center_test = np.array([1.0, 1.0, 1.0])
mask_sphere = grid_points_in_sphere(Xgg, Ygg, Zgg, center_test, 0.5)
assert np.any(mask_sphere), '[TC33] sphere mask non-empty FAILED'

# ---- TC34: solve_waveguide_modes betas shape correct ----
N_w_test = 64
eps_prof_test = np.ones(N_w_test) * 2.0
k0_test = 1.0e7
h_test = 1e-9
betas_test, modes_test = solve_waveguide_modes(eps_prof_test, k0_test, h_test, num_modes=3, boundary='PEC')
assert len(betas_test) == 3, '[TC34] waveguide modes betas count FAILED'
assert modes_test.shape == (N_w_test, 3), '[TC34] waveguide modes shape FAILED'

# ---- TC35: effective_permittivity_mim_waveguide returns real positive ----
c_test = 2.99792458e8
omega_test_wg = 4.0e15
eps_metal_test = drude_permittivity(omega_test_wg, 1.37e16, 4.05e13, 9.84)
eps_eff_mim = effective_permittivity_mim_waveguide(
    eps_metal_test, 1.0, width_metal=10e-9, width_dielectric=5e-9,
    wavelength=2*np.pi*c_test/omega_test_wg
)
assert np.isreal(eps_eff_mim) and np.real(eps_eff_mim) > 0, '[TC35] effective permittivity real positive FAILED'

# ---- TC36: hot_carrier_generation_rate non-negative ----
field_hc = np.ones((8, 8)) * 1.0e5
G_hc_test = hot_carrier_generation_rate(field_hc, 4.0e15, eps_metal_test, dx=10e-9, dy=10e-9)
assert np.all(G_hc_test >= 0), '[TC36] hot carrier generation non-negative FAILED'

# ---- TC37: extract_multiresolution_hotspots returns expected types ----
np.random.seed(42)
field_mr = np.abs(np.random.randn(32, 32))
scales_mr, coeffs_mr = extract_multiresolution_hotspots(field_mr, threshold_factor=1.5)
assert isinstance(scales_mr, list), '[TC37] multiresolution scales list FAILED'
assert isinstance(coeffs_mr, dict), '[TC37] multiresolution coefficients dict FAILED'

# ---- TC38: mie_cross_sections array omega returns array output ----
omegas_arr = np.linspace(3.0e15, 6.0e15, 5)
sig_ext_arr, sig_sca_arr = mie_cross_sections(omegas_arr, 30e-9)
assert sig_ext_arr.shape == omegas_arr.shape, '[TC38] mie array output shape FAILED'

# ---- TC39: spectral_response_dipole non-negative intensity ----
np.random.seed(42)
dt_spec = 1e-16
N_spec = 128
t_spec = np.arange(N_spec) * dt_spec
p_spec = np.column_stack([np.sin(4e15 * t_spec), np.zeros(N_spec), np.zeros(N_spec)])
freqs_spec, intens_spec = spectral_response_dipole(p_spec, dt_spec)
assert np.all(intens_spec >= 0), '[TC39] spectral response non-negative FAILED'

# ---- TC40: generate_sphere_surface_grid output shapes ----
x_s, y_s, z_s, theta_s, phi_s, area_s = generate_sphere_surface_grid(n_theta=16, n_phi=32, radius=30e-9)
assert x_s.shape == (16, 32), '[TC40] sphere grid x shape FAILED'
assert area_s.shape == (16, 32), '[TC40] sphere grid area shape FAILED'

# ---- TC41: find_connected_components_2d returns integer labels ----
labels_test = find_connected_components_2d(np.abs(np.random.randn(6, 6)), 0.0)[0]
assert labels_test.dtype == int, '[TC41] connected components labels int FAILED'

# ---- TC42: calculate_collection_efficiency in [0,1] with fixed seed ----
np.random.seed(42)
eff_test = calculate_collection_efficiency(
    particle_radius=30e-9, mfp=10e-9, tau=10e-15,
    barrier_height=0.8, plasmon_energy=2.0, num_walkers=100
)
assert 0.0 <= eff_test <= 1.0, '[TC42] collection efficiency in [0,1] FAILED'

# ---- TC43: Legendre nodes in [-1,1] ----
x_nodes, _ = legendre_nodes_weights_1d(10)
assert np.all(x_nodes >= -1.0) and np.all(x_nodes <= 1.0), '[TC43] Legendre nodes in [-1,1] FAILED'

# ---- TC44: expand_bracket finds sign change ----
from resonance_finder import expand_bracket
def f_bracket(x):
    return x - 2.0
a_exp, b_exp = expand_bracket(f_bracket, 0.0, 1.0)
assert f_bracket(a_exp) * f_bracket(b_exp) <= 0, '[TC44] expand bracket sign change FAILED'

# ---- TC45: r83s_matvec correct shape ----
n_r83 = 10
a_r83 = np.array([0.5, 2.0, 0.5])
x_r83 = np.ones(n_r83)
b_r83 = r83s_matvec(n_r83, a_r83, x_r83)
assert len(b_r83) == n_r83, '[TC45] r83s_matvec output shape FAILED'

# ---- TC46: conjugate_gradient_r83s solves known system ----
n_cg = 5
a_cg = np.array([1.0, 2.0, 1.0])
x_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b_cg = r83s_matvec(n_cg, a_cg, x_true)
x0_cg = np.zeros(n_cg)
x_sol = conjugate_gradient_r83s(n_cg, a_cg, b_cg, x0_cg)
assert np.allclose(x_sol, x_true, atol=1e-10), '[TC46] conjugate gradient solves system FAILED'

# ---- TC47: build_coupling_matrix square shape ----
np.random.seed(42)
N_coup = 4
pos_coup = np.random.rand(N_coup, 3) * 50e-9
alpha_coup = np.full(N_coup, complex(1e-30, 0.0))
A_coup = build_coupling_matrix(pos_coup, alpha_coup, 4.0e15, eps_medium=1.0)
assert A_coup.shape == (3*N_coup, 3*N_coup), '[TC47] coupling matrix shape FAILED'

# ---- TC48: build_coupling_graph returns arc list ----
adj, arcs = build_coupling_graph(pos_coup, 4.0e15, eps_medium=1.0, threshold=1.0e20)
assert isinstance(arcs, list), '[TC48] coupling graph arcs list FAILED'

# ---- TC49: spectral_partition_laplacian returns valid assignment ----
from domain_partition import spectral_partition_laplacian
adj_mat = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
assign_spec = spectral_partition_laplacian(adj_mat, 2)
assert len(assign_spec) == 3, '[TC49] spectral partition length FAILED'
assert np.min(assign_spec) >= 0 and np.max(assign_spec) < 2, '[TC49] spectral partition range FAILED'

# ---- TC50: integrate_over_box constant function ----
a_int = np.array([0.0, 0.0, 0.0])
b_int = np.array([1.0, 1.0, 1.0])
f_const = lambda x, y, z: np.ones_like(x)
result_const = integrate_over_box(f_const, a_int, b_int, nx=4, ny=4, nz=4)
assert abs(result_const - 1.0) < 1e-10, '[TC50] integrate constant FAILED'

print('\n全部 50 个测试通过!\n')
