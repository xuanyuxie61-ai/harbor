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
    # TODO: Set the probe frequency for the coupled-dipole calculation.
    # Then compute the metal permittivity at this frequency using the Drude model,
    # and compute the polarizability for each nanoparticle using Clausius-Mossotti.
    # The polarizabilities must be passed as a 1D array of length Np.
    raise NotImplementedError("Hole 3: Probe-frequency selection and polarizability array construction are missing.")

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
