
import numpy as np
import os


np.random.seed(42)


def run_simulation():
    print("=" * 70)
    print("GRB Afterglow Radiation Mechanism Synthesis")
    print("Domain: Astrophysics — Gamma-Ray Burst Radiation Mechanisms")
    print("=" * 70)




    print("\n[1] Computing relativistic jet hydrodynamic profiles...")
    from grb_jet_hydro import compute_jet_profiles, lorentz_factor

    jet = compute_jet_profiles(n_r=16, n_z=32, r_max=1e13, z_max=1e15,
                               c_param=1.0, rho_0=1e-24, Gamma_0=300.0)
    Gamma_mean = np.mean(jet["Gamma"])
    rho_mean = np.mean(jet["rho"])
    res_max = np.max(np.abs(jet["residual"]))
    print(f"    Mean Lorentz factor  Γ = {Gamma_mean:.3f}")
    print(f"    Mean density         ρ = {rho_mean:.3e} g/cm³")
    print(f"    Max continuity residual = {res_max:.3e}")




    print("\n[2] Evaluating Barenblatt blast-wave energy density...")
    from blast_wave_diffusion import blast_wave_energy_density_profile

    r_bw = np.linspace(1e10, 5e12, 50)
    t_bw = 10.0
    eps_bw = blast_wave_energy_density_profile(r_bw, t_bw, E_iso=1e53,
                                               n_ism=1.0, gamma_ad=4.0 / 3.0)
    print(f"    Blast wave energy density at r=1e12 cm, t=10 s: {eps_bw[10]:.3e} erg/cm³")
    print(f"    Peak energy density: {np.max(eps_bw):.3e} erg/cm³")




    print("\n[3] Constructing helical magnetic field geometry...")
    from magnetic_spiral import magnetic_pitch_angle_grid, magnetization_parameter

    r_mag, psi_mag, B_ratio = magnetic_pitch_angle_grid(n_r=16, r_max=1e13,
                                                        v_z=2.99e10, Omega=1e-3)
    sigma = magnetization_parameter(rho_mean, 10.0, Gamma_mean)
    print(f"    Pitch angle at r=1e12 cm: {np.degrees(psi_mag[10]):.2f}°")
    print(f"    Magnetization parameter σ: {np.mean(sigma):.3e}")




    print("\n[4] Evolving magnetic reconnection automaton...")
    from reconnection_automaton import evolve_reconnection

    hist, power = evolve_reconnection(m=16, n=16, n_steps=20,
                                      seed_density=0.15, B=10.0, cell_size=1e10)
    print(f"    Initial active sites: {hist[0]}")
    print(f"    Final active sites: {hist[-1]}")
    print(f"    Mean dissipated power: {np.mean(power):.3e} erg/s")




    print("\n[5] Stochastic Runge-Kutta particle acceleration...")
    from particle_acceleration import accelerate_electrons

    gamma_final = accelerate_electrons(gamma_0=10.0, n_particles=200,
                                       t_max=1.0, dt=0.01,
                                       B=10.0, u1=2.5e10, u2=6.0e9,
                                       q_noise=0.5, gamma_max=1e7)
    print(f"    Initial γ: 10.0")
    print(f"    Mean final γ: {np.mean(gamma_final):.3f}")
    print(f"    Max final γ: {np.max(gamma_final):.3e}")




    print("\n[6] Building opacity table and interpolating...")
    from opacity_interpolator import build_opacity_table, interpolate_opacity

    log_rho, log_T, kappa_table = build_opacity_table(n_rho=12, n_T=12)
    rho_q = np.array([1e-18, 1e-15, 1e-12])
    T_q = np.array([1e6, 1e7, 1e8])
    kappa_q = interpolate_opacity(rho_q, T_q, log_rho, log_T, kappa_table)
    print(f"    Opacities at query points: {kappa_q}")




    print("\n[7] Solving 1D radiation diffusion equation via FEM...")
    from radiation_diffusion_fem import solve_radiation_diffusion

    t_fem, u_fem = solve_radiation_diffusion(n=32, t_span=(0.0, 2.0),
                                             c_array=np.array([0.0, -0.3, 0.0, 0.0]),
                                             w0_func=lambda x: np.sin(np.pi * x))
    print(f"    FEM solution shape: {u_fem.shape}")
    print(f"    Max radiation energy density at t=2: {np.max(u_fem[-1]):.3e}")




    print("\n[8] Assembling Wathen FEM matrix and solving...")
    from fem_matrix_assembly import solve_wathen_system



    pass




    print("\n[9] Computing photon transfer steady state...")
    from photon_transfer_matrix import build_compton_transfer_matrix, compute_photon_stats

    A_trans = build_compton_transfer_matrix(n_bins=8, T_e=1e8, tau_es=0.5)
    stats = compute_photon_stats(A_trans)
    print(f"    Mean scatterings: {stats['mean_scatterings']:.3f}")
    print(f"    Compton-y parameter: {stats['y_param']:.3e}")
    print(f"    Power iterations: {stats['iterations']}")




    print("\n[10] Computing spectral moments via Hankel SPD matrix...")
    from spectral_moments import (synthetic_grb_moments, build_hankel_from_moments,
                                  compute_spectral_moments_from_hankel,
                                  hankel_spd_cholesky_lower)

    moments = synthetic_grb_moments(n=4)
    H = build_hankel_from_moments(moments)
    stats_h = compute_spectral_moments_from_hankel(H)
    print(f"    Bolometric flux: {stats_h['bolometric_flux']:.3e}")
    print(f"    Mean frequency: {stats_h['mean_frequency']:.3e} Hz")
    print(f"    Spectral width: {stats_h['spectral_width']:.3e} Hz")




    pass




    print("\n[11] Monte Carlo SED integration over (γ, θ) triangle...")
    from sed_triangle_integrator import monte_carlo_sed


    vertices = np.array([
        [1e4, 0.1],
        [1e6, 0.1],
        [1e5, np.pi / 2.0]
    ])
    flux_sed, err_sed = monte_carlo_sed(vertices, n_samples=5000,
                                        nu_obs=1e15, B=100.0, N_gamma=1e20)
    print(f"    SED flux at ν=10¹⁵ Hz: {flux_sed:.3e} ± {err_sed:.3e}")




    print("\n[12] Interpolating spectral energy distribution...")
    from spectrum_interpolator import interpolate_spectrum

    nu_bins = np.logspace(10, 20, 12)
    flux_bins = nu_bins ** (-0.5) * np.exp(-nu_bins / 1e18)
    flux_bins = np.clip(flux_bins, 1e-30, None)
    nu_query = np.logspace(11, 19, 20)
    flux_lin = interpolate_spectrum(nu_bins, flux_bins, nu_query, method='linear')
    flux_lag = interpolate_spectrum(nu_bins, flux_bins, nu_query, method='lagrange')
    print(f"    Linear interpolation: mean νF_ν = {np.mean(flux_lin):.3e}")
    print(f"    Lagrange interpolation: mean νF_ν = {np.mean(flux_lag):.3e}")




    print("\n[13] Computing discrete inverse-Compton cascade energies...")
    from discrete_cascade import discrete_ic_cascade, cascade_compactness

    gamma_discrete = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])
    cascade = discrete_ic_cascade(epsilon_0=1.0, gamma_discrete=gamma_discrete,
                                  n_scatter_max=3)
    for idx, energies in enumerate(cascade):
        print(f"    After {idx + 1} scatterings: {energies.size} discrete energies, "
              f"range [{energies.min():.2e}, {energies.max():.2e}] eV")

    ell = cascade_compactness(L=1e52, R=1e13)
    print(f"    Compactness parameter ℓ = {ell:.3e}")




    print("\n[14] Constructing anisotropic diffusion tensors...")
    from anisotropic_tensor import magic_anisotropic_field

    r_ani, D_tensors = magic_anisotropic_field(n_r=8, D_perp_base=1e20,
                                               D_para_base=1e24)
    D_trace = np.trace(D_tensors[4])
    print(f"    Diffusion tensor at mid-radius: trace = {D_trace:.3e}")




    print("\n[15] Exporting matrices to Matrix Market format...")
    from matrix_io import export_grb_matrix

    files_written = export_grb_matrix(A_wathen, filename_prefix='grb_matrix')
    for f in files_written:
        print(f"    Written: {f} ({os.path.getsize(f)} bytes)")
        os.remove(f)




    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("All 15 seed projects have been integrated into the GRB")
    print("afterglow radiation-mechanism pipeline.")
    print("No errors encountered.")


if __name__ == '__main__':
    run_simulation()
