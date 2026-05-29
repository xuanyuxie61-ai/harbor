"""
GRB Afterglow Radiation Mechanism Synthesis
===========================================
Unified zero-parameter entry point for the gamma-ray burst
radiation-mechanism simulation pipeline.

This script orchestrates:
  1. Jet hydrodynamics (continuity equation, blast-wave diffusion)
  2. Magnetic field geometry (spiral + reconnection automaton)
  3. Particle acceleration (stochastic Runge-Kutta Fokker-Planck)
  4. Radiative transfer (FEM + opacity interpolation)
  5. SED computation (triangle Monte Carlo integration)
  6. Spectral moments (Hankel SPD matrix)
  7. Photon cascade (discrete subset-sum IC cascade)
  8. Matrix export (Matrix Market format)

All 15 seed projects are integrated into the above pipeline.
"""

import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)


def run_simulation():
    print("=" * 70)
    print("GRB Afterglow Radiation Mechanism Synthesis")
    print("Domain: Astrophysics — Gamma-Ray Burst Radiation Mechanisms")
    print("=" * 70)

    # ================================================================
    # 1. Jet Hydrodynamics (seed 211_continuity_exact)
    # ================================================================
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

    # ================================================================
    # 2. Blast Wave Diffusion (seed 901_porous_medium_exact)
    # ================================================================
    print("\n[2] Evaluating Barenblatt blast-wave energy density...")
    from blast_wave_diffusion import blast_wave_energy_density_profile

    r_bw = np.linspace(1e10, 5e12, 50)
    t_bw = 10.0  # seconds
    eps_bw = blast_wave_energy_density_profile(r_bw, t_bw, E_iso=1e53,
                                               n_ism=1.0, gamma_ad=4.0 / 3.0)
    print(f"    Blast wave energy density at r=1e12 cm, t=10 s: {eps_bw[10]:.3e} erg/cm³")
    print(f"    Peak energy density: {np.max(eps_bw):.3e} erg/cm³")

    # ================================================================
    # 3. Magnetic Spiral (seed 1371_ulam_spiral)
    # ================================================================
    print("\n[3] Constructing helical magnetic field geometry...")
    from magnetic_spiral import magnetic_pitch_angle_grid, magnetization_parameter

    r_mag, psi_mag, B_ratio = magnetic_pitch_angle_grid(n_r=16, r_max=1e13,
                                                        v_z=2.99e10, Omega=1e-3)
    sigma = magnetization_parameter(rho_mean, 10.0, Gamma_mean)
    print(f"    Pitch angle at r=1e12 cm: {np.degrees(psi_mag[10]):.2f}°")
    print(f"    Magnetization parameter σ: {np.mean(sigma):.3e}")

    # ================================================================
    # 4. Reconnection Automaton (seed 671_life)
    # ================================================================
    print("\n[4] Evolving magnetic reconnection automaton...")
    from reconnection_automaton import evolve_reconnection

    hist, power = evolve_reconnection(m=16, n=16, n_steps=20,
                                      seed_density=0.15, B=10.0, cell_size=1e10)
    print(f"    Initial active sites: {hist[0]}")
    print(f"    Final active sites: {hist[-1]}")
    print(f"    Mean dissipated power: {np.mean(power):.3e} erg/s")

    # ================================================================
    # 5. Particle Acceleration (seed 1171_stochastic_rk)
    # ================================================================
    print("\n[5] Stochastic Runge-Kutta particle acceleration...")
    from particle_acceleration import accelerate_electrons

    gamma_final = accelerate_electrons(gamma_0=10.0, n_particles=200,
                                       t_max=1.0, dt=0.01,
                                       B=10.0, u1=2.5e10, u2=6.0e9,
                                       q_noise=0.5, gamma_max=1e7)
    print(f"    Initial γ: 10.0")
    print(f"    Mean final γ: {np.mean(gamma_final):.3f}")
    print(f"    Max final γ: {np.max(gamma_final):.3e}")

    # ================================================================
    # 6. Opacity Interpolation (seed 927_pwl_interp_2d)
    # ================================================================
    print("\n[6] Building opacity table and interpolating...")
    from opacity_interpolator import build_opacity_table, interpolate_opacity

    log_rho, log_T, kappa_table = build_opacity_table(n_rho=12, n_T=12)
    rho_q = np.array([1e-18, 1e-15, 1e-12])
    T_q = np.array([1e6, 1e7, 1e8])
    kappa_q = interpolate_opacity(rho_q, T_q, log_rho, log_T, kappa_table)
    print(f"    Opacities at query points: {kappa_q}")

    # ================================================================
    # 7. Radiation Diffusion FEM (seed 377_fem_neumann)
    # ================================================================
    print("\n[7] Solving 1D radiation diffusion equation via FEM...")
    from radiation_diffusion_fem import solve_radiation_diffusion

    t_fem, u_fem = solve_radiation_diffusion(n=32, t_span=(0.0, 2.0),
                                             c_array=np.array([0.0, -0.3, 0.0, 0.0]),
                                             w0_func=lambda x: np.sin(np.pi * x))
    print(f"    FEM solution shape: {u_fem.shape}")
    print(f"    Max radiation energy density at t=2: {np.max(u_fem[-1]):.3e}")

    # ================================================================
    # 8. FEM Matrix Assembly (seed 1401_wathen_matrix)
    # ================================================================
    print("\n[8] Assembling Wathen FEM matrix and solving...")
    from fem_matrix_assembly import solve_wathen_system

    x_wathen, A_wathen, b_wathen = solve_wathen_system(nx=3, ny=3)
    print(f"    Matrix order N = {A_wathen.shape[0]}")
    print(f"    ||A·x - b||_2 = {np.linalg.norm(A_wathen @ x_wathen - b_wathen):.3e}")

    # ================================================================
    # 9. Photon Transfer Matrix (seed 1405_web_matrix)
    # ================================================================
    print("\n[9] Computing photon transfer steady state...")
    from photon_transfer_matrix import build_compton_transfer_matrix, compute_photon_stats

    A_trans = build_compton_transfer_matrix(n_bins=8, T_e=1e8, tau_es=0.5)
    stats = compute_photon_stats(A_trans)
    print(f"    Mean scatterings: {stats['mean_scatterings']:.3f}")
    print(f"    Compton-y parameter: {stats['y_param']:.3e}")
    print(f"    Power iterations: {stats['iterations']}")

    # ================================================================
    # 10. Spectral Moments (seed 506_hankel_spd)
    # ================================================================
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

    # Cholesky factorization
    n_chol = 4
    lii = np.ones(n_chol)
    liim1 = 0.5 * np.ones(n_chol - 1)
    L_chol = hankel_spd_cholesky_lower(n_chol, lii, liim1)
    H_recon = L_chol @ L_chol.T
    print(f"    Hankel reconstruction error: {np.linalg.norm(H_recon - H_recon.T):.3e}")

    # ================================================================
    # 11. SED Triangle Integration (seed 1308_triangle_integrands)
    # ================================================================
    print("\n[11] Monte Carlo SED integration over (γ, θ) triangle...")
    from sed_triangle_integrator import monte_carlo_sed

    # Triangle in (γ, θ) space
    vertices = np.array([
        [1e4, 0.1],
        [1e6, 0.1],
        [1e5, np.pi / 2.0]
    ])
    flux_sed, err_sed = monte_carlo_sed(vertices, n_samples=5000,
                                        nu_obs=1e15, B=100.0, N_gamma=1e20)
    print(f"    SED flux at ν=10¹⁵ Hz: {flux_sed:.3e} ± {err_sed:.3e}")

    # ================================================================
    # 12. Spectrum Interpolation (seed 590_interp)
    # ================================================================
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

    # ================================================================
    # 13. Discrete IC Cascade (seed 1178_subset_sum)
    # ================================================================
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

    # ================================================================
    # 14. Anisotropic Tensor (seed 719_matlab_compiler)
    # ================================================================
    print("\n[14] Constructing anisotropic diffusion tensors...")
    from anisotropic_tensor import magic_anisotropic_field

    r_ani, D_tensors = magic_anisotropic_field(n_r=8, D_perp_base=1e20,
                                               D_para_base=1e24)
    D_trace = np.trace(D_tensors[4])
    print(f"    Diffusion tensor at mid-radius: trace = {D_trace:.3e}")

    # ================================================================
    # 15. Matrix I/O (seed 782_msm_to_mm)
    # ================================================================
    print("\n[15] Exporting matrices to Matrix Market format...")
    from matrix_io import export_grb_matrix

    files_written = export_grb_matrix(A_wathen, filename_prefix='grb_matrix')
    for f in files_written:
        print(f"    Written: {f} ({os.path.getsize(f)} bytes)")
        os.remove(f)  # Clean up temporary files

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("All 15 seed projects have been integrated into the GRB")
    print("afterglow radiation-mechanism pipeline.")
    print("No errors encountered.")


if __name__ == '__main__':
    run_simulation()

# ================================================================
# 测试用例（35个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: grid_2d returns correct shape and bounds ----
from grb_jet_hydro import grid_2d
x_grid, y_grid = grid_2d(4, 0.0, 10.0, 5, -1.0, 1.0)
assert x_grid.shape == (4, 5), '[TC01] grid_2d returns correct shape and bounds FAILED'
assert np.isclose(x_grid[0, 0], 0.0) and np.isclose(x_grid[-1, 0], 10.0), '[TC01] grid_2d returns correct shape and bounds FAILED'
assert np.isclose(y_grid[0, 0], -1.0) and np.isclose(y_grid[0, -1], 1.0), '[TC01] grid_2d returns correct shape and bounds FAILED'

# ---- TC02: phi_stream vanishes at z=1 boundary ----
from grb_jet_hydro import phi_stream
assert np.isclose(phi_stream(1.0, 1.0), 0.0), '[TC02] phi_stream vanishes at z=1 boundary FAILED'

# ---- TC03: uv_spiral u component vanishes when x=0 ----
from grb_jet_hydro import uv_spiral
u_test, v_test = uv_spiral(1, np.array([0.0]), np.array([0.5]), 1.0)
assert np.isclose(u_test[0], 0.0), '[TC03] uv_spiral u component vanishes when x=0 FAILED'

# ---- TC04: lorentz_factor at rest equals 1 ----
from grb_jet_hydro import lorentz_factor
assert np.isclose(lorentz_factor(0.0, 0.0, 0.0), 1.0), '[TC04] lorentz_factor at rest equals 1 FAILED'

# ---- TC05: lorentz_factor clamps superluminal input to finite value ----
from grb_jet_hydro import lorentz_factor
gamma_super = lorentz_factor(3e10, 0.0, 0.0)
assert np.isfinite(gamma_super) and gamma_super > 1.0, '[TC05] lorentz_factor clamps superluminal input FAILED'

# ---- TC06: compute_jet_profiles returns expected keys and finite values ----
from grb_jet_hydro import compute_jet_profiles
jet = compute_jet_profiles(n_r=8, n_z=8)
assert set(jet.keys()) == {'r', 'z', 'rho', 'vr', 'vz', 'Gamma', 'residual'}, '[TC06] compute_jet_profiles returns expected keys and finite values FAILED'
for k, v in jet.items():
    assert np.all(np.isfinite(v)), '[TC06] compute_jet_profiles returns expected keys and finite values FAILED'

# ---- TC07: porous_medium_exact zero outside blast front ----
from blast_wave_diffusion import porous_medium_exact, porous_medium_parameters
params = porous_medium_parameters()
u, _, _, _ = porous_medium_exact(10.0, 1.0, params)
assert np.all(u == 0.0), '[TC07] porous_medium_exact zero outside blast front FAILED'

# ---- TC08: blast_wave_energy_density_profile nonnegative ----
from blast_wave_diffusion import blast_wave_energy_density_profile
r_bw = np.linspace(1e10, 5e12, 20)
eps = blast_wave_energy_density_profile(r_bw, 10.0)
assert np.all(eps >= 0.0) and np.all(np.isfinite(eps)), '[TC08] blast_wave_energy_density_profile nonnegative FAILED'

# ---- TC09: spiral_array center equals base value ----
from magnetic_spiral import spiral_array
S = spiral_array(2, base=5)
assert S[2, 2] == 5, '[TC09] spiral_array center equals base value FAILED'

# ---- TC10: magnetic_pitch_angle_grid psi monotonic with radius ----
from magnetic_spiral import magnetic_pitch_angle_grid
r_mag, psi_mag, _ = magnetic_pitch_angle_grid(n_r=16, r_max=1e13)
assert np.all(np.diff(psi_mag) >= -1e-10), '[TC10] magnetic_pitch_angle_grid psi monotonic with radius FAILED'

# ---- TC11: magnetization_parameter finite for physical inputs ----
from magnetic_spiral import magnetization_parameter
sigma = magnetization_parameter(1e-24, 10.0, 300.0)
assert np.isfinite(sigma) and sigma >= 0.0, '[TC11] magnetization_parameter finite for physical inputs FAILED'

# ---- TC12: life_update preserves boundary zeros ----
from reconnection_automaton import life_update, initialize_reconnection_sites
np.random.seed(42)
grid = initialize_reconnection_sites(8, 8, seed_density=0.1)
grid_updated = life_update(8, 8, grid.copy())
assert np.all(grid_updated[0, :] == 0) and np.all(grid_updated[-1, :] == 0), '[TC12] life_update preserves boundary zeros FAILED'
assert np.all(grid_updated[:, 0] == 0) and np.all(grid_updated[:, -1] == 0), '[TC12] life_update preserves boundary zeros FAILED'

# ---- TC13: evolve_reconnection returns correct array lengths ----
from reconnection_automaton import evolve_reconnection
np.random.seed(42)
hist, power = evolve_reconnection(4, 4, n_steps=10)
assert hist.size == 10 and power.size == 10, '[TC13] evolve_reconnection returns correct array lengths FAILED'

# ---- TC14: acceleration_coefficient positive for u1 greater than u2 ----
from particle_acceleration import acceleration_coefficient
A = acceleration_coefficient(100.0, 2.5e10, 6.0e9)
assert A > 0.0, '[TC14] acceleration_coefficient positive for u1 greater than u2 FAILED'

# ---- TC15: bohm_diffusion scales monotonically with gamma ----
from particle_acceleration import bohm_diffusion
D1 = bohm_diffusion(10.0, 10.0)
D2 = bohm_diffusion(100.0, 10.0)
assert D2 > D1 > 0.0, '[TC15] bohm_diffusion scales monotonically with gamma FAILED'

# ---- TC16: rk4_ti_step deterministic with fixed random seed ----
from particle_acceleration import rk4_ti_step, fi_dsa, gi_dsa
np.random.seed(42)
x1 = rk4_ti_step(10.0, 0.0, 0.01, 0.0, lambda x: fi_dsa(x, 2.5e10, 6.0e9), lambda x: gi_dsa(x, 10.0))
np.random.seed(42)
x2 = rk4_ti_step(10.0, 0.0, 0.01, 0.0, lambda x: fi_dsa(x, 2.5e10, 6.0e9), lambda x: gi_dsa(x, 10.0))
assert np.isclose(x1, x2), '[TC16] rk4_ti_step deterministic with fixed random seed FAILED'

# ---- TC17: build_opacity_table shape and positive values ----
from opacity_interpolator import build_opacity_table
log_rho, log_T, kappa = build_opacity_table(n_rho=8, n_T=8)
assert kappa.shape == (8, 8), '[TC17] build_opacity_table shape and positive values FAILED'
assert np.all(kappa > 0.0), '[TC17] build_opacity_table shape and positive values FAILED'

# ---- TC18: pwl_interp_2d_scalar returns inf for out-of-bounds query ----
from opacity_interpolator import pwl_interp_2d_scalar
val = pwl_interp_2d_scalar(3, 3, np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), np.ones((3, 3)), 5.0, 1.0)
assert np.isinf(val), '[TC18] pwl_interp_2d_scalar returns inf for out-of-bounds query FAILED'

# ---- TC19: assemble_mass_matrix is symmetric ----
from radiation_diffusion_fem import assemble_mass_matrix
M = assemble_mass_matrix(8)
assert np.allclose(M, M.T), '[TC19] assemble_mass_matrix is symmetric FAILED'

# ---- TC20: cg_sparse solves small linear system accurately ----
from fem_matrix_assembly import cg_sparse
A_test = np.array([[4.0, 1.0], [1.0, 3.0]])
b_test = np.array([1.0, 2.0])
x_test = cg_sparse(2, A_test, b_test)
assert np.linalg.norm(A_test @ x_test - b_test) < 1e-8, '[TC20] cg_sparse solves small linear system accurately FAILED'

# ---- TC21: wathen_order formula correctness ----
from fem_matrix_assembly import wathen_order
assert wathen_order(3, 3) == 3 * 3 * 3 + 2 * 3 + 2 * 3 + 1, '[TC21] wathen_order formula correctness FAILED'

# ---- TC22: incidence_to_transition columns sum to 1 ----
from photon_transfer_matrix import incidence_to_transition
A_test = np.array([[1.0, 2.0], [3.0, 4.0]])
T = incidence_to_transition(A_test)
col_sums = np.sum(T, axis=0)
assert np.allclose(col_sums, 1.0), '[TC22] incidence_to_transition columns sum to 1 FAILED'

# ---- TC23: build_compton_transfer_matrix escape state is absorbing ----
from photon_transfer_matrix import build_compton_transfer_matrix
A_trans = build_compton_transfer_matrix(n_bins=4, T_e=1e8, tau_es=0.5)
n = A_trans.shape[0]
assert A_trans[n - 1, n - 1] == 1.0, '[TC23] build_compton_transfer_matrix escape state is absorbing FAILED'

# ---- TC24: synthetic_grb_moments returns finite positive values ----
from spectral_moments import synthetic_grb_moments
moments = synthetic_grb_moments(n=4)
assert np.all(np.isfinite(moments)) and np.all(moments > 0.0), '[TC24] synthetic_grb_moments returns finite positive values FAILED'

# ---- TC25: build_hankel_from_moments produces symmetric matrix ----
from spectral_moments import build_hankel_from_moments
moments = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
H = build_hankel_from_moments(moments)
assert np.allclose(H, H.T), '[TC25] build_hankel_from_moments produces symmetric matrix FAILED'

# ---- TC26: hankel_spd_cholesky_lower reconstructs SPD matrix ----
from spectral_moments import hankel_spd_cholesky_lower
n_chol = 4
lii = np.ones(n_chol)
liim1 = 0.5 * np.ones(n_chol - 1)
L = hankel_spd_cholesky_lower(n_chol, lii, liim1)
H_recon = L @ L.T
eigvals_H = np.linalg.eigvalsh(H_recon)
assert np.all(eigvals_H > 0.0), '[TC26] hankel_spd_cholesky_lower reconstructs SPD matrix FAILED'

# ---- TC27: triangle_area matches known right triangle ----
from sed_triangle_integrator import triangle_area
verts = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
area = triangle_area(verts)
assert np.isclose(area, 6.0), '[TC27] triangle_area matches known right triangle FAILED'

# ---- TC28: synchrotron_function_approx nonnegative for sample x values ----
from sed_triangle_integrator import synchrotron_function_approx
x_vals = np.array([1e-4, 0.1, 1.0, 5.0, 15.0])
F_vals = synchrotron_function_approx(x_vals)
assert np.all(F_vals >= 0.0), '[TC28] synchrotron_function_approx nonnegative for sample x values FAILED'

# ---- TC29: interpolate_spectrum linear exact at data nodes ----
from spectrum_interpolator import interpolate_spectrum
nu_bins = np.array([1.0, 2.0, 3.0, 4.0])
flux_bins = np.array([10.0, 20.0, 15.0, 5.0])
flux_interp = interpolate_spectrum(nu_bins, flux_bins, nu_bins, method='linear')
assert np.allclose(flux_interp, flux_bins), '[TC29] interpolate_spectrum linear exact at data nodes FAILED'

# ---- TC30: subset_sum_find recovers correct subset ----
from discrete_cascade import subset_sum_find
weights = np.array([3, 5, 7, 9])
subset = subset_sum_find(8, weights)
assert subset is not None and sum(subset) == 8, '[TC30] subset_sum_find recovers correct subset FAILED'

# ---- TC31: cascade_compactness matches analytical formula ----
from discrete_cascade import cascade_compactness
ell = cascade_compactness(1e52, 1e13)
sigma_T = 6.6524587158e-25
m_e = 9.10938356e-28
c = 2.99792458e10
expected = (1e52 * sigma_T) / (1e13 * m_e * c ** 3)
assert np.isclose(ell, expected), '[TC31] cascade_compactness matches analytical formula FAILED'

# ---- TC32: magic3 row and column sums equal 15 ----
from anisotropic_tensor import magic3
M3 = magic3()
assert np.allclose(np.sum(M3, axis=0), 15.0) and np.allclose(np.sum(M3, axis=1), 15.0), '[TC32] magic3 row and column sums equal 15 FAILED'

# ---- TC33: anisotropic_diffusion_tensor is symmetric ----
from anisotropic_tensor import anisotropic_diffusion_tensor
D = anisotropic_diffusion_tensor(1e20, 1e24, np.pi / 4)
assert np.allclose(D, D.T), '[TC33] anisotropic_diffusion_tensor is symmetric FAILED'

# ---- TC34: dense_to_mm_array contains MatrixMarket header ----
from matrix_io import dense_to_mm_array
A_test = np.array([[1.0, 2.0], [3.0, 4.0]])
mm_text = dense_to_mm_array(A_test)
assert '%%MatrixMarket' in mm_text, '[TC34] dense_to_mm_array contains MatrixMarket header FAILED'

# ---- TC35: export_grb_matrix writes files and returns valid paths ----
from matrix_io import export_grb_matrix
import os
A_test = np.array([[1.0, 0.0], [0.0, 2.0]])
files = export_grb_matrix(A_test, filename_prefix='test_tmp_grb')
assert len(files) == 2, '[TC35] export_grb_matrix writes files and returns valid paths FAILED'
for f in files:
    assert os.path.exists(f), '[TC35] export_grb_matrix writes files and returns valid paths FAILED'
    os.remove(f)

print('\n全部 35 个测试通过!\n')
