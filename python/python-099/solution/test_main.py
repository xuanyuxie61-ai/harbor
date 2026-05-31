"""
main.py
=======
Unified entry point for the Plasma Stealth Absorber Coating (PSAC) project.

Scientific Problem
------------------
Design and evaluate a multi-layer, non-uniform plasma coating for
wideband electromagnetic stealth (radar cross-section reduction).

The coating consists of a stratified collisional plasma slab whose
electron density profile n_e(z) and collision frequency nu(z) are
engineered to absorb incident microwave radiation across multiple
frequency bands (1–40 GHz).

Core physics models:
  1. Drude dielectric function:
         eps(omega) = 1 - omega_p^2/(omega^2+nu^2)
                      - i*nu*omega_p^2/(omega*(omega^2+nu^2))
     where omega_p = sqrt(n_e*e^2/(m_e*eps_0)).

  2. Fresnel reflection / transmission at layer interfaces, combined
     via the transfer-matrix method (TMM) for the full stack.

  3. Electromagnetic power absorption density:
         P_abs(z) = 0.5 * omega * eps_0 * eps_i(z) * |E(z)|^2.

  4. Total absorbed power computed by 3-D Gauss-Legendre quadrature.

  5. Radar Cross Section (RCS) reduction:
         Delta_RCS = 10*log10( R_coating / R_metal )   [dB].

Workflow
--------
  A. Load / generate plasma coating design parameters.
  B. Construct electron density profile via PWL approximation.
  C. Compute complex permittivity profile (Drude model).
  D. Solve electromagnetic field (transfer matrix + finite differences).
  E. Evaluate reflection spectrum and RCS reduction.
  F. Compute 3-D energy absorption via Gaussian quadrature.
  G. Perform Haar wavelet multi-resolution analysis of reflection data.
  H. Optimize coating parameters via quasi-Monte Carlo (Sobol + lattice rules).
  I. Encode multi-band design using Chinese Remainder Theorem.
  J. Propagate parameter uncertainties via Monte Carlo probability sampling.
  K. Print comprehensive numerical report.

All calculations are performed with zero command-line arguments.
"""

import numpy as np
import math
import time

# Project modules
import plasma_drude_model as pdm
import fresnel_coefficients as fc
import layered_field_solver as lfs
import energy_absorption as ea
import wavelet_decomposition as wvd
import qmc_optimizer as qmc
import density_profile as dprof
import wideband_crt as wcrt
from utils import (
    check_runtime_environment,
    newton_solve,
    jacobi_solve,
    matrix_to_st,
    llsq_fit,
    safe_divide,
    clamp,
    ensure_positive,
)


def print_header(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_subheader(title: str):
    print(f"\n--- {title} ---")


def section_a_environment():
    """A. Runtime environment check (from 824_octopus)."""
    print_header("A. RUNTIME ENVIRONMENT")
    env = check_runtime_environment()
    for k, v in env.items():
        print(f"  {k}: {str(v)[:60]}")


def section_b_design_parameters():
    """B. Define coating geometry and plasma parameters."""
    print_header("B. COATING DESIGN PARAMETERS")

    # Coating geometry
    coating_thickness = 5.0e-3          # 5 mm
    num_layers = 40
    z = np.linspace(0.0, coating_thickness, num_layers)

    # Plasma parameters
    peak_density = 1.0e18               # m^-3
    electron_temp = 5000.0              # K
    gas_pressure = 100.0                # Pa

    # Collision frequency from temperature and pressure
    nu_base = pdm.collision_frequency_from_temperature(electron_temp, gas_pressure)

    # Target frequency bands for stealth (GHz -> Hz)
    f_bands = np.array([2.0e9, 5.0e9, 10.0e9, 18.0e9, 35.0e9])
    band_widths = np.array([0.5e9, 1.0e9, 2.0e9, 3.0e9, 5.0e9])

    print(f"  Coating thickness      : {coating_thickness*1e3:.3f} mm")
    print(f"  Number of layers       : {num_layers}")
    print(f"  Peak electron density  : {peak_density:.3e} m^-3")
    print(f"  Electron temperature   : {electron_temp:.1f} K")
    print(f"  Gas pressure           : {gas_pressure:.1f} Pa")
    print(f"  Collision frequency    : {nu_base:.3e} rad/s")
    print(f"  Target stealth bands   : {f_bands/1e9} GHz")

    return {
        "z": z,
        "coating_thickness": coating_thickness,
        "peak_density": peak_density,
        "electron_temp": electron_temp,
        "gas_pressure": gas_pressure,
        "nu_base": nu_base,
        "f_bands": f_bands,
        "band_widths": band_widths,
    }


def section_c_density_profile(params: dict):
    """C. Construct density profile using PWL approximation (from 925_pwl_approx_1d)."""
    print_header("C. ELECTRON DENSITY PROFILE (PWL)")

    z = params["z"]
    peak = params["peak_density"]
    width = params["coating_thickness"] * 0.3

    # Generate a Gaussian-like synthetic profile
    n_e_raw = dprof.generate_density_profile(z, peak, width, profile_type="gaussian")

    # Approximate with fewer control points via PWL (from 925_pwl_approx_1d)
    n_control = 8
    xc = np.linspace(z.min(), z.max(), n_control)
    yc = dprof.pwl_approx_1d(z, n_e_raw, xc)
    n_e_pwl = dprof.pwl_interp_1d(xc, yc, z)

    print(f"  Control points         : {n_control}")
    print(f"  Density at z=0         : {n_e_pwl[0]:.3e} m^-3")
    print(f"  Density at z=max       : {n_e_pwl[-1]:.3e} m^-3")
    print(f"  Max density            : {np.max(n_e_pwl):.3e} m^-3")
    print(f"  Min density            : {np.min(n_e_pwl):.3e} m^-3")

    return {**params, "n_e": n_e_pwl, "xc": xc, "yc": yc}


def section_d_permittivity(params: dict):
    """D. Compute complex permittivity profile (Drude model, from pdm)."""
    print_header("D. COMPLEX PERMITTIVITY PROFILE (DRUDE)")

    z = params["z"]
    n_e = params["n_e"]
    nu = params["nu_base"]
    f_bands = params["f_bands"]

    # Use center frequency for profile evaluation
    f_center = f_bands[f_bands.size // 2]
    omega_center = 2.0 * math.pi * f_center

    eps_profile = pdm.effective_permittivity_profile(z, n_e, np.full_like(z, nu), omega_center)

    print(f"  Evaluation frequency   : {f_center/1e9:.2f} GHz")
    print(f"  eps_r range            : [{np.min(eps_profile.real):.4f}, {np.max(eps_profile.real):.4f}]")
    print(f"  eps_i range            : [{np.min(eps_profile.imag):.4f}, {np.max(eps_profile.imag):.4f}]")

    return {**params, "eps_profile": eps_profile, "omega_center": omega_center}


def section_e_transfer_matrix(params: dict):
    """E. Transfer-matrix reflection spectrum (from fc, Fresnel/TMM)."""
    print_header("E. TRANSFER-MATRIX REFLECTION SPECTRUM")

    z = params["z"]
    n_e = params["n_e"]
    nu = params["nu_base"]
    f_bands = params["f_bands"]
    coating_thickness = params["coating_thickness"]

    num_layers = z.size - 1
    d_layers = np.diff(z)

    R_spectrum = np.zeros(f_bands.size, dtype=float)
    r_spectrum = np.zeros(f_bands.size, dtype=complex)

    for idx, f in enumerate(f_bands):
        omega = 2.0 * math.pi * f
        # Refractive index of each layer: n_j = sqrt(eps_j)
        n_layers = np.zeros(num_layers, dtype=complex)
        for j in range(num_layers):
            eps_j = pdm.drude_permittivity(n_e[j], nu, omega)
            n_layers[j] = np.sqrt(eps_j)

        r_total = fc.multilayer_reflection_stack(n_layers, d_layers, omega, theta0=0.0, polarization="TE")
        R = fc.reflection_power_ratio(r_total)
        R_spectrum[idx] = R
        r_spectrum[idx] = r_total

    # Bare metal reflection (assume perfect conductor, R ≈ 1.0)
    R_metal = np.ones_like(R_spectrum)

    reductions = np.array([ea.rcs_reduction_db(R_spectrum[i], R_metal[i]) for i in range(f_bands.size)])
    avg_reduction = ea.frequency_averaged_rcs_reduction(R_spectrum, R_metal, f_bands)

    print(f"  {'Band (GHz)':>12} {'R_coating':>12} {'RCS red (dB)':>14}")
    for i in range(f_bands.size):
        print(f"  {f_bands[i]/1e9:12.2f} {R_spectrum[i]:12.4e} {reductions[i]:14.2f}")
    print(f"  Average RCS reduction  : {avg_reduction:.2f} dB")

    return {**params, "R_spectrum": R_spectrum, "r_spectrum": r_spectrum, "reductions": reductions, "avg_reduction": avg_reduction}


def section_f_finite_difference(params: dict):
    """F. Finite-difference field solver + Jacobi iteration (from 603_jacobi)."""
    print_header("F. FINITE-DIFFERENCE FIELD SOLVER (JACOBI)")

    z = params["z"]
    eps_profile = params["eps_profile"]
    omega_center = params["omega_center"]

    # Use direct solver for small N; Jacobi for demonstration
    if z.size <= 60:
        E_fd = lfs.solve_fd_direct(z, eps_profile, omega_center)
        res = float(np.linalg.norm(lfs.build_fd_matrix(z, eps_profile, omega_center)[0] @ E_fd - lfs.build_fd_matrix(z, eps_profile, omega_center)[1]))
        it = 0
        conv = True
        method = "Direct (dense)"
    else:
        E_fd, res, it, conv = lfs.solve_fd_jacobi(z, eps_profile, omega_center, max_iter=5000, tol=1e-8)
        method = "Jacobi iterative"

    P_abs = lfs.compute_power_density(E_fd, eps_profile, omega_center)

    # Export FD matrix to sparse triplet (from 783_msm_to_st)
    rows, cols, vals, shape = lfs.fd_matrix_to_st(z, eps_profile, omega_center)
    nnz = rows.size

    print(f"  Solver method          : {method}")
    print(f"  Grid points            : {z.size}")
    print(f"  Residual norm          : {res:.3e}")
    print(f"  Jacobi iterations      : {it}")
    print(f"  Converged              : {conv}")
    print(f"  FD matrix (ST) shape   : {shape}")
    print(f"  Non-zero entries       : {nnz}")
    print(f"  Max |E|                : {np.max(np.abs(E_fd)):.4e}")
    print(f"  Max P_abs              : {np.max(P_abs):.4e} W/m^3")

    return {**params, "E_fd": E_fd, "P_abs_fd": P_abs, "fd_residual": res, "fd_converged": conv}


def section_g_energy_absorption(params: dict):
    """G. 3-D energy absorption integral (from 232_cube_felippa_rule)."""
    print_header("G. 3-D ENERGY ABSORPTION INTEGRAL")

    z = params["z"]
    P_abs = params["P_abs_fd"]
    coating_thickness = params["coating_thickness"]

    # Create an interpolable power density function
    # P_abs is only defined along z; assume uniform in x and y
    P_max = float(np.max(P_abs))
    P_min = float(np.min(P_abs))

    def power_density_3d(x, y, zq):
        # 1-D interpolation along z
        if zq <= z[0]:
            p = P_abs[0]
        elif zq >= z[-1]:
            p = P_abs[-1]
        else:
            j = int(np.searchsorted(z, zq) - 1)
            j = clamp(j, 0, z.size - 2)
            h = z[j + 1] - z[j]
            if abs(h) < 1e-14:
                p = P_abs[j]
            else:
                w = (zq - z[j]) / h
                p = (1.0 - w) * P_abs[j] + w * P_abs[j + 1]
        # Uniform in x, y
        return float(p)

    Lx = 0.1  # 10 cm
    Ly = 0.1
    total_absorbed = ea.absorbed_energy_in_coating(
        power_density_3d, Lx, Ly, coating_thickness, order=3
    )
    area = Lx * Ly
    incident_power_density = 1.0  # 1 W/m^2 reference
    eta = ea.absorption_efficiency(total_absorbed, incident_power_density, area)

    print(f"  Slab dimensions        : {Lx*1e2:.1f} cm x {Ly*1e2:.1f} cm x {coating_thickness*1e3:.1f} mm")
    print(f"  Total absorbed power   : {total_absorbed:.4e} W")
    print(f"  Absorption efficiency  : {eta:.6f}")

    return {**params, "total_absorbed": total_absorbed, "absorption_efficiency": eta}


def section_h_wavelet_analysis(params: dict):
    """H. Haar wavelet decomposition of reflection spectrum (from 496_haar_transform)."""
    print_header("H. HAAR WAVELET ANALYSIS OF REFLECTION SPECTRUM")

    R_spectrum = params["R_spectrum"]

    v = wvd.haar_1d_transform(R_spectrum)
    energies = wvd.multiscale_energy_distribution(R_spectrum)
    peaks = wvd.detect_reflection_peaks_haar(R_spectrum, threshold_factor=1.5)

    print(f"  Original signal length : {R_spectrum.size}")
    print(f"  Number of scales       : {energies.size}")
    print(f"  Energy per scale       : {energies}")
    print(f"  Detected peak indices  : {peaks}")

    # Reconstruct to verify invertibility
    R_recon = wvd.haar_1d_inverse(v)
    recon_error = float(np.linalg.norm(R_spectrum - R_recon))
    print(f"  Reconstruction error   : {recon_error:.3e}")

    return {**params, "wavelet_coeffs": v, "wavelet_energies": energies, "peaks": peaks}


def section_i_qmc_optimization(params: dict):
    """I. QMC parameter optimization (from 1097_sobol, 654_lattice_rule)."""
    print_header("I. QMC COATING PARAMETER OPTIMIZATION")

    z = params["z"]
    nu_base = params["nu_base"]
    f_bands = params["f_bands"]
    coating_thickness = params["coating_thickness"]

    # Objective: minimize average reflection power over target bands
    # Parameter vector: [peak_density_log10, profile_width_frac, nu_scale]
    def objective(p):
        peak_d = 10.0 ** clamp(p[0], 15.0, 20.0)
        width_frac = clamp(p[1], 0.05, 0.95)
        nu_scale = clamp(p[2], 0.1, 10.0)

        width = coating_thickness * width_frac
        n_e_test = dprof.generate_density_profile(z, peak_d, width, profile_type="gaussian")
        nu_test = nu_base * nu_scale

        num_layers = z.size - 1
        d_layers = np.diff(z)
        avg_R = 0.0
        for f in f_bands:
            omega = 2.0 * math.pi * f
            n_layers = np.zeros(num_layers, dtype=complex)
            for j in range(num_layers):
                eps_j = pdm.drude_permittivity(n_e_test[j], nu_test, omega)
                n_layers[j] = np.sqrt(eps_j)
            r_total = fc.multilayer_reflection_stack(n_layers, d_layers, omega, theta0=0.0, polarization="TE")
            avg_R += fc.reflection_power_ratio(r_total)
        avg_R /= f_bands.size
        return avg_R

    bounds = np.array([[15.0, 19.0], [0.1, 0.8], [0.5, 5.0]])
    best_p, best_val = qmc.optimize_coating_parameters_qmc(
        objective, dim=3, n_samples=256, param_bounds=bounds, seed=42
    )

    print(f"  Sobol sample size      : 256")
    print(f"  Best peak density      : {10.0**best_p[0]:.3e} m^-3")
    print(f"  Best width fraction    : {best_p[1]:.4f}")
    print(f"  Best nu scale          : {best_p[2]:.4f}")
    print(f"  Best avg reflection    : {best_val:.4e}")

    # Lattice-rule integral of the objective over the parameter space
    def lattice_objective(x):
        p = bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])
        return objective(p)

    lattice_avg = qmc.lattice_rule_integrate(lattice_objective, dim_num=3, m=128)
    print(f"  Lattice-rule avg obj   : {lattice_avg:.4e}")

    return {**params, "best_params": best_p, "best_reflection": best_val, "lattice_avg": lattice_avg}


def section_j_crt_encoding(params: dict):
    """J. Multi-band CRT encoding (from 170_chinese_remainder_theorem)."""
    print_header("J. MULTI-BAND CRT ENCODING")

    f_bands = params["f_bands"]
    band_widths = params["band_widths"]

    design = wcrt.design_multiband_coating_parameters(f_bands, band_widths, base_plasma_freq=2.0 * math.pi * 10e9)

    print(f"  Target bands (Hz)      : {f_bands}")
    print(f"  Moduli                 : {design['moduli']}")
    print(f"  Remainders             : {design['remainders']}")
    print(f"  Composite index        : {design['composite_index']}")

    # Verify decode
    decoded = wcrt.decode_frequency_bands(design["composite_index"], design["moduli"], design["bin_width_hz"])
    print(f"  Decoded frequencies    : {decoded}")
    print(f"  Encoding error (max)   : {np.max(np.abs(decoded - f_bands)):.3e} Hz")

    return {**params, "crt_design": design}


def section_k_uncertainty_quantification(params: dict):
    """K. Monte Carlo uncertainty propagation (from 918_prob)."""
    print_header("K. UNCERTAINTY QUANTIFICATION (MONTE CARLO)")

    z = params["z"]
    nu_base = params["nu_base"]
    f_bands = params["f_bands"]
    coating_thickness = params["coating_thickness"]
    best_p = params["best_params"]

    # Mean parameters around the QMC optimum
    peak_d_mean = 10.0 ** best_p[0]
    width_frac_mean = best_p[1]
    nu_scale_mean = best_p[2]

    param_means = np.array([peak_d_mean, width_frac_mean * coating_thickness, nu_scale_mean * nu_base])
    param_stds = np.array([peak_d_mean * 0.1, coating_thickness * 0.05, nu_scale_mean * nu_base * 0.2])

    def model(p):
        peak_d, width, nu = p[0], p[1], p[2]
        n_e = dprof.generate_density_profile(z, peak_d, width, profile_type="gaussian")
        num_layers = z.size - 1
        d_layers = np.diff(z)
        avg_R = 0.0
        for f in f_bands:
            omega = 2.0 * math.pi * f
            n_layers = np.zeros(num_layers, dtype=complex)
            for j in range(num_layers):
                eps_j = pdm.drude_permittivity(n_e[j], nu, omega)
                n_layers[j] = np.sqrt(eps_j)
            r_total = fc.multilayer_reflection_stack(n_layers, d_layers, omega, theta0=0.0, polarization="TE")
            avg_R += fc.reflection_power_ratio(r_total)
        return avg_R / f_bands.size

    mean_val, std_val, samples = qmc.monte_carlo_uncertainty_propagation(
        model, param_means, param_stds, n_mc=500, seed=123
    )

    print(f"  MC samples             : 500")
    print(f"  Mean reflection        : {mean_val:.4e}")
    print(f"  Std deviation          : {std_val:.4e}")
    print(f"  Coefficient of variation: {safe_divide(std_val, mean_val, 0.0):.4f}")

    return {**params, "mc_mean": mean_val, "mc_std": std_val}


def section_l_electron_statistics(params: dict):
    """L. Electron spatial distribution statistics (from 1125_sphere_positive_distance)."""
    print_header("L. ELECTRON SCATTERING STATISTICS")

    stats = dprof.sphere_positive_distance_stats(n=200, seed=99)
    mean_cos, std_cos, cos_samples = dprof.electron_scattering_angle_distribution(n_samples=500, seed=99)

    print(f"  Sphere distance mean   : {stats['mean']:.4f}")
    print(f"  Sphere distance std    : {stats['std']:.4f}")
    print(f"  Forward scattering (<cos>) : {mean_cos:.4f}")
    print(f"  Angular std            : {std_cos:.4f}")

    return {**params, "sphere_stats": stats, "mean_cos_theta": mean_cos}


def section_m_inversion_and_newton(params: dict):
    """M. Least-squares inversion + Newton dispersion solve (from 692_llsq, 808_nonlin_newton)."""
    print_header("M. PARAMETER INVERSION & NONLINEAR DISPERSION")

    f_bands = params["f_bands"]
    R_spectrum = params["R_spectrum"]

    # Invert reflection data for effective omega_p^2 (from 692_llsq)
    A_fit, residual, rms = fc.invert_reflection_for_epsilon(f_bands, R_spectrum, theta=0.0, polarization="TE")
    omega_p_inv = math.sqrt(max(A_fit, 0.0))

    print(f"  LSQ fit coefficient A  : {A_fit:.4e}")
    print(f"  Inferred omega_p       : {omega_p_inv:.3e} rad/s")
    print(f"  Inferred n_e (approx)  : {(omega_p_inv**2 * 9.10938356e-31 * 8.854187817e-12) / (1.602176634e-19**2):.3e} m^-3")
    print(f"  LSQ residual           : {residual:.3e}")
    print(f"  RMS error              : {rms:.3e}")

    # Newton solve for nonlinear surface-wave dispersion
    # Use the lowest frequency band (2 GHz) where eps_p < 0 (overdense)
    n_e_peak = params["peak_density"]
    nu = params["nu_base"]
    f_test = f_bands[0]
    omega_test = 2.0 * math.pi * f_test
    k_sol, it, conv = pdm.nonlinear_dispersion_relation(omega_test, n_e_peak, nu)

    print(f"  Nonlinear k_solution   : {k_sol:.4e} m^-1")
    print(f"  Newton iterations      : {it}")
    print(f"  Converged              : {conv}")

    return {**params, "inferred_omega_p": omega_p_inv, "k_dispersion": k_sol}


def section_n_summary(params: dict):
    """N. Final summary."""
    print_header("N. PROJECT SUMMARY")
    print(f"  Average RCS reduction  : {params['avg_reduction']:.2f} dB")
    print(f"  Absorption efficiency  : {params['absorption_efficiency']:.6f}")
    print(f"  Total absorbed power   : {params['total_absorbed']:.4e} W")
    print(f"  FD solver converged    : {params['fd_converged']}")
    print(f"  Best QMC reflection    : {params['best_reflection']:.4e}")
    print(f"  CRT composite index    : {params['crt_design']['composite_index']}")
    print(f"  MC mean reflection     : {params['mc_mean']:.4e}")
    print(f"  Inferred omega_p       : {params['inferred_omega_p']:.3e} rad/s")
    print(f"  Wavelet peaks detected : {params['peaks']}")
    print("\n  All sections completed successfully.")


def main():
    """Main execution pipeline."""
    t_start = time.time()

    # A. Environment
    section_a_environment()

    # B. Parameters
    params = section_b_design_parameters()

    # C. Density profile (PWL)
    params = section_c_density_profile(params)

    # D. Permittivity (Drude)
    params = section_d_permittivity(params)

    # E. Transfer matrix reflection
    params = section_e_transfer_matrix(params)

    # F. Finite difference + Jacobi
    params = section_f_finite_difference(params)

    # G. 3-D energy absorption
    params = section_g_energy_absorption(params)

    # H. Haar wavelet analysis
    params = section_h_wavelet_analysis(params)

    # I. QMC optimization
    params = section_i_qmc_optimization(params)

    # J. CRT encoding
    params = section_j_crt_encoding(params)

    # K. Uncertainty quantification
    params = section_k_uncertainty_quantification(params)

    # L. Electron statistics
    params = section_l_electron_statistics(params)

    # M. Inversion + Newton
    params = section_m_inversion_and_newton(params)

    # N. Summary
    section_n_summary(params)

    t_elapsed = time.time() - t_start
    print(f"\n  Total execution time   : {t_elapsed:.3f} s")
    print("=" * 72)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（28个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# Additional imports for functions not brought into main.py namespace
from utils import fresnel_cos, fresnel_sin, llsq_fit_through_origin, gauss_legendre_1d

# ---- TC01: plasma_frequency returns positive finite value for typical density ----
omega_p = pdm.plasma_frequency(1.0e18)
assert np.isfinite(omega_p) and omega_p > 0.0, '[TC01] plasma_frequency positive finite FAILED'

# ---- TC02: drude_permittivity for overdense plasma has eps_r < 1 ----
eps = pdm.drude_permittivity(1.0e18, 1.0e10, 2.0 * math.pi * 10.0e9)
assert eps.real < 1.0, '[TC02] drude_permittivity eps_r < 1 FAILED'
assert eps.imag >= 0.0, '[TC02] drude_permittivity eps_i non-negative FAILED'

# ---- TC03: drude_permittivity approaches vacuum at very high frequency ----
eps_high = pdm.drude_permittivity(1.0e18, 1.0e10, 2.0 * math.pi * 1000.0e9)
assert abs(eps_high.real - 1.0) < 1e-3, '[TC03] high-frequency eps_r ~ 1 FAILED'
assert abs(eps_high.imag) < 1e-3, '[TC03] high-frequency eps_i ~ 0 FAILED'

# ---- TC04: wave_number has positive imaginary part indicating attenuation ----
k = pdm.wave_number(1.0e18, 1.0e10, 2.0 * math.pi * 10.0e9)
assert k.imag > 0.0, '[TC04] wave_number imag positive FAILED'

# ---- TC05: collision_frequency increases with electron temperature ----
nu_300 = pdm.collision_frequency_from_temperature(300.0, 100.0)
nu_5000 = pdm.collision_frequency_from_temperature(5000.0, 100.0)
assert nu_5000 > nu_300, '[TC05] collision_frequency monotonic with T_e FAILED'

# ---- TC06: skin_depth decreases with increasing electron density ----
delta_low = pdm.skin_depth(1.0e16, 1.0e10, 2.0 * math.pi * 10.0e9)
delta_high = pdm.skin_depth(1.0e19, 1.0e10, 2.0 * math.pi * 10.0e9)
assert delta_high < delta_low, '[TC06] skin_depth decreases with n_e FAILED'

# ---- TC07: effective_permittivity_profile output shape matches input z ----
z_test = np.linspace(0.0, 1.0e-3, 20)
n_e_test = np.full(20, 1.0e17)
nu_test = np.full(20, 1.0e10)
eps_prof = pdm.effective_permittivity_profile(z_test, n_e_test, nu_test, 2.0 * math.pi * 10.0e9)
assert eps_prof.shape == z_test.shape, '[TC07] permittivity_profile shape FAILED'

# ---- TC08: Fresnel reflection at normal incidence between identical media is zero ----
r = fc.fresnel_reflection_coefficient(1.0 + 0j, 1.0 + 0j, 0.0, "TE")
assert abs(r) < 1e-12, '[TC08] Fresnel r=0 for identical media FAILED'

# ---- TC09: reflection_power_ratio of zero amplitude reflection is zero ----
R = fc.reflection_power_ratio(0.0 + 0j)
assert abs(R) < 1e-15, '[TC09] reflection_power_ratio zero FAILED'

# ---- TC10: multilayer stack with single vacuum layer has zero reflection ----
n_layers = np.array([1.0 + 0j])
d_layers = np.array([1.0e-3])
r_total = fc.multilayer_reflection_stack(n_layers, d_layers, 2.0 * math.pi * 1.0e9, theta0=0.0, polarization="TE")
assert abs(r_total) < 1e-10, '[TC10] vacuum layer reflection zero FAILED'

# ---- TC11: field_transfer_matrix output length equals z input length ----
z_fd = np.linspace(0.0, 5.0e-3, 15)
eps_fd = np.ones(15, dtype=complex)
E_fd = lfs.field_transfer_matrix(z_fd, eps_fd, 2.0 * math.pi * 10.0e9, E0=1.0, theta=0.0)
assert E_fd.shape == z_fd.shape, '[TC11] field_transfer_matrix shape FAILED'

# ---- TC12: build_fd_matrix returns square matrix of correct size ----
z_mat = np.linspace(0.0, 1.0e-3, 10)
eps_mat = np.ones(10, dtype=complex)
A_mat, b_mat = lfs.build_fd_matrix(z_mat, eps_mat, 2.0 * math.pi * 10.0e9)
assert A_mat.shape == (10, 10), '[TC12] FD matrix shape FAILED'
assert b_mat.shape == (10,), '[TC12] FD rhs shape FAILED'

# ---- TC13: compute_power_density is non-negative for all points ----
E_test = np.ones(10, dtype=complex)
eps_test = np.ones(10, dtype=complex) * (1.0 - 0.5j)
P_test = lfs.compute_power_density(E_test, eps_test, 2.0 * math.pi * 10.0e9)
assert np.all(P_test >= 0.0), '[TC13] power_density non-negative FAILED'

# ---- TC14: RCS reduction is negative when coating reflects less than metal ----
red_db = ea.rcs_reduction_db(0.1, 1.0)
assert red_db < 0.0, '[TC14] RCS reduction negative FAILED'

# ---- TC15: absorption_efficiency lies in [0, 1] ----
eta = ea.absorption_efficiency(5.0, 10.0, 1.0)
assert 0.0 <= eta <= 1.0, '[TC15] absorption_efficiency range FAILED'

# ---- TC16: 3D Gauss-Legendre integrates constant function exactly ----
def const_func(x, y, z):
    return 2.0
integral_val = ea.integrate_3d_cube_gauss(const_func, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), order_x=1, order_y=1, order_z=1)
assert abs(integral_val - 2.0) < 1e-12, '[TC16] constant 3D integral FAILED'

# ---- TC17: Haar transform followed by inverse reproduces original signal ----
signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
v_haar = wvd.haar_1d_transform(signal)
signal_recon = wvd.haar_1d_inverse(v_haar)
recon_err = float(np.linalg.norm(signal - signal_recon))
assert recon_err < 1e-12, '[TC17] Haar invertibility FAILED'

# ---- TC18: multiscale_energy_distribution preserves total energy (Parseval) ----
sig = np.array([1.5, -0.5, 2.0, 1.0])
energies = wvd.multiscale_energy_distribution(sig)
total_energy = float(np.sum(energies))
original_energy = float(np.sum(sig ** 2))
assert abs(total_energy - original_energy) < 1e-12, '[TC18] Parseval energy conservation FAILED'

# ---- TC19: Sobol sequence is deterministic for same parameters ----
pts1 = qmc.sobol_sequence(3, 16, skip=0)
pts2 = qmc.sobol_sequence(3, 16, skip=0)
assert np.allclose(pts1, pts2), '[TC19] Sobol sequence determinism FAILED'

# ---- TC20: lattice_rule_integrate is exact for constant function ----
const_val = qmc.lattice_rule_integrate(lambda x: 4.0, dim_num=2, m=64)
assert abs(const_val - 4.0) < 1e-12, '[TC20] lattice_rule constant integral FAILED'

# ---- TC21: Gaussian density profile peak equals specified peak_density ----
z_prof = np.linspace(0.0, 1.0e-3, 51)
n_e_prof = dprof.generate_density_profile(z_prof, 2.0e18, 2.0e-4, profile_type="gaussian")
assert abs(np.max(n_e_prof) - 2.0e18) < 1.0e12, '[TC21] gaussian peak density FAILED'

# ---- TC22: pwl_interp_1d interpolates control points exactly ----
xc_pwl = np.array([0.0, 1.0, 2.0, 3.0])
yc_pwl = np.array([0.0, 1.0, 4.0, 9.0])
yi_pwl = dprof.pwl_interp_1d(xc_pwl, yc_pwl, xc_pwl)
assert np.allclose(yi_pwl, yc_pwl), '[TC22] pwl_interp exact at control points FAILED'

# ---- TC23: Chinese Remainder Theorem satisfies all original congruences ----
remainders = np.array([2, 3, 1])
moduli = np.array([3, 5, 7])
x_sol = wcrt.chinese_remainder_theorem(remainders, moduli)
for i in range(remainders.size):
    assert x_sol % int(moduli[i]) == int(remainders[i]), '[TC23] CRT congruence FAILED'

# ---- TC24: encode/decode frequency bands roundtrip consistency ----
freqs = np.array([1.0e9, 2.5e9, 5.0e9])
rem, mods, comp = wcrt.encode_frequency_bands(freqs, bin_width_hz=1.0e8)
decoded = wcrt.decode_frequency_bands(comp, mods, bin_width_hz=1.0e8)
assert np.allclose(decoded, freqs, rtol=1e-10), '[TC24] CRT encode/decode roundtrip FAILED'

# ---- TC25: Fresnel integrals at origin are zero ----
assert abs(fresnel_cos(0.0)) < 1e-15, '[TC25] fresnel_cos(0) FAILED'
assert abs(fresnel_sin(0.0)) < 1e-15, '[TC25] fresnel_sin(0) FAILED'

# ---- TC26: Newton solve converges for simple quadratic equation ----
def f_q(x):
    return x * x - 4.0
def fp_q(x):
    return 2.0 * x
root, f_root, iters, conv = newton_solve(f_q, fp_q, 3.0)
assert conv and abs(root - 2.0) < 1e-10, '[TC26] Newton solve quadratic FAILED'

# ---- TC27: Jacobi solver converges for diagonally dominant system ----
A_jac = np.array([[4.0, 1.0], [1.0, 4.0]], dtype=float)
b_jac = np.array([5.0, 5.0], dtype=float)
x_sol_jac, res_jac, it_jac, conv_jac = jacobi_solve(A_jac, b_jac, max_iter=5000, tol=1e-10)
assert conv_jac and res_jac < 1e-9, '[TC27] Jacobi solver convergence FAILED'

# ---- TC28: llsq_fit_through_origin recovers exact slope for y=3x ----
x_llsq = np.array([1.0, 2.0, 3.0, 4.0])
y_llsq = 3.0 * x_llsq
slope_llsq, resid_llsq = llsq_fit_through_origin(x_llsq, y_llsq)
assert abs(slope_llsq - 3.0) < 1e-12, '[TC28] llsq_fit_through_origin slope FAILED'
assert resid_llsq < 1e-12, '[TC28] llsq_fit_through_origin residual FAILED'

print('\n全部 28 个测试通过!\n')
