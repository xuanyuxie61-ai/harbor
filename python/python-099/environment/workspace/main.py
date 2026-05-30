
import numpy as np
import math
import time


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
    print_header("A. RUNTIME ENVIRONMENT")
    env = check_runtime_environment()
    for k, v in env.items():
        print(f"  {k}: {str(v)[:60]}")


def section_b_design_parameters():
    print_header("B. COATING DESIGN PARAMETERS")


    coating_thickness = 5.0e-3
    num_layers = 40
    z = np.linspace(0.0, coating_thickness, num_layers)


    peak_density = 1.0e18
    electron_temp = 5000.0
    gas_pressure = 100.0


    nu_base = pdm.collision_frequency_from_temperature(electron_temp, gas_pressure)


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
    print_header("C. ELECTRON DENSITY PROFILE (PWL)")

    z = params["z"]
    peak = params["peak_density"]
    width = params["coating_thickness"] * 0.3


    n_e_raw = dprof.generate_density_profile(z, peak, width, profile_type="gaussian")


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
    print_header("D. COMPLEX PERMITTIVITY PROFILE (DRUDE)")

    z = params["z"]
    n_e = params["n_e"]
    nu = params["nu_base"]
    f_bands = params["f_bands"]


    f_center = f_bands[f_bands.size // 2]
    omega_center = 2.0 * math.pi * f_center

    eps_profile = pdm.effective_permittivity_profile(z, n_e, np.full_like(z, nu), omega_center)

    print(f"  Evaluation frequency   : {f_center/1e9:.2f} GHz")
    print(f"  eps_r range            : [{np.min(eps_profile.real):.4f}, {np.max(eps_profile.real):.4f}]")
    print(f"  eps_i range            : [{np.min(eps_profile.imag):.4f}, {np.max(eps_profile.imag):.4f}]")

    return {**params, "eps_profile": eps_profile, "omega_center": omega_center}


def section_e_transfer_matrix(params: dict):
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






        raise NotImplementedError("Hole 3: layer refractive index and reflection calculation not implemented.")


    R_metal = np.ones_like(R_spectrum)

    reductions = np.array([ea.rcs_reduction_db(R_spectrum[i], R_metal[i]) for i in range(f_bands.size)])
    avg_reduction = ea.frequency_averaged_rcs_reduction(R_spectrum, R_metal, f_bands)

    print(f"  {'Band (GHz)':>12} {'R_coating':>12} {'RCS red (dB)':>14}")
    for i in range(f_bands.size):
        print(f"  {f_bands[i]/1e9:12.2f} {R_spectrum[i]:12.4e} {reductions[i]:14.2f}")
    print(f"  Average RCS reduction  : {avg_reduction:.2f} dB")

    return {**params, "R_spectrum": R_spectrum, "r_spectrum": r_spectrum, "reductions": reductions, "avg_reduction": avg_reduction}


def section_f_finite_difference(params: dict):
    print_header("F. FINITE-DIFFERENCE FIELD SOLVER (JACOBI)")

    z = params["z"]
    eps_profile = params["eps_profile"]
    omega_center = params["omega_center"]


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
    print_header("G. 3-D ENERGY ABSORPTION INTEGRAL")

    z = params["z"]
    P_abs = params["P_abs_fd"]
    coating_thickness = params["coating_thickness"]



    P_max = float(np.max(P_abs))
    P_min = float(np.min(P_abs))

    def power_density_3d(x, y, zq):

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

        return float(p)

    Lx = 0.1
    Ly = 0.1
    total_absorbed = ea.absorbed_energy_in_coating(
        power_density_3d, Lx, Ly, coating_thickness, order=3
    )
    area = Lx * Ly
    incident_power_density = 1.0
    eta = ea.absorption_efficiency(total_absorbed, incident_power_density, area)

    print(f"  Slab dimensions        : {Lx*1e2:.1f} cm x {Ly*1e2:.1f} cm x {coating_thickness*1e3:.1f} mm")
    print(f"  Total absorbed power   : {total_absorbed:.4e} W")
    print(f"  Absorption efficiency  : {eta:.6f}")

    return {**params, "total_absorbed": total_absorbed, "absorption_efficiency": eta}


def section_h_wavelet_analysis(params: dict):
    print_header("H. HAAR WAVELET ANALYSIS OF REFLECTION SPECTRUM")

    R_spectrum = params["R_spectrum"]

    v = wvd.haar_1d_transform(R_spectrum)
    energies = wvd.multiscale_energy_distribution(R_spectrum)
    peaks = wvd.detect_reflection_peaks_haar(R_spectrum, threshold_factor=1.5)

    print(f"  Original signal length : {R_spectrum.size}")
    print(f"  Number of scales       : {energies.size}")
    print(f"  Energy per scale       : {energies}")
    print(f"  Detected peak indices  : {peaks}")


    R_recon = wvd.haar_1d_inverse(v)
    recon_error = float(np.linalg.norm(R_spectrum - R_recon))
    print(f"  Reconstruction error   : {recon_error:.3e}")

    return {**params, "wavelet_coeffs": v, "wavelet_energies": energies, "peaks": peaks}


def section_i_qmc_optimization(params: dict):
    print_header("I. QMC COATING PARAMETER OPTIMIZATION")

    z = params["z"]
    nu_base = params["nu_base"]
    f_bands = params["f_bands"]
    coating_thickness = params["coating_thickness"]



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


    def lattice_objective(x):
        p = bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])
        return objective(p)

    lattice_avg = qmc.lattice_rule_integrate(lattice_objective, dim_num=3, m=128)
    print(f"  Lattice-rule avg obj   : {lattice_avg:.4e}")

    return {**params, "best_params": best_p, "best_reflection": best_val, "lattice_avg": lattice_avg}


def section_j_crt_encoding(params: dict):
    print_header("J. MULTI-BAND CRT ENCODING")

    f_bands = params["f_bands"]
    band_widths = params["band_widths"]

    design = wcrt.design_multiband_coating_parameters(f_bands, band_widths, base_plasma_freq=2.0 * math.pi * 10e9)

    print(f"  Target bands (Hz)      : {f_bands}")
    print(f"  Moduli                 : {design['moduli']}")
    print(f"  Remainders             : {design['remainders']}")
    print(f"  Composite index        : {design['composite_index']}")


    decoded = wcrt.decode_frequency_bands(design["composite_index"], design["moduli"], design["bin_width_hz"])
    print(f"  Decoded frequencies    : {decoded}")
    print(f"  Encoding error (max)   : {np.max(np.abs(decoded - f_bands)):.3e} Hz")

    return {**params, "crt_design": design}


def section_k_uncertainty_quantification(params: dict):
    print_header("K. UNCERTAINTY QUANTIFICATION (MONTE CARLO)")

    z = params["z"]
    nu_base = params["nu_base"]
    f_bands = params["f_bands"]
    coating_thickness = params["coating_thickness"]
    best_p = params["best_params"]


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
    print_header("L. ELECTRON SCATTERING STATISTICS")

    stats = dprof.sphere_positive_distance_stats(n=200, seed=99)
    mean_cos, std_cos, cos_samples = dprof.electron_scattering_angle_distribution(n_samples=500, seed=99)

    print(f"  Sphere distance mean   : {stats['mean']:.4f}")
    print(f"  Sphere distance std    : {stats['std']:.4f}")
    print(f"  Forward scattering (<cos>) : {mean_cos:.4f}")
    print(f"  Angular std            : {std_cos:.4f}")

    return {**params, "sphere_stats": stats, "mean_cos_theta": mean_cos}


def section_m_inversion_and_newton(params: dict):
    print_header("M. PARAMETER INVERSION & NONLINEAR DISPERSION")

    f_bands = params["f_bands"]
    R_spectrum = params["R_spectrum"]


    A_fit, residual, rms = fc.invert_reflection_for_epsilon(f_bands, R_spectrum, theta=0.0, polarization="TE")
    omega_p_inv = math.sqrt(max(A_fit, 0.0))

    print(f"  LSQ fit coefficient A  : {A_fit:.4e}")
    print(f"  Inferred omega_p       : {omega_p_inv:.3e} rad/s")
    print(f"  Inferred n_e (approx)  : {(omega_p_inv**2 * 9.10938356e-31 * 8.854187817e-12) / (1.602176634e-19**2):.3e} m^-3")
    print(f"  LSQ residual           : {residual:.3e}")
    print(f"  RMS error              : {rms:.3e}")



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
    t_start = time.time()


    section_a_environment()


    params = section_b_design_parameters()


    params = section_c_density_profile(params)


    params = section_d_permittivity(params)


    params = section_e_transfer_matrix(params)


    params = section_f_finite_difference(params)


    params = section_g_energy_absorption(params)


    params = section_h_wavelet_analysis(params)


    params = section_i_qmc_optimization(params)


    params = section_j_crt_encoding(params)


    params = section_k_uncertainty_quantification(params)


    params = section_l_electron_statistics(params)


    params = section_m_inversion_and_newton(params)


    section_n_summary(params)

    t_elapsed = time.time() - t_start
    print(f"\n  Total execution time   : {t_elapsed:.3f} s")
    print("=" * 72)


if __name__ == "__main__":
    main()
