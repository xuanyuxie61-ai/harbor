#!/usr/bin/env python3
"""
Dusty Plasma Crystallization Phase Transition Simulator
========================================================

A PhD-level scientific computing project simulating the phase transition
of a complex (dusty) plasma from a gaseous/disordered state to a crystalline
ordered state (plasma crystal / Coulomb crystal).

Scientific Background
---------------------
In a complex plasma, micron-sized dust particles levitate in the plasma sheath,
acquire high negative charges (typically 10^3 - 10^4 elementary charges), and
interact via a screened Coulomb (Yukawa) potential:

    U(r) = (Q_eff^2 / (4 * pi * eps0 * r)) * exp(-r / lambda_D)

The thermodynamic state is characterized by the coupling parameter:

    Gamma = (Q_eff^2 / (4*pi*eps0 * a_WS * k_B * T_dust)) * exp(-kappa)

where a_WS is the Wigner-Seitz radius and kappa = a_WS / lambda_D.

Phase transition to the crystalline state occurs at Gamma_c ~ 170 for
isotropic 3D Coulomb systems (Ikezi, 1986), with Gamma_c increasing for
Yukawa systems (Hamaguchi et al., 1997).

This simulation performs:
  1. Lattice construction and geometric analysis
  2. Energy minimization via Monte Carlo
  3. Dynamical matrix construction and phonon analysis
  4. Thermal property computation via quadrature methods
  5. Phase transition detection via Lindemann criterion
  6. Particle trajectory integration with nonlinear forcing
  7. Numerical verification of all subroutines

Seed projects integrated (15 total):
  - 737_matrix_analyze    -> Matrix property analysis
  - 1098_solve            -> Gaussian elimination solver
  - 1049_rubber_band_ode  -> Piecewise ODE dynamics
  - 669_levenshtein_matrix-> Configuration distance DP
  - 1309_triangle_interpolate -> Barycentric interpolation
  - 604_jacobi_eigenvalue -> Jacobi eigenvalue method
  - 1406_wedge_exactness  -> Wedge integral exactness
  - 1159_st_to_msm        -> Sparse triplet conversion
  - 281_diff2_center      -> Centered finite differences
  - 1039_rng_cliff        -> Cliff random number generator
  - 1367_tsp_random       -> Monte Carlo random search
  - 323_e_spigot          -> Spigot algorithm for e
  - 150_cg_lab_triangles  -> Signed point-line distance
  - 1206_test_eigen       -> Test matrix generation
  - 464_gen_hermite_exactness -> Generalized Hermite quadrature
"""

import numpy as np
import time

from lattice_geometry import (
    build_hexagonal_lattice, build_square_lattice,
    signed_point_line_distance, triangle_area,
    barycentric_interpolate, uniform_in_triangle
)
from yukawa_physics import (
    coupling_parameter, debye_length, wigner_seitz_radius,
    diff2_center, e_spigot, yukawa_potential, yukawa_force_magnitude
)
from particle_dynamics import (
    cliff_sequence, dust_trajectory_deriv,
    integrate_trajectories, compute_mean_square_displacement,
    compute_kinetic_temperature
)
from linear_algebra import (
    gaussian_solve, sparse_triplet_to_dense, analyze_matrix_properties
)
from dynamical_analysis import (
    jacobi_eigenvalue, generate_test_symmetric_matrix,
    verify_eigensystem, compute_dynamical_matrix, compute_phonon_frequencies
)
from config_optimizer import (
    total_yukawa_energy, monte_carlo_relax, configuration_distance
)
from thermal_quadrature import (
    generalized_hermite_integral, wedge_monomial_integral,
    thermal_average_displacement, partition_function_harmonic,
    specific_heat_harmonic, gauss_hermite_quadrature
)
from phase_transition import (
    lindemann_parameter, radial_distribution_function,
    structure_factor, detect_phase, compute_bond_orientational_order,
    debye_waller_factor
)


def run_simulation():
    print("=" * 75)
    print("  Dusty Plasma Crystallization Phase Transition Simulator")
    print("  (PhD-level Scientific Computing Project)")
    print("=" * 75)
    t_start = time.time()
    
    # ========================================================================
    # PHYSICAL PARAMETERS (typical laboratory complex plasma)
    # ========================================================================
    # Dust particle parameters
    a_dust = 5.0e-6        # dust radius [m]
    rho_dust = 2000.0      # dust material density [kg/m^3]
    m_d = (4.0 / 3.0) * np.pi * a_dust**3 * rho_dust  # dust mass [kg]
    Q_num = 1850.0         # charge in units of elementary charge
    e_charge = 1.602176634e-19
    Q_eff = Q_num * e_charge  # effective dust charge [C]
    
    # Plasma parameters
    n_e = 5.0e14           # electron density [m^-3]
    T_e_eV = 2.0           # electron temperature [eV]
    T_e_K = T_e_eV * 11604.525  # convert to Kelvin
    T_dust = 300.0         # dust kinetic temperature [K]
    
    # Dust number density and Wigner-Seitz radius
    n_dust = 3.0e9         # dust number density [m^-3]
    a_ws = wigner_seitz_radius(n_dust)
    
    # Debye length and coupling parameter
    lambda_D = debye_length(n_e, T_e_K)
    gamma = coupling_parameter(Q_eff, n_dust, T_dust, lambda_D)
    kappa = a_ws / lambda_D
    
    print("\n[Physical Parameters]")
    print(f"  Dust mass           : {m_d:.3e} kg")
    print(f"  Dust charge         : {Q_num:.0f} e = {Q_eff:.3e} C")
    print(f"  Dust density        : {n_dust:.1e} m^-3")
    print(f"  Electron density    : {n_e:.1e} m^-3")
    print(f"  Electron temp       : {T_e_eV:.1f} eV = {T_e_K:.0f} K")
    print(f"  Dust temperature    : {T_dust:.0f} K")
    print(f"  Debye length        : {lambda_D*1e6:.2f} um")
    print(f"  Wigner-Seitz radius : {a_ws*1e6:.2f} um")
    print(f"  Kappa (a_WS/lD)     : {kappa:.3f}")
    print(f"  Coupling Gamma      : {gamma:.2f}")
    
    # ========================================================================
    # STEP 1: Lattice Construction and Geometric Analysis
    # ========================================================================
    print("\n" + "=" * 75)
    print("STEP 1: Lattice Construction & Geometric Analysis")
    print("=" * 75)
    
    n_rows, n_cols = 4, 4
    hex_pos = build_hexagonal_lattice(n_rows, n_cols, a_ws)
    sq_pos = build_square_lattice(n_rows, n_cols, a_ws)
    
    # Perturb z-coordinate for quasi-2D simulation
    hex_pos[:, 2] += (np.random.rand(len(hex_pos)) - 0.5) * a_ws * 0.02
    sq_pos[:, 2] += (np.random.rand(len(sq_pos)) - 0.5) * a_ws * 0.02
    
    print(f"  Hexagonal lattice   : {len(hex_pos)} particles")
    print(f"  Square lattice      : {len(sq_pos)} particles")
    
    # Signed distance to lattice plane (seed 150)
    if len(hex_pos) >= 5:
        p1, p2 = hex_pos[0, :2], hex_pos[1, :2]
        test_p = hex_pos[4, :2]  # non-collinear point
        sd = signed_point_line_distance(p1, p2, test_p)
        print(f"  Signed dist to plane: {sd:.3e} m")
    
    # Triangle area and barycentric interpolation (seed 1309)
    if len(hex_pos) >= 5:
        # Use non-collinear points from hexagonal lattice: (0,0), (a,0), (0.5a, sqrt(3)/2*a)
        v1, v2, v3 = hex_pos[0, :2], hex_pos[1, :2], hex_pos[4, :2]
        A_tri = triangle_area(v1, v2, v3)
        print(f"  Triangle area       : {abs(A_tri):.3e} m^2")
        
        if abs(A_tri) > 1e-20:
            query = np.array([[0.5*(v1[0]+v2[0]), 0.5*(v1[1]+v2[1])]])
            val_interp = barycentric_interpolate(query, v1, v2, v3, 1.0, 2.0, 3.0)
            print(f"  Barycentric interp  : {val_interp[0]:.3f}")
            
            rand_pts = uniform_in_triangle(v1, v2, v3, 10)
            print(f"  Uniform triangle pts: mean=({rand_pts[:,0].mean():.3e}, {rand_pts[:,1].mean():.3e})")
    
    # ========================================================================
    # STEP 2: Energy Minimization via Monte Carlo
    # ========================================================================
    print("\n" + "=" * 75)
    print("STEP 2: Energy Minimization (Monte Carlo)")
    print("=" * 75)
    
    E_hex_init = total_yukawa_energy(hex_pos, Q_eff, lambda_D)
    print(f"  Initial hex energy  : {E_hex_init:.6e} J")
    
    hex_eq, E_hex_eq, n_accept = monte_carlo_relax(
        hex_pos, Q_eff, lambda_D,
        n_steps=800, step_size=a_ws * 0.03, T=0.0
    )
    print(f"  Relaxed hex energy  : {E_hex_eq:.6e} J")
    print(f"  MC acceptance rate  : {n_accept / 800 * 100:.1f}%")
    
    # Configuration distance between initial and relaxed (seed 669)
    dist_cfg = configuration_distance(hex_pos, hex_eq)
    print(f"  Config distance     : {dist_cfg:.3e}")
    
    # ========================================================================
    # STEP 3: Dynamical Matrix and Phonon Analysis
    # ========================================================================
    print("\n" + "=" * 75)
    print("STEP 3: Dynamical Matrix & Phonon Analysis")
    print("=" * 75)
    
    Dmat = compute_dynamical_matrix(hex_eq, Q_eff, lambda_D, m_d)
    print(f"  Dynamical matrix    : {Dmat.shape}")
    
    # Matrix property analysis (seed 737)
    props = analyze_matrix_properties(Dmat)
    print(f"  Frobenius norm      : {props['frobenius_norm']:.3e}")
    print(f"  Is symmetric        : {props.get('is_symmetric', False)}")
    print(f"  Is diagonal dom.    : {props.get('is_diagonally_dominant', False)}")
    print(f"  Is SPD              : {props.get('is_spd', False)}")
    print(f"  Is normal           : {props.get('is_normal', False)}")
    
    # Linear solve test (seed 1098)
    b_test = np.random.randn(Dmat.shape[0])
    reg = 1e4 * np.eye(Dmat.shape[0])
    x_test = gaussian_solve(Dmat + reg, b_test)
    residual = np.linalg.norm((Dmat + reg) @ x_test - b_test)
    print(f"  Linear solve residual: {residual:.3e}")
    
    # Jacobi eigenvalue method (seed 604)
    V_jac, d_jac, it_num, rot_num = jacobi_eigenvalue(Dmat + reg, it_max=3000)
    print(f"  Jacobi iterations   : {it_num}")
    print(f"  Jacobi rotations    : {rot_num}")
    print(f"  Eigenvalue range    : [{np.min(d_jac):.3e}, {np.max(d_jac):.3e}]")
    
    err_jac = verify_eigensystem(Dmat + reg, V_jac, d_jac)
    print(f"  Eigensystem error   : {err_jac:.3e}")
    
    # Phonon frequencies
    omegas = compute_phonon_frequencies(Dmat)
    print(f"  Phonon frequencies  : min={omegas[0]:.3e}, max={omegas[-1]:.3e} rad/s")
    print(f"  Zero modes          : {np.sum(omegas < 1e-6)} (expected: 3)")
    
    # ========================================================================
    # STEP 4: Test Matrix Validation
    # ========================================================================
    print("\n" + "=" * 75)
    print("STEP 4: Test Matrix Validation")
    print("=" * 75)
    
    A_test, lambdas_true, Q_test = generate_test_symmetric_matrix(10, mean=1e5, std=1e4)
    V_test, d_test, it_test, rot_test = jacobi_eigenvalue(A_test, it_max=5000)
    err_test = verify_eigensystem(A_test, V_test, d_test)
    print(f"  Test matrix size    : 10x10")
    print(f"  Jacobi iterations   : {it_test}")
    print(f"  Eigensystem error   : {err_test:.3e}")
    print(f"  Eigenvalue match    : {np.allclose(np.sort(d_test), np.sort(lambdas_true), rtol=1e-3)}")
    
    # ========================================================================
    # STEP 5: Thermal Properties via Quadrature
    # ========================================================================
    print("\n" + "=" * 75)
    print("STEP 5: Thermal Averages & Quadrature")
    print("=" * 75)
    
    # Generalized Hermite integrals (seed 464)
    print("  Generalized Hermite integrals H(n, alpha=0):")
    for n in [0, 2, 4, 6, 8]:
        val = generalized_hermite_integral(n, alpha=0.0)
        print(f"    H({n}, 0) = {val:.6f}")
    
    # Wedge monomial integrals (seed 1406)
    print("  Unit wedge monomial integrals:")
    test_exps = [(0, 0, 0), (1, 0, 0), (0, 1, 1), (2, 0, 0), (1, 1, 2)]
    for e in test_exps:
        val = wedge_monomial_integral(e)
        print(f"    Wedge{e} = {val:.6f}")
    
    # Thermal displacement and thermodynamic quantities
    omega_nonzero = omegas[omegas > 1e-6]
    if len(omega_nonzero) > 0:
        u2_mode = thermal_average_displacement(omega_nonzero[0], T_dust, m_d)
        print(f"  <u^2> (fundamental) : {u2_mode:.3e} m^2")
        
        Z = partition_function_harmonic(omega_nonzero[:12], T_dust)
        print(f"  Partition Z (12 modes): {Z:.3e}")
        
        C_v = specific_heat_harmonic(omega_nonzero[:12], T_dust)
        print(f"  Specific heat C_v   : {C_v:.3e} J/K")
    
    # Gauss-Hermite quadrature test
    def test_func(x):
        return x**4
    quad_val = gauss_hermite_quadrature(test_func, n_nodes=16, alpha=0.0)
    exact_val = generalized_hermite_integral(4, 0.0)
    print(f"  GH quadrature x^4   : {quad_val:.6f} (exact: {exact_val:.6f})")
    
    # ========================================================================
    # STEP 6: Phase Transition Detection
    # ========================================================================
    print("\n" + "=" * 75)
    print("STEP 6: Phase Transition Detection")
    print("=" * 75)
    
    # Lindemann parameter from thermal phonon fluctuations
    # <u^2> = (1/N) * sum_k (hbar/(m_d*omega_k)) * (n_B + 1/2)
    if len(omega_nonzero) > 0:
        u2_thermal = np.sum([thermal_average_displacement(w, T_dust, m_d) 
                             for w in omega_nonzero]) / len(hex_eq)
        L = np.sqrt(max(u2_thermal, 0.0)) / a_ws
    else:
        L = 1.0
    print(f"  Lindemann L         : {L:.4f}")
    
    # Radial distribution function
    r_bins, g_r = radial_distribution_function(hex_eq, dr=a_ws * 0.05, r_max=a_ws * 5.0)
    # Normalize by average density for this finite cluster
    # Approximate cluster volume as sphere of radius 2*a_ws
    cluster_vol = (4.0/3.0) * np.pi * (2.0 * a_ws)**3
    avg_density = len(hex_eq) / cluster_vol
    g_r = g_r / avg_density
    valid_peaks = g_r[r_bins > a_ws * 0.5]
    if len(valid_peaks) > 0:
        peak_idx = np.argmax(valid_peaks) + np.searchsorted(r_bins, a_ws * 0.5)
        print(f"  g(r) first peak     : r={r_bins[peak_idx]*1e6:.1f} um, g={g_r[peak_idx]:.2f}")
    
    # Structure factor
    q_vec = np.array([2.0 * np.pi / a_ws, 0.0, 0.0])
    S_q = structure_factor(q_vec, hex_eq)
    print(f"  Structure factor S(q): {S_q:.3f}")
    
    # Bond orientational order
    psi6 = compute_bond_orientational_order(hex_eq[:, :2])
    print(f"  Bond order psi_6    : {psi6:.4f}")
    
    # Debye-Waller factor
    u_rms = np.sqrt(max(np.mean(np.sum((hex_eq - hex_pos)**2, axis=1)), 0.0))
    dwf = debye_waller_factor(q_vec, u_rms)
    print(f"  Debye-Waller factor : {dwf:.4f}")
    
    # Phase detection
    phase = detect_phase(gamma, L, gamma_c=170.0, lindemann_c=0.10)
    print(f"  Detected phase      : {phase}")
    
    # ========================================================================
    # STEP 7: Particle Dynamics Integration
    # ========================================================================
    print("\n" + "=" * 75)
    print("STEP 7: Particle Trajectory Integration")
    print("=" * 75)
    
    # Cliff RNG statistics (seed 1039)
    rng_vals = cliff_sequence(200, seed=0.314159)
    print(f"  Cliff RNG mean      : {np.mean(rng_vals):.4f}")
    print(f"  Cliff RNG std       : {np.std(rng_vals):.4f}")
    
    # Setup dynamics
    N_part = len(hex_eq)
    y0 = np.zeros(6 * N_part)
    y0[:3*N_part] = hex_eq.flatten()
    # Small thermal velocities
    v_thermal = np.sqrt(3.0 * 1.380649e-23 * T_dust / m_d)
    y0[3*N_part:] = (np.random.rand(3*N_part) - 0.5) * v_thermal * 0.1
    
    dyn_params = {
        'N': N_part,
        'm_d': m_d,
        'Q_eff': Q_eff,
        'lambda_D': lambda_D,
        'nu_n': 15.0,         # neutral collision frequency [Hz]
        'g': 9.81,            # gravitational acceleration [m/s^2]
        'F_ion_base': 5e-13,  # base ion drag [N]
        'z_eq': 0.0
    }
    
    # Short-time integration
    t_end = 5e-5  # 50 microseconds
    dt = 1e-7     # 0.1 microsecond timestep
    y_final = integrate_trajectories(y0, t_end, dt, dyn_params)
    
    final_pos = y_final[:3*N_part].reshape((-1, 3))
    final_vel = y_final[3*N_part:].reshape((-1, 3))
    
    msd = compute_mean_square_displacement(hex_eq, final_pos)
    T_kin = compute_kinetic_temperature(final_vel, m_d)
    print(f"  Integration time    : {t_end*1e6:.1f} us")
    print(f"  Mean sq. displ.     : {msd:.3e} m^2")
    print(f"  Kinetic temperature : {T_kin:.1f} K")
    
    # ========================================================================
    # STEP 8: Numerical Methods Verification
    # ========================================================================
    print("\n" + "=" * 75)
    print("STEP 8: Numerical Methods Verification")
    print("=" * 75)
    
    # Centered finite difference on Yukawa potential (seed 281)
    def pot_test(r):
        return yukawa_potential(r, Q_eff, lambda_D)
    
    r_test = a_ws
    d2U_fd = diff2_center(pot_test, r_test, h=a_ws * 0.0005)
    
    # Analytic second derivative for comparison
    # TODO: Compute the exact analytic second derivative d2U/dr2 of the Yukawa potential.
    # HINT: For U(r) = (Q^2/4*pi*eps0*r) * exp(-r/lambda_D), derive d2U/dr2 analytically.
    #       The result can be expressed as U(r) * [2/r^2 + 2/(lambda_D*r) + 1/lambda_D^2].
    #       You need to compute eps0, U0, and d2U_exact.
    raise NotImplementedError("Hole 3: Analytic second derivative validation in main.py is not implemented.")
    
    rel_err = abs(d2U_fd - d2U_exact) / abs(d2U_exact) if abs(d2U_exact) > 1e-30 else 0.0
    print(f"  d2U/dr2 (FD)        : {d2U_fd:.3e}")
    print(f"  d2U/dr2 (exact)     : {d2U_exact:.3e}")
    print(f"  Relative error      : {rel_err:.3e}")
    
    # Spigot algorithm for e (seed 323)
    e_digits = e_spigot(25)
    e_str = ''.join(str(d) for d in e_digits)
    e_spigot_val = float(e_str[0] + '.' + e_str[1:])
    print(f"  e (spigot, 25 digs) : {e_spigot_val:.20f}")
    print(f"  e (NumPy)           : {np.e:.20f}")
    print(f"  e difference        : {abs(e_spigot_val - np.e):.3e}")
    
    # ========================================================================
    # STEP 9: Sparse Matrix Conversion
    # ========================================================================
    print("\n" + "=" * 75)
    print("STEP 9: Sparse Matrix Conversion")
    print("=" * 75)
    
    # Build sparse interaction matrix (seed 1159)
    rows, cols, vals = [], [], []
    N = len(hex_eq)
    cutoff = 3.0 * lambda_D
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(hex_eq[i] - hex_eq[j])
            if r < cutoff:
                val = yukawa_potential(r, Q_eff, lambda_D)
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([val, val])
    
    sparse_dense = sparse_triplet_to_dense(rows, cols, vals, N, N)
    nnz = len(vals)
    print(f"  Sparse matrix size  : {N}x{N}")
    print(f"  Non-zero elements   : {nnz}")
    print(f"  Sparsity            : {nnz / (N*N) * 100:.1f}%")
    print(f"  Frobenius norm      : {np.linalg.norm(sparse_dense):.3e}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 75)
    print("SIMULATION SUMMARY")
    print("=" * 75)
    print(f"  Particles           : {N}")
    print(f"  Coupling Gamma      : {gamma:.2f}")
    print(f"  Kappa               : {kappa:.3f}")
    print(f"  Phase               : {phase}")
    print(f"  Lindemann L         : {L:.4f}")
    print(f"  Bond order psi_6    : {psi6:.4f}")
    
    if phase == "CRYSTALLINE":
        print("  >> CRYSTALLINE: Plasma crystal formed (Gamma > Gamma_c, L < L_c)")
    elif phase == "LIQUID":
        print("  >> LIQUID: Intermediate coupling, short-range order")
    else:
        print("  >> GASEOUS: Weak coupling, disordered state")
    
    t_elapsed = time.time() - t_start
    print(f"\n  Execution time      : {t_elapsed:.2f} s")
    print("=" * 75)


if __name__ == "__main__":
    np.random.seed(42)
    run_simulation()
