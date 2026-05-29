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
    # d2U/dr2 = U(r) * [2/r^2 + 2/(lambda_D*r) + 1/lambda_D^2]
    eps0 = 8.854187817e-12
    U0 = yukawa_potential(r_test, Q_eff, lambda_D)
    d2U_exact = U0 * (2.0/r_test**2 + 2.0/(r_test*lambda_D) + 1.0/lambda_D**2)
    
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

# ================================================================
# 测试用例（34个，assert模式，涉及随机值均使用固定种子）
# ================================================================

from linear_algebra import frobenius_norm, is_symmetric
from yukawa_physics import yukawa_force_vector
from scipy.special import gamma as gamma_func

# ---- TC01: triangle_area returns correct signed area for right triangle ----
A_tc01 = triangle_area([0, 0], [1, 0], [0, 1])
assert abs(A_tc01 - 0.5) < 1e-10, '[TC01] triangle_area right triangle FAILED'

# ---- TC02: signed_point_line_distance is zero for point on line ----
d_tc02 = signed_point_line_distance([0, 0], [1, 0], [0.5, 0])
assert abs(d_tc02) < 1e-10, '[TC02] signed_point_line_distance on line FAILED'

# ---- TC03: build_hexagonal_lattice produces expected number of particles ----
hex_tc03 = build_hexagonal_lattice(3, 4, 1e-6)
assert hex_tc03.shape == (12, 3), '[TC03] build_hexagonal_lattice shape FAILED'

# ---- TC04: build_square_lattice produces expected number of particles ----
sq_tc04 = build_square_lattice(3, 4, 1e-6)
assert sq_tc04.shape == (12, 3), '[TC04] build_square_lattice shape FAILED'

# ---- TC05: yukawa_potential is positive for positive charges ----
U_tc05 = yukawa_potential(1e-5, 1e-15, 1e-4)
assert U_tc05 > 0.0, '[TC05] yukawa_potential positivity FAILED'

# ---- TC06: yukawa_potential returns zero at r <= 0 ----
U_tc06 = yukawa_potential(0.0, 1e-15, 1e-4)
assert U_tc06 == 0.0, '[TC06] yukawa_potential zero at r=0 FAILED'

# ---- TC07: debye_length matches analytical formula ----
lambda_D_tc07 = debye_length(1e15, 11604.525)
expected_tc07 = np.sqrt(8.854187817e-12 * 1.380649e-23 * 11604.525 / (1e15 * (1.602176634e-19)**2))
assert abs(lambda_D_tc07 - expected_tc07) < 1e-15 * max(abs(expected_tc07), 1.0), '[TC07] debye_length formula FAILED'

# ---- TC08: wigner_seitz_radius matches analytical formula ----
a_ws_tc08 = wigner_seitz_radius(1e9)
expected_tc08 = (3.0 / (4.0 * np.pi * 1e9)) ** (1.0 / 3.0)
assert abs(a_ws_tc08 - expected_tc08) < 1e-20, '[TC08] wigner_seitz_radius formula FAILED'

# ---- TC09: coupling_parameter returns finite positive scalar ----
Q_test = 1.602176634e-19 * 1000
n_test = 1e9
T_test = 300.0
lD_test = 1e-4
gamma_tc09 = coupling_parameter(Q_test, n_test, T_test, lD_test)
assert np.isfinite(gamma_tc09) and gamma_tc09 > 0.0, '[TC09] coupling_parameter finite positive FAILED'

# ---- TC10: diff2_center recovers exact second derivative for quadratic ----
d2_tc10 = diff2_center(lambda x: 3.0*x**2 + 2.0*x + 1.0, 1.0, h=1e-4)
assert abs(d2_tc10 - 6.0) < 1e-6, '[TC10] diff2_center quadratic FAILED'

# ---- TC11: e_spigot first digits match known expansion of e ----
digits_tc11 = e_spigot(10)
expected_tc11 = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4]
assert digits_tc11 == expected_tc11, '[TC11] e_spigot digits FAILED'

# ---- TC12: gaussian_solve solves simple 2x2 system exactly ----
A_tc12 = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=float)
b_tc12 = np.array([5.0, 8.0], dtype=float)
x_tc12 = gaussian_solve(A_tc12, b_tc12)
expected_x_tc12 = np.linalg.solve(A_tc12, b_tc12)
assert np.allclose(x_tc12, expected_x_tc12, rtol=1e-10), '[TC12] gaussian_solve 2x2 FAILED'

# ---- TC13: frobenius_norm of identity matrix equals sqrt(n) ----
I_tc13 = np.eye(5)
assert abs(frobenius_norm(I_tc13) - np.sqrt(5)) < 1e-12, '[TC13] frobenius_norm identity FAILED'

# ---- TC14: is_symmetric correctly identifies symmetric matrix ----
sym_tc14 = np.array([[1.0, 2.0], [2.0, 3.0]])
assert is_symmetric(sym_tc14) == True, '[TC14] is_symmetric true case FAILED'

# ---- TC15: is_symmetric correctly rejects non-symmetric matrix ----
non_sym_tc15 = np.array([[1.0, 2.0], [3.0, 4.0]])
assert is_symmetric(non_sym_tc15) == False, '[TC15] is_symmetric false case FAILED'

# ---- TC16: jacobi_eigenvalue eigenvalues match numpy for test matrix ----
np.random.seed(42)
A_tc16, lam_true_tc16, _ = generate_test_symmetric_matrix(6, mean=10.0, std=1.0)
V_tc16, d_tc16, _, _ = jacobi_eigenvalue(A_tc16, it_max=5000)
assert np.allclose(np.sort(d_tc16), np.sort(lam_true_tc16), rtol=1e-3), '[TC16] jacobi eigenvalue match FAILED'

# ---- TC17: verify_eigensystem residual is small for test matrix ----
err_tc17 = verify_eigensystem(A_tc16, V_tc16, d_tc16)
assert err_tc17 < 1e-6, '[TC17] verify_eigensystem residual FAILED'

# ---- TC18: compute_phonon_frequencies returns non-negative values ----
hex_tc18 = build_hexagonal_lattice(3, 3, 1e-6)
Q_tc18 = 1.602176634e-19 * 500
lD_tc18 = 5e-5
m_tc18 = 1e-15
Dmat_tc18 = compute_dynamical_matrix(hex_tc18, Q_tc18, lD_tc18, m_tc18)
omega_tc18 = compute_phonon_frequencies(Dmat_tc18)
assert np.all(omega_tc18 >= 0.0), '[TC18] phonon frequencies non-negative FAILED'

# ---- TC19: generalized_hermite_integral zero for odd exponent ----
H_odd_tc19 = generalized_hermite_integral(3, 0.0)
assert H_odd_tc19 == 0.0, '[TC19] generalized_hermite_integral odd FAILED'

# ---- TC20: generalized_hermite_integral exact for even exponent alpha=0 ----
H_4_tc20 = generalized_hermite_integral(4, 0.0)
expected_H4_tc20 = gamma_func(2.5)
assert abs(H_4_tc20 - expected_H4_tc20) < 1e-12, '[TC20] generalized_hermite_integral even FAILED'

# ---- TC21: wedge_monomial_integral zero for odd z exponent ----
W_odd_tc21 = wedge_monomial_integral((0, 0, 1))
assert W_odd_tc21 == 0.0, '[TC21] wedge_monomial_integral odd z FAILED'

# ---- TC22: thermal_average_displacement finite at zero temperature ----
u2_tc22 = thermal_average_displacement(1e6, 0.0, m_d=1e-15)
assert np.isfinite(u2_tc22) and u2_tc22 > 0.0, '[TC22] thermal displacement zero T FAILED'

# ---- TC23: structure_factor equals N for q=0 ----
pos_tc23 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
S_tc23 = structure_factor(np.array([0.0, 0.0, 0.0]), pos_tc23)
assert abs(S_tc23 - 2.0) < 1e-10, '[TC23] structure_factor q=0 FAILED'

# ---- TC24: detect_phase returns CRYSTALLINE for strong coupling low L ----
phase_tc24 = detect_phase(500.0, 0.05)
assert phase_tc24 == "CRYSTALLINE", '[TC24] detect_phase crystalline FAILED'

# ---- TC25: detect_phase returns GASEOUS for weak coupling ----
phase_tc25 = detect_phase(50.0, 0.25)
assert phase_tc25 == "GASEOUS", '[TC25] detect_phase gaseous FAILED'

# ---- TC26: debye_waller_factor in range (0,1] for finite u_rms ----
dwf_tc26 = debye_waller_factor(np.array([1e7, 0, 0]), 1e-9)
assert 0.0 < dwf_tc26 <= 1.0, '[TC26] debye_waller_factor range FAILED'

# ---- TC27: compute_mean_square_displacement zero for identical positions ----
pos_tc27 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
msd_tc27 = compute_mean_square_displacement(pos_tc27, pos_tc27)
assert abs(msd_tc27) < 1e-20, '[TC27] msd zero FAILED'

# ---- TC28: compute_kinetic_temperature zero for zero velocities ----
vel_tc28 = np.zeros((5, 3))
T_kin_tc28 = compute_kinetic_temperature(vel_tc28, 1e-15)
assert abs(T_kin_tc28) < 1e-20, '[TC28] kinetic temperature zero FAILED'

# ---- TC29: cliff_sequence reproducible with fixed seed ----
np.random.seed(42)
seq1_tc29 = cliff_sequence(10, seed=0.3)
seq2_tc29 = cliff_sequence(10, seed=0.3)
assert np.allclose(seq1_tc29, seq2_tc29, rtol=1e-14), '[TC29] cliff_sequence reproducibility FAILED'

# ---- TC30: total_yukawa_energy zero for single particle ----
pos_tc30 = np.array([[0.0, 0.0, 0.0]])
E_tc30 = total_yukawa_energy(pos_tc30, 1e-15, 1e-4)
assert E_tc30 == 0.0, '[TC30] total energy single particle FAILED'

# ---- TC31: configuration_distance zero for identical configs ----
np.random.seed(42)
cfg_tc31 = np.random.rand(5, 3)
dist_tc31 = configuration_distance(cfg_tc31, cfg_tc31.copy())
assert abs(dist_tc31) < 1e-10, '[TC31] config distance identical FAILED'

# ---- TC32: sparse_triplet_to_dense correct shape and values ----
dense_tc32 = sparse_triplet_to_dense([0, 1], [1, 0], [2.0, 3.0], 2, 2)
assert dense_tc32.shape == (2, 2), '[TC32] sparse shape FAILED'
assert dense_tc32[0, 1] == 2.0 and dense_tc32[1, 0] == 3.0, '[TC32] sparse values FAILED'

# ---- TC33: yukawa_force_vector zero for zero separation ----
fvec_tc33 = yukawa_force_vector(np.array([0.0, 0.0, 0.0]), 1e-15, 1e-4)
assert np.allclose(fvec_tc33, np.zeros(3)), '[TC33] yukawa_force_vector zero separation FAILED'

# ---- TC34: lindemann_parameter zero for perfect match ----
pos_tc34 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
L_tc34 = lindemann_parameter(pos_tc34, pos_tc34, 1.0)
assert abs(L_tc34) < 1e-15, '[TC34] Lindemann zero FAILED'

print('\n全部 34 个测试通过!\n')
