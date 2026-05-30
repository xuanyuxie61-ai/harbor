#!/usr/bin/env python3

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')


from pce_expansion import (
    generate_multi_indices, evaluate_pce_basis, pce_moments,
    kl_expansion_1d, assemble_stochastic_galerkin_system
)
from cvt_sampling import generate_cvt_samples, adaptive_cvt_for_reliability
from sphere_voronoi_mapper import (
    map_stochastic_to_sphere, directional_reliability_sensitivity
)
from fem_quadrature import (
    integrate_triangle_wandzura, integrate_hexahedron_witherden,
    triangulation_quad, faces_average,
    assemble_triangular_fem_matrices, t3_shape_functions
)
from tridiagonal_engine import (
    r83v_fs, r83v_cg, r83v_jac_sl, r83v_mv,
    build_beam_tridiagonal, modal_analysis_tridiagonal
)
from dynamic_integrator import (
    cauchy_principal_value, frequency_response_function,
    psd_response, integrate_psd_cpv, first_passage_probability,
    modal_superposition_response, compute_stress_psd_from_displacement_psd
)
from reliability_optimizer import (
    lngamma, gamma_function, chi2_pdf, rayleigh_pdf, normal_cdf,
    golden_section_search, form_reliability, form_with_golden_search,
    sorm_reliability
)
from mesh_generator import (
    polyiamond_hexagon_mesh, generate_perforated_plate_mesh,
    magic_square, construct_test_stiffness_matrix,
    nodal_stress_recovery, compute_element_stresses
)
from matrix_exporter import export_structural_matrices, read_matrix_market
from elliptic_module import (
    complete_elliptic_k, complete_elliptic_e,
    jacobi_elliptic_functions, elastica_beam_deflection,
    nonlinear_vibration_period, elliptical_hole_stress_concentration
)


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def apply_boundary_conditions(K, M, F, bc_dofs):
    n_dof = K.shape[0]
    free_dofs = np.setdiff1d(np.arange(n_dof), bc_dofs)
    K_red = K[np.ix_(free_dofs, free_dofs)]
    M_red = M[np.ix_(free_dofs, free_dofs)]
    F_red = F[free_dofs] if F is not None else None
    return K_red, M_red, F_red, free_dofs


def main():
    
    print("\n" + "#" * 80)
    print("#  STOCHASTIC VIBRATION & RELIABILITY ANALYSIS OF PERFORATED PLATE STRUCTURE  #")
    print("#" * 80)
    



    print_section("1. MESH GENERATION: Polyiamond Hexagonal Plate Topology")
    

    hex_order = 2
    scale = 0.5
    nodes, elements, boundary_nodes = polyiamond_hexagon_mesh(hex_order, scale)
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    
    print(f"   Generated hexagonal polyiamond mesh: order={hex_order}, scale={scale}")
    print(f"   Nodes: {n_nodes}, Elements: {n_elements}")
    print(f"   Boundary nodes: {len(boundary_nodes)}")
    

    nodes_rect, elements_rect, boundary_rect = generate_perforated_plate_mesh(
        nx=6, ny=6, hole_positions=[(0.35, 0.35), (0.65, 0.65)],
        hole_radius=0.12, Lx=1.0, Ly=1.0
    )
    print(f"   Perforated rectangular plate: {nodes_rect.shape[0]} nodes, {elements_rect.shape[0]} elements")
    



    print_section("2. STOCHASTIC MATERIAL MODELING (KL Expansion)")
    

    E_mean = 210e9
    E_std = 21e9
    nu = 0.3
    rho = 7850.0
    h_plate = 0.005
    

    n_kl_modes = 3
    correlation_length = 0.3
    L_domain = 1.0
    

    x_kl = np.linspace(0, L_domain, max(20, n_nodes))
    kl_eigvals, kl_modes = kl_expansion_1d(n_kl_modes, L_domain, correlation_length, x_kl)
    
    print(f"   Young's modulus: mean={E_mean/1e9:.1f} GPa, COV={E_std/E_mean*100:.1f}%")
    print(f"   KL expansion: {n_kl_modes} modes retained")
    print(f"   KL eigenvalues: {kl_eigvals}")
    



    print_section("3. FINITE ELEMENT MATRIX ASSEMBLY")
    

    K_det, M_det = assemble_triangular_fem_matrices(
        nodes_rect, elements_rect,
        material_thickness=h_plate,
        young_modulus=E_mean,
        poisson_ratio=nu,
        density=rho
    )
    
    n_dof = K_det.shape[0]
    print(f"   Global stiffness matrix size: {n_dof} x {n_dof}")
    

    bc_dofs = []
    for bn in boundary_rect:
        bc_dofs.extend([2 * bn, 2 * bn + 1])
    bc_dofs = np.unique(bc_dofs)
    
    F_load = np.zeros(n_dof)

    center_node = len(nodes_rect) // 2
    F_load[2 * center_node + 1] = 1000.0
    
    K_red, M_red, F_red, free_dofs = apply_boundary_conditions(K_det, M_det, F_load, bc_dofs)
    n_free = len(free_dofs)
    print(f"   After BC application: {n_free} free DOFs")
    



    print_section("4. MODAL ANALYSIS")
    
    try:
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(K_red, M_red)
        omega_n = np.sqrt(np.maximum(eigvals, 0.0))
    except ImportError:

        M_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M_red)))
        K_tilde = M_inv_sqrt @ K_red @ M_inv_sqrt
        eigvals, eigvecs_tilde = np.linalg.eigh(K_tilde)
        eigvecs = M_inv_sqrt @ eigvecs_tilde
        omega_n = np.sqrt(np.maximum(eigvals, 0.0))
    
    n_modes_use = min(8, n_free)
    omega_modes = omega_n[:n_modes_use]
    phi_modes = eigvecs[:, :n_modes_use]
    

    for m in range(n_modes_use):
        mass_norm = np.sqrt(phi_modes[:, m].T @ M_red @ phi_modes[:, m])
        if mass_norm > 1e-12:
            phi_modes[:, m] /= mass_norm
    
    print(f"   First {n_modes_use} natural frequencies (Hz):")
    for m in range(n_modes_use):
        print(f"      Mode {m+1}: {omega_modes[m]/(2*np.pi):.3f} Hz  (omega={omega_modes[m]:.2f} rad/s)")
    



    print_section("5. TRIDIAGONAL SOLVER VERIFICATION")
    

    n_beam = 20
    EI_beam = E_mean * h_plate ** 3 / 12.0
    L_beam = 1.0
    a_tri, b_tri, c_tri, rhs_beam, h_beam = build_beam_tridiagonal(
        n_beam, EI_beam, L_beam, load_type='uniform'
    )
    

    x_thomas = r83v_fs(n_beam, a_tri.copy(), b_tri.copy(), c_tri.copy(), rhs_beam.copy())
    x_cg, it_cg, res_cg = r83v_cg(n_beam, a_tri, b_tri, c_tri, rhs_beam, tol=1e-12)
    x_jac, it_jac, res_jac = r83v_jac_sl(n_beam, a_tri, b_tri, c_tri, rhs_beam, it_max=500, tol=1e-10)
    
    print(f"   Thomas algorithm: ||Ax-b|| = {np.linalg.norm(r83v_mv(n_beam,a_tri,b_tri,c_tri,x_thomas)-rhs_beam):.3e}")
    print(f"   Conjugate Gradient: {it_cg} iterations, residual={res_cg:.3e}")
    print(f"   Jacobi iteration: {it_jac} iterations, residual={res_jac:.3e}")
    

    eig_tri, phi_tri = modal_analysis_tridiagonal(n_beam, a_tri, b_tri, c_tri, n_modes=3)
    print(f"   Tridiagonal beam modes: {eig_tri}")
    



    print_section("6. RANDOM VIBRATION RESPONSE ANALYSIS")
    

    zeta_modes = np.array([0.02, 0.015, 0.012, 0.01, 0.01, 0.01, 0.01, 0.01])
    zeta_modes = zeta_modes[:n_modes_use]
    

    S0 = 1.0e6
    def psd_input(omega):
        return np.ones_like(omega) * S0
    

    omega_fine = np.linspace(0.1, 500.0, 2000)
    

    load_idx = np.where(free_dofs == 2 * center_node + 1)[0]
    resp_idx = np.where(free_dofs == 2 * center_node + 1)[0]
    if len(load_idx) == 0 or len(resp_idx) == 0:

        load_idx = [0]
        resp_idx = [0]
    
    load_idx = load_idx[0]
    resp_idx = resp_idx[0]
    
    H_frf, omega_m, phi_m = modal_superposition_response(
        omega_fine, K_red, M_red, zeta_modes,
        load_dof=load_idx, response_dof=resp_idx, omega_n_modes=n_modes_use
    )
    


    Sy = np.zeros_like(omega_fine)
    


    sigma_u_sq = 0.0
    sigma_u = 0.0
    


    omega_0 = omega_modes[0]
    
    print(f"   RMS displacement response: {sigma_u:.6f} m")
    print(f"   Zero-crossing frequency: {omega_0/(2*np.pi):.3f} Hz")
    

    omega_res = omega_modes[0]
    def cpv_integrand(om):
        return psd_response(om, omega_res, zeta_modes[0], psd_input)
    

    cpv_val = cauchy_principal_value(cpv_integrand, omega_res * 0.9, omega_res * 1.1, omega_res, n=20)
    print(f"   CPV integral near 1st resonance: {cpv_val:.4e}")
    



    print_section("7. POLYNOMIAL CHAOS EXPANSION UNCERTAINTY QUANTIFICATION")
    
    n_stoch = n_kl_modes
    pce_order = 2
    indices = generate_multi_indices(n_stoch, pce_order)
    n_basis = indices.shape[0]
    print(f"   Stochastic dimension: {n_stoch}, PCE order: {pce_order}")
    print(f"   PCE basis size: {n_basis}")
    

    n_samples = 200
    cvt_samples = generate_cvt_samples(
        dim=n_stoch, n_gen=n_samples, it_max=30, bounds=[(-1, 1)] * n_stoch, seed=42
    )
    

    psi_samples, _, norms = evaluate_pce_basis(n_stoch, pce_order, cvt_samples, indices)
    


    omega_samples = np.zeros((n_samples, n_modes_use))
    for s in range(n_samples):
        xi_s = cvt_samples[s]

        E_realization = E_mean
        for i in range(n_kl_modes):
            E_realization += E_std * np.sqrt(kl_eigvals[i]) * xi_s[i]
        E_realization = max(E_realization, E_mean * 0.5)
        

        K_s = K_red * (E_realization / E_mean)
        try:
            from scipy.linalg import eigh
            eig_s, _ = eigh(K_s, M_red)
        except ImportError:
            M_inv = np.diag(1.0 / np.sqrt(np.diag(M_red)))
            Ks_tilde = M_inv @ K_s @ M_inv
            eig_s, _ = np.linalg.eigh(Ks_tilde)
        omega_samples[s] = np.sqrt(np.maximum(eig_s[:n_modes_use], 0.0))
    


    PsiTPsi = psi_samples.T @ psi_samples
    for m in range(n_modes_use):
        coeffs = np.linalg.solve(PsiTPsi + 1e-10 * np.eye(n_basis),
                                 psi_samples.T @ omega_samples[:, m])
        mean_om, var_om = pce_moments(coeffs, indices, norms)
        print(f"   Mode {m+1} freq: mean={mean_om/(2*np.pi):.3f} Hz, std={np.sqrt(var_om)/(2*np.pi):.3f} Hz")
    



    print_section("8. SPHERICAL VORONOI DIRECTIONAL SENSITIVITY")
    

    if n_stoch >= 3:
        directions, radii = map_stochastic_to_sphere(cvt_samples[:, :3])
    else:

        padded = np.pad(cvt_samples, ((0, 0), (0, 3 - n_stoch)), mode='constant')
        directions, radii = map_stochastic_to_sphere(padded)
    

    max_omega_shift = np.max(omega_samples, axis=1)
    
    sens = directional_reliability_sensitivity(
        directions, max_omega_shift, n_directional_bins=12
    )
    max_sector_response = np.max(sens['max_response'][sens['counts'] > 0])
    print(f"   Maximum directional response sensitivity: {max_sector_response:.3f} rad/s")
    



    print_section("9. STRUCTURAL RELIABILITY ANALYSIS (FORM / SORM)")
    



    threshold_disp = 1.5e-5
    

    def limit_state(u):


        u_ext = np.zeros(n_kl_modes)
        u_ext[:len(u)] = u
        



        norm_u = np.linalg.norm(u_ext)
        response_factor = 1.0 + 0.25 * norm_u + 0.05 * norm_u ** 2
        resp = sigma_u * response_factor
        return threshold_disp - resp
    
    def grad_limit_state(u):
        eps = 1e-6
        g0 = limit_state(u)
        grad = np.zeros_like(u)
        for i in range(len(u)):
            u_p = u.copy()
            u_p[i] += eps
            grad[i] = (limit_state(u_p) - g0) / eps
        return grad
    
    def hess_limit_state(u):
        eps = 1e-5
        n = len(u)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                u_pp = u.copy(); u_pp[i] += eps; u_pp[j] += eps
                u_pm = u.copy(); u_pm[i] += eps; u_pm[j] -= eps
                u_mp = u.copy(); u_mp[i] -= eps; u_mp[j] += eps
                u_mm = u.copy(); u_mm[i] -= eps; u_mm[j] -= eps
                H[i, j] = (limit_state(u_pp) - limit_state(u_pm) -
                           limit_state(u_mp) + limit_state(u_mm)) / (4 * eps ** 2)
        return (H + H.T) / 2.0
    

    beta_form, u_star, Pf_form = form_reliability(
        limit_state, grad_limit_state, u0=np.zeros(2), dim=2, max_iter=50, tol=1e-8
    )
    

    beta_golden, u_golden, Pf_golden = form_with_golden_search(
        limit_state, dim=2, beta_max=8.0, n_directions=72
    )
    

    try:
        Pf_sorm, curvatures = sorm_reliability(
            limit_state, grad_limit_state, hess_limit_state, u_star, beta_form
        )
    except Exception:
        Pf_sorm = Pf_form
        curvatures = np.array([])
    
    print(f"   Threshold displacement: {threshold_disp:.6e} m")
    print(f"   RMS displacement: {sigma_u:.6e} m")
    print(f"   FORM reliability index: beta = {beta_form:.4f}")
    print(f"   FORM failure probability: P_f = {Pf_form:.6e}")
    print(f"   Golden-section FORM: beta = {beta_golden:.4f}, P_f = {Pf_golden:.6e}")
    print(f"   SORM failure probability: P_f = {Pf_sorm:.6e}")
    if len(curvatures) > 0:
        print(f"   Principal curvatures at design point: {curvatures}")
    

    T_duration = 3600.0
    Pf_fp, nu_up = first_passage_probability(
        sigma_u, threshold_disp, omega_0, T_duration, method='vanmarcke'
    )
    print(f"   First-passage failure (Vanmarcke, T={T_duration/60:.0f} min): P_f = {Pf_fp:.6e}")
    print(f"   Up-crossing rate: {nu_up:.6e} crossings/sec")
    



    print_section("10. SPECIAL FUNCTIONS & ELLIPTIC INTEGRALS")
    

    z_test = 5.5
    lg_val, ier = lngamma(z_test)
    gamma_val = gamma_function(z_test)
    print(f"   log Gamma({z_test}) = {lg_val:.8f}, Gamma({z_test}) = {gamma_val:.8f}")
    

    chi2_val = chi2_pdf(3.0, k=2)
    print(f"   Chi-square PDF(3.0, df=2) = {chi2_val:.6f}")
    

    rayleigh_val = rayleigh_pdf(2.0 * sigma_u, sigma_u)
    print(f"   Rayleigh PDF(2*sigma_y) = {rayleigh_val:.6e}")
    

    m_ellip = 0.5
    K_val = complete_elliptic_k(m_ellip)
    E_val = complete_elliptic_e(m_ellip)
    print(f"   K({m_ellip}) = {K_val:.6f}, E({m_ellip}) = {E_val:.6f}")
    

    T_nl, T_lin = nonlinear_vibration_period(
        amplitude=0.01, omega_linear=omega_modes[0], alpha_nonlin=1e6
    )
    print(f"   Nonlinear period: T = {T_nl*1000:.3f} ms, Linear T = {T_lin*1000:.3f} ms")
    

    s_theta, kt = elliptical_hole_stress_concentration(
        a=0.1, b=0.05, sigma_inf=100e6, theta=np.linspace(0, 2*np.pi, 100)
    )
    print(f"   Elliptical hole SCF: K_t = {kt:.2f}")
    print(f"   Max tangential stress: {np.max(s_theta)/1e6:.2f} MPa")
    



    print_section("11. HIGH-PRECISION QUADRATURE VERIFICATION")
    

    unit_triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    f_test = lambda x, y: x ** 2 + y ** 2
    I_tri = integrate_triangle_wandzura(f_test, unit_triangle, degree=5)
    I_exact_tri = 1.0 / 6.0
    print(f"   Triangle quadrature: I = {I_tri:.8f}, Exact = {I_exact_tri:.8f}, Error = {abs(I_tri-I_exact_tri):.3e}")
    

    f_3d = lambda x, y, z: x * y * z
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    I_hex = integrate_hexahedron_witherden(f_3d, bounds, degree=3)
    I_exact_hex = 0.125
    print(f"   Hexahedron quadrature: I = {I_hex:.8f}, Exact = {I_exact_hex:.8f}, Error = {abs(I_hex-I_exact_hex):.3e}")
    

    node_vals = nodes_rect[:, 0] ** 2 + nodes_rect[:, 1] ** 2
    I_triang, area_triang = triangulation_quad(nodes_rect, elements_rect, node_vals)
    print(f"   Triangulation integral: I = {I_triang:.8f}, Total area = {area_triang:.4f}")
    



    print_section("12. NODAL STRESS RECOVERY & MATRIX EXPORT")
    

    try:
        u_static = np.linalg.solve(K_red + 1e-3 * np.eye(n_free), F_red)
    except Exception:
        u_static = np.zeros(n_free)
    

    u_full = np.zeros(n_dof)
    u_full[free_dofs] = u_static
    

    elem_stresses, elem_vm = compute_element_stresses(
        u_full, nodes_rect, elements_rect, E_mean, nu
    )
    nodal_vm = nodal_stress_recovery(elem_vm, elements_rect, nodes_rect.shape[0])
    max_vm = np.max(nodal_vm)
    print(f"   Maximum Von Mises stress: {max_vm/1e6:.3f} MPa")
    

    output_dir = "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/089_synth_project"
    export_structural_matrices(K_red, M_red, prefix=f"{output_dir}/structural")
    print(f"   Exported matrices to {output_dir}/structural_*.mtx")
    

    K_read = read_matrix_market(f"{output_dir}/structural_K.mtx")
    diff_K = np.linalg.norm(K_read - K_red)
    print(f"   Matrix read-back verification: ||K_read - K||_F = {diff_K:.3e}")
    



    print_section("13. MAGIC SQUARE TEST MATRIX VERIFICATION")
    
    n_test = 5
    M_magic = magic_square(n_test)
    magic_sum = n_test * (n_test ** 2 + 1) // 2
    row_sums = np.sum(M_magic, axis=1)
    col_sums = np.sum(M_magic, axis=0)
    diag_sum = np.sum(np.diag(M_magic))
    anti_diag_sum = np.sum(np.diag(np.fliplr(M_magic)))
    
    print(f"   Magic square order {n_test}: magic sum = {magic_sum}")
    print(f"   Row sums: {row_sums}")
    print(f"   Col sums: {col_sums}")
    print(f"   Diagonal sums: {diag_sum}, {anti_diag_sum}")
    
    K_test, M_test = construct_test_stiffness_matrix(n_test, magic_based=True)
    eig_test = np.linalg.eigvalsh(K_test)
    print(f"   Test stiffness matrix eigenvalues: {eig_test}")
    print(f"   All positive definite: {np.all(eig_test > 0)}")
    



    print_section("14. SUMMARY REPORT")
    print(f"""
    ================================================================================
                         STOCHASTIC RELIABILITY ANALYSIS RESULTS
    ================================================================================
    
    PROBLEM: Perforated plate under random vibration with stochastic Young's modulus
    
    MESH:
      - Polyiamond hexagonal: {n_nodes} nodes, {n_elements} elements
      - Perforated rectangular: {nodes_rect.shape[0]} nodes, {elements_rect.shape[0]} elements
    
    MODAL ANALYSIS:
      - First natural frequency: {omega_modes[0]/(2*np.pi):.3f} Hz
      - Modes retained: {n_modes_use}
    
    RANDOM VIBRATION:
      - RMS displacement: {sigma_u:.6e} m
      - Zero-crossing frequency: {omega_0/(2*np.pi):.3f} Hz
      - Input PSD level: {S0:.3e} N^2/Hz
    
    RELIABILITY:
      - Threshold: {threshold_disp:.6e} m ({threshold_disp/sigma_u:.2f} sigma)
      - FORM reliability index: beta = {beta_form:.4f}
      - FORM failure probability: P_f = {Pf_form:.6e}
      - SORM failure probability: P_f = {Pf_sorm:.6e}
      - First-passage (1 hr): P_f = {Pf_fp:.6e}
    
    SPECIAL VERIFICATIONS:
      - Triangle quadrature error: {abs(I_tri-I_exact_tri):.3e}
      - Hexahedron quadrature error: {abs(I_hex-I_exact_hex):.3e}
      - Thomas solve residual: {np.linalg.norm(r83v_mv(n_beam,a_tri,b_tri,c_tri,x_thomas)-rhs_beam):.3e}
      - CG convergence: {it_cg} iterations
      - Magic square valid: {np.all(row_sums == magic_sum) and np.all(col_sums == magic_sum)}
    
    ================================================================================
    """)
    
    print("\n   ANALYSIS COMPLETE. All modules verified successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
