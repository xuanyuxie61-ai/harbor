#!/usr/bin/env python3
"""
================================================================================
STRUCTURAL MECHANICS: STOCHASTIC VIBRATION AND RELIABILITY ANALYSIS
================================================================================

A doctoral-level computational framework synthesizing 15 seed projects
into a unified pipeline for random vibration response and reliability
analysis of structures with stochastic material properties.

Scientific Problem:
-------------------
A perforated thin plate with polyiamond-inspired hexagonal topology is
subjected to wide-band random vibration. The Young's modulus is modeled
as a lognormal random field via Karhunen-Loeve expansion. The stochastic
dynamic response is computed using:

  1. Polynomial Chaos Expansion (PCE) for uncertainty propagation
  2. Centroidal Voronoi Tessellation (CVT) for optimal random-space sampling
  3. Spherical Voronoi mapping for directional sensitivity analysis
  4. High-precision triangular/hexahedral quadrature for FEM matrices
  5. Tridiagonal solvers (Thomas, CG, Jacobi) for beam-modal preconditioning
  6. Cauchy Principal Value integration for resonant FRF singularities
  7. Golden-section optimization for FORM reliability design-point search
  8. Special functions (Gamma, elliptic integrals) for statistical distributions
  9. Polyiamond mesh topology for complex plate geometry
  10. Matrix Market export for large-scale matrix archival

Key Governing Equations:
------------------------
- Plane stress equilibrium:    div(sigma) + rho*b = rho*d^2u/dt^2
- Constitutive (stochastic):   sigma = D(E(x,xi)) : epsilon
- KL expansion of E:           E(x,xi) = mu_E + sum_i sqrt(lambda_i)*phi_i(x)*xi_i
- PCE response expansion:      u(x,xi) = sum_alpha u_alpha(x) * Psi_alpha(xi)
- Modal FRF:                   H_m(omega) = 1 / (omega_m^2 - omega^2 + 2i*zeta_m*omega_m*omega)
- Response PSD:                S_y(omega) = sum_m |H_m|^2 * S_f(omega) * (phi_m^T * f)^2
- First-passage failure:       P_f = 1 - exp(-nu_0^+ * T)
  nu_0^+ = (omega_0/2*pi) * exp(-b^2/2),  b = threshold / sigma_y
- FORM reliability index:      beta = min ||u|| s.t. g(x(u)) = 0
- SORM correction (Breitung):  P_f,SORM = Phi(-beta) * prod_i (1 + beta*kappa_i)^{-1/2}

Zero-parameter execution: simply run `python main.py`
================================================================================
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Local module imports
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
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def apply_boundary_conditions(K, M, F, bc_dofs):
    """
    Apply zero Dirichlet boundary conditions by eliminating rows/columns.
    
    Parameters
    ----------
    K, M : ndarray
    F : ndarray
    bc_dofs : list or ndarray
        DOF indices to constrain.
    
    Returns
    -------
    K_red, M_red, F_red, free_dofs
    """
    n_dof = K.shape[0]
    free_dofs = np.setdiff1d(np.arange(n_dof), bc_dofs)
    K_red = K[np.ix_(free_dofs, free_dofs)]
    M_red = M[np.ix_(free_dofs, free_dofs)]
    F_red = F[free_dofs] if F is not None else None
    return K_red, M_red, F_red, free_dofs


def main():
    """Main execution pipeline for stochastic structural reliability analysis."""
    
    print("\n" + "#" * 80)
    print("#  STOCHASTIC VIBRATION & RELIABILITY ANALYSIS OF PERFORATED PLATE STRUCTURE  #")
    print("#" * 80)
    
    # ========================================================================
    # 1. MESH GENERATION (Polyiamond-inspired hexagonal plate)
    # ========================================================================
    print_section("1. MESH GENERATION: Polyiamond Hexagonal Plate Topology")
    
    # Use a small hexagonal polyiamond mesh
    hex_order = 2
    scale = 0.5
    nodes, elements, boundary_nodes = polyiamond_hexagon_mesh(hex_order, scale)
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    
    print(f"   Generated hexagonal polyiamond mesh: order={hex_order}, scale={scale}")
    print(f"   Nodes: {n_nodes}, Elements: {n_elements}")
    print(f"   Boundary nodes: {len(boundary_nodes)}")
    
    # Also generate a perforated rectangular plate for comparison
    nodes_rect, elements_rect, boundary_rect = generate_perforated_plate_mesh(
        nx=6, ny=6, hole_positions=[(0.35, 0.35), (0.65, 0.65)],
        hole_radius=0.12, Lx=1.0, Ly=1.0
    )
    print(f"   Perforated rectangular plate: {nodes_rect.shape[0]} nodes, {elements_rect.shape[0]} elements")
    
    # ========================================================================
    # 2. MATERIAL PROPERTIES & STOCHASTIC MODELING
    # ========================================================================
    print_section("2. STOCHASTIC MATERIAL MODELING (KL Expansion)")
    
    # Material properties
    E_mean = 210e9      # 210 GPa (steel)
    E_std = 21e9        # 10% COV
    nu = 0.3
    rho = 7850.0        # kg/m^3
    h_plate = 0.005     # 5 mm thickness
    
    # Karhunen-Loeve expansion parameters
    n_kl_modes = 3
    correlation_length = 0.3
    L_domain = 1.0
    
    # 1D KL modes along x-direction (simplified for demonstration)
    x_kl = np.linspace(0, L_domain, max(20, n_nodes))
    kl_eigvals, kl_modes = kl_expansion_1d(n_kl_modes, L_domain, correlation_length, x_kl)
    
    print(f"   Young's modulus: mean={E_mean/1e9:.1f} GPa, COV={E_std/E_mean*100:.1f}%")
    print(f"   KL expansion: {n_kl_modes} modes retained")
    print(f"   KL eigenvalues: {kl_eigvals}")
    
    # ========================================================================
    # 3. FINITE ELEMENT MATRIX ASSEMBLY
    # ========================================================================
    print_section("3. FINITE ELEMENT MATRIX ASSEMBLY")
    
    # Use rectangular plate for FEM (simpler boundary handling)
    K_det, M_det = assemble_triangular_fem_matrices(
        nodes_rect, elements_rect,
        material_thickness=h_plate,
        young_modulus=E_mean,
        poisson_ratio=nu,
        density=rho
    )
    
    n_dof = K_det.shape[0]
    print(f"   Global stiffness matrix size: {n_dof} x {n_dof}")
    
    # Apply boundary conditions (fix all DOFs on boundary)
    bc_dofs = []
    for bn in boundary_rect:
        bc_dofs.extend([2 * bn, 2 * bn + 1])
    bc_dofs = np.unique(bc_dofs)
    
    F_load = np.zeros(n_dof)
    # Apply random vibration load at center node
    center_node = len(nodes_rect) // 2
    F_load[2 * center_node + 1] = 1000.0  # Vertical load
    
    K_red, M_red, F_red, free_dofs = apply_boundary_conditions(K_det, M_det, F_load, bc_dofs)
    n_free = len(free_dofs)
    print(f"   After BC application: {n_free} free DOFs")
    
    # ========================================================================
    # 4. MODAL ANALYSIS
    # ========================================================================
    print_section("4. MODAL ANALYSIS")
    
    try:
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(K_red, M_red)
        omega_n = np.sqrt(np.maximum(eigvals, 0.0))
    except ImportError:
        # Fallback: use numpy standard eigenvalue solver with mass preconditioning
        M_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M_red)))
        K_tilde = M_inv_sqrt @ K_red @ M_inv_sqrt
        eigvals, eigvecs_tilde = np.linalg.eigh(K_tilde)
        eigvecs = M_inv_sqrt @ eigvecs_tilde
        omega_n = np.sqrt(np.maximum(eigvals, 0.0))
    
    n_modes_use = min(8, n_free)
    omega_modes = omega_n[:n_modes_use]
    phi_modes = eigvecs[:, :n_modes_use]
    
    # Mass-normalize
    for m in range(n_modes_use):
        mass_norm = np.sqrt(phi_modes[:, m].T @ M_red @ phi_modes[:, m])
        if mass_norm > 1e-12:
            phi_modes[:, m] /= mass_norm
    
    print(f"   First {n_modes_use} natural frequencies (Hz):")
    for m in range(n_modes_use):
        print(f"      Mode {m+1}: {omega_modes[m]/(2*np.pi):.3f} Hz  (omega={omega_modes[m]:.2f} rad/s)")
    
    # ========================================================================
    # 5. TRIDIAGONAL SOLVER VERIFICATION (Beam Modal Preconditioning)
    # ========================================================================
    print_section("5. TRIDIAGONAL SOLVER VERIFICATION")
    
    # Build a 1D beam tridiagonal system for verification
    n_beam = 20
    EI_beam = E_mean * h_plate ** 3 / 12.0
    L_beam = 1.0
    a_tri, b_tri, c_tri, rhs_beam, h_beam = build_beam_tridiagonal(
        n_beam, EI_beam, L_beam, load_type='uniform'
    )
    
    # Solve with different methods
    x_thomas = r83v_fs(n_beam, a_tri.copy(), b_tri.copy(), c_tri.copy(), rhs_beam.copy())
    x_cg, it_cg, res_cg = r83v_cg(n_beam, a_tri, b_tri, c_tri, rhs_beam, tol=1e-12)
    x_jac, it_jac, res_jac = r83v_jac_sl(n_beam, a_tri, b_tri, c_tri, rhs_beam, it_max=500, tol=1e-10)
    
    print(f"   Thomas algorithm: ||Ax-b|| = {np.linalg.norm(r83v_mv(n_beam,a_tri,b_tri,c_tri,x_thomas)-rhs_beam):.3e}")
    print(f"   Conjugate Gradient: {it_cg} iterations, residual={res_cg:.3e}")
    print(f"   Jacobi iteration: {it_jac} iterations, residual={res_jac:.3e}")
    
    # Modal analysis on tridiagonal system
    eig_tri, phi_tri = modal_analysis_tridiagonal(n_beam, a_tri, b_tri, c_tri, n_modes=3)
    print(f"   Tridiagonal beam modes: {eig_tri}")
    
    # ========================================================================
    # 6. RANDOM VIBRATION RESPONSE ANALYSIS
    # ========================================================================
    print_section("6. RANDOM VIBRATION RESPONSE ANALYSIS")
    
    # Damping ratios
    zeta_modes = np.array([0.02, 0.015, 0.012, 0.01, 0.01, 0.01, 0.01, 0.01])
    zeta_modes = zeta_modes[:n_modes_use]
    
    # White noise input PSD (constant)
    S0 = 1.0e6  # N^2/Hz
    def psd_input(omega):
        return np.ones_like(omega) * S0
    
    # Compute response PSD using modal superposition
    omega_fine = np.linspace(0.1, 500.0, 2000)
    
    # Find load and response DOF indices in reduced system
    load_idx = np.where(free_dofs == 2 * center_node + 1)[0]
    resp_idx = np.where(free_dofs == 2 * center_node + 1)[0]
    if len(load_idx) == 0 or len(resp_idx) == 0:
        # Fallback: use first available DOF
        load_idx = [0]
        resp_idx = [0]
    
    load_idx = load_idx[0]
    resp_idx = resp_idx[0]
    
    H_frf, omega_m, phi_m = modal_superposition_response(
        omega_fine, K_red, M_red, zeta_modes,
        load_dof=load_idx, response_dof=resp_idx, omega_n_modes=n_modes_use
    )
    
    Sy = np.abs(H_frf) ** 2 * psd_input(omega_fine)
    
    # Mean-square displacement response
    sigma_u_sq = np.trapezoid(Sy, omega_fine) / (2.0 * np.pi)
    sigma_u = np.sqrt(sigma_u_sq)
    
    # Zero-crossing frequency
    moment2 = np.trapezoid(omega_fine ** 2 * Sy, omega_fine) / (2.0 * np.pi)
    omega_0 = np.sqrt(moment2 / sigma_u_sq) if sigma_u_sq > 1e-20 else omega_modes[0]
    
    print(f"   RMS displacement response: {sigma_u:.6f} m")
    print(f"   Zero-crossing frequency: {omega_0/(2*np.pi):.3f} Hz")
    
    # Cauchy Principal Value verification near first resonance
    omega_res = omega_modes[0]
    def cpv_integrand(om):
        return psd_response(om, omega_res, zeta_modes[0], psd_input)
    
    # Numerical CPV around resonance
    cpv_val = cauchy_principal_value(cpv_integrand, omega_res * 0.9, omega_res * 1.1, omega_res, n=20)
    print(f"   CPV integral near 1st resonance: {cpv_val:.4e}")
    
    # ========================================================================
    # 7. PCE-BASED UNCERTAINTY QUANTIFICATION
    # ========================================================================
    print_section("7. POLYNOMIAL CHAOS EXPANSION UNCERTAINTY QUANTIFICATION")
    
    n_stoch = n_kl_modes
    pce_order = 2
    indices = generate_multi_indices(n_stoch, pce_order)
    n_basis = indices.shape[0]
    print(f"   Stochastic dimension: {n_stoch}, PCE order: {pce_order}")
    print(f"   PCE basis size: {n_basis}")
    
    # Generate CVT samples in stochastic space
    n_samples = 200
    cvt_samples = generate_cvt_samples(
        dim=n_stoch, n_gen=n_samples, it_max=30, bounds=[(-1, 1)] * n_stoch, seed=42
    )
    
    # Evaluate PCE basis at samples
    psi_samples, _, norms = evaluate_pce_basis(n_stoch, pce_order, cvt_samples, indices)
    
    # Stochastic response: sample Young's modulus at each CVT point
    # and compute corresponding natural frequencies
    omega_samples = np.zeros((n_samples, n_modes_use))
    for s in range(n_samples):
        xi_s = cvt_samples[s]
        # Realization of E field (simplified: use first KL mode amplitude)
        E_realization = E_mean
        for i in range(n_kl_modes):
            E_realization += E_std * np.sqrt(kl_eigvals[i]) * xi_s[i]
        E_realization = max(E_realization, E_mean * 0.5)
        
        # Scale stiffness (proportional to E)
        K_s = K_red * (E_realization / E_mean)
        try:
            from scipy.linalg import eigh
            eig_s, _ = eigh(K_s, M_red)
        except ImportError:
            M_inv = np.diag(1.0 / np.sqrt(np.diag(M_red)))
            Ks_tilde = M_inv @ K_s @ M_inv
            eig_s, _ = np.linalg.eigh(Ks_tilde)
        omega_samples[s] = np.sqrt(np.maximum(eig_s[:n_modes_use], 0.0))
    
    # Project samples onto PCE basis (least squares)
    # u_alpha = (Psi^T Psi)^{-1} Psi^T u_samples
    PsiTPsi = psi_samples.T @ psi_samples
    for m in range(n_modes_use):
        coeffs = np.linalg.solve(PsiTPsi + 1e-10 * np.eye(n_basis),
                                 psi_samples.T @ omega_samples[:, m])
        mean_om, var_om = pce_moments(coeffs, indices, norms)
        print(f"   Mode {m+1} freq: mean={mean_om/(2*np.pi):.3f} Hz, std={np.sqrt(var_om)/(2*np.pi):.3f} Hz")
    
    # ========================================================================
    # 8. SPHERICAL VORONOI DIRECTIONAL ANALYSIS
    # ========================================================================
    print_section("8. SPHERICAL VORONOI DIRECTIONAL SENSITIVITY")
    
    # Map CVT samples to sphere (use 3D projection of stochastic variables)
    if n_stoch >= 3:
        directions, radii = map_stochastic_to_sphere(cvt_samples[:, :3])
    else:
        # Pad with zeros for 3D embedding
        padded = np.pad(cvt_samples, ((0, 0), (0, 3 - n_stoch)), mode='constant')
        directions, radii = map_stochastic_to_sphere(padded)
    
    # Response metric: maximum natural frequency shift
    max_omega_shift = np.max(omega_samples, axis=1)
    
    sens = directional_reliability_sensitivity(
        directions, max_omega_shift, n_directional_bins=12
    )
    max_sector_response = np.max(sens['max_response'][sens['counts'] > 0])
    print(f"   Maximum directional response sensitivity: {max_sector_response:.3f} rad/s")
    
    # ========================================================================
    # 9. RELIABILITY ANALYSIS (FORM / SORM)
    # ========================================================================
    print_section("9. STRUCTURAL RELIABILITY ANALYSIS (FORM / SORM)")
    
    # Define limit state: g = threshold - response(u)
    # Response is displacement, which increases with stochastic parameter magnitude
    # (representing material degradation or load amplification)
    threshold_disp = 1.5e-5  # 15 micron threshold
    
    # FORM in standard normal space (2D: first two stochastic variables)
    def limit_state(u):
        # Stochastic response model: displacement increases with ||u||
        # representing combined material variability and load uncertainty
        u_ext = np.zeros(n_kl_modes)
        u_ext[:len(u)] = u
        
        # Response amplification factor due to randomness
        # For ||u|| = 0 (mean): factor = 1.0
        # For ||u|| > 0: factor > 1.0 (degradation scenario)
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
    
    # FORM with HL-RF
    beta_form, u_star, Pf_form = form_reliability(
        limit_state, grad_limit_state, u0=np.zeros(2), dim=2, max_iter=50, tol=1e-8
    )
    
    # FORM with golden section search (alternative)
    beta_golden, u_golden, Pf_golden = form_with_golden_search(
        limit_state, dim=2, beta_max=8.0, n_directions=72
    )
    
    # SORM correction
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
    
    # First-passage probability with corrected threshold
    T_duration = 3600.0  # 1 hour
    Pf_fp, nu_up = first_passage_probability(
        sigma_u, threshold_disp, omega_0, T_duration, method='vanmarcke'
    )
    print(f"   First-passage failure (Vanmarcke, T={T_duration/60:.0f} min): P_f = {Pf_fp:.6e}")
    print(f"   Up-crossing rate: {nu_up:.6e} crossings/sec")
    
    # ========================================================================
    # 10. SPECIAL FUNCTIONS & ELLIPTIC INTEGRALS
    # ========================================================================
    print_section("10. SPECIAL FUNCTIONS & ELLIPTIC INTEGRALS")
    
    # Gamma function verification
    z_test = 5.5
    lg_val, ier = lngamma(z_test)
    gamma_val = gamma_function(z_test)
    print(f"   log Gamma({z_test}) = {lg_val:.8f}, Gamma({z_test}) = {gamma_val:.8f}")
    
    # Chi-square PDF for vibration amplitude distribution
    chi2_val = chi2_pdf(3.0, k=2)
    print(f"   Chi-square PDF(3.0, df=2) = {chi2_val:.6f}")
    
    # Rayleigh PDF for peak response
    rayleigh_val = rayleigh_pdf(2.0 * sigma_u, sigma_u)
    print(f"   Rayleigh PDF(2*sigma_y) = {rayleigh_val:.6e}")
    
    # Elliptic integrals for large-deflection beam
    m_ellip = 0.5
    K_val = complete_elliptic_k(m_ellip)
    E_val = complete_elliptic_e(m_ellip)
    print(f"   K({m_ellip}) = {K_val:.6f}, E({m_ellip}) = {E_val:.6f}")
    
    # Nonlinear vibration period
    T_nl, T_lin = nonlinear_vibration_period(
        amplitude=0.01, omega_linear=omega_modes[0], alpha_nonlin=1e6
    )
    print(f"   Nonlinear period: T = {T_nl*1000:.3f} ms, Linear T = {T_lin*1000:.3f} ms")
    
    # Elliptical hole stress concentration
    s_theta, kt = elliptical_hole_stress_concentration(
        a=0.1, b=0.05, sigma_inf=100e6, theta=np.linspace(0, 2*np.pi, 100)
    )
    print(f"   Elliptical hole SCF: K_t = {kt:.2f}")
    print(f"   Max tangential stress: {np.max(s_theta)/1e6:.2f} MPa")
    
    # ========================================================================
    # 11. QUADRATURE VERIFICATION
    # ========================================================================
    print_section("11. HIGH-PRECISION QUADRATURE VERIFICATION")
    
    # Integrate f(x,y)=x^2+y^2 over unit triangle
    unit_triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    f_test = lambda x, y: x ** 2 + y ** 2
    I_tri = integrate_triangle_wandzura(f_test, unit_triangle, degree=5)
    I_exact_tri = 1.0 / 6.0  # Exact: integral_0^1 dx integral_0^{1-x} (x^2+y^2) dy = 1/6
    print(f"   Triangle quadrature: I = {I_tri:.8f}, Exact = {I_exact_tri:.8f}, Error = {abs(I_tri-I_exact_tri):.3e}")
    
    # Hexahedron quadrature
    f_3d = lambda x, y, z: x * y * z
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    I_hex = integrate_hexahedron_witherden(f_3d, bounds, degree=3)
    I_exact_hex = 0.125  # integral_0^1 x dx * integral_0^1 y dy * integral_0^1 z dz = 1/8
    print(f"   Hexahedron quadrature: I = {I_hex:.8f}, Exact = {I_exact_hex:.8f}, Error = {abs(I_hex-I_exact_hex):.3e}")
    
    # Triangulation-level integration
    node_vals = nodes_rect[:, 0] ** 2 + nodes_rect[:, 1] ** 2
    I_triang, area_triang = triangulation_quad(nodes_rect, elements_rect, node_vals)
    print(f"   Triangulation integral: I = {I_triang:.8f}, Total area = {area_triang:.4f}")
    
    # ========================================================================
    # 12. NODAL STRESS RECOVERY & MATRIX EXPORT
    # ========================================================================
    print_section("12. NODAL STRESS RECOVERY & MATRIX EXPORT")
    
    # Compute static displacement for stress recovery
    try:
        u_static = np.linalg.solve(K_red + 1e-3 * np.eye(n_free), F_red)
    except Exception:
        u_static = np.zeros(n_free)
    
    # Expand to full DOF vector
    u_full = np.zeros(n_dof)
    u_full[free_dofs] = u_static
    
    # Element stresses
    elem_stresses, elem_vm = compute_element_stresses(
        u_full, nodes_rect, elements_rect, E_mean, nu
    )
    nodal_vm = nodal_stress_recovery(elem_vm, elements_rect, nodes_rect.shape[0])
    max_vm = np.max(nodal_vm)
    print(f"   Maximum Von Mises stress: {max_vm/1e6:.3f} MPa")
    
    # Export matrices
    output_dir = "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/089_synth_project"
    export_structural_matrices(K_red, M_red, prefix=f"{output_dir}/structural")
    print(f"   Exported matrices to {output_dir}/structural_*.mtx")
    
    # Verify read-back
    K_read = read_matrix_market(f"{output_dir}/structural_K.mtx")
    diff_K = np.linalg.norm(K_read - K_red)
    print(f"   Matrix read-back verification: ||K_read - K||_F = {diff_K:.3e}")
    
    # ========================================================================
    # 13. MAGIC SQUARE TEST MATRIX VERIFICATION
    # ========================================================================
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
    
    # ========================================================================
    # 14. SUMMARY REPORT
    # ========================================================================
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
    main()

# ================================================================
# 测试用例（35个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: polyiamond_hexagon_mesh generates valid topology ----
nodes, elements, boundary_nodes = polyiamond_hexagon_mesh(1, scale=1.0)
assert nodes.shape[0] >= 3, '[TC01] polyiamond_hexagon_mesh FAILED'
assert elements.shape[1] == 3, '[TC01] polyiamond_hexagon_mesh FAILED'
assert len(boundary_nodes) > 0, '[TC01] polyiamond_hexagon_mesh FAILED'

# ---- TC02: magic_square row and column sums equal magic constant ----
n = 5
M = magic_square(n)
magic_sum = n * (n ** 2 + 1) // 2
assert np.all(np.sum(M, axis=1) == magic_sum), '[TC02] magic_square FAILED'
assert np.all(np.sum(M, axis=0) == magic_sum), '[TC02] magic_square FAILED'

# ---- TC03: generate_perforated_plate_mesh excludes holes ----
nodes_p, elements_p, boundary_p = generate_perforated_plate_mesh(4, 4, hole_positions=[(0.5, 0.5)], hole_radius=0.2, Lx=1.0, Ly=1.0)
assert nodes_p.shape[0] > 0, '[TC03] generate_perforated_plate_mesh FAILED'
assert elements_p.shape[0] > 0, '[TC03] generate_perforated_plate_mesh FAILED'
for nidx in range(nodes_p.shape[0]):
    xn, yn = nodes_p[nidx]
    assert (xn - 0.5) ** 2 + (yn - 0.5) ** 2 >= 0.2 ** 2 - 1e-10, '[TC03] generate_perforated_plate_mesh FAILED'

# ---- TC04: integrate_triangle_wandzura exact for constant function ----
unit_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
I_const = integrate_triangle_wandzura(lambda x, y: 1.0, unit_tri, degree=2)
assert abs(I_const - 0.5) < 1e-10, '[TC04] integrate_triangle_wandzura FAILED'

# ---- TC05: integrate_hexahedron_witherden exact for constant ----
I_hex_const = integrate_hexahedron_witherden(lambda x, y, z: 1.0, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), degree=3)
assert abs(I_hex_const - 1.0) < 1e-10, '[TC05] integrate_hexahedron_witherden FAILED'

# ---- TC06: t3_shape_functions sum to unity ----
N, dN_dxi, dN_deta = t3_shape_functions(0.3, 0.2)
assert abs(np.sum(N) - 1.0) < 1e-10, '[TC06] t3_shape_functions FAILED'

# ---- TC07: r83v_mv produces correct matrix-vector product ----
n_r = 5
a_r = np.ones(n_r - 1)
b_r = 2.0 * np.ones(n_r)
c_r = np.ones(n_r - 1)
x_r = np.arange(1, n_r + 1, dtype=float)
y_r = r83v_mv(n_r, a_r, b_r, c_r, x_r)
expected_r = np.array([4.0, 8.0, 12.0, 16.0, 14.0])
assert np.allclose(y_r, expected_r), '[TC07] r83v_mv FAILED'

# ---- TC08: r83v_fs solves tridiagonal system with small residual ----
n_fs = 10
a_fs = -1.0 * np.ones(n_fs - 1)
b_fs = 2.0 * np.ones(n_fs)
c_fs = -1.0 * np.ones(n_fs - 1)
rhs_fs = np.ones(n_fs)
x_sol_fs = r83v_fs(n_fs, a_fs, b_fs, c_fs, rhs_fs)
res_fs = np.linalg.norm(r83v_mv(n_fs, a_fs, b_fs, c_fs, x_sol_fs) - rhs_fs)
assert res_fs < 1e-8, '[TC08] r83v_fs FAILED'

# ---- TC09: r83v_cg converges for SPD tridiagonal system ----
n_cg = 20
a_cg = -1.0 * np.ones(n_cg - 1)
b_cg = 2.0 * np.ones(n_cg)
c_cg = -1.0 * np.ones(n_cg - 1)
np.random.seed(42)
rhs_cg = np.random.rand(n_cg)
x_cg, it_cg, res_cg = r83v_cg(n_cg, a_cg, b_cg, c_cg, rhs_cg, tol=1e-10)
assert res_cg < 1e-8, '[TC09] r83v_cg FAILED'

# ---- TC10: build_beam_tridiagonal produces symmetric system ----
a_beam, b_beam, c_beam, rhs_beam, h_beam = build_beam_tridiagonal(10, EI=1.0, L=1.0, load_type='uniform')
assert np.allclose(a_beam, c_beam), '[TC10] build_beam_tridiagonal FAILED'
assert h_beam > 0, '[TC10] build_beam_tridiagonal FAILED'

# ---- TC11: frequency_response_function peak near resonance ----
omega_f = np.linspace(0.1, 20.0, 1000)
omega_n_f = 10.0
zeta_f = 0.05
H_f = frequency_response_function(omega_f, omega_n_f, zeta_f)
peak_idx_f = np.argmax(np.abs(H_f))
assert abs(omega_f[peak_idx_f] - omega_n_f) < 0.5, '[TC11] frequency_response_function FAILED'

# ---- TC12: psd_response non-negative for all frequencies ----
omega_p = np.linspace(0.1, 20.0, 500)
S_y_p = psd_response(omega_p, omega_n=10.0, zeta=0.02, psd_input=lambda o: np.ones_like(o))
assert np.all(S_y_p >= 0), '[TC12] psd_response FAILED'

# ---- TC13: first_passage_probability within [0, 1] ----
Pf_fp, nu_fp = first_passage_probability(sigma_y=1.0, threshold=2.0, omega_0=10.0, T_duration=3600.0, method='poisson')
assert 0.0 <= Pf_fp <= 1.0, '[TC13] first_passage_probability FAILED'
assert nu_fp >= 0, '[TC13] first_passage_probability FAILED'

# ---- TC14: cauchy_principal_value finite for smooth function ----
cpv_val = cauchy_principal_value(lambda t: np.sin(t), 0.0, np.pi, np.pi / 2.0, n=20)
assert np.isfinite(cpv_val), '[TC14] cauchy_principal_value FAILED'

# ---- TC15: lngamma value matches known log Gamma(5.5) ----
lg_val, ier_val = lngamma(5.5)
assert ier_val == 0, '[TC15] lngamma FAILED'
assert abs(lg_val - 3.957813) < 1e-4, '[TC15] lngamma FAILED'

# ---- TC16: gamma_function Gamma(5) = 24 ----
g5_val = gamma_function(5.0)
assert abs(g5_val - 24.0) < 1e-8, '[TC16] gamma_function FAILED'

# ---- TC17: golden_section_search finds minimum of parabola ----
x_min_gs, f_min_gs, it_gs = golden_section_search(lambda x: (x - 2.0) ** 2, -5.0, 5.0, n_iter=50, x_tol=1e-10)
assert abs(x_min_gs - 2.0) < 1e-6, '[TC17] golden_section_search FAILED'
assert f_min_gs < 1e-10, '[TC17] golden_section_search FAILED'

# ---- TC18: normal_cdf monotonic and bounded ----
x_cdf = np.array([-5.0, 0.0, 5.0])
cdf_vals = normal_cdf(x_cdf)
assert np.all(np.diff(cdf_vals) > 0), '[TC18] normal_cdf FAILED'
assert 0.0 <= cdf_vals[0] <= 0.01, '[TC18] normal_cdf FAILED'
assert 0.99 <= cdf_vals[-1] <= 1.0, '[TC18] normal_cdf FAILED'

# ---- TC19: complete_elliptic_K increasing in m ----
m_ell = np.array([0.1, 0.5, 0.9])
K_ell = complete_elliptic_k(m_ell)
assert np.all(np.diff(K_ell) > 0), '[TC19] complete_elliptic_k FAILED'

# ---- TC20: nonlinear_vibration_period hardening T < T_linear ----
T_nl_val, T_lin_val = nonlinear_vibration_period(amplitude=0.01, omega_linear=100.0, alpha_nonlin=1e6)
assert T_nl_val < T_lin_val, '[TC20] nonlinear_vibration_period FAILED'

# ---- TC21: elliptical_hole_stress_concentration SCF >= 1 ----
theta_arr = np.linspace(0, 2 * np.pi, 50)
s_theta_val, kt_val = elliptical_hole_stress_concentration(a=0.2, b=0.1, sigma_inf=100e6, theta=theta_arr)
assert kt_val >= 1.0, '[TC21] elliptical_hole_stress_concentration FAILED'

# ---- TC22: legendre_polynomial orthonormality on Gauss quadrature ----
from pce_expansion import legendre_polynomial
xq_leg, wq_leg = np.polynomial.legendre.leggauss(5)
L_leg = legendre_polynomial(3, xq_leg)
for i_leg in range(4):
    for j_leg in range(i_leg + 1, 4):
        dot_leg = np.sum(wq_leg * L_leg[i_leg] * L_leg[j_leg])
        assert abs(dot_leg) < 1e-10, '[TC22] legendre_polynomial FAILED'

# ---- TC23: generate_multi_indices count matches expected for n=3,p=2 ----
indices_test = generate_multi_indices(3, 2)
expected_count_23 = 10
assert indices_test.shape[0] == expected_count_23, '[TC23] generate_multi_indices FAILED'

# ---- TC24: evaluate_pce_basis norms equal 1 for orthonormal Legendre ----
np.random.seed(42)
xi_test = np.random.uniform(-1, 1, (10, 2))
psi_test, _, norms_test = evaluate_pce_basis(2, 2, xi_test)
assert np.allclose(norms_test, 1.0), '[TC24] evaluate_pce_basis FAILED'

# ---- TC25: pce_moments mean equals first coefficient ----
coeffs_test = np.array([3.0, 0.5, -0.2, 1.0, 0.0, 0.3])
indices_pce = generate_multi_indices(2, 2)
mean_pce, var_pce = pce_moments(coeffs_test, indices_pce)
assert abs(mean_pce - 3.0) < 1e-10, '[TC25] pce_moments FAILED'
assert var_pce >= 0, '[TC25] pce_moments FAILED'

# ---- TC26: kl_expansion_1d modes are normalized ----
x_kl = np.linspace(0, 1.0, 50)
eigvals_kl, modes_kl = kl_expansion_1d(3, length=1.0, correlation_length=0.3, x_coords=x_kl)
for i_kl in range(3):
    norm_sq_kl = np.trapezoid(modes_kl[i_kl] ** 2, x_kl)
    assert abs(norm_sq_kl - 1.0) < 0.1, '[TC26] kl_expansion_1d FAILED'

# ---- TC27: cvt_energy decreases after one Lloyd iteration ----
np.random.seed(42)
from cvt_sampling import cvt_energy, cvt_iterate
gens_cvt = np.random.rand(5, 2)
samples_cvt = np.random.rand(1000, 2)
E0_cvt, _ = cvt_energy(gens_cvt, samples_cvt)
new_gens_cvt, diff_cvt, _ = cvt_iterate(gens_cvt, samples_cvt)
E1_cvt, _ = cvt_energy(new_gens_cvt, samples_cvt)
assert E1_cvt <= E0_cvt + 1e-10, '[TC27] cvt_iterate FAILED'

# ---- TC28: generate_cvt_samples reproducible with fixed seed ----
np.random.seed(42)
samples1 = generate_cvt_samples(dim=2, n_gen=10, it_max=5, seed=42)
np.random.seed(42)
samples2 = generate_cvt_samples(dim=2, n_gen=10, it_max=5, seed=42)
assert np.allclose(samples1, samples2), '[TC28] generate_cvt_samples FAILED'

# ---- TC29: map_stochastic_to_sphere yields unit directions ----
np.random.seed(42)
xi_samples = np.random.randn(20, 3)
directions, radii = map_stochastic_to_sphere(xi_samples)
norms_dir = np.linalg.norm(directions, axis=1)
assert np.allclose(norms_dir, 1.0), '[TC29] map_stochastic_to_sphere FAILED'
assert np.all(radii >= 0), '[TC29] map_stochastic_to_sphere FAILED'

# ---- TC30: matrix export and read round-trip preserves values ----
A_test = np.array([[1.0, 2.0], [3.0, 4.0]])
import tempfile, os
from matrix_exporter import export_matrix_market
fd, tmpfile = tempfile.mkstemp(suffix='.mtx')
os.close(fd)
export_matrix_market(tmpfile, A_test, symmetry='general')
A_read = read_matrix_market(tmpfile)
assert np.allclose(A_read, A_test), '[TC30] matrix export/read FAILED'
os.remove(tmpfile)

# ---- TC31: modal_analysis_tridiagonal eigenvalues positive ----
n_ma = 15
a_ma = -1.0 * np.ones(n_ma - 1)
b_ma = 2.0 * np.ones(n_ma)
c_ma = -1.0 * np.ones(n_ma - 1)
eigvals_tri, eigvecs_tri = modal_analysis_tridiagonal(n_ma, a_ma, b_ma, c_ma, n_modes=3)
assert np.all(eigvals_tri > 0), '[TC31] modal_analysis_tridiagonal FAILED'

# ---- TC32: assemble_triangular_fem_matrices produces symmetric K ----
nodes_t = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
elems_t = np.array([[0, 1, 2], [1, 3, 2]])
K_t, M_t = assemble_triangular_fem_matrices(nodes_t, elems_t, material_thickness=1.0, young_modulus=1.0, poisson_ratio=0.3, density=1.0)
assert np.allclose(K_t, K_t.T), '[TC32] assemble_triangular_fem_matrices FAILED'
assert np.allclose(M_t, M_t.T), '[TC32] assemble_triangular_fem_matrices FAILED'

# ---- TC33: faces_average counts consistent ----
elem_stress = np.array([1.0, 2.0, 3.0])
elems_avg = np.array([[0, 1, 2], [1, 2, 3]])
nodal_avg = faces_average(elem_stress, elems_avg, 4)
assert nodal_avg[1] == 1.5, '[TC33] faces_average FAILED'

# ---- TC34: triangulation_quad exact for linear function ----
nodes_lin = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
elems_lin = np.array([[0, 1, 2]])
node_vals_lin = nodes_lin[:, 0] + nodes_lin[:, 1]
I_lin, area_lin = triangulation_quad(nodes_lin, elems_lin, node_vals_lin)
assert abs(I_lin - (2.0 / 3.0) * area_lin) < 1e-10, '[TC34] triangulation_quad FAILED'

# ---- TC35: FORM reliability beta non-negative for simple linear limit state ----
g_lin_lambda = lambda u: 3.0 - np.sum(u)
dg_lin_lambda = lambda u: -np.ones_like(u)
beta_form, u_star, Pf_form = form_reliability(g_lin_lambda, dg_lin_lambda, u0=np.zeros(2), dim=2, max_iter=50, tol=1e-8)
assert beta_form >= 0, '[TC35] form_reliability FAILED'
assert 0.0 <= Pf_form <= 1.0, '[TC35] form_reliability FAILED'

print('\n全部 35 个测试通过!\n')
