
import numpy as np
import math
import time

from utils_physics import (
    init_physics_system, generate_prime_seed, rot13_encode,
    geometric_units, safe_divide, fermi_momentum_to_density
)
from eos_legendre import (
    associated_legendre_polynomial_value,
    SkyrmeEOS, PolytropicEOS, build_composite_eos,
    legendre_angular_expansion
)
from tov_solver import (
    TOVIntegrator, compute_mass_radius_relation, compute_tidal_deformability,
    build_eps_of_P_simplified
)
from crust_integrals import (
    triangle_area, triangle_unit_monomial_integral,
    quadrilateral_witherden_rule, quadrilateral_unit_monomial_integral,
    wedge01_monomial_integral, pasta_free_energy_triangle_lattice,
    pasta_free_energy_quadrilateral_sheet, pasta_free_energy_wedge_cylinder
)
from lattice_grid import (
    polygon_grid_count, polygon_grid_points,
    generate_crust_lattice_hexagonal,
    compute_crust_shear_modulus,
    r8mat_solve
)
from neutrino_diffusion import (
    solve_neutrino_diffusion_1d,
    compute_neutrino_luminosity,
    compute_deleptonization_timescale
)
from matrix_stability import (
    r8lt_det, r8lt_inverse,
    hilbert_matrix, hilbert_inverse,
    matrix_condition_number_1d,
    test_numerical_stability_on_hilbert,
    analyze_eigenvalue_stability
)
from quadrature_verify import (
    hermite_integral, hermite_quadrature_exactness,
    gauss_hermite_nodes_weights, gauss_laguerre_nodes_weights,
    verify_eos_integrals
)


def run_phase_1_eos_and_legendre():
    print("\n" + "=" * 70)
    print("PHASE 1: Equation of State & Legendre Polynomial Expansion")
    print("=" * 70)

    eos = SkyrmeEOS()


    rho_vals = np.linspace(0.05, 0.8, 10)
    print(f"\n{'rho (fm^-3)':>12} {'eps (MeV/fm^3)':>18} {'P (MeV/fm^3)':>18} {'cs^2/c^2':>12}")
    print("-" * 65)
    for rho in rho_vals:
        eps = eos.energy_density(rho)
        P = eos.pressure(rho)
        cs2 = eos.sound_speed_squared(rho)
        print(f"{rho:12.4f} {eps:18.4f} {P:18.4f} {cs2:12.6f}")


    print("\n--- Legendre Angular Expansion Test ---")
    coeffs = np.array([1.0, 0.5, -0.3, 0.1])
    cos_theta = np.linspace(-1.0, 1.0, 21)
    V_vals = legendre_angular_expansion(coeffs, cos_theta)
    print(f"Potential V(cosθ) at θ=0: {V_vals[0]:.6f} MeV")
    print(f"Potential V(cosθ) at θ=π: {V_vals[-1]:.6f} MeV")


    mu_n, mu_p = eos.chemical_potential(0.16, delta=0.2)
    print(f"\nChemical potentials at saturation (δ=0.2):")
    print(f"  μ_n = {mu_n:.4f} MeV")
    print(f"  μ_p = {mu_p:.4f} MeV")

    return eos


def run_phase_2_tov_structure(eos):
    print("\n" + "=" * 70)
    print("PHASE 2: TOV Equation & Mass-Radius Relation")
    print("=" * 70)






    raise NotImplementedError("Hole 3: 请补全物态方程构建与 TOV 参数扫描调用")

    print(f"\n{'Pc (MeV/fm^3)':>16} {'R (km)':>12} {'M (M_sun)':>12} {'M/R (geom)':>14}")
    print("-" * 58)
    for i in range(len(mr_data['Pc'])):
        Pc = mr_data['Pc'][i]
        R_km = mr_data['R_km'][i]
        M_sun = mr_data['M_sun'][i]
        MR = mr_data['M_over_R'][i]
        if not (np.isnan(R_km) or np.isnan(M_sun)):
            print(f"{Pc:16.2f} {R_km:12.3f} {M_sun:12.4f} {MR:14.6f}")


    print("\n--- Tidal Deformability ---")
    Pc_test = 200.0
    Lambda = compute_tidal_deformability(eps_of_P, Pc_test)
    if not np.isnan(Lambda):
        print(f"Tidal deformability Λ at Pc={Pc_test:.2f} MeV/fm^3: {Lambda:.4e}")
    else:
        print("Tidal deformability computation failed for this central pressure.")

    return mr_data


def run_phase_3_crust_integrals():
    print("\n" + "=" * 70)
    print("PHASE 3: Crust Nuclear Pasta Phase Multi-Region Integration")
    print("=" * 70)


    print("\n--- Triangle Monomial Integral Exactness ---")
    test_expons = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    for expon in test_expons:
        exact = triangle_unit_monomial_integral(expon)
        print(f"  x^{expon[0]} y^{expon[1]}: exact = {exact:.10e}")


    print("\n--- Quadrilateral Witherden Rule Verification ---")
    for p in [1, 3, 5]:
        n, x, y, w = quadrilateral_witherden_rule(p)

        max_err = 0.0
        for px in range(p + 1):
            for py in range(p + 1 - px):
                exact = quadrilateral_unit_monomial_integral((px, py))
                quad = np.sum(w * (x**px) * (y**py))
                err = abs(quad - exact)
                max_err = max(max_err, err)
        print(f"  Order p={p:2d}: n={n:2d} points, max monomial error = {max_err:.2e}")


    print("\n--- Wedge Monomial Integral ---")
    test_wedge = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 2), (1, 1, 2)]
    for e in test_wedge:
        val = wedge01_monomial_integral(e)
        print(f"  x^{e[0]} y^{e[1]} z^{e[2]}: integral = {val:.10e}")


    print("\n--- Nuclear Pasta Free Energy Estimates ---")
    F_gnocchi = pasta_free_energy_triangle_lattice(
        lattice_spacing=50.0, surface_tension=1.0, nuclear_radius=8.0
    )
    F_lasagna = pasta_free_energy_quadrilateral_sheet(
        sheet_width=10.0, sheet_spacing=50.0, surface_tension=1.0
    )
    F_spaghetti = pasta_free_energy_wedge_cylinder(
        cylinder_radius=8.0, cell_size=50.0, surface_tension=1.0
    )
    print(f"  Gnocchi phase (triangle lattice):  {F_gnocchi:.4f} MeV/fm^2")
    print(f"  Lasagna phase (quad sheet):        {F_lasagna:.4f} MeV/fm^2")
    print(f"  Spaghetti phase (wedge cylinder):  {F_spaghetti:.4f} MeV/fm^3")


def run_phase_4_lattice_grid():
    print("\n" + "=" * 70)
    print("PHASE 4: Crust Lattice Grid Generation & Elasticity")
    print("=" * 70)


    print("\n--- Hexagonal Lattice Generation ---")
    nodes, elements = generate_crust_lattice_hexagonal(
        lattice_constant=30.0, n_layers=3
    )
    print(f"  Generated {nodes.shape[0]} nodes and {len(elements)} triangular elements.")


    print("\n--- Polygon Grid (Wigner-Seitz Cell) ---")
    n_sub = 4
    nv = 6
    v = np.array([
        [0.0, 0.0],
        [30.0, 0.0],
        [45.0, 25.98],
        [30.0, 51.96],
        [0.0, 51.96],
        [-15.0, 25.98]
    ])
    ng = polygon_grid_count(n_sub, nv)
    xg = polygon_grid_points(n_sub, nv, v, ng)
    print(f"  Polygon grid: {n_sub} subdivisions -> {ng} points")
    print(f"  Centroid at: ({xg[0, 0]:.2f}, {xg[0, 1]:.2f}) fm")


    shear = compute_crust_shear_modulus(nodes, elements, young_modulus=1.0e35)
    print(f"\n  Estimated crust shear modulus: {shear:.4e} Pa")


    print("\n--- Linear System Test (Gauss-Jordan) ---")
    n_test = 5
    A_test = np.eye(n_test) + 0.1 * np.random.rand(n_test, n_test)
    b_test = np.ones(n_test)
    aug = np.hstack([A_test, b_test.reshape(-1, 1)])
    sol, info = r8mat_solve(n_test, 1, aug)
    if info == 0:
        x_sol = sol[:, -1]
        residual = np.linalg.norm(A_test @ x_sol - b_test)
        print(f"  Gauss-Jordan solve residual: {residual:.2e}")
    else:
        print(f"  Gauss-Jordan failed at step {info}")


def run_phase_5_neutrino_diffusion():
    print("\n" + "=" * 70)
    print("PHASE 5: Neutrino Diffusion & Cooling")
    print("=" * 70)

    print("\nSolving 1D neutrino diffusion-convection-reaction equations...")
    x, t, sol = solve_neutrino_diffusion_1d(
        nx=100, nt=2000, t_final=5.0, L=5.0e4
    )


    Y_e_initial = sol[0, :, 0]
    Y_e_final = sol[-1, :, 0]
    T_initial = sol[0, :, 1]
    T_final = sol[-1, :, 1]

    print(f"\n  Initial Y_e range: [{Y_e_initial.min():.4f}, {Y_e_initial.max():.4f}]")
    print(f"  Final Y_e range:   [{Y_e_final.min():.4f}, {Y_e_final.max():.4f}]")
    print(f"  Initial T range:   [{T_initial.min():.4e}, {T_initial.max():.4e}] K")
    print(f"  Final T range:     [{T_final.min():.4e}, {T_final.max():.4e}] K")


    dx = x[1] - x[0]
    L_nu = compute_neutrino_luminosity(T_final, dx, R_star=1.0e6)
    print(f"\n  Neutrino luminosity: {L_nu:.4e} erg/s")


    tau = compute_deleptonization_timescale(
        Y_e_initial=0.3, Y_e_final=0.1, weak_rate=1.0e-2
    )
    print(f"  Deleptonization timescale: {tau:.2f} s")


def run_phase_6_matrix_stability():
    print("\n" + "=" * 70)
    print("PHASE 6: Matrix Stability & Ill-Conditioned System Analysis")
    print("=" * 70)


    print("\n--- Lower Triangular Matrix Determinant ---")
    n_lt = 8
    L_mat = np.tril(np.random.rand(n_lt, n_lt) + 0.5)
    det_lt = r8lt_det(n_lt, L_mat)
    det_np = np.linalg.det(L_mat)
    print(f"  Custom r8lt_det:   {det_lt:.10e}")
    print(f"  NumPy det:         {det_np:.10e}")
    print(f"  Relative diff:     {abs(det_lt - det_np) / abs(det_np):.2e}")


    L_inv = r8lt_inverse(n_lt, L_mat)
    identity_check = L_mat @ L_inv
    off_diag_max = np.max(np.abs(identity_check - np.eye(n_lt)))
    print(f"  Inverse accuracy (off-diag max): {off_diag_max:.2e}")


    print("\n--- Hilbert Matrix Numerical Stability ---")
    stability_results = test_numerical_stability_on_hilbert(n_max=8)
    print(f"{'n':>4} {'cond(H)':>16} {'||x_num - x_ex||':>20}")
    print("-" * 45)
    for n, res in stability_results.items():
        print(f"{n:4d} {res['condition_number']:16.4e} {res['error']:20.4e}")


    print("\n--- Eigenvalue Stability Test ---")
    A_stable = np.array([[-2.0, 1.0], [0.5, -1.5]])
    eigs, is_stable = analyze_eigenvalue_stability(A_stable)
    print(f"  Test matrix eigenvalues: {eigs[0]:.4f}, {eigs[1]:.4f}")
    print(f"  System is {'STABLE' if is_stable else 'UNSTABLE'}")


def run_phase_7_quadrature_verification():
    print("\n" + "=" * 70)
    print("PHASE 7: Quadrature Rule Exactness Verification")
    print("=" * 70)


    print("\n--- Gauss-Hermite Quadrature Verification ---")
    for n in [4, 8, 16]:
        x, w = gauss_hermite_nodes_weights(n)
        errors = hermite_quadrature_exactness(n, x, w, p_max=2 * n - 1)
        max_err = np.max(errors)
        print(f"  n={n:2d}: max relative error for polynomials up to degree {2*n-1} = {max_err:.2e}")


    print("\n--- Gauss-Laguerre Quadrature Verification ---")
    for n in [4, 8]:
        x, w = gauss_laguerre_nodes_weights(n)

        quad_0 = np.sum(w)
        print(f"  n={n:2d}: ∫e^{{-x}}dx via quadrature = {quad_0:.15f} (exact=1.0)")


    print("\n--- EOS Critical Integral Verification ---")
    verify_res = verify_eos_integrals()
    for key, val in verify_res.items():
        print(f"  {key}: {val:.4e}")


def run_phase_8_synthesis_summary():
    print("\n" + "=" * 70)
    print("PHASE 8: Synthesis Summary & Encoded Output")
    print("=" * 70)

    seed = generate_prime_seed(lower=1000, upper=5000)
    print(f"\n  Generated prime random seed: {seed}")

    summary = (
        "NeutronStarEOSDenseMatterSynthesisCompletedSuccessfully"
    )
    encoded = rot13_encode(summary)
    print(f"  ROT13 encoded status: {encoded}")
    print(f"  Decoded verification: {rot13_encode(encoded)}")

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETED SUCCESSFULLY")
    print("=" * 70)


def main():
    t_start = time.time()
    init_physics_system()


    eos = run_phase_1_eos_and_legendre()


    mr_data = run_phase_2_tov_structure(eos)


    run_phase_3_crust_integrals()


    run_phase_4_lattice_grid()


    run_phase_5_neutrino_diffusion()


    run_phase_6_matrix_stability()


    run_phase_7_quadrature_verification()


    run_phase_8_synthesis_summary()

    t_elapsed = time.time() - t_start
    print(f"\nTotal execution time: {t_elapsed:.3f} seconds\n")


if __name__ == "__main__":
    main()
