
import numpy as np
import time

from nuclide_sampling import sample_nuclide_mass_chain, build_r_process_nuclide_set
from reaction_rates import build_reaction_rate_table
from nuclear_network import solve_network_bdf2, compute_abundance_peaks
from neutron_transport import neutron_diffusion_solution, neutron_capture_rate_profile
from spectral_expansion import spectral_expand_reaction_rate, spectral_evaluate_reaction_rate
from circulant_solver import circulant_solve, build_circulant_dif2
from nonlinear_root import solve_neutron_chemical_potential
from conformal_mapping import map_accretion_streamline, temperature_field_conformal
from spherical_geometry import icosahedron_vertices, spherical_delaunay_triangulation
from nuclide_encoding import atbash_mirror_map, build_nuclide_grid_path, gaussian_prime_spiral_trajectory
from quadrature_rules import integrate_tetrahedron, wedge_exactness_monomial_integral
from fem_approximation import fem1d_approximate, fem1d_evaluate
from voronoi_partition import partition_nuclear_chart, interpolate_nuclear_data


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_nuclide_sampling():
    print_section("Step 1: Nuclide Space Sampling & Path Generation")
    a_values = sample_nuclide_mass_chain(a_min=80, a_max=240, n_nuclides=40,
                                          density_profile='r_process_path')
    nuclides = build_r_process_nuclide_set(a_values, beta_stability_offset=10)
    print(f"  Sampled {len(nuclides)} r-process nuclides")


    spiral = gaussian_prime_spiral_trajectory(0 + 0j, 1, max_steps=200)
    print(f"  Gaussian prime spiral length: {len(spiral)}")


    mirrored = atbash_mirror_map(nuclides[:5])
    print(f"  Mirror nuclei example: {nuclides[0]} -> {mirrored[0]}")


    generators, labels = partition_nuclear_chart(nuclides, n_partitions=4)
    print(f"  Nuclear chart partitioned into {len(np.unique(labels))} regions")

    return nuclides


def run_reaction_rates(nuclides):
    print_section("Step 2: Temperature-Dependent Reaction Rates & Spectral Expansion")
    T9_range = np.linspace(0.5, 3.0, 50)


    S_n_table = {}
    T_half_table = {}
    for z, n, a in nuclides:
        S_n_table[(z, a)] = 8.0 - 0.02 * (a - 100)
        T_half_table[(z, a)] = max(0.01, 10.0 * np.exp(-0.01 * (a - 80)))

    rates = build_reaction_rate_table(nuclides, T9_range, S_n_table, T_half_table)
    print(f"  Reaction rate table built for {len(nuclides)} nuclides")


    key = (nuclides[0][0], nuclides[0][2])
    cap_rates = rates['capture'][key]
    coeffs, t_min, t_max = spectral_expand_reaction_rate(T9_range, cap_rates, degree=8)
    print(f"  Spectral expansion coefficients (degree 8): max |c_k| = {np.max(np.abs(coeffs)):.3e}")


    tau_test = np.linspace(0, 1, 20)
    recon = spectral_evaluate_reaction_rate(tau_test, coeffs, t_min, t_max)
    T_test = tau_test * (t_max - t_min) + t_min
    exact_interp = np.interp(T_test, T9_range, cap_rates)
    err = np.max(np.abs(recon - exact_interp) / (np.abs(exact_interp) + 1e-30))
    print(f"  Spectral reconstruction max relative error: {err:.3e}")

    return rates, T9_range, S_n_table, T_half_table


def run_neutron_transport():
    print_section("Step 3: Neutron Transport & Chemical Potential")
    r = np.linspace(1e3, 1e6, 500)
    R_star = 1e6
    D = 1e5
    sigma_a = 1e-3
    S0 = 1e20
    phi = neutron_diffusion_solution(r, R_star, D, sigma_a, S0)
    print(f"  Neutron flux at center: {phi[0]:.3e} cm^-2 s^-1")
    print(f"  Neutron flux at surface: {phi[-1]:.3e} cm^-2 s^-1")


    n_n = 1e30
    sigma_cap = 1e-24
    capture_profile = neutron_capture_rate_profile(r, phi, n_n, sigma_cap)
    print(f"  Max capture rate: {np.max(capture_profile):.3e} cm^-3 s^-1")


    mu_n, info = solve_neutron_chemical_potential(n_n, 1e9)
    print(f"  Neutron chemical potential: {mu_n:.3e} erg (eta={info.get('eta', 'N/A')})")

    return phi, capture_profile


def run_nuclear_network(nuclides, rates, T9_range=None):
    print_section("Step 4: Nuclear Reaction Network Evolution")
    n_nuc = len(nuclides)
    Y0 = np.ones(n_nuc) / n_nuc
    rho = 1e8
    n_n = 1e30
    t_end = 5.0







    temp_profile = None
    t_hist, Y_hist = solve_network_bdf2(nuclides, rates, rho, n_n, Y0,
                                         t_end=t_end, n_steps=200,
                                         temp_profile=temp_profile)
    print(f"  Network solved: {len(t_hist)} time steps")
    print(f"  Final total abundance: {np.sum(Y_hist[-1]):.6f}")


    A_centers, abundances = compute_abundance_peaks(Y_hist[-1], nuclides)
    peak_idx = np.argmax(abundances)
    print(f"  Dominant abundance peak at A ≈ {A_centers[peak_idx]:.0f}")

    return t_hist, Y_hist, A_centers, abundances


def run_geometric_tools():
    print_section("Step 5: Geometric Mapping & Spherical Mesh")

    theta = np.linspace(0, 2 * np.pi, 100)
    w_r, w_i = map_accretion_streamline(1.2, theta, offset=0.15)
    print(f"  Accretion streamline mapped to {len(w_r)} points")


    verts = icosahedron_vertices()
    faces = spherical_delaunay_triangulation(verts)
    print(f"  Icosahedron triangulation: {len(verts)} vertices, {len(faces)} faces")


    rho_grid = np.linspace(1.0, 10.0, 50)
    T_field = temperature_field_conformal(rho_grid, 0.0, 1e9, 1e8)
    print(f"  Temperature field range: [{np.min(T_field):.3e}, {np.max(T_field):.3e}] K")


def run_numerical_integrals():
    print_section("Step 6: Numerical Quadrature Verification")

    f_test = lambda x, y, z: x * y * z
    val_tet = integrate_tetrahedron(f_test, n_per_dim=8)
    exact_tet = 1.0 / 720.0
    print(f"  Tetrahedron integral: {val_tet:.6e}, exact: {exact_tet:.6e}, err: {abs(val_tet - exact_tet):.3e}")


    exact_wedge = wedge_exactness_monomial_integral(2, 1, 0)
    print(f"  Wedge exact integral (x^2*y): {exact_wedge:.6e}")


def run_fem_and_circulant():
    print_section("Step 7: FEM Approximation & Circulant Solver")

    T_data = np.random.rand(80)
    R_data = np.exp(-2.0 / T_data) + 0.05 * np.random.randn(80)
    mesh = np.linspace(0.1, 3.0, 25)
    coeffs_fem = fem1d_approximate(mesh, T_data, R_data,
                                    weight_approx=1.0, weight_deriv=0.1,
                                    weight_boundary=1e4,
                                    boundary_values=(0.0, 0.0))
    T_test = np.linspace(0.1, 3.0, 100)
    R_fit = fem1d_evaluate(T_test, mesh, coeffs_fem)
    print(f"  FEM fit evaluated at {len(T_test)} points, range: [{np.min(R_fit):.3e}, {np.max(R_fit):.3e}]")


    n = 32
    a_circ = np.array([3.0, -1.0] + [0.0] * (n - 3) + [-1.0])
    b = np.random.rand(n)
    x_sol = circulant_solve(a_circ, b)

    from circulant_solver import circulant_matvec
    residual = np.linalg.norm(circulant_matvec(a_circ, x_sol) - b)
    print(f"  Circulant solver residual: {residual:.3e}")


def run_interpolation(nuclides):
    print_section("Step 8: Nuclear Data Interpolation")
    coords = np.array([(z, n) for z, n, a in nuclides], dtype=float)

    data = np.array([8.0 - 0.02 * (a - 100) for z, n, a in nuclides])

    query = np.array([[30, 50], [40, 65], [50, 82]], dtype=float)
    interp_vals = interpolate_nuclear_data(query, coords, data)
    print(f"  Interpolated S_n at query points: {interp_vals}")


def main():
    print("\n" + "#" * 70)
    print("#  r-Process Nucleosynthesis Multi-Scale Simulation Platform")
    print("#  Nuclear Astrophysics: Neutron Star Merger Environment")
    print("#" * 70)
    start_time = time.time()


    nuclides = run_nuclide_sampling()


    rates, T9_range, S_n_table, T_half_table = run_reaction_rates(nuclides)


    phi, capture_profile = run_neutron_transport()


    t_hist, Y_hist, A_centers, abundances = run_nuclear_network(nuclides, rates, T9_range)


    run_geometric_tools()


    run_numerical_integrals()


    run_fem_and_circulant()


    run_interpolation(nuclides)

    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"#  Simulation completed successfully in {elapsed:.2f} seconds")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
