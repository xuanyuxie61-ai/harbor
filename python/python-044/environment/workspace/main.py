
import numpy as np
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from biot_equations import PoroelasticMaterial, BiotConsolidation, compute_characteristic_frequencies
from mesh_generator import (
    generate_structured_triangle_mesh, generate_quadratic_nodes,
    mesh_quality_metrics, identify_boundary_nodes,
    fibonacci_spiral_points, cvt_lloyd_1d, chebyshev_zero_nodes
)
from quadrature_rules import line_rule, triangle_rule, line_monomial_integral
from fem2d_assembler import fem2d_biot_assemble, apply_dirichlet_bc
from time_integrator import midpoint_implicit_step, imex_splitting_step, exponential_integrator_exact
from hermite_interpolator import reconstruct_wave_field_1d
from sparse_matrix_utils import build_adjacency_matrix, write_hb_format, estimate_condition_number
from kmeans_clustering import cluster_velocity_zones
from histogram_analyzer import energy_spectrum_bins, analyze_wave_front_histogram
from tet3d_basis import tetrahedron_volume, basis_mn_tet4
from wave_analyzer import (
    biot_wave_velocities_low_freq, biot_dispersion_relation,
    compute_quality_factor, separate_fast_slow_waves
)


def main():
    print("=" * 70)
    print("  Poroelastic Wave Propagation Simulation")
    print("  Biot's Theory | FEM | Fast/Slow P-wave Analysis")
    print("=" * 70)




    print("\n[1] Defining poroelastic material properties...")
    material = PoroelasticMaterial(
        lam=6.0e9,
        mu=9.0e9,
        phi=0.20,
        kappa=1.0e-13,
        eta=1.0e-3,
        K_s=36.0e9,
        K_f=2.25e9,
        rho_s=2650.0,
        rho_f=1000.0,
    )
    print(material.summary())


    char_freq = compute_characteristic_frequencies(material, length_scale=1.0)
    print("\n  Characteristic frequencies:")
    for k, v in char_freq.items():
        print(f"    {k:20s} = {v:.6e}")




    print("\n[2] Generating 2D triangular mesh...")
    nx, ny = 11, 11
    nodes_fine, elements_fine = generate_structured_triangle_mesh(
        xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, nx=nx, ny=ny
    )

    nx, ny = 5, 5
    nodes, elements = generate_structured_triangle_mesh(
        xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, nx=nx, ny=ny
    )
    nodes6, elements6 = generate_quadratic_nodes(nodes, elements)
    n_nodes = nodes.shape[0]
    n_nodes6 = nodes6.shape[0]
    n_elements = elements.shape[0]
    print(f"  Linear mesh:    {n_nodes} nodes, {n_elements} elements")
    print(f"  Quadratic mesh: {n_nodes6} nodes")


    quality = mesh_quality_metrics(nodes, elements)
    print(f"  Mesh quality: mean={quality['quality_mean']:.4f}, "
          f"min={quality['quality_min']:.4f}, max_diam={quality['diameter_max']:.4f}")



    bc = identify_boundary_nodes(nodes6, 0.0, 1.0, 0.0, 1.0)
    bc_all = bc["all"]
    print(f"  Boundary nodes: {len(bc_all)}")




    print("\n[3] Placing seismic sources via Fibonacci spiral...")
    sources = fibonacci_spiral_points(n=5, radius=0.4, center=(0.5, 0.5))
    print(f"  Source locations: {sources}")




    print("\n[4] Computing CVT boundary sampling...")

    def density_func(s):

        return 1.0 + 5.0 * np.abs(s)

    g_cvt, energy_cvt, motion_cvt = cvt_lloyd_1d(
        n=7, it_num=10, s_num=200, density_func=density_func, init=2
    )
    print(f"  CVT generators: {g_cvt}")
    print(f"  Final energy:   {energy_cvt[-1]:.6e}")




    print("\n[5] Validating quadrature rules...")
    w, x = line_rule(0.0, 1.0, 5)

    quad_val = np.sum(w * x ** 4)
    exact_val = line_monomial_integral(0.0, 1.0, 4)
    print(f"  Quadrature of x^4 on [0,1]: {quad_val:.12f} (exact: {exact_val:.12f})")
    print(f"  Absolute error: {abs(quad_val - exact_val):.2e}")


    w_tri, xi_tri, eta_tri = triangle_rule(3)
    print(f"  Triangle rule weights sum: {np.sum(w_tri):.6f} (expected 0.5)")




    print("\n[6] Assembling global FEM matrices...")


    K_uu_p2, C_p2, M_p_p2, K_p_p2, M_uu_p2 = fem2d_biot_assemble(
        nodes6, elements6, elements, material, quad_order=3
    )
    print(f"  P2/P1 K_uu shape: {K_uu_p2.shape}, cond est: {estimate_condition_number(K_uu_p2):.4e}")
    print(f"  P2/P1 C shape:    {C_p2.shape}")
    print(f"  P2/P1 M_p shape:  {M_p_p2.shape}")


    K_uu, C, M_p, K_p, M_uu = fem2d_biot_assemble(
        nodes, elements, elements, material, quad_order=3
    )
    print(f"  P1/P1 K_uu shape: {K_uu.shape}, cond est: {estimate_condition_number(K_uu):.4e}")
    print(f"  P1/P1 C shape:    {C.shape}")
    print(f"  P1/P1 M_p shape:  {M_p.shape}")
    print(f"  P1/P1 K_p shape:  {K_p.shape}")




    print("\n[7] Solving quasi-static consolidation...")
    n_steps = 20
    t_final = 1.0
    dt = t_final / n_steps

    n_dof_u = 2 * n_nodes
    n_dof_p = n_nodes


    u = np.zeros(n_dof_u)
    p = np.zeros(n_dof_p)


    center_node = (ny // 2) * nx + (nx // 2)
    F_p_base = np.zeros(n_dof_p)
    F_p_base[center_node] = 1.0e-6

    F_u = np.zeros(n_dof_u)


    bc_linear = identify_boundary_nodes(nodes, 0.0, 1.0, 0.0, 1.0)
    bc_all_linear = bc_linear["all"]


    reg_scale = 1e-6 * np.max(np.diag(K_uu))
    K_uu_reg = K_uu + reg_scale * np.eye(K_uu.shape[0])
    K_uu_bc, F_u_bc = apply_dirichlet_bc(K_uu_reg, F_u, bc_all_linear, 0.0, ndof_per_node=2)


    p_history = np.zeros((n_steps + 1, n_dof_p))
    u_history = np.zeros((n_steps + 1, n_dof_u))
    p_history[0, :] = p
    u_history[0, :] = u

    print(f"  Time step dt={dt:.4e}, n_steps={n_steps}")

    for step in range(1, n_steps + 1):
        t = step * dt

        F_p = F_p_base * np.sin(np.pi * t / t_final) ** 2


        rhs_u = F_u_bc - C @ p

        for node in bc_all_linear:
            rhs_u[2 * node] = 0.0
            rhs_u[2 * node + 1] = 0.0


        try:
            u_new = np.linalg.solve(K_uu_bc, rhs_u)
        except (np.linalg.LinAlgError, ValueError):
            reg = 1e-6 * np.eye(n_dof_u)
            u_new = np.linalg.lstsq(K_uu_bc + reg, rhs_u, rcond=None)[0]


        for node in bc_all_linear:
            u_new[2 * node] = 0.0
            u_new[2 * node + 1] = 0.0








        A_p = None
        rhs_p = None
        try:
            p_new = np.linalg.solve(A_p, rhs_p)
        except (np.linalg.LinAlgError, ValueError):
            reg = 1e-6 * np.eye(n_dof_p)
            p_new = np.linalg.lstsq(A_p + reg, rhs_p, rcond=None)[0]

        u = u_new
        p = p_new
        u_history[step, :] = u
        p_history[step, :] = p

    print(f"  Final max pressure: {np.max(p):.4e} Pa")
    print(f"  Final max displacement: {np.max(np.abs(u)):.4e} m")
    if not np.isfinite(np.max(p)) or not np.isfinite(np.max(u)):
        print("  WARNING: Non-finite values detected. Replacing with zeros for analysis.")
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        u_history = np.nan_to_num(u_history, nan=0.0, posinf=0.0, neginf=0.0)
        p_history = np.nan_to_num(p_history, nan=0.0, posinf=0.0, neginf=0.0)




    print("\n[8] Reconstructing pressure profile with Hermite interpolation...")

    j_mid = ny // 2
    line_nodes = [j_mid * nx + i for i in range(nx)]
    x_line = nodes[line_nodes, 0]
    p_line = p[line_nodes]

    dp_line = np.zeros(nx)
    dp_line[1:-1] = (p_line[2:] - p_line[:-2]) / (x_line[2:] - x_line[:-2])
    dp_line[0] = (p_line[1] - p_line[0]) / (x_line[1] - x_line[0])
    dp_line[-1] = (p_line[-1] - p_line[-2]) / (x_line[-1] - x_line[-2])

    x_eval = np.linspace(0.0, 1.0, 41)
    p_recon, dp_recon = reconstruct_wave_field_1d(x_line, p_line, dp_line, x_eval)
    recon_error = np.max(np.abs(p_recon - np.interp(x_eval, x_line, p_line)))
    print(f"  Hermite reconstruction max deviation from linear interp: {recon_error:.4e}")




    print("\n[9] Analyzing fast/slow P-wave separation...")
    u_2d = u.reshape((n_nodes, 2))
    fast_mask, slow_mask, ratio = separate_fast_slow_waves(
        p, u_2d, nodes, material, dt, dx_est=1.0 / (nx - 1)
    )
    print(f"  Fast-wave dominated nodes: {np.sum(fast_mask)}")
    print(f"  Slow-wave dominated nodes: {np.sum(slow_mask)}")
    print(f"  Mean p/u ratio: {np.mean(ratio):.4e}")




    print("\n[10] Computing dispersion relations and quality factors...")
    omega_vals = np.logspace(-2, 4, 20)
    v_fast_arr, v_slow_arr, alpha_fast_arr, alpha_slow_arr = biot_dispersion_relation(
        omega_vals, material
    )
    Q_fast, Q_slow = compute_quality_factor(omega_vals, material)
    print(f"  Low-freq fast wave velocity: {np.real(v_fast_arr[0]):.4f} m/s")
    print(f"  Low-freq slow wave velocity: {np.real(v_slow_arr[0]):.4f} m/s")
    print(f"  Quality factor Q_fast (low-freq): {Q_fast[0]:.2f}")
    print(f"  Quality factor Q_slow (low-freq): {Q_slow[0]:.2f}")




    print("\n[11] Computing energy spectrum statistics...")
    energy_stats = energy_spectrum_bins(p, u_2d, material, bin_num=16)
    print(f"  Mean energy density:  {energy_stats['energy_mean']:.4e} J/m³")
    print(f"  Total energy:         {energy_stats['total_energy']:.4e} J")
    print(f"  Energy skewness:      {energy_stats['skewness']:.4f}")
    print(f"  Energy kurtosis:      {energy_stats['kurtosis']:.4f}")


    time_array = np.linspace(0.0, t_final, n_steps + 1)
    wf_stats = analyze_wave_front_histogram(p_history, time_array, bin_num=12)
    print(f"  Max pressure over time range: [{wf_stats['max_pressure'].min():.4e}, "
          f"{wf_stats['max_pressure'].max():.4e}] Pa")




    print("\n[12] K-means clustering for lithological zonation...")
    zones, centers = cluster_velocity_zones(nodes, u_2d, k=3)
    for z in range(3):
        count = np.sum(zones == z)
        print(f"  Zone {z}: {count} nodes, center=({centers[z,0]:.3f}, {centers[z,1]:.3f}, |v|={centers[z,2]:.3e})")




    print("\n[13] Building adjacency matrix and HB format output...")
    adj = build_adjacency_matrix(elements, n_nodes=n_nodes)
    print(f"  Adjacency matrix shape: {adj.shape}, nonzero entries: {np.count_nonzero(adj)}")

    hb_filename = os.path.join(os.path.dirname(__file__), "poroelastic_matrix.hb")
    write_hb_format(hb_filename, K_uu[:min(50, K_uu.shape[0]), :min(50, K_uu.shape[1])],
                    title="PoroelasticStiffness", key="KUU001")
    print(f"  Written HB format to: {hb_filename}")




    print("\n[14] Validating 3D tetrahedral basis functions...")
    tet = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    vol = tetrahedron_volume(tet)
    print(f"  Reference tetrahedron volume/6: {vol:.6f} (expected 1.0)")


    p_cent = np.array([[0.25], [0.25], [0.25]])
    phi_cent = basis_mn_tet4(tet, 1, p_cent)
    print(f"  Basis at centroid: {phi_cent.flatten()}, sum={np.sum(phi_cent):.6f} (expected 1.0)")




    print("\n[15] Verifying exponential integrator...")
    t_ode, y_ode = exponential_integrator_exact(alpha=-0.5, t0=0.0, y0=2.0, tstop=5.0, n_steps=50)
    y_exact = 2.0 * np.exp(-0.5 * (t_ode - 0.0))
    max_err = np.max(np.abs(y_ode - y_exact))
    print(f"  Exponential integrator max error: {max_err:.2e} (expected ~0)")




    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Domain:          [0,1] x [0,1] m²")
    print(f"  Mesh:            {nx}x{ny} grid, {n_elements} triangles")
    print(f"  Material:        Sandstone-like, phi={material.phi:.2f}")
    print(f"  Time steps:      {n_steps}, dt={dt:.4e} s")
    print(f"  Final pressure:  {np.max(p):.4e} Pa")
    print(f"  Final |u|:       {np.max(np.abs(u)):.4e} m")
    print(f"  Fast Vp:         {material.V_p_fast:.2f} m/s")
    print(f"  Slow Vp:         {material.V_p_slow:.4f} m/s")
    print(f"  Shear Vs:        {material.V_s:.2f} m/s")
    print(f"  P2/P1 cond:      {estimate_condition_number(K_uu_p2):.4e}")
    print("=" * 70)

    return {
        "material": material,
        "nodes": nodes,
        "elements": elements,
        "pressure": p,
        "displacement": u,
        "p_history": p_history,
        "u_history": u_history,
        "energy_stats": energy_stats,
        "zones": zones,
        "fast_mask": fast_mask,
        "slow_mask": slow_mask,
    }


if __name__ == "__main__":
    result = main()
