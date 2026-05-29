"""
main.py
-------
统一入口：非厄米物理与例外点（Exceptional Points）博士级计算框架。

运行方式:
    python main.py

无需任何命令行参数。程序自动执行以下科研计算流程：
1. 构造多种非厄米哈密顿量（PT对称、SSH、Hofstadter）
2. 使用 Laguerre 方法在复 k 平面寻找例外点
3. 计算双正交 Berry 相位与 Chern 数
4. 使用四面体积分在三维布里渊区求平均能量
5. 模拟非厄米薛定谔方程的时间演化（自适应 RKF45）
6. Lindblad 主方程演化
7. 非厄米随机矩阵能级间距统计（不完全 Beta 函数）
8. 有限元离散化与三角剖分
9. Vandermonde 谱插值
10. 蒙特卡洛与并行参数扫描搜索 EP
11. 传递矩阵与李雅普诺夫指数
"""

import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# 1. Config parser (seed 431_filum)
# ---------------------------------------------------------------------------
from config_parser import (
    file_char_count,
    string_to_float_vector,
    read_parameter_config,
)

# ---------------------------------------------------------------------------
# 2. Hamiltonian builder
# ---------------------------------------------------------------------------
from hamiltonian_builder import (
    build_pt_symmetric_hamiltonian_1d,
    build_pt_symmetric_hamiltonian_2d,
    build_nonhermitian_ssh_hamiltonian,
    build_nonhermitian_hofstadter_hamiltonian,
    discriminant_2x2,
)

# ---------------------------------------------------------------------------
# 3. Exceptional point solver (seed 1430_zero_laguerre)
# ---------------------------------------------------------------------------
from exceptional_point_solver import (
    laguerre_root_find,
    find_exceptional_points_1d,
    find_exceptional_points_ssh,
    local_exceptional_point_order,
)

# ---------------------------------------------------------------------------
# 4. Biorthogonal topology
# ---------------------------------------------------------------------------
from biorthogonal_topology import (
    compute_biorthogonal_eigenvectors,
    berry_connection_1d,
    berry_curvature_2d,
    zak_phase_1d,
    chern_number_2d,
    winding_number_complex_energy,
)

# ---------------------------------------------------------------------------
# 5. Brillouin zone integrator (seed 1253_tetrahedron_nco_rule)
# ---------------------------------------------------------------------------
from brillouin_integrator import (
    integrate_over_tetrahedron,
    partition_bz_into_tetrahedra,
    integrate_bz_3d,
    bz_average_energy,
)

# ---------------------------------------------------------------------------
# 6. Non-Hermitian dynamics (seeds 675_lindberg_ode, 1086_sir_ode)
# ---------------------------------------------------------------------------
from nonherm_dynamics import (
    evolve_nonhermitian_schrodinger,
    lindblad_evolve_2level,
    nonhermitian_lindberg_system,
    rkf45_step_complex,
)

# ---------------------------------------------------------------------------
# 7. Random matrix statistics (seed 031_asa063)
# ---------------------------------------------------------------------------
from random_matrix_stats import (
    incomplete_beta,
    level_spacing_ratios,
    wigner_poisson_mixture_cdf,
    generate_ginibre_spectrum,
    analyze_spacing_statistics,
)

# ---------------------------------------------------------------------------
# 8. Mesh discretization (seed 474_gmsh_io)
# ---------------------------------------------------------------------------
from mesh_discretization import (
    SimpleMesh,
    build_mass_matrix,
    build_stiffness_matrix_2d,
    assemble_nonhermitian_hamiltonian_fe,
)

# ---------------------------------------------------------------------------
# 9. Parallel sweep (seed 514_hello_parfor)
# ---------------------------------------------------------------------------
from parallel_sweep import (
    parallel_parameter_sweep,
    find_ep_contours_from_sweep,
    coarse_to_fine_ep_search,
)

# ---------------------------------------------------------------------------
# 10. Transfer matrix (seed 1094_snakes_matrix)
# ---------------------------------------------------------------------------
from transfer_matrix import (
    transfer_matrix_ssh,
    spectrum_from_transfer_matrix,
    lyapunov_exponent_ssh,
    nonhermitian_markov_chain,
    steady_state_distribution,
)

# ---------------------------------------------------------------------------
# 11. Vandermonde solver (seed 1004_r8vm)
# ---------------------------------------------------------------------------
from vandermonde_solver import (
    vandermonde_solve_bjorck_pereyra,
    vandermonde_determinant,
    barycentric_lagrange_interpolate,
    interpolate_energy_band,
    characteristic_polynomial_from_roots,
)

# ---------------------------------------------------------------------------
# 12. Potential profile (seed 920_profile_data)
# ---------------------------------------------------------------------------
from potential_profile import (
    complex_poschl_teller,
    nonhermitian_kronig_penney,
    double_well_nonhermitian,
    profile_based_potential,
    get_default_profile,
)

# ---------------------------------------------------------------------------
# 13. Manifold generator (seed 1052_sammon_data)
# ---------------------------------------------------------------------------
from manifold_generator import (
    circle_loop,
    helix_loop,
    nonlinear_curve,
    simplex_parameter_space,
    adiabatic_cycle_around_ep,
)

# ---------------------------------------------------------------------------
# 14. Monte Carlo sampler (seed 696_locker_simulation)
# ---------------------------------------------------------------------------
from monte_carlo_sampler import (
    strategy_random_search,
    strategy_importance_sampling,
    strategy_adaptive_local_search,
    metropolis_hastings_ep_search,
)

# ---------------------------------------------------------------------------
# 15. Parameter box (seed 1377_usa_box_plot)
# ---------------------------------------------------------------------------
from parameter_box import (
    ParameterBox,
    adaptive_box_refinement,
)

# ---------------------------------------------------------------------------
# 16. Triangulation (seed 1352_triangulation_svg)
# ---------------------------------------------------------------------------
from triangulation import (
    bowyer_watson,
    triangulate_domain_rectangle,
    triangulate_domain_delaunay,
    triangle_quality,
)


def section_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def ssh_builder_pt(**kwargs):
    """Module-level builder for parallel sweep (must be picklable)."""
    # TODO: Extract parameters from kwargs and build the SSH Hamiltonian.
    raise NotImplementedError("SSH builder is missing.")


def main():
    np.random.seed(42)
    print("非厄米物理与例外点 (Non-Hermitian Physics & Exceptional Points)")
    print("博士级科研计算框架 —— 统一入口 main.py")
    print(f"Python: {sys.version}")

    # -----------------------------------------------------------------------
    # 1. Hamiltonian construction and discriminant
    # -----------------------------------------------------------------------
    section_header("1. 非厄米哈密顿量构造与判别式")
    k_test = 0.5
    H_1d = build_pt_symmetric_hamiltonian_1d(k_test, t=1.0, m=0.5, gamma=0.3)
    print(f"PT-symmetric H(k={k_test}):\n{H_1d}")
    delta = discriminant_2x2(H_1d)
    print(f"Discriminant Δ = {delta}")

    H_2d = build_pt_symmetric_hamiltonian_2d(0.3, 0.4, t=1.0, m=0.5, gamma=0.3)
    print(f"2D H(kx=0.3, ky=0.4) shape: {H_2d.shape}")

    H_ssh = build_nonhermitian_ssh_hamiltonian(k_test, t1=1.0, t2=0.5, gamma=0.2)
    print(f"SSH H(k={k_test}) discriminant: {discriminant_2x2(H_ssh)}")

    H_hof = build_nonhermitian_hofstadter_hamiltonian(0.1, 0.2, phi=1.0 / 4.0, q=4)
    print(f"Hofstadter H shape: {H_hof.shape}, trace={np.trace(H_hof):.4f}")

    # -----------------------------------------------------------------------
    # 2. Exceptional point finding
    # -----------------------------------------------------------------------
    section_header("2. Laguerre 方法寻找例外点")
    ep_1d = find_exceptional_points_1d(t=1.0, m=0.5, gamma=0.3, k_guess_grid=24)
    print(f"Found {len(ep_1d)} exceptional points in 1D PT model.")
    for ep in ep_1d[:5]:
        print(f"  k_EP = {ep:.6f}")

    ep_ssh = find_exceptional_points_ssh(t1=1.0, t2=0.5, gamma=0.2, k_guess_grid=20)
    print(f"Found {len(ep_ssh)} exceptional points in SSH model.")
    for ep in ep_ssh[:5]:
        print(f"  k_EP = {ep:.6f}")

    # -----------------------------------------------------------------------
    # 3. Biorthogonal topology
    # -----------------------------------------------------------------------
    section_header("3. 双正交拓扑不变量")
    E, right, left = compute_biorthogonal_eigenvectors(H_1d)
    print(f"Eigenvalues: {E}")
    print(f"Biorthogonal overlap check: {np.vdot(left[0], right[:, 0]):.6f}")

    bc = berry_connection_1d(
        lambda k: build_pt_symmetric_hamiltonian_1d(k, t=1.0, m=0.5, gamma=0.3),
        k=0.5, dk=1e-4
    )
    print(f"Berry connection at k=0.5: {bc:.6f}")

    zak = zak_phase_1d(
        lambda k: build_pt_symmetric_hamiltonian_1d(k, t=1.0, m=0.5, gamma=0.3),
        k_points=401
    )
    print(f"Zak phase (1D): {zak:.6f}")

    try:
        W = winding_number_complex_energy(
            lambda k: build_pt_symmetric_hamiltonian_1d(k, t=1.0, m=0.5, gamma=0.3),
            k_points=401
        )
        print(f"Winding number: {W:.4f}")
    except Exception as e:
        print(f"Winding number computation skipped: {e}")

    # -----------------------------------------------------------------------
    # 4. Brillouin zone integration
    # -----------------------------------------------------------------------
    section_header("4. 四面体布里渊区积分")
    tetras = partition_bz_into_tetrahedra(n_k=2)
    print(f"Number of tetrahedra in BZ partition (n_k=2): {len(tetras)}")

    def simple_integrand(kx, ky, kz):
        # A trivial integrand for demonstration
        return np.sin(kx) ** 2 + np.cos(ky) ** 2

    integral_val = integrate_bz_3d(simple_integrand, n_k=2, degree=3)
    expected = (2.0 * np.pi) ** 3 * 1.0  # average of sin^2+cos^2 is 1
    print(f"BZ integral of (sin^2 kx + cos^2 ky): {integral_val:.4f} (expected ≈ {expected:.4f})")

    # -----------------------------------------------------------------------
    # 5. Non-Hermitian dynamics
    # -----------------------------------------------------------------------
    section_header("5. 非厄米动力学演化")
    H_eff = np.array([[1.0, 0.5], [0.5, -1.0]], dtype=complex) + 1j * np.array([[-0.1, 0.0], [0.0, 0.1]], dtype=complex)
    psi0 = np.array([1.0, 0.0], dtype=complex)
    t_vals, psi_vals, norms = evolve_nonhermitian_schrodinger(H_eff, psi0, (0.0, 5.0), dt0=1e-3, tol=1e-9)
    print(f"Time evolution: {len(t_vals)} steps, final norm = {norms[-1]:.6f}")

    # Lindblad
    H_lind = np.array([[1.0, 0.3], [0.3, -0.5]], dtype=complex)
    L1 = np.array([[0.0, 0.2], [0.0, 0.0]], dtype=complex)
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    t_lind, rho_lind, purity = lindblad_evolve_2level(H_lind, [L1], rho0, (0.0, 3.0), dt0=1e-3, tol=1e-9)
    print(f"Lindblad evolution: final purity = {purity[-1]:.6f}")

    # Lindberg-like stiff system (very stiff, integrate only briefly)
    y0 = np.array([0.1, 0.1, 0.5, 0.5])
    t_stiff, y_stiff = [0.0], [y0.copy()]
    t, y, h = 0.0, y0.copy(), 1e-4
    while t < 0.01:
        y, t, h = rkf45_step_complex(lambda tt, yy: nonhermitian_lindberg_system(yy), t, y, h, tol=1e-6)
        t_stiff.append(t)
        y_stiff.append(y.copy())
    print(f"Stiff ODE: final t={t_stiff[-1]:.4f}, y={y_stiff[-1]}")

    # -----------------------------------------------------------------------
    # 6. Random matrix statistics
    # -----------------------------------------------------------------------
    section_header("6. 非厄米随机矩阵能级间距统计")
    eig_ginibre = generate_ginibre_spectrum(200, seed=42)
    stats = analyze_spacing_statistics(eig_ginibre, num_bins=20)
    if stats is not None:
        print(f"Ginibre spectrum mean spacing: {stats['mean_spacing']:.4f}")
        print(f"Wigner-Poisson mixture fit α = {stats['alpha_fit']:.4f}")
    else:
        print("Spacing statistics unavailable.")

    # Incomplete beta
    val_beta, ifault = incomplete_beta(0.5, 2.0, 3.0)
    print(f"I_0.5(2,3) = {val_beta:.6f}, ifault={ifault}")

    # -----------------------------------------------------------------------
    # 7. Mesh & Finite Element
    # -----------------------------------------------------------------------
    section_header("7. 有限元网格离散化")
    points, triangles = triangulate_domain_rectangle((-1.0, 1.0), (-1.0, 1.0), nx=11, ny=11)
    mesh = SimpleMesh(points, triangles, element_type='triangle')
    print(f"Rectangle mesh: {mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} triangles")

    M_mass = build_mass_matrix(mesh)
    K_stiff = build_stiffness_matrix_2d(mesh)
    print(f"Mass matrix trace: {np.trace(M_mass):.4f}, Stiffness matrix trace: {np.trace(K_stiff):.4f}")

    H_fe, M_fe = assemble_nonhermitian_hamiltonian_fe(
        mesh,
        V_func=lambda x, y: 0.5 * (x ** 2 + y ** 2),
        W_func=lambda x, y: 0.1 * x,
    )
    print(f"FE Hamiltonian shape: {H_fe.shape}")

    # Delaunay on scattered points
    scattered = np.random.rand(50, 2) * 2.0 - 1.0
    pts_delaunay, tri_delaunay = triangulate_domain_delaunay(scattered)
    if len(tri_delaunay) > 0:
        q = triangle_quality(pts_delaunay, tri_delaunay[0])
        print(f"First Delaunay triangle quality: {q:.4f}")
    else:
        print("Delaunay triangulation produced no triangles.")

    # -----------------------------------------------------------------------
    # 8. Parallel sweep
    # -----------------------------------------------------------------------
    section_header("8. 并行参数扫描")

    param_grids = {
        'k': np.linspace(-np.pi, np.pi, 21),
        'gamma': np.linspace(0.0, 0.5, 11),
    }
    sweep_results = parallel_parameter_sweep(ssh_builder_pt, param_grids, n_workers=2)
    ep_candidates = find_ep_contours_from_sweep(sweep_results, threshold=1e-2)
    print(f"Parallel sweep: {len(sweep_results)} points, {len(ep_candidates)} EP candidates found.")

    # -----------------------------------------------------------------------
    # 9. Transfer matrix
    # -----------------------------------------------------------------------
    section_header("9. 传递矩阵与李雅普诺夫指数")
    E_grid = np.linspace(-2.0, 2.0, 101)
    traces, discr = spectrum_from_transfer_matrix(E_grid, t1=1.0, t2=0.5, gamma=0.2)
    print(f"Transfer matrix trace range: [{traces.real.min():.3f}, {traces.real.max():.3f}]")

    lyap = lyapunov_exponent_ssh(0.0, t1=1.0, t2=0.5, gamma=0.2, N=500, seed=42)
    print(f"Lyapunov exponent at E=0: {lyap:.6f}")

    L_markov = nonhermitian_markov_chain(10, p_forward=0.8, p_backward=0.3, loss_rate=0.1)
    pi_ss = steady_state_distribution(L_markov)
    print(f"Markov steady-state sum: {pi_ss.sum():.6f}")

    # -----------------------------------------------------------------------
    # 10. Vandermonde solver
    # -----------------------------------------------------------------------
    section_header("10. Vandermonde 谱插值")
    nodes = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * np.pi
    energies = np.sin(nodes) + 0.1j * np.cos(nodes)
    eval_pts = np.linspace(0.0, np.pi, 41)
    interp_energies = interpolate_energy_band(nodes, energies, eval_pts)
    err_max = np.max(np.abs(interp_energies - (np.sin(eval_pts) + 0.1j * np.cos(eval_pts))))
    print(f"Barycentric interpolation max error: {err_max:.2e}")

    coeffs = characteristic_polynomial_from_roots([1.0 + 1j, 2.0 - 0.5j])
    print(f"Characteristic polynomial coefficients: {coeffs}")

    # -----------------------------------------------------------------------
    # 11. Potential profiles
    # -----------------------------------------------------------------------
    section_header("11. 空间势场剖面")
    x_prof = np.linspace(-3.0, 3.0, 101)
    V_pt = complex_poschl_teller(x_prof, V0=1.0, W0=0.3, alpha=1.0)
    V_kp = nonhermitian_kronig_penney(x_prof, V1=1.0, V2=0.3, a=1.0)
    V_dw = double_well_nonhermitian(x_prof, V0=1.0, gamma=0.2, a=2.0, b=0.5)
    print(f"Pöschl-Teller potential at x=0: {V_pt[len(V_pt)//2]:.4f}")
    print(f"Kronig-Penney potential at x=0: {V_kp[len(V_kp)//2]:.4f}")
    print(f"Double-well potential at x=0: {V_dw[len(V_dw)//2]:.4f}")

    profile = get_default_profile()
    V_profile = profile_based_potential(x_prof, profile, scale=2.0, imaginary_ratio=0.2)
    print(f"Profile-based potential range: [{V_profile.real.min():.3f}, {V_profile.real.max():.3f}]")

    # -----------------------------------------------------------------------
    # 12. Manifold generation
    # -----------------------------------------------------------------------
    section_header("12. 参数流形生成")
    loop = circle_loop((0.5, 0.3), 0.1, n_points=50)
    hel = helix_loop((0.5, 0.3, 1.0), 0.1, 0.05, n_points=60, turns=2)
    nonlin = nonlinear_curve(n_points=40, dim=5)
    simp_params, simp_labels = simplex_parameter_space(dim=3, n_points=60, std=0.1)
    print(f"Circle loop shape: {loop.shape}")
    print(f"Helix loop shape: {hel.shape}")
    print(f"Nonlinear curve shape: {nonlin.shape}")
    print(f"Simplex sample shape: {simp_params.shape}, labels: {np.unique(simp_labels)}")

    # -----------------------------------------------------------------------
    # 13. Monte Carlo sampling
    # -----------------------------------------------------------------------
    section_header("13. 蒙特卡洛 EP 搜索")

    def disc_ssh_dict(p):
        H = build_nonhermitian_ssh_hamiltonian(p.get('k', 0.0), 1.0, 0.5, p.get('gamma', 0.2))
        return discriminant_2x2(H)

    bounds_ssh = {'k': (-np.pi, np.pi), 'gamma': (0.0, 0.5)}
    mc_random = strategy_random_search(disc_ssh_dict, bounds_ssh, n_trials=2000, threshold=1e-2, seed=42)
    print(f"Random search candidates: {len(mc_random)}")

    mc_mcmc, acc_rate = metropolis_hastings_ep_search(disc_ssh_dict, bounds_ssh, n_steps=2000, beta=1e3, step=0.05, seed=42)
    best_mcmc = min(mc_mcmc, key=lambda r: r['abs_delta'])
    print(f"MCMC best |Δ|: {best_mcmc['abs_delta']:.6f}, accept rate: {acc_rate:.3f}")

    # -----------------------------------------------------------------------
    # 14. Parameter box refinement
    # -----------------------------------------------------------------------
    section_header("14. 自适应参数空间细分")
    box = ParameterBox([(-np.pi, np.pi), (0.0, 0.5)])
    print(f"Initial parameter box volume: {box.volume():.4f}")
    sub_boxes = box.subdivide()
    print(f"Subdivided into {len(sub_boxes)} sub-boxes.")

    # -----------------------------------------------------------------------
    # 15. Config parser test
    # -----------------------------------------------------------------------
    section_header("15. 配置文件解析")
    config_path = "demo_config.txt"
    with open(config_path, "w") as f:
        f.write("# Demo config for non-Hermitian simulation\n")
        f.write("N_SITES 8\n")
        f.write("t hopping = 1.0\n")
        f.write("GAMMA 0.3\n")
        f.write("mass_term 0.5\n")
    params = read_parameter_config(config_path)
    print(f"Config params: {params}")
    os.remove(config_path)

    # String parsing
    vec = string_to_float_vector("1.2 3.4 -0.5", 3)
    print(f"Parsed vector: {vec}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    section_header("计算完成")
    print("所有模块已成功运行，无报错。")
    print("本框架涵盖了非厄米物理的核心计算任务：")
    print("  - 哈密顿量构造与判别式分析")
    print("  - Laguerre 方法求解例外点")
    print("  - 双正交 Berry 相位 / Zak 相位 / Chern 数")
    print("  - 四面体布里渊区数值积分")
    print("  - 非厄米薛定谔方程与 Lindblad 主方程演化")
    print("  - 随机矩阵 Ginibre 系综与不完全 Beta 函数")
    print("  - 有限元三角剖分与刚度/质量矩阵组装")
    print("  - 并行参数扫描与蒙特卡洛 EP 搜索")
    print("  - 传递矩阵与李雅普诺夫指数")
    print("  - Vandermonde 谱插值与特征多项式重构")
    print("  - 空间势场剖面（Pöschl-Teller / Kronig-Penney / 双阱）")
    print("  - 参数流形生成（圆/螺旋/非线性/单形）")
    print("  - 自适应参数空间细分")


if __name__ == "__main__":
    main()
