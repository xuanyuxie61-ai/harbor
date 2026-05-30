
import numpy as np
import time

from spin_lattice import PyrochloreLattice, PrimeFrustratedLattice, connected_components_2d_spin_map
from exchange_matrix import (
    exchange_laplacian_1d,
    exchange_laplacian_2d,
    exchange_energy,
    apply_exchange_operator,
    add_disorder,
)
from spin_quaternion import (
    random_spin_quaternion,
    q_to_spin_vector,
    spin_vector_to_q,
)
from energy_landscape import (
    greedy_relaxation,
    simulated_annealing_spin_glass,
    line_search_spin_rotation,
)
from eigen_analysis import (
    power_method,
    inverse_iteration,
    spectral_gap_and_soft_modes,
    spin_wave_dispersion_1d,
    correlation_length_from_gap,
)
from spin_dynamics import (
    euler_integrate_llg,
    trapezoidal_integrate_llg,
    integrate_brusselator_pump,
    domain_wall_magnetization,
    compute_magnetization_trajectory,
)
from domain_analysis import (
    extract_domain_statistics,
    spin_orientation_histogram,
    radial_distribution_function_2d,
    entropy_rate_from_trajectory,
    analyze_triangle_spin_distribution,
)
from adaptive_fem_solver import adaptive_fem_order_parameter
from utils import (
    histogram_stats_1d,
    triangle_area_histogram_2d,
    skyline_mv,
    build_skyline_from_tridiagonal,
    safe_divide,
    rms_norm,
    EPS_MACHINE,
)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(label: str, value):
    print(f"  {label:<40s}: {value}")


def main():
    np.random.seed(42)
    start_time = time.time()




    print_header("STEP 1: 晶格构造 (Pyrochlore + Prime-Frustrated)")
    lattice_3d = PyrochloreLattice(L=3, J1=1.0, J2=0.2, disorder_std=0.2)
    lattice_2d = PrimeFrustratedLattice(L=12, J0=1.0, alpha=0.3, seed=42)
    print_result("3D Pyrochlore sites", lattice_3d.N)
    print_result("2D Prime-Frustrated sites", lattice_2d.N)
    print_result("3D bonds count", len(lattice_3d.bonds))




    print_header("STEP 2: 交换矩阵与离散拉普拉斯算子")
    J_3d = lattice_3d.J
    J_2d = lattice_2d.to_full_matrix()
    J_2d = add_disorder(J_2d, std=0.15, seed=123)


    L1d_periodic = exchange_laplacian_1d(n=64, h=0.1, bc="periodic")
    print_result("1D Laplacian shape", L1d_periodic.shape)
    print_result("1D Laplacian max eigenvalue (Gershgorin)", float(np.max(np.sum(np.abs(L1d_periodic), axis=1))))


    L1d_dir = exchange_laplacian_1d(n=64, h=0.1, bc="dirichlet")
    na, diag_idx, a_sky = build_skyline_from_tridiagonal(
        np.diag(L1d_dir, -1), np.diag(L1d_dir), np.diag(L1d_dir, 1)
    )
    x_test = np.random.randn(64)
    y_sky = skyline_mv(64, diag_idx, a_sky, x_test)
    y_full = L1d_dir @ x_test
    print_result("Skyline vs Full matvec error", float(np.max(np.abs(y_sky - y_full))))




    print_header("STEP 3: 四元数自旋初始化")
    N_3d = lattice_3d.N
    spins_3d = np.zeros((N_3d, 3), dtype=float)
    for i in range(N_3d):
        q = random_spin_quaternion(seed=i)
        spins_3d[i] = q_to_spin_vector(q)
    print_result("Initial spin norm mean", float(np.mean(np.linalg.norm(spins_3d, axis=1))))
    print_result("Initial total magnetization Mz", float(np.mean(spins_3d[:, 2])))




    print_header("STEP 4: 能量景观优化 (Simulated Annealing + Greedy Relaxation)")
    spins_best, e_best, sa_history = simulated_annealing_spin_glass(
        J_3d, spins_3d, T_init=3.0, T_final=1e-3, cooling_rate=0.99, steps_per_T=50
    )
    print_result("SA best energy", e_best)
    print_result("SA steps recorded", len(sa_history))

    spins_relaxed, e_relaxed, relax_history = greedy_relaxation(
        J_3d, spins_best, n_sweeps=8, tol=1e-7
    )
    print_result("Greedy relaxed energy", e_relaxed)
    print_result("Greedy sweeps performed", len(relax_history))


    axis_test = np.array([0.0, 0.0, 1.0])
    theta_opt, s_new, e_min = line_search_spin_rotation(
        J_3d, spins_relaxed, site_idx=0, axis=axis_test
    )
    print_result("Brent line-search optimal angle (rad)", theta_opt)
    print_result("Brent line-search energy", e_min)




    print_header("STEP 5: 特征值谱与自旋波色散")

    lam_min, gap, eigs, eigvecs = spectral_gap_and_soft_modes(J_2d, n_soft=3)
    print_result("Min eigenvalue", lam_min)
    print_result("Spectral gap", gap)
    print_result("Correlation length (MF)", correlation_length_from_gap(gap, J=1.0, a=1.0))


    lam_max, v_max, iters = power_method(np.abs(J_2d), max_iter=300, tol=1e-10)
    print_result("Power method dominant eigenvalue", lam_max)
    print_result("Power method iterations", iters)


    lam_soft, v_soft, iters_inv = inverse_iteration(J_2d, shift=0.0, max_iter=200)
    print_result("Inverse iteration soft eigenvalue", lam_soft)
    print_result("Inverse iteration iterations", iters_inv)


    k_pts = np.linspace(0, np.pi, 50)
    omega_sw = spin_wave_dispersion_1d(J=1.0, S=1.0, a=1.0, k_points=k_pts)
    print_result("Spin wave max frequency", float(np.max(omega_sw)))




    print_header("STEP 6: LLG 自旋动力学 (Euler + Trapezoidal)")

    t_arr_eu, traj_eu = euler_integrate_llg(
        J_3d, spins_relaxed, t_span=(0.0, 2.0), n_steps=400, gamma=1.0, alpha=0.15
    )
    Mx_eu, My_eu, Mz_eu = compute_magnetization_trajectory(traj_eu)
    print_result("Euler Mz final", float(Mz_eu[-1]))
    print_result("Euler Mz RMS norm", rms_norm(Mz_eu))

    t_arr_tr, traj_tr = trapezoidal_integrate_llg(
        J_3d, spins_relaxed, t_span=(0.0, 2.0), n_steps=100, gamma=1.0, alpha=0.15
    )
    Mx_tr, My_tr, Mz_tr = compute_magnetization_trajectory(traj_tr)
    print_result("Trapezoidal Mz final", float(Mz_tr[-1]))




    print_header("STEP 7: Brusselator 非线性自旋泵")
    t_pump, y_pump = integrate_brusselator_pump(a=1.0, b=3.0, n_steps=2000)
    print_result("Pump u final", float(y_pump[-1, 0]))
    print_result("Pump v final", float(y_pump[-1, 1]))




    print_header("STEP 8: Fisher-KPP 磁畴壁行波解")
    x_wall = np.linspace(-10.0, 10.0, 200)
    Mz_wall, dMz_dt, dMz_dx, d2Mz_dx2 = domain_wall_magnetization(t=1.0, x=x_wall, Ms=1.0)
    print_result("Domain wall center Mz", float(Mz_wall[len(Mz_wall) // 2]))
    print_result("Domain wall width proxy", float(np.sum(np.abs(dMz_dx) > 0.01)))




    print_header("STEP 9: 磁畴分析与统计直方图")

    L_proj = int(np.sqrt(N_3d))
    if L_proj * L_proj <= N_3d:
        spin_map = spins_relaxed[: L_proj * L_proj, 2].reshape((L_proj, L_proj))
    else:
        spin_map = spins_relaxed[:N_3d, 2].reshape((1, N_3d))

    dom_stats = extract_domain_statistics(spin_map, threshold=0.3)
    print_result("Number of domains", dom_stats["n_domains"])
    print_result("Max domain size", dom_stats["max_domain_size"])
    print_result("Mean domain size", dom_stats["mean_domain_size"])
    print_result("Domain size entropy", dom_stats["domain_size_entropy"])


    counts, centers, orient_stats = spin_orientation_histogram(spins_relaxed, n_bins=12)
    print_result("Polar angle mean (rad)", orient_stats["mean"])
    print_result("Polar angle variance", orient_stats["variance"])


    spins_xy = spins_relaxed[:, :2]
    norms = np.linalg.norm(spins_xy, axis=1, keepdims=True) + EPS_MACHINE
    spins_xy_tri = np.abs(spins_xy) / norms
    histo_tri, info_tri = analyze_triangle_spin_distribution(spins_xy_tri, n_sub=4)
    print_result("Triangle histogram average", info_tri["average"])
    print_result("Triangle histogram variance", info_tri["variance"])


    N2 = lattice_2d.N
    pos_2d = np.zeros((N2, 2))
    L2 = lattice_2d.L
    for i in range(N2):
        ix = i // L2
        iy = i % L2
        pos_2d[i] = [ix / L2, iy / L2]
    spins_2d = np.zeros((N2, 3))
    for i in range(N2):
        q = random_spin_quaternion(seed=1000 + i)
        spins_2d[i] = q_to_spin_vector(q)
    r_centers, g_r = radial_distribution_function_2d(pos_2d, spins_2d, max_r=0.5, n_bins=30)
    print_result("Radial correlation at r=0.1", float(g_r[2] if len(g_r) > 2 else 0.0))


    entropy_rate = entropy_rate_from_trajectory(Mz_eu, delay=1)
    print_result("Entropy rate (from Mz traj)", entropy_rate)




    print_header("STEP 10: 自适应 FEM 求解 Ginzburg-Landau 序参量")

    def A_func(x: float) -> float:
        return 0.1 + 0.05 * np.sin(np.pi * x) ** 2

    def B_func(x: float) -> float:
        return 1.0 - 0.8 * np.exp(-10.0 * (x - 0.5) ** 2)

    def F_func(x: float) -> float:
        return 0.5 * np.sin(2.0 * np.pi * x)

    nodes, solution, energy_density, fem_history = adaptive_fem_order_parameter(
        A_func=A_func,
        B_func=B_func,
        F_func=F_func,
        m_left=0.0,
        m_right=1.0,
        n_initial=8,
        max_refinements=5,
        error_threshold=0.01,
        max_nodes=150,
    )
    print_result("Final FEM nodes", nodes.size)
    print_result("Solution range", f"[{float(np.min(solution)):.4f}, {float(np.max(solution)):.4f}]")
    print_result("FEM refinement steps", len(fem_history))
    for rec in fem_history:
        print(f"    Step {rec['step']}: nodes={rec['n_nodes']}, max_error={rec['max_error']:.4e}")




    print_header("SUMMARY: 数值鲁棒性与性能指标")
    elapsed = time.time() - start_time
    print_result("Total execution time (s)", f"{elapsed:.3f}")
    print_result("3D system size", N_3d)
    print_result("2D system size", N2)
    print_result("Skyline compression ratio", f"{na / (64 * 64):.3f}")
    print_result("SA energy reduction", f"{sa_history[0] - e_best:.4f}")
    print_result("Greedy energy reduction", f"{relax_history[0] - e_relaxed:.4e}")
    print_result("Domain wall RMS gradient", f"{rms_norm(dMz_dx):.4e}")

    print("\n" + "=" * 70)
    print("  所有计算模块执行完毕，无报错。")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
