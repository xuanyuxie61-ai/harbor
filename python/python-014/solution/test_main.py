"""
main.py
=======
三维阻挫自旋玻璃基态与动力学相变的全栈计算框架。

科学领域：凝聚态物理 — 自旋玻璃与阻挫磁性。

本程序为零参数统一入口，执行以下全流程：
1. 构造烧绿石晶格与素数调制阻挫晶格
2. 建立交换相互作用矩阵（含键无序）
3. 四元数表示下的随机自旋初始化
4. 模拟退火搜索基态 + 贪心松弛精化
5. 能量景观局部极小分析（Brent 线搜索）
6. 自旋动力学 LLG 方程积分（显式 Euler + 隐式梯形）
7. 特征值谱分析与自旋波色散
8. 磁畴连通分量识别与统计直方图
9. Brusselator 型非线性自旋泵
10. Fisher-KPP 磁畴壁行波解
11. 自适应有限元求解 Ginzburg-Landau 序参量
12. 输出所有关键物理量与数值指标
"""

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

    # ======================================================================
    # 1. 晶格构造
    # ======================================================================
    print_header("STEP 1: 晶格构造 (Pyrochlore + Prime-Frustrated)")
    lattice_3d = PyrochloreLattice(L=3, J1=1.0, J2=0.2, disorder_std=0.2)
    lattice_2d = PrimeFrustratedLattice(L=12, J0=1.0, alpha=0.3, seed=42)
    print_result("3D Pyrochlore sites", lattice_3d.N)
    print_result("2D Prime-Frustrated sites", lattice_2d.N)
    print_result("3D bonds count", len(lattice_3d.bonds))

    # ======================================================================
    # 2. 交换矩阵与拉普拉斯算子
    # ======================================================================
    print_header("STEP 2: 交换矩阵与离散拉普拉斯算子")
    J_3d = lattice_3d.J
    J_2d = lattice_2d.to_full_matrix()
    J_2d = add_disorder(J_2d, std=0.15, seed=123)

    # 一维拉普拉斯（用于自旋波参考）
    L1d_periodic = exchange_laplacian_1d(n=64, h=0.1, bc="periodic")
    print_result("1D Laplacian shape", L1d_periodic.shape)
    print_result("1D Laplacian max eigenvalue (Gershgorin)", float(np.max(np.sum(np.abs(L1d_periodic), axis=1))))

    # Skyline 格式测试（使用 Dirichlet 边界三对角矩阵）
    L1d_dir = exchange_laplacian_1d(n=64, h=0.1, bc="dirichlet")
    na, diag_idx, a_sky = build_skyline_from_tridiagonal(
        np.diag(L1d_dir, -1), np.diag(L1d_dir), np.diag(L1d_dir, 1)
    )
    x_test = np.random.randn(64)
    y_sky = skyline_mv(64, diag_idx, a_sky, x_test)
    y_full = L1d_dir @ x_test
    print_result("Skyline vs Full matvec error", float(np.max(np.abs(y_sky - y_full))))

    # ======================================================================
    # 3. 自旋初始化（四元数）
    # ======================================================================
    print_header("STEP 3: 四元数自旋初始化")
    N_3d = lattice_3d.N
    spins_3d = np.zeros((N_3d, 3), dtype=float)
    for i in range(N_3d):
        q = random_spin_quaternion(seed=i)
        spins_3d[i] = q_to_spin_vector(q)
    print_result("Initial spin norm mean", float(np.mean(np.linalg.norm(spins_3d, axis=1))))
    print_result("Initial total magnetization Mz", float(np.mean(spins_3d[:, 2])))

    # ======================================================================
    # 4. 能量景观优化：模拟退火 + 贪心松弛
    # ======================================================================
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

    # 单点 Brent 线搜索示例
    axis_test = np.array([0.0, 0.0, 1.0])
    theta_opt, s_new, e_min = line_search_spin_rotation(
        J_3d, spins_relaxed, site_idx=0, axis=axis_test
    )
    print_result("Brent line-search optimal angle (rad)", theta_opt)
    print_result("Brent line-search energy", e_min)

    # ======================================================================
    # 5. 特征值分析与自旋波
    # ======================================================================
    print_header("STEP 5: 特征值谱与自旋波色散")
    # 对 2D 小矩阵做全谱分析
    lam_min, gap, eigs, eigvecs = spectral_gap_and_soft_modes(J_2d, n_soft=3)
    print_result("Min eigenvalue", lam_min)
    print_result("Spectral gap", gap)
    print_result("Correlation length (MF)", correlation_length_from_gap(gap, J=1.0, a=1.0))

    # 幂法
    lam_max, v_max, iters = power_method(np.abs(J_2d), max_iter=300, tol=1e-10)
    print_result("Power method dominant eigenvalue", lam_max)
    print_result("Power method iterations", iters)

    # 逆迭代求软模
    lam_soft, v_soft, iters_inv = inverse_iteration(J_2d, shift=0.0, max_iter=200)
    print_result("Inverse iteration soft eigenvalue", lam_soft)
    print_result("Inverse iteration iterations", iters_inv)

    # 一维自旋波色散
    k_pts = np.linspace(0, np.pi, 50)
    omega_sw = spin_wave_dispersion_1d(J=1.0, S=1.0, a=1.0, k_points=k_pts)
    print_result("Spin wave max frequency", float(np.max(omega_sw)))

    # ======================================================================
    # 6. 自旋动力学：LLG 方程
    # ======================================================================
    print_header("STEP 6: LLG 自旋动力学 (Euler + Trapezoidal)")
    # 对 3D 小系统做短时间演化
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

    # ======================================================================
    # 7. Brusselator 型自旋泵
    # ======================================================================
    print_header("STEP 7: Brusselator 非线性自旋泵")
    t_pump, y_pump = integrate_brusselator_pump(a=1.0, b=3.0, n_steps=2000)
    print_result("Pump u final", float(y_pump[-1, 0]))
    print_result("Pump v final", float(y_pump[-1, 1]))

    # ======================================================================
    # 8. Fisher-KPP 磁畴壁行波
    # ======================================================================
    print_header("STEP 8: Fisher-KPP 磁畴壁行波解")
    x_wall = np.linspace(-10.0, 10.0, 200)
    Mz_wall, dMz_dt, dMz_dx, d2Mz_dx2 = domain_wall_magnetization(t=1.0, x=x_wall, Ms=1.0)
    print_result("Domain wall center Mz", float(Mz_wall[len(Mz_wall) // 2]))
    print_result("Domain wall width proxy", float(np.sum(np.abs(dMz_dx) > 0.01)))

    # ======================================================================
    # 9. 磁畴分析与直方图统计
    # ======================================================================
    print_header("STEP 9: 磁畴分析与统计直方图")
    # 将 3D 自旋投影到 2D 平面做连通分量分析
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

    # 自旋取向直方图
    counts, centers, orient_stats = spin_orientation_histogram(spins_relaxed, n_bins=12)
    print_result("Polar angle mean (rad)", orient_stats["mean"])
    print_result("Polar angle variance", orient_stats["variance"])

    # 三角形分布统计
    spins_xy = spins_relaxed[:, :2]
    norms = np.linalg.norm(spins_xy, axis=1, keepdims=True) + EPS_MACHINE
    spins_xy_tri = np.abs(spins_xy) / norms
    histo_tri, info_tri = analyze_triangle_spin_distribution(spins_xy_tri, n_sub=4)
    print_result("Triangle histogram average", info_tri["average"])
    print_result("Triangle histogram variance", info_tri["variance"])

    # 径向关联函数（仅对 2D 投影位置）
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

    # 熵产生率
    entropy_rate = entropy_rate_from_trajectory(Mz_eu, delay=1)
    print_result("Entropy rate (from Mz traj)", entropy_rate)

    # ======================================================================
    # 10. 自适应有限元：Ginzburg-Landau 序参量
    # ======================================================================
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

    # ======================================================================
    # 11. 综合性能与数值鲁棒性汇总
    # ======================================================================
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

# ================================================================
# 测试用例（28个，assert模式，涉及随机值均使用固定种子）
# ================================================================

import numpy as np

# ---- TC01: safe_divide 正常除法返回正确值 ----
result = safe_divide(10.0, 2.0, fallback=0.0)
assert abs(result - 5.0) < 1e-12, '[TC01] safe_divide normal division FAILED'

# ---- TC02: safe_divide 除零时返回 fallback ----
result = safe_divide(5.0, 0.0, fallback=999.0)
assert result == 999.0, '[TC02] safe_divide zero fallback FAILED'

# ---- TC03: clip_spin_norm 退化零矢量对齐 z 轴 ----
from utils import clip_spin_norm
s_zero = np.array([0.0, 0.0, 0.0])
s_clip = clip_spin_norm(s_zero, target_norm=1.0)
assert np.allclose(s_clip, np.array([0.0, 0.0, 1.0])), '[TC03] clip_spin_norm degenerate FAILED'

# ---- TC04: rms_norm 空数组返回零 ----
assert rms_norm(np.array([])) == 0.0, '[TC04] rms_norm empty array FAILED'

# ---- TC05: skyline_mv 与稠密矩阵乘法一致 ----
np.random.seed(42)
L5 = exchange_laplacian_1d(n=8, h=0.1, bc="dirichlet")
na, diag_idx, a_sky = build_skyline_from_tridiagonal(np.diag(L5, -1), np.diag(L5), np.diag(L5, 1))
x_vec = np.random.randn(8)
y_sky = skyline_mv(8, diag_idx, a_sky, x_vec)
y_dense = L5 @ x_vec
assert np.allclose(y_sky, y_dense), '[TC05] skyline_mv consistency FAILED'

# ---- TC06: is_prime 正确检测素数与非素数 ----
from utils import is_prime
assert is_prime(17) == True, '[TC06] is_prime prime check FAILED'
assert is_prime(1) == False, '[TC06] is_prime non-prime check FAILED'

# ---- TC07: exchange_laplacian_1d Dirichlet 对称正定 ----
L_dir = exchange_laplacian_1d(n=5, h=1.0, bc="dirichlet")
assert np.allclose(L_dir, L_dir.T), '[TC07] Laplacian symmetry FAILED'
assert np.all(np.diag(L_dir) > 0), '[TC07] Laplacian positive diagonal FAILED'

# ---- TC08: exchange_laplacian_1d periodic 行和为零 ----
L_per = exchange_laplacian_1d(n=6, h=0.5, bc="periodic")
assert np.allclose(np.sum(L_per, axis=1), 0.0), '[TC08] periodic Laplacian row sum FAILED'

# ---- TC09: exchange_energy Ising 标量情形 ----
J_ising = np.array([[0.0, 1.0], [1.0, 0.0]])
spins_ising = np.array([1.0, -1.0])
e_ising = exchange_energy(J_ising, spins_ising)
assert abs(e_ising - (-1.0)) < 1e-10, '[TC09] exchange energy Ising FAILED'

# ---- TC10: apply_exchange_operator Heisenberg 输出形状与值 ----
J_id = np.eye(4)
spins_heis = np.ones((4, 3))
H_eff = apply_exchange_operator(J_id, spins_heis)
assert H_eff.shape == (4, 3), '[TC10] apply_exchange shape FAILED'
assert np.allclose(H_eff, spins_heis), '[TC10] apply_exchange identity FAILED'

# ---- TC11: q_multiply 单位四元数保持右操作数 ----
from spin_quaternion import q_multiply, q_normalize
q_i = np.array([1.0, 0.0, 0.0, 0.0])
q_j = np.array([0.0, 1.0, 0.0, 0.0])
q_res = q_multiply(q_i, q_j)
assert np.allclose(q_res, q_j), '[TC11] q_multiply identity FAILED'

# ---- TC12: q_to_rotation_matrix 与 rotation_matrix_to_q 互逆 ----
from spin_quaternion import q_to_rotation_matrix, rotation_matrix_to_q, random_spin_quaternion
np.random.seed(42)
q_orig = random_spin_quaternion(seed=123)
R_mat = q_to_rotation_matrix(q_orig)
q_back = rotation_matrix_to_q(R_mat)
dot_abs = abs(np.dot(q_normalize(q_orig), q_normalize(q_back)))
assert dot_abs > 0.999, '[TC12] quat-rotmat roundtrip FAILED'

# ---- TC13: q_rotate_vector 保持矢量长度不变 ----
from spin_quaternion import q_rotate_vector, axis_angle_to_q
q_rot = axis_angle_to_q(np.array([0.0, 0.0, 1.0]), np.pi / 4.0)
v_in = np.array([2.0, 3.0, 4.0])
v_out = q_rotate_vector(q_rot, v_in)
assert np.allclose(np.linalg.norm(v_out), np.linalg.norm(v_in)), '[TC13] rotation preserves norm FAILED'

# ---- TC14: random_spin_quaternion 固定种子结果可复现 ----
np.random.seed(42)
q_a = random_spin_quaternion(seed=77)
q_b = random_spin_quaternion(seed=77)
assert np.allclose(q_a, q_b), '[TC14] random quaternion reproducibility FAILED'

# ---- TC15: local_min_brent 精确求 x^2 极小 ----
from energy_landscape import local_min_brent
theta_opt, e_min, calls = local_min_brent(lambda x: x * x, -2.0, 2.0)
assert abs(theta_opt) < 0.01, '[TC15] Brent min x^2 position FAILED'
assert abs(e_min) < 1e-4, '[TC15] Brent min x^2 value FAILED'

# ---- TC16: spin_wave_dispersion_1d k=0 频率严格为零 ----
k_vals = np.array([0.0, np.pi / 4, np.pi / 2])
omega_sw = spin_wave_dispersion_1d(J=1.0, S=1.0, a=1.0, k_points=k_vals)
assert abs(omega_sw[0]) < 1e-12, '[TC16] spin wave omega(0) FAILED'
assert omega_sw[-1] > omega_sw[0], '[TC16] spin wave monotonicity FAILED'

# ---- TC17: correlation_length_from_gap 零间隙发散为 inf ----
xi_inf = correlation_length_from_gap(0.0, J=1.0, a=1.0)
assert np.isinf(xi_inf), '[TC17] correlation length inf FAILED'

# ---- TC18: power_method 对角矩阵主导特征值精确 ----
np.random.seed(42)
A_diag = np.diag([1.0, 3.0, 7.0])
lam_dom, v_dom, iters_pm = power_method(A_diag, max_iter=300, tol=1e-10)
assert abs(lam_dom - 7.0) < 0.01, '[TC18] power method dominant eigenvalue FAILED'

# ---- TC19: spectral_gap_and_soft_modes 谱隙非负 ----
np.random.seed(42)
J_sym = np.random.randn(5, 5)
J_sym = (J_sym + J_sym.T) * 0.5
lam_min, gap_val, eigs_all, eigvecs = spectral_gap_and_soft_modes(J_sym, n_soft=2)
assert gap_val >= 0.0, '[TC19] spectral gap non-negative FAILED'

# ---- TC20: euler_integrate_llg 保持自旋单位范数 ----
np.random.seed(42)
J_llg = np.eye(3) * 0.05
spins_init = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
t_e, traj_e = euler_integrate_llg(J_llg, spins_init, t_span=(0.0, 0.2), n_steps=20, gamma=1.0, alpha=0.1)
norms_final = np.linalg.norm(traj_e[-1], axis=1)
assert np.allclose(norms_final, 1.0, atol=1e-5), '[TC20] Euler spin norm conservation FAILED'

# ---- TC21: fisher_kpp_domain_wall_exact 左右边界渐进行为 ----
from spin_dynamics import fisher_kpp_domain_wall_exact
x_bound = np.array([-15.0, 15.0])
u_kpp, ut_kpp, ux_kpp, uxx_kpp = fisher_kpp_domain_wall_exact(t=0.0, x=x_bound)
assert u_kpp[0] > 0.99, '[TC21] KPP left boundary FAILED'
assert u_kpp[-1] < 0.01, '[TC21] KPP right boundary FAILED'

# ---- TC22: connected_components_2d_spin_map 正确计数连通区域 ----
spin_map_test = np.array([[0.8, 0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.9]])
labels_map = connected_components_2d_spin_map(spin_map_test, threshold=0.5)
assert int(labels_map.max()) == 2, '[TC22] connected components count FAILED'

# ---- TC23: solve_tridiagonal 与稠密求解器结果一致 ----
from adaptive_fem_solver import solve_tridiagonal
n_td = 5
aD_td = np.ones(n_td) * 2.0
aL_td = np.zeros(n_td)
aR_td = np.zeros(n_td)
aL_td[1:] = -1.0
aR_td[:-1] = -1.0
rhs_td = np.ones(n_td)
sol_td = solve_tridiagonal(aL_td, aD_td, aR_td, rhs_td)
A_full_td = np.diag(aD_td) + np.diag(aR_td[:-1], 1) + np.diag(aL_td[1:], -1)
sol_np_td = np.linalg.solve(A_full_td, rhs_td)
assert np.allclose(sol_td, sol_np_td), '[TC23] tridiagonal solve FAILED'

# ---- TC24: adaptive_fem_order_parameter 输出结构正确 ----
nodes_fem, sol_fem, edens_fem, hist_fem = adaptive_fem_order_parameter(
    A_func=lambda x: 1.0,
    B_func=lambda x: 0.5,
    F_func=lambda x: x,
    m_left=0.0,
    m_right=1.0,
    n_initial=4,
    max_refinements=2,
    error_threshold=0.1,
    max_nodes=50,
)
assert nodes_fem.size == sol_fem.size, '[TC24] FEM nodes-solution size FAILED'
assert len(hist_fem) > 0, '[TC24] FEM history non-empty FAILED'

# ---- TC25: integrate_brusselator_pump 输出时序形状正确 ----
t_p, y_p = integrate_brusselator_pump(a=1.0, b=3.0, n_steps=100)
assert y_p.shape == (101, 2), '[TC25] Brusselator output shape FAILED'
assert t_p.size == 101, '[TC25] Brusselator time size FAILED'

# ---- TC26: extract_domain_statistics 空图返回零域 ----
empty_spin_map = np.zeros((4, 4))
dom_stats = extract_domain_statistics(empty_spin_map, threshold=0.5)
assert dom_stats["n_domains"] == 0, '[TC26] empty domain stats FAILED'

# ---- TC27: histogram_stats_1d 空数组安全回退 ----
counts_h, edges_h, stats_h = histogram_stats_1d(np.array([]), bins=10)
assert stats_h["mean"] == 0.0, '[TC27] histogram empty mean FAILED'

# ---- TC28: triangle_area_histogram_2d 子区域数量正确 ----
pts_tri = np.array([[0.1, 0.1], [0.3, 0.2], [0.15, 0.05]])
histo_tri, info_tri = triangle_area_histogram_2d(pts_tri, n_sub=3)
assert histo_tri.size == 9, '[TC28] triangle histogram size FAILED'

print('\n全部 28 个测试通过!\n')
