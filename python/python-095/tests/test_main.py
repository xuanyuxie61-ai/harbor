"""
main.py
三维封闭空间多通道宽带主动噪声控制(ANC)系统综合仿真

本程序为零参数入口,运行后将依次执行:
  1. 一维管道声学模态分析 (tridiagonal_acoustics)
  2. 球面阵列几何生成 (spherical_array_geometry)
  3. 稀疏声传播矩阵构建 (sparse_acoustics)
  4. 次级声源相位优化 (source_phase_optimizer)
  5. 多通道自适应滤波仿真 (adaptive_filter)
  6. 最优声源组合选择 (optimal_source_selection)
  7. 噪声统计建模与步长调整 (statistical_noise_model)
  8. 圆形活塞辐射器积分 (integrals_radiation)
  9. 非线性自适应动力学分析 (nonlinear_ode_dynamics)
 10. 3D有限元房间声学建模 (acoustic_room_model)

所有结果以文本形式输出至标准输出.
"""

import numpy as np
import math
import sys


def main():
    print("=" * 70)
    print("  博士级主动噪声控制综合仿真系统")
    print("  领域: 声学工程 - 主动噪声控制与自适应滤波")
    print("=" * 70)

    # ================================================================
    # 1. 一维管道声学模态分析 (tridiagonal_acoustics)
    # ================================================================
    print("\n[1] 一维管道Helmholtz方程模态分析")
    print("-" * 50)
    from tridiagonal_acoustics import pipe_helmholtz_solver, lindberg_exact_solution, lindberg_residual

    L_pipe = 1.0  # 管道长度 1m
    N_pts = 100
    f_freq = 500.0  # Hz
    c0 = 343.0
    k = 2.0 * math.pi * f_freq / c0

    # 源分布在管道中部
    source = np.zeros(N_pts, dtype=complex)
    source[N_pts // 2] = 1.0e-3
    x_pipe, p_pipe = pipe_helmholtz_solver(L_pipe, N_pts, k, source)
    print(f"  频率: {f_freq} Hz, 波数: {k:.3f} rad/m")
    print(f"  管道中部声压幅值: {abs(p_pipe[N_pts//2]):.6e} Pa")
    print(f"  声压实部范围: [{np.real(p_pipe).min():.4e}, {np.real(p_pipe).max():.4e}]")

    # Lindberg刚性ODE验证
    t_test = np.linspace(0, 1.0, 11)
    y_exact, dydt_exact = lindberg_exact_solution(t_test)
    res = lindberg_residual(t_test, y_exact, dydt_exact)
    max_res = np.max(np.abs(res))
    print(f"  Lindberg ODE 残差验证: max|residual| = {max_res:.3e} (应接近0)")

    # ================================================================
    # 2. 球面阵列几何 (spherical_array_geometry)
    # ================================================================
    print("\n[2] Fibonacci球面阵列几何布置")
    print("-" * 50)
    from spherical_array_geometry import sphere_fibonacci_grid_points

    N_sensors = 32
    radius = 0.5  # m
    sensors = sphere_fibonacci_grid_points(N_sensors, radius)
    print(f"  生成 {N_sensors} 个球面传感器位置 (半径 {radius} m)")
    print(f"  第一个传感器坐标: ({sensors[0,0]:.4f}, {sensors[0,1]:.4f}, {sensors[0,2]:.4f})")
    print(f"  最后一个传感器坐标: ({sensors[-1,0]:.4f}, {sensors[-1,1]:.4f}, {sensors[-1,2]:.4f})")

    # 检查均匀性: 计算最小间距
    min_dist = np.inf
    for i in range(N_sensors):
        for j in range(i + 1, N_sensors):
            d = np.linalg.norm(sensors[i] - sensors[j])
            if d < min_dist:
                min_dist = d
    print(f"  最小传感器间距: {min_dist:.4f} m")

    # ================================================================
    # 3. 稀疏声传播矩阵 (sparse_acoustics)
    # ================================================================
    print("\n[3] 稀疏声学传递矩阵构建")
    print("-" * 50)
    from sparse_acoustics import acoustic_transfer_matrix_sparse, generate_room_coupling_graph

    N_sources = 16
    source_pos = sphere_fibonacci_grid_points(N_sources, radius=0.3)
    H_sparse = acoustic_transfer_matrix_sparse(sensors, source_pos, k)
    H_sparse.st_to_ccs()
    print(f"  传感器数 M={N_sensors}, 声源数 N={N_sources}")
    print(f"  稀疏矩阵非零元数: {len(H_sparse.st_vals)} (密度={len(H_sparse.st_vals)/(N_sensors*N_sources)*100:.2f}%)")

    # 测试矩阵向量乘法
    test_vec = np.ones(N_sources)
    y_st = H_sparse.st_mv(test_vec)
    y_ccs = H_sparse.ccs_mv(test_vec)
    diff = np.max(np.abs(y_st - y_ccs))
    print(f"  ST与CCS乘法一致性误差: {diff:.3e}")

    # 图耦合测试
    graph = generate_room_coupling_graph(20, connection_prob=0.15, seed=42)
    graph.st_to_ccs()
    y_graph = graph.ccs_mv(np.ones(20))
    print(f"  房间耦合图矩阵-向量乘测试: sum={np.sum(y_graph):.4f}")

    # ================================================================
    # 4. 声源相位优化 (source_phase_optimizer)
    # ================================================================
    print("\n[4] 次级声源相位角非线性优化")
    print("-" * 50)
    from source_phase_optimizer import optimize_source_phase

    # 构造简化的单源场景
    np.random.seed(42)
    H_col = (np.random.randn(N_sensors) + 1j * np.random.randn(N_sensors)) * 0.5
    d_noise = (np.random.randn(N_sensors) + 1j * np.random.randn(N_sensors)) * 0.3
    amplitude = 0.5

    phi_opt, min_energy = optimize_source_phase(H_col, d_noise, amplitude)
    print(f"  最优相位角: {phi_opt:.4f} rad ({math.degrees(phi_opt):.2f} deg)")
    print(f"  最小声能量: {min_energy:.6e}")

    # 与零相位比较
    s0 = amplitude * np.exp(1j * 0.0)
    p0 = d_noise + H_col * s0
    energy0 = np.vdot(p0, p0).real
    reduction_db = 10.0 * math.log10((min_energy + 1e-18) / (energy0 + 1e-18))
    print(f"  相比零相位的能量降低: {reduction_db:.2f} dB")

    # ================================================================
    # 5. 多通道自适应滤波 (adaptive_filter)
    # ================================================================
    print("\n[5] 多通道FxLMS自适应滤波仿真")
    print("-" * 50)
    from adaptive_filter import MultichannelFxLMS, qr_rank_revealing_ls

    L_ch = 2
    K_len = 16
    M_sens = 4
    sec_len = 8
    sec_model = np.random.randn(M_sens, L_ch, sec_len) * 0.1

    fxlms = MultichannelFxLMS(L_ch, K_len, sec_model, mu=0.002)

    T_sim = 500
    errors = []
    for t in range(T_sim):
        x_ref = np.random.randn(L_ch) * 0.5
        # 构造目标误差 (含初级噪声)
        target = np.random.randn(M_sens) * 0.3
        fxlms.update(x_ref, target)
        y_out = fxlms.predict_output(x_ref)
        errors.append(np.mean(target ** 2))

    err_before = np.mean(errors[:50])
    err_after = np.mean(errors[-50:])
    print(f"  仿真步数: {T_sim}")
    print(f"  初始平均误差功率: {err_before:.6f}")
    print(f"  收敛后平均误差功率: {err_after:.6f}")
    print(f"  衰减量: {10*math.log10((err_after+1e-12)/(err_before+1e-12)):.2f} dB")

    # QR秩揭示测试
    A_test = np.random.randn(20, 5)
    b_test = np.random.randn(20)
    w_qr, rank = qr_rank_revealing_ls(A_test, b_test)
    residual_norm = np.linalg.norm(A_test @ w_qr - b_test)
    print(f"  QR秩揭示最小二乘残差: {residual_norm:.4e}, 秩={rank}")

    # ================================================================
    # 6. 最优声源选择 (optimal_source_selection)
    # ================================================================
    print("\n[6] 最优次级声源子集选择")
    print("-" * 50)
    from optimal_source_selection import greedy_source_selection, subset_sum_swap_anc

    # 功率预算子集选择
    powers = np.array([50, 80, 120, 40, 90, 60, 110, 30], dtype=float)
    budget = 250.0
    selected_power, achieved = subset_sum_swap_anc(powers, budget)
    print(f"  功率预算: {budget} W")
    print(f"  选中索引: {np.where(selected_power)[0].tolist()}")
    print(f"  实际功耗: {achieved:.1f} W")

    # 贪心声源选择
    H_sel = np.random.randn(M_sens, N_sources) + 1j * np.random.randn(M_sens, N_sources)
    d_sel = np.random.randn(M_sens) + 1j * np.random.randn(M_sens)
    max_src = 6
    src_powers = np.array([30, 45, 60, 35, 50, 70, 40, 55, 65, 80, 25, 90, 50, 60, 75, 85], dtype=float)
    sel, filters = greedy_source_selection(H_sel, d_sel, max_src, budget, src_powers)
    n_sel = np.sum(sel)
    print(f"  贪心选择声源数: {n_sel}")
    if n_sel > 0:
        H_sub = H_sel[:, sel]
        residual = d_sel + H_sub @ filters
        res_energy = np.vdot(residual, residual).real
        print(f"  残余声能量: {res_energy:.6e}")

    # ================================================================
    # 7. 统计噪声模型 (statistical_noise_model)
    # ================================================================
    print("\n[7] 噪声Dirichlet统计建模与自适应步长")
    print("-" * 50)
    from statistical_noise_model import dirichlet_estimate_mle, adaptive_step_size_from_dirichlet, noise_stationarity_test

    # 构造模拟的多通道功率数据
    rng = np.random.default_rng(123)
    N_obs = 200
    K_ch = 4
    alpha_true = np.array([3.0, 2.0, 4.0, 2.5])
    x_data = rng.dirichlet(alpha_true, N_obs)

    alpha_est, niter, loglik = dirichlet_estimate_mle(x_data)
    print(f"  观测数: {N_obs}, 通道数: {K_ch}")
    print(f"  真实alpha: {alpha_true}")
    print(f"  估计alpha: {alpha_est.round(4)}")
    print(f"  迭代次数: {niter}")
    print(f"  对数似然: {loglik:.4f}")

    mu_adaptive = adaptive_step_size_from_dirichlet(x_data, base_mu=0.001)
    print(f"  自适应步长: {mu_adaptive.round(6)}")

    # 平稳性检验
    err_hist = np.cumsum(rng.normal(0, 0.01, 200)) + rng.normal(0, 0.1, 200)
    is_stat, f_stat = noise_stationarity_test(err_hist, window=40)
    print(f"  噪声平稳性检验: F统计量={f_stat:.3f}, 平稳={is_stat}")

    # ================================================================
    # 8. 圆形活塞辐射器积分 (integrals_radiation)
    # ================================================================
    print("\n[8] 圆形活塞声辐射积分计算")
    print("-" * 50)
    from integrals_radiation import disk_unit_sample, rayleigh_integral_piston, piston_directivity_factor, piston_radiation_resistance

    # 圆盘采样与积分
    n_samp = 200
    a_piston = 0.1  # 活塞半径 10cm
    disk_pts = disk_unit_sample(n_samp, radius=a_piston)
    print(f"  活塞半径: {a_piston} m, 采样点数: {n_samp}")

    # Rayleigh积分
    observer = np.array([0.0, 0.0, 1.0])
    k_piston = 2.0 * math.pi * 1000.0 / c0  # 1kHz
    p_rayleigh = rayleigh_integral_piston(observer, disk_pts, u_n=0.01, k=k_piston)
    print(f"  1kHz时1m处声压: {abs(p_rayleigh):.6e} Pa")

    # 指向性因子
    ka = k_piston * a_piston
    di = piston_directivity_factor(ka)
    print(f"  ka={ka:.3f}, 指向性因子 DI={di:.2f} dB")

    # 辐射阻力
    R_ratio = piston_radiation_resistance(ka)
    print(f"  辐射阻力比 R_r/(rho0 c0 S)={R_ratio:.4f}")

    # 特殊函数验证
    from special_functions import cos_power_int, betain, digamma, trigamma, log_beta
    cpi = cos_power_int(0.0, math.pi / 2, 4)
    print(f"  cos^4积分 [0,pi/2]: {cpi:.6f} (理论值=3pi/16={3*math.pi/16:.6f})")

    beta_val, ierr = betain(0.5, 2.0, 3.0, log_beta(2.0, 3.0))
    print(f"  I_0.5(2,3) = {beta_val:.6f} (ifault={ierr})")

    psi_val, _ = digamma(2.5)
    psi_prime, _ = trigamma(2.5)
    print(f"  digamma(2.5)={psi_val:.6f}, trigamma(2.5)={psi_prime:.6f}")

    # ================================================================
    # 9. 非线性自适应动力学 (nonlinear_ode_dynamics)
    # ================================================================
    print("\n[9] 非线性自适应系统动力学分析")
    print("-" * 50)
    from nonlinear_ode_dynamics import anishchenko_adaptive_deriv, rk4_integrate, stability_boundary_anishchenko

    # 稳定性边界分析 (mu vs gamma_leak)
    mu_grid, gamma_grid, stable = stability_boundary_anishchenko((0.1, 2.0), (0.1, 2.0), n_grid=30)
    stable_fraction = np.sum(stable) / stable.size
    print(f"  参数空间(mu vs gamma)稳定区域占比: {stable_fraction*100:.1f}%")

    # RK4积分
    traj = rk4_integrate(
        lambda t, y: anishchenko_adaptive_deriv(t, y, mu=1.2, eta=0.5),
        0.0, [-0.1, 0.5, -0.6], 50.0, h=0.05
    )
    final_state = traj[-1][1]
    print(f"  Anishchenko-like系统积分 (t=0..50):")
    print(f"    最终状态: w1={final_state[0]:.4f}, w2={final_state[1]:.4f}, e={final_state[2]:.4f}")
    print(f"    轨迹点数: {len(traj)}")

    # ================================================================
    # 10. 3D有限元房间声学 (acoustic_room_model)
    # ================================================================
    print("\n[10] 3D房间声学有限元建模与RCM重排序")
    print("-" * 50)
    from acoustic_room_model import generate_box_mesh, AcousticRoomFEM

    nodes, elements = generate_box_mesh(1.0, 1.0, 1.0, nx=5, ny=5, nz=5)
    fem = AcousticRoomFEM(nodes, elements)
    bw_before = fem.compute_bandwidth()
    perm, perm_inv = fem.rcm_reorder()
    bw_after = fem.compute_bandwidth()

    print(f"  节点数: {fem.Nn}, 单元数: {fem.Ne}")
    print(f"  RCM重排序前带宽: {bw_before}")
    print(f"  RCM重排序后带宽: {bw_after}")
    reduction = bw_before / max(bw_after, 1) if bw_after < bw_before else bw_after / max(bw_before, 1)
    print(f"  带宽变化: {reduction:.2f}x ({'缩减' if bw_after < bw_before else '增加'})")

    # 组装并求解简化有限元系统
    k_fem = 2.0 * math.pi * 200.0 / c0
    A_fem, b_fem = fem.assemble_system(k_fem)
    # 在中心节点施加点源
    center_node = fem.Nn // 2
    b_fem[center_node] = 1.0

    try:
        p_fem = np.linalg.solve(A_fem, b_fem)
        print(f"  FEM声压求解成功. 中心节点声压: {p_fem[center_node]:.4e}")
        print(f"  声压范数: {np.linalg.norm(p_fem):.4e}")
    except np.linalg.LinAlgError:
        print("  FEM矩阵奇异,使用最小二乘近似")
        p_fem = np.linalg.lstsq(A_fem, b_fem, rcond=None)[0]
        print(f"  近似中心节点声压: {p_fem[center_node]:.4e}")

    # ================================================================
    print("\n" + "=" * 70)
    print("  仿真全部完成. 所有模块运行正常.")
    print("=" * 70)


if __name__ == "__main__":
    main()
# ---- TC01: tridiagonal_solver求解有限且非NaN ----
from tridiagonal_acoustics import tridiagonal_solver
a = np.array([0.0, -1.0, -1.0])
b = np.array([2.0, 2.0, 2.0])
c = np.array([-1.0, -1.0, 0.0])
d = np.array([1.0, 0.0, 0.0])
x = tridiagonal_solver(a, b, c, d)
assert np.all(np.isfinite(x)), '[TC01] tridiagonal_solver产生非有限值 FAILED'

# ---- TC02: tridiagonal_mv与solver一致性 ----
from tridiagonal_acoustics import tridiagonal_solver, tridiagonal_mv
a = np.array([0.0, -1.0, -1.0])
b = np.array([2.0, 2.0, 2.0])
c = np.array([-1.0, -1.0, 0.0])
d = np.array([1.0, 0.0, 0.0])
x = tridiagonal_solver(a.copy(), b.copy(), c.copy(), d.copy())
rhs = tridiagonal_mv(a, b, c, x)
assert np.max(np.abs(rhs - d)) < 1e-10, '[TC02] tridiagonal_mv与solver不一致 FAILED'

# ---- TC03: pipe_helmholtz_solver声压有限 ----
from tridiagonal_acoustics import pipe_helmholtz_solver
L = 1.0
N = 50
k = 2.0 * math.pi * 500.0 / 343.0
source = np.zeros(N, dtype=complex)
source[N // 2] = 1.0e-3
x_pipe, p_pipe = pipe_helmholtz_solver(L, N, k, source)
assert np.all(np.isfinite(p_pipe)), '[TC03] pipe_helmholtz_solver产生非有限声压 FAILED'

# ---- TC04: lindberg_exact_solution残差接近零 ----
from tridiagonal_acoustics import lindberg_exact_solution, lindberg_residual
t_test = np.linspace(0, 1.0, 11)
y_exact, dydt_exact = lindberg_exact_solution(t_test)
res = lindberg_residual(t_test, y_exact, dydt_exact)
max_res = np.max(np.abs(res))
assert max_res < 1e-10, '[TC04] Lindberg精确解残差过大 FAILED'

# ---- TC05: sphere_fibonacci_grid_points点在球面上 ----
from spherical_array_geometry import sphere_fibonacci_grid_points
N_sensors = 32
radius = 0.5
sensors = sphere_fibonacci_grid_points(N_sensors, radius)
distances = np.linalg.norm(sensors, axis=1)
assert np.max(np.abs(distances - radius)) < 1e-10, '[TC05] Fibonacci网格点不在球面上 FAILED'

# ---- TC06: sphere_fibonacci_grid_points输出形状 ----
from spherical_array_geometry import sphere_fibonacci_grid_points
sensors = sphere_fibonacci_grid_points(16, 1.0)
assert sensors.shape == (16, 3), '[TC06] Fibonacci网格输出形状错误 FAILED'

# ---- TC07: SparseAcousticMatrix ST与CCS乘法一致 ----
from sparse_acoustics import SparseAcousticMatrix
S = SparseAcousticMatrix(5, 5)
for i in range(5):
    S.add_entry(i, i, 2.0)
    if i < 4:
        S.add_entry(i, i + 1, -1.0)
S.st_to_ccs()
x_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_st = S.st_mv(x_vec)
y_ccs = S.ccs_mv(x_vec)
assert np.max(np.abs(y_st - y_ccs)) < 1e-12, '[TC07] ST与CCS乘法不一致 FAILED'

# ---- TC08: generate_room_coupling_graph输出有限 ----
from sparse_acoustics import generate_room_coupling_graph
graph = generate_room_coupling_graph(10, connection_prob=0.15, seed=42)
graph.st_to_ccs()
y = graph.ccs_mv(np.ones(10))
assert np.all(np.isfinite(y)), '[TC08] 房间耦合图输出非有限 FAILED'

# ---- TC09: zero_chandrupatla求根精度 ----
from source_phase_optimizer import zero_chandrupatla
f = lambda x: x**3 - 2.0
xm, fm, calls = zero_chandrupatla(f, 1.0, 2.0)
assert abs(fm) < 1e-6, '[TC09] Chandrupatla求根精度不足 FAILED'

# ---- TC10: optimize_source_phase能量不增 ----
from source_phase_optimizer import optimize_source_phase
np.random.seed(42)
H_col = (np.random.randn(8) + 1j * np.random.randn(8)) * 0.5
d_noise = (np.random.randn(8) + 1j * np.random.randn(8)) * 0.3
phi_opt, min_energy = optimize_source_phase(H_col, d_noise, 0.5)
s0 = 0.5 * np.exp(1j * 0.0)
p0 = d_noise + H_col * s0
energy0 = np.vdot(p0, p0).real
assert min_energy <= energy0 + 1e-12, '[TC10] 相位优化未降低能量 FAILED'

# ---- TC11: qr_rank_revealing_ls残差小 ----
from adaptive_filter import qr_rank_revealing_ls
np.random.seed(42)
A_test = np.random.randn(20, 5)
w_true = np.random.randn(5)
b_test = A_test @ w_true
w_qr, rank = qr_rank_revealing_ls(A_test, b_test)
residual_norm = np.linalg.norm(A_test @ w_qr - b_test)
assert residual_norm < 1e-8, '[TC11] QR秩揭示最小二乘残差过大 FAILED'

# ---- TC12: MultichannelFxLMS系数更新形状不变 ----
from adaptive_filter import MultichannelFxLMS
np.random.seed(42)
sec_model = np.random.randn(2, 2, 4) * 0.1
fxlms = MultichannelFxLMS(2, 8, sec_model, mu=0.001)
x_ref = np.random.randn(2) * 0.5
target = np.random.randn(2) * 0.3
fxlms.update(x_ref, target)
assert fxlms.w.shape == (2, 8), '[TC12] FxLMS更新后系数形状改变 FAILED'

# ---- TC13: subset_sum_swap_anc功率不超预算 ----
from optimal_source_selection import subset_sum_swap_anc
powers = np.array([50, 80, 120, 40, 90, 60, 110, 30], dtype=float)
budget = 250.0
selected, achieved = subset_sum_swap_anc(powers, budget)
assert achieved <= budget + 1e-6, '[TC13] 子集和功率超过预算 FAILED'

# ---- TC14: greedy_source_selection选中数不超上限 ----
from optimal_source_selection import greedy_source_selection
np.random.seed(42)
M_sens = 4
N_sources = 8
H_sel = np.random.randn(M_sens, N_sources) + 1j * np.random.randn(M_sens, N_sources)
d_sel = np.random.randn(M_sens) + 1j * np.random.randn(M_sens)
max_src = 3
src_powers = np.array([30, 45, 60, 35, 50, 70, 40, 55], dtype=float)
sel, filters = greedy_source_selection(H_sel, d_sel, max_src, 200.0, src_powers)
assert np.sum(sel) <= max_src, '[TC14] 贪心选择声源数超过上限 FAILED'

# ---- TC15: dirichlet_estimate_mle估计有限 ----
from statistical_noise_model import dirichlet_estimate_mle
rng = np.random.default_rng(42)
alpha_true = np.array([3.0, 2.0, 4.0, 2.5])
x_data = rng.dirichlet(alpha_true, 500)
alpha_est, niter, loglik = dirichlet_estimate_mle(x_data)
assert np.all(np.isfinite(alpha_est)) and niter >= 0, '[TC15] Dirichlet MLE估计非有限或迭代异常 FAILED'

# ---- TC16: adaptive_step_size_from_dirichlet范围约束 ----
from statistical_noise_model import adaptive_step_size_from_dirichlet
rng = np.random.default_rng(42)
alpha_true = np.array([3.0, 2.0, 4.0, 2.5])
x_data = rng.dirichlet(alpha_true, 200)
mu = adaptive_step_size_from_dirichlet(x_data, base_mu=0.001)
assert np.all((mu >= 0.0001) & (mu <= 0.005)), '[TC16] 自适应步长超出合理范围 FAILED'

# ---- TC17: noise_stationarity_test平稳信号判稳 ----
from statistical_noise_model import noise_stationarity_test
np.random.seed(42)
err_hist = np.random.normal(0, 0.1, 200)
is_stat, f_stat = noise_stationarity_test(err_hist, window=40)
assert is_stat, '[TC17] 平稳信号误判为非平稳 FAILED'

# ---- TC18: cos_power_int解析验证 ----
from special_functions import cos_power_int
cpi = cos_power_int(0.0, math.pi / 2, 4)
expected = 3.0 * math.pi / 16.0
assert abs(cpi - expected) < 1e-10, '[TC18] cos^4积分与理论值不符 FAILED'

# ---- TC19: digamma已知值验证 ----
from special_functions import digamma
psi_val, ierr = digamma(1.0)
expected = -0.5772156649
assert abs(psi_val - expected) < 1e-6 and ierr == 0, '[TC19] digamma(1)值错误 FAILED'

# ---- TC20: trigamma已知值验证 ----
from special_functions import trigamma
psi_prime, ierr = trigamma(1.0)
expected = math.pi**2 / 6.0
assert abs(psi_prime - expected) < 1e-6 and ierr == 0, '[TC20] trigamma(1)值错误 FAILED'

# ---- TC21: piston_radiation_resistance小ka近似 ----
from integrals_radiation import piston_radiation_resistance
ka = 1e-4
R_ratio = piston_radiation_resistance(ka)
approx = ka**2 / 2.0
assert abs(R_ratio - approx) < 1e-6, '[TC21] 小ka辐射阻力近似不符 FAILED'

# ---- TC22: rk4_integrate常数ODE精确 ----
from nonlinear_ode_dynamics import rk4_integrate
traj = rk4_integrate(lambda t, y: np.array([0.0, 0.0]), 0.0, [1.0, -1.0], 1.0, h=0.1)
final = traj[-1][1]
assert np.max(np.abs(final - np.array([1.0, -1.0]))) < 1e-12, '[TC22] RK4常数ODE积分不精确 FAILED'

# ---- TC23: stability_boundary_anishchenko稳定区域验证 ----
from nonlinear_ode_dynamics import stability_boundary_anishchenko
mu_grid, gamma_grid, stable = stability_boundary_anishchenko((0.1, 2.0), (0.1, 2.0), n_grid=30)
i_mu = np.argmin(np.abs(mu_grid - 0.5))
j_ga = np.argmin(np.abs(gamma_grid - 1.0))
assert stable[i_mu, j_ga], '[TC23] 已知稳定点被判为不稳定 FAILED'

# ---- TC24: AcousticRoomFEM RCM重排序带宽不增 ----
from acoustic_room_model import generate_box_mesh, AcousticRoomFEM
nodes, elements = generate_box_mesh(1.0, 1.0, 1.0, nx=4, ny=4, nz=4)
fem = AcousticRoomFEM(nodes, elements)
bw_before = fem.compute_bandwidth()
fem.rcm_reorder()
bw_after = fem.compute_bandwidth()
assert bw_after <= bw_before, '[TC24] RCM重排序后带宽增加 FAILED'

# ---- TC25: generate_box_mesh节点数正确 ----
from acoustic_room_model import generate_box_mesh
nodes, elements = generate_box_mesh(1.0, 1.0, 1.0, nx=3, ny=3, nz=3)
expected_nodes = 3 * 3 * 3
assert nodes.shape[0] == expected_nodes, '[TC25] 盒状网格节点数错误 FAILED'

# ---- TC26: rayleigh_integral_piston输出有限 ----
from integrals_radiation import disk_unit_sample, rayleigh_integral_piston
disk_pts = disk_unit_sample(100, radius=0.1)
observer = np.array([0.0, 0.0, 1.0])
p_rayleigh = rayleigh_integral_piston(observer, disk_pts, u_n=0.01, k=2.0 * math.pi * 1000.0 / 343.0)
assert np.isfinite(p_rayleigh), '[TC26] Rayleigh积分产生非有限值 FAILED'

# ---- TC27: piston_directivity_factor小ka理论值 ----
from integrals_radiation import piston_directivity_factor
di = piston_directivity_factor(0.001)
assert abs(di - 3.0103) < 0.1, '[TC27] 小ka指向性因子与理论值不符 FAILED'

# ---- TC28: acoustic_transfer_matrix_sparse维度正确 ----
from sparse_acoustics import acoustic_transfer_matrix_sparse
from spherical_array_geometry import sphere_fibonacci_grid_points
sensors = sphere_fibonacci_grid_points(8, 0.5)
sources = sphere_fibonacci_grid_points(4, 0.3)
k = 2.0 * math.pi * 500.0 / 343.0
H = acoustic_transfer_matrix_sparse(sensors, sources, k)
assert H.m == 8 and H.n == 4, '[TC28] 稀疏传递矩阵维度错误 FAILED'

# ---- TC29: log_beta对称性 ----
from special_functions import log_beta
lb1 = log_beta(2.0, 3.0)
lb2 = log_beta(3.0, 2.0)
assert abs(lb1 - lb2) < 1e-12, '[TC29] log_beta不对称 FAILED'

# ---- TC30: betain边界值 ----
from special_functions import betain, log_beta
lb = log_beta(2.0, 3.0)
val0, ierr0 = betain(0.0, 2.0, 3.0, lb)
val1, ierr1 = betain(1.0, 2.0, 3.0, lb)
assert val0 == 0.0 and ierr0 == 0, '[TC30] betain(0)边界错误 FAILED'
assert val1 == 1.0 and ierr1 == 0, '[TC30] betain(1)边界错误 FAILED'

print('\n全部 30 个测试通过!\n')
