
import numpy as np
import math
import sys


def main():
    print("=" * 70)
    print("  博士级主动噪声控制综合仿真系统")
    print("  领域: 声学工程 - 主动噪声控制与自适应滤波")
    print("=" * 70)




    print("\n[1] 一维管道Helmholtz方程模态分析")
    print("-" * 50)
    from tridiagonal_acoustics import pipe_helmholtz_solver, lindberg_exact_solution, lindberg_residual

    L_pipe = 1.0
    N_pts = 100
    f_freq = 500.0
    c0 = 343.0
    k = 2.0 * math.pi * f_freq / c0


    source = np.zeros(N_pts, dtype=complex)
    source[N_pts // 2] = 1.0e-3
    x_pipe, p_pipe = pipe_helmholtz_solver(L_pipe, N_pts, k, source)
    print(f"  频率: {f_freq} Hz, 波数: {k:.3f} rad/m")
    print(f"  管道中部声压幅值: {abs(p_pipe[N_pts//2]):.6e} Pa")
    print(f"  声压实部范围: [{np.real(p_pipe).min():.4e}, {np.real(p_pipe).max():.4e}]")


    t_test = np.linspace(0, 1.0, 11)
    y_exact, dydt_exact = lindberg_exact_solution(t_test)
    res = lindberg_residual(t_test, y_exact, dydt_exact)
    max_res = np.max(np.abs(res))
    print(f"  Lindberg ODE 残差验证: max|residual| = {max_res:.3e} (应接近0)")




    print("\n[2] Fibonacci球面阵列几何布置")
    print("-" * 50)
    from spherical_array_geometry import sphere_fibonacci_grid_points

    N_sensors = 32
    radius = 0.5
    sensors = sphere_fibonacci_grid_points(N_sensors, radius)
    print(f"  生成 {N_sensors} 个球面传感器位置 (半径 {radius} m)")
    print(f"  第一个传感器坐标: ({sensors[0,0]:.4f}, {sensors[0,1]:.4f}, {sensors[0,2]:.4f})")
    print(f"  最后一个传感器坐标: ({sensors[-1,0]:.4f}, {sensors[-1,1]:.4f}, {sensors[-1,2]:.4f})")


    min_dist = np.inf
    for i in range(N_sensors):
        for j in range(i + 1, N_sensors):
            d = np.linalg.norm(sensors[i] - sensors[j])
            if d < min_dist:
                min_dist = d
    print(f"  最小传感器间距: {min_dist:.4f} m")




    print("\n[3] 稀疏声学传递矩阵构建")
    print("-" * 50)
    from sparse_acoustics import acoustic_transfer_matrix_sparse, generate_room_coupling_graph










    raise NotImplementedError("Hole 3: 稀疏声学传递矩阵构建与验证 待实现")




    print("\n[4] 次级声源相位角非线性优化")
    print("-" * 50)
    from source_phase_optimizer import optimize_source_phase


    np.random.seed(42)
    H_col = (np.random.randn(N_sensors) + 1j * np.random.randn(N_sensors)) * 0.5
    d_noise = (np.random.randn(N_sensors) + 1j * np.random.randn(N_sensors)) * 0.3
    amplitude = 0.5

    phi_opt, min_energy = optimize_source_phase(H_col, d_noise, amplitude)
    print(f"  最优相位角: {phi_opt:.4f} rad ({math.degrees(phi_opt):.2f} deg)")
    print(f"  最小声能量: {min_energy:.6e}")


    s0 = amplitude * np.exp(1j * 0.0)
    p0 = d_noise + H_col * s0
    energy0 = np.vdot(p0, p0).real
    reduction_db = 10.0 * math.log10((min_energy + 1e-18) / (energy0 + 1e-18))
    print(f"  相比零相位的能量降低: {reduction_db:.2f} dB")




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


    A_test = np.random.randn(20, 5)
    b_test = np.random.randn(20)
    w_qr, rank = qr_rank_revealing_ls(A_test, b_test)
    residual_norm = np.linalg.norm(A_test @ w_qr - b_test)
    print(f"  QR秩揭示最小二乘残差: {residual_norm:.4e}, 秩={rank}")




    print("\n[6] 最优次级声源子集选择")
    print("-" * 50)
    from optimal_source_selection import greedy_source_selection, subset_sum_swap_anc


    powers = np.array([50, 80, 120, 40, 90, 60, 110, 30], dtype=float)
    budget = 250.0
    selected_power, achieved = subset_sum_swap_anc(powers, budget)
    print(f"  功率预算: {budget} W")
    print(f"  选中索引: {np.where(selected_power)[0].tolist()}")
    print(f"  实际功耗: {achieved:.1f} W")


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




    print("\n[7] 噪声Dirichlet统计建模与自适应步长")
    print("-" * 50)
    from statistical_noise_model import dirichlet_estimate_mle, adaptive_step_size_from_dirichlet, noise_stationarity_test


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


    err_hist = np.cumsum(rng.normal(0, 0.01, 200)) + rng.normal(0, 0.1, 200)
    is_stat, f_stat = noise_stationarity_test(err_hist, window=40)
    print(f"  噪声平稳性检验: F统计量={f_stat:.3f}, 平稳={is_stat}")




    print("\n[8] 圆形活塞声辐射积分计算")
    print("-" * 50)
    from integrals_radiation import disk_unit_sample, rayleigh_integral_piston, piston_directivity_factor, piston_radiation_resistance


    n_samp = 200
    a_piston = 0.1
    disk_pts = disk_unit_sample(n_samp, radius=a_piston)
    print(f"  活塞半径: {a_piston} m, 采样点数: {n_samp}")


    observer = np.array([0.0, 0.0, 1.0])
    k_piston = 2.0 * math.pi * 1000.0 / c0
    p_rayleigh = rayleigh_integral_piston(observer, disk_pts, u_n=0.01, k=k_piston)
    print(f"  1kHz时1m处声压: {abs(p_rayleigh):.6e} Pa")


    ka = k_piston * a_piston
    di = piston_directivity_factor(ka)
    print(f"  ka={ka:.3f}, 指向性因子 DI={di:.2f} dB")


    R_ratio = piston_radiation_resistance(ka)
    print(f"  辐射阻力比 R_r/(rho0 c0 S)={R_ratio:.4f}")


    from special_functions import cos_power_int, betain, digamma, trigamma, log_beta
    cpi = cos_power_int(0.0, math.pi / 2, 4)
    print(f"  cos^4积分 [0,pi/2]: {cpi:.6f} (理论值=3pi/16={3*math.pi/16:.6f})")

    beta_val, ierr = betain(0.5, 2.0, 3.0, log_beta(2.0, 3.0))
    print(f"  I_0.5(2,3) = {beta_val:.6f} (ifault={ierr})")

    psi_val, _ = digamma(2.5)
    psi_prime, _ = trigamma(2.5)
    print(f"  digamma(2.5)={psi_val:.6f}, trigamma(2.5)={psi_prime:.6f}")




    print("\n[9] 非线性自适应系统动力学分析")
    print("-" * 50)
    from nonlinear_ode_dynamics import anishchenko_adaptive_deriv, rk4_integrate, stability_boundary_anishchenko


    mu_grid, gamma_grid, stable = stability_boundary_anishchenko((0.1, 2.0), (0.1, 2.0), n_grid=30)
    stable_fraction = np.sum(stable) / stable.size
    print(f"  参数空间(mu vs gamma)稳定区域占比: {stable_fraction*100:.1f}%")


    traj = rk4_integrate(
        lambda t, y: anishchenko_adaptive_deriv(t, y, mu=1.2, eta=0.5),
        0.0, [-0.1, 0.5, -0.6], 50.0, h=0.05
    )
    final_state = traj[-1][1]
    print(f"  Anishchenko-like系统积分 (t=0..50):")
    print(f"    最终状态: w1={final_state[0]:.4f}, w2={final_state[1]:.4f}, e={final_state[2]:.4f}")
    print(f"    轨迹点数: {len(traj)}")




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


    k_fem = 2.0 * math.pi * 200.0 / c0
    A_fem, b_fem = fem.assemble_system(k_fem)

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


    print("\n" + "=" * 70)
    print("  仿真全部完成. 所有模块运行正常.")
    print("=" * 70)


if __name__ == "__main__":
    main()
