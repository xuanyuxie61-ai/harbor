
import numpy as np
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parameters import (
    R0, a_minor, B0, KAPPA, DELTA, q0, q_edge,
    N_E_AXIS, T_E_AXIS, Z_EFF, N_FFT
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_equilibrium():
    from equilibrium_solver import solve_grad_shafranov, compute_magnetic_field

    print_section("1. Grad-Shafranov 平衡求解")
    psi, R_grid, Z_grid, info = solve_grad_shafranov(
        max_iter=300, tol=1e-7, relaxation=0.3
    )
    print(f"  迭代次数: {info['iterations']}")
    print(f"  最终残差: {info['final_error']:.6e}")
    print(f"  磁轴磁通 ψ_axis: {info['psi_axis']:.6f} Wb/rad")
    print(f"  边界磁通 ψ_edge: {info['psi_edge']:.6f} Wb/rad")
    print(f"  安全因子 q(边缘): {info['q_profile'][-2]:.3f}")

    B_R, B_Z, B_phi = compute_magnetic_field(psi, R_grid, Z_grid)
    B_total = np.sqrt(B_R ** 2 + B_Z ** 2 + B_phi ** 2)
    print(f"  轴心磁场强度: {B_total[len(R_grid)//2, len(Z_grid)//2]:.4f} T")
    return psi, R_grid, Z_grid, info, B_total


def run_particle_drift():
    from particle_drift import simulate_guiding_center, compute_adiabatic_invariant

    print_section("2. 引导中心漂移运动模拟 (类比刚体 Euler 方程)")
    t_arr, y_arr, energy = simulate_guiding_center(n_steps=3000)
    print(f"  模拟时长: {t_arr[-1]:.2f} s")
    print(f"  初始速度: v_R={y_arr[0,0]:.4f}, v_Z={y_arr[0,1]:.4f}, v∥={y_arr[0,2]:.4f} m/s")
    print(f"  末态速度: v_R={y_arr[-1,0]:.4f}, v_Z={y_arr[-1,1]:.4f}, v∥={y_arr[-1,2]:.4f} m/s")
    print(f"  归一化动能范围: [{energy.min():.4f}, {energy.max():.4f}]")


    B_dummy = np.full(len(t_arr), B0)
    mu_arr, mu_rstd = compute_adiabatic_invariant(y_arr, B_dummy)
    print(f"  磁矩相对标准差: {mu_rstd:.6e} (衡量绝热不变性)")
    return t_arr, y_arr, energy


def run_fusion_kinetics():
    from fusion_kinetics import simulate_fusion_burn, compute_bremsstrahlung, lawson_criterion

    print_section("3. D-T 聚变反应动力学")
    t_arr, y_arr, P_fus, Q_factor = simulate_fusion_burn(Ti_keV=15.0, n_steps=2000)
    print(f"  模拟时长: {t_arr[-1]:.1f} s")
    print(f"  稳态燃料密度: n_D={y_arr[-1,0]:.4e}, n_T={y_arr[-1,1]:.4e} m^-3")
    print(f"  稳态灰密度: n_He={y_arr[-1,2]:.4e} m^-3")
    print(f"  聚变功率密度: {P_fus[-1]:.4e} W/m³")
    print(f"  增益因子 Q: {Q_factor[-1]:.4f}")


    n_e = y_arr[-1, 0] + 2.0 * y_arr[-1, 2]
    P_brem = compute_bremsstrahlung(n_e, T_E_AXIS, Z_EFF)
    print(f"  轫致辐射损失: {P_brem:.4e} W/m³")


    ntau = lawson_criterion(15.0)
    print(f"  15 keV Lawson nτ_E 判据: {ntau:.4e} s·m⁻³")
    return t_arr, y_arr, P_fus, Q_factor


def run_transport_dde():
    from transport_dde import simulate_energy_transport, compute_confinement_time_scaling
    from transport_dde import compute_particle_diffusivity

    print_section("4. 能量输运延迟微分方程 (Mackey-Glass 型 ITB 模型)")
    t_arr, W_arr, P_loss, info = simulate_energy_transport(n_steps=2000)
    print(f"  延迟时间 τ: {info['delay_tau']:.3f} s")
    print(f"  Mackey-Glass 非线性指数 n: {info['mackey_glass_n']:.2f}")
    print(f"  平均能量密度: {info['mean_energy_density']:.4e} J/m³")
    print(f"  最大能量密度: {info['max_energy_density']:.4e} J/m³")
    print(f"  Lyapunov 指数近似: {info['lyapunov_approx']:.6f}")


    tau_E = compute_confinement_time_scaling(
        I_p=15.0, B_t=B0, n_e20=N_E_AXIS / 1e20,
        P_loss=50.0, R=R0, a=a_minor, kappa=KAPPA
    )
    print(f"  ITER89-P τ_E: {tau_E:.4f} s")


    nu_ei = 1.0e3
    rho_i = 3.0e-3
    D_neo = compute_particle_diffusivity(q=2.0, R0=R0, a=a_minor,
                                         nu_ei=nu_ei, rho_i=rho_i)
    print(f"  新经典扩散系数 D_neo: {D_neo:.4e} m²/s")
    return t_arr, W_arr, P_loss, info


def run_quadrature_and_fem():
    from quadrature_engine import (
        gauss_quadrature, toroidal_volume_integral,
        line_fekete_points, triangle_quadrature,
        assemble_stiffness_triangle
    )
    from geometry_utils import generate_triangular_mesh

    print_section("5. 高阶求积与有限元刚度矩阵")


    def fusion_power_density(r, theta):
        return 1.0e6 * (1.0 - (r / a_minor) ** 2) ** 2

    P_total = toroidal_volume_integral(fusion_power_density, R0, a_minor, KAPPA,
                                       n_radial=32, n_theta=32)
    print(f"  环向聚变功率体积积分: {P_total:.4e} W")


    vert1 = np.array([R0, 0.0])
    vert2 = np.array([R0 + a_minor, 0.0])
    vert3 = np.array([R0, a_minor * KAPPA])
    tri_integral = triangle_quadrature(
        lambda x, y: 1.0, vert1, vert2, vert3, precision=7
    )
    print(f"  参考三角形面积 (求积): {tri_integral:.6f} m²")


    xf, wf, Vf = line_fekete_points(m=8, a=R0 - a_minor, b=R0 + a_minor, n_sample=200)
    print(f"  Fekete 点数量: {len(xf)}")
    print(f"  Fekete 权重范围: [{wf.min():.6e}, {wf.max():.6e}]")


    vertices, triangles = generate_triangular_mesh(n_r=6, n_theta=12)
    K_loc = assemble_stiffness_triangle(vertices[0], vertices[1], vertices[2])
    print(f"  示例局部刚度矩阵条件数: {np.linalg.cond(K_loc):.4f}")
    print(f"  全局网格: {len(vertices)} 顶点, {len(triangles)} 三角形")
    return P_total, xf, wf, vertices, triangles


def run_spectral_analysis():
    from spectral_analysis import (
        generate_turbulent_signal, compute_fft_spectrum,
        compute_growth_rate_from_spectrum, alfvén_dispersion
    )

    print_section("6. 等离子体湍流 FFT 谱分析")
    signal, true_params = generate_turbulent_signal(n_t=N_FFT, dt=1.0e-4)
    freqs, power = compute_fft_spectrum(signal, dt=1.0e-4)
    print(f"  信号采样点数: {len(signal)}")
    print(f"  频率分辨率: {freqs[1] - freqs[0]:.2f} Hz")


    peak_idx = np.argmax(power[1:]) + 1
    print(f"  主导频率峰值: {freqs[peak_idx]:.2f} Hz")
    print(f"  对应功率密度: {power[peak_idx]:.4e}")



    power_history = power[:len(power) // 4]
    if len(power_history) > 10:
        gamma, r2 = compute_growth_rate_from_spectrum(power_history, dt=1.0e-4)
        print(f"  谱功率增长率估计: γ = {gamma:.4f} 1/s, R² = {r2:.4f}")


    k_par = 1.0 / R0
    k_perp = 2.0 / a_minor
    omega_A, v_A = alfvén_dispersion(k_par, k_perp, B0, rho_m=1.0e-19)
    print(f"  阿尔芬速度 v_A: {v_A:.4e} m/s")
    print(f"  剪切阿尔芬频率 ω_A: {omega_A:.4e} rad/s")
    return freqs, power


def run_collision_transport():
    from collision_transport import (
        electron_ion_collision_frequency, mean_free_path,
        hypersphere_velocity_sampling, rectangle_collision_distance_stats,
        compute_transport_coefficients, coulomb_logarithm
    )

    print_section("7. 库仑碰撞统计与输运系数")
    lnL = coulomb_logarithm(N_E_AXIS, T_E_AXIS)
    nu_ei = electron_ion_collision_frequency(N_E_AXIS, T_E_AXIS, Z_EFF)
    mfp = mean_free_path(N_E_AXIS, T_E_AXIS, Z_EFF)
    print(f"  Coulomb 对数 ln Λ: {lnL:.3f}")
    print(f"  电子-离子碰撞频率 ν_ei: {nu_ei:.4e} Hz")
    print(f"  电子平均自由程 λ_e: {mfp:.4e} m")


    stats_3d = hypersphere_velocity_sampling(m_dim=3, n_samples=5000, T_e_eV=T_E_AXIS)
    print(f"  3D 速度空间夹角均值: {stats_3d['theta_mean_deg']:.2f}°")
    print(f"  3D 速度空间夹角标准差: {stats_3d['theta_std_rad']:.4f} rad")

    stats_5d = hypersphere_velocity_sampling(m_dim=5, n_samples=5000, T_e_eV=T_E_AXIS)
    print(f"  5D 速度空间夹角均值: {stats_5d['theta_mean_deg']:.2f}° (高维趋近 90°)")


    dist_stats = rectangle_collision_distance_stats(a=a_minor, b=a_minor * KAPPA, n_samples=20000)
    print(f"  磁面patch平均碰撞距离: {dist_stats['mean_distance']:.4f} m")
    print(f"  距离标准差: {dist_stats['std_distance']:.4f} m")


    coeffs = compute_transport_coefficients(N_E_AXIS, T_E_AXIS, B0, Z_EFF)
    print(f"  经典扩散系数 D_cl: {coeffs['D_classical_m2s']:.4e} m²/s")
    print(f"  新经典扩散系数 D_neo: {coeffs['D_neoclassical_m2s']:.4e} m²/s")
    print(f"  电子热导率 χ_e: {coeffs['chi_e_m2s']:.4e} m²/s")
    print(f"  离子热导率 χ_i: {coeffs['chi_i_m2s']:.4e} m²/s")
    return stats_3d, dist_stats, coeffs


def run_geometry_analysis():
    from geometry_utils import (
        point_in_flux_surface, compute_poloidal_area,
        compute_toroidal_volume, fekete_points_on_flux_surface,
        compute_curvature_and_torsion
    )

    print_section("8. 磁面几何与谱元插值")


    test_points = [(R0, 0.0), (R0 + 2.0 * a_minor, 0.0), (R0, a_minor * KAPPA * 0.5)]
    for R_test, Z_test in test_points:
        inside, R_poly, Z_poly = point_in_flux_surface(R_test, Z_test)
        status = "内部" if inside else "外部"
        print(f"  点 ({R_test:.2f}, {Z_test:.2f}): {status}")


    area, _, _ = compute_poloidal_area(n_theta=256)
    volume, volume_approx = compute_toroidal_volume(n_theta=128, n_radial=64)
    print(f"  极向截面面积: {area:.4f} m²")
    print(f"  环向等离子体体积: {volume:.4f} m³")
    print(f"  解析近似体积: {volume_approx:.4f} m³")
    print(f"  体积相对误差: {abs(volume - volume_approx) / volume * 100:.2f}%")


    theta_f, R_f, Z_f, w_f = fekete_points_on_flux_surface(m=12, n_sample=300)
    print(f"  LCFS 上 Fekete 节点数: {len(theta_f)}")
    print(f"  首节点: θ={theta_f[0]:.4f} rad, R={R_f[0]:.4f} m, Z={Z_f[0]:.4f} m")


    theta_sample = np.linspace(0, 2.0 * np.pi, 200)
    kappa, rho_c = compute_curvature_and_torsion(theta_sample)
    print(f"  最大曲率: {kappa.max():.6f} 1/m")
    print(f"  最小曲率半径: {rho_c.min():.4f} m")
    return area, volume, theta_f, R_f, Z_f


def run_matrix_algebra():
    from matrix_algebra import (
        r8utt_det, r8utt_solve, write_matrix_market, read_matrix_market,
        write_hb_file, assemble_global_stiffness, solve_stiffness_system,
        condition_number_estimate
    )
    from geometry_utils import generate_triangular_mesh

    print_section("9. 稀疏矩阵代数与格式转换")


    n = 8
    a_utt = np.array([2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    det_val = r8utt_det(n, a_utt)
    print(f"  UTT 行列式 det(A) = a0^n = {det_val:.4f}")
    b_test = np.ones(n)
    x_utt = r8utt_solve(n, a_utt, b_test)
    residual = np.linalg.norm(np.tril(np.ones((n, n)) * a_utt[0], 0) @ x_utt - b_test
                              + np.triu(np.outer(np.arange(n), np.ones(n)), 1) @ x_utt * 0)

    A_utt = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            A_utt[i, j] = a_utt[j - i]
    res_true = np.linalg.norm(A_utt @ x_utt - b_test)
    print(f"  UTT 求解残差: {res_true:.6e}")


    vertices, triangles = generate_triangular_mesh(n_r=5, n_theta=10)
    K_global = assemble_global_stiffness(vertices, triangles)
    print(f"  全局刚度矩阵维度: {K_global.shape}")
    print(f"  刚度矩阵条件数估计: {condition_number_estimate(K_global):.4e}")


    b_fem = np.random.randn(K_global.shape[0])
    x_fem, info_cg = solve_stiffness_system(K_global, b_fem)
    print(f"  CG 迭代次数: {info_cg['iterations']}")
    print(f"  CG 最终残差: {info_cg['residual']:.6e}")


    mm_path = "/tmp/tokamak_stiffness.mtx"
    write_matrix_market(mm_path, K_global[:20, :20], title="Tokamak Stiffness")
    A_read, info_mm = read_matrix_market(mm_path)
    print(f"  Matrix Market 读写验证: 非零元 {info_mm['nnz']}, 维度 {A_read.shape}")


    hb_path = "/tmp/tokamak_stiffness.hb"
    write_hb_file(hb_path, K_global[:15, :15], title="Tokamak", key="TOK1")
    hb_size = os.path.getsize(hb_path)
    print(f"  HB 文件大小: {hb_size} bytes")
    return K_global, x_fem


def run_mhd_stability():
    from mhd_stability import (
        build_mhd_transition_matrix, mhd_markov_evolution,
        compute_ideal_mhd_delta_w, compute_mercier_criterion,
        compute_critical_beta
    )
    from equilibrium_solver import f_profile, pressure_profile

    print_section("10. MHD 稳定性分析 (Markov 状态转移 + δW)")


    P, labels = build_mhd_transition_matrix()
    initial = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    history, absorption_time = mhd_markov_evolution(P, initial, n_steps=200)
    print(f"  MHD 状态数: {len(labels)}")
    for i, lab in enumerate(labels):
        print(f"    {i}: {lab:20s} 稳态概率 {history[-1, i]:.4f}")
    print(f"  期望破裂时间: {absorption_time:.1f} 时间步")





    raise NotImplementedError("此处需补全 MHD 稳定性分析所需的剖面构建与磁场计算")

    delta_w, stability = compute_ideal_mhd_delta_w(
        m_mode=2, n_mode=1, q_profile=q_prof, r_grid=r_grid,
        p_profile=p_prof, B_theta=B_theta, B_phi=B_phi_prof
    )
    print(f"  (m=2, n=1) 理想 δW: {delta_w:.4e} J")
    print(f"  稳定性判定: {stability}")


    D_M, unstable_regions = compute_mercier_criterion(
        q_prof, r_grid, p_prof, B_phi_prof, B_theta
    )
    print(f"  Mercier 不稳定区域数: {len(unstable_regions)}")
    for r_s, r_e in unstable_regions[:3]:
        print(f"    r ∈ [{r_s:.3f}, {r_e:.3f}] m")


    beta_c = compute_critical_beta(q_prof, r_grid, B_phi_prof)
    print(f"  临界比压 β_c (Troyon): {beta_c:.2f}%")
    return P, history, delta_w, beta_c


def main():
    np.random.seed(42)
    print("\n" + "#" * 70)
    print("#  托卡马克磁约束聚变综合模拟系统")
    print("#  Tokamak Magnetic Confinement Fusion Integrated Simulator")
    print("#" * 70)
    print(f"\n  运行时间: {__import__('datetime').datetime.now().isoformat()}")
    print(f"  NumPy 版本: {np.__version__}")

    try:

        psi, R_grid, Z_grid, eq_info, B_total = run_equilibrium()


        t_drift, y_drift, energy_drift = run_particle_drift()


        t_fus, y_fus, P_fus, Q_fus = run_fusion_kinetics()


        t_trans, W_trans, P_loss_trans, info_trans = run_transport_dde()


        P_int, xf, wf, verts, tris = run_quadrature_and_fem()


        freqs, power = run_spectral_analysis()


        stats_3d, dist_stats, coeffs = run_collision_transport()


        area, volume, theta_f, R_f, Z_f = run_geometry_analysis()


        K_global, x_fem = run_matrix_algebra()


        P_mhd, hist_mhd, delta_w, beta_c = run_mhd_stability()


        print("\n" + "#" * 70)
        print("#  计算总结")
        print("#" * 70)
        print(f"  Grad-Shafranov 平衡: 收敛于 {eq_info['iterations']} 次迭代")
        print(f"  聚变增益因子 Q: {Q_fus[-1]:.4f}")
        print(f"  等离子体体积: {volume:.4f} m³")
        print(f"  MHD 临界比压: {beta_c:.2f}%")
        print(f"  程序正常结束，无报错。")
        print("#" * 70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] 运行过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
