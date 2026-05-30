import numpy as np
import sys


from spectral_methods import (
    polynomial_multiply, chebyshev_polynomial, legendre_polynomial,
    spectral_differentiation_matrix, apply_angular_spectral_derivative
)
from matrix_solvers import (
    R8SDMatrix, r8sd_cg, R8PBLMatrix, build_poisson_r8sd,
    build_band_spd_matrix
)
from mesh_generation import (
    triangle_grid, triangle_grid_count, generate_disk_cross_section_mesh,
    triangulation_mask, mesh_to_fem_format, compute_triangle_area,
    mask_disk_jet_region, mask_black_hole_horizon
)
from quadrature_rules import (
    legendre_set, legendre_3d_set, monomial_integral_3d,
    test_quadrature_exactness, cylindrical_quadrature
)
from rbf_interpolation import (
    rbf_weights, rbf_interpolate, rbf_gradient_2d,
    phi_multiquadric, phi_gaussian
)
from hydrodynamics import (
    rk3_integrate, gradient_2d, gradient_1d, laplacian_1d,
    divergence_cylindrical, compute_cfl_timestep
)
from fem_radial import (
    lagrange_basis_1d, lagrange_basis_derivative_1d,
    fem1d_mass_matrix, fem1d_stiffness_matrix,
    solve_fem1d_radial, fem_interpolate_1d
)
from monte_carlo_transport import (
    wedge01_sample, wedge01_monomial_integral, wedge_monte_carlo_integral,
    ball_unit_sample, ball_distance_stats, ball_distance_pdf,
    sample_jet_particles, mc_jet_energy_transport, compute_correlation_function
)
from particle_dynamics import (
    velocity_verlet_step, compute_pair_forces, compute_kinetic_energy,
    run_particle_simulation, accretion_disk_particle_model
)
from accretion_physics import (
    G_GRAV, C_LIGHT, M_SUN, SIGMA_SB, K_BOLTZMANN,
    keplerian_angular_velocity, sound_speed, scale_height,
    shakura_sunyaev_sigma, viscous_torque,
    schwarzschild_potential, paczynski_wiita_potential,
    jet_launching_criterion, magnetic_braking_torque,
    disk_spectrum_nu, disk_instability_criterion, compute_radial_velocity
)
from utils import (
    magic4_matrix, normalized_magic_weights, ball_unit_sample as utils_ball_sample,
    distance_stats, safe_divide, clip_with_warning
)


def run_simulation():
    print("=" * 80)
    print("  多维吸积盘流体力学数值模拟与磁离心喷流形成机制研究")
    print("=" * 80)




    M_bh = 10.0 * M_SUN
    M_dot = 1e14
    alpha = 0.1
    r_isco = 6.0 * G_GRAV * M_bh / C_LIGHT ** 2


    r_in = r_isco
    r_out = 500.0 * r_isco
    z_max = 0.1 * r_out

    print(f"\n[物理参数]")
    print(f"  黑洞质量 M_bh = {M_bh / M_SUN:.2f} M_sun")
    print(f"  ISCO 半径 r_isco = {r_isco / 1e3:.3f} km")
    print(f"  吸积率 M_dot = {M_dot:.3e} kg/s")
    print(f"  粘滞参数 alpha = {alpha}")
    print(f"  模拟区域: r in [{r_in/1e3:.2f}, {r_out/1e3:.2f}] km")




    print("\n" + "-" * 60)
    print("Step 1: Shakura-Sunyaev 薄盘径向结构")
    print("-" * 60)

    n_r = 64
    r_grid = np.linspace(r_in, r_out, n_r)
    dr = r_grid[1] - r_grid[0]

    Sigma, T_disk, H_disk = shakura_sunyaev_sigma(r_grid, M_dot, M_bh, alpha)
    v_r = compute_radial_velocity(Sigma, r_grid, M_bh, alpha, M_dot)
    omega_k = keplerian_angular_velocity(r_grid, M_bh)
    cs_disk = sound_speed(T_disk)

    print(f"  径向网格数: {n_r}")
    print(f"  表面密度范围: [{np.min(Sigma):.3e}, {np.max(Sigma):.3e}] kg/m^2")
    print(f"  温度范围: [{np.min(T_disk):.2e}, {np.max(T_disk):.2e}] K")
    print(f"  标高范围: [{np.min(H_disk)/1e3:.3f}, {np.max(H_disk)/1e3:.3f}] km")
    print(f"  径向速度范围: [{np.min(v_r):.3e}, {np.max(v_r):.3e}] m/s")




    print("\n" + "-" * 60)
    print("Step 2: 1D FEM 径向结构方程求解")
    print("-" * 60)

    def ss_source(r):

        return 1e-6 * np.exp(-r / r_out)

    r_fem, Sigma_fem = solve_fem1d_radial(
        r_in, r_out, n_elements=16, order=2,
        source_func=ss_source,
        bc_left={'type': 'dirichlet', 'value': Sigma[0]},
        bc_right={'type': 'dirichlet', 'value': Sigma[-1]}
    )
    print(f"  FEM 节点数: {len(r_fem)}")
    print(f"  FEM 表面密度范围: [{np.min(Sigma_fem):.3e}, {np.max(Sigma_fem):.3e}] kg/m^2")




    print("\n" + "-" * 60)
    print("Step 3: 泊松方程 CG 求解器")
    print("-" * 60)


    A_poisson = build_poisson_r8sd(n_r, dr)




    rho_midplane = None
    rho_midplane = np.where(rho_midplane < 0, 0, rho_midplane)
    b_poisson = 4.0 * np.pi * G_GRAV * rho_midplane

    phi_grav, cg_info = r8sd_cg(A_poisson, b_poisson, tol=1e-12)
    print(f"  CG 迭代次数: {cg_info['iterations']}")
    print(f"  CG 残差: {cg_info['residual']:.3e}")
    print(f"  CG 收敛: {cg_info['converged']}")
    print(f"  引力势范围: [{np.min(phi_grav):.3e}, {np.max(phi_grav):.3e}]")


    phi_analytic = schwarzschild_potential(r_grid, M_bh)
    rel_error = np.mean(np.abs(phi_grav - phi_analytic) / (np.abs(phi_analytic) + 1e-30))
    print(f"  与 Newton 势平均相对误差: {rel_error:.3e}")




    print("\n" + "-" * 60)
    print("Step 4: R8PBL 带状 SPD 求解器 (压力方程)")
    print("-" * 60)


    ml = 2
    A_band = build_band_spd_matrix(n_r, ml, condition_hint=1.0)


    b_pressure = -rho_midplane * np.gradient(v_r, dr)
    b_pressure = b_pressure[:n_r]

    phi_pressure = A_band.cholesky_band_solve(b_pressure)
    print(f"  带状矩阵阶数: {n_r}, 带宽: {ml}")
    print(f"  压力场范围: [{np.min(phi_pressure):.3e}, {np.max(phi_pressure):.3e}]")




    print("\n" + "-" * 60)
    print("Step 5: 谱方法 (切比雪夫/勒让德多项式 + 角向导数)")
    print("-" * 60)


    T5 = chebyshev_polynomial(5)
    P5 = legendre_polynomial(5)
    conv = polynomial_multiply(T5, P5)
    print(f"  T_5 系数: {T5}")
    print(f"  P_5 系数: {P5}")
    print(f"  T_5 * P_5 次数: {len(conv)-1}")


    D_cheb, x_cheb = spectral_differentiation_matrix(8)
    f_test = np.sin(np.arccos(x_cheb))
    df_test = D_cheb @ f_test
    print(f"  谱微分矩阵条件数: {np.linalg.cond(D_cheb):.3e}")


    phi_grid = np.linspace(0, 2 * np.pi, 128)
    f_phi = np.sin(3 * phi_grid)
    df_dphi = apply_angular_spectral_derivative(f_phi, N_modes=8)
    df_exact = 3 * np.cos(3 * phi_grid)
    err_angular = np.max(np.abs(df_dphi - df_exact))
    print(f"  角向谱导数最大误差: {err_angular:.3e}")




    print("\n" + "-" * 60)
    print("Step 6: 吸积盘截面网格生成与掩码")
    print("-" * 60)

    nodes, elements = generate_disk_cross_section_mesh(
        r_in / 1e3, r_out / 1e3, z_max / 1e3, n_r=20, n_z=10
    )
    print(f"  原始网格: {len(nodes)} 节点, {len(elements)} 单元")


    r_jet_launch = 3.0 * r_isco / 1e3
    z_jet = 0.05 * r_out / 1e3
    nodes_masked, elements_masked = mask_disk_jet_region(
        nodes, elements, r_jet_launch, z_jet
    )
    print(f"  掩码后网格: {len(nodes_masked)} 节点, {len(elements_masked)} 单元")


    fem_mesh = mesh_to_fem_format(nodes_masked, elements_masked)
    print(f"  FEM 格式: {fem_mesh['n_nodes']} 节点, {fem_mesh['n_elements']} 单元")


    tri_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tri_pts = triangle_grid(5, tri_vertices)
    print(f"  三角形细分网格点数: {len(tri_pts)}")




    print("\n" + "-" * 60)
    print("Step 7: 3D 高斯-勒让德求积规则验证")
    print("-" * 60)

    box = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    errors = test_quadrature_exactness(3, 3, 3, box, max_total_degree=4)
    max_err = max(errors.values())
    print(f"  3x3x3 GL 求积对总次数<=4的多项式最大相对误差: {max_err:.3e}")


    cyl_pts, cyl_w = cylindrical_quadrature(4, 8, 3, r_in, r_out, -z_max, z_max)
    print(f"  柱坐标求积节点数: {len(cyl_pts)}")


    mass_integrand = np.zeros(len(cyl_pts))
    for i in range(len(cyl_pts)):
        r_p = cyl_pts[i, 0]
        idx_r = min(int((r_p - r_in) / (r_out - r_in) * (n_r - 1)), n_r - 2)
        idx_r = max(0, idx_r)
        t = (r_p - r_grid[idx_r]) / (r_grid[idx_r + 1] - r_grid[idx_r])
        t = max(0.0, min(1.0, t))
        sigma_interp = (1 - t) * Sigma[idx_r] + t * Sigma[idx_r + 1]
        mass_integrand[i] = sigma_interp

    total_mass = np.dot(cyl_w, mass_integrand)
    print(f"  盘总质量估计: {total_mass / M_SUN:.3e} M_sun")




    print("\n" + "-" * 60)
    print("Step 8: RBF 场重构")
    print("-" * 60)


    n_data = 20
    r_data = np.linspace(r_in, r_out, n_data)
    z_data = np.zeros(n_data)
    data_points = np.column_stack([r_data / 1e5, z_data / 1e5])
    data_values = np.log10(Sigma[::max(1, n_r // n_data)][:n_data] + 1e-30)

    rbf_w = rbf_weights(data_points, data_values, r0=1.0, basis='multiquadric')


    n_query = 50
    rq = np.linspace(r_in, r_out, n_query) / 1e5
    zq = np.zeros(n_query)
    query_pts = np.column_stack([rq, zq])

    sigma_interp = rbf_interpolate(query_pts, data_points, rbf_w, r0=1.0, basis='multiquadric')
    print(f"  RBF 数据点数: {n_data}")
    print(f"  RBF 查询点数: {n_query}")
    print(f"  重构表面密度范围: [{10**np.min(sigma_interp):.3e}, {10**np.max(sigma_interp):.3e}] kg/m^2")




    print("\n" + "-" * 60)
    print("Step 9: RK3 流体动力学时间演化")
    print("-" * 60)


    def disk_evolution_rhs(t, Sigma_vec):
        Sigma_vec = np.asarray(Sigma_vec, dtype=np.float64)
        n = len(Sigma_vec)
        dSigma_dt = np.zeros(n, dtype=np.float64)


        nu = 1e10
        sqrt_r = np.sqrt(r_grid)
        sqr = Sigma_vec * sqrt_r


        if n > 2:
            d_sqr = np.zeros(n)
            d_sqr[1:-1] = (sqr[2:] - sqr[:-2]) / (2.0 * dr)
            d_sqr[0] = (sqr[1] - sqr[0]) / dr
            d_sqr[-1] = (sqr[-1] - sqr[-2]) / dr

            r_half = np.zeros(n)
            r_half[1:-1] = (r_grid[2:] ** 1.5 - r_grid[:-2] ** 1.5) / (2.0 * dr)
            r_half[0] = (r_grid[1] ** 1.5 - r_grid[0] ** 1.5) / dr
            r_half[-1] = (r_grid[-1] ** 1.5 - r_grid[-2] ** 1.5) / dr

            flux = r_half * d_sqr
            dSigma_dt[1:-1] = (3.0 / r_grid[1:-1]) * (flux[2:] - flux[:-2]) / (2.0 * dr)

        return dSigma_dt

    t_span = [0.0, 1e6]
    n_steps = 50
    Sigma0 = Sigma.copy()

    t_hist, Sigma_hist = rk3_integrate(disk_evolution_rhs, t_span, Sigma0, n_steps)
    Sigma_final = Sigma_hist[-1]

    print(f"  时间演化: {t_span[0]:.1e} -> {t_span[1]:.1e} s")
    print(f"  时间步数: {n_steps}")
    print(f"  初始 Sigma 总和: {np.sum(Sigma0):.3e}")
    print(f"  最终 Sigma 总和: {np.sum(Sigma_final):.3e}")
    print(f"  质量守恒误差: {abs(np.sum(Sigma_final) - np.sum(Sigma0)) / np.sum(Sigma0):.3e}")


    grad_Sigma = gradient_1d(Sigma_final, dr)
    print(f"  表面密度梯度范围: [{np.min(grad_Sigma):.3e}, {np.max(grad_Sigma):.3e}]")




    print("\n" + "-" * 60)
    print("Step 10: 喷流区域蒙特卡洛采样")
    print("-" * 60)


    def test_integrand(x):
        return x[0] ** 2 + x[1] * x[2]

    mc_est, mc_err = wedge_monte_carlo_integral(10000, test_integrand, seed=42)
    exact_val = wedge01_monomial_integral([2, 0, 0]) + wedge01_monomial_integral([0, 1, 1])
    print(f"  楔形体 MC 积分: {mc_est:.6f} +/- {mc_err:.6f}")
    print(f"  精确值: {exact_val:.6f}")
    print(f"  相对误差: {abs(mc_est - exact_val) / abs(exact_val):.3e}")


    n_jet_particles = 500
    r_launch = 5.0 * r_isco
    theta_open = np.pi / 12.0
    v_jet = 0.1 * C_LIGHT

    jet_pos, jet_vel = sample_jet_particles(
        n_jet_particles, r_launch, theta_open, v_jet, seed=42
    )
    print(f"  喷流粒子数: {n_jet_particles}")
    print(f"  发射半径: {r_launch/1e3:.1f} km")
    print(f"  半开角: {np.degrees(theta_open):.1f} 度")
    print(f"  喷射速度: {v_jet/1e3:.1f} km/s")


    energies, escaped = mc_jet_energy_transport(1000, r_out, np.max(T_disk), seed=42)
    escape_fraction = np.sum(escaped) / len(escaped)
    print(f"  光子逃逸比例: {escape_fraction:.3f}")
    print(f"  平均光子能量(相对): {np.mean(energies):.3f}")


    ball_pts = ball_unit_sample(200, dim=3, seed=42)
    ball_dists = ball_distance_stats(500, seed=42)
    print(f"  球内随机点对距离均值: {ball_dists['mean']:.4f}")
    print(f"  理论均值: {36.0/35.0:.4f}")


    r_bins = np.linspace(0, 2.0, 11)
    xi = compute_correlation_function(ball_pts, r_bins)
    print(f"  两点关联函数范围: [{np.min(xi):.3f}, {np.max(xi):.3f}]")




    print("\n" + "-" * 60)
    print("Step 11: 粒子动力学模拟 (速度 Verlet)")
    print("-" * 60)


    md_result = run_particle_simulation(
        n_particles=50, dim=3, n_steps=20, dt=0.01,
        box_size=1.0, potential='sinsq', seed=42
    )
    E0 = md_result['total_energy'][0]
    E_final = md_result['total_energy'][-1]
    dE = abs(E_final - E0) / abs(E0)
    print(f"  粒子数: 50")
    print(f"  步数: 20")
    print(f"  初始总能量: {E0:.6f}")
    print(f"  最终总能量: {E_final:.6f}")
    print(f"  相对能量变化: {dE:.3e}")


    dust_result = accretion_disk_particle_model(
        n_dust=30, r_in=r_in/1e5, r_out=r_out/1e5,
        z_scale=0.01*r_out/1e5, n_steps=10, dt=1e4, seed=42
    )
    print(f"  尘埃粒子数: 30")
    print(f"  最终尘埃位置范围: r in [{np.min(np.linalg.norm(dust_result['final_positions'][:, :2], axis=1)):.2e}, "
          f"{np.max(np.linalg.norm(dust_result['final_positions'][:, :2], axis=1)):.2e}] x10^5 m")




    print("\n" + "-" * 60)
    print("Step 12: Blandford-Payne 喷流判据与光谱")
    print("-" * 60)


    B_z = 1e5


    launched, v_A, v_esc = jet_launching_criterion(r_grid, B_z, None, M_bh)
    n_launched = np.sum(launched)
    print(f"  垂直磁场 B_z = {B_z:.1e} T")
    print(f"  Alfven 速度范围: [{np.min(v_A):.3e}, {np.max(v_A):.3e}] m/s")
    print(f"  逃逸速度范围: [{np.min(v_esc):.3e}, {np.max(v_esc):.3e}] m/s")
    print(f"  可发射喷流的位置数: {n_launched} / {n_r}")


    B_phi = 1e2
    B_r = 1e1
    T_mag = magnetic_braking_torque(r_grid, B_phi, B_r, Sigma, M_bh)
    print(f"  磁制动扭矩范围: [{np.min(T_mag):.3e}, {np.max(T_mag):.3e}] N*m/m")


    unstable, t_visc, t_cool = disk_instability_criterion(Sigma, T_disk, r_grid, M_bh, alpha)
    n_unstable = np.sum(unstable)
    print(f"  热不稳定区域数: {n_unstable} / {n_r}")


    nu_spec = np.logspace(14, 19, 50)
    L_nu = disk_spectrum_nu(nu_spec, r_in, r_out, M_dot, M_bh)
    peak_idx = np.argmax(L_nu)
    print(f"  光谱峰值频率: {nu_spec[peak_idx]:.3e} Hz")
    print(f"  对应波长: {C_LIGHT / nu_spec[peak_idx] * 1e9:.2f} nm")




    print("\n" + "-" * 60)
    print("Step 13: 幻方权重与数值鲁棒性验证")
    print("-" * 60)

    M4 = magic4_matrix(4)
    w4 = normalized_magic_weights(4)
    print(f"  4阶幻方矩阵和: {np.sum(M4)}")
    print(f"  归一化幻方权重和: {np.sum(w4):.6f}")


    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0, 0.0])
    safe_res = safe_divide(a, b, fill_value=999.0)
    print(f"  安全除法结果: {safe_res}")


    clipped = clip_with_warning(np.array([-1.0, 0.5, 2.0]), 0.0, 1.0)
    print(f"  裁剪结果: {clipped}")




    print("\n" + "=" * 80)
    print("  模拟完成 - 所有模块正常运行，无报错")
    print("=" * 80)
    print(f"\n  核心物理量汇总:")
    print(f"    - 盘质量: {total_mass / M_SUN:.3e} M_sun")
    print(f"    - 最高温度: {np.max(T_disk):.3e} K")
    print(f"    - 最小粘滞时标: {np.min(t_visc):.3e} s")
    print(f"    - 喷流发射位置: {n_launched} 个")
    print(f"    - 热不稳定区域: {n_unstable} 个")
    print(f"    - 光谱峰值: {C_LIGHT / nu_spec[peak_idx] * 1e9:.1f} nm")
    print(f"    - RK3 质量守恒误差: {abs(np.sum(Sigma_final) - np.sum(Sigma0)) / np.sum(Sigma0):.3e}")
    print(f"    - Verlet 能量相对变化: {dE:.3e}")
    print(f"    - CG 求解收敛: {cg_info['converged']}")
    print("=" * 80)

    return {
        'r_grid': r_grid,
        'Sigma': Sigma,
        'T_disk': T_disk,
        'phi_grav': phi_grav,
        'Sigma_final': Sigma_final,
        'jet_positions': jet_pos,
        'jet_velocities': jet_vel,
        'spectrum_nu': nu_spec,
        'spectrum_Lnu': L_nu,
        'cg_info': cg_info,
        'md_energy_error': dE
    }


if __name__ == '__main__':
    try:
        results = run_simulation()
        print("\n[成功] main.py 运行完毕，无报错。")
        sys.exit(0)
    except Exception as e:
        print(f"\n[错误] 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
