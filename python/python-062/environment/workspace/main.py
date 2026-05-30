
import numpy as np
import sys
import time


def print_banner():
    print("=" * 78)
    print("  边界层湍流与大涡模拟 (PBL-LES) 合成科研代码项目")
    print("  领域：大气科学 —— 边界层湍流与大涡模拟")
    print("=" * 78)
    print()


def run_simulation():
    print_banner()
    t0_total = time.time()




    print("[1/8] 初始化物理参数与计算网格 ...")
    nx, ny, nz = 24, 24, 16
    Lx, Ly, Lz = 1000.0, 1000.0, 500.0
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    dt = 0.2
    n_steps = 5
    rho = 1.225
    nu_mol = 1.5e-5
    g = 9.81

    print(f"      网格: {nx} x {ny} x {nz}, 分辨率: {dx:.1f}m x {dy:.1f}m x {dz:.1f}m")
    print(f"      时间步长: {dt}s, 总步数: {n_steps}")




    print("[2/8] 初始化湍流场（对数风剖面 + 随机脉动）...")
    from les_core import initialize_turbulent_field
    u, v, w, theta = initialize_turbulent_field(
        nx, ny, nz, dx, dy, dz,
        u_mean=8.0, v_mean=0.0,
        turbulence_intensity=0.02,
        theta_mean=298.0, theta_gradient=0.005
    )




    print("[3/8] 构建网格拓扑与识别边界 ...")
    from mesh_topology import build_mesh_graph, mesh_quality_metrics
    from mesh_boundary import extract_boundary_nodes_3d, apply_surface_layer_bc


    nodes = np.zeros((nx * ny * nz, 3), dtype=np.float64)
    idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes[idx] = [i * dx, j * dy, k * dz]
                idx += 1



    element_nodes = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                n000 = i + nx * (j + ny * k)
                n100 = (i + 1) + nx * (j + ny * k)
                n010 = i + nx * ((j + 1) + ny * k)
                n110 = (i + 1) + nx * ((j + 1) + ny * k)
                n001 = i + nx * (j + ny * (k + 1))
                n101 = (i + 1) + nx * (j + ny * (k + 1))
                n011 = i + nx * ((j + 1) + ny * (k + 1))
                n111 = (i + 1) + nx * ((j + 1) + ny * (k + 1))

                element_nodes.append([n000, n100, n010, n001])
                element_nodes.append([n111, n011, n101, n110])
                element_nodes.append([n100, n010, n110, n001])
                element_nodes.append([n100, n110, n101, n001])
                element_nodes.append([n110, n101, n001, n011])

    element_nodes = np.array(element_nodes, dtype=int)
    n_elem = element_nodes.shape[0]
    print(f"      四面体单元数: {n_elem}")

    adjacency, degrees = build_mesh_graph(element_nodes, nx * ny * nz)
    print(f"      平均节点度: {np.mean(degrees):.1f}")

    bdry_info = extract_boundary_nodes_3d(element_nodes, nodes, lower_z=0.0, upper_z=Lz)
    lower_nodes = bdry_info['lower']
    print(f"      下边界节点数: {len(lower_nodes)}")


    u, v, w = apply_surface_layer_bc(
        u.flatten(), v.flatten(), w.flatten(), theta.flatten(),
        nodes, lower_nodes, u_star=0.4, z0=0.1
    )
    u = u.reshape((nx, ny, nz))
    v = v.reshape((nx, ny, nz))
    w = w.reshape((nx, ny, nz))




    print("[4/8] LES 时间步进（对流-扩散-SGS-投影）...")
    from les_core import convection_term, laplacian_3d, projection_step
    from sgs_model import smagorinsky_model, dynamic_smagorinsky_model
    from time_integrator import adaptive_timestep

    for step in range(n_steps):

        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(np.isfinite(w))):
            print(f"      [警告] 步 {step} 检测到非有限值，重新初始化速度场")
            u, v, w, theta = initialize_turbulent_field(
                nx, ny, nz, dx, dy, dz,
                u_mean=8.0, v_mean=0.0,
                turbulence_intensity=0.05,
                theta_mean=298.0, theta_gradient=0.005
            )


        dt = adaptive_timestep(u, v, w, dx, dy, dz, nu_mol + 0.1, cfl=0.3)
        if dt < 1e-6:
            dt = 1e-6










        raise NotImplementedError("HOLE 2: 请实现 LES 时间步进核心循环")



        u, v, w = apply_surface_layer_bc(
            u.flatten(), v.flatten(), w.flatten(), theta.flatten(),
            nodes, lower_nodes, u_star=0.4, z0=0.1
        )
        u = u.reshape((nx, ny, nz))
        v = v.reshape((nx, ny, nz))
        w = w.reshape((nx, ny, nz))


        u = np.clip(u, -50.0, 50.0)
        v = np.clip(v, -50.0, 50.0)
        w = np.clip(w, -20.0, 20.0)

        if step % 1 == 0:
            print(f"      步 {step:3d}/{n_steps}, dt={dt:.4f}s, nu_sgs_mean={np.mean(nu_sgs):.4f}, "
                  f"投影收敛={converged}, u_max={np.max(np.abs(u)):.2f}")

    print(f"      时间步进完成，最终 CFL 约束 dt={dt:.4f}s")




    print("[5/8] 计算湍流统计量 ...")
    from turbulence_stats import (
        compute_tke, compute_reynolds_stresses, compute_heat_flux,
        longitudinal_structure_function, compute_kolmogorov_scales
    )
    from quadrature_rules import compute_energy_dissipation_rate

    tke, tke_mean = compute_tke(u, v, w)
    R = compute_reynolds_stresses(u, v, w)
    qx, qy, qz = compute_heat_flux(u, v, w, theta)
    epsilon = compute_energy_dissipation_rate(u, v, w, dx, dy, dz, nu_mol)

    eta, tau_eta, u_eta, Re_lambda = compute_kolmogorov_scales(epsilon, nu_mol)

    r, D_ll = longitudinal_structure_function(u, axis=0, max_lag=nx // 4)

    print(f"      平均湍动能 TKE = {tke_mean:.4f} m²/s²")
    print(f"      Reynolds 应力 u'u' = {R['uu']:.4f}, w'w' = {R['ww']:.4f}")
    print(f"      热通量 w'θ' = {qz:.4f} K·m/s")
    print(f"      能量耗散率 ε = {epsilon:.6e} m²/s³")
    print(f"      Kolmogorov 尺度: η={eta:.4e}m, τ_η={tau_eta:.4e}s, u_η={u_eta:.4e}m/s")
    print(f"      Taylor Reynolds 数 Re_λ = {Re_lambda:.1f}")




    print("[6/8] 拉格朗日粒子追踪与随机 Langevin 扩散 ...")
    from stochastic_model import initialize_particles, langevin_step_euler, ensemble_concentration
    from particle_tracker import track_particles_rk2
    from lagrange_interp import interpolate_velocity_to_particles

    n_particles = 200
    particles, velocities = initialize_particles(
        n_particles,
        domain_x=(0, Lx), domain_y=(0, Ly), domain_z=(0, Lz),
        release_height=20.0
    )


    x_grid = np.arange(nx) * dx
    y_grid = np.arange(ny) * dy
    z_grid = np.arange(nz) * dz

    u_p, v_p, w_p = interpolate_velocity_to_particles(
        particles, (x_grid, y_grid, z_grid), u, v, w, order=3
    )


    sigma_w_field = np.sqrt(np.maximum(R['ww'], 1e-6))
    sigma_w_p = np.full(n_particles, sigma_w_field)
    epsilon_p = np.full(n_particles, epsilon)

    particles_new, velocities_new = langevin_step_euler(
        particles, velocities, sigma_w_p, epsilon_p, dt=1.0
    )

    print(f"      粒子数: {n_particles}")
    print(f"      平均垂直位移: {np.mean(particles_new[:, 2] - particles[:, 2]):.3f} m")




    print("[7/8] 分形维数与间歇性分析 ...")
    from fractal_analysis import box_counting, compute_intermittency_factor, richardson_cascade_spectrum


    u_slice = u[:, :, nz // 2]
    D_f, scales, counts = box_counting(u_slice, threshold=np.mean(u_slice), max_level=4)
    mu = compute_intermittency_factor(u_slice, window_size=4)


    k_spec = np.linspace(1, nx // 2, nx // 2)
    E_k = richardson_cascade_spectrum(k_spec, epsilon, C=1.5, mu=mu)

    print(f"      速度场盒计数维数 D_f = {D_f:.3f} (理论值 ~2.0–2.8)")
    print(f"      间歇性因子 μ = {mu:.3f}")
    print(f"      能谱峰值波数 k_max = {k_spec[np.argmax(E_k)]:.1f}")




    print("[8/8] 波数空间三波相互作用枚举 ...")
    from wave_interactions import enumerate_triads, shell_energy_flux

    triads = enumerate_triads(k_max=3, dim=2)
    print(f"      截断球内三波组数: {len(triads)}")


    np.random.seed(7)
    velocities_spec = {}
    for k1 in range(-3, 4):
        for k2 in range(-3, 4):
            if k1 == 0 and k2 == 0:
                continue
            k_mag = np.sqrt(k1**2 + k2**2)
            if k_mag > 3:
                continue

            amp = (epsilon ** (1.0 / 3.0)) * (k_mag ** (-1.0 / 3.0))
            phase = np.random.rand(2) * 2 * np.pi
            velocities_spec[(k1, k2)] = amp * np.array([
                np.exp(1j * phase[0]),
                np.exp(1j * phase[1])
            ])

    k_bins = np.array([0.5, 1.5, 2.5, 3.5])
    Pi = shell_energy_flux(triads, velocities_spec, k_bins)
    print(f"      能量通量 Π(k): {Pi}")
    print(f"      理论惯性子区通量应近似为常数 ε = {epsilon:.4e}")




    print("[附加] 球谐函数基展开演示 ...")
    from spherical_harmonics import spherical_harmonic_basis

    theta_ang = np.linspace(0, np.pi, 12)
    phi_ang = np.linspace(0, 2 * np.pi, 24)
    TH, PH = np.meshgrid(theta_ang, phi_ang, indexing='ij')

    Y_2_1_c, Y_2_1_s = spherical_harmonic_basis(2, 1, TH, PH)
    print(f"      Y_2^1 实部积分: {np.mean(Y_2_1_c):.6f} (应 ≈ 0)")




    print("[附加] Gauss-Patterson 高精度求积演示 ...")
    from quadrature_rules import patterson_integrate_1d, patterson_integrate_3d


    result_1d = patterson_integrate_1d(lambda x: x**2, 0.0, 1.0, order=7)
    print(f"      ∫_0^1 x² dx = {result_1d:.10f} (精确值 1/3 = 0.3333333333)")


    result_3d = patterson_integrate_3d(
        lambda x, y, z: x * y * z,
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), order=7
    )
    print(f"      ∫_0^1 x*y*z dV = {result_3d:.10f} (精确值 1/8 = 0.1250000000)")




    print("[附加] 有限元基函数与 Laplacian 矩阵组装演示 ...")
    from fem_basis import basis_mn_tet4, basis_gradient_tet4, fem_laplacian_matrix

    test_tet = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    test_points = np.array([[0.1, 0.1, 0.1], [0.25, 0.25, 0.25]], dtype=np.float64)
    phi_test = basis_mn_tet4(test_tet, test_points)
    grad_test = basis_gradient_tet4(test_tet)
    print(f"      测试点基函数值: {phi_test[:, 0]}")
    print(f"      基函数梯度范数: {[np.linalg.norm(g) for g in grad_test]}")




    t_total = time.time() - t0_total
    print()
    print("=" * 78)
    print(f"  模拟完成，总耗时: {t_total:.2f} 秒")
    print("=" * 78)


if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"\n[ERROR] 运行过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
