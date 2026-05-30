
import numpy as np
import time


from vorticity_transport import VorticityTransportSolver
from poisson_fem import PoissonFEM2D
from structure_dynamics import StructureDynamics
from jacobi_spectral import (
    gauss_jacobi_rule,
    boundary_layer_map,
    integrate_boundary_layer,
    test_jacobi_spectral,
)
from cvt_adaptation import CVT1DPeriodic, CVT2DReflect, wake_density_function
from stability_coupling import analyze_stability
from error_estimator import (
    compute_solution_quality_metrics,
    l1_norm_discrete,
    l2_norm_discrete,
    richardson_error_estimate,
)
from probability_model import (
    estimate_vortex_shedding_frequency,
    fit_beta_parameters,
    phase_pdf,
    phase_cdf,
    incomplete_beta,
)
from geometry_utils import (
    parse_stl_ascii,
    compute_face_normals,
    stla_check,
    sensor_placement_optimization,
    vortex_caustic_map,
    generate_simple_cylinder_stl_lines,
)


def main():
    print("=" * 72)
    print("  圆柱涡激振动流固耦合数值模拟系统")
    print("  Vortex-Induced Vibration Fluid-Structure Interaction Solver")
    print("=" * 72)
    t_start = time.time()




    U_inf = 1.0
    D_cyl = 0.2
    nu = 0.002
    Re = U_inf * D_cyl / nu

    Lx = 4.0
    Ly = 2.0
    nx = 81
    ny = 41
    dt = 0.004
    n_steps = 500

    mass = 1.0
    stiffness = 39.47841760435743
    damping = 2.0 * 0.01 * np.sqrt(mass * stiffness)
    mass_ratio = 2.0
    k_nl = 100.0

    print(f"\n[参数设定]")
    print(f"  雷诺数 Re = {Re:.1f}")
    print(f"  计算域  = {Lx:.1f} x {Ly:.1f} m")
    print(f"  网格    = {nx} x {ny}")
    print(f"  时间步长 dt = {dt:.4f} s, 总步数 = {n_steps}")
    print(f"  结构固有频率 f_n = {np.sqrt(stiffness/mass)/(2*np.pi):.3f} Hz")




    cylinder_params = {
        'cx': Lx * 0.25,
        'cy': Ly * 0.5,
        'r': D_cyl / 2.0,
    }

    fluid_solver = VorticityTransportSolver(
        nx=nx, ny=ny, lx=Lx, ly=Ly,
        nu=nu, u_inf=U_inf, dt=dt,
        cylinder_params=cylinder_params
    )

    poisson_solver = PoissonFEM2D(
        nx=nx, ny=ny, lx=Lx, ly=Ly,
        solid_mask=fluid_solver.solid_mask
    )




    structure = StructureDynamics(
        mass=mass,
        damping=damping,
        stiffness=stiffness,
        mass_ratio=mass_ratio,
        u_inf=U_inf,
        diameter=D_cyl,
        fn=1.0,
        nonlinear_params={'k_nl': k_nl},
        time_integrator='cn_rk2',
    )
    structure.set_time_step(dt)




    time_history = np.zeros(n_steps)
    cl_history = np.zeros(n_steps)
    cd_history = np.zeros(n_steps)
    disp_history = np.zeros((n_steps, 2))
    vel_history = np.zeros((n_steps, 2))



    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if not fluid_solver.solid_mask[j, i]:
                dx_ = fluid_solver.X[j, i] - cylinder_params['cx']
                dy_ = fluid_solver.Y[j, i] - cylinder_params['cy']
                dist = np.sqrt(dx_ ** 2 + dy_ ** 2)
                if dist > cylinder_params['r'] and dist < 2.0 * cylinder_params['r']:
                    sign = 1.0 if dy_ > 0 else -1.0
                    fluid_solver.omega[j, i] = (
                        sign * 3.0 * U_inf / D_cyl
                        * np.exp(-(dist - cylinder_params['r']) ** 2
                                  / (0.4 * cylinder_params['r']) ** 2)
                    )
    fluid_solver.apply_boundary_conditions()




    print("\n[时间推进开始]")
    for step in range(n_steps):
        t = step * dt
        time_history[step] = t


        fluid_solver.time_step(step)


        dirichlet_mask = np.zeros((ny, nx), dtype=bool)
        dirichlet_values = np.zeros((ny, nx))


        dirichlet_mask[:, 0] = True
        dirichlet_values[:, 0] = U_inf * fluid_solver.Y[:, 0]

        dirichlet_mask[0, :] = True
        dirichlet_values[0, :] = 0.0
        dirichlet_mask[ny - 1, :] = True
        dirichlet_values[ny - 1, :] = U_inf * Ly

        dirichlet_mask[fluid_solver.solid_mask] = True
        dirichlet_values[fluid_solver.solid_mask] = 0.0

        psi_new = poisson_solver.solve(
            rhs_field=fluid_solver.omega,
            dirichlet_mask=dirichlet_mask,
            dirichlet_values=dirichlet_values,
        )
        fluid_solver.psi = psi_new


        fluid_solver.apply_boundary_conditions()


        fluid_solver.compute_velocity_from_psi()


        c_d, c_l = fluid_solver.compute_force_coefficients()
        cl_history[step] = c_l
        cd_history[step] = c_d









        raise NotImplementedError("Hole 2: 升阻力到物理力的转换与结构步调用尚未实现")


        if (step + 1) % 100 == 0:
            print(f"  Step {step+1:4d}/{n_steps}, t={t:.3f}s, "
                  f"C_L={c_l:+.4f}, C_D={c_d:+.4f}, "
                  f"Y_disp={disp_history[step,1]:+.4e}")

    print("[时间推进完成]")




    print("\n[Jacobi 谱后处理]")
    n_gauss = 16
    alpha_j = 0.5
    beta_j = 0.0
    nodes_gj, weights_gj = gauss_jacobi_rule(n_gauss, alpha_j, beta_j)


    x_wake = cylinder_params['cx'] + 2.0 * D_cyl
    wake_profile, y_profile = fluid_solver.get_wake_profile(x_wake)
    valid = ~np.isnan(wake_profile)

    if np.sum(valid) > n_gauss:

        y_valid = y_profile[valid]
        w_valid = wake_profile[valid]

        y_min, y_max = np.min(y_valid), np.max(y_valid)
        y_mapped = 0.5 * (y_max - y_min) * (nodes_gj + 1.0) + y_min
        w_interp = np.interp(y_mapped, y_valid, w_valid, left=0.0, right=0.0)
        wake_enstrophy = np.sum(weights_gj * w_interp ** 2) * 0.5 * (y_max - y_min)
        print(f"  尾流区涡量拟能 (Jacobi-Gauss {n_gauss}点) = {wake_enstrophy:.6e}")
    else:
        wake_enstrophy = 0.0
        print(f"  尾流区有效数据点不足，跳过谱积分。")


    test_jacobi_spectral()




    print("\n[CVT 自适应网格评估]")

    cvt1d = CVT1DPeriodic(num_generators=20, domain_length=Ly, it_max=30)
    gen_1d = cvt1d.iterate()
    print(f"  1D 周期 CVT 收敛步数 = {len(cvt1d.energy_history)}, "
          f"最终能量 = {cvt1d.energy_history[-1]:.6e}")


    density_func = lambda x, y: wake_density_function(
        np.array([[x]]), np.array([[y]]),
        cylinder_params['cx'], cylinder_params['cy'], cylinder_params['r'],
        base_density=1.0, peak_density=8.0
    )[0, 0]
    cvt2d = CVT2DReflect(
        num_generators=50,
        bounds=((cylinder_params['cx'], Lx), (0.0, Ly)),
        density_func=density_func,
        it_max=20,
        sample_num=5000,
    )
    gen_2d = cvt2d.iterate()
    print(f"  2D 反射 CVT 收敛步数 = {len(cvt2d.energy_history)}, "
          f"最终能量 = {cvt2d.energy_history[-1]:.6e}")




    print("\n[耦合稳定性分析]")
    stab_report = analyze_stability(
        nx=20, nu=nu, mass=mass, damping=damping,
        stiffness=stiffness, rho_f=rho_f, D_cyl=D_cyl,
    )
    print(f"  对数范数 \mu_1(J)  = {stab_report['mu_1']:+.4f}")
    print(f"  对数范数 \mu_2(J)  = {stab_report['mu_2']:+.4f}")
    print(f"  对数范数 \mu_\infty(J) = {stab_report['mu_inf']:+.4f}")
    print(f"  谱横坐标           = {stab_report['spectral_abscissa']:+.4f}")
    print(f"  Gershgorin 界      = [{stab_report['gershgorin_min']:+.4f}, "
          f"{stab_report['gershgorin_max']:+.4f}]")
    print(f"  谱稳定判据         = {'稳定' if stab_report['stable_spectral'] else '可能不稳定'}")




    print("\n[误差估计与解质量]")
    metrics = compute_solution_quality_metrics(
        fluid_solver.omega, fluid_solver.psi,
        fluid_solver.u, fluid_solver.v,
        fluid_solver.dx, fluid_solver.dy,
        solid_mask=fluid_solver.solid_mask,
    )
    print(f"  涡量 L2 范数       = {metrics['omega_l2']:.6e}")
    print(f"  流函数 L2 范数     = {metrics['psi_l2']:.6e}")
    print(f"  动能估计           = {metrics['kinetic_energy']:.6e}")
    print(f"  涡量拟能           = {metrics['enstrophy']:.6e}")
    print(f"  散度 RMS           = {metrics['divergence_rms']:.6e}")



    omega_coarse = fluid_solver.omega[::2, ::2].copy()

    omega_coarse_fine = np.zeros_like(fluid_solver.omega)
    for j in range(omega_coarse.shape[0]):
        for i in range(omega_coarse.shape[1]):
            omega_coarse_fine[j * 2, i * 2] = omega_coarse[j, i]

    from scipy.ndimage import zoom
    try:
        omega_coarse_fine = zoom(omega_coarse, (fluid_solver.ny / omega_coarse.shape[0],
                                                  fluid_solver.nx / omega_coarse.shape[1]),
                                 order=1)
    except Exception:
        omega_coarse_fine = fluid_solver.omega.copy()

    rich_err = richardson_error_estimate(
        fluid_solver.omega, omega_coarse_fine, ratio=2.0, order=2.0
    )
    err_l1 = l1_norm_discrete(rich_err, fluid_solver.dx, fluid_solver.dy)
    err_l2 = l2_norm_discrete(rich_err, fluid_solver.dx, fluid_solver.dy)
    print(f"  Richardson L1 误差 = {err_l1:.6e}")
    print(f"  Richardson L2 误差 = {err_l2:.6e}")




    print("\n[涡脱落统计概率建模]")

    start_idx = n_steps // 2
    cl_steady = cl_history[start_idx:]

    f_est, st_est = estimate_vortex_shedding_frequency(cl_steady, dt)
    print(f"  估计涡脱落频率     = {f_est:.4f} Hz")
    print(f"  Strouhal 数        = {st_est:.4f} (归一化)")


    analytic_signal = np.fft.ifft(
        np.fft.fft(cl_steady) * (np.arange(len(cl_steady)) >= 0) * 2.0
    )
    phase = np.angle(analytic_signal)
    phase = np.mod(phase, 2.0 * np.pi)

    p_beta, q_beta = fit_beta_parameters(phase)
    print(f"  Beta 分布参数      = p={p_beta:.3f}, q={q_beta:.3f}")


    theta_test = np.pi
    cdf_val = phase_cdf(theta_test, p_beta, q_beta)
    pdf_val = phase_pdf(theta_test, p_beta, q_beta)
    print(f"  相位 CDF(\pi)       = {cdf_val:.4f}")
    print(f"  相位 PDF(\pi)       = {pdf_val:.4f}")


    ib_test = incomplete_beta(0.5, 2.0, 3.0)
    print(f"  不完全 Beta 自检   = I_0.5(2,3) = {ib_test:.6f} (参考 ~0.6875)")




    print("\n[几何处理与传感器优化]")

    stl_lines = generate_simple_cylinder_stl_lines(diameter=D_cyl, num_facets=24)
    verts, norms = parse_stl_ascii(stl_lines)
    if verts.size > 0:
        check_code = stla_check(verts, norms)
        print(f"  STL 解析面片数     = {len(verts)}")
        print(f"  STL 一致性检查     = {'通过' if check_code == 0 else f'错误码 {check_code}'}")
        recomputed_norms = compute_face_normals(verts)
        norm_diff = np.max(np.linalg.norm(recomputed_norms - norms, axis=1))
        print(f"  法向量重构最大偏差 = {norm_diff:.3e}")


    candidates = [(j, i) for j in range(1, ny - 1) for i in range(1, nx - 1)
                  if not fluid_solver.solid_mask[j, i]]
    selected, achieved = sensor_placement_optimization(
        candidates, fluid_solver.omega, budget_num=15, influence_radius=2.0 * D_cyl
    )
    print(f"  候选测点数         = {len(candidates)}")
    print(f"  选中测点数         = {np.sum(selected)}")
    print(f"  总信息权重         = {achieved:.4f}")


    caustic_conn, wnum = vortex_caustic_map(
        n_points=100, m_ratio=7,
        cylinder_center=(cylinder_params['cx'], cylinder_params['cy']),
        radius_scale=D_cyl * 3.0,
    )
    print(f"  Caustic 拓扑环绕数 = {wnum}")




    t_elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print("  模拟完成")
    print(f"  总耗时 = {t_elapsed:.3f} s")
    print(f"  最终横向振幅 = {np.max(np.abs(disp_history[:,1])):.4e} m")
    print(f"  最终流向振幅 = {np.max(np.abs(disp_history[:,0])):.4e} m")
    print(f"  升力系数 RMS = {np.sqrt(np.mean(cl_steady**2)):.4f}")
    print("=" * 72)

    return {
        'time_history': time_history,
        'cl_history': cl_history,
        'cd_history': cd_history,
        'disp_history': disp_history,
        'vel_history': vel_history,
        'stab_report': stab_report,
        'metrics': metrics,
    }


if __name__ == "__main__":
    main()
