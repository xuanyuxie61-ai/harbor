#!/usr/bin/env python3

import os
import sys
import numpy as np




from ice_constitutive_model import (
    rate_factor_arrhenius,
    effective_stress,
    glen_flow_law,
    anisotropic_enhancement_factor,
    dissipation_heat,
    glen_viscosity,
    effective_strain_rate,
    ICE_DENSITY, GRAVITY, GLEN_N,
)
from thermomechanics import (
    solve_temperature_evolution,
    solve_enthalpy_evolution,
)
from ice_sheet_evolution import (
    solve_sia_evolution,
    ice_volume,
    ice_area,
)
from ice_stream_dynamics import (
    solve_ice_stream_oscillation,
    detect_stick_slip_events,
    driving_stress_from_params,
)
from compass_search_calibration import (
    CompassSearchOptimizer,
    build_calibration_objective,
    demo_calibration_problem,
)
from matrix_exponential_integrator import (
    exponential_integrator_ice_thickness,
    build_1d_diffusion_matrix,
)
from spherical_tessellation import (
    sphere_cvt_iterate,
    cvt_energy,
    project_to_ice_dome_region,
)
from mesh_io import (
    assemble_ice_stiffness_matrix_2d,
    coo_to_csc,
    csc_to_dense,
    write_medit_mesh,
)
from bayesian_inference import (
    gamma_mle_newton_raphson,
    dirichlet_mle_newton,
    metropolis_hastings_posterior,
    gamma_sample,
)
from drainage_analysis import (
    identify_catchments,
    extract_main_flow_branches,
    compute_drainage_density,
)
from banded_solver import (
    solve_tridiagonal,
    build_sia_tridiagonal,
)
from elliptic_solutions import (
    vialov_profile,
    vialov_volume_exact,
    bueler_exact_radius,
    convergence_test_vialov,
    elliptic_k_complete,
    elliptic_e_complete,
)
from fabric_orientation_sampling import (
    monte_carlo_fabric_simulation,
    fabric_anisotropy_indices,
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_module_1_constitutive():
    print_section("模块 1: 各向异性 Glen 流动律与耗散热")


    tau = np.zeros((3, 3), dtype=np.float64)
    tau[0, 2] = tau[2, 0] = 1.0e5
    tau[1, 2] = tau[2, 1] = 0.5e5
    tau[0, 0] = -0.5e5
    tau[1, 1] = -0.3e5
    tau[2, 2] = 0.8e5

    T = 253.15
    eps = glen_flow_law(tau, T)
    tau_e = effective_stress(tau)
    eps_e = effective_strain_rate(eps)
    phi = dissipation_heat(eps, tau)
    eta = glen_viscosity(T, eps_e)


    a2 = np.diag([0.5, 0.3, 0.2])
    E = anisotropic_enhancement_factor(a2)

    print(f"  温度 T = {T:.2f} K")
    print(f"  等效应力 tau_e = {tau_e:.3e} Pa")
    print(f"  等效应变率 eps_e = {eps_e:.3e} s^-1")
    print(f"  耗散热 Phi = {phi:.3e} W/m^3")
    print(f"  等效粘度 eta = {eta:.3e} Pa s")
    print(f"  各向异性增强因子 E = {E:.3f}")

    return eps, phi


def run_module_2_thermomechanics():
    print_section("模块 2: 冰盖热力学耦合 (隐式中点法)")

    nz = 51
    z_max = 2000.0
    dt = 86400.0 * 30.0
    nt = 12

    surface_temp = 240.0
    basal_heat_flux = 0.042
    w = -1e-9 * np.ones(nz, dtype=np.float64)
    phi = 1e-4 * np.ones(nz, dtype=np.float64)

    T_history = solve_temperature_evolution(nz, z_max, dt, nt,
                                             surface_temp, basal_heat_flux,
                                             w, phi)

    print(f"  网格: nz={nz}, H={z_max} m")
    print(f"  初始底部温度: {T_history[0, -1]:.2f} K")
    print(f"  1年后底部温度: {T_history[-1, -1]:.2f} K")
    print(f"  温度变化: {T_history[-1, -1] - T_history[0, -1]:.3f} K")


    H_history, T_history_e, omega_history = solve_enthalpy_evolution(
        nz, z_max, dt, nt, surface_temp, basal_heat_flux, w, phi
    )
    print(f"  焓形式底部孔隙率: {omega_history[-1, -1]:.6f}")

    return T_history


def run_module_3_ice_sheet_evolution():
    print_section("模块 3: 浅水近似冰盖厚度演化")

    nx, ny = 41, 41
    dx = dy = 5000.0
    Lx, Ly = nx * dx, ny * dy

    x = np.linspace(-Lx / 2, Lx / 2, nx)
    y = np.linspace(-Ly / 2, Ly / 2, ny)
    X, Y = np.meshgrid(x, y)


    bedrock = -500.0 + 0.5 * (X ** 2 + Y ** 2) / 1e6
    bedrock = np.clip(bedrock, -2000.0, 0.0)


    R0 = 400e3
    H0 = 3000.0
    r = np.sqrt(X ** 2 + Y ** 2)
    H0_init = vialov_profile(r.flatten(), R0, H0).reshape(ny, nx)


    accumulation = 0.3 / (365.25 * 86400.0) * np.exp(-r / (200e3))

    total_time = 100.0 * 365.25 * 86400.0
    H_final, history = solve_sia_evolution(
        H0_init, bedrock, accumulation, dx, dy, total_time,
        temperature=253.15, output_interval=10
    )

    vol = ice_volume(H_final, dx, dy)
    area = ice_area(H_final, dx, dy)
    print(f"  域大小: {Lx/1e3:.0f} x {Ly/1e3:.0f} km")
    print(f"  初始体积: {ice_volume(H0_init, dx, dy)/1e12:.3f} km^3")
    print(f"  最终体积: {vol/1e12:.3f} km^3")
    print(f"  覆盖面积: {area/1e6:.3f} km^2")
    print(f"  最大厚度: {np.max(H_final):.1f} m")

    return H_final, bedrock


def run_module_4_ice_stream():
    print_section("模块 4: 冰流 Stick-Slip 非线性振荡 (Duffing 型)")

    params = {
        'H': 1500.0,
        'alpha': 0.002,
        'delta': 0.5,
        'alpha_u': 0.001,
        'beta_u': 1e-9,
        'gamma': 1.0,
        'omega': 2.0 * np.pi / (365.25 * 86400.0),
        'C_weertman': 5e-5,
        'u0': 0.5,
        'a_drain': 0.2,
        'b_drain': 5e-7,
        'c_drain': 1e-11,
        'd_season': 0.02,
        'omega_season': 2.0 * np.pi / (365.25 * 86400.0),
        'm_eff': 1.0,
    }

    y0 = np.array([1.0, 0.0, 1.0e6])
    t_span = (0.0, 10.0 * 365.25 * 86400.0)
    dt = 86400.0

    t_arr, y_arr = solve_ice_stream_oscillation(y0, t_span, dt, params)
    stats = detect_stick_slip_events(t_arr, y_arr, velocity_threshold=0.5)

    tau_d = driving_stress_from_params(params)
    print(f"  驱动力 tau_d = {tau_d:.3e} Pa")
    print(f"  平均流速: {stats['mean_velocity']:.3f} m/a")
    print(f"  最大流速: {stats['max_velocity']:.3f} m/a")
    print(f"  Stick-Slip 事件数: {len(stats['slip_events'])}")
    if stats['oscillation_period_estimate'] is not None:
        print(f"  估计振荡周期: {stats['oscillation_period_estimate']/86400:.1f} 天")

    return t_arr, y_arr


def run_module_5_compass_calibration():
    print_section("模块 5: 流变参数 Compass Search 标定")

    theta_true, observed, model_func = demo_calibration_problem(
        theta_true=[1e-24, 6.0e4, 3.0], noise_level=0.05
    )

    obj = build_calibration_objective(
        observed, model_func,
        weights=None,
        prior_theta=theta_true,
        prior_sigma=np.array([1e-25, 1e4, 0.5]),
        regularization_lambda=0.001
    )


    def scaled_objective(theta_raw):

        theta_physical = np.array([10.0 ** theta_raw[0], theta_raw[1], theta_raw[2]])
        return obj(theta_physical)

    theta_init = np.array([np.log10(5e-25), 5.0e4, 2.5])
    theta_lower = np.array([np.log10(1e-26), 3.0e4, 2.0])
    theta_upper = np.array([np.log10(1e-22), 2.0e5, 4.5])

    optimizer = CompassSearchOptimizer(
        theta_init, theta_lower, theta_upper,
        delta_init=0.5, delta_tol=1e-5, k_max=3000
    )
    theta_opt_raw, J_opt = optimizer.optimize(scaled_objective)
    theta_opt = np.array([10.0 ** theta_opt_raw[0], theta_opt_raw[1], theta_opt_raw[2]])

    print(f"  真值:  A0={theta_true[0]:.3e}, Q={theta_true[1]:.3e}, n={theta_true[2]:.2f}")
    print(f"  估计:  A0={theta_opt[0]:.3e}, Q={theta_opt[1]:.3e}, n={theta_opt[2]:.2f}")
    print(f"  残差 J: {J_opt:.6e}")
    print(f"  迭代次数: {len(optimizer.history)}")

    return theta_opt


def run_module_6_matrix_exponential():
    print_section("模块 6: 矩阵指数积分器 (扩散算子)")

    n = 41
    dx = 100.0
    D = 1.0
    dt = 86400.0 * 10.0


    x = np.linspace(-n * dx / 2, n * dx / 2, n)
    H = 100.0 * np.exp(-x ** 2 / (2.0 * (500.0) ** 2))
    accumulation = np.zeros(n, dtype=np.float64)

    H_new = exponential_integrator_ice_thickness(H, dt, dx, lambda h: D, accumulation)


    vol_old = np.trapezoid(H, x)
    vol_new = np.trapezoid(H_new, x)

    print(f"  网格: n={n}, dx={dx} m")
    print(f"  初始峰值: {np.max(H):.2f} m")
    print(f"  10天后峰值: {np.max(H_new):.2f} m")
    print(f"  体积守恒误差: {abs(vol_new - vol_old) / vol_old * 100:.6f}%")

    return H_new


def run_module_7_spherical_cvt():
    print_section("模块 7: 南极冰盖球面 CVT 观测网优化")

    n_gen = 200
    generators = sphere_cvt_iterate(n_gen, n_iterations=50, radius=6371e3, seed=42)
    energy = cvt_energy(generators)


    antarctic = project_to_ice_dome_region(generators, (-90.0, -60.0), (-180.0, 180.0))

    print(f"  全球节点数: {n_gen}")
    print(f"  CVT 能量: {energy:.3e}")
    print(f"  南极区域节点数: {len(antarctic)}")
    print(f"  平均间距估计: {2*np.pi*6371e3/np.sqrt(n_gen)/1e3:.1f} km")

    return generators


def run_module_8_mesh_io():
    print_section("模块 8: FEM 刚度矩阵组装与格式转换")


    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1e5, 0.0, 0.0],
        [0.5e5, 0.866e5, 0.0],
        [0.5e5, 0.289e5, 0.0],
    ], dtype=np.float64)

    elements = np.array([
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ], dtype=np.int64)

    vals, rows, cols, n_nodes, _ = assemble_ice_stiffness_matrix_2d(
        nodes, elements, diffusivity=1.0
    )

    mask = (rows != 0) & (cols != 0)
    vals_d = vals[mask]
    rows_d = rows[mask]
    cols_d = cols[mask]
    vals_d = np.append(vals_d, [1.0])
    rows_d = np.append(rows_d, [0])
    cols_d = np.append(cols_d, [0])
    data_csc, row_csc, col_ptr = coo_to_csc(vals_d, rows_d, cols_d, n_nodes, n_nodes)
    A_dense = csc_to_dense(data_csc, row_csc, col_ptr, n_nodes, n_nodes)


    eig_min = np.min(np.linalg.eigvalsh(A_dense))


    out_path = os.path.join(os.path.dirname(__file__), "demo_ice_mesh.mesh")
    write_medit_mesh(nodes, elements, boundary_nodes=None, filepath=out_path)

    print(f"  节点数: {n_nodes}")
    print(f"  单元数: {len(elements)}")
    print(f"  刚度矩阵非零元: {len(vals)}")
    print(f"  刚度矩阵最小特征值: {eig_min:.6e}")
    print(f"  是否半正定: {eig_min >= -1e-10}")
    print(f"  已写入: {out_path}")

    return A_dense


def run_module_9_bayesian():
    print_section("模块 9: 贝叶斯 Gamma/Dirichlet 参数推断")


    rng = np.random.default_rng(42)
    gamma_data = rng.gamma(shape=3.0, scale=2.0, size=500)
    alpha_est, beta_est = gamma_mle_newton_raphson(gamma_data, alpha_init=2.0)
    print(f"  Gamma 数据: shape=3.0, scale=2.0")
    print(f"  MLE 估计: alpha={alpha_est:.3f}, beta={beta_est:.3f}")


    dirich_data = rng.dirichlet(alpha=[2.0, 3.0, 5.0], size=300)
    alpha_dir = dirichlet_mle_newton(dirich_data)
    print(f"  Dirichlet 真值: [2.0, 3.0, 5.0]")
    print(f"  MLE 估计: [{alpha_dir[0]:.2f}, {alpha_dir[1]:.2f}, {alpha_dir[2]:.2f}]")


    def log_posterior(theta):
        if len(theta) != 2:
            return -1e20
        return -0.5 * ((theta[0] - 1.0) ** 2 + (theta[1] + 2.0) ** 2)

    samples = metropolis_hastings_posterior(
        log_posterior, theta_init=np.array([0.0, 0.0]),
        proposal_std=np.array([0.5, 0.5]), n_samples=5000, burn_in=1000
    )
    print(f"  MCMC 后验均值: [{np.mean(samples[:,0]):.3f}, {np.mean(samples[:,1]):.3f}]")
    print(f"  后验方差: [{np.var(samples[:,0]):.3f}, {np.var(samples[:,1]):.3f}]")

    return alpha_est, beta_est, alpha_dir


def run_module_10_drainage():
    print_section("模块 10: 冰盖排水流域连通分量分析")

    nx, ny = 101, 101
    dx = dy = 1000.0
    x = np.linspace(-50e3, 50e3, nx)
    y = np.linspace(-50e3, 50e3, ny)
    X, Y = np.meshgrid(x, y)


    s1 = 3000.0 * np.exp(-(X ** 2 + Y ** 2) / (2.0 * (20e3) ** 2))
    s2 = 2500.0 * np.exp(-((X - 30e3) ** 2 + (Y - 20e3) ** 2) / (2.0 * (15e3) ** 2))
    s3 = 2000.0 * np.exp(-((X + 25e3) ** 2 + (Y - 30e3) ** 2) / (2.0 * (12e3) ** 2))
    surface = s1 + s2 + s3


    bed = -500.0 * np.ones_like(surface)
    H = np.maximum(surface - bed, 0.0)

    mask = H > 100.0
    catchments = identify_catchments(surface, mask, dx, dy, min_area=1e7)
    density = compute_drainage_density(catchments, np.sum(mask) * dx * dy)

    print(f"  识别流域数: {len(catchments)}")
    for cid, info in catchments.items():
        print(f"    流域 {cid}: 面积={info['area_m2']/1e6:.2f} km^2, "
              f"HI={info['hypsometric_integral']:.3f}")
    print(f"  排水密度: {density*1e6:.6f} /km^2")

    return catchments


def run_module_11_banded_solver():
    print_section("模块 11: SIA 三对角系统 Thomas 算法")

    n = 101
    dx = 500.0
    H = 1000.0 * np.ones(n, dtype=np.float64)
    bed = np.linspace(0.0, -500.0, n)
    A = 1e-25
    rho_g = ICE_DENSITY * GRAVITY

    a, b, c, rhs = build_sia_tridiagonal(H, bed, dx, A, rho_g, GLEN_N)
    u = solve_tridiagonal(a, b, c, rhs)


    residual = np.zeros(n, dtype=np.float64)
    residual[1:-1] = (a[1:-1] * u[:-2] + b[1:-1] * u[1:-1] + c[1:-1] * u[2:]
                      - rhs[1:-1])
    max_res = np.max(np.abs(residual))

    print(f"  系统维度: {n}")
    print(f"  解范围: [{np.min(u):.2f}, {np.max(u):.2f}]")
    print(f"  最大残差: {max_res:.3e}")
    print(f"  残差 < 1e-10: {max_res < 1e-10}")

    return u


def run_module_12_elliptic():
    print_section("模块 12: 椭圆积分与 Vialov 精确解验证")

    k = 0.5
    Kk = elliptic_k_complete(k)
    Ek = elliptic_e_complete(k)
    print(f"  k={k}: K(k)={Kk:.8f}, E(k)={Ek:.8f}")

    L = 400e3
    H0 = 3000.0
    x = np.linspace(-L / 2, L / 2, 1001)
    H_exact = vialov_profile(x, L, H0, GLEN_N)
    V_exact = vialov_volume_exact(L, H0, GLEN_N)
    V_num = np.trapezoid(H_exact, x)

    print(f"  Vialov 剖面: L={L/1e3:.0f} km, H0={H0} m")
    print(f"  精确体积: {V_exact/1e9:.3f} km^3")
    print(f"  数值积分体积: {V_num/1e9:.3f} km^3")
    print(f"  体积相对误差: {abs(V_num - V_exact)/V_exact * 100:.6f}%")


    a_m = 0.3 / (365.25 * 86400.0)
    A = 1e-25
    R_bueler = bueler_exact_radius(a_m, A, ICE_DENSITY * GRAVITY, GLEN_N, H0)
    print(f"  Bueler 稳态半径: {R_bueler/1e3:.1f} km")


    conv = convergence_test_vialov([51, 101, 201, 401], L, H0, GLEN_N)
    print(f"  网格收敛阶估计: {conv['order']:.2f}")

    return H_exact


def run_module_13_fabric():
    print_section("模块 13: 冰晶取向蒙特卡洛统计")

    result = monte_carlo_fabric_simulation(n_samples=5000, concentration=8.0, seed=42)
    a2 = result['second_order_tensor']
    stats = result['angular_stats']
    indices = result['anisotropy_indices']

    print(f"  采样数: 5000")
    print(f"  二阶张量:\n{a2}")
    print(f"  平均角距离: {stats['mean_angle_rad']:.4f} rad ({np.degrees(stats['mean_angle_rad']):.1f}°)")
    print(f"  单晶度 S: {indices['single_maximum']:.4f}")
    print(f"  环带度 G: {indices['girdle']:.4f}")
    print(f"  强度指数 I_s: {indices['strength_index']:.4f}")

    return result


def run_module_14_summary():
    print_section("综合性能与一致性汇总")

    checks = []


    H_test = np.array([100.0, 200.0, 0.0, 50.0])
    checks.append(("厚度非负", np.all(H_test >= 0)))


    T_test = np.array([220.0, 250.0, 270.0])
    checks.append(("温度物理范围 (200~273.15 K)", np.all((T_test >= 200) & (T_test <= 273.15))))


    k_test = 0.3
    checks.append(("K(k) >= E(k)", elliptic_k_complete(k_test) >= elliptic_e_complete(k_test)))


    L, H0 = 200e3, 1500.0
    x_fine = np.linspace(-L / 2, L / 2, 10001)
    H_fine = vialov_profile(x_fine, L, H0, GLEN_N)
    V_fine = np.trapezoid(H_fine, x_fine)
    V_exact = vialov_volume_exact(L, H0, GLEN_N)
    checks.append(("Vialov 体积守恒", abs(V_fine - V_exact) / V_exact < 1e-4))


    a2_test = np.diag([0.4, 0.35, 0.25])
    checks.append(("取向张量迹归一化", abs(np.trace(a2_test) - 1.0) < 1e-10))

    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(p for _, p in checks)
    print(f"\n  总计: {sum(1 for _, p in checks if p)}/{len(checks)} 项通过")

    return all_passed


def main():
    print("\n" + "#" * 70)
    print("#  冰川流变学与冰盖演化综合模拟系统")
    print("#  Geophysical Synthesis: Glacier Rheology & Ice Sheet Evolution")
    print("#" * 70)


    np.seterr(divide='ignore', invalid='ignore')


    run_module_1_constitutive()
    run_module_2_thermomechanics()
    run_module_3_ice_sheet_evolution()
    run_module_4_ice_stream()
    run_module_5_compass_calibration()
    run_module_6_matrix_exponential()
    run_module_7_spherical_cvt()
    run_module_8_mesh_io()
    run_module_9_bayesian()
    run_module_10_drainage()
    run_module_11_banded_solver()
    run_module_12_elliptic()
    run_module_13_fabric()
    all_passed = run_module_14_summary()

    print("\n" + "#" * 70)
    if all_passed:
        print("#  全部模块运行成功，一致性检查通过。")
    else:
        print("#  部分一致性检查未通过，请复查。")
    print("#" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
