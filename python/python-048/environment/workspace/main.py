
import numpy as np
import sys


from special_math import trigamma, fracture_size_pdf, stress_intensity_factor, log_likelihood_fracture_size
from quadrature_rules import filon_cos_quad, hexagon_stroud_rule, prism_jaskowiec_rule, product_rule_1d
from fracture_geometry import create_planar_fracture_patch, FracturePatch
from stochastic_fracture import (
    generate_fracture_network_params,
    connectivity_mod2,
    lights_out_matrix,
    r8vec_normal_01_sorted,
    cliff_sequence,
)
from pore_pressure_solver import PorePressureSolver, euler_integrate
from seismic_green import SeismicGreen
from moment_tensor import MomentTensor, vandermonde_matrix, apply_caesar_rotation
from source_inversion import source_location_grid_search, moment_tensor_inversion, connectivity_source_cluster
from sensitivity_analysis import (
    sensitive_deriv,
    sensitive_exact,
    lyapunov_exponent_euler,
    FracturePropagationSensitivity,
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.random.seed(42)
    print("微地震监测与压裂裂缝网络多物理场耦合反演系统")
    print("Microseismic Monitoring & Hydraulic Fracture Network Inversion")
    print("运行模式: 零参数全自动演示")




    print_section("Step 1: 裂缝尺寸统计模型与特殊函数")
    a_obs = np.array([2.5, 3.1, 4.8, 5.2, 6.0, 8.5, 12.3, 15.0, 22.0, 35.0])
    a_min = 2.0
    D_f = 2.2
    ll = log_likelihood_fracture_size(a_obs, a_min, D_f)
    print(f"  观测裂缝尺寸: {a_obs}")
    print(f"  分形维数 D_f = {D_f}, 最小尺寸 a_min = {a_min} m")
    print(f"  对数似然 ln L = {ll:.4f}")


    tg_val, tg_err = trigamma(D_f)
    print(f"  trigamma(D_f) = {tg_val:.6f} (ifault={tg_err})")


    KI = stress_intensity_factor(sigma=20.0e6, a=5.0, geometry_factor=1.12)
    print(f"  示例应力强度因子 K_I = {KI/1e6:.3f} MPa·m^(1/2)")




    print_section("Step 2: 随机裂缝网络参数生成")
    n_frac = 12
    params = generate_fracture_network_params(n_frac, a_min=a_min, D_f=D_f)
    print(f"  生成 {n_frac} 条裂缝")
    print(f"  裂缝半长范围: [{params['lengths'].min():.2f}, {params['lengths'].max():.2f}] m")
    print(f"  走向角均值: {params['strikes'].mean():.2f}°")


    mrow, ncol = 5, 5
    occupied = (np.random.rand(mrow * ncol) > 0.55).astype(int)
    A_lo = lights_out_matrix(mrow, ncol)
    conn = connectivity_mod2(occupied, mrow, ncol)
    print(f"  离散网格连通性测试 ({mrow}x{ncol}): 存在贯通簇 = {conn}")




    print_section("Step 3: Bezier 裂缝曲面片几何建模")
    patches = []
    for i in range(min(4, n_frac)):
        center = params["positions"][i]

        strike = np.deg2rad(params["strikes"][i])
        dip = np.deg2rad(params["dips"][i])
        nvec = np.array([
            -np.sin(strike) * np.sin(dip),
            np.cos(strike) * np.sin(dip),
            -np.cos(dip)
        ])
        length = params["lengths"][i]
        height = length * 0.6
        patch = create_planar_fracture_patch(center, nvec, length, height, patch_id=i)
        patches.append(patch)
        print(f"  Patch {i}: 面积={patch.area:.2f} m², 重心=({patch.centroid()[0]:.1f}, {patch.centroid()[1]:.1f}, {patch.centroid()[2]:.1f})")




    print_section("Step 4: 孔隙压力扩散与裂缝腔体流动耦合")

    k_perm = 1.0e-15
    mu_fluid = 1.0e-3
    phi_poro = 0.15
    c_total = 1.0e-9
    D = k_perm / (mu_fluid * phi_poro * c_total)
    print(f"  水力扩散系数 D = {D:.4e} m²/s")

    solver = PorePressureSolver(
        x_min=-100.0, x_max=100.0, nx=81,
        D=D, source_rate=2.0e2, source_x=0.0
    )
    tspan = (0.0, 3600.0)
    n_steps = 800
    t_hist, p_hist = solver.solve(tspan, n_steps)
    p_final = p_hist[-1, :]
    print(f"  模拟时长: {tspan[1]} s, 时间步数: {n_steps}")
    print(f"  最终压力范围: [{p_final.min():.2e}, {p_final.max():.2e}] Pa")


    w_aperture = 2.0e-3
    v_flow = solver.cavity_flow_velocity(p_final, w_aperture, mu_fluid)
    print(f"  裂缝腔体最大流速: {v_flow.max():.4e} m/s")


    sigma_n = 50.0e6
    cfs = solver.coulomb_failure_stress(p_final, sigma_n, mu_fric=0.6, cohesion=2.0e6)
    n_failure = np.sum(cfs >= 0)
    print(f"  库仑破裂应力 CFS >= 0 的网格数: {n_failure} / {solver.x.size}")




    print_section("Step 5: 地震波格林函数与合成远场位移")
    green = SeismicGreen(rho=2650.0, alpha=4500.0, beta=2600.0)


    source_loc = np.array([10.0, 5.0, -2000.0])
    station = np.array([150.0, 80.0, -1500.0])


    mt = MomentTensor.from_strike_dip_rake(strike_deg=30.0, dip_deg=90.0, rake_deg=0.0, M0=1.0e14)
    print(f"  震源位置: {source_loc}")
    print(f"  接收台站: {station}")
    print(f"  地震矩 M_0 = {mt.seismic_moment:.4e} N·m")
    print(f"  矩震级 M_w = {mt.moment_magnitude:.3f}")


    tp = green.travel_time_p(station, source_loc)
    ts = green.travel_time_s(station, source_loc)
    print(f"  P 波走时: {tp:.4f} s, S 波走时: {ts:.4f} s")


    t_eval = tp + 0.05
    u_t = green.time_domain_displacement_filon(station, source_loc, mt.M, t_eval,
                                                omega_max=150.0, n_omega=301)
    print(f"  t={t_eval:.3f}s 时远场位移: [{u_t[0]:.6e}, {u_t[1]:.6e}, {u_t[2]:.6e}] m")


    def test_filon_func(x_arr):
        return np.exp(-x_arr**2)
    filon_val = filon_cos_quad(test_filon_func, 0.0, 2.0, 101, t=5.0)
    print(f"  Filon 测试积分 ∫_0^2 exp(-x²) cos(5x) dx ≈ {filon_val:.6f}")




    print_section("Step 6: 震源定位与矩张量反演")


    n_stations = 6
    stations = source_loc + np.random.randn(n_stations, 3) * 100.0
    stations[:, 2] = np.linspace(-1800.0, -1600.0, n_stations)
    noise_tt = np.random.randn(n_stations) * 0.02
    observed_tt = np.array([green.travel_time_p(s, source_loc) for s in stations]) + noise_tt

    best_loc, best_misfit = source_location_grid_search(
        stations, observed_tt, velocity=green.alpha,
        grid_bounds=((source_loc[0]-80, source_loc[0]+80),
                     (source_loc[1]-80, source_loc[1]+80),
                     (source_loc[2]-100, source_loc[2]+100)),
        grid_dims=(9, 9, 9)
    )
    print(f"  真实震源: {source_loc}")
    print(f"  反演震源: {best_loc}")
    print(f"  定位残差: {np.linalg.norm(best_loc - source_loc):.3f} m")
    print(f"  走时残差范数: {best_misfit:.6f}")


    synth_disp = np.zeros((n_stations, 3))
    for k in range(n_stations):
        u_syn = green.displacement_spectrum_farfield(stations[k], source_loc, mt.M, omega=50.0)
        synth_disp[k, :] = np.real(u_syn)

    mt_inv = moment_tensor_inversion(stations, synth_disp, source_loc, green)
    print(f"  反演地震矩 M_0 = {mt_inv.seismic_moment:.4e} N·m")
    print(f"  反演矩震级 M_w = {mt_inv.moment_magnitude:.3f}")


    mt_rot = apply_caesar_rotation(mt.M, k=1)
    print(f"  Caesar 旋转后矩张量迹: {np.trace(mt_rot):.4e}")


    poly_coeffs = mt_inv.interpolate_radiation_vandermonde(n_samples=8)
    print(f"  Vandermonde 辐射花样多项式系数前 4 项: {poly_coeffs[:4]}")




    print_section("Step 7: 裂缝扩展敏感性分析")
    y0 = np.array([0.01, 0.0])
    delta_y0 = np.array([1.0e-4, 0.0])
    tspan_sens = (0.0, 3.0)
    n_steps_sens = 600
    lam_est = lyapunov_exponent_euler(y0, delta_y0, tspan_sens, n_steps_sens)
    print(f"  敏感型 ODE 估计 Lyapunov 指数: λ = {lam_est:.4f}")
    print(f"  理论 Lyapunov 指数: λ = 1.0000")


    y_exact = sensitive_exact(tspan_sens[1], y0)
    _, y_euler = euler_integrate(sensitive_deriv, tspan_sens, y0, n_steps_sens)
    y_euler_final = y_euler[-1, :]
    err = np.linalg.norm(y_euler_final - y_exact)
    print(f"  Euler 终值误差 (vs 精确解): {err:.6e}")


    fps = FracturePropagationSensitivity(kappa=0.8, sigma_noise=0.05)
    dJ_dkappa = fps.adjoint_sensitivity(y0, tspan_sens, n_steps_sens, parameter_index=0)
    print(f"  伴随方法估计目标泛函对 κ 的梯度: {dJ_dkappa:.6f}")




    print_section("Step 8: 高阶求积规则数值验证")


    hex_x, hex_y, hex_w = hexagon_stroud_rule(p=3)
    hex_area = np.sum(hex_w)
    print(f"  六边形 Stroud p=3 规则: {hex_x.size} 点, 权重和 = {hex_area:.6f} (理论 = {3*np.sqrt(3)/2:.6f})")


    prism_x, prism_y, prism_z, prism_w = prism_jaskowiec_rule(p=3)
    prism_vol = np.sum(prism_w)
    print(f"  三棱柱 Jaskowiec p=3 规则: {prism_x.size} 点, 权重和 = {prism_vol:.6f} (理论 = 0.5)")


    x1d = np.array([0.0, np.sqrt(3.0/5.0), -np.sqrt(3.0/5.0)])
    w1d = np.array([8.0/9.0, 5.0/9.0, 5.0/9.0])
    X_prod, W_prod = product_rule_1d([x1d, x1d], [w1d, w1d])
    print(f"  二维乘积 Gauss-Legendre 规则: {W_prod.size} 点, 权重和 = {np.sum(W_prod):.6f} (理论 = 4.0)")




    print_section("Step 9: 破裂区三维连通性聚类")
    nx, ny, nz = 10, 10, 5
    grid_occ = np.random.rand(nx * ny * nz) > 0.88
    clusters = connectivity_source_cluster(grid_occ.astype(int), (nx, ny, nz))
    print(f"  网格规模: {nx}x{ny}x{nz}, 破裂单元数: {np.sum(grid_occ)}")
    print(f"  识别连通簇数量: {len(clusters)}")
    for ci, cl in enumerate(clusters[:3]):
        print(f"    簇 {ci+1}: 包含 {cl.shape[0]} 个单元")




    print_section("计算结果汇总")
    print(f"  裂缝数量: {n_frac}")
    print(f"  孔隙压力最大增幅: {p_final.max():.3e} Pa")
    print(f"  微震触发网格数: {n_failure}")
    print(f"  震源定位误差: {np.linalg.norm(best_loc - source_loc):.2f} m")
    print(f"  矩震级（真实/反演）: {mt.moment_magnitude:.2f} / {mt_inv.moment_magnitude:.2f}")
    print(f"  Lyapunov 指数估计: {lam_est:.4f}")
    print("\n  === 程序正常结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
