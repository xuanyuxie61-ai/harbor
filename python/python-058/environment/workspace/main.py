
import numpy as np
import time
import sys


from thermodynamics import (
    saturation_vapor_pressure, dewpoint_from_vapor_pressure,
    potential_temperature, virtual_temperature,
    specific_humidity_from_t_rh, compute_cape_cin,
    saturation_adjustment, log_gamma, lambert_w
)
from vertical_quadrature import (
    precipitable_water, mass_weighted_integral, gauss_legendre_quadrature
)
from moisture_flux import (
    moisture_flux_convergence_2d, gradient_2d_centered,
    laplacian_9point_torus, divergence_2d
)
from convection_dynamics import ConvectionDynamics
from anelastic_solver import solve_anelastic_pressure
from microphysics import StochasticMicrophysics, gamma_moment, gamma_distribution_pdf
from uncertainty_quantification import EnsembleSparseGridUQ, scale_to_physical
from ensemble_generator import EnsembleParameterSampler, SoundingProfileInterpolator
from precipitation_estimator import PrecipitationVolumeEstimator
from surface_geometry import (
    regular_surface_mesh, integrate_over_triangles,
    surface_sensible_heat_flux, surface_latent_heat_flux, triangle_area
)
from observation_optimizer import ObservationNetworkOptimizer


def print_section(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_metric(name: str, value: float, unit: str = ""):
    unit_str = f" {unit}" if unit else ""
    val_str = f"{value:>12.4e}" if abs(value) < 0.0001 and value != 0.0 else f"{value:>12.4f}"
    print(f"  {name:.<50} {val_str}{unit_str}")


def generate_standard_sounding():
    p = np.array([100000, 92500, 85000, 70000, 60000, 50000,
                  40000, 30000, 25000, 20000, 15000, 10000], dtype=float)
    T = np.array([298.0, 292.0, 288.0, 278.0, 268.0, 258.0,
                  245.0, 230.0, 220.0, 215.0, 205.0, 195.0])
    rh = np.array([0.85, 0.80, 0.75, 0.60, 0.50, 0.40,
                   0.30, 0.20, 0.15, 0.10, 0.05, 0.02])
    u = np.array([2.0, 3.0, 5.0, 8.0, 12.0, 15.0,
                  18.0, 22.0, 25.0, 28.0, 30.0, 32.0])
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                  5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
    qv = np.array([specific_humidity_from_t_rh(T[i], rh[i], p[i]) for i in range(len(p))])
    return p, T, qv, rh, u, v


def main():
    start_time = time.time()
    print("\n" + "█" * 72)
    print("█" + " " * 70 + "█")
    print("█" + "   Mesoscale Convective System (MCS) Numerical Forecasting".center(70) + "█")
    print("█" + "   Synthesis System — Python Scientific Computing Project 058".center(70) + "█")
    print("█" + " " * 70 + "█")
    print("█" * 72)




    print_section("阶段 1: 探空数据前处理与热力学诊断 (Thermodynamic Preprocessing)")

    p_levels, T_env, qv_env, rh_env, u_env, v_env = generate_standard_sounding()
    nlev = len(p_levels)
    p_sfc = p_levels[0]
    T_sfc = T_env[0]
    qv_sfc = qv_env[0]

    print(f"  探空层数: {nlev}")
    print(f"  地面气压: {p_sfc/100:.1f} hPa, 地面温度: {T_sfc:.1f} K")






    raise NotImplementedError("HOLE 3a: CAPE/CIN 调用与解包尚未实现")



    pw = precipitable_water(p_levels, qv_env, p_sfc, T_sfc)


    Tv_env = virtual_temperature(T_env, qv_env)


    theta_env = np.array([potential_temperature(T_env[i], p_levels[i]) for i in range(nlev)])

    print_metric("CAPE (对流有效位能)", cape, "J/kg")
    print_metric("CIN (对流抑制能量)", cin, "J/kg")
    print_metric("LCL 气压", p_lcl / 100.0, "hPa")
    print_metric("LFC 气压", p_lfc / 100.0, "hPa")
    print_metric("EL 气压", p_el / 100.0, "hPa")
    print_metric("PW (可降水量)", pw, "kg/m²")


    T_adj, qv_adj = saturation_adjustment(T_sfc, qv_sfc * 1.5, p_sfc)
    print_metric("饱和调整后温度", T_adj, "K")
    print_metric("饱和调整后比湿", qv_adj * 1000.0, "g/kg")




    print_section("阶段 2: 集合预报参数采样 (Ensemble Parameter Sampling)")

    param_names = ["qv_perturb", "tau_cond", "bl_coeff", "sfc_flux"]
    param_bounds = [(-0.3, 0.3), (100.0, 600.0), (0.5e-3, 2.5e-3), (50.0, 400.0)]
    sampler = EnsembleParameterSampler(param_names, param_bounds, samples_per_dim=2)
    print(f"  集合成员数: {sampler.n_ensemble} (维度={sampler.dim})")


    Td_env = np.array([dewpoint_from_vapor_pressure(
        qv_env[i] * p_levels[i] / (0.622 + 0.378 * qv_env[i])
    ) for i in range(nlev)])
    sounding_interp = SoundingProfileInterpolator(p_levels, T_env, Td_env, u_env, v_env)


    p_mid = np.array([95000, 80000, 65000, 55000, 45000, 35000, 28000, 22000, 17000, 12000], dtype=float)
    p_out, T_out, Td_out, u_out, v_out = sounding_interp.interpolate(p_mid)
    print(f"  插值到 {len(p_mid)} 个中间层完成")




    print_section("阶段 3: 对流动力学积分 (Convective Reaction-Diffusion Dynamics)")

    nx_dyn, ny_dyn = 64, 64
    dx_dyn = 2000.0
    dyn = ConvectionDynamics(nx=nx_dyn, ny=ny_dyn, dx=dx_dyn,
                              alpha=0.25, beta=0.001,
                              delta=1e-5, epsilon=0.002,
                              Du=5.0e3, Dv=1.0e2)

    dt_dyn = 0.5
    nsteps = 40
    U_final, V_final = dyn.integrate(dt_dyn, nsteps)
    convective_energy = dyn.total_convective_energy()

    print(f"  动力网格: {nx_dyn} x {ny_dyn}, 格距: {dx_dyn/1000:.1f} km")
    print(f"  积分步长: {dt_dyn} s, 总步数: {nsteps}")
    print_metric("积分后对流能量", convective_energy, "m²/s²")
    print_metric("U 场最大值", float(np.max(U_final)))
    print_metric("V 场最大值", float(np.max(V_final)))




    print_section("阶段 4: 滞弹性压力方程求解 (Anelastic Pressure Solver)")


    nx_pr = 32
    nz_pr = 16
    dx_pr = 4000.0
    dz_pr = 500.0
    x_pr = np.arange(nx_pr) * dx_pr
    z_pr = np.arange(nz_pr) * dz_pr


    rho0 = 1.225 * np.exp(-z_pr / 8500.0)
    rho0_2d = np.tile(rho0[:, np.newaxis], (1, nx_pr))


    buoyancy = np.zeros((nz_pr, nx_pr))
    for j in range(nz_pr):
        for i in range(nx_pr):

            xc = nx_pr // 2
            zc = nz_pr // 4
            r2 = ((i - xc) / 8.0)**2 + ((j - zc) / 4.0)**2
            buoyancy[j, i] = 0.05 * np.exp(-r2) * (1.0 + 0.1 * np.sin(i))


    dBdx, dBdz = gradient_2d_centered(buoyancy, dx_pr, dz_pr)
    rhs_pr = rho0_2d * (dBdx + dBdz)

    rhs_pr += 0.01 * np.random.randn(nz_pr, nx_pr)

    pressure = solve_anelastic_pressure(nx_pr, nz_pr, dx_pr, dz_pr, rho0_2d, rhs_pr)

    print(f"  压力求解网格: {nx_pr} x {nz_pr}")
    print_metric("压力扰动最大值", float(np.max(pressure)), "Pa")
    print_metric("压力扰动最小值", float(np.min(pressure)), "Pa")
    print_metric("压力扰动标准差", float(np.std(pressure)), "Pa")




    print_section("阶段 5: 水汽通量辐合诊断 (Moisture Flux Convergence)")


    X, Z = np.meshgrid(np.arange(nx_pr) * dx_pr, z_pr)
    qv_2d = qv_sfc * np.exp(-Z / 3000.0) * (1.0 + 0.2 * np.sin(X / 20000.0))
    u_2d = 5.0 + 2.0 * np.sin(X / 20000.0) + 1.0 * np.cos(Z / 5000.0)
    v_2d = 3.0 + 1.5 * np.cos(X / 15000.0)
    w_2d = 0.1 * np.sin(X / 10000.0) * np.exp(-Z / 2000.0)


    mfc_2d = moisture_flux_convergence_2d(qv_2d, u_2d, v_2d, rho0_2d, dx_pr, dz_pr)
    div_h = divergence_2d(u_2d, v_2d, dx_pr, dz_pr)

    print_metric("最大水汽通量辐合", float(np.max(mfc_2d)), "kg/(m³·s)")
    print_metric("平均水平散度", float(np.mean(div_h)), "1/s")




    print_section("阶段 6: 随机微物理参数化 (Stochastic Microphysics)")

    micro = StochasticMicrophysics(chaos_order=4, n_quad=12)

    print(f"  Laguerre 混沌阶数: 4")






    raise NotImplementedError("HOLE 3b: 微物理验证与降水率估算尚未实现")





    print_section("阶段 7: 稀疏网格不确定性量化 (Sparse Grid UQ)")



    uq_dim = 3
    uq_level = 2
    uq = EnsembleSparseGridUQ(dim=uq_dim, level=uq_level)
    print(f"  UQ 维度: {uq_dim}, 稀疏网格层数: {uq_level}")
    print(f"  稀疏网格节点数: {uq.n_points}")









    raise NotImplementedError("HOLE 3c: 预报响应函数 forecast_response 尚未实现")


    uq_bounds = [(-2.0, 2.0), (-0.2, 0.2), (150.0, 450.0)]
    mean_cape, std_cape = uq.compute_statistics(forecast_response, uq_bounds)

    print_metric("CAPE 预报均值 (UQ)", mean_cape, "J/kg")
    print_metric("CAPE 预报标准差 (UQ)", std_cape, "J/kg")




    print_section("阶段 8: 降水体积蒙特卡洛估算 (Precipitation Volume MC)")

    precip_est = PrecipitationVolumeEstimator(n_samples_per_cell=128)



    nz_rain = 8
    ny_rain = 32
    nx_rain = 32
    dx_rain = 4000.0
    dy_rain = 4000.0
    dz_rain = np.ones(nz_rain) * 500.0

    precip_field = np.zeros((nz_rain, ny_rain, nx_rain))
    for k in range(nz_rain):
        for j in range(ny_rain):
            for i in range(nx_rain):
                xc, yc = nx_rain // 2, ny_rain // 2
                r2 = ((i - xc) / 10.0)**2 + ((j - yc) / 10.0)**2

                base_rate = 2.0e-7 * np.exp(-r2) * max(0.0, 1.0 - k / nz_rain)
                precip_field[k, j, i] = base_rate

    total_precip_mass_flux = precip_est.estimate_from_gridded_field(
        precip_field, dx_rain, dy_rain, dz_rain
    )

    domain_area = nx_rain * ny_rain * dx_rain * dy_rain
    rho_water = 1000.0
    equiv_depth = total_precip_mass_flux / (rho_water * domain_area) * 1000.0 * 3600.0
    print_metric("总降水质量通量", total_precip_mass_flux, "kg/s")
    print_metric("等效区域平均降水率", equiv_depth, "mm/hr")




    print_section("阶段 9: 地表通量面积分 (Surface Flux Integration)")


    triangles = regular_surface_mesh(nx=16, ny=16,
                                      xlim=(0.0, 200000.0),
                                      ylim=(0.0, 200000.0))
    print(f"  地表三角形单元数: {len(triangles)}")


    def sensible_flux_func(pt: np.ndarray) -> float:
        x, y = pt[0], pt[1]
        t_sfc = 305.0 + 3.0 * np.sin(x / 20000.0) * np.cos(y / 20000.0)
        t_air = 298.0
        ws = 3.0 + 2.0 * np.sin(y / 30000.0)
        return surface_sensible_heat_flux(t_sfc, t_air, ws)

    def latent_flux_func(pt: np.ndarray) -> float:
        x, y = pt[0], pt[1]
        q_sfc = 0.020 + 0.003 * np.sin(x / 25000.0)
        q_air = 0.015
        ws = 3.0 + 2.0 * np.sin(y / 30000.0)
        return surface_latent_heat_flux(q_sfc, q_air, ws)

    total_H = integrate_over_triangles(triangles, sensible_flux_func)
    total_LE = integrate_over_triangles(triangles, latent_flux_func)
    total_area = sum(triangle_area(t[0], t[1], t[2]) for t in triangles)

    print_metric("总感热通量", total_H, "W")
    print_metric("总潜热通量", total_LE, "W")
    print_metric("地表总面积", total_area / 1e6, "km²")
    print_metric("平均感热通量密度", total_H / total_area, "W/m²")
    print_metric("平均潜热通量密度", total_LE / total_area, "W/m²")




    print_section("阶段 10: 观测网络优化 (CCVT Observation Network)")

    obs_opt = ObservationNetworkOptimizer(domain_km=(0.0, 200.0, 0.0, 200.0))
    radar_pos = obs_opt.optimize_radar_placement(n_radars=4)
    station_pos = obs_opt.optimize_station_network(n_stations=12)

    radar_score = obs_opt.coverage_score(radar_pos)
    station_score = obs_opt.coverage_score(station_pos)

    print(f"  雷达站点数: {len(radar_pos)}")
    print_metric("雷达覆盖均匀度", radar_score)
    print(f"  气象站数: {len(station_pos)}")
    print_metric("气象站覆盖均匀度", station_score)




    print_section("阶段 11: 综合诊断与数值验证 (Integrated Diagnostics)")


    def test_integrand(x: float) -> float:
        return np.exp(-x**2)

    gl_result = gauss_legendre_quadrature(test_integrand, 0.0, 2.0, n=16)
    exact_ref = 0.5 * np.sqrt(np.pi) * 0.9953222650189527
    print_metric("Gauss-Legendre 测试积分", gl_result)
    print_metric("参考值", exact_ref)
    print_metric("相对误差", abs(gl_result - exact_ref) / (abs(exact_ref) + 1e-12))


    w_test = lambert_w(1.0)
    print_metric("Lambert W(1)", w_test)
    print_metric("W(1)*exp(W(1))", w_test * np.exp(w_test))


    lg_test = log_gamma(5.0)
    print_metric("log Γ(5)", lg_test)
    print_metric("参考值 log(24)", np.log(24.0))


    elapsed = time.time() - start_time
    print("\n" + "█" * 72)
    print(f"  全部计算完成. 总耗时: {elapsed:.3f} 秒")
    print("  中尺度对流系统数值预报综合模拟系统 — 运行成功")
    print("█" * 72 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
