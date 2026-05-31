"""
================================================================================
中尺度对流系统数值预报综合模拟系统
Mesoscale Convective System (MCS) Numerical Forecasting Synthesis System
================================================================================

博士级科研代码合成项目 (PROJECT_58)
领域: 大气科学 — 中尺度对流系统数值预报

集成 15 个种子项目核心算法, 构建从前处理、动力核心、微物理参数化、
不确定性量化到观测网络优化的完整 MCS 预报流程.

运行方式:
    python main.py
    (零参数输入, 自动执行完整预报流程并输出诊断结果)
================================================================================
"""

import numpy as np
import time
import sys

# ── 导入各子模块 ────────────────────────────────────────────
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
    """打印分节标题."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_metric(name: str, value: float, unit: str = ""):
    """格式化输出诊断量."""
    unit_str = f" {unit}" if unit else ""
    val_str = f"{value:>12.4e}" if abs(value) < 0.0001 and value != 0.0 else f"{value:>12.4f}"
    print(f"  {name:.<50} {val_str}{unit_str}")


def generate_standard_sounding():
    """
    生成标准探空廓线 (模拟典型的 MCS 前兆环境).
    气压从 1000 hPa 递减至 100 hPa.
    """
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

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 1: 探空数据前处理与热力学诊断
    # ═══════════════════════════════════════════════════════════════════════
    print_section("阶段 1: 探空数据前处理与热力学诊断 (Thermodynamic Preprocessing)")

    p_levels, T_env, qv_env, rh_env, u_env, v_env = generate_standard_sounding()
    nlev = len(p_levels)
    p_sfc = p_levels[0]
    T_sfc = T_env[0]
    qv_sfc = qv_env[0]

    print(f"  探空层数: {nlev}")
    print(f"  地面气压: {p_sfc/100:.1f} hPa, 地面温度: {T_sfc:.1f} K")

    # 热力学诊断
    cape, cin, p_lcl, p_lfc, p_el = compute_cape_cin(
        p_levels, T_env, qv_env, p_sfc, T_sfc, qv_sfc
    )

    # 可降水量 (高精度 Gauss-Legendre 垂直积分)
    pw = precipitable_water(p_levels, qv_env, p_sfc, T_sfc)

    # 虚温环境廓线
    Tv_env = virtual_temperature(T_env, qv_env)

    # 潜在温度廓线
    theta_env = np.array([potential_temperature(T_env[i], p_levels[i]) for i in range(nlev)])

    print_metric("CAPE (对流有效位能)", cape, "J/kg")
    print_metric("CIN (对流抑制能量)", cin, "J/kg")
    print_metric("LCL 气压", p_lcl / 100.0, "hPa")
    print_metric("LFC 气压", p_lfc / 100.0, "hPa")
    print_metric("EL 气压", p_el / 100.0, "hPa")
    print_metric("PW (可降水量)", pw, "kg/m²")

    # 饱和调整验证 (不动点迭代) — 使用过饱和条件触发凝结
    T_adj, qv_adj = saturation_adjustment(T_sfc, qv_sfc * 1.5, p_sfc)
    print_metric("饱和调整后温度", T_adj, "K")
    print_metric("饱和调整后比湿", qv_adj * 1000.0, "g/kg")

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 2: 集合参数采样 (Hypercube Grid + Nearest Interp)
    # ═══════════════════════════════════════════════════════════════════════
    print_section("阶段 2: 集合预报参数采样 (Ensemble Parameter Sampling)")

    param_names = ["qv_perturb", "tau_cond", "bl_coeff", "sfc_flux"]
    param_bounds = [(-0.3, 0.3), (100.0, 600.0), (0.5e-3, 2.5e-3), (50.0, 400.0)]
    sampler = EnsembleParameterSampler(param_names, param_bounds, samples_per_dim=2)
    print(f"  集合成员数: {sampler.n_ensemble} (维度={sampler.dim})")

    # 探空插值器
    Td_env = np.array([dewpoint_from_vapor_pressure(
        qv_env[i] * p_levels[i] / (0.622 + 0.378 * qv_env[i])
    ) for i in range(nlev)])
    sounding_interp = SoundingProfileInterpolator(p_levels, T_env, Td_env, u_env, v_env)

    # 对中间层插值
    p_mid = np.array([95000, 80000, 65000, 55000, 45000, 35000, 28000, 22000, 17000, 12000], dtype=float)
    p_out, T_out, Td_out, u_out, v_out = sounding_interp.interpolate(p_mid)
    print(f"  插值到 {len(p_mid)} 个中间层完成")

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 3: 对流动力学积分 (Reaction-Diffusion PDE)
    # ═══════════════════════════════════════════════════════════════════════
    print_section("阶段 3: 对流动力学积分 (Convective Reaction-Diffusion Dynamics)")

    nx_dyn, ny_dyn = 64, 64
    dx_dyn = 2000.0  # 2 km
    dyn = ConvectionDynamics(nx=nx_dyn, ny=ny_dyn, dx=dx_dyn,
                              alpha=0.25, beta=0.001,
                              delta=1e-5, epsilon=0.002,
                              Du=5.0e3, Dv=1.0e2)

    dt_dyn = 0.5  # 数值稳定性约束
    nsteps = 40
    U_final, V_final = dyn.integrate(dt_dyn, nsteps)
    convective_energy = dyn.total_convective_energy()

    print(f"  动力网格: {nx_dyn} x {ny_dyn}, 格距: {dx_dyn/1000:.1f} km")
    print(f"  积分步长: {dt_dyn} s, 总步数: {nsteps}")
    print_metric("积分后对流能量", convective_energy, "m²/s²")
    print_metric("U 场最大值", float(np.max(U_final)))
    print_metric("V 场最大值", float(np.max(V_final)))

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 4: 滞弹性压力求解 (Sparse CG)
    # ═══════════════════════════════════════════════════════════════════════
    print_section("阶段 4: 滞弹性压力方程求解 (Anelastic Pressure Solver)")

    # 构建简化的 x-z 剖面密度与浮力
    nx_pr = 32
    nz_pr = 16
    dx_pr = 4000.0
    dz_pr = 500.0
    x_pr = np.arange(nx_pr) * dx_pr
    z_pr = np.arange(nz_pr) * dz_pr

    # 基态密度 (静力近似)
    rho0 = 1.225 * np.exp(-z_pr / 8500.0)  # 指数密度递减
    rho0_2d = np.tile(rho0[:, np.newaxis], (1, nx_pr))

    # 浮力场 (与对流强度关联)
    buoyancy = np.zeros((nz_pr, nx_pr))
    for j in range(nz_pr):
        for i in range(nx_pr):
            # 简化的浮力分布: 中心区域有正浮力
            xc = nx_pr // 2
            zc = nz_pr // 4
            r2 = ((i - xc) / 8.0)**2 + ((j - zc) / 4.0)**2
            buoyancy[j, i] = 0.05 * np.exp(-r2) * (1.0 + 0.1 * np.sin(i))

    # 右端项: ∇·(ρ0 B)
    dBdx, dBdz = gradient_2d_centered(buoyancy, dx_pr, dz_pr)
    rhs_pr = rho0_2d * (dBdx + dBdz)
    # 加入变形项近似
    rhs_pr += 0.01 * np.random.randn(nz_pr, nx_pr)

    pressure = solve_anelastic_pressure(nx_pr, nz_pr, dx_pr, dz_pr, rho0_2d, rhs_pr)

    print(f"  压力求解网格: {nx_pr} x {nz_pr}")
    print_metric("压力扰动最大值", float(np.max(pressure)), "Pa")
    print_metric("压力扰动最小值", float(np.min(pressure)), "Pa")
    print_metric("压力扰动标准差", float(np.std(pressure)), "Pa")

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 5: 水汽通量辐合诊断 (3D Gradient)
    # ═══════════════════════════════════════════════════════════════════════
    print_section("阶段 5: 水汽通量辐合诊断 (Moisture Flux Convergence)")

    # 构建简化的 2D 水汽与风场 (x-z 剖面)
    X, Z = np.meshgrid(np.arange(nx_pr) * dx_pr, z_pr)
    qv_2d = qv_sfc * np.exp(-Z / 3000.0) * (1.0 + 0.2 * np.sin(X / 20000.0))
    u_2d = 5.0 + 2.0 * np.sin(X / 20000.0) + 1.0 * np.cos(Z / 5000.0)
    v_2d = 3.0 + 1.5 * np.cos(X / 15000.0)
    w_2d = 0.1 * np.sin(X / 10000.0) * np.exp(-Z / 2000.0)

    # 仅计算水平辐合 (2D 简化)
    mfc_2d = moisture_flux_convergence_2d(qv_2d, u_2d, v_2d, rho0_2d, dx_pr, dz_pr)
    div_h = divergence_2d(u_2d, v_2d, dx_pr, dz_pr)

    print_metric("最大水汽通量辐合", float(np.max(mfc_2d)), "kg/(m³·s)")
    print_metric("平均水平散度", float(np.mean(div_h)), "1/s")

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 6: 随机微物理参数化 (Laguerre Chaos)
    # ═══════════════════════════════════════════════════════════════════════
    print_section("阶段 6: 随机微物理参数化 (Stochastic Microphysics)")

    micro = StochasticMicrophysics(chaos_order=4, n_quad=12)

    print(f"  Laguerre 混沌阶数: 4")

    # Gamma 分布雨滴谱验证
    D_test = np.linspace(0.1, 5.0, 50)
    N_D = gamma_distribution_pdf(D_test, N0=8e6, mu=2.0, lam=3.0)
    M3 = gamma_moment(3, N0=8e6, mu=2.0, lam=3.0)
    print_metric("雨滴谱 3 阶矩", M3, "mm³·m⁻³")

    # 降水率估算
    ql_test = 0.5e-3  # 云水混合比 kg/kg
    precip_rate = micro.precipitation_rate(ql_test)
    print_metric("单点降水率估算", precip_rate * 3600, "mm/hr")

    # 凝结率混沌展开 (使用过饱和条件触发非零凝结)
    qvs_sfc = saturation_vapor_pressure(T_sfc) * 0.622 / p_sfc
    cond_coeffs = micro.condensate_rate_ensemble(qv_sfc * 1.2, qvs_sfc, tau_mean=300.0, tau_std=60.0)
    print_metric("凝结率混沌 0 阶系数", cond_coeffs[0] * 1000.0, "g/(kg·s)")

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 7: 稀疏网格不确定性量化 (Sparse Grid UQ)
    # ═══════════════════════════════════════════════════════════════════════
    print_section("阶段 7: 稀疏网格不确定性量化 (Sparse Grid UQ)")

    # 对 CAPE 的预报不确定性进行量化
    # 定义参数空间: [初始温度扰动, 水汽扰动比例, 微物理时间尺度]
    uq_dim = 3
    uq_level = 2
    uq = EnsembleSparseGridUQ(dim=uq_dim, level=uq_level)
    print(f"  UQ 维度: {uq_dim}, 稀疏网格层数: {uq_level}")
    print(f"  稀疏网格节点数: {uq.n_points}")

    # 定义一个简化的预报响应函数: CAPE 对参数的敏感性
    def forecast_response(params: np.ndarray) -> float:
        dT = params[0] if len(params) > 0 else 0.0
        dqv = params[1] if len(params) > 1 else 0.0
        tau = params[2] if len(params) > 2 else 300.0
        # 简化的参数化响应
        cape_pert = cape * (1.0 + 0.05 * dT + 0.3 * dqv) * (300.0 / max(tau, 50.0))**0.1
        return max(0.0, cape_pert)

    uq_bounds = [(-2.0, 2.0), (-0.2, 0.2), (150.0, 450.0)]
    mean_cape, std_cape = uq.compute_statistics(forecast_response, uq_bounds)

    print_metric("CAPE 预报均值 (UQ)", mean_cape, "J/kg")
    print_metric("CAPE 预报标准差 (UQ)", std_cape, "J/kg")

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 8: 降水体积蒙特卡洛估算 (Tetrahedral MC)
    # ═══════════════════════════════════════════════════════════════════════
    print_section("阶段 8: 降水体积蒙特卡洛估算 (Precipitation Volume MC)")

    precip_est = PrecipitationVolumeEstimator(n_samples_per_cell=128)

    # 从格点降水场估算总降水体积
    # 构建简化的 3D 降水生成率场 (kg/(m³·s))
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
                # 降水生成率: 2e-7 kg/(m³·s), 随高度和水平距离衰减
                base_rate = 2.0e-7 * np.exp(-r2) * max(0.0, 1.0 - k / nz_rain)
                precip_field[k, j, i] = base_rate

    total_precip_mass_flux = precip_est.estimate_from_gridded_field(
        precip_field, dx_rain, dy_rain, dz_rain
    )
    # 转换为等效降水深度率: mass_flux / (rho_water * area) * 1000(mm/m) * 3600(s/hr)
    domain_area = nx_rain * ny_rain * dx_rain * dy_rain
    rho_water = 1000.0  # kg/m³
    equiv_depth = total_precip_mass_flux / (rho_water * domain_area) * 1000.0 * 3600.0
    print_metric("总降水质量通量", total_precip_mass_flux, "kg/s")
    print_metric("等效区域平均降水率", equiv_depth, "mm/hr")

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 9: 地表通量面积分 (Triangle Geometry)
    # ═══════════════════════════════════════════════════════════════════════
    print_section("阶段 9: 地表通量面积分 (Surface Flux Integration)")

    # 生成地表三角形网格
    triangles = regular_surface_mesh(nx=16, ny=16,
                                      xlim=(0.0, 200000.0),
                                      ylim=(0.0, 200000.0))
    print(f"  地表三角形单元数: {len(triangles)}")

    # 定义地表通量函数 (随空间变化)
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

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 10: 观测网络优化 (CCVT)
    # ═══════════════════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 11: 高斯求积验证与综合诊断
    # ═══════════════════════════════════════════════════════════════════════
    print_section("阶段 11: 综合诊断与数值验证 (Integrated Diagnostics)")

    # 使用 Gauss-Legendre 验证积分精度
    def test_integrand(x: float) -> float:
        return np.exp(-x**2)

    gl_result = gauss_legendre_quadrature(test_integrand, 0.0, 2.0, n=16)
    exact_ref = 0.5 * np.sqrt(np.pi) * 0.9953222650189527  # 0.5*sqrt(pi)*erf(2)
    print_metric("Gauss-Legendre 测试积分", gl_result)
    print_metric("参考值", exact_ref)
    print_metric("相对误差", abs(gl_result - exact_ref) / (abs(exact_ref) + 1e-12))

    # Lambert W 验证
    w_test = lambert_w(1.0)
    print_metric("Lambert W(1)", w_test)
    print_metric("W(1)*exp(W(1))", w_test * np.exp(w_test))

    # log-Gamma 验证
    lg_test = log_gamma(5.0)
    print_metric("log Γ(5)", lg_test)
    print_metric("参考值 log(24)", np.log(24.0))

    # 总运行时间
    elapsed = time.time() - start_time
    print("\n" + "█" * 72)
    print(f"  全部计算完成. 总耗时: {elapsed:.3f} 秒")
    print("  中尺度对流系统数值预报综合模拟系统 — 运行成功")
    print("█" * 72 + "\n")

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 补充导入（测试用例所需）
# ================================================================
from vertical_quadrature import legendre_gauss_nodes_weights
from anelastic_solver import SparseMatrixCOO, conjugate_gradient
from ensemble_generator import hypercube_grid, nearest_interp_1d
from microphysics import laguerre_polynomials
from precipitation_estimator import tetrahedron01_volume

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: saturation_vapor_pressure 输出为正的有限值 ----
es = saturation_vapor_pressure(300.0)
assert es > 0.0 and np.isfinite(es), '[TC01] saturation_vapor_pressure FAILED'

# ---- TC02: lambert_w(1) 满足 W*exp(W)=1 ----
w1 = lambert_w(1.0)
assert abs(w1 * np.exp(w1) - 1.0) < 1e-10, '[TC02] lambert_w FAILED'

# ---- TC03: log_gamma(5) 等于 ln(24) ----
lg5 = log_gamma(5.0)
assert abs(lg5 - np.log(24.0)) < 1e-10, '[TC03] log_gamma FAILED'

# ---- TC04: potential_temperature 随气压降低而增大 ----
theta1 = potential_temperature(300.0, 100000.0)
theta2 = potential_temperature(300.0, 50000.0)
assert theta2 > theta1, '[TC04] potential_temperature FAILED'

# ---- TC05: virtual_temperature 非负 ----
tv = virtual_temperature(280.0, np.array([0.0, 0.01, 0.02]))
assert np.all(tv >= 0.0), '[TC05] virtual_temperature FAILED'

# ---- TC06: specific_humidity_from_t_rh 边界截断 ----
q = specific_humidity_from_t_rh(350.0, 1.0, 100000.0)
assert 0.0 <= q <= 0.05, '[TC06] specific_humidity_from_t_rh FAILED'

# ---- TC07: dewpoint_from_vapor_pressure 逆一致性 ----
es = saturation_vapor_pressure(295.0)
td = dewpoint_from_vapor_pressure(es)
assert abs(td - 295.0) < 0.5, '[TC07] dewpoint_from_vapor_pressure FAILED'

# ---- TC08: saturation_adjustment 未饱和返回原值 ----
T_adj, qv_adj = saturation_adjustment(300.0, 0.005, 100000.0)
assert abs(T_adj - 300.0) < 1e-6 and abs(qv_adj - 0.005) < 1e-6, '[TC08] saturation_adjustment FAILED'

# ---- TC09: StochasticMicrophysics precipitation_rate 非负 ----
micro = StochasticMicrophysics(chaos_order=2, n_quad=8)
rate = micro.precipitation_rate(0.5e-3)
assert rate >= 0.0 and np.isfinite(rate), '[TC09] StochasticMicrophysics precipitation_rate FAILED'

# ---- TC10: EnsembleSparseGridUQ 标准差非负 ----
uq = EnsembleSparseGridUQ(dim=2, level=2)
mean, std = uq.compute_statistics(lambda p: p[0] + p[1], [(-1.0, 1.0), (-1.0, 1.0)])
assert std >= 0.0 and np.isfinite(std), '[TC10] EnsembleSparseGridUQ FAILED'

# ---- TC11: gauss_legendre_quadrature 精确积分三次多项式 ----
result = gauss_legendre_quadrature(lambda x: x**3, 0.0, 1.0, n=4)
assert abs(result - 0.25) < 1e-14, '[TC11] gauss_legendre_quadrature FAILED'

# ---- TC12: legendre_gauss_nodes_weights 权重和为 2 ----
x, w = legendre_gauss_nodes_weights(8)
assert abs(np.sum(w) - 2.0) < 1e-12, '[TC12] legendre_gauss_nodes_weights FAILED'

# ---- TC13: gradient_2d_centered 常数场梯度为零 ----
const_field = np.ones((8, 8))
dfdx, dfdy = gradient_2d_centered(const_field, 1000.0, 1000.0)
assert np.allclose(dfdx, 0.0) and np.allclose(dfdy, 0.0), '[TC13] gradient_2d_centered FAILED'

# ---- TC14: divergence_2d 均匀场散度为零 ----
u = np.ones((8, 8)) * 5.0
v = np.ones((8, 8)) * 3.0
div = divergence_2d(u, v, 1000.0, 1000.0)
assert np.allclose(div, 0.0, atol=1e-12), '[TC14] divergence_2d FAILED'

# ---- TC15: laplacian_9point_torus 常数场 Laplacian 为零 ----
const_field = np.ones((8, 8))
lap = laplacian_9point_torus(const_field, 1000.0, 1000.0)
assert np.allclose(lap, 0.0, atol=1e-12), '[TC15] laplacian_9point_torus FAILED'

# ---- TC16: SparseMatrixCOO 矩阵向量乘法正确性 ----
A = SparseMatrixCOO(3, 3)
A.append(0, 0, 2.0)
A.append(1, 1, 3.0)
A.append(2, 2, 4.0)
x_vec = np.array([1.0, 2.0, 3.0])
y_vec = A.mv(x_vec)
assert np.allclose(y_vec, np.array([2.0, 6.0, 12.0])), '[TC16] SparseMatrixCOO.mv FAILED'

# ---- TC17: conjugate_gradient 单位矩阵精确解 ----
A = SparseMatrixCOO(3, 3)
for i in range(3): A.append(i, i, 1.0)
b = np.array([1.0, 2.0, 3.0])
x_sol, iters, res = conjugate_gradient(A, b)
assert np.allclose(x_sol, b) and res < 1e-8, '[TC17] conjugate_gradient FAILED'

# ---- TC18: solve_anelastic_pressure 输出形状正确 ----
nx, nz = 4, 3
rho = np.ones((nz, nx))
rhs = np.zeros((nz, nx))
p = solve_anelastic_pressure(nx, nz, 1000.0, 500.0, rho, rhs)
assert p.shape == (nz, nx), '[TC18] solve_anelastic_pressure FAILED'

# ---- TC19: ConvectionDynamics 积分后 U,V 在 [0,1] 内 ----
dyn = ConvectionDynamics(nx=8, ny=8, dx=2000.0)
U, V = dyn.integrate(dt=0.5, nsteps=3)
assert np.all(U >= 0.0) and np.all(U <= 1.0), '[TC19] ConvectionDynamics U range FAILED'
assert np.all(V >= 0.0) and np.all(V <= 1.0), '[TC19] ConvectionDynamics V range FAILED'

# ---- TC20: ConvectionDynamics 对流能量非负有限 ----
dyn2 = ConvectionDynamics(nx=8, ny=8, dx=2000.0)
dyn2.integrate(dt=0.5, nsteps=3)
ece = dyn2.total_convective_energy()
assert ece >= 0.0 and np.isfinite(ece), '[TC20] ConvectionDynamics energy FAILED'

# ---- TC21: hypercube_grid 输出尺寸与张量积一致 ----
grid = hypercube_grid(2, [3, 2], [(0.0, 1.0), (-1.0, 1.0)])
assert grid.shape == (6, 2), '[TC21] hypercube_grid FAILED'

# ---- TC22: nearest_interp_1d 精确命中数据点 ----
xd = np.array([1.0, 2.0, 3.0])
yd = np.array([10.0, 20.0, 30.0])
yi = nearest_interp_1d(xd, yd, np.array([2.0]))
assert abs(yi[0] - 20.0) < 1e-12, '[TC22] nearest_interp_1d FAILED'

# ---- TC23: gamma_distribution_pdf 输出非负有限 ----
D = np.linspace(0.1, 5.0, 10)
pdf = gamma_distribution_pdf(D, N0=8e6, mu=2.0, lam=3.0)
assert np.all(pdf >= 0.0) and np.all(np.isfinite(pdf)), '[TC23] gamma_distribution_pdf FAILED'

# ---- TC24: gamma_moment k=0 解析验证 ----
M0 = gamma_moment(0, N0=8e6, mu=2.0, lam=3.0)
expected = 8e6 * 2.0 / 27.0
assert abs(M0 - expected) / expected < 1e-6, '[TC24] gamma_moment FAILED'

# ---- TC25: laguerre_polynomials L_0 恒为 1 ----
x_test = np.array([0.0, 1.0, 2.0])
L = laguerre_polynomials(0, x_test)
assert np.allclose(L[0, :], 1.0), '[TC25] laguerre_polynomials FAILED'

# ---- TC26: scale_to_physical 端点映射正确 ----
pts = np.array([[-1.0], [0.0], [1.0]])
scaled = scale_to_physical(pts, [(0.0, 10.0)])
assert abs(scaled[0, 0] - 0.0) < 1e-12 and abs(scaled[2, 0] - 10.0) < 1e-12, '[TC26] scale_to_physical FAILED'

# ---- TC27: triangle_area 平面直角三角形面积 ----
v0 = np.array([0.0, 0.0, 0.0])
v1 = np.array([3.0, 0.0, 0.0])
v2 = np.array([0.0, 4.0, 0.0])
A = triangle_area(v0, v1, v2)
assert abs(A - 6.0) < 1e-12, '[TC27] triangle_area FAILED'

# ---- TC28: tetrahedron01_volume 解析值 1/6 ----
vol = tetrahedron01_volume()
assert abs(vol - 1.0 / 6.0) < 1e-12, '[TC28] tetrahedron01_volume FAILED'

# ---- TC29: surface_sensible_heat_flux 暖面向上为正 ----
H = surface_sensible_heat_flux(305.0, 298.0, 5.0)
assert H > 0.0, '[TC29] surface_sensible_heat_flux FAILED'

# ---- TC30: main 集成测试返回 0 ----
ret = main()
assert ret == 0, '[TC30] main FAILED'

print('\n全部 30 个测试通过!\n')
