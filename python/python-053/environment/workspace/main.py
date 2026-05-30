
import numpy as np
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calendar_utils import (
    ymd_to_jed, seasonal_phase, phase_locking_index, enso_event_timing
)
from sparse_matrix_utils import (
    ge_to_crs, ge_to_st, build_laplacian_2d_crs, crs_matvec
)
from sst_interpolator import (
    lagrange_interp_2d, chebyshev_nodes_1d, interpolate_sst_field, nino34_index
)
from ocean_heat_content import (
    prism_unit_monomial_integral, prism_witherden_rule,
    integrate_ohc_over_prism, thermocline_depth_from_profile, warm_water_volume
)
from equatorial_wave_solver import (
    solve_burgers_etdrk4, solve_coupled_wave_envelope,
    wave_energy, recharge_discharge_timescale
)
from recharge_discharge_oscillator import (
    solve_rdo, find_equilibrium, oscillation_period_approx, classify_dynamics
)
from chaos_analysis import (
    lyapunov_exponent_1d, correlation_dimension,
    levy_dragon_ifs, cross_chaos_ifs,
    enso_lyapunov_exponent, bifurcation_diagram
)
from enso_predictor import (
    classify_enso_state, state_name, transition_probability, ENSOState,
    monte_carlo_enso_forecast, forecast_skill, probabilistic_event_forecast
)
from multiscale_analysis import (
    cwt_1d, global_wavelet_spectrum, red_noise_spectrum,
    find_scale_peaks, nested_multiscale_analysis
)
from quadrature_validation import (
    validate_hypercube_quadrature, validate_hermite_quadrature_1d,
    gauss_legendre_points_weights_1d, gauss_hermite_points_weights_1d,
    ensemble_mean_integral
)
from parameter_inversion import (
    solve_steady_heat_2d, piecewise_diffusivity,
    gradient_descent_inversion, sensitivity_analysis
)
from io_utils import (
    write_mesh_data, read_mesh_data,
    write_sst_timeseries, read_sst_timeseries,
    compute_checksum
)


def section_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_calendar_utils():
    section_header("1. 日历工具与 ENSO 季节锁相分析 (calendar_utils)")


    jed = ymd_to_jed(1997, 12, 1)
    print(f"  1997/12/01 的儒略日数 JED = {jed:.4f}")


    theta = seasonal_phase(1997, 12, 1)
    print(f"  该日期的季节相位角 θ = {theta:.4f} rad ({np.degrees(theta):.1f}°)")


    event_months = [12, 11, 12, 1, 12, 11, 1, 12, 12, 1]
    pli = phase_locking_index(event_months)
    print(f"  模拟 ENSO 事件月份: {event_months}")
    print(f"  季节锁相强度 PLI = {pli:.4f} (越接近 1 锁相越强)")


    events = [(1997, 12, 1, 2.5), (1982, 12, 15, 2.3), (2015, 12, 10, 2.6),
              (2009, 12, 5, 1.2), (1972, 12, 20, 1.8)]
    timing = enso_event_timing(events)
    print(f"  历史 El Nino 事件分析:")
    print(f"    平均间隔 = {timing['mean_interval_years']:.2f} 年")
    print(f"    间隔标准差 = {timing['std_interval_years']:.2f} 年")
    print(f"    锁相强度 = {timing['phase_locking_index']:.4f}")


def demo_sparse_matrix():
    section_header("2. 稀疏矩阵与海洋 Laplacian 离散 (sparse_matrix_utils)")

    nx, ny = 20, 15
    dx, dy = 2.0e5, 2.0e5


    n, nz, row_ptr, col, val = build_laplacian_2d_crs(nx, ny, dx, dy, boundary="dirichlet")
    print(f"  网格: {nx} x {ny}, 总节点 = {n}")
    print(f"  Laplacian 非零元数量 nz = {nz} (稀疏度 = {nz / (n * n) * 100:.2f}%)")


    x_vec = np.random.rand(n)
    y_vec = crs_matvec(n, row_ptr, col, val, x_vec)
    print(f"  CRS 矩阵-向量乘法测试通过，结果范数 = {np.linalg.norm(y_vec):.4e}")


    A_dense = np.array([[1.0, 0.0, 2.0],
                        [0.0, 3.0, 0.0],
                        [4.0, 0.0, 5.0]])
    nz_st, ist, jst, Ast = ge_to_st(A_dense)
    print(f"  GE->ST 转换: {nz_st} 个非零元")


    n_crs, nz_crs, rp, c, v = ge_to_crs(A_dense)
    print(f"  GE->CRS 转换: {nz_crs} 个非零元")


def demo_sst_interpolation():
    section_header("3. 海表温度空间插值与 Niño 3.4 指数 (sst_interpolator)")


    lon = np.linspace(120.0, 280.0, 41)
    lat = np.linspace(-20.0, 20.0, 21)


    LON, LAT = np.meshgrid(lon, lat, indexing='ij')
    sst = 25.0 + 3.0 * np.cos(np.radians(LON - 200.0)) * np.exp(-0.5 * (LAT / 10.0) ** 2)


    nino = nino34_index(sst - sst.mean(), lon, lat)
    print(f"  合成 SST 场的 Niño 3.4 指数 = {nino:.4f} °C")


    xd = chebyshev_nodes_1d(6, 120.0, 280.0)
    yd = chebyshev_nodes_1d(5, -20.0, 20.0)
    zd = np.random.rand(xd.shape[0] * yd.shape[0])
    xi = np.array([200.0, 220.0])
    yi = np.array([0.0, 5.0])
    zi = lagrange_interp_2d(xd.shape[0] - 1, yd.shape[0] - 1, xd, yd, zd, xi, yi)
    print(f"  二维 Lagrange 插值测试: 在 (200°, 0°) 处插值 = {zi[0]:.4f}")


    lon_target = np.array([190.0, 210.0, 230.0])
    lat_target = np.array([0.0, 2.0, -2.0])
    sst_interp = interpolate_sst_field(lon, lat, sst, lon_target, lat_target,
                                       degree_lon=3, degree_lat=3)
    print(f"  SST 场插值到目标点: {sst_interp}")


def demo_ocean_heat_content():
    section_header("4. 三维海洋热含量与温跃层计算 (ocean_heat_content)")


    exact = prism_unit_monomial_integral((1, 1, 1))
    print(f"  单位棱柱 x*y*z 精确积分 = {exact:.8f} (理论值 = 1/48 = 0.0208333)")


    x_r, y_r, z_r, w_r = prism_witherden_rule(p=5)
    numerical = np.sum(w_r * x_r * y_r * z_r)
    print(f"  Witherden p=5 数值积分 = {numerical:.8f}, 相对误差 = {abs(numerical - exact) / exact:.2e}")


    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0e6, 0.0, 0.0],
        [0.0, 1.0e6, 0.0],
        [0.0, 0.0, -200.0],
        [1.0e6, 0.0, -200.0],
        [0.0, 1.0e6, -200.0],
    ])

    def temp_profile(x, y, z):

        t_surface = 28.0
        t_deep = 10.0
        depth_ratio = abs(z) / 200.0
        return t_surface - (t_surface - t_deep) * depth_ratio

    ohc = integrate_ohc_over_prism(temp_profile, vertices, rho0=1025.0, cp=3993.0)
    print(f"  单个棱柱 OHC = {ohc:.4e} J")


    z_levels = np.array([0.0, -20.0, -50.0, -100.0, -150.0, -200.0])
    t_profile = np.array([28.0, 27.0, 25.0, 22.0, 18.0, 10.0])
    d20 = thermocline_depth_from_profile(z_levels, t_profile, t_crit=20.0)
    print(f"  20°C 温跃层深度 D_20 = {d20:.1f} m")


    nx, ny = 10, 5
    lon_grid = np.linspace(120.0, 280.0, nx)
    lat_grid = np.linspace(-5.0, 5.0, ny)
    d20_field = np.ones((nx, ny)) * 120.0
    clim_depth = np.ones((nx, ny)) * 100.0
    wwv = warm_water_volume(d20_field, lon_grid, lat_grid, clim_depth,
                            dx=2.0e5, dy=2.0e5)
    print(f"  暖水体积 WWV = {wwv:.4e} m³")


def demo_equatorial_waves():
    section_header("5. 赤道 Kelvin-Rossby 波 ETDRK4 谱求解 (equatorial_wave_solver)")

    nx, nt = 128, 20
    vis = 0.03
    tmax = 3.0

    x, tt, uu = solve_coupled_wave_envelope(nx, nt, vis, tmax, coupling_strength=0.8)
    print(f"  耦合波包络求解完成: nx={nx}, nt={nt}, tmax={tmax}")
    print(f"  空间域: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  最终时刻波场最大值 = {np.max(uu[:, -1]):.4f}, 最小值 = {np.min(uu[:, -1]):.4f}")


    dx = x[1] - x[0]
    energy = wave_energy(uu, dx)
    print(f"  初始能量 = {energy[0]:.4f}, 最终能量 = {energy[-1]:.4f}")


    c_k = 2.5
    c_r = -0.8
    basin_width = 1.5e7
    tau_r = recharge_discharge_timescale(c_k, c_r, basin_width)
    print(f"  太平洋 basin 充放电时间尺度 τ_R = {tau_r / (365.25 * 24 * 3600):.2f} 年")


def demo_recharge_discharge():
    section_header("6. Recharge-Discharge Oscillator (recharge_discharge_oscillator)")

    t, u = solve_rdo(years=25.0, n_steps=25000,
                     h_w0=0.5, t_e0=0.3,
                     r=0.25, alpha=0.5, R=1.0, epsilon=0.3, gamma=0.4,
                     seasonal_amp=0.15)

    print(f"  RDO 积分完成: {t[-1]:.1f} 年, {len(t)} 步")
    print(f"  最终状态: h_W = {u[-1]:.4f}, T_E = {u[-1]:.4f}")


    trivial, nontrivial = find_equilibrium(r=0.25, alpha=0.5, R=1.0, epsilon=0.3, gamma=0.4)
    print(f"  平凡平衡点: h_W={trivial[0]:.4f}, T_E={trivial[1]:.4f}")
    if nontrivial is not None:
        print(f"  非平凡平衡点: h_W={nontrivial[0, 0]:.4f}, T_E={nontrivial[0, 1]:.4f}")


    T_osc = oscillation_period_approx(r=0.25, alpha=0.5, R=1.0, epsilon=0.3, gamma=0.4)
    print(f"  解析近似振荡周期 = {T_osc:.2f} 年")


    dyn_type = classify_dynamics(r=0.25, alpha=0.5, R=1.0, epsilon=0.3, gamma=0.4)
    print(f"  动力学分类: {dyn_type}")


def demo_chaos_analysis():
    section_header("7. 海气耦合系统混沌分析 (chaos_analysis)")


    lyap = enso_lyapunov_exponent(r=0.25, alpha=0.5, R=1.0, epsilon=0.3, gamma=0.4)
    print(f"  ENSO Poincare 映射 Lyapunov 指数 λ = {lyap:.6f}")
    if lyap > 0:
        print(f"  (λ > 0, 系统在该参数区表现出混沌行为)")
    else:
        print(f"  (λ < 0, 系统趋于稳定周期轨道)")


    points_levy = levy_dragon_ifs(n_iter=5000)
    print(f"  Levy Dragon IFS 生成: {points_levy.shape[0]} 个点")
    print(f"    覆盖范围: x∈[{points_levy[:, 0].min():.3f}, {points_levy[:, 0].max():.3f}], "
          f"y∈[{points_levy[:, 1].min():.3f}, {points_levy[:, 1].max():.3f}]")


    points_cross = cross_chaos_ifs(n_iter=5000)
    print(f"  Cross Chaos IFS 生成: {points_cross.shape[0]} 个点")


    t_rdo, u_rdo = solve_rdo(years=30.0, n_steps=30000, seasonal_amp=0.0)
    d2 = correlation_dimension(u_rdo, r_min=1e-3, r_max=2.0)
    print(f"  RDO 轨迹关联维数 D2 ≈ {d2:.3f}")


    gammas = np.linspace(0.2, 0.8, 30)
    params, attractors = bifurcation_diagram("gamma", gammas)
    print(f"  分岔图扫描完成: {len(params)} 个参数点")


def demo_enso_prediction():
    section_header("8. ENSO 概率预测与不确定性量化 (enso_predictor)")


    nino34_current = 1.2
    month_current = 10
    current_state = classify_enso_state(nino34_current)
    print(f"  当前 Niño 3.4 = {nino34_current:.2f}°C, 状态 = {state_name(current_state)}")


    probs = transition_probability(current_state, month_current)
    print(f"  下月状态转移概率:")
    labels = ["Strong Nina", "Weak Nina", "Neutral", "Weak Nino", "Strong Nino"]
    for label, p in zip(labels, probs):
        print(f"    {label:15s}: {p:.4f}")


    forecast = monte_carlo_enso_forecast(nino34_current, month_current,
                                         n_ensemble=500, n_months=12,
                                         noise_std=0.35)
    print(f"  12 个月集合预测 (N=500):")
    for m in [3, 6, 12]:
        if m <= len(forecast["mean_trajectory"]):
            print(f"    {m}个月均值 = {forecast['mean_trajectory'][m-1]:.3f} ± "
                  f"{forecast['std_trajectory'][m-1]:.3f} °C")


    p_nino = probabilistic_event_forecast(nino34_current, month_current,
                                          ENSOState.WEAK_NINO, lead_months=6, n_trials=5000)
    p_nina = probabilistic_event_forecast(nino34_current, month_current,
                                          ENSOState.WEAK_NINA, lead_months=6, n_trials=5000)
    print(f"  6 个月后弱 El Nino 概率 = {p_nino:.3f}")
    print(f"  6 个月后弱 La Nina 概率 = {p_nina:.3f}")


    obs = forecast["mean_trajectory"] + np.random.normal(0, 0.2, len(forecast["mean_trajectory"]))
    skill = forecast_skill(forecast["mean_trajectory"], obs)
    print(f"  预测 skill: RMSE={skill['rmse']:.3f}, 相关系数={skill['correlation']:.3f}")


def demo_multiscale():
    section_header("9. ENSO 多尺度小波分析 (multiscale_analysis)")


    t_months = np.arange(0, 480)
    nino34_synth = (
        1.2 * np.sin(2.0 * np.pi * t_months / 48.0)
        + 0.3 * np.sin(2.0 * np.pi * t_months / 12.0)
        + np.random.normal(0, 0.3, len(t_months))
    )

    result = nested_multiscale_analysis(nino34_synth, dt=1.0, n_levels=6)
    print(f"  合成 Niño 3.4 序列分析 (40 年月数据):")
    print(f"    AR(1) 系数 ρ = {result['ar1_rho']:.4f}")
    print(f"    显著周期峰:")
    for peak in result['significant_peaks'][:5]:
        print(f"      周期 = {peak['period_months']:.1f} 月, 尺度 = {peak['scale']:.2f}, "
              f"功率 = {peak['power']:.4e}")


def demo_quadrature():
    section_header("10. 高维求积规则精确度验证 (quadrature_validation)")


    result_h = validate_hermite_quadrature_1d(n_points=8, degree_max=15)
    print(f"  1D Gauss-Hermite (n=8): 最大精确阶数 = {result_h['max_degree_passed']}")


    result_c = validate_hypercube_quadrature(dim=2, n_points=5, degree_max=9)
    print(f"  2D Gauss-Legendre (n=5 per dim): 总点数 = {result_c['total_points']}, "
          f"最大精确阶数 = {result_c['max_degree_passed']}")


    quad_pts = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    quad_wts = np.array([0.25, 0.25, 0.25, 0.25])
    ensemble = np.random.randn(100, 4)
    mu, sigma_sq = ensemble_mean_integral(ensemble, quad_pts, quad_wts)
    print(f"  集合均值积分: μ = {mu:.4f}, σ² = {sigma_sq:.4f}")


def demo_parameter_inversion():
    section_header("11. 热扩散参数反演 (parameter_inversion)")

    nx, ny = 15, 10
    dx, dy = 2.0e5, 2.0e5


    D_true, tau_true, coupling_true = 2000.0, 30.0 * 24 * 3600, 0.02


    heat_source = np.ones((nx, ny)) * 50.0
    T_obs = solve_steady_heat_2d(nx, ny, dx, dy, D_true, tau_true,
                                 heat_source + coupling_true * np.ones((nx, ny)))


    theta_init = np.array([1000.0, 15.0 * 24 * 3600, 0.01])
    theta_prior = np.array([1500.0, 20.0 * 24 * 3600, 0.015])
    theta_opt, history = gradient_descent_inversion(
        T_obs, nx, ny, dx, dy, heat_source,
        theta_init, theta_prior, lr=0.5, n_iter=80, lam=0.01
    )

    print(f"  真实参数: D_h = {D_true:.1f} m²/s, τ = {tau_true/(24*3600):.1f} days, coupling = {coupling_true:.4f}")
    print(f"  反演结果: D_h = {theta_opt[0]:.1f} m²/s, τ = {theta_opt[1]/(24*3600):.1f} days, coupling = {theta_opt[2]:.4f}")
    print(f"  最终目标函数 J = {history[-1]:.4e}")


    sens = sensitivity_analysis(theta_opt, T_obs, nx, ny, dx, dy, heat_source)
    print(f"  参数灵敏度:")
    for k, v in sens.items():
        print(f"    {k:15s}: {v:.4f}")


def demo_io():
    section_header("12. 海洋模式数据读写 (io_utils)")


    n_nodes = 100
    n_elements = 50
    node_coords = np.random.rand(2, n_nodes) * 1.0e6
    node_data = np.random.rand(3, n_nodes)
    element_nodes = np.random.randint(0, n_nodes, size=(3, n_elements))

    filename = "/tmp/enso_test_mesh.dat"
    write_mesh_data(filename, "ENSO_Test",
                    ["X", "Y", "SST", "h_anom", "u_zonal"],
                    node_coords, node_data, element_nodes)


    data = read_mesh_data(filename)
    print(f"  写入/读取网格数据测试通过:")
    print(f"    标题: {data['title']}")
    print(f"    节点数: {data['n_nodes']}, 单元数: {data['n_elements']}")


    times = np.arange(0, 24)
    nino = 0.5 * np.sin(2.0 * np.pi * times / 12.0) + np.random.normal(0, 0.1, 24)
    ts_file = "/tmp/enso_nino34.txt"
    write_sst_timeseries(ts_file, times, nino,
                         metadata={"model": "ENSO_v1", "region": "Nino34"})
    t_read, nino_read, meta = read_sst_timeseries(ts_file)
    print(f"  时间序列读写测试通过: {len(t_read)} 个时间点")
    print(f"  元数据: {meta}")


    cs = compute_checksum(nino)
    print(f"  数据校验和 = {cs}")


    try:
        os.remove(filename)
        os.remove(ts_file)
    except OSError:
        pass


def main():
    print("\n")
    print("*" * 70)
    print("*  海气耦合模式与 ENSO 预测系统 — 博士级科研计算综合演示")
    print("*  科学领域: 海洋科学 — 海气耦合模式与 ENSO 预测")
    print("*" * 70)
    print("\n  本系统整合 15 个种子项目的核心算法，涵盖:")
    print("    - 赤道波动动力学 (ETDRK4 谱方法)")
    print("    - Recharge-Discharge Oscillator (非线性 ODE)")
    print("    - 三维海洋热含量计算 (棱柱体高斯求积)")
    print("    - 海表温度空间插值 (二维 Lagrange 张量积)")
    print("    - 热力学参数反演 (梯度下降 + 正则化)")
    print("    - 稀疏矩阵存储 (CRS/ST 格式)")
    print("    - 混沌与分岔分析 (Lyapunov 指数 + IFS)")
    print("    - 概率集合预测 (蒙特卡洛)")
    print("    - 多尺度小波分析 (Morlet CWT)")
    print("    - 高维求积验证 (Gauss-Legendre/Hermite)")
    print("    - 日历与季节锁相 (儒略日)")
    print("    - 数据 I/O (结构化网格格式)")
    print("\n  运行环境: Python 3.x + NumPy")
    print("  零参数直接运行，无需外部数据。")


    demo_calendar_utils()
    demo_sparse_matrix()
    demo_sst_interpolation()
    demo_ocean_heat_content()
    demo_equatorial_waves()
    demo_recharge_discharge()
    demo_chaos_analysis()
    demo_enso_prediction()
    demo_multiscale()
    demo_quadrature()
    demo_parameter_inversion()
    demo_io()

    print("\n" + "=" * 70)
    print("  所有模块运行完毕，系统正常。")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
