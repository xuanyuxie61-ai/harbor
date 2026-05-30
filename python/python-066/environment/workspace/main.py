
import numpy as np
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh_topology import Mesh1D, Mesh2D, parse_well_locations
from stochastic_field import lognormal_k_field, quasirandom_k_parameters, rejection_sample_1d
from fem_transport import FEMTransportSolver1D, FlowSolver1D
from mrmt_generator import MRMTModel, polynomial_multiply
from travel_time_tree import TravelTimeTree, discrete_dynamical_stability_map
from inverse_estimator import InverseEstimator, sherman_morrison_solve
from uncertainty_engine import MonteCarloEngine, convergence_analysis
from fractal_media import FractalPorousMedia, sample_uniform_hexagon, hexagon_grid
from interpolation_engine import ConcentrationInterpolator, cubic_spline_natural
from quadrature_engine import integrate_decay_convolution, integrate_2d_rectangle, test_quadrature_exactness
from data_parser import parse_numeric_table, serialize_time_series, format_well_report
from privacy_encoder import encode_well_id, decode_well_id, encode_coordinate_list


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.random.seed(42)
    print("\n")
    print("*" * 70)
    print("*  HydroGeoSim-66: 博士级地下水溶质运移综合模拟平台")
    print("*  科学领域: 水文地质 — 地下水流与溶质运移模拟")
    print("*" * 70)




    print_section("1. 计算网格与监测井设置")
    L_domain = 100.0
    nx = 80
    mesh = Mesh1D(0.0, L_domain, nx)
    print(f"   一维网格: {mesh.n_nodes} 节点, {mesh.n_elements} 单元")
    print(f"   网格质量: h_min={mesh.quality_report()['h_min']:.3f}m, "
          f"h_max={mesh.quality_report()['h_max']:.3f}m")


    well_positions = [10.0, 30.0, 55.0, 80.0]
    well_ids = [f"MW-{i+1:02d}" for i in range(len(well_positions))]
    print(f"   监测井: {well_ids}")
    print(f"   位置 (m): {well_positions}")


    mesh2d = Mesh2D((0.0, L_domain), (0.0, 20.0), 20, 4, triangular=True)
    print(f"   二维辅助网格: {mesh2d.n_nodes} 节点, {mesh2d.n_elements} 单元")




    print_section("2. 随机水力传导度场生成")
    mu_y = -2.0
    sigma_y = 1.0
    corr_len = 10.0
    K_field = lognormal_k_field(mesh.nodes, mu_y, sigma_y, corr_len, seed=42)
    print(f"   K 场统计: 均值={np.mean(K_field):.4e},  std={np.std(K_field):.4e}")
    print(f"   K 范围: [{np.min(K_field):.4e}, {np.max(K_field):.4e}] m/d")


    param_samples = quasirandom_k_parameters(8, dim=3)
    print(f"   QMC 参数扫描: 生成 {len(param_samples)} 组 (μ, σ, λ)")


    from math import log, exp, pi, sqrt
    def lognormal_pdf(k):
        if k <= 0:
            return 0.0
        return (1.0 / (k * sigma_y * sqrt(2.0 * pi))) * \
               exp(-0.5 * ((log(k) - mu_y) / sigma_y) ** 2)
    k_samples = rejection_sample_1d(lognormal_pdf, 3.0, 1e-4, 2.0, 200, seed=42)
    print(f"   接受-拒绝采样: {len(k_samples)} 个 K 样本, "
          f"均值={np.mean(k_samples):.4e}")




    print_section("3. 稳态地下水流动求解")
    h_left = 15.0
    h_right = 10.0
    porosity = 0.25
    flow_solver = FlowSolver1D(mesh.nodes)
    h_field = flow_solver.solve_steady(K_field, h_left, h_right)
    v_field = flow_solver.compute_velocity(K_field, h_field, porosity)
    print(f"   水头范围: [{np.min(h_field):.3f}, {np.max(h_field):.3f}] m")
    print(f"   达西流速范围: [{np.min(v_field):.4e}, {np.max(v_field):.4e}] m/d")
    print(f"   平均流速: {np.mean(v_field):.4e} m/d")




    print_section("4. 对流-弥散-反应方程隐式 FEM 求解")
    D_disp = 0.5
    R_retard = 1.5
    lambda_decay = 0.005
    t_end = 200.0
    dt = 2.0


    C0 = np.zeros(mesh.n_nodes)
    C0[mesh.nodes <= 5.0] = 1.0


    v_mean = float(np.mean(np.abs(v_field)))
    transport_solver = FEMTransportSolver1D(mesh.nodes, theta=1.0)
    t_hist, C_hist = transport_solver.solve_transient(
        C0, t_end, dt,
        D=D_disp, v=v_mean, R=R_retard, lam=lambda_decay,
        bc_nodes=[0, mesh.n_nodes - 1], bc_values=[0.0, 0.0],
        verbose=False
    )
    print(f"   时间步数: {len(t_hist)-1}, 最终 t={t_hist[-1]:.1f} d")
    print(f"   初始总质量: {transport_solver.compute_mass_balance(C0):.6f}")
    print(f"   最终总质量: {transport_solver.compute_mass_balance(C_hist[-1]):.6f}")
    print(f"   质量守恒误差: {abs(transport_solver.compute_mass_balance(C_hist[-1]) - transport_solver.compute_mass_balance(C0)):.6e}")




    print_section("5. 多速率质量转移 (MRMT) 模型")
    alphas = np.array([0.01, 0.05, 0.2, 1.0])
    betas = np.array([0.4, 0.3, 0.2, 0.1])
    mrmt = MRMTModel(alphas, betas, R_m=1.0)
    C_mobile = C_hist[-1, :]

    C_m_tseries = np.mean(C_hist, axis=1)
    S_immobile = mrmt.compute_immobile_concentration(C_m_tseries, dt)
    print(f"   MRMT 速率数: {len(alphas)}")
    print(f"   流动区最终平均浓度: {C_m_tseries[-1]:.6f}")
    print(f"   不动区最终平均浓度: {S_immobile[-1]:.6f}")
    print(f"   等效滞留因子 (s→0): {mrmt.effective_retardation(0.0):.3f}")
    print(f"   等效滞留因子 (s→∞): {mrmt.effective_retardation(100.0):.3f}")


    layer_response = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
    total_response = polynomial_multiply(layer_response, layer_response)
    print(f"   生成函数卷积: 单层响应长度 {len(layer_response)}, "
          f"叠加后长度 {len(total_response)}")




    print_section("6. 随机行走粒子追踪与运移时间分布")
    x_obs = 55.0
    tree = TravelTimeTree(x0=x_obs, t0=0.0, v_func=lambda x: v_mean,
                          D=D_disp, dt=2.0)
    btree = tree.build_backward_tree(max_levels=5, n_branches=2)
    print(f"   反向追踪树深度: {len(btree['levels'])} 层")
    for level_idx, level_nodes in enumerate(btree["levels"]):
        print(f"     层级 {level_idx}: {len(level_nodes)} 个可能源区节点")

    ttd = tree.compute_travel_time_distribution(
        x_source=0.0, n_particles=2000, max_steps=300,
        x_bounds=(-10.0, L_domain + 10.0)
    )
    reached = ttd[ttd > 0]
    print(f"   粒子追踪: 释放 2000 个粒子, {len(reached)} 个到达观测点")
    if len(reached) > 0:
        print(f"   平均运移时间: {np.mean(reached):.2f} d")
        print(f"   运移时间标准差: {np.std(reached):.2f} d")


    x_grid = np.linspace(0, L_domain, 50)
    lyap = discrete_dynamical_stability_map(lambda x: v_mean, x_grid, dt=2.0)
    print(f"   Lyapunov 指数范围: [{np.nanmin(lyap):.4f}, {np.nanmax(lyap):.4f}]")




    print_section("7. 非线性最小二乘参数反演")

    true_params = np.array([D_disp, v_mean, lambda_decay])
    param_names = ["D_disp", "v_mean", "lambda_decay"]


    obs_wells_idx = [int(p / L_domain * nx) for p in well_positions]
    obs_wells_idx = [min(i, nx) for i in obs_wells_idx]

    def forward_model(beta):







        raise NotImplementedError("Hole 2: 请实现正演模型 forward_model")

    C_observed = forward_model(true_params)

    noise = np.random.default_rng(42).normal(0, 0.01, size=len(C_observed))
    C_observed = np.maximum(C_observed + noise, 0.0)

    def residual(beta):
        return forward_model(beta) - C_observed

    estimator = InverseEstimator(
        residual, param_names,
        param_lower_bounds=np.array([0.01, 0.01, 0.0001]),
        param_upper_bounds=np.array([5.0, 2.0, 0.05])
    )
    result_lm = estimator.solve_lm(
        beta0=np.array([0.3, 0.3, 0.001]),
        max_iter=30, tol=1e-4
    )
    print(f"   真实参数: D={true_params[0]:.4f}, v={true_params[1]:.4f}, λ={true_params[2]:.4f}")
    print(f"   反演结果: D={result_lm['beta_opt'][0]:.4f}, "
          f"v={result_lm['beta_opt'][1]:.4f}, λ={result_lm['beta_opt'][2]:.4f}")
    print(f"   最终 RMS: {result_lm['final_rms']:.6e}")
    print(f"   迭代次数: {result_lm['n_iter']}")


    A_test = np.eye(3) * 2.0 + 0.1
    b_test = np.array([1.0, 2.0, 3.0])
    x_base = np.linalg.solve(A_test, b_test)
    u = np.array([0.1, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    A_inv_u = np.linalg.solve(A_test, u)
    x_sm = sherman_morrison_solve(A_inv_u, v, x_base, alpha=1.0)
    x_direct = np.linalg.solve(A_test + np.outer(u, v), b_test)
    sm_error = np.linalg.norm(x_sm - x_direct)
    print(f"   Sherman-Morrison 验证误差: {sm_error:.2e}")




    print_section("8. 不确定性量化 (Monte Carlo)")

    def transport_model_uq(beta):
        D, v, lam = beta
        if D <= 0 or v < 0 or lam < 0:
            return 0.0
        try:
            ts, Cs = transport_solver.solve_transient(
                C0, t_end, dt, D, v, R_retard, lam,
                bc_nodes=[0, nx], bc_values=[0.0, 0.0]
            )

            idx = int(55.0 / L_domain * nx)
            idx = min(idx, nx)
            return float(np.mean(Cs[-5:, idx]))
        except Exception:
            return 0.0

    def param_sampler_uq(u):

        return np.array([
            0.1 + u[0] * 0.9,
            0.1 + u[1] * 0.9,
            0.001 + u[2] * 0.009
        ])

    mc_engine = MonteCarloEngine(transport_model_uq, param_sampler_uq, n_params=3)
    mc_result = mc_engine.run_mc(N=64, seed=42)
    print(f"   MC 样本数: 64")
    print(f"   平均浓度: {mc_result['mean']:.6f}")
    print(f"   浓度标准差: {mc_result['std']:.6f}")
    print(f"   浓度范围: [{mc_result['min']:.6f}, {mc_result['max']:.6f}]")

    exceed = mc_engine.estimate_exceedance_probability(
        mc_result["samples_output"], threshold=0.01)
    print(f"   超标概率 (C > 0.01): {exceed['exceedance_probability']:.4f} "
          f"± {exceed['standard_error']:.4f}")


    conv = convergence_analysis(transport_model_uq, param_sampler_uq, 3,
                                sample_sizes=[16, 32, 64])
    print(f"   收敛分析 (均值随样本量):")
    for item in conv["convergence"]:
        print(f"     N={item['N']:3d}: mean={item['mean']:.6f}, SE={item['std_err']:.6f}")




    print_section("9. 分形多孔介质表征")
    fpm = FractalPorousMedia(n_iterations=3, n_points=2000)
    K_frac = fpm.generate_sierpinski_carpet_permeability(grid_res=81)
    print(f"   Sierpinski carpet 渗透率场: {K_frac.shape[0]}×{K_frac.shape[1]} 网格")
    print(f"   高渗透通道占比: {np.mean(K_frac > 0.5):.4f}")
    print(f"   分形维数 (理论): log(8)/log(3) ≈ 1.893")

    pts = fpm.generate_ifs_attractor(n_points=2000)
    D_est = fpm.fractal_dimension_boxcount(pts, n_boxes=15)
    print(f"   IFS 吸引子估计分形维数: {D_est:.3f}")

    hex_samples = sample_uniform_hexagon(500, radius=5.0, seed=42)
    print(f"   六边形均匀采样: {len(hex_samples)} 个点")
    hcoords, helems = hexagon_grid(4, 3, radius=2.0)
    print(f"   六边形网格: {len(hcoords)} 个节点, {len(helems)} 个单元")




    print_section("10. 浓度插值与突破曲线重建")
    interp = ConcentrationInterpolator(mesh.nodes, C_hist[0, :], max_order=6)
    C_at_wells = []
    for wp in well_positions:
        c = interp.interpolate(wp)
        C_at_wells.append(c)
    print(f"   监测井插值浓度 (t=0): {['%.4f' % c for c in C_at_wells]}")


    btc_55 = interp.reconstruct_breakthrough_curve(
        x_obs=55.0, C_history=C_hist, t_values=t_hist
    )
    print(f"   55m 处突破曲线: 峰值浓度={np.max(btc_55):.6f}, "
          f"到达时间={t_hist[np.argmax(btc_55)]:.1f} d")


    if len(t_hist) > 3:
        btc_smooth = cubic_spline_natural(t_hist, btc_55, t_hist)
        print(f"   样条平滑后峰值: {np.max(btc_smooth):.6f}")




    print_section("11. 高精度数值积分验证")
    quad_tests = test_quadrature_exactness()
    print(f"   Gauss-Laguerre 精确性测试: {'通过' if quad_tests['laguerre_exactness'] else '失败'}")
    print(f"   2D Gauss-Legendre 精确性测试: {'通过' if quad_tests['legendre2d_exactness'] else '失败'}")


    conv_result = integrate_decay_convolution(C_m_tseries, dt, lam=lambda_decay, n_quad=32)
    print(f"   衰减卷积积分 (λ={lambda_decay}): {conv_result:.6f}")


    def f_test2d(x, y):
        return np.exp(-0.01 * (x + y))
    int2d = integrate_2d_rectangle(f_test2d, (0.0, L_domain), (0.0, 20.0), nx=8, ny=8)
    print(f"   二维指数积分: {int2d:.4f}")




    print_section("12. 数据解析与格式化")
    well_data = []
    for wid, wp in zip(well_ids, well_positions):
        well_data.append({
            "well_id": wid,
            "x": wp,
            "y": 0.0,
            "depth": 30.0,
            "head": float(np.interp(wp, mesh.nodes, h_field)),
            "concentration": float(np.interp(wp, mesh.nodes, C_hist[-1, :]))
        })
    report = format_well_report(well_data)
    print(report)


    ts_text = serialize_time_series(t_hist, btc_55,
                                     variable_name="breakthrough_curve_55m",
                                     unit="mg/L")
    parsed_ts = parse_numeric_table(ts_text.split('\n'))
    print(f"   时间序列解析: {len(parsed_ts)} 行数据")




    print_section("13. 监测井数据隐私编码")
    encoded_wells = encode_coordinate_list(
        [(w["x"], w["y"]) for w in well_data],
        [w["well_id"] for w in well_data]
    )
    print(f"   原始井号 -> 编码井号:")
    for rec in encoded_wells["encoded_wells"]:
        dec = decode_well_id(rec["encoded_id"])
        print(f"     {rec['original_id']:8s} -> {rec['encoded_id']:8s} (解码验证: {dec})")




    print("\n" + "*" * 70)
    print("*  HydroGeoSim-66 综合模拟完成")
    print("*  所有模块运行正常，无报错")
    print("*" * 70 + "\n")


if __name__ == "__main__":
    main()
