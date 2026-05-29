"""
main.py
================================================================================
HydroGeoSim-66: 地下水溶质运移综合模拟平台
================================================================================
科学领域：水文地质 — 地下水流与溶质运移模拟

本项目融合 15 个种子项目的核心算法，构建了一个面向前沿科学问题的
博士级地下水溶质运移模拟系统。主要包含：

  1. 随机水力传导度场生成（低差异序列 + 接受-拒绝采样）
  2. 一维稳态地下水流动求解
  3. 对流-弥散-反应方程的隐式有限元求解
  4. 多速率质量转移（MRMT）模型
  5. 随机行走粒子追踪与运移时间分布
  6. 非线性最小二乘参数反演（Gauss-Newton / Levenberg-Marquardt）
  7. 不确定性量化（蒙特卡洛 / 拟蒙特卡洛）
  8. 分形多孔介质表征
  9. 高精度浓度插值与突破曲线重建
 10. 监测数据隐私编码

运行方式：
    python main.py
（无需任何命令行参数，零配置可运行）
================================================================================
"""

import numpy as np
import sys
import os

# 将当前目录加入路径以导入自定义模块
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

    # ========================================================================
    # 1. 网格生成与监测井设置（基于 749_medit_to_ice）
    # ========================================================================
    print_section("1. 计算网格与监测井设置")
    L_domain = 100.0  # m
    nx = 80
    mesh = Mesh1D(0.0, L_domain, nx)
    print(f"   一维网格: {mesh.n_nodes} 节点, {mesh.n_elements} 单元")
    print(f"   网格质量: h_min={mesh.quality_report()['h_min']:.3f}m, "
          f"h_max={mesh.quality_report()['h_max']:.3f}m")

    # 监测井位置
    well_positions = [10.0, 30.0, 55.0, 80.0]
    well_ids = [f"MW-{i+1:02d}" for i in range(len(well_positions))]
    print(f"   监测井: {well_ids}")
    print(f"   位置 (m): {well_positions}")

    # 2D 网格用于二维积分测试
    mesh2d = Mesh2D((0.0, L_domain), (0.0, 20.0), 20, 4, triangular=True)
    print(f"   二维辅助网格: {mesh2d.n_nodes} 节点, {mesh2d.n_elements} 单元")

    # ========================================================================
    # 2. 随机水力传导度场（基于 1021_rejection_sample + 803_niederreiter2）
    # ========================================================================
    print_section("2. 随机水力传导度场生成")
    mu_y = -2.0      # ln(K) 均值 => K_geo ≈ 0.135 m/d
    sigma_y = 1.0    # ln(K) 标准差
    corr_len = 10.0  # 相关长度 (m)
    K_field = lognormal_k_field(mesh.nodes, mu_y, sigma_y, corr_len, seed=42)
    print(f"   K 场统计: 均值={np.mean(K_field):.4e},  std={np.std(K_field):.4e}")
    print(f"   K 范围: [{np.min(K_field):.4e}, {np.max(K_field):.4e}] m/d")

    # 使用 Niederreiter 低差异序列扫描参数空间
    param_samples = quasirandom_k_parameters(8, dim=3)
    print(f"   QMC 参数扫描: 生成 {len(param_samples)} 组 (μ, σ, λ)")

    # 接受-拒绝采样：从对数正态分布采样渗透率值
    from math import log, exp, pi, sqrt
    def lognormal_pdf(k):
        if k <= 0:
            return 0.0
        return (1.0 / (k * sigma_y * sqrt(2.0 * pi))) * \
               exp(-0.5 * ((log(k) - mu_y) / sigma_y) ** 2)
    k_samples = rejection_sample_1d(lognormal_pdf, 3.0, 1e-4, 2.0, 200, seed=42)
    print(f"   接受-拒绝采样: {len(k_samples)} 个 K 样本, "
          f"均值={np.mean(k_samples):.4e}")

    # ========================================================================
    # 3. 稳态地下水流动求解（基于 391_fem1d_heat_implicit）
    # ========================================================================
    print_section("3. 稳态地下水流动求解")
    h_left = 15.0   # m
    h_right = 10.0  # m
    porosity = 0.25
    flow_solver = FlowSolver1D(mesh.nodes)
    h_field = flow_solver.solve_steady(K_field, h_left, h_right)
    v_field = flow_solver.compute_velocity(K_field, h_field, porosity)
    print(f"   水头范围: [{np.min(h_field):.3f}, {np.max(h_field):.3f}] m")
    print(f"   达西流速范围: [{np.min(v_field):.4e}, {np.max(v_field):.4e}] m/d")
    print(f"   平均流速: {np.mean(v_field):.4e} m/d")

    # ========================================================================
    # 4. 溶质运移方程求解（基于 391_fem1d_heat_implicit）
    # ========================================================================
    print_section("4. 对流-弥散-反应方程隐式 FEM 求解")
    D_disp = 0.5     # 弥散系数 (m²/d)
    R_retard = 1.5   # 滞留因子
    lambda_decay = 0.005  # 一级衰变速率 (1/d)
    t_end = 200.0    # d
    dt = 2.0         # d

    # 初始条件：源区（0-5m）浓度为 1.0
    C0 = np.zeros(mesh.n_nodes)
    C0[mesh.nodes <= 5.0] = 1.0

    # 使用平均流速进行确定性求解
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

    # ========================================================================
    # 5. 多速率质量转移 MRMT（基于 158_change_polynomial）
    # ========================================================================
    print_section("5. 多速率质量转移 (MRMT) 模型")
    alphas = np.array([0.01, 0.05, 0.2, 1.0])  # 1/d
    betas = np.array([0.4, 0.3, 0.2, 0.1])
    mrmt = MRMTModel(alphas, betas, R_m=1.0)
    C_mobile = C_hist[-1, :]  # 取最终时刻作为稳态近似
    # 将节点浓度近似为时间序列（简化：假设空间均匀衰变）
    C_m_tseries = np.mean(C_hist, axis=1)  # 空间平均浓度随时间变化
    S_immobile = mrmt.compute_immobile_concentration(C_m_tseries, dt)
    print(f"   MRMT 速率数: {len(alphas)}")
    print(f"   流动区最终平均浓度: {C_m_tseries[-1]:.6f}")
    print(f"   不动区最终平均浓度: {S_immobile[-1]:.6f}")
    print(f"   等效滞留因子 (s→0): {mrmt.effective_retardation(0.0):.3f}")
    print(f"   等效滞留因子 (s→∞): {mrmt.effective_retardation(100.0):.3f}")

    # 生成函数卷积：模拟多层响应叠加
    layer_response = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
    total_response = polynomial_multiply(layer_response, layer_response)
    print(f"   生成函数卷积: 单层响应长度 {len(layer_response)}, "
          f"叠加后长度 {len(total_response)}")

    # ========================================================================
    # 6. 粒子追踪与运移时间分布（基于 196_collatz）
    # ========================================================================
    print_section("6. 随机行走粒子追踪与运移时间分布")
    x_obs = 55.0  # 观测点
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

    # 离散动力系统稳定性分析
    x_grid = np.linspace(0, L_domain, 50)
    lyap = discrete_dynamical_stability_map(lambda x: v_mean, x_grid, dt=2.0)
    print(f"   Lyapunov 指数范围: [{np.nanmin(lyap):.4f}, {np.nanmax(lyap):.4f}]")

    # ========================================================================
    # 7. 参数反演估计（基于 1220_test_nls + 995_r8sm）
    # ========================================================================
    print_section("7. 非线性最小二乘参数反演")
    # 合成观测数据：在已知参数下运行模型，然后反推
    true_params = np.array([D_disp, v_mean, lambda_decay])  # D, v, λ
    param_names = ["D_disp", "v_mean", "lambda_decay"]

    # 在几个观测井位置生成合成观测
    obs_wells_idx = [int(p / L_domain * nx) for p in well_positions]
    obs_wells_idx = [min(i, nx) for i in obs_wells_idx]

    def forward_model(beta):
        D, v, lam = beta
        if D <= 0 or v < 0 or lam < 0:
            return np.full(len(obs_wells_idx), 1e6)
        try:
            ts, Cs = transport_solver.solve_transient(
                C0, t_end, dt, D, v, R_retard, lam,
                bc_nodes=[0, nx], bc_values=[0.0, 0.0]
            )
            # 取稳态近似（最后几个时间步平均）
            C_final = np.mean(Cs[-5:, :], axis=0)
            return C_final[obs_wells_idx]
        except Exception:
            return np.full(len(obs_wells_idx), 1e6)

    C_observed = forward_model(true_params)
    # 添加观测噪声
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

    # Sherman-Morrison 快速重解测试
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

    # ========================================================================
    # 8. 不确定性量化（基于 189_clock_solitaire_simulation）
    # ========================================================================
    print_section("8. 不确定性量化 (Monte Carlo)")

    def transport_model_uq(beta):
        """UQ 包装器：beta = [D, v, lambda]"""
        D, v, lam = beta
        if D <= 0 or v < 0 or lam < 0:
            return 0.0
        try:
            ts, Cs = transport_solver.solve_transient(
                C0, t_end, dt, D, v, R_retard, lam,
                bc_nodes=[0, nx], bc_values=[0.0, 0.0]
            )
            # 输出：观测井 55m 处的最终浓度
            idx = int(55.0 / L_domain * nx)
            idx = min(idx, nx)
            return float(np.mean(Cs[-5:, idx]))
        except Exception:
            return 0.0

    def param_sampler_uq(u):
        # D ~ U(0.1, 1.0), v ~ U(0.1, 1.0), λ ~ U(0.001, 0.01)
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

    # 收敛分析
    conv = convergence_analysis(transport_model_uq, param_sampler_uq, 3,
                                sample_sizes=[16, 32, 64])
    print(f"   收敛分析 (均值随样本量):")
    for item in conv["convergence"]:
        print(f"     N={item['N']:3d}: mean={item['mean']:.6f}, SE={item['std_err']:.6f}")

    # ========================================================================
    # 9. 分形多孔介质（基于 526_hexagon_chaos）
    # ========================================================================
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

    # ========================================================================
    # 10. 浓度插值与突破曲线（基于 592_interp_equal）
    # ========================================================================
    print_section("10. 浓度插值与突破曲线重建")
    interp = ConcentrationInterpolator(mesh.nodes, C_hist[0, :], max_order=6)
    C_at_wells = []
    for wp in well_positions:
        c = interp.interpolate(wp)
        C_at_wells.append(c)
    print(f"   监测井插值浓度 (t=0): {['%.4f' % c for c in C_at_wells]}")

    # 重构 55m 处的突破曲线
    btc_55 = interp.reconstruct_breakthrough_curve(
        x_obs=55.0, C_history=C_hist, t_values=t_hist
    )
    print(f"   55m 处突破曲线: 峰值浓度={np.max(btc_55):.6f}, "
          f"到达时间={t_hist[np.argmax(btc_55)]:.1f} d")

    # 三次样条平滑
    if len(t_hist) > 3:
        btc_smooth = cubic_spline_natural(t_hist, btc_55, t_hist)
        print(f"   样条平滑后峰值: {np.max(btc_smooth):.6f}")

    # ========================================================================
    # 11. 高精度数值积分（基于 466_gen_laguerre_exactness + 1143_square_exactness）
    # ========================================================================
    print_section("11. 高精度数值积分验证")
    quad_tests = test_quadrature_exactness()
    print(f"   Gauss-Laguerre 精确性测试: {'通过' if quad_tests['laguerre_exactness'] else '失败'}")
    print(f"   2D Gauss-Legendre 精确性测试: {'通过' if quad_tests['legendre2d_exactness'] else '失败'}")

    # 衰减卷积积分
    conv_result = integrate_decay_convolution(C_m_tseries, dt, lam=lambda_decay, n_quad=32)
    print(f"   衰减卷积积分 (λ={lambda_decay}): {conv_result:.6f}")

    # 二维矩形积分测试
    def f_test2d(x, y):
        return np.exp(-0.01 * (x + y))
    int2d = integrate_2d_rectangle(f_test2d, (0.0, L_domain), (0.0, 20.0), nx=8, ny=8)
    print(f"   二维指数积分: {int2d:.4f}")

    # ========================================================================
    # 12. 数据解析与格式化（基于 1419_xy_display）
    # ========================================================================
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

    # 时间序列序列化
    ts_text = serialize_time_series(t_hist, btc_55,
                                     variable_name="breakthrough_curve_55m",
                                     unit="mg/L")
    parsed_ts = parse_numeric_table(ts_text.split('\n'))
    print(f"   时间序列解析: {len(parsed_ts)} 行数据")

    # ========================================================================
    # 13. 隐私编码（基于 1045_rot13）
    # ========================================================================
    print_section("13. 监测井数据隐私编码")
    encoded_wells = encode_coordinate_list(
        [(w["x"], w["y"]) for w in well_data],
        [w["well_id"] for w in well_data]
    )
    print(f"   原始井号 -> 编码井号:")
    for rec in encoded_wells["encoded_wells"]:
        dec = decode_well_id(rec["encoded_id"])
        print(f"     {rec['original_id']:8s} -> {rec['encoded_id']:8s} (解码验证: {dec})")

    # ========================================================================
    # 完成
    # ========================================================================
    print("\n" + "*" * 70)
    print("*  HydroGeoSim-66 综合模拟完成")
    print("*  所有模块运行正常，无报错")
    print("*" * 70 + "\n")


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: Mesh1D 基本属性与质量报告 ----
mesh1d = Mesh1D(0.0, 100.0, 20)
qr = mesh1d.quality_report()
assert qr['n_nodes'] == 21 and qr['n_elements'] == 20, '[TC01] Mesh1D 基本属性 FAILED'
assert abs(qr['aspect_ratio'] - 1.0) < 1e-10, '[TC01] Mesh1D 均匀网格 aspect_ratio FAILED'

# ---- TC02: Mesh1D refine 后节点数正确 ----
mesh_fine = mesh1d.refine(2)
assert mesh_fine.n_nodes == 41, '[TC02] Mesh1D refine FAILED'

# ---- TC03: Mesh2D 三角形网格总面积等于矩形面积 ----
mesh2d = Mesh2D((0.0, 10.0), (0.0, 5.0), 4, 2, triangular=True)
qr2 = mesh2d.quality_report()
assert abs(qr2['total_area'] - 50.0) < 1e-10, '[TC03] Mesh2D 总面积 FAILED'

# ---- TC04: parse_well_locations 过滤注释与空行 ----
wells = parse_well_locations(["MW-01  10.0  20.0  30.0", "# comment", "", "MW-02  15.0  25.0"])
assert len(wells) == 2, '[TC04] parse_well_locations 数量 FAILED'
assert wells[0]['well_id'] == 'MW-01' and wells[0]['depth'] == 30.0, '[TC04] parse_well_locations 内容 FAILED'

# ---- TC05: lognormal_k_field 形状与正值 ----
x_grid = np.linspace(0, 10, 20)
np.random.seed(42)
K_field = lognormal_k_field(x_grid, mu=-2.0, sigma=1.0, correlation_length=2.0)
assert K_field.shape == x_grid.shape, '[TC05] K_field 形状 FAILED'
assert np.all(K_field > 0), '[TC05] K_field 必须全为正 FAILED'

# ---- TC06: rejection_sample_1d 样本数与范围 ----
def uniform_pdf(x):
    return 0.5 if -1.0 <= x <= 1.0 else 0.0
samples = rejection_sample_1d(uniform_pdf, 0.5, -1.0, 1.0, 100, seed=42)
assert len(samples) == 100, '[TC06] rejection_sample 样本数 FAILED'
assert np.all((samples >= -1.0) & (samples <= 1.0)), '[TC06] rejection_sample 范围 FAILED'

# ---- TC07: NiederreiterGenerator 生成范围在 [0,1] ----
from stochastic_field import NiederreiterGenerator
gen = NiederreiterGenerator(dim=3, seed=0)
pts = gen.generate(50)
assert pts.shape == (50, 3), '[TC07] Niederreiter 形状 FAILED'
assert np.all((pts >= 0) & (pts <= 1)), '[TC07] Niederreiter 范围 FAILED'

# ---- TC08: FlowSolver1D 稳态边界条件精确满足 ----
nodes = np.linspace(0.0, 100.0, 51)
K_const = np.ones_like(nodes)
flow_solver = FlowSolver1D(nodes)
h_field = flow_solver.solve_steady(K_const, h_left=15.0, h_right=10.0)
assert abs(h_field[0] - 15.0) < 1e-10, '[TC08] FlowSolver 左边界 FAILED'
assert abs(h_field[-1] - 10.0) < 1e-10, '[TC08] FlowSolver 右边界 FAILED'

# ---- TC09: FEMTransportSolver1D 纯弥散浓度非负与边界满足 ----
solver = FEMTransportSolver1D(nodes, theta=1.0)
C0 = np.zeros_like(nodes)
C0[nodes <= 10.0] = 1.0
t_hist, C_hist = solver.solve_transient(C0, t_end=10.0, dt=2.0, D=1.0, v=0.0, R=1.0, lam=0.0, bc_nodes=[0, len(nodes)-1], bc_values=[0.0, 0.0])
assert np.all(C_hist >= -1e-10), '[TC09] FEM 浓度非负 FAILED'
assert abs(C_hist[-1, 0] - 0.0) < 1e-10 and abs(C_hist[-1, -1] - 0.0) < 1e-10, '[TC09] FEM 边界条件 FAILED'

# ---- TC10: polynomial_multiply 解析验证 (1+2x)(1+3x)=1+5x+6x^2 ----
p = np.array([1.0, 2.0])
q = np.array([1.0, 3.0])
r = polynomial_multiply(p, q)
assert np.allclose(r, [1.0, 5.0, 6.0]), '[TC10] polynomial_multiply FAILED'

# ---- TC11: MRMTModel 等效滞留因子 s=0 退化为 R_m ----
alphas = np.array([0.01, 0.1, 1.0])
betas = np.array([0.5, 0.3, 0.2])
mrmt = MRMTModel(alphas, betas, R_m=1.0)
R_eff0 = mrmt.effective_retardation(0.0)
assert abs(R_eff0 - 1.0) < 1e-10, '[TC11] MRMT s=0 等效滞留因子 FAILED'

# ---- TC12: TravelTimeTree 反向树层级数正确 ----
tree = TravelTimeTree(x0=50.0, t0=0.0, v_func=lambda x: 0.5, D=0.0, dt=2.0)
btree = tree.build_backward_tree(max_levels=3, n_branches=1)
assert len(btree['levels']) == 4, '[TC12] TravelTimeTree 层级数 FAILED'

# ---- TC13: discrete_dynamical_stability_map 常流速 Lyapunov 为零 ----
x_grid = np.linspace(0, 50, 20)
lyap = discrete_dynamical_stability_map(lambda x: 0.5, x_grid, dt=1.0, n_iter=50)
assert np.allclose(lyap, np.log(1.0), atol=0.1), '[TC13] 常流速 Lyapunov FAILED'

# ---- TC14: sherman_morrison_solve 与直接求解一致 ----
A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
x_base = np.linalg.solve(A, b)
u = np.array([1.0, 0.0])
v = np.array([0.0, 1.0])
A_inv_u = np.linalg.solve(A, u)
x_sm = sherman_morrison_solve(A_inv_u, v, x_base, alpha=1.0)
x_direct = np.linalg.solve(A + np.outer(u, v), b)
assert np.allclose(x_sm, x_direct), '[TC14] Sherman-Morrison FAILED'

# ---- TC15: InverseEstimator LM 线性拟合收敛 ----
x_data = np.array([0.0, 1.0, 2.0, 3.0])
y_data = np.array([1.0, 3.0, 5.0, 7.0])
def lin_residual(beta):
    a, b = beta
    return a * x_data + b - y_data
est = InverseEstimator(lin_residual, ['a', 'b'])
res_lm = est.solve_lm(np.array([0.0, 0.0]), max_iter=20, tol=1e-8)
assert res_lm['final_rms'] < 1e-6, '[TC15] LM 线性拟合 FAILED'

# ---- TC16: MonteCarloEngine 均值与方差非负 ----
def simple_model(beta):
    return beta[0] + beta[1]
def simple_sampler(u):
    return np.array([u[0], u[1]])
mc = MonteCarloEngine(simple_model, simple_sampler, n_params=2)
mc_res = mc.run_mc(100, seed=42)
assert mc_res['mean'] >= 0.0, '[TC16] MC 均值非负 FAILED'
assert mc_res['variance'] >= 0.0, '[TC16] MC 方差非负 FAILED'

# ---- TC17: estimate_exceedance_probability 概率在 [0,1] ----
exceed = mc.estimate_exceedance_probability(mc_res['samples_output'], 1.0)
assert 0.0 <= exceed['exceedance_probability'] <= 1.0, '[TC17] 超标概率范围 FAILED'

# ---- TC18: FractalPorousMedia Sierpinski carpet 形状 ----
fpm = FractalPorousMedia(n_iterations=2, n_points=1000)
K_frac = fpm.generate_sierpinski_carpet_permeability(grid_res=81)
assert K_frac.shape == (81, 81), '[TC18] Sierpinski carpet 形状 FAILED'

# ---- TC19: sample_uniform_hexagon 样本在六边形内 ----
np.random.seed(42)
hex_samples = sample_uniform_hexagon(200, radius=2.0, seed=42)
assert hex_samples.shape == (200, 2), '[TC19] 六边形采样形状 FAILED'
dists = np.linalg.norm(hex_samples, axis=1)
assert np.all(dists <= 2.0 + 1e-6), '[TC19] 六边形采样超出范围 FAILED'

# ---- TC20: divided_differences 线性多项式精确插值 ----
from interpolation_engine import divided_differences, newton_evaluate
x_nodes = np.array([0.0, 1.0, 2.0])
y_nodes = np.array([3.0, 5.0, 7.0])
coeffs = divided_differences(x_nodes, y_nodes)
val = newton_evaluate(x_nodes, coeffs, 1.5)
assert abs(val - 6.0) < 1e-10, '[TC20] Newton 线性插值 FAILED'

# ---- TC21: ConcentrationInterpolator 节点处精确重构 ----
x_fine = np.linspace(0, 10, 11)
C_fine = np.sin(x_fine)
interp = ConcentrationInterpolator(x_fine, C_fine, max_order=5)
for xi in x_fine:
    ci = interp.interpolate(xi)
    assert abs(ci - np.sin(xi)) < 1e-6, '[TC21] 插值节点精确性 FAILED'

# ---- TC22: cubic_spline_natural 边界外插等于端点值 ----
xq = np.array([-1.0, 11.0])
ys = cubic_spline_natural(x_fine, C_fine, xq)
assert abs(ys[0] - C_fine[0]) < 1e-10, '[TC22] 样条左外插 FAILED'
assert abs(ys[1] - C_fine[-1]) < 1e-10, '[TC22] 样条右外插 FAILED'

# ---- TC23: gauss_laguerre_nodes_weights 节点非负权值非负 ----
from quadrature_engine import gauss_laguerre_nodes_weights
x_lg, w_lg = gauss_laguerre_nodes_weights(8, alpha=0.0)
assert np.all(w_lg >= 0), '[TC23] Laguerre 权值非负 FAILED'
assert np.all(x_lg >= 0), '[TC23] Laguerre 节点非负 FAILED'

# ---- TC24: integrate_2d_rectangle 常函数解析验证 ----
def f_const(x, y):
    return 2.0
int_val = integrate_2d_rectangle(f_const, (0.0, 2.0), (0.0, 3.0), nx=4, ny=4)
assert abs(int_val - 12.0) < 1e-10, '[TC24] 2D 常函数积分 FAILED'

# ---- TC25: parse_numeric_table 过滤注释、空行与尾部注释 ----
lines = ["# header", "1.0 2.0", "", "3.0 4.0 # trailing"]
arr = parse_numeric_table(lines)
assert arr.shape == (2, 2), '[TC25] parse_numeric_table 形状 FAILED'
assert np.allclose(arr, [[1.0, 2.0], [3.0, 4.0]]), '[TC25] parse_numeric_table 值 FAILED'

# ---- TC26: serialize_time_series 与 parse_time_series 往返一致 ----
from data_parser import parse_time_series
t_arr = np.array([0.0, 1.0, 2.0])
v_arr = np.array([10.0, 20.0, 30.0])
ts_text = serialize_time_series(t_arr, v_arr, variable_name='test', unit='m')
parsed = parse_time_series(ts_text)
assert len(parsed['t']) == 3, '[TC26] 时间序列往返长度 FAILED'
assert np.allclose(parsed['values'], v_arr), '[TC26] 时间序列往返值 FAILED'

# ---- TC27: encode_well_id 与 decode_well_id ROT13/ROT5 自逆性 ----
orig_id = "MW-2024-A05"
enc_id = encode_well_id(orig_id)
dec_id = decode_well_id(enc_id)
assert dec_id == orig_id, '[TC27] ROT13/ROT5 自逆性 FAILED'

# ---- TC28: encode_coordinate_list 长度与坐标一致性 ----
coords = [(10.0, 20.0), (30.0, 40.0)]
wids = ["W1", "W2"]
enc_res = encode_coordinate_list(coords, wids)
assert len(enc_res['encoded_wells']) == 2, '[TC28] encode_coordinate_list 长度 FAILED'
assert enc_res['encoded_wells'][0]['x'] == 10.0, '[TC28] encode_coordinate_list 坐标 FAILED'

# ---- TC29: integrate_decay_convolution 衰变常数越大积分值越小 ----
C_exp = np.exp(-np.linspace(0, 5, 50) * 0.2)
conv1 = integrate_decay_convolution(C_exp, dt=0.1, lam=0.5, n_quad=16)
conv2 = integrate_decay_convolution(C_exp, dt=0.1, lam=1.0, n_quad=16)
assert conv2 < conv1, '[TC29] 衰减卷积单调性 FAILED'

# ---- TC30: hexagon_grid 生成节点与单元 ----
hcoords, helems = hexagon_grid(3, 2, radius=1.0)
assert hcoords.shape[0] == 6, '[TC30] hexagon_grid 节点数 FAILED'
assert len(helems) > 0, '[TC30] hexagon_grid 单元数 FAILED'

print('\n全部 30 个测试通过!\n')
