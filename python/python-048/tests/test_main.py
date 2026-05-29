"""
main.py
微地震监测与压裂裂缝网络多物理场耦合反演系统

统一入口，零参数可运行。

本程序完成以下博士级科学计算流程:
1. 随机裂缝网络生成（有序正态抽样 + Cliff RNG + Bezier 曲面片）
2. 孔隙压力扩散与裂缝腔体流动耦合求解（扩散 PDE + Euler 时间推进）
3. 库仑破裂判据评估，定位潜在微震源
4. 地震波格林函数计算与合成波形（Filon 振荡积分）
5. 震源定位网格搜索与矩张量最小二乘反演
6. 裂缝扩展 Lyapunov 敏感性分析
7. 六边形/三棱柱高阶求积验证

科学领域: 地球物理 — 微地震监测与压裂裂缝网络
"""

import numpy as np
import sys

# 导入各模块
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

    # ======================================================================
    # 1. 特殊函数与裂缝统计模型 (040_asa121)
    # ======================================================================
    print_section("Step 1: 裂缝尺寸统计模型与特殊函数")
    a_obs = np.array([2.5, 3.1, 4.8, 5.2, 6.0, 8.5, 12.3, 15.0, 22.0, 35.0])
    a_min = 2.0
    D_f = 2.2
    ll = log_likelihood_fracture_size(a_obs, a_min, D_f)
    print(f"  观测裂缝尺寸: {a_obs}")
    print(f"  分形维数 D_f = {D_f}, 最小尺寸 a_min = {a_min} m")
    print(f"  对数似然 ln L = {ll:.4f}")

    # Trigamma 计算
    tg_val, tg_err = trigamma(D_f)
    print(f"  trigamma(D_f) = {tg_val:.6f} (ifault={tg_err})")

    # 应力强度因子
    KI = stress_intensity_factor(sigma=20.0e6, a=5.0, geometry_factor=1.12)
    print(f"  示例应力强度因子 K_I = {KI/1e6:.3f} MPa·m^(1/2)")

    # ======================================================================
    # 2. 随机裂缝网络生成 (1007_random_sorted, 1039_rng_cliff, 672_lights_out)
    # ======================================================================
    print_section("Step 2: 随机裂缝网络参数生成")
    n_frac = 12
    params = generate_fracture_network_params(n_frac, a_min=a_min, D_f=D_f)
    print(f"  生成 {n_frac} 条裂缝")
    print(f"  裂缝半长范围: [{params['lengths'].min():.2f}, {params['lengths'].max():.2f}] m")
    print(f"  走向角均值: {params['strikes'].mean():.2f}°")

    # Lights Out 连通性测试（离散网格）
    mrow, ncol = 5, 5
    occupied = (np.random.rand(mrow * ncol) > 0.55).astype(int)
    A_lo = lights_out_matrix(mrow, ncol)
    conn = connectivity_mod2(occupied, mrow, ncol)
    print(f"  离散网格连通性测试 ({mrow}x{ncol}): 存在贯通簇 = {conn}")

    # ======================================================================
    # 3. Bezier 裂缝曲面几何 (083_bezier_surface)
    # ======================================================================
    print_section("Step 3: Bezier 裂缝曲面片几何建模")
    patches = []
    for i in range(min(4, n_frac)):
        center = params["positions"][i]
        # 由走向、倾角计算法向量
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

    # ======================================================================
    # 4. 孔隙压力扩散与腔体流动 (283_diffusion_pde, 343_euler, 142_cavity_flow_movie)
    # ======================================================================
    print_section("Step 4: 孔隙压力扩散与裂缝腔体流动耦合")
    # 储层参数
    k_perm = 1.0e-15  # m^2 (1 Darcy ≈ 1e-12 m^2)
    mu_fluid = 1.0e-3  # Pa·s (水)
    phi_poro = 0.15
    c_total = 1.0e-9  # Pa^{-1}
    D = k_perm / (mu_fluid * phi_poro * c_total)
    print(f"  水力扩散系数 D = {D:.4e} m²/s")

    solver = PorePressureSolver(
        x_min=-100.0, x_max=100.0, nx=81,
        D=D, source_rate=2.0e2, source_x=0.0
    )
    tspan = (0.0, 3600.0)  # 1 小时
    n_steps = 800
    t_hist, p_hist = solver.solve(tspan, n_steps)
    p_final = p_hist[-1, :]
    print(f"  模拟时长: {tspan[1]} s, 时间步数: {n_steps}")
    print(f"  最终压力范围: [{p_final.min():.2e}, {p_final.max():.2e}] Pa")

    # 裂缝腔体流速
    w_aperture = 2.0e-3  # 2 mm
    v_flow = solver.cavity_flow_velocity(p_final, w_aperture, mu_fluid)
    print(f"  裂缝腔体最大流速: {v_flow.max():.4e} m/s")

    # 库仑破裂应力评估
    sigma_n = 50.0e6  # Pa
    cfs = solver.coulomb_failure_stress(p_final, sigma_n, mu_fric=0.6, cohesion=2.0e6)
    n_failure = np.sum(cfs >= 0)
    print(f"  库仑破裂应力 CFS >= 0 的网格数: {n_failure} / {solver.x.size}")

    # ======================================================================
    # 5. 地震波格林函数与合成波形 (430_filon_rule)
    # ======================================================================
    print_section("Step 5: 地震波格林函数与合成远场位移")
    green = SeismicGreen(rho=2650.0, alpha=4500.0, beta=2600.0)

    # 定义一个微震源与接收台站
    source_loc = np.array([10.0, 5.0, -2000.0])
    station = np.array([150.0, 80.0, -1500.0])

    # 构造矩张量（走滑型）
    mt = MomentTensor.from_strike_dip_rake(strike_deg=30.0, dip_deg=90.0, rake_deg=0.0, M0=1.0e14)
    print(f"  震源位置: {source_loc}")
    print(f"  接收台站: {station}")
    print(f"  地震矩 M_0 = {mt.seismic_moment:.4e} N·m")
    print(f"  矩震级 M_w = {mt.moment_magnitude:.3f}")

    # 走时
    tp = green.travel_time_p(station, source_loc)
    ts = green.travel_time_s(station, source_loc)
    print(f"  P 波走时: {tp:.4f} s, S 波走时: {ts:.4f} s")

    # 时间域位移（Filon/Simpson 积分）
    t_eval = tp + 0.05
    u_t = green.time_domain_displacement_filon(station, source_loc, mt.M, t_eval,
                                                omega_max=150.0, n_omega=301)
    print(f"  t={t_eval:.3f}s 时远场位移: [{u_t[0]:.6e}, {u_t[1]:.6e}, {u_t[2]:.6e}] m")

    # Filon 积分独立验证
    def test_filon_func(x_arr):
        return np.exp(-x_arr**2)
    filon_val = filon_cos_quad(test_filon_func, 0.0, 2.0, 101, t=5.0)
    print(f"  Filon 测试积分 ∫_0^2 exp(-x²) cos(5x) dx ≈ {filon_val:.6f}")

    # ======================================================================
    # 6. 震源定位与矩张量反演 (1381_vandermonde, 132_caesar, 672_lights_out)
    # ======================================================================
    print_section("Step 6: 震源定位与矩张量反演")

    # 构建多个台站与含噪观测走时
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

    # 合成位移用于矩张量反演
    synth_disp = np.zeros((n_stations, 3))
    for k in range(n_stations):
        u_syn = green.displacement_spectrum_farfield(stations[k], source_loc, mt.M, omega=50.0)
        synth_disp[k, :] = np.real(u_syn)

    mt_inv = moment_tensor_inversion(stations, synth_disp, source_loc, green)
    print(f"  反演地震矩 M_0 = {mt_inv.seismic_moment:.4e} N·m")
    print(f"  反演矩震级 M_w = {mt_inv.moment_magnitude:.3f}")

    # Caesar 循环置换分析
    mt_rot = apply_caesar_rotation(mt.M, k=1)
    print(f"  Caesar 旋转后矩张量迹: {np.trace(mt_rot):.4e}")

    # Vandermonde 辐射花样插值
    poly_coeffs = mt_inv.interpolate_radiation_vandermonde(n_samples=8)
    print(f"  Vandermonde 辐射花样多项式系数前 4 项: {poly_coeffs[:4]}")

    # ======================================================================
    # 7. 敏感性分析 (1064_sensitive_ode)
    # ======================================================================
    print_section("Step 7: 裂缝扩展敏感性分析")
    y0 = np.array([0.01, 0.0])
    delta_y0 = np.array([1.0e-4, 0.0])
    tspan_sens = (0.0, 3.0)
    n_steps_sens = 600
    lam_est = lyapunov_exponent_euler(y0, delta_y0, tspan_sens, n_steps_sens)
    print(f"  敏感型 ODE 估计 Lyapunov 指数: λ = {lam_est:.4f}")
    print(f"  理论 Lyapunov 指数: λ = 1.0000")

    # 精确解对比
    y_exact = sensitive_exact(tspan_sens[1], y0)
    _, y_euler = euler_integrate(sensitive_deriv, tspan_sens, y0, n_steps_sens)
    y_euler_final = y_euler[-1, :]
    err = np.linalg.norm(y_euler_final - y_exact)
    print(f"  Euler 终值误差 (vs 精确解): {err:.6e}")

    # 裂缝前缘伴随敏感性
    fps = FracturePropagationSensitivity(kappa=0.8, sigma_noise=0.05)
    dJ_dkappa = fps.adjoint_sensitivity(y0, tspan_sens, n_steps_sens, parameter_index=0)
    print(f"  伴随方法估计目标泛函对 κ 的梯度: {dJ_dkappa:.6f}")

    # ======================================================================
    # 8. 高阶求积规则验证 (530_hexagon_stroud, 916_prism_jaskowiec, 919_product_rule)
    # ======================================================================
    print_section("Step 8: 高阶求积规则数值验证")

    # 六边形 Stroud 规则
    hex_x, hex_y, hex_w = hexagon_stroud_rule(p=3)
    hex_area = np.sum(hex_w)
    print(f"  六边形 Stroud p=3 规则: {hex_x.size} 点, 权重和 = {hex_area:.6f} (理论 = {3*np.sqrt(3)/2:.6f})")

    # 三棱柱 Jaskowiec 规则
    prism_x, prism_y, prism_z, prism_w = prism_jaskowiec_rule(p=3)
    prism_vol = np.sum(prism_w)
    print(f"  三棱柱 Jaskowiec p=3 规则: {prism_x.size} 点, 权重和 = {prism_vol:.6f} (理论 = 0.5)")

    # 乘积规则
    x1d = np.array([0.0, np.sqrt(3.0/5.0), -np.sqrt(3.0/5.0)])
    w1d = np.array([8.0/9.0, 5.0/9.0, 5.0/9.0])
    X_prod, W_prod = product_rule_1d([x1d, x1d], [w1d, w1d])
    print(f"  二维乘积 Gauss-Legendre 规则: {W_prod.size} 点, 权重和 = {np.sum(W_prod):.6f} (理论 = 4.0)")

    # ======================================================================
    # 9. 连通性聚类分析 (672_lights_out)
    # ======================================================================
    print_section("Step 9: 破裂区三维连通性聚类")
    nx, ny, nz = 10, 10, 5
    grid_occ = np.random.rand(nx * ny * nz) > 0.88
    clusters = connectivity_source_cluster(grid_occ.astype(int), (nx, ny, nz))
    print(f"  网格规模: {nx}x{ny}x{nz}, 破裂单元数: {np.sum(grid_occ)}")
    print(f"  识别连通簇数量: {len(clusters)}")
    for ci, cl in enumerate(clusters[:3]):
        print(f"    簇 {ci+1}: 包含 {cl.shape[0]} 个单元")

    # ======================================================================
    # 10. 结果汇总
    # ======================================================================
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
    main()

# ================================================================
# 测试用例（27个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: trigamma(1) 应等于 pi^2/6 ----
val_tg, err_tg = trigamma(1.0)
assert err_tg == 0, '[TC01] trigamma(1) 错误码非零 FAILED'
assert np.isclose(val_tg, np.pi ** 2 / 6.0, rtol=1e-5), '[TC01] trigamma(1) 数值不匹配 FAILED'

# ---- TC02: fracture_size_pdf 对小于 a_min 的尺寸返回 0 ----
pdf_vals = fracture_size_pdf(np.array([1.0, 2.0, 3.0]), a_min=2.5, D_f=2.2)
assert np.all(pdf_vals[:2] == 0.0), '[TC02] 小于 a_min 的 pdf 非零 FAILED'
assert pdf_vals[2] > 0.0, '[TC02] 大于 a_min 的 pdf 非正 FAILED'

# ---- TC03: stress_intensity_factor 对负裂缝长度返回 0 ----
KI_neg = stress_intensity_factor(sigma=20.0e6, a=-1.0, geometry_factor=1.12)
assert KI_neg == 0.0, '[TC03] 负 a 的 KI 未返回 0 FAILED'

# ---- TC04: log_likelihood_fracture_size 对含非法观测返回 -inf ----
ll_invalid = log_likelihood_fracture_size(np.array([2.0, 1.5]), a_min=2.0, D_f=2.2)
assert ll_invalid == -np.inf, '[TC04] 含 a<a_min 时对数似然未返回 -inf FAILED'

# ---- TC05: create_planar_fracture_patch 面积必须为正 ----
patch_tc05 = create_planar_fracture_patch(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 10.0, 5.0, patch_id=0)
assert patch_tc05.area > 0.0, '[TC05] 裂缝片面积非正 FAILED'

# ---- TC06: FracturePatch 重心接近几何中心 ----
patch_tc06 = create_planar_fracture_patch(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]), 4.0, 2.0, patch_id=1)
cent_tc06 = patch_tc06.centroid()
assert np.allclose(cent_tc06, np.array([1.0, 2.0, 3.0]), atol=0.3), '[TC06] 重心偏离中心过多 FAILED'

# ---- TC07: filon_cos_quad 对常数函数 f=1, t=0 等于区间长度 ----
filon_val_tc07 = filon_cos_quad(lambda x_arr: np.ones_like(x_arr), 0.0, 2.0, 101, t=0.0)
assert np.isclose(filon_val_tc07, 2.0, rtol=1e-10), '[TC07] Filon 常数函数积分值错误 FAILED'

# ---- TC08: hexagon_stroud_rule p=3 权重和等于六边形理论面积 ----
hex_x_tc08, hex_y_tc08, hex_w_tc08 = hexagon_stroud_rule(p=3)
assert np.isclose(np.sum(hex_w_tc08), 3.0 * np.sqrt(3.0) / 2.0, rtol=1e-10), '[TC08] 六边形权重和不等于理论面积 FAILED'

# ---- TC09: prism_jaskowiec_rule p=0 权重和等于理论体积 0.5 ----
prism_x_tc09, prism_y_tc09, prism_z_tc09, prism_w_tc09 = prism_jaskowiec_rule(p=0)
assert np.isclose(np.sum(prism_w_tc09), 0.5, rtol=1e-10), '[TC09] 三棱柱权重和不等于 0.5 FAILED'

# ---- TC10: product_rule_1d 二维乘积权重和等于各维权重和乘积 ----
x1d_tc10 = np.array([0.0, 0.5])
w1d_tc10 = np.array([0.5, 0.5])
X_tc10, W_tc10 = product_rule_1d([x1d_tc10, x1d_tc10], [w1d_tc10, w1d_tc10])
assert np.isclose(np.sum(W_tc10), 1.0, rtol=1e-10), '[TC10] 乘积规则权重和错误 FAILED'
assert X_tc10.shape == (2, 4), '[TC10] 乘积规则节点形状错误 FAILED'

# ---- TC11: r8vec_normal_01_sorted 输出单调不降 ----
np.random.seed(42)
sorted_norm = r8vec_normal_01_sorted(20)
assert np.all(np.diff(sorted_norm) >= -1e-12), '[TC11] 有序正态抽样非单调 FAILED'

# ---- TC12: cliff_sequence 相同种子产生相同序列 ----
seq1_tc12 = cliff_sequence(8, seed=0.123456)
seq2_tc12 = cliff_sequence(8, seed=0.123456)
assert np.allclose(seq1_tc12, seq2_tc12, rtol=1e-12), '[TC12] Cliff 序列不可复现 FAILED'

# ---- TC13: lights_out_matrix 输出维度正确 ----
A_lo_tc13 = lights_out_matrix(3, 4)
assert A_lo_tc13.shape == (12, 12), '[TC13] Lights Out 矩阵维度错误 FAILED'

# ---- TC14: connectivity_mod2 对全占据网格存在贯通簇 ----
occ_tc14 = np.ones(12, dtype=int)
conn_tc14 = connectivity_mod2(occ_tc14, 3, 4)
assert conn_tc14 == True, '[TC14] 全占据网格未判定贯通 FAILED'

# ---- TC15: euler_integrate 对零导数保持初值不变 ----
t_tc15, y_tc15 = euler_integrate(lambda t, y: np.zeros(2), (0.0, 1.0), np.array([3.0, 4.0]), 10)
assert np.allclose(y_tc15[-1, :], np.array([3.0, 4.0]), rtol=1e-12), '[TC15] 零导数 Euler 积分改变初值 FAILED'

# ---- TC16: PorePressureSolver.solve 输出有限值且无 NaN ----
solver_tc16 = PorePressureSolver(x_min=-10.0, x_max=10.0, nx=21, D=1.0e-4, source_rate=1.0, source_x=0.0)
t_hist_tc16, p_hist_tc16 = solver_tc16.solve((0.0, 10.0), 100)
assert np.all(np.isfinite(p_hist_tc16[-1, :])), '[TC16] 孔隙压力解含非有限值 FAILED'

# ---- TC17: SeismicGreen P 波走时小于 S 波走时 ----
green_tc17 = SeismicGreen(rho=2650.0, alpha=4500.0, beta=2600.0)
tp_tc17 = green_tc17.travel_time_p(np.array([100.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
ts_tc17 = green_tc17.travel_time_s(np.array([100.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
assert tp_tc17 < ts_tc17, '[TC17] P 波走时不小于 S 波走时 FAILED'

# ---- TC18: displacement_spectrum_farfield 返回复数数组 ----
mt_tc18 = MomentTensor.from_strike_dip_rake(strike_deg=0.0, dip_deg=90.0, rake_deg=0.0, M0=1.0e12)
u_spec_tc18 = green_tc17.displacement_spectrum_farfield(np.array([100.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), mt_tc18.M, omega=10.0)
assert np.iscomplexobj(u_spec_tc18), '[TC18] 位移谱非复数类型 FAILED'

# ---- TC19: MomentTensor.from_strike_dip_rake 构造对称矩阵 ----
mt_tc19 = MomentTensor.from_strike_dip_rake(strike_deg=30.0, dip_deg=60.0, rake_deg=10.0, M0=1.0e14)
assert np.allclose(mt_tc19.M, mt_tc19.M.T, rtol=1e-12), '[TC19] 矩张量不对称 FAILED'

# ---- TC20: MomentTensor.seismic_moment 为正 ----
assert mt_tc19.seismic_moment > 0.0, '[TC20] 地震矩非正 FAILED'

# ---- TC21: apply_caesar_rotation k=3 为恒等变换 ----
M_tc21 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
M_rot3_tc21 = apply_caesar_rotation(M_tc21, k=3)
assert np.allclose(M_rot3_tc21, M_tc21, rtol=1e-12), '[TC21] Caesar 旋转 3 次未回到原矩阵 FAILED'

# ---- TC22: vandermonde_matrix 第一行全为 1 ----
V_tc22 = vandermonde_matrix(3, np.array([1.0, 2.0, 3.0]))
assert np.allclose(V_tc22[0, :], 1.0, rtol=1e-12), '[TC22] Vandermonde 第一行不全为 1 FAILED'

# ---- TC23: source_location_grid_search 对完美数据定位误差为零 ----
stations_tc23 = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
source_tc23 = np.array([10.0, 10.0, 10.0])
tt_tc23 = np.linalg.norm(stations_tc23 - source_tc23, axis=1) / 5000.0
loc_tc23, misfit_tc23 = source_location_grid_search(stations_tc23, tt_tc23, 5000.0,
                                                     ((0.0, 20.0), (0.0, 20.0), (0.0, 20.0)), (5, 5, 5))
assert misfit_tc23 < 1e-8, '[TC23] 完美数据定位残差非零 FAILED'

# ---- TC24: connectivity_source_cluster 单点孤立簇识别正确 ----
grid_tc24 = np.zeros(27, dtype=int)
grid_tc24[13] = 1
clusters_tc24 = connectivity_source_cluster(grid_tc24, (3, 3, 3))
assert len(clusters_tc24) == 1, '[TC24] 单点簇数量错误 FAILED'
assert clusters_tc24[0].shape[0] == 1, '[TC24] 单点簇单元数错误 FAILED'

# ---- TC25: lyapunov_exponent_euler 对敏感型 ODE 估计接近 1.0 ----
y0_tc25 = np.array([0.01, 0.0])
dy0_tc25 = np.array([1.0e-4, 0.0])
lam_tc25 = lyapunov_exponent_euler(y0_tc25, dy0_tc25, (0.0, 3.0), 600)
assert np.isclose(lam_tc25, 1.0, rtol=0.18), '[TC25] Lyapunov 指数估计偏离 1.0 过多 FAILED'

# ---- TC26: sensitive_exact 与 euler_integrate 终值误差小 ----
y0_tc26 = np.array([1.0, 0.0])
y_exact_tc26 = sensitive_exact(1.0, y0_tc26)
_, y_euler_tc26 = euler_integrate(sensitive_deriv, (0.0, 1.0), y0_tc26, 2000)
err_tc26 = np.linalg.norm(y_euler_tc26[-1, :] - y_exact_tc26)
assert err_tc26 < 0.01, '[TC26] Euler 终值与精确解误差过大 FAILED'

# ---- TC27: FracturePropagationSensitivity.adjoint_sensitivity 输出有限 ----
fps_tc27 = FracturePropagationSensitivity(kappa=0.5, sigma_noise=0.05)
dJ_tc27 = fps_tc27.adjoint_sensitivity(np.array([0.1, 0.0]), (0.0, 1.0), 100, parameter_index=0)
assert np.isfinite(dJ_tc27), '[TC27] 伴随敏感性梯度非有限 FAILED'

print("\n全部 27 个测试通过!\n")
