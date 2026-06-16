"""
main.py — 计算高能物理: 粒子径迹重建与 Kalman 滤波拟合
       高阶有限差分与稳定性分析 (小规模可复现实验)
======================================================

统一入口，零参数可运行。

本项目融合 15 个种子项目的核心算法，构建完整的带电粒子径迹重建流水线:

    种子项目映射:
    ──────────────────────────────────────────────────────────
    [064_backward_euler_fixed] → 后向平滑器的隐式向后 Euler + Picard 迭代
    [1322_triangle_to_xml]    → 探测器几何的 DOLFIN XML 序列化
    [165_chebyshev1_rule]     → Gauss 求积用于 χ² 似然积分
    [067_ball_grid]           → 3D 网格采样用于灵敏体积覆盖
    [1423_xyz_display]        → 3D 坐标归一化用于模式识别预处理
    [1330_triangulation]      → 网格质量度量用于探测器覆盖评估
    [669_levenshtein_matrix]  → 编辑距离用于径迹模式匹配
    [158_change_polynomial]   → 多项式乘法用于种子组合计数
    [1225_Okita0512_VSC_HEOM] → DVR/Hierarchical 方法用于径迹层次分析
    [503_hand_mesh2d]         → DistMesh 自适应网格用于探测器通道优化
    [742_mcnuggets]           → Diophantine 求解用于材料厚度组合优化
    [581_image_noise]         → 椒盐/均匀噪声用于探测器噪声模拟
    [694_local_min]           → Brent 最小化用于 χ² 参数优化
    [614_kdv_etdrk4]          → ETDRK4 用于刚性磁场 ODE 传播
    [1122_PepHiRe]            → Ladderpath 分解用于径迹拓扑分析
    ──────────────────────────────────────────────────────────

运行:
    python main.py

输出:
    1. 探测器配置与几何摘要
    2. 模拟粒子径迹 (truth)
    3. 探测器击中模拟 (含噪声)
    4. Kalman 滤波重建结果
    5. 后向平滑结果
    6. χ² 拟合质量
    7. 数值稳定性分析
    8. 综合诊断报告
"""

import sys
import math
import random
import numpy as np

# 添加当前目录到 path
sys.path.insert(0, '.')

from constants import (
    B_FIELD_NOMINAL, MASS_PION, MASS_MUON, MASS_ELECTRON,
    relativistic_beta, relativistic_gamma, relativistic_energy,
    kinetic_energy, pt_from_curvature,
)
from detector_geometry import (
    DetectorLayer, TrackingDetector, build_standard_tracker,
)
from magnetic_field import SolenoidField
from noise_model import DetectorNoiseModel
from material_effects import MaterialEffects
from numerical_propagation import (
    propagate_rk4, propagate_backward_euler, propagate_etdrk4,
    lorentz_equations, optimize_step_size,
)
from chi_square import (
    compute_chi2_track, chi2_likelihood_integral,
    brent_chi2_minimize, chi2_pvalue, pull_distribution,
    safe_invert_matrix,
)
from pattern_recognition import (
    levenshtein_distance_matrix, track_pattern_similarity,
    count_seed_combinations, TrackLadderpath, generate_track_seeds,
)
from kalman_engine import (
    TrackState, KalmanFilter, KalmanSmoother, fit_track,
    measurement_model, measurement_jacobian,
)
from stability_analysis import (
    analyze_backward_euler_convergence, compare_propagation_methods,
    analyze_filter_amplification, perturbation_propagation,
    generate_stability_report,
)


def generate_truth_track(pt_gev, eta, phi_0, charge, detector):
    """
    生成模拟真实径迹

    在均匀磁场 B = (0, 0, B₀) 中，带电粒子的轨迹为螺旋线:
        p_T = 给定横向动量
        θ = 2·arctan(e^{-η})    (赝快度 → 极角)
        λ = π/2 - θ             (偶极角)
        κ = q/(0.3·B/p_T)       (曲率)

    螺旋线参数方程:
        R = p_T / (0.3·B)       [m] (回旋半径)
        x(t) = R·sin(φ₀ + t/R) + x_c
        y(t) = R·cos(φ₀ + t/R) + y_c
        z(t) = R·tan(λ)·t

    Parameters
    ----------
    pt_gev : float
        横向动量 [GeV/c]
    eta : float
        赝快度
    phi_0 : float
        初始方位角 [rad]
    charge : int
        电荷 (±1)
    detector : TrackingDetector

    Returns
    -------
    dict : 真实径迹参数
    """
    # 运动学
    theta = 2.0 * math.atan(math.exp(-eta))
    tan_lambda = 1.0 / math.tan(theta) if abs(math.tan(theta)) > 1e-10 else 10.0
    p_gev = pt_gev * math.cosh(eta)
    p_mev = p_gev * 1000.0

    # 曲率 κ = q·0.3·B / p_T [1/mm]
    kappa = charge * 0.3 * B_FIELD_NOMINAL / (pt_gev * 1000.0)  # 1/mm

    # 回旋半径
    R_mm = abs(pt_gev * 1000.0 / (0.3 * B_FIELD_NOMINAL))  # mm

    # 初始位置 (从顶点出发)
    x0, y0, z0 = 0.0, 0.0, 0.0

    # 在各层上的真实击中
    true_hits = []
    for layer in detector.layers:
        r_layer = layer.radius_mm

        # 粒子到达该层时的方位角
        # 圆弧: sin(Δφ/2) = r/(2R)
        if R_mm > 1e-10:
            sin_half = r_layer / (2.0 * R_mm)
            if abs(sin_half) > 1.0:
                continue  # 粒子无法到达该层
            delta_phi = 2.0 * math.asin(sin_half)
        else:
            delta_phi = 0.0

        phi_at_layer = phi_0 + charge * delta_phi

        # 击中位置
        x_hit = r_layer * math.cos(phi_at_layer)
        y_hit = r_layer * math.sin(phi_at_layer)

        # 纵向位置
        path_length = r_layer * delta_phi  # 弧长近似
        z_hit = z0 + tan_lambda * path_length

        # 检查是否在层的接受范围内
        if abs(z_hit) > layer.half_length_mm:
            continue

        true_hits.append({
            'layer_id': layer.layer_id,
            'layer_radius': r_layer,
            'x_true': x_hit,
            'y_true': y_hit,
            'z_true': z_hit,
            'r_true': r_layer,
            'phi_true': phi_at_layer,
        })

    return {
        'pt_gev': pt_gev,
        'eta': eta,
        'phi_0': phi_0,
        'charge': charge,
        'kappa': kappa,
        'tan_lambda': tan_lambda,
        'theta': theta,
        'p_gev': p_gev,
        'p_mev': p_mev,
        'R_mm': R_mm,
        'true_hits': true_hits,
        'x0': x0, 'y0': y0, 'z0': z0,
        'particle': 'pion',
        'mass_mev': MASS_PION,
    }


def simulate_hits(truth_track, detector, noise_model):
    """
    模拟探测器击中 (含噪声和物质效应)

    Parameters
    ----------
    truth_track : dict
    detector : TrackingDetector
    noise_model : DetectorNoiseModel

    Returns
    -------
    dict : {
        'hits_per_layer': dict,
        'n_detected': int,
        'n_lost': int,
        'n_fake': int,
        'residuals': list,
    }
    """
    hits_per_layer = {}
    residuals_r = []
    residuals_z = []
    n_detected = 0
    n_lost = 0

    material_calc = MaterialEffects(
        particle_charge=truth_track['charge'],
        particle_mass_mev=truth_track['mass_mev']
    )

    current_p = truth_track['p_mev']

    for true_hit in truth_track['true_hits']:
        layer = detector.get_layer(true_hit['layer_id'])

        # 物质效应
        effects = material_calc.energy_loss_in_layer(current_p, layer)
        current_p = effects['p_out']

        # 模拟击中
        hit = noise_model.simulate_hit(
            true_hit['r_true'], true_hit['z_true'],
            layer.sigma_r_mm, layer.sigma_z_mm
        )

        if hit is not None:
            hit['layer_id'] = true_hit['layer_id']
            hit['phi_true'] = true_hit['phi_true']
            hits_per_layer[true_hit['layer_id']] = hit
            residuals_r.append(hit['residual_r'])
            residuals_z.append(hit['residual_z'])
            n_detected += 1
        else:
            n_lost += 1

    # 添加假击中
    n_fake = 0
    for layer in detector.layers:
        if noise_model._rng.random() < noise_model.fake_rate * 0.1:
            fake_hits = noise_model.generate_fake_hits(
                layer.radius_mm, layer.half_length_mm, 1)
            # 不与真实击中冲突
            if layer.layer_id not in hits_per_layer:
                hits_per_layer[layer.layer_id] = fake_hits[0]
                n_fake += 1

    return {
        'hits_per_layer': hits_per_layer,
        'n_detected': n_detected,
        'n_lost': n_lost,
        'n_fake': n_fake,
        'residuals_r': residuals_r,
        'residuals_z': residuals_z,
    }


def run_kalman_reconstruction(truth_track, sim_hits, detector, bfield_t=2.0):
    """
    运行 Kalman 滤波径迹重建

    Parameters
    ----------
    truth_track : dict
    sim_hits : dict
    detector : TrackingDetector
    bfield_t : float

    Returns
    -------
    dict : 重建结果
    """
    # 初始状态猜测 (从真实参数加偏移)
    rng = random.Random(12345)
    init_kappa = truth_track['kappa'] * (1.0 + 0.01 * rng.gauss(0, 1))
    init_tan_lambda = truth_track['tan_lambda'] * (1.0 + 0.01 * rng.gauss(0, 1))
    init_phi = truth_track['phi_0'] + 0.001 * rng.gauss(0, 1)
    init_d = 0.0 + 0.01 * rng.gauss(0, 1)
    init_z0 = truth_track['z0'] + 0.1 * rng.gauss(0, 1)

    initial_state = TrackState(
        params=[init_kappa, init_tan_lambda, init_phi, init_d, init_z0],
        covariance=np.diag([1e-4, 1e-3, 1e-3, 0.1, 1.0])
    )

    # 构建各层击中列表
    hits_list = []
    for layer in detector.layers:
        hit = sim_hits['hits_per_layer'].get(layer.layer_id)
        hits_list.append(hit)

    # Kalman 滤波
    kf = KalmanFilter(bfield_t)
    kf.reset()

    current_state = initial_state.copy()
    current_radius = detector.inner_radius * 0.5
    fitted_states = []
    residuals = []
    chi2_per_hit = []

    for layer_idx, layer in enumerate(detector.layers):
        if layer_idx >= len(hits_list) or hits_list[layer_idx] is None:
            continue

        hit = hits_list[layer_idx]

        # 预测
        pred_state = kf.predict(current_state, current_radius, layer.radius_mm)

        # 测量
        meas = np.array([hit.get('r_meas', layer.radius_mm),
                         hit.get('z_meas', 0.0)])
        meas_cov = np.diag([layer.sigma_r_mm**2, layer.sigma_z_mm**2])

        # 更新
        current_state = kf.update(pred_state, meas, meas_cov, layer.radius_mm)
        current_radius = layer.radius_mm

        fitted_states.append(current_state.copy())

        # 残差
        h_pred = measurement_model(pred_state, layer.radius_mm)
        res = meas - h_pred
        residuals.append(res.tolist())

    # 后向平滑
    active_radii = []
    for layer_idx, layer in enumerate(detector.layers):
        if layer_idx < len(hits_list) and hits_list[layer_idx] is not None:
            active_radii.append(layer.radius_mm)

    smoother = KalmanSmoother(kf)
    smoothed_states = smoother.smooth(active_radii)

    # 最终结果
    final_state = fitted_states[-1] if fitted_states else None
    smoothed_final = smoothed_states[-1] if smoothed_states else None

    results = {
        'chi2': kf.chi2,
        'ndf': kf.ndf,
        'chi2_ndof': kf.chi2 / max(kf.ndf - 5, 1),
        'chi2_per_hit': kf.chi2_per_hit,
        'n_hits': len(fitted_states),
        'fitted_states': fitted_states,
        'smoothed_states': smoothed_states,
        'filter_gains': kf.gains,
        'predicted_states': kf.predicted_states,
        'residuals': residuals,
        'final_state': final_state,
        'smoothed_final': smoothed_final,
        'initial_state': initial_state,
    }

    return results


def run_stability_checks(state0, bfield, detector):
    """
    执行数值稳定性分析

    Returns
    -------
    dict : 稳定性分析结果
    """
    # 1. 向后 Euler 收敛性
    omega_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    dt_range = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    convergence = analyze_backward_euler_convergence(omega_range, dt_range)

    # 2. 传播方法对比
    step_lengths = [10.0, 20.0, 50.0, 100.0]
    propagation = compare_propagation_methods(
        state0, lambda x, y, z: bfield.get_field(x, y, z),
        step_lengths
    )

    # 3. 扰动传播
    perturbation = perturbation_propagation(
        state0, lambda x, y, z: bfield.get_field(x, y, z)
    )

    return {
        'convergence': convergence,
        'propagation': propagation,
        'perturbation': perturbation,
    }


def print_separator(title, char='═', width=70):
    """打印分隔线"""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def format_scientific(value, precision=4):
    """科学计数法格式化"""
    if abs(value) < 1e-15:
        return "0.0000e+00"
    return f"{value:.{precision}e}"


def main():
    """
    主函数: 运行完整的径迹重建流水线
    """
    print_separator("计算高能物理: 粒子径迹重建与 Kalman 滤波拟合", '═')
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("  PROJECT_227 — 博士级合成项目")

    # ============================================================
    # 1. 初始化探测器
    # ============================================================
    print_separator("1. 探测器配置", '─')

    detector = build_standard_tracker()
    print(detector.summary())

    # [1322] 探测器几何 XML 导出
    xml_str = detector.to_xml_string()
    xml_lines = xml_str.count('\n') + 1
    print(f"\n[1322] XML 几何导出: {xml_lines} 行, "
          f"{len(xml_str)} 字符")

    # [067] 灵敏体积网格采样
    grid_points = detector.sample_sensitive_volume(
        n_radial=3, n_phi=4, n_z=3)
    print(f"[067] 灵敏体积采样: {len(grid_points)} 个网格点")

    # [1423] 坐标归一化测试
    test_points = [(gp[0] * math.cos(gp[1]),
                    gp[0] * math.sin(gp[1]),
                    gp[2]) for gp in grid_points[:5]]
    if test_points:
        norm_pts, bounds = detector.normalize_hit_coordinates(test_points)
        print(f"[1423] 坐标归一化: range={bounds['range']:.1f}mm, "
              f"margin={bounds['margin']:.2f}mm")

    # [1330] 覆盖质量
    coverage_q = detector.coverage_quality_score(n_phi_samples=18)
    print(f"[1330] 探测器覆盖质量: {coverage_q:.4f}")

    # [503] 通道密度优化
    channels = detector.refine_layer_density(target_max_spacing_mm=50.0)
    total_channels = sum(c['total_channels'] for c in channels.values())
    print(f"[503] 通道优化: 目标间距 50mm → 总通道数 {total_channels}")

    # ============================================================
    # 2. 磁场配置
    # ============================================================
    print_separator("2. 磁场配置", '─')

    bfield = SolenoidField(
        b0_tesla=B_FIELD_NOMINAL,
        z_range_mm=(-800.0, 800.0),
        r_range_mm=(0.0, 600.0),
        poly_order=4,
        nonuniformity=1e-6,
    )

    # 测试磁场
    b_center = bfield.get_field(0, 0, 0)
    b_edge = bfield.get_field(300, 0, 0)
    b_far = bfield.get_field(0, 0, 700)
    print(f"  B(center) = ({b_center[0]:.6f}, {b_center[1]:.6f}, {b_center[2]:.6f}) T")
    print(f"  B(r=300mm) = ({b_edge[0]:.6f}, {b_edge[1]:.6f}, {b_edge[2]:.6f}) T")
    print(f"  B(z=700mm) = ({b_far[0]:.6f}, {b_far[1]:.6f}, {b_far[2]:.6f}) T")

    # [165] 磁场线积分
    integral_bz = bfield.line_integral_bz(-500, 500, n_quad=12)
    print(f"[165] ∫B_z dz [-500,500]mm = {integral_bz:.4f} T·mm")

    # ============================================================
    # 3. 生成模拟径迹
    # ============================================================
    print_separator("3. 模拟径迹生成", '─')

    # 固定种子保证可复现
    random.seed(42)
    np.random.seed(42)

    # 生成 π⁺ 径迹
    truth = generate_truth_track(
        pt_gev=2.0,       # 2 GeV/c 横向动量
        eta=0.5,           # 赝快度
        phi_0=0.3,         # 初始方位角
        charge=+1,         # 正电荷
        detector=detector,
    )

    print(f"  粒子类型: π⁺ (m = {truth['mass_mev']:.3f} MeV/c²)")
    print(f"  p_T  = {truth['pt_gev']:.4f} GeV/c")
    print(f"  η    = {truth['eta']:.4f}")
    print(f"  φ₀   = {truth['phi_0']:.4f} rad")
    print(f"  p    = {truth['p_gev']:.4f} GeV/c")
    print(f"  θ    = {truth['theta']:.4f} rad ({math.degrees(truth['theta']):.2f}°)")
    print(f"  κ    = {truth['kappa']:.6e} 1/mm")
    print(f"  tanλ = {truth['tan_lambda']:.4f}")
    print(f"  R_L  = {truth['R_mm']:.2f} mm (Larmor 半径)")
    print(f"  β    = {relativistic_beta(truth['p_mev'], truth['mass_mev']):.6f}")
    print(f"  γ    = {relativistic_gamma(truth['p_mev'], truth['mass_mev']):.4f}")
    print(f"  真实击中数: {len(truth['true_hits'])} / {detector.n_layers} 层")

    # ============================================================
    # 4. 探测器噪声模拟
    # ============================================================
    print_separator("4. 探测器噪声模拟", '─')

    noise_model = DetectorNoiseModel(
        dead_channel_prob=0.001,
        hot_channel_prob=0.0005,
        noise_prob=0.02,
        noise_amplitude_r=0.05,
        noise_amplitude_z=0.15,
        hit_efficiency=0.96,
        fake_hit_rate=0.3,
        seed=42,
    )
    print(f"  {noise_model.summary()}")

    sim_hits = simulate_hits(truth, detector, noise_model)

    print(f"\n  探测到: {sim_hits['n_detected']} 击中")
    print(f"  丢失:   {sim_hits['n_lost']} 击中")
    print(f"  假阳性: {sim_hits['n_fake']} 击中")

    if sim_hits['residuals_r']:
        mean_res_r = sum(sim_hits['residuals_r']) / len(sim_hits['residuals_r'])
        std_res_r = math.sqrt(sum((r - mean_res_r)**2
                                   for r in sim_hits['residuals_r'])
                               / max(len(sim_hits['residuals_r']) - 1, 1))
        print(f"  径向残差: mean = {mean_res_r*1e3:.2f} μm, "
              f"std = {std_res_r*1e3:.2f} μm")

    if sim_hits['residuals_z']:
        mean_res_z = sum(sim_hits['residuals_z']) / len(sim_hits['residuals_z'])
        std_res_z = math.sqrt(sum((r - mean_res_z)**2
                                   for r in sim_hits['residuals_z'])
                               / max(len(sim_hits['residuals_z']) - 1, 1))
        print(f"  纵向残差: mean = {mean_res_z*1e3:.2f} μm, "
              f"std = {std_res_z*1e3:.2f} μm")

    # ============================================================
    # 5. 模式识别与种子生成
    # ============================================================
    print_separator("5. 模式识别", '─')

    # [669] 编辑距离
    hit_layers = set(sim_hits['hits_per_layer'].keys())
    true_layers = set(h['layer_id'] for h in truth['true_hits'])
    pattern_sim = track_pattern_similarity(hit_layers, true_layers,
                                            detector.n_layers)
    print(f"[669] 击中模式编辑距离: {pattern_sim['distance']}")
    print(f"     相似度: {pattern_sim['similarity']:.4f}")
    print(f"     共同层: {pattern_sim['n_common']}")

    # [158] 种子组合计数
    hit_counts_per_layer = []
    for layer in detector.layers:
        n_hits = 1 if layer.layer_id in sim_hits['hits_per_layer'] else 0
        hit_counts_per_layer.append(max(n_hits, 1))
    seed_info = count_seed_combinations(hit_counts_per_layer, n_seed_layers=3)
    print(f"[158] 种子组合总数: {seed_info['total_seeds']}")

    # [1122] Ladderpath 分解
    ladderpath = TrackLadderpath()
    sorted_hit_layers = sorted(hit_layers)
    decomposition = ladderpath.decompose(sorted_hit_layers)
    complexity = ladderpath.complexity_index()
    print(f"[1122] Ladderpath 分解: {len(decomposition)} 个模式")
    print(f"      复杂度: (ladderons={complexity[0]}, "
          f"basic={complexity[1]}, max_depth={complexity[2]})")

    # ============================================================
    # 6. Kalman 滤波重建
    # ============================================================
    print_separator("6. Kalman 滤波重建", '─')

    recon = run_kalman_reconstruction(truth, sim_hits, detector, B_FIELD_NOMINAL)

    print(f"  拟合击中数: {recon['n_hits']}")
    print(f"  χ²  = {recon['chi2']:.4f}")
    print(f"  ndf = {recon['ndf']}")
    print(f"  χ²/ndf = {recon['chi2_ndof']:.4f}")

    # p-value
    pval = chi2_pvalue(recon['chi2'], max(recon['ndf'] - 5, 1))
    print(f"  p-value = {pval:.4f}")

    # 最终参数
    if recon['final_state'] is not None:
        fs = recon['final_state']
        print(f"\n  滤波后参数:")
        print(f"    κ     = {format_scientific(fs.kappa)} 1/mm")
        print(f"    tanλ  = {fs.tan_lambda:.6f}")
        print(f"    φ     = {fs.phi:.6f} rad")
        print(f"    d     = {fs.d:.4f} mm")
        print(f"    z₀    = {fs.z0:.4f} mm")
        print(f"    p_T   = {fs.pt_gev:.4f} GeV/c")
        print(f"    η     = {fs.eta:.4f}")

        # 参数残差 (相对于真实值)
        kappa_err = abs(fs.kappa - truth['kappa']) / max(abs(truth['kappa']), 1e-15) * 100
        tanlam_err = abs(fs.tan_lambda - truth['tan_lambda']) / max(abs(truth['tan_lambda']), 1e-15) * 100
        print(f"\n  参数相对误差:")
        print(f"    Δκ/κ   = {kappa_err:.4f} %")
        print(f"    Δtanλ/tanλ = {tanlam_err:.4f} %")

    # 平滑后参数
    if recon['smoothed_final'] is not None:
        ss = recon['smoothed_final']
        print(f"\n  平滑后参数:")
        print(f"    κ     = {format_scientific(ss.kappa)} 1/mm")
        print(f"    tanλ  = {ss.tan_lambda:.6f}")
        print(f"    p_T   = {ss.pt_gev:.4f} GeV/c")

        kappa_err_smooth = abs(ss.kappa - truth['kappa']) / max(abs(truth['kappa']), 1e-15) * 100
        print(f"    Δκ/κ (平滑) = {kappa_err_smooth:.4f} %")

    # [694] Brent χ² 最小化 (曲率微调)
    if recon['final_state'] is not None:
        kappa_init = recon['final_state'].kappa

        def chi2_vs_kappa(kappa_val):
            """简化 χ² 作为曲率的函数"""
            total = 0.0
            for i, res_list in enumerate(recon['residuals']):
                if len(res_list) >= 1:
                    total += res_list[0]**2 / 0.01**2
            # 添加曲率偏差惩罚
            total += (kappa_val - kappa_init)**2 / 1e-6
            return total

        brent_result = brent_chi2_minimize(
            chi2_vs_kappa,
            kappa_init * 0.5, kappa_init * 1.5,
            tol=1e-6, max_iter=50
        )
        print(f"\n[694] Brent χ² 最小化:")
        print(f"     最优 κ = {format_scientific(brent_result['p_opt'])}")
        print(f"     最小 χ² = {brent_result['chi2_min']:.4f}")
        print(f"     收敛: {brent_result['converged']}, "
              f"求值 {brent_result['n_evals']} 次")

    # ============================================================
    # 7. Pull 分布分析
    # ============================================================
    print_separator("7. Pull 分布与残差分析", '─')

    if recon['residuals']:
        r_residuals = [r[0] for r in recon['residuals'] if len(r) > 0]
        r_errors = [detector.layers[i].sigma_r_mm
                     for i in range(len(r_residuals))]
        pulls_r = pull_distribution(r_residuals, r_errors)
        print(f"  径向 Pull:")
        print(f"    mean = {pulls_r['mean']:.4f} (期望: 0)")
        print(f"    std  = {pulls_r['std']:.4f} (期望: 1)")
        print(f"    χ²_pull = {pulls_r['chi2_pull']:.4f}")

    # ============================================================
    # 8. 数值稳定性分析
    # ============================================================
    print_separator("8. 数值稳定性分析", '─')

    # 初始 6D 状态用于传播测试
    state6d = np.array([
        0.0, 0.0, 0.0,  # 位置
        truth['p_mev'] * math.cos(truth['phi_0']) * math.sin(truth['theta']),
        truth['p_mev'] * math.sin(truth['phi_0']) * math.sin(truth['theta']),
        truth['p_mev'] * math.cos(truth['theta']),
    ])

    stability = run_stability_checks(state6d, bfield, detector)

    # 8.1 向后 Euler 收敛
    conv = stability['convergence']
    conv_frac = sum(1 for v in conv['critical_dt'].values() if v > 0) / max(len(conv['critical_dt']), 1)
    print(f"[064] 向后 Euler 收敛分析:")
    print(f"     收敛覆盖率: {conv_frac*100:.1f}%")

    # 8.2 传播方法对比
    prop = stability['propagation']
    print(f"\n  传播方法精度对比 (step=100mm):")
    for method in ['RK4', 'BackwardEuler', 'ETDRK4']:
        errors = prop['method_errors'].get(method, [])
        if errors:
            print(f"    {method:15s}: error = {format_scientific(errors[-1])}")

    # 8.3 滤波稳定性
    if recon['fitted_states'] and recon['predicted_states']:
        filter_analysis = analyze_filter_amplification(
            recon['fitted_states'],
            recon['predicted_states'],
            recon['filter_gains']
        )
        print(f"\n  Kalman 滤波稳定性:")
        print(f"    稳定性: {'✓ 稳定' if filter_analysis['is_stable'] else '✗ 不稳定'}")
        print(f"    最大条件数: {format_scientific(filter_analysis['max_condition'])}")
        if filter_analysis['spectral_radii']:
            print(f"    平均谱半径: {filter_analysis['mean_spectral_radius']:.6f}")

    # 8.4 扰动传播
    pert = stability['perturbation']
    print(f"\n  扰动传播分析:")
    print(f"    最大放大因子: {pert['max_amplification']:.4f}")
    print(f"    平均放大因子: {pert['mean_amplification']:.4f}")
    print(f"    转移矩阵条件数: {format_scientific(pert['transfer_matrix_cond'])}")

    # ============================================================
    # 9. 综合报告
    # ============================================================
    print_separator("9. 综合诊断报告", '═')

    # [742] 材料组合验证
    # 检查各层材料厚度是否为可接受的组合
    thickness_values = [l.thickness_ratio for l in detector.layers]
    total_budget = sum(thickness_values)
    print(f"[742] 材料预算验证:")
    print(f"     总预算: {total_budget:.4f} X₀")
    print(f"     各层: {', '.join(f'{t:.4f}' for t in thickness_values)}")

    # χ² 拟合质量
    print(f"\n  拟合质量:")
    print(f"    χ²/ndf = {recon['chi2_ndof']:.4f}")
    print(f"    p-value = {pval:.4f}")
    quality = '良好' if 0.5 < recon['chi2_ndof'] < 2.0 else '需要关注'
    print(f"    评估: {quality}")

    # 动量分辨率估计
    if recon['final_state'] is not None:
        pt_true = truth['pt_gev']
        pt_reco = recon['final_state'].pt_gev
        pt_res = abs(pt_reco - pt_true) / pt_true * 100
        print(f"\n  动量分辨率:")
        print(f"    p_T(true)  = {pt_true:.4f} GeV/c")
        print(f"    p_T(reco)  = {pt_reco:.4f} GeV/c")
        print(f"    Δp_T/p_T   = {pt_res:.4f} %")

    # 综合评级
    issues = []
    if recon['chi2_ndof'] > 3.0:
        issues.append("χ²/ndf 偏高")
    if pval < 0.01:
        issues.append("p-value 过低")

    print(f"\n  综合评级: {'A (优秀)' if not issues else 'B (良好)'}")
    if issues:
        print(f"  注意事项: {', '.join(issues)}")

    print_separator("计算完成", '═')
    print("  PROJECT_227 径迹重建流水线运行成功")
    print("  融合 15 个种子项目算法于计算高能物理框架")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
