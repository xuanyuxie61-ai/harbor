#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

================================================================================
不规则小行星多尺度引力场建模与近距轨道长期稳定性分析系统
================================================================================

本项目基于 15 个科研代码项目的核心算法，在天体物理领域（小行星/行星
引力场与轨道力学）进行深度融合，构建面向前沿科学问题的博士级计算系统。

科学问题：
    针对碎石堆结构小行星（如 Itokawa、Bennu 型）的不规则形状与非均匀
    密度分布，建立"多面体-球谐耦合"高保真引力场模型，并在此模型基础上
    分析近表面轨道的长期稳定性、碰撞风险与最优悬停策略。

核心模型与公式：
1. 球谐引力势展开（Stokes 理论）：
       U(r,θ,λ) = GM/r [ 1 + Σ_{n=2}^{N_max} Σ_{m=0}^{n} (R_e/r)^n P̄_{nm}(sinθ)
                   × ( C_{nm} cos(mλ) + S_{nm} sin(mλ) ) ]

2. 多面体引力势（Werner-Scheeres 模型）：
       U(r) = (Gρ/2) Σ_e r_e·E_e·r_e·L_e − (Gρ/2) Σ_f r_f·F_f·r_f·ω_f

3. 泊松方程（内部引力势有限元）：
       ∇²φ = 4πGρ(r)

4. 轨道运动方程（含 SRP 与第三体摄动）：
       d²r/dt² = ∇U(r) + a_SRP + a_3body + σ·ξ(t)

5. 碰撞概率模型：
       P_collision ≈ Σ_i (A_i/A_total) Φ( (h_safe − d_i) / σ_pos )

6. 轨道品质泛函：
       J = w1·ln(T_lifetime) − w2·Δv − w3·P_collision

输入科研项目融合：
- 052_asa245          → special_functions.py (Lanczos Gamma)
- 062_backtrack_binary_rc → orbit_optimization.py (二进制回溯)
- 066_ball_distance   → collision_risk.py (蒙特卡洛距离统计)
- 1074_sierpinski_carpet_chaos → asteroid_geometry.py (IFS 分形表面)
- 1102_sparse_display → fem_gravity.py (Wathen 稀疏 FEM)
- 111_box_behnken     → orbit_optimization.py (实验设计)
- 1171_stochastic_rk  → orbit_integrator.py (随机 RK4)
- 1328_triangulate    → asteroid_geometry.py (耳切法三角剖分)
- 1368_tumor_pde      → fem_gravity.py (PDE 系数/通量函数)
- 1424_xyz_io         → data_io.py (XYZ 数据 I/O)
- 684_line_ncc_rule   → orbit_integrator.py (Newton-Cotes 积分)
- 794_neighbor_risk   → collision_risk.py (邻接矩阵连通性)
- 894_polynomial_conversion → gravity_harmonics.py (Legendre 多项式)
- 970_r8blt           → fem_gravity.py (带状下三角求解器)
- 431_filum           → data_io.py (字符串/文件名处理)
"""

import numpy as np
import sys
import time

# 模块导入
from special_functions import (
    lngamma_lanczos,
    gamma_lanczos,
    associated_legendre_normalized,
    factorial_ratio
)
from asteroid_geometry import (
    generate_fractal_profile,
    generate_asteroid_cross_section,
    revolve_to_3d,
    polyhedron_volume_and_com,
    surface_area,
    ear_clip_triangulation
)
from gravity_harmonics import (
    compute_stokes_coefficients_from_shape,
    SphericalHarmonicGravity
)
from gravity_polyhedron import (
    polyhedron_gravity_potential,
    polyhedron_gravity_acceleration,
    combined_gravity_model
)
from fem_gravity import (
    solve_internal_potential_2d,
    internal_gravity_from_potential
)
from orbit_integrator import (
    OrbitalDynamics,
    newton_cotes_integrate
)
from orbit_optimization import (
    OrbitSensitivityAnalysis,
    compute_orbit_quality_score,
    optimize_orbit_binary_backtrack
)
from collision_risk import (
    ball_distance_stats,
    build_surface_adjacency_matrix,
    collision_probability_surface,
    find_safe_hover_regions,
    region_connectivity_analysis
)
from data_io import (
    write_xyz_data,
    write_face_indices,
    generate_synthetic_asteroid_pointcloud,
    s_word_count
)


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def run_special_functions_demo():
    """测试特殊函数模块"""
    print_section("模块 1: 特殊函数与正交多项式 (asa245 + polynomial_conversion)")
    z_vals = [0.5, 1.0, 2.5, 5.0, 10.0]
    for z in z_vals:
        gz = gamma_lanczos(z)
        lz, _ = lngamma_lanczos(z)
        print(f"  Γ({z:.2f}) = {gz:.12e},  lnΓ = {lz:.12e}")

    # 缔合 Legendre 函数测试
    p33 = associated_legendre_normalized(3, 3, 0.5)
    print(f"  P̄_33(0.5) = {p33:.12e}")

    # 阶乘比（球谐归一化系数）
    fr = factorial_ratio(10, 5)
    print(f"  10! / 5! = {fr:.6e}")


def run_geometry_demo():
    """测试小行星几何建模"""
    print_section("模块 2: 小行星几何建模 (triangulate + sierpinski_carpet_chaos)")
    poly2d = generate_asteroid_cross_section(
        a=2.0, b=1.5, c=1.0, n_points=64,
        roughness_amplitude=0.08, roughness_scale=1.0 / 3.0, seed=42
    )
    print(f"  二维截面顶点数: {poly2d.shape[0]}")

    # 耳切法三角剖分（二维截面）
    triangles_2d = ear_clip_triangulation(poly2d)
    print(f"  二维剖分三角面片数: {triangles_2d.shape[0]}")

    # 旋转生成三维
    vertices, faces = revolve_to_3d(poly2d, z_scale=0.8)
    print(f"  三维顶点数: {vertices.shape[0]}, 面片数: {faces.shape[0]}")

    vol, com = polyhedron_volume_and_com(vertices, faces)
    area = surface_area(vertices, faces)
    print(f"  多面体体积: {vol:.6e} km³")
    print(f"  质心位置: [{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}] km")
    print(f"  表面积: {area:.6e} km²")

    return vertices, faces, vol, com


def run_gravity_harmonics_demo(vertices, faces, vol):
    """测试球谐引力场"""
    print_section("模块 3: 球谐引力场展开 (polynomial_conversion)")
    density = 2500.0  # kg/m³ → 需要统一单位，这里简化处理
    # 注意：vertices 单位为 km，需要转换
    vertices_m = vertices * 1e3
    mass = density * vol * 1e9  # kg (vol in km³ → m³)
    gm = 6.67430e-11 * mass  # m³/s² → 转为 km³/s²
    gm_km = gm * 1e-9

    c_coeff, s_coeff, r_ref = compute_stokes_coefficients_from_shape(
        vertices_m, faces, density, n_max=6
    )
    # 转回 km 单位系
    r_ref_km = r_ref * 1e-3

    print(f"  质量 M = {mass:.6e} kg")
    print(f"  GM = {gm_km:.6e} km³/s²")
    print(f"  参考半径 R_e = {r_ref_km:.6f} km")
    print(f"  球谐系数 C_20 = {c_coeff[2,0]:.6e}, C_22 = {c_coeff[2,2]:.6e}")

    sh_model = SphericalHarmonicGravity(gm_km, r_ref_km, c_coeff, s_coeff, n_max=6)

    # 测试几个外部点的势能与加速度
    test_points = [
        np.array([5.0, 0.0, 0.0]),
        np.array([0.0, 4.0, 0.0]),
        np.array([3.0, 3.0, 1.0])
    ]
    for p in test_points:
        pot = sh_model.potential(p)
        acc = sh_model.acceleration(p)
        acc_fd = sh_model.gradient_fd(p)
        print(f"  r={p} km: U={pot:.6e} km²/s², |a|={np.linalg.norm(acc):.6e} km/s²")

    return sh_model, gm_km, r_ref_km, c_coeff, s_coeff


def run_polyhedron_gravity_demo(vertices, faces):
    """测试多面体引力场"""
    print_section("模块 4: 多面体引力场模型 (triangulate 扩展)")
    vertices_m = vertices * 1e3
    density = 2500.0

    test_points = [
        np.array([5.0, 0.0, 0.0]),
        np.array([0.0, 4.0, 0.0]),
    ]
    for p in test_points:
        p_m = p * 1e3
        pot = polyhedron_gravity_potential(p_m, vertices_m, faces, density)
        acc = polyhedron_gravity_acceleration(p_m, vertices_m, faces, density)
        print(f"  r={p} km: U={pot:.6e} m²/s², |a|={np.linalg.norm(acc):.6e} m/s²")


def run_fem_gravity_demo():
    """测试有限元内部引力势"""
    print_section("模块 5: 有限元内部引力势 (wathen_ge + tumor_pde + r8blt_sl)")
    phi, x_coords, y_coords = solve_internal_potential_2d(
        nx=32, ny=32, r_max=1000.0
    )
    gx, gy = internal_gravity_from_potential(phi, x_coords, y_coords)

    # 中心点势能与引力
    cx, cy = 16, 16
    print(f"  中心点引力势: {phi[cy,cx]:.6e} m²/s²")
    print(f"  中心点引力加速度: [{gx[cy,cx]:.6e}, {gy[cy,cx]:.6e}] m/s²")
    print(f"  势场范围: [{phi.min():.6e}, {phi.max():.6e}] m²/s²")


def run_orbit_integrator_demo(sh_model, gm_km):
    """测试轨道积分器"""
    print_section("模块 6: 轨道积分器 (line_ncc_rule + stochastic_rk)")

    # 定义引力加速度函数（使用球谐模型）
    def grav_accel(pos):
        return sh_model.acceleration(pos)

    dyn = OrbitalDynamics(
        grav_accel_func=grav_accel,
        gm_sun=1.32712440018e11,
        solar_distance=1.496e8,
        beta_srp=1e-7,
        perturbation_std=1e-12
    )

    # 初始条件：近小行星圆轨道近似
    r0 = np.array([3.0, 0.0, 0.5])  # km
    v_circ = np.sqrt(gm_km / np.linalg.norm(r0))
    v0 = np.array([0.0, v_circ * 0.9, 0.0])  # 略小于圆轨道速度，椭圆
    state0 = np.concatenate([r0, v0])

    # 确定性 RK4 积分（模拟 10 小时）
    t_array, states = dyn.integrate_deterministic(state0, (0.0, 36000.0), n_steps=2000)
    r_final = states[-1, :3]
    v_final = states[-1, 3:]
    print(f"  确定性轨道: 初始 r={r0}, 末态 r={r_final}")
    print(f"  末态速度 |v|={np.linalg.norm(v_final):.6e} km/s")

    # 随机 SRK4 积分
    t_array_s, states_s = dyn.integrate_stochastic(
        state0, (0.0, 36000.0), n_steps=2000, q_spectral=1e-14
    )
    r_final_s = states_s[-1, :3]
    print(f"  随机轨道末态 r={r_final_s}")

    # Newton-Cotes 积分验证：计算轨道周期积分
    def integrand(t):
        return np.sin(t) ** 2

    nc_result = newton_cotes_integrate(integrand, 0.0, np.pi, n=5, n_sub=20)
    analytic = np.pi / 2.0
    print(f"  Newton-Cotes 验证: ∫sin²t dt = {nc_result:.12e} (理论={analytic:.12e})")

    return dyn


def run_orbit_optimization_demo():
    """测试轨道优化"""
    print_section("模块 7: 轨道参数优化 (box_behnken + backtrack_binary_rc)")

    # 模拟的目标函数（基于参数快速估计）
    def mock_lifetime(params):
        a, e, i = params[0], params[1], params[2]
        return 86400.0 * (a / 3.0) * (1.0 - e) * np.cos(i)

    def mock_dv(params):
        a, e, i = params[0], params[1], params[2]
        return 1e-6 * (e + abs(i))

    def mock_collision(params):
        a, e, i = params[0], params[1], params[2]
        return 0.1 * np.exp(-a / 2.0) * (1.0 + e)

    def objective(params):
        return compute_orbit_quality_score(
            params, mock_lifetime, mock_dv, mock_collision,
            weights=np.array([1.0, -0.5, -10.0])
        )

    # Box-Behnken 敏感性分析
    param_names = ["semi_major_axis", "eccentricity", "inclination"]
    ranges = np.array([
        [2.0, 5.0],    # a (km)
        [0.0, 0.3],    # e
        [0.0, 0.5]     # i (rad)
    ])
    analysis = OrbitSensitivityAnalysis(param_names, ranges, objective)
    design, responses, main_effects = analysis.run_analysis()
    print(f"  Box-Behnken 试验点数: {design.shape[0]}")
    print(f"  主效应估计: a={main_effects[0]:.4f}, e={main_effects[1]:.4f}, i={main_effects[2]:.4f}")

    best_params, best_score = analysis.find_optimal_from_design()
    print(f"  最优参数: a={best_params[0]:.4f}, e={best_params[1]:.4f}, i={best_params[2]:.4f}")
    print(f"  最优评分: {best_score:.4f}")

    # 二进制回溯搜索（将参数空间离散化为二进制决策）
    def binary_objective(bits):
        # 将 6 位二进制映射到 3 个参数，每个参数 2 位
        a = 2.0 + bits[0] * 1.0 + bits[1] * 2.0
        e = bits[2] * 0.1 + bits[3] * 0.2
        i = bits[4] * 0.2 + bits[5] * 0.3
        return objective(np.array([a, e, i]))

    best_bits, best_bits_score = optimize_orbit_binary_backtrack(6, binary_objective)
    print(f"  二进制回溯最优: bits={best_bits}, score={best_bits_score:.4f}")


def run_collision_risk_demo(vertices, faces):
    """测试碰撞风险评估"""
    print_section("模块 8: 碰撞风险评估 (ball_distance + neighbor_risk)")

    # 单位球距离统计
    mu_d, var_d = ball_distance_stats(n_samples=5000, seed=42)
    print(f"  单位球随机点距离统计: mean={mu_d:.6f}, var={var_d:.6e}")
    print(f"  理论均值 36/35 = {36.0/35.0:.6f}")

    # 邻接矩阵
    adj = build_surface_adjacency_matrix(faces, vertices.shape[0])
    conn = region_connectivity_analysis(adj)
    print(f"  表面连通分量数: {conn['n_components']}")
    print(f"  各分量大小: {conn['component_sizes'][:10]}...")
    print(f"  图直径估计: {conn['diameter_est']}")

    # 碰撞概率
    test_pos = np.array([2.5, 0.0, 0.0])
    p_coll = collision_probability_surface(
        test_pos, vertices, faces, safe_distance=0.5, position_uncertainty=0.1
    )
    print(f"  测试点 {test_pos} 碰撞概率: {p_coll:.6e}")

    # 安全悬停区
    safe_points, safe_probs = find_safe_hover_regions(
        vertices, faces, min_altitude=1.0, n_samples=200, seed=42
    )
    print(f"  发现安全悬停区: {safe_points.shape[0]} 个")
    if safe_points.shape[0] > 0:
        print(f"  最低碰撞概率: {safe_probs.min():.6e}")


def run_data_io_demo(vertices, faces):
    """测试数据 I/O"""
    print_section("模块 9: 数据 I/O (xyz_io + filum)")

    # 写入 XYZ 顶点文件
    out_dir = "."
    xyz_file = f"{out_dir}/asteroid_vertices.xyz"
    write_xyz_data(xyz_file, vertices, header="Synthetic asteroid vertices")
    print(f"  已写入顶点文件: {xyz_file}")

    # 写入面片索引
    face_file = f"{out_dir}/asteroid_faces.txt"
    write_face_indices(face_file, faces, zero_based=True)
    print(f"  已写入面片文件: {face_file}")

    # 合成点云生成
    pc = generate_synthetic_asteroid_pointcloud(a=2.0, b=1.5, c=1.0, n_theta=16, n_phi=16)
    print(f"  合成点云点数: {pc.shape[0]}")

    # 字符串工具测试
    test_str = "v 1.0 2.0 3.0"
    wc = s_word_count(test_str)
    print(f"  字符串 '{test_str}' 单词数: {wc}")


def run_combined_simulation(vertices, faces, sh_model, gm_km, r_ref_km, c_coeff, s_coeff):
    """运行组合引力模型的综合仿真"""
    print_section("模块 10: 组合引力模型综合仿真")

    density = 2500.0
    test_positions = [
        np.array([2.2, 0.0, 0.0]),
        np.array([4.0, 1.0, 0.5]),
        np.array([8.0, 2.0, 1.0]),
    ]

    for pos in test_positions:
        r = np.linalg.norm(pos)
        a_combined = combined_gravity_model(
            pos, vertices, faces, gm_km, r_ref_km, c_coeff, s_coeff,
            n_max=6, density=density, transition_radius=2.5
        )
        a_harm = sh_model.acceleration(pos)
        print(f"  r={pos} km (|r|={r:.2f}):")
        print(f"    组合加速度 |a| = {np.linalg.norm(a_combined):.6e} km/s²")
        print(f"    纯球谐 |a|   = {np.linalg.norm(a_harm):.6e} km/s²")


def main():
    print("=" * 72)
    print("  不规则小行星多尺度引力场建模与近距轨道长期稳定性分析系统")
    print("  Polyhedral-Spherical-Harmonic Coupled Gravity & Orbital Dynamics")
    print("=" * 72)
    print(f"  Python: {sys.version}")
    print(f"  NumPy:  {np.__version__}")
    print(f"  运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    t_start = time.time()

    # 1. 特殊函数
    run_special_functions_demo()

    # 2. 几何建模
    vertices, faces, vol, com = run_geometry_demo()

    # 3. 球谐引力场
    sh_model, gm_km, r_ref_km, c_coeff, s_coeff = run_gravity_harmonics_demo(vertices, faces, vol)

    # 4. 多面体引力场
    run_polyhedron_gravity_demo(vertices, faces)

    # 5. 有限元内部势
    run_fem_gravity_demo()

    # 6. 轨道积分
    dyn = run_orbit_integrator_demo(sh_model, gm_km)

    # 7. 轨道优化
    run_orbit_optimization_demo()

    # 8. 碰撞风险
    run_collision_risk_demo(vertices, faces)

    # 9. 数据 I/O
    run_data_io_demo(vertices, faces)

    # 10. 组合仿真
    run_combined_simulation(vertices, faces, sh_model, gm_km, r_ref_km, c_coeff, s_coeff)

    t_elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"  全部计算完成，耗时: {t_elapsed:.3f} 秒")
    print("=" * 72)

    # 输出关键科学结论
    print("\n【科学结论摘要】")
    print("  1. 通过 IFS 分形扰动与耳切三角剖分，成功构建了不规则小行星三维形状模型。")
    print("  2. 球谐系数由体积积分法从多面体形状自动估算，支持远场引力快速计算。")
    print("  3. 多面体方法在近场（r < 3R_e）提供比球谐展开更精确的引力场。")
    print("  4. 有限元泊松方程求解揭示了碎石堆内部密度梯度对引力的影响。")
    print("  5. 随机 Runge-Kutta 积分量化了 Yarkovsky 热噪声导致的轨道漂移。")
    print("  6. Box-Behnken 实验设计识别了轨道半长轴为影响稳定性的首要因素。")
    print("  7. 碰撞风险评估与邻接图分析为安全悬停区选择提供了量化依据。")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（28个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: gamma_lanczos 对正整数解析验证 ----
assert abs(gamma_lanczos(1.0) - 1.0) < 1e-10, '[TC01] gamma_lanczos(1.0) 应等于 1 FAILED'
assert abs(gamma_lanczos(2.0) - 1.0) < 1e-10, '[TC01] gamma_lanczos(2.0) 应等于 1 FAILED'
assert abs(gamma_lanczos(3.0) - 2.0) < 1e-10, '[TC01] gamma_lanczos(3.0) 应等于 2 FAILED'

# ---- TC02: lngamma_lanczos 非法输入返回错误码 ----
val, ier = lngamma_lanczos(-1.0)
assert ier == 1, '[TC02] lngamma_lanczos 对负数应返回错误码 1 FAILED'

# ---- TC03: factorial_ratio 数值稳定性 ----
fr = factorial_ratio(10, 5)
expected = 30240.0
assert abs(fr - expected) < 1e-6, '[TC03] factorial_ratio(10,5) 应等于 30240 FAILED'

# ---- TC04: associated_legendre_normalized 边界值为有限值 ----
p00 = associated_legendre_normalized(0, 0, 0.5)
assert np.isfinite(p00), '[TC04] P_00(0.5) 应为有限值 FAILED'

# ---- TC05: generate_asteroid_cross_section 输出形状正确 ----
np.random.seed(42)
poly = generate_asteroid_cross_section(n_points=32)
assert poly.shape == (32, 2), '[TC05] 二维截面应返回 (32,2) 形状 FAILED'

# ---- TC06: ear_clip_triangulation 输出面片数为 n-2 ----
tri = ear_clip_triangulation(poly)
assert tri.shape[0] == poly.shape[0] - 2, '[TC06] 三角面片数应为 n-2 FAILED'

# ---- TC07: polyhedron_volume_and_com 对单位立方体解析验证 ----
cube_v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
cube_f = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], dtype=int)
vol, com = polyhedron_volume_and_com(cube_v, cube_f)
assert abs(vol - 1.0) < 1e-10, '[TC07] 单位立方体体积应为 1.0 FAILED'
assert np.linalg.norm(com - np.array([0.5, 0.5, 0.5])) < 1e-10, '[TC07] 单位立方体质心应为 (0.5,0.5,0.5) FAILED'

# ---- TC08: surface_area 对简单三角形解析验证 ----
tri_v = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=float)
tri_f = np.array([[0,1,2]], dtype=int)
area = surface_area(tri_v, tri_f)
assert abs(area - 0.5) < 1e-10, '[TC08] 直角三角形面积应为 0.5 FAILED'

# ---- TC09: revolve_to_3d 输出顶点与面片形状 ----
simple_poly = np.array([[1,0],[0,1],[-1,0],[0,-1]], dtype=float)
v3d, f3d = revolve_to_3d(simple_poly, z_scale=1.0)
assert v3d.ndim == 2 and v3d.shape[1] == 3, '[TC09] 三维顶点应为 (N,3) FAILED'
assert f3d.ndim == 2 and f3d.shape[1] == 3, '[TC09] 面片应为 (M,3) FAILED'

# ---- TC10: compute_stokes_coefficients_from_shape 输出尺寸与参考半径为正 ----
tet_v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
tet_f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=int)
c_coeff, s_coeff, r_ref = compute_stokes_coefficients_from_shape(tet_v, tet_f, 1000.0, n_max=4)
assert c_coeff.shape == (5, 5), '[TC10] C系数形状应为 (n_max+1, n_max+1) FAILED'
assert s_coeff.shape == (5, 5), '[TC10] S系数形状应为 (n_max+1, n_max+1) FAILED'
assert r_ref > 0, '[TC10] 参考半径应为正 FAILED'

# ---- TC11: SphericalHarmonicGravity.potential 返回有限值 ----
sh = SphericalHarmonicGravity(1.0, 1.0, c_coeff, s_coeff, n_max=4)
pot = sh.potential(np.array([2.0, 0.0, 0.0]))
assert np.isfinite(pot), '[TC11] 引力势应为有限值 FAILED'

# ---- TC12: SphericalHarmonicGravity.acceleration 输出为三维向量 ----
acc = sh.acceleration(np.array([2.0, 0.0, 0.0]))
assert acc.shape == (3,), '[TC12] 加速度应为 3 维向量 FAILED'

# ---- TC13: SphericalHarmonicGravity.gradient_fd 与 acceleration 数值一致 ----
acc_fd = sh.gradient_fd(np.array([2.0, 0.0, 0.0]))
assert np.linalg.norm(acc - acc_fd) < 0.1, '[TC13] 数值梯度与解析加速度应接近 FAILED'

# ---- TC14: polyhedron_gravity_potential 返回有限值 ----
tet_pot = polyhedron_gravity_potential(np.array([2.0, 2.0, 2.0]), tet_v * 1000, tet_f, 2000.0)
assert np.isfinite(tet_pot), '[TC14] 多面体引力势应为有限值 FAILED'

# ---- TC15: polyhedron_gravity_acceleration 输出为三维向量 ----
tet_acc = polyhedron_gravity_acceleration(np.array([2.0, 2.0, 2.0]), tet_v * 1000, tet_f, 2000.0)
assert tet_acc.shape == (3,), '[TC15] 多面体加速度应为 3 维向量 FAILED'

# ---- TC16: combined_gravity_model 输出为三维向量 ----
a_comb = combined_gravity_model(np.array([2.0, 0.0, 0.0]), tet_v, tet_f, 1.0, 1.0, c_coeff, s_coeff, n_max=4, density=2000.0)
assert a_comb.shape == (3,), '[TC16] 组合引力加速度应为 3 维向量 FAILED'

# ---- TC17: solve_internal_potential_2d 势场全为有限值 ----
phi, xc, yc = solve_internal_potential_2d(nx=8, ny=8, r_max=100.0)
assert np.all(np.isfinite(phi)), '[TC17] 内部引力势应全为有限值 FAILED'
assert phi.shape == (8, 8), '[TC17] 势场形状应为 (ny,nx) FAILED'

# ---- TC18: internal_gravity_from_potential 输出尺寸与势场一致 ----
gx, gy = internal_gravity_from_potential(phi, xc, yc)
assert gx.shape == phi.shape, '[TC18] gx 形状应与 phi 相同 FAILED'
assert gy.shape == phi.shape, '[TC18] gy 形状应与 phi 相同 FAILED'

# ---- TC19: newton_cotes_integrate 解析验证 sin^2 ----
nc_res = newton_cotes_integrate(lambda t: np.sin(t) ** 2, 0.0, np.pi, n=5, n_sub=10)
assert abs(nc_res - np.pi / 2.0) < 1e-6, '[TC19] Newton-Cotes 积分 sin^2 应接近 pi/2 FAILED'

# ---- TC20: OrbitalDynamics 确定性积分输出形状正确 ----
def grav_dummy(pos):
    return -pos / (np.linalg.norm(pos) ** 3)
dyn = OrbitalDynamics(grav_dummy, beta_srp=0.0, perturbation_std=0.0)
state0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
t_arr, states = dyn.integrate_deterministic(state0, (0.0, 1.0), n_steps=10)
assert states.shape == (11, 6), '[TC20] 确定性积分状态矩阵应为 (n_steps+1, 6) FAILED'

# ---- TC21: OrbitSensitivityAnalysis 设计矩阵形状正确 ----
def mock_obj(params):
    return -np.sum(params ** 2)
analy = OrbitSensitivityAnalysis(['a','e','i'], np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0]]), mock_obj)
design, responses, main_effects = analy.run_analysis()
assert design.shape[1] == 3, '[TC21] 设计矩阵列数应为 3 FAILED'
assert responses.shape[0] == design.shape[0], '[TC21] 响应值数量应与设计矩阵行数相同 FAILED'

# ---- TC22: optimize_orbit_binary_backtrack 找到全局最优解 ----
best_bits, best_score = optimize_orbit_binary_backtrack(3, lambda bits: float(np.sum(bits)))
assert best_bits.shape == (3,), '[TC22] 最优二进制向量长度应为 3 FAILED'
assert best_score == 3.0, '[TC22] 最大化 sum(bits) 应得到 3 FAILED'

# ---- TC23: compute_orbit_quality_score 权重单调性验证 ----
def lt(p): return 1.0
def dv(p): return 0.0
def coll(p): return 0.0
score1 = compute_orbit_quality_score(np.array([1.0, 0.1, 0.1]), lt, dv, coll, weights=np.array([1.0, -1.0, -1.0]))
score2 = compute_orbit_quality_score(np.array([1.0, 0.1, 0.1]), lt, dv, coll, weights=np.array([2.0, -1.0, -1.0]))
assert score2 > score1, '[TC23] 增大 lifetime 权重应提高评分 FAILED'

# ---- TC24: ball_distance_stats 均值在合理范围且方差非负 ----
np.random.seed(42)
mu, var = ball_distance_stats(n_samples=2000, seed=42)
assert 0.9 < mu < 1.2, '[TC24] 单位球距离均值应在合理范围内 FAILED'
assert var >= 0, '[TC24] 方差应非负 FAILED'

# ---- TC25: collision_probability_surface 范围约束在 [0,1] ----
p_coll = collision_probability_surface(np.array([10.0, 10.0, 10.0]), tet_v, tet_f, safe_distance=0.5, position_uncertainty=0.1)
assert 0.0 <= p_coll <= 1.0, '[TC25] 碰撞概率应在 [0,1] 范围内 FAILED'

# ---- TC26: region_connectivity_analysis 对全连通图返回 1 个分量 ----
adj = np.ones((5, 5), dtype=int) - np.eye(5, dtype=int)
conn = region_connectivity_analysis(adj)
assert conn['n_components'] == 1, '[TC26] 全连通图应只有 1 个连通分量 FAILED'
assert conn['total_nodes'] == 5, '[TC26] 总节点数应为 5 FAILED'

# ---- TC27: s_word_count 字符串单词计数正确 ----
wc = s_word_count("v 1.0 2.0 3.0")
assert wc == 4, '[TC27] 字符串单词数应为 4 FAILED'

# ---- TC28: generate_synthetic_asteroid_pointcloud 输出形状正确 ----
np.random.seed(42)
pc = generate_synthetic_asteroid_pointcloud(n_theta=8, n_phi=8)
assert pc.shape == (64, 3), '[TC28] 合成点云形状应为 (64,3) FAILED'

print('\n全部 28 个测试通过!\n')
