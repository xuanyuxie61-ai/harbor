#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_main.py — Harbor 测试脚本

从 /tests/ 目录运行，测试 /app/workspace/ 中的科学计算代码。
通过 sys.path 引入 workspace 模块，使用绝对路径，不依赖相对路径。
"""

import sys
import os
import json

# === 路径设置：确保可以导入 /app/workspace 中的模块 ===
WORKSPACE_DIR = "/app/workspace"
TESTS_DIR = "/tests"
LOGS_DIR = "/logs/verifier"

if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

# 创建日志输出目录
os.makedirs(LOGS_DIR, exist_ok=True)

import numpy as np

# === 从 /app/workspace 导入所有模块 ===
from special_functions import (
    lngamma_lanczos,
    gamma_lanczos,
    associated_legendre_normalized,
    factorial_ratio
)
from asteroid_geometry import (
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


def run_all_tests():
    """运行全部 28 个测试用例，返回 (通过数, 总数, 失败详情)"""
    passed = 0
    total = 28
    failures = []

    # ---- TC01: gamma_lanczos 对正整数解析验证 ----
    try:
        assert abs(gamma_lanczos(1.0) - 1.0) < 1e-10
        assert abs(gamma_lanczos(2.0) - 1.0) < 1e-10
        assert abs(gamma_lanczos(3.0) - 2.0) < 1e-10
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC01] gamma_lanczos 解析验证 FAILED: {e}")

    # ---- TC02: lngamma_lanczos 非法输入返回错误码 ----
    try:
        val, ier = lngamma_lanczos(-1.0)
        assert ier == 1
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC02] lngamma_lanczos 错误码 FAILED: {e}")

    # ---- TC03: factorial_ratio 数值稳定性 ----
    try:
        fr = factorial_ratio(10, 5)
        expected = 30240.0
        assert abs(fr - expected) < 1e-6
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC03] factorial_ratio FAILED: {e}")

    # ---- TC04: associated_legendre_normalized 边界值为有限值 ----
    try:
        p00 = associated_legendre_normalized(0, 0, 0.5)
        assert np.isfinite(p00)
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC04] Legendre 边界值 FAILED: {e}")

    # ---- TC05: generate_asteroid_cross_section 输出形状正确 ----
    try:
        np.random.seed(42)
        poly = generate_asteroid_cross_section(n_points=32)
        assert poly.shape == (32, 2)
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC05] 截面形状 FAILED: {e}")

    # ---- TC06: ear_clip_triangulation 输出面片数为 n-2 ----
    try:
        tri = ear_clip_triangulation(poly)
        assert tri.shape[0] == poly.shape[0] - 2
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC06] 三角剖分 FAILED: {e}")

    # ---- TC07: polyhedron_volume_and_com 对单位立方体解析验证 ----
    try:
        cube_v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
        cube_f = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], dtype=int)
        vol, com = polyhedron_volume_and_com(cube_v, cube_f)
        assert abs(vol - 1.0) < 1e-10
        assert np.linalg.norm(com - np.array([0.5, 0.5, 0.5])) < 1e-10
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC07] 多面体体积/质心 FAILED: {e}")

    # ---- TC08: surface_area 对简单三角形解析验证 ----
    try:
        tri_v = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=float)
        tri_f = np.array([[0,1,2]], dtype=int)
        area = surface_area(tri_v, tri_f)
        assert abs(area - 0.5) < 1e-10
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC08] 表面积 FAILED: {e}")

    # ---- TC09: revolve_to_3d 输出顶点与面片形状 ----
    try:
        simple_poly = np.array([[1,0],[0,1],[-1,0],[0,-1]], dtype=float)
        v3d, f3d = revolve_to_3d(simple_poly, z_scale=1.0)
        assert v3d.ndim == 2 and v3d.shape[1] == 3
        assert f3d.ndim == 2 and f3d.shape[1] == 3
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC09] 3D 旋转 FAILED: {e}")

    # ---- TC10: compute_stokes_coefficients_from_shape 输出尺寸与参考半径为正 ----
    try:
        tet_v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        tet_f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=int)
        c_coeff, s_coeff, r_ref = compute_stokes_coefficients_from_shape(tet_v, tet_f, 1000.0, n_max=4)
        assert c_coeff.shape == (5, 5)
        assert s_coeff.shape == (5, 5)
        assert r_ref > 0
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC10] Stokes 系数 FAILED: {e}")

    # ---- TC11: SphericalHarmonicGravity.potential 返回有限值 ----
    try:
        sh = SphericalHarmonicGravity(1.0, 1.0, c_coeff, s_coeff, n_max=4)
        pot = sh.potential(np.array([2.0, 0.0, 0.0]))
        assert np.isfinite(pot)
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC11] 球谐引力势 FAILED: {e}")

    # ---- TC12: SphericalHarmonicGravity.acceleration 输出为三维向量 ----
    try:
        acc = sh.acceleration(np.array([2.0, 0.0, 0.0]))
        assert acc.shape == (3,)
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC12] 球谐加速度 FAILED: {e}")

    # ---- TC13: SphericalHarmonicGravity.gradient_fd 与 acceleration 数值一致 ----
    try:
        acc_fd = sh.gradient_fd(np.array([2.0, 0.0, 0.0]))
        assert np.linalg.norm(acc - acc_fd) < 0.1
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC13] 球谐梯度 FAILED: {e}")

    # ---- TC14: polyhedron_gravity_potential 返回有限值 ----
    try:
        tet_pot = polyhedron_gravity_potential(np.array([2.0, 2.0, 2.0]), tet_v * 1000, tet_f, 2000.0)
        assert np.isfinite(tet_pot)
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC14] 多面体引力势 FAILED: {e}")

    # ---- TC15: polyhedron_gravity_acceleration 输出为三维向量 ----
    try:
        tet_acc = polyhedron_gravity_acceleration(np.array([2.0, 2.0, 2.0]), tet_v * 1000, tet_f, 2000.0)
        assert tet_acc.shape == (3,)
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC15] 多面体加速度 FAILED: {e}")

    # ---- TC16: combined_gravity_model 输出为三维向量 ----
    try:
        a_comb = combined_gravity_model(np.array([2.0, 0.0, 0.0]), tet_v, tet_f, 1.0, 1.0, c_coeff, s_coeff, n_max=4, density=2000.0)
        assert a_comb.shape == (3,)
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC16] 组合引力 FAILED: {e}")

    # ---- TC17: solve_internal_potential_2d 势场全为有限值 ----
    try:
        phi, xc, yc = solve_internal_potential_2d(nx=8, ny=8, r_max=100.0)
        assert np.all(np.isfinite(phi))
        assert phi.shape == (8, 8)
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC17] FEM 势场 FAILED: {e}")

    # ---- TC18: internal_gravity_from_potential 输出尺寸与势场一致 ----
    try:
        gx, gy = internal_gravity_from_potential(phi, xc, yc)
        assert gx.shape == phi.shape
        assert gy.shape == phi.shape
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC18] FEM 引力 FAILED: {e}")

    # ---- TC19: newton_cotes_integrate 解析验证 sin^2 ----
    try:
        nc_res = newton_cotes_integrate(lambda t: np.sin(t) ** 2, 0.0, np.pi, n=5, n_sub=10)
        assert abs(nc_res - np.pi / 2.0) < 1e-6
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC19] Newton-Cotes FAILED: {e}")

    # ---- TC20: OrbitalDynamics 确定性积分输出形状正确 ----
    try:
        def grav_dummy(pos):
            return -pos / (np.linalg.norm(pos) ** 3)
        dyn = OrbitalDynamics(grav_dummy, beta_srp=0.0, perturbation_std=0.0)
        state0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        t_arr, states = dyn.integrate_deterministic(state0, (0.0, 1.0), n_steps=10)
        assert states.shape == (11, 6)
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC20] 轨道积分 FAILED: {e}")

    # ---- TC21: OrbitSensitivityAnalysis 设计矩阵形状正确 ----
    try:
        def mock_obj(params):
            return -np.sum(params ** 2)
        analy = OrbitSensitivityAnalysis(['a','e','i'], np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0]]), mock_obj)
        design, responses, main_effects = analy.run_analysis()
        assert design.shape[1] == 3
        assert responses.shape[0] == design.shape[0]
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC21] Box-Behnken FAILED: {e}")

    # ---- TC22: optimize_orbit_binary_backtrack 找到全局最优解 ----
    try:
        best_bits, best_score = optimize_orbit_binary_backtrack(3, lambda bits: float(np.sum(bits)))
        assert best_bits.shape == (3,)
        assert best_score == 3.0
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC22] 二进制回溯 FAILED: {e}")

    # ---- TC23: compute_orbit_quality_score 权重单调性验证 ----
    try:
        def lt(p): return 1.0
        def dv(p): return 0.0
        def coll(p): return 0.0
        score1 = compute_orbit_quality_score(np.array([1.0, 0.1, 0.1]), lt, dv, coll, weights=np.array([1.0, -1.0, -1.0]))
        score2 = compute_orbit_quality_score(np.array([1.0, 0.1, 0.1]), lt, dv, coll, weights=np.array([2.0, -1.0, -1.0]))
        assert score2 > score1
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC23] 品质评分 FAILED: {e}")

    # ---- TC24: ball_distance_stats 均值在合理范围且方差非负 ----
    try:
        np.random.seed(42)
        mu, var = ball_distance_stats(n_samples=2000, seed=42)
        assert 0.9 < mu < 1.2
        assert var >= 0
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC24] 距离统计 FAILED: {e}")

    # ---- TC25: collision_probability_surface 范围约束在 [0,1] ----
    try:
        p_coll = collision_probability_surface(np.array([10.0, 10.0, 10.0]), tet_v, tet_f, safe_distance=0.5, position_uncertainty=0.1)
        assert 0.0 <= p_coll <= 1.0
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC25] 碰撞概率 FAILED: {e}")

    # ---- TC26: region_connectivity_analysis 对全连通图返回 1 个分量 ----
    try:
        adj = np.ones((5, 5), dtype=int) - np.eye(5, dtype=int)
        conn = region_connectivity_analysis(adj)
        assert conn['n_components'] == 1
        assert conn['total_nodes'] == 5
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC26] 连通性分析 FAILED: {e}")

    # ---- TC27: s_word_count 字符串单词计数正确 ----
    try:
        wc = s_word_count("v 1.0 2.0 3.0")
        assert wc == 4
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC27] 单词计数 FAILED: {e}")

    # ---- TC28: generate_synthetic_asteroid_pointcloud 输出形状正确 ----
    try:
        np.random.seed(42)
        pc = generate_synthetic_asteroid_pointcloud(n_theta=8, n_phi=8)
        assert pc.shape == (64, 3)
        passed += 1
    except AssertionError as e:
        failures.append(f"[TC28] 点云生成 FAILED: {e}")

    return passed, total, failures


def write_reward(passed: int, total: int, failures: list):
    """按照 Harbor 规范在 /logs/verifier/ 下生成 reward 文件"""
    score = passed / total if total > 0 else 0.0
    reward_data = {
        "score": round(score, 4),
        "passed": passed,
        "total": total,
        "status": "pass" if score == 1.0 else "partial",
        "failures": failures
    }

    # 写入 JSON 格式
    reward_json = os.path.join(LOGS_DIR, "reward.json")
    with open(reward_json, "w", encoding="utf-8") as f:
        json.dump(reward_data, f, indent=2, ensure_ascii=False)

    # 同时写入纯文本版本
    reward_txt = os.path.join(LOGS_DIR, "reward.txt")
    with open(reward_txt, "w", encoding="utf-8") as f:
        f.write(f"score={reward_data['score']}\n")
        f.write(f"passed={passed}\n")
        f.write(f"total={total}\n")
        f.write(f"status={reward_data['status']}\n")

    print(f"\n奖励文件已生成: {reward_json}")
    print(f"奖励文件已生成: {reward_txt}")


if __name__ == "__main__":
    print("=" * 72)
    print("  Harbor 测试: 不规则小行星多尺度引力场建模")
    print(f"  Workspace: {WORKSPACE_DIR}")
    print(f"  Tests:     {TESTS_DIR}")
    print(f"  Logs:      {LOGS_DIR}")
    print("=" * 72)

    passed, total, failures = run_all_tests()

    print(f"\n全部 {total} 个测试: {passed} 通过, {total - passed} 失败")
    if failures:
        for f in failures:
            print(f"  {f}")
    else:
        print("全部测试通过!")

    write_reward(passed, total, failures)
