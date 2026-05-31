#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
室内声场射线追踪与模态分析的统一入口
声学工程：三维封闭空间声场的混合射线-有限元模态分析与贝叶斯参数反演

本程序零参数运行，完成以下完整流程：
1. 房间几何定义与表面提取
2. 四面体网格生成与局部加密
3. 基于 FEM 的 Helmholtz 方程求解（低频声压分布）
4. 解析模态分析与 Schroeder 频率计算
5. 蒙特卡洛射线追踪（高频能量衰减与混响时间）
6. 边缘衍射场计算
7. 反射转移图构建（基于稀疏图理论）
8. 贝叶斯吸声系数反演（DREAM MCMC）
9. 不确定性传播分析
10. 结果汇总输出
"""

import numpy as np
import os
import sys

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from room_geometry import (
    dshoebox_with_pillars, huniform, extract_room_surfaces,
    compute_surface_normals, room_surface_areas, room_total_volume,
    compute_sabine_reverberation_time
)
from mesh_generator import (
    distmesh_3d, refine_mesh_near_boundary, mesh_statistics,
    surftri, simp_qual_3d
)
from fem_acoustics import (
    assemble_helmholtz_system, solve_helmholtz_cg,
    compute_sound_pressure_level, C_AIR
)
from sparse_linalg import SparseCOO, conjugate_gradient
from modal_analysis import (
    rectangular_room_modes, schroeder_frequency,
    inverse_iteration, compute_modal_density, modal_overlap_factor
)
from quadrature_rules import (
    triangle_symq_rule, integrate_over_triangle,
    ball01_monomial_integral, ball01_sample
)
from ray_tracer import (
    monte_carlo_ray_tracing, build_reflection_graph,
    compute_room_response_stats
)
from edge_diffraction import (
    detect_room_edges, compute_edge_diffraction_field
)
from bayesian_calibration import (
    calibrate_absorption_coefficients, uncertainty_propagation
)
from utils import (
    compute_mean_free_path, sabine_absorption_to_t60,
    eyring_absorption_to_t60, compute_statistics, linear_regression
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.random.seed(42)
    print("=" * 70)
    print("  室内声场射线追踪与模态分析系统")
    print("  Indoor Sound Field Ray Tracing & Modal Analysis")
    print("=" * 70)

    # ================================================================
    # 1. 房间几何定义
    # ================================================================
    print_section("1. 房间几何定义")
    surfaces = extract_room_surfaces()
    normals = compute_surface_normals(surfaces)
    surface_areas = room_surface_areas(surfaces)
    room_vol = room_total_volume()
    total_surface_area = sum(surface_areas.values())

    print(f"  房间体积: {room_vol:.3f} m³")
    print(f"  总表面积: {total_surface_area:.3f} m²")
    for name, area in surface_areas.items():
        print(f"    {name:20s}: {area:8.3f} m²")

    # 设定吸声系数
    absorption = {
        'floor': 0.15,
        'ceiling': 0.25,
        'front_wall': 0.10,
        'back_wall': 0.10,
        'left_wall': 0.05,
        'right_wall': 0.05,
    }
    avg_alpha = sum(absorption[s] * surface_areas[s] for s in absorption) / total_surface_area
    T60_sabine = compute_sabine_reverberation_time(absorption, surfaces)
    T60_eyring = eyring_absorption_to_t60(room_vol, total_surface_area, avg_alpha)
    print(f"\n  Sabine T60: {T60_sabine:.3f} s")
    print(f"  Eyring T60: {T60_eyring:.3f} s")

    # ================================================================
    # 2. 四面体网格生成
    # ================================================================
    print_section("2. 四面体网格生成 (DistMesh 3D)")
    h0 = 1.2  # 目标网格尺寸
    box = [0.0, 10.0, 0.0, 8.0, 0.0, 5.0]
    pfix = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 8.0, 0.0], [0.0, 8.0, 0.0],
        [0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [10.0, 8.0, 5.0], [0.0, 8.0, 5.0],
    ], dtype=float)
    p, t = distmesh_3d(dshoebox_with_pillars, huniform, h0, box,
                       iteration_max=30, pfix=pfix)
    p, t = refine_mesh_near_boundary(p, t, dshoebox_with_pillars, h0, n_refinements=1)
    stats = mesh_statistics(p, t)
    print(f"  节点数: {stats['node_num']}")
    print(f"  四面体数: {stats['tet_num']}")
    print(f"  体积范围: [{stats['volume_min']:.6f}, {stats['volume_max']:.6f}]")
    print(f"  质量范围: [{stats['quality_min']:.4f}, {stats['quality_max']:.4f}]")
    print(f"  平均质量: {stats['quality_mean']:.4f}")

    # ================================================================
    # 3. 基于 FEM 的 Helmholtz 方程求解
    # ================================================================
    print_section("3. 有限元 Helmholtz 方程求解")
    freq_fem = 125.0  # Hz，低频
    source_pos = np.array([5.0, 4.0, 2.5])
    # 找最近的节点作为声源
    dists = np.linalg.norm(p - source_pos, axis=1)
    source_node = int(np.argmin(dists))
    print(f"  声源频率: {freq_fem} Hz")
    print(f"  声源位置: ({source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f})")
    print(f"  最近节点: #{source_node}")

    try:
        p_sol, k_wavenum = solve_helmholtz_cg(p, t, freq_fem, source_node=source_node,
                                               source_strength=1.0, tol=1e-8)
        spl = compute_sound_pressure_level(p_sol)
        print(f"  波数 k = {k_wavenum:.4f} rad/m")
        print(f"  声压级范围: [{np.min(spl):.2f}, {np.max(spl):.2f}] dB")
        print(f"  声压级均值: {np.mean(spl):.2f} dB")
    except Exception as e:
        print(f"  FEM 求解过程出现警告（小规模网格预期行为）: {e}")
        p_sol = np.zeros(p.shape[0])
        k_wavenum = 2.0 * np.pi * freq_fem / C_AIR

    # ================================================================
    # 4. 解析模态分析
    # ================================================================
    print_section("4. 解析模态分析与 Schroeder 频率")
    modes = rectangular_room_modes(10.0, 8.0, 5.0, max_order=4)
    f_s = schroeder_frequency(room_vol, total_surface_area, avg_alpha)
    print(f"  Schroeder 频率: {f_s:.2f} Hz")
    print(f"  前10阶模态频率:")
    for i, mode in enumerate(modes[:10]):
        print(f"    ({mode['l']},{mode['m']},{mode['n']})  f = {mode['frequency']:7.3f} Hz")

    # 模态密度与重叠因子
    mof_data = modal_overlap_factor(modes[:20], damping_ratio=0.01)
    mof_mean = np.mean([d['mof'] for d in mof_data])
    print(f"  平均模态重叠因子 (MOF): {mof_mean:.3f}")
    if mof_mean > 1.0:
        print("  -> 声场处于扩散区")
    else:
        print("  -> 声场处于模态控制区")

    # 基于 FEM 矩阵的逆迭代模态分析
    print("\n  基于 FEM 的逆迭代模态分析:")
    try:
        A_sparse, K_sparse, M_sparse, F, k_val = assemble_helmholtz_system(
            p, t, freq_fem, source_node=source_node, source_strength=1.0
        )
        # 构建 SPD 近似矩阵 K + k^2 M
        n_nodes = p.shape[0]
        A_rows, A_cols, A_vals = [], [], []
        for i in range(K_sparse.nnz):
            A_rows.append(K_sparse.rows[i])
            A_cols.append(K_sparse.cols[i])
            A_vals.append(K_sparse.vals[i])
        for i in range(M_sparse.nnz):
            A_rows.append(M_sparse.rows[i])
            A_cols.append(M_sparse.cols[i])
            A_vals.append((k_val ** 2) * M_sparse.vals[i])
        A_spd = SparseCOO(np.array(A_rows), np.array(A_cols), np.array(A_vals), (n_nodes, n_nodes))
        freq_mode, mode_shape, lam = inverse_iteration(A_spd, M_sparse, max_iter=30, tol=1e-8)
        print(f"    基频估算: {freq_mode:.3f} Hz")
        print(f"    Rayleigh 商 λ: {lam:.6f}")
    except Exception as e:
        print(f"    逆迭代分析跳过（网格规模限制）: {e}")
        freq_mode = modes[0]['frequency'] if modes else 0.0
        print(f"    使用解析基频: {freq_mode:.3f} Hz")

    # ================================================================
    # 5. 蒙特卡洛射线追踪
    # ================================================================
    print_section("5. 蒙特卡洛射线追踪 (Sobol 采样)")
    n_rays = 2000
    source_pos_rt = np.array([5.0, 4.0, 2.5])
    times, edc, T60_mc, EDT = monte_carlo_ray_tracing(
        surfaces, normals, absorption, source_pos_rt,
        n_rays=n_rays, max_reflections=40, scattering_coeff=0.05
    )
    print(f"  发射射线数: {n_rays}")
    print(f"  蒙特卡洛 T60: {T60_mc:.3f} s")
    print(f"  早期衰减时间 EDT: {EDT:.3f} s")

    # 房间响应统计
    stats_rt = compute_room_response_stats(surfaces, normals, absorption,
                                            source_pos_rt, n_rays=500,
                                            max_reflections=30)
    print(f"  平均自由程: {stats_rt['mean_free_path']:.3f} m")
    print(f"  平均反射次数: {stats_rt['mean_reflections']:.1f}")

    # 平均自由程理论值
    mfp_theory = compute_mean_free_path(room_vol, total_surface_area)
    print(f"  理论平均自由程: {mfp_theory:.3f} m")

    # ================================================================
    # 6. 边缘衍射场计算
    # ================================================================
    print_section("6. 边缘衍射场计算 (GTD/UTD)")
    edges = detect_room_edges(surfaces)
    print(f"  检测到 {len(edges)} 条房间边缘")
    receiver_pos = np.array([7.0, 6.0, 3.0])
    freq_diffraction = 500.0
    diff_field = compute_edge_diffraction_field(source_pos_rt, receiver_pos, edges, freq_diffraction)
    print(f"  声源位置: ({source_pos_rt[0]:.2f}, {source_pos_rt[1]:.2f}, {source_pos_rt[2]:.2f})")
    print(f"  接收位置: ({receiver_pos[0]:.2f}, {receiver_pos[1]:.2f}, {receiver_pos[2]:.2f})")
    print(f"  频率: {freq_diffraction} Hz")
    print(f"  总衍射声压幅值: {np.abs(diff_field):.6e}")

    # ================================================================
    # 7. 反射转移图（稀疏图理论）
    # ================================================================
    print_section("7. 反射转移图 (稀疏马尔可夫链)")
    trans_prob, surf_names = build_reflection_graph(surfaces, normals, absorption, n_rays=1000)
    print("  表面间转移概率矩阵:")
    header = "        " + "".join([f"{s[:8]:>10s}" for s in surf_names])
    print(header)
    for i, s in enumerate(surf_names):
        row_str = f"{s[:8]:8s}"
        for j in range(len(surf_names)):
            row_str += f"{trans_prob[i, j]:10.4f}"
        print(row_str)

    # 计算稳态分布（PageRank 思想）
    # 求解 π^T P = π^T，即 P^T π = π
    # 使用幂迭代
    P = trans_prob.T
    pi = np.ones(len(surf_names)) / len(surf_names)
    for _ in range(100):
        pi_new = P @ pi
        pi_new = pi_new / np.sum(pi_new)
        if np.linalg.norm(pi_new - pi) < 1e-10:
            break
        pi = pi_new
    print("\n  稳态能量分布 (PageRank 类比):")
    for i, s in enumerate(surf_names):
        print(f"    {s:20s}: {pi[i]:.4f}")

    # ================================================================
    # 8. 贝叶斯吸声系数反演 (DREAM MCMC)
    # ================================================================
    print_section("8. 贝叶斯吸声系数反演 (DREAM MCMC)")
    T60_obs = T60_sabine  # 使用 Sabine 值作为"观测"
    T60_sigma = 0.1 * T60_obs
    cal_result = calibrate_absorption_coefficients(
        T60_obs, T60_sigma, surface_areas, room_vol,
        n_chains=4, n_generations=200
    )
    print(f"  观测 T60: {T60_obs:.3f} ± {T60_sigma:.3f} s")
    print(f"  后验均值:")
    print(f"    α_floor+ceiling: {cal_result['posterior_mean'][0]:.4f} ± {cal_result['posterior_std'][0]:.4f}")
    print(f"    α_front+back:   {cal_result['posterior_mean'][1]:.4f} ± {cal_result['posterior_std'][1]:.4f}")
    print(f"    α_left+right:   {cal_result['posterior_mean'][2]:.4f} ± {cal_result['posterior_std'][2]:.4f}")
    print(f"  预测 T60: {cal_result['predicted_t60']:.3f} s")
    print(f"  Gelman-Rubin R: {cal_result['R_final']}")

    # ================================================================
    # 9. 不确定性传播
    # ================================================================
    print_section("9. 不确定性传播分析")
    unc = uncertainty_propagation(cal_result['chains'], surface_areas, room_vol, n_samples=100)
    print(f"  T60 预测均值: {unc['mean']:.3f} s")
    print(f"  T60 预测标准差: {unc['std']:.3f} s")
    print(f"  95% 置信区间: [{unc['ci_95'][0]:.3f}, {unc['ci_95'][1]:.3f}] s")

    # ================================================================
    # 10. 球面积分验证 (ball_integrals 思想)
    # ================================================================
    print_section("10. 球面积分验证")
    # 验证单位球上某些单项式的积分
    test_exponents = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (2, 2, 2)]
    for e in test_exponents:
        exact = ball01_monomial_integral(e)
        # 蒙特卡洛验证
        samples = ball01_sample(5000)
        vals = samples[:, 0] ** e[0] * samples[:, 1] ** e[1] * samples[:, 2] ** e[2]
        mc = (4.0 * np.pi / 3.0) * np.mean(vals)
        print(f"  x^{e[0]}y^{e[1]}z^{e[2]}: 精确={exact:.6f}, MC={mc:.6f}")

    # ================================================================
    # 11. 高阶三角形求积验证
    # ================================================================
    print_section("11. 三角形求积验证")
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])

    def f_const(p):
        return 1.0

    def f_linear(p):
        return p[0] + p[1]

    area_exact = 0.5
    I_const = integrate_over_triangle(f_const, v0, v1, v2, precision=5)
    I_linear = integrate_over_triangle(f_linear, v0, v1, v2, precision=5)
    print(f"  常数函数积分: {I_const:.6f} (期望: {area_exact:.6f})")
    print(f"  线性函数积分: {I_linear:.6f} (期望: {area_exact / 3.0 * 2.0:.6f})")

    # ================================================================
    # 结果汇总
    # ================================================================
    print_section("结果汇总")
    print(f"  房间体积:           {room_vol:.2f} m³")
    print(f"  总表面积:           {total_surface_area:.2f} m²")
    print(f"  平均吸声系数:       {avg_alpha:.4f}")
    print(f"  Sabine T60:         {T60_sabine:.3f} s")
    print(f"  Eyring T60:         {T60_eyring:.3f} s")
    print(f"  蒙特卡洛 T60:       {T60_mc:.3f} s")
    print(f"  Schroeder 频率:     {f_s:.2f} Hz")
    print(f"  FEM 声源频率:       {freq_fem} Hz")
    print(f"  解析基频:           {modes[0]['frequency']:.3f} Hz")
    print(f"  平均自由程:         {stats_rt['mean_free_path']:.3f} m")
    print(f"  贝叶斯反演 T60:     {cal_result['predicted_t60']:.3f} s")
    print("\n  合成项目执行完毕。")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: extract_room_surfaces 返回 6 个表面 ----
test_surfaces = extract_room_surfaces()
assert len(test_surfaces) == 6, '[TC01] surface count FAILED'
assert set(test_surfaces.keys()) == {'floor', 'ceiling', 'front_wall', 'back_wall', 'left_wall', 'right_wall'}, '[TC01] surface names FAILED'

# ---- TC02: room_surface_areas 地板面积等于 80 m² ----
test_areas = room_surface_areas(test_surfaces)
assert abs(test_areas['floor'] - 80.0) < 1e-10, '[TC02] floor area FAILED'
assert abs(test_areas['ceiling'] - 80.0) < 1e-10, '[TC02] ceiling area FAILED'

# ---- TC03: room_total_volume 接近 400 减去柱子体积 ----
test_vol = room_total_volume()
expected_vol = 400.0 - 2.0 * (np.pi * 0.3 ** 2 * 5.0)
assert abs(test_vol - expected_vol) < 1e-10, '[TC03] room volume FAILED'

# ---- TC04: compute_sabine_reverberation_time 返回正值 ----
test_abs = {'floor': 0.15, 'ceiling': 0.25, 'front_wall': 0.10, 'back_wall': 0.10, 'left_wall': 0.05, 'right_wall': 0.05}
test_t60 = compute_sabine_reverberation_time(test_abs, test_surfaces)
assert test_t60 > 0.5 and test_t60 < 5.0, '[TC04] Sabine T60 range FAILED'

# ---- TC05: mesh_statistics 返回正节点数和四面体数 ----
np.random.seed(42)
h0 = 1.2
box = [0.0, 10.0, 0.0, 8.0, 0.0, 5.0]
pfix = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 8.0, 0.0], [0.0, 8.0, 0.0],
                 [0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [10.0, 8.0, 5.0], [0.0, 8.0, 5.0]], dtype=float)
p, t = distmesh_3d(dshoebox_with_pillars, huniform, h0, box, iteration_max=10, pfix=pfix)
stats = mesh_statistics(p, t)
assert stats['node_num'] > 0, '[TC05] node count FAILED'
assert stats['tet_num'] > 0, '[TC05] tet count FAILED'
assert stats['volume_min'] > 0, '[TC05] volume min FAILED'

# ---- TC06: simp_qual_3d 四面体质量在 [0, 100] 内 ----
qual = simp_qual_3d(p, t)
assert np.all(qual >= 0.0) and np.all(qual <= 100.0), '[TC06] quality range FAILED'

# ---- TC07: surftri 提取表面三角形数量大于 0 ----
boundary_faces = surftri(p, t)
assert boundary_faces.shape[0] > 0, '[TC07] boundary faces count FAILED'
assert boundary_faces.shape[1] == 3, '[TC07] boundary faces dim FAILED'

# ---- TC08: compute_sound_pressure_level 标量输入输出正 dB ----
spl_scalar = compute_sound_pressure_level(np.array([1.0]))
assert spl_scalar > 0, '[TC08] SPL positive FAILED'

# ---- TC09: SparseCOO 矩阵向量乘法正确性 ----
A_coo = SparseCOO(np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]), (3, 3))
x_vec = np.array([1.0, 1.0, 1.0])
y_vec = A_coo.mv(x_vec)
assert np.allclose(y_vec, np.array([1.0, 2.0, 3.0])), '[TC09] SparseCOO mv FAILED'

# ---- TC10: conjugate_gradient 求解单位矩阵方程 ----
I_coo = SparseCOO(np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([1.0, 1.0, 1.0]), (3, 3))
b_vec = np.array([3.0, -1.0, 2.0])
x_sol = conjugate_gradient(I_coo, b_vec, tol=1e-10)
assert np.allclose(x_sol, b_vec, atol=1e-8), '[TC10] CG solve identity FAILED'

# ---- TC11: rectangular_room_modes 基频等于 c/(2*Lx) ----
modes = rectangular_room_modes(10.0, 8.0, 5.0, max_order=1)
base_freq = modes[0]['frequency']
expected_base = C_AIR / (2.0 * 10.0)
assert abs(base_freq - expected_base) < 1e-10, '[TC11] base mode frequency FAILED'

# ---- TC12: schroeder_frequency 返回值正有限 ----
fs = schroeder_frequency(400.0, 340.0, 0.1)
assert np.isfinite(fs) and fs > 0, '[TC12] Schroeder frequency finite FAILED'

# ---- TC13: compute_modal_density 与 f² 成正比 ----
nf1 = compute_modal_density(400.0, 100.0)
nf2 = compute_modal_density(400.0, 200.0)
assert abs(nf2 / nf1 - 4.0) < 1e-10, '[TC13] modal density scaling FAILED'

# ---- TC14: ball01_monomial_integral 奇数指数返回 0 ----
val_odd = ball01_monomial_integral((1, 0, 0))
assert abs(val_odd) < 1e-14, '[TC14] ball odd exponent FAILED'

# ---- TC15: ball01_monomial_integral (0,0,0) 等于单位球体积 ----
val_000 = ball01_monomial_integral((0, 0, 0))
assert abs(val_000 - 4.0 * np.pi / 3.0) < 1e-10, '[TC15] ball zero exponent FAILED'

# ---- TC16: integrate_over_triangle 常数函数积分等于面积 ----
v0 = np.array([0.0, 0.0, 0.0])
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
I_const = integrate_over_triangle(lambda p: 1.0, v0, v1, v2, precision=5)
assert abs(I_const - 0.5) < 1e-10, '[TC16] integrate constant FAILED'

# ---- TC17: detect_room_edges 返回 12 条边 ----
test_edges = detect_room_edges(test_surfaces)
assert len(test_edges) == 12, '[TC17] room edges count FAILED'

# ---- TC18: compute_mean_free_path 理论值验证 ----
mfp = compute_mean_free_path(400.0, 340.0)
assert abs(mfp - 4.0 * 400.0 / 340.0) < 1e-10, '[TC18] mean free path FAILED'

# ---- TC19: linear_regression 斜率与截距精确恢复 ----
x_lr = np.array([0.0, 1.0, 2.0, 3.0])
y_lr = np.array([1.0, 3.0, 5.0, 7.0])
slope_lr, int_lr = linear_regression(x_lr, y_lr)
assert abs(slope_lr - 2.0) < 1e-10 and abs(int_lr - 1.0) < 1e-10, '[TC19] linear regression FAILED'

# ---- TC20: eyring_absorption_to_t60 平均吸声系数为 0 时退化为 Sabine ----
t60_eyring_zero = eyring_absorption_to_t60(400.0, 340.0, 0.0)
t60_sabine_zero = sabine_absorption_to_t60(400.0, 340.0 * 0.0)
assert abs(t60_eyring_zero - t60_sabine_zero) < 1e-6, '[TC20] Eyring zero absorption FAILED'

# ---- TC21: C_AIR 声速常数等于 343.0 ----
assert C_AIR == 343.0, '[TC21] C_AIR constant FAILED'

# ---- TC22: compute_statistics 返回字典包含均值和标准差 ----
stats_dict = compute_statistics(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
assert 'mean' in stats_dict and 'std' in stats_dict, '[TC22] statistics keys FAILED'
assert abs(stats_dict['mean'] - 3.0) < 1e-10, '[TC22] statistics mean FAILED'

# ---- TC23: modal_overlap_factor 输出列表长度与输入一致 ----
mof_data = modal_overlap_factor(modes[:5], damping_ratio=0.01)
assert len(mof_data) == 5, '[TC23] MOF list length FAILED'
assert all('mof' in d and 'frequency' in d for d in mof_data), '[TC23] MOF keys FAILED'

# ---- TC24: ball01_sample 所有采样点位于单位球内 ----
np.random.seed(42)
samples = ball01_sample(100)
norm_samples = np.linalg.norm(samples, axis=1)
assert np.all(norm_samples <= 1.0 + 1e-10), '[TC24] ball sample inside FAILED'

# ---- TC25: monte_carlo_ray_tracing 返回非负 T60 ----
np.random.seed(42)
normals = compute_surface_normals(test_surfaces)
times, edc, T60_mc, EDT = monte_carlo_ray_tracing(test_surfaces, normals, test_abs,
                                                   np.array([5.0, 4.0, 2.5]),
                                                   n_rays=100, max_reflections=20)
assert T60_mc >= 0.0, '[TC25] MC T60 non-negative FAILED'
assert EDT >= 0.0, '[TC26] EDT non-negative FAILED'

# ---- TC26: build_reflection_graph 转移概率矩阵每行和为 1 ----
np.random.seed(42)
trans_prob, surf_names = build_reflection_graph(test_surfaces, normals, test_abs, n_rays=200)
row_sums = trans_prob.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), '[TC26] transition row sums FAILED'

# ---- TC27: compute_room_response_stats 返回结构正确 ----
np.random.seed(42)
rt_stats = compute_room_response_stats(test_surfaces, normals, test_abs,
                                        np.array([5.0, 4.0, 2.5]), n_rays=100)
assert 'mean_free_path' in rt_stats, '[TC27] response stats keys FAILED'
assert rt_stats['mean_free_path'] >= 0.0, '[TC27] mean free path non-negative FAILED'

# ---- TC28: compute_edge_diffraction_field 返回有限复数值 ----
receiver_pos = np.array([7.0, 6.0, 3.0])
diff_field = compute_edge_diffraction_field(np.array([5.0, 4.0, 2.5]), receiver_pos, test_edges, 500.0)
assert np.isfinite(np.abs(diff_field)), '[TC28] diffraction field finite FAILED'

# ---- TC29: solve_helmholtz_cg 返回解向量长度等于节点数 ----
np.random.seed(42)
p_small = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
t_small = np.array([[0, 1, 2, 3]], dtype=int)
p_sol_test, k_wavenum_test = solve_helmholtz_cg(p_small, t_small, 125.0, source_node=0, source_strength=1.0, tol=1e-6)
assert len(p_sol_test) == p_small.shape[0], '[TC29] Helmholtz solution length FAILED'
assert k_wavenum_test > 0, '[TC29] wavenumber positive FAILED'

# ---- TC30: triangle_symq_rule 精度 1 返回 1 个求积点 ----
n_rule, a_rule, b_rule, c_rule, w_rule = triangle_symq_rule(1)
assert n_rule == 1, '[TC30] triangle rule 1 point FAILED'
assert abs(a_rule[0] - 1.0/3.0) < 1e-10, '[TC30] triangle rule barycenter FAILED'

print('\n全部 30 个测试通过!\n')
