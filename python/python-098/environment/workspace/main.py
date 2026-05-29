#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
超表面电磁调控与全息的博士级综合计算框架入口。

运行方式:  python main.py
（零参数，内部自动执行全部计算流程）

本程序融合 15 个种子项目的核心算法，在电磁学：超表面电磁调控与全息
前沿领域内完成以下全流程：
  1. 基于 CVT 的非周期超表面布局优化（florida_cvt_pop, sphere_cvt）
  2. 基于 Legendre/Chebyshev 谱展开的相位剖面设计（r8poly, polpak）
  3. 基于球谐函数的远场全息图案展开（polpak/spherical_harmonic）
  4. 基于 SVD 的低秩散射算符压缩（svd_gray）
  5. 基于 RK4/RK23 的梯度折射率波传播模拟（ode_rk4, rk23）
  6. 基于分段/非线性振子的 meta-atom 响应建模（rubber_band_ode, pendulum_ode_period）
  7. 基于结构化/非结构化网格的仿真域剖分（ice_to_medit, gmsh_to_fem）
  8. 基于 XYZ 点云的 meta-atom 几何 I/O（xyz_io）
  9. 基于蒙特卡洛的制造公差分析（craps_simulation）
  10. 基于单重积分的贴片矩量验证（square_integrals）
  11. 基于 Gerchberg-Saxton 的全息优化与配置 I/O（scip_solution_read）
"""

import os
import sys
import numpy as np

# 将当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cvt_placer import (
    lloyd_relaxation_square, lloyd_relaxation_sphere,
    compute_voronoi_areas_square, compute_voronoi_areas_sphere
)
from phase_spectrum import (
    horner_eval, legendre_polynomials_array, design_hologram_phase_2d,
    chebyshev_nodes, reconstruct_phase_from_spectrum
)
from spherical_expansion import (
    spherical_harmonic_y, expand_far_field_spherical,
    reconstruct_far_field, scattering_coefficients_mie
)
from scattering_operator import (
    build_scattering_operator, svd_compress_scattering,
    evaluate_compression_error, apply_scattering_operator
)
from wave_propagation import (
    rk4_integrate, rk23_integrate, propagate_plane_wave_scalar,
    propagate_coupled_modes, angular_spectrum_propagate,
    effective_medium_profile
)
from nonlinear_resonator import (
    rubber_band_resonator, pendulum_meta_atom, duffing_resonator,
    effective_phase_shift_nonlinear, nonlinear_transmission_coefficient,
    pendulum_period_small_angle, pendulum_period_elliptic
)
from mesh_handler import (
    write_simple_mesh, read_simple_mesh, detect_dimension,
    compute_mesh_quality, generate_unit_cube_mesh
)
from geometry_io import (
    write_xyz_data, read_xyz_data, generate_meta_atom_cloud,
    compute_point_cloud_stats, write_xyz_with_phases
)
from tolerance_analysis import (
    monte_carlo_phase_error, estimate_yield, craps_exact_probability,
    tolerance_sensitivity_analysis
)
from moment_integrals import (
    square01_monomial_integral, squaresym_monomial_integral,
    integrate_2d_gauss_legendre, compute_field_moments,
    verify_monomial_integrals, patch_impedance_moment
)
from hologram_io import (
    write_binary_hologram, read_binary_hologram,
    binary_to_phase_config, phase_to_binary_config,
    gerchberg_saxton_iteration, validate_hologram_config
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("  超表面电磁调控与全息 — 博士级综合计算框架")
    print("  Metasurface Electromagnetic Control and Holography")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 物理常数与系统参数
    # ------------------------------------------------------------------
    c = 2.99792458e8          # 光速 (m/s)
    lambda0 = 633e-9          # 工作波长：He-Ne 红光 (m)
    k0 = 2.0 * np.pi / lambda0
    omega = 2.0 * np.pi * c / lambda0
    aperture_size = 20e-6     # 超表面孔径 20 μm
    n_pixels = 32             # 32×32 像素化超表面
    dx = aperture_size / n_pixels
    meta_atom_height = 600e-9 # 纳米柱高度
    n_si = 3.48               # 硅折射率
    n_air = 1.0

    print(f"\n[系统参数]")
    print(f"  波长 λ = {lambda0*1e9:.1f} nm")
    print(f"  孔径 L = {aperture_size*1e6:.1f} μm")
    print(f"  像素数 N = {n_pixels}×{n_pixels}")
    print(f"  像素尺寸 dx = {dx*1e9:.2f} nm")
    print(f"  硅折射率 n_Si = {n_si}")

    # ------------------------------------------------------------------
    # 1. CVT 非周期布局优化（florida_cvt_pop, sphere_cvt）
    # ------------------------------------------------------------------
    print_section("1. CVT 非周期超表面布局优化")

    # 目标远场为一个高斯斑，密度正比于目标强度
    def target_density(x, y):
        sigma = 0.3
        val = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        return max(val, 1e-6)

    generators_square = lloyd_relaxation_square(
        n_generators=64, n_steps=12, density_func=target_density,
        n_samples=10000, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), seed=42
    )
    areas_sq = compute_voronoi_areas_square(generators_square, n_samples=30000)
    print(f"  二维 CVT 生成元数: {generators_square.shape[0]}")
    print(f"  Voronoi 面积均值: {np.mean(areas_sq):.6f}, 标准差: {np.std(areas_sq):.6f}")

    generators_sphere = lloyd_relaxation_sphere(
        n_generators=64, n_steps=12, n_samples=10000, seed=42
    )
    areas_sp = compute_voronoi_areas_sphere(generators_sphere, n_samples=30000)
    print(f"  球面 CVT 生成元数: {generators_sphere.shape[0]}")
    print(f"  球面 Voronoi 面积均值: {np.mean(areas_sp):.6f} sr, 标准差: {np.std(areas_sp):.6f}")

    # ------------------------------------------------------------------
    # 2. Legendre 谱相位设计（r8poly, polpak/legendre_poly）
    # ------------------------------------------------------------------
    print_section("2. Legendre 谱展开相位剖面设计")

    x_grid = np.linspace(-aperture_size / 2.0, aperture_size / 2.0, n_pixels)
    y_grid = x_grid.copy()

    # HOLE 3 (Part A): 构造 Legendre 谱系数并调用相位设计函数
    # 需要定义 max_degree、设置 coeffs 矩阵（透镜项 + 螺旋相位项），
    # 调用 design_hologram_phase_2d 生成 phase_profile，并 wrap 到 [-π, π]
    raise NotImplementedError("Hole 3 Part A: main.py 中相位系数设置与相位设计调用需要实现")

    # 验证 Chebyshev 节点上的 Horner 求值
    cheb_x = chebyshev_nodes(8)
    poly_c = [1.0, -2.0, 3.0, -0.5]  # 1 - 2x + 3x² - 0.5x³
    horner_vals = [horner_eval(poly_c, xi) for xi in cheb_x]
    print(f"  Horner 求值验证（前3点）: {horner_vals[:3]}")

    # ------------------------------------------------------------------
    # 3. 球谐函数远场展开（polpak/spherical_harmonic）
    # ------------------------------------------------------------------
    print_section("3. 球谐函数远场展开与 Mie 散射")

    theta_grid = np.linspace(0.01, np.pi - 0.01, 24)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, 48)
    # 构造测试远场：Y_2^0(θ,φ) ∝ (3cos²θ - 1)
    TH, PH = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    target_far_field = (3.0 * np.cos(TH) ** 2 - 1.0) * np.exp(1j * PH)

    l_max = 4
    coeffs_sph = expand_far_field_spherical(
        target_far_field, theta_grid, phi_grid, l_max
    )
    energy = sum(abs(a) ** 2 for a in coeffs_sph.values())
    print(f"  球谐展开阶数 l_max = {l_max}")
    print(f"  展开系数总数: {len(coeffs_sph)}")
    print(f"  展开能量 Σ|a_lm|² = {energy:.6f}")

    # Mie 散射系数
    a_mie = scattering_coefficients_mie(l_max, k0, 100e-9,
                                         n_si ** 2, 1.0)
    print(f"  Mie 电偶极子系数 a_1 = {a_mie[1]:.6e}")

    # ------------------------------------------------------------------
    # 4. SVD 散射算符压缩（svd_gray）
    # ------------------------------------------------------------------
    print_section("4. SVD 低秩散射算符压缩")

    # HOLE 3 (Part B): 调用 build_scattering_operator 构建散射算符
    # 需要传入 n_pixels, aperture_size, wavelength, phase_profile 等参数
    raise NotImplementedError("Hole 3 Part B: main.py 中散射算符构建调用需要实现")

    rank_list = [16, 32, 64, 128]
    err_results = evaluate_compression_error(S, rank_list)
    for res in err_results:
        print(f"  秩 R={res['rank']:3d}: 相对误差 = {res['relative_error']:.6e}")

    S_approx, U, s, Vh, metrics = svd_compress_scattering(S, rank=64)
    print(f"  最优压缩比: {metrics['compression_ratio']:.4f}")
    print(f"  部分奇异值和占比: {metrics['partial_sum_ratio']:.4f}")

    # 验证散射算符作用
    field_in = np.ones(S.shape[1], dtype=complex)
    field_out = apply_scattering_operator(S, field_in)
    print(f"  散射输出场总功率: {np.sum(np.abs(field_out)**2):.6e}")

    # ------------------------------------------------------------------
    # 5. RK4/RK23 波传播（ode_rk4, rk23）
    # ------------------------------------------------------------------
    print_section("5. 梯度有效折射率波传播 (RK4/RK23)")

    z_span = (0.0, meta_atom_height)
    n_steps = 100

    # 线性梯度折射率
    n_eff = lambda z: effective_medium_profile(
        z, n_substrate=n_si, n_air=n_air, thickness=meta_atom_height,
        profile_type='linear'
    )
    t_rk4, E_rk4 = propagate_plane_wave_scalar(
        k0, z_span, n_eff, E0=1.0 + 0.0j, n_steps=n_steps
    )
    print(f"  RK4 传播: z ∈ [{z_span[0]*1e9:.0f}, {z_span[1]*1e9:.0f}] nm")
    print(f"  初始幅度 |E(0)| = {abs(E_rk4[0]):.6f}")
    print(f"  最终幅度 |E(z_end)| = {abs(E_rk4[-1]):.6f}")

    # RK23 误差估计
    def f_wave(z, E_vec):
        nz = n_eff(z)
        nz = complex(max(np.real(nz), 1.0), min(np.imag(nz), 0.0))
        return np.array([1j * k0 * nz * E_vec[0]], dtype=complex)

    t_rk23, E_rk23, e_rk23 = rk23_integrate(f_wave, z_span,
                                            np.array([1.0 + 0.0j], dtype=complex),
                                            n_steps=n_steps)
    max_err_rk23 = float(np.max(e_rk23))
    print(f"  RK23 最大局部截断误差估计: {max_err_rk23:.6e}")

    # 角谱传播验证
    field_2d = np.exp(1j * phase_profile)
    propagated = angular_spectrum_propagate(field_2d, k0, z=10e-6, dx=dx, dy=dx)
    print(f"  角谱传播后总功率守恒偏差: {abs(np.sum(np.abs(propagated)**2) - np.sum(np.abs(field_2d)**2)):.6e}")

    # ------------------------------------------------------------------
    # 6. 非线性 Meta-Atom 响应（rubber_band_ode, pendulum_ode_period）
    # ------------------------------------------------------------------
    print_section("6. 非线性 Meta-Atom 响应建模")

    # Duffing 振子
    from wave_propagation import rk4_integrate
    t_duff, y_duff = rk4_integrate(
        lambda t, y: duffing_resonator(t, y, alpha=1.0, beta=0.1,
                                       gamma=0.05, omega=1.0, F=1.0),
        (0.0, 20.0 * np.pi),
        np.array([0.1, 0.0]),
        n_steps=800
    )
    print(f"  Duffing 振子稳态幅度: {np.max(np.abs(y_duff[:,0])):.6f}")

    # 非线性相位偏移
    I_range = np.linspace(0.0, 10.0, 11)
    phi_nonlinear = []
    params_nl = {'omega0': 1.0, 'omega': 1.0, 'gamma': 0.05,
                 'kappa': 0.1, 'I_sat': 1.0}
    for I in I_range:
        phi_eff = effective_phase_shift_nonlinear(I, params_nl)
        phi_nonlinear.append(phi_eff)
    print(f"  非线性相位偏移范围: [{min(phi_nonlinear):.4f}, {max(phi_nonlinear):.4f}] rad")

    # 摆周期
    T_small = pendulum_period_small_angle(g=9.81, l=0.1)
    T_large = pendulum_period_elliptic(theta0=np.pi / 3.0, g=9.81, l=0.1)
    print(f"  小角度摆周期 T_small = {T_small:.6f} s")
    print(f"  大角度(π/3)摆周期 T_large = {T_large:.6f} s")

    # ------------------------------------------------------------------
    # 7. 网格处理（ice_to_medit, gmsh_to_fem）
    # ------------------------------------------------------------------
    print_section("7. 仿真域网格生成与质量评估")

    nodes, elements, elem_types = generate_unit_cube_mesh(nx=4, ny=4, nz=4)
    mesh_filename = os.path.join(os.path.dirname(__file__), "test_mesh.txt")
    write_simple_mesh(mesh_filename, nodes, elements, elem_types)
    nodes_r, node_labels_r, elements_r, elem_types_r, elem_labels_r = read_simple_mesh(mesh_filename)
    dim = detect_dimension(nodes_r)
    qualities, avg_q, min_q = compute_mesh_quality(nodes_r, elements_r, elem_types_r)
    print(f"  网格节点数: {nodes_r.shape[0]}")
    print(f"  网格单元数: {elements_r.shape[0]}")
    print(f"  检测维度: {dim}D")
    print(f"  平均网格质量: {avg_q:.6f}")
    print(f"  最差网格质量: {min_q:.6f}")

    # ------------------------------------------------------------------
    # 8. XYZ 点云 I/O（xyz_io）
    # ------------------------------------------------------------------
    print_section("8. Meta-Atom 三维点云 I/O")

    cloud = generate_meta_atom_cloud(
        n=50, x_range=(-aperture_size / 2.0, aperture_size / 2.0),
        y_range=(-aperture_size / 2.0, aperture_size / 2.0),
        z_range=(0.0, meta_atom_height), radius=200e-9, seed=42
    )
    xyz_filename = os.path.join(os.path.dirname(__file__), "meta_atoms.xyz")
    write_xyz_data(xyz_filename, cloud)
    cloud_r = read_xyz_data(xyz_filename)
    stats = compute_point_cloud_stats(cloud_r)
    print(f"  生成 meta-atom 数量: {cloud_r.shape[0]}")
    print(f"  点云重心: ({stats['centroid'][0]*1e6:.3f}, "
          f"{stats['centroid'][1]*1e6:.3f}, {stats['centroid'][2]*1e9:.3f}) μm/μm/nm")
    print(f"  最小间距: {stats['min_distance']*1e9:.2f} nm")

    # 写出带相位标签的 XYZ
    phases_cloud = np.random.uniform(-np.pi, np.pi, cloud_r.shape[0])
    xyzp_filename = os.path.join(os.path.dirname(__file__), "meta_atoms_phase.xyz")
    write_xyz_with_phases(xyzp_filename, cloud_r, phases_cloud)

    # ------------------------------------------------------------------
    # 9. 蒙特卡洛制造公差分析（craps_simulation）
    # ------------------------------------------------------------------
    print_section("9. 蒙特卡洛制造公差分析")

    target_amp = np.ones((n_pixels, n_pixels), dtype=float)
    # 高斯目标
    cx, cy = n_pixels // 2, n_pixels // 2
    for i in range(n_pixels):
        for j in range(n_pixels):
            target_amp[i, j] = np.exp(-((i - cx) ** 2 + (j - cy) ** 2) / (2.0 * (n_pixels / 4.0) ** 2))

    mc_result = monte_carlo_phase_error(
        n_trials=300,
        sigma_phase=0.05,
        n_pixels=n_pixels,
        phase_design=phase_profile,
        target_far_field=target_amp,
        propagate_func=None,
        seed=42
    )
    yield_rate = estimate_yield(mc_result['errors'], threshold=0.15)
    print(f"  蒙特卡洛试验次数: 300")
    print(f"  相位噪声 σ_φ = 0.05 rad")
    print(f"  平均重构误差: {mc_result['mean_error']:.6f}")
    print(f"  误差标准差: {mc_result['std_error']:.6f}")
    print(f"  良率 (ε < 0.15): {yield_rate*100:.2f}%")
    print(f"  参考 craps 精确概率: {craps_exact_probability():.6f}")

    # ------------------------------------------------------------------
    # 10. 贴片矩量验证（square_integrals）
    # ------------------------------------------------------------------
    print_section("10. 方形贴片矩量与数值积分验证")

    max_order = 4
    max_err = verify_monomial_integrals(max_order)
    print(f"  解析/数值积分最大相对误差 (order≤{max_order}): {max_err:.6e}")

    # 贴片阻抗矩
    moments, k0_patch = patch_impedance_moment(
        patch_size=dx, wavelength=lambda0, order=2
    )
    print(f"  贴片尺寸: {dx*1e9:.2f} nm")
    print(f"  屋顶函数矩 M_00 = {moments[0,0]:.6e}")

    # Gauss-Legendre 数值积分示例
    def test_func(x, y):
        return np.sin(np.pi * x) * np.cos(np.pi * y)
    val_gl = integrate_2d_gauss_legendre(test_func, xlim=(-1.0, 1.0),
                                          ylim=(-1.0, 1.0), n=8)
    # 解析值: ∫_{-1}^1 sin(πx) dx = 0, ∫_{-1}^1 cos(πy) dy = 0 => 0
    print(f"  Gauss-Legendre 数值积分 sin(πx)cos(πy) 于 [-1,1]² = {val_gl:.6e}")

    # ------------------------------------------------------------------
    # 11. Gerchberg-Saxton 全息优化（scip_solution_read）
    # ------------------------------------------------------------------
    print_section("11. Gerchberg-Saxton 全息相位优化")

    # 重新定义目标幅度（与第9节相同的高斯斑）
    target_amplitude = np.ones((n_pixels, n_pixels), dtype=float)
    cx, cy = n_pixels // 2, n_pixels // 2
    for i in range(n_pixels):
        for j in range(n_pixels):
            target_amplitude[i, j] = np.exp(-((i - cx) ** 2 + (j - cy) ** 2) / (2.0 * (n_pixels / 4.0) ** 2))

    initial_phase = np.random.uniform(-np.pi, np.pi, (n_pixels, n_pixels))
    phase_gs, error_history = gerchberg_saxton_iteration(
        target_amplitude, initial_phase, n_iter=15
    )
    print(f"  GS 迭代次数: 15")
    print(f"  初始误差: {error_history[0]:.6f}")
    print(f"  最终误差: {error_history[-1]:.6f}")

    # 将优化结果写入二进制配置
    bin_config = phase_to_binary_config(phase_gs, phase_levels=2)
    holo_filename = os.path.join(os.path.dirname(__file__), "hologram_config.txt")
    write_binary_hologram(holo_filename, bin_config)
    config_read = read_binary_hologram(holo_filename, n_pixels=n_pixels * n_pixels)
    print(f"  全息配置写入/读取一致性: {np.array_equal(bin_config, config_read)}")

    # 约束验证
    is_valid, violations = validate_hologram_config(
        bin_config.reshape(n_pixels, n_pixels),
        constraints={'max_ones': n_pixels * n_pixels, 'min_ones': 0}
    )
    print(f"  配置约束验证: {'通过' if is_valid else '失败'}")
    if not is_valid:
        for v in violations[:3]:
            print(f"    违规: {v}")

    # ------------------------------------------------------------------
    # 综合结果汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  计算完成。所有模块已协同运行，无报错。")
    print("=" * 70)
    print(f"\n  输出文件:")
    print(f"    - {mesh_filename}")
    print(f"    - {xyz_filename}")
    print(f"    - {xyzp_filename}")
    print(f"    - {holo_filename}")
    print("\n")


if __name__ == "__main__":
    main()
