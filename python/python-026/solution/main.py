# -*- coding: utf-8 -*-
"""
main.py

激光-等离子体相互作用多尺度数值模拟系统 —— 统一入口。

本程序基于 15 个科研代码项目的核心算法融合而成，
面向惯性约束聚变（ICF）中高功率激光在等离子体中传播与能量沉积的
前沿博士级科学计算问题。

运行方式:
    python main.py

无需任何命令行参数。
"""

import numpy as np
import sys
import os

# 将当前目录加入路径（确保模块导入正确）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics_constants import (
    plasma_frequency, critical_density, refractive_index,
    laser_E0_from_intensity, laser_intensity_from_E0,
    inverse_bremsstrahlung_absorption, E_CHARGE, E_MASS,
    EPSILON_0, C_LIGHT, K_BOLTZMANN
)
from plasma_grid import rect_grid_2d, grid_spacing_quality, cell_volumes_2d
from density_profile import (
    piecewise_constant_density_2d, icf_density_profile,
    density_gradient_pwc, total_plasma_mass
)
from ray_tracer import RayTracer
from quadrature_engine import (
    integrate_1d_gauss_legendre, integrate_energy_deposition_along_ray
)
from dispersion_solver import solve_langmuir_wave_dispersion, srs_three_wave_coupling_roots
from parameter_sampling import sample_laser_plasma_parameters, sample_quality_metrics
from target_geometry import (
    icf_target_surface_mesh, solid_angle_subtended_by_quad,
    spherical_distance, stereographic_projection_sphere_to_plane
)
from polarization_dynamics import (
    evolve_polarization_along_ray, faraday_rotation_integral,
    circle_map_matrix_polarization
)
from sparse_field_solver import solve_poisson_2d, compute_electric_field_from_potential
from data_integrity import checksum_plasma_state


def main():
    print("=" * 70)
    print("  激光-等离子体相互作用多尺度数值模拟系统")
    print("  Multi-scale Numerical Simulation of Laser-Plasma Interactions")
    print("=" * 70)
    print()

    # ============================================================
    # 步骤 1: 初始化物理参数
    # ============================================================
    print("[步骤 1] 初始化激光与等离子体物理参数")

    # 激光参数 (Nd:YAG 激光)
    I0 = 5.0e18          # W/m^2
    lambda0 = 1.064e-6   # m
    w0 = 20.0e-6         # m, 焦斑半径
    tau_pulse = 1.0e-9   # s, 脉冲宽度
    omega0 = 2.0 * np.pi * C_LIGHT / lambda0
    E0 = laser_E0_from_intensity(I0)
    nc = critical_density(omega0)

    print(f"    激光波长 λ = {lambda0*1e6:.3f} μm")
    print(f"    激光角频率 ω_0 = {omega0:.4e} rad/s")
    print(f"    激光强度 I_0 = {I0:.4e} W/m²")
    print(f"    电场振幅 E_0 = {E0:.4e} V/m")
    print(f"    临界密度 n_c = {nc:.4e} m⁻³")

    # 等离子体参数
    n0 = 5.0e25          # m^{-3}
    Te_eV = 1000.0       # eV
    Te_K = Te_eV * E_CHARGE / K_BOLTZMANN
    R0 = 100.0e-6        # m, 靶丸半径
    Ls = 10.0e-6         # m, 密度标长
    Z_ion = 1

    print(f"    峰值电子密度 n_0 = {n0:.4e} m⁻³")
    print(f"    电子温度 T_e = {Te_eV:.1f} eV ({Te_K:.1e} K)")
    print(f"    靶丸半径 R_0 = {R0*1e6:.1f} μm")
    print(f"    密度标长 L_s = {Ls*1e6:.1f} μm")
    print()

    # ============================================================
    # 步骤 2: 生成空间网格 (基于 line_grid)
    # ============================================================
    print("[步骤 2] 生成 2D 空间离散网格")

    nx, ny = 81, 81
    x_bounds = (-150e-6, 150e-6)
    y_bounds = (-150e-6, 150e-6)
    X, Y, dx, dy = rect_grid_2d(nx, ny, x_bounds, y_bounds, cx=1, cy=1)
    quality = grid_spacing_quality(X[:, 0])
    print(f"    网格规模: {nx} × {ny}")
    print(f"    网格间距: dx = {dx*1e6:.3f} μm, dy = {dy*1e6:.3f} μm")
    print(f"    网格质量比 (max/min): {quality['ratio']:.3f}")
    print()

    # ============================================================
    # 步骤 3: 构建分段常数等离子体密度场 (基于 pwc_plot_2d)
    # ============================================================
    print("[步骤 3] 构建分段常数等离子体密度场")

    xc = X[:, 0]
    yc = Y[0, :]
    nxc, nyc = nx - 1, ny - 1

    def density_func(x, y):
        return icf_density_profile(x, y, n0, R0, Ls, f_plateau=0.2,
                                    perturbation_amplitude=0.02,
                                    perturbation_scale=2.0e-6)

    ne_cells, xc_cells, yc_cells = piecewise_constant_density_2d(
        xc, yc, nxc, nyc, density_func
    )
    print(f"    单元数: {nxc} × {nyc} = {nxc*nyc}")
    print(f"    密度范围: [{np.min(ne_cells):.3e}, {np.max(ne_cells):.3e}] m⁻³")

    mass = total_plasma_mass(ne_cells, xc_cells, yc_cells)
    print(f"    等离子体面密度估算: {mass:.3e} kg/m")

    grad_x, grad_y = density_gradient_pwc(ne_cells, xc_cells, yc_cells)
    print(f"    密度梯度最大值: |∇n_e|_max = {max(np.max(np.abs(grad_x)), np.max(np.abs(grad_y))):.3e} m⁻⁴")
    print()

    # ============================================================
    # 步骤 4: 密度插值函数封装 (用于射线追踪)
    # ============================================================
    print("[步骤 4] 建立密度插值接口")

    from density_profile import bilinear_interpolate_density

    def density_interp(x, y):
        return bilinear_interpolate_density(ne_cells, xc_cells, yc_cells, x, y)

    # 验证插值在中心点
    ne_center = density_interp(0.0, 0.0)
    print(f"    中心密度插值: n_e(0,0) = {ne_center:.3e} m⁻³")
    print()

    # ============================================================
    # 步骤 5: 激光射线追踪 (基于 stetter_ode / RK4)
    # ============================================================
    print("[步骤 5] 激光射线追踪 (Hamiltonian 射线方程)")

    tracer = RayTracer(omega0, eta_min=1e-4, max_steps=20000)
    domain_bounds = (x_bounds, y_bounds)

    # 生成一束激光射线（沿 x 轴方向，从左侧入射）
    n_rays = 5
    y_positions = np.linspace(-30e-6, 30e-6, n_rays)
    positions = np.column_stack((np.full(n_rays, x_bounds[0]), y_positions))
    directions = np.tile(np.array([1.0, 0.0]), (n_rays, 1))

    results = tracer.trace_beam(positions, directions, density_interp, domain_bounds)

    for i, res in enumerate(results):
        print(f"    射线 {i+1}: 初始 y = {y_positions[i]*1e6:.1f} μm, "
              f"路径长 = {res['path_length']*1e6:.2f} μm, 状态 = {res['status']}")
    print()

    # ============================================================
    # 步骤 6: 沿射线能量沉积计算 (基于 quad_rule / Gauss-Legendre)
    # ============================================================
    print("[步骤 6] 沿射线能量沉积计算 (Gauss-Legendre 积分)")

    total_dep = 0.0
    for i, res in enumerate(results):
        traj = res['trajectory']
        s_vals = res['s_vals']
        if len(s_vals) < 2:
            continue
        ne_path = np.array([density_interp(traj[k, 0], traj[k, 1]) for k in range(len(traj))])
        # 假设强度随光程指数衰减（简化模型）
        I_path = I0 * np.exp(-np.cumsum(
            np.array([inverse_bremsstrahlung_absorption(ne_path[k], Te_K, omega0, Z_ion)
                      for k in range(len(ne_path))]) * np.append(0, np.diff(s_vals))
        ))
        I_path = np.clip(I_path, 0.0, I0 * 10.0)

        dep = integrate_energy_deposition_along_ray(
            s_vals, I_path, ne_path, Te_K, omega0, Z_ion
        )
        total_dep += dep
        print(f"    射线 {i+1}: 沉积能量密度 = {dep:.3e} J/m²")

    print(f"    束总沉积能量密度 = {total_dep:.3e} J/m²")
    print()

    # ============================================================
    # 步骤 7: 等离子体波色散关系求解 (基于 wdk)
    # ============================================================
    print("[步骤 7] 等离子体波色散关系求解 (WDK 复根算法)")

    k_wave = 2.0 * np.pi / lambda0  # 激光波数
    omega_r, gamma, root_sel, all_roots = solve_langmuir_wave_dispersion(
        ne_center, Te_K, k_wave, omega0
    )
    print(f"    Langmuir 波: ω_r = {omega_r:.4e} rad/s, γ = {gamma:.4e} rad/s")
    print(f"    选定复根: {root_sel}")
    print(f"    全部 {len(all_roots)} 个复根:")
    for idx, r in enumerate(all_roots):
        print(f"      root[{idx}] = {r:.4e}")

    # SRS 三波耦合
    omega_s_r, gamma_srs, roots_srs = srs_three_wave_coupling_roots(
        ne_center, Te_K, 0.5 * k_wave, omega0, E0
    )
    print(f"    SRS 散射波: ω_s = {omega_s_r:.4e} rad/s, γ_SRS = {gamma_srs:.4e} rad/s")
    print()

    # ============================================================
    # 步骤 8: 参数空间拉丁超立方采样 (基于 latin_edge + opt_sample)
    # ============================================================
    print("[步骤 8] 激光-等离子体参数空间拉丁超立方采样")

    n_samples = 20
    params, param_names, param_bounds = sample_laser_plasma_parameters(n_samples, seed=42)
    metrics = sample_quality_metrics(params)
    print(f"    采样点数: {n_samples}")
    print(f"    参数维度: {len(param_names)}")
    for name, bound in zip(param_names, param_bounds):
        print(f"      {name}: [{bound[0]:.2e}, {bound[1]:.2e}]")
    print(f"    采样质量: 最小两两距离 = {metrics['min_pairwise_dist']:.4e}")
    print(f"    采样质量: 平均两两距离 = {metrics['mean_pairwise_dist']:.4e}")
    print()

    # ============================================================
    # 步骤 9: 靶丸几何建模 (基于 cities + sphere_stereograph + quadrilateral)
    # ============================================================
    print("[步骤 9] ICF 靶丸表面几何建模")

    n_theta, n_phi = 21, 41
    vertices, face_areas, total_area = icf_target_surface_mesh(R0, n_theta, n_phi)
    print(f"    靶丸半径: R_0 = {R0*1e6:.1f} μm")
    print(f"    网格面片数: {(n_theta-1)*(n_phi-1)}")
    print(f"    计算表面积: {total_area*1e6:.4f} mm²")
    print(f"    理论表面积: {4*np.pi*R0**2*1e6:.4f} mm²")
    print(f"    相对误差: {abs(total_area - 4*np.pi*R0**2)/(4*np.pi*R0**2)*100:.2f}%")

    # 球面距离示例
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = np.pi / 2.0, 0.0
    d = spherical_distance(lat1, lon1, lat2, lon2, R0)
    print(f"    球面距离示例 (0,0) -> (π/2,0): {d*1e6:.2f} μm")

    # 立体投影示例
    p_surf = np.array([0.0, R0, 0.0])
    p_norm = p_surf / R0  # 单位球
    q_proj = stereographic_projection_sphere_to_plane(p_norm)
    print(f"    立体投影示例 (0, R0, 0) -> 平面: ({q_proj[0]:.4f}, {q_proj[1]:.4f})")
    print()

    # ============================================================
    # 步骤 10: 偏振态演化 (基于 circle_map)
    # ============================================================
    print("[步骤 10] 激光偏振态在等离子体中的演化")

    z_pol = np.linspace(0.0, 200e-6, 200)
    def ne_func_z(z):
        return icf_density_profile(z, 0.0, n0, R0, Ls, f_plateau=0.2)
    def B_func_z(z):
        # 假设沿 z 轴的磁场，在中心最强
        B0 = 10.0  # T
        return np.array([0.0, 0.0, B0 * np.exp(-(z - 100e-6)**2 / (2 * (30e-6)**2))])

    E0_jones = np.array([1.0, 0.0])  # 线偏振
    E_hist, S_hist = evolve_polarization_along_ray(
        omega0, ne_func_z, B_func_z, z_pol, E0_jones
    )

    # 法拉第旋转角
    B_par = np.array([B_func_z(z)[2] for z in z_pol])
    ne_path = np.array([ne_func_z(z) for z in z_pol])
    theta_F = faraday_rotation_integral(ne_path, B_par, z_pol, omega0)
    print(f"    总法拉第旋转角: θ_F = {theta_F:.4e} rad = {np.degrees(theta_F):.4f}°")

    # Circle map: 琼斯矩阵对单位圆的映射
    # 构造一个代表性的有效琼斯矩阵 (累积效应)
    T_eff = np.eye(2)
    for i in range(1, len(z_pol)):
        dz = z_pol[i] - z_pol[i - 1]
        from physics_constants import plasma_frequency
        from polarization_dynamics import jones_matrix_propagation
        ne_mid = 0.5 * (ne_path[i] + ne_path[i - 1])
        B_mid = B_func_z(0.5 * (z_pol[i] + z_pol[i - 1]))
        omega_p = plasma_frequency(ne_mid)
        B_mag = np.linalg.norm(B_mid)
        omega_c = E_CHARGE * B_mag / E_MASS if B_mag > 0 else 0.0
        b_hat = B_mid / B_mag if B_mag > 0 else np.array([0.0, 0.0, 1.0])
        T_step = jones_matrix_propagation(omega0, omega_p, omega_c, b_hat, dz)
        T_eff = T_step @ T_eff

    T_eff_real = np.real(T_eff)
    _, _, aspect = circle_map_matrix_polarization(T_eff_real, n_points=100)
    print(f"    偏振椭圆长短轴比: {aspect:.4f}")
    print(f"    最终 Stokes 参数: S = [{S_hist[-1,0]:.4f}, {S_hist[-1,1]:.4f}, "
          f"{S_hist[-1,2]:.4f}, {S_hist[-1,3]:.4f}]")
    print()

    # ============================================================
    # 步骤 11: 稀疏矩阵泊松求解 (基于 sparse_parfor)
    # ============================================================
    print("[步骤 11] 等离子体泊松方程稀疏矩阵求解")

    # 在内部网格上求解 (ne_cells 有 nxc×nyc 个单元，nxc=nx-1, nyc=ny-1)
    # 去掉一圈边界后内部单元为 (nxc-2) × (nyc-2)
    nx_in = nxc - 2
    ny_in = nyc - 2
    # 准中性偏离: n_i = n_e * (1 + delta)，delta 为小扰动 (模拟电荷分离)
    delta_quasi = 0.01 * np.sin(np.linspace(0, 2*np.pi, nx_in))[:, None] * \
                  np.cos(np.linspace(0, 2*np.pi, ny_in))[None, :]
    rho_grid = E_CHARGE * ne_cells[1:-1, 1:-1] * delta_quasi
    phi, residual, info = solve_poisson_2d(
        rho_grid, nx_in, ny_in, dx, dy, tol=1e-12, max_iter=10000
    )
    Ex, Ey = compute_electric_field_from_potential(phi, dx, dy)
    print(f"    内部网格: {nx_in} × {ny_in}")
    print(f"    求解器信息码: {info}")
    print(f"    相对残差: {residual:.4e}")
    print(f"    电势范围: [{np.min(phi):.4e}, {np.max(phi):.4e}] V")
    print(f"    电场最大值: |E|_max = {np.max(np.sqrt(Ex**2 + Ey**2)):.4e} V/m")
    print()

    # ============================================================
    # 步骤 12: 数据完整性校验 (基于 luhn)
    # ============================================================
    print("[步骤 12] 数值数据完整性校验")

    Te_grid = np.full_like(ne_cells, Te_K)
    # 扩展 phi 到与 ne_cells 相同尺寸（边界补零）
    phi_padded = np.zeros((nxc, nyc))
    phi_padded[1:-1, 1:-1] = phi
    checksum = checksum_plasma_state(ne_cells, Te_grid, phi_padded)
    print(f"    等离子体状态校验码:")
    print(f"      {checksum}")
    print()

    # ============================================================
    # 步骤 13: 综合结果汇总
    # ============================================================
    print("=" * 70)
    print("  综合结果汇总")
    print("=" * 70)
    print(f"  激光临界密度:          n_c = {nc:.4e} m⁻³")
    print(f"  中心等离子体频率:      ω_p = {plasma_frequency(ne_center):.4e} rad/s")
    print(f"  中心折射率:            η   = {refractive_index(ne_center, omega0):.6f}")
    print(f"  Langmuir 波频率:       ω_r = {omega_r:.4e} rad/s")
    print(f"  Langmuir 波阻尼率:     γ   = {gamma:.4e} rad/s")
    print(f"  SRS 增长率:            γ_0 = {gamma_srs:.4e} rad/s")
    print(f"  法拉第旋转角:          θ_F = {np.degrees(theta_F):.4f}°")
    print(f"  总能量沉积密度:        E_dep = {total_dep:.4e} J/m²")
    print(f"  靶丸表面积:            A = {total_area*1e6:.4f} mm²")
    print(f"  泊松求解残差:          res = {residual:.4e}")
    print("=" * 70)
    print("  模拟完成，所有模块运行正常。")
    print("=" * 70)


if __name__ == "__main__":
    main()
