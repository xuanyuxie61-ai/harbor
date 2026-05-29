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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# ---- TC01: plasma_frequency zero density returns zero ----
result = plasma_frequency(0.0)
assert abs(result) < 1e-20, '[TC01] plasma_frequency zero density FAILED'

# ---- TC02: critical_density matches analytical formula ----
omega_test = 1.0e15
nc_test = critical_density(omega_test)
nc_expected = EPSILON_0 * E_MASS * omega_test**2 / E_CHARGE**2
assert abs(nc_test - nc_expected) / max(abs(nc_expected), 1.0) < 1e-10, '[TC02] critical_density formula FAILED'

# ---- TC03: refractive_index vacuum is 1 and cutoff is 0 ----
omega_test = 1.0e15
nc_test = critical_density(omega_test)
eta_vac = refractive_index(0.0, omega_test)
eta_cut = refractive_index(nc_test, omega_test)
assert abs(eta_vac - 1.0) < 1e-12, '[TC03] refractive_index vacuum FAILED'
assert abs(eta_cut) < 1e-12, '[TC03] refractive_index cutoff FAILED'

# ---- TC04: laser intensity E0 round-trip consistency ----
I_test = 1.0e18
E0_test = laser_E0_from_intensity(I_test)
I_back = laser_intensity_from_E0(E0_test)
assert abs(I_back - I_test) / I_test < 1e-12, '[TC04] laser intensity E0 round-trip FAILED'

# ---- TC05: inverse_bremsstrahlung finite for small positive density ----
Te_test = 1000.0 * E_CHARGE / K_BOLTZMANN
omega_test = 1.0e15
kappa = inverse_bremsstrahlung_absorption(1.0e10, Te_test, omega_test)
assert np.isfinite(kappa), '[TC05] inverse_bremsstrahlung finite FAILED'
assert kappa >= 0.0, '[TC05] inverse_bremsstrahlung non-negative FAILED'

# ---- TC06: rect_grid_2d shape and spacing correctness ----
X, Y, dx, dy = rect_grid_2d(11, 21, (-1e-4, 1e-4), (-2e-4, 2e-4), cx=1, cy=1)
assert X.shape == (11, 21), '[TC06] rect_grid_2d X shape FAILED'
assert Y.shape == (11, 21), '[TC06] rect_grid_2d Y shape FAILED'
assert abs(dx - 2e-5) < 1e-20, '[TC06] rect_grid_2d dx FAILED'
assert abs(dy - 2e-5) < 1e-20, '[TC06] rect_grid_2d dy FAILED'

# ---- TC07: grid_spacing_quality uniform grid ratio is 1 ----
x_uniform = np.linspace(0.0, 1.0, 101)
qual = grid_spacing_quality(x_uniform)
assert abs(qual['ratio'] - 1.0) < 1e-12, '[TC07] grid_spacing_quality ratio FAILED'
assert abs(qual['mean_dx'] - 0.01) < 1e-12, '[TC07] grid_spacing_quality mean_dx FAILED'

# ---- TC08: cell_volumes_2d shape matches grid ----
X, Y, dx, dy = rect_grid_2d(5, 7, (0.0, 1.0), (0.0, 2.0), cx=1, cy=1)
vols = cell_volumes_2d(X, Y)
assert vols.shape == (4, 6), '[TC08] cell_volumes_2d shape FAILED'

# ---- TC09: icf_density_profile center approaches peak density ----
n0_test = 1.0e25
R0_test = 50.0e-6
Ls_test = 5.0e-6
ne_center = icf_density_profile(0.0, 0.0, n0_test, R0_test, Ls_test, f_plateau=0.3, perturbation_amplitude=0.0)
assert abs(ne_center - n0_test) / n0_test < 1e-3, '[TC09] icf_density_profile center FAILED'

# ---- TC10: piecewise_constant_density_2d shape and value ----
xc_test = np.linspace(-1e-4, 1e-4, 6)
yc_test = np.linspace(-1e-4, 1e-4, 6)
def const_density(x, y):
    x_arr = np.asarray(x, dtype=float)
    return np.full_like(x_arr, 1.0e24)
ne_cells, xc_out, yc_out = piecewise_constant_density_2d(xc_test, yc_test, 5, 5, const_density)
assert ne_cells.shape == (5, 5), '[TC10] piecewise_constant_density_2d shape FAILED'
assert np.allclose(ne_cells, 1.0e24), '[TC10] piecewise_constant_density_2d value FAILED'

# ---- TC11: density_gradient_pwc uniform field zero gradient ----
ne_cells = np.ones((5, 5)) * 1.0e24
xc = np.linspace(0.0, 1.0, 6)
yc = np.linspace(0.0, 1.0, 6)
gx, gy = density_gradient_pwc(ne_cells, xc, yc)
assert gx.shape == (4, 5), '[TC11] density_gradient_pwc grad_x shape FAILED'
assert gy.shape == (5, 4), '[TC11] density_gradient_pwc grad_y shape FAILED'
assert np.allclose(gx, 0.0), '[TC11] density_gradient_pwc uniform gx FAILED'
assert np.allclose(gy, 0.0), '[TC11] density_gradient_pwc uniform gy FAILED'

# ---- TC12: total_plasma_mass uniform density exact integral ----
xc_test = np.linspace(0.0, 1.0, 3)
yc_test = np.linspace(0.0, 2.0, 3)
ne_cells = np.ones((2, 2)) * 1.0e24
mass = total_plasma_mass(ne_cells, xc_test, yc_test, ion_mass=1.0)
assert abs(mass - 2.0e24) < 1e-10, '[TC12] total_plasma_mass uniform FAILED'

# ---- TC13: RayTracer vacuum straight ray exits domain ----
tracer = RayTracer(1.0e15, eta_min=1e-4, max_steps=5000)
def vacuum_density(x, y):
    return 0.0
domain = ((-1e-3, 1e-3), (-1e-3, 1e-3))
pos = np.array([[-5e-4, 0.0]])
dir_vec = np.array([[1.0, 0.0]])
results = tracer.trace_beam(pos, dir_vec, vacuum_density, domain)
assert len(results) == 1, '[TC13] RayTracer results length FAILED'
assert results[0]['status'] == 'domain_exit', '[TC13] RayTracer status FAILED'
assert results[0]['path_length'] > 1e-4, '[TC13] RayTracer path length FAILED'

# ---- TC14: integrate_1d_gauss_legendre constant exact ----
result = integrate_1d_gauss_legendre(lambda x: 3.0, 0.0, 2.0, n=5)
assert abs(result - 6.0) < 1e-14, '[TC14] integrate_1d_gauss_legendre constant FAILED'

# ---- TC15: integrate_energy_deposition zero density zero energy ----
s_test = np.linspace(0.0, 1.0, 11)
I_test_arr = np.ones(11) * 1.0e18
ne_zero = np.zeros(11)
Te_test = 1000.0
omega_test = 1.0e15
dep = integrate_energy_deposition_along_ray(s_test, I_test_arr, ne_zero, Te_test, omega_test)
assert abs(dep) < 1e-20, '[TC15] integrate_energy_deposition zero density FAILED'

# ---- TC16: solve_langmuir_wave_dispersion finite positive frequency ----
ne_test = 1.0e25
Te_test = 1000.0 * E_CHARGE / K_BOLTZMANN
k_test = 1.0e7
omega0_test = 1.0e15
omega_r, gamma, root_sel, all_roots = solve_langmuir_wave_dispersion(ne_test, Te_test, k_test, omega0_test)
assert np.isfinite(omega_r), '[TC16] solve_langmuir omega_r finite FAILED'
assert omega_r > 0, '[TC16] solve_langmuir omega_r positive FAILED'
assert len(all_roots) == 4, '[TC16] solve_langmuir root count FAILED'

# ---- TC17: srs_three_wave_coupling_roots finite output ----
ne_test = 1.0e25
Te_test = 1000.0 * E_CHARGE / K_BOLTZMANN
k_s = 1.0e7
omega0_test = 1.0e15
E0_test = 1.0e10
omega_s_r, gamma_srs, roots = srs_three_wave_coupling_roots(ne_test, Te_test, k_s, omega0_test, E0_test)
assert np.isfinite(omega_s_r), '[TC17] srs omega_s finite FAILED'
assert omega_s_r > 0, '[TC17] srs omega_s positive FAILED'
assert len(roots) == 4, '[TC17] srs root count FAILED'

# ---- TC18: sample_laser_plasma_parameters shape and bounds ----
params, names, bounds = sample_laser_plasma_parameters(10, seed=42)
assert params.shape == (10, 6), '[TC18] sample shape FAILED'
for i, (pmin, pmax) in enumerate(bounds):
    assert np.all(params[:, i] >= pmin), f'[TC18] sample lower bound dim {i} FAILED'
    assert np.all(params[:, i] <= pmax), f'[TC18] sample upper bound dim {i} FAILED'

# ---- TC19: sample_quality_metrics consistency ----
params, names, bounds = sample_laser_plasma_parameters(5, seed=42)
metrics = sample_quality_metrics(params)
assert metrics['min_pairwise_dist'] >= 0, '[TC19] sample_quality min dist FAILED'
assert metrics['max_pairwise_dist'] >= metrics['min_pairwise_dist'], '[TC19] sample_quality max>=min FAILED'

# ---- TC20: spherical_distance quarter circumference exact ----
R_test = 1.0
d = spherical_distance(0.0, 0.0, np.pi / 2.0, 0.0, R_test)
assert abs(d - np.pi / 2.0 * R_test) < 1e-12, '[TC20] spherical_distance quarter FAILED'

# ---- TC21: stereographic_projection known points ----
p_south = np.array([0.0, 0.0, -1.0])
q_south = stereographic_projection_sphere_to_plane(p_south)
assert np.allclose(q_south, [0.0, 0.0]), '[TC21] stereographic south pole FAILED'
p_equator = np.array([1.0, 0.0, 0.0])
q_equator = stereographic_projection_sphere_to_plane(p_equator)
assert np.allclose(q_equator, [2.0, 0.0]), '[TC21] stereographic equator FAILED'

# ---- TC22: icf_target_surface_mesh area converges to sphere ----
R_test = 100.0e-6
vertices, face_areas, total_area = icf_target_surface_mesh(R_test, 41, 81)
expected_area = 4.0 * np.pi * R_test**2
rel_err = abs(total_area - expected_area) / expected_area
assert rel_err < 0.05, '[TC22] icf_target_surface_mesh area FAILED'

# ---- TC23: solid_angle_subtended_by_quad finite non-negative ----
q = np.array([[1.0, 0.0, -1.0, 0.0],
              [0.0, 1.0, 0.0, -1.0],
              [0.0, 0.0, 0.0, 0.0]])
omega = solid_angle_subtended_by_quad(q, np.array([0.0, 0.0, 1.0]))
assert omega >= 0.0, '[TC23] solid_angle non-negative FAILED'
assert np.isfinite(omega), '[TC23] solid_angle finite FAILED'

# ---- TC24: faraday_rotation_integral zero B field ----
z_test = np.linspace(0.0, 1.0, 51)
ne_test_arr = np.ones(51) * 1.0e24
B_zero = np.zeros(51)
theta_F = faraday_rotation_integral(ne_test_arr, B_zero, z_test, 1.0e15)
assert abs(theta_F) < 1e-20, '[TC24] faraday_rotation_integral zero B FAILED'

# ---- TC25: circle_map_matrix_polarization identity aspect ratio ----
x_in, x_out, aspect = circle_map_matrix_polarization(np.eye(2), n_points=100)
assert abs(aspect - 1.0) < 1e-10, '[TC25] circle_map identity aspect FAILED'

# ---- TC26: solve_poisson_2d zero charge zero potential ----
nx_test, ny_test = 5, 5
dx_test, dy_test = 1.0e-6, 1.0e-6
rho_zero = np.zeros((nx_test, ny_test))
phi, residual, info = solve_poisson_2d(rho_zero, nx_test, ny_test, dx_test, dy_test)
assert np.allclose(phi, 0.0, atol=1e-10), '[TC26] solve_poisson zero phi FAILED'
assert residual < 1e-10, '[TC26] solve_poisson zero residual FAILED'

# ---- TC27: compute_electric_field_from_potential linear gradient ----
phi_linear = np.outer(np.arange(5), np.ones(5)).astype(float)
Ex, Ey = compute_electric_field_from_potential(phi_linear, 1.0, 1.0)
assert np.allclose(Ex[1:-1, 1:-1], -1.0, atol=1e-10), '[TC27] electric_field Ex FAILED'
assert np.allclose(Ey[1:-1, 1:-1], 0.0, atol=1e-10), '[TC27] electric_field Ey FAILED'

# ---- TC28: checksum_plasma_state reproducible ----
ne_test_arr = np.ones((3, 3)) * 1.0e24
Te_test_arr = np.ones((3, 3)) * 1000.0
phi_test_arr = np.zeros((3, 3))
cs1 = checksum_plasma_state(ne_test_arr, Te_test_arr, phi_test_arr)
cs2 = checksum_plasma_state(ne_test_arr, Te_test_arr, phi_test_arr)
assert cs1 == cs2, '[TC28] checksum reproducibility FAILED'

# ---- TC29: evolve_polarization_along_ray shape and no-B conservation ----
z_vals = np.linspace(0.0, 1.0e-6, 11)
def ne_func_z(z):
    return 1.0e24
def B_func_z(z):
    return np.array([0.0, 0.0, 0.0])
E0_jones = np.array([1.0, 0.0])
E_hist, S_hist = evolve_polarization_along_ray(1.0e15, ne_func_z, B_func_z, z_vals, E0_jones)
assert E_hist.shape == (11, 2), '[TC29] evolve_polarization E shape FAILED'
assert S_hist.shape == (11, 4), '[TC29] evolve_polarization S shape FAILED'
assert abs(S_hist[0, 0] - S_hist[-1, 0]) < 1e-3 * S_hist[0, 0], '[TC29] evolve_polarization intensity drift FAILED'

# ---- TC30: critical_density and plasma_frequency mutual consistency ----
omega_test = 1.0e15
nc_test = critical_density(omega_test)
omega_p_test = plasma_frequency(nc_test)
assert abs(omega_p_test - omega_test) / omega_test < 1e-10, '[TC30] critical_density plasma_frequency consistency FAILED'
print("\n全部 30 个测试通过!\n")
