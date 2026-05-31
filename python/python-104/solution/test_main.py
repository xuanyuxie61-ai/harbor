"""
main.py — 自适应光学波前校正系统的高保真数值模拟

统一入口, 零参数可运行.

执行流程:
  1. 系统参数初始化
  2. 生成大气湍流相位屏 (Kolmogorov + PME修正 + 光化学-热扰动)
  3. Shack-Hartmann波前传感器采样与斜率提取
  4. Zernike模态分解与zonal波前重构
  5. 变形镜面形计算与快速倾斜镜动力学响应
  6. 闭环PI控制迭代校正
  7. 光学传递函数、Strehl比、焦散奇点分析
  8. 控制参数扫描优化
  9. 结果输出与日志记录
"""

import os
import sys
import numpy as np

# 导入各模块
import data_io
import iterative_utils
import zernike_modes
import atmosphere_turbulence
import wavefront_propagation
import shack_hartmann_sensor
import wavefront_reconstruction
import optical_transfer
import deformable_mirror
import adaptive_sampling
import closed_loop_control


def main():
    print("=" * 70)
    print("Adaptive Optics Wavefront Correction System — High-Fidelity Simulation")
    print("=" * 70)

    # ============================================================
    # 1. 系统参数初始化
    # ============================================================
    print("\n[1/9] Initializing system parameters...")

    params = {
        'telescope_diameter_m': 1.0,
        'wavelength_m': 500e-9,
        'seeing_arcsec': 0.8,
        'grid_size': 128,
        'n_subapertures': 8,
        'n_zernike_modes': 21,
        'n_dm_actuators': 64,
        'fsm_bandwidth_hz': 100.0,
        'control_dt_ms': 0.5,
        'n_closed_loop_steps': 200,
        'seed': 42,
    }

    D = params['telescope_diameter_m']
    wavelength = params['wavelength_m']
    grid_size = params['grid_size']
    pixel_scale = D / grid_size
    n_subap = params['n_subapertures']
    n_modes = params['n_zernike_modes']
    n_actuators = params['n_dm_actuators']
    seed = params['seed']

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    data_io.log_system_parameters(
        os.path.join(output_dir, 'system_parameters.log'), params
    )
    print("  Parameters logged.")

    # ============================================================
    # 2. 生成大气湍流相位屏
    # ============================================================
    print("\n[2/9] Generating atmospheric turbulent phase screen...")

    phase_turb, r0, mask = atmosphere_turbulence.generate_turbulent_phase_screen(
        grid_size=grid_size,
        D_aperture=D,
        wavelength=wavelength,
        seeing=params['seeing_arcsec'],
        apply_pme_correction=True,
        apply_thermal_perturbation=True,
        seed=seed
    )

    # 计算Fried参数理论值
    Cn2_int = 1e-14 * D  # 简化的C_n^2积分
    r0_theory = atmosphere_turbulence.fried_parameter(wavelength, Cn2_int)
    print(f"  Fried parameter r0 = {r0:.4f} m (theory: {r0_theory:.4f} m)")
    print(f"  Phase RMS = {np.std(phase_turb[mask]):.4f} rad")

    # ============================================================
    # 3. Shack-Hartmann传感器采样
    # ============================================================
    print("\n[3/9] Shack-Hartmann wavefront sensor sampling...")

    subaps = shack_hartmann_sensor.generate_subaperture_grid(
        grid_size, n_subap, geometry='square'
    )
    print(f"  Number of subapertures: {len(subaps)}")

    sx, sy = shack_hartmann_sensor.compute_subaperture_slopes(
        phase_turb, subaps, pixel_scale,
        focal_length=0.1, noise_photon=0.005, noise_read=0.2
    )
    slope_vec = shack_hartmann_sensor.slopes_to_vector(sx, sy)
    print(f"  Slope vector norm: {np.linalg.norm(slope_vec):.4f}")

    data_io.write_subaperture_slopes(
        os.path.join(output_dir, 'subaperture_slopes.txt'), sx, sy
    )

    # ============================================================
    # 4. Zernike模态分解与波前重构
    # ============================================================
    print("\n[4/9] Zernike modal decomposition and wavefront reconstruction...")

    basis_flat, mask_z, x_vec, y_vec = zernike_modes.compute_zernike_basis(
        grid_size, n_modes
    )

    coeffs_turb = zernike_modes.zernike_decompose(phase_turb, mask, basis_flat)
    print(f"  Turbulence Zernike coeffs (first 5): {coeffs_turb[:5]}")

    # Modal重构 (从斜率)
    coeffs_recon_modal = wavefront_reconstruction.reconstruct_wavefront_modal(
        sx, sy, subaps, basis_flat, mask, pixel_scale
    )
    phase_recon_modal = zernike_modes.zernike_reconstruct(
        coeffs_recon_modal, basis_flat, mask
    )

    # Zonal重构
    phase_recon_zonal = wavefront_reconstruction.reconstruct_wavefront_zonal(
        sx, sy, subaps, grid_size, pixel_scale, method='cg'
    )

    # 单形约束搜索 (源自asa299)
    c_simplex, err_simplex = zernike_modes.zernike_coefficient_simplex_search(
        phase_turb, basis_flat, mask, T_max=3
    )
    print(f"  Simplex search best error: {err_simplex:.4e}")

    data_io.write_zernike_coefficients(
        os.path.join(output_dir, 'zernike_coefficients_turbulence.txt'),
        coeffs_turb
    )

    # ============================================================
    # 5. 变形镜与快速倾斜镜响应
    # ============================================================
    print("\n[5/9] Deformable mirror and fast steering mirror response...")

    dm = deformable_mirror.DeformableMirror(
        n_actuators=n_actuators,
        grid_size=grid_size,
        aperture_radius=D / 2.0,
        influence_sigma=0.08,
        use_magic_square_layout=True
    )

    # 计算DM电压 (简单映射: Zernike系数 -> 电压)
    R = dm.voltage_to_zernike_response(basis_flat, mask)
    voltages, _, _, _ = np.linalg.lstsq(R, coeffs_turb, rcond=None)
    voltages = np.clip(voltages, -1.0, 1.0)
    dm_surface = dm.compute_surface(voltages)

    # FSM双轴动力学响应
    fsm = deformable_mirror.FastSteeringMirrorDynamics(
        g=9.81, m1=0.01, m2=0.01, l1=0.05, l2=0.05,
        damping1=0.8, damping2=0.8, coupling=0.05
    )
    control_x = np.ones(500) * 0.01
    control_y = np.ones(500) * 0.005
    theta1_fsm, theta2_fsm = fsm.simulate_response(control_x, control_y, dt=1e-4)
    fsm_tilt_surface = theta1_fsm[-1] * x_vec[None, :] + theta2_fsm[-1] * y_vec[:, None]
    fsm_tilt_surface = fsm_tilt_surface[:grid_size, :grid_size]

    total_correction = dm_surface + fsm_tilt_surface
    phase_corrected = phase_turb - total_correction
    phase_corrected[mask] -= np.mean(phase_corrected[mask])

    strehl_before = wavefront_propagation.compute_strehl_ratio(phase_turb, wavelength, mask)
    strehl_after = wavefront_propagation.compute_strehl_ratio(phase_corrected, wavelength, mask)
    print(f"  Strehl ratio (before correction): {strehl_before:.4f}")
    print(f"  Strehl ratio (after correction):  {strehl_after:.4f}")

    # ============================================================
    # 6. 闭环PI控制迭代校正
    # ============================================================
    print("\n[6/9] Closed-loop PI control iteration...")

    turb_cov = zernike_modes.kolmogorov_zernike_covariance(n_modes, D / r0)
    dt_control = params['control_dt_ms'] * 1e-3
    n_steps = params['n_closed_loop_steps']

    final_strehl, mean_strehl, res_hist, str_hist = closed_loop_control.simulate_modal_control_loop(
        n_modes=n_modes,
        turb_covariance=turb_cov,
        Kp=0.6,
        Ki=0.15,
        bandwidth_hz=params['fsm_bandwidth_hz'],
        dt=dt_control,
        n_steps=n_steps,
        noise_std=0.02
    )

    print(f"  Final Strehl (closed-loop): {final_strehl:.4f}")
    print(f"  Mean Strehl (closed-loop):  {mean_strehl:.4f}")
    print(f"  Final residual RMS: {res_hist[-1]:.4f}")

    # HH控制电路响应验证
    hh = closed_loop_control.HHControlCircuit(C=1.0, dt=1e-5)
    for _ in range(1000):
        hh.step(75.0 if hh.V < -50 else 0.0)
    print(f"  HH circuit final V: {hh.V:.2f} mV")

    # ============================================================
    # 7. 光学传递函数与焦散分析
    # ============================================================
    print("\n[7/9] Optical transfer function and caustic singularity analysis...")

    P = optical_transfer.compute_pupil_function(grid_size, mask, phase_corrected)
    otf = optical_transfer.compute_otf_from_pupil(P)
    mtf = optical_transfer.compute_mtf_from_otf(otf)

    psf, x_coords = optical_transfer.compute_psf_from_pupil(
        P, pixel_scale, wavelength, focal_length=10.0
    )
    ee50 = optical_transfer.compute_encircled_energy(psf, x_coords, radius=0.5e-6)
    print(f"  Encircled energy (0.5 um): {ee50:.4f}")

    # 焦散奇点检测
    singularity, det_H = wavefront_propagation.detect_caustic_singularities(
        phase_turb, pixel_scale, mask
    )
    n_singularities = np.sum(singularity)
    print(f"  Caustic singularities detected: {n_singularities}")

    # 焦散线密度 (几何模拟)
    caustic_lines = wavefront_propagation.caustic_line_density(n=50, m=7)
    print(f"  Caustic network lines: {len(caustic_lines)}")

    # 四面体相位矩
    moments = optical_transfer.tetrahedral_phase_moments(phase_turb, mask, max_order=2)
    print(f"  Phase moments: M_00={moments.get((0,0),0):.4f}, M_10={moments.get((1,0),0):.4f}, M_01={moments.get((0,1),0):.4f}")

    # ============================================================
    # 8. 控制参数扫描优化
    # ============================================================
    print("\n[8/9] Control parameter sweep optimization...")

    def simulation_wrapper(Kp, Ki, bw, dt, n_steps):
        fs, ms, _, _ = closed_loop_control.simulate_modal_control_loop(
            n_modes=n_modes, turb_covariance=turb_cov,
            Kp=Kp, Ki=Ki, bandwidth_hz=bw, dt=dt, n_steps=n_steps, noise_std=0.02
        )
        return ms

    Kp_grid = np.linspace(0.1, 1.0, 5)
    Ki_grid = np.linspace(0.05, 0.3, 5)
    bw_grid = np.array([50.0, 100.0, 200.0])

    best_params, best_strehl, perf = closed_loop_control.parameter_sweep_optimization(
        Kp_grid, Ki_grid, bw_grid, simulation_wrapper, dt=dt_control, n_steps=100
    )
    print(f"  Optimal parameters: Kp={best_params[0]:.2f}, Ki={best_params[1]:.2f}, BW={best_params[2]:.1f} Hz")
    print(f"  Optimal mean Strehl: {best_strehl:.4f}")

    # ============================================================
    # 9. 自适应采样与CVT
    # ============================================================
    print("\n[9/9] Adaptive sampling and CVT optimization...")

    # CVT圆盘采样
    cvt_points = adaptive_sampling.cvt_disk_uniform(
        n_generators=32, radius=D / 2.0, n_iterations=20, seed=seed
    )
    print(f"  CVT generators: {len(cvt_points)}")

    # 自适应相位采样
    adaptive_pts = adaptive_sampling.adaptive_phase_sampling(
        phase_turb, mask, n_target_points=40, n_iterations=15
    )
    print(f"  Adaptive sampling points: {len(adaptive_pts)}")

    adaptive_sampling.log_sampling_info(
        os.path.join(output_dir, 'adaptive_sampling_points.txt'),
        adaptive_pts, iteration=15, residual=res_hist[-1]
    )

    # Collatz迭代步长分析
    for res_val in [1.0, 0.1, 0.01, 0.001]:
        step = iterative_utils.collatz_step_size(res_val, base_step=0.5)
        print(f"  Collatz step (res={res_val}): {step:.6f}")

    # ============================================================
    # 结果汇总
    # ============================================================
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"  - system_parameters.log")
    print(f"  - subaperture_slopes.txt")
    print(f"  - zernike_coefficients_turbulence.txt")
    print(f"  - adaptive_sampling_points.txt")
    print(f"\nKey metrics:")
    print(f"  Initial Strehl:     {strehl_before:.4f}")
    print(f"  Corrected Strehl:   {strehl_after:.4f}")
    print(f"  Closed-loop Strehl: {final_strehl:.4f}")
    print(f"  Optimal (Kp,Ki,BW): ({best_params[0]:.2f}, {best_params[1]:.2f}, {best_params[2]:.1f})")
    print(f"  Caustic singularities: {n_singularities}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    ret = main()
    np.random.seed(42)

    # ================================================================
    # 测试用例（35个，assert模式，涉及随机值均使用固定种子）
    # ================================================================
    # ---- TC01: atmosphere_turbulence.fried_parameter 正值与有限性验证 ----
    r0_01 = atmosphere_turbulence.fried_parameter(500e-9, 1e-14)
    assert r0_01 > 0.0 and np.isfinite(r0_01), '[TC01] atmosphere_turbulence.fried_parameter 正值与有限性验证 FAILED'
    
    # ---- TC02: atmosphere_turbulence.hufnagel_valley_cnsquared 输出非负性 ----
    h_vals_02 = np.array([0.0, 1000.0, 5000.0, 10000.0])
    cn2_02 = atmosphere_turbulence.hufnagel_valley_cnsquared(h_vals_02)
    assert np.all(cn2_02 >= 0.0), '[TC02] atmosphere_turbulence.hufnagel_valley_cnsquared 输出非负性 FAILED'
    
    # ---- TC03: atmosphere_turbulence.generate_phase_screen 相同种子可复现性 ----
    np.random.seed(42)
    ph3a, m3a = atmosphere_turbulence.generate_phase_screen(64, 0.01, 0.1, seed=42)
    np.random.seed(42)
    ph3b, m3b = atmosphere_turbulence.generate_phase_screen(64, 0.01, 0.1, seed=42)
    assert np.allclose(ph3a, ph3b), '[TC03] atmosphere_turbulence.generate_phase_screen 相同种子可复现性 FAILED'
    assert ph3a.shape == (64, 64) and m3a.shape == (64, 64), '[TC03] atmosphere_turbulence.generate_phase_screen 相同种子可复现性 FAILED'
    
    # ---- TC04: atmosphere_turbulence.barenblatt_pme_solution 零边界与外区域归零 ----
    x4 = np.array([0.0, 1.0, 10.0])
    u4 = atmosphere_turbulence.barenblatt_pme_solution(x4, 1.0, m=3.0, c=0.01, delta=1.0)
    assert np.all(u4 >= 0.0), '[TC04] atmosphere_turbulence.barenblatt_pme_solution 零边界与外区域归零 FAILED'
    assert u4[2] == 0.0, '[TC04] atmosphere_turbulence.barenblatt_pme_solution 零边界与外区域归零 FAILED'
    
    # ---- TC05: shack_hartmann_sensor.generate_subaperture_grid 方形网格子孔径数量 ----
    subs5 = shack_hartmann_sensor.generate_subaperture_grid(64, 4, geometry='square')
    assert len(subs5) == 16, '[TC05] shack_hartmann_sensor.generate_subaperture_grid 方形网格子孔径数量 FAILED'
    
    # ---- TC06: shack_hartmann_sensor.slopes_to_vector 与 vector_to_slopes 互逆一致性 ----
    sx6 = np.array([1.0, 2.0, 3.0])
    sy6 = np.array([4.0, 5.0, 6.0])
    svec6 = shack_hartmann_sensor.slopes_to_vector(sx6, sy6)
    sx6b, sy6b = shack_hartmann_sensor.vector_to_slopes(svec6)
    assert np.allclose(sx6, sx6b) and np.allclose(sy6, sy6b), '[TC06] shack_hartmann_sensor.slopes_to_vector 与 vector_to_slopes 互逆一致性 FAILED'
    
    # ---- TC07: zernike_modes.zernike_radial R_0^0 恒等于1解析验证 ----
    rho7 = np.array([0.0, 0.3, 0.7, 1.0])
    R7 = zernike_modes.zernike_radial(0, 0, rho7)
    assert np.allclose(R7, 1.0), '[TC07] zernike_modes.zernike_radial R_0^0 恒等于1解析验证 FAILED'
    
    # ---- TC08: zernike_modes.noll_to_nm j=1 对应 (n=0,m=0) ----
    n8, m8 = zernike_modes.noll_to_nm(1)
    assert n8 == 0 and m8 == 0, '[TC08] zernike_modes.noll_to_nm j=1 对应 (n=0,m=0) FAILED'
    
    # ---- TC09: zernike_modes.compute_zernike_basis 输出维度与掩码形状 ----
    basis9, mask9, xv9, yv9 = zernike_modes.compute_zernike_basis(32, 6)
    assert basis9.shape == (32*32, 6), '[TC09] zernike_modes.compute_zernike_basis 输出维度与掩码形状 FAILED'
    assert mask9.shape == (32, 32), '[TC09] zernike_modes.compute_zernike_basis 输出维度与掩码形状 FAILED'
    assert len(xv9) == 32 and len(yv9) == 32, '[TC09] zernike_modes.compute_zernike_basis 输出维度与掩码形状 FAILED'
    
    # ---- TC10: zernike_modes.zernike_decompose 与 zernike_reconstruct 互逆一致性 ----
    basis10, mask10, _, _ = zernike_modes.compute_zernike_basis(32, 6)
    phase10 = np.zeros((32, 32))
    phase10[mask10] = basis10[mask10.ravel(), 2]
    coeffs10 = zernike_modes.zernike_decompose(phase10, mask10, basis10)
    recon10 = zernike_modes.zernike_reconstruct(coeffs10, basis10, mask10)
    assert np.allclose(phase10[mask10], recon10[mask10], atol=1e-6), '[TC10] zernike_modes.zernike_decompose 与 zernike_reconstruct 互逆一致性 FAILED'
    
    # ---- TC11: zernike_modes.kolmogorov_zernike_covariance 对角正定性与对称性 ----
    cov11 = zernike_modes.kolmogorov_zernike_covariance(10, 2.0)
    assert np.all(np.diag(cov11) > 0), '[TC11] zernike_modes.kolmogorov_zernike_covariance 对角正定性与对称性 FAILED'
    assert np.all(cov11 == cov11.T), '[TC11] zernike_modes.kolmogorov_zernike_covariance 对角正定性与对称性 FAILED'
    
    # ---- TC12: zernike_modes.simplex_lattice_enum 组合数公式验证 ----
    lattice12 = zernike_modes.simplex_lattice_enum(3, 4)
    assert len(lattice12) == 35, '[TC12] zernike_modes.simplex_lattice_enum 组合数公式验证 FAILED'
    assert np.all(np.sum(lattice12, axis=1) <= 4), '[TC12] zernike_modes.simplex_lattice_enum 组合数公式验证 FAILED'
    
    # ---- TC13: wavefront_reconstruction.R83VOperator_matvec 线性叠加性 ----
    A13 = wavefront_reconstruction.build_southwell_matrix_1d(5, 0.1)
    x1_13 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    x2_13 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    y_sum13 = A13.matvec(x1_13 + x2_13)
    y1_13 = A13.matvec(x1_13)
    y2_13 = A13.matvec(x2_13)
    assert np.allclose(y_sum13, y1_13 + y2_13), '[TC13] wavefront_reconstruction.R83VOperator_matvec 线性叠加性 FAILED'
    
    # ---- TC14: wavefront_reconstruction.R83VOperator_conjugate_gradient_solve 残差足够小 ----
    A14 = wavefront_reconstruction.build_southwell_matrix_1d(5, 0.1)
    b14 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x14 = A14.conjugate_gradient_solve(b14, tol=1e-12)
    res14 = A14.residual(x14, b14)
    assert np.linalg.norm(res14) < 1e-8, '[TC14] wavefront_reconstruction.R83VOperator_conjugate_gradient_solve 残差足够小 FAILED'
    
    # ---- TC15: wavefront_reconstruction.reconstruct_wavefront_1d 零斜率重构为常数 ----
    slopes15 = np.zeros(5)
    recon15 = wavefront_reconstruction.reconstruct_wavefront_1d(slopes15, 0.1, method='cg')
    assert len(recon15) == 6, '[TC15] wavefront_reconstruction.reconstruct_wavefront_1d 零斜率重构为常数 FAILED'
    assert np.allclose(np.diff(recon15), 0.0, atol=1e-8), '[TC15] wavefront_reconstruction.reconstruct_wavefront_1d 零斜率重构为常数 FAILED'
    
    # ---- TC16: deformable_mirror.magic4_matrix 幻方常数行和列和对角线验证 ----
    M16 = deformable_mirror.magic4_matrix(4)
    magic_sum16 = np.sum(M16[0, :])
    assert np.all(np.sum(M16, axis=1) == magic_sum16), '[TC16] deformable_mirror.magic4_matrix 幻方常数行和列和对角线验证 FAILED'
    assert np.all(np.sum(M16, axis=0) == magic_sum16), '[TC16] deformable_mirror.magic4_matrix 幻方常数行和列和对角线验证 FAILED'
    assert np.sum(np.diag(M16)) == magic_sum16, '[TC16] deformable_mirror.magic4_matrix 幻方常数行和列和对角线验证 FAILED'
    
    # ---- TC17: deformable_mirror.gaussian_influence_function 中心值为1且偏心衰减 ----
    inf17_center = deformable_mirror.gaussian_influence_function(0.0, 0.0, 0.0, 0.0, 0.1)
    inf17_off = deformable_mirror.gaussian_influence_function(0.1, 0.0, 0.0, 0.0, 0.1)
    assert abs(inf17_center - 1.0) < 1e-10, '[TC17] deformable_mirror.gaussian_influence_function 中心值为1且偏心衰减 FAILED'
    assert inf17_off < 1.0, '[TC17] deformable_mirror.gaussian_influence_function 中心值为1且偏心衰减 FAILED'
    
    # ---- TC18: deformable_mirror.DeformableMirror_compute_surface 零电压输出为零 ----
    dm18 = deformable_mirror.DeformableMirror(16, 32, aperture_radius=0.5, influence_sigma=0.05, use_magic_square_layout=True)
    surf18 = dm18.compute_surface(np.zeros(16))
    assert surf18.shape == (32, 32), '[TC18] deformable_mirror.DeformableMirror_compute_surface 零电压输出为零 FAILED'
    assert np.allclose(surf18, 0.0), '[TC18] deformable_mirror.DeformableMirror_compute_surface 零电压输出为零 FAILED'
    
    # ---- TC19: deformable_mirror.FastSteeringMirrorDynamics_simulate_response 输出长度匹配 ----
    fsm19 = deformable_mirror.FastSteeringMirrorDynamics()
    t1_19, t2_19 = fsm19.simulate_response(np.ones(100)*0.001, np.ones(100)*0.001, dt=1e-4)
    assert len(t1_19) == 100 and len(t2_19) == 100, '[TC19] deformable_mirror.FastSteeringMirrorDynamics_simulate_response 输出长度匹配 FAILED'
    
    # ---- TC20: closed_loop_control.PIController_update 积分windup限幅验证 ----
    ctrl20 = closed_loop_control.PIController(Kp=1.0, Ki=10.0, dt=0.1, integral_limit=1.0)
    for _ in range(100):
        u20 = ctrl20.update(1.0)
    assert abs(ctrl20.integral) <= 1.0001, '[TC20] closed_loop_control.PIController_update 积分windup限幅验证 FAILED'
    assert u20 <= 11.0, '[TC20] closed_loop_control.PIController_update 积分windup限幅验证 FAILED'
    
    # ---- TC21: closed_loop_control.BandwidthLimitedActuator_step 阶跃响应稳态趋近1 ----
    act21 = closed_loop_control.BandwidthLimitedActuator(100.0, 1e-3)
    for _ in range(5000):
        out21 = act21.step(1.0)
    assert abs(out21 - 1.0) < 0.01, '[TC21] closed_loop_control.BandwidthLimitedActuator_step 阶跃响应稳态趋近1 FAILED'
    
    # ---- TC22: closed_loop_control.HHControlCircuit_step 膜电位有界性 ----
    hh22 = closed_loop_control.HHControlCircuit(C=1.0, dt=1e-5)
    for _ in range(2000):
        v22 = hh22.step(10.0)
    assert np.isfinite(v22) and -100.0 < v22 < 100.0, '[TC22] closed_loop_control.HHControlCircuit_step 膜电位有界性 FAILED'
    
    # ---- TC23: optical_transfer.tetrahedron01_monomial_integral 零次体积为1/6 ----
    vol23 = optical_transfer.tetrahedron01_monomial_integral(0, 0, 0)
    assert abs(vol23 - 1.0/6.0) < 1e-14, '[TC23] optical_transfer.tetrahedron01_monomial_integral 零次体积为1/6 FAILED'
    
    # ---- TC24: optical_transfer.compute_pupil_function 瞳孔区域内模恒为1 ----
    N24 = 32
    mask24 = np.zeros((N24, N24), dtype=bool)
    mask24[8:24, 8:24] = True
    phase24 = np.zeros((N24, N24))
    P24 = optical_transfer.compute_pupil_function(N24, mask24, phase24)
    mod24 = np.abs(P24[mask24])
    assert np.allclose(mod24, 1.0), '[TC24] optical_transfer.compute_pupil_function 瞳孔区域内模恒为1 FAILED'
    
    # ---- TC25: optical_transfer.compute_mtf_from_otf 输出归一化范围[0,1] ----
    otf25 = np.ones((32, 32), dtype=complex)
    mtf25 = optical_transfer.compute_mtf_from_otf(otf25)
    assert np.all(mtf25 >= 0.0) and np.all(mtf25 <= 1.0 + 1e-10), '[TC25] optical_transfer.compute_mtf_from_otf 输出归一化范围[0,1] FAILED'
    
    # ---- TC26: wavefront_propagation.compute_strehl_ratio 平坦波前应接近1 ----
    N26 = 64
    x26 = np.linspace(-1, 1, N26)
    X26, Y26 = np.meshgrid(x26, x26)
    mask26 = (X26**2 + Y26**2) <= 1.0
    flat26 = np.zeros((N26, N26))
    S26 = wavefront_propagation.compute_strehl_ratio(flat26, 500e-9, mask26)
    assert abs(S26 - 1.0) < 1e-6, '[TC26] wavefront_propagation.compute_strehl_ratio 平坦波前应接近1 FAILED'
    
    # ---- TC27: wavefront_propagation.detect_caustic_singularities 输出形状与输入一致 ----
    N27 = 32
    phase27 = np.zeros((N27, N27))
    mask27 = np.ones((N27, N27), dtype=bool)
    sing27, detH27 = wavefront_propagation.detect_caustic_singularities(phase27, 0.01, mask27)
    assert sing27.shape == (N27, N27) and detH27.shape == (N27, N27), '[TC27] wavefront_propagation.detect_caustic_singularities 输出形状与输入一致 FAILED'
    
    # ---- TC28: adaptive_sampling.triangle_area 直角三角形面积解析验证 ----
    area28 = adaptive_sampling.triangle_area([0, 0], [3, 0], [0, 4])
    assert abs(area28 - 6.0) < 1e-10, '[TC28] adaptive_sampling.triangle_area 直角三角形面积解析验证 FAILED'
    
    # ---- TC29: adaptive_sampling.sample_triangle_uniform 采样点位于三角形内 ----
    np.random.seed(42)
    pts29 = adaptive_sampling.sample_triangle_uniform([0, 0], [1, 0], [0, 1], 200, seed=42)
    assert pts29.shape == (200, 2), '[TC29] adaptive_sampling.sample_triangle_uniform 采样点位于三角形内 FAILED'
    assert np.all(pts29 >= -1e-9), '[TC29] adaptive_sampling.sample_triangle_uniform 采样点位于三角形内 FAILED'
    assert np.all(pts29[:, 0] + pts29[:, 1] <= 1.0001), '[TC29] adaptive_sampling.sample_triangle_uniform 采样点位于三角形内 FAILED'
    
    # ---- TC30: iterative_utils.collatz_step_size 大步与小步单调递减性 ----
    s_large30 = iterative_utils.collatz_step_size(1e6, base_step=1.0)
    s_small30 = iterative_utils.collatz_step_size(1e-12, base_step=1.0)
    assert s_large30 < s_small30, '[TC30] iterative_utils.collatz_step_size 大步与小步单调递减性 FAILED'
    assert abs(s_small30 - 0.5) < 1e-10, '[TC30] iterative_utils.collatz_step_size 大步与小步单调递减性 FAILED'
    
    # ---- TC31: iterative_utils.collatz_sequence_length 已知值验证 ----
    assert iterative_utils.collatz_sequence_length(1) == 1, '[TC31] iterative_utils.collatz_sequence_length 已知值验证 FAILED'
    assert iterative_utils.collatz_sequence_length(2) == 2, '[TC31] iterative_utils.collatz_sequence_length 已知值验证 FAILED'
    
    # ---- TC32: iterative_utils.adaptive_relaxation 输出在指定区间内 ----
    omega32 = iterative_utils.adaptive_relaxation([1.0, 0.5], omega_min=0.1, omega_max=1.9)
    assert 0.1 <= omega32 <= 1.9, '[TC32] iterative_utils.adaptive_relaxation 输出在指定区间内 FAILED'
    
    # ---- TC33: data_io.write_xy_data 与 read_xy_data 往返一致性 ----
    import os as _os33
    tmp33 = '/tmp/test_xy_104_new.txt'
    data_io.write_xy_data(tmp33, [0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
    x33, y33 = data_io.read_xy_data(tmp33)
    assert len(x33) == 3 and np.allclose(x33, [0.0, 1.0, 2.0]) and np.allclose(y33, [3.0, 4.0, 5.0]), '[TC33] data_io.write_xy_data 与 read_xy_data 往返一致性 FAILED'
    _os33.remove(tmp33)
    
    # ---- TC34: data_io.write_zernike_coefficients 与 read_zernike_coefficients 往返一致性 ----
    import os as _os34
    tmp34 = '/tmp/test_zern_104_new.txt'
    data_io.write_zernike_coefficients(tmp34, np.array([0.5, -0.3, 0.1]))
    c34 = data_io.read_zernike_coefficients(tmp34)
    assert len(c34) == 3 and np.allclose(c34, [0.5, -0.3, 0.1]), '[TC34] data_io.write_zernike_coefficients 与 read_zernike_coefficients 往返一致性 FAILED'
    _os34.remove(tmp34)
    
    # ---- TC35: data_io.log_system_parameters 与 read_system_parameters 往返一致性 ----
    import os as _os35
    tmp35 = '/tmp/test_params_104_new.txt'
    params35 = {'a': 1.0, 'b': 2, 'c': 'test'}
    data_io.log_system_parameters(tmp35, params35)
    read35 = data_io.read_system_parameters(tmp35)
    assert read35['a'] == 1.0 and read35['b'] == 2 and read35['c'] == 'test', '[TC35] data_io.log_system_parameters 与 read_system_parameters 往返一致性 FAILED'
    _os35.remove(tmp35)
    
    # ---- TC36: main() 零参数集成测试返回值验证 ----
    assert ret == 0, '[TC36] main() 零参数集成测试返回值验证 FAILED'

    print('\n全部 36 个测试通过!\n')
    sys.exit(ret)

